"""Jedna, wspólna ewaluacja dla każdego modelu (.pt albo .onnx).

Przykład:
    python scripts/evaluate.py stare/baseline_v0_best.pt --name baseline_v0 --seed 42 \
        --config configs/baseline.yaml --emissions path/do/emissions.csv

Dopisuje wiersz do results/results.csv i per-klasową tabelę do results/results_per_class.csv.
Energię raportujemy w kWh (CO2 zależy od wykrytego regionu serwera i nie nadaje się do porównań).
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from common import ROOT, DATASET, git_commit, host_info, local_data_yaml

RESULTS = ROOT / "results" / "results.csv"
PER_CLASS = ROOT / "results" / "results_per_class.csv"


def energy_kwh(emissions_csv):
    """Suma energii po uruchomieniach (run_id); ostatni wiersz każdego uruchomienia = wartość skumulowana."""
    if not emissions_csv or not Path(emissions_csv).is_file():
        return None, None
    df = pd.read_csv(emissions_csv)
    last = df.groupby("run_id")[["duration", "energy_consumed"]].max().sum()
    return float(last["energy_consumed"]), float(last["duration"]) / 60


def cpu_latency(model, images, repeats, imgsz):
    """ms na jedno zdjęcie (preprocessing + inferencja + NMS), batch=1, CPU. Zwraca średnią i odchylenie między powtórzeniami."""
    model.predict(str(images[0]), imgsz=imgsz, device="cpu", verbose=False)  # rozgrzewka
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        for img in images:
            model.predict(str(img), imgsz=imgsz, device="cpu", verbose=False)
        runs.append((time.perf_counter() - start) / len(images) * 1000)
    s = pd.Series(runs)
    return float(s.mean()), float(s.std())


def append(path, df):
    """Dopisuje wiersze; przy zmianie kolumn nie rozjeżdża pliku (łączy po nazwach kolumn)."""
    if path.is_file():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--name", required=True, help="nazwa modelu/eksperymentu w tabeli")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="urządzenie do walidacji (domyślnie auto)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=None, help="ścieżka do configu, z którego wytrenowano model")
    ap.add_argument("--emissions", default=None, help="emissions.csv z treningu (CodeCarbon)")
    ap.add_argument("--latency-images", type=int, default=100)
    ap.add_argument("--latency-repeats", type=int, default=5)
    args = ap.parse_args()

    from ultralytics import YOLO  # import po parsowaniu argumentów, żeby --help działał szybko

    weights = Path(args.weights)
    data = local_data_yaml(ROOT / "results" / ".tmp")
    model = YOLO(str(weights))
    m = model.val(data=str(data), split=args.split, imgsz=args.imgsz, batch=args.batch, device=args.device,
                  workers=0, plots=False, verbose=False, project=str(ROOT / "results" / ".tmp"), name=args.name,
                  exist_ok=True)

    p, r = m.box.p, m.box.r  # per klasa (przy progu maksymalizującym F1 w Ultralytics)
    f1_per_class = 2 * p * r / (p + r + 1e-9)
    images = sorted((DATASET / args.split / "images").glob("*"))[: args.latency_images]
    ms, ms_std = cpu_latency(model, images, args.latency_repeats, args.imgsz)
    kwh, train_min = energy_kwh(args.emissions)

    row = {
        "model": args.name, "weights": str(weights), "split": args.split, "n_images": len(list((DATASET / args.split / "images").glob("*"))),
        "seed": args.seed, "config": args.config, "commit": git_commit(),
        "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
        "precision": float(m.box.mp), "recall": float(m.box.mr), "F1_macro": float(f1_per_class.mean()),
        "ms_cpu": ms, "ms_cpu_std": ms_std, "FPS_cpu": 1000 / ms, "latency_images": len(images),
        "size_MB": weights.stat().st_size / 1e6,
        "train_kWh": kwh, "train_min": train_min,
        **{f"host_{k}": v for k, v in host_info().items()},
    }
    if weights.suffix == ".pt":
        from ultralytics.utils.torch_utils import get_flops
        row["GFLOPs"] = float(get_flops(model.model, args.imgsz))
        row["params_M"] = sum(x.numel() for x in model.model.parameters()) / 1e6

    per_class = pd.DataFrame({
        "model": args.name, "class": [m.names[int(c)] for c in m.box.ap_class_index],
        "mAP50-95": [float(m.box.maps[int(c)]) for c in m.box.ap_class_index],
    })

    RESULTS.parent.mkdir(exist_ok=True)
    append(RESULTS, pd.DataFrame([row]))
    append(PER_CLASS, per_class)
    print(pd.DataFrame([row]).T.to_string())
    print(per_class.to_string(index=False))


if __name__ == "__main__":
    main()

"""Trening z configu YAML + pomiar energii (CodeCarbon) + automatyczna ewaluacja wspólnym evaluate.py.

    python scripts/train.py configs/baseline.yaml --name baseline_s42
    python scripts/train.py configs/baseline.yaml --name baseline_s1 --seed 1
    python scripts/train.py configs/baseline.yaml --name no_flip --set fliplr=0.0   # jedna zmiana na eksperyment

Wyniki: runs/<name>/ (weights/, results.csv, emissions.csv, config_used.yaml, environment.json).
"""
import argparse
import subprocess
import sys
from datetime import datetime

import yaml

from common import ROOT, git_commit, host_info, local_data_yaml, write_json


def parse_overrides(items):
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        out[key] = yaml.safe_load(value)  # '0.5' -> float, 'true' -> bool, 'AdamW' -> str
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", metavar="KEY=VAL", help="nadpisania configu (loguj je w nazwie eksperymentu)")
    ap.add_argument("--country", default="POL")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg.update(parse_overrides(args.set))
    if args.seed is not None:
        cfg["seed"] = args.seed
    run = ROOT / "runs" / args.name
    if run.exists():
        sys.exit(f"{run} już istnieje - wybierz inną --name (nie nadpisujemy wyników).")
    run.mkdir(parents=True)
    data = local_data_yaml(run)
    (run / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(run / "environment.json", {**host_info(), "commit": git_commit(), "config": args.config,
                                          "overrides": args.set or [], "started": datetime.now().isoformat()})

    import torch
    from ultralytics import YOLO

    device = args.device or (0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    tracker = None
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(project_name=args.name, country_iso_code=args.country, output_dir=str(run),
                                          output_file="emissions.csv", measure_power_secs=15, tracking_mode="machine",
                                          save_to_file=True, save_to_api=False, log_level="warning")
        tracker.start()
    except ImportError:
        print("UWAGA: brak codecarbon - energia nie będzie mierzona (pip install codecarbon).")

    cfg = dict(cfg)
    model = YOLO(cfg.pop("model"))
    try:
        model.train(data=str(data), device=device, project=str(run.parent), name=args.name, exist_ok=True,
                    amp=torch.cuda.is_available(), **cfg)
    finally:
        if tracker:
            tracker.stop()

    best = run / "weights" / "best.pt"
    if not args.no_eval and best.is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), str(best), "--name", args.name,
                        "--seed", str(cfg.get("seed")), "--config", args.config, "--emissions",
                        str(run / "emissions.csv")], check=True)


if __name__ == "__main__":
    main()

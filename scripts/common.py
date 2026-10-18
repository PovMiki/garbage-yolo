"""Wspólne pomoce dla train.py i evaluate.py."""
import json
import platform
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset_clean"


def local_data_yaml(out_dir: Path) -> Path:
    """data.yaml z poprawioną ścieżką `path` (w repo jest absolutna ścieżka z czyjegoś komputera)."""
    cfg = yaml.safe_load((DATASET / "data.yaml").read_text())
    cfg["path"] = str(DATASET.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "data_local.yaml"
    target.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return target


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def host_info() -> dict:
    info = {"machine": platform.machine(), "system": platform.platform(), "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    return info


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

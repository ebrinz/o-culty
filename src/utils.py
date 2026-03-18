from pathlib import Path
import json
import traceback
from datetime import datetime, timezone
import yaml
import torch


def load_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def log_error(stage: str, source: str, item_id: str, error: Exception, errors_dir: Path | None = None) -> None:
    if errors_dir is None:
        errors_dir = Path(__file__).parent.parent / "data" / "errors"
    errors_dir = Path(errors_dir)
    errors_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "source": source,
        "item_id": item_id,
        "error_type": type(error).__name__,
        "traceback": traceback.format_exception(error)[-1].strip() if traceback.format_exception(error) else str(error),
    }
    with open(errors_dir / f"{stage}.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_device(device_config: str = "auto") -> str:
    if device_config != "auto":
        return device_config
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

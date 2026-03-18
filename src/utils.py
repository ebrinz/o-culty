from pathlib import Path
import yaml


def load_config(path: Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)

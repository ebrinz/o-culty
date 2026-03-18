import pytest
from pathlib import Path
from src.utils import load_config


def test_load_config_returns_dict(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("scraping:\n  sacred_texts:\n    enabled: true\n    delay_seconds: 1.5\n")
    config = load_config(cfg_file)
    assert isinstance(config, dict)
    assert config["scraping"]["sacred_texts"]["enabled"] is True
    assert config["scraping"]["sacred_texts"]["delay_seconds"] == 1.5


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


import json
from src.utils import log_error


def test_log_error_creates_jsonl(tmp_path):
    log_error(stage="scraping", source="sacred-texts", item_id="test-item", error=ValueError("test error"), errors_dir=tmp_path)
    log_file = tmp_path / "scraping.jsonl"
    assert log_file.exists()
    entry = json.loads(log_file.read_text().strip())
    assert entry["stage"] == "scraping"
    assert entry["source"] == "sacred-texts"
    assert entry["item_id"] == "test-item"
    assert entry["error_type"] == "ValueError"
    assert "test error" in entry["traceback"]


def test_log_error_appends(tmp_path):
    log_error("scraping", "src1", "id1", ValueError("e1"), errors_dir=tmp_path)
    log_error("scraping", "src2", "id2", ValueError("e2"), errors_dir=tmp_path)
    log_file = tmp_path / "scraping.jsonl"
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2


from src.utils import get_device


def test_get_device_auto():
    device = get_device("auto")
    assert device in ("mps", "cuda", "cpu")


def test_get_device_explicit():
    assert get_device("cpu") == "cpu"

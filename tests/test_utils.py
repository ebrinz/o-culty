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

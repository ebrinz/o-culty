import json
import pytest
from pathlib import Path
from src.scraper.base import BaseScraper

class MockScraper(BaseScraper):
    def scrape(self):
        pass

def test_manifest_loads_empty(tmp_path):
    scraper = MockScraper(name="test", output_dir=tmp_path, delay=0)
    assert scraper.manifest == {"items": {}}

def test_manifest_save_and_reload(tmp_path):
    scraper = MockScraper(name="test", output_dir=tmp_path, delay=0)
    scraper.manifest["items"]["item1"] = {"title": "Test", "status": "downloaded"}
    scraper.save_manifest()
    scraper2 = MockScraper(name="test", output_dir=tmp_path, delay=0)
    assert "item1" in scraper2.manifest["items"]

def test_is_downloaded(tmp_path):
    scraper = MockScraper(name="test", output_dir=tmp_path, delay=0)
    assert scraper.is_downloaded("item1") is False
    scraper.manifest["items"]["item1"] = {"status": "downloaded"}
    scraper.save_manifest()
    assert scraper.is_downloaded("item1") is True

def test_mark_downloaded(tmp_path):
    scraper = MockScraper(name="test", output_dir=tmp_path, delay=0)
    scraper.mark_downloaded("item1", {"title": "Test"})
    assert scraper.manifest["items"]["item1"]["status"] == "downloaded"
    assert scraper.manifest["items"]["item1"]["title"] == "Test"

def test_mark_failed(tmp_path):
    scraper = MockScraper(name="test", output_dir=tmp_path, delay=0)
    scraper.mark_failed("item1", "404 Not Found")
    assert scraper.manifest["items"]["item1"]["status"] == "failed"
    assert scraper.manifest["items"]["item1"]["error"] == "404 Not Found"

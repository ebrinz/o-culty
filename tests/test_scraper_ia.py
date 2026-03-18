import pytest
from pathlib import Path
from src.scraper.internet_archive import InternetArchiveScraper

def test_scraper_init(tmp_path):
    scraper = InternetArchiveScraper(output_dir=tmp_path / "raw" / "internet-archive", search_terms=["occult"], max_results_per_term=5, formats=["pdf", "txt"], delay=0)
    assert scraper.name == "internet-archive"
    assert scraper.search_terms == ["occult"]

def test_build_search_query():
    scraper = InternetArchiveScraper.__new__(InternetArchiveScraper)
    scraper.formats = ["pdf", "txt"]
    query = scraper.build_search_query("occult")
    assert "occult" in query
    assert "mediatype:texts" in query

def test_skips_already_downloaded(tmp_path):
    scraper = InternetArchiveScraper(output_dir=tmp_path / "raw" / "internet-archive", search_terms=["occult"], max_results_per_term=5, formats=["pdf", "txt"], delay=0)
    scraper.mark_downloaded("test-item", {"title": "Test"})
    assert scraper.is_downloaded("test-item") is True

import pytest
from pathlib import Path
from src.scraper.gutenberg import GutenbergScraper

SEARCH_RESULTS_HTML = """
<html><body>
<li class="booklink">
  <a class="link" href="/ebooks/1234">The Book of Magic</a>
  <span class="subtitle">by John Dee</span>
</li>
<li class="booklink">
  <a class="link" href="/ebooks/5678">Alchemy Revealed</a>
  <span class="subtitle">by Anonymous</span>
</li>
</body></html>
"""

def test_parse_search_results():
    scraper = GutenbergScraper.__new__(GutenbergScraper)
    results = scraper.parse_search_results(SEARCH_RESULTS_HTML)
    assert len(results) == 2
    assert results[0]["ebook_id"] == "1234"
    assert results[0]["title"] == "The Book of Magic"

def test_scraper_init(tmp_path):
    scraper = GutenbergScraper(output_dir=tmp_path / "raw" / "gutenberg", search_terms=["occult"], delay=0)
    assert scraper.name == "gutenberg"

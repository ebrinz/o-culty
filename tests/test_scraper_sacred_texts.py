import pytest
from pathlib import Path
from src.scraper.sacred_texts import SacredTextsScraper

TRADITION_INDEX_HTML = """
<html><body>
<a href="/eso/book1/index.htm">Book One</a>
<a href="/eso/book2/index.htm">Book Two</a>
</body></html>
"""

BOOK_INDEX_HTML = """
<html><body>
<a href="chap1.htm">Chapter 1</a>
<a href="chap2.htm">Chapter 2</a>
</body></html>
"""

def test_parse_tradition_index():
    scraper = SacredTextsScraper.__new__(SacredTextsScraper)
    scraper.BASE_URL = "https://sacred-texts.com"
    links = scraper.parse_tradition_index(TRADITION_INDEX_HTML, "https://sacred-texts.com/eso/")
    assert len(links) == 2
    assert "https://sacred-texts.com/eso/book1/index.htm" in links

def test_parse_book_index():
    scraper = SacredTextsScraper.__new__(SacredTextsScraper)
    scraper.BASE_URL = "https://sacred-texts.com"
    links = scraper.parse_book_index(BOOK_INDEX_HTML, "https://sacred-texts.com/eso/book1/index.htm")
    assert len(links) == 2
    assert "https://sacred-texts.com/eso/book1/chap1.htm" in links

def test_scraper_skips_already_downloaded(tmp_path):
    scraper = SacredTextsScraper(output_dir=tmp_path / "raw" / "sacred-texts", traditions=["eso"], delay=0)
    scraper.mark_downloaded("eso/book1", {"title": "Book One"})
    assert scraper.is_downloaded("eso/book1") is True

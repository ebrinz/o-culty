import pytest
from pathlib import Path
from src.scraper.sacred_texts import SacredTextsScraper

# Real sacred-texts.com section pages carry a site-wide nav to every other
# archive alongside the books that actually belong to the section.
TRADITION_INDEX_HTML = """
<html><body>
<a href="/eso/book1/index.htm">Book One</a>
<a href="/eso/book2/index.htm">Book Two</a>
<a href="/eso/index.htm">Esoteric Index</a>
<a href="/eso/book1/index.htm">Book One (repeat link)</a>
<a href="/bud/index.htm">Buddhism</a>
<a href="/hin/index.htm">Hinduism</a>
<a href="/afr/index.htm">Africa</a>
<a href="https://example.com/eso/other/index.htm">Offsite</a>
</body></html>
"""

BOOK_INDEX_HTML = """
<html><body>
<a href="chap1.htm">Chapter 1</a>
<a href="chap2.htm">Chapter 2</a>
<a href="chap1.htm">Chapter 1 (repeat link)</a>
<a href="/help/faq.htm">FAQ</a>
<a href="/eso/book2/chap1.htm">Another book's chapter</a>
<a href="https://example.com/eso/book1/chap9.htm">Offsite</a>
</body></html>
"""


def _bare_scraper():
    scraper = SacredTextsScraper.__new__(SacredTextsScraper)
    scraper.BASE_URL = "https://sacred-texts.com"
    return scraper


class FakeResponse:
    def __init__(self, text):
        self.text = text


def test_parse_tradition_index():
    links = _bare_scraper().parse_tradition_index(
        TRADITION_INDEX_HTML, "https://sacred-texts.com/eso/", "/eso/"
    )
    assert links == [
        "https://sacred-texts.com/eso/book1/index.htm",
        "https://sacred-texts.com/eso/book2/index.htm",
    ]


def test_parse_tradition_index_excludes_other_sections():
    """Site nav links to unrelated archives must not be scraped as this tradition."""
    links = _bare_scraper().parse_tradition_index(
        TRADITION_INDEX_HTML, "https://sacred-texts.com/eso/", "/eso/"
    )
    for unrelated in ("/bud/", "/hin/", "/afr/"):
        assert not any(unrelated in link for link in links)


def test_parse_tradition_index_excludes_section_index_itself():
    links = _bare_scraper().parse_tradition_index(
        TRADITION_INDEX_HTML, "https://sacred-texts.com/eso/", "/eso/"
    )
    assert "https://sacred-texts.com/eso/index.htm" not in links


def test_parse_tradition_index_excludes_offsite_links():
    links = _bare_scraper().parse_tradition_index(
        TRADITION_INDEX_HTML, "https://sacred-texts.com/eso/", "/eso/"
    )
    assert all(link.startswith("https://sacred-texts.com/") for link in links)


def test_parse_tradition_index_nested_prefix():
    """A tradition mapped to a sub-path only collects books beneath it."""
    html = """
    <html><body>
    <a href="/eso/enoch/book/index.htm">Enochian Book</a>
    <a href="/eso/other/index.htm">Sibling Esoteric Book</a>
    </body></html>
    """
    links = _bare_scraper().parse_tradition_index(
        html, "https://sacred-texts.com/eso/enoch/", "/eso/enoch/"
    )
    assert links == ["https://sacred-texts.com/eso/enoch/book/index.htm"]


def test_parse_tradition_index_prefix_without_slashes():
    links = _bare_scraper().parse_tradition_index(
        TRADITION_INDEX_HTML, "https://sacred-texts.com/eso/", "eso"
    )
    assert len(links) == 2


def test_parse_book_index():
    links = _bare_scraper().parse_book_index(
        BOOK_INDEX_HTML, "https://sacred-texts.com/eso/book1/index.htm"
    )
    assert links == [
        "https://sacred-texts.com/eso/book1/chap1.htm",
        "https://sacred-texts.com/eso/book1/chap2.htm",
    ]


def test_parse_book_index_excludes_pages_outside_the_book():
    """Nav, footer and sibling-book links are not chapters of this book."""
    links = _bare_scraper().parse_book_index(
        BOOK_INDEX_HTML, "https://sacred-texts.com/eso/book1/index.htm"
    )
    assert all(link.startswith("https://sacred-texts.com/eso/book1/") for link in links)


def test_parse_title_strips_site_prefix_and_trailing_index():
    html = "<html><head><title>Sacred Texts: The Aurora of the Philosophers Index</title></head><body></body></html>"
    assert _bare_scraper().parse_title(html, "fallback") == "The Aurora of the Philosophers"


def test_parse_title_collapses_whitespace():
    html = "<html><head><title>\n  The   Kybalion\n</title></head><body></body></html>"
    assert _bare_scraper().parse_title(html, "fallback") == "The Kybalion"


def test_parse_title_falls_back_to_h1():
    html = "<html><head><title>Index</title></head><body><h1>Corpus Hermeticum</h1></body></html>"
    assert _bare_scraper().parse_title(html, "fallback") == "Corpus Hermeticum"


def test_parse_title_falls_back_to_given_default():
    html = "<html><head></head><body><p>no title anywhere</p></body></html>"
    assert _bare_scraper().parse_title(html, "alc_arr_index.htm") == "alc_arr_index.htm"


def test_scrape_book_records_real_titles(tmp_path, monkeypatch):
    scraper = SacredTextsScraper(output_dir=tmp_path / "raw" / "sacred-texts", traditions=["alchemy"], delay=0)
    pages = {
        "https://sacred-texts.com/alc/arr/index.htm": "<html><head><title>Sacred Texts: A New Light of Alchymie Index</title></head><body>"
                                                      '<a href="chap1.htm">1</a><a href="chap2.htm">2</a></body></html>',
        "https://sacred-texts.com/alc/arr/chap1.htm": "<html><head><title>Chapter I: Of Nature</title></head><body>text</body></html>",
        "https://sacred-texts.com/alc/arr/chap2.htm": "<html><head><title>Chapter II: Of Sulphur</title></head><body>text</body></html>",
    }
    monkeypatch.setattr(scraper, "fetch_url", lambda url: FakeResponse(pages[url]))

    scraper._scrape_book("alchemy/alc_arr_index.htm", "https://sacred-texts.com/alc/arr/index.htm", "alchemy")

    entry = scraper.manifest["items"]["alchemy/alc_arr_index.htm"]
    assert entry["title"] == "A New Light of Alchymie"
    assert entry["tradition"] == "alchemy"
    assert entry["source_url"] == "https://sacred-texts.com/alc/arr/index.htm"
    assert entry["chapter_titles"]["chapter_000"] == "Chapter I: Of Nature"
    assert entry["chapter_titles"]["chapter_001"] == "Chapter II: Of Sulphur"


def test_scraper_skips_already_downloaded(tmp_path):
    scraper = SacredTextsScraper(output_dir=tmp_path / "raw" / "sacred-texts", traditions=["eso"], delay=0)
    scraper.mark_downloaded("eso/book1", {"title": "Book One"})
    assert scraper.is_downloaded("eso/book1") is True


def test_parse_title_keeps_titles_ending_in_a_real_word():
    """Only the site's trailing 'Index' marker is stripped, not real title words."""
    html = "<html><head><title>Sacred Texts: The Book of Contents</title></head><body></body></html>"
    assert _bare_scraper().parse_title(html, "fallback") == "The Book of Contents"

import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from src.scraper.base import BaseScraper
from src.utils import log_error

logger = logging.getLogger(__name__)

TRADITION_PATHS = {
    "hermetic": "/eso/", "kabbalah": "/jud/", "alchemy": "/alc/",
    "grimoire": "/grim/", "enochian": "/eso/enoch/", "rosicrucian": "/sro/",
    "thelema": "/oto/", "gnostic": "/gno/",
}

class SacredTextsScraper(BaseScraper):
    BASE_URL = "https://sacred-texts.com"

    def __init__(self, output_dir: Path, traditions: list[str], delay: float = 1.5):
        super().__init__(name="sacred-texts", output_dir=output_dir, delay=delay)
        self.traditions = traditions

    def parse_tradition_index(self, html: str, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith(("/index.htm", "/index.html")):
                full_url = urljoin(base_url, href)
                if self.BASE_URL in full_url:
                    links.append(full_url)
        return links

    def parse_book_index(self, html: str, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith((".htm", ".html")) and "index" not in href:
                full_url = urljoin(base_url, href)
                if self.BASE_URL in full_url:
                    links.append(full_url)
        return links

    def scrape(self) -> None:
        for tradition in self.traditions:
            path = TRADITION_PATHS.get(tradition)
            if path is None:
                logger.warning(f"Unknown tradition: {tradition}, skipping")
                continue
            self._scrape_tradition(tradition, path)

    def _scrape_tradition(self, tradition: str, path: str) -> None:
        index_url = urljoin(self.BASE_URL, path)
        logger.info(f"Scraping tradition: {tradition} from {index_url}")
        resp = self.fetch_url(index_url)
        if resp is None:
            log_error("scraping", self.name, tradition, ConnectionError(f"Failed to fetch {index_url}"))
            return
        book_urls = self.parse_tradition_index(resp.text, index_url)
        logger.info(f"Found {len(book_urls)} books for {tradition}")
        for book_url in book_urls:
            book_id = self._url_to_id(book_url, tradition)
            if self.is_downloaded(book_id):
                continue
            self._scrape_book(book_id, book_url, tradition)

    def _scrape_book(self, book_id: str, book_url: str, tradition: str) -> None:
        self.rate_limit()
        resp = self.fetch_url(book_url)
        if resp is None:
            self.mark_failed(book_id, f"Failed to fetch {book_url}")
            return
        chapter_urls = self.parse_book_index(resp.text, book_url)
        if not chapter_urls:
            chapter_urls = [book_url]
        book_dir = self.output_dir / book_id
        book_dir.mkdir(parents=True, exist_ok=True)
        for i, chapter_url in enumerate(chapter_urls):
            self.rate_limit()
            chapter_resp = self.fetch_url(chapter_url)
            if chapter_resp is None:
                continue
            chapter_file = book_dir / f"chapter_{i:03d}.html"
            chapter_file.write_text(chapter_resp.text, encoding="utf-8")
        self.mark_downloaded(book_id, {"title": book_id.split("/")[-1], "tradition": tradition, "source_url": book_url, "chapters": len(chapter_urls), "file_type": "html"})

    def _url_to_id(self, url: str, tradition: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_")
        return f"{tradition}/{path}"

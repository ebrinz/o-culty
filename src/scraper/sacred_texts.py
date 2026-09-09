import logging
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm
from src.scraper.base import BaseScraper
from src.utils import log_error

logger = logging.getLogger(__name__)

TRADITION_PATHS = {
    "hermetic": "/eso/", "kabbalah": "/jud/", "alchemy": "/alc/",
    "grimoire": "/grim/", "enochian": "/eso/enoch/", "rosicrucian": "/sro/",
    "thelema": "/oto/", "gnostic": "/gno/",
}

INDEX_SUFFIXES = ("/index.htm", "/index.html")

# sacred-texts.com titles look like "Sacred Texts: Alchemy" or "The Aurora of
# the Philosophers Index" — strip the site prefix and the trailing "Index".
SITE_PREFIX_RE = re.compile(r"^\s*sacred[\s-]*texts\s*[:|>-]+\s*", re.I)
TRAILING_INDEX_RE = re.compile(r"[\s:|-]*index\s*$", re.I)

class SacredTextsScraper(BaseScraper):
    BASE_URL = "https://sacred-texts.com"

    def __init__(self, output_dir: Path, traditions: list[str], delay: float = 1.5):
        super().__init__(name="sacred-texts", output_dir=output_dir, delay=delay)
        self.traditions = traditions

    @staticmethod
    def _normalize_prefix(path: str) -> str:
        stripped = path.strip("/")
        return f"/{stripped}/" if stripped else "/"

    def _same_site(self, url: str) -> bool:
        return urlparse(url).netloc == urlparse(self.BASE_URL).netloc

    def parse_tradition_index(self, html: str, base_url: str, path_prefix: str) -> list[str]:
        """Book index links living under path_prefix.

        Every sacred-texts.com section page carries a site-wide nav listing all
        the other sections, so an unfiltered scan pulls in unrelated archives
        (/bud/, /hin/, /afr/) and tags them with whatever tradition is being
        crawled. Restricting to the tradition's own path prefix keeps only the
        books that actually belong to it.
        """
        prefix = self._normalize_prefix(path_prefix)
        soup = BeautifulSoup(html, "html.parser")
        links = []
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.endswith(INDEX_SUFFIXES):
                continue
            full_url = urljoin(base_url, href)
            if not self._same_site(full_url):
                continue
            path = urlparse(full_url).path
            if not path.startswith(prefix):
                continue
            # the section's own index page, not a book under it
            if path in (f"{prefix}index.htm", f"{prefix}index.html"):
                continue
            if full_url in seen:
                continue
            seen.add(full_url)
            links.append(full_url)
        return links

    def parse_book_index(self, html: str, base_url: str) -> list[str]:
        """Chapter links belonging to the book whose index page this is.

        Book pages carry the same site-wide nav as section pages, so chapters
        are limited to the book's own directory to keep unrelated pages out.
        """
        book_prefix = base_url.rsplit("/", 1)[0] + "/"
        book_path = urlparse(book_prefix).path
        soup = BeautifulSoup(html, "html.parser")
        links = []
        seen = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.endswith((".htm", ".html")) or "index" in href:
                continue
            full_url = urljoin(base_url, href)
            if not self._same_site(full_url):
                continue
            if not urlparse(full_url).path.startswith(book_path):
                continue
            if full_url in seen:
                continue
            seen.add(full_url)
            links.append(full_url)
        return links

    def parse_title(self, html: str, fallback: str) -> str:
        """Human-readable title from a page's <title>, falling back to its <h1>."""
        soup = BeautifulSoup(html, "html.parser")
        candidates = []
        if soup.title is not None:
            candidates.append(soup.title.get_text())
        h1 = soup.find("h1")
        if h1 is not None:
            candidates.append(h1.get_text())
        for raw in candidates:
            title = re.sub(r"\s+", " ", raw).strip()
            title = SITE_PREFIX_RE.sub("", title)
            title = TRAILING_INDEX_RE.sub("", title).strip(" :|-")
            if title:
                return title
        return fallback

    def scrape(self) -> None:
        for tradition in tqdm(self.traditions, desc="sacred-texts traditions"):
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
        book_urls = self.parse_tradition_index(resp.text, index_url, path)
        logger.info(f"Found {len(book_urls)} books under {path} for {tradition}")
        for book_url in tqdm(book_urls, desc=f"  {tradition} books", leave=False):
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
        book_title = self.parse_title(resp.text, book_id.split("/")[-1])
        chapter_urls = self.parse_book_index(resp.text, book_url)
        if not chapter_urls:
            chapter_urls = [book_url]
        book_dir = self.output_dir / book_id
        book_dir.mkdir(parents=True, exist_ok=True)
        chapter_titles = {}
        for i, chapter_url in enumerate(chapter_urls):
            self.rate_limit()
            chapter_resp = self.fetch_url(chapter_url)
            if chapter_resp is None:
                continue
            chapter_name = f"chapter_{i:03d}"
            (book_dir / f"{chapter_name}.html").write_text(chapter_resp.text, encoding="utf-8")
            chapter_titles[chapter_name] = self.parse_title(chapter_resp.text, book_title)
        self.mark_downloaded(book_id, {"title": book_title, "tradition": tradition, "source_url": book_url, "chapters": len(chapter_urls), "chapter_titles": chapter_titles, "file_type": "html"})
        logger.info(f"Downloaded: {book_id} - {book_title} ({len(chapter_urls)} chapters)")

    def _url_to_id(self, url: str, tradition: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_")
        return f"{tradition}/{path}"

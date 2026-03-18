import logging
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm import tqdm
from src.scraper.base import BaseScraper
from src.utils import log_error

logger = logging.getLogger(__name__)

class GutenbergScraper(BaseScraper):
    BASE_URL = "https://www.gutenberg.org"
    SEARCH_URL = "https://www.gutenberg.org/ebooks/search/?query={query}&submit_search=Go%21"

    def __init__(self, output_dir: Path, search_terms: list[str], delay: float = 1.5):
        super().__init__(name="gutenberg", output_dir=output_dir, delay=delay)
        self.search_terms = search_terms

    def parse_search_results(self, html: str) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select("li.booklink"):
            link = item.select_one("a.link")
            if link and link.get("href"):
                href = link["href"]
                ebook_id = href.strip("/").split("/")[-1]
                title = link.get_text(strip=True)
                author_span = item.select_one("span.subtitle")
                author = author_span.get_text(strip=True) if author_span else "unknown"
                results.append({"ebook_id": ebook_id, "title": title, "author": author})
        return results

    def scrape(self) -> None:
        for term in tqdm(self.search_terms, desc="gutenberg terms"):
            self._scrape_term(term)

    def _scrape_term(self, term: str) -> None:
        url = self.SEARCH_URL.format(query=term)
        logger.info(f"Searching Gutenberg: {term}")
        self.rate_limit()
        resp = self.fetch_url(url)
        if resp is None:
            log_error("scraping", self.name, term, ConnectionError(f"Failed search: {term}"))
            return
        results = self.parse_search_results(resp.text)
        logger.info(f"Found {len(results)} results for '{term}'")
        for result in tqdm(results, desc=f"  '{term}' books", leave=False):
            ebook_id = result["ebook_id"]
            if self.is_downloaded(ebook_id):
                continue
            self._download_text(ebook_id, result)

    def _download_text(self, ebook_id: str, metadata: dict) -> None:
        txt_url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
        self.rate_limit()
        resp = self.fetch_url(txt_url)
        if resp is None:
            self.mark_failed(ebook_id, f"Failed to download text for {ebook_id}")
            return
        out_file = self.output_dir / f"{ebook_id}.txt"
        out_file.write_text(resp.text, encoding="utf-8")
        self.mark_downloaded(ebook_id, {"title": metadata.get("title", ebook_id), "author": metadata.get("author", "unknown"), "source_url": f"https://www.gutenberg.org/ebooks/{ebook_id}", "file_type": "txt"})
        logger.info(f"Downloaded: {ebook_id} - {metadata.get('title', '')}")

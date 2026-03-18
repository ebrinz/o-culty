import logging
from pathlib import Path
import internetarchive as ia
from src.scraper.base import BaseScraper
from src.utils import log_error

logger = logging.getLogger(__name__)

class InternetArchiveScraper(BaseScraper):
    def __init__(self, output_dir: Path, search_terms: list[str], max_results_per_term: int = 50, formats: list[str] | None = None, delay: float = 1.5):
        super().__init__(name="internet-archive", output_dir=output_dir, delay=delay)
        self.search_terms = search_terms
        self.max_results_per_term = max_results_per_term
        self.formats = formats or ["pdf", "txt"]

    def build_search_query(self, term: str) -> str:
        return f"{term} AND mediatype:texts"

    def scrape(self) -> None:
        for term in self.search_terms:
            self._scrape_term(term)

    def _scrape_term(self, term: str) -> None:
        query = self.build_search_query(term)
        logger.info(f"Searching Internet Archive: {query}")
        try:
            results = ia.search_items(query)
        except Exception as e:
            log_error("scraping", self.name, term, e)
            return
        count = 0
        for result in results:
            if count >= self.max_results_per_term:
                break
            item_id = result["identifier"]
            if self.is_downloaded(item_id):
                count += 1
                continue
            self._download_item(item_id)
            count += 1

    def _download_item(self, item_id: str) -> None:
        self.rate_limit()
        try:
            item = ia.get_item(item_id)
            item_dir = self.output_dir / item_id
            item_dir.mkdir(parents=True, exist_ok=True)
            downloaded_files = []
            for file_info in item.files:
                name = file_info.get("name", "")
                ext = Path(name).suffix.lower().lstrip(".")
                fmt = file_info.get("format", "").lower()
                if ext in self.formats or any(f in fmt for f in self.formats):
                    target = item_dir / name
                    if not target.exists():
                        item.download(files=[name], destdir=str(self.output_dir), no_directory=False, verbose=False)
                        downloaded_files.append(name)
            if downloaded_files:
                metadata = item.metadata or {}
                self.mark_downloaded(item_id, {"title": metadata.get("title", item_id), "author": metadata.get("creator", "unknown"), "source_url": f"https://archive.org/details/{item_id}", "files": downloaded_files, "file_type": "mixed"})
            else:
                self.mark_failed(item_id, "No matching files found")
        except Exception as e:
            self.mark_failed(item_id, str(e))
            log_error("scraping", self.name, item_id, e)

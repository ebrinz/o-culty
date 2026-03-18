import json
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils import log_error

logger = logging.getLogger(__name__)

class BaseScraper(ABC):
    def __init__(self, name: str, output_dir: Path, delay: float = 1.5):
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest = self._load_manifest()
        self.session = self._build_session()

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"items": {}}

    def save_manifest(self) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": "o-culty/0.1 (occult research project)"})
        return session

    def is_downloaded(self, item_id: str) -> bool:
        item = self.manifest["items"].get(item_id)
        return item is not None and item.get("status") == "downloaded"

    def mark_downloaded(self, item_id: str, metadata: dict) -> None:
        self.manifest["items"][item_id] = {**metadata, "status": "downloaded", "download_date": datetime.now(timezone.utc).isoformat()}
        self.save_manifest()

    def mark_failed(self, item_id: str, error: str) -> None:
        self.manifest["items"][item_id] = {"status": "failed", "error": error, "failed_date": datetime.now(timezone.utc).isoformat()}
        self.save_manifest()

    def rate_limit(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    def fetch_url(self, url: str) -> requests.Response | None:
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    @abstractmethod
    def scrape(self) -> None: ...

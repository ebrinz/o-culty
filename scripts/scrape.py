import argparse
import json
import logging
from pathlib import Path
from src.utils import load_config
from src.scraper.sacred_texts import SacredTextsScraper
from src.scraper.internet_archive import InternetArchiveScraper
from src.scraper.gutenberg import GutenbergScraper

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s: %(message)s")


def print_summary(data_dir: Path):
    """Print a summary of what's been downloaded across all sources."""
    total_downloaded = 0
    total_failed = 0
    for manifest_path in data_dir.rglob("manifest.json"):
        source = manifest_path.parent.name
        manifest = json.loads(manifest_path.read_text())
        items = manifest.get("items", {})
        downloaded = sum(1 for v in items.values() if v.get("status") == "downloaded")
        failed = sum(1 for v in items.values() if v.get("status") == "failed")
        total_downloaded += downloaded
        total_failed += failed
        print(f"  {source}: {downloaded} downloaded, {failed} failed")
    print(f"\n  Total: {total_downloaded} downloaded, {total_failed} failed")


def main():
    parser = argparse.ArgumentParser(description="Scrape occult primary sources")
    parser.add_argument("--source", choices=["sacred-texts", "internet-archive", "gutenberg", "all"], default="all")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--status", action="store_true", help="Just print download status, don't scrape")
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = Path("data/raw")

    if args.status:
        print("\nScrape status:")
        print_summary(data_dir)
        return

    scrapers = []
    if args.source in ("sacred-texts", "all"):
        cfg = config["scraping"]["sacred_texts"]
        if cfg.get("enabled", True):
            scrapers.append(SacredTextsScraper(output_dir=data_dir / "sacred-texts", traditions=cfg["traditions"], delay=cfg.get("delay_seconds", 1.5)))
    if args.source in ("internet-archive", "all"):
        cfg = config["scraping"]["internet_archive"]
        if cfg.get("enabled", True):
            scrapers.append(InternetArchiveScraper(output_dir=data_dir / "internet-archive", search_terms=cfg["search_terms"], max_results_per_term=cfg.get("max_results_per_term", 50), formats=cfg.get("formats", ["pdf", "txt"]), delay=1.5))
    if args.source in ("gutenberg", "all"):
        cfg = config["scraping"]["gutenberg"]
        if cfg.get("enabled", True):
            scrapers.append(GutenbergScraper(output_dir=data_dir / "gutenberg", search_terms=cfg["search_terms"], delay=1.5))
    for scraper in scrapers:
        print(f"\n>>> {scraper.name}")
        scraper.scrape()

    print("\n--- Summary ---")
    print_summary(data_dir)


if __name__ == "__main__":
    main()

import argparse
import logging
from pathlib import Path
from src.utils import load_config
from src.scraper.sacred_texts import SacredTextsScraper
from src.scraper.internet_archive import InternetArchiveScraper
from src.scraper.gutenberg import GutenbergScraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Scrape occult primary sources")
    parser.add_argument("--source", choices=["sacred-texts", "internet-archive", "gutenberg", "all"], default="all")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = Path("data/raw")
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
        logging.info(f"Running scraper: {scraper.name}")
        scraper.scrape()
        logging.info(f"Completed: {scraper.name}")

if __name__ == "__main__":
    main()

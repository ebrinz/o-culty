"""CLI entry point for processing raw scraped files into clean text."""
import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm

from src.utils import load_config, log_error
from src.processor.html_extractor import extract_text_from_html
from src.processor.pdf_extractor import extract_text_from_pdf, is_scanned_pdf
from src.processor.ocr import ocr_pdf
from src.processor.cleaner import normalize_text, detect_language, is_duplicate, normalize_title, is_garbled

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_file(path: Path, config: dict, no_ocr: bool = False) -> dict | None:
    suffix = path.suffix.lower()
    qg = config.get("processing", {}).get("quality_gate", {})
    min_len = qg.get("min_text_length", 200)
    max_garble = qg.get("max_garbled_ratio", 0.10)

    try:
        if suffix in (".html", ".htm"):
            result = extract_text_from_html(path.read_text(encoding="utf-8", errors="replace"))
            text = result["text"]
            chapters = result["chapters"]
            ocr_used = False
        elif suffix == ".pdf":
            result = extract_text_from_pdf(path)
            text = result["text"]
            chapters = []

            # Quality gate: if text-based extraction looks garbled or empty, try OCR
            if not no_ocr and (result["is_scanned"] or is_garbled(text, max_garble)):
                languages = config.get("processing", {}).get("ocr_languages", ["en", "la"])
                ocr_result = ocr_pdf(path, languages)
                # Use OCR if it produced more usable text
                if len(ocr_result["text"].strip()) > len(text.strip()):
                    text = ocr_result["text"]
                ocr_used = True
            else:
                ocr_used = False
        elif suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace")
            chapters = []
            ocr_used = False
        else:
            return None

        text = normalize_text(text)

        # Quality gate: skip if too short or still garbled after OCR
        if len(text.strip()) < min_len:
            return None
        if is_garbled(text, max_garble):
            log_error("processing", str(path.parent.name), str(path.name),
                      ValueError(f"Text still garbled after extraction (ratio > {max_garble})"))
            return None

        language = detect_language(text[:1000])
        return {"text": text, "chapters": chapters, "language": language, "ocr_used": ocr_used}
    except Exception as e:
        log_error("processing", str(path.parent.name), str(path.name), e)
        return None


def build_metadata(file_path: Path, source_name: str, manifest_entry: dict, processing_result: dict) -> dict:
    text_id = f"{source_name}_{file_path.stem}"
    return {
        "id": text_id,
        "title": manifest_entry.get("title", file_path.stem),
        "author": manifest_entry.get("author", "unknown"),
        "tradition": manifest_entry.get("tradition", "unknown"),
        "source": source_name,
        "source_url": manifest_entry.get("source_url", ""),
        "file_type": file_path.suffix.lstrip("."),
        "ocr_used": processing_result["ocr_used"],
        "language": processing_result["language"],
        "chapters": processing_result["chapters"],
        "char_count": len(processing_result["text"]),
    }


def content_hash(text: str) -> str:
    """Hash first 2000 chars of normalized text for dedup."""
    import hashlib
    sample = text.strip()[:2000].lower()
    return hashlib.sha256(sample.encode()).hexdigest()


def dedup_check(text_hash: str, existing_hashes: dict[str, str]) -> str | None:
    """Check if content hash already exists."""
    for existing_id, existing_hash in existing_hashes.items():
        if text_hash == existing_hash:
            return existing_id
    return None


def main():
    parser = argparse.ArgumentParser(description="Process raw files into clean text")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--source", default="all")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR for scanned PDFs (fast text-only pass)")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF files entirely (process only txt/html)")
    args = parser.parse_args()
    config = load_config(args.config)
    raw_dir = Path("data/raw")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes: dict[str, str] = {}

    sources = [d for d in sorted(raw_dir.iterdir()) if d.is_dir()] if args.source == "all" else [raw_dir / args.source]

    stats = {"processed": 0, "skipped_quality": 0, "skipped_dup": 0, "errors": 0}

    for source_dir in sources:
        if not source_dir.exists():
            continue
        source_name = source_dir.name
        manifest_path = source_dir / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text()).get("items", {})

        allowed_ext = {".html", ".htm", ".txt"} if args.skip_pdf else {".html", ".htm", ".pdf", ".txt"}
        files = sorted(
            f for f in source_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in allowed_ext and f.name != "manifest.json"
        )

        for file_path in tqdm(files, desc=f"Processing {source_name}"):
            # Build a unique text_id using relative path to avoid collisions
            rel_parts = file_path.relative_to(source_dir).with_suffix("").parts
            text_id = f"{source_name}_{'_'.join(rel_parts)}"
            # Skip if already processed
            if (out_dir / f"{text_id}.txt").exists():
                continue

            rel_key = str(file_path.relative_to(source_dir)).split("/")[0]
            # Try with and without extension for manifest lookup
            manifest_entry = manifest.get(rel_key, {}) or manifest.get(Path(rel_key).stem, {})
            title = manifest_entry.get("title", file_path.stem)

            result = process_file(file_path, config, no_ocr=args.no_ocr)
            if result is None:
                stats["skipped_quality"] += 1
                continue

            # Dedup by content hash (catches same text from different sources)
            h = content_hash(result["text"])
            dup_of = dedup_check(h, seen_hashes)
            if dup_of is not None:
                stats["skipped_dup"] += 1
                continue

            metadata = build_metadata(file_path, source_name, manifest_entry, result)
            metadata["id"] = text_id
            seen_hashes[text_id] = h

            text_id = metadata["id"]
            text_file = out_dir / f"{text_id}.txt"
            meta_file = out_dir / f"{text_id}.json"
            text_file.write_text(result["text"], encoding="utf-8")
            meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            stats["processed"] += 1

    print(f"\n--- Processing Summary ---")
    print(f"  Processed: {stats['processed']}")
    print(f"  Skipped (quality): {stats['skipped_quality']}")
    print(f"  Skipped (duplicate): {stats['skipped_dup']}")


if __name__ == "__main__":
    main()

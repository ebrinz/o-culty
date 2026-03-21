"""Export processed txt+json pairs into a Parquet dataset for Hugging Face.

Off-topic documents are moved to data/misc/ instead of being included in the corpus.
Duplicate documents (by content hash) are deduplicated, keeping the longest version.
"""
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Title substrings that indicate off-topic content (matched case-insensitively)
OFF_TOPIC = [
    "global positioning system",
    "gps pseudolites",
    "zero taxes",
    "chiesa viva",
    "secrets of nasa",
    "secrets of nassa",
    "chernobyl",
    "nuclear disaster hoax",
    "americanisation of the world",
    "ringmakers of saturn",
    "new testament in indonesian",
    "new testament in urdu",
    "alkitab",
    "perjanjian baru",
    "hindustani",
    "denis fahey collection",
    "016 cd r&",
    "synarchy movement of empire",
    "australian personal computer",
    "ijelfeb",
    "ijasrjun",
]


def _str(val, default=""):
    """Coerce value to string — join lists, stringify others."""
    if isinstance(val, list):
        return "; ".join(str(v) for v in val)
    if val is None:
        return default
    return str(val)


def is_off_topic(title: str) -> bool:
    tl = title.lower()
    return any(kw in tl for kw in OFF_TOPIC)


def content_hash(text: str) -> str:
    """Hash first 5000 chars of text for dedup."""
    return hashlib.sha256(text.strip()[:5000].encode()).hexdigest()


def dedup_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Deduplicate by content hash, keeping the longest version."""
    seen: dict[str, int] = {}  # hash -> index in deduped list
    deduped = []
    dup_count = 0

    for row in rows:
        h = content_hash(row["text"])
        if h in seen:
            # Keep the longer version
            existing_idx = seen[h]
            if row["char_count"] > deduped[existing_idx]["char_count"]:
                deduped[existing_idx] = row
            dup_count += 1
        else:
            seen[h] = len(deduped)
            deduped.append(row)

    return deduped, dup_count


def main():
    processed_dir = Path("data/processed")
    misc_dir = Path("data/misc")
    misc_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path("data/corpus.parquet")

    json_files = sorted(processed_dir.glob("*.json"))
    print(f"Found {len(json_files)} metadata files")

    rows = []
    misc_count = 0
    for meta_path in tqdm(json_files, desc="Building corpus"):
        txt_path = meta_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        title = _str(meta.get("title", ""))

        if is_off_topic(title):
            shutil.move(str(meta_path), str(misc_dir / meta_path.name))
            shutil.move(str(txt_path), str(misc_dir / txt_path.name))
            misc_count += 1
            continue

        text = txt_path.read_text(encoding="utf-8")

        rows.append({
            "id": _str(meta.get("id", meta_path.stem)),
            "text": text,
            "title": title,
            "author": _str(meta.get("author"), "unknown"),
            "tradition": _str(meta.get("tradition"), "unknown"),
            "source": _str(meta.get("source", "")),
            "source_url": _str(meta.get("source_url", "")),
            "language": _str(meta.get("language", "en")),
            "file_type": _str(meta.get("file_type", "")),
            "ocr_used": bool(meta.get("ocr_used", False)),
            "char_count": int(meta.get("char_count", len(text))),
        })

    rows, dup_count = dedup_rows(rows)
    print(f"Deduplicated: removed {dup_count} duplicates")

    schema = pa.schema([
        ("id", pa.string()),
        ("text", pa.large_string()),
        ("title", pa.string()),
        ("author", pa.string()),
        ("tradition", pa.string()),
        ("source", pa.string()),
        ("source_url", pa.string()),
        ("language", pa.string()),
        ("file_type", pa.string()),
        ("ocr_used", pa.bool_()),
        ("char_count", pa.int64()),
    ])

    table = pa.table({col: [r[col] for r in rows] for col in schema.names}, schema=schema)
    pq.write_table(table, out_path, compression="zstd")

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {len(rows)} rows to {out_path} ({file_size_mb:.1f} MB)")
    print(f"Moved {misc_count} off-topic files to {misc_dir}/")

    # Write manifest summary
    sources = {}
    langs = {}
    for r in rows:
        sources[r["source"]] = sources.get(r["source"], 0) + 1
        langs[r["language"]] = langs.get(r["language"], 0) + 1

    manifest = {
        "total_documents": len(rows),
        "total_chars": sum(r["char_count"] for r in rows),
        "file_size_mb": round(file_size_mb, 1),
        "compression": "zstd",
        "sources": dict(sorted(sources.items(), key=lambda x: -x[1])),
        "languages": dict(sorted(langs.items(), key=lambda x: -x[1])),
        "columns": list(schema.names),
        "duplicates_removed": dup_count,
        "off_topic_moved_to_misc": misc_count,
    }
    manifest_path = Path("data/manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()

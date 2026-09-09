"""List what the corpus contains, and what repos the HuggingFace account can see.

Two subcommands:

    repos   what dataset/model/space repos the authenticated account has access to
    texts   the individual documents inside a corpus parquet (hub or local)

The corpus `text` column holds ~1.45B characters, so `texts` reads the parquet
column-pruned and never touches it — listing titles costs megabytes, not gigabytes.
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_REPO = "ebrinz/text-cult"
DEFAULT_LOCAL = Path("data/corpus.parquet")

# Everything except `text` — see module docstring.
LIST_COLUMNS = [
    "id", "title", "author", "tradition", "source",
    "source_url", "language", "file_type", "text_format", "ocr_used", "char_count",
]
DISPLAY_COLUMNS = ["title", "author", "tradition", "source", "language", "char_count"]


def _require_hub():
    """Import huggingface_hub with an actionable message if it is missing."""
    try:
        from huggingface_hub import HfApi, HfFileSystem
    except ImportError:
        sys.exit("huggingface_hub is not installed. Run: pip install -e .")
    return HfApi, HfFileSystem


def truncate(value, width: int) -> str:
    """Trim to width, marking elision with a single-character ellipsis."""
    text = "" if value is None else str(value)
    if width <= 0 or len(text) <= width:
        return text
    return text[: width - 1] + "…"


def human_count(n: int) -> str:
    """Format a character count as 1.4B / 85.7M / 12.3K."""
    for limit, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if n >= limit:
            return f"{n / limit:.1f}{suffix}"
    return str(n)


def matches(row: dict, field: str, needle: str) -> bool:
    """Case-insensitive substring match on a row field."""
    return needle.lower() in str(row.get(field, "")).lower()


def filter_rows(rows: list[dict], args) -> list[dict]:
    """Apply the --tradition/--source/--author/--language/--title/--min-chars filters."""
    out = rows
    for field, needle in (
        ("tradition", args.tradition),
        ("source", args.source),
        ("author", args.author),
        ("language", args.language),
        ("title", args.title),
    ):
        if needle:
            out = [r for r in out if matches(r, field, needle)]
    if args.min_chars:
        out = [r for r in out if int(r.get("char_count", 0)) >= args.min_chars]
    return out


def sort_rows(rows: list[dict], key: str, descending: bool) -> list[dict]:
    """Sort by a column, coercing char_count numerically and text case-insensitively."""
    if key == "char_count":
        return sorted(rows, key=lambda r: int(r.get("char_count", 0)), reverse=descending)
    return sorted(rows, key=lambda r: str(r.get(key, "")).lower(), reverse=descending)


def group_counts(rows: list[dict], field: str) -> list[tuple[str, int, int]]:
    """Return (value, document count, total chars) per distinct value, largest first."""
    groups: dict[str, list[int]] = {}
    for r in rows:
        bucket = groups.setdefault(str(r.get(field, "")) or "unknown", [0, 0])
        bucket[0] += 1
        bucket[1] += int(r.get("char_count", 0))
    ordered = sorted(groups.items(), key=lambda kv: -kv[1][0])
    return [(name, docs, chars) for name, (docs, chars) in ordered]


def read_parquet_columns(source, columns: list[str], filesystem=None) -> list[dict]:
    """Read only `columns` from a parquet file, skipping any the schema lacks."""
    schema = pq.ParquetFile(source, filesystem=filesystem).schema_arrow
    present = [c for c in columns if c in schema.names]
    missing = [c for c in columns if c not in schema.names]
    if missing:
        logger.warning("parquet is missing expected columns: %s", ", ".join(missing))
    table = pq.read_table(source, columns=present, filesystem=filesystem)
    return table.to_pylist()


def load_local(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(
            f"No parquet at {path}.\n"
            f"Build it with `python scripts/export_parquet.py`, or read the hub copy "
            f"by dropping --local."
        )
    return read_parquet_columns(str(path), LIST_COLUMNS)


def load_hub(repo: str, token: str | None) -> list[dict]:
    """Read every parquet file in a dataset repo, column-pruned, over HfFileSystem."""
    HfApi, HfFileSystem = _require_hub()
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
    except Exception as e:
        sys.exit(
            f"Could not read dataset '{repo}': {type(e).__name__}: {e}\n"
            f"If it is private, log in first: huggingface-cli login"
        )
    parquets = sorted(f for f in files if f.endswith(".parquet"))
    if not parquets:
        sys.exit(f"Dataset '{repo}' contains no .parquet files (found {len(files)} files).")

    fs = HfFileSystem(token=token)
    rows: list[dict] = []
    for name in parquets:
        logger.info("reading %s", name)
        rows.extend(read_parquet_columns(f"datasets/{repo}/{name}", LIST_COLUMNS, filesystem=fs))
    return rows


def write_rows(rows: list[dict], fmt: str, columns: list[str], stream) -> None:
    """Emit rows as an aligned table, csv/tsv, or json/jsonl, limited to `columns`."""
    if fmt in ("json", "jsonl"):
        projected = [{c: r.get(c) for c in columns} for r in rows]
        if fmt == "json":
            json.dump(projected, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            return
        for r in projected:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
        return
    if fmt in ("csv", "tsv"):
        writer = csv.DictWriter(
            stream, fieldnames=columns, extrasaction="ignore",
            delimiter="\t" if fmt == "tsv" else ",", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        return

    # Aligned table: cap the widest text columns so long titles cannot wrap the terminal.
    caps = {"title": 60, "author": 28, "source_url": 50}
    widths = {}
    for col in columns:
        cell_width = max((len(truncate(r.get(col), caps.get(col, 0) or 10**6)) for r in rows), default=0)
        widths[col] = max(len(col), cell_width)
    stream.write("  ".join(c.ljust(widths[c]) for c in columns).rstrip() + "\n")
    stream.write("  ".join("-" * widths[c] for c in columns) + "\n")
    for r in rows:
        cells = [truncate(r.get(c), caps.get(c, 0) or 10**6).ljust(widths[c]) for c in columns]
        stream.write("  ".join(cells).rstrip() + "\n")


def cmd_repos(args):
    """List the repos the authenticated account can see, private ones included."""
    HfApi, _ = _require_hub()
    api = HfApi(token=args.token)

    author = args.author
    try:
        me = api.whoami()
        print(f"Authenticated as: {me.get('name', '?')} ({me.get('type', '?')})")
        author = author or me.get("name")
    except Exception:
        print("Not authenticated — showing public repos only.")
        print("Log in with `huggingface-cli login` to include private repos.\n")
        if not author:
            sys.exit("No token available, so --author is required to know whose repos to list.")

    listers = [
        ("datasets", api.list_datasets),
        ("models", api.list_models),
        ("spaces", api.list_spaces),
    ]
    for label, lister in listers:
        try:
            items = list(lister(author=author))
        except Exception as e:
            logger.warning("could not list %s: %s", label, e)
            continue
        print(f"\n{label} ({len(items)}) for {author}:")
        if not items:
            print("  (none)")
            continue
        for item in sorted(items, key=lambda i: i.id):
            visibility = "private" if getattr(item, "private", False) else "public"
            modified = getattr(item, "last_modified", None)
            stamp = modified.strftime("%Y-%m-%d") if modified else "?"
            downloads = getattr(item, "downloads", None)
            extra = f", {downloads} downloads" if downloads else ""
            print(f"  {item.id}  [{visibility}, updated {stamp}{extra}]")


def cmd_texts(args):
    """List the documents in the corpus, filtered and sorted per the flags."""
    rows = load_local(args.local) if args.local else load_hub(args.repo, args.token)
    total = len(rows)

    rows = filter_rows(rows, args)
    rows = sort_rows(rows, args.sort, args.desc)
    shown = len(rows)

    if args.stats:
        for field in ("source", "tradition", "language"):
            counts = group_counts(rows, field)
            print(f"\n{field} ({len(counts)} distinct):")
            width = max((len(name) for name, _, _ in counts), default=0)
            for name, docs, chars in counts:
                print(f"  {name.ljust(width)}  {docs:>6} docs  {human_count(chars):>8} chars")
        print(f"\nTotal: {shown} documents, {human_count(sum(int(r.get('char_count', 0)) for r in rows))} chars")
        return

    if args.limit:
        rows = rows[: args.limit]

    columns = args.columns.split(",") if args.columns else DISPLAY_COLUMNS
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            write_rows(rows, args.format, columns, f)
        print(f"Wrote {len(rows)} rows to {args.out}")
    else:
        write_rows(rows, args.format, columns, sys.stdout)

    if args.format == "table":
        suffix = f" (of {total} in corpus)" if shown != total else ""
        note = f", showing first {len(rows)}" if args.limit and len(rows) < shown else ""
        print(f"\n{shown} documents matched{suffix}{note}.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="List HuggingFace repos and corpus documents")
    parser.add_argument("--token", default=None, help="HF token (defaults to HF_TOKEN or a cached login)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_repos = sub.add_parser("repos", help="list repos the account can access")
    p_repos.add_argument("--author", default=None, help="account to list (defaults to the logged-in user)")
    p_repos.set_defaults(func=cmd_repos)

    p_texts = sub.add_parser("texts", help="list documents in the corpus")
    p_texts.add_argument("--repo", default=DEFAULT_REPO, help=f"dataset repo to read (default: {DEFAULT_REPO})")
    p_texts.add_argument("--local", type=Path, nargs="?", const=DEFAULT_LOCAL, default=None,
                         help=f"read a local parquet instead of the hub (default: {DEFAULT_LOCAL})")
    p_texts.add_argument("--tradition", help="filter by tradition substring")
    p_texts.add_argument("--source", help="filter by source substring")
    p_texts.add_argument("--author", help="filter by author substring")
    p_texts.add_argument("--language", help="filter by language substring")
    p_texts.add_argument("--title", help="filter by title substring")
    p_texts.add_argument("--min-chars", type=int, default=0, help="only documents at least this long")
    p_texts.add_argument("--sort", default="title", choices=LIST_COLUMNS, help="sort column (default: title)")
    p_texts.add_argument("--desc", action="store_true", help="sort descending")
    p_texts.add_argument("--limit", type=int, default=0, help="show at most N documents (0 = all)")
    p_texts.add_argument("--columns", help=f"comma-separated columns (default: {','.join(DISPLAY_COLUMNS)})")
    p_texts.add_argument("--format", default="table", choices=["table", "csv", "tsv", "json", "jsonl"])
    p_texts.add_argument("--out", type=Path, help="write to a file instead of stdout")
    p_texts.add_argument("--stats", action="store_true", help="print grouped counts instead of a document list")
    p_texts.set_defaults(func=cmd_texts)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

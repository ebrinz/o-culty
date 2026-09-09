import io
import json
from argparse import Namespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.list_hf import (
    LIST_COLUMNS,
    filter_rows,
    group_counts,
    human_count,
    matches,
    read_parquet_columns,
    sort_rows,
    truncate,
    write_rows,
)

ROWS = [
    {"id": "a", "title": "The Kybalion", "author": "Three Initiates", "tradition": "hermetic",
     "source": "sacred-texts", "language": "en", "char_count": 240000},
    {"id": "b", "title": "Sepher Yetzirah", "author": "Wescott", "tradition": "kabbalah",
     "source": "internet-archive", "language": "en", "char_count": 45000},
    {"id": "c", "title": "Transcendental Magic", "author": "Eliphas Levi", "tradition": "occult-general",
     "source": "gutenberg", "language": "fr", "char_count": 900000},
]


def no_filters(**overrides):
    base = dict(tradition=None, source=None, author=None, language=None, title=None, min_chars=0)
    base.update(overrides)
    return Namespace(**base)


def test_truncate_marks_elision():
    assert truncate("abcdefghij", 5) == "abcd…"
    assert truncate("abc", 10) == "abc"
    assert truncate(None, 5) == ""


def test_human_count_scales():
    assert human_count(1_450_000_000) == "1.4B"
    assert human_count(85_700_000) == "85.7M"
    assert human_count(999) == "999"


def test_matches_is_case_insensitive():
    assert matches(ROWS[0], "title", "kybalion")
    assert not matches(ROWS[0], "title", "grimoire")


def test_filter_by_tradition_substring():
    assert [r["id"] for r in filter_rows(ROWS, no_filters(tradition="herm"))] == ["a"]


def test_filter_by_min_chars():
    assert [r["id"] for r in filter_rows(ROWS, no_filters(min_chars=100000))] == ["a", "c"]


def test_filters_combine():
    assert filter_rows(ROWS, no_filters(language="en", min_chars=500000)) == []


def test_sort_by_char_count_descending():
    assert [r["id"] for r in sort_rows(ROWS, "char_count", True)] == ["c", "a", "b"]


def test_sort_by_title_is_case_insensitive():
    assert [r["id"] for r in sort_rows(ROWS, "title", False)] == ["b", "a", "c"]


def test_group_counts_orders_by_document_count():
    counts = group_counts(ROWS + [dict(ROWS[1], id="d")], "source")
    assert counts[0] == ("internet-archive", 2, 90000)


def test_group_counts_labels_blanks_unknown():
    assert group_counts([{"tradition": "", "char_count": 1}], "tradition") == [("unknown", 1, 1)]


@pytest.mark.parametrize("fmt", ["table", "csv", "tsv", "json", "jsonl"])
def test_write_rows_honors_column_selection(fmt):
    buf = io.StringIO()
    write_rows(ROWS, fmt, ["title", "char_count"], buf)
    out = buf.getvalue()
    assert "Kybalion" in out
    assert "Three Initiates" not in out


def test_write_rows_json_is_parseable():
    buf = io.StringIO()
    write_rows(ROWS, "json", ["title"], buf)
    assert json.loads(buf.getvalue()) == [{"title": r["title"]} for r in ROWS]


def test_write_rows_table_aligns_columns():
    buf = io.StringIO()
    write_rows(ROWS, "table", ["title", "language"], buf)
    lines = buf.getvalue().splitlines()
    offset = lines[0].index("language")
    assert all(line[offset:].startswith(("en", "fr", "-")) for line in lines[1:])


def write_corpus(path):
    """Write a parquet with the real corpus schema, including the bulky text column."""
    schema = pa.schema([
        ("id", pa.string()), ("text", pa.large_string()), ("title", pa.string()),
        ("author", pa.string()), ("tradition", pa.string()), ("source", pa.string()),
        ("source_url", pa.string()), ("language", pa.string()), ("file_type", pa.string()),
        ("ocr_used", pa.bool_()), ("char_count", pa.int64()),
    ])
    data = {
        "id": ["a"], "text": ["x" * 10000], "title": ["The Kybalion"], "author": ["Three Initiates"],
        "tradition": ["hermetic"], "source": ["sacred-texts"], "source_url": ["https://example/1"],
        "language": ["en"], "file_type": ["html"], "ocr_used": [False], "char_count": [240000],
    }
    pq.write_table(pa.table(data, schema=schema), path, compression="zstd")


def test_read_parquet_columns_skips_text(tmp_path):
    path = tmp_path / "corpus.parquet"
    write_corpus(path)
    rows = read_parquet_columns(str(path), LIST_COLUMNS)
    assert "text" not in rows[0]
    assert rows[0]["title"] == "The Kybalion"


def test_read_parquet_columns_tolerates_missing_columns(tmp_path):
    path = tmp_path / "partial.parquet"
    pq.write_table(pa.table({"id": ["a"], "title": ["T"]}), path)
    assert read_parquet_columns(str(path), LIST_COLUMNS) == [{"id": "a", "title": "T"}]

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "process_script", Path(__file__).parent.parent / "scripts" / "process.py"
)
process = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(process)


def test_manifest_lookup_sacred_texts_book_directory():
    """Chapter files resolve to the book entry that holds their metadata."""
    manifest = {"alchemy/alc_arr_index.htm": {"title": "A New Light of Alchymie", "tradition": "alchemy"}}
    entry = process.manifest_lookup(manifest, Path("alchemy/alc_arr_index.htm/chapter_000.html"))
    assert entry["title"] == "A New Light of Alchymie"
    assert entry["tradition"] == "alchemy"


def test_manifest_lookup_internet_archive_item_directory():
    manifest = {"some-item": {"title": "Some Item"}}
    entry = process.manifest_lookup(manifest, Path("some-item/scan.pdf"))
    assert entry["title"] == "Some Item"


def test_manifest_lookup_gutenberg_flat_file():
    manifest = {"12345": {"title": "The Kybalion"}}
    entry = process.manifest_lookup(manifest, Path("12345.txt"))
    assert entry["title"] == "The Kybalion"


def test_manifest_lookup_missing_entry_returns_empty():
    assert process.manifest_lookup({}, Path("alchemy/book/chapter_000.html")) == {}


def test_resolve_title_prefers_chapter_title():
    entry = {"title": "A New Light of Alchymie", "chapter_titles": {"chapter_000": "Chapter I: Of Nature"}}
    assert process.resolve_title(Path("chapter_000.html"), entry) == "Chapter I: Of Nature"


def test_resolve_title_falls_back_to_book_title():
    entry = {"title": "A New Light of Alchymie", "chapter_titles": {"chapter_000": "Chapter I"}}
    assert process.resolve_title(Path("chapter_007.html"), entry) == "A New Light of Alchymie"


def test_resolve_title_falls_back_to_filename():
    assert process.resolve_title(Path("chapter_000.html"), {}) == "chapter_000"


def test_build_metadata_carries_book_metadata_to_chapter():
    """The end of the chain the bug broke: chapter rows keep book title and tradition."""
    entry = {
        "title": "A New Light of Alchymie",
        "tradition": "alchemy",
        "source_url": "https://sacred-texts.com/alc/arr/index.htm",
        "chapter_titles": {"chapter_000": "Chapter I: Of Nature"},
    }
    meta = process.build_metadata(
        Path("data/raw/sacred-texts/alchemy/alc_arr_index.htm/chapter_000.html"),
        "sacred-texts",
        entry,
        {"ocr_used": False, "language": "en", "chapters": [], "text": "some text"},
    )
    assert meta["title"] == "Chapter I: Of Nature"
    assert meta["tradition"] == "alchemy"
    assert meta["source_url"] == "https://sacred-texts.com/alc/arr/index.htm"

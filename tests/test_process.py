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


# --- PDF extraction: docling primary, PyMuPDF fallback -----------------------

def _config(**processing):
    base = {"ocr_engine": "easyocr", "ocr_languages": ["en", "la"],
            "quality_gate": {"min_text_length": 10, "max_garbled_ratio": 0.10}}
    base.update(processing)
    return {"processing": base}


def test_extract_pdf_uses_docling_by_default(monkeypatch):
    monkeypatch.setattr(process, "extract_markdown_from_pdf",
                        lambda *a, **k: {"text": "# T\n\nbody", "page_count": 2,
                                         "ocr_used": True, "text_format": "markdown"})
    result = process.extract_pdf(Path("book.pdf"), _config())
    assert result["text_format"] == "markdown"


def test_extract_pdf_passes_configured_ocr_settings(monkeypatch):
    seen = {}

    def spy(path, do_ocr, ocr_engine, languages):
        seen.update(do_ocr=do_ocr, ocr_engine=ocr_engine, languages=languages)
        return {"text": "x", "page_count": 1, "ocr_used": do_ocr, "text_format": "markdown"}

    monkeypatch.setattr(process, "extract_markdown_from_pdf", spy)
    process.extract_pdf(Path("book.pdf"), _config(ocr_engine="rapidocr", ocr_languages=["la"]))
    assert seen == {"do_ocr": True, "ocr_engine": "rapidocr", "languages": ["la"]}


def test_extract_pdf_no_ocr_flag_disables_ocr(monkeypatch):
    seen = {}
    monkeypatch.setattr(process, "extract_markdown_from_pdf",
                        lambda path, do_ocr, ocr_engine, languages: seen.update(do_ocr=do_ocr) or
                        {"text": "x", "page_count": 1, "ocr_used": do_ocr, "text_format": "markdown"})
    process.extract_pdf(Path("book.pdf"), _config(), no_ocr=True)
    assert seen["do_ocr"] is False


def test_extract_pdf_falls_back_when_docling_missing(monkeypatch):
    """A missing docling must cost structure, not the document."""
    def unavailable(*a, **k):
        raise process.DoclingUnavailable("not installed")

    monkeypatch.setattr(process, "extract_markdown_from_pdf", unavailable)
    monkeypatch.setattr(process, "extract_text_from_pdf",
                        lambda p: {"text": "plain text", "page_count": 1, "is_scanned": False})
    result = process.extract_pdf(Path("book.pdf"), _config())
    assert result == {"text": "plain text", "page_count": 1, "ocr_used": False, "text_format": "text"}


def test_extract_pdf_falls_back_when_docling_errors(monkeypatch, tmp_path):
    """One unconvertible PDF must not drop the document."""
    def boom(*a, **k):
        raise RuntimeError("corrupt xref table")

    monkeypatch.setattr(process, "extract_markdown_from_pdf", boom)
    monkeypatch.setattr(process, "extract_text_from_pdf",
                        lambda p: {"text": "plain text", "page_count": 1, "is_scanned": False})
    monkeypatch.setattr(process, "log_error", lambda *a, **k: None)
    result = process.extract_pdf(tmp_path / "book.pdf", _config())
    assert result["text_format"] == "text"
    assert result["text"] == "plain text"


def test_extract_pdf_skips_docling_when_disabled(monkeypatch):
    monkeypatch.setattr(process, "extract_markdown_from_pdf",
                        lambda *a, **k: pytest.fail("docling should not be called"))
    monkeypatch.setattr(process, "extract_text_from_pdf",
                        lambda p: {"text": "plain text", "page_count": 1, "is_scanned": False})
    assert process.extract_pdf(Path("b.pdf"), _config(), use_docling=False)["text_format"] == "text"


# --- markdown survives the quality gate --------------------------------------

MARKDOWN_WITH_TABLE = """## Planetary Correspondences

| Planet | Metal | Day |
|--------|-------|-----|
| Sol | Gold | Sunday |
| Luna | Silver | Monday |

The operation of the Sun is governed by gold, as the philosophers taught.
"""


def test_markdown_pdf_survives_the_garble_gate(monkeypatch, tmp_path):
    """The table pipes would read as encoding damage without markdown-aware checking."""
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(process, "extract_markdown_from_pdf",
                        lambda *a, **k: {"text": MARKDOWN_WITH_TABLE, "page_count": 1,
                                         "ocr_used": False, "text_format": "markdown"})
    result = process.process_file(pdf, _config())
    assert result is not None, "markdown table was wrongly rejected as garbled"
    assert result["text_format"] == "markdown"


def test_markdown_pdf_keeps_its_structure(monkeypatch, tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(process, "extract_markdown_from_pdf",
                        lambda *a, **k: {"text": MARKDOWN_WITH_TABLE, "page_count": 1,
                                         "ocr_used": False, "text_format": "markdown"})
    text = process.process_file(pdf, _config())["text"]
    assert "## Planetary Correspondences" in text
    assert "| Sol | Gold | Sunday |" in text


def test_build_metadata_records_text_format():
    meta = process.build_metadata(
        Path("data/raw/internet-archive/item/scan.pdf"), "internet-archive", {"title": "T"},
        {"ocr_used": True, "language": "en", "chapters": [], "text": "x", "text_format": "markdown"},
    )
    assert meta["text_format"] == "markdown"


def test_build_metadata_defaults_text_format_for_older_results():
    meta = process.build_metadata(
        Path("a/b/c.html"), "sacred-texts", {"title": "T"},
        {"ocr_used": False, "language": "en", "chapters": [], "text": "x"},
    )
    assert meta["text_format"] == "text"

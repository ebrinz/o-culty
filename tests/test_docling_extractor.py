import sys
from pathlib import Path

import pytest

from src.processor import docling_extractor as de
from src.processor.docling_extractor import (
    OCR_ENGINES, DoclingUnavailable, _ocr_options, build_converter, extract_markdown_from_pdf,
)


def test_ocr_engines_cover_the_configured_default():
    """config.yaml ships ocr_engine: easyocr — it must map to a backend."""
    assert "easyocr" in OCR_ENGINES


def test_unknown_ocr_engine_raises_before_touching_docling():
    with pytest.raises(ValueError, match="Unknown ocr_engine"):
        _ocr_options("nonsuch", ["en"])


def test_build_converter_raises_docling_unavailable_when_import_fails(monkeypatch):
    """Callers rely on this specific error to fall back rather than lose the document."""
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    monkeypatch.setattr(de, "_CONVERTERS", {})
    with pytest.raises(DoclingUnavailable):
        build_converter(do_ocr=False)


def test_extract_missing_file_raises_before_loading_models(monkeypatch):
    monkeypatch.setattr(de, "build_converter", lambda **kw: pytest.fail("should not build a converter"))
    with pytest.raises(FileNotFoundError):
        extract_markdown_from_pdf(Path("/nonexistent/none.pdf"))


class FakeDocument:
    def __init__(self, markdown, pages):
        self._markdown = markdown
        self.pages = pages

    def export_to_markdown(self):
        return self._markdown


class FakeResult:
    def __init__(self, document):
        self.document = document


class FakeConverter:
    def __init__(self, markdown="# Title\n\nBody text.", pages=(1, 2, 3)):
        self.document = FakeDocument(markdown, {p: object() for p in pages})
        self.converted = []

    def convert(self, source):
        self.converted.append(source)
        return FakeResult(self.document)


def test_extract_returns_markdown_and_page_count(tmp_path, monkeypatch):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    fake = FakeConverter()
    monkeypatch.setattr(de, "build_converter", lambda **kw: fake)

    result = extract_markdown_from_pdf(pdf, do_ocr=False)

    assert result["text"] == "# Title\n\nBody text."
    assert result["text_format"] == "markdown"
    assert result["page_count"] == 3
    assert result["ocr_used"] is False
    assert fake.converted == [str(pdf)]


def test_extract_reports_ocr_enabled(tmp_path, monkeypatch):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(de, "build_converter", lambda **kw: FakeConverter())
    assert extract_markdown_from_pdf(pdf, do_ocr=True)["ocr_used"] is True


def test_extract_passes_ocr_settings_to_the_converter(tmp_path, monkeypatch):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return FakeConverter()

    monkeypatch.setattr(de, "build_converter", spy)
    extract_markdown_from_pdf(pdf, do_ocr=True, ocr_engine="tesseract", languages=["en", "la"])

    assert seen == {"do_ocr": True, "ocr_engine": "tesseract", "languages": ["en", "la"]}


def test_build_converter_memoises_per_configuration():
    """Each converter loads layout models; rebuilding per document would dominate runtime."""
    pytest.importorskip("docling")
    first = build_converter(do_ocr=False)
    second = build_converter(do_ocr=False)
    assert first is second

import pytest
from pathlib import Path
from src.processor.pdf_extractor import extract_text_from_pdf, is_scanned_pdf
import fitz

@pytest.fixture
def text_pdf(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Chapter I: The Emerald Tablet\n\nThat which is above is like that which is below.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Chapter II: The Seven Principles\n\nThe All is Mind; the Universe is Mental.")
    path = tmp_path / "test.pdf"
    doc.save(str(path))
    doc.close()
    return path

def test_extract_text_from_pdf(text_pdf):
    result = extract_text_from_pdf(text_pdf)
    assert "Emerald Tablet" in result["text"]
    assert "Seven Principles" in result["text"]
    assert result["page_count"] == 2

def test_is_scanned_detects_text_pdf(text_pdf):
    assert is_scanned_pdf(text_pdf) is False

def test_extract_nonexistent_pdf():
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(Path("/nonexistent.pdf"))

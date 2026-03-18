import pytest
from unittest.mock import patch, MagicMock
from src.processor.ocr import ocr_pdf_page

def test_ocr_pdf_page_returns_string():
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        (None, "As above", 0.95),
        (None, "so below", 0.90),
    ]
    with patch("src.processor.ocr._get_reader", return_value=mock_reader):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        doc.close()
        result = ocr_pdf_page(img_bytes)
        assert "As above" in result
        assert "so below" in result

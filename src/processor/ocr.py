import logging
from pathlib import Path
import fitz

logger = logging.getLogger(__name__)
_reader = None

def _get_reader(languages: list[str] | None = None):
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(languages or ["en", "la"], gpu=True)
    return _reader

def ocr_pdf_page(image_bytes: bytes, languages: list[str] | None = None) -> str:
    reader = _get_reader(languages)
    results = reader.readtext(image_bytes)
    lines = [text for _, text, conf in results if conf > 0.3]
    return " ".join(lines)

def ocr_pdf(path: Path, languages: list[str] | None = None) -> dict:
    path = Path(path)
    doc = fitz.open(str(path))
    pages_text = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        text = ocr_pdf_page(img_bytes, languages)
        pages_text.append(text)
        logger.debug(f"OCR page {i+1}/{len(doc)}: {len(text)} chars")
    doc.close()
    return {"text": "\n\n".join(pages_text), "page_count": len(pages_text)}

import io
import logging
from pathlib import Path

import fitz
from PIL import Image

logger = logging.getLogger(__name__)


def ocr_pdf_page(image_bytes: bytes) -> str:
    """OCR a single page using macOS Vision framework via ocrmac."""
    from ocrmac import ocrmac
    img = Image.open(io.BytesIO(image_bytes))
    result = ocrmac.OCR(img, language_preference=["en-US"]).recognize()
    return " ".join(r[0] for r in result)


def ocr_pdf(path: Path, languages: list[str] | None = None) -> dict:
    path = Path(path)
    doc = fitz.open(str(path))
    pages_text = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        text = ocr_pdf_page(img_bytes)
        pages_text.append(text)
        logger.debug(f"OCR page {i+1}/{len(doc)}: {len(text)} chars")
    doc.close()
    return {"text": "\n\n".join(pages_text), "page_count": len(pages_text)}

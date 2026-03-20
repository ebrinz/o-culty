import logging
from pathlib import Path
import fitz

logger = logging.getLogger(__name__)


def extract_text_from_pdf(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    doc = fitz.open(str(path))
    pages_text = []
    total_chars = 0
    for page in doc:
        text = page.get_text()
        pages_text.append(text)
        total_chars += len(text.strip())
    doc.close()
    full_text = "\n\n".join(pages_text)
    is_scanned = total_chars < 10 * len(pages_text)
    return {"text": full_text, "page_count": len(pages_text), "is_scanned": is_scanned}

def is_scanned_pdf(path: Path) -> bool:
    result = extract_text_from_pdf(path)
    return result["is_scanned"]

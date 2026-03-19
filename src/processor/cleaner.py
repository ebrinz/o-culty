import re
import unicodedata
from Levenshtein import ratio as levenshtein_ratio

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n\n", text)
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

_LATIN_WORDS = {
    "et", "est", "in", "non", "sed", "ad", "per", "cum", "vel", "enim",
    "autem", "quod", "qui", "quae", "quam", "quis", "omnis", "omnia",
    "esse", "sunt", "erat", "esse", "ille", "illa", "illud", "hoc", "haec",
    "deus", "dei", "terra", "caelum", "vitae", "anima", "spiritus",
}

def _is_latin(text: str) -> bool:
    words = {w.lower().strip(".,;:!?\"'()") for w in text.split()}
    overlap = words & _LATIN_WORDS
    return len(overlap) >= 3

def detect_language(text: str) -> str:
    if _is_latin(text):
        return "la"
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"

def normalize_title(title: str) -> str:
    title = title.lower().strip()
    title = re.sub(r"\s+", " ", title)
    return title

def is_duplicate(title_a: str, title_b: str, threshold: float = 0.85) -> bool:
    a = normalize_title(title_a)
    b = normalize_title(title_b)
    return levenshtein_ratio(a, b) >= threshold


def is_garbled(text: str, max_ratio: float = 0.10) -> bool:
    """Check if extracted text is garbled (broken font encoding, OCR garbage).

    Looks at ratio of non-printable / unusual characters. Garbled PDFs with
    broken font mappings produce text like 'OFDLAD' instead of 'OF DEAD'.
    """
    if not text:
        return True
    sample = text[:2000]
    weird = sum(1 for c in sample if ord(c) > 127 or c in "£¬|»«®©™{}[]<>\\^~`")
    return (weird / len(sample)) > max_ratio

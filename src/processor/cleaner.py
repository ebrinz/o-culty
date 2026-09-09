import re
import unicodedata
from Levenshtein import ratio as levenshtein_ratio

def _rejoin_hyphenated(text: str) -> str:
    """Rejoin words split across lines by hyphens (e.g. 'dis-\\ntinct' -> 'distinct')."""
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def _strip_ocr_noise(text: str) -> str:
    """Remove isolated short lines that are OCR scan artifacts (e.g. 'I', '|', '\\')."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Blank lines are paragraph structure, not scan noise. They are shorter
        # than the noise threshold below, so they need saying explicitly.
        if not stripped:
            cleaned.append(line)
            continue
        # Drop lines that are just 1-2 non-alphanumeric chars or single letters
        if len(stripped) <= 2 and not re.match(r"^[A-Za-z0-9]{2}$", stripped):
            if not re.match(r"^[IVXLCDM]+$", stripped):  # keep roman numerals
                continue
        cleaned.append(line)
    return "\n".join(cleaned)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _rejoin_hyphenated(text)
    text = _strip_ocr_noise(text)
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n\n", text)
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def normalize_markdown(text: str) -> str:
    """Tidy markdown without flattening the structure the extractor captured.

    normalize_text() strips each line's leading whitespace and drops very short
    lines, which would destroy nested lists, table rows and heading markers. So
    markdown gets a gentler pass: unicode normalisation, de-hyphenation, page
    numbers, trailing whitespace and runs of blank lines.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _rejoin_hyphenated(text)
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Table separator rows ("|---|:--:|") carry no prose and are all "weird" chars.
_MD_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]{3,}\|?\s*$", re.M)


def strip_markdown_syntax(text: str) -> str:
    """Reduce markdown to its prose, so text-quality checks judge words not syntax.

    Table pipes and link brackets are exactly the characters is_garbled() counts
    as encoding damage, so a well-extracted table would otherwise look garbled.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = _MD_TABLE_SEPARATOR_RE.sub("", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)
    text = re.sub(r"[*_`]", "", text)
    text = text.replace("|", " ")
    return text


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


def is_garbled(text: str, max_ratio: float = 0.10, markdown: bool = False) -> bool:
    """Check if extracted text is garbled (broken font encoding, OCR garbage).

    Looks at ratio of non-printable / unusual characters AND ratio of sparse
    lines (1-3 chars) which indicate OCR scan noise.

    Set markdown=True for structured extractions: table pipes and link brackets
    are counted as encoding damage, so a cleanly extracted table would otherwise
    be rejected as garbage.
    """
    if not text:
        return True
    if markdown:
        text = strip_markdown_syntax(text)
        if not text.strip():
            return True
    sample = text[:2000]
    weird = sum(1 for c in sample if ord(c) > 127 or c in "£¬|»«®©™{}[]<>\\^~`")
    if (weird / len(sample)) > max_ratio:
        return True
    # Check for sparse-line pattern (scan noise like isolated 'I', '|', etc.)
    lines = [l.strip() for l in sample.splitlines() if l.strip()]
    if len(lines) >= 10:
        sparse = sum(1 for l in lines if len(l) <= 3)
        if (sparse / len(lines)) > 0.50:
            return True
    return False

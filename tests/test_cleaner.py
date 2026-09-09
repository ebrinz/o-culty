import pytest
from src.processor.cleaner import (
    normalize_text, detect_language, is_duplicate, normalize_title, is_garbled,
    normalize_markdown, strip_markdown_syntax,
)

def test_normalize_unicode():
    assert "fi" in normalize_text("ﬁrst")

def test_normalize_whitespace():
    result = normalize_text("hello   \t  world\n\n\n\nfoo")
    assert "   " not in result
    assert "\n\n" in result

def test_strip_page_numbers():
    result = normalize_text("Some text here.\n\n42\n\nMore text.")
    assert "\n42\n" not in result

def test_detect_language_english():
    lang = detect_language("This is a simple English sentence about the nature of reality.")
    assert lang == "en"

def test_detect_language_latin():
    lang = detect_language("In principio creavit Deus caelum et terram.")
    assert lang == "la"

def test_normalize_title():
    assert normalize_title("The  Corpus  Hermeticum") == "the corpus hermeticum"
    assert normalize_title("  AGRIPPA's Three Books ") == "agrippa's three books"

def test_is_duplicate_exact():
    assert is_duplicate("the corpus hermeticum", "the corpus hermeticum") is True

def test_is_duplicate_fuzzy():
    assert is_duplicate("the corpus hermeticum", "corpus hermeticum the", threshold=0.7) is True

def test_is_not_duplicate():
    assert is_duplicate("the corpus hermeticum", "the key of solomon") is False


def test_is_garbled_clean_text():
    assert is_garbled("This is perfectly normal English text about alchemy.") is False


def test_is_garbled_broken_encoding():
    assert is_garbled("TH£ BOOK Of TR£ASVR£ SP1R1TS ¬ »«®© £¬|»« ®©™") is True


def test_is_garbled_empty():
    assert is_garbled("") is True


MARKDOWN_TABLE = """## Table of Planetary Correspondences

| Planet | Metal | Day |
|--------|-------|-----|
| Sol | Gold | Sunday |
| Luna | Silver | Monday |
| Mars | Iron | Tuesday |
| Venus | Copper | Friday |
"""


def test_is_garbled_rejects_markdown_table_without_the_flag():
    """Guards the reason the flag exists: pipes read as encoding damage."""
    assert is_garbled(MARKDOWN_TABLE) is True


def test_is_garbled_accepts_markdown_table_with_the_flag():
    assert is_garbled(MARKDOWN_TABLE, markdown=True) is False


def test_is_garbled_still_catches_garbage_in_markdown_mode():
    assert is_garbled("\x00�� �{}<>\\^~ ��� ��" * 20, markdown=True) is True


def test_is_garbled_markdown_mode_rejects_syntax_only_input():
    assert is_garbled("# \n\n| |\n|---|\n", markdown=True) is True


def test_strip_markdown_syntax_keeps_prose():
    stripped = strip_markdown_syntax("## Heading\n\n- a bullet\n\n[label](http://x)\n\n`code`\n")
    assert "Heading" in stripped
    assert "a bullet" in stripped
    assert "label" in stripped
    assert "http://x" not in stripped
    assert "#" not in stripped


def test_strip_markdown_syntax_drops_docling_image_placeholders():
    assert "image" not in strip_markdown_syntax("Text before\n\n<!-- image -->\n\nText after")


def test_normalize_markdown_preserves_structure():
    md = "# Heading\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n-   nested\n    -   deeper\n"
    out = normalize_markdown(md)
    assert "# Heading" in out
    assert "| a | b |" in out
    assert "    -   deeper" in out


def test_normalize_markdown_collapses_blank_lines_and_trailing_space():
    out = normalize_markdown("a   \n\n\n\n\nb")
    assert out == "a\n\nb"


def test_normalize_text_would_destroy_markdown_structure():
    """Why markdown needs its own normaliser rather than reusing normalize_text."""
    md = "# Heading\n\n-   nested\n    -   deeper\n"
    assert "    -   deeper" not in normalize_text(md)


def test_normalize_text_preserves_paragraph_breaks():
    """Blank lines are structure; _strip_ocr_noise used to delete every one."""
    out = normalize_text("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
    assert out == "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."


def test_normalize_text_still_drops_scan_noise_between_paragraphs():
    out = normalize_text("Real prose here.\n\n|\n\\\na\n\nMore real prose.")
    assert "\n|\n" not in out
    assert "\n\\\n" not in out
    assert "Real prose here." in out
    assert "More real prose." in out


def test_normalize_text_keeps_roman_numerals():
    assert "IV" in normalize_text("Chapter text.\n\nIV\n\nMore chapter text.")


def test_normalize_text_collapses_long_blank_runs():
    assert normalize_text("alpha\n\n\n\n\n\nbeta") == "alpha\n\nbeta"

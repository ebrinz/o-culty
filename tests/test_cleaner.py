import pytest
from src.processor.cleaner import normalize_text, detect_language, is_duplicate, normalize_title, is_garbled

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

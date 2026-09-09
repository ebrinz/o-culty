import pytest
from src.processor.html_extractor import extract_text_from_html

SAMPLE_HTML = """
<html>
<head><title>Chapter I</title></head>
<body>
<div class="navbar"><a href="index.htm">Back</a></div>
<h2>Chapter I: Poemandres</h2>
<p>Once on a time, when I had begun to think about the things that are,
and my thoughts had soared high aloft.</p>
<p>My bodily senses had been restrained, as happens to those
oppressed by sleep.</p>
<div class="footer">sacred-texts.com 2024</div>
</body>
</html>
"""

def test_extract_strips_nav_and_footer():
    result = extract_text_from_html(SAMPLE_HTML)
    assert "navbar" not in result["text"].lower()
    assert "sacred-texts.com" not in result["text"]

def test_extract_preserves_content():
    result = extract_text_from_html(SAMPLE_HTML)
    assert "Poemandres" in result["text"]
    assert "thoughts had soared" in result["text"]

def test_extract_finds_chapters():
    result = extract_text_from_html(SAMPLE_HTML)
    assert len(result["chapters"]) >= 1
    assert "Poemandres" in result["chapters"][0]

def test_extract_empty_html():
    result = extract_text_from_html("<html><body></body></html>")
    assert result["text"] == ""


def test_extract_separates_paragraphs():
    """Paragraph breaks used to vanish, leaving one undifferentiated run of lines."""
    result = extract_text_from_html(SAMPLE_HTML)
    paragraphs = [p for p in result["text"].split("\n\n") if p.strip()]
    assert len(paragraphs) == 3  # heading + two paragraphs
    assert paragraphs[0] == "Chapter I: Poemandres"
    assert paragraphs[1].startswith("Once on a time")
    assert paragraphs[2].startswith("My bodily senses")


def test_extract_keeps_sentences_with_inline_markup_intact():
    """separator="\\n" used to split a sentence at every inline tag."""
    html = '<html><body><p>The Tablet, <i>as rendered</i> by <a href="x">Newton</a>.</p></body></html>'
    assert extract_text_from_html(html)["text"] == "The Tablet, as rendered by Newton."


def test_extract_does_not_insert_space_inside_a_word():
    html = "<html><body><p>un<b>bold</b>ed</p></body></html>"
    assert extract_text_from_html(html)["text"] == "unbolded"


def test_extract_treats_br_as_a_line_break_not_a_paragraph():
    html = "<html><body><p>first line<br>second line</p></body></html>"
    assert extract_text_from_html(html)["text"] == "first line\nsecond line"


def test_extract_separates_table_cells():
    html = "<html><body><table><tr><td>Sol</td><td>Gold</td></tr></table></body></html>"
    assert "Sol\tGold" in extract_text_from_html(html)["text"]


def test_extract_does_not_duplicate_nested_block_text():
    html = "<html><body><div><div><p>Only once.</p></div></div></body></html>"
    assert extract_text_from_html(html)["text"].count("Only once.") == 1


def test_extract_collapses_runs_of_blank_lines():
    html = "<html><body><p>one</p><div></div><div></div><p>two</p></body></html>"
    assert extract_text_from_html(html)["text"] == "one\n\ntwo"


def test_extract_html_without_body_returns_empty():
    assert extract_text_from_html("<p>orphan</p>")["text"] == ""

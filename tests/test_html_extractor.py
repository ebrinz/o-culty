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

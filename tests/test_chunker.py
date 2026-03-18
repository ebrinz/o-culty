import pytest
from src.chunker.chunker import chunk_text, split_into_chapters

SAMPLE_TEXT = """Chapter I: Poemandres

Once on a time, when I had begun to think about the things that are,
and my thoughts had soared high aloft. My bodily senses had been restrained.

Chapter II: To Asclepius

I was filled with joy unspeakable. For the sense of my body was at rest.
And I saw an endless vision in which everything had become light."""

def test_split_into_chapters():
    chapters = split_into_chapters(SAMPLE_TEXT)
    assert len(chapters) == 2
    assert "Poemandres" in chapters[0]["title"]
    assert "Asclepius" in chapters[1]["title"]

def test_split_no_chapters():
    chapters = split_into_chapters("Just some text with no chapter headings.")
    assert len(chapters) == 1
    assert chapters[0]["title"] == ""

def test_chunk_text_basic():
    chunks = chunk_text(
        text=SAMPLE_TEXT,
        metadata={"id": "test", "title": "Test"},
        chunk_size=50,
        overlap=10,
        tokenizer_name="nomic-ai/nomic-embed-text-v1.5",
    )
    assert len(chunks) > 0
    assert all("chunk_id" in c for c in chunks)
    assert all("text" in c for c in chunks)
    assert all("chapter" in c for c in chunks)
    assert chunks[0]["id"] == "test"

def test_chunk_preserves_metadata():
    chunks = chunk_text(
        text="Short text.",
        metadata={"id": "test", "title": "Test Book", "tradition": "hermetic"},
        chunk_size=512,
        overlap=64,
        tokenizer_name="nomic-ai/nomic-embed-text-v1.5",
    )
    assert len(chunks) == 1
    assert chunks[0]["title"] == "Test Book"
    assert chunks[0]["tradition"] == "hermetic"

def test_short_chapter_stays_single():
    chunks = chunk_text(
        text="Chapter I: Short\n\nJust a tiny bit of text.",
        metadata={"id": "test", "title": "Test"},
        chunk_size=512,
        overlap=64,
        tokenizer_name="nomic-ai/nomic-embed-text-v1.5",
    )
    assert len(chunks) == 1

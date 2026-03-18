import pytest
import numpy as np
import pandas as pd
from src.analysis.search import semantic_search


@pytest.fixture
def sample_corpus():
    vectors = np.random.randn(5, 768).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    metadata = pd.DataFrame({
        "chunk_id": [f"chunk_{i}" for i in range(5)],
        "text": ["The transmutation of metals", "Kabbalistic tree of life", "Alchemical gold from lead", "The nature of the divine", "Enochian angel language"],
        "tradition": ["alchemy", "kabbalah", "alchemy", "hermetic", "enochian"],
        "title": ["Book A", "Book B", "Book A", "Book C", "Book D"],
    })
    return vectors, metadata


def test_search_returns_results(sample_corpus):
    vectors, metadata = sample_corpus
    query_vec = np.random.randn(768).astype(np.float32)
    results = semantic_search(query_vec, vectors, metadata, top_k=3)
    assert len(results) == 3
    assert "score" in results[0]
    assert "text" in results[0]


def test_search_filter_by_tradition(sample_corpus):
    vectors, metadata = sample_corpus
    query_vec = np.random.randn(768).astype(np.float32)
    results = semantic_search(query_vec, vectors, metadata, top_k=10, filter_tradition="alchemy")
    assert all(r["tradition"] == "alchemy" for r in results)


def test_search_results_sorted_descending(sample_corpus):
    vectors, metadata = sample_corpus
    query_vec = np.random.randn(768).astype(np.float32)
    results = semantic_search(query_vec, vectors, metadata, top_k=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

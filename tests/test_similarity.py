import pytest
import numpy as np
import pandas as pd
from src.analysis.similarity import compute_similarity_matrix


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    vectors = rng.randn(9, 768).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    metadata = pd.DataFrame({
        "chunk_id": [f"c{i}" for i in range(9)],
        "tradition": ["alchemy"] * 3 + ["kabbalah"] * 3 + ["hermetic"] * 3,
        "title": ["Book A"] * 3 + ["Book B"] * 3 + ["Book C"] * 3,
    })
    return vectors, metadata


def test_similarity_by_tradition(sample_data):
    vectors, metadata = sample_data
    labels, matrix = compute_similarity_matrix(vectors, metadata, level="tradition")
    assert matrix.shape == (3, 3)
    assert len(labels) == 3
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-5)


def test_similarity_by_title(sample_data):
    vectors, metadata = sample_data
    labels, matrix = compute_similarity_matrix(vectors, metadata, level="title")
    assert matrix.shape == (3, 3)


def test_similarity_symmetric(sample_data):
    vectors, metadata = sample_data
    _, matrix = compute_similarity_matrix(vectors, metadata, level="tradition")
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-5)

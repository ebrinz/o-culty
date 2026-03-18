import pytest
import numpy as np
import pandas as pd
from src.analysis.visualize import reduce_dimensions, create_scatter_plot, create_heatmap


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    vectors = rng.randn(50, 768).astype(np.float32)
    metadata = pd.DataFrame({
        "chunk_id": [f"c{i}" for i in range(50)],
        "text": [f"Sample text {i}" for i in range(50)],
        "tradition": ["alchemy"] * 25 + ["kabbalah"] * 25,
        "title": ["Book A"] * 25 + ["Book B"] * 25,
    })
    return vectors, metadata


def test_reduce_dimensions_umap(sample_data):
    vectors, _ = sample_data
    coords = reduce_dimensions(vectors, method="umap", n_components=2)
    assert coords.shape == (50, 2)


def test_reduce_dimensions_tsne(sample_data):
    vectors, _ = sample_data
    coords = reduce_dimensions(vectors, method="tsne", n_components=2)
    assert coords.shape == (50, 2)


def test_create_scatter_returns_figure(sample_data):
    vectors, metadata = sample_data
    coords = reduce_dimensions(vectors, method="umap", n_components=2)
    fig = create_scatter_plot(coords, metadata, color_by="tradition")
    assert fig is not None
    assert hasattr(fig, "data")


def test_create_heatmap_returns_figure():
    labels = ["alchemy", "kabbalah", "hermetic"]
    matrix = np.eye(3)
    fig = create_heatmap(labels, matrix, title="Test")
    assert fig is not None
    assert hasattr(fig, "data")

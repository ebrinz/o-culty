import pytest
import numpy as np
import pandas as pd
from src.analysis.cluster import cluster_embeddings, label_clusters


@pytest.fixture
def clusterable_data():
    rng = np.random.RandomState(42)
    cluster_a = rng.randn(50, 768).astype(np.float32) * 0.1
    cluster_a[:, 0] += 3.0
    cluster_b = rng.randn(50, 768).astype(np.float32) * 0.1
    cluster_b[:, 0] -= 3.0
    vectors = np.concatenate([cluster_a, cluster_b])
    metadata = pd.DataFrame({
        "chunk_id": [f"chunk_{i}" for i in range(100)],
        "text": [f"Text about topic A number {i}" for i in range(50)] + [f"Text about topic B number {i}" for i in range(50)],
        "tradition": ["alchemy"] * 50 + ["kabbalah"] * 50,
    })
    return vectors, metadata


def test_cluster_finds_groups(clusterable_data):
    vectors, metadata = clusterable_data
    labels = cluster_embeddings(vectors, min_cluster_size=10)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == 100
    unique = set(labels)
    unique.discard(-1)
    assert len(unique) >= 2


def test_label_clusters():
    texts = ["gold transmutation alchemy", "gold lead transform", "tree sephiroth divine"]
    labels = np.array([0, 0, 1])
    cluster_labels = label_clusters(texts, labels, top_n=2)
    assert 0 in cluster_labels
    assert 1 in cluster_labels
    assert isinstance(cluster_labels[0], list)

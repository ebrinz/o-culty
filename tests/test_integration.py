"""Integration test: chunk -> embed -> analyze on synthetic data."""
import pytest
import numpy as np
import pandas as pd

from src.chunker.chunker import chunk_text
from src.embedder.embedder import Embedder
from src.analysis.search import semantic_search
from src.analysis.similarity import compute_similarity_matrix
from src.analysis.visualize import reduce_dimensions

HERMETIC_TEXT = """Chapter I: The Divine Mind

The All is Mind; the Universe is Mental. As above, so below; as below, so above.
The lips of wisdom are closed, except to the ears of Understanding.
Nothing rests; everything moves; everything vibrates.
Everything is Dual; everything has poles; everything has its pair of opposites."""

ALCHEMICAL_TEXT = """Chapter I: The Emerald Tablet

That which is above is like that which is below, for performing the miracle of the One Thing.
The Sun is its father, the Moon its mother. The Wind carries it in its belly.
Separate the Earth from the Fire, the subtle from the gross, gently and with great judgment."""

@pytest.fixture(scope="module")
def embedder():
    return Embedder(model_name="nomic-ai/nomic-embed-text-v1.5", device="cpu")

def test_full_pipeline(embedder):
    # Chunk
    hermetic_chunks = chunk_text(
        text=HERMETIC_TEXT,
        metadata={"id": "hermetic_kybalion", "title": "The Kybalion", "tradition": "hermetic"},
        chunk_size=128, overlap=32,
    )
    alchemical_chunks = chunk_text(
        text=ALCHEMICAL_TEXT,
        metadata={"id": "alchemy_emerald", "title": "Emerald Tablet", "tradition": "alchemy"},
        chunk_size=128, overlap=32,
    )
    all_chunks = hermetic_chunks + alchemical_chunks
    assert len(all_chunks) >= 2

    # Embed
    texts = [c["text"] for c in all_chunks]
    vectors = embedder.embed(texts)
    assert vectors.shape[0] == len(all_chunks)
    assert vectors.shape[1] == 768

    # Build metadata DataFrame
    metadata = pd.DataFrame(all_chunks)

    # Search
    query_vec = embedder.embed(["transmutation of metals"])[0]
    results = semantic_search(query_vec, vectors, metadata, top_k=3)
    assert len(results) > 0
    assert "score" in results[0]

    # Similarity
    labels, sim_matrix = compute_similarity_matrix(vectors, metadata, level="tradition")
    assert len(labels) == 2
    assert sim_matrix.shape == (2, 2)

    # Dimensionality reduction (just verify it doesn't crash)
    if len(vectors) >= 5:
        coords = reduce_dimensions(vectors, method="umap", n_components=2)
        assert coords.shape == (len(vectors), 2)

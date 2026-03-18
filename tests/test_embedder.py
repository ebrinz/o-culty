import pytest
import numpy as np
from src.embedder.embedder import Embedder

@pytest.fixture(scope="module")
def embedder():
    return Embedder(model_name="nomic-ai/nomic-embed-text-v1.5", device="cpu")

def test_embed_single(embedder):
    vectors = embedder.embed(["Hello world"])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (1, 768)

def test_embed_batch(embedder):
    texts = ["Hello world", "Goodbye world", "The nature of reality"]
    vectors = embedder.embed(texts)
    assert vectors.shape == (3, 768)

def test_embed_similar_texts_closer(embedder):
    texts = [
        "The transmutation of base metals into gold",
        "Alchemical transformation of lead to gold",
        "The weather today is sunny and warm",
    ]
    vectors = embedder.embed(texts)
    from numpy.linalg import norm
    def cosine(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))
    sim_related = cosine(vectors[0], vectors[1])
    sim_unrelated = cosine(vectors[0], vectors[2])
    assert sim_related > sim_unrelated

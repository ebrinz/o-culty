import numpy as np
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP


def cluster_embeddings(
    vectors: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    umap_dim: int = 50,
    umap_n_neighbors: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce with UMAP first, then cluster with HDBSCAN. Returns (labels, reduced_vectors)."""
    print(f"  UMAP: {vectors.shape[1]}d -> {umap_dim}d ({vectors.shape[0]} points)...")
    reducer = UMAP(n_components=umap_dim, n_neighbors=umap_n_neighbors, metric="cosine", verbose=True)
    reduced = reducer.fit_transform(vectors)

    print(f"  HDBSCAN: clustering {reduced.shape[0]} points in {umap_dim}d...")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)
    return labels, reduced


def label_clusters(
    texts: list[str],
    labels: np.ndarray,
    top_n: int = 5,
) -> dict[int, list[str]]:
    unique_labels = sorted(set(labels))
    unique_labels = [l for l in unique_labels if l != -1]
    cluster_docs = {}
    for text, label in zip(texts, labels):
        if label == -1:
            continue
        cluster_docs.setdefault(label, []).append(text)
    corpus = []
    cluster_order = []
    for label in unique_labels:
        corpus.append(" ".join(cluster_docs.get(label, [])))
        cluster_order.append(label)
    if not corpus:
        return {}
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    cluster_labels = {}
    for i, label in enumerate(cluster_order):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]
        cluster_labels[label] = [feature_names[j] for j in top_indices]
    return cluster_labels

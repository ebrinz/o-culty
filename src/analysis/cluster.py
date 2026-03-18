import numpy as np
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


def cluster_embeddings(
    vectors: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
) -> np.ndarray:
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(vectors)
    return labels


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

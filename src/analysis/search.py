import numpy as np
import pandas as pd


def semantic_search(
    query_vector: np.ndarray,
    corpus_vectors: np.ndarray,
    metadata: pd.DataFrame,
    top_k: int = 10,
    filter_tradition: str | None = None,
    filter_title: str | None = None,
) -> list[dict]:
    query_vector = query_vector / np.linalg.norm(query_vector)
    mask = np.ones(len(metadata), dtype=bool)
    if filter_tradition is not None:
        mask &= (metadata["tradition"] == filter_tradition).values
    if filter_title is not None:
        mask &= (metadata["title"] == filter_title).values
    if not mask.any():
        return []
    filtered_vectors = corpus_vectors[mask]
    filtered_meta = metadata[mask].reset_index(drop=True)
    scores = filtered_vectors @ query_vector
    k = min(top_k, len(scores))
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    results = []
    for idx in top_indices:
        row = filtered_meta.iloc[idx].to_dict()
        row["score"] = float(scores[idx])
        results.append(row)
    return results

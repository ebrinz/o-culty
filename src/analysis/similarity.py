import numpy as np
import pandas as pd


def compute_similarity_matrix(
    vectors: np.ndarray,
    metadata: pd.DataFrame,
    level: str = "tradition",
) -> tuple[list[str], np.ndarray]:
    groups = sorted(metadata[level].unique())
    mean_vectors = []
    for group in groups:
        mask = (metadata[level] == group).values
        group_vectors = vectors[mask]
        mean_vec = group_vectors.mean(axis=0)
        mean_vec = mean_vec / np.linalg.norm(mean_vec)
        mean_vectors.append(mean_vec)
    mean_matrix = np.stack(mean_vectors)
    sim_matrix = mean_matrix @ mean_matrix.T
    return list(groups), sim_matrix

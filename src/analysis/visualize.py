import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def reduce_dimensions(
    vectors: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    if method == "umap":
        from umap import UMAP
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=min(30, len(vectors) - 1),
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return reducer.fit_transform(vectors)


def create_scatter_plot(
    coords: np.ndarray,
    metadata: pd.DataFrame,
    color_by: str = "tradition",
    title: str = "Occult Text Embeddings",
) -> go.Figure:
    plot_df = metadata.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    hover_cols = [c for c in ["text", "tradition", "title"] if c in plot_df.columns]
    if coords.shape[1] == 3:
        plot_df["z"] = coords[:, 2]
        fig = px.scatter_3d(plot_df, x="x", y="y", z="z", color=color_by, hover_data=hover_cols, title=title)
    else:
        fig = px.scatter(plot_df, x="x", y="y", color=color_by, hover_data=hover_cols, title=title)
    fig.update_layout(template="plotly_dark", width=1200, height=800)
    return fig


def create_heatmap(
    labels: list[str],
    matrix: np.ndarray,
    title: str = "Similarity Matrix",
) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale="Viridis",
        text=np.round(matrix, 3),
        texttemplate="%{text}",
    ))
    fig.update_layout(title=title, template="plotly_dark", width=800, height=800)
    return fig

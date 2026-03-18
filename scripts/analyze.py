import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import load_config
from src.embedder.embedder import Embedder
from src.analysis.search import semantic_search
from src.analysis.cluster import cluster_embeddings, label_clusters
from src.analysis.similarity import compute_similarity_matrix
from src.analysis.visualize import reduce_dimensions, create_scatter_plot, create_heatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_corpus(embed_dir: Path):
    vectors = np.load(embed_dir / "vectors.npy")
    metadata = pd.read_parquet(embed_dir / "metadata.parquet")
    return vectors, metadata


def cmd_search(args, config):
    embed_dir = Path("data/embeddings")
    vectors, metadata = load_corpus(embed_dir)
    embed_cfg = config.get("embedding", {})
    embedder = Embedder(model_name=embed_cfg.get("model", "nomic-ai/nomic-embed-text-v1.5"), device=embed_cfg.get("device", "auto"))
    query_vec = embedder.embed([args.query])[0]
    results = semantic_search(query_vec, vectors, metadata, top_k=args.top_k, filter_tradition=args.tradition)
    print(f"\nSearch: '{args.query}' (top {args.top_k})\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.get('tradition', '?')}] {r.get('title', '?')} (score: {r['score']:.4f})")
        text_preview = r.get('text', '')[:200].replace('\n', ' ')
        print(f"   {text_preview}...\n")


def cmd_cluster(args, config):
    embed_dir = Path("data/embeddings")
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    vectors, metadata = load_corpus(embed_dir)
    analysis_cfg = config.get("analysis", {})
    labels = cluster_embeddings(vectors, min_cluster_size=analysis_cfg.get("hdbscan_min_cluster_size", 10))
    cluster_labels_map = label_clusters(metadata["text"].tolist(), labels)
    metadata["cluster"] = labels
    metadata.to_parquet(embed_dir / "metadata.parquet", index=False)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"\nFound {n_clusters} clusters ({n_noise} noise points)\n")
    for cid, terms in sorted(cluster_labels_map.items()):
        count = (labels == cid).sum()
        print(f"  Cluster {cid} ({count} chunks): {', '.join(terms)}")


def cmd_similarity(args, config):
    embed_dir = Path("data/embeddings")
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    vectors, metadata = load_corpus(embed_dir)
    labels, matrix = compute_similarity_matrix(vectors, metadata, level=args.level)
    np.save(results_dir / f"similarity_{args.level}.npy", matrix)
    print(f"\nSimilarity matrix ({args.level} level):\n")
    header = "".ljust(25) + "  ".join(l[:12].ljust(12) for l in labels)
    print(header)
    for i, label in enumerate(labels):
        row = label[:25].ljust(25) + "  ".join(f"{matrix[i, j]:.3f}".ljust(12) for j in range(len(labels)))
        print(row)
    fig = create_heatmap(labels, matrix, title=f"Similarity by {args.level}")
    out_path = results_dir / f"similarity_{args.level}.html"
    fig.write_html(str(out_path))
    print(f"\nHeatmap saved to {out_path}")


def cmd_visualize(args, config):
    embed_dir = Path("data/embeddings")
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    vectors, metadata = load_corpus(embed_dir)
    analysis_cfg = config.get("analysis", {})
    coords = reduce_dimensions(
        vectors,
        method=args.method,
        n_components=2,
        n_neighbors=analysis_cfg.get("umap_n_neighbors", 15),
        min_dist=analysis_cfg.get("umap_min_dist", 0.1),
    )
    fig = create_scatter_plot(coords, metadata, color_by=args.color_by)
    out_path = results_dir / f"scatter_{args.color_by}_{args.method}.html"
    fig.write_html(str(out_path))
    print(f"Visualization saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze occult text embeddings")
    parser.add_argument("--config", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sp_search = subparsers.add_parser("search", help="Semantic search")
    sp_search.add_argument("query", type=str)
    sp_search.add_argument("--top-k", type=int, default=10)
    sp_search.add_argument("--tradition", type=str, default=None)
    subparsers.add_parser("cluster", help="Cluster embeddings")
    sp_sim = subparsers.add_parser("similarity", help="Compute similarity matrix")
    sp_sim.add_argument("--level", choices=["tradition", "title"], default="tradition")
    sp_viz = subparsers.add_parser("visualize", help="UMAP/t-SNE visualization")
    sp_viz.add_argument("--color-by", default="tradition")
    sp_viz.add_argument("--method", choices=["umap", "tsne"], default="umap")
    args = parser.parse_args()
    config = load_config(args.config)
    commands = {"search": cmd_search, "cluster": cmd_cluster, "similarity": cmd_similarity, "visualize": cmd_visualize}
    commands[args.command](args, config)


if __name__ == "__main__":
    main()

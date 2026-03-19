"""CLI entry point for generating embeddings from chunks."""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import load_config
from src.embedder.embedder import Embedder

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROFILES = ["search", "cluster"]


def embed_profile(profile: str, config: dict, force: bool = False):
    embed_cfg = config["embedding"][profile]
    model_name = embed_cfg["model"]
    batch_size = embed_cfg.get("batch_size", 64)
    device = embed_cfg.get("device", "auto")

    chunks_dir = Path(f"data/chunks/{profile}")
    embed_dir = Path(f"data/embeddings/{profile}")
    by_source_dir = embed_dir / "by_source"
    embed_dir.mkdir(parents=True, exist_ok=True)
    by_source_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(chunks_dir.glob("*.parquet"))
    if not chunk_files:
        print(f"  {profile}: no chunks found in {chunks_dir}")
        return

    print(f"  Loading {model_name}...")
    embedder = Embedder(model_name=model_name, device=device, batch_size=batch_size)

    all_vectors = []
    all_metadata = []

    for chunk_file in tqdm(chunk_files, desc=f"Embedding ({profile})"):
        source_npy = by_source_dir / f"{chunk_file.stem}.npy"
        df = pd.read_parquet(chunk_file)

        if source_npy.exists() and not force:
            vectors = np.load(source_npy)
        else:
            texts = df["text"].tolist()
            vectors = embedder.embed(texts)
            np.save(source_npy, vectors)

        all_vectors.append(vectors)
        all_metadata.append(df)

    combined_vectors = np.concatenate(all_vectors, axis=0)
    combined_metadata = pd.concat(all_metadata, ignore_index=True)

    np.save(embed_dir / "vectors.npy", combined_vectors)
    combined_metadata.to_parquet(embed_dir / "metadata.parquet", index=False)
    embedder.save_model_info(embed_dir / "model_info.json")

    print(f"  {profile}: {combined_vectors.shape[0]} chunks, {combined_vectors.shape[1]} dims -> {embed_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from chunks")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profile", choices=PROFILES + ["all"], default="all")
    parser.add_argument("--force", action="store_true", help="Re-embed all")
    args = parser.parse_args()
    config = load_config(args.config)

    profiles = PROFILES if args.profile == "all" else [args.profile]
    for profile in profiles:
        print(f"\n>>> Embedding profile: {profile}")
        embed_profile(profile, config, force=args.force)


if __name__ == "__main__":
    main()

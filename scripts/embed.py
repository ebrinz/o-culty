"""CLI entry point for generating embeddings from chunks.

Supports checkpointing — saves progress every mega-batch so it can resume
if interrupted.
"""
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
MEGA_BATCH = 10000


def embed_profile(profile: str, config: dict, force: bool = False):
    embed_cfg = config["embedding"][profile]
    model_name = embed_cfg["model"]
    batch_size = embed_cfg.get("batch_size", 64)
    device = embed_cfg.get("device", "auto")
    max_seq_length = embed_cfg.get("max_seq_length")

    chunks_dir = Path(f"data/chunks/{profile}")
    embed_dir = Path(f"data/embeddings/{profile}")
    embed_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = embed_dir / "vectors.npy"
    metadata_path = embed_dir / "metadata.parquet"
    checkpoint_dir = embed_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if vectors_path.exists() and not force:
        print(f"  {profile}: already exists, use --force to re-embed")
        return

    chunk_files = sorted(chunks_dir.glob("*.parquet"))
    if not chunk_files:
        print(f"  {profile}: no chunks found in {chunks_dir}")
        return

    # Load texts only for embedding (metadata saved separately at the end)
    print(f"  Loading {len(chunk_files)} chunk files (text only)...")
    texts = []
    for f in tqdm(chunk_files, desc="Loading chunks"):
        texts.extend(pd.read_parquet(f, columns=["text"])["text"].tolist())
    total = len(texts)
    print(f"  Total chunks: {total}")

    # Check for existing checkpoints — count actual vectors, not batch count
    existing = sorted(checkpoint_dir.glob("batch_*.npy"))
    start_idx = sum(np.load(f).shape[0] for f in existing) if existing else 0
    if existing:
        print(f"  Resuming from checkpoint: {len(existing)} batches ({start_idx} chunks done)")

    # Clear MPS cache before starting
    if device == "auto" or device == "mps":
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("  Cleared MPS cache")

    # Load model
    print(f"  Loading {model_name}...")
    embedder = Embedder(model_name=model_name, device=device, batch_size=batch_size, max_seq_length=max_seq_length)

    # Embed in mega-batches with checkpointing
    remaining = total - start_idx
    n_new_batches = (remaining + MEGA_BATCH - 1) // MEGA_BATCH if remaining > 0 else 0
    next_batch_num = len(existing)
    total_batches = next_batch_num + n_new_batches
    for j in tqdm(range(n_new_batches), desc="Embedding", initial=next_batch_num, total=total_batches):
        i = start_idx + j * MEGA_BATCH
        batch_texts = texts[i:i + MEGA_BATCH]
        vecs = embedder.embed(batch_texts)
        np.save(checkpoint_dir / f"batch_{next_batch_num + j:05d}.npy", vecs)

    # Combine all checkpoints into final output
    print("  Combining checkpoints...")
    all_checkpoints = sorted(checkpoint_dir.glob("batch_*.npy"))
    all_vectors = [np.load(f) for f in all_checkpoints]
    vectors = np.concatenate(all_vectors, axis=0)

    n_chunks, n_dims = vectors.shape
    np.save(vectors_path, vectors)
    del texts, vectors, all_vectors  # free memory before loading full metadata
    print("  Saving metadata...")
    combined = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    combined.to_parquet(metadata_path, index=False)
    embedder.save_model_info(embed_dir / "model_info.json")

    # Clean up checkpoints
    for f in all_checkpoints:
        f.unlink()
    checkpoint_dir.rmdir()

    print(f"  {profile}: {n_chunks} chunks, {n_dims} dims -> {embed_dir}")


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

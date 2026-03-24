"""Embed only new chunk files and append to existing search vectors.

Usage:
    python scripts/embed_incremental.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils import load_config
from src.embedder.embedder import Embedder

PROFILE = "search"
MEGA_BATCH = 10000


def main():
    config = load_config()
    embed_cfg = config["embedding"][PROFILE]

    chunks_dir = Path(f"data/chunks/{PROFILE}")
    embed_dir = Path(f"data/embeddings/{PROFILE}")
    vectors_path = embed_dir / "vectors.npy"
    metadata_path = embed_dir / "metadata.parquet"

    # Load existing metadata to find which chunk files are already embedded
    existing_meta = pd.read_parquet(metadata_path, columns=["id"])
    existing_doc_ids = set(existing_meta["id"].unique())
    print(f"Existing embeddings cover {len(existing_doc_ids)} documents ({len(existing_meta)} chunks)")

    # Find new chunk files (filename stem = doc id)
    all_chunk_files = sorted(chunks_dir.glob("*.parquet"))
    new_chunk_files = [f for f in all_chunk_files if f.stem not in existing_doc_ids]
    print(f"New chunk files to embed: {len(new_chunk_files)}")

    if not new_chunk_files:
        print("Nothing to do.")
        return

    # Load new chunks
    new_texts = []
    new_meta_dfs = []
    for f in tqdm(new_chunk_files, desc="Loading new chunks"):
        df = pd.read_parquet(f)
        new_texts.extend(df["text"].tolist())
        new_meta_dfs.append(df)
    new_meta = pd.concat(new_meta_dfs, ignore_index=True)
    print(f"New chunks to embed: {len(new_texts)}")

    # Embed
    model_name = embed_cfg["model"]
    batch_size = embed_cfg.get("batch_size", 64)
    device = embed_cfg.get("device", "auto")
    max_seq_length = embed_cfg.get("max_seq_length")

    print(f"Loading {model_name}...")
    embedder = Embedder(
        model_name=model_name, device=device,
        batch_size=batch_size, max_seq_length=max_seq_length,
    )

    # Checkpointing: save progress every mega-batch so we can resume
    checkpoint_dir = embed_dir / "incremental_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    existing_ckpts = sorted(checkpoint_dir.glob("batch_*.npy"))
    start_idx = sum(np.load(f).shape[0] for f in existing_ckpts) if existing_ckpts else 0
    if existing_ckpts:
        print(f"Resuming from {len(existing_ckpts)} checkpoints ({start_idx} chunks done)")

    remaining = len(new_texts) - start_idx
    n_batches = (remaining + MEGA_BATCH - 1) // MEGA_BATCH if remaining > 0 else 0
    next_batch_num = len(existing_ckpts)

    for j in tqdm(range(n_batches), desc="Embedding", initial=next_batch_num,
                  total=next_batch_num + n_batches):
        i = start_idx + j * MEGA_BATCH
        batch = new_texts[i:i + MEGA_BATCH]
        vecs = embedder.embed(batch)
        np.save(checkpoint_dir / f"batch_{next_batch_num + j:05d}.npy", vecs)

    # Combine checkpoints
    all_ckpts = sorted(checkpoint_dir.glob("batch_*.npy"))
    new_vectors = np.concatenate([np.load(f) for f in all_ckpts], axis=0)
    print(f"New vectors shape: {new_vectors.shape}")

    # Append to existing
    existing_vectors = np.load(vectors_path)
    print(f"Existing vectors shape: {existing_vectors.shape}")

    total_count = len(existing_vectors) + len(new_vectors)
    combined_vectors = np.concatenate([existing_vectors, new_vectors], axis=0)
    np.save(vectors_path, combined_vectors)
    del existing_vectors, combined_vectors, new_vectors  # free memory
    print(f"Saved combined vectors: {total_count} total")

    existing_full_meta = pd.read_parquet(metadata_path)
    combined_meta = pd.concat([existing_full_meta, new_meta], ignore_index=True)
    combined_meta.to_parquet(metadata_path, index=False)
    print(f"Combined metadata: {len(combined_meta)} rows")

    # Clean up checkpoints
    for f in all_ckpts:
        f.unlink()
    checkpoint_dir.rmdir()
    print("Cleaned up checkpoints.")

    print("Done!")


if __name__ == "__main__":
    main()

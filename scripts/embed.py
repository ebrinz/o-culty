import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import load_config
from src.embedder.embedder import Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from chunks")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Re-embed all")
    args = parser.parse_args()
    config = load_config(args.config)
    chunks_dir = Path("data/chunks")
    embed_dir = Path("data/embeddings")
    by_source_dir = embed_dir / "by_source"
    embed_dir.mkdir(parents=True, exist_ok=True)
    by_source_dir.mkdir(parents=True, exist_ok=True)
    embed_cfg = config.get("embedding", {})
    model_name = embed_cfg.get("model", "nomic-ai/nomic-embed-text-v1.5")
    batch_size = embed_cfg.get("batch_size", 64)
    device = embed_cfg.get("device", "auto")
    embedder = Embedder(model_name=model_name, device=device, batch_size=batch_size)
    chunk_files = sorted(chunks_dir.glob("*.parquet"))
    all_vectors = []
    all_metadata = []
    for chunk_file in chunk_files:
        source_npy = by_source_dir / f"{chunk_file.stem}.npy"
        df = pd.read_parquet(chunk_file)
        if source_npy.exists() and not args.force:
            logger.info(f"Loading cached embeddings: {chunk_file.stem}")
            vectors = np.load(source_npy)
        else:
            texts = df["text"].tolist()
            logger.info(f"Embedding: {chunk_file.stem} ({len(texts)} chunks)")
            vectors = embedder.embed(texts)
            np.save(source_npy, vectors)
        all_vectors.append(vectors)
        all_metadata.append(df)
    if not all_vectors:
        logger.warning("No chunks found to embed")
        return
    combined_vectors = np.concatenate(all_vectors, axis=0)
    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    np.save(embed_dir / "vectors.npy", combined_vectors)
    combined_metadata.to_parquet(embed_dir / "metadata.parquet", index=False)
    embedder.save_model_info(embed_dir / "model_info.json")
    logger.info(f"Done: {combined_vectors.shape[0]} chunks, {combined_vectors.shape[1]} dims")

if __name__ == "__main__":
    main()

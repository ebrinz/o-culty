"""CLI entry point for chunking processed texts into search and cluster chunks."""
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils import load_config
from src.chunker.chunker import chunk_text

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROFILES = ["search", "cluster"]


def main():
    parser = argparse.ArgumentParser(description="Chunk processed texts")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profile", choices=PROFILES + ["all"], default="all",
                        help="Which chunk profile to run (default: all)")
    args = parser.parse_args()
    config = load_config(args.config)
    processed_dir = Path("data/processed")
    meta_files = sorted(processed_dir.glob("*.json"))

    profiles = PROFILES if args.profile == "all" else [args.profile]

    for profile in profiles:
        chunk_cfg = config["chunking"][profile]
        embed_cfg = config["embedding"][profile]
        chunk_size = chunk_cfg["chunk_size_tokens"]
        overlap = chunk_cfg["overlap_tokens"]
        model_name = embed_cfg["model"]

        chunks_dir = Path(f"data/chunks/{profile}")
        chunks_dir.mkdir(parents=True, exist_ok=True)

        total_chunks = 0
        for meta_file in tqdm(meta_files, desc=f"Chunking ({profile}: {chunk_size} tokens)"):
            text_file = meta_file.with_suffix(".txt")
            if not text_file.exists():
                continue

            out_path = chunks_dir / f"{meta_file.stem}.parquet"
            if out_path.exists():
                continue

            metadata = json.loads(meta_file.read_text())
            text = text_file.read_text(encoding="utf-8")

            chunks = chunk_text(
                text=text, metadata=metadata,
                chunk_size=chunk_size, overlap=overlap,
                tokenizer_name=model_name,
            )
            if not chunks:
                continue

            df = pd.DataFrame(chunks)
            df.to_parquet(out_path, index=False)
            total_chunks += len(chunks)

        print(f"  {profile}: {total_chunks} chunks from {len(meta_files)} texts")


if __name__ == "__main__":
    main()

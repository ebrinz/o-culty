import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from src.utils import load_config
from src.chunker.chunker import chunk_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Chunk processed texts")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    processed_dir = Path("data/processed")
    chunks_dir = Path("data/chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_cfg = config.get("chunking", {})
    chunk_size = chunk_cfg.get("chunk_size_tokens", 512)
    overlap = chunk_cfg.get("overlap_tokens", 64)
    model_name = config.get("embedding", {}).get("model", "nomic-ai/nomic-embed-text-v1.5")
    meta_files = sorted(processed_dir.glob("*.json"))
    for meta_file in meta_files:
        text_file = meta_file.with_suffix(".txt")
        if not text_file.exists():
            logger.warning(f"Missing text file for {meta_file.name}")
            continue
        metadata = json.loads(meta_file.read_text())
        text = text_file.read_text(encoding="utf-8")
        chunks = chunk_text(text=text, metadata=metadata, chunk_size=chunk_size, overlap=overlap, tokenizer_name=model_name)
        if not chunks:
            continue
        df = pd.DataFrame(chunks)
        out_path = chunks_dir / f"{metadata['id']}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Chunked: {metadata['id']} -> {len(chunks)} chunks")

if __name__ == "__main__":
    main()

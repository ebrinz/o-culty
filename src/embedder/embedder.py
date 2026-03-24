import json
import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import get_device

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", device: str = "auto", batch_size: int = 64, max_seq_length: int | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = get_device(device)
        logger.info(f"Loading model {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.device == "mps":
            return self._embed_mps(texts)
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=len(texts) > 100,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def _truncate(self, texts: list[str]) -> list[str]:
        """Pre-truncate texts to max_seq_length tokens (nomic's custom tokenizer ignores the ST setting)."""
        max_len = self.model.max_seq_length
        tokenizer = self.model.tokenizer
        truncated = []
        for t in texts:
            ids = tokenizer.encode(t, add_special_tokens=False)
            if len(ids) > max_len - 2:  # leave room for [CLS] and [SEP]
                t = tokenizer.decode(ids[:max_len - 2], skip_special_tokens=True)
            truncated.append(t)
        return truncated

    def _embed_mps(self, texts: list[str]) -> np.ndarray:
        """Embed in sub-batches with periodic MPS cache clearing to avoid fragmentation OOM."""
        import torch
        from tqdm import tqdm
        MPS_CHUNK = 1000  # clear cache every N texts, not every micro-batch
        parts = []
        for i in tqdm(range(0, len(texts), MPS_CHUNK), desc="Batches", leave=False):
            chunk = texts[i:i + MPS_CHUNK]
            vecs = self.model.encode(
                chunk, batch_size=self.batch_size,
                convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False,
            )
            parts.append(vecs.astype(np.float32))
            torch.mps.empty_cache()
        return np.concatenate(parts, axis=0)

    def save_model_info(self, path: Path) -> None:
        info = {"model": self.model_name, "dimensions": self.dimensions, "device": self.device}
        with open(path, "w") as f:
            json.dump(info, f, indent=2)

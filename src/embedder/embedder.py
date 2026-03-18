import json
import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils import get_device

logger = logging.getLogger(__name__)

class Embedder:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", device: str = "auto", batch_size: int = 64):
        self.device = get_device(device)
        self.model_name = model_name
        self.batch_size = batch_size
        logger.info(f"Loading model {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=len(texts) > 100,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def save_model_info(self, path: Path) -> None:
        info = {"model": self.model_name, "dimensions": self.dimensions, "device": self.device}
        with open(path, "w") as f:
            json.dump(info, f, indent=2)

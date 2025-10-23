from __future__ import annotations
from typing import Optional
import torch

from .interfaces import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            *,
            device: Optional[str] = None,
            batch_size: int = 64,
            normalize: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._batch_size = batch_size
        # self_normalize = normalize

        _probe = self._model.encode(["probe"], convert_to_numpy=True, normalize_embeddings=False)
        self._dim = int(_probe.shape[1])

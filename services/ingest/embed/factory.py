from __future__ import annotations
from typing import Literal, Any

from .interfaces import Embedder
from .sbert_embedder import SentenceTransformerEmbedder

EmbedBackend = Literal["sbert"]

def create_embedder(
        backend: EmbedBackend = "sbert",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs: Any,
) -> Embedder:
    if backend == "sbert":
        return SentenceTransformerEmbedder(model_name=model_name, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")

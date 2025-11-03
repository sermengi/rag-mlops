from __future__ import annotations
from typing import Optional, List, Iterable, cast
import torch
import numpy as np

from .interfaces import Embedder


def _batch_iter(xs: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), batch_size):
        yield xs[i : i + batch_size]


class SentenceTransformerEmbedder(Embedder):
    def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            *,
            device: Optional[str] = None,
            batch_size: int = 64,
            normalize: bool = True,
            max_length: int | None = 1024) -> None:
        from sentence_transformers import SentenceTransformer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize

        if max_length is not None:
            cap = int(max_length)
            try:
                self._model.max_seq_length = cap
            except Exception:
                pass

            maybe_setter = getattr(self._model, "set_max_seq_length", None)
            if callable(maybe_setter):
                maybe_setter(cap)

        _probe = self._model.encode(["probe"], convert_to_numpy=True, normalize_embeddings=False)
        self._dim = int(_probe.shape[1])

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vecs: List[np.ndarray] = []

        for batch in _batch_iter(texts, self._batch_size):
            arr = self._model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            if self._normalize:
                denom = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)
                arr = arr / denom

            vecs.append(arr)

        out = np.vstack(vecs) if vecs else np.zeros((0, self._dim), dtype=np.float32)
        return cast(List[List[float]], out.astype(np.float32).tolist())

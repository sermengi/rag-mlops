from __future__ import annotations
from typing import List, Protocol


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...
    @property
    def dimension(self) -> int: ...
    def embed_texts(self, text: List[str]) -> List[List[float]]: ...

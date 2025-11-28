from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Distance = Literal["Cosine", "Dot", "Euclid"]

@dataclass(frozen=True)
class CollectionSpec:
    name: str
    vector_size: int
    distance: Distance = "Cosine"

DEFAULT_COLLECTION = CollectionSpec(
    name="documents",
    vector_size=384,
    distance="Cosine",
)

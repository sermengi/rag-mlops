from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class EmbeddedChunk:
    doc_id: str
    chunk_index: int
    vector: List[float]
    embedding_model: str
    payload: Dict[str, object]

@dataclass(frozen=True)
class EmbedBatchResult:
    doc_id: str
    model_name: str
    vectors: List[EmbeddedChunk]

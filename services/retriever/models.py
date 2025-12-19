from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    top_k: int = 5
    doc_id: Optional[str] = None

@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    chunk_index: int
    text: str
    score: float
    page_start: int | None = None
    page_end: int | None = None
    source: str | None = None
    payload: Dict[str, Any] | None = None

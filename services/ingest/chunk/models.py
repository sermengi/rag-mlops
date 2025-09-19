from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    page_start: int
    page_end: int
    section: Optional[str] = None  # reserved for future semantic/section-aware chunking

@dataclass(frozen=True)
class ChunkedDoc:
    doc_id: str
    chunks: List[Chunk]
    chunk_size: int
    chunk_overlap: int
    tokenizer_name: str

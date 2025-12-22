from __future__ import annotations
from typing import List, Optional, Dict, Any
import re

from services.ingest.embed.interfaces import Embedder
from services.ingest.index.qdrant_client import search as qdrant_search
from qdrant_client import QdrantClient

from .models import RetrievalQuery, RetrievedChunk

_WHITESPACE_RE = re.compile(r"\s+")

class Retriever:
    def __init__(
            self,
            *,
            qdrant: QdrantClient,
            embedder: Embedder,
            collection_name: str,
            ):
        self._qdrant = qdrant
        self._embedder = embedder
        self._collection = collection_name

    def retrieve(self, rq: RetrievalQuery) -> List[RetrievedChunk]:
        qtext = normalize_query_text(rq.text)
        query_vec = self._embedder.embed_texts([qtext])[0]

        filter_eq: Optional[Dict[str, Any]] = None
        if rq.doc_id:
            filter_eq = {"doc_id": rq.doc_id}

        hits = qdrant_search(
            self._qdrant,
            self._collection,
            query_vector=query_vec,
            top_k=rq.top_k,
            filter_eq=filter_eq
        )

        out: List[RetrievedChunk] = []
        for h in hits:
            payload = h.get("payload") or {}
            out.append(
                RetrievedChunk(
                    doc_id=str(payload.get("doc_id", "")),
                    chunk_index=int(payload.get("chunk_index", -1)),
                    text=str(payload.get("text", "")),
                    score=float(h.get("score", 0.0)),
                    page_start=_optional_int(payload.get("page_start")),
                    page_end=_optional_int(payload.get("page_end")),
                    source=_optional_str(payload.get("source")),
                    payload=payload,
                )
            )

        filtered = [o for o in out if o.text and o.text.strip()]
        return filtered

def _optional_int(v: Any) -> Optional[int]:
    try:
        return None if v is None else int(v)
    except Exception:
        return None

def _optional_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def normalize_query_text(text: str) -> str:
    """
    Minimal query normalization:
    - trim ends
    - unify newlines
    - collapse all whitespace runs to a single space
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text

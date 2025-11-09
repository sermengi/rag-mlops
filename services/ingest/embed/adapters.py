from __future__ import annotations
from typing import List, Optional

from .interfaces import Embedder
from .models import EmbeddedChunk, EmbedBatchResult
from ..chunk.models import ChunkedDoc

def embed_chunked_doc(
        embedder: Embedder,
        doc: ChunkedDoc,
        *,
        max_chars: Optional[int] = 1500
        ) -> EmbedBatchResult:
    texts: List[str] = []

    for c in doc.chunks:
        text = c.text
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]
        texts.append(text)

    vectors = embedder.embed_texts(texts)

    embedded_chunks: List[EmbeddedChunk] = []
    for c, v, truncated_text in zip(doc.chunks, vectors, texts):
        payload = {
            "doc_id": c.doc_id,
            "chunk_index": c.chunk_index,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "token_count": c.token_count,
            "text": truncated_text,
        }
        embedded_chunks.append(
            EmbeddedChunk(
                doc_id=c.doc_id,
                chunk_index=c.chunk_index,
                vector=v,
                embedding_model=embedder.model_name,
                payload=payload,
            )
        )

    return EmbedBatchResult(
        doc_id=doc.doc_id,
        model_name=embedder.model_name,
        vectors=embedded_chunks,
    )

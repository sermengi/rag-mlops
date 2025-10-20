from __future__ import annotations
from typing import List

from .tokenizer import Tokenizer
from ..pdf_reader import RawDoc
from .models import Chunk, ChunkedDoc
from .pager import concat_pages, char_span_to_page_span
from .tokenizer import get_tokenizer


MIN_CHARS = 50  # drop very small/boilerplate chunks
MAX_CHARS = 4000  # safety cap after decode

def _build_token_to_char_index(tokens: List[int], tok: Tokenizer, full_text: str) -> List[int]:
    char_index: List[int] = [0]
    buf_tokens: List[int] = []
    buf_str = ""

    for i, t in enumerate(tokens, start=1):
        buf_tokens.append(t)
        new_str = tok.decode([t])
        buf_str += new_str
        char_index.append(len(buf_str))

    return char_index


def _window_token_bounds(n_tokens: int, chunk_size: int, overlap: int) -> List[tuple[int, int]]:
    assert 0 < overlap < chunk_size, "overlap must be greater than 0 and smaller than chunk size"
    step = chunk_size - overlap
    bounds: List[tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + chunk_size, n_tokens)
        bounds.append((start, end))
        if end == n_tokens:
            break
        start += step

    return bounds


def make_chunks_fixed(full_text: str, tok: Tokenizer, chunk_size: int, overlap: int) -> List[tuple[int, int, str, int]]:
    tokens = tok.encode(full_text)
    if not tokens:
        return []
    char_index = _build_token_to_char_index(tokens, tok, full_text)
    windows = _window_token_bounds(len(tokens), chunk_size, overlap)
    out: List[tuple[int, int, str, int]] = []
    for (ts, te) in windows:
        char_start = char_index[ts]
        char_end = char_index[te]
        snippet = tok.decode(tokens[ts: te])[:MAX_CHARS]
        token_count = te - ts
        if len(snippet.strip()) < MIN_CHARS:
            continue
        out.append((char_start, char_end, snippet, token_count))

    return out


def chunk_rawdoc(
        raw: RawDoc,
        tokenizer_name: str = "cl100k_base",
        chunk_size: int = 800,
        overlap: int = 160,
    ) -> ChunkedDoc:
    full_text, page_spans = concat_pages(raw.pages)
    tok = get_tokenizer(tokenizer_name)
    windows = make_chunks_fixed(full_text, tok, chunk_size, overlap)

    chunks: List[Chunk] = []
    for idx, (c_start, c_end, snippet, tcount) in enumerate(windows):
        p_start, p_end = char_span_to_page_span(c_start, c_end, page_spans)
        snippet = f"[Pages {p_start}-{p_end}] " + snippet
        chunks.append(
            Chunk(
                doc_id=raw.doc_id,
                chunk_index=idx,
                text=snippet,
                token_count=tcount,
                char_start=c_start,
                char_end=c_end,
                page_start=p_start,
                page_end=p_end,
                section=None,
            )
        )

    return ChunkedDoc(
        doc_id=raw.doc_id,
        chunks=chunks,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        tokenizer_name=tok.name,
    )

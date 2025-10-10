from __future__ import annotations
from typing import List

from .tokenizer import Tokenizer


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

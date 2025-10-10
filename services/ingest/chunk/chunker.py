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

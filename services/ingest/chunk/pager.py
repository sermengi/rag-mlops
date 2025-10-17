from __future__ import annotations
from typing import List, Optional, Tuple

from services.ingest.pdf_reader import PageText

PAGE_BREAK = "\n\n[PAGE_BREAK]\n\n"

def concat_pages(pages: List[PageText]) -> tuple[str, List[tuple[int, int]]]:
    parts: List[str] = []
    spans: List[tuple[int, int]] = []
    cursor = 0
    for i, p in enumerate(pages):
        start = cursor
        parts.append(p.text.strip())
        cursor += len(p.text)
        spans.append((start, cursor))
        if i < len(pages) - 1:
            parts.append(PAGE_BREAK)
            cursor += len(PAGE_BREAK)
    return "".join(parts), spans

def char_span_to_page_span(char_start: int, char_end: int, page_spans: List[Tuple[int, int]]) -> Tuple[int, int]:
    first: Optional[int] = None
    last: Optional[int] = None

    for idx, (p_start, p_end) in enumerate(page_spans, start=1):
        if not (char_end <= p_start or char_start >= p_end):
            first = idx if first is None else first
            last = idx

    if first is None:
        first = last = 1

    assert first is not None and last is not None
    return first, last

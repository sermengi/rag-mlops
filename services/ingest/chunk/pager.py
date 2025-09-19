from __future__ import annotations
from typing import List

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

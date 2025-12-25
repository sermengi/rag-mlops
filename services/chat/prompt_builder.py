from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

from services.retriever.models import RetrievedChunk

@dataclass(frozen=True)
class PromptArtifacts:
    messages: List[Dict[str, str]]
    sources: List[Dict[str, object]]


class PromptBuilder:
    def __init__(self, *, max_content_chars: int = 12_000) -> None:
        self._max_content_chars = max_content_chars

    def _system_prompt(self) -> str:
        return (
            "You are a helpful assistant.\n"
            "Answer the user's question using ONLY the provided context.\n"
            "If the context does not contain the answer, say you don't know.\n"
            "Cite sources using the bracketed labels like [S1], [S2].\n"
            "Do not invent citations.\n"
        )

    def _user_prompt(self, *, question: str, context: str) -> str:
        return (
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Answer (with citations like [S1]):"
        )

    def _format_context(self, chunks: List[RetrievedChunk]) -> tuple[str, List[Dict[str, object]]]:
        sources: List[Dict[str, object]] = []
        blocks: List[str] = []
        total = 0

        for i, c in enumerate(chunks, start=1):
            label = f"S[{i}]"

            sources.append(
                {
                    "label": label,
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "source": c.source,
                    "score": c.score,
                }
            )

            header = f"[{label}] doc_id={c.doc_id} chunk={c.chunk_index}"
            if c.page_start is not None and c.page_end is not None:
                header += f" pages={c.page_start}-{c.page_end}"
            if c.source:
                header += f" source={c.source}"

            text = (c.text or "").strip()

            block = f"{header}\n{text}"
            if total + len(block) > self._max_content_chars:
                break

            blocks.append(block)
            total += len(block) + 2

        context = "\n\n".join(blocks).strip()
        return context, sources

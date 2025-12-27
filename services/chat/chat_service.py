from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Literal

from services.retriever import Retriever, RetrievalQuery
from services.chat.prompt_builder import PromptBuilder
from services.llm.openai_compat import OpenAICompatibleClient

import re

_CITATION_RE = re.compile(r"\[(S\d+)\]")

CitationPolicy = Literal["off", "strip", "strict"]

def extract_citation_labels(text: str) -> Set[str]:
    return set(_CITATION_RE.findall(text))

def remove_invalid_citations(text: str, *, allowed: Set[str]) -> str:
    def repl(m: re.Match[str]) -> str:
        label = m.group(1)
        return m.group(0) if label in allowed else ""
    return _CITATION_RE.sub(repl, text)

@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: List[Dict[str, object]]
    retrived: int
    invalid_citations: List[str] = field(default_factory=list)

class ChatService:
    def __init__(
            self,
            *,
            retriever: Retriever,
            prompt_builder: PromptBuilder,
            llm: OpenAICompatibleClient,
            citation_policy: CitationPolicy = "strip"
        ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._citation_policy = citation_policy

    def chat(
            self,
            *,
            question: str,
            top_k: int = 5,
            doc_id: Optional[str] = None,
        ) -> ChatResult:
        chunks = self._retriever.retrieve(RetrievalQuery(text=question, top_k=top_k, doc_id=doc_id))

        artifacts = self._prompt_builder.build(question=question, chunks=chunks)

        resp = self._llm.chat(messages=artifacts.messages, temperature=0.2, max_tokens=500)
        answer = resp.text

        allowed: Set[str] = {
            v
            for s in artifacts.sources
            for (k, v) in s.items()
            if k == "label" and isinstance(v, str)}
        used: Set[str] = extract_citation_labels(answer)
        invalid: List[str] = sorted(used - allowed)

        if invalid and self._citation_policy != "off":
            if self._citation_policy == "strict":
                raise ValueError(f"Answer contained invalid citations: {invalid}. Allowed: {sorted(allowed)}")
            elif self._citation_policy == "strip":
                answer = remove_invalid_citations(answer, allowed=allowed)

        return ChatResult(
            answer=resp.text,
            sources=artifacts.sources,
            retrived=len(chunks),
            invalid_citations=invalid,
            )

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict

from services.retriever import Retriever, RetrievalQuery
from services.chat.prompt_builder import PromptBuilder
from services.llm.openai_compat import OpenAICompatibleClient

@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: List[Dict[str, object]]
    retrived: int

class ChatService:
    def __init__(
            self,
            *,
            retriever: Retriever,
            prompt_builder: PromptBuilder,
            llm: OpenAICompatibleClient,
        ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm = llm

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

        return ChatResult(answer=resp.text, sources=artifacts.sources, retrived=len(chunks))

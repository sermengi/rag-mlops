from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import os
import httpx

@dataclass(frozen=True)
class LLMResponse:
    text: str
    raw: Dict[str, Any]

class OpenAICompatibleClient:
    def __init__(
            self,
            *,
            base_url: str,
            api_key: str,
            model: str,
            timeout_s: float = 60.0
        ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_s

    def chat(
            self,
            messages: List[Dict[str, str]],
            *,
            temperature: float = 0.2,
            max_tokens: int = 500,
        ) -> LLMResponse:
        url = f"{self._base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        text = data["choices"][0]["message"]["content"]
        return LLMResponse(text=text, raw=data)


def client_from_env() -> OpenAICompatibleClient:
    base_url = os.environ["OPENAI_BASE_URL"]
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=model)

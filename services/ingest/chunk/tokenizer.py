from __future__ import annotations
from typing import List, Protocol, runtime_checkable, cast


@runtime_checkable
class Tokenizer(Protocol):
    name: str
    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...


class _EncProtocol(Protocol):
    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...


class _TikTok:
    def __init__(self, enc: _EncProtocol, name: str) -> None:
        self._enc: _EncProtocol = enc
        self.name: str = name

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self._enc.decode(tokens)


def get_tokenizer(name: str = "cl100k_base") -> Tokenizer:
    import tiktoken
    # tiktoken is untyped; tell mypy what interface we rely on
    enc = cast(_EncProtocol, tiktoken.get_encoding(name))
    return _TikTok(enc, name)

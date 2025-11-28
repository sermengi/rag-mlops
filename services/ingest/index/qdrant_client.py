from __future__ import annotations
from typing import Optional
from qdrant_client import QdrantClient


def connect_qdrant(
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        https: bool = False
) -> QdrantClient:
    if https:
        return QdrantClient(url=f"https://{host}", api_key=api_key, prefer_grpc=False)
    return QdrantClient(host=host, port=port, api_key=api_key, prefer_grpc=False)

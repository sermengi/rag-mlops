from __future__ import annotations
from typing import Optional
from qdrant_client import QdrantClient

from qdrant_client.models import Distance, VectorParams

from .schemas import CollectionSpec


def connect_qdrant(
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        https: bool = False
) -> QdrantClient:
    if https:
        return QdrantClient(url=f"https://{host}", api_key=api_key, prefer_grpc=False)
    return QdrantClient(host=host, port=port, api_key=api_key, prefer_grpc=False)

def ensure_collection(client: QdrantClient, spec: CollectionSpec) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if spec.name in existing:
        return

    client.create_collection(
        collection_name=spec.name,
        vectors_config=VectorParams(size=spec.vector_size, distance=Distance(spec.distance))
    )

    for field, schema_type in [("doc_id", "keyword"), ("source", "keyword")]:
        try:
            client.create_payload_index(
                collection_name=spec.name,
                field_name=field,
                field_schema=schema_type
            )
        except Exception:
            pass

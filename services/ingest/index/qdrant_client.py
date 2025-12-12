from __future__ import annotations
from typing import Any, List, Iterable, Dict, Optional
from qdrant_client import QdrantClient
import time
import uuid

from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Condition, FieldCondition, MatchValue, Filter
)

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

def make_point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{chunk_index}"))

def upsert_embedded_chunks(
        client: QdrantClient,
        collection: str,
        embedded_chunks: Iterable[Dict[str, Any]],
        *,
        batch_size: int = 128,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
) -> int:
    buf: List[PointStruct] = []
    count = 0

    def flush() -> None:
        nonlocal buf, count
        if not buf:
            return
        for attempt in range(max_retries):
            try:
                client.upsert(collection_name=collection, points=buf)
                count += len(buf)
                buf = []
                return
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_backoff_s * (2**attempt))

    for item in embedded_chunks:
        pid = make_point_id(item["doc_id"], int(item["chunk_index"]))
        vec = item["vector"]
        payload = dict(item.get("payload", {}))
        payload.setdefault("doc_id", item["doc_id"])
        payload.setdefault("chunk_index", item["chunk_index"])
        buf.append(PointStruct(id=pid, vector=vec, payload=payload))
        if len(buf) >= batch_size:
            flush()
    flush()
    return count

def search(
        client: QdrantClient,
        collection: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_eq: Optional[Dict[str, Any]] = None,
        ) -> List[Dict[str, Any]]:
    qfilter = None
    if filter_eq:
        conditions: List[Condition] = []
        for k, v in filter_eq.items():
            conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
        qfilter = Filter(must=conditions)

    res = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
    )
    return [{"id": r.id, "score": r.score, "payload": r.payload} for r in res]

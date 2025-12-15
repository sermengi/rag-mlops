from services.ingest.index import (
    connect_qdrant,
    ensure_collection,
    upsert_embedded_chunks,
    search,
)

from services.ingest.index.schemas import CollectionSpec

def test_qdrant_upsert_and_search_filter_by_doc_id() -> None:
    client = connect_qdrant()
    spec = CollectionSpec(name="test_docs", vector_size=3, distance="Cosine")
    ensure_collection(client, spec)

    points = [
        # doc: demo
        {
            "doc_id": "demo",
            "chunk_index": 0,
            "vector": [1.0, 0.0, 0.0],
            "payload": {"doc_id": "demo", "chunk_index": 0, "text": "alpha", "source": "file://demo.pdf"},
        },
        {
            "doc_id": "demo",
            "chunk_index": 1,
            "vector": [0.0, 1.0, 0.0],
            "payload": {"doc_id": "demo", "chunk_index": 1, "text": "beta", "source": "file://demo.pdf"},
        },
        # doc: other
        {
            "doc_id": "other",
            "chunk_index": 0,
            "vector": [1.0, 0.0, 0.0],
            "payload": {"doc_id": "other", "chunk_index": 0, "text": "alpha-other", "source": "file://other.pdf"},
        },
    ]
    upsert_embedded_chunks(client, spec.name, points, batch_size=64)

    res_no_filter = search(client, spec.name, [1.0, 0.0, 0.0], top_k=3)
    assert len(res_no_filter) >= 1
    assert any(r["payload"]["doc_id"] == "demo" for r in res_no_filter)
    assert any(r["payload"]["doc_id"] == "other" for r in res_no_filter)

    res_demo_only = search(
        client,
        spec.name,
        [1.0, 0.0, 0.0],
        top_k=3,
        filter_eq={"doc_id": "demo"},
    )
    assert len(res_demo_only) >= 1
    assert all(r["payload"]["doc_id"] == "demo" for r in res_demo_only)

    res_other_only = search(
        client,
        spec.name,
        [1.0, 0.0, 0.0],
        top_k=3,
        filter_eq={"doc_id": "other"},
    )
    assert len(res_other_only) >= 1
    assert all(r["payload"]["doc_id"] == "other" for r in res_other_only)

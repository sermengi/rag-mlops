from services.ingest.index import connect_qdrant, ensure_collection, DEFAULT_COLLECTION, upsert_embedded_chunks, search

def test_qdrant_upsert_and_search() -> None:
    client = connect_qdrant()
    spec = DEFAULT_COLLECTION
    spec = type(spec)(name="test_docs", vector_size=3, distance="Cosine")
    ensure_collection(client, spec)

    points = [
        {
            "doc_id": "demo",
            "chunk_index": 0,
            "vector": [1.0, 0.0, 0.0],
            "payload": {"doc_id": "demo", "chunk_index": 0, "text": "alpha"},
        },
        {
            "doc_id": "demo",
            "chunk_index": 1,
            "vector": [0.0, 1.0, 0.0],
            "payload": {"doc_id": "demo", "chunk_index": 1, "text": "beta"},
        },
    ]
    upsert_embedded_chunks(client, spec.name, points, batch_size=64)

    res = search(client, spec.name, [1.0, 0.0, 0.0], top_k=1, filter_eq={"doc_id": "demo"})
    assert len(res) == 1
    assert res[0]["payload"]["chunk_index"] == 0

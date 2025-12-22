
from services.ingest.index import connect_qdrant, ensure_collection, upsert_embedded_chunks
from services.ingest.index.schemas import CollectionSpec
from services.ingest.embed.factory import create_embedder
from services.retriever import Retriever, RetrievalQuery

def test_retriever_returns_chunks_with_filter() -> None:
    client = connect_qdrant()

    embedder = create_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    spec = CollectionSpec(name="test_retriever", vector_size=embedder.dimension, distance="Cosine")
    ensure_collection(client, spec)

    def upsert_doc(doc_id: str, texts: list[str]) -> None:
        vecs = embedder.embed_texts(texts)
        points = []
        for i, (t, v) in enumerate(zip(texts, vecs)):
            points.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "vector": v,
                "payload": {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": t,
                    "source": f"file://{doc_id}.pdf",
                    "page_start": 1,
                    "page_end": 1,
                }
            })
        upsert_embedded_chunks(client, spec.name, points)

    upsert_doc("demo", ["cats are great pets", "dogs are loyal animals"])
    upsert_doc("other", ["pizza is tasty", "pasta is popular in italy"])

    retriever = Retriever(qdrant=client, embedder=embedder, collection_name=spec.name)

    hits = retriever.retrieve(RetrievalQuery(text="tell me about cats", top_k=3, doc_id="demo"))
    assert len(hits) >= 1
    assert all(h.doc_id == "demo" for h in hits)
    assert any("cat" in h.text.lower() for h in hits)

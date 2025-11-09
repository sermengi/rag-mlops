from services.ingest.chunk.models import Chunk, ChunkedDoc
from services.ingest.embed.factory import create_embedder
from services.ingest.embed.adapters import embed_chunked_doc

def test_embedder_smoke() -> None:
    # Fake small chunked doc
    chunks = [
        Chunk(doc_id="demo", chunk_index=0, text="Hello world.", token_count=3,
              char_start=0, char_end=12, page_start=1, page_end=1),
        Chunk(doc_id="demo", chunk_index=1, text="This is another chunk about embeddings.",
              token_count=7, char_start=13, char_end=55, page_start=1, page_end=1),
    ]
    cd = ChunkedDoc(doc_id="demo", chunks=chunks, chunk_size=800, chunk_overlap=160, tokenizer_name="cl100k_base")

    embedder = create_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    result = embed_chunked_doc(embedder, cd)

    assert result.doc_id == "demo"
    assert result.model_name.startswith("sentence-transformers")
    assert len(result.vectors) == len(chunks)
    # shape & types
    first = result.vectors[0]
    assert isinstance(first.vector, list) and len(first.vector) == embedder.dimension

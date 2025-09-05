from fastapi import FastAPI, HTTPException, status, Depends
from qdrant_client import QdrantClient
import os
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional


class Health(BaseModel):
    status: str

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    app.state.qdrant = QdrantClient(url=qdrant_url, timeout=2)
    try:
        yield
    finally:
        client: Optional[QdrantClient] = getattr(app.state, "qdrant", None)
        if client:
            client.close()

app = FastAPI(title="rag-mlops API", version="0.1.0", lifespan=lifespan)

def get_qdrant() -> QdrantClient:
    client: Optional[QdrantClient] = getattr(app.state, "qdrant", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant client not ready"
        )
    return client

@app.get("/health", response_model=Health)
def health_check() -> Health:
    return Health(status="ok")

@app.get("/qdrant/health", response_model=Health)
def qdrant_health(qdrant: QdrantClient = Depends(get_qdrant)) -> Health:
    try:
        qdrant.get_collections()
        return Health(status="ok")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant not reachable"
        )

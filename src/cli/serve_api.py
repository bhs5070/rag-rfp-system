from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.rag_rfp.generate.generator import RAGGenerator
from src.rag_rfp.index.chroma_store import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR
from src.rag_rfp.retrieve.retriever import ChunkRetriever


app = FastAPI(title="RAG-RFP API")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AskRequest(BaseModel):
    query: str = Field(..., description="User question")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class ContextItem(BaseModel):
    doc_id: str | None = None
    chunk_index: int | None = None
    chunk_id: str | None = None
    page: int | None = None
    n_chars: int | None = None
    score: float | None = None
    text: str


class AskResponse(BaseModel):
    answer: str
    model: str
    retrieval: Dict[str, Any]
    contexts: List[ContextItem]


@lru_cache(maxsize=1)
def get_retriever() -> ChunkRetriever:
    return ChunkRetriever(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", str(DEFAULT_PERSIST_DIR)),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
    )


@lru_cache(maxsize=1)
def get_generator() -> RAGGenerator:
    return RAGGenerator(retriever=get_retriever())


@app.get("/health")
def health() -> Dict[str, Any]:
    retriever = get_retriever()
    return {
        "status": "ok",
        "embedding_model": retriever.embedding_model,
        "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        "collection_name": retriever.collection_name,
        "persist_dir": str(retriever.persist_dir),
        "chunk_count": retriever.collection.count(),
    }


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    generator = get_generator()
    answer_obj = generator.ask(request.query, top_k=request.top_k)

    contexts = [
        ContextItem(
            doc_id=context.get("doc_id"),
            chunk_index=context.get("chunk_index"),
            chunk_id=context.get("chunk_id"),
            page=context.get("page"),
            n_chars=context.get("n_chars"),
            score=context.get("score"),
            text=context.get("text", ""),
        )
        for context in answer_obj.contexts
    ]

    return AskResponse(
        answer=answer_obj.answer,
        model=generator.model,
        retrieval={
            "chunking": "semantic",
            "vector_db": "ChromaDB",
            "embedding_model": get_retriever().embedding_model,
            "top_k": request.top_k,
        },
        contexts=contexts,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

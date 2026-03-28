from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from src.rag_rfp.index.chroma_store import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR


class ChunkRetriever:
    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        self.embedding_model = embedding_model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
        self.persist_dir = Path(persist_dir or os.getenv("CHROMA_PERSIST_DIR", str(DEFAULT_PERSIST_DIR)))
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", DEFAULT_COLLECTION_NAME)

        self.client = OpenAI(api_key=api_key)
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.chroma_client.get_collection(self.collection_name)

    def _embed_query(self, question: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[question],
        )
        return response.data[0].embedding

    def embed_queries(self, questions: List[str], batch_size: int = 128) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(questions), batch_size):
            batch = questions[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def search_by_embedding(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        contexts: List[Dict[str, Any]] = []
        for index, (text, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            metadata = metadata or {}
            raw_distance = float(distance) if distance is not None else 0.0
            contexts.append(
                {
                    "text": text,
                    "doc_id": metadata.get("doc_id"),
                    "chunk_index": metadata.get("chunk_index", index),
                    "chunk_id": metadata.get("chunk_id"),
                    "page": metadata.get("page"),
                    "n_chars": metadata.get("size", len(text)),
                    "score": 1.0 - raw_distance,
                    "distance": raw_distance,
                    "retrieval_mode": "chromadb_dense_semantic",
                    "raw_metadata": metadata,
                }
            )

        return contexts

    def search(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self._embed_query(question)
        return self.search_by_embedding(query_embedding, top_k=top_k)

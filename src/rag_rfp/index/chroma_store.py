from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "advanced_chunks" / "chunks_semantic.jsonl"
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "eval" / "embedding_cache" / "openai-text-embedding-3-large__semantic.npy"
DEFAULT_PERSIST_DIR = PROJECT_ROOT / "data" / "vectorstores" / "chroma_semantic"
DEFAULT_COLLECTION_NAME = "rfp_semantic_large"


def load_chunks(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


def chunk_metadata(chunk: Dict, chunk_index: int) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "doc_id": str(chunk.get("doc_id", "")),
        "filename": str(chunk.get("filename", "")),
        "chunk_id": str(chunk.get("chunk_id", f"chunk-{chunk_index}")),
        "chunk_type": str(chunk.get("chunk_type", "")),
        "chunk_index": int(chunk_index),
        "size": int(chunk.get("size", len(str(chunk.get("text", ""))))),
    }
    if chunk.get("page") is not None:
        metadata["page"] = int(chunk["page"])
    if chunk.get("aspect") is not None:
        metadata["aspect"] = str(chunk["aspect"])
    if chunk.get("section") is not None:
        metadata["section"] = str(chunk["section"])
    return metadata


class ChromaSemanticIndex:
    def __init__(self, persist_dir: Path = DEFAULT_PERSIST_DIR, collection_name: str = DEFAULT_COLLECTION_NAME) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

    def get_or_create_collection(self) -> Collection:
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def build_from_embeddings(
        self,
        chunks: Sequence[Dict],
        embeddings: np.ndarray,
        batch_size: int = 500,
        reset: bool = True,
    ) -> Collection:
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) and embedding count ({len(embeddings)}) must match.")

        if reset:
            self.reset_collection()
        collection = self.get_or_create_collection()

        ids = [str(chunk.get("chunk_id", f"chunk-{index}")) for index, chunk in enumerate(chunks)]
        documents = [str(chunk.get("text", "")) for chunk in chunks]
        metadatas = [chunk_metadata(chunk, index) for index, chunk in enumerate(chunks)]
        vectors = embeddings.astype(np.float32).tolist()

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=vectors[start:end],
            )

        return collection

    def stats(self) -> Dict[str, object]:
        collection = self.get_or_create_collection()
        return {
            "persist_dir": str(self.persist_dir),
            "collection_name": self.collection_name,
            "count": collection.count(),
        }


def build_semantic_chroma_index(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    embeddings_path: Path = DEFAULT_EMBEDDINGS_PATH,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 500,
    reset: bool = True,
) -> Dict[str, object]:
    chunks = load_chunks(chunks_path)
    embeddings = load_embeddings(embeddings_path)
    index = ChromaSemanticIndex(persist_dir=persist_dir, collection_name=collection_name)
    collection = index.build_from_embeddings(chunks, embeddings, batch_size=batch_size, reset=reset)

    return {
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "chunk_count": len(chunks),
        "vector_count": collection.count(),
        "chunks_path": str(chunks_path),
        "embeddings_path": str(embeddings_path),
    }

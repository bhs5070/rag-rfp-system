from __future__ import annotations

import json
from pathlib import Path

import typer

from src.rag_rfp.index.chroma_store import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_PERSIST_DIR,
    build_semantic_chroma_index,
)


app = typer.Typer(help="Build a ChromaDB index from semantic chunks and cached embeddings.")


@app.command()
def main(
    chunks_path: Path = typer.Option(DEFAULT_CHUNKS_PATH, help="Path to semantic chunk JSONL."),
    embeddings_path: Path = typer.Option(DEFAULT_EMBEDDINGS_PATH, help="Path to cached semantic embeddings (.npy)."),
    persist_dir: Path = typer.Option(DEFAULT_PERSIST_DIR, help="ChromaDB persist directory."),
    collection_name: str = typer.Option(DEFAULT_COLLECTION_NAME, help="Chroma collection name."),
    batch_size: int = typer.Option(500, help="Upsert batch size."),
    reset: bool = typer.Option(True, help="Recreate the collection before indexing."),
) -> None:
    payload = build_semantic_chroma_index(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        persist_dir=persist_dir,
        collection_name=collection_name,
        batch_size=batch_size,
        reset=reset,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()

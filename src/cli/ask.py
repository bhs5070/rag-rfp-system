# src/rag_rfp/ask.py
import typer
from pathlib import Path

from rag_rfp.retrieve.retriever import ChunkRetriever
from rag_rfp.generate.generator import RAGGenerator


app = typer.Typer()


@app.command()
def ask(
    question: str = typer.Argument(..., help="사용자 질문"),
    top_k: int = typer.Option(5, help="검색할 상위 chunk 개수"),
):
    """
    Gemma-2-2B-IT + BGE-M3 + FAISS 기반 RAG 질의 응답.
    """

    print("\n[1] Loading retriever (FAISS index + metadata)...")
    retriever = ChunkRetriever()

    print("[2] Loading Gemma-2-2B-IT model...")
    generator = RAGGenerator(
        retriever=retriever,
        model_id="google/gemma-2-2b-it",  # 정확한 모델 ID
        top_k=top_k
    )

    print("\n[3] Running RAG...")
    result = generator.ask(question, top_k=top_k)

    print("\n\n===== 📘 RAG Answer =====")
    print(result.answer)

    print("\n\n===== 📄 Used Contexts (Top-k) =====")
    for c in result.contexts:
        print(f"- doc: {c['doc_id']}, chunk: {c['chunk_index']}, score={c['score']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    app()

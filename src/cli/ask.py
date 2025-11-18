# src/cli/ask.py

from __future__ import annotations
import sys

from rag_rfp.retrieve.retriever import ChunkRetriever
from rag_rfp.generate.generator import RAGGenerator


def main() -> None:
    print("=== RAG CLI (chunks_512_64_final 기반) ===")
    print("종료하려면 빈 줄 또는 Ctrl+C 를 입력하세요.\n")

    retriever = ChunkRetriever()
    generator = RAGGenerator(retriever=retriever)

    while True:
        try:
            question = input("질문 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not question:
            print("종료합니다.")
            break

        answer_obj = generator.ask(question)

        print("\n[답변]\n")
        print(answer_obj.answer)

        print("\n[사용된 컨텍스트 요약]\n")
        for ctx in answer_obj.contexts[:3]:
            print(
                f"- doc={ctx.get('doc_id')}, chunk={ctx.get('chunk_index')}, "
                f"chars={ctx.get('n_chars')}, score={ctx.get('score'):.4f}"
            )
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
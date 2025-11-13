import os, json
import typer
from loguru import logger
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

from src.rag_rfp.index.vectordb import FaissIndex
from src.rag_rfp.index.embed import embed_texts
from src.rag_rfp.retrieve.rerank import CrossEncoderReranker
from src.rag_rfp.generate.generator import generate_answer

app = typer.Typer()
load_dotenv()  # .env 있으면 로드

class AskArgs(BaseModel):
    q: str
    top_k: int = 8
    rerank: bool = True
    index_dir: str = os.getenv("VECTOR_DIR", "./vectorstore")

@app.command()
def main(
    q: str = typer.Option(..., help="질문"),
    top_k: int = typer.Option(8, help="검색 개수"),
    rerank: bool = typer.Option(True, help="리랭크 여부"),
    index_dir: str = typer.Option(os.getenv("VECTOR_DIR", "./vectorstore"), help="인덱스 경로")
):
    args = AskArgs(q=q, top_k=top_k, rerank=rerank, index_dir=index_dir)
    index = FaissIndex.load(args.index_dir)
    qv = embed_texts([args.q])[0]
    hits: List[Dict] = index.search(qv, k=args.top_k)

    if args.rerank and hits:
        reranker = CrossEncoderReranker()
        hits = reranker.rerank(args.q, hits, top_k=args.top_k)

    answer = generate_answer(args.q, hits)
    print("\n=== Answer ===\n")
    print(answer)
    print("\n=== Contexts ===\n")
    for h in hits:
        print(json.dumps({"doc_id": h.get("doc_id"), "page": h.get("page"), "score": h.get("re_score", h.get("score"))}, ensure_ascii=False))

if __name__ == "__main__":
    app()

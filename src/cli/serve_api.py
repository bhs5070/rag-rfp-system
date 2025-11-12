from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json, os
from pathlib import Path
from src.rag_rfp.index.vectordb import FaissIndex
from src.rag_rfp.index.embed import embed_texts
from src.rag_rfp.retrieve.rerank import CrossEncoderReranker
from src.rag_rfp.generate.generator import generate_answer

app = FastAPI(title="RAG-RFP API")

class AskReq(BaseModel):
    query: str
    top_k: int = 8
    rerank: bool = True

IDX_DIR = Path(os.getenv("VECTOR_DIR", "./vectorstore"))
index = FaissIndex.load(str(IDX_DIR))
reranker = CrossEncoderReranker()

@app.post("/ask")
def ask(req: AskReq):
    qv = embed_texts([req.query])[0]
    hits = index.search(qv, k=req.top_k)
    if req.rerank:
        hits = reranker.rerank(req.query, hits, top_k=req.top_k)
    answer = generate_answer(req.query, hits)
    return {"answer": answer, "contexts": [{"doc_id":h["doc_id"],"page":h["page"],"score":h.get("re_score",h.get("score"))} for h in hits]}

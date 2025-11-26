import os
import json
from typing import Dict, List

from retriever_core import RFPRetrieverCore
from lc_custom_retriever import CustomRFPRetriever
from evaluate_generator import evaluate_one  # /eval 에서 사용

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


###########################################
# CONFIG
###########################################
FAISS_INDEX_PATH = "/home/bhs1581/rag-rfp-system/chunking/vector_db/vectordb_multi_aspect_index.faiss"
CHUNK_PATH = "/home/bhs1581/rag-rfp-system/chunking/chunks/chunks_multi_aspect (1).jsonl"

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


###########################################
# LOAD INDEX + CHUNKS
###########################################

def load_faiss(path):
    idx = faiss.read_index(path)
    print(f"✅ FAISS loaded: {idx.ntotal} vectors")
    return idx

def load_chunks(path):
    texts, mapping = [], {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            texts.append(obj["text"])
            mapping[i] = obj["doc_id"]
    print(f"✅ Loaded {len(texts)} chunks")
    return texts, mapping


print("\n=== Loading Vector DB ===")
index = load_faiss(FAISS_INDEX_PATH)
chunk_texts, chunk_map = load_chunks(CHUNK_PATH)


###########################################
# LOAD RERANKER
###########################################

print("\n=== Loading Reranker ===")
reranker_model_name = "BAAI/bge-reranker-base"
tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Reranker on {device}")


###########################################
# INIT RETRIEVAL CORE
###########################################

core = RFPRetrieverCore(
    faiss_index=index,
    chunk_texts=chunk_texts,
    chunk_mapping=chunk_map,
    openai_api_key=OPENAI_API_KEY,
    reranker_model=model,
    reranker_tokenizer=tokenizer,
    device=device,
)
print("✅ RetrieverCore ready\n")


###########################################
# LANGCHAIN RETRIEVER WRAPPER
###########################################

retriever = CustomRFPRetriever(
    core=core,
    is_multistep=True,
    top_k=10
)


###########################################
# LLM PIPELINE
###########################################

llm = ChatOpenAI(
    model="gpt-5-mini",
)

prompt = ChatPromptTemplate.from_template("""
당신은 한국 RFP 문서 기반 QA 모델입니다.
반드시 context만 사용하여 답변하세요.

<context>
{context}
</context>

질문: {question}

문서에 정보가 없다면 출력:
"해당 질문의 답변을 문서에서 찾지 못했습니다."
""")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

print("✅ RAG pipeline assembled\n")


###########################################
# INPUT SANITIZER
###########################################

def clean_input(t):
    return (
        t.encode("utf-8", "ignore")
         .decode("utf-8", "ignore")
         .replace("\ufffd", "")
         .strip()
    )


###########################################
# RERANK SCORE THRESHOLD
###########################################
THRESHOLD = -3.3


###########################################
# INTERACTIVE LOOP (WITH /eval SUPPORT)
###########################################

def interactive_loop():
    print("=== Intelligent RAG System Started ===")
    print("💬 입력하세요 (종료: exit):")

    while True:
        raw = input("\n🔎 Query: ").strip()
        query = clean_input(raw)

        if query.lower() in ["exit", "quit"]:
            print("👋 종료합니다.")
            break

        # ----------------------------------------------------
        #  📌 /eval 모드 (generator 평가)
        # ----------------------------------------------------
        if query.startswith("/eval"):
            real_q = clean_input(query.replace("/eval", "").strip())

            print("\n📊 Evaluating generator...\n")

            # 1) Retrieval
            docs = retriever._get_relevant_documents(real_q)
            rerank_score = core.last_rerank_score

            # 2) Generation
            answer = rag_chain.invoke(real_q).content

            # 3) 평가 실행
            result = evaluate_one(
                question=real_q,
                answer=answer,
                retrieved_docs=docs
            )

            print("=== Evaluation Result ===")
            for k, v in result.items():
                print(f"{k}: {v}")
            print("=========================\n")
            continue
        # ----------------------------------------------------

        # ----------------------------------------------------
        # 🔥 일반 쿼리 처리
        # ----------------------------------------------------
        print("\n⏳ Retrieving...\n")

        docs = retriever._get_relevant_documents(query)
        rerank_score = core.last_rerank_score

        # threshold fallback
        if len(docs) == 0 or rerank_score < THRESHOLD:
            print("🧠 Answer:")
            print("해당 질문의 답변을 문서에서 찾지 못했습니다.")
            continue

        print("⏳ Generating answer...\n")
        answer = rag_chain.invoke(query)

        print("🧠 Answer:")
        print(answer.content)

        print("\n📚 Sources:")
        for i, d in enumerate(docs):
            print(f"[{i}] DocID={d.metadata.get('doc_id')}  Chunk={d.metadata.get('chunk_index')}")


if __name__ == "__main__":
    interactive_loop()

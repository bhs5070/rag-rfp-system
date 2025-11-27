import os
import json
from typing import List, Dict

import streamlit as st
import faiss
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from retriever_core import RFPRetrieverCore
from lc_custom_retriever import CustomRFPRetriever
from evaluate_generator import evaluate_one


# =========================
# CONFIG
# =========================
FAISS_INDEX_PATH = "/home/bhs1581/rag-rfp-system/chunking/vector_db/vectordb_multi_aspect_index.faiss"
CHUNK_PATH = "/home/bhs1581/rag-rfp-system/chunking/chunks/chunks_multi_aspect (1).jsonl"
DOC_ORIGINAL_BASE = "/home/bhs1581/rag-rfp-system/original_docs"   # 원문 폴더

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
THRESHOLD = -3.3

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()


# =========================
# UTIL
# =========================
def clean_text(t: str):
    if t is None:
        return ""
    return (
        str(t)
        .encode("utf-8", "ignore")
        .decode("utf-8", "ignore")
        .replace("\ufffd", "")
        .strip()
    )


def load_original_doc(doc_id: str):
    """문서 원본 보기 기능"""
    filename = f"{doc_id}.txt"
    path = os.path.join(DOC_ORIGINAL_BASE, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "(원문 문서를 찾을 수 없습니다.)"


# =========================
# CACHED LOADING
# =========================
@st.cache_resource
def load_faiss_index(path):
    return faiss.read_index(path)


@st.cache_resource
def load_chunks(path):
    texts = []
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            texts.append(obj["text"])
            mapping[i] = obj["doc_id"]
    return texts, mapping


@st.cache_resource
def load_reranker():
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def load_llm(model_name: str):
    """gpt-5 계열은 temperature 지원 안 하므로 분리"""

    if model_name in ["gpt-5", "gpt-5-mini"]:
        return ChatOpenAI(model=model_name)
    else:
        return ChatOpenAI(model=model_name, temperature=0.2)


@st.cache_resource
def init_core_and_retriever():
    index = load_faiss_index(FAISS_INDEX_PATH)
    chunk_texts, chunk_map = load_chunks(CHUNK_PATH)
    tokenizer, model, device = load_reranker()

    core = RFPRetrieverCore(
        faiss_index=index,
        chunk_texts=chunk_texts,
        chunk_mapping=chunk_map,
        openai_api_key=OPENAI_API_KEY,
        reranker_model=model,
        reranker_tokenizer=tokenizer,
        device=device,
    )

    retriever = CustomRFPRetriever(
        core=core,
        is_multistep=True,
        top_k=10,
    )

    return core, retriever, chunk_texts, chunk_map


# =========================
# STREAMLIT UI SETUP
# =========================
st.set_page_config(page_title="RFP RAG System", layout="wide")

st.title("📑 RFP 기반 RAG 시스템")


# 모델 선택 기능
model_name = st.sidebar.selectbox(
    "LLM 모델 선택",
    ["gpt-5-mini", "gpt-5", "gpt-4o-mini"],
    index=0,
)


core, retriever, chunk_texts, chunk_map = init_core_and_retriever()
llm = load_llm(model_name)


# RAG Prompt
prompt = ChatPromptTemplate.from_template("""
당신은 한국 RFP 문서 기반 QA 모델입니다.
반드시 아래 context만 사용하여 정확하게 답변하세요.

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


# =========================
# CHAT UI
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []


st.subheader("💬 RAG 기반 Chat Interface")


# 기존 메시지 렌더링
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# 사용자 입력
user_input = st.chat_input("질문을 입력하세요")

if user_input:
    q = clean_text(user_input)

    st.session_state.messages.append({"role": "user", "content": q})
    st.chat_message("user").write(q)

    # RAG Retrieval
    with st.spinner("🔍 검색 및 답변 생성 중..."):

        docs = retriever._get_relevant_documents(q)
        rerank_score = core.last_rerank_score

        if len(docs) == 0 or rerank_score < THRESHOLD:
            answer_text = "해당 질문의 답변을 문서에서 찾지 못했습니다."
        else:
            answer_obj = rag_chain.invoke(q)
            answer_text = clean_text(answer_obj.content)

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    st.chat_message("assistant").write(answer_text)

    # 문서 원문 보기 기능
    if st.checkbox("📄 문서 원문 보기"):
        for d in docs:
            doc_id = d.metadata.get("doc_id")
            st.markdown(f"### 📘 DocID: {doc_id}")
            st.write(load_original_doc(doc_id))


# =========================
# Optional: 한 질문 평가 기능
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Generator 평가 (/eval one)")

eval_q = st.sidebar.text_input("평가 질문 입력")

if st.sidebar.button("🧪 실행") and eval_q.strip():
    q = clean_text(eval_q)
    docs = retriever._get_relevant_documents(q)

    if len(docs) == 0:
        answer_text = "해당 질문의 답변을 문서에서 찾지 못했습니다."
    else:
        answer_obj = rag_chain.invoke(q)
        answer_text = clean_text(answer_obj.content)

    st.sidebar.write("### 🧠 Answer")
    st.sidebar.write(answer_text)

    st.sidebar.write("### 📚 Sources")
    for d in docs:
        st.sidebar.write(f"- {d.metadata.get('doc_id')} / chunk {d.metadata.get('chunk_index')}")

    st.sidebar.write("### 📊 평가 결과")
    result = evaluate_one(q, answer_text, docs)
    st.sidebar.json(result)

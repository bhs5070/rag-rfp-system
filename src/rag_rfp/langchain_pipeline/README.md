# LangChain Pipeline (Experimental)

이 디렉토리는 `rag-rfp-system` 내에서 **LangChain 기반 RAG 실험용 파이프라인**을 모아둔 공간입니다.  
기존의 `rag_rfp/retrieve/` 에 있는 custom retriever를 기반으로 하되,  
파이프라인 전체를 LangChain Runnable/LCEL 형태로 재구성하여 단순화 및 실험 용도로 사용합니다.

---

## 📌 목적

- Retrieval 엔진(`RFPRetrieverCore`)을 LangChain 환경에 자연스럽게 통합
- 기존 custom pipeline 대비 **실험 속도 향상**
- Hybrid Search, Reranker, FAISS 등을 **LangChain wrapper**로 묶어서 재사용성 강화
- Custom retriever와 LangChain 오케스트레이션의 **하이브리드 구조** 구축

---

## 📁 구성 파일

### `main.py`
LangChain 기반 RAG 파이프라인 실행 엔트리 포인트입니다.
- RFPRetrieverCore 로드  
- Custom LangChain Retriever 생성  
- LLM (OpenAI) 로드  
- 검색 → 문맥 조합 → 답변 생성 end-to-end 파이프라인 수행  

---

### `retriever_core.py`
Custom Retrieval 엔진의 핵심 로직.

포함된 기능:
- Multi-step Hybrid Search  
- Dense Search (FAISS)  
- Sparse Search (BM25)  
- RRF Fusion  
- Cross-Encoder Reranker  
- Query Rewriting  
- Embedding (text-embedding-3-small)  

LangChain retriever는 이 core를 backend로 사용하여 Document 리스트를 반환합니다.

---

### `lc_custom_retriever.py`
LangChain의 `BaseRetriever`를 상속한 Custom Retriever Wrapper.

역할:
- `RFPRetrieverCore.retrieve()` 결과를 LangChain `Document` 객체로 변환
- LangChain Runnable/Chains에서 사용 가능하도록 인터페이스 통일

---

### `evaluate_generator.py`
LangChain Pipeline 기반 LLM Generator 품질 평가 스크립트.

예시:
- 특정 질문 리스트에 대해 응답 생성
- 응답 품질 평가  
- 비교 실험 수행

---

# 🚀 사용 방법

### 1. core 초기화
`retriever_core.py` 내부에서 다음을 로딩하도록 구성:

- 청크 텍스트 (`chunk_texts`)
- 청크 매핑 (`chunk_mapping`)
- FAISS 인덱스
- Reranker 모델 + tokenizer
- OpenAI key

### 2. 파이프라인 실행 (main.py)

```bash
python src/langchain_pipeline/main.py
```

---

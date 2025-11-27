# LangChain 기반 RAG Chat 실험 환경 (rag-rfp-system/langchain-pipeline)

이 디렉토리는 `rag-rfp-system` 내에서 **LangChain 기반 RAG (Retrieval-Augmented Generation) 실험용 파이프라인과 Streamlit UI**를 모아둔 공간입니다.
검색(Retrieval) – 랭킹(Reranking) – 생성(Generation) – 평가(Evaluation) 프로세스를 빠르게 실험하고 시각화하기 위한 환경을 제공합니다.

---

## 📌 프로젝트 목표 (Objectives)

* **LangChain 통합:** Retrieval 엔진(`RFPRetrieverCore`)을 LangChain 환경에 **자연스럽게 통합**하여 사용 편의성 극대화.
* **실험 효율성:** 기존 Custom Pipeline 대비 **실험 속도 및 유지보수성 향상**.
* **재사용성 강화:** **Hybrid Search, Reranker, FAISS** 등을 LangChain Wrapper로 묶어 모듈화 및 재사용성 강화.
* **하이브리드 구조:** Custom Retriever와 LangChain Runnable/Chain 기반 **유연한 하이브리드 RAG 검색 구조** 구축.
* **데모 및 평가 환경:** Streamlit UI를 통해 **RAG Chat 및 평가 기능**을 즉시 테스트 가능한 **데모 환경 제공**.

---

## 📁 구성 파일 (File Structure & Roles)

### `main.py`
LangChain 기반 RAG 파이프라인의 메인 실행 엔트리 포인트입니다.
* `RFPRetrieverCore` 초기화 및 `CustomRFPRetriever` 생성.
* **GPT-5 / GPT-5-mini** 기반 LLM 로드 및 RAG Chain 구성.
* 검색 → 문맥 구성 → 답변 생성의 전체 파이프라인 실행.
* 터미널에서 `/eval <질문>` 명령 기반 Generator **평가 기능** 포함.

### `retriever_core.py`
Retrieval 엔진의 핵심 로직 구현 모듈입니다.
* **Multi-step Hybrid Search** (Dense + Sparse + Query Rewriting) 구현.
* **Dense Search (FAISS)** 및 **Sparse Search (BM25)**.
* 검색 결과 **RRF (Reciprocal Rank Fusion)** 통합.
* **BGE Cross-Encoder Reranker**를 활용한 최종 랭킹.
* surrogate-safe embedding 및 reranker confidence 기반 Fallback 로직 포함.

### `lc_custom_retriever.py`
* LangChain의 `BaseRetriever`를 상속받은 Custom Retriever 구현체.
* `RFPRetrieverCore`의 검색 결과를 LangChain `Document` 객체로 변환.
* LangChain의 Runnable 및 Chain에서 **바로 연결하여 사용 가능**하도록 지원.

### `evaluate_generator.py`
* **LLM-as-a-Judge** 방식을 사용한 Generator 평가 로직.
* **Faithfulness, Groundedness, Quality** 등의 지표 산출.
* Source 기반의 **Hallucination 판별** 기능.
* `main.py`의 `/eval` 명령 및 Streamlit UI와 연동.

### `rag_chat_app.py`
Streamlit 기반의 **실시간 RAG Chat 데모 UI**를 제공합니다.
* 질문 검색 및 답변 생성 결과 표시.
* **Source Document 보기** 및 **평가 실행** (Evaluation).
* LLM 모델 스위치, 원문 비교, 채팅 히스토리 관리 UI 포함.

---

## 🚀 사용 방법 (How to Run)

### 1) 터미널 기반 RAG 실행
RAG 파이프라인의 로직 테스트 및 콘솔 기반 평가에 사용합니다.
```bash
python main.py
2) Streamlit 기반 RAG Chat 실행
실시간 데모 및 시각화된 QA, 평가 수행에 사용합니다.

Bash

streamlit run rag_chat_app.py
실행 후 다음 주소로 접속합니다.

http://localhost:8501 
# VM 환경에서 실행 시: http://<VM-public-ip>:8501
3) Eval 명령어 사용법
main.py 또는 Streamlit Chat UI에서 평가를 수행합니다.

Bash

/eval <평가할_질문>
🗺️ 디렉토리 구조
langchain_pipeline/
│
├── main.py                   # RAG 파이프라인 엔트리
├── rag_chat_app.py           # Streamlit UI 데모
├── retriever_core.py         # 핵심 검색 로직 (Hybrid Search, Reranker)
├── lc_custom_retriever.py    # LangChain BaseRetriever Wrapper
└── evaluate_generator.py     # LLM-as-a-Judge 평가 모듈
🎯 비고 (Notes)
이 디렉토리는 빠른 실험, 테스트, 기능 검증을 위해 설계되었습니다.

기능 개발 속도 향상, 모델 간 비교, UI/UX 실험에 최적화되어 있습니다.

팀원들이 Streamlit을 통해 손쉽게 QA 및 정성/정량 평가 수행이 가능합니다.

Production-level의 안정적인 파이프라인과 별개의 가벼운 구조로 운영됩니다.


---

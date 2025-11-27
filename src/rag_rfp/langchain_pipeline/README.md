# LangChain Pipeline + Streamlit RAG Chat (Experimental)

이 디렉토리는 `rag-rfp-system` 내에서 **LangChain 기반 RAG 실험용 파이프라인과 Streamlit UI**를 모아둔 공간입니다.  
검색–랭킹–생성–평가를 빠르게 실험하고 시각화하기 위한 환경을 제공합니다.

---

## 📌 목적
- Retrieval 엔진(`RFPRetrieverCore`)을 LangChain 환경에 자연스럽게 통합  
- 기존 custom pipeline 대비 **실험 속도 및 유지보수성 향상**  
- Hybrid Search, Reranker, FAISS 등을 LangChain wrapper로 묶어 **재사용성 강화**  
- Custom retriever + LangChain Runnable/Chain 기반 **하이브리드 검색 구조 구축**  
- Streamlit UI를 통해 **RAG Chat + 평가 기능을 즉시 테스트 가능한 데모 제공**

---

## 📁 구성 파일

### `main.py`
LangChain 기반 RAG 파이프라인 실행 엔트리 포인트입니다.
- RFPRetrieverCore 초기화  
- CustomRFPRetriever 생성  
- GPT-5 / GPT-5-mini 기반 LLM 로드  
- 검색 → 문맥 구성 → 답변 생성  
- `/eval` 명령 기반 generator 평가 기능 포함  

---

### `retriever_core.py`
Retrieval 엔진의 핵심 로직 구현:
- Multi-step Hybrid Search (Dense + Sparse + Query Rewriting)  
- Dense Search (FAISS)  
- Sparse Search (BM25)  
- RRF Fusion  
- BGE Cross-Encoder Reranker  
- surrogate-safe embedding  
- reranker confidence 기반 fallback 포함  

---

### `lc_custom_retriever.py`
- LangChain `BaseRetriever` 상속  
- RFPRetrieverCore 검색 결과를 LangChain `Document`로 변환  
- Runnable / Chains에서 바로 사용 가능  

---

### `evaluate_generator.py`
- LLM-as-a-Judge 방식 Generator 평가  
- Faithfulness / Groundedness / Quality 산출  
- Source 기반 hallucination 판별  
- `/eval` 및 Streamlit UI와 연동  

---

### `rag_chat_app.py`
Streamlit 기반의 실시간 RAG Chat 데모 UI:
- 검색  
- 답변 생성  
- Source 보기  
- 평가 실행  
- 모델 스위치, 원문 보기, 채팅 UI  

---

## 🚀 사용 방법

### 1) 터미널 기반 RAG 실행
```bash
python main.py


**### 2) Streamlit 기반 RAG Chat 실행**
streamlit run rag_chat_app.py

http://<VM-public-ip>:8501

**### 3) eval 명령어 사용법**
/eval <질문>

## 디렉토리 구조 

langchain_pipeline/
│
├── main.py
├── rag_chat_app.py
├── retriever_core.py
├── lc_custom_retriever.py
└── evaluate_generator.py

🎯 비고

- 이 디렉토리는 실험/테스트용으로 설계됨
- 빠른 기능 개발, 모델 비교, UI 실험에 적합
- 팀원들이 Streamlit으로 손쉽게 QA 및 평가 수행 가능
- Production-level 파이프라인과 별개의 가벼운 구조

---


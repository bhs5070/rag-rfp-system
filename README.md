# 🧠 RAG-RFP 시스템
RFP(제안요청서) 문서 요약 및 정보 추출을 위한 Retrieval-Augmented Generation 시스템

PDF/HWP 문서에서 “사업 목적, 예산, 일정, 수행 범위, 수행 기관 자격 요건” 등 핵심 정보를  
검색 기반 + 생성 기반 모델을 결합하여 자동으로 찾아주는 RAG 시스템입니다.

---

# 🏗️ 프로젝트 구조

```
data/
 ├─ eval         # 평가 데이터
 └─ data_list.csv      # RFP 문서 메타데이터

configs/
 └─ config.sample.yaml # 설정 템플릿 (local 복사 필요)

src/
 ├─ rag_rfp/
 │   ├─ io/            # PDF/HWP 파싱 및 텍스트 정규화
 │   ├─ prep/          # 청킹 로직
 │   ├─ index/         # 임베딩 생성 및 FAISS 인덱스 구축
 │   ├─ retrieve/      # 검색 모듈 (retriever_custom.py)
 │   ├─ generate/      # LLM 기반 답변 생성
 │   └─ eval/          # 성능 평가 (Recall@K 등)
 │
 ├─ langchain_pipeline/ # LangChain 기반 RAG 실험 파이프라인
 │     ├─ main.py
 │     ├─ retriever_core.py
 │     ├─ lc_custom_retriever.py
 │     └─ evaluate_generator.py
 │
 └─ cli/               # 파이프라인 실행용 CLI 스크립트

Makefile               # ingest/index/ask/serve 명령어 자동화
environment.yml        # Conda 환경 설정
requirements.txt       # Python dependency
```

---

# 🚀 실행 안내

## 1) 가상환경 설정

### venv
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Conda
```bash
conda env create -f environment.yml
conda activate rag-rfp
```

---

## 2) API Key 설정

프로젝트 루트에 `.env` 생성:

```
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-large
```

---

## 3) 데이터 준비

원본 문서를 `data/raw/` 에 저장  
메타데이터를 `data/data_list.csv` 에 작성

예시:
```
doc_id,filename,path,title,issuer,publication_date,page_count,language,doc_type
1,sample.pdf,raw/sample.pdf,샘플 RFP,행정기관,2024-10-10,3,ko,RFP
```

---

## 4) 파이프라인 실행

```bash
make ingest    # 파싱 → 정제 → 청킹
make index     # 임베딩 및 FAISS 인덱스 구축
make ask       # CLI 직접 질의
make serve     # FastAPI 서버 실행 (127.0.0.1:8000/docs)
```

---

# 🧩 주요 구성 요소

| 모듈명 | 파일 경로 | 역할 |
|--------|-----------|------|
| Parser | `src/rag_rfp/io/parse_pdf.py`, `parse_hwp.py` | PDF/HWP 텍스트 추출 |
| Normalizer | `src/rag_rfp/io/normalize.py` | 텍스트 정규화 |
| Chunker | `src/rag_rfp/prep/chunk.py` | 문서 청킹 |
| Embedder | `src/rag_rfp/index/embed.py` | 텍스트 임베딩 생성 |
| Vector DB | `src/rag_rfp/index/vectordb.py` | FAISS 인덱스 구축 |
| Retriever (최종) | `src/rag_rfp/retrieve/retriever_custom.py` | Hybrid + Reranker + Multi-step 검색 |
| Generator | `src/rag_rfp/generate/generator.py` | 검색 기반 응답 생성 |
| Evaluation | `src/rag_rfp/eval/` | 성능 측정 스크립트 |

---

# 🧪 LangChain Pipeline (Experimental)

LangChain 기반 RAG 실험용 코드는  
기존 pipeline과 독립적으로 아래 폴더에 정리되어 있습니다:

```
src/langchain_pipeline/
```

포함 기능:
- LangChain 커스텀 Retriever (LCEL 기반)
- Multi-step Hybrid Search
- BM25 + Dense + Reranker
- LLM 기반 Generator 평가 스크립트

자세한 설명: `src/langchain_pipeline/README.md`

---

# 🔧 수정 / 확장 가이드

| 변경 목표 | 수정 파일 | 설명 |
|-----------|-----------|------|
| 임베딩 모델 교체 | `index/embed.py` | OpenAI → SentenceTransformer 등 |
| 검색 전략 교체 | `retrieve/retriever_custom.py` | BM25, Hybrid, Multi-step 등 |
| LLM 교체 | `generate/generator.py` | GPT-4o / Claude / Mistral 등 |
| 청킹 규칙 변경 | `prep/chunk.py` | max_tokens, stride 조정 |
| 파라미터 변경 | `configs/config.local.yaml` | top_k, temperature 등 |
| API 확장 | `cli/serve_api.py` | FastAPI endpoint 추가 |

---

# 🧰 Make 명령어 요약

| 명령어 | 설명 |
|--------|------|
| `make setup` | 환경 초기화 |
| `make ingest` | 문서 파싱 및 청킹 |
| `make index` | 임베딩 + 인덱스 구축 |
| `make ask` | CLI 질의 |
| `make serve` | FastAPI 서버 실행 |

---

# 👥 협업 규칙

### 브랜치 전략
- 신규 기능: `feat/<기능명>`
- 버그 수정: `fix/<이슈>`
- 예: `feat/retrieval-multistep`, `fix/chunk-offset`

### 커밋 메시지 (Conventional)
- `feat:` 기능 추가  
- `fix:` 버그 수정  
- `chore:` 문서/환경 정리  

### Pull 권장
```bash
git pull origin main --rebase
```

---

# 📈 향후 개선 방향

- Retrieval 평가 지표 확장 (nDCG, MRR 등)
- LangChain / LangGraph 기반 워크플로우 구성
- Chroma/LanceDB 등 다른 벡터 DB 실험
- Streamlit / FastAPI 기반 UI 개발
- Docker 이미지화 및 배포 자동화

---


# 🧠 RAG-RFP 시스템
RFP(제안요청서) 문서 요약 및 정보 추출을 위한 Retrieval-Augmented Generation 시스템

## 🚀 개요
이 프로젝트는 RAG (Retrieval-Augmented Generation) 구조를 기반으로
PDF와 HWP 형식의 RFP 문서에서 주요 정보를 자동으로 추출하고 요약하는 시스템입니다.

문서 내의 “제출 마감일, 예산, 사업 범위, 수행 기관 자격 요건” 등 핵심 정보를
검색과 생성 모델을 결합하여 효과적으로 응답할 수 있도록 설계되었습니다.

## 🏗️ 프로젝트 구조
data/  
 ├─ raw/               # 원본 RFP 문서 (PDF, HWP)  
 ├─ interim/           # 중간 가공 데이터  
 ├─ processed/         # 전처리 및 청킹된 데이터 (JSONL)  
 └─ data_list.csv      # 문서 메타데이터 목록  
configs/  
 └─ config.sample.yaml # 예시 설정 파일 (복사 후 config.local.yaml로 사용)  
src/  
 ├─ rag_rfp/  
 │   ├─ io/            # PDF/HWP 파싱 및 텍스트 정제  
 │   ├─ prep/          # 청킹 및 전처리 로직  
 │   ├─ index/         # 임베딩 및 벡터 인덱스 (FAISS)  
 │   ├─ retrieve/      # 검색 및 리랭커 모듈  
 │   ├─ generate/      # LLM 기반 답변 생성 모듈  
 │   └─ eval/          # 성능 평가 스크립트 (선택)  
 └─ cli/               # 커맨드라인 실행 모듈  
Makefile               # 파이프라인 명령어 자동화  
environment.yml         # Conda 환경 설정  
requirements.txt        # Python 라이브러리 목록  

## ⚙️ 실행 방법
### 1️⃣ 환경 설정
venv 사용 시  
python -m venv .venv  
source .venv/bin/activate  
pip install -r requirements.txt  

Conda 사용 시  
conda env create -f environment.yml  
conda activate rag-rfp  

### 2️⃣ API Key 설정
프로젝트 루트(rag-rfp-system/)에 .env 파일을 만들고 아래 내용 추가 👇  

OPENAI_API_KEY=sk-발급받은_API키  
OPENAI_CHAT_MODEL=gpt-4o-mini  
OPENAI_EMBED_MODEL=text-embedding-3-large  

### 3️⃣ 데이터 준비
data/raw/ 폴더에 RFP 원본 파일(PDF, HWP) 저장  
data/data_list.csv 파일에 문서 정보를 등록  

예시:  
doc_id,filename,path,title,issuer,publication_date,page_count,language,doc_type  
1,sample.pdf,raw/sample.pdf,샘플 RFP,행정기관,2024-10-10,3,ko,RFP  

### 4️⃣ 파이프라인 실행  
make ingest   # 문서 파싱 → 전처리 → 청킹  
make index    # 임베딩 생성 및 인덱스 구축  
make ask      # CLI에서 직접 질의  
make serve    # FastAPI 서버 실행 (http://127.0.0.1:8000/docs)  

## 🧩 주요 구성요소
모듈	파일 경로	역할  
| **모듈명**        | **파일 경로**                                     | **역할**                   |
| -------------- | --------------------------------------------- | ------------------------ |
| **Parser**     | `src/rag_rfp/io/parse_pdf.py`, `parse_hwp.py` | PDF/HWP 문서에서 텍스트 추출      |
| **Normalizer** | `src/rag_rfp/io/normalize.py`                 | 텍스트 정제, 공백 및 노이즈 제거      |
| **Chunker**    | `src/rag_rfp/prep/chunk.py`                   | 문서를 청크 단위로 분리            |
| **Embedder**   | `src/rag_rfp/index/embed.py`                  | 텍스트 → 벡터 임베딩 변환          |
| **Vector DB**  | `src/rag_rfp/index/vectordb.py`               | FAISS 기반 벡터 인덱스 관리       |
| **Retriever**  | `src/rag_rfp/retrieve/retriever.py`           | Top-K 관련 문서 검색 수행        |
| **Reranker**   | `src/rag_rfp/retrieve/rerank.py`              | 교차 인코더 기반 리랭킹 수행         |
| **Generator**  | `src/rag_rfp/generate/generator.py`           | 검색 결과 기반 답변 생성 (LLM)     |
| **CLI 도구**     | `src/cli/`                                    | 전체 파이프라인 실행을 위한 명령어 스크립트 |


## 🧠 수정/확장 가이드
변경 목표	수정 파일	설명  
| **변경 목표**  | **수정 파일**                   | **설명**                                  |
| ---------- | --------------------------- | --------------------------------------- |
| 임베딩 모델 변경  | `index/embed.py`            | OpenAI → SentenceTransformer 등 로컬 모델 교체 |
| 검색 전략 변경   | `retrieve/retriever.py`     | BM25 / Hybrid Search 등 새로운 검색 방식 추가     |
| LLM 모델 교체  | `generate/generator.py`     | GPT-4o → Mistral, Claude 등으로 변경         |
| 청킹 규칙 수정   | `prep/chunk.py`             | `max_tokens`, `stride` 등 청킹 규칙 조정       |
| 하이퍼파라미터 변경 | `configs/config.local.yaml` | `temperature`, `top_k` 등 파라미터 수정        |
| API 추가     | `cli/serve_api.py`          | FastAPI 엔드포인트 추가 및 수정 가능                |
 

## 🧰 개발 명령어 요약
명령어	설명  
| **명령어**       | **설명**           |
| ------------- | ---------------- |
| `make setup`  | 가상환경 생성 및 의존성 설치 |
| `make ingest` | 데이터 파싱 및 청킹 수행   |
| `make index`  | 벡터 인덱스 구축        |
| `make serve`  | FastAPI 서버 실행    |
| `make ask`    | CLI로 직접 질의 수행    |

## 👥 협업 규칙
브랜치 전략:
새로운 기능은 feat/기능명, 수정은 fix/수정내용 형태로 분기
예: feat/retrieval-bm25, fix/chunk-overlap

커밋 메시지 규칙: Conventional Commits
feat: 리랭커 모듈 추가
fix: 청킹 오프셋 오류 수정
chore: Makefile 주석 정리

Pull 시 권장:
git pull origin main --rebase

## 📈 향후 개선 방향
평가 지표 추가 (Recall@K, nDCG 등)  
LangChain / Chroma 기반 백엔드 실험  
Streamlit / FastAPI 프론트엔드 UI 연동  
Docker 컨테이너화 및 배포 자동화

## 📄 개인 협업 일지
박병현: https://famous-gorilla-33.notion.site/AI-_-_-2a8c7c1a0092809fb74ac1cef219e972?source=copy_link

손원후: [https://www.notion.so/2b9869855e42806a8824ced736e15303](https://www.notion.so/2b9869855e42806a8824ced736e15303?source=copy_link)

이솔형: https://www.notion.so/2-2a724d5698b681709a85d118b4b925e4?source=copy_link

배현석: [https://www.notion.so/2a670bb8574780b2b6b8ca4a55e0baa6?source=copy_link](https://www.notion.so/2a670bb8574780b2b6b8ca4a55e0baa6?source=copy_link)

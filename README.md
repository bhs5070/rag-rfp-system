# RFP Analyzer - RFP 문서 분석 RAG 시스템

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RFP(제안요청서) 문서를 자동으로 파싱, 청킹, 임베딩하여 핵심 요구사항을 즉시 검색할 수 있는 RAG 시스템입니다.

**프로젝트 기간**: 2025.09 ~ 2025.11 (3개월)

---

## 📊 핵심 성과

| 항목 | 목표 | 달성 | 평가 |
|------|------|------|------|
| Recall@5 | ≥ 80% | **90.3%** | ✅ 목표 초과 달성 |
| Recall@1 | ≥ 60% | **83.3%** | ✅ 임베딩 전환으로 향상 |
| 비용 절감 | - | **80%** ($0.0095 → $0.0019) | ✅ 대폭 절감 |
| Faithfulness | ≥ 0.95 | **0.979** | ✅ 환각 최소화 |
| Answer Relevancy | ≥ 0.90 | **0.979** | ✅ 목표 초과 달성 |

---

## 🎯 주요 기능

### 1. PDF 문서 자동 파싱
- PyMuPDF 기반 RFP 문서 텍스트 추출
- 100개 RFP 문서 처리 (평균 100페이지 이상)

### 2. Semantic Chunking
- **7가지 chunking 전략 정량 비교** (Page, Paragraph, Original, Structure-aware, Semantic 등)
- **Semantic Chunking 채택**: 문맥 기반 청킹으로 의미 보존
- **청크 수 21.9% 감소**: 10,401 → 8,123 chunks
- **Recall@5 90.3% 달성**: 최고 검색 정확도

### 3. 고성능 검색
- **text-embedding-3-large**: Recall@1 13.9%p 향상 (66.7% → 80.6%)
- **ChromaDB**: 빠른 벡터 검색 (평균 4.9ms/query)
- **Dense Retrieval**: BM25, Reranker 제거 후 최고 성능

### 4. 비용 최적화 생성
- **GPT-4.1-mini**: 비용 80% 절감 ($0.0095 → $0.0019)
- **품질 유지**: 1.3%만 하락 (0.992 → 0.979)
- **Faithfulness 0.979**: 환각 최소화

### 5. 웹 UI
- FastAPI 기반 단일 페이지 웹 인터페이스
- 실시간 질의응답
- 검색 결과 출처 표시

---

## 🛠 기술 스택

### Backend
- **Framework**: FastAPI, Python 3.11+
- **Vector DB**: ChromaDB
- **API**: RESTful API

### AI 모델
- **Embedding**: text-embedding-3-large (OpenAI)
- **Generation**: GPT-4.1-mini
- **Chunking**: Semantic Chunking (8,123 chunks)

### 데이터 처리
- **PDF Parsing**: PyMuPDF
- **Evaluation**: 72개 평가셋 기반 정량 비교

---

## 🚀 빠른 시작

### 사전 요구사항

- Python 3.11 이상
- OpenAI API Key

### 1. 저장소 클론

```bash
git clone https://github.com/yourusername/rag-rfp-system.git
cd rag-rfp-system
```

### 2. 가상환경 설정

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env` 파일 생성:

```env
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-large
```

### 5. 인덱스 생성

```bash
PYTHONPATH=. python src/cli/build_index.py
```

### 6. 웹 서버 실행

```bash
PYTHONPATH=. python -m uvicorn src.cli.serve_api:app --host 0.0.0.0 --port 8000 --reload
```

### 7. 접속

브라우저에서 `http://localhost:8000` 접속

---

## 📁 프로젝트 구조

```
rag-rfp-system/
├── src/
│   ├── cli/
│   │   ├── build_index.py      # 인덱스 생성
│   │   ├── ask.py              # CLI 질의응답
│   │   └── serve_api.py        # 웹 API 서버
│   ├── rag_rfp/
│   │   ├── io/                 # PDF 파싱
│   │   ├── prep/               # 청킹, 임베딩
│   │   ├── index/              # ChromaDB 인덱싱
│   │   ├── retrieve/           # 검색
│   │   ├── generate/           # 답변 생성
│   │   └── eval/               # 평가 스크립트
│   └── langchain_pipeline/     # LangChain 기반 파이프라인
├── data/
│   └── eval/                   # 평가 데이터
├── docs/
│   ├── portfolio-retrieval-summary.md    # 검색 성능 요약
│   ├── retrieval-evaluation-log.md       # 실험 로그
│   └── generation-model-evaluation.md    # 생성 모델 평가
├── configs/
│   └── config.sample.yaml      # 설정 템플릿
├── requirements.txt
└── README.md
```

---

## 💡 핵심 기술적 성과

### 1. Semantic Chunking 최적화

**7가지 chunking 전략 정량 비교**

| Strategy | Chunks | Recall@1 | Recall@5 | MRR@10 | Latency (ms) |
|----------|--------|----------|----------|--------|--------------|
| **semantic** | **8,123** | **83.3%** | **90.3%** | **0.860** | **4.9** |
| original | 10,401 | 81.9% | 87.5% | 0.844 | 5.5 |
| structure_aware | 11,452 | 81.9% | 88.9% | 0.848 | 7.5 |
| page | 7,456 | 80.6% | 84.7% | 0.832 | 3.4 |
| paragraph | 7,389 | 80.6% | 86.1% | 0.826 | 6.5 |

**채택 근거**:
- 최고 Recall@5 (90.3%), Recall@1 (83.3%)
- 청크 수 21.9% 감소
- Latency 실사용 가능 (4.9ms)

### 2. 임베딩 모델 최적화

**text-embedding-3-large 전환 효과**

| Model | Recall@1 | Recall@5 | MRR@10 | 향상 |
|-------|----------|----------|--------|------|
| text-embedding-3-small | 66.7% | 87.5% | 0.730 | - |
| **text-embedding-3-large** | **80.6%** | **90.3%** | **0.836** | **+13.9%p** |

- Recall@1: +13.9%p 향상
- MRR: +14.5% 개선

### 3. 생성 모델 Trade-off 분석

**5가지 생성 모델 품질·비용·속도 비교**

| Model | Answer Relevancy | Faithfulness | Cost/query |
|-------|-----------------|--------------|------------|
| GPT-4.1 | 0.992 | 0.979 | $0.0095 |
| **GPT-4.1-mini** | **0.979** | **0.979** | **$0.0019** |
| GPT-4o-mini | 0.936 | - | $0.0006 |

**GPT-4.1-mini 채택 근거**:
- 비용 80% 절감
- 품질 1.3%만 하락
- Faithfulness 유지 (환각 최소화)

### 4. Retrieval 전략 실험

**실패한 실험**:
- **Reranker (BGE)**: Recall@1 80% → 32% (급락), Latency 1,200배 증가
- **Hybrid Search (BM25+Dense)**: Recall@1 80.6% → 73.6% (하락)

**최종 채택**: Dense E5 임베딩 단독

---

## 📈 평가 지표

### Retrieval 성능

| 메트릭 | 결과 | 의미 |
|--------|------|------|
| **Recall@1** | 83.3% | Top 1에 정답 포함 비율 |
| **Recall@5** | 90.3% | Top 5에 정답 포함 비율 |
| **Recall@10** | 93.1% | Top 10에 정답 포함 비율 |
| **MRR@10** | 0.860 | 평균 정답 순위 (역수) |

### Generation 품질

| 메트릭 | 결과 | 의미 |
|--------|------|------|
| **Answer Relevancy** | 0.979 | 답변 관련성 |
| **Faithfulness** | 0.979 | 환각 최소화 (원본 충실도) |
| **Avg Latency** | 3,858ms | 평균 응답 시간 |

### 비용 효율

- **쿼리당 비용**: $0.0019
- **월 비용 (1만 쿼리)**: $19
- **비용 절감**: 80% (GPT-4.1 대비)

---

## 📖 문서

### 평가 및 실험

- [검색 성능 요약](docs/portfolio-retrieval-summary.md) - Chunking, 임베딩, Retrieval 전략 비교
- [실험 로그](docs/retrieval-evaluation-log.md) - 전체 실험 과정 상세 기록
- [생성 모델 평가](docs/generation-model-evaluation.md) - 5가지 모델 Trade-off 분석

### 결과 데이터

- [Chunking 전략 결과](data/eval/results/chunking_strategy_latency_results.md)
- [생성 모델 비교](data/eval/results/generation_model_comparison.md)

---

## 🎓 핵심 교훈

### 1. 정량적 실험의 중요성

**모든 의사결정에 데이터 근거**:
- 7가지 chunking 전략 비교 → Semantic 채택
- 5가지 생성 모델 비교 → GPT-4.1-mini 채택
- 3가지 retrieval 전략 비교 → Dense only 채택

**실패한 실험도 문서화**:
- Reranker, Hybrid Search 실패 원인 분석
- 실패 이유를 학습 자료로 활용

### 2. 도메인 특성 이해

**RFP 문서 특성**:
- 요구사항이 문단 단위 구성
- 전문 용어 빈번 사용
- Semantic 경계가 명확

**최적 전략**:
- Semantic Chunking: 요구사항 단위 보존
- Dense E5 임베딩: 전문 용어 이해 우수
- text-embedding-3-large: 문맥 유사도 정확

### 3. Trade-off 인식

- **성능 vs 비용**: GPT-4.1-mini 선택으로 비용 80% 절감, 품질 1.3%만 하락
- **정확도 vs Latency**: Reranker 제거로 80배 빠른 속도 확보
- **복잡도 vs 효과**: Simple is Best - Dense E5 단독이 최고 성능

---

## 🔧 환경 변수

`.env` 파일 설정:

```env
# Required
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-large

# Optional
LOG_LEVEL=INFO
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

---

## 📝 라이선스

이 프로젝트는 [MIT License](LICENSE) 하에 배포됩니다.

---

## 👤 작성자

**배현석** - AI Engineer

- GitHub: [yourusername](https://github.com/yourusername)
- Email: bhs5070@gmail.com

---

## 🙏 감사의 말

이 프로젝트는 개인 학습 및 포트폴리오 목적으로 개발되었습니다.

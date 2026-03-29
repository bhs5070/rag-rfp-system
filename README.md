# RFP Analyzer

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

RAG (Retrieval-Augmented Generation) 아키텍처를 활용한 RFP(제안요청서) 문서 분석 시스템

## 개요

RFP Analyzer는 RFP 문서를 자동으로 파싱, 청킹, 인덱싱하여 핵심 요구사항을 즉시 검색할 수 있게 합니다. 시맨틱 청킹과 최신 임베딩 모델을 통해 90.3%의 Recall@5를 달성하면서 운영 비용을 80% 절감했습니다.

### 주요 성능 지표

| 지표 | 결과 | 상태 |
|------|------|------|
| **Recall@5** | 90.3% | ✅ 프로덕션 준비 완료 |
| **Recall@1** | 83.3% | ✅ 높은 정밀도 |
| **Answer Relevancy** | 0.979 | ✅ 낮은 환각률 |
| **Faithfulness** | 0.979 | ✅ 높은 출처 신뢰도 |
| **쿼리당 비용** | $0.0019 | ✅ 80% 비용 절감 |

---

## 주요 기능

### 시맨틱 청킹
- **문맥 인식 분할**: 요구사항 경계를 보존하는 지능형 분할
- **21.9% 청크 감소**: 10,401개 → 8,123개 청크
- **90.3% Recall@5**: 고정 크기 전략 대비 우수한 성능

### 고성능 검색
- **text-embedding-3-large**: 탁월한 의미 이해
- **ChromaDB** 벡터 스토어: 평균 4.9ms 쿼리 지연시간
- **Dense retrieval**: 기술 문서에 최적화된 검색

### 비용 최적화 생성
- **GPT-4.1-mini**: 품질과 비용의 균형
- **80% 비용 절감**: GPT-4.1 대비 ($0.0095 → $0.0019/쿼리)
- **0.979 충실도 점수**: 환각 현상 최소화

### 웹 인터페이스
- FastAPI 기반 RESTful API
- 실시간 쿼리 처리
- 출처 인용 및 투명성 제공

---

## 빠른 시작

### 사전 요구사항

- Python 3.11+
- OpenAI API 키

### 설치

```bash
# 저장소 클론
git clone https://github.com/bhs5070/rag-rfp-system.git
cd rag-rfp-system

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 설정

`.env` 파일 생성:

```env
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-large
```

### 인덱스 빌드

```bash
PYTHONPATH=. python src/cli/build_index.py
```

### 서버 실행

```bash
PYTHONPATH=. python -m uvicorn src.cli.serve_api:app --host 0.0.0.0 --port 8000 --reload
```

`http://localhost:8000`에서 접속

---

## 아키텍처

```
┌─────────────────────────────────────────────┐
│          PDF 문서 인제스천                   │
│  PyMuPDF → 텍스트 추출 → 정규화              │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│          시맨틱 청킹                         │
│  문장 분할 → 유사도 기반 병합                │
│  8,123개 청크 (평균 875자)                   │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│     임베딩 (text-embedding-3-large)         │
│  3072차원 벡터 → ChromaDB                    │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         Dense Retrieval (Top-K=5)           │
│  코사인 유사도 → 평균 4.9ms 지연시간         │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│      답변 생성 (GPT-4.1-mini)               │
│  컨텍스트 + 쿼리 → 응답 (평균 3.8초)         │
└─────────────────────────────────────────────┘
```

---

## 기술적 의사결정

### 청킹 전략

72개 쿼리 벤치마크에서 7가지 청킹 전략 평가:

| 전략 | 청크 수 | Recall@5 | 지연시간 (ms) | 결정 |
|------|---------|----------|---------------|------|
| **Semantic** | **8,123** | **90.3%** | **4.9** | ✅ 선택됨 |
| Original | 10,401 | 87.5% | 5.5 | - |
| Structure-aware | 11,452 | 88.9% | 7.5 | - |
| Page | 7,456 | 84.7% | 3.4 | - |
| Paragraph | 7,389 | 86.1% | 6.5 | - |

**선택 근거**: 시맨틱 청킹이 가장 높은 재현율을 달성하면서 청크 수를 줄이고 낮은 지연시간을 유지했습니다.

### 임베딩 모델

| 모델 | Recall@1 | Recall@5 | MRR@10 | 개선율 |
|------|----------|----------|--------|--------|
| text-embedding-3-small | 66.7% | 87.5% | 0.730 | 기준선 |
| **text-embedding-3-large** | **80.6%** | **90.3%** | **0.836** | **+13.9pp** |

**선택 근거**: 13.9pp의 Recall@1 개선과 14.5%의 MRR 향상으로 약간의 비용 증가를 정당화했습니다.

### 생성 모델

| 모델 | Answer Relevancy | Faithfulness | 쿼리당 비용 |
|------|------------------|--------------|-------------|
| GPT-4.1 | 0.992 | 0.979 | $0.0095 |
| **GPT-4.1-mini** | **0.979** | **0.979** | **$0.0019** |
| GPT-4o-mini | 0.936 | - | $0.0006 |

**선택 근거**: 1.3%의 품질 저하만으로 80% 비용 절감을 달성하고 충실도를 유지했습니다.

### 기각된 접근법

**하이브리드 검색 (BM25 + Dense)**
- Recall@1: 80.6% → 73.6% (↓7pp)
- 지연시간: 4ms → 25ms (↑6.25배)
- **결정**: 기각 - Dense 단독이 하이브리드보다 우수

**리랭커 (BGE)**
- Recall@1: 80% → 32% (↓48pp)
- 지연시간: 4ms → 4,856ms (↑1,214배)
- **결정**: 기각 - 치명적인 성능 저하

---

## 평가

### 검색 성능

100개 RFP 문서(8,123개 청크)에 대한 72개 쿼리 벤치마크:

| 지표 | 결과 | 정의 |
|------|------|------|
| Recall@1 | 83.3% | Top-1에 관련 문서 포함 |
| Recall@5 | 90.3% | Top-5에 관련 문서 포함 |
| Recall@10 | 93.1% | Top-10에 관련 문서 포함 |
| MRR@10 | 0.860 | 첫 번째 관련 결과의 평균 역순위 |

### 생성 품질

| 지표 | 결과 | 정의 |
|------|------|------|
| Answer Relevancy | 0.979 | 예상 답변과의 의미적 유사도 |
| Faithfulness | 0.979 | 검색된 컨텍스트와의 일관성 |
| 평균 지연시간 | 3,858ms | 종단간 응답 시간 |

### 비용 효율성

- **쿼리당**: $0.0019
- **월간 (10K 쿼리)**: $19
- **비용 절감**: GPT-4.1 기준선 대비 80%

---

## 프로젝트 구조

```
rag-rfp-system/
├── src/
│   ├── cli/
│   │   ├── build_index.py          # 인덱스 구축
│   │   ├── ask.py                  # CLI 쿼리 인터페이스
│   │   └── serve_api.py            # 웹 API 서버
│   ├── rag_rfp/
│   │   ├── io/                     # PDF 파싱 및 정규화
│   │   ├── prep/                   # 청킹 및 임베딩
│   │   ├── index/                  # ChromaDB 인덱싱
│   │   ├── retrieve/               # Dense retrieval
│   │   ├── generate/               # 답변 생성
│   │   └── eval/                   # 평가 스크립트
│   └── langchain_pipeline/         # LangChain 통합
├── data/
│   └── eval/
│       └── results/                # 벤치마크 결과
├── docs/
│   ├── retrieval-evaluation-log.md     # 실험 로그
│   └── generation-model-evaluation.md  # 모델 비교
├── configs/
│   └── config.sample.yaml          # 설정 템플릿
├── requirements.txt
└── README.md
```

---

## 문서

### 평가 보고서
- [검색 평가](docs/retrieval-evaluation-log.md) - 청킹 및 임베딩 실험 종합
- [생성 모델 분석](docs/generation-model-evaluation.md) - 5개 모델 품질-비용 트레이드오프 연구

### 벤치마크 결과
- [청킹 전략](data/eval/results/chunking_strategy_latency_results.md)
- [생성 모델](data/eval/results/generation_model_comparison.md)

---

## 설정

`.env` 파일 옵션:

```env
# 필수
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-large

# 선택사항
LOG_LEVEL=INFO
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
```

---

## API 레퍼런스

### REST 엔드포인트

**POST /query**
```json
{
  "question": "시스템 요구사항은 무엇인가요?",
  "top_k": 5
}
```

응답:
```json
{
  "answer": "...",
  "sources": [...],
  "confidence": 0.95,
  "latency_ms": 3850
}
```

---

## 성능 최적화

### 시맨틱 청킹 알고리즘

1. **문장 분할**: 문장 경계로 문서 분할
2. **임베딩 생성**: text-embedding-3-large로 각 문장 인코딩
3. **유사도 계산**: 연속된 문장 간 코사인 유사도 계산
4. **임계값 기반 병합**: 유사도 ≥ 0.8인 문장 병합
5. **크기 제약**: 청크당 최소 200자, 최대 2000자

**결과**: 21.9% 적은 청크, 2.8pp Recall@5 개선

### Dense Retrieval 최적화

- **ChromaDB HNSW 인덱스**: O(log N) 검색 복잡도
- **평균 4.9ms 지연시간**: 8,123개 청크 규모
- **CPU 기반 임베딩**: VRAM 사용량 최소화

---

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

---

## 연락처

**배현석 (Hyeonseok Bae)**

- GitHub: [@bhs5070](https://github.com/bhs5070)
- Email: bhs5070@gmail.com

---

## 감사의 글

다음 프로젝트를 활용하여 개발되었습니다:
- [FastAPI](https://fastapi.tiangolo.com/) - 모던 웹 프레임워크
- [ChromaDB](https://www.trychroma.com/) - 벡터 데이터베이스
- [LangChain](https://www.langchain.com/) - LLM 오케스트레이션
- [OpenAI](https://openai.com/) - 임베딩 및 생성 모델

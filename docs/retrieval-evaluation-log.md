# Retrieval Evaluation Log

## 목적

이 문서는 `rag-rfp-system` 개인 프로젝트 재시작 과정에서 수행한 검색 성능 실험을 정리한다.

현재 의사결정은 다음 두 가지다.

- 청킹 기본값: `semantic`
- 벡터 DB 방향: `ChromaDB`

임베딩 기본값은 현재까지의 결과 기준으로 `text-embedding-3-large`가 가장 적합했다.

## 데이터와 환경

- 코퍼스: `pdf_files 3` 폴더의 원본문서 100개
- 텍스트 추출 결과: [all_extracted_texts.jsonl](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/interim/all_extracted_texts.jsonl#L1)
- 평가셋: [eval.jsonl](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/eval/eval.jsonl#L1)
- 평가 단위: 문서 단위 retrieval
- 실행 환경: `.venv311` / Python 3.11

추출 및 청킹 산출물:

- [chunking_stats.json](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/processed/advanced_chunks/chunking_stats.json#L1)
- [chunks_multi_aspect.jsonl](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/processed/advanced_chunks/chunks_multi_aspect.jsonl#L1)
- [chunks_structure_aware.jsonl](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/processed/advanced_chunks/chunks_structure_aware.jsonl#L1)
- [chunks_parent_child.jsonl](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/processed/advanced_chunks/chunks_parent_child.jsonl#L1)
- [chunks_sliding_window.jsonl](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/processed/advanced_chunks/chunks_sliding_window.jsonl#L1)
- [chunks_paragraph.jsonl](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/processed/advanced_chunks/chunks_paragraph.jsonl#L1)
- [chunks_page.jsonl](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/processed/advanced_chunks/chunks_page.jsonl#L1)
- [chunks_semantic.jsonl](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/processed/advanced_chunks/chunks_semantic.jsonl#L1)

## 지표 정의

이번 문서의 지표는 모두 문서 단위 중복 제거 기준이다.

- `Recall@k`: 상위 `k`개 고유 문서 안에 정답 문서가 있으면 1, 없으면 0
- `Precision@k`: 상위 `k`개 고유 문서 중 정답 문서 비율
- `MRR@10`: 상위 10개 고유 문서 안에서 정답 문서가 처음 등장한 순위의 역수
- `Latency`: 임베딩 생성 시간을 제외한 retrieval-only 시간

주의:

- 평가셋은 질문당 정답 문서가 1개다.
- 그래서 `Precision@k`는 사실상 `Recall`의 보조 지표로 보면 된다.
- 이전 대화에서 나온 일부 recall 수치와 다른 이유는, 이번 문서에서 문서 중복을 제거한 고유 문서 순위 기준으로 다시 계산했기 때문이다.

## 실험 코드

- 기본 평가 유틸: [evaluate_embedding_chunking_grid.py](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/src/rag_rfp/eval/evaluate_embedding_chunking_grid.py#L1)
- multi-aspect 조합 평가: [evaluate_multi_aspect_combinations.py](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/src/rag_rfp/eval/evaluate_multi_aspect_combinations.py#L1)
- summary/hybrid/reranker 평가: [evaluate_summary_retrieval_strategies.py](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/src/rag_rfp/eval/evaluate_summary_retrieval_strategies.py#L1)
- semantic/page/paragraph 포함 청킹 비교: [evaluate_chunking_strategy_latency.py](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/src/rag_rfp/eval/evaluate_chunking_strategy_latency.py#L1)
- 청킹 생성 코드: [advanced_rfp_chunking.py](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/src/rag_rfp/prep/advanced_rfp_chunking.py#L1)

## 1. `text-embedding-3-small` 기준 기존 청킹 전략 비교

조건:

- 모델: `text-embedding-3-small`
- 전략: `parent_child`, `structure_aware`, `multi_aspect`, `sliding_window`

| Model | Chunking | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `text-embedding-3-small` | `parent_child` | 0.7083 | 0.7639 | 0.7778 | 0.8889 | 0.7083 | 0.2546 | 0.1556 | 0.0889 | 0.7513 |
| `text-embedding-3-small` | `structure_aware` | 0.5694 | 0.7361 | 0.7639 | 0.8750 | 0.5694 | 0.2454 | 0.1528 | 0.0875 | 0.6651 |
| `text-embedding-3-small` | `multi_aspect` | 0.6667 | 0.7778 | 0.8333 | 0.8611 | 0.6667 | 0.2593 | 0.1667 | 0.0861 | 0.7297 |
| `text-embedding-3-small` | `sliding_window` | 0.6528 | 0.7361 | 0.7778 | 0.8472 | 0.6528 | 0.2454 | 0.1556 | 0.0847 | 0.7081 |

해석:

- `R@1`, `MRR@10`은 `parent_child`가 가장 높다.
- `R@5`는 `multi_aspect`가 가장 높다.
- 즉 `small` 단계에서는 "어떤 `k`를 더 중요하게 보느냐"에 따라 우세 전략이 달랐다.

## 2. `small` vs `large` 비교

조건:

- 전략: `multi_aspect`
- 모델: `text-embedding-3-small`, `text-embedding-3-large`

| Model | Chunking | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `text-embedding-3-large` | `multi_aspect` | 0.8056 | 0.8472 | 0.8889 | 0.9028 | 0.8056 | 0.2824 | 0.1778 | 0.0903 | 0.8358 |
| `text-embedding-3-small` | `multi_aspect` | 0.6667 | 0.7778 | 0.8333 | 0.8611 | 0.6667 | 0.2593 | 0.1667 | 0.0861 | 0.7297 |

해석:

- `large`가 모든 지표에서 `small`보다 우세하다.
- 이후 모든 본 실험은 `text-embedding-3-large`를 기준선으로 잡았다.

## 3. multi-aspect 내부 조합 비교

조건:

- 모델: `text-embedding-3-large`
- 비교 대상: `original`, `keywords`, `summary` 및 조합들

원본 결과 파일:

- [multi_aspect_combo_results.md](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/eval/results/multi_aspect_combo_results.md#L1)

| Aspect Combo | Chunks | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR@10 | Avg ms/query | P95 ms/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `keywords+summary` | 20800 | 0.7778 | 0.8611 | 0.8889 | 0.9306 | 0.7778 | 0.2870 | 0.1778 | 0.0931 | 0.8269 | 7.198 | 8.092 |
| `original+summary` | 20802 | 0.8056 | 0.8750 | 0.9028 | 0.9167 | 0.8056 | 0.2917 | 0.1806 | 0.0917 | 0.8456 | 9.694 | 13.454 |
| `summary` | 10401 | 0.7639 | 0.8472 | 0.8889 | 0.9167 | 0.7639 | 0.2824 | 0.1778 | 0.0917 | 0.8149 | 3.745 | 4.481 |
| `original` | 10401 | 0.8194 | 0.8472 | 0.8750 | 0.9167 | 0.8194 | 0.2824 | 0.1750 | 0.0917 | 0.8440 | 4.264 | 5.119 |
| `original+keywords+summary` | 31201 | 0.8056 | 0.8472 | 0.8889 | 0.9028 | 0.8056 | 0.2824 | 0.1778 | 0.0903 | 0.8358 | 10.341 | 11.187 |
| `original+keywords` | 20800 | 0.8056 | 0.8472 | 0.8472 | 0.9028 | 0.8056 | 0.2824 | 0.1694 | 0.0903 | 0.8289 | 8.758 | 13.386 |
| `keywords` | 10399 | 0.6806 | 0.8194 | 0.8194 | 0.8333 | 0.6806 | 0.2731 | 0.1639 | 0.0833 | 0.7471 | 5.441 | 5.972 |

해석:

- `original`은 `R@1`, `MRR@10`, latency가 강하다.
- `keywords+summary`는 `R@10`이 가장 높다.
- `summary`는 속도와 성능의 균형이 좋다.
- 이 단계까지는 `original`과 `summary`가 실전 후보였다.

## 4. summary 기반 retrieval 전략 비교

조건:

- 모델: `text-embedding-3-large`
- 비교 대상:
  - `large+original`
  - `large+summary`
  - `large+keywords+summary`
  - `large+summary+hybrid`
  - `large+summary+reranker`
  - `large+summary+hybrid+reranker`

원본 결과 파일:

- [summary_retrieval_strategy_results.md](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/eval/results/summary_retrieval_strategy_results.md#L1)

| Setup | Chunks | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR@10 | Avg ms/query | P95 ms/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `large+keywords+summary` | 20800 | 0.7778 | 0.8611 | 0.8889 | 0.9306 | 0.7778 | 0.2870 | 0.1778 | 0.0931 | 0.8269 | 8.169 | 11.619 |
| `large+summary+hybrid` | 10401 | 0.7361 | 0.8889 | 0.9167 | 0.9167 | 0.7361 | 0.2963 | 0.1833 | 0.0917 | 0.8072 | 24.671 | 33.339 |
| `large+summary` | 10401 | 0.7639 | 0.8472 | 0.8889 | 0.9167 | 0.7639 | 0.2824 | 0.1778 | 0.0917 | 0.8149 | 3.896 | 4.315 |
| `large+original` | 10401 | 0.8194 | 0.8472 | 0.8750 | 0.9167 | 0.8194 | 0.2824 | 0.1750 | 0.0917 | 0.8440 | 3.267 | 3.995 |
| `large+summary+hybrid+reranker` | 10401 | 0.3194 | 0.6389 | 0.6944 | 0.8056 | 0.3194 | 0.2130 | 0.1389 | 0.0806 | 0.4850 | 6127.252 | 16326.703 |
| `large+summary+reranker` | 10401 | 0.2222 | 0.5000 | 0.5833 | 0.7361 | 0.2222 | 0.1667 | 0.1167 | 0.0736 | 0.3854 | 4856.374 | 14014.266 |

해석:

- `hybrid`는 `summary`의 `R@3`, `R@5`를 올리지만 latency가 커진다.
- 현재 붙인 reranker는 성능도 나쁘고 latency도 매우 크다.
- 따라서 reranker는 현 시점 채택 대상이 아니다.

## 5. semantic / structural / paragraph / page 포함 청킹 전략 비교

조건:

- 모델: `text-embedding-3-large`
- 비교 대상:
  - `original` (`multi_aspect`의 원문 청크만 사용)
  - `structure_aware`
  - `paragraph`
  - `page`
  - `semantic`

원본 결과 파일:

- [chunking_strategy_latency_results.md](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/eval/results/chunking_strategy_latency_results.md#L1)

| Strategy | Chunks | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR@10 | Avg ms/query | P95 ms/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `semantic` | 8123 | 0.8333 | 0.8889 | 0.9028 | 0.9306 | 0.8333 | 0.2963 | 0.1806 | 0.0931 | 0.8604 | 4.900 | 6.035 |
| `original` | 10401 | 0.8194 | 0.8472 | 0.8750 | 0.9167 | 0.8194 | 0.2824 | 0.1750 | 0.0917 | 0.8440 | 5.498 | 5.991 |
| `page` | 7456 | 0.8056 | 0.8472 | 0.8472 | 0.9028 | 0.8056 | 0.2824 | 0.1694 | 0.0903 | 0.8322 | 3.416 | 4.529 |
| `structure_aware` | 11452 | 0.8194 | 0.8750 | 0.8889 | 0.8889 | 0.8194 | 0.2917 | 0.1778 | 0.0889 | 0.8477 | 7.543 | 13.869 |
| `paragraph` | 7389 | 0.8056 | 0.8333 | 0.8611 | 0.8889 | 0.8056 | 0.2778 | 0.1722 | 0.0889 | 0.8262 | 6.500 | 14.955 |

해석:

- 현재 최고 전략은 `semantic`
- `semantic`은 `R@1`, `R@3`, `R@5`, `R@10`, `MRR@10`에서 모두 가장 강하다
- latency도 `structure_aware`, `paragraph`보다 낮고 `original`과 큰 차이가 없다

## 왜 `semantic`이 가장 좋았는가

현재 `semantic`은 완전한 embedding-based semantic chunking은 아니고, 규칙 기반 의미 청킹이다.

- 문단/문장을 먼저 단위화
- 용어 겹침이 낮아지는 지점에서 경계 분리
- 제목처럼 보이는 라인도 경계로 사용
- 이후 적당한 길이로 다시 합침

관련 구현:

- [advanced_rfp_chunking.py](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/src/rag_rfp/prep/advanced_rfp_chunking.py#L325)
- [advanced_rfp_chunking.py](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/src/rag_rfp/prep/advanced_rfp_chunking.py#L389)
- [advanced_rfp_chunking.py](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/src/rag_rfp/prep/advanced_rfp_chunking.py#L421)

이번 RFP 코퍼스에서는 길이 기준 `original`보다 주제 경계가 더 잘 보존되어 query와 chunk 간 의미 순도가 높아진 것으로 해석된다.

## 최종 결정

현재까지의 평가를 기준으로 기본 검색 조합은 다음이 가장 적절하다.

- 청킹: `semantic`
- 임베딩: `text-embedding-3-large`
- 벡터 DB: `ChromaDB`

채택 이유:

- `semantic`이 최고 recall / precision / MRR를 기록
- `text-embedding-3-large`가 `small`보다 전 구간 우세
- `reranker`는 현재 구성에서 비효율적
- `hybrid`는 의미가 있지만 기본값으로는 latency 비용이 큼

## 다음 작업

1. `semantic` 청크를 ChromaDB에 적재하는 인덱싱 경로 구현
2. 현재 검색기 기본값을 `text-embedding-3-large + semantic + ChromaDB`로 고정
3. 필요 시 `semantic + hybrid`를 옵션으로 추가

# Portfolio Retrieval Summary

## 개요

이 문서는 `rag-rfp-system` 개인 프로젝트 재정비 과정에서 수행한 retrieval 성능 비교 실험을 포트폴리오 참고용으로 정리한 요약본이다.

최종적으로 채택한 방향은 다음과 같다.

- Chunking: `semantic`
- Embedding model: `text-embedding-3-large`
- Vector DB: `ChromaDB`

평가 지표는 문서 단위 기준으로 계산했다.

- `Recall@k`
- `Precision@k`
- `MRR@10`
- `Latency (ms/query)`

## 실험 조건

- 문서 코퍼스: PDF 100개
- 평가셋: `data/eval/eval.jsonl`
- 임베딩 비교: `text-embedding-3-small`, `text-embedding-3-large`
- 청킹 비교:
  - `parent_child`
  - `structure_aware`
  - `multi_aspect`
  - `sliding_window`
  - `original`
  - `page`
  - `paragraph`
  - `semantic`

## 1. `text-embedding-3-small` 기준 기존 청킹 전략 비교

| Model | Chunking | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `text-embedding-3-small` | `parent_child` | 0.7083 | 0.7639 | 0.7778 | 0.8889 | 0.7083 | 0.2546 | 0.1556 | 0.0889 | 0.7513 |
| `text-embedding-3-small` | `structure_aware` | 0.5694 | 0.7361 | 0.7639 | 0.8750 | 0.5694 | 0.2454 | 0.1528 | 0.0875 | 0.6651 |
| `text-embedding-3-small` | `multi_aspect` | 0.6667 | 0.7778 | 0.8333 | 0.8611 | 0.6667 | 0.2593 | 0.1667 | 0.0861 | 0.7297 |
| `text-embedding-3-small` | `sliding_window` | 0.6528 | 0.7361 | 0.7778 | 0.8472 | 0.6528 | 0.2454 | 0.1556 | 0.0847 | 0.7081 |

요약:

- `Recall@1`, `MRR@10`은 `parent_child`가 가장 높았다.
- `Recall@5`는 `multi_aspect`가 가장 높았다.

## 2. `small` vs `large` 비교 (`multi_aspect`)

| Model | Chunking | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `text-embedding-3-large` | `multi_aspect` | 0.8056 | 0.8472 | 0.8889 | 0.9028 | 0.8056 | 0.2824 | 0.1778 | 0.0903 | 0.8358 |
| `text-embedding-3-small` | `multi_aspect` | 0.6667 | 0.7778 | 0.8333 | 0.8611 | 0.6667 | 0.2593 | 0.1667 | 0.0861 | 0.7297 |

요약:

- `text-embedding-3-large`가 모든 지표에서 `small`보다 우세했다.

## 3. `multi_aspect` 내부 조합 비교 (`text-embedding-3-large`)

| Aspect Combo | Chunks | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR@10 | Avg ms/query | P95 ms/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `keywords+summary` | 20800 | 0.7778 | 0.8611 | 0.8889 | 0.9306 | 0.7778 | 0.2870 | 0.1778 | 0.0931 | 0.8269 | 7.198 | 8.092 |
| `original+summary` | 20802 | 0.8056 | 0.8750 | 0.9028 | 0.9167 | 0.8056 | 0.2917 | 0.1806 | 0.0917 | 0.8456 | 9.694 | 13.454 |
| `summary` | 10401 | 0.7639 | 0.8472 | 0.8889 | 0.9167 | 0.7639 | 0.2824 | 0.1778 | 0.0917 | 0.8149 | 3.745 | 4.481 |
| `original` | 10401 | 0.8194 | 0.8472 | 0.8750 | 0.9167 | 0.8194 | 0.2824 | 0.1750 | 0.0917 | 0.8440 | 4.264 | 5.119 |
| `original+keywords+summary` | 31201 | 0.8056 | 0.8472 | 0.8889 | 0.9028 | 0.8056 | 0.2824 | 0.1778 | 0.0903 | 0.8358 | 10.341 | 11.187 |
| `original+keywords` | 20800 | 0.8056 | 0.8472 | 0.8472 | 0.9028 | 0.8056 | 0.2824 | 0.1694 | 0.0903 | 0.8289 | 8.758 | 13.386 |
| `keywords` | 10399 | 0.6806 | 0.8194 | 0.8194 | 0.8333 | 0.6806 | 0.2731 | 0.1639 | 0.0833 | 0.7471 | 5.441 | 5.972 |

요약:

- `original`은 `Recall@1`, `MRR@10`, latency가 강했다.
- `keywords+summary`는 `Recall@10`이 가장 높았다.
- `summary`는 속도와 성능 균형이 좋았다.

## 4. `summary` 기반 retrieval 전략 비교 (`text-embedding-3-large`)

| Setup | Chunks | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR@10 | Avg ms/query | P95 ms/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `large+keywords+summary` | 20800 | 0.7778 | 0.8611 | 0.8889 | 0.9306 | 0.7778 | 0.2870 | 0.1778 | 0.0931 | 0.8269 | 8.169 | 11.619 |
| `large+summary+hybrid` | 10401 | 0.7361 | 0.8889 | 0.9167 | 0.9167 | 0.7361 | 0.2963 | 0.1833 | 0.0917 | 0.8072 | 24.671 | 33.339 |
| `large+summary` | 10401 | 0.7639 | 0.8472 | 0.8889 | 0.9167 | 0.7639 | 0.2824 | 0.1778 | 0.0917 | 0.8149 | 3.896 | 4.315 |
| `large+original` | 10401 | 0.8194 | 0.8472 | 0.8750 | 0.9167 | 0.8194 | 0.2824 | 0.1750 | 0.0917 | 0.8440 | 3.267 | 3.995 |
| `large+summary+hybrid+reranker` | 10401 | 0.3194 | 0.6389 | 0.6944 | 0.8056 | 0.3194 | 0.2130 | 0.1389 | 0.0806 | 0.4850 | 6127.252 | 16326.703 |
| `large+summary+reranker` | 10401 | 0.2222 | 0.5000 | 0.5833 | 0.7361 | 0.2222 | 0.1667 | 0.1167 | 0.0736 | 0.3854 | 4856.374 | 14014.266 |

요약:

- `hybrid`는 `summary`의 `Recall@3`, `Recall@5`를 개선했다.
- reranker는 성능과 latency 모두 비효율적이었다.

## 5. 최종 청킹 전략 비교 (`text-embedding-3-large`)

| Strategy | Chunks | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@1 | Precision@3 | Precision@5 | Precision@10 | MRR@10 | Avg ms/query | P95 ms/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `semantic` | 8123 | 0.8333 | 0.8889 | 0.9028 | 0.9306 | 0.8333 | 0.2963 | 0.1806 | 0.0931 | 0.8604 | 4.900 | 6.035 |
| `original` | 10401 | 0.8194 | 0.8472 | 0.8750 | 0.9167 | 0.8194 | 0.2824 | 0.1750 | 0.0917 | 0.8440 | 5.498 | 5.991 |
| `page` | 7456 | 0.8056 | 0.8472 | 0.8472 | 0.9028 | 0.8056 | 0.2824 | 0.1694 | 0.0903 | 0.8322 | 3.416 | 4.529 |
| `structure_aware` | 11452 | 0.8194 | 0.8750 | 0.8889 | 0.8889 | 0.8194 | 0.2917 | 0.1778 | 0.0889 | 0.8477 | 7.543 | 13.869 |
| `paragraph` | 7389 | 0.8056 | 0.8333 | 0.8611 | 0.8889 | 0.8056 | 0.2778 | 0.1722 | 0.0889 | 0.8262 | 6.500 | 14.955 |

요약:

- 최종 성능 기준 최고 전략은 `semantic`
- `semantic`이 `Recall`, `Precision`, `MRR`에서 가장 강했다
- latency도 실사용 가능한 수준이었다

## 최종 결론

현재 프로젝트 기준 최적 조합은 다음과 같다.

- `semantic` chunking
- `text-embedding-3-large`
- `ChromaDB`

이 조합을 선택한 이유:

- `semantic`이 최고 retrieval 성능을 기록
- `text-embedding-3-large`가 `small`보다 일관되게 우세
- reranker는 현재 설정에서 비효율적
- hybrid는 옵션으로는 의미가 있지만 기본값으로 두기엔 latency 비용이 있음

## 참고 파일

- 상세 실험 로그: [retrieval-evaluation-log.md](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/docs/retrieval-evaluation-log.md#L1)
- 청킹 전략 결과: [chunking_strategy_latency_results.md](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/eval/results/chunking_strategy_latency_results.md#L1)
- multi-aspect 조합 결과: [multi_aspect_combo_results.md](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/eval/results/multi_aspect_combo_results.md#L1)
- summary 전략 결과: [summary_retrieval_strategy_results.md](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/eval/results/summary_retrieval_strategy_results.md#L1)

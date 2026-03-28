# Generation Model Evaluation

## 목적

이 문서는 `semantic + text-embedding-3-large + ChromaDB` retrieval을 고정한 상태에서,
generation 모델 후보를 비교하고 최종 모델을 선택한 과정을 정리한다.

평가 기준은 다음 4개다.

- `answer relevancy`
- `faithfulness`
- `latency`
- `cost`

## 평가 설정

- Retrieval 고정:
  - chunking: `semantic`
  - embedding: `text-embedding-3-large`
  - vector DB: `ChromaDB`
- 평가셋: `data/eval/eval.jsonl`
- 문항 수: 72
- generation 후보:
  - `gpt-4.1`
  - `gpt-4o`
  - `gpt-4.1-mini`
  - `gpt-4.1-nano`
  - `gpt-4o-mini`

## 평가 결과

원본 결과 파일:

- [generation_model_comparison.md](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/eval/results/generation_model_comparison.md#L1)
- [generation_model_comparison.json](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/data/eval/results/generation_model_comparison.json#L1)

| Model | Answer Relevancy | Faithfulness | Avg Latency ms | P95 Latency ms | Avg Prompt Tokens | Avg Completion Tokens | Total Cost USD | Avg Cost / Query USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gpt-4.1` | 0.9921 | 0.9853 | 3722.246 | 6878.563 | 3658.2 | 275.7 | 0.6856 | 0.0095 |
| `gpt-4.1-mini` | 0.9792 | 0.9792 | 3858.336 | 7482.462 | 3658.2 | 251.2 | 0.1343 | 0.0019 |
| `gpt-4o` | 0.9479 | 0.9636 | 2197.884 | 3921.970 | 3658.2 | 147.2 | 0.7645 | 0.0106 |
| `gpt-4o-mini` | 0.9364 | 0.9446 | 3485.028 | 6125.143 | 3658.2 | 156.6 | 0.0463 | 0.0006 |
| `gpt-4.1-nano` | 0.9047 | 0.8962 | 1703.898 | 3058.366 | 3658.2 | 172.2 | 0.0313 | 0.0004 |

## 해석

### 1. 최고 품질 모델

- `gpt-4.1`
- `answer relevancy`와 `faithfulness` 모두 최고

하지만 기본 운영 모델로 쓰기엔 비용이 높다.

- 총 비용: `0.6856 USD`
- query당 평균 비용: `0.0095 USD`

## 2. 가장 좋은 실전 트레이드오프

- `gpt-4.1-mini`

이유:

- `gpt-4.1`에 매우 근접한 품질
  - relevancy: `0.9921 -> 0.9792`
  - faithfulness: `0.9853 -> 0.9792`
- 비용은 크게 절감
  - 총 비용: `0.6856 -> 0.1343 USD`
  - query당 평균 비용: `0.0095 -> 0.0019 USD`
- latency는 오히려 `gpt-4.1`보다 크게 좋아지진 않았지만, 품질/비용 균형이 가장 좋다

즉,

- `gpt-4.1`은 최고 성능
- `gpt-4.1-mini`는 최고 효율

실제 서비스/포트폴리오 기본값은 `gpt-4.1-mini`가 맞다.

## 3. 다른 후보들에 대한 판단

### `gpt-4o`

- latency는 빠른 편
- 하지만 이번 실험에서는 `gpt-4.1-mini`보다 품질이 낮고 비용도 더 높다
- 따라서 기본 모델로 채택할 이유가 약하다

### `gpt-4o-mini`

- 저렴하지만 `gpt-4.1-mini`보다 품질 차이가 더 난다
- 초저비용 데모용 후보 정도로는 의미가 있다

### `gpt-4.1-nano`

- 가장 싸고 가장 빠른 축에 가깝지만
- relevancy / faithfulness 하락이 커서 현재 프로젝트 기본값으로는 부적합하다

## 최종 의사결정

현재 프로젝트의 기본 generation 모델은 다음으로 정한다.

- `gpt-4.1-mini`

전체 파이프라인 기본 조합:

- chunking: `semantic`
- embedding: `text-embedding-3-large`
- vector DB: `ChromaDB`
- generation: `gpt-4.1-mini`

## 반영 사항

- 환경 변수 기본값: `OPENAI_CHAT_MODEL=gpt-4.1-mini`
- generator 기본값: 환경 변수 미지정 시 `gpt-4.1-mini`
- config 샘플 기본값: `gpt-4.1-mini`

관련 파일:

- [generator.py](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/src/rag_rfp/generate/generator.py#L1)
- [config.sample.yaml](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/configs/config.sample.yaml#L1)
- [.env](/Users/hyeonseokbae/Desktop/portfolio/rag-rfp-system/.env#L1)

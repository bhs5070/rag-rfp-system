# Recall Summary

## 1. Chunking Comparison

Condition:
- embedding model: `text-embedding-3-small`
- corpus: full 100-document corpus
- chunking candidates: `parent_child`, `structure_aware`, `multi_aspect`, `sliding_window`

| Model | Chunking | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- | --- |
| openai/text-embedding-3-small | multi_aspect | 0.6667 | 0.7639 | 0.8194 | 0.8611 |
| openai/text-embedding-3-small | structure_aware | 0.5694 | 0.6944 | 0.7500 | 0.8611 |
| openai/text-embedding-3-small | sliding_window | 0.6528 | 0.7361 | 0.7778 | 0.8472 |
| openai/text-embedding-3-small | parent_child | 0.7083 | 0.7639 | 0.7778 | 0.8333 |

Best chunking:
- `multi_aspect`

## 2. Embedding Model Comparison

Condition:
- chunking: `multi_aspect`
- corpus: eval set에 등장하는 9개 문서 subset
- reason: full 100-document corpus 기준 로컬 대형 모델은 현재 맥북 MPS 메모리 한계로 비교가 비현실적으로 느리거나 OOM 발생

| Model | Chunking | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- | --- |
| local/jhgan-ko-sbert-multitask | multi_aspect | 0.8194 | 0.8750 | 0.9167 | 0.9722 |
| openai/text-embedding-3-small | multi_aspect | 0.8194 | 0.9028 | 0.9306 | 0.9583 |

Failed due to MPS OOM:
- `local/intfloat-multilingual-e5-large`
- `local/BAAI-bge-m3`

## 3. Recommendation

## 3. Small vs Large

Condition:
- chunking: `multi_aspect`
- corpus: full 100-document corpus

| Model | Chunking | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| --- | --- | --- | --- | --- | --- |
| openai/text-embedding-3-large | multi_aspect | 0.8056 | 0.8333 | 0.8750 | 0.9028 |
| openai/text-embedding-3-small | multi_aspect | 0.6667 | 0.7639 | 0.8194 | 0.8611 |

## 4. Recommendation

- production / practical baseline: `text-embedding-3-small` + `multi_aspect`
- local Korean-first experiment baseline: `jhgan/ko-sbert-multitask` + `multi_aspect`
- best full-corpus OpenAI result: `text-embedding-3-large` + `multi_aspect`

Why:
- `text-embedding-3-small` is stable and strong on the full 100-document comparison used for chunking selection.
- `jhgan/ko-sbert-multitask` performed very well on the subset comparison, but full-corpus evaluation has not been completed yet.
- larger local multilingual models hit current MacBook MPS memory limits.

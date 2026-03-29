# Retrieval Performance Summary

## Overview

This document summarizes the retrieval performance experiments conducted to optimize the RAG-RFP system's search accuracy and efficiency.

Final configuration:

- **Chunking**: Semantic
- **Embedding model**: text-embedding-3-large
- **Vector DB**: ChromaDB

Evaluation metrics (document-level):

- Recall@k
- Precision@k
- MRR@10
- Latency (ms/query)

## Experiment Setup

- **Document corpus**: 100 RFP PDFs
- **Evaluation set**: data/eval/eval.jsonl
- **Embedding models**: text-embedding-3-small, text-embedding-3-large
- **Chunking strategies**: parent_child, structure_aware, multi_aspect, sliding_window, original, page, paragraph, semantic

## Final Chunking Strategy Comparison (text-embedding-3-large)

| Strategy | Chunks | Recall@1 | Recall@5 | MRR@10 | Avg ms/query |
|----------|--------|----------|----------|--------|--------------|
| **semantic** | **8,123** | **0.833** | **0.903** | **0.860** | **4.9** |
| original | 10,401 | 0.819 | 0.875 | 0.844 | 5.5 |
| structure_aware | 11,452 | 0.819 | 0.889 | 0.848 | 7.5 |
| page | 7,456 | 0.806 | 0.847 | 0.832 | 3.4 |
| paragraph | 7,389 | 0.806 | 0.861 | 0.826 | 6.5 |

**Key finding**: semantic achieved best overall performance with 21.9% fewer chunks.

## Embedding Model Impact

| Model | Recall@1 | Recall@5 | MRR@10 | Improvement |
|-------|----------|----------|--------|-------------|
| text-embedding-3-small | 0.667 | 0.875 | 0.730 | Baseline |
| **text-embedding-3-large** | **0.806** | **0.903** | **0.836** | **+13.9pp** |

**Key finding**: Large model improved Recall@1 by 13.9 percentage points.

## Rejected Approaches

**Hybrid Search (BM25 + Dense)**
- Recall@1: 0.806 → 0.736 (↓7pp)
- Latency: 4ms → 25ms (↑6.25×)
- **Decision**: Dense-only outperformed

**Reranker (BGE)**
- Recall@1: 0.80 → 0.32 (↓48pp)
- Latency: 4ms → 4,856ms (↑1,214×)
- **Decision**: Catastrophic performance degradation

## Final Configuration Rationale

1. **semantic** chunking achieved highest retrieval accuracy
2. **text-embedding-3-large** significantly improved precision
3. **Dense retrieval** proved most efficient for RFP domain
4. **ChromaDB** provided optimal latency/accuracy tradeoff

See [retrieval-evaluation-log.md](retrieval-evaluation-log.md) for detailed experiments.

# RFP Analyzer

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An intelligent document analysis system for RFP (Request for Proposal) documents using RAG (Retrieval-Augmented Generation) architecture.

## Overview

RFP Analyzer automatically parses, chunks, and indexes RFP documents to enable instant retrieval of critical requirements. Built with semantic chunking and state-of-the-art embedding models, it achieves 90.3% Recall@5 while reducing operational costs by 80%.

### Key Performance Metrics

| Metric | Result | Status |
|--------|--------|--------|
| **Recall@5** | 90.3% | ✅ Production Ready |
| **Recall@1** | 83.3% | ✅ High Precision |
| **Answer Relevancy** | 0.979 | ✅ Low Hallucination |
| **Faithfulness** | 0.979 | ✅ Source Fidelity |
| **Cost per Query** | $0.0019 | ✅ 80% Cost Reduction |

---

## Features

### Semantic Chunking
- **Context-aware segmentation** preserving requirement boundaries
- **21.9% chunk reduction** (10,401 → 8,123 chunks)
- **90.3% Recall@5** outperforming fixed-size strategies

### High-Performance Retrieval
- **text-embedding-3-large** for superior semantic understanding
- **ChromaDB** vector store with 4.9ms average query latency
- **Dense retrieval** optimized for technical documents

### Cost-Optimized Generation
- **GPT-4.1-mini** balancing quality and cost
- **80% cost reduction** vs GPT-4.1 ($0.0095 → $0.0019/query)
- **0.979 faithfulness score** minimizing hallucination

### Web Interface
- FastAPI-powered RESTful API
- Real-time query processing
- Source citation and transparency

---

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Installation

```bash
# Clone repository
git clone https://github.com/bhs5070/rag-rfp-system.git
cd rag-rfp-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create `.env` file:

```env
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-large
```

### Build Index

```bash
PYTHONPATH=. python src/cli/build_index.py
```

### Run Server

```bash
PYTHONPATH=. python -m uvicorn src.cli.serve_api:app --host 0.0.0.0 --port 8000 --reload
```

Access at `http://localhost:8000`

---

## Architecture

```
┌─────────────────────────────────────────────┐
│          PDF Document Ingestion             │
│  PyMuPDF → Text Extraction → Normalization  │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│          Semantic Chunking                  │
│  Sentence Segmentation → Similarity-based   │
│  Merging → 8,123 chunks (avg 875 chars)     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│     Embedding (text-embedding-3-large)      │
│  3072-dimensional vectors → ChromaDB        │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         Dense Retrieval (Top-K=5)           │
│  Cosine Similarity → 4.9ms avg latency      │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│      Answer Generation (GPT-4.1-mini)       │
│  Context + Query → Response (3.8s avg)      │
└─────────────────────────────────────────────┘
```

---

## Technical Decisions

### Chunking Strategy

We evaluated 7 chunking strategies on a 72-query benchmark:

| Strategy | Chunks | Recall@5 | Latency (ms) | Decision |
|----------|--------|----------|--------------|----------|
| **Semantic** | **8,123** | **90.3%** | **4.9** | ✅ Selected |
| Original | 10,401 | 87.5% | 5.5 | - |
| Structure-aware | 11,452 | 88.9% | 7.5 | - |
| Page | 7,456 | 84.7% | 3.4 | - |
| Paragraph | 7,389 | 86.1% | 6.5 | - |

**Rationale**: Semantic chunking achieved highest recall while reducing chunk count and maintaining low latency.

### Embedding Model

| Model | Recall@1 | Recall@5 | MRR@10 | Improvement |
|-------|----------|----------|--------|-------------|
| text-embedding-3-small | 66.7% | 87.5% | 0.730 | Baseline |
| **text-embedding-3-large** | **80.6%** | **90.3%** | **0.836** | **+13.9pp** |

**Rationale**: 13.9pp Recall@1 improvement and 14.5% MRR gain justified the marginal cost increase.

### Generation Model

| Model | Answer Relevancy | Faithfulness | Cost/Query |
|-------|-----------------|--------------|------------|
| GPT-4.1 | 0.992 | 0.979 | $0.0095 |
| **GPT-4.1-mini** | **0.979** | **0.979** | **$0.0019** |
| GPT-4o-mini | 0.936 | - | $0.0006 |

**Rationale**: 80% cost reduction with only 1.3% quality degradation and maintained faithfulness.

### Rejected Approaches

**Hybrid Search (BM25 + Dense)**
- Recall@1: 80.6% → 73.6% (↓7pp)
- Latency: 4ms → 25ms (↑6.25×)
- **Decision**: Rejected - Dense-only outperformed hybrid

**Reranker (BGE)**
- Recall@1: 80% → 32% (↓48pp)
- Latency: 4ms → 4,856ms (↑1,214×)
- **Decision**: Rejected - Catastrophic performance degradation

---

## Evaluation

### Retrieval Performance

72-query benchmark on 100 RFP documents (8,123 chunks):

| Metric | Result | Definition |
|--------|--------|------------|
| Recall@1 | 83.3% | Top-1 contains relevant document |
| Recall@5 | 90.3% | Top-5 contains relevant document |
| Recall@10 | 93.1% | Top-10 contains relevant document |
| MRR@10 | 0.860 | Mean reciprocal rank of first relevant result |

### Generation Quality

| Metric | Result | Definition |
|--------|--------|------------|
| Answer Relevancy | 0.979 | Semantic similarity to expected answer |
| Faithfulness | 0.979 | Consistency with retrieved context |
| Avg Latency | 3,858ms | End-to-end response time |

### Cost Efficiency

- **Per Query**: $0.0019
- **Monthly (10K queries)**: $19
- **Cost Reduction**: 80% vs GPT-4.1 baseline

---

## Project Structure

```
rag-rfp-system/
├── src/
│   ├── cli/
│   │   ├── build_index.py          # Index construction
│   │   ├── ask.py                  # CLI query interface
│   │   └── serve_api.py            # Web API server
│   ├── rag_rfp/
│   │   ├── io/                     # PDF parsing & normalization
│   │   ├── prep/                   # Chunking & embedding
│   │   ├── index/                  # ChromaDB indexing
│   │   ├── retrieve/               # Dense retrieval
│   │   ├── generate/               # Answer generation
│   │   └── eval/                   # Evaluation scripts
│   └── langchain_pipeline/         # LangChain integration
├── data/
│   └── eval/
│       └── results/                # Benchmark results
├── docs/
│   ├── retrieval-evaluation-log.md     # Experiment log
│   └── generation-model-evaluation.md  # Model comparison
├── configs/
│   └── config.sample.yaml          # Configuration template
├── requirements.txt
└── README.md
```

---

## Documentation

### Evaluation Reports
- [Retrieval Evaluation](docs/retrieval-evaluation-log.md) - Comprehensive chunking and embedding experiments
- [Generation Model Analysis](docs/generation-model-evaluation.md) - 5-model quality-cost tradeoff study

### Benchmark Results
- [Chunking Strategies](data/eval/results/chunking_strategy_latency_results.md)
- [Generation Models](data/eval/results/generation_model_comparison.md)

---

## Configuration

`.env` file options:

```env
# Required
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-large

# Optional
LOG_LEVEL=INFO
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
```

---

## API Reference

### REST Endpoints

**POST /query**
```json
{
  "question": "What are the system requirements?",
  "top_k": 5
}
```

Response:
```json
{
  "answer": "...",
  "sources": [...],
  "confidence": 0.95,
  "latency_ms": 3850
}
```

---

## Performance Optimization

### Semantic Chunking Algorithm

1. **Sentence Segmentation**: Split documents by sentence boundaries
2. **Embedding Generation**: Encode each sentence with text-embedding-3-large
3. **Similarity Computation**: Calculate cosine similarity between consecutive sentences
4. **Threshold-based Merging**: Merge sentences with similarity ≥ 0.8
5. **Size Constraints**: Min 200 chars, max 2000 chars per chunk

**Results**: 21.9% fewer chunks, 2.8pp Recall@5 improvement

### Dense Retrieval Optimization

- **ChromaDB HNSW index** for O(log N) search complexity
- **4.9ms average latency** at 8,123 chunk scale
- **CPU-based embedding** to minimize VRAM footprint

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

**Hyeonseok Bae**

- GitHub: [@bhs5070](https://github.com/bhs5070)
- Email: bhs5070@gmail.com

---

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [LangChain](https://www.langchain.com/) - LLM orchestration
- [OpenAI](https://openai.com/) - Embedding and generation models

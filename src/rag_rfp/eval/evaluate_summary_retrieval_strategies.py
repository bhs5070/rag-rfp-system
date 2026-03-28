from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.rag_rfp.eval.evaluate_embedding_chunking_grid import (
    DEFAULT_CHUNK_DIR,
    DEFAULT_EVAL_PATH,
    DEFAULT_OUTPUT_DIR,
    EvalSample,
    l2_normalize,
    load_eval_samples,
    load_chunks,
    ranked_unique_doc_ids,
    summarize_ranked_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMBED_CACHE_DIR = PROJECT_ROOT / "data" / "eval" / "embedding_cache"
MULTI_ASPECT_CHUNKS = DEFAULT_CHUNK_DIR / "chunks_multi_aspect.jsonl"
QUERY_EMBEDDINGS = EMBED_CACHE_DIR / "openai-text-embedding-3-large__eval_queries.npy"
CHUNK_EMBEDDINGS = EMBED_CACHE_DIR / "openai-text-embedding-3-large__multi_aspect.npy"
RERANKER_MODEL = "BAAI/bge-reranker-base"


def tokenize(text: str) -> List[str]:
    return re.findall(r"[가-힣A-Za-z0-9]+", text.lower())


def select_aspects(chunks: Sequence[Dict], vectors: np.ndarray, aspects: set[str]) -> tuple[List[Dict], np.ndarray]:
    indices = [idx for idx, chunk in enumerate(chunks) if chunk.get("aspect") in aspects]
    return [chunks[idx] for idx in indices], vectors[indices]


def compute_recall_and_latency(
    samples: Sequence[EvalSample],
    query_vectors: np.ndarray,
    chunks: Sequence[Dict],
    chunk_vectors: np.ndarray,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    chunk_vectors = l2_normalize(chunk_vectors)
    query_vectors = l2_normalize(query_vectors)
    chunk_doc_ids = [chunk["doc_id"] for chunk in chunks]

    totals = {f"recall@{k}": 0.0 for k in k_values}
    totals.update({f"precision@{k}": 0.0 for k in k_values})
    totals["mrr@10"] = 0.0
    latencies: List[float] = []

    for query_idx, sample in enumerate(samples):
        start = time.perf_counter()
        similarities = query_vectors[query_idx] @ chunk_vectors.T
        retrieved_doc_ids = ranked_unique_doc_ids(similarities, chunk_doc_ids, top_k=max(max(k_values), 10))
        latencies.append((time.perf_counter() - start) * 1000.0)

        metrics = summarize_ranked_metrics(retrieved_doc_ids, sample.gt_doc_id, k_values=k_values, mrr_cutoff=10)
        for metric_name, value in metrics.items():
            totals[metric_name] += value

    total = len(samples)
    return {
        "chunks": float(len(chunks)),
        **{metric_name: metric_value / total for metric_name, metric_value in totals.items()},
        "latency_avg_ms": float(np.mean(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }


class SummaryHybridEvaluator:
    def __init__(self, samples: Sequence[EvalSample], query_vectors: np.ndarray, chunks: Sequence[Dict], vectors: np.ndarray) -> None:
        self.samples = samples
        self.query_vectors = l2_normalize(query_vectors)
        self.chunks = list(chunks)
        self.chunk_vectors = l2_normalize(vectors)
        self.chunk_doc_ids = [chunk["doc_id"] for chunk in self.chunks]
        self.chunk_texts = [chunk["text"] for chunk in self.chunks]
        self.tokenized_corpus = [tokenize(text) for text in self.chunk_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def dense_candidates(self, query_idx: int, top_n: int) -> List[int]:
        similarities = self.query_vectors[query_idx] @ self.chunk_vectors.T
        if top_n >= len(self.chunks):
            return np.argsort(similarities)[::-1].tolist()
        part = np.argpartition(similarities, -top_n)[-top_n:]
        ordered = part[np.argsort(similarities[part])[::-1]]
        return ordered.tolist()

    def sparse_candidates(self, query_text: str, top_n: int) -> List[int]:
        scores = self.bm25.get_scores(tokenize(query_text))
        if top_n >= len(self.chunks):
            return np.argsort(scores)[::-1].tolist()
        part = np.argpartition(scores, -top_n)[-top_n:]
        ordered = part[np.argsort(scores[part])[::-1]]
        return ordered.tolist()

    def hybrid_candidates(self, query_idx: int, query_text: str, top_n: int, rrf_k: int = 60) -> List[int]:
        fused: Dict[int, float] = {}
        for rank, idx in enumerate(self.dense_candidates(query_idx, top_n), start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        for rank, idx in enumerate(self.sparse_candidates(query_text, top_n), start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        return [idx for idx, _score in sorted(fused.items(), key=lambda item: item[1], reverse=True)]


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rerank(self, query: str, texts: Sequence[str], candidate_indices: Sequence[int], top_k: int) -> List[int]:
        if not candidate_indices:
            return []
        pairs = [[query, texts[idx]] for idx in candidate_indices]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(dim=1).detach().cpu().numpy()
        ranked = sorted(zip(candidate_indices, logits), key=lambda item: item[1], reverse=True)
        return [idx for idx, _score in ranked[:top_k]]


def evaluate_strategy(
    name: str,
    samples: Sequence[EvalSample],
    retriever: SummaryHybridEvaluator,
    strategy: str,
    candidate_top_n: int = 50,
    final_top_k: int = 10,
    reranker: CrossEncoderReranker | None = None,
) -> Dict[str, float]:
    totals = {f"recall@{k}": 0.0 for k in (1, 3, 5, 10)}
    totals.update({f"precision@{k}": 0.0 for k in (1, 3, 5, 10)})
    totals["mrr@10"] = 0.0
    latencies: List[float] = []

    for query_idx, sample in enumerate(samples):
        start = time.perf_counter()

        if strategy == "summary_hybrid":
            ranked_indices = retriever.hybrid_candidates(query_idx, sample.question, candidate_top_n)[:final_top_k]
        elif strategy == "summary_reranker":
            dense = retriever.dense_candidates(query_idx, candidate_top_n)
            ranked_indices = reranker.rerank(sample.question, retriever.chunk_texts, dense, final_top_k) if reranker else []
        elif strategy == "summary_hybrid_reranker":
            fused = retriever.hybrid_candidates(query_idx, sample.question, candidate_top_n)
            ranked_indices = reranker.rerank(sample.question, retriever.chunk_texts, fused[:candidate_top_n], final_top_k) if reranker else []
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        retrieved_doc_ids = []
        seen: set[str] = set()
        for idx in ranked_indices:
            doc_id = retriever.chunk_doc_ids[idx]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            retrieved_doc_ids.append(doc_id)
        latencies.append((time.perf_counter() - start) * 1000.0)

        metrics = summarize_ranked_metrics(retrieved_doc_ids, sample.gt_doc_id, k_values=(1, 3, 5, 10), mrr_cutoff=10)
        for metric_name, value in metrics.items():
            totals[metric_name] += value

    total = len(samples)
    return {
        "strategy": name,
        "chunks": float(len(retriever.chunks)),
        **{metric_name: metric_value / total for metric_name, metric_value in totals.items()},
        "latency_avg_ms": float(np.mean(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }


def markdown_table(rows: Sequence[Dict[str, float]]) -> str:
    headers = [
        "Setup",
        "Chunks",
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "Precision@1",
        "Precision@3",
        "Precision@5",
        "Precision@10",
        "MRR@10",
        "Avg ms/query",
        "P95 ms/query",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["strategy"]),
                    str(int(row["chunks"])),
                    f'{row["recall@1"]:.4f}',
                    f'{row["recall@3"]:.4f}',
                    f'{row["recall@5"]:.4f}',
                    f'{row["recall@10"]:.4f}',
                    f'{row["precision@1"]:.4f}',
                    f'{row["precision@3"]:.4f}',
                    f'{row["precision@5"]:.4f}',
                    f'{row["precision@10"]:.4f}',
                    f'{row["mrr@10"]:.4f}',
                    f'{row["latency_avg_ms"]:.3f}',
                    f'{row["latency_p95_ms"]:.3f}',
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def run() -> Dict[str, object]:
    samples = load_eval_samples(DEFAULT_EVAL_PATH)
    all_chunks = load_chunks(MULTI_ASPECT_CHUNKS)
    query_vectors = np.load(QUERY_EMBEDDINGS)
    all_vectors = np.load(CHUNK_EMBEDDINGS)

    rows: List[Dict[str, float]] = []

    for name, aspects in [
        ("large+original", {"original"}),
        ("large+summary", {"summary"}),
        ("large+keywords+summary", {"keywords", "summary"}),
    ]:
        chunks, vectors = select_aspects(all_chunks, all_vectors, aspects)
        metrics = compute_recall_and_latency(samples, query_vectors, chunks, vectors)
        rows.append({"strategy": name, **metrics})

    summary_chunks, summary_vectors = select_aspects(all_chunks, all_vectors, {"summary"})
    hybrid = SummaryHybridEvaluator(samples, query_vectors, summary_chunks, summary_vectors)
    reranker = CrossEncoderReranker()

    rows.append(evaluate_strategy("large+summary+hybrid", samples, hybrid, "summary_hybrid"))
    rows.append(evaluate_strategy("large+summary+reranker", samples, hybrid, "summary_reranker", reranker=reranker))
    rows.append(
        evaluate_strategy(
            "large+summary+hybrid+reranker",
            samples,
            hybrid,
            "summary_hybrid_reranker",
            reranker=reranker,
        )
    )

    rows.sort(key=lambda row: (row["recall@10"], row["recall@5"], row["recall@3"], row["recall@1"]), reverse=True)
    payload = {
        "model": "openai/text-embedding-3-large",
        "candidate_top_n": 50,
        "final_top_k": 10,
        "reranker_model": RERANKER_MODEL,
        "results": rows,
        "table": markdown_table(rows),
        "best": rows[0] if rows else None,
    }

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (DEFAULT_OUTPUT_DIR / "summary_retrieval_strategy_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (DEFAULT_OUTPUT_DIR / "summary_retrieval_strategy_results.md").write_text(payload["table"], encoding="utf-8")
    return payload


if __name__ == "__main__":
    payload = run()
    print(payload["table"])

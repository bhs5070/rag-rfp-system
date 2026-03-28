from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

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

ASPECT_COMBINATIONS = [
    ("original", {"original"}),
    ("keywords", {"keywords"}),
    ("summary", {"summary"}),
    ("original+keywords", {"original", "keywords"}),
    ("original+summary", {"original", "summary"}),
    ("keywords+summary", {"keywords", "summary"}),
    ("original+keywords+summary", {"original", "keywords", "summary"}),
]


def evaluate_combo(
    samples: Sequence[EvalSample],
    query_vectors: np.ndarray,
    chunks: Sequence[Dict],
    chunk_vectors: np.ndarray,
    aspects: set[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    selected_indices = [idx for idx, chunk in enumerate(chunks) if chunk.get("aspect") in aspects]
    selected_chunks = [chunks[idx] for idx in selected_indices]
    selected_vectors = chunk_vectors[selected_indices]
    selected_doc_ids = [chunk["doc_id"] for chunk in selected_chunks]

    selected_vectors = l2_normalize(selected_vectors)
    query_vectors = l2_normalize(query_vectors)

    totals = {f"recall@{k}": 0.0 for k in k_values}
    totals.update({f"precision@{k}": 0.0 for k in k_values})
    totals["mrr@10"] = 0.0
    per_query_ms: List[float] = []

    for row_index, sample in enumerate(samples):
        start = time.perf_counter()
        similarities = query_vectors[row_index] @ selected_vectors.T
        retrieved_doc_ids = ranked_unique_doc_ids(similarities, selected_doc_ids, top_k=max(max(k_values), 10))
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_query_ms.append(elapsed_ms)

        metrics = summarize_ranked_metrics(retrieved_doc_ids, sample.gt_doc_id, k_values=k_values, mrr_cutoff=10)
        for metric_name, value in metrics.items():
            totals[metric_name] += value

    total = len(samples)
    return {
        "chunk_count": float(len(selected_chunks)),
        **{metric_name: metric_value / total for metric_name, metric_value in totals.items()},
        "latency_avg_ms": float(np.mean(per_query_ms)),
        "latency_p95_ms": float(np.percentile(per_query_ms, 95)),
    }


def markdown_table(rows: Sequence[Dict[str, float]]) -> str:
    headers = [
        "Aspect Combo",
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
                    str(row["combo"]),
                    str(int(row["chunk_count"])),
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
    chunks = load_chunks(MULTI_ASPECT_CHUNKS)
    query_vectors = np.load(QUERY_EMBEDDINGS)
    chunk_vectors = np.load(CHUNK_EMBEDDINGS)

    rows: List[Dict[str, float]] = []
    for combo_name, aspects in ASPECT_COMBINATIONS:
        metrics = evaluate_combo(samples, query_vectors, chunks, chunk_vectors, aspects)
        rows.append({"combo": combo_name, **metrics})

    rows.sort(
        key=lambda row: (row["recall@10"], row["recall@5"], row["recall@3"], row["recall@1"]),
        reverse=True,
    )

    payload = {
        "model": "openai/text-embedding-3-large",
        "chunking": "multi_aspect",
        "latency_note": "retrieval-only latency using cached embeddings and in-memory numpy similarity",
        "results": rows,
        "table": markdown_table(rows),
        "best": rows[0] if rows else None,
    }

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (DEFAULT_OUTPUT_DIR / "multi_aspect_combo_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (DEFAULT_OUTPUT_DIR / "multi_aspect_combo_results.md").write_text(payload["table"], encoding="utf-8")
    return payload


if __name__ == "__main__":
    payload = run()
    print(payload["table"])

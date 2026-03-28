from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.rag_rfp.eval.evaluate_embedding_chunking_grid import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CHUNK_DIR,
    DEFAULT_EVAL_PATH,
    DEFAULT_OUTPUT_DIR,
    OpenAIEmbedder,
    cache_paths,
    encode_with_cache,
    l2_normalize,
    load_chunks,
    load_eval_samples,
    ranked_unique_doc_ids,
    summarize_ranked_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_KEY = "openai/text-embedding-3-large"
MODEL_NAME = "text-embedding-3-large"
MULTI_ASPECT_PATH = DEFAULT_CHUNK_DIR / "chunks_multi_aspect.jsonl"

STRATEGIES: List[Tuple[str, str]] = [
    ("original", "length_based"),
    ("structure_aware", "file"),
    ("paragraph", "file"),
    ("page", "file"),
    ("semantic", "file"),
]


def select_original_chunks(all_chunks: Sequence[Dict]) -> List[Dict]:
    return [chunk for chunk in all_chunks if chunk.get("aspect") == "original"]


def load_strategy_chunks(strategy_key: str) -> List[Dict]:
    if strategy_key == "original":
        return select_original_chunks(load_chunks(MULTI_ASPECT_PATH))
    return load_chunks(DEFAULT_CHUNK_DIR / f"chunks_{strategy_key}.jsonl")


def load_strategy_vectors(
    strategy_key: str,
    chunks: Sequence[Dict],
    cache_dir: Path,
) -> np.ndarray:
    if strategy_key == "original":
        npy_path, meta_path = cache_paths(cache_dir, MODEL_KEY, "multi_aspect")
        if not npy_path.exists() or not meta_path.exists():
            raise RuntimeError("multi_aspect cache is missing for text-embedding-3-large.")
        all_vectors = np.load(npy_path)
        all_chunks = load_chunks(MULTI_ASPECT_PATH)
        selected_indices = [idx for idx, chunk in enumerate(all_chunks) if chunk.get("aspect") == "original"]
        return all_vectors[selected_indices]

    embedder = OpenAIEmbedder(MODEL_NAME)
    texts = [chunk["text"] for chunk in chunks]
    return encode_with_cache(embedder, texts, cache_dir, MODEL_KEY, strategy_key)


def evaluate_strategy(
    strategy_key: str,
    samples,
    query_vectors: np.ndarray,
    chunks: Sequence[Dict],
    chunk_vectors: np.ndarray,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    query_vectors = l2_normalize(query_vectors)
    chunk_vectors = l2_normalize(chunk_vectors)
    chunk_doc_ids = [chunk["doc_id"] for chunk in chunks]

    totals = {f"recall@{k}": 0.0 for k in k_values}
    totals.update({f"precision@{k}": 0.0 for k in k_values})
    totals["mrr@10"] = 0.0
    per_query_ms: List[float] = []

    for row_index, sample in enumerate(samples):
        start = time.perf_counter()
        similarities = query_vectors[row_index] @ chunk_vectors.T
        retrieved_doc_ids = ranked_unique_doc_ids(similarities, chunk_doc_ids, top_k=max(max(k_values), 10))
        per_query_ms.append((time.perf_counter() - start) * 1000.0)

        metrics = summarize_ranked_metrics(retrieved_doc_ids, sample.gt_doc_id, k_values=k_values, mrr_cutoff=10)
        for metric_name, value in metrics.items():
            totals[metric_name] += value

    total = len(samples)
    return {
        "strategy": strategy_key,
        "chunk_count": float(len(chunks)),
        **{metric_name: metric_value / total for metric_name, metric_value in totals.items()},
        "latency_avg_ms": float(np.mean(per_query_ms)),
        "latency_p95_ms": float(np.percentile(per_query_ms, 95)),
    }


def markdown_table(rows: Sequence[Dict[str, float]]) -> str:
    headers = [
        "Strategy",
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


def run(
    eval_path: Path = DEFAULT_EVAL_PATH,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, object]:
    samples = load_eval_samples(eval_path)
    questions = [sample.question for sample in samples]
    embedder = OpenAIEmbedder(MODEL_NAME)
    query_vectors = encode_with_cache(embedder, questions, cache_dir, MODEL_KEY, "eval_queries")

    rows: List[Dict[str, float]] = []
    for strategy_key, _source in STRATEGIES:
        chunks = load_strategy_chunks(strategy_key)
        vectors = load_strategy_vectors(strategy_key, chunks, cache_dir)
        rows.append(evaluate_strategy(strategy_key, samples, query_vectors, chunks, vectors))

    rows.sort(key=lambda row: (row["recall@10"], row["recall@5"], row["recall@3"], row["recall@1"]), reverse=True)

    payload = {
        "model": MODEL_NAME,
        "rows": rows,
        "table": markdown_table(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chunking_strategy_latency_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "chunking_strategy_latency_results.md").write_text(payload["table"], encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run()
    print(result["table"])

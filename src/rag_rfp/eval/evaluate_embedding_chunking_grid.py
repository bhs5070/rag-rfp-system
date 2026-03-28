from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "eval.jsonl"
DEFAULT_CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "advanced_chunks"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "eval" / "embedding_cache"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "results"

CHUNK_FILES = {
    "parent_child": "chunks_parent_child.jsonl",
    "structure_aware": "chunks_structure_aware.jsonl",
    "multi_aspect": "chunks_multi_aspect.jsonl",
    "sliding_window": "chunks_sliding_window.jsonl",
    "paragraph": "chunks_paragraph.jsonl",
    "page": "chunks_page.jsonl",
    "semantic": "chunks_semantic.jsonl",
}

MODEL_SPECS = {
    "openai/text-embedding-3-small": {"kind": "openai", "name": "text-embedding-3-small"},
    "openai/text-embedding-3-large": {"kind": "openai", "name": "text-embedding-3-large"},
    "local/jhgan-ko-sbert-multitask": {"kind": "local", "name": "jhgan/ko-sbert-multitask"},
    "local/intfloat-multilingual-e5-large": {"kind": "local", "name": "intfloat/multilingual-e5-large"},
    "local/BAAI-bge-m3": {"kind": "local", "name": "BAAI/bge-m3"},
}


@dataclass
class EvalSample:
    sample_id: str
    question: str
    gt_doc_id: str


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value)


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def load_eval_samples(path: Path) -> List[EvalSample]:
    rows: List[EvalSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                EvalSample(
                    sample_id=item["id"],
                    question=normalize_text(item["question"].strip().strip('"')),
                    gt_doc_id=normalize_text(item["gt_doc_id"]),
                )
            )
    return rows


def load_chunks(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        chunks = [json.loads(line) for line in handle if line.strip()]
    for chunk in chunks:
        chunk["text"] = normalize_text(chunk["text"])
        chunk["doc_id"] = normalize_text(chunk["doc_id"])
    return chunks


class BaseEmbedder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model_name: str) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        batch_size = 128
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            while True:
                try:
                    response = self.client.embeddings.create(model=self.model_name, input=batch)
                    vectors.extend(item.embedding for item in response.data)
                    break
                except Exception as exc:
                    message = str(exc)
                    if "rate_limit_exceeded" not in message and "Rate limit reached" not in message:
                        raise

                    wait_seconds = 2.0
                    match = re.search(r"Please try again in ([0-9.]+)(ms|s)", message)
                    if match:
                        amount = float(match.group(1))
                        unit = match.group(2)
                        wait_seconds = amount / 1000.0 if unit == "ms" else amount
                    time.sleep(max(wait_seconds, 1.0))
        return np.asarray(vectors, dtype=np.float32)


class LocalEmbedder(BaseEmbedder):
    def __init__(self, model_name: str) -> None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(model_key: str) -> BaseEmbedder:
    spec = MODEL_SPECS[model_key]
    if spec["kind"] == "openai":
        return OpenAIEmbedder(spec["name"])
    return LocalEmbedder(spec["name"])


def cache_paths(cache_dir: Path, model_key: str, corpus_key: str) -> tuple[Path, Path]:
    stem = f"{slugify(model_key)}__{slugify(corpus_key)}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.meta.json"


def encode_with_cache(
    embedder: BaseEmbedder,
    texts: Sequence[str],
    cache_dir: Path,
    model_key: str,
    corpus_key: str,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy_path, meta_path = cache_paths(cache_dir, model_key, corpus_key)

    if npy_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("count") == len(texts):
            return np.load(npy_path)

    vectors = embedder.encode(texts)
    np.save(npy_path, vectors)
    meta_path.write_text(json.dumps({"count": len(texts)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return vectors


def top_k_doc_ids(similarities: np.ndarray, doc_ids: Sequence[str], top_k: int) -> List[str]:
    if top_k >= len(doc_ids):
        indices = np.argsort(similarities)[::-1]
    else:
        part = np.argpartition(similarities, -top_k)[-top_k:]
        indices = part[np.argsort(similarities[part])[::-1]]
    return [doc_ids[index] for index in indices]


def ranked_unique_doc_ids(similarities: np.ndarray, doc_ids: Sequence[str], top_k: int | None = None) -> List[str]:
    indices = np.argsort(similarities)[::-1]
    ranked: List[str] = []
    seen: set[str] = set()

    for index in indices:
        doc_id = doc_ids[index]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append(doc_id)
        if top_k is not None and len(ranked) >= top_k:
            break

    return ranked


def summarize_ranked_metrics(
    ranked_doc_ids: Sequence[str],
    gt_doc_id: str,
    k_values: Sequence[int] = (1, 3, 5, 10),
    mrr_cutoff: int = 10,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    for k in k_values:
        window = list(ranked_doc_ids[:k])
        hit = 1.0 if gt_doc_id in window else 0.0
        metrics[f"recall@{k}"] = hit
        metrics[f"precision@{k}"] = hit / k

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:mrr_cutoff], start=1):
        if doc_id == gt_doc_id:
            reciprocal_rank = 1.0 / rank
            break
    metrics[f"mrr@{mrr_cutoff}"] = reciprocal_rank
    return metrics


def evaluate_recall(
    query_vectors: np.ndarray,
    chunk_vectors: np.ndarray,
    chunk_doc_ids: Sequence[str],
    samples: Sequence[EvalSample],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    query_vectors = l2_normalize(query_vectors)
    chunk_vectors = l2_normalize(chunk_vectors)
    scores = query_vectors @ chunk_vectors.T

    totals = {f"recall@{k}": 0.0 for k in k_values}
    totals.update({f"precision@{k}": 0.0 for k in k_values})
    totals["mrr@10"] = 0.0

    for row_index, sample in enumerate(samples):
        ranked_doc_ids = ranked_unique_doc_ids(scores[row_index], chunk_doc_ids, top_k=max(max(k_values), 10))
        metrics = summarize_ranked_metrics(ranked_doc_ids, sample.gt_doc_id, k_values=k_values, mrr_cutoff=10)
        for metric_name, value in metrics.items():
            totals[metric_name] += value

    total = len(samples)
    return {metric_name: metric_value / total for metric_name, metric_value in totals.items()}


def markdown_table(rows: Sequence[Dict[str, object]]) -> str:
    headers = [
        "Model",
        "Chunking",
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "Precision@1",
        "Precision@3",
        "Precision@5",
        "Precision@10",
        "MRR@10",
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
                    str(row["model"]),
                    str(row["chunking"]),
                    f'{row["recall@1"]:.4f}',
                    f'{row["recall@3"]:.4f}',
                    f'{row["recall@5"]:.4f}',
                    f'{row["recall@10"]:.4f}',
                    f'{row["precision@1"]:.4f}',
                    f'{row["precision@3"]:.4f}',
                    f'{row["precision@5"]:.4f}',
                    f'{row["precision@10"]:.4f}',
                    f'{row["mrr@10"]:.4f}',
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def run(
    model_keys: Iterable[str] | None = None,
    chunking_keys: Iterable[str] | None = None,
    eval_path: Path = DEFAULT_EVAL_PATH,
    chunk_dir: Path = DEFAULT_CHUNK_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    restrict_to_eval_docs: bool = False,
) -> Dict[str, object]:
    load_dotenv()
    model_keys = list(model_keys or MODEL_SPECS.keys())
    chunking_keys = list(chunking_keys or CHUNK_FILES.keys())

    samples = load_eval_samples(eval_path)
    questions = [sample.question for sample in samples]
    eval_doc_ids = {sample.gt_doc_id for sample in samples}

    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    for model_key in model_keys:
        try:
            embedder = build_embedder(model_key)
            query_vectors = encode_with_cache(embedder, questions, cache_dir, model_key, "eval_queries")
        except Exception as exc:
            failures.append({"model": model_key, "chunking": "*", "error": str(exc)})
            continue

        for chunking_key in chunking_keys:
            chunk_path = chunk_dir / CHUNK_FILES[chunking_key]
            chunks = load_chunks(chunk_path)
            if restrict_to_eval_docs:
                chunks = [chunk for chunk in chunks if chunk["doc_id"] in eval_doc_ids]
            corpus_key = f"{chunking_key}__eval_docs" if restrict_to_eval_docs else chunking_key
            texts = [chunk["text"] for chunk in chunks]
            doc_ids = [chunk["doc_id"] for chunk in chunks]

            try:
                chunk_vectors = encode_with_cache(embedder, texts, cache_dir, model_key, corpus_key)
                metrics = evaluate_recall(query_vectors, chunk_vectors, doc_ids, samples)
                results.append(
                    {
                        "model": model_key,
                        "chunking": chunking_key,
                        **metrics,
                    }
                )
            except Exception as exc:
                failures.append({"model": model_key, "chunking": chunking_key, "error": str(exc)})

    results.sort(key=lambda row: (row["recall@10"], row["recall@5"], row["recall@3"], row["recall@1"]), reverse=True)
    payload = {
        "results": results,
        "failures": failures,
        "best": results[0] if results else None,
        "table": markdown_table(results) if results else "",
        "restrict_to_eval_docs": restrict_to_eval_docs,
    }

    (output_dir / "embedding_chunking_grid_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if results:
        (output_dir / "embedding_chunking_grid_results.md").write_text(payload["table"], encoding="utf-8")
    return payload


if __name__ == "__main__":
    payload = run()
    print(payload["table"])
    if payload["failures"]:
        print("\nFailures:")
        print(json.dumps(payload["failures"], ensure_ascii=False, indent=2))

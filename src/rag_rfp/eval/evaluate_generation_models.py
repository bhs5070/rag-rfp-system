from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from src.rag_rfp.retrieve.retriever import ChunkRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "eval.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "results"

GENERATION_MODELS = [
    "gpt-4.1",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o-mini",
]

JUDGE_MODEL = "gpt-5-mini"
TOP_K = 5

MODEL_PRICING_PER_1M = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass
class EvalSample:
    sample_id: str
    question: str
    reference_answer: str


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
                    question=item["question"].strip(),
                    reference_answer=item.get("answer", "").strip(),
                )
            )
    return rows


def build_context_text(contexts: Sequence[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[doc={ctx.get('doc_id')}, chunk={ctx.get('chunk_index')}]\n{ctx.get('text', '')}"
        for ctx in contexts
    )


def extract_usage_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)

    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

    return int(prompt_tokens or 0), int(completion_tokens or 0)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = MODEL_PRICING_PER_1M[model]
    input_cost = (prompt_tokens / 1_000_000.0) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000.0) * pricing["output"]
    return input_cost + output_cost


def ask_generation_model(
    client: OpenAI,
    model: str,
    question: str,
    contexts: Sequence[Dict[str, Any]],
) -> tuple[str, float, int, int]:
    system_prompt = (
        "너는 공공 RFP 문서를 기반으로 답변하는 어시스턴트다. "
        "반드시 제공된 컨텍스트 안에서만 근거를 찾아서 한국어로 답변해라. "
        "컨텍스트에 없는 내용을 추측하지 마라."
    )
    user_prompt = (
        f"사용자 질문:\n{question}\n\n"
        f"관련 문서 컨텍스트:\n{build_context_text(contexts)}\n\n"
        "위 컨텍스트만 근거로 질문에 답하라. "
        "핵심 정보를 짧은 bullet 형태로 정리하라."
    )

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    answer = response.choices[0].message.content or ""
    prompt_tokens, completion_tokens = extract_usage_tokens(response)
    return answer, elapsed_ms, prompt_tokens, completion_tokens


def judge_answer(
    client: OpenAI,
    question: str,
    answer: str,
    contexts: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    context_text = build_context_text(contexts)
    prompt = f"""
당신은 RAG 응답을 평가하는 심사자다.

질문, 모델 답변, 검색 컨텍스트를 보고 아래 두 지표를 0~1 사이 실수로 평가하라.

- answer_relevancy: 답변이 질문에 직접적으로 잘 답하고 있는가
- faithfulness: 답변이 검색 컨텍스트에 충실하며, 근거 없는 내용을 만들어내지 않았는가

[Question]
{question}

[Answer]
{answer}

[Retrieved Context]
{context_text}

반드시 아래 JSON 형식만 출력하라.

{{
  "answer_relevancy": 0.0,
  "faithfulness": 0.0,
  "summary": "짧은 한국어 한 줄 평가"
}}
""".strip()

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content or ""
    try:
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start : end + 1])
    except Exception:
        return {
            "answer_relevancy": 0.0,
            "faithfulness": 0.0,
            "summary": "judge parsing error",
        }


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = [
        "Model",
        "Answer Relevancy",
        "Faithfulness",
        "Avg Latency ms",
        "P95 Latency ms",
        "Avg Prompt Tokens",
        "Avg Completion Tokens",
        "Total Cost USD",
        "Avg Cost / Query USD",
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
                    f'{row["answer_relevancy"]:.4f}',
                    f'{row["faithfulness"]:.4f}',
                    f'{row["latency_avg_ms"]:.3f}',
                    f'{row["latency_p95_ms"]:.3f}',
                    f'{row["prompt_tokens_avg"]:.1f}',
                    f'{row["completion_tokens_avg"]:.1f}',
                    f'{row["cost_total_usd"]:.4f}',
                    f'{row["cost_avg_usd"]:.4f}',
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def save_payload(output_dir: Path, payload: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "generation_model_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "generation_model_comparison.md").write_text(payload["table"], encoding="utf-8")


def run(
    eval_path: Path = DEFAULT_EVAL_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    generation_models: Sequence[str] = GENERATION_MODELS,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    retriever = ChunkRetriever()
    samples = load_eval_samples(eval_path)
    query_embeddings = retriever.embed_queries([sample.question for sample in samples])
    context_cache = {
        sample.sample_id: retriever.search_by_embedding(query_embedding, top_k=top_k)
        for sample, query_embedding in zip(samples, query_embeddings)
    }

    rows: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for model in generation_models:
        try:
            print(f"[model] {model} start")
            relevancy_scores: List[float] = []
            faithfulness_scores: List[float] = []
            latencies: List[float] = []
            prompt_tokens_list: List[int] = []
            completion_tokens_list: List[int] = []
            total_cost = 0.0

            for index, sample in enumerate(samples, start=1):
                contexts = context_cache[sample.sample_id]
                answer, elapsed_ms, prompt_tokens, completion_tokens = ask_generation_model(
                    client=client,
                    model=model,
                    question=sample.question,
                    contexts=contexts,
                )
                judged = judge_answer(
                    client=client,
                    question=sample.question,
                    answer=answer,
                    contexts=contexts,
                )

                relevancy = float(judged.get("answer_relevancy", 0.0))
                faithfulness = float(judged.get("faithfulness", 0.0))
                cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)

                relevancy_scores.append(relevancy)
                faithfulness_scores.append(faithfulness)
                latencies.append(elapsed_ms)
                prompt_tokens_list.append(prompt_tokens)
                completion_tokens_list.append(completion_tokens)
                total_cost += cost

                details.append(
                    {
                        "model": model,
                        "sample_id": sample.sample_id,
                        "question": sample.question,
                        "answer": answer,
                        "judge": judged,
                        "latency_ms": elapsed_ms,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cost_usd": cost,
                    }
                )

                if index % 10 == 0 or index == len(samples):
                    print(f"[model] {model} progress {index}/{len(samples)}")

            row = {
                "model": model,
                "answer_relevancy": statistics.mean(relevancy_scores) if relevancy_scores else 0.0,
                "faithfulness": statistics.mean(faithfulness_scores) if faithfulness_scores else 0.0,
                "latency_avg_ms": statistics.mean(latencies) if latencies else 0.0,
                "latency_p95_ms": percentile(latencies, 0.95),
                "prompt_tokens_avg": statistics.mean(prompt_tokens_list) if prompt_tokens_list else 0.0,
                "completion_tokens_avg": statistics.mean(completion_tokens_list) if completion_tokens_list else 0.0,
                "cost_total_usd": total_cost,
                "cost_avg_usd": total_cost / len(samples) if samples else 0.0,
            }
            rows.append(row)
            print(f"[model] {model} done")
        except Exception as exc:
            failures.append({"model": model, "error": str(exc)})
            print(f"[model] {model} failed: {exc}")

        interim_rows = sorted(rows, key=lambda row: (row["faithfulness"], row["answer_relevancy"]), reverse=True)
        interim_payload = {
            "retrieval_setup": "semantic + text-embedding-3-large + ChromaDB",
            "judge_model": JUDGE_MODEL,
            "models": list(generation_models),
            "results": interim_rows,
            "failures": failures,
            "details": details,
            "table": markdown_table(interim_rows),
        }
        save_payload(output_dir, interim_payload)

    rows.sort(key=lambda row: (row["faithfulness"], row["answer_relevancy"]), reverse=True)
    payload = {
        "retrieval_setup": "semantic + text-embedding-3-large + ChromaDB",
        "judge_model": JUDGE_MODEL,
        "models": list(generation_models),
        "results": rows,
        "failures": failures,
        "details": details,
        "table": markdown_table(rows),
    }

    save_payload(output_dir, payload)
    return payload


if __name__ == "__main__":
    payload = run()
    print(payload["table"])

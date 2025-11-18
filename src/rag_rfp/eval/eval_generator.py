# src/rag_rfp/eval/eval_generator.py

import json
from pathlib import Path
from typing import List

from rouge_score import rouge_scorer

from rag_rfp.retrieve.retriever import ChunkRetriever
from rag_rfp.generate.generator import RAGGenerator


# 프로젝트 루트: /rag-rfp-system
# 이 파일 경로: /rag-rfp-system/src/rag_rfp/eval/eval_generator.py
# parents[0] = .../src/rag_rfp/eval
# parents[1] = .../src/rag_rfp
# parents[2] = .../src
# parents[3] = .../rag-rfp-system   ✅
BASE_DIR = Path(__file__).resolve().parents[3]
EVAL_PATH = BASE_DIR / "data" / "eval" / "rag_eval.jsonl"


def load_eval_dataset(path: Path) -> List[dict]:
    """rag_eval.jsonl 읽어서 리스트로 반환"""
    samples: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(obj)
    return samples


def simple_f1(pred: str, gold: str) -> float:
    """
    아주 단순한 whitespace 단위 F1 점수.
    RFP 답변 텍스트용으로 대충 경향만 보는 용도.
    """
    pred_tokens = pred.split()
    gold_tokens = gold.split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)

    overlap = pred_set & gold_set
    if not overlap:
        return 0.0

    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def run_generator_eval():
    """
    Generator 성능 평가:
    - ROUGE-L
    - Token F1
    ask.py 와 동일하게 generator.ask(question) 을 사용.
    """
    # 1) retriever / generator 초기화
    retriever = ChunkRetriever()
    generator = RAGGenerator(retriever=retriever)

    # 2) 평가 데이터셋 로드
    samples = load_eval_dataset(EVAL_PATH)
    print(f"Loaded {len(samples)} eval samples from {EVAL_PATH}")

    rouge_l_scores: List[float] = []
    f1_scores: List[float] = []

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # 3) 샘플별로 질문 → 답변 생성 → 점수 계산
    for i, sample in enumerate(samples, start=1):
        question = sample["question"]
        gold_answer = sample["answer"]

        # ask.py 와 동일한 방식으로 generator 호출
        answer_obj = generator.ask(question)
        pred_answer = answer_obj.answer

        # ROUGE-L
        rouge_scores = scorer.score(gold_answer, pred_answer)
        rouge_l = rouge_scores["rougeL"].fmeasure

        # F1
        f1 = simple_f1(pred_answer, gold_answer)

        rouge_l_scores.append(rouge_l)
        f1_scores.append(f1)

        if i % 5 == 0 or i == len(samples):
            print(
                f"[{i}/{len(samples)}] "
                f"ROUGE-L: {rouge_l:.4f}, F1: {f1:.4f}"
            )

    # 4) 평균 점수 출력
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    print("\nGenerator Evaluation Metrics:")
    print(f"Avg ROUGE-L: {avg_rouge_l:.4f}")
    print(f"Avg F1 Score: {avg_f1:.4f}")


if __name__ == "__main__":
    run_generator_eval()
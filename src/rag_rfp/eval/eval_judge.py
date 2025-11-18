# src/rag_rfp/eval/eval_judge.py

import json
import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from rag_rfp.retrieve.retriever import ChunkRetriever
from rag_rfp.generate.generator import RAGGenerator


# 프로젝트 루트: /rag-rfp-system
# 이 파일 경로: /rag-rfp-system/src/rag_rfp/eval/eval_judge.py
BASE_DIR = Path(__file__).resolve().parents[3]
EVAL_PATH = BASE_DIR / "data" / "eval" / "rag_eval.jsonl"

# Judge 에 사용할 모델 (권한 있는 걸로 설정)
JUDGE_MODEL = "gpt-5-mini"  # 필요하면 gpt-4.1, gpt-5 등으로 교체


# OpenAI 클라이언트 초기화 (.env: OPENAI_API_KEY)
load_dotenv(BASE_DIR / ".env")
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise RuntimeError("OPENAI_API_KEY not found in .env")
client = OpenAI(api_key=api_key)


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


def ask_judge(
    question: str, gold_answer: str, pred_answer: str
) -> Tuple[float, str]:
    """
    LLM Judge에게 1~12 점수+설명을 받아오는 함수.
    점수만 평균낼 때 쓰고, 설명은 필요시 로그 확인용.
    """
    system_prompt = (
        "You are an expert evaluator for Korean RFP question answering systems. "
        "You will be given a user question, a reference (ideal) answer, and a model's answer. "
        "Evaluate how good the model's answer is compared to the reference answer on a 1-12 scale, "
        "where 1 is completely wrong or useless, and 12 is essentially perfect. "
        "Focus on factual correctness and coverage of key points rather than style."
    )

    user_prompt = f"""
[질문]
{question}

[정답(Reference Answer)]
{gold_answer}

[모델 답변(Model Answer)]
{pred_answer}

위 정보를 바탕으로, 모델 답변의 품질을 1~12 사이의 정수 점수로 평가해 주세요.
점수 기준:
1-3: 거의 틀렸거나 전혀 도움이 되지 않음
4-6: 일부 관련 정보는 있지만 핵심이 많이 빠져 있음
7-9: 대부분 맞지만 몇 가지 중요한 디테일이 부족하거나 약간 부정확함
10-12: 사실상 완벽하거나 거의 완벽한 답변

반드시 아래 JSON 형식으로만 출력하세요:

{{
  "score": 10,
  "explanation": "왜 이 점수를 줬는지 한국어로 간단히 설명"
}}
""".strip()

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    text_out = resp.choices[0].message.content.strip()

    # JSON 파싱
    try:
        start = text_out.find("{")
        end = text_out.rfind("}")
        if start != -1 and end != -1:
            text_out = text_out[start : end + 1]

        obj = json.loads(text_out)
        score = float(obj.get("score", 0))
        explanation = obj.get("explanation", "").strip()
    except Exception:
        # 파싱 실패 시 fallback: 점수 0, 전체 텍스트를 설명으로
        score = 0.0
        explanation = text_out

    return score, explanation


def run_judge_eval():
    """
    LLM Judge를 이용한 End-to-End 평가.
    - generator.ask(question) 으로 답변 생성
    - Judge 모델이 1~12 점수 부여
    - 평균 Judge Score 출력
    """
    # 1) retriever / generator 초기화
    retriever = ChunkRetriever()
    generator = RAGGenerator(retriever=retriever)

    # 2) 평가 데이터셋 로드
    samples = load_eval_dataset(EVAL_PATH)
    print(f"Loaded {len(samples)} eval samples from {EVAL_PATH}")

    judge_scores: List[float] = []

    # 3) 샘플별로 질문 → 답변 생성 → Judge 점수 계산
    for i, sample in enumerate(samples, start=1):
        question = sample["question"]
        gold_answer = sample["answer"]

        # ask.py 와 동일하게 generator 호출
        answer_obj = generator.ask(question)
        pred_answer = answer_obj.answer

        score, explanation = ask_judge(question, gold_answer, pred_answer)
        judge_scores.append(score)

        print(
            f"[{i}/{len(samples)}] Judge Score: {score:.2f} / 12  "
            f"(Q: {question[:20]}..., Ref len: {len(gold_answer)}, Pred len: {len(pred_answer)})"
        )
        # 필요하면 explanation도 출력하고 싶을 때:
        # print("  →", explanation)

    # 4) 평균 점수 출력
    avg_score = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0

    print("\nE2E LLM Judge Evaluation:")
    print(f"Average Judge Score (1~12): {avg_score:.4f}")


if __name__ == "__main__":
    run_judge_eval()
# evaluate_generator.py

import json
from typing import List, Any
from openai import OpenAI

client = OpenAI()

JUDGE_MODEL = "gpt-5-mini"   # 필요하면 gpt-5-mini로 변경 가능

def judge_answer(question: str, answer: str, sources: List[str]) -> dict:
    """
    LLM-as-a-judge 평가
    - JSON 강제 출력
    - JSON 아닌 경우 자동 추출하여 파싱
    """

    src_text = "\n----\n".join(sources) if sources else "(no sources)"

    prompt = f"""
당신은 RFP 기반 QA 시스템을 평가하는 전문 심사위원입니다.

아래 요소에 대해 0~1 사이 점수를 부여하세요.

[Question]
{question}

[Answer]
{answer}

[Retrieved Source]
{src_text}

반드시 아래의 JSON 형식만 출력하세요.
설명 절대 금지. JSON 외 텍스트 절대 출력 금지.

출력 형식:
{{
  "faithfulness": 0.0~1.0,
  "groundedness": 0.0~1.0,
  "quality": 0.0~1.0,
  "summary": "한 줄 요약"
}}
"""

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    output = resp.choices[0].message.content.strip()

    # ---------------------------
    # 1차 시도: 바로 JSON 파싱
    # ---------------------------
    try:
        return json.loads(output)
    except:
        pass

    # ---------------------------
    # 2차 시도: 문자열에서 JSON만 추출
    # ---------------------------
    import re
    try:
        json_str = re.search(r"\{[\s\S]*\}", output).group(0)
        return json.loads(json_str)
    except:
        pass

    # ---------------------------
    # 완전 실패 → 안전한 기본값
    # ---------------------------
    return {
        "faithfulness": 0.0,
        "groundedness": 0.0,
        "quality": 0.0,
        "summary": "JSON parsing error"
    }



def evaluate_one(question=None, answer=None, retrieved_docs=None) -> dict:
    """
    /eval 경로에서 한 개의 Q-A에 대해
    - Retriever Hit / Precision@1
    - Generator LLM-judge 점수(faithfulness, groundedness, quality)
    를 계산하는 함수.
    """

    q = question or ""
    model_answer = answer or ""
    docs = retrieved_docs or []

    # 1) Retriever 평가 (간단 버전)
    retriever_hit = len(docs) > 0
    precision_at_1 = 1.0 if retriever_hit else 0.0

    # 2) Source 텍스트 추출 (LangChain Document or str 모두 지원)
    sources: List[str] = []
    for d in docs:
        if isinstance(d, str):
            sources.append(d)
        else:
            # LangChain Document 타입일 경우
            content = getattr(d, "page_content", None)
            if content is None:
                content = str(d)
            sources.append(content)

    # 3) LLM-as-a-judge로 Generator 평가
    judge = judge_answer(q, model_answer, sources)

    return {
        "retriever_hit": retriever_hit,
        "retriever_precision@1": precision_at_1,
        "faithfulness": judge.get("faithfulness", 0.0),
        "groundedness": judge.get("groundedness", 0.0),
        "quality": judge.get("quality", 0.0),
        "summary": judge.get("summary", ""),
    }


if __name__ == "__main__":
    # 간단 테스트용
    dummy_docs = ["이것은 테스트 소스 문장입니다."]
    out = evaluate_one(
        question="테스트 질문입니다.",
        answer="테스트 답변입니다.",
        retrieved_docs=dummy_docs,
    )
    print(out)

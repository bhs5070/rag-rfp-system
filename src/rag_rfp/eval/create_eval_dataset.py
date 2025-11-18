import json
import random
from pathlib import Path

from dotenv import load_dotenv
import os

from openai import OpenAI  # 새 SDK


# ===== 기본 설정 =====

# 이 파일: src/rag_rfp/eval/create_eval_dataset.py
# BASE_DIR: /rag-rfp-system
BASE_DIR = Path(__file__).resolve().parents[3]

CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks_512_64_final.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "eval" / "rag_eval.jsonl"

NUM_SAMPLES = 30  # 몇 개의 질문/정답을 만들지 (원하면 바꿔도 됨)
MODEL_NAME = "gpt-5-mini"  # 권한 있는 모델로 변경 가능


# ===== OpenAI 클라이언트 초기화 =====

load_dotenv(BASE_DIR / ".env")
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise RuntimeError("OPENAI_API_KEY not found. Check your .env at project root.")

client = OpenAI(api_key=api_key)


# ===== 유틸 함수들 =====

def load_chunks(path: Path):
    """
    chunks_512_64_final.jsonl 은 JSONL 형식:
      {"file": "...pdf", "chunk": "텍스트..."}
      {"file": "...pdf", "chunk": "텍스트..."}
      ...

    여기서:
      - id  : file 이름 + 라인 인덱스를 조합해서 사용
      - text: chunk 필드 사용
    """
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 혹시 중간에 깨진 줄이 있어도 전체가 죽지 않게 스킵
                # print("WARN: bad json line, skip")
                continue

            text = (
                obj.get("chunk")
                or obj.get("text")
                or obj.get("content")
            )
            if not text:
                continue

            file_name = obj.get("file", "")
            # file 명 + 라인 인덱스로 고유 id 생성
            chunk_id = file_name if file_name else str(idx)

            chunks.append({"id": chunk_id, "text": text})

    return chunks


def generate_qa_from_chunk(text: str):
    """
    한 개의 chunk text를 받아서 (question, answer)를 생성한다.
    chat.completions 기반 버전.
    """
    system_prompt = (
        "You are an assistant that creates evaluation questions for a Korean RFP document. "
        "You receive a single paragraph of the RFP and must create exactly one natural question "
        "a user might ask about this paragraph, and one ideal answer based ONLY on the paragraph."
    )

    user_prompt = f"""
다음은 한국어 RFP(제안요청서) 문서의 일부입니다.

[컨텍스트]
{text}

위 컨텍스트 내용만을 기반으로,
1) 사용자가 물어볼 법한 자연스러운 질문 1개와
2) 그에 대한 모범 답변 1개를 한국어로 만들어 주세요.

반드시 아래 JSON 형식으로만 출력하세요:

{{
  "question": "질문 내용",
  "answer": "답변 내용"
}}
""".strip()

    # 🔥 responses.create 대신 chat.completions.create 사용
    resp = client.chat.completions.create(
        model=MODEL_NAME,   # "gpt-4o-mini" 로 설정해 둔 값
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
        question = obj.get("question", "").strip()
        answer = obj.get("answer", "").strip()
    except Exception:
        # 파싱 실패 시 fallback
        question = "이 문단의 주요 내용을 요약하면 무엇인가요?"
        answer = text_out

    return question, answer


def main():
    # 1) chunks 로드
    print(f"Loading chunks from: {CHUNKS_PATH}")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Total chunks loaded: {len(chunks)}")

    if not chunks:
        print("No chunks loaded. Check CHUNKS_PATH and JSONL format.")
        return

    # 2) 샘플링
    num = min(NUM_SAMPLES, len(chunks))
    sampled = random.sample(chunks, num)
    print(f"Sampling {num} chunks for eval dataset generation.")

    # 3) 출력 폴더 생성
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 4) 각 chunk에서 Q/A 생성 후 rag_eval.jsonl에 기록
    with OUTPUT_PATH.open("w", encoding="utf-8") as f_out:
        for i, ch in enumerate(sampled, start=1):
            chunk_id = ch["id"]
            text = ch["text"]

            print(f"[{i}/{num}] Generating QA for chunk_id={chunk_id}...")

            question, answer = generate_qa_from_chunk(text)

            sample = {
                "id": f"q{i}",
                "question": question,
                "answer": answer,
                "relevant_chunk_ids": [chunk_id],
            }

            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\nDone! Saved eval dataset to: {OUTPUT_PATH}")
    print("이제 eval_retriever.py / eval_generator.py / eval_judge.py 를 돌릴 수 있습니다.")


if __name__ == "__main__":
    main()
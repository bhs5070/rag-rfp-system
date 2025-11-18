# src/rag_rfp/generate/generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# .env 로드
load_dotenv()


@dataclass
class RAGAnswer:
    """LLM 답변 + 사용된 컨텍스트를 함께 들고 있는 결과 구조체."""
    answer: str
    contexts: List[Dict[str, Any]]


class RAGGenerator:
    """
    Retriever → LLM 조합으로 RAG 파이프라인을 구현하는 Generator 클래스.
    """

    def __init__(self, retriever, model: str = "gpt-5-mini", top_k: int = 5):
        self.client = OpenAI()
        self.retriever = retriever
        self.model = model
        self.top_k = top_k

    def ask(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """
        1) Retriever로 관련 chunk 검색
        2) 검색된 chunk를 하나의 컨텍스트 텍스트로 합치고
        3) LLM에 프롬프트로 전달해 답변 생성
        """
        k = top_k or self.top_k

        # 1) 관련 chunk 검색
        contexts = self.retriever.search(question, top_k=k)

        # 2) 컨텍스트 텍스트 구성
        context_text = "\n\n".join(
            [
                f"[doc={c.get('doc_id')}, chunk={c.get('chunk_index')}]\n{c['text']}"
                for c in contexts
            ]
        )

        # 3) 프롬프트 구성 (공공 RFP 기준)
        system_prompt = (
            "너는 공공 RFP 문서를 기반으로 답변하는 어시스턴트야. "
            "반드시 제공된 컨텍스트 안에서만 근거를 찾아서 대답해. "
            "컨텍스트에 없는 내용을 추측하거나, 확인되지 않은 정보를 말하지 마."
        )

        user_content = (
            f"사용자 질문:\n{question}\n\n"
            f"관련 문서 컨텍스트:\n{context_text}\n\n"
            "위 컨텍스트를 근거로, 질문에 대한 답변을 한국어로 요약해서 알려줘. "
            "가능하면 bullet 형식으로 정리해줘."
        )

        # 4) OpenAI ChatCompletion 호출
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        answer = resp.choices[0].message.content or ""
        return RAGAnswer(answer=answer, contexts=contexts)
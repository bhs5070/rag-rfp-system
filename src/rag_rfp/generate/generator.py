# src/rag_rfp/generate/generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

# .env 로드 (OPENAI_API_KEY 사용)
load_dotenv()


@dataclass
class RAGAnswer:
    """LLM 답변 + 사용된 컨텍스트를 함께 들고 있는 결과 구조체."""
    answer: str
    contexts: List[Dict[str, Any]]


class RAGGenerator:
    """
    Retriever → LLM 조합으로 RAG 파이프라인을 구현하는 Generator 클래스.

    retriever 타입은 두 가지를 모두 지원한다.
    1) 기존 커스텀 retriever:
       - .search(question, top_k) → List[dict] 형태를 반환
         예) {"text": "...", "doc_id": "...", "chunk_index": ...}

    2) LangChain BaseRetriever 기반 객체 (예: CustomRFPRetriever):
       - _get_relevant_documents(query) 구현
       - 반환값: List[Document] (page_content, metadata 포함)
         metadata에는 "doc_id", "chunk_index" 등이 들어 있다고 가정
    """

    def __init__(self, retriever, model: str = "gpt-5-mini", top_k: int = 5):
        self.client = OpenAI()
        self.retriever = retriever
        self.model = model
        self.top_k = top_k

    # ------------------------------------------------------------------
    # 내부 유틸: retriever 타입에 따라 공통 포맷(List[dict])으로 변환
    # ------------------------------------------------------------------
    def _retrieve_contexts(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        retriever 타입에 상관없이, 최종적으로는 다음 형태의 리스트를 리턴하도록 통합한다.

        {
            "text": str,
            "doc_id": Any,
            "chunk_index": int | Any,
            "retrieval_mode": str | None,
            "raw_metadata": dict (있으면)
        }
        """

        # 1) 기존 커스텀 retriever: .search(question, top_k)
        if hasattr(self.retriever, "search"):
            return self.retriever.search(question, top_k=top_k)

        # 2) LangChain CustomRFPRetriever 등:
        #    _get_relevant_documents(query) 구현되어 있음
        if hasattr(self.retriever, "_get_relevant_documents"):
            docs = self.retriever._get_relevant_documents(question)
        else:
            raise TypeError(
                "지원하지 않는 retriever 타입입니다. "
                "search() 또는 _get_relevant_documents() 중 하나가 필요합니다."
            )

        contexts: List[Dict[str, Any]] = []
        # LangChain Document 리스트 → dict 리스트로 변환
        for d in docs[:top_k]:
            meta = getattr(d, "metadata", {}) or {}
            text = getattr(d, "page_content", "")

            contexts.append(
                {
                    "text": text,
                    "doc_id": meta.get("doc_id"),
                    "chunk_index": meta.get("chunk_index"),
                    "retrieval_mode": meta.get("retrieval_mode"),
                    "raw_metadata": meta,
                }
            )

        return contexts

    # ------------------------------------------------------------------
    # 외부에서 사용하는 메인 메서드
    # ------------------------------------------------------------------
    def ask(self, question: str, top_k: int | None = None) -> RAGAnswer:
        """
        1) Retriever로 관련 chunk/문서 검색
        2) 검색된 컨텍스트를 하나의 텍스트로 합치고
        3) LLM에 프롬프트로 전달해 답변 생성
        """
        k = top_k or self.top_k

        # 1) 관련 컨텍스트 검색 (retriever 타입에 따라 자동 분기)
        contexts = self._retrieve_contexts(question, top_k=k)

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
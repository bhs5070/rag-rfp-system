# src/rag_rfp/generate/generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


@dataclass
class RAGAnswer:
    """LLM 답변 + 사용된 컨텍스트를 포함하는 구조체."""
    answer: str
    contexts: List[Dict[str, Any]]


class RAGGenerator:
    """
    Retriever → Gemma-2-2B-IT 조합의 온프레미스 RAG Generator.
    """

    def __init__(
        self,
        retriever,
        model_id: str = "google/gemma-2-2b-it",  # ← 정확하게 정정
        top_k: int = 5,
        device: str = "cuda"
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.device = device

        # 🔹 Gemma 2-2B-IT 모델 / 토크나이저 로드
        #    - padding_side="left" 권장
        #    - trust_remote_code=True 필요 (Gemma의 HF 모델 구조 때문)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,        # L4 GPU에 최적
            device_map="auto",                # 자동 GPU 매핑
            trust_remote_code=True
        )

    def ask(self, question: str, top_k: int | None = None) -> RAGAnswer:
        k = top_k or self.top_k

        # 1) Retriever로 top-k chunk 검색
        contexts = self.retriever.search(question, top_k=k)

        # 2) 검색된 컨텍스트 텍스트 구성
        context_text = "\n\n".join(
            [
                f"[doc={c.get('doc_id')}, chunk={c.get('chunk_index')}]\n{c['text']}"
                for c in contexts
            ]
        )

        # 3) Gemma에서 잘 동작하는 단일 prompt 구성
        prompt = (
            "너는 공공 RFP 문서를 기반으로 답변하는 한국어 어시스턴트야.\n"
            "반드시 제공된 컨텍스트 안에 있는 내용만 근거로 답변해.\n"
            "컨텍스트에 없는 내용을 절대 추측하거나 지어내지 마.\n\n"
            f"사용자 질문:\n{question}\n\n"
            f"관련 문서 컨텍스트:\n{context_text}\n\n"
            "위 정보를 바탕으로 질문에 답변해줘.\n"
            "가능하면 bullet 형식으로 간단히 정리해줘."
        )

        # 4) Gemma 모델 실행
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,      # RAG는 낮게 유지 (환각 방지)
            do_sample=False,      # 재현성 & 정확도 우선 (권장)
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return RAGAnswer(answer=answer, contexts=contexts)

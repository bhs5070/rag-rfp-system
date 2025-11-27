# src/rag_rfp/generate/generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# 🔹 팀원이 넘겨준 retriever.py (하이브리드 검색 코드) 불러오기
#   예: 프로젝트 구조가 src/rag_rfp/retrieve/retriever.py 라고 가정
from rag_rfp.retrieve import retriever as hybrid_retriever


@dataclass
class RAGAnswer:
    """LLM 답변 + 사용된 컨텍스트를 포함하는 구조체."""
    answer: str
    contexts: List[Dict[str, Any]]


class RAGGenerator:
    """
    RAG Generator (Gemma 2-2B-IT 기반)

    두 가지 모드를 지원한다.

    1) 외부 retriever 주입 모드
       - __init__(retriever=외부_retriever, ...)
       - 외부 retriever는 .retrieve(question, top_k) 메서드를 가져야 한다.

    2) 하이브리드 내장 모드 (지금 팀원이 준 retriever.py 사용)
       - __init__(retriever=None, ...)
       - bge-m3 + FAISS + BM25 + RRF (hybrid_search) 사용
       - retriever.py 의 설정값(CHUNK_FILE_PATH, FAISS_INDEX_PATH 등)을 그대로 사용
    """

    def __init__(
        self,
        retriever: Optional[object] = None,
        model_id: str = "google/gemma-2-2b-it",  # Gemma 2-2B-IT
        top_k: int = 5,
        device: str = "cuda",
    ):
        self.external_retriever = retriever  # None이면 하이브리드 내장 모드
        self.top_k = top_k

        # GPU 없으면 자동으로 cpu로 fallback
        if device == "cuda" and not torch.cuda.is_available():
            print("[RAGGenerator] CUDA 미사용 환경 감지 → device='cpu'로 변경")
            device = "cpu"
        self.device = device

        # ==========================================
        # 1) Gemma 2-2B-IT 모델 / 토크나이저 로드
        # ==========================================
        print("[RAGGenerator] Loading Gemma-2-2B-IT...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device != "cuda":
            # device_map=None 인 경우 직접 to(device)
            self.model.to(self.device)

        # ==========================================
        # 2) retriever 모드 설정
        #    - 외부 retriever 가 없으면, 하이브리드 내장 모드 초기화
        # ==========================================
        if self.external_retriever is None:
            # 하이브리드 내장 모드
            print("[RAGGenerator] 외부 retriever 미지정 → 하이브리드 내장 모드 활성화")
            self._init_hybrid_retriever()
        else:
            print("[RAGGenerator] 외부 retriever 사용 모드 활성화")

    # ==========================================================
    # 하이브리드 retriever 초기화 (팀원 retriever.py 기반)
    # ==========================================================
    def _init_hybrid_retriever(self):
        """
        팀원이 넘겨준 retriever.py 에 정의된 설정과 함수를 그대로 사용해서
        bge-m3 + FAISS + BM25 + RRF 환경을 초기화한다.
        """
        print("[RAGGenerator] Loading bge-m3 model for hybrid retrieval...")
        self.hybrid_model = SentenceTransformer(
            "BAAI/bge-m3",
            trust_remote_code=True,
        )
        self.hybrid_model = self.hybrid_model.to(self.device)

        print("[RAGGenerator] Loading chunks and BM25 (retriever.load_chunk_mapping)...")
        # 이 함수 내부에서 BM25_MODEL 전역변수도 초기화됨
        self.chunk_mapping, self.chunks = hybrid_retriever.load_chunk_mapping(
            hybrid_retriever.CHUNK_FILE_PATH
        )

        print("[RAGGenerator] Loading FAISS index (retriever.load_faiss_index)...")
        self.faiss_index = hybrid_retriever.load_faiss_index(
            hybrid_retriever.FAISS_INDEX_PATH
        )

        print("[RAGGenerator] Hybrid retriever initialization complete.")

    # ==========================================================
    # 하이브리드 검색 → 컨텍스트 리스트 생성
    # ==========================================================
    def _hybrid_retrieve(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        팀원이 준 hybrid_search(model, index, query)를 호출해서
        상위 chunk 인덱스를 얻고, 이를 컨텍스트 dict 리스트로 변환.
        """
        # hybrid_retriever.HYBRID_TOP_K 만큼 뽑고 그 중 상위 top_k만 사용
        indices = hybrid_retriever.hybrid_search(
            self.hybrid_model,
            self.faiss_index,
            question,
        )[:top_k]

        contexts: List[Dict[str, Any]] = []
        for idx in indices:
            chunk = self.chunks[idx]  # load_chunk_mapping 에서 읽어온 원본 chunk dict

            # chunk 의 구조를 가정:
            # {
            #   "doc_id": ...,
            #   "text": ...,
            #   "chunk_index": ... (있을 수도 있고 없을 수도 있음)
            #   ...
            # }
            doc_id = chunk.get("doc_id")
            chunk_index = chunk.get("chunk_index", idx)

            ctx = {
                "text": chunk.get("text", ""),
                "doc_id": doc_id,
                "chunk_id": idx,
                "meta": {
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                },
            }
            contexts.append(ctx)

        return contexts

    # ==========================================================
    # 메인 진입점: 질문 → RAGAnswer
    # ==========================================================
    def ask(self, question: str, top_k: Optional[int] = None) -> RAGAnswer:
        """질문을 받아 RAGAnswer 반환."""
        k = top_k or self.top_k

        # 1) 컨텍스트 검색
        if self.external_retriever is not None:
            # 외부 retriever 사용
            contexts = self.external_retriever.retrieve(question, top_k=k)
        else:
            # 하이브리드 내장 모드 사용
            contexts = self._hybrid_retrieve(question, top_k=k)

        # 2) 검색된 컨텍스트 텍스트 구성
        def _format_header(c: Dict[str, Any]) -> str:
            """
            doc_id / chunk_index / chunk_id 를 상황에 따라 유연하게 표시.
            """
            meta = c.get("meta", {}) or {}
            doc_id = meta.get("doc_id", c.get("doc_id", "unknown_doc"))
            chunk_index = meta.get(
                "chunk_index",
                c.get("chunk_index", c.get("chunk_id", "unknown_chunk")),
            )
            return f"[doc={doc_id}, chunk={chunk_index}]"

        context_text = "\n\n".join(
            f"{_format_header(c)}\n{c.get('text', '')}"
            for c in contexts
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
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,      # RAG는 낮게 유지 (환각 방지)
            do_sample=False,      # 재현성 & 정확도 우선
        )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 프롬프트까지 포함된 전체 텍스트에서, 실제 답변 부분만 잘라내기
        if full_text.startswith(prompt):
            answer = full_text[len(prompt):].strip()
        else:
            # 혹시 tokenizer 처리로 prompt가 약간 달라졌다면 전체 텍스트 반환
            answer = full_text.strip()

        return RAGAnswer(answer=answer, contexts=contexts)

    def __call__(self, question: str, top_k: Optional[int] = None) -> RAGAnswer:
        """
        generator(question) 형태로도 쓸 수 있게 호출 연산자 오버라이드.
        (기존 코드 호환용)
        """
        return self.ask(question, top_k=top_k)


# 🔁 기존 코드와의 호환성을 위한 별칭
Generator = RAGGenerator
# retriever_core.py

import faiss
import numpy as np
import time
from typing import List, Dict
from collections import defaultdict
from rank_bm25 import BM25Okapi
import re
from openai import OpenAI
import torch
import os
import pickle


class RFPRetrieverCore:
    """
    RFP Retrieval 엔진 통합 버전
    """

    def __init__(
        self,
        faiss_index,
        chunk_texts: List[str],
        chunk_mapping: Dict[int, str],
        openai_api_key: str,
        reranker_model,
        reranker_tokenizer,
        device=None,
        rrf_k: int = 60,
        hybrid_k: int = 20,
        max_k_dense: int = 20,
        max_k_sparse: int = 20,
        transform_prefix: str = "RFP의 필수 정보를 찾고 있습니다: "
    ):

        self.index = faiss_index
        self.chunk_texts = chunk_texts
        self.chunk_mapping = chunk_mapping

        self.client = OpenAI(api_key=openai_api_key)

        self.reranker_model = reranker_model
        self.reranker_tokenizer = reranker_tokenizer

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reranker_model.to(self.device)

        self.rrf_k = rrf_k
        self.hybrid_k = hybrid_k
        self.max_k_dense = max_k_dense
        self.max_k_sparse = max_k_sparse
        self.transform_prefix = transform_prefix

        self.query_cache: Dict[str, List[str]] = {}
        self.last_rerank_score: float = 0.0

        print("Initializing BM25 corpus...")
        self.bm25 = self.load_or_build_bm25()
        print("BM25 initialized successfully.")

    # -----------------------------------------------------
    # Tokenizer + BM25 Cache
    # -----------------------------------------------------
    def fast_tokenize(self, text: str):
        return re.findall(r"[가-힣A-Za-z0-9]+", text)

    def load_or_build_bm25(self):
        bm25_path = "/home/bhs1581/rag-rfp-system/chunking/vector_db/bm25.pkl"

        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                return pickle.load(f)

        print("⚠️ BM25 building for the first time...")
        tokenized = [self.fast_tokenize(t) for t in self.chunk_texts]
        bm25 = BM25Okapi(tokenized)

        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)

        return bm25

    # -----------------------------------------------------
    # Query Rewriting
    # -----------------------------------------------------
    def transform_query(self, query: str) -> List[str]:

        if query in self.query_cache:
            return self.query_cache[query]

        print(f"[LLM Query Rewriting] {query}")

        prompt = f"""
        당신은 RFP 문서를 검색하기 위한 전문 검색어 생성기입니다.
        아래 사용자 입력을 기반으로 RFP 문서 검색에 적합한 3개의 검색 쿼리를 생성하세요.
        각 줄에 하나씩 출력하세요.

        사용자 쿼리: "{query}"
        """

        try:
            time.sleep(0.2)
            resp = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            lines = resp.choices[0].message.content.split("\n")

            cleaned = []
            for q in lines:
                q = (
                    q.encode("utf-8", "ignore")
                      .decode("utf-8", "ignore")
                      .replace("\ufffd", "")
                      .strip()
                )
                if q:
                    cleaned.append(q)

            if not cleaned:
                cleaned = [query]

            self.query_cache[query] = cleaned
            return cleaned

        except Exception:
            return [query]

    # -----------------------------------------------------
    # Embedding
    # -----------------------------------------------------
    def embed(self, text: str) -> np.ndarray:

        safe = (
            text.encode("utf-8", "ignore")
                .decode("utf-8", "ignore")
                .replace("\ufffd", "")
        )

        resp = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[safe]
        )
        vec = np.array(resp.data[0].embedding, dtype="float32")
        return vec / (np.linalg.norm(vec) + 1e-10)

    # -----------------------------------------------------
    # Dense (FAISS)
    # -----------------------------------------------------
    def dense_search(self, vec: np.ndarray) -> Dict[int, float]:
        faiss.normalize_L2(vec.reshape(1, -1))
        _, I = self.index.search(vec.reshape(1, -1), self.max_k_dense)

        out = {}
        for rank, idx in enumerate(I[0]):
            if idx != -1:
                out[idx] = rank + 1
        return out

    # -----------------------------------------------------
    # Sparse (BM25)
    # -----------------------------------------------------
    def sparse_search(self, query: str) -> Dict[int, float]:
        tokens = self.fast_tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1]

        out = {}
        for rank, idx in enumerate(ranked[:self.max_k_sparse]):
            out[idx] = rank + 1
        return out

    # -----------------------------------------------------
    # RRF
    # -----------------------------------------------------
    def rrf_fusion(self, rank_list: List[Dict[int, float]]) -> List[int]:
        fused = defaultdict(float)

        for ranks in rank_list:
            for idx, rank in ranks.items():
                fused[idx] += 1 / (self.rrf_k + rank)

        return sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[: self.hybrid_k]

    # -----------------------------------------------------
    # Reranker
    # -----------------------------------------------------
    def rerank(self, query: str, indices: List[int], top_k=10):

        if not indices:
            self.last_rerank_score = 0.0
            return []

        clean_query = (
            query.encode("utf-8", "ignore")
                 .decode("utf-8", "ignore")
                 .replace("\ufffd", "")
        )

        pair_strings = []
        for idx in indices:
            chunk = (
                self.chunk_texts[idx]
                    .encode("utf-8", "ignore")
                    .decode("utf-8", "ignore")
                    .replace("\ufffd", "")
            )
            pair_strings.append(f"{clean_query} [SEP] {chunk}")

        inputs = self.reranker_tokenizer(
            pair_strings,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.reranker_model(**inputs).logits
            scores = logits.squeeze(1).cpu().numpy()

        ranked = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)

        self.last_rerank_score = float(max(scores))

        return [idx for idx, _ in ranked][:top_k]

    # -----------------------------------------------------
    # Hybrid Search
    # -----------------------------------------------------
    def multi_step_hybrid_search(self, query: str) -> List[int]:

        queries = self.transform_query(query)
        all_ranks = []

        for q in queries:
            vec = self.embed(f"{self.transform_prefix}{q}")
            dense = self.dense_search(vec)
            sparse = self.sparse_search(q)

            all_ranks.append(dense)
            all_ranks.append(sparse)

        return self.rrf_fusion(all_ranks)

    # -----------------------------------------------------
    # Final Retrieve
    # -----------------------------------------------------
    def retrieve(self, query: str, top_k=10):
        fused = self.multi_step_hybrid_search(query)
        return self.rerank(query, fused, top_k=top_k)

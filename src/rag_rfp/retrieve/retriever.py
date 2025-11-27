from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


# ==== 경로 설정 ====
BASE_DIR = Path(__file__).resolve().parents[3]  # rag-rfp-system/
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "index"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace 캐시 경로를 HOME 아래에 강제로 지정 (GCP 권한 문제 회피)
os.environ["TRANSFORMERS_CACHE"] = str(BASE_DIR / ".hf_qwen3_cache")
os.environ["HF_HOME"] = str(BASE_DIR / ".hf_qwen3_cache")

# 청킹 결과 / 인덱스 / 메타데이터 경로
CHUNKS_PATH = DATA_DIR / "vectordb_multi_aspect_chunks.jsonl"
FAISS_INDEX_PATH = OUTPUT_DIR / "multi_aspect.faiss"
META_PATH = OUTPUT_DIR / "multi_aspect_meta.parquet"

class ChunkRetriever:
    """
    vectordb_multi_aspect_chunks.jsonl -> Qwen3-Embedding-0.6B 임베딩 -> FAISS Index 생성/로드
    """

    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = META_PATH,
        batch_size: int = 16,
    ) -> None:
        # 🔹 Qwen3-Embedding-0.6B 로드
        #  - cache_folder 를 HOME 아래로 지정해서 /mnt/hf_cache 권한 문제 피하기
        self.embedder = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B", 
            device="cpu",
            cache_folder=str(BASE_DIR / ".hf_qwen3_cache"),
        )

        self.chunks_path = chunks_path
        self.index_path = index_path
        self.meta_path = meta_path
        self.batch_size = batch_size

        self.index: faiss.IndexFlatIP | None = None
        self.meta_df: pd.DataFrame | None = None
        self.dim: int | None = None

        self._load_or_build()

    # ================== 공통 유틸 ================== #

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        텍스트를 안전하게 정제 (깨진 유니코드 방지용)
        """
        if not isinstance(text, str):
            text = str(text)
        return text.encode("utf-8", errors="replace").decode("utf-8")

    # ================== 내부 로직 ================== #

    def _load_or_build(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            print("[ChunkRetriever] Found existing index. Loading...")
            self._load_index_and_meta()
        else:
            print("[ChunkRetriever] Index/meta not found. Building from JSONL...")
            self._build_from_jsonl()
            self._save_index_and_meta()

    def _build_from_jsonl(self) -> None:
        """
        현재 chunks_512_64_final.jsonl 구조:
        {"file": "<원본 파일명>", "chunk": "<잘린 텍스트>"}
        또는 {"doc_id": ..., "text": ...} 형태도 지원
        """

        rows: List[Dict[str, Any]] = []
        texts: List[str] = []

        with self.chunks_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)

                # 우리 파일 기준 키 매핑
                doc_id = obj["doc_id"]
                raw_text = obj["text"]
                chunk_id = obj.get("chunk_id")          # 새 필드
                aspect = obj.get("aspect")
                chunk_type = obj.get("chunk_type")
                filename = obj.get("filename")


                if not raw_text:
                    continue

                text = self._clean_text(raw_text)
                chunk_index = len(rows)

                rows.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "filename": filename,
                        "chunk_index": chunk_index,
                        "aspect": aspect,
                        "chunk_type": chunk_type,
                        "size": obj.get("size"),
                        "text": text,
                        "n_chars": len(text),
                    }
                )
                texts.append(text)

        if not rows:
            raise ValueError(f"no valid chunks found in: {self.chunks_path}")

        print(f"[ChunkRetriever] Loaded {len(rows)} chunks. Embedding with Qwen3-Embedding-0.6B...")

        # 🔹 전체 청크를 임베딩
        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = self._embed_passages(batch)
            all_vecs.append(vecs)

        vec_matrix = np.vstack(all_vecs).astype("float32")
        self.dim = vec_matrix.shape[1]

        # 🔹 코사인 유사도용 내적 인덱스 (임베딩은 이미 L2 normalize 됨)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vec_matrix)

        self.meta_df = pd.DataFrame(rows)

    # -------- 임베딩 헬퍼들 -------- #

    def _embed_passages(self, texts: List[str]) -> np.ndarray:
        """
        RFP 문서 청크(문서/패시지) 임베딩.
        Qwen3-Embedding-0.6B: passage 쪽은 기본 encode 사용.
        """
        safe_texts = [self._clean_text(t) for t in texts]
        vecs = self.embedder.encode(
            safe_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # FAISS IP + L2 normalize = cosine
        )
        return vecs.astype("float32")

    def _embed_query(self, query: str) -> np.ndarray:
        """
        검색 쿼리 임베딩.
        Qwen3-Embedding-0.6B는 query에 prompt_name=\"query\" 권장.
        """
        safe_q = self._clean_text(query)
        vec = self.embedder.encode(
            [safe_q],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt_name="query",
        )
        return vec.astype("float32")

    # -------- 인덱스 저장/로드 -------- #

    def _save_index_and_meta(self) -> None:
        assert self.index is not None and self.meta_df is not None

        faiss.write_index(self.index, str(self.index_path))
        self.meta_df.to_parquet(self.meta_path, index=False)

        print(f"[ChunkRetriever] Saved index -> {self.index_path}")
        print(f"[ChunkRetriever] Saved meta  -> {self.meta_path}")

    def _load_index_and_meta(self) -> None:
        self.index = faiss.read_index(str(self.index_path))
        self.meta_df = pd.read_parquet(self.meta_path)
        self.dim = self.index.d

        print(
            f"[ChunkRetriever] Loaded index ({self.index.ntotal} vectors, dim={self.dim}), "
            f"meta_rows={len(self.meta_df)}"
        )

    # ================== Public API ================== #

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.meta_df is None:
            raise RuntimeError("Index/meta is not loaded")

        # 🔹 쿼리 임베딩 (query prompt 사용)
        q_vec = self._embed_query(query)  # shape: (1, dim)
        scores, idx = self.index.search(q_vec, top_k)

        scores = scores[0]
        idx = idx[0]

        results: List[Dict[str, Any]] = []
        for i, score in zip(idx, scores):
            if i < 0:
                continue
            row = self.meta_df.iloc[int(i)]
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "filename": row["filename"],
                    "chunk_index": int(row["chunk_index"]),
                    "aspect": row["aspect"],
                    "chunk_type": row["chunk_type"],
                    "text": row["text"],
                    "n_chars": int(row["n_chars"]),
                    "score": float(score),
                }
            )
        return results

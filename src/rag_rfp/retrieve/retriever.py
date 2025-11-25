# src/rag_rfp/retrieve/retriever.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ==== 경로 설정 ====
BASE_DIR = Path(__file__).resolve().parents[3]  # rag-rfp-system/
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs" / "index"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 청킹 결과 / 인덱스 / 메타데이터 경로
CHUNKS_PATH = DATA_DIR / "chunks_512_64_final.jsonl"
FAISS_INDEX_PATH = OUTPUT_DIR / "chunks_512_64.faiss"
META_PATH = OUTPUT_DIR / "chunks_512_64_meta.parquet"

# 사용할 임베딩 모델은 SentenceTransformer("BAAI/bge-m3")로 로드함

class ChunkRetriever:
    """
    chunks_512_64_final.jsonl// v2 // v6 -> bge m3 임베딩 -> FAISS Index 생성/로드
    """

    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = META_PATH,
        batch_size: int = 128,
    ) -> None:

        self.embedder = SentenceTransformer("BAAI/bge-m3")
        
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
        OpenAI API로 보내기 전에 텍스트를 안전하게 정제
        (깨진 유니코드 문자 때문에 나는 UnicodeEncodeError 방지)
        """
        if not isinstance(text, str):
            text = str(text)
        # 문제가 되는 코드 포인트는 �(replacement char)로 치환
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
                doc_id = obj.get("doc_id") or obj.get("file")
                raw_text = obj.get("text") or obj.get("chunk")

                if not raw_text:
                    continue

                # 🔹 여기서 한 번 정제
                text = self._clean_text(raw_text)

                chunk_index = len(rows)

                rows.append(
                    {
                        "doc_id": doc_id,
                        "chunk_index": chunk_index,
                        "text": text,
                        "n_chars": len(text),
                    }
                )
                texts.append(text)

        if not rows:
            raise ValueError(f"no valid chunks found in: {self.chunks_path}")

        print(f"[ChunkRetriever] Loaded {len(rows)} chunks. Embedding...")

        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = self._embed_batch(batch)
            all_vecs.append(vecs)

        vec_matrix = np.vstack(all_vecs).astype("float32")
        self.dim = vec_matrix.shape[1]

        # 코사인 유사도 ~ 내적으로 사용
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vec_matrix)

        self.meta_df = pd.DataFrame(rows)

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        # 🔹 임베딩 들어가기 전에 한 번 더 방어적으로 정제
        safe_texts = [self._clean_text(t) for t in texts]

        vecs = self.embedder.encode(
            safe_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")
        return vecs
    
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

        # 질문도 정제해서 임베딩
        q_vec = self._embed_batch([query])
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
                    "doc_id": row["doc_id"],
                    "chunk_index": int(row["chunk_index"]),
                    "text": row["text"],
                    "n_chars": int(row["n_chars"]),
                    "score": float(score),
                }
            )
        return results
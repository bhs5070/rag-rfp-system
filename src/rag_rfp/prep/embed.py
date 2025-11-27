import os
from typing import List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---- 모델 경로 설정: 팀 공용 HF 캐시 충돌 방지용 ----
CACHE_DIR = os.path.expanduser("~/.cache/hf_qwen3")

# ---- Qwen3-Embedding-0.6B 로드 ----
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
_model = SentenceTransformer(
    EMBED_MODEL,
    cache_folder=CACHE_DIR,
)

def embed_texts(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    """
    Qwen3-Embedding-0.6B를 이용해 RFP 문서 청크를 임베딩.
    반환 형태: List[List[float]]
    """

    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]

        # Qwen3 passage/document embedding (normalize=True → cosine similarity)
        vecs = _model.encode(
            batch,
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=True,
        )

        embeddings.extend(vecs.tolist())

    return embeddings

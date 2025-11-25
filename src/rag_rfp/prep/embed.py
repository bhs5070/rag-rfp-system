import os
from typing import List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 모델은 함수 밖에서 미리 로드 → 속도 향상
EMBED_MODEL = "BAAI/bge-m3"
_model = SentenceTransformer(EMBED_MODEL)

def embed_texts(texts: List[str], batch_size: int = 128) -> List[List[float]]:
    """
    BGE-M3를 이용해 텍스트 리스트를 임베딩으로 변환.
    반환 형태: List[List[float]]
    """
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        
        vecs = _model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True  # 검색 정확도 향상
        )
        
        embeddings.extend(vecs.tolist())

    return embeddings

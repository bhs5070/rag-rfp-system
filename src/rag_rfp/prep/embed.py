import os
from typing import List
from tqdm import tqdm

def embed_texts(texts: List[str]) -> List[List[float]]:
    provider = os.getenv("EMB_PROVIDER", "openai")
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
        out = []
        for i in tqdm(range(0, len(texts), 128)):
            batch = texts[i:i + 128]
            resp = client.embeddings.create(model=model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out
    else:
        # SentenceTransformer 로컬 임베딩 fallback
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-base"))
        return m.encode(texts, convert_to_numpy=False, normalize_embeddings=True).tolist()

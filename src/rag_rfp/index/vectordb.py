from typing import List, Dict, Any
import faiss
import numpy as np
import os, json

class FaissIndex:
    def __init__(self, dim: int, path: str):
        self.dim = dim
        self.path = path
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[Dict[str, Any]] = []

    def add(self, embeddings: List[List[float]], metas: List[Dict[str, Any]]):
        X = np.array(embeddings).astype("float32")
        faiss.normalize_L2(X)
        self.index.add(X)
        self.meta.extend(metas)

    def search(self, q: List[float], k: int = 8):
        xq = np.array([q]).astype("float32")
        faiss.normalize_L2(xq)
        D, I = self.index.search(xq, k)
        hits = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            item = {**self.meta[idx], "score": float(dist)}
            hits.append(item)
        return hits

    def save(self):
        os.makedirs(self.path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.path, "index.faiss"))
        with open(os.path.join(self.path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        idx = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(dim=idx.d, path=path)
        obj.index = idx
        obj.meta = meta
        return obj

from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def rerank(self, query: str, hits: List[Dict], top_k: int = 8) -> List[Dict]:
        if not hits:
            return hits
        pairs = [(query, h["text"]) for h in hits]
        batch = self.tok(pairs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            scores = self.model(**batch).logits.squeeze(-1)
        for h, s in zip(hits, scores.tolist()):
            h["re_score"] = float(s)
        return sorted(hits, key=lambda x: x.get("re_score", x.get("score", 0.0)), reverse=True)[:top_k]

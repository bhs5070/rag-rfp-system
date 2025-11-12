from typing import List, Dict
from ..index.embed import embed_texts
from ..index.vectordb import FaissIndex

class Retriever:
    def __init__(self, index: FaissIndex):
        self.index = index

    def search(self, query: str, top_k: int = 8) -> List[Dict]:
        qv = embed_texts([query])[0]
        return self.index.search(qv, k=top_k)

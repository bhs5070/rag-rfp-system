from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from langchain_pipeline.retriever_core import RFPRetrieverCore


class CustomRFPRetriever(BaseRetriever):
    core: RFPRetrieverCore = Field(...)
    is_multistep: bool = True
    top_k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None
    ) -> List[Document]:

        # 1) retrieval
        if self.is_multistep:
            indices = self.core.retrieve(query, top_k=self.top_k)
        else:
            indices = self._single_step(query)

        docs = []
        for idx in indices:

            # 안전하게 int로 캐스팅
            idx = int(idx)

            text = self.core.chunk_texts[idx]
            doc_id = self.core.chunk_mapping.get(idx, "Unknown_Doc")

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "chunk_index": idx,
                        "doc_id": doc_id,
                        "retrieval_mode": "multi_step" if self.is_multistep else "single_step",
                    }
                )
            )

        return docs

    # Single-step hybrid search
    def _single_step(self, query: str):
        dense_query = f"{self.core.transform_prefix}{query}"
        vec = self.core.embed(dense_query)

        dense = self.core.dense_search(vec)
        sparse = self.core.sparse_search(query)

        fused = self.core.rrf_fusion([dense, sparse])
        reranked = self.core.rerank(query, fused, top_k=self.top_k)

        return reranked

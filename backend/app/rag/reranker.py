import logging
from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.rag.config import settings

logger = logging.getLogger(__name__)

class LocalCrossEncoderReranker:
    """
    Local Cross-Encoder reranker that scores relevance between query and retrieved candidates.
    No API dependencies, works 100% offline.
    """

    def __init__(self, model_name: str = settings.RERANKER_MODEL):
        self.model_name = model_name
        logger.info(f"Initializing Local Cross-Encoder Reranker ('{model_name}')...")
        try:
            self.model = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Document],
        top_k: int = settings.FINAL_RERANK_TOP_K
    ) -> List[Document]:
        """
        Scores (query, candidate_text) pairs and returns top_k highest-scoring Documents.
        """
        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.model.predict(pairs)

        scored_candidates = []
        for doc, score in zip(candidates, scores):
            meta = doc.metadata.copy()
            meta["rerank_score"] = float(score)
            scored_candidates.append(Document(page_content=doc.page_content, metadata=meta))

        # Sort descending by rerank score
        scored_candidates.sort(key=lambda d: d.metadata["rerank_score"], reverse=True)
        return scored_candidates[:top_k]

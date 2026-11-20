from sentence_transformers import CrossEncoder

from src.ingestion.store import COLLECTION_NAME
from src.retrieval.hybrid import hybrid_search

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None


def get_model():
    """Load the reranker model once, then reuse it."""
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(question, chunks, top_n=5):
    """Read each chunk together with the question, score it, and keep the best top_n."""
    if not chunks:
        return []
    pairs = [(question, chunk["text"]) for chunk in chunks]
    scores = get_model().predict(pairs)
    scored = [{**chunk, "rerank_score": float(score)} for chunk, score in zip(chunks, scores)]
    scored.sort(key=lambda chunk: chunk["rerank_score"], reverse=True)
    return scored[:top_n]


def retrieve(question, top_n=5, collection_name=COLLECTION_NAME, dense_weight=0.5, sparse_weight=0.5, candidates=20):
    """All of Phase 2: hybrid search for a shortlist, then rerank it down to the best top_n."""
    shortlist = hybrid_search(
        question,
        k=candidates,
        collection_name=collection_name,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        candidates=candidates,
    )
    return rerank(question, shortlist, top_n)u

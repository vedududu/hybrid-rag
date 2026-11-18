import re
from rank_bm25 import BM25Okapi

from src.ingestion.store import get_collection

_index = None
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by",
    "for", "with", "as", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "i", "you", "we", "they", "he", "she", "my",
    "your", "our", "what", "which", "who", "how", "why", "when", "where", "do",
    "does", "did", "can", "could", "should", "would", "will", "there", "here",
    "so", "than", "then", "about", "into", "me", "us", "them",
}


def tokenize(text):
    """Split text into lowercase words for keyword search, dropping filler words."""
    return [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]



def build_index():
    """Read every chunk from ChromaDB and build a BM25 keyword index over them."""
    data = get_collection().get(include=["documents", "metadatas"])
    tokenized = [tokenize(text) for text in data["documents"]]
    return {
        "bm25": BM25Okapi(tokenized),
        "ids": data["ids"],
        "texts": data["documents"],
        "metadatas": data["metadatas"],
    }


def get_index():
    """Build the index once, then reuse it."""
    global _index
    if _index is None:
        _index = build_index()
    return _index


def bm25_search(question, k=10):
    """Find the k chunks whose words best match the question."""
    index = get_index()
    scores = index["bm25"].get_scores(tokenize(question))
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    hits = []
    for i in best:
        hits.append({
            **index["metadatas"][i],
            "id": index["ids"][i],
            "text": index["texts"][i],
            "score": float(scores[i]),
        })
    return hits

from src.ingestion.bm25_index import bm25_search
from src.ingestion.store import COLLECTION_NAME, search

RRF_K = 60


def rrf_fuse(result_lists, weights, rrf_k=RRF_K):
    """Merge several ranked lists into one, rewarding chunks that rank well in more than one."""
    scores = {}
    chunks = {}
    for results, weight in zip(result_lists, weights):
        for rank, hit in enumerate(results, start=1):
            chunk_id = hit["id"]
            scores[chunk_id] = scores.get(chunk_id, 0) + weight / (rrf_k + rank)
            chunks[chunk_id] = {**chunks.get(chunk_id, {}), **hit}
    ranked = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [{**chunks[chunk_id], "rrf_score": scores[chunk_id]} for chunk_id in ranked]


def hybrid_search(question, k=20, collection_name=COLLECTION_NAME, dense_weight=0.5, sparse_weight=0.5, candidates=20):
    """Search by meaning and by words, then merge both lists with RRF."""
    dense = search(question, k=candidates, collection_name=collection_name)
    sparse = bm25_search(question, k=candidates, collection_name=collection_name)
    return rrf_fuse([dense, sparse], [dense_weight, sparse_weight])[:k]

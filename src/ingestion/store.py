import chromadb

from src.ingestion.embedder import embed_query, embed_texts

DB_PATH = "chroma_db"
COLLECTION_NAME = "fastapi_docs"


def get_collection():
    """Open (or create) the ChromaDB collection that holds our chunks."""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks, batch_size=64):
    """Embed chunks and save them in ChromaDB, a batch at a time."""
    collection = get_collection()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c["text"] for c in batch]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embed_texts(texts),
            metadatas=[
                {
                    "source": c["source"],
                    "path": c["path"],
                    "type": c["type"],
                    "chunk_index": c["chunk_index"],
                }
                for c in batch
            ],
        )
        print(f"{min(start + batch_size, len(chunks))}/{len(chunks)} stored")

def search(question, k=10):
    """Find the k chunks closest in meaning to the question."""
    result = get_collection().query(
        query_embeddings=[embed_query(question)],
        n_results=k,
    )
    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({**meta, "text": text, "distance": distance})
    return hits

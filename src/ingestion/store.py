import chromadb

from src.ingestion.embedder import embed_query, embed_texts

DB_PATH = "chroma_db"
COLLECTION_NAME = "fastapi_docs"


def get_collection(name=COLLECTION_NAME):  # NEW: name
    """Open (or create) a ChromaDB collection by name."""
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name=name,  # NEW
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks, collection_name=COLLECTION_NAME, batch_size=64):  # NEW: collection_name
    """Embed chunks and save them in a ChromaDB collection, a batch at a time."""
    collection = get_collection(collection_name)  # NEW
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [c["text"] for c in batch]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=texts,
            embeddings=embed_texts(texts),
            metadatas=[  # NEW: keep every label except id and text
                {key: value for key, value in c.items() if key not in ("id", "text")}
                for c in batch
            ],
        )
        print(f"{min(start + batch_size, len(chunks))}/{len(chunks)} stored")


def search(question, k=10, collection_name=COLLECTION_NAME):  # NEW: collection_name
    """Find the k chunks closest in meaning to the question."""
    result = get_collection(collection_name).query(  # NEW
        query_embeddings=[embed_query(question)],
        n_results=k,
    )
    hits = []
    for chunk_id, text, meta, distance in zip(  # NEW: chunk_id
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({**meta, "id": chunk_id, "text": text, "distance": distance})  # NEW: "id"
    return hits

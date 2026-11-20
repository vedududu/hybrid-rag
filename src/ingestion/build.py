import chromadb

from src.ingestion.chunker import chunk_by_headings, chunk_document, chunk_semantic
from src.ingestion.dedup import clean_collection
from src.ingestion.loader import load_all
from src.ingestion.store import DB_PATH, add_chunks

STRATEGIES = {
    "fastapi_docs": chunk_document,
    "fastapi_docs_headings": chunk_by_headings,
    "fastapi_docs_semantic": chunk_semantic,
}


def build_collection(collection_name, folder="data/raw"):
    """Delete a collection and rebuild it from scratch: load, chunk, store, clean."""
    chunker = STRATEGIES[collection_name]
    client = chromadb.PersistentClient(path=DB_PATH)
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    docs = load_all(folder)
    chunks = [c for d in docs for c in chunker(d)]
    print(f"{collection_name}: {len(chunks)} chunks from {len(docs)} documents")
    add_chunks(chunks, collection_name=collection_name)
    report = clean_collection(collection_name)
    print(f"{collection_name}: cleaned {report}")
    return report


if __name__ == "__main__":
    for name in STRATEGIES:
        build_collection(name)

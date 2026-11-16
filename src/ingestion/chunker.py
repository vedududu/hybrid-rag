from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document(doc, chunk_size=800, chunk_overlap=100):
    """Split a loaded document into overlapping chunks that keep their source info."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for i, text in enumerate(splitter.split_text(doc["text"])):
        chunks.append({
            "id": f"{doc['path']}::{i}",
            "text": text,
            "source": doc["source"],
            "path": doc["path"],
            "type": doc["type"],
            "chunk_index": i,
        })
    return chunks

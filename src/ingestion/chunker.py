import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


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


HEADINGS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def clean_heading(heading):
    """Remove FastAPI's link tags like '{ #path-parameters }' and HTML tags from a heading."""
    heading = re.sub(r"<[^>]+>", "", heading)
    return re.sub(r"\s*\{\s*#[^}]*\}", "", heading).strip()



def chunk_by_headings(doc, chunk_size=800, chunk_overlap=100):
    """Split a markdown document at its headings; cut long sections further."""
    if doc["type"] != ".md":
        return chunk_document(doc, chunk_size, chunk_overlap)

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADINGS, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = []
    for section in header_splitter.split_text(doc["text"]):
        heading = " > ".join(clean_heading(h) for h in section.metadata.values())
        for text in size_splitter.split_text(section.page_content):
            i = len(chunks)
            chunks.append({
                "id": f"{doc['path']}::headings::{i}",
                "text": text,
                "source": doc["source"],
                "path": doc["path"],
                "type": doc["type"],
                "chunk_index": i,
                "heading": heading,
            })
    return chunks


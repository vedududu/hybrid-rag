from src.ingestion.embedder import embed_texts
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

def split_sentences(text):
    """Split text into sentences, keeping each code block whole."""
    sentences = []
    for part in re.split(r"(```.*?```)", text, flags=re.DOTALL):
        if part.startswith("```"):
            sentences.append(part.strip())
            continue
        for line in part.split("\n"):
            for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
                if sentence:
                    sentences.append(sentence)
    return sentences


def sentence_similarities(sentences, window=1):
    """How similar each sentence's neighbourhood is to the next one's (higher = same topic)."""
    windows = [
        " ".join(sentences[max(0, i - window): i + window + 1])
        for i in range(len(sentences))
    ]
    vectors = embed_texts(windows)
    return [
        sum(a * b for a, b in zip(vectors[i], vectors[i + 1]))
        for i in range(len(vectors) - 1)
    ]


def find_cut_points(similarities, percentile=10):
    """Return the gap positions to cut at: the lowest `percentile`% of similarities."""
    if not similarities:
        return []
    threshold = sorted(similarities)[len(similarities) * percentile // 100]
    return [i for i, s in enumerate(similarities) if s < threshold]


def group_sentences(sentences, cut_points):
    """Join sentences into pieces, starting a new piece after each cut point."""
    pieces = []
    current = []
    for i, sentence in enumerate(sentences):
        current.append(sentence)
        if i in cut_points:
            pieces.append("\n".join(current))
            current = []
    if current:
        pieces.append("\n".join(current))
    return pieces


def fix_sizes(pieces, min_size=200, max_size=800, overlap=100):
    """Cut pieces that are too big, then glue pieces that are too small onto a neighbour."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_size, chunk_overlap=overlap)
    cut = []
    for piece in pieces:
        cut.extend(splitter.split_text(piece))

    glued = []
    for piece in cut:
        if glued and len(piece) < min_size:
            glued[-1] += "\n" + piece
        else:
            glued.append(piece)
    if len(glued) > 1 and len(glued[0]) < min_size:
        glued[1] = glued[0] + "\n" + glued[1]
        glued.pop(0)
    return glued


def chunk_semantic(doc, percentile=10, min_size=200, max_size=800, chunk_overlap=100):
    """Split a document where the topic changes, then fix chunk sizes."""
    sentences = split_sentences(doc["text"])
    if not sentences:
        return []
    cuts = find_cut_points(sentence_similarities(sentences), percentile)
    pieces = fix_sizes(group_sentences(sentences, cuts), min_size, max_size, chunk_overlap)
    chunks = []
    for i, text in enumerate(pieces):
        chunks.append({
            "id": f"{doc['path']}::semantic::{i}",
            "text": text,
            "source": doc["source"],
            "path": doc["path"],
            "type": doc["type"],
            "chunk_index": i,
        })
    return chunks

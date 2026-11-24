from src.ingestion.store import COLLECTION_NAME, get_collection


def next_chunk_id(chunk_id):
    """The id of the chunk right after this one in the same file, e.g. 'x.md::3' -> 'x.md::4'."""
    base, number = chunk_id.rsplit("::", 1)
    return f"{base}::{int(number) + 1}"


def add_next_chunks(chunks, collection_name=COLLECTION_NAME):
    """Give each retrieved chunk the text of the chunk that follows it, so answers cut at a boundary stay whole."""
    retrieved = {chunk["id"] for chunk in chunks}
    wanted = [next_chunk_id(chunk["id"]) for chunk in chunks]
    found = get_collection(collection_name).get(ids=[i for i in wanted if i not in retrieved])
    following = dict(zip(found["ids"], found["documents"]))
    expanded = []
    for chunk, next_id in zip(chunks, wanted):
        extra = following.get(next_id)
        expanded.append({**chunk, "text": chunk["text"] + "\n" + extra if extra else chunk["text"]})
    return expanded
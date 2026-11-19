import numpy as np

from src.ingestion.store import get_collection


def clean_collection(collection_name, threshold=0.98, min_chars=50):
    """Remove tiny chunks and near-duplicates from a collection, recording where duplicates also appeared."""
    collection = get_collection(collection_name)
    data = collection.get(include=["embeddings", "documents", "metadatas"])
    ids, texts, metadatas = data["ids"], data["documents"], data["metadatas"]
    vectors = np.array(data["embeddings"])

    remove = {i for i, text in enumerate(texts) if len(text.strip()) < min_chars}
    tiny = len(remove)

    similarity = vectors @ vectors.T
    also_in = {}
    for i in range(len(ids)):
        if i in remove:
            continue
        for j in np.where(similarity[i, i + 1:] >= threshold)[0] + i + 1:
            j = int(j)
            if j in remove:
                continue
            remove.add(j)
            if metadatas[j]["source"] != metadatas[i]["source"]:
                also_in.setdefault(i, set()).add(metadatas[j]["source"])

    if remove:
        collection.delete(ids=[ids[i] for i in remove])
    for i, sources in also_in.items():
        existing = metadatas[i].get("also_in", "")
        sources |= {s for s in existing.split(", ") if s}
        collection.update(ids=[ids[i]], metadatas=[{**metadatas[i], "also_in": ", ".join(sorted(sources))}])

    return {"tiny": tiny, "duplicates": len(remove) - tiny, "kept": len(ids) - len(remove)}

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def get_model():
    """Load the embedding model once, then reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts):
    """Turn a list of texts into a list of embeddings (384 numbers each)."""
    model = get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_query(question):
    """Turn one question into an embedding, adding the prefix BGE expects."""
    return embed_texts([QUERY_PREFIX + question])[0]

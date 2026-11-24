import re
import time

from src.generation.answer import ask_stream
MAX_ENTRIES = 500
MAX_AGE_SECONDS = 24 * 60 * 60

_cache = {}


def cache_key(question):
    """Make small differences (capitals, extra spaces, a final ? . or !) count as the same question."""
    text = re.sub(r"\s+", " ", question.strip().lower())
    return text.rstrip("?.! ")


def cached_ask_stream(question):
    """Like ask_stream, but a question answered recently is replied to straight from the cache."""
    key = cache_key(question)
    entry = _cache.get(key)
    if entry and time.time() - entry["saved_at"] < MAX_AGE_SECONDS:
        yield {"type": "done", "result": {**entry["result"], "question": question, "cached": True}}
        return
    for update in ask_stream(question):
        if update["type"] == "done":
            if len(_cache) >= MAX_ENTRIES:
                _cache.pop(next(iter(_cache)))
            _cache[key] = {"saved_at": time.time(), "result": update["result"]}
            update = {"type": "done", "result": {**update["result"], "cached": False}}
        yield update


def cached_ask(question):
    """Return a saved answer if this question was answered recently; otherwise answer it and save the result."""
    for update in cached_ask_stream(question):
        if update["type"] == "done":
            return update["result"]
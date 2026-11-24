import re
import time

from src.generation.answer import ask

MAX_ENTRIES = 500
MAX_AGE_SECONDS = 24 * 60 * 60

_cache = {}


def cache_key(question):
    """Make small differences (capitals, extra spaces, a final ? . or !) count as the same question."""
    text = re.sub(r"\s+", " ", question.strip().lower())
    return text.rstrip("?.! ")


def cached_ask(question):
    """Return a saved answer if this question was answered recently; otherwise run ask() and save the result."""
    key = cache_key(question)
    entry = _cache.get(key)
    if entry and time.time() - entry["saved_at"] < MAX_AGE_SECONDS:
        return {**entry["result"], "question": question, "cached": True}
    result = ask(question)
    if len(_cache) >= MAX_ENTRIES:
        _cache.pop(next(iter(_cache)))
    _cache[key] = {"saved_at": time.time(), "result": result}
    return {**result, "cached": False}
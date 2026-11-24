import math
import time

from src.generation.judge import find_citations, judge_answer
from src.generation.llm import ask_llm, stream_llm
from src.generation.prompt import NO_ANSWER, SYSTEM_PROMPT, build_prompt, clean_citations
from src.ingestion.store import COLLECTION_NAME
from src.retrieval.reranker import retrieve
from src.retrieval.neighbours import add_next_chunks


GATE = -7.0
HIGH = 0.8
MEDIUM = 0.5


def confidence_level(confidence):
    """Turn a confidence number into a label."""
    if confidence >= HIGH:
        return "high"
    if confidence >= MEDIUM:
        return "medium"
    return "low"


def is_no_answer(text):
    """True if the AI replied with the 'I don't know' message."""
    return text.strip().rstrip(".").lower() == NO_ANSWER.rstrip(".").lower()


def cited_sources(answer, chunks):
    """List the sources the answer actually cites, with their numbers."""
    return [
        {"number": n, "source": chunks[n - 1]["source"], "path": chunks[n - 1]["path"]}
        for n in find_citations(answer)
        if 1 <= n <= len(chunks)
    ]


def refuse(question, reason):
    """The reply for questions we can't answer from the docs."""
    return {"question": question, "answer": NO_ANSWER, "answered": False, "reason": reason,
            "confidence": 0.0, "level": "none", "sources": [], "checks": [], "bad_citations": []}


def timed(stats, step, work):
    """Run work() and record how many seconds it took as stats["seconds"][step]."""
    started = time.perf_counter()
    result = work()
    stats["seconds"][step] = round(time.perf_counter() - started, 2)
    return result


def add_stats(result, stats, usage):
    """Attach the timings and token counts to a result."""
    stats["tokens"] = dict(zip(["writer", "judge"], usage))
    stats["total_tokens"] = sum(call["in"] + call["out"] for call in usage)
    stats["total_seconds"] = round(sum(stats["seconds"].values()), 2)
    return {**result, "stats": stats}


def ask_stream(question, collection_name=COLLECTION_NAME):
    """Answer a question step by step, handing out progress updates; the last update holds the full result."""
    stats, usage = {"seconds": {}}, []
    yield {"type": "stage", "text": "Searching the docs…"}
    chunks = timed(stats, "search", lambda: retrieve(question, collection_name=collection_name))
    best = chunks[0]["rerank_score"] if chunks else -math.inf
    if best < GATE:
        yield {"type": "done", "result": add_stats(refuse(question, "gate: no chunk was relevant enough"), stats, usage)}
        return
    chunks = add_next_chunks(chunks, collection_name)
    yield {"type": "stage", "text": f"Found {len(chunks)} relevant passages in {stats['seconds']['search']} s. "
                                    "The AI is reading them and thinking…"}

    started = time.perf_counter()
    pieces = []
    for update in stream_llm(build_prompt(question, chunks), system=SYSTEM_PROMPT, usage_log=usage):
        if update["type"] == "text":
            pieces.append(update["text"])
        yield update
    stats["seconds"]["writer"] = round(time.perf_counter() - started, 2)
    answer = clean_citations("".join(pieces))
    if is_no_answer(answer):
        result = refuse(question, "writer: the sources don't contain the answer")
        yield {"type": "done", "result": add_stats(result, stats, usage)}
        return

    yield {"type": "stage", "text": "Fact-checking every claim against the sources…"}
    report = timed(stats, "judge", lambda: judge_answer(answer, chunks, usage_log=usage))
    # Confidence = share of claims the judge confirmed (Phase 4 showed the reranker score is a poor guide).
    confidence = report["support_score"]
    result = {"question": question, "answer": answer, "answered": True, "reason": "answered",
              "confidence": round(confidence, 2), "level": confidence_level(confidence),
              "sources": cited_sources(answer, chunks), "checks": report["checks"],
              "bad_citations": report["bad_citations"]}
    yield {"type": "done", "result": add_stats(result, stats, usage)}


def ask(question, collection_name=COLLECTION_NAME):
    """Answer a question in one go: run ask_stream and keep only its final result."""
    for update in ask_stream(question, collection_name):
        if update["type"] == "done":
            return update["result"]
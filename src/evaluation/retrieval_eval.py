import json
import time
from pathlib import Path

from src.generation.answer import GATE
from src.retrieval.reranker import retrieve

QUESTIONS_FILE = "data/eval/questions.json"
RESULTS_FILE = "data/eval/retrieval_results.json"
DOCS_FOLDER = Path("data/raw")
SHELVES = ["fastapi_docs", "fastapi_docs_headings", "fastapi_docs_semantic"]


def load_questions(path=QUESTIONS_FILE):
    """Read the test questions (utf-8-sig skips the invisible BOM marker)."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def doc_name(chunk):
    """Write a chunk's file the same way as expected_sources, e.g. 'tutorial/cors.md'."""
    return Path(chunk["path"]).relative_to(DOCS_FOLDER).as_posix()


def first_hit(chunks, expected_sources):
    """The position (1-5) of the first chunk from a right file, or None if there is none."""
    for position, chunk in enumerate(chunks, start=1):
        if doc_name(chunk) in expected_sources:
            return position
    return None


def evaluate_shelf(shelf, questions):
    """Run retrieval for every question on one shelf and record what came back."""
    rows = []
    for q in questions:
        chunks = retrieve(q["question"], collection_name=shelf)
        best = chunks[0]["rerank_score"] if chunks else float("-inf")
        rows.append({
            "id": q["id"],
            "answerable": q["answerable"],
            "hit_position": first_hit(chunks, q["expected_sources"]),
            "best_score": round(best, 2),
            "passes_gate": best >= GATE,
            "found": [doc_name(chunk) for chunk in chunks],
        })
    return rows


def summarize(rows):
    """Turn one shelf's rows into the scores: hit rate, MRR and the two Gate checks."""
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    hits = [r for r in answerable if r["hit_position"]]
    return {
        "hit_rate": len(hits) / len(answerable),
        "mrr": sum(1 / r["hit_position"] for r in hits) / len(answerable),
        "gate_passed_answerable": sum(r["passes_gate"] for r in answerable) / len(answerable),
        "gate_blocked_unanswerable": sum(not r["passes_gate"] for r in unanswerable) / len(unanswerable),
        "missed": [r["id"] for r in answerable if not r["hit_position"]],
    }


def main():
    """Test every shelf, print a scoreboard and save the details."""
    questions = load_questions()
    report = {}
    for shelf in SHELVES:
        start = time.time()
        rows = evaluate_shelf(shelf, questions)
        summary = summarize(rows)
        report[shelf] = {"summary": summary, "questions": rows}
        print(
            f"{shelf:22}  hit@5 {summary['hit_rate']:4.0%}  MRR {summary['mrr']:.2f}  "
            f"gate: passed {summary['gate_passed_answerable']:4.0%} answerable, "
            f"blocked {summary['gate_blocked_unanswerable']:4.0%} unanswerable  "
            f"({time.time() - start:.0f}s)"
        )
        print(f"{'':22}  missed: {', '.join(summary['missed']) or 'none'}")
    Path(RESULTS_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Details saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()

import json
import sys  # NEW
import time
from pathlib import Path

from src.generation.answer import GATE
from src.retrieval.reranker import retrieve

QUESTIONS_FILE = "data/eval/questions.json"
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


def squash(text):  # NEW
    """Ignore differences in spaces and line breaks when comparing text."""
    return " ".join(text.split())


def first_evidence_hit(chunks, question):  # NEW
    """The position (1-5) of the first chunk from a right file that CONTAINS an evidence quote, or None."""
    quotes = [squash(quote) for quote in question.get("evidence", [])]
    for position, chunk in enumerate(chunks, start=1):
        text = squash(chunk["text"])
        if doc_name(chunk) in question["expected_sources"] and any(quote in text for quote in quotes):
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
            "evidence_position": first_evidence_hit(chunks, q),  # NEW
            "best_score": round(best, 2),
            "passes_gate": best >= GATE,
            "found": [doc_name(chunk) for chunk in chunks],
        })
    return rows


def summarize(rows):
    """Turn one shelf's rows into the scores: file level, chunk level and the two Gate checks."""
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    hits = [r for r in answerable if r["hit_position"]]
    evidence_hits = [r for r in answerable if r["evidence_position"]]  # NEW
    return {
        "hit_rate": len(hits) / len(answerable),
        "mrr": sum(1 / r["hit_position"] for r in hits) / len(answerable),
        "chunk_hit_rate": len(evidence_hits) / len(answerable),  # NEW
        "chunk_mrr": sum(1 / r["evidence_position"] for r in evidence_hits) / len(answerable),  # NEW
        "gate_passed_answerable": sum(r["passes_gate"] for r in answerable) / len(answerable),
        "gate_blocked_unanswerable": sum(not r["passes_gate"] for r in unanswerable) / len(unanswerable),
        "missed": [r["id"] for r in answerable if not r["hit_position"]],
        "chunk_missed": [r["id"] for r in answerable if not r["evidence_position"]],  # NEW
    }


def main(questions_file=QUESTIONS_FILE):  # CHANGED
    """Test every shelf on one question file, print a scoreboard and save the details."""
    questions = load_questions(questions_file)  # CHANGED
    results_file = Path(questions_file).with_name(Path(questions_file).name.replace("questions", "retrieval_results"))  # NEW
    report = {}
    for shelf in SHELVES:
        start = time.time()
        rows = evaluate_shelf(shelf, questions)
        summary = summarize(rows)
        report[shelf] = {"summary": summary, "questions": rows}
        print(  # CHANGED
            f"{shelf:22}  file:  hit@5 {summary['hit_rate']:4.0%}  MRR {summary['mrr']:.2f}   "
            f"chunk: hit@5 {summary['chunk_hit_rate']:4.0%}  MRR {summary['chunk_mrr']:.2f}   "
            f"gate: passed {summary['gate_passed_answerable']:4.0%}, blocked {summary['gate_blocked_unanswerable']:4.0%}  "
            f"({time.time() - start:.0f}s)"
        )
        print(f"{'':22}  chunk-level misses: {', '.join(summary['chunk_missed']) or 'none'}")  # CHANGED
    results_file.write_text(json.dumps(report, indent=2), encoding="utf-8")  # CHANGED
    print(f"Details saved to {results_file}")  # CHANGED


if __name__ == "__main__":
    main(*sys.argv[1:])  # CHANGED
import json
import sys
import time
from pathlib import Path

from src.evaluation.retrieval_eval import QUESTIONS_FILE, first_evidence_hit, first_hit, load_questions
from src.ingestion.bm25_index import bm25_search
from src.ingestion.store import search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.neighbours import add_next_chunks
from src.retrieval.reranker import retrieve

TOP_K = 5

METHODS = {
    "meaning only": lambda q: search(q, k=TOP_K),
    "keywords only": lambda q: bm25_search(q, k=TOP_K),
    "hybrid (RRF)": lambda q: hybrid_search(q, k=TOP_K),
    "hybrid + reranker": lambda q: retrieve(q, top_n=TOP_K),
    "hybrid + reranker + next chunk": lambda q: add_next_chunks(retrieve(q, top_n=TOP_K)),
}


def score_method(find, questions):
    """Run one search method on every answerable question; score it by right file and by right chunk."""
    file_positions, chunk_positions = {}, {}
    for q in questions:
        chunks = find(q["question"])
        file_positions[q["id"]] = first_hit(chunks, q["expected_sources"])
        chunk_positions[q["id"]] = first_evidence_hit(chunks, q)
    n = len(questions)
    file_hits = [p for p in file_positions.values() if p]
    chunk_hits = [p for p in chunk_positions.values() if p]
    return {
        "file_hit_at_5": len(file_hits) / n,
        "file_mrr": sum(1 / p for p in file_hits) / n,
        "chunk_hit_at_1": sum(p == 1 for p in chunk_hits) / n,
        "chunk_hit_at_5": len(chunk_hits) / n,
        "chunk_mrr": sum(1 / p for p in chunk_hits) / n,
        "chunk_missed": [qid for qid, p in chunk_positions.items() if not p],
    }




def main(questions_file=QUESTIONS_FILE):
    """Compare the four search methods on the answerable questions of one file and save the table."""
    questions = [q for q in load_questions(questions_file) if q["answerable"]]
    results_file = Path(questions_file).with_name(Path(questions_file).name.replace("questions", "method_comparison"))
    report = {}
    print(f"{len(questions)} answerable questions from {questions_file}")
    print(f"{'method':32} {'file hit@5':>10} {'file MRR':>9} {'chunk hit@1':>12} {'chunk hit@5':>12} {'chunk MRR':>10}")
    for name, find in METHODS.items():
        start = time.time()
        r = report[name] = score_method(find, questions)
        print(f"{name:32} {r['file_hit_at_5']:10.0%} {r['file_mrr']:9.2f} {r['chunk_hit_at_1']:12.0%} "
              f"{r['chunk_hit_at_5']:12.0%} {r['chunk_mrr']:10.2f}  ({time.time() - start:.0f}s)")
    results_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved to {results_file}")


if __name__ == "__main__":
    main(*sys.argv[1:])
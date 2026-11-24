import json
import sys  # NEW
import time
from pathlib import Path

from openai import RateLimitError

from src.evaluation.grader import grade
from src.evaluation.retrieval_eval import QUESTIONS_FILE, load_questions  # CHANGED
from src.generation.answer import ask

RESULTS_FILE = "data/eval/answer_results.json"
WAIT_SECONDS = 60
MAX_TRIES = 5


def load_done(path=RESULTS_FILE):
    """Read the results saved by earlier runs, so finished questions are skipped."""
    if not Path(path).exists():
        return {}
    return {row["id"]: row for row in json.loads(Path(path).read_text(encoding="utf-8"))}


def save(rows, path=RESULTS_FILE):
    """Write all results so far to the results file."""
    Path(path).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def support(result):
    """The judge's support score (share of claims supported), or None if the answer wasn't judged."""
    if not result["answered"]:
        return None
    checks = result["checks"]
    return sum(1 for c in checks if c.get("supported") is True) / len(checks) if checks else 0.0


def run_one(item):
    """Ask and grade one question, waiting and retrying if we hit the free plan's speed limit."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            result = ask(item["question"])
            return result, grade(item, result)
        except RateLimitError:
            print(f"     speed limit reached - waiting {WAIT_SECONDS}s (try {attempt} of {MAX_TRIES})")
            time.sleep(WAIT_SECONDS)
    raise RuntimeError("still limited after several tries - probably the daily limit, run again later")


def average(numbers):
    """The average of a list, or 0 for an empty list."""
    return sum(numbers) / len(numbers) if numbers else 0.0


def print_scoreboard(rows, total):
    """Print the final scores, the confidence check and the list of failures."""
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    judged = [r["support"] for r in rows if r["support"] is not None]
    print(f"\n===== SCOREBOARD ({len(rows)} of {total} questions) =====")
    print(f"Correctness (all)         {average([r['score'] for r in rows]):.0%}")
    print(f"  answerable questions    {average([r['score'] for r in answerable]):.0%}  ({len(answerable)})")
    print(f"  unanswerable questions  {average([r['score'] for r in unanswerable]):.0%}  ({len(unanswerable)})")
    print(f"Faithfulness              {average(judged):.0%}  (judge's support score, {len(judged)} answers)")
    print("Confidence check (is HIGH really more often right?):")
    for level in ["high", "medium", "low"]:
        group = [r for r in rows if r["level"] == level]
        right = sum(1 for r in group if r["verdict"] == "correct")
        print(f"  {level:6}  {len(group):2} answers, {right:2} correct")
    failures = [r for r in rows if r["verdict"] != "correct"]
    print(f"Not correct ({len(failures)}):")
    for r in failures:
        print(f"  {r['id']}  {r['verdict']:10} {r['grader_reason'][:90]}")


def main(questions_file=QUESTIONS_FILE):  # CHANGED
    """Ask and grade every question in one file, saving after each one, then print the scoreboard."""
    questions = load_questions(questions_file)  # CHANGED
    results_file = Path(questions_file).with_name(Path(questions_file).name.replace("questions", "answer_results"))  # NEW
    done = load_done(results_file)  # CHANGED
    for item in questions:
        if item["id"] in done:
            continue
        start = time.time()
        try:
            result, verdict = run_one(item)
        except RuntimeError as error:
            print(f"Stopped at {item['id']}: {error}")
            break
        done[item["id"]] = {
            "id": item["id"],
            "question": item["question"],
            "answerable": item["answerable"],
            "answered": result["answered"],
            "reason": result["reason"],
            "answer": result["answer"],
            "level": result["level"],
            "confidence": result["confidence"],
            "support": support(result),
            "verdict": verdict["verdict"],
            "score": verdict["score"],
            "grader_reason": verdict["reason"],
        }
        save(list(done.values()), results_file)  # CHANGED
        print(f"{item['id']}  {verdict['verdict']:10} level {result['level']:6} ({time.time() - start:.0f}s)")
    print_scoreboard(list(done.values()), len(questions))


if __name__ == "__main__":
    main(*sys.argv[1:])  # CHANGED
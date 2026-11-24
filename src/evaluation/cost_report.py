import json
import sys
from pathlib import Path

PRICE_PER_MILLION_IN = 0.15   # US$ for gpt-oss-120b on Groq's paid tier – check groq.com/pricing
PRICE_PER_MILLION_OUT = 0.60  # (the free tier costs nothing, but has daily limits)


def average(numbers):
    """The average of a list, or 0 for an empty list."""
    return sum(numbers) / len(numbers) if numbers else 0.0


def main(results_file="data/eval/answer_results.json"):
    """Average the time and token counts saved by run_eval, overall and per type of outcome."""
    rows = [r for r in json.loads(Path(results_file).read_text(encoding="utf-8")) if r.get("stats")]
    print(f"{len(rows)} questions with stats in {results_file}\n")
    groups = {
        "answered": [r for r in rows if r["answered"]],
        "refused by the Gate": [r for r in rows if r["reason"].startswith("gate")],
        "refused by the writer": [r for r in rows if r["reason"].startswith("writer")],
    }
    for name, group in groups.items():
        if not group:
            continue
        stats = [r["stats"] for r in group]
        tokens_in = average([sum(c["in"] for c in s["tokens"].values()) for s in stats])
        tokens_out = average([sum(c["out"] for c in s["tokens"].values()) for s in stats])
        cost = (tokens_in * PRICE_PER_MILLION_IN + tokens_out * PRICE_PER_MILLION_OUT) / 1_000_000
        print(f"{name} ({len(group)} questions)")
        for step in ["search", "writer", "judge"]:
            times = [s["seconds"][step] for s in stats if step in s["seconds"]]
            if times:
                print(f"   {step:7} {average(times):5.2f} s")
        print(f"   total   {average([s['total_seconds'] for s in stats]):5.2f} s   "
              f"tokens: {tokens_in:,.0f} in + {tokens_out:,.0f} out   ≈ ${cost:.5f} per question")


if __name__ == "__main__":
    main(*sys.argv[1:])
import json
import re

from rich import prompt

from src.generation.llm import ask_llm
from src.generation.prompt import format_sources

JUDGE_PROMPT = """You are a strict fact-checker. You check an answer against numbered sources.

Split the answer into its factual claims. Skip code blocks and filler sentences.
For each claim, decide whether the sources it cites fully support it.
Judge ONLY by the sources, never by your own knowledge.
A claim with no citation is not supported.

Reply with ONLY a JSON list and no other text, like this:
[{"claim": "short summary of the claim", "citations": [1], "supported": true}]"""

CITATION = re.compile(r"\[(\d+)\]")
CODE = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)


def find_citations(answer):
    """Return the citation numbers used in the answer, ignoring anything inside code."""
    text = CODE.sub("", answer)
    return sorted({int(n) for n in CITATION.findall(text)})


def bad_citations(answer, num_sources):
    """Citation numbers that point to a source that doesn't exist."""
    return [n for n in find_citations(answer) if not 1 <= n <= num_sources]


def parse_checks(reply):
    """Pull the JSON list out of the judge's reply; an unreadable reply gives an empty list."""
    start, end = reply.find("["), reply.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        checks = json.loads(reply[start:end + 1])
    except json.JSONDecodeError:
        return []
    return checks if isinstance(checks, list) else []


def judge_answer(answer, chunks, usage_log=None):
    """Check an answer's citations: real numbers (code) and real support (AI judge)."""
    bad = bad_citations(answer, len(chunks))
    prompt = f"Sources:\n\n{format_sources(chunks)}\n\nAnswer to check:\n{answer}"
    checks = parse_checks(ask_llm(prompt, system=JUDGE_PROMPT, usage_log=usage_log))
    supported = sum(1 for check in checks if check.get("supported") is True)
    score = supported / len(checks) if checks else 0.0
    return {"checks": checks, "bad_citations": bad, "support_score": score}


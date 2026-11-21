import json

from src.generation.llm import ask_llm

GRADER_PROMPT = """You are a fair but strict examiner. You compare a student's answer with a reference answer.

Grades:
- "correct": it contains the key facts of the reference answer and nothing that contradicts it.
- "partial": it gets some key facts right but misses an important one, or is too vague.
- "wrong": it misses the main point, contradicts the reference, or does not answer the question.

Judge the meaning, not the wording. Extra details are fine if they do not contradict the reference.
Ignore citation marks like [1].
If the reference answer is "I don't know.", the student is "correct" only if it says it doesn't know
or that the sources don't cover the question; it is "wrong" if it gives an answer anyway.

Reply with ONLY a JSON object and no other text, like this:
{"verdict": "correct", "reason": "one short sentence"}"""

SCORES = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def parse_verdict(reply):
    """Pull the JSON verdict out of the grader's reply, or None if it can't be read."""
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        verdict = json.loads(reply[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(verdict, dict) or verdict.get("verdict") not in SCORES:
        return None
    return verdict


def grade(item, result):
    """Mark one result from ask() against its answer key: correct, partial or wrong."""
    if not result["answered"]:
        verdict = "wrong" if item["answerable"] else "correct"
        return {"verdict": verdict, "score": SCORES[verdict], "reason": f"refused - {result['reason']}"}

    prompt = (f"Question: {item['question']}\n\n"
              f"Reference answer:\n{item['reference_answer']}\n\n"
              f"Student's answer:\n{result['answer']}")
    verdict = parse_verdict(ask_llm(prompt, system=GRADER_PROMPT))
    if verdict is None:
        return {"verdict": "unreadable", "score": 0.0, "reason": "the grader's reply had no readable verdict"}
    return {"verdict": verdict["verdict"], "score": SCORES[verdict["verdict"]], "reason": verdict.get("reason", "")}

import re

NO_ANSWER = "I don't know."

SYSTEM_PROMPT = f"""You answer questions about FastAPI using ONLY the numbered sources you are given.

Rules:
1. Use only facts from the sources. Never use outside knowledge.
2. After every sentence that uses a source, cite it with its number in plain square brackets, like [1] or [2][3].
3. If the sources do not contain the answer, reply with exactly: {NO_ANSWER}
4. Copy code exactly as it appears in the sources.
5. Be short and clear."""


def format_sources(chunks):
    """Number each chunk so the AI can cite it as [1], [2], ..."""
    parts = []
    for number, chunk in enumerate(chunks, start=1):
        parts.append(f"[{number}] (from {chunk['source']})\n{chunk['text']}")
    return "\n\n".join(parts)


def build_prompt(question, chunks):
    """Put the numbered sources and the question together into one message."""
    return f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"


def clean_citations(answer):
    """Turn citations written as 【2】 into the normal [2] style."""
    return re.sub(r"【(\d+)(?:†[^】]*)?】", r"[\1]", answer)
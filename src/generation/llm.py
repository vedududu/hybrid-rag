import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_name": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
    },
}

PROVIDER = "groq"

_client = None


def get_client():
    """Connect to the chosen provider once, then reuse the connection."""
    global _client
    if _client is None:
        settings = PROVIDERS[PROVIDER]
        _client = OpenAI(api_key=os.environ[settings["key_name"]], base_url=settings["base_url"])
    return _client


def ask_llm(prompt, system=None, temperature=0, usage_log=None):
    """Send a prompt to the LLM and return its reply as text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = get_client().chat.completions.create(
        model=PROVIDERS[PROVIDER]["model"],
        messages=messages,
        temperature=temperature,
    )
    if usage_log is not None:
        usage_log.append({"in": response.usage.prompt_tokens, "out": response.usage.completion_tokens})
    return response.choices[0].message.content



def stream_llm(prompt, system=None, temperature=0, usage_log=None):
    """Like ask_llm, but hands out the reply piece by piece: the AI's thinking first, then its answer text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    stream = get_client().chat.completions.create(
        model=PROVIDERS[PROVIDER]["model"],
        messages=messages,
        temperature=temperature,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            thinking = getattr(delta, "reasoning", None)
            if thinking:
                yield {"type": "thinking", "text": thinking}
            if delta.content:
                yield {"type": "text", "text": delta.content}
        if chunk.usage and usage_log is not None:
            usage_log.append({"in": chunk.usage.prompt_tokens, "out": chunk.usage.completion_tokens})
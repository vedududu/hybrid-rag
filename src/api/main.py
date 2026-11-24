import json
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.feedback import save_feedback
from src.generation.cache import cached_ask, cached_ask_stream

app = FastAPI(title="Hybrid RAG over the FastAPI docs")


class Question(BaseModel):
    """What a request to /ask must contain."""

    model_config = {"str_strip_whitespace": True}

    question: str = Field(min_length=3, max_length=500)


@app.get("/health")
def health():
    """Quick check that the API is running. No search, no AI."""
    return {"status": "ok"}


@app.post("/ask")
def ask_question(body: Question):
    """Answer a question about FastAPI, with sources and a confidence level."""
    return cached_ask(body.question)


@app.post("/ask/stream")
def ask_question_stream(body: Question):
    """Same as /ask, but sends progress updates as they happen – one JSON object per line."""
    lines = (json.dumps(update) + "\n" for update in cached_ask_stream(body.question))
    return StreamingResponse(lines, media_type="application/x-ndjson")



class Feedback(BaseModel):
    """A visitor's thumbs-up or thumbs-down on one answer."""

    question: str = Field(min_length=1, max_length=500)
    answer: str = Field(default="", max_length=10000)
    vote: Literal["up", "down"]
    level: str | None = None
    confidence: float | None = None
    comment: str | None = Field(default=None, max_length=1000)


@app.post("/feedback")
def give_feedback(body: Feedback):
    """Store a visitor's vote on an answer (Supabase if set up, otherwise a local file)."""
    return {"saved": True, "stored_in": save_feedback(**body.model_dump())}
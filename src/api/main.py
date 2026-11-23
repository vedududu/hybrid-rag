from fastapi import FastAPI
from pydantic import BaseModel, Field  # NEW

from src.generation.answer import ask  # NEW

app = FastAPI(title="Hybrid RAG over the FastAPI docs")


class Question(BaseModel):  # NEW
    """What a request to /ask must contain."""  # NEW

    model_config = {"str_strip_whitespace": True}  # NEW

    question: str = Field(min_length=3, max_length=500)  # NEW


@app.get("/health")
def health():
    """Quick check that the API is running. No search, no AI."""
    return {"status": "ok"}


@app.post("/ask")  # NEW
def ask_question(body: Question):  # NEW
    """Answer a question about FastAPI, with sources and a confidence level."""  # NEW
    return ask(body.question)  # NEW
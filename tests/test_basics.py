"""Fast checks of our own logic. No Groq calls and no AI models are loaded, so they run in seconds."""
from fastapi.testclient import TestClient

import src.generation.answer as answer
import src.generation.cache as cache
from src.api.main import app
from src.evaluation.retrieval_eval import first_evidence_hit
from src.generation.judge import find_citations
from src.generation.prompt import clean_citations
from src.retrieval.hybrid import rrf_fuse
from src.retrieval.neighbours import next_chunk_id

client = TestClient(app)


def test_clean_citations_turns_odd_brackets_into_normal_ones():
    assert clean_citations("Use status_code 【2】【3】.") == "Use status_code [2][3]."


def test_find_citations_ignores_numbers_inside_code():
    assert find_citations("Return it [1]. Example: `items[0]` and [3].") == [1, 3]


def test_rrf_rewards_chunks_found_by_both_searches():
    meaning = [{"id": "a"}, {"id": "b"}]
    keywords = [{"id": "b"}, {"id": "c"}]
    ranked = [chunk["id"] for chunk in rrf_fuse([meaning, keywords], [0.5, 0.5])]
    assert ranked[0] == "b"
    assert sorted(ranked) == ["a", "b", "c"]


def test_next_chunk_id_works_for_every_shelf():
    assert next_chunk_id("tutorial/cors.md::3") == "tutorial/cors.md::4"
    assert next_chunk_id("tutorial/cors.md::headings::7") == "tutorial/cors.md::headings::8"


def test_evidence_must_be_in_a_chunk_from_the_right_file():
    question = {"expected_sources": ["tutorial/cors.md"], "evidence": ["allow_origins=origins"]}
    wrong_file = {"path": "data/raw/tutorial/other.md", "text": "allow_origins=origins"}
    right_file = {"path": "data/raw/tutorial/cors.md", "text": "app.add_middleware(\n    allow_origins=origins)"}
    assert first_evidence_hit([wrong_file, right_file], question) == 2


def test_cache_answers_a_repeated_question_without_running_the_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(cache, "ask", lambda q: calls.append(q) or {"question": q, "answer": "x"})
    monkeypatch.setattr(cache, "_cache", {})
    first = cache.cached_ask("How do I add CORS?")
    second = cache.cached_ask("  how do i add CORS  ")
    assert len(calls) == 1
    assert (first["cached"], second["cached"]) == (False, True)


def test_gate_refuses_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(answer, "retrieve", lambda question, collection_name: [{"rerank_score": -11.0}])
    monkeypatch.setattr(answer, "ask_llm", lambda *args, **kwargs: 1 / 0)  # would crash if called
    result = answer.ask("How do I roast a chicken?")
    assert result["answered"] is False
    assert result["reason"].startswith("gate")


def test_api_health_and_input_validation():
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422
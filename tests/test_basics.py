"""Fast checks of our own logic. No Groq calls and no AI models are loaded, so they run in seconds."""
import json

from fastapi.testclient import TestClient

import src.api.main as api
import src.feedback as feedback
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
    assert clean_citations("See 【3†L1-L9】【4†L1-L7】.") == "See [3][4]."


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

    def fake_pipeline(question):
        calls.append(question)
        yield {"type": "done", "result": {"question": question, "answer": "x"}}

    monkeypatch.setattr(cache, "ask_stream", fake_pipeline)
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


def test_stream_route_sends_one_json_update_per_line(monkeypatch):
    def fake_stream(question):
        yield {"type": "stage", "text": "Searching the docs…"}
        yield {"type": "done", "result": {"answer": "x"}}

    monkeypatch.setattr(api, "cached_ask_stream", fake_stream)
    reply = client.post("/ask/stream", json={"question": "How do I add CORS?"})
    lines = [line for line in reply.text.splitlines() if line]
    assert [json.loads(line)["type"] for line in lines] == ["stage", "done"]



def test_feedback_is_saved_to_a_file_when_supabase_is_not_set_up(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setattr(feedback, "FEEDBACK_FILE", tmp_path / "feedback.jsonl")
    assert feedback.save_feedback("How do I add CORS?", "Use CORSMiddleware [1].", "up", "high", 1.0) == "file"
    row = json.loads((tmp_path / "feedback.jsonl").read_text(encoding="utf-8"))
    assert (row["vote"], row["level"]) == ("up", "high")


def test_feedback_route_accepts_only_up_or_down(monkeypatch):
    monkeypatch.setattr(api, "save_feedback", lambda **row: "file")
    good = client.post("/feedback", json={"question": "How do I add CORS?", "answer": "x", "vote": "down"})
    bad = client.post("/feedback", json={"question": "How do I add CORS?", "answer": "x", "vote": "meh"})
    assert good.json() == {"saved": True, "stored_in": "file"}
    assert bad.status_code == 422
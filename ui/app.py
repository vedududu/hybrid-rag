"""Web page for the Hybrid RAG system.

Two ways to run:
- API mode (default, local + Docker): the page only displays things; it sends questions to the FastAPI service
  and shows its progress updates live (POST /ask/stream).
- Direct mode (RAG_DIRECT=1, used on Streamlit Community Cloud): there is no separate API there,
  so the page runs the same ask() pipeline itself.
"""
import hashlib
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
import streamlit as st


def load_secrets_into_env():
    """On Streamlit Cloud, settings arrive as st.secrets – copy them into environment variables for the rest of the code."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ.setdefault(key, value)
    except Exception:
        pass  # no secrets file – normal when running locally or in Docker


load_secrets_into_env()

API_URL = os.getenv("API_URL", "http://localhost:8000")
DIRECT_MODE = os.getenv("RAG_DIRECT") == "1"
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "150"))  # protects the free Groq allowance on a public demo
DOCS_SITE = "https://fastapi.tiangolo.com/"
THINKING_SHOWN = 400   # how many characters of the AI's live thinking to show at once
THINKING_PAUSE = 0.012  # seconds between thinking pieces (Groq sends them almost instantly; this is only for show)
TYPING_PAUSE = 0.025    # seconds between answer pieces, so the answer is typed out instead of flashing in

EXAMPLES = [
    "How do I return a 201 status for a create endpoint?",
    "How do I add CORS to my app?",
    "How do I run code when the app starts up?",
    "How do I roast a chicken?",
]

LEVELS = {
    "high": ("HIGH", "#16A34A", "All or nearly all claims were confirmed in the sources."),
    "medium": ("MEDIUM", "#D97706", "Most claims were confirmed; check the sources for the rest."),
    "low": ("LOW", "#DC2626", "Few claims could be confirmed – treat this answer with care."),
    "none": ("NO ANSWER", "#6B7280", ""),
}

REASONS = {
    "gate": "None of the documentation looked relevant enough, so no AI was asked (the Gate).",
    "writer": "The AI read the most relevant documentation and found that it doesn't contain the answer.",
}

st.set_page_config(page_title="Ask the FastAPI docs", page_icon="📘", layout="centered")

st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem; color: white;
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 55%, #0D9488 100%);
    }
    .hero h1 { color: white; margin: 0; padding: 0; font-size: 2.1rem; }
    .hero p { margin: 0.35rem 0 0 0; opacity: 0.9; font-size: 1rem; }
    .badge {
        display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px; color: white;
        font-weight: 700; font-size: 0.85rem; letter-spacing: 0.04em;
    }
    .badge-note { opacity: 0.75; font-size: 0.85rem; margin-left: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- talking to the API ----------
@st.cache_data(ttl=10, show_spinner=False)
def api_is_up():
    """Ask GET /health whether the API is running (re-checked at most every 10 seconds)."""
    try:
        return requests.get(f"{API_URL}/health", timeout=2).ok
    except requests.RequestException:
        return False


def updates_from_api(question):
    """Ask POST /ask/stream and hand out each progress update as it arrives (one JSON object per line)."""
    with requests.post(f"{API_URL}/ask/stream", json={"question": question}, stream=True, timeout=180) as reply:
        reply.raise_for_status()
        for line in reply.iter_lines():
            if line:
                yield json.loads(line)


# ---------- direct mode (Streamlit Cloud) ----------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Import the RAG pipeline once and keep it in memory for every visitor."""
    try:  # Streamlit Cloud's Linux has an sqlite3 too old for ChromaDB – swap in a newer copy if installed
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.generation.cache import cached_ask_stream
    return cached_ask_stream


def get_updates(question):
    """Progress updates for one question, from the pipeline inside this app or from the API."""
    if DIRECT_MODE:
        return load_pipeline()(question)
    return updates_from_api(question)


def explain_error(error):
    """Turn whatever went wrong into a short, friendly message."""
    if isinstance(error, requests.ConnectionError) and not isinstance(error, requests.exceptions.ChunkedEncodingError):
        return f"Can't reach the API at {API_URL}. Is uvicorn running?"
    if isinstance(error, requests.Timeout):
        return "The answer took too long. The first question after start-up is slow – please try again."
    if isinstance(error, requests.HTTPError) and error.response is not None and error.response.status_code == 422:
        return "Please type a real question (between 3 and 500 characters)."
    if type(error).__name__ == "RateLimitError":
        return "The free AI service is busy right now (rate limit). Please wait a minute and try again."
    return f"Something went wrong while answering ({type(error).__name__}). Please try again."


def answer_with_live_progress(question):
    """Show each stage and the answer text as they arrive. Returns (result, error_message)."""
    if not 3 <= len(question) <= 500:
        return None, "Please type a real question (between 3 and 500 characters)."
    stage, thinking_box, text = st.empty(), st.empty(), st.empty()
    first_time = DIRECT_MODE and "result" not in st.session_state
    stage.info("Starting…" + (" The first question after the app wakes up takes 1–2 minutes while the AI "
                              "models load." if first_time else ""), icon="⏳")
    thought, written = "", ""

    def clear_progress():
        stage.empty()
        thinking_box.empty()
        text.empty()

    try:
        for update in get_updates(question):
            if update["type"] == "stage":
                stage.info(update["text"], icon="⏳")
            elif update["type"] == "thinking":
                thought += update["text"]
                thinking_box.caption("💭 " + ("…" + thought[-THINKING_SHOWN:] if len(thought) > THINKING_SHOWN else thought))
                time.sleep(THINKING_PAUSE)
            elif update["type"] == "text":
                thinking_box.empty()
                written += update["text"]
                text.markdown(re.sub(r"【(\d+)(?:†[^】]*)?】", r"[\1]", written) + " ▌")
                time.sleep(TYPING_PAUSE)
            elif update["type"] == "done":
                clear_progress()
                return update["result"], None
    except Exception as error:
        clear_progress()
        return None, explain_error(error)
    clear_progress()
    return None, "The answer stopped unexpectedly. Please try again."


# ---------- feedback ----------
def send_feedback(result, vote, comment):
    """Store a vote: through the API, or directly when running on Streamlit Cloud. Returns True if saved."""
    row = {"question": result["question"], "answer": result["answer"], "vote": vote,
           "level": result["level"], "confidence": result["confidence"], "comment": comment or None}
    try:
        if DIRECT_MODE:
            load_pipeline()  # makes the project's src folder importable
            from src.feedback import save_feedback
            save_feedback(**row)
        else:
            requests.post(f"{API_URL}/feedback", json=row, timeout=15).raise_for_status()
        return True
    except Exception:
        return False


def feedback_section(result):
    """Thumbs up/down under an answer, plus an optional comment after a thumbs down."""
    key = "feedback_" + hashlib.md5((result["question"] + result["answer"]).encode("utf-8")).hexdigest()[:12]
    if st.session_state.get(key + "_saved"):
        st.caption("Thanks for your feedback! 🙏")
        return
    st.caption("Was this answer helpful?")
    vote = st.feedback("thumbs", key=key)
    if vote is None:
        return
    comment = ""
    if vote == 0:
        comment = st.text_input("What was wrong or missing? (optional)", key=key + "_comment", max_chars=1000)
        if not st.button("Send feedback", key=key + "_send"):
            return
    if send_feedback(result, "up" if vote == 1 else "down", comment):
        st.session_state[key + "_saved"] = True
        st.caption("Thanks for your feedback! 🙏")
    else:
        st.caption("Sorry – your feedback couldn't be saved right now.")


# ---------- daily limit ----------
@st.cache_resource
def usage_counter():
    """One counter shared by every visitor of this running app."""
    return {"day": None, "count": 0}


def within_daily_limit():
    counter = usage_counter()
    today = date.today().isoformat()
    if counter["day"] != today:
        counter.update(day=today, count=0)
    if counter["count"] >= DAILY_LIMIT:
        return False
    counter["count"] += 1
    return True


# ---------- small display helpers ----------
def docs_link(path):
    """Turn data\\raw\\tutorial\\cors.md into the matching page on the FastAPI website."""
    page = path.replace("\\", "/").split("data/raw/", 1)[-1]
    if not page.endswith(".md"):
        return None
    page = page[: -len(".md")]
    page = page[: -len("index")] if page.endswith("index") else page + "/"
    return DOCS_SITE + page


def show_badge(result):
    name, colour, note = LEVELS.get(result["level"], LEVELS["none"])
    label = name if result["level"] == "none" else f"{name} · {result['confidence']:.2f}"
    st.markdown(
        f'<span class="badge" style="background:{colour}">{label}</span>'
        f'<span class="badge-note">{note}</span>',
        unsafe_allow_html=True,
    )


def show_result(result):
    st.markdown(f"#### {result['question']}")
    show_badge(result)
    if result.get("cached"):
        st.caption("⚡ Answered from cache – this question was asked recently, so no AI call was needed.")
    elif result.get("stats"):
        stats = result["stats"]
        st.caption(f"⏱ {stats['total_seconds']:.1f} s · {stats['total_tokens']:,} AI tokens")
    st.write("")

    if not result["answered"]:
        kind = result["reason"].split(":")[0]
        st.info(f"**I don't know.**  \n{REASONS.get(kind, result['reason'])}", icon="🤷")
        return

    with st.container(border=True):
        st.markdown(result["answer"])

    if result["bad_citations"]:
        st.warning(f"The answer cites sources that don't exist: {result['bad_citations']}", icon="⚠️")

    sources = result["sources"]
    with st.expander(f"📚 Sources ({len(sources)})", expanded=True):
        for s in sources:
            link = docs_link(s["path"])
            name = f"[{s['source']}]({link})" if link else s["source"]
            st.markdown(f"**[{s['number']}]** {name}  \n<small>`{s['path']}`</small>", unsafe_allow_html=True)

    checks = result["checks"]
    confirmed = sum(c["supported"] for c in checks)
    with st.expander(f"🔍 Fact-check: {confirmed} of {len(checks)} claims confirmed"):
        for c in checks:
            icon = "✅" if c.get("supported") else "❌"
            cited = ", ".join(f"[{n}]" for n in c["citations"]) or "no citation"
            st.markdown(f"{icon} {c['claim']}  \n<small>cited: {cited}</small>", unsafe_allow_html=True)

    feedback_section(result)


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        "1. **Hybrid search** – meaning (embeddings) + keywords (BM25), merged with RRF\n"
        "2. **Reranker** picks the 5 best chunks\n"
        "3. **Gate** – says *I don't know* if nothing is relevant\n"
        "4. **Writer** answers only from those chunks, with citations\n"
        "5. **Judge** fact-checks every claim → confidence"
    )
    st.markdown("### Tested on 50 questions")
    c1, c2 = st.columns(2)
    c1.metric("Correct", "84%")
    c2.metric("Honest refusals", "10 / 10")
    st.caption("Faithfulness 84% – share of claims backed by the sources.")
    st.divider()
    if DIRECT_MODE:
        st.info("Running the pipeline inside this app (cloud demo mode).", icon="☁️")
    elif api_is_up():
        st.success(f"API online · {API_URL}", icon="🟢")
    else:
        st.error(f"API offline · {API_URL}", icon="🔴")


# ---------- main page ----------
st.markdown(
    '<div class="hero"><h1>📘 Ask the FastAPI docs</h1>'
    "<p>Answers come only from the official documentation – with sources, a fact-check, "
    "and an honest <i>I don't know</i> when the docs don't cover it.</p></div>",
    unsafe_allow_html=True,
)


def use_example(text):
    st.session_state["question"] = text
    st.session_state["ask_now"] = True


st.caption("Try an example:")
cols = st.columns(2)
for i, example in enumerate(EXAMPLES):
    cols[i % 2].button(example, on_click=use_example, args=(example,), width="stretch")

with st.form("ask_form", clear_on_submit=False):
    question = st.text_input("Your question", key="question", placeholder="e.g. How do I add CORS to my app?")
    submitted = st.form_submit_button("Ask", type="primary", width="stretch")

if submitted or st.session_state.pop("ask_now", False):
    if within_daily_limit():
        result, error = answer_with_live_progress(st.session_state["question"].strip())
    else:
        result, error = None, f"This demo has reached its limit of {DAILY_LIMIT} questions for today. Please come back tomorrow!"
    if error:
        st.session_state.pop("result", None)
        st.error(error, icon="🚫")
    else:
        st.session_state["result"] = result

if "result" in st.session_state:
    st.divider()
    show_result(st.session_state["result"])

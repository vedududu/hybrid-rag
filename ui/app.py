"""Web page for the Hybrid RAG system. It only displays things – all the real work happens in the API."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
DOCS_SITE = "https://fastapi.tiangolo.com/"

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


def ask_api(question):
    """Send the question to POST /ask. Returns (result, error_message) – one of them is None."""
    try:
        reply = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=180)
    except requests.ConnectionError:
        return None, f"Can't reach the API at {API_URL}. Is uvicorn running?"
    except requests.Timeout:
        return None, "The API took too long to answer. The first question after start-up is slow – try again."
    if reply.status_code == 422:
        return None, "Please type a real question (between 3 and 500 characters)."
    if not reply.ok:
        return None, f"The API had a problem (status {reply.status_code}). Check the uvicorn terminal for details."
    return reply.json(), None


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
            icon = "✅" if c["supported"] else "❌"
            cited = ", ".join(f"[{n}]" for n in c["citations"]) or "no citation"
            st.markdown(f"{icon} {c['claim']}  \n<small>cited: {cited}</small>", unsafe_allow_html=True)


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
    if api_is_up():
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
    with st.spinner("Searching the docs, writing an answer and fact-checking it…"):
        result, error = ask_api(st.session_state["question"].strip())
    if error:
        st.session_state.pop("result", None)
        st.error(error, icon="🚫")
    else:
        st.session_state["result"] = result

if "result" in st.session_state:
    st.divider()
    show_result(st.session_state["result"])

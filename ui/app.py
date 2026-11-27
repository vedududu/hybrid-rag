"""Web page for the Hybrid RAG system – a chat over the FastAPI docs.

Two ways to run:
- API mode (default, local + Docker): the page only displays things; it sends questions to the FastAPI service
  and shows its progress updates live (POST /ask/stream).
- Direct mode (RAG_DIRECT=1, used on Streamlit Community Cloud): there is no separate API there,
  so the page runs the same pipeline itself.
"""
import base64
import hashlib
import html
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

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
GITHUB_URL = "https://github.com/vedududu/hybrid-rag"
# The "ask me out… for an interview" card (a joke, always on screen: right side, or in the sidebar on small screens)
CONTACT_EMAIL = "vedant3715@gmail.com"
INTERVIEW_MAILTO = (f"mailto:{CONTACT_EMAIL}?subject=" + quote("Interview invite (via your RAG demo) 💚")
                    + "&body=" + quote("Hi Vedant,\n\nI tried your RAG demo and would like to talk about a role at "
                                       "our company.\n\n"))
MAX_TURNS = 20          # how many questions the chat keeps on screen
THINKING_SHOWN = 260    # how many characters of the AI's live thinking to show at once
THINKING_PAUSE = 0.012  # seconds between thinking pieces (Groq sends them almost instantly; this is only for show)
TYPING_PAUSE = 0.025    # seconds between answer pieces, so the answer is typed out instead of flashing in

EXAMPLES = [
    ("⚡", "How do I return a 201 status for a create endpoint?"),
    ("🌐", "How do I add CORS to my app?"),
    ("🚀", "How do I run code when the app starts up?"),
    ("🍗", "How do I roast a chicken?"),
]
HEADLINE_STATS = [("95%","correct on a 100-question held-out exam"), ("20/20", "honest “I don't know”"),
                  ("~8 s", "per answer")]
STEPS = [("search", "Search", "meaning + keywords"), ("rerank", "Rerank", "best 5 passages"),
         ("gate", "Gate", "relevant enough?"), ("think", "Think", "AI reads sources"),
         ("write", "Write", "with citations"), ("check", "Fact-check", "every claim")]
LEVELS = {
    "high": ("HIGH", "#B8F34A", "All or nearly all claims were confirmed in the sources."),
    "medium": ("MEDIUM", "#FFB020", "Most claims were confirmed – check the sources for the rest."),
    "low": ("LOW", "#FF5A5F", "Few claims could be confirmed – treat this answer with care."),
    "none": ("NO ANSWER", "#8B93A1", ""),
}
REASONS = {
    "gate": "None of the documentation looked relevant enough, so no AI was asked (the Gate).",
    "writer": "The AI read the most relevant documentation and found that it doesn't contain the answer.",
}
CITATION = re.compile(r"【(\d+)(?:†[^】]*)?】")

LOGO = Path(__file__).resolve().parent / "logo.png"
LOGO_URI = "data:image/png;base64," + base64.b64encode(LOGO.read_bytes()).decode()

st.set_page_config(page_title="Ask the FastAPI docs", page_icon=str(LOGO), layout="centered",
                   initial_sidebar_state="expanded")

HIRE_CARD = ('<svg class="heart" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 '
            '2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 '
            '3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>'
            '<div class="q1">Would you like to ask Vedant out…</div>'
            '<div class="q2">…for an <span>interview?</span></div>'
            '<div class="small">Builds AI that admits when it doesn&#39;t know. Emotionally available, technically too.</div>'
            f'<div class="btns"><a class="yes" href="{INTERVIEW_MAILTO}" target="_blank">Yes! 💚</a>'
            '<span class="no"><span class="a">Maybe later</span><span class="b">Sure? 🥺</span></span></div>')


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root { --lime:#B8F34A; --lime-soft:rgba(184,243,74,.14); --orange:#FF8A3D; --card:#101318; --line:#232831;
        --muted:#8B93A1; --mono:'JetBrains Mono', ui-monospace, monospace; --head:'Space Grotesk', sans-serif; }
.stApp { background: radial-gradient(900px 420px at 8% -8%, rgba(184,243,74,.10), transparent 60%),
                     radial-gradient(700px 380px at 105% 10%, rgba(255,138,61,.07), transparent 60%), #07080A; }
[data-testid="stSidebar"] { background:#0B0D10; border-right:1px solid var(--line); }
.block-container { padding-top: 4.6rem; }

/* ---------- header ---------- */
.eyebrow { font-family:var(--mono); font-size:.72rem; letter-spacing:.22em; color:var(--muted); text-transform:uppercase; }
.eyebrow:after { content:""; display:block; width:48px; height:2px; background:var(--lime); margin-top:.55rem;
                 box-shadow:0 0 10px var(--lime); }
.title { font-family:var(--head); font-weight:700; font-size:3.1rem; line-height:1.02; margin:.9rem 0 .2rem 0; color:#F2F4F7; }
.title span { color:var(--lime); text-shadow:0 0 28px rgba(184,243,74,.45); }
.title.small { font-size:1.65rem; margin:.4rem 0 .6rem 0; }
.subtitle { font-family:var(--head); font-size:1.35rem; color:var(--orange); font-weight:700; margin-bottom:.9rem; }
.chips { display:flex; flex-wrap:wrap; gap:.5rem; margin:.4rem 0 1.1rem 0; }
.chip { font-family:var(--mono); font-size:.78rem; color:#D7DBE2; border:1px solid var(--line); background:var(--card);
        padding:.32rem .7rem; border-radius:999px; }
.chip b { color:var(--lime); font-weight:600; }
.stats { display:grid; grid-template-columns:repeat(3, 1fr); gap:.7rem; margin:.3rem 0 1.4rem 0; }
.stat { border:1px solid var(--line); background:var(--card); border-radius:14px; padding:.8rem .9rem; }
.stat .n { font-family:var(--head); font-size:1.6rem; font-weight:700; color:var(--lime); }
.stat .l { font-family:var(--mono); font-size:.72rem; color:var(--muted); }
.label { font-family:var(--mono); font-size:.72rem; letter-spacing:.18em; color:var(--muted); text-transform:uppercase;
         margin:.4rem 0 .5rem 0; }

/* ---------- example buttons ---------- */
div[data-testid="stButton"] button { border-radius:12px; border:1px solid var(--line); background:var(--card);
    text-align:left; justify-content:flex-start; }
div[data-testid="stButton"] button:hover { border-color:var(--lime); box-shadow:0 0 16px rgba(184,243,74,.18); }

/* ---------- chat bubbles ---------- */
[data-testid="stChatMessage"] { background:var(--card); border:1px solid var(--line); border-radius:18px;
    padding:.9rem 1rem; margin-bottom:.8rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction:row-reverse;
    background:#171B21; width:fit-content; max-width:80%; margin-left:auto; border-color:rgba(184,243,74,.22); }
[data-testid="stChatMessageAvatarUser"] { background:var(--lime) !important; color:#07080A !important; }
[data-testid="stChatMessageAvatarCustom"] { background:#1A1F27 !important; border:1px solid rgba(184,243,74,.35); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    text-align:right; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarCustom"]) {
    border-color:rgba(184,243,74,.28); box-shadow:0 0 30px rgba(184,243,74,.05); }
[data-testid="stChatInput"] { border:1px solid rgba(184,243,74,.35) !important; border-radius:14px !important;
    box-shadow:0 0 22px rgba(184,243,74,.10); }

/* ---------- pipeline tracker ---------- */
.pipe { display:grid; grid-template-columns:repeat(6, 1fr); gap:.4rem; margin:.2rem 0 .8rem 0; }
.step { border:1px solid var(--line); border-radius:10px; padding:.42rem .45rem; background:#0C0E12; min-width:0; }
.step .t { font-family:var(--mono); font-size:.72rem; font-weight:600; color:var(--muted); white-space:nowrap; }
.step .s { font-family:var(--mono); font-size:.62rem; color:#5E6673; line-height:1.25; }
.step.done { border-color:rgba(184,243,74,.45); }
.step.done .t { color:var(--lime); }
.step.active { border-color:var(--lime); box-shadow:0 0 14px rgba(184,243,74,.35); animation:pulse 1.1s ease-in-out infinite; }
.step.active .t { color:#F2F4F7; }
.step.stop { border-color:var(--orange); } .step.stop .t { color:var(--orange); }
.step.skip { opacity:.35; }
@keyframes pulse { 0%,100% { box-shadow:0 0 6px rgba(184,243,74,.25);} 50% { box-shadow:0 0 18px rgba(184,243,74,.6);} }

/* ---------- streaming placeholder ---------- */
.stream-label { font-family:var(--mono); font-size:.75rem; color:var(--lime); margin:.2rem 0 .45rem 0; }
.bar { height:7px; border-radius:6px; margin:.42rem 0; background:linear-gradient(90deg, rgba(184,243,74,.10),
       rgba(184,243,74,.55), rgba(184,243,74,.10)); background-size:200% 100%; animation:shimmer 1.4s linear infinite; }
@keyframes shimmer { from { background-position:200% 0; } to { background-position:-200% 0; } }
.thought { font-family:var(--mono); font-size:.74rem; color:#6F7886; margin-top:.5rem; line-height:1.45; }

/* ---------- answer header ---------- */
.ahead { display:flex; align-items:center; gap:.9rem; margin-bottom:.6rem; }
.ring { width:58px; height:58px; border-radius:50%; display:grid; place-items:center; flex:none; }
.ring div { width:46px; height:46px; border-radius:50%; background:var(--card); display:grid; place-items:center;
            font-family:var(--mono); font-size:.8rem; font-weight:600; }
.level { font-family:var(--mono); font-size:.78rem; font-weight:600; letter-spacing:.12em; }
.note { font-size:.85rem; color:var(--muted); }
.meta { font-family:var(--mono); font-size:.72rem; color:var(--muted); margin-top:.15rem; }
.meta b { color:var(--lime); font-weight:600; }

/* ---------- sources + claims ---------- */
.srcs { display:grid; grid-template-columns:repeat(auto-fill, minmax(210px, 1fr)); gap:.5rem; margin:.3rem 0 .4rem 0; }
.src { display:block; border:1px solid var(--line); border-radius:12px; padding:.55rem .7rem; background:#0C0E12;
       text-decoration:none !important; transition:all .15s; }
.src:hover { border-color:var(--lime); box-shadow:0 0 14px rgba(184,243,74,.2); }
.src .num { font-family:var(--mono); font-size:.68rem; color:#07080A; background:var(--lime); border-radius:6px;
            padding:.05rem .38rem; margin-right:.45rem; font-weight:600; }
.src .name { font-family:var(--mono); font-size:.8rem; color:#E8EAED; }
.src .go { font-family:var(--mono); font-size:.66rem; color:var(--muted); margin-top:.25rem; }
.claim { border-left:3px solid; border-radius:8px; padding:.4rem .65rem; margin:.35rem 0; background:#0C0E12; font-size:.86rem; }
.claim.ok { border-color:var(--lime); } .claim.bad { border-color:#FF5A5F; }
.claim small { font-family:var(--mono); color:var(--muted); }
.refusal { border:1px solid rgba(255,138,61,.45); border-radius:14px; padding:.8rem 1rem; background:rgba(255,138,61,.06); }
.refusal b { color:var(--orange); font-family:var(--head); font-size:1.05rem; }
.errorcard { border:1px solid rgba(255,90,95,.5); border-radius:14px; padding:.7rem 1rem; background:rgba(255,90,95,.07); }

/* ---------- sidebar ---------- */
.brand { font-family:var(--head); font-weight:700; font-size:1.15rem; color:#F2F4F7; display:flex; align-items:center; gap:.55rem; }
.brand img { width:30px; height:30px; filter:drop-shadow(0 0 6px rgba(184,243,74,.45)); }
.status { font-family:var(--mono); font-size:.72rem; color:var(--muted); margin:.3rem 0 1rem 0; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.4rem; }
.dot.on { background:var(--lime); box-shadow:0 0 8px var(--lime); } .dot.off { background:#FF5A5F; }
.arch { border:1px dashed var(--line); border-radius:14px; padding:.7rem; }
.node { border:1px solid var(--line); border-radius:10px; padding:.38rem .55rem; background:#101318;
        font-family:var(--mono); font-size:.72rem; color:#D7DBE2; }
.node small { color:var(--muted); display:block; font-size:.62rem; }
.arrow { text-align:center; color:var(--lime); font-size:.8rem; line-height:1.1; text-shadow:0 0 6px var(--lime); }
.mini { display:grid; grid-template-columns:1fr 1fr; gap:.45rem; }
.mini div { border:1px solid var(--line); border-radius:10px; padding:.45rem .55rem; background:#101318; }
.mini b { font-family:var(--head); color:var(--lime); font-size:1.1rem; display:block; }
.mini span { font-family:var(--mono); font-size:.62rem; color:var(--muted); }
.footer { font-family:var(--mono); font-size:.7rem; color:#5E6673; text-align:center; margin:1.2rem 0 .4rem 0; }
.footer a { color:var(--muted); }

/* ---------- "ask me out… for an interview" popup ---------- */
/* --gap = the empty strip right of the chat column (sidebar 300px + column 704px); the card sizes itself to fit it */
.hire { --gap:calc((100vw - 1004px) / 2); --w:clamp(196px, calc(var(--gap) - 24px), 272px);
        position:fixed; top:30vh; width:var(--w); right:max(10px, calc((var(--gap) - var(--w)) / 2));
        container-type:inline-size; z-index:1000; padding:1.05rem 1.1rem .95rem 1.1rem;
        background:linear-gradient(160deg, #12161C, #0B0D10); border:1px solid rgba(184,243,74,.38); border-radius:18px;
        box-shadow:0 0 0 1px rgba(184,243,74,.06), 0 18px 50px rgba(0,0,0,.55), 0 0 36px rgba(184,243,74,.10);
        }
.hire.side { display:none; }
.hire .heart { width:46px; height:46px; display:block; margin:0 0 .55rem 0;
               filter:drop-shadow(0 0 6px rgba(184,243,74,.9)) drop-shadow(0 0 16px rgba(184,243,74,.45));
               animation:heartbeat 1.25s ease-in-out infinite; transform-origin:center; }
.hire .heart path { fill:var(--lime); transition:fill .25s; }
@keyframes heartbeat { 0%,100% { transform:scale(1); } 14% { transform:scale(1.22); } 28% { transform:scale(1); }
                       42% { transform:scale(1.14); } 70% { transform:scale(1); } }
.hire .q1 { font-family:var(--mono); font-size:.74rem; color:var(--muted); letter-spacing:.02em; }
.hire .q2 { font-family:var(--head); font-weight:700; font-size:1.32rem; line-height:1.15; color:#F2F4F7; margin:.15rem 0 .45rem 0; }
.hire .q2 span { color:var(--lime); text-shadow:0 0 14px rgba(184,243,74,.45); }
.hire .small { font-size:.78rem; color:#AEB4BE; line-height:1.4; margin-bottom:.85rem; }
.hire .btns { display:flex; gap:.5rem; }
.hire .btns > * { flex:1; cursor:pointer; text-align:center; text-decoration:none !important; border-radius:10px; padding:.48rem .4rem;
                font-family:var(--mono); font-size:.74rem; font-weight:600; transition:all .2s; }
.hire .yes { background:var(--lime); color:#07080A !important; box-shadow:0 0 14px rgba(184,243,74,.35); }
.hire .yes:hover { box-shadow:0 0 22px rgba(184,243,74,.7); transform:translateY(-1px); }
.hire .no { border:1px solid var(--line); color:#8B93A1 !important; }
.hire .no .b { display:none; }
.hire .no:hover { border-color:rgba(255,90,95,.5); color:#FF5A5F !important; }
.hire .no:hover .a { display:none; } .hire .no:hover .b { display:inline; }
.hire:has(.yes:hover) .heart { animation-duration:.55s; }                   /* excited */
.hire:has(.no:hover) .heart { animation:none; transform:scale(.9); filter:none; }  /* heartbroken */
.hire:has(.no:hover) .heart path { fill:#3A404A; }
@container (max-width:230px) {   /* narrow card: smaller type, buttons stacked */
  .hire .heart { width:36px; height:36px; } .hire .q1 { font-size:.68rem; } .hire .q2 { font-size:1.12rem; }
  .hire .small { font-size:.74rem; } .hire .btns { flex-direction:column; gap:.4rem; } }
@media (max-width:1399px) {   /* no room on the right: show the copy at the bottom of the sidebar instead */
  .hire.float { display:none; }
  .hire.side { display:block; position:static; width:auto; margin-top:1rem; } }
@media (prefers-reduced-motion:reduce) { .hire .heart { animation:none; } }
</style>
""" + f'<div class="hire float">{HIRE_CARD}</div>', unsafe_allow_html=True)


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


# ---------- the pipeline tracker ----------
def tracker_html(states, notes):
    """The six pipeline steps as glowing tiles: todo / active / done / stop / skip."""
    tiles = []
    for key, name, hint in STEPS:
        mark = {"done": "✓ ", "stop": "■ ", "active": "● "}.get(states[key], "")
        tiles.append(f'<div class="step {states[key]}"><div class="t">{mark}{name}</div>'
                     f'<div class="s">{html.escape(notes.get(key, hint))}</div></div>')
    return '<div class="pipe">' + "".join(tiles) + "</div>"


def tracker_for_result(result):
    """A finished tracker for an answer in the chat history, using the timings the pipeline measured."""
    seconds = (result.get("stats") or {}).get("seconds", {})
    states = {key: "done" for key, _, _ in STEPS}
    notes = {}
    if "search" in seconds:
        notes["search"] = f"{seconds['search']:.1f} s"
    if "writer" in seconds:
        notes["write"] = f"{seconds['writer']:.1f} s"
    if "judge" in seconds:
        notes["check"] = f"{seconds['judge']:.1f} s"
    reason = result.get("reason", "")
    if reason.startswith("gate"):
        states.update(gate="stop", think="skip", write="skip", check="skip")
        notes["gate"] = "not relevant"
    elif reason.startswith("writer"):
        states["check"] = "skip"
        notes["write"] = "not in sources"
    if result.get("cached"):
        notes = {key: "from cache" for key, _, _ in STEPS}
    return tracker_html(states, notes)


def run_question(question):
    """Ask one question and show the pipeline working live. Returns (result, error_message)."""
    if not 3 <= len(question) <= 500:
        return None, "Please type a real question (between 3 and 500 characters)."
    order = [key for key, _, _ in STEPS]
    states = {key: "todo" for key in order}
    notes, started = {}, {}
    tracker, stream_box, text = st.empty(), st.empty(), st.empty()

    def move_to(step):
        """Mark every step before `step` as done (with its time) and light up `step`."""
        now = time.perf_counter()
        for key in order[: order.index(step)]:
            if states[key] != "done":
                if key in started:
                    notes.setdefault(key, f"{now - started[key]:.1f} s")
                states[key] = "done"
        if states[step] != "active":
            states[step] = "active"
            started[step] = now
        tracker.markdown(tracker_html(states, notes), unsafe_allow_html=True)

    def clear_live_view():
        tracker.empty()
        stream_box.empty()
        text.empty()

    def streaming_placeholder(thought=""):
        shown = ("…" + thought[-THINKING_SHOWN:]) if len(thought) > THINKING_SHOWN else thought
        stream_box.markdown('<div class="stream-label">● Streaming response…</div>'
                            '<div class="bar" style="width:92%"></div><div class="bar" style="width:78%"></div>'
                            '<div class="bar" style="width:60%"></div>'
                            + (f'<div class="thought">💭 {html.escape(shown)}</div>' if shown else ""),
                            unsafe_allow_html=True)

    first_time = DIRECT_MODE and not st.session_state.get("history")
    move_to("search")
    if first_time:
        notes["search"] = "loading models…"
        tracker.markdown(tracker_html(states, notes), unsafe_allow_html=True)
    thought, written = "", ""
    try:
        for update in get_updates(question):
            kind = update["type"]
            if kind == "stage" and update["text"].startswith("Found"):
                found = re.search(r"in ([\d.]+) s", update["text"])
                notes["search"] = f"{found.group(1)} s" if found else notes.get("search", "")
                notes["rerank"], notes["gate"] = "5 passages", "passed"
                move_to("think")
                streaming_placeholder()
            elif kind == "stage" and update["text"].startswith("Fact-checking"):
                move_to("check")
            elif kind == "thinking":
                thought += update["text"]
                streaming_placeholder(thought)
                time.sleep(THINKING_PAUSE)
            elif kind == "text":
                if states["write"] != "active":
                    move_to("write")
                    stream_box.empty()
                written += update["text"]
                text.markdown(CITATION.sub(r"[\1]", written) + " ▌")
                time.sleep(TYPING_PAUSE)
            elif kind == "done":
                clear_live_view()
                return update["result"], None
    except Exception as error:
        clear_live_view()
        return None, explain_error(error)
    clear_live_view()
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


def feedback_section(result, turn):
    """Thumbs up/down under an answer, plus an optional comment after a thumbs down."""
    key = f"feedback_{turn}_" + hashlib.md5((result["question"] + result["answer"]).encode("utf-8")).hexdigest()[:10]
    if st.session_state.get(key + "_saved"):
        st.caption("Thanks for your feedback! 🙏")
        return
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
        st.rerun()  # redraw the page so the buttons are replaced by the thank-you note
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


# ---------- showing an answer ----------
def docs_link(path):
    """Turn data\\raw\\tutorial\\cors.md into the matching page on the FastAPI website."""
    page = path.replace("\\", "/").split("data/raw/", 1)[-1]
    if not page.endswith(".md"):
        return None
    page = page[: -len(".md")]
    page = page[: -len("index")] if page.endswith("index") else page + "/"
    return DOCS_SITE + page


def pretty_path(path):
    """data\\raw\\tutorial\\security\\simple-oauth2.md -> tutorial › security › simple-oauth2"""
    page = path.replace("\\", "/").split("data/raw/", 1)[-1].removesuffix(".md")
    return " › ".join(page.split("/"))


def answer_header(result):
    """Confidence ring, level, what it means, and time/tokens."""
    name, colour, note = LEVELS.get(result["level"], LEVELS["none"])
    percent = round(result["confidence"] * 100)
    stats = result.get("stats") or {}
    if result.get("cached"):
        meta = "<b>⚡ from cache</b> · no AI call needed"
    elif stats:
        meta = f"<b>⏱ {stats.get('total_seconds', 0):.1f} s</b> · {stats.get('total_tokens', 0):,} AI tokens"
    else:
        meta = ""
    ring_label = f"{percent}%" if result["answered"] else "–"
    st.markdown(
        f'<div class="ahead"><div class="ring" style="background:conic-gradient({colour} {percent}%, #232831 0)">'
        f'<div style="color:{colour}">{ring_label}</div></div>'
        f'<div><div class="level" style="color:{colour}">{name + " CONFIDENCE" if result["answered"] else name}</div>'
        f'<div class="note">{note or "No answer was given."}</div><div class="meta">{meta}</div></div></div>',
        unsafe_allow_html=True)


def render_answer(result, turn):
    """One finished answer inside its chat bubble."""
    answer_header(result)
    st.markdown(tracker_for_result(result), unsafe_allow_html=True)
    if not result["answered"]:
        kind = result["reason"].split(":")[0]
        st.markdown(f'<div class="refusal"><b>🤷 I don\'t know.</b><br>'
                    f'{html.escape(REASONS.get(kind, result["reason"]))}</div>', unsafe_allow_html=True)
        return
    st.markdown(result["answer"])
    if result["bad_citations"]:
        st.warning(f"The answer cites sources that don't exist: {result['bad_citations']}", icon="⚠️")
    sources = result["sources"]
    if sources:
        cards = []
        for s in sources:
            link = docs_link(s["path"])
            inner = (f'<span class="num">{s["number"]}</span><span class="name">{html.escape(pretty_path(s["path"]))}'
                     f'</span><div class="go">{"open on fastapi.tiangolo.com ↗" if link else html.escape(s["source"])}</div>')
            cards.append(f'<a class="src" href="{link}" target="_blank">{inner}</a>' if link
                         else f'<div class="src">{inner}</div>')
        st.markdown(f'<div class="label">Sources · {len(sources)}</div><div class="srcs">{"".join(cards)}</div>',
                    unsafe_allow_html=True)
    checks = result["checks"]
    confirmed = sum(1 for c in checks if c.get("supported"))
    with st.expander(f"🔍 Fact-check · {confirmed} of {len(checks)} claims confirmed"):
        for c in checks:
            ok = bool(c.get("supported"))
            cited = ", ".join(f"[{n}]" for n in c.get("citations", [])) or "no citation"
            st.markdown(f'<div class="claim {"ok" if ok else "bad"}">{"✅" if ok else "❌"} {html.escape(c["claim"])}'
                        f'<br><small>cited: {cited}</small></div>', unsafe_allow_html=True)
    feedback_section(result, turn)


# ---------- sidebar ----------
with st.sidebar:
    st.markdown(f'<div class="brand"><img src="{LOGO_URI}" alt="">RAG Assistant</div>', unsafe_allow_html=True)
    if DIRECT_MODE:
        status = '<span class="dot on"></span>Online · cloud demo mode'
    elif api_is_up():
        status = f'<span class="dot on"></span>Online · API {html.escape(API_URL)}'
    else:
        status = f'<span class="dot off"></span>Offline · API {html.escape(API_URL)}'
    st.markdown(f'<div class="status">{status}</div>', unsafe_allow_html=True)
    st.markdown('<div class="label">At a glance</div><div class="arch">'
                '<div class="node">Your question</div><div class="arrow">↓</div>'
                '<div class="node">Hybrid search<small>embeddings + BM25 → RRF</small></div><div class="arrow">↓</div>'
                '<div class="node">Reranker<small>MiniLM cross-encoder · top 5</small></div><div class="arrow">↓</div>'
                '<div class="node">Gate<small>refuse if nothing is relevant</small></div><div class="arrow">↓</div>'
                '<div class="node">Writer<small>gpt-oss-120b on Groq · citations</small></div><div class="arrow">↓</div>'
                '<div class="node">Judge<small>checks every claim → confidence</small></div></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="label" style="margin-top:1rem">Tested</div><div class="mini">'
                '<div><b>95%</b><span>correct (exam)</span></div><div><b>20/20</b><span>honest refusals</span></div>'
                '<div><b>81%</b><span>faithfulness</span></div><div><b>84%</b><span>passage found</span></div>'
                '</div>', unsafe_allow_html=True)
    st.write("")
    if st.button("🧹  Clear chat", width="stretch"):
        st.session_state["history"] = []
        st.rerun()
    st.caption("Each question is answered on its own – the assistant doesn't remember earlier questions.")
    st.markdown(f'<div class="footer"><a href="{GITHUB_URL}" target="_blank">GitHub ↗</a></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="hire side">{HIRE_CARD}</div>', unsafe_allow_html=True)  # small screens only


# ---------- main page ----------
history = st.session_state.setdefault("history", [])


def ask_example(text):
    st.session_state["pending"] = text


if not history:
    st.markdown('<div class="eyebrow">Hybrid RAG · FastAPI docs</div>'
                '<div class="title">Ask the<br><span>FastAPI docs</span></div>'
                '<div class="subtitle">Search · Rerank · Fact-check</div>'
                '<div class="chips"><span class="chip"><b>◉</b> Live streaming</span>'
                '<span class="chip"><b>✓</b> Cited &amp; fact-checked</span>'
                '<span class="chip"><b>∅</b> Honest “I don\'t know”</span>'
                '<span class="chip"><b>⚡</b> Cached repeats</span></div>'
                '<div class="stats">'
                + "".join(f'<div class="stat"><div class="n">{n}</div><div class="l">{label}</div></div>'
                          for n, label in HEADLINE_STATS)
                + '</div><div class="label">Try an example</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (icon, example) in enumerate(EXAMPLES):
        cols[i % 2].button(f"{icon}  {example}", on_click=ask_example, args=(example,), width="stretch",
                           key=f"example_{i}")
else:
    st.markdown('<div class="eyebrow">Hybrid RAG · FastAPI docs</div>'
                '<div class="title small">Ask the <span>FastAPI docs</span></div>', unsafe_allow_html=True)

for turn, item in enumerate(history):
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant", avatar="🤖"):
        if item.get("error"):
            st.markdown(f'<div class="errorcard">🚫 {html.escape(item["error"])}</div>', unsafe_allow_html=True)
        else:
            render_answer(item["result"], turn)

typed = st.chat_input("Ask anything about FastAPI…")
question = (typed or st.session_state.pop("pending", None) or "").strip()
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🤖"):
        if within_daily_limit():
            result, error = run_question(question)
        else:
            result, error = None, f"This demo has reached its limit of {DAILY_LIMIT} questions for today. Please come back tomorrow!"
    history.append({"question": question, "result": result, "error": error})
    del history[:-MAX_TURNS]
    st.rerun()

st.markdown(f'<div class="footer">FastAPI docs · ChromaDB · BM25 · MiniLM reranker · gpt-oss-120b on Groq · '
            f'<a href="{GITHUB_URL}" target="_blank">source on GitHub ↗</a></div>', unsafe_allow_html=True)

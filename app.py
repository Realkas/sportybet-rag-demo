"""
Streamlit RAG demo — chat over SportyBet Nigeria's public support knowledge base
using NVIDIA NIM embeddings for retrieval and Claude for generation.

Local run:
    export NVIDIA_API_KEY=...
    export ANTHROPIC_API_KEY=...
    streamlit run app.py

Deployed (Streamlit Community Cloud): keys come from st.secrets, configured
in the app dashboard — never committed to the repo.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from rag_core import LocalKnowledgeBase, answer_question, TOP_K

load_dotenv()

MAX_QUESTIONS_PER_SESSION = 15  # cost guard when running on the owner's baked-in keys

SAMPLE_QUESTIONS = [
    "Мінімальний і максимальний депозит?",
    "Як вивести кошти і скільки це триває?",
    "Як пройти верифікацію акаунта?",
    "Як працює self-exclusion?",
]


def _secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


st.set_page_config(
    page_title="SportyBet Nigeria — Support Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Design system ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,500&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --ink: #1A1613;
        --paper: #F3EEE3;
        --lamp: #E8A33D;
        --pen: #C1432E;
        --ink-soft: #4A4038;
        --muted: #C9C2B4;
        --hairline: rgba(243, 238, 227, 0.12);
    }

    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }

    html, body, .stApp {
        background: var(--ink) !important;
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .block-container { max-width: 860px; padding-top: 2.5rem; }

    /* ---- hero ---- */
    .hero { animation: heroIn 0.6s ease-out; margin-bottom: 0.5rem; }
    @keyframes heroIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.75rem;
        color: var(--lamp);
        margin: 0 0 0.9rem;
    }
    .hero-title {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: clamp(2.3rem, 5vw, 3.5rem);
        line-height: 1.05;
        letter-spacing: -0.01em;
        color: var(--paper);
        margin: 0 0 1.1rem;
    }
    .hero-sub {
        font-size: 1.03rem;
        line-height: 1.65;
        color: var(--muted);
        max-width: 58ch;
        margin: 0 0 1.6rem;
    }

    .redline-card {
        background: var(--paper);
        border-left: 3px solid var(--pen);
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin-bottom: 1.6rem;
    }
    .redline-q { font-weight: 600; color: var(--ink); margin: 0 0 0.5rem; font-size: 0.95rem; }
    .redline-wrong {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.86rem;
        color: var(--pen);
        margin: 0 0 0.35rem;
        opacity: 0.85;
    }
    .redline-wrong s {
        text-decoration: none;
        background-image: linear-gradient(var(--pen), var(--pen));
        background-repeat: no-repeat;
        background-size: 0% 2px;
        background-position: 0 55%;
        animation: strike 0.5s ease-out 0.7s forwards;
    }
    @keyframes strike { to { background-size: 100% 2px; } }
    .redline-right {
        color: var(--ink);
        font-size: 0.95rem;
        margin: 0;
        opacity: 0;
        transform: translateY(4px);
        animation: rise 0.4s ease-out 1.3s forwards;
    }
    @keyframes rise { to { opacity: 1; transform: none; } }

    .hero-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.03em;
        color: var(--muted);
        opacity: 0.75;
        margin: 0 0 1.8rem;
    }

    @media (prefers-reduced-motion: reduce) {
        .hero, .redline-wrong s, .redline-right { animation: none !important; }
        .redline-wrong s { background-size: 100% 2px !important; }
        .redline-right { opacity: 1 !important; transform: none !important; }
    }

    hr.divider { border: none; border-top: 1px solid var(--hairline); margin: 0 0 1.6rem; }

    /* ---- chat ---- */
    .chip-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0 0 0.6rem;
    }
    .stButton > button {
        background: transparent;
        color: var(--paper);
        border: 1px solid var(--hairline);
        border-radius: 999px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.85rem;
        padding: 0.5rem 0.9rem;
        transition: border-color 0.15s ease, color 0.15s ease;
    }
    .stButton > button:hover { border-color: var(--lamp); color: var(--lamp); }
    .stButton > button:focus-visible { outline: 2px solid var(--lamp); outline-offset: 2px; }

    [data-testid="stChatMessageContent"] {
        background: var(--paper);
        color: var(--ink);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        border: 1px solid var(--hairline);
    }
    [data-testid="stChatMessageContent"] p { color: var(--ink); }

    [data-testid="stChatInput"] {
        background: var(--paper);
        border-radius: 14px;
        border: 1px solid var(--hairline);
    }
    [data-testid="stChatInput"] textarea { color: var(--ink) !important; }

    [data-testid="stExpander"] { border: none; background: transparent; }
    [data-testid="stExpander"] summary {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: var(--ink-soft);
    }
    .citation {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(26, 22, 19, 0.1);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
    }
    .citation:last-child { border-bottom: none; }
    .citation-tag { color: var(--pen); }
    .citation-score { color: var(--ink-soft); opacity: 0.85; }

    .quota-pill {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--muted);
        border: 1px solid var(--hairline);
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        margin-bottom: 1rem;
    }

    [data-testid="stAlert"] {
        background: var(--paper) !important;
        border-radius: 10px;
        border: 1px solid var(--hairline);
    }
    [data-testid="stAlert"] p { color: var(--ink) !important; }
    </style>

    <div class="hero">
        <p class="eyebrow">SportyBet Nigeria · Support assistant</p>
        <h1 class="hero-title">It only speaks<br>from the file.</h1>
        <p class="hero-sub">
            A retrieval-augmented assistant built on SportyBet Nigeria's public
            support pages — deposits, withdrawals, verification, responsible gaming.
            Every answer is checked against the source documents first.
            When they don't say, it says so too.
        </p>
        <div class="redline-card">
            <p class="redline-q">«Чи можу я скасувати self-exclusion достроково?»</p>
            <p class="redline-wrong"><s>Так, зверніться в підтримку і вам розблокують.</s></p>
            <p class="redline-right">Self-exclusion необоротний до кінця обраного
            періоду — акаунт не відновлюється достроково.</p>
        </div>
        <p class="hero-meta">
            EMBEDDINGS — NVIDIA NIM (nv-embedqa-e5-v5) &nbsp;·&nbsp;
            GENERATION — CLAUDE HAIKU 4.5 &nbsp;·&nbsp;
            RETRIEVAL — COSINE SIMILARITY, IN-MEMORY
        </p>
    </div>
    <hr class="divider">
    """,
    unsafe_allow_html=True,
)


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for source, score in sources:
            st.markdown(
                f'<div class="citation"><span class="citation-tag">{source}</span>'
                f'<span class="citation-score">sim {score:.2f}</span></div>',
                unsafe_allow_html=True,
            )


# ---------- API key handling ----------
default_nvidia_key = os.environ.get("NVIDIA_API_KEY", "") or _secret("NVIDIA_API_KEY")
default_anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "") or _secret("ANTHROPIC_API_KEY")
using_owner_keys = bool(default_nvidia_key and default_anthropic_key)

with st.sidebar:
    st.header("Configuration")
    if using_owner_keys:
        st.success("Using built-in demo API keys.")
        with st.expander("Use your own API keys instead"):
            own_nvidia_key = st.text_input("NVIDIA_API_KEY", value="", type="password")
            own_anthropic_key = st.text_input("ANTHROPIC_API_KEY", value="", type="password")
        nvidia_key = own_nvidia_key or default_nvidia_key
        anthropic_key = own_anthropic_key or default_anthropic_key
    else:
        own_nvidia_key = st.text_input("NVIDIA_API_KEY", value="", type="password")
        own_anthropic_key = st.text_input("ANTHROPIC_API_KEY", value="", type="password")
        nvidia_key = own_nvidia_key
        anthropic_key = own_anthropic_key
    docs_dir = st.text_input("Documents folder", value="docs")
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Documents in the folder are chunked and embedded (NVIDIA NIM).\n"
        "2. Your question is embedded and compared by cosine similarity.\n"
        f"3. The top {TOP_K} matching chunks are sent to Claude as context.\n"
        "4. Claude answers *only* from that context — or says it doesn't know."
    )

if not nvidia_key or not anthropic_key:
    st.info("Enter both API keys in the sidebar to start.")
    st.stop()

using_owner_keys_only = using_owner_keys and not (own_nvidia_key or own_anthropic_key)

# ---------- Build / cache the knowledge base ----------
if "kb" not in st.session_state or st.session_state.get("kb_docs_dir") != docs_dir:
    progress_bar = st.progress(0, text="Indexing documents...")

    def progress_cb(done, total):
        progress_bar.progress(done / total, text=f"Embedding chunks: {done}/{total}")

    try:
        kb = LocalKnowledgeBase.build(docs_dir, nvidia_key, progress_cb=progress_cb)
    except Exception as e:
        st.error(f"Failed to build knowledge base: {e}")
        st.stop()

    progress_bar.empty()
    st.session_state.kb = kb
    st.session_state.kb_docs_dir = docs_dir

    if not kb.chunks:
        st.warning(f"No .txt/.md files found in '{docs_dir}/'. Add some documents and reload.")
    else:
        st.success(f"Indexed {len(kb.chunks)} chunks from '{docs_dir}/'.")

kb = st.session_state.kb

# ---------- Chat interface ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if using_owner_keys_only:
    remaining = max(MAX_QUESTIONS_PER_SESSION - st.session_state.question_count, 0)
    st.markdown(
        f'<span class="quota-pill">Demo questions left this session: {remaining}</span>',
        unsafe_allow_html=True,
    )

if not st.session_state.messages:
    st.markdown('<p class="chip-label">Спробуй одне з цих</p>', unsafe_allow_html=True)
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, sample_q in zip(cols, SAMPLE_QUESTIONS):
        with col:
            if st.button(sample_q, key=f"chip_{sample_q}", use_container_width=True):
                st.session_state.pending_question = sample_q
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

question = st.chat_input("Ask a question about the knowledge base...")
if not question and st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    quota_exceeded = (
        using_owner_keys_only and st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION
    )
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if quota_exceeded:
            answer = (
                f"Demo limit reached ({MAX_QUESTIONS_PER_SESSION} questions per session). "
                "Reload the page, or add your own API keys in the sidebar to continue."
            )
            retrieved = []
            st.warning(answer)
        else:
            if using_owner_keys_only:
                st.session_state.question_count += 1
            with st.spinner("Retrieving and generating..."):
                try:
                    retrieved = kb.retrieve(question, nvidia_key, top_k=TOP_K)
                    answer = answer_question(question, retrieved, anthropic_key)
                except Exception as e:
                    answer = f"Error: {e}"
                    retrieved = []

            st.markdown(answer)
            if retrieved:
                render_sources([(c.source, float(s)) for c, s in retrieved])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [(c.source, float(s)) for c, s in retrieved] if retrieved else [],
        }
    )

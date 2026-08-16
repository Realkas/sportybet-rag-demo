# RAG Knowledge Base Demo — SportyBet Nigeria

A small, self-contained RAG (retrieval-augmented generation) chatbot with a web UI.
Built as a portfolio piece — same architecture pattern as the production RAG
Telegram bot (Supabase pgvector + NVIDIA NIM embeddings + Claude agent), but
with a local in-memory vector store instead of Supabase, so it runs anywhere
with zero infrastructure and doesn't touch any production data.

The knowledge base is compiled from SportyBet Nigeria's public Help/FAQ pages —
deposits, withdrawals, KYC verification, payout limits, and Responsible Gaming /
Self-Exclusion — deliberately including a sensitive topic (self-exclusion) to show
how the retrieval + guardrail pattern handles it without hallucinating or stalling.

## What it does

1. Reads every `.txt`/`.md` file in `docs/`, splits them into overlapping chunks.
2. Embeds each chunk with **NVIDIA NIM** (`nv-embedqa-e5-v5`).
3. On each question, embeds the question and retrieves the top-k most similar
   chunks by cosine similarity.
4. Sends those chunks to **Claude** (`claude-haiku-4-5`) as context, with a
   system prompt that forces it to answer only from what's retrieved — and to
   say "I don't know" / route to Live Chat rather than hallucinate account-specific
   details (balance, transaction status, verification outcome) the knowledge base
   can't cover.
5. Shows the retrieved sources (with similarity scores) alongside every answer,
   so retrieval quality is visible, not a black box.

## Setup

```bash
pip install -r requirements.txt

export NVIDIA_API_KEY=your_key_here      # free tier: https://build.nvidia.com
export ANTHROPIC_API_KEY=your_key_here   # https://console.anthropic.com

streamlit run app.py
```

Or copy `.env.example` to `.env` and fill in the keys instead of exporting
them. Or skip both and paste keys directly into the sidebar when the app opens.

## Try it

Ask things like:

- "Мінімальний і максимальний депозит?"
- "Як вивести кошти і скільки це триває?"
- "Як пройти верифікацію акаунта?"
- "Як працює self-exclusion?" — watch how it explains the mechanism (temporary,
  irreversible until the period ends, balance still withdrawable) without
  stalling or refusing on a Responsible Gaming question.
- "Скільки грошей на моєму балансі?" — a question the public docs can't answer,
  to see the honest "route to Live Chat" behavior instead of a guess.

Drop your own `.txt`/`.md` files into `docs/` to index different content —
the app re-indexes automatically when the folder path changes.

## Deploying publicly (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at this repo, `app.py` as the entrypoint.
3. In the app's **Settings → Secrets**, add:
   ```toml
   NVIDIA_API_KEY = "..."
   ANTHROPIC_API_KEY = "..."
   ```
   Keys live only in Streamlit's secrets store — never committed to the repo.
4. Visitors get a working demo with no key entry required; they can still
   paste their own keys in the sidebar to bypass the built-in demo cap.
5. A per-session cap (`MAX_QUESTIONS_PER_SESSION` in `app.py`, default 15)
   limits API cost from the shared demo keys.

## Why this exists

This is a from-scratch rebuild of the retrieval + generation pattern used in
an existing production RAG bot, packaged as a standalone web demo for
portfolio/interview purposes — this variant targets a support-bot use case in
a regulated, sensitive-content domain (betting/gaming):

- **Retrieval**: chunking, embeddings, cosine-similarity ranking — same NIM
  embedding provider as production, swapped from Supabase pgvector to an
  in-memory NumPy matrix so the demo needs no database.
- **Generation**: Claude with a strict "answer only from context" system
  prompt, tuned to route account-specific and sensitive (self-exclusion)
  questions to a human channel instead of guessing.
- **Interface**: Streamlit, chosen because it's the fastest way to put a
  working chat UI in front of retrieval logic and inspect what's actually
  being retrieved per question.

## Notes on scaling this up

- Swap `LocalKnowledgeBase` (in-memory NumPy) for Supabase pgvector or another
  vector DB once the document count grows past a few thousand chunks — the
  `retrieve()` interface stays the same.
- Chunking is a simple sliding window; a production version would likely
  chunk by semantic boundaries (headings, paragraphs) instead.
- No conversation memory yet — each question is independent. Multi-turn
  context would need the chat history added to each retrieval call.

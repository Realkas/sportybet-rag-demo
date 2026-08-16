"""
rag_core.py — RAG pipeline: chunking, NVIDIA NIM embeddings, cosine-similarity
retrieval, and Claude generation. No external vector DB — this is an in-memory
local demo standing in for the Supabase pgvector store used in production.

Requires two environment variables:
  NVIDIA_API_KEY   — from build.nvidia.com (free tier available)
  ANTHROPIC_API_KEY — from console.anthropic.com
"""

import os
import re
import json
import numpy as np
import requests
from pathlib import Path
from dataclasses import dataclass, field
from anthropic import Anthropic

NVIDIA_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
NVIDIA_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"  # 1024-dim NIM embedding model
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

CHUNK_SIZE = 600       # characters per chunk
CHUNK_OVERLAP = 100     # overlap between consecutive chunks
TOP_K = 4                # chunks retrieved per query


@dataclass
class Chunk:
    text: str
    source: str
    embedding: np.ndarray = field(default=None, repr=False)


def chunk_text(text: str, source: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Simple sliding-window chunker on whitespace-normalized text."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source))
        start += size - overlap
    return chunks


def load_documents(docs_dir: str) -> list[Chunk]:
    """Read every .txt/.md file in docs_dir and chunk it."""
    all_chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).glob("*")):
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        all_chunks.extend(chunk_text(text, source=path.name))
    return all_chunks


def embed_texts(texts: list[str], api_key: str, input_type: str = "passage") -> np.ndarray:
    """
    Call NVIDIA NIM embeddings endpoint. input_type is 'passage' for documents
    being indexed and 'query' for the user's question — the NIM QA embedding
    model is asymmetric and expects this distinction for best retrieval quality.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    payload = {
        "input": texts,
        "model": NVIDIA_EMBED_MODEL,
        "input_type": input_type,
        "encoding_format": "float",
    }
    resp = requests.post(NVIDIA_EMBED_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    vectors = [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix_norms @ query_norm


class LocalKnowledgeBase:
    """In-memory stand-in for the Supabase pgvector store."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    @classmethod
    def build(cls, docs_dir: str, nvidia_api_key: str, progress_cb=None) -> "LocalKnowledgeBase":
        chunks = load_documents(docs_dir)
        if not chunks:
            return cls([])
        # Embed in batches of 32 to stay within typical request limits
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vecs = embed_texts([c.text for c in batch], nvidia_api_key, input_type="passage")
            for c, v in zip(batch, vecs):
                c.embedding = v
            if progress_cb:
                progress_cb(min(i + batch_size, len(chunks)), len(chunks))
        return cls(chunks)

    def retrieve(self, query: str, nvidia_api_key: str, top_k: int = TOP_K) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_vec = embed_texts([query], nvidia_api_key, input_type="query")[0]
        matrix = np.stack([c.embedding for c in self.chunks])
        sims = cosine_similarity(query_vec, matrix)
        ranked = sorted(zip(self.chunks, sims), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]


SYSTEM_PROMPT = """You are a knowledge-base assistant. You answer ONLY using the \
provided context excerpts below. 

Rules:
- If the context contains the answer, respond clearly and cite which source file(s) you used.
- If the context does NOT contain enough information to answer, say so explicitly \
  — do not guess, do not use outside knowledge, do not make anything up.
- Keep answers concise and direct.
"""


def answer_question(question: str, retrieved: list[tuple[Chunk, float]], anthropic_api_key: str) -> str:
    client = Anthropic(api_key=anthropic_api_key)

    if not retrieved:
        context_block = "(no documents in knowledge base)"
    else:
        context_block = "\n\n".join(
            f"[Source: {c.source} | similarity: {score:.2f}]\n{c.text}"
            for c, score in retrieved
        )

    user_msg = f"Context excerpts:\n\n{context_block}\n\nQuestion: {question}"

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return response.content[0].text

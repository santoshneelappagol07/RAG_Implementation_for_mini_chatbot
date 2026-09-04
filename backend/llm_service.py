"""
llm_service.py
----------------
Connects directly to Gemini using the google.genai SDK (the official,
actively maintained Google Gen AI Python SDK).

Previously used the deprecated google.generativeai package.
The new google.genai SDK uses a Client-based pattern with
types.GenerateContentConfig for clean configuration.

Performance optimizations:
  1. max_output_tokens=256 to cap answer length
  2. temperature=0 for deterministic (faster) decoding
  3. System instruction enforces 2-3 sentence answers
"""

from google import genai
from google.genai import types

try:
    from backend.config import GEMINI_API_KEY, CHAT_MODEL
except ImportError:
    from config import GEMINI_API_KEY, CHAT_MODEL

# ── Create client once at import time ─────────────────────────────────────
_client = genai.Client(api_key=GEMINI_API_KEY)

# ── RAG config (used when FAISS chunks are available) ─────────────────────
_rag_config = types.GenerateContentConfig(
    temperature=0,
    max_output_tokens=512,
    system_instruction=(
        "Answer using ONLY the provided context excerpts. "
        "Cite page numbers in brackets like [Page 2]. "
        "If the answer is not in the context, say so. "
        "Be concise but complete — give a thorough answer without unnecessary filler."
    ),
)

# ── Fallback config (full text stuffing when no vector index exists) ──────
_fallback_config = types.GenerateContentConfig(
    temperature=0,
    max_output_tokens=512,
    system_instruction=(
        "Answer questions using ONLY the PDF content provided. "
        "If the answer isn't in the content, say you don't know. "
        "Be concise but complete — give a thorough answer without unnecessary filler."
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────

def format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Combines retrieved chunks and their page numbers into a structured context string.
    """
    if not chunks:
        return "No relevant context found in document."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "").strip()
        parts.append(f"[Page {page}]:\n{text}")

    return "\n\n".join(parts)


# ── RAG (chunks → Gemini → answer) ───────────────────────────────────────

def answer_question_with_rag(chunks: list[dict], question: str) -> str:
    """
    Sends only the top relevant chunks + question to Gemini and returns
    a cited answer. Uses the direct SDK for minimal overhead.
    """
    context = format_chunks_for_prompt(chunks)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    response = _client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=_rag_config,
    )
    return response.text


def stream_answer_with_rag(chunks: list[dict], question: str):
    """
    Streaming version of RAG answer — yields tokens as Gemini generates them.
    """
    context = format_chunks_for_prompt(chunks)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    for chunk in _client.models.generate_content_stream(
        model=CHAT_MODEL,
        contents=prompt,
        config=_rag_config,
    ):
        if chunk.text:
            yield chunk.text


# ── Fallback (full text → Gemini → answer) ────────────────────────────────

def answer_question_from_text(pdf_text: str, question: str) -> str:
    """
    Fallback: Sends the PDF's full text + question to Gemini.
    """
    prompt = f"PDF CONTENT:\n{pdf_text}\n\nQUESTION: {question}"
    response = _client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=_fallback_config,
    )
    return response.text


def stream_answer_from_text(pdf_text: str, question: str):
    """
    Fallback: Streaming version with full text.
    """
    prompt = f"PDF CONTENT:\n{pdf_text}\n\nQUESTION: {question}"
    for chunk in _client.models.generate_content_stream(
        model=CHAT_MODEL,
        contents=prompt,
        config=_fallback_config,
    ):
        if chunk.text:
            yield chunk.text

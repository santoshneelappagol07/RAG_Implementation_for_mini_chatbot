"""
llm_service.py
----------------
Connects to Gemini using the google.genai SDK (official Google Gen AI Python SDK).

Features:
  1. Async & Non-blocking generation via `_client.aio`
  2. Async Streaming SSE generation
  3. Gemini Context Caching (`client.aio.caches.create`) for large documents
  4. Deterministic low-latency configuration (temperature=0, max_output_tokens=512)
"""

import logging
from typing import Optional, List, Dict, Any, AsyncGenerator, Generator

from google import genai
from google.genai import types

try:
    from backend.config import (
        GEMINI_API_KEY, CHAT_MODEL,
        GEMINI_CONTEXT_CACHE_ENABLED, GEMINI_CONTEXT_CACHE_TTL,
    )
except ImportError:
    from config import (
        GEMINI_API_KEY, CHAT_MODEL,
        GEMINI_CONTEXT_CACHE_ENABLED, GEMINI_CONTEXT_CACHE_TTL,
    )

logger = logging.getLogger(__name__)

# ── Create client once at import time ─────────────────────────────────────
_client = genai.Client(api_key=GEMINI_API_KEY)

# ── Base configs ──────────────────────────────────────────────────────────
_rag_system_instruction = (
    "You are an expert AI assistant analyzing a PDF document. "
    "Answer questions accurately, thoroughly, and clearly using the provided context excerpts. "
    "Cite page numbers in brackets like [Page 2]. "
    "Use structured Markdown formatting (headers, bold text, bullet points, numbered lists) for clarity. "
    "When asked to list topics, explain concepts, or summarize, provide complete and comprehensive details."
)

_rag_config = types.GenerateContentConfig(
    temperature=0.2,
    max_output_tokens=2048,
    system_instruction=_rag_system_instruction,
    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
)

_fallback_system_instruction = (
    "You are an expert AI assistant analyzing a PDF document. "
    "Answer questions accurately, thoroughly, and clearly using the provided PDF content. "
    "Use structured Markdown formatting (headers, bold text, bullet points, numbered lists) for clarity. "
    "When asked to list topics, explain concepts, or summarize, provide a complete, well-organized, and comprehensive response."
)

_fallback_config = types.GenerateContentConfig(
    temperature=0.2,
    max_output_tokens=2048,
    system_instruction=_fallback_system_instruction,
    thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
)


# ── Helpers ───────────────────────────────────────────────────────────────

def format_chunks_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """
    Combines retrieved chunks and their page numbers into a structured context string.
    """
    if not chunks:
        return "No relevant context found in document."

    parts = []
    for chunk in chunks:
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "").strip()
        parts.append(f"[Page {page}]:\n{text}")

    return "\n\n".join(parts)


# ── 1. Async RAG & Full-Text Generation ───────────────────────────────────

async def answer_question_with_rag_async(chunks: List[Dict[str, Any]], question: str) -> str:
    """
    Asynchronously queries Gemini with top relevant chunks + question.
    """
    context = format_chunks_for_prompt(chunks)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    response = await _client.aio.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=_rag_config,
    )
    return response.text or ""


async def stream_answer_with_rag_async(
    chunks: List[Dict[str, Any]], question: str
) -> AsyncGenerator[str, None]:
    """
    Asynchronously streams RAG answer tokens from Gemini.
    """
    context = format_chunks_for_prompt(chunks)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    response_stream = await _client.aio.models.generate_content_stream(
        model=CHAT_MODEL,
        contents=prompt,
        config=_rag_config,
    )
    async for chunk in response_stream:
        if chunk.text:
            yield chunk.text


async def answer_question_from_text_async(pdf_text: str, question: str) -> str:
    """
    Fallback async: Queries Gemini with full document text.
    """
    prompt = f"PDF CONTENT:\n{pdf_text}\n\nQUESTION: {question}"
    response = await _client.aio.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=_fallback_config,
    )
    return response.text or ""


async def stream_answer_from_text_async(
    pdf_text: str, question: str
) -> AsyncGenerator[str, None]:
    """
    Fallback async streaming: Streams tokens for full document text.
    """
    prompt = f"PDF CONTENT:\n{pdf_text}\n\nQUESTION: {question}"
    response_stream = await _client.aio.models.generate_content_stream(
        model=CHAT_MODEL,
        contents=prompt,
        config=_fallback_config,
    )
    async for chunk in response_stream:
        if chunk.text:
            yield chunk.text


# ── 2. Gemini Context Caching ─────────────────────────────────────────────

async def create_gemini_context_cache_async(
    document_id: int,
    pdf_text: str,
    ttl_seconds: int = GEMINI_CONTEXT_CACHE_TTL,
) -> Optional[types.CachedContent]:
    """
    Creates a server-side Gemini Context Cache for large document text.
    Gracefully handles token limits if document is under the API minimum threshold.
    """
    if not GEMINI_CONTEXT_CACHE_ENABLED or not pdf_text:
        return None

    try:
        cache = await _client.aio.caches.create(
            model=CHAT_MODEL,
            config=types.CreateCachedContentConfig(
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=pdf_text)],
                    )
                ],
                ttl=f"{ttl_seconds}s",
                display_name=f"pdfchat_doc_{document_id}",
                system_instruction=_fallback_system_instruction,
            ),
        )
        logger.info("✅ Created Gemini Context Cache: %s for document %d", cache.name, document_id)
        return cache
    except Exception as exc:
        logger.info("ℹ️  Context Cache not created for document %d (likely < min tokens): %s", document_id, exc)
        return None


async def answer_with_context_cache_async(cache_name: str, question: str) -> str:
    """
    Asynchronously queries Gemini using a previously created server-side context cache.
    """
    config = types.GenerateContentConfig(
        cached_content=cache_name,
        temperature=0.2,
        max_output_tokens=2048,
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
    )
    response = await _client.aio.models.generate_content(
        model=CHAT_MODEL,
        contents=question,
        config=config,
    )
    return response.text or ""


async def stream_answer_with_context_cache_async(
    cache_name: str, question: str
) -> AsyncGenerator[str, None]:
    """
    Asynchronously streams tokens using a server-side context cache.
    """
    config = types.GenerateContentConfig(
        cached_content=cache_name,
        temperature=0.2,
        max_output_tokens=2048,
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
    )
    response_stream = await _client.aio.models.generate_content_stream(
        model=CHAT_MODEL,
        contents=question,
        config=config,
    )
    async for chunk in response_stream:
        if chunk.text:
            yield chunk.text


# ── 3. Synchronous Fallbacks (backward compatibility) ─────────────────────

def answer_question_with_rag(chunks: List[Dict[str, Any]], question: str) -> str:
    context = format_chunks_for_prompt(chunks)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    response = _client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=_rag_config,
    )
    return response.text or ""


def stream_answer_with_rag(chunks: List[Dict[str, Any]], question: str) -> Generator[str, None, None]:
    context = format_chunks_for_prompt(chunks)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    for chunk in _client.models.generate_content_stream(
        model=CHAT_MODEL,
        contents=prompt,
        config=_rag_config,
    ):
        if chunk.text:
            yield chunk.text


def answer_question_from_text(pdf_text: str, question: str) -> str:
    prompt = f"PDF CONTENT:\n{pdf_text}\n\nQUESTION: {question}"
    response = _client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=_fallback_config,
    )
    return response.text or ""


def stream_answer_from_text(pdf_text: str, question: str) -> Generator[str, None, None]:
    prompt = f"PDF CONTENT:\n{pdf_text}\n\nQUESTION: {question}"
    for chunk in _client.models.generate_content_stream(
        model=CHAT_MODEL,
        contents=prompt,
        config=_fallback_config,
    ):
        if chunk.text:
            yield chunk.text


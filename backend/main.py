"""
main.py
--------
FastAPI app with:
  1. Fully Non-Blocking Async Endpoints (`async def`)
  2. Multi-Tier Caching:
     - Tier 1: Exact Hash Cache (< 1ms)
     - Tier 2: Semantic Similarity Cache (Embedding Cosine Similarity, < 15ms)
  3. Gemini Context Caching for large documents
  4. FAISS Vector Search RAG for rapid, cited answers
  5. Async Streaming via Server-Sent Events (SSE)

Run with:  uvicorn backend.main:app --reload
Docs UI at: http://127.0.0.1:8000/docs
"""

import asyncio
import logging
import time
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

try:
    from backend.pdf_service import save_pdf_to_disk, extract_pages_from_pdf
    from backend.llm_service import (
        answer_question_with_rag_async,
        stream_answer_with_rag_async,
        answer_question_from_text_async,
        stream_answer_from_text_async,
        create_gemini_context_cache_async,
        answer_with_context_cache_async,
        stream_answer_with_context_cache_async,
    )
    from backend.vector_service import (
        create_vector_index,
        get_relevant_chunks,
        delete_vector_index,
        get_query_embedding_async,
    )
    from backend.database import Base, engine, get_db, init_db
    from backend.models import Document
    from backend.cache_service import (
        get_cached_answer,
        set_cached_answer,
        get_semantic_cached_answer,
        set_semantic_cached_answer,
        invalidate_document_cache,
        get_cache_stats,
    )
except ImportError:
    from pdf_service import save_pdf_to_disk, extract_pages_from_pdf
    from llm_service import (
        answer_question_with_rag_async,
        stream_answer_with_rag_async,
        answer_question_from_text_async,
        stream_answer_from_text_async,
        create_gemini_context_cache_async,
        answer_with_context_cache_async,
        stream_answer_with_context_cache_async,
    )
    from vector_service import (
        create_vector_index,
        get_relevant_chunks,
        delete_vector_index,
        get_query_embedding_async,
    )
    from database import Base, engine, get_db, init_db
    from models import Document
    from cache_service import (
        get_cached_answer,
        set_cached_answer,
        get_semantic_cached_answer,
        set_semantic_cached_answer,
        invalidate_document_cache,
        get_cache_stats,
    )

logger = logging.getLogger("pdfchat_api")

app = FastAPI(
    title="PDF Chatbot API (Async + Semantic Cache + Context Cache + FAISS)",
    version="2.0.0",
)

# Auto-create schema for models and self-heal missing columns if database is reachable
try:
    init_db()
except Exception as e:
    logger.warning("Database schema creation deferred (database connection pending): %s", e)



class AskRequest(BaseModel):
    document_id: int
    question: str


class AskResponse(BaseModel):
    document_id: int
    question: str
    answer: str
    source: str  # "redis_exact", "semantic_cache", "gemini_context_cache", "llm_rag", "llm_fallback"
    cached: bool = False
    sources: List[int] = []
    similarity_score: Optional[float] = None
    matched_question: Optional[str] = None


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Accepts a PDF upload, extracts text, indexes chunks in FAISS,
    and initializes Gemini Context Caching when applicable.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    file_bytes = await file.read()
    try:
        file_path = await asyncio.to_thread(save_pdf_to_disk, file.filename, file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save PDF to disk: {str(e)}")

    try:
        pages = await asyncio.to_thread(extract_pages_from_pdf, file_path)
        full_text = "\n\n".join(page["text"] for page in pages)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF text: {str(e)}")

    # 1. Store document metadata in database
    try:
        document = Document(
            filename=file.filename,
            file_path=file_path,
            extracted_text=full_text,
        )
        db.add(document)
        db.commit()
        db.refresh(document)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error saving document: {str(e)}")

    # 2. Build FAISS vector index in background thread
    try:
        chunks_count = await asyncio.to_thread(create_vector_index, document.id, pages)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings and build vector index: {str(e)}",
        )

    # 3. Attempt Gemini Context Caching (creates cache if document token count threshold met)
    context_cached = False
    cache_info = None
    try:
        cache_obj = await create_gemini_context_cache_async(document.id, full_text)
        if cache_obj:
            document.gemini_cache_name = cache_obj.name
            db.commit()
            context_cached = True
            cache_info = cache_obj.name
    except Exception as e:
        logger.warning("Gemini context caching initialization note: %s", e)

    return {
        "document_id": document.id,
        "filename": document.filename,
        "file_path": document.file_path,
        "pages_count": len(pages),
        "chunks_indexed": chunks_count,
        "characters_extracted": len(full_text),
        "gemini_context_cache_created": context_cached,
        "gemini_cache_name": cache_info,
        "message": "PDF uploaded, indexed with FAISS vector store, and ready for /ask.",
    }


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, db: Session = Depends(get_db)):
    """
    RAG & Multi-Tier Cached Question Answering:
    1. Tier 1: Check Exact Cache (< 1ms)
    2. Tier 2: Check Semantic Similarity Cache (< 15ms)
    3. Tier 3: Query Gemini via Context Cache or FAISS Vector RAG
    4. Store result in both exact and semantic cache
    """
    total_start = time.perf_counter()

    # ── 1. Check Exact Cache (Tier 1) ───────────────────────────────
    t0 = time.perf_counter()
    exact_cached = get_cached_answer(request.document_id, request.question)
    t_exact = time.perf_counter() - t0

    if exact_cached is not None:
        logger.info("⏱️  Exact Cache HIT: %.4fs", time.perf_counter() - total_start)
        return AskResponse(
            document_id=request.document_id,
            question=request.question,
            answer=exact_cached,
            source="redis_exact",
            cached=True,
            sources=[],
        )

    # ── 2. Compute Embedding & Check Semantic Cache (Tier 2) ────────
    t0 = time.perf_counter()
    query_embedding: Optional[List[float]] = None
    try:
        query_embedding = await get_query_embedding_async(request.question)
    except Exception as e:
        logger.warning("Could not generate query embedding: %s", e)

    if query_embedding:
        sem_answer, sem_score, sem_question = get_semantic_cached_answer(
            request.document_id, query_embedding,
        )
        if sem_answer is not None:
            t_sem = time.perf_counter() - t0
            logger.info("⏱️  Semantic Cache HIT (score=%.3f) in %.4fs", sem_score, t_sem)
            return AskResponse(
                document_id=request.document_id,
                question=request.question,
                answer=sem_answer,
                source="semantic_cache",
                cached=True,
                sources=[],
                similarity_score=round(sem_score, 4),
                matched_question=sem_question,
            )

    # ── 3. Database Lookup ──────────────────────────────────────────
    t0 = time.perf_counter()
    document = await asyncio.to_thread(
        lambda: db.query(Document).filter(Document.id == request.document_id).first()
    )
    t_db = time.perf_counter() - t0

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document with id {request.document_id}. Upload one first via /upload.",
        )

    # ── 4. Generation via Gemini Context Cache or FAISS RAG ─────────
    sources: List[int] = []
    source_type = "llm_rag"

    # Option A: Active Gemini Context Cache
    if document.gemini_cache_name:
        try:
            t0 = time.perf_counter()
            answer = await answer_with_context_cache_async(
                document.gemini_cache_name, request.question,
            )
            source_type = "gemini_context_cache"
            t_llm = time.perf_counter() - t0
        except Exception as e:
            logger.warning("Context cache query failed, falling back to RAG: %s", e)
            document.gemini_cache_name = None

    # Option B: FAISS RAG
    if not document.gemini_cache_name:
        t0 = time.perf_counter()
        chunks = await asyncio.to_thread(
            get_relevant_chunks, request.document_id, request.question
        )
        t_faiss = time.perf_counter() - t0
        sources = sorted(list({c["page_number"] for c in chunks if "page_number" in c}))

        t0 = time.perf_counter()
        try:
            if chunks:
                answer = await answer_question_with_rag_async(chunks, request.question)
                source_type = "llm_rag"
            else:
                answer = await answer_question_from_text_async(
                    document.extracted_text, request.question
                )
                source_type = "llm_fallback"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini LLM error: {str(e)}")
        t_llm = time.perf_counter() - t0

    # ── 5. Cache answer for future queries (Exact + Semantic) ───────
    set_cached_answer(request.document_id, request.question, answer)
    if query_embedding:
        set_semantic_cached_answer(
            request.document_id, request.question, query_embedding, answer,
        )

    t_total = time.perf_counter() - total_start
    logger.info("⏱️  TOTAL /ask: %.3fs (source=%s)", t_total, source_type)

    return AskResponse(
        document_id=document.id,
        question=request.question,
        answer=answer,
        source=source_type,
        cached=False,
        sources=sources,
    )


@app.post("/ask/stream")
async def ask_question_stream(request: AskRequest, db: Session = Depends(get_db)):
    """
    Async Streaming RAG version of /ask with Server-Sent Events (SSE).
    """
    # ── 1. Check Exact Cache ────────────────────────────────────────
    exact_cached = get_cached_answer(request.document_id, request.question)
    if exact_cached is not None:
        async def cached_generator():
            yield f"data: {exact_cached}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(cached_generator(), media_type="text/event-stream")

    # ── 2. Check Semantic Cache ─────────────────────────────────────
    query_embedding = None
    try:
        query_embedding = await get_query_embedding_async(request.question)
    except Exception:
        pass

    if query_embedding:
        sem_answer, sem_score, _ = get_semantic_cached_answer(
            request.document_id, query_embedding
        )
        if sem_answer is not None:
            async def sem_generator():
                yield f"data: {sem_answer}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sem_generator(), media_type="text/event-stream")

    # ── 3. Database Lookup ──────────────────────────────────────────
    document = await asyncio.to_thread(
        lambda: db.query(Document).filter(Document.id == request.document_id).first()
    )
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document with id {request.document_id}. Upload one first via /upload.",
        )

    # ── 4. Retrieve FAISS chunks if needed ──────────────────────────
    chunks = await asyncio.to_thread(
        get_relevant_chunks, request.document_id, request.question
    )

    # ── 5. Async SSE Stream Generator ───────────────────────────────
    async def event_generator():
        full_answer = []
        try:
            if document.gemini_cache_name:
                stream_fn = stream_answer_with_context_cache_async(
                    document.gemini_cache_name, request.question
                )
            elif chunks:
                stream_fn = stream_answer_with_rag_async(chunks, request.question)
            else:
                stream_fn = stream_answer_from_text_async(
                    document.extracted_text, request.question
                )

            async for token in stream_fn:
                full_answer.append(token)
                yield f"data: {token}\n\n"

            # Cache the complete answer upon stream completion
            if full_answer:
                complete_text = "".join(full_answer)
                set_cached_answer(request.document_id, request.question, complete_text)
                if query_embedding:
                    set_semantic_cached_answer(
                        request.document_id, request.question, query_embedding, complete_text
                    )

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Cache & Vector management endpoints ──────────────────────────────────

@app.get("/cache/stats")
def cache_stats():
    """
    Returns exact and semantic cache hits, misses, and hit rate.
    """
    return get_cache_stats()


@app.delete("/cache/{document_id}")
def clear_document_cache(document_id: int):
    """
    Deletes cached answers in Redis/Memory and frees FAISS vector index.
    """
    deleted = invalidate_document_cache(document_id)
    vector_cleared = delete_vector_index(document_id)
    return {
        "document_id": document_id,
        "keys_deleted": deleted,
        "vector_index_deleted": vector_cleared,
        "message": f"Cleared {deleted} cached answer(s) and FAISS index for document {document_id}.",
    }


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "PDF Chatbot API is running with Async, Semantic Caching, and Gemini Context Caching.",
    }


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
  6. Email OTP Authentication with session management

Run with:  uvicorn backend.main:app --reload
Docs UI at: http://127.0.0.1:8000/docs
"""

import asyncio
import logging
import os
import time
from typing import Optional, List, Union, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pymongo.database import Database
from pydantic import BaseModel, EmailStr, model_validator

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
    from backend.database import get_db, init_db
    from backend.models import Document
    from backend.cache_service import (
        get_cached_answer,
        set_cached_answer,
        get_semantic_cached_answer,
        set_semantic_cached_answer,
        invalidate_document_cache,
        get_cache_stats,
    )
    from backend.auth_service import (
        generate_otp, verify_otp, send_otp_email,
        create_session, validate_session, get_session_remaining,
        invalidate_session, get_current_user,
    )
    from backend.config import OTP_EXPIRY_SECONDS, SESSION_EXPIRY_SECONDS
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
    from database import get_db, init_db
    from models import Document
    from cache_service import (
        get_cached_answer,
        set_cached_answer,
        get_semantic_cached_answer,
        set_semantic_cached_answer,
        invalidate_document_cache,
        get_cache_stats,
    )
    from auth_service import (
        generate_otp, verify_otp, send_otp_email,
        create_session, validate_session, get_session_remaining,
        invalidate_session, get_current_user,
    )
    from config import OTP_EXPIRY_SECONDS, SESSION_EXPIRY_SECONDS

from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("pdfchat_api")

app = FastAPI(
    title="PDF Chatbot API (Async + Semantic Cache + Context Cache + FAISS)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to MongoDB Atlas and verify index creation
try:
    init_db()
except Exception as e:
    logger.warning("MongoDB Atlas initialization deferred: %s", e)



# ── Auth request/response models ─────────────────────────────────────────

class SendOtpRequest(BaseModel):
    email: EmailStr


class VerifyOtpRequest(BaseModel):
    email: EmailStr
    otp: str


# ── Existing request/response models ────────────────────────────────────

class AskRequest(BaseModel):
    document_id: Optional[Union[str, int]] = None
    id: Optional[Union[str, int]] = None
    question: str

    @model_validator(mode="before")
    @classmethod
    def populate_document_id(cls, values: Any) -> Any:
        if isinstance(values, dict):
            doc_id = values.get("document_id") or values.get("id")
            if not doc_id:
                raise ValueError("Either 'document_id' or 'id' must be provided.")
            values["document_id"] = doc_id
            values["id"] = doc_id
        return values


class AskResponse(BaseModel):
    document_id: Union[str, int]
    id: Optional[Union[str, int]] = None
    question: str
    answer: str
    source: str  # "redis_exact", "semantic_cache", "gemini_context_cache", "llm_rag", "llm_fallback"
    cached: bool = False
    sources: List[int] = []
    similarity_score: Optional[float] = None
    matched_question: Optional[str] = None

    @model_validator(mode="after")
    def populate_id(self) -> "AskResponse":
        if self.id is None:
            self.id = self.document_id
        return self


# ═══════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS (public — no token required)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/auth/send-otp", tags=["Authentication"])
async def send_otp(request: SendOtpRequest):
    """
    Send a 6-digit OTP to the provided email address.
    The OTP expires after 10 minutes.
    """
    otp = generate_otp(request.email)
    try:
        await asyncio.to_thread(send_otp_email, request.email, otp)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": f"OTP sent to {request.email}",
        "expires_in_seconds": OTP_EXPIRY_SECONDS,
    }


@app.post("/auth/verify-otp", tags=["Authentication"])
async def verify_otp_endpoint(request: VerifyOtpRequest):
    """
    Verify the OTP and receive a session token.
    Use this token in the Authorization header: Bearer <token>
    """
    if not verify_otp(request.email, request.otp):
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired OTP. Please request a new one via /auth/send-otp.",
        )

    token = create_session(request.email)
    return {
        "message": "Login successful",
        "email": request.email,
        "token": token,
        "token_type": "bearer",
        "session_expires_in_seconds": SESSION_EXPIRY_SECONDS,
    }


@app.get("/auth/me", tags=["Authentication"])
async def auth_me(current_user: str = Depends(get_current_user)):
    """
    Returns the currently authenticated user info and session time remaining.
    Requires a valid Bearer token.
    """
    return {
        "email": current_user,
        "authenticated": True,
    }


@app.post("/auth/logout", tags=["Authentication"])
async def logout(
    current_user: str = Depends(get_current_user),
):
    """
    Invalidate the current session token (logout).
    """
    # Find and remove all sessions for this user
    try:
        from backend.auth_service import _session_store
    except ImportError:
        from auth_service import _session_store
    tokens_to_remove = [
        token for token, rec in _session_store.items()
        if rec["email"] == current_user
    ]
    for token in tokens_to_remove:
        invalidate_session(token)

    return {
        "message": f"Logged out {current_user}",
        "email": current_user,
        "sessions_invalidated": len(tokens_to_remove),
    }


# ═══════════════════════════════════════════════════════════════════════
# PROTECTED ENDPOINTS (Bearer token required)
# ═══════════════════════════════════════════════════════════════════════

@app.post("/upload", tags=["Documents"])
async def upload_pdf(
    file: UploadFile = File(...),
    db: Database = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
    """
    Accepts a PDF upload, extracts text, indexes chunks in FAISS,
    stores metadata in MongoDB Atlas, and initializes Gemini Context Caching when applicable.
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

    # 1. Store document metadata in MongoDB Atlas
    try:
        document = Document(
            filename=file.filename,
            file_path=file_path,
            extracted_text=full_text,
        )
        await asyncio.to_thread(Document.insert, db, document)
    except Exception as e:
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
            await asyncio.to_thread(document.update_cache, db, cache_obj.name)
            context_cached = True
            cache_info = cache_obj.name
    except Exception as e:
        logger.warning("Gemini context caching initialization note: %s", e)

    return {
        "id": document.id,
        "document_id": document.id,
        "filename": document.filename,
        "file_path": document.file_path,
        "pages_count": len(pages),
        "chunks_indexed": chunks_count,
        "characters_extracted": len(full_text),
        "gemini_context_cache_created": context_cached,
        "gemini_cache_name": cache_info,
        "message": "PDF uploaded, stored in MongoDB Atlas, indexed with FAISS vector store, and ready for /ask.",
    }


@app.post("/ask", response_model=AskResponse, tags=["Documents"])
async def ask_question(
    request: AskRequest,
    db: Database = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
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
        Document.find_by_id, db, request.document_id
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
            await asyncio.to_thread(document.update_cache, db, None)

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


@app.post("/ask/stream", tags=["Documents"])
async def ask_question_stream(
    request: AskRequest,
    db: Database = Depends(get_db),
    current_user: str = Depends(get_current_user),
):
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
        Document.find_by_id, db, request.document_id
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

@app.get("/cache/stats", tags=["Cache"])
def cache_stats(current_user: str = Depends(get_current_user)):
    """
    Returns exact and semantic cache hits, misses, and hit rate.
    """
    return get_cache_stats()


@app.delete("/cache/{document_id}", tags=["Cache"])
def clear_document_cache(document_id: str, current_user: str = Depends(get_current_user)):
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


@app.get("/api/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "message": "PDF Chatbot API is running with Auth, Async, Semantic Caching, and Gemini Context Caching.",
        "auth_info": "Send OTP via POST /auth/send-otp to get started.",
    }


# ── Serve React Frontend Static Files ──────────────────────────────────
# The built React app lives in frontend/dist/ after `npm run build`.
# In Docker, it's copied into the container.
_frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")

if os.path.isdir(_frontend_dist):
    # Serve static assets (JS, CSS, images) under /assets/
    app.mount("/assets", StaticFiles(directory=os.path.join(_frontend_dist, "assets")), name="static-assets")

    @app.get("/")
    @app.get("/{catch_all:path}")
    async def serve_frontend(catch_all: str = ""):
        """Serve React SPA — all non-API routes get index.html."""
        index_file = os.path.join(_frontend_dist, "index.html")
        if os.path.isfile(index_file):
            return FileResponse(index_file)
        return {"status": "ok", "message": "Frontend not built yet. Run: cd frontend && npm run build"}
else:
    @app.get("/", tags=["Health"])
    async def root():
        return {
            "status": "ok",
            "message": "PDF Chatbot API running. Frontend dist/ not found — build with: cd frontend && npm run build",
        }


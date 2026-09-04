"""
main.py
--------
The FastAPI app: two endpoints.

  POST /upload  -> accepts a PDF file, saves it to disk, extracts its text,
                   and INSERTs a row into MySQL (filename, file_path, extracted_text)
  POST /ask     -> accepts {document_id, question}, SELECTs the stored text
                   from MySQL by id, sends it + the question to Gemini, returns the answer

Run with:  uvicorn app.main:app --reload
Docs UI at: http://127.0.0.1:8000/docs
"""

from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

try:
    from backend.pdf_service import save_pdf_to_disk, extract_text_from_pdf, extract_pages_from_pdf
    from backend.llm_service import (
        answer_question_with_rag, stream_answer_with_rag,
        answer_question_from_text, stream_answer_from_text,
    )
    from backend.vector_service import (
        create_vector_index, get_relevant_chunks, delete_vector_index,
    )
    from backend.database import Base, engine, get_db
    from backend.models import Document
    from backend.cache_service import (
        get_cached_answer, set_cached_answer,
        invalidate_document_cache, get_cache_stats,
    )
except ImportError:
    from pdf_service import save_pdf_to_disk, extract_text_from_pdf, extract_pages_from_pdf
    from llm_service import (
        answer_question_with_rag, stream_answer_with_rag,
        answer_question_from_text, stream_answer_from_text,
    )
    from vector_service import (
        create_vector_index, get_relevant_chunks, delete_vector_index,
    )
    from database import Base, engine, get_db
    from models import Document
    from cache_service import (
        get_cached_answer, set_cached_answer,
        invalidate_document_cache, get_cache_stats,
    )

app = FastAPI(title="PDF Chatbot with RAG (Gemini + FAISS + MySQL + Redis)")

# Creates the `documents` table if it doesn't exist yet, based on the
# Document model. Fine for learning/dev.
Base.metadata.create_all(bind=engine)


class AskRequest(BaseModel):
    document_id: int
    question: str


class AskResponse(BaseModel):
    document_id: int
    question: str
    answer: str
    source: str  # "redis" or "llm_rag"
    cached: bool = False
    sources: List[int] = []


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Accepts a PDF upload, extracts text page-by-page, saves PDF to disk,
    stores record in MySQL, and indexes chunks into FAISS with Gemini embeddings.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    file_bytes = await file.read()
    try:
        file_path = save_pdf_to_disk(file.filename, file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save PDF to disk: {str(e)}")

    try:
        pages = extract_pages_from_pdf(file_path)
        full_text = "\n\n".join(page["text"] for page in pages)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF text: {str(e)}")

    # 1. Store document metadata + extracted text in MySQL
    try:
        document = Document(filename=file.filename, file_path=file_path, extracted_text=full_text)
        db.add(document)
        db.commit()
        db.refresh(document)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error saving document: {str(e)}")

    # 2. Build FAISS vector index (Page-Aware Recursive Chunking + text-embedding-004)
    try:
        chunks_count = create_vector_index(document.id, pages)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embeddings and build vector index: {str(e)}",
        )

    return {
        "document_id": document.id,
        "filename": document.filename,
        "file_path": document.file_path,
        "pages_count": len(pages),
        "chunks_indexed": chunks_count,
        "characters_extracted": len(full_text),
        "message": "PDF uploaded, chunked, embedded, and indexed with FAISS & MySQL. Ready for /ask.",
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest, db: Session = Depends(get_db)):
    """
    RAG-powered question answering with performance timing:
    1. Checks Redis cache for instant answer (< 5ms).
    2. Searches FAISS for top relevant chunks with page metadata (< 2ms).
    3. Prompts Gemini with only the relevant excerpts for rapid, cited answers.
    4. Caches answer in Redis for future requests.
    """
    import time
    import logging
    logger = logging.getLogger("ask_timing")
    total_start = time.perf_counter()

    # ── 1. Check cache first ────────────────────────────────────────
    t0 = time.perf_counter()
    cached_answer = get_cached_answer(request.document_id, request.question)
    t_cache = time.perf_counter() - t0
    logger.info("⏱️  Cache check: %.3fs", t_cache)

    if cached_answer is not None:
        logger.info("⏱️  TOTAL (cache hit): %.3fs", time.perf_counter() - total_start)
        return AskResponse(
            document_id=request.document_id,
            question=request.question,
            answer=cached_answer,
            source="redis",
            cached=True,
            sources=[],
        )

    # ── 2. Cache miss → look up document in MySQL ───────────────────
    t0 = time.perf_counter()
    try:
        document = db.query(Document).filter(Document.id == request.document_id).first()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")
    t_db = time.perf_counter() - t0
    logger.info("⏱️  DB lookup: %.3fs", t_db)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document with id {request.document_id}. Upload one first via /upload.",
        )

    # ── 3. Retrieve relevant chunks from FAISS vector store ────────
    t0 = time.perf_counter()
    chunks = get_relevant_chunks(request.document_id, request.question)
    t_faiss = time.perf_counter() - t0
    logger.info("⏱️  FAISS retrieval (%d chunks): %.3fs", len(chunks), t_faiss)
    sources = sorted(list({c["page_number"] for c in chunks if "page_number" in c}))

    # ── 4. Call Gemini via RAG (or fallback to full text if no index) ──
    t0 = time.perf_counter()
    try:
        if chunks:
            answer = answer_question_with_rag(chunks, request.question)
            source_type = "llm_rag"
        else:
            answer = answer_question_from_text(document.extracted_text, request.question)
            source_type = "llm_fallback"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini LLM error: {str(e)}")
    t_llm = time.perf_counter() - t0
    logger.info("⏱️  LLM generation: %.3fs", t_llm)

    # ── 5. Store in cache for next time ─────────────────────────────
    t0 = time.perf_counter()
    set_cached_answer(request.document_id, request.question, answer)
    t_cache_set = time.perf_counter() - t0
    logger.info("⏱️  Cache store: %.3fs", t_cache_set)

    t_total = time.perf_counter() - total_start
    logger.info(
        "⏱️  TOTAL: %.3fs | cache=%.3fs db=%.3fs faiss=%.3fs llm=%.3fs cache_set=%.3fs",
        t_total, t_cache, t_db, t_faiss, t_llm, t_cache_set,
    )

    return AskResponse(
        document_id=document.id,
        question=request.question,
        answer=answer,
        source=source_type,
        cached=False,
        sources=sources,
    )


@app.post("/ask/stream")
def ask_question_stream(request: AskRequest, db: Session = Depends(get_db)):
    """
    Streaming RAG version of /ask. Retrieves top chunks and streams tokens
    via Server-Sent Events (SSE) for near-instant perceived response time.
    """
    # ── 1. Check cache first ────────────────────────────────────────
    cached_answer = get_cached_answer(request.document_id, request.question)
    if cached_answer is not None:
        def cached_event_generator():
            yield f"data: {cached_answer}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(cached_event_generator(), media_type="text/event-stream")

    # ── 2. Cache miss → look up document in MySQL ───────────────────
    try:
        document = db.query(Document).filter(Document.id == request.document_id).first()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document with id {request.document_id}. Upload one first via /upload.",
        )

    # ── 3. Retrieve relevant chunks from FAISS ─────────────────────
    chunks = get_relevant_chunks(request.document_id, request.question)

    # ── 4. Stream from Gemini and cache full answer afterward ───────
    def event_generator():
        full_answer = []
        try:
            stream_fn = stream_answer_with_rag(chunks, request.question) if chunks else stream_answer_from_text(document.extracted_text, request.question)
            for token in stream_fn:
                full_answer.append(token)
                yield f"data: {token}\n\n"

            # Cache the complete answer once streaming finishes
            if full_answer:
                set_cached_answer(
                    request.document_id, request.question, "".join(full_answer),
                )
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Cache & Vector management endpoints ──────────────────────────────────

@app.get("/cache/stats")
def cache_stats():
    """
    Returns Redis cache hit/miss counters and connectivity status.
    Useful for monitoring whether caching is working.
    """
    return get_cache_stats()


@app.delete("/cache/{document_id}")
def clear_document_cache(document_id: int):
    """
    Deletes all cached answers in Redis and frees FAISS vector index from RAM/disk.
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
    return {"status": "ok", "message": "PDF Chatbot API is running. See /docs for the interactive UI."}

# Implementation Plan: Full RAG Pipeline with Page-Aware Chunking, Gemini Embeddings, and FAISS Vector Search

Transform the current "prompt-stuffing" architecture into a high-performance **Retrieval-Augmented Generation (RAG)** pipeline. This plan incorporates **Page-Aware Recursive Character Chunking**, **Google Gemini Embeddings (`text-embedding-004`)**, and **In-Memory FAISS (CPU)** vector search for sub-second responses and page-level source citations.

---

## Architecture Overview

```
[Upload Phase]
PDF File ──> Page-by-Page Extraction (pypdf) 
         ──> RecursiveCharacterTextSplitter (chunk_size=800, overlap=120) with page metadata
         ──> Gemini Embeddings (text-embedding-004)
         ──> FAISS Vector Index (persisted to storage/vectors/{doc_id}/)
         ──> Document record in MySQL

[Query Phase (/ask & /ask/stream)]
User Question ──> Check Redis Cache (instant return if hit)
              ──> Cache Miss: Load FAISS Index for document_id (cached in memory)
              ──> Vector Similarity Search (Top-K = 3 chunks) with page numbers
              ──> Augmented Prompt (Context + Sources + Question)
              ──> Gemini Flash (Streaming or Non-streaming)
              ──> Save answer to Redis Cache
```

---

## User Review Required

> [!IMPORTANT]
> **New Dependencies Required**:
> We need to add `faiss-cpu`, `langchain-text-splitters`, and `langchain-community` to [requirements.txt](file:///c:/Users/santosh/OneDrive/Desktop/Fastapi/projects/pdf-chatbot-gemini/requirements.txt) and install them via `pip install faiss-cpu langchain-text-splitters langchain-community`.

> [!NOTE]
> **Backward Compatibility**:
> MySQL will continue to store the full extracted text in `documents` table as a fallback and record of truth, while FAISS handles vector similarity retrieval. Existing `/ask` and `/ask/stream` contracts remain intact while returning faster answers with page source citations.

---

## Proposed Changes

### 1. Dependencies & Configuration

#### [MODIFY] [requirements.txt](file:///c:/Users/santosh/OneDrive/Desktop/Fastapi/projects/pdf-chatbot-gemini/requirements.txt)
- Add:
  - `faiss-cpu` (fast local CPU vector database)
  - `langchain-text-splitters` (for `RecursiveCharacterTextSplitter`)
  - `langchain-community` (for LangChain FAISS integration)

#### [MODIFY] [config.py](file:///c:/Users/santosh/OneDrive/Desktop/Fastapi/projects/pdf-chatbot-gemini/backend/config.py)
- Define `EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")`
- Define `VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "vectors")`
- Define `TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "3"))`
- Define `CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))`
- Define `CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))`

---

### 2. Services & PDF Processing

#### [MODIFY] [pdf_service.py](file:///c:/Users/santosh/OneDrive/Desktop/Fastapi/projects/pdf-chatbot-gemini/backend/pdf_service.py)
- Add `extract_pages_from_pdf(file_path: str) -> list[dict]`:
  - Returns structured pages: `[{"page_number": 1, "text": "..."}, {"page_number": 2, "text": "..."}]`
  - Keep `extract_text_from_pdf` for backwards-compatibility and database storage.

#### [NEW] [vector_service.py](file:///c:/Users/santosh/OneDrive/Desktop/Fastapi/projects/pdf-chatbot-gemini/backend/vector_service.py)
- Responsible for all Chunking, Embedding, and FAISS indexing:
  - `create_vector_index(document_id: int, pages: list[dict]) -> int`:
    - Takes extracted pages with page metadata.
    - Applies `RecursiveCharacterTextSplitter` per page (preserves page numbers in metadata).
    - Computes embeddings via `GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)`.
    - Builds a FAISS index and persists it to `storage/vectors/{document_id}/`.
  - `get_relevant_chunks(document_id: int, question: str, k: int = 3) -> list[dict]`:
    - Loads or retrieves the cached FAISS index from disk/RAM.
    - Executes `similarity_search_with_score(question, k=k)`.
    - Returns text and metadata (including `page_number`).
  - `delete_vector_index(document_id: int)`: Cleans up vector files on document deletion.

---

### 3. LLM & Prompt Integration

#### [MODIFY] [llm_service.py](file:///c:/Users/santosh/OneDrive/Desktop/Fastapi/projects/pdf-chatbot-gemini/backend/llm_service.py)
- Refactor prompt template to support RAG with source citations:
  ```
  SYSTEM: You are a helpful assistant that answers questions using ONLY the retrieved context from the PDF.
  Each chunk is tagged with its page number.
  When answering, mention the page number(s) where you found the information (e.g., [Page X]).
  If the answer cannot be found in the context, clearly state that you don't know.
  
  CONTEXT:
  {context}
  ```
- Add:
  - `format_chunks_for_prompt(chunks: list[dict]) -> str`
  - `answer_question_with_rag(chunks: list[dict], question: str) -> str`
  - `stream_answer_with_rag(chunks: list[dict], question: str)`

---

### 4. API Endpoints

#### [MODIFY] [main.py](file:///c:/Users/santosh/OneDrive/Desktop/Fastapi/projects/pdf-chatbot-gemini/backend/main.py)
- Update `/upload`:
  - Extracts pages via `extract_pages_from_pdf`.
  - Builds and saves FAISS vector index via `create_vector_index(document.id, pages)`.
  - Returns `chunks_created` in the upload response.
- Update `/ask`:
  - Checks Redis cache.
  - On miss: retrieves top 3 chunks using `get_relevant_chunks(request.document_id, request.question)`.
  - Invokes `answer_question_with_rag(chunks, request.question)`.
  - Caches and returns answer.
- Update `/ask/stream`:
  - Streams response tokens from `stream_answer_with_rag`.
- Update models to include optional `sources: list[int]` (e.g., list of cited page numbers).

---

## Verification Plan

### Automated Tests & Sanity Checks
1. **Dependency Installation**:
   - `pip install faiss-cpu langchain-text-splitters langchain-community`
   - Run Python verification script testing FAISS + Gemini Embeddings imports.
2. **Chunking & Index Creation Test**:
   - Create a test script in `scratch/test_rag.py` to create chunks from a sample PDF, generate embeddings, and build the FAISS index.
3. **Retrieval Verification**:
   - Query FAISS with a test question and verify that top relevant chunks with correct `page_number` are retrieved in $< 5\text{ms}$.
4. **End-to-End API Test**:
   - Run FastAPI server test with a sample PDF upload.
   - Test `/upload`, `/ask`, and `/ask/stream`.
   - Verify Redis caching works seamlessly with RAG responses.

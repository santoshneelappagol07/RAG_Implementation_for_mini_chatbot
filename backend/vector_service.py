"""
vector_service.py
-----------------
Handles the RAG vector search lifecycle:
1. Page-Aware Recursive Character Chunking
2. Embedding generation via Google Gemini (text-embedding-004)
3. In-memory & serialized vector storage via FAISS (CPU)
4. Fast Top-K similarity retrieval with page metadata
"""

import os
import shutil
from typing import Optional, List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores.faiss import FAISS  # pyright: ignore[reportMissingImports]  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

try:
    from backend.config import (
        GEMINI_API_KEY,
        EMBEDDING_MODEL,
        VECTOR_STORE_DIR,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        TOP_K_CHUNKS,
    )
except ImportError:
    from config import (
        GEMINI_API_KEY,
        EMBEDDING_MODEL,
        VECTOR_STORE_DIR,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        TOP_K_CHUNKS,
    )

# In-memory cache to keep active FAISS indexes in RAM for zero disk-load latency
_vector_store_cache: Dict[int, FAISS] = {}

# Reusable embedding instance
_embeddings_instance: Optional[GoogleGenerativeAIEmbeddings] = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Returns a singleton instance of Google's Gemini embeddings client.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY,
        )
    return _embeddings_instance


def get_query_embedding(text: str) -> List[float]:
    """Generates embedding vector for a single query string."""
    embeddings = get_embeddings()
    return embeddings.embed_query(text)


async def get_query_embedding_async(text: str) -> List[float]:
    """Asynchronously generates embedding vector for a query string."""
    embeddings = get_embeddings()
    return await embeddings.aembed_query(text)


def _get_doc_vector_dir(document_id: int) -> str:

    return os.path.join(VECTOR_STORE_DIR, str(document_id))


def create_vector_index(document_id: int, pages: List[Dict[str, Any]]) -> int:
    """
    Splits page text using Recursive Character Chunking, attaches page metadata,
    computes Gemini embeddings, and builds a FAISS CPU index saved to disk & RAM.

    Returns the total number of chunks indexed.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: List[LCDocument] = []
    for page in pages:
        page_num = page.get("page_number", 1)
        raw_text = page.get("text", "")
        if not raw_text.strip():
            continue

        chunks = splitter.split_text(raw_text)
        for idx, chunk in enumerate(chunks):
            documents.append(
                LCDocument(
                    page_content=chunk,
                    metadata={
                        "document_id": document_id,
                        "page_number": page_num,
                        "chunk_index": idx,
                    },
                )
            )

    if not documents:
        raise ValueError("Cannot build vector index: no text chunks were generated.")

    embeddings = get_embeddings()
    batch_size = 20
    vector_store = None

    import time
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        for attempt in range(4):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(documents=batch, embedding=embeddings)
                else:
                    vector_store.add_documents(batch)
                break
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 3:
                    wait_sec = 20 * (attempt + 1)
                    time.sleep(wait_sec)
                else:
                    raise
        if i + batch_size < len(documents):
            time.sleep(0.5)

    # Persist index to storage/vectors/{document_id}/
    target_dir = _get_doc_vector_dir(document_id)
    os.makedirs(target_dir, exist_ok=True)
    vector_store.save_local(target_dir)

    # Keep in RAM cache for subsequent queries
    _vector_store_cache[document_id] = vector_store

    return len(documents)


def load_vector_index(document_id: int) -> Optional[FAISS]:
    """
    Loads FAISS index for document_id from RAM cache, or loads from disk if present.
    Returns None if no index exists yet for this document.
    """
    if document_id in _vector_store_cache:
        return _vector_store_cache[document_id]

    target_dir = _get_doc_vector_dir(document_id)
    index_file = os.path.join(target_dir, "index.faiss")
    if not os.path.exists(index_file):
        return None

    embeddings = get_embeddings()
    # allow_dangerous_deserialization is safe here because these are our own local files
    vector_store = FAISS.load_local(
        target_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    _vector_store_cache[document_id] = vector_store
    return vector_store


def get_relevant_chunks(
    document_id: int, question: str, k: int = TOP_K_CHUNKS
) -> List[Dict[str, Any]]:
    """
    Performs similarity search in FAISS on the CPU and returns top K chunks
    along with page metadata and similarity scores.
    """
    vector_store = load_vector_index(document_id)
    if vector_store is None:
        return []

    # similarity_search_with_score returns List[Tuple[Document, float]]
    results = vector_store.similarity_search_with_score(question, k=k)

    chunks = []
    for doc, score in results:
        chunks.append({
            "text": doc.page_content,
            "page_number": doc.metadata.get("page_number", 1),
            "score": float(score),
        })

    return chunks


def delete_vector_index(document_id: int) -> bool:
    """
    Removes the FAISS index from RAM and disk when a document is deleted.
    """
    _vector_store_cache.pop(document_id, None)
    target_dir = _get_doc_vector_dir(document_id)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        return True
    return False

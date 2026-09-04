"""
pdf_service.py
---------------
Handles everything about the PDF file itself: saving it to disk and
reading text out of it. No LangChain or Gemini logic lives here on purpose —
this file only knows about "PDF in, text out". That separation makes it easy
to swap local disk storage for MySQL later without touching the RAG logic.
"""

import os
from pypdf import PdfReader

try:
    from backend.config import STORAGE_DIR
except ImportError:
    from config import STORAGE_DIR


def save_pdf_to_disk(filename: str, file_bytes: bytes) -> str:
    """
    Writes the uploaded PDF's raw bytes to STORAGE_DIR.

    Later, when you move to MySQL, this function is what you'd change:
    instead of writing to disk, you'd INSERT a row (filename, file_bytes or
    file_path, uploaded_at) into a `documents` table. Nothing outside this
    function needs to know that changed.
    """
    file_path = os.path.join(STORAGE_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return file_path


def extract_pages_from_pdf(file_path: str) -> list[dict]:
    """
    Reads a PDF from disk and returns a list of dictionaries per page:
    [{"page_number": 1, "text": "page 1 text..."}, ...]

    This powers Page-Aware Chunking: chunks created from each page will retain
    their exact page number metadata so the LLM and UI can cite sources.
    """
    reader = PdfReader(file_path)
    pages = []
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            pages.append({
                "page_number": idx + 1,
                "text": page_text.strip(),
            })

    if not pages:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned/image-only PDF, which needs OCR (not covered here)."
        )

    return pages


def extract_text_from_pdf(file_path: str) -> str:
    """
    Reads a PDF from disk and returns all its text as one big string.
    Kept for MySQL extracted_text storage and backward compatibility.
    """
    pages = extract_pages_from_pdf(file_path)
    return "\n\n".join(page["text"] for page in pages)

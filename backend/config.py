"""
config.py
---------
Single place that reads environment variables and defines shared constants.
Keeping this separate means every other file just does `from app.config import X`
instead of scattering os.getenv() calls everywhere.
"""

import os
from dotenv import load_dotenv

# Loads variables from local .env files into the process environment
_current_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_current_dir)
load_dotenv(os.path.join(_root_dir, ".env"), override=True)
load_dotenv(os.path.join(_current_dir, ".env"), override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
    )

# The actual PDF file bytes still live on disk — only metadata + extracted
# text move into MongoDB Atlas. This keeps the database lean.
STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "pdfs")
os.makedirs(STORAGE_DIR, exist_ok=True)

# The Gemini chat model. Centralized here so you can swap models in one
# place later.
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# --- RAG & Vector Store configuration ---
# Embedding model used to convert text chunks into dense semantic vectors.
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# Directory where FAISS vector indexes are serialized per document:
# storage/vectors/{document_id}/index.faiss and index.pkl
VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "vectors")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Chunking parameters (RecursiveCharacterTextSplitter)
# Rich context per chunk while keeping prompt concise
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Number of relevant chunks retrieved per user question
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "4"))

# --- MongoDB Atlas connection settings ---
# Connection string (URI) for MongoDB Atlas (e.g. mongodb+srv://user:pass@cluster0.mongodb.net/?retryWrites=true&w=majority)
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    os.getenv("MONGO_URI", "mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority"),
)
MONGODB_DB = os.getenv("MONGODB_DB", os.getenv("MONGO_DB", "pdf_chatbot"))

# --- Redis caching settings ---
# Used by cache_service.py to cache LLM answers and avoid repeated
# Gemini API calls for the same (document_id, question) pair.
# Supports local Redis or cloud instances (Azure Cache for Redis, Upstash, AWS ElastiCache).
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None   # None = no auth
REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() in ("true", "1", "yes")
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))  # seconds

# --- Semantic Caching settings ---
# When enabled, questions with high semantic similarity to a previously
# answered question for the same document are served from cache.
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.90"))  # cosine similarity

# --- Gemini Context Caching settings ---
# Used to cache large document contexts directly on Google servers (TTL in seconds).
GEMINI_CONTEXT_CACHE_ENABLED = os.getenv("GEMINI_CONTEXT_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
GEMINI_CONTEXT_CACHE_TTL = int(os.getenv("GEMINI_CONTEXT_CACHE_TTL", "3600"))

# --- SMTP / OTP Authentication settings ---
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USERNAME)
OTP_EXPIRY_SECONDS = int(os.getenv("OTP_EXPIRY_SECONDS", "600"))       # 10 minutes (600s)
SESSION_EXPIRY_SECONDS = int(
    os.getenv("SESSION_EXPIRY_SECONDS", str(int(os.getenv("SESSION_EXPIRY_MINUTES", "60")) * 60))
)  # 1 hour (3600s)


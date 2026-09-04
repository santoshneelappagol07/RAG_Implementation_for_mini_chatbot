"""
config.py
---------
Single place that reads environment variables and defines shared constants.
Keeping this separate means every other file just does `from app.config import X`
instead of scattering os.getenv() calls everywhere.
"""

import os
from dotenv import load_dotenv
from sqlalchemy.engine import URL

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
# text move into MySQL. This keeps the database lean.
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

# --- MySQL connection settings ---
# Each piece is its own env var so you can change host/user/password
# independently (e.g. localhost while developing, a cloud DB in production)
# without touching code.
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "pdf_chatbot")

# SQLAlchemy connection string (DSN): dialect+driver://user:password@host:port/dbname
# "mysql+pymysql" tells SQLAlchemy: talk MySQL's protocol, using the pymysql driver.
DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=MYSQL_USER,
    password=MYSQL_PASSWORD,
    host=MYSQL_HOST,
    port=int(MYSQL_PORT),
    database=MYSQL_DB,
)

# --- Redis caching settings ---
# Used by cache_service.py to cache LLM answers and avoid repeated
# Gemini API calls for the same (document_id, question) pair.
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None   # None = no auth
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


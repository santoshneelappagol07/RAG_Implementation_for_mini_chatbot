"""
cache_service.py
-----------------
Thin wrapper around Redis for caching LLM answers.

The cache key is a SHA-256 hash of (document_id, question), so identical
questions about the same document always hit the same key. This avoids
repeating expensive Gemini API calls for answers we already have.

Design notes:
- Every public function catches redis.RedisError and returns a safe
  fallback (None / False) instead of crashing. If Redis is down, the app
  still works — just without caching.
- We use the synchronous `redis.Redis` client (not `redis.asyncio`)
  because the FastAPI endpoints that call this are synchronous `def`
  routes, not `async def`. Mixing async Redis calls inside a sync
  endpoint would require an event-loop dance that adds complexity
  without benefit here.
"""

import hashlib
import json
import logging

import redis

try:
    from backend.config import (
        REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, REDIS_CACHE_TTL,
    )
except ImportError:
    from config import (
        REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, REDIS_CACHE_TTL,
    )

logger = logging.getLogger(__name__)

# ── Redis client ────────────────────────────────────────────────────────
# decode_responses=True  →  get() returns str instead of bytes
# socket_connect_timeout →  don't hang forever if Redis is unreachable
_redis_client: redis.Redis | None = None

try:
    _redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )
    # Quick connectivity check on import — logs a warning but doesn't crash.
    _redis_client.ping()
    logger.info("✅ Connected to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
except Exception as exc:
    logger.warning(
        "⚠️  Redis unavailable (%s). Caching disabled — app will still "
        "work, but every question hits the LLM.", exc,
    )
    _redis_client = None

# Simple in-memory counters for the /cache/stats endpoint.
_stats = {"hits": 0, "misses": 0}

# ── Key helpers ─────────────────────────────────────────────────────────
_KEY_PREFIX = "pdfchat:"


def _make_key(document_id: int, question: str) -> str:
    """
    Build a deterministic cache key from (document_id, question).

    We hash the pair instead of using it literally because questions can
    be arbitrarily long and may contain characters Redis key encoding
    would struggle with.
    """
    raw = f"{document_id}:{question.strip().lower()}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"{_KEY_PREFIX}{document_id}:{digest}"


# ── Public API ──────────────────────────────────────────────────────────

def get_cached_answer(document_id: int, question: str) -> str | None:
    """
    Look up a cached answer for this (document, question) pair.
    Returns the cached answer string, or None on miss / Redis error.
    """
    if _redis_client is None:
        return None
    try:
        data = _redis_client.get(_make_key(document_id, question))
        if data is not None:
            _stats["hits"] += 1
            parsed = json.loads(data)
            return parsed.get("answer")
        _stats["misses"] += 1
        return None
    except redis.RedisError as exc:
        logger.warning("Redis GET error: %s", exc)
        _stats["misses"] += 1
        return None


def set_cached_answer(
    document_id: int,
    question: str,
    answer: str,
    ttl: int | None = None,
) -> bool:
    """
    Store an answer in Redis with an expiry (TTL).
    Returns True on success, False on failure.
    """
    if _redis_client is None:
        return False
    try:
        payload = json.dumps({"document_id": document_id, "question": question, "answer": answer})
        _redis_client.setex(
            name=_make_key(document_id, question),
            time=ttl or REDIS_CACHE_TTL,
            value=payload,
        )
        return True
    except redis.RedisError as exc:
        logger.warning("Redis SET error: %s", exc)
        return False


def invalidate_document_cache(document_id: int) -> int:
    """
    Delete all cached answers for a given document_id.
    Returns the number of keys deleted (0 if Redis is down).
    """
    if _redis_client is None:
        return 0
    try:
        pattern = f"{_KEY_PREFIX}{document_id}:*"
        keys = list(_redis_client.scan_iter(match=pattern, count=200))
        if keys:
            return _redis_client.delete(*keys)
        return 0
    except redis.RedisError as exc:
        logger.warning("Redis DELETE error: %s", exc)
        return 0


def get_cache_stats() -> dict:
    """Return hit/miss counters and Redis connectivity status."""
    connected = False
    if _redis_client is not None:
        try:
            _redis_client.ping()
            connected = True
        except redis.RedisError:
            pass

    return {
        "redis_connected": connected,
        "cache_hits": _stats["hits"],
        "cache_misses": _stats["misses"],
        "hit_rate": (
            round(_stats["hits"] / (_stats["hits"] + _stats["misses"]) * 100, 1)
            if (_stats["hits"] + _stats["misses"]) > 0
            else 0.0
        ),
    }

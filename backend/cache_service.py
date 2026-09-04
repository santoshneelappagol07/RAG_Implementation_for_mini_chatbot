"""
cache_service.py
-----------------
Thin wrapper around Redis for caching LLM answers with Multi-Tier Caching:
  Tier 1: Exact Hash Cache (SHA-256 of document_id + question, O(1) < 1ms)
  Tier 2: Semantic Similarity Cache (Cosine similarity over Gemini question embeddings, < 15ms)

Design notes:
- Every public function catches redis.RedisError and returns a safe
  fallback (None / False) instead of crashing. If Redis is down, an
  in-memory semantic cache and standard fallback ensure the app still works.
- Fully compatible with sync and async callers.
"""

import hashlib
import json
import logging
import math
import time
from typing import Optional, List, Dict, Any, Tuple

import redis

try:
    from backend.config import (
        REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, REDIS_CACHE_TTL,
        SEMANTIC_CACHE_ENABLED, SEMANTIC_CACHE_THRESHOLD,
    )
except ImportError:
    from config import (
        REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD, REDIS_CACHE_TTL,
        SEMANTIC_CACHE_ENABLED, SEMANTIC_CACHE_THRESHOLD,
    )

logger = logging.getLogger(__name__)

# ── Redis client ────────────────────────────────────────────────────────
_redis_client: Optional[redis.Redis] = None

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
    _redis_client.ping()
    logger.info("✅ Connected to Redis at %s:%s", REDIS_HOST, REDIS_PORT)
except Exception as exc:
    logger.warning(
        "⚠️  Redis unavailable (%s). Using in-memory caching fallback.", exc,
    )
    _redis_client = None

# Multi-tier statistics
_stats = {
    "exact_hits": 0,
    "semantic_hits": 0,
    "misses": 0,
}

# In-memory fallbacks when Redis is not running
_memory_exact_cache: Dict[str, str] = {}
_memory_semantic_cache: Dict[int, List[Dict[str, Any]]] = {}

# ── Key helpers ─────────────────────────────────────────────────────────
_KEY_PREFIX = "pdfchat:"
_SEMANTIC_KEY_PREFIX = "pdfchat:semantic:"


def _make_key(document_id: int, question: str) -> str:
    """Deterministic exact cache key."""
    raw = f"{document_id}:{question.strip().lower()}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return f"{_KEY_PREFIX}{document_id}:{digest}"


def _make_semantic_key(document_id: int) -> str:
    """Redis key storing semantic vectors for a document."""
    return f"{_SEMANTIC_KEY_PREFIX}{document_id}"


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two vector embeddings."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


# ── Tier 1: Exact Hash Cache ─────────────────────────────────────────────

def get_cached_answer(document_id: int, question: str) -> Optional[str]:
    """
    Tier 1 Cache: Look up exact answer for this (document, question) pair.
    Returns cached answer string or None.
    """
    cache_key = _make_key(document_id, question)
    if _redis_client is not None:
        try:
            data = _redis_client.get(cache_key)
            if data is not None:
                _stats["exact_hits"] += 1
                parsed = json.loads(data)
                return parsed.get("answer")
        except redis.RedisError as exc:
            logger.warning("Redis GET error: %s", exc)

    if cache_key in _memory_exact_cache:
        _stats["exact_hits"] += 1
        return _memory_exact_cache[cache_key]

    return None


def set_cached_answer(
    document_id: int,
    question: str,
    answer: str,
    ttl: Optional[int] = None,
) -> bool:
    """Store an exact answer in cache."""
    cache_key = _make_key(document_id, question)
    _memory_exact_cache[cache_key] = answer

    if _redis_client is None:
        return True
    try:
        payload = json.dumps({"document_id": document_id, "question": question, "answer": answer})
        _redis_client.setex(
            name=cache_key,
            time=ttl or REDIS_CACHE_TTL,
            value=payload,
        )
        return True
    except redis.RedisError as exc:
        logger.warning("Redis SET error: %s", exc)
        return False


# ── Tier 2: Semantic Similarity Cache ───────────────────────────────────

def get_semantic_cached_answer(
    document_id: int,
    query_embedding: List[float],
    threshold: float = SEMANTIC_CACHE_THRESHOLD,
) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Tier 2 Cache: Searches previously answered questions for this document
    using cosine similarity on embeddings.

    Returns:
        (answer, max_similarity_score, matched_question)
    """
    if not SEMANTIC_CACHE_ENABLED or not query_embedding:
        return None, 0.0, None

    entries: List[Dict[str, Any]] = []

    # Fetch entries from Redis
    if _redis_client is not None:
        try:
            raw_entries = _redis_client.lrange(_make_semantic_key(document_id), 0, -1)
            for item in raw_entries:
                entries.append(json.loads(item))
        except redis.RedisError as exc:
            logger.warning("Redis semantic LRANGE error: %s", exc)

    # Combine with in-memory entries
    if document_id in _memory_semantic_cache:
        entries.extend(_memory_semantic_cache[document_id])

    if not entries:
        _stats["misses"] += 1
        return None, 0.0, None

    best_entry = None
    best_score = -1.0

    for entry in entries:
        cached_embedding = entry.get("embedding")
        if not cached_embedding:
            continue
        score = _cosine_similarity(query_embedding, cached_embedding)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is not None and best_score >= threshold:
        _stats["semantic_hits"] += 1
        return best_entry.get("answer"), best_score, best_entry.get("question")

    _stats["misses"] += 1
    return None, max(best_score, 0.0), None


def set_semantic_cached_answer(
    document_id: int,
    question: str,
    query_embedding: List[float],
    answer: str,
    ttl: Optional[int] = None,
) -> bool:
    """
    Saves a question, its vector embedding, and answer into the semantic cache.
    """
    if not SEMANTIC_CACHE_ENABLED or not query_embedding:
        return False

    payload = {
        "document_id": document_id,
        "question": question,
        "embedding": query_embedding,
        "answer": answer,
        "timestamp": time.time(),
    }

    # In-memory store
    if document_id not in _memory_semantic_cache:
        _memory_semantic_cache[document_id] = []
    _memory_semantic_cache[document_id].append(payload)

    # Redis store
    if _redis_client is not None:
        try:
            key = _make_semantic_key(document_id)
            _redis_client.rpush(key, json.dumps(payload))
            _redis_client.expire(key, ttl or REDIS_CACHE_TTL)
            return True
        except redis.RedisError as exc:
            logger.warning("Redis semantic RPUSH error: %s", exc)
            return False

    return True


# ── Cache Invalidation & Stats ──────────────────────────────────────────

def invalidate_document_cache(document_id: int) -> int:
    """Delete all exact and semantic cached answers for a given document_id."""
    deleted = 0
    # Clear memory
    _memory_semantic_cache.pop(document_id, None)
    keys_to_del = [k for k in _memory_exact_cache if k.startswith(f"{_KEY_PREFIX}{document_id}:")]
    for k in keys_to_del:
        _memory_exact_cache.pop(k, None)
        deleted += 1

    if _redis_client is not None:
        try:
            pattern = f"{_KEY_PREFIX}{document_id}:*"
            keys = list(_redis_client.scan_iter(match=pattern, count=200))
            keys.append(_make_semantic_key(document_id))
            if keys:
                deleted += _redis_client.delete(*keys)
        except redis.RedisError as exc:
            logger.warning("Redis DELETE error: %s", exc)

    return deleted


def get_cache_stats() -> dict:
    """Return hit/miss counters and Redis connectivity status."""
    connected = False
    if _redis_client is not None:
        try:
            _redis_client.ping()
            connected = True
        except redis.RedisError:
            pass

    total_hits = _stats["exact_hits"] + _stats["semantic_hits"]
    total_reqs = total_hits + _stats["misses"]

    return {
        "redis_connected": connected,
        "exact_cache_hits": _stats["exact_hits"],
        "semantic_cache_hits": _stats["semantic_hits"],
        "total_cache_hits": total_hits,
        "cache_misses": _stats["misses"],
        "hit_rate": round(total_hits / total_reqs * 100, 1) if total_reqs > 0 else 0.0,
    }


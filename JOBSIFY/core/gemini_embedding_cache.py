"""
Persistent embedding cache for direct Gemini Embedding API calls.

Wraps client.models.embed_content() calls in ranker.py and job_matcher.py
so that identical texts are never re-embedded across restarts or rerank cycles.

Cache is SQLite-backed (same pattern as LLMCache in llm_invoker.py) and
thread-safe. TTL defaults to 24 hours — resume and JD text rarely changes
between rerank runs, so cache hit rates are expected to be high (>90%).

Usage:
    from core.gemini_embedding_cache import cached_embed_texts, cached_embed_text

    # Batch (replaces client.models.embed_content(model=..., contents=list))
    vectors = cached_embed_texts(client, model, list_of_texts)

    # Single text (replaces _get_text_embedding / _embed helpers)
    vector = cached_embed_text(client, model, text)
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import closing
from typing import List, Optional, Any

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
_ENABLED = os.getenv("ENABLE_GEMINI_EMBED_CACHE", "true").strip().lower() == "true"
_TTL_SECONDS = int(os.getenv("GEMINI_EMBED_CACHE_TTL_SECONDS", str(60 * 60 * 24)))  # 24 h
_DB_PATH = os.getenv("GEMINI_EMBED_CACHE_PATH", "/tmp/gemini_embed_cache.sqlite")
_CLEANUP_INTERVAL = 3600  # clean expired rows once per hour

# ── Internal state ─────────────────────────────────────────────────────────────
_lock = threading.Lock()
_last_cleanup: float = 0.0
_initialized = False


def _init_db() -> None:
    """Create the cache table if it doesn't exist (idempotent)."""
    global _initialized
    if _initialized:
        return
    try:
        with closing(sqlite3.connect(_DB_PATH, check_same_thread=False)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gemini_embed_cache (
                    k     TEXT PRIMARY KEY,
                    v     TEXT NOT NULL,
                    ts    INTEGER NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON gemini_embed_cache(ts)")
            conn.commit()
        _initialized = True
        log.debug("✅ Gemini embedding cache DB initialised at %s", _DB_PATH)
    except Exception as exc:
        log.warning("⚠️ Could not initialise Gemini embedding cache DB: %s", exc)


def _make_key(model: str, text: str) -> str:
    payload = f"{model}||{text.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _now() -> int:
    return int(time.time())


def _cache_get(key: str) -> Optional[List[float]]:
    try:
        with closing(sqlite3.connect(_DB_PATH, check_same_thread=False)) as conn:
            row = conn.execute(
                "SELECT v, ts FROM gemini_embed_cache WHERE k = ?", (key,)
            ).fetchone()
            if not row:
                return None
            v, ts = row
            if _now() - ts > _TTL_SECONDS:
                conn.execute("DELETE FROM gemini_embed_cache WHERE k = ?", (key,))
                conn.commit()
                return None
            return json.loads(v)
    except Exception as exc:
        log.debug("Embed cache GET failed: %s", exc)
        return None


def _cache_set(key: str, vector: List[float]) -> None:
    try:
        with closing(sqlite3.connect(_DB_PATH, check_same_thread=False)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO gemini_embed_cache (k, v, ts) VALUES (?, ?, ?)",
                (key, json.dumps(vector), _now()),
            )
            conn.commit()
    except Exception as exc:
        log.debug("Embed cache SET failed: %s", exc)


def _maybe_cleanup() -> None:
    """Periodically delete expired rows to keep the DB lean."""
    global _last_cleanup
    now = _now()
    if now - _last_cleanup < _CLEANUP_INTERVAL:
        return
    _last_cleanup = now
    try:
        cutoff = now - _TTL_SECONDS
        with closing(sqlite3.connect(_DB_PATH, check_same_thread=False)) as conn:
            deleted = conn.execute(
                "DELETE FROM gemini_embed_cache WHERE ts < ?", (cutoff,)
            ).rowcount
            conn.commit()
        if deleted:
            log.debug("🧹 Gemini embed cache: removed %d expired entries", deleted)
    except Exception as exc:
        log.debug("Embed cache cleanup failed: %s", exc)


# ── Public API ─────────────────────────────────────────────────────────────────

def cached_embed_texts(
    client: Any,
    model: str,
    texts: List[str],
) -> List[List[float]]:
    """
    Batch-embed a list of texts, serving cached vectors where available.

    Replaces:
        client.models.embed_content(model=model, contents=texts)

    Returns a list of float vectors in the same order as `texts`.
    Falls back to a direct API call for any text not in cache, and caches
    the results. Returns [] on complete failure.
    """
    if not texts:
        return []

    _init_db()
    _maybe_cleanup()

    if not _ENABLED:
        return _call_api_batch(client, model, texts)

    keys = [_make_key(model, t) for t in texts]
    results: List[Optional[List[float]]] = [None] * len(texts)
    miss_indices: List[int] = []

    with _lock:
        for i, key in enumerate(keys):
            vec = _cache_get(key)
            if vec is not None:
                results[i] = vec
            else:
                miss_indices.append(i)

    if not miss_indices:
        log.debug("🟩 Gemini embed cache: all %d texts hit", len(texts))
        return [r for r in results]  # type: ignore[return-value]

    log.debug(
        "🟥 Gemini embed cache: %d miss(es) out of %d texts",
        len(miss_indices),
        len(texts),
    )

    miss_texts = [texts[i] for i in miss_indices]
    computed = _call_api_batch(client, model, miss_texts)

    if len(computed) != len(miss_texts):
        # API returned unexpected count — fall back to full uncached call
        log.warning(
            "⚠️ Embed API returned %d vectors for %d texts; skipping cache",
            len(computed),
            len(miss_texts),
        )
        return _call_api_batch(client, model, texts)

    with _lock:
        for idx, vec in zip(miss_indices, computed):
            _cache_set(keys[idx], vec)
            results[idx] = vec

    return results  # type: ignore[return-value]


def cached_embed_text(
    client: Any,
    model: str,
    text: str,
) -> List[float]:
    """
    Embed a single text string, served from cache when possible.

    Replaces:
        client.models.embed_content(model=model, contents=text)

    Returns a float vector, or [] on failure.
    """
    vecs = cached_embed_texts(client, model, [text])
    return vecs[0] if vecs else []


# ── Internal helpers ───────────────────────────────────────────────────────────

def _call_api_batch(
    client: Any,
    model: str,
    texts: List[str],
) -> List[List[float]]:
    """Call the Gemini embed API and return a list of float vectors."""
    try:
        result = client.models.embed_content(model=model, contents=texts)
        vectors: List[List[float]] = []
        if hasattr(result, "embeddings") and result.embeddings:
            for emb in result.embeddings:
                if hasattr(emb, "values"):
                    vectors.append(list(emb.values))
                elif isinstance(emb, list):
                    vectors.append(list(emb))
                else:
                    vectors.append(list(emb))
        return vectors
    except Exception as exc:
        log.warning("⚠️ Gemini embed API call failed: %s", exc)
        return []

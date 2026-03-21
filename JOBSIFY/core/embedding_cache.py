"""
Optional in-process embedding cache to avoid re-embedding the same text (Section 8 Issue 2).
Issue 8.2: Enabled by default for cost savings. Disable via ENABLE_EMBEDDING_CACHE=false.
"""
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, List, Optional

log = logging.getLogger(__name__)

# Issue 8.2: Enable embedding cache by default for cost/latency savings
_ENABLED = os.getenv("ENABLE_EMBEDDING_CACHE", "true").strip().lower() == "true"
_MAX_ENTRIES = int(os.getenv("EMBEDDING_CACHE_MAX_ENTRIES", "10000"))
_TTL_SECONDS = int(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", "3600"))
_MODEL_VERSION = os.getenv("EMBEDDING_CACHE_MODEL_VERSION", "v1")


def _text_key(text: str) -> str:
    normalized = (text or "").strip()
    return hashlib.sha256((normalized + _MODEL_VERSION).encode()).hexdigest()


class _EmbeddingCache:
    """LRU cache for embedding vectors with TTL."""

    def __init__(self, max_entries: int = _MAX_ENTRIES, ttl_seconds: int = _TTL_SECONDS):
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, tuple[List[float], float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[List[float]]:
        with self._lock:
            if key not in self._cache:
                return None
            vec, ts = self._cache[key]
            if time.time() - ts > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return vec

    def set(self, key: str, vec: List[float]) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (vec, time.time())
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)


_cache_instance: Optional[_EmbeddingCache] = None


def _get_cache() -> Optional[_EmbeddingCache]:
    global _cache_instance
    if not _ENABLED:
        return None
    if _cache_instance is None:
        _cache_instance = _EmbeddingCache()
    return _cache_instance


def wrap_embedding_fn(embedding_fn: Any) -> Any:
    """
    Wrap an embedding function so that repeated inputs are served from cache when
    ENABLE_EMBEDDING_CACHE=true. The wrapped object is callable with the same
    signature (list of strings -> list of vectors) and forwards other attrs to the inner fn.
    """
    if not _ENABLED:
        return embedding_fn

    cache = _get_cache()
    if cache is None:
        return embedding_fn

    class _CachedWrapper:
        def __call__(self, input: List[str]) -> List[List[float]]:
            if not input:
                return embedding_fn(input) if callable(embedding_fn) else []
            keys = [_text_key(t) for t in input]
            results: List[Optional[List[float]]] = [cache.get(k) for k in keys]
            to_compute: List[int] = [i for i, r in enumerate(results) if r is None]
            if not to_compute:
                return [r for r in results if r is not None]
            texts_to_compute = [input[i] for i in to_compute]
            computed = embedding_fn(texts_to_compute) if callable(embedding_fn) else []
            if len(computed) != len(texts_to_compute):
                return embedding_fn(input) if callable(embedding_fn) else []
            for idx, vec in zip(to_compute, computed):
                cache.set(keys[idx], vec)
                results[idx] = vec
            return list(results)

        def __getattr__(self, name: str) -> Any:
            return getattr(embedding_fn, name)

    return _CachedWrapper()

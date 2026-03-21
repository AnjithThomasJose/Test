from __future__ import annotations

import time
from typing import Dict


class TokenBucketRateLimiter:
    """Simple in-memory token bucket rate limiter keyed by identifier (e.g., IP)."""

    def __init__(self, max_tokens: int, refill_rate: float):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second
        self._buckets: Dict[str, Dict[str, float]] = {}

    def allow(self, key: str) -> bool:
        now = time.time()
        bucket = self._buckets.get(key)
        if not bucket:
            self._buckets[key] = {"tokens": float(self.max_tokens - 1), "last": now}
            return True

        elapsed = max(0.0, now - bucket["last"])
        bucket["last"] = now
        # Refill
        bucket["tokens"] = min(self.max_tokens, bucket["tokens"] + elapsed * self.refill_rate)

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False



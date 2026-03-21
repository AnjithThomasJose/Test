from __future__ import annotations

import json
import re
from typing import Any, Dict

PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),  # Credit card
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # Email
    re.compile(r"\b(?:\+?\d[\d(). -]{8,}\d)\b"),  # Phone
]


def add_security_headers(response) -> None:
    headers = response.headers
    headers.setdefault("X-Content-Type-Options", "nosniff")
    headers.setdefault("X-Frame-Options", "DENY")
    headers.setdefault("Referrer-Policy", "no-referrer")
    headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'; base-uri 'none'")


def sanitize_response_json(data: Any) -> Any:
    """Redact common PII patterns in JSON responses. Best-effort, non-recursive for large payloads."""
    if isinstance(data, dict):
        return {k: sanitize_response_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_response_json(v) for v in data]
    if isinstance(data, str):
        redacted = data
        for rx in PII_PATTERNS:
            redacted = rx.sub("[REDACTED]", redacted)
        return redacted
    return data



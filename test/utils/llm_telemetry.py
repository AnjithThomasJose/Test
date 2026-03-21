"""
LLM Telemetry - Structured logging for LLM API calls.

This module provides utilities to log LLM invocation metrics for observability,
including latency, success status, JSON validity, and fallback usage.
"""

import time
import logging
from typing import Optional, Dict, Any

log = logging.getLogger("llm.telemetry")


def log_invoke_llm(
    session_id: Optional[str],
    stage: str,
    model: Optional[str],
    start_ts: float,
    ok: bool,
    json_valid: bool,
    fallback_used: bool
) -> None:
    """
    Log structured telemetry for an LLM invocation.
    
    Args:
        session_id: Optional session identifier
        stage: Stage identifier (e.g., "summary", "question_generation", "scoring")
        model: Optional model name/identifier
        start_ts: Timestamp when LLM call started (from time.time())
        ok: Whether the LLM call succeeded
        json_valid: Whether the returned JSON was valid
        fallback_used: Whether a fallback was used instead of LLM output
    """
    latency_ms = int((time.time() - start_ts) * 1000)
    
    # Build log entry (structured for easy parsing)
    log_data: Dict[str, Any] = {
        "session_id": session_id or "unknown",
        "stage": stage,
        "model": model or "unknown",
        "latency_ms": latency_ms,
        "ok": ok,
        "json_valid": json_valid,
        "fallback_used": fallback_used
    }
    
    # Log at appropriate level
    if not ok or fallback_used:
        log.warning(f"LLM call: {log_data}")
    else:
        log.info(f"LLM call: {log_data}")


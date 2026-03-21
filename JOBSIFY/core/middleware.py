from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from .error_handler import with_error_handling, error_handler

log = logging.getLogger(__name__)


def _get_agent_timeout_seconds(agent_name: str) -> Optional[float]:
    """Section 6 Issue 4: optional per-agent timeout from env. Default None = no timeout."""
    per_agent = os.getenv(f"AGENT_{agent_name.upper()}_TIMEOUT_SECONDS", "").strip()
    if per_agent:
        try:
            return float(per_agent)
        except ValueError:
            pass
    default = os.getenv("AGENT_TIMEOUT_SECONDS", "").strip()
    if not default:
        return None
    try:
        return float(default)
    except ValueError:
        return None

AgentFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass
class MiddlewareConfig:
    enable_tenant_validation: bool = True
    enable_rate_limit: bool = True
    enable_circuit_breaker: bool = True
    enable_evaluation: bool = True
    enable_error_handling: bool = True
    enable_retry: bool = True
    enable_fallback: bool = True


class SimpleCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_ts = 0.0

    def is_open(self, now_ts: float) -> bool:
        if self.state == "open" and (now_ts - self.last_failure_ts) >= self.reset_timeout:
            self.state = "half_open"
        return self.state == "open"

    def record_success(self):
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self, now_ts: float):
        self.failure_count += 1
        self.last_failure_ts = now_ts
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class TenantScopedRateLimiter:
    def __init__(self, max_tokens: int = 10, refill_rate: float = 1.0):
        from .rate_limit import TokenBucketRateLimiter
        self._limiter = TokenBucketRateLimiter(max_tokens=max_tokens, refill_rate=refill_rate)

    def allow(self, tenant_key: str) -> bool:
        return self._limiter.allow(tenant_key)


class MiddlewareManager:
    """Applies security, resilience, and evaluation around agent functions."""

    def __init__(self, config: Optional[MiddlewareConfig] = None):
        self.config = config or MiddlewareConfig()
        self._breakers: Dict[str, SimpleCircuitBreaker] = {}
        self._limiters: Dict[str, TenantScopedRateLimiter] = {}

    def _get_breaker(self, tenant: str, agent: str) -> SimpleCircuitBreaker:
        key = f"{tenant}:{agent}"
        if key not in self._breakers:
            self._breakers[key] = SimpleCircuitBreaker()
        return self._breakers[key]

    def _get_limiter(self, tenant: str) -> TenantScopedRateLimiter:
        if tenant not in self._limiters:
            self._limiters[tenant] = TenantScopedRateLimiter()
        return self._limiters[tenant]

    def apply(
        self, 
        agent_name: str, 
        agent_fn: AgentFn, 
        evaluation_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        fallback_fn: Optional[Callable] = None
    ) -> AgentFn:
        async def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
            tenant_id = state.get("security_context", {}).get("tenant_id") or state.get("tenant_id", "default")
            
            # Step counting and recursion warning (Issue 1.4 fix)
            from core.config import GRAPH_RECURSION_WARNING_THRESHOLD
            current_step = state.get("_graph_step_count", 0) + 1
            uid = state.get("uid", "unknown")
            if current_step == GRAPH_RECURSION_WARNING_THRESHOLD:
                log.warning(
                    f"⚠️ RECURSION WARNING: Graph execution reached {current_step} steps "
                    f"(warning threshold) for uid={uid}. Current node: {agent_name}. "
                    f"Consider optimizing workflow or increasing GRAPH_RECURSION_LIMIT."
                )
            elif current_step > GRAPH_RECURSION_WARNING_THRESHOLD and current_step % 10 == 0:
                log.warning(
                    f"⚠️ RECURSION WARNING: Graph execution at {current_step} steps "
                    f"for uid={uid}. Current node: {agent_name}."
                )
            
            # 🔍 DEBUG: Middleware entry
            log.info(f"🔍 MIDDLEWARE_DEBUG: {agent_name} middleware entry with tenant_id: {tenant_id}")

            # Tenant validation
            if self.config.enable_tenant_validation:
                import re
                if not re.match(r"^[a-zA-Z0-9_\-]{8,64}$", str(tenant_id)):
                    log.error(f"Invalid tenant_id: {tenant_id}")
                    log.error(f"🔍 MIDDLEWARE_DEBUG: {agent_name} failed tenant validation")
                    return {
                        "success": False,
                        "error_code": "validation_medium",
                        "error_message": "Invalid tenant_id format",
                        "error_category": "validation",
                        "agent_status": f"{agent_name}_validation_error"
                    }
                log.info(f"🔍 MIDDLEWARE_DEBUG: {agent_name} passed tenant validation")

            # Rate limit
            if self.config.enable_rate_limit:
                limiter = self._get_limiter(tenant_id)
                if not limiter.allow(tenant_id):
                    log.warning(f"Rate limit exceeded for tenant: {tenant_id}")
                    log.warning(f"🔍 MIDDLEWARE_DEBUG: {agent_name} rate limited")
                    return {
                        "success": False,
                        "error_code": "rate_limit_medium",
                        "error_message": "Rate limit exceeded",
                        "error_category": "resource",
                        "agent_status": f"{agent_name}_rate_limited"
                    }
                log.info(f"🔍 MIDDLEWARE_DEBUG: {agent_name} passed rate limit check")

            # Circuit breaker
            breaker = self._get_breaker(tenant_id, agent_name) if self.config.enable_circuit_breaker else None
            now_ts = __import__("time").time()
            if breaker and breaker.is_open(now_ts):
                log.warning(f"Circuit breaker open for {agent_name}")
                log.warning(f"🔍 MIDDLEWARE_DEBUG: {agent_name} circuit breaker open")
                return {
                    "success": False,
                    "error_code": "circuit_breaker_high",
                    "error_message": "Circuit breaker is open",
                    "error_category": "system",
                    "agent_status": f"{agent_name}_circuit_open"
                }
            log.info(f"🔍 MIDDLEWARE_DEBUG: {agent_name} passed circuit breaker check")

            # Execute agent with centralized error handling (Section 6 Issue 4: optional timeout)
            try:
                log.info(f"🔍 MIDDLEWARE_DEBUG: {agent_name} executing agent function")
                _t = __import__("time")
                _start_ts = _t.time()
                if self.config.enable_error_handling:
                    coro = with_error_handling(
                        agent_name=agent_name,
                        fallback_fn=fallback_fn if self.config.enable_fallback else None,
                        enable_retry=self.config.enable_retry
                    )(agent_fn)(state)
                else:
                    coro = agent_fn(state)
                timeout_sec = _get_agent_timeout_seconds(agent_name)
                if timeout_sec is not None and timeout_sec > 0:
                    result = await asyncio.wait_for(asyncio.shield(coro), timeout=timeout_sec)
                else:
                    result = await coro
                _end_ts = _t.time()
                
                log.info(f"🔍 MIDDLEWARE_DEBUG: {agent_name} agent function completed successfully")
                if breaker:
                    breaker.record_success()
                    
                # Annotate result with processing time for universal node timing
                try:
                    if isinstance(result, dict):
                        timing_seconds = round(max(0.0, _end_ts - _start_ts), 2)
                        timing_payload = {
                            "agent": agent_name,
                            "processing_time_seconds": timing_seconds,
                        }
                        # Non-breaking: place under a namespaced key
                        result["_node_timing"] = timing_payload
                        
                        # Log agent completion for progress tracking (visible in QA environment)
                        log.warning(f"{agent_name} completed in {timing_seconds} seconds")
                except Exception:
                    pass

            except asyncio.TimeoutError:
                log.error(f"🔍 MIDDLEWARE_DEBUG: {agent_name} timed out")
                if breaker:
                    breaker.record_failure(now_ts)
                timeout_sec_val = timeout_sec if timeout_sec is not None else 0
                return {
                    "success": False,
                    "error_code": "timeout_high",
                    "error_message": f"Agent timed out after {timeout_sec_val}s",
                    "error_category": "resource",
                    "agent_status": f"{agent_name}_timeout"
                }
            except Exception as e:
                log.error(f"🔍 MIDDLEWARE_DEBUG: {agent_name} agent function failed with exception: {e}")
                if breaker:
                    breaker.record_failure(now_ts)
                
                # If error handling is disabled, use original behavior
                if not self.config.enable_error_handling:
                    raise
                
                # Otherwise, the error handler should have caught it
                log.error(f"Unexpected error in {agent_name}: {e}")
                return {
                    "success": False,
                    "error_code": "unexpected_error_high",
                    "error_message": f"Unexpected error: {str(e)}",
                    "error_category": "unknown",
                    "agent_status": f"{agent_name}_error"
                }

            # Evaluate output
            if self.config.enable_evaluation and evaluation_fn and result is not None:
                try:
                    evaluation_fn(result)
                    log.info(f"🔍 MIDDLEWARE_DEBUG: {agent_name} evaluation completed successfully")
                except Exception as eval_error:
                    # Never fail the pipeline due to evaluation; annotate instead
                    log.warning(f"Evaluation failed for {agent_name}: {eval_error}")
                    log.warning(f"🔍 MIDDLEWARE_DEBUG: {agent_name} evaluation failed: {eval_error}")
                    result["evaluation_warning"] = True
                    result["evaluation_error"] = str(eval_error)

            # Propagate step count for recursion monitoring (Issue 1.4)
            if isinstance(result, dict):
                result["_graph_step_count"] = current_step

            log.info(f"🔍 MIDDLEWARE_DEBUG: {agent_name} middleware completed, returning result")
            return result

        return wrapped



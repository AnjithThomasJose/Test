from langgraph.graph import StateGraph
from langgraph.errors import GraphRecursionError
from typing import Dict, Any, List, TypedDict, Annotated, Optional, Union, Tuple
import sys
import os
import operator
import re
import asyncio
import time
import hashlib
import json
import logging
import copy
from functools import partial
from collections import deque, OrderedDict
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse
import threading
import concurrent.futures
import atexit
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chroma import (
    move_job_to_closed, 
    move_job_to_active,
    batch_move_jobs_to_closed,
    batch_move_jobs_to_active,
)
from core.security import (
    sanitize_url,
    sanitize_url_async,
    RESUME_URL_ALLOWED_SCHEMES as ALLOWED_SCHEMES,
    get_resume_url_allowlist_hosts,
)

ALLOWED_HOSTS = get_resume_url_allowlist_hosts()

# Reducer to safely merge string or list inputs into a single list
def merge_names(existing, new):
    if existing is None:
        existing = []
    elif isinstance(existing, str):
        existing = [existing]
    if new is None:
        new = []
    elif isinstance(new, str):
        new = [new]
    return (existing or []) + (new or [])

def merge_lists(existing, new):
    """Merge two lists, concatenating them. Handles None values and ensures list type."""
    if existing is None:
        existing = []
    elif not isinstance(existing, list):
        existing = [existing]
    if new is None:
        new = []
    elif not isinstance(new, list):
        new = [new]
    return (existing or []) + (new or [])


def _state_merge_max_env(name: str, default: int, lo: int = 1, hi: int = 10_000) -> int:
    try:
        v = int(os.getenv(name, str(default)))
        return max(lo, min(hi, v))
    except ValueError:
        return default


def _bounded_list_reducer(existing, new, *, max_items: int):
    """Merge lists, dedupe dicts by stable JSON key, then keep the last max_items entries."""

    def _item_key(item: Any) -> str:
        if isinstance(item, dict):
            try:
                return json.dumps(item, sort_keys=True, default=str)
            except Exception:
                return str(id(item))
        return str(item)

    merged = merge_lists(existing, new)
    if not merged:
        return merged
    deduped: Dict[str, Any] = {}
    order: List[str] = []
    for x in merged:
        k = _item_key(x)
        if k not in deduped:
            order.append(k)
        deduped[k] = x
    out = [deduped[k] for k in order]
    if len(out) > max_items:
        log.debug("merge_lists_bounded: truncating list from %s to %s items", len(out), max_items)
        out = out[-max_items:]
    return out


_merge_education = partial(_bounded_list_reducer, max_items=_state_merge_max_env("STATE_MERGE_MAX_EDUCATION", 20))
_merge_work_experience = partial(_bounded_list_reducer, max_items=_state_merge_max_env("STATE_MERGE_MAX_WORK_EXPERIENCE", 50))
_merge_skills = partial(_bounded_list_reducer, max_items=_state_merge_max_env("STATE_MERGE_MAX_SKILLS", 100))
_merge_certifications = partial(_bounded_list_reducer, max_items=_state_merge_max_env("STATE_MERGE_MAX_CERTIFICATIONS", 30))
_merge_projects = partial(_bounded_list_reducer, max_items=_state_merge_max_env("STATE_MERGE_MAX_PROJECTS", 30))
_merge_extras = partial(_bounded_list_reducer, max_items=_state_merge_max_env("STATE_MERGE_MAX_EXTRAS", 30))
_merge_agent_execution_trace = partial(_bounded_list_reducer, max_items=_state_merge_max_env("STATE_MERGE_MAX_AGENT_TRACE", 30))
_merge_agents_invoked = partial(_bounded_list_reducer, max_items=_state_merge_max_env("STATE_MERGE_MAX_AGENTS_INVOKED", 50))


def take_last(existing, new):
    """Reducer that takes the last (newest) value, overwriting the existing one.
    
    This is a named function (not lambda) to enable:
    - State serialization/pickling for checkpointing
    - Distributed execution with Redis/database checkpointers
    - Better debugging with meaningful __name__ attribute
    """
    return new


def accumulate_int(existing, new):
    """Reducer that sums integer values. Each node contributes its increment."""
    return (existing or 0) + (new or 0)

# Import all agent nodes
# Note: Logging not yet initialized here, so we'll log after setup_loggers() is called
from agents.validate_resume import validate_resume_agent, is_valid_resume
import asyncio
from agents.validate_jd import validate_jd_agent, is_valid_jd
from agents.groq_resume_parser import groq_resume_parser_agent
from agents.skill_proficiency_analyzer import (
    skill_proficiency_analyzer_agent,
    extract_skills_from_certificate_names,
    _normalize_skill_name,
)
from agents.resume_assembler import resume_assembler_agent
from agents.resume_summary_agent import resume_summary_agent
from agents.interest_filler_agent import interest_filler_agent
from agents.resume_analysis import resume_analysis_agent
from agents.job_matcher import job_matcher_agent, job_matcher_preprocessor, compare_candidate_with_job
from agents.resume_score import resume_scorer_agent
from agents.skill_and_career_advisor import skill_and_career_advisor_agent
from agents.enhanced_role_fit_agent import enhanced_role_fit_agent
from agents.market_and_course_recommender import market_and_course_recommender_agent
from agents.assessment_recommender import assessment_recommender_agent
from agents.assessment_question_generator import assessment_question_generator_agent
from agents.assessment_evaluator import assessment_evaluator_agent
from agents.report_generator import report_generator_agent
from agents.assessment_validator import assessment_validator_agent
from agents.groq_jd_parser import groq_jd_parser_agent
from agents.prescreening_questions_agent import prescreening_questions_agent
from agents.ranker import ranker_agent
from agents.notification_agent import notification_agent_http
from agents.resume_content_generator import resume_content_generator_agent
from agents.assessment_builder_agent import assessment_builder_agent

# Import utility and setup functions
from utils.resume_utils import (
    download_resume_text_async,
    _quality_diag,
    is_text_weak,
    _pdf_page_count,
)
from utils.memory_manager import memory_manager
from utils.session_manager import session_manager
from log_handler import setup_loggers
from core.langfuse_tracing import merge_langfuse_into_config

# Safe LangSmith tracing - handle errors gracefully
import logging
log = logging.getLogger('main')

try:
    from langsmith.run_helpers import traceable as _traceable
    import os
    # Only enable tracing if properly configured
    _LANGSMITH_ENABLED = (
        os.getenv("LANGCHAIN_TRACING_V2") == "true" or 
        os.getenv("LANGSMITH_TRACING") == "true"
    ) and bool(os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"))
    
    if _LANGSMITH_ENABLED:
        # Helper function to safely process outputs for LangSmith
        def _safe_process_outputs(outputs):
            """Ensure outputs are always valid and serializable for LangSmith."""
            if outputs is None:
                return {"output": None, "_note": "Function returned None"}
            
            def _sanitize_value(value, max_depth=10):
                """Recursively sanitize values to prevent NoneType subscriptable errors."""
                if max_depth <= 0:
                    return "<max_depth_reached>"
                
                if value is None:
                    return None
                
                if isinstance(value, (str, int, float, bool)):
                    return value
                
                if isinstance(value, dict):
                    sanitized = {}
                    for k, v in value.items():
                        try:
                            sanitized[k] = _sanitize_value(v, max_depth - 1)
                        except Exception as e:
                            sanitized[k] = f"<error_sanitizing: {str(e)}>"
                    return sanitized
                
                if isinstance(value, (list, tuple)):
                    sanitized = []
                    for item in value:
                        try:
                            sanitized.append(_sanitize_value(item, max_depth - 1))
                        except Exception as e:
                            sanitized.append(f"<error_sanitizing: {str(e)}>")
                    return sanitized
                
                # For other types, convert to string representation
                try:
                    if hasattr(value, '__dict__'):
                        return {"_type": type(value).__name__, "_repr": str(value)}
                    return str(value)
                except Exception:
                    return f"<{type(value).__name__} object>"
            
            # If outputs is already a dict, ensure it's safe
            if isinstance(outputs, dict):
                try:
                    return _sanitize_value(outputs)
                except Exception as e:
                    log.debug(f"Error sanitizing dict outputs for LangSmith: {e}")
                    return {"output": "<error_sanitizing_dict>", "_error": str(e)}
            
            # For non-dict outputs, wrap them safely
            try:
                sanitized = _sanitize_value(outputs)
                if isinstance(sanitized, dict):
                    return sanitized
                return {"output": sanitized}
            except Exception as e:
                log.debug(f"Error processing outputs for LangSmith: {e}")
                return {"output": "<unserializable>", "_error": str(e)}
        
        # Wrap traceable to catch and suppress errors during execution
        def traceable(*args, **kwargs):
            # Extract existing process_outputs if provided
            existing_process_outputs = kwargs.pop("process_outputs", None)
            
            # Create a combined process_outputs function
            def combined_process_outputs(outputs):
                """Safely process outputs for LangSmith, handling all edge cases."""
                try:
                    # First apply user's custom processor if provided
                    if existing_process_outputs:
                        try:
                            outputs = existing_process_outputs(outputs)
                        except Exception as e:
                            log.debug(f"Error in custom process_outputs: {e}, using safe fallback")
                    
                    # Then apply our safe processor
                    return _safe_process_outputs(outputs)
                except Exception as e:
                    # Ultimate fallback - if anything goes wrong, return a safe structure
                    log.debug(f"Error in combined_process_outputs: {e}, using ultimate fallback")
                    return {
                        "output": "<error_processing_outputs>",
                        "_error": str(e),
                        "_error_type": type(e).__name__
                    }
            
            # Add our safe process_outputs handler
            kwargs["process_outputs"] = combined_process_outputs
            
            def decorator(func):
                traced_func = _traceable(*args, **kwargs)(func)
                # Wrap the traced function to catch runtime errors
                if asyncio.iscoroutinefunction(traced_func):
                    async def safe_wrapper(*fargs, **fkwargs):
                        try:
                            result = await traced_func(*fargs, **fkwargs)
                            # Ensure result is never None before returning
                            if result is None:
                                log.debug(f"Function {func.__name__} returned None, wrapping for LangSmith")
                                return {"output": None, "_note": "Function returned None"}
                            return result
                        except (TypeError, AttributeError) as e:
                            if "'NoneType' object is not subscriptable" in str(e) or "Failed to post run" in str(e):
                                log.debug(f"LangSmith tracing error (non-critical): {e}, continuing without trace")
                                result = await func(*fargs, **fkwargs)
                                # Ensure result is never None
                                return result if result is not None else {"output": None, "_note": "Function returned None"}
                            raise
                    return safe_wrapper
                else:
                    def safe_wrapper(*fargs, **fkwargs):
                        try:
                            result = traced_func(*fargs, **fkwargs)
                            # Ensure result is never None before returning
                            if result is None:
                                log.debug(f"Function {func.__name__} returned None, wrapping for LangSmith")
                                return {"output": None, "_note": "Function returned None"}
                            return result
                        except (TypeError, AttributeError) as e:
                            if "'NoneType' object is not subscriptable" in str(e) or "Failed to post run" in str(e):
                                log.debug(f"LangSmith tracing error (non-critical): {e}, continuing without trace")
                                result = func(*fargs, **fkwargs)
                                # Ensure result is never None
                                return result if result is not None else {"output": None, "_note": "Function returned None"}
                            raise
                    return safe_wrapper
            return decorator
    else:
        # No-op decorator when tracing is disabled
        def traceable(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
except Exception as e:
    # If langsmith is not available, create a no-op decorator
    log.debug(f"LangSmith not available: {e}, using no-op traceable decorator")
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from chroma import (
    fetch_structured_resume,
    get_resume_doc,
    upsert_resume_doc,
    get_job_matcher_ranking,
    upsert_job_matcher_ranking,
    get_candidate_job_ranking,
    upsert_candidate_job_ranking,
)
from core.middleware import MiddlewareManager
from core.evaluation import evaluate_agent_output_for_pii
from core.field_filter import field_filter_manager, filter_agent_output, get_workflow_summary, learn_from_agent_execution, learn_from_workflow, get_adaptive_insights
from core.concurrency import get_max_concurrency
from core.task_queue import can_short_circuit
from core.observability import create_task_metrics, log_task_metrics
from core.config import GRAPH_RECURSION_LIMIT

# Get existing loggers instead of creating new ones
error_log = logging.getLogger('error')

def log_session_update(uid: str, step: str, data: dict, success: bool = True):
    """Helper function to log session updates"""
    if success:
        log.info(f"✅ SESSION UPDATE: UID={uid}, Step={step}, Data keys={list(data.keys())}")
        log.info(f"📊 SESSION DATA: {data}")
    else:
        log.error(f"❌ SESSION UPDATE FAILED: UID={uid}, Step={step}, Data keys={list(data.keys())}")
        log.error(f"📊 SESSION DATA: {data}")

def log_agent_execution(agent_name: str, uid: str, input_data: dict, output_data: dict, success: bool = True):
    """Helper function to log agent execution and data storage"""
    if success:
        log.info(f"🤖 AGENT EXECUTED: {agent_name} for UID={uid}")
        log.info(f"📥 AGENT INPUT: {list(input_data.keys())}")
        log.info(f"📤 AGENT OUTPUT: {list(output_data.keys())}")
        log.info(f"📊 AGENT OUTPUT DATA: {output_data}")
    else:
        log.error(f"❌ AGENT FAILED: {agent_name} for UID={uid}")
        log.error(f"📥 AGENT INPUT: {list(input_data.keys())}")
        log.error(f"📤 AGENT OUTPUT: {list(output_data.keys())}")
        log.error(f"📊 AGENT OUTPUT DATA: {output_data}")

# Production Constants
TENANT_ID_REGEX = re.compile(r'^[a-zA-Z0-9_\-]{3,64}$')
USER_ID_REGEX = re.compile(r'^[a-zA-Z0-9_\-]{3,64}$')
MAX_PAYLOAD_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PROMPT_SIZE = 30 * 1024    # 30K chars
MAX_RESPONSE_SIZE = 50 * 1024  # 50K chars
CONFIDENCE_THRESHOLD = 0.7
QUALITY_THRESHOLD = 0.6
TIMEOUT_SECONDS = 60
LLM_RETRY_ATTEMPTS = 3
LLM_BASE_BACKOFF = 0.4
CACHE_TTL_MINUTES = 30
MAX_CACHE_ENTRIES = 1000
MAX_TENANT_MEMORY_ENTRIES = 1000

# PII Patterns for redaction
PII_PATTERNS = {
    'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
    'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    'phone': re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b')
}

DANGEROUS = [
    re.compile(r'<script[^>]*>.*?</script>', re.I|re.S),
    re.compile(r'(?i)javascript\s*:'),
    re.compile(r'<[^>]+\s(on\w+\s*=)'),
    re.compile(r'(?i)\beval\s*\('),
    re.compile(r'(?i)\bexec\s*\('),
    re.compile(r'__import__', re.I),
]

def merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dictionaries, creating a new dict without mutating inputs.
    This is a pure function required by LangGraph for concurrent updates."""
    # Safety check: detect coroutines that shouldn't be in state
    import inspect
    if not isinstance(a, dict):
        if inspect.iscoroutine(a) or inspect.isawaitable(a):
            log.warning(f"⚠️ WARNING: merge_dicts received coroutine as 'a' parameter, converting to empty dict")
            a = {}
        else:
            # Not a dict and not a coroutine - unexpected, but try to handle gracefully
            log.warning(f"⚠️ WARNING: merge_dicts received non-dict as 'a' parameter (type: {type(a).__name__}), converting to empty dict")
            a = {}
    if not isinstance(b, dict):
        if inspect.iscoroutine(b) or inspect.isawaitable(b):
            log.warning(f"⚠️ WARNING: merge_dicts received coroutine as 'b' parameter, converting to empty dict")
            b = {}
        else:
            # Not a dict and not a coroutine - unexpected, but try to handle gracefully
            log.warning(f"⚠️ WARNING: merge_dicts received non-dict as 'b' parameter (type: {type(b).__name__}), converting to empty dict")
            b = {}
    
    if not a:
        return b.copy() if b else {}
    if not b:
        return a.copy() if a else {}
    
    # Create a new dictionary to avoid mutating inputs
    result = a.copy()
    for key, value in b.items():
        # Safety check: skip coroutine values
        if inspect.iscoroutine(value) or inspect.isawaitable(value):
            log.warning(f"⚠️ WARNING: Skipping coroutine value for key '{key}' in merge_dicts")
            continue
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


# Production Utility Classes
class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    reset_timeout: int = 60
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    probe_in_progress: bool = False
    _lock: threading.Lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    return False
                return True
            return False

    def record_success(self):
        with self._lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
            self.probe_in_progress = False

    def record_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN

class TokenBucket:
    def __init__(self, max_tokens: int = 10, refill_rate: float = 1.0):
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.refill_rate = refill_rate
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.time()
            time_passed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + time_passed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class TenantAwareCache:
    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES, ttl_minutes: int = CACHE_TTL_MINUTES):
        self.cache = {}
        # LRU order: oldest at front (popitem last=False), MRU at end — O(1) touch/evict
        self._lru_order: "OrderedDict[str, None]" = OrderedDict()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_minutes * 60
        self._lock = threading.Lock()

    def _generate_key(self, tenant_id: str, **kwargs) -> str:
        key_data = {"tenant_id": tenant_id, **kwargs}
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def get(self, tenant_id: str, **kwargs) -> Optional[Any]:
        key = self._generate_key(tenant_id, **kwargs)
        now = time.time()
        with self._lock:
            if key in self.cache:
                entry_time, value = self.cache[key]
                if now - entry_time < self.ttl_seconds:
                    self._lru_order.move_to_end(key)
                    return value
                del self.cache[key]
                self._lru_order.pop(key, None)
        return None

    def set(self, tenant_id: str, value: Any, **kwargs):
        key = self._generate_key(tenant_id, **kwargs)
        now = time.time()
        with self._lock:
            if len(self.cache) >= self.max_entries and key not in self.cache:
                oldest_key, _ = self._lru_order.popitem(last=False)
                self.cache.pop(oldest_key, None)

            self.cache[key] = (now, value)
            self._lru_order.pop(key, None)
            self._lru_order[key] = None

class PerformanceMetrics:
    def __init__(self, max_entries: int = 5000):
        self.response_times = deque(maxlen=max_entries)
        self.confidence_scores = deque(maxlen=max_entries)
        self.method_counts = {"deterministic": 0, "llm": 0, "cached": 0, "error": 0, "rate_limited": 0}
        self.cache_hits = 0
        self.cache_misses = 0
        self.timeouts = 0
        self.circuit_breaker_events = deque(maxlen=100)
        self._lock = threading.Lock()

    def record_response_time(self, duration: float):
        with self._lock:
            self.response_times.append(duration)

    def record_confidence(self, confidence: float):
        with self._lock:
            self.confidence_scores.append(confidence)

    def record_method(self, method: str):
        with self._lock:
            if method in self.method_counts:
                self.method_counts[method] += 1

    def record_cache_hit(self):
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self):
        with self._lock:
            self.cache_misses += 1

    def record_timeout(self):
        with self._lock:
            self.timeouts += 1

    def record_circuit_breaker_event(self, event: str):
        with self._lock:
            self.circuit_breaker_events.append({"event": event, "timestamp": time.time()})

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            response_times = list(self.response_times)
            confidence_scores = list(self.confidence_scores)
            
            return {
                "response_time_p95": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                "response_time_avg": sum(response_times) / len(response_times) if response_times else 0,
                "confidence_avg": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                "method_split": dict(self.method_counts),
                "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                "timeout_rate": self.timeouts / len(response_times) if response_times else 0,
                "circuit_breaker_events": list(self.circuit_breaker_events)
            }

# Issue 8.1: BoundedTenantDict to prevent unbounded memory growth
class BoundedTenantDict:
    """
    Thread-safe dictionary with LRU eviction and TTL cleanup.
    
    Prevents memory exhaustion in long-running servers with many tenants.
    """
    def __init__(self, max_entries: int = 500, ttl_seconds: int = 7200):
        self._dict: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._access_order: "OrderedDict[str, None]" = OrderedDict()
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries (must be called with lock held)."""
        now = time.time()
        expired = [k for k, ts in self._timestamps.items() if now - ts > self._ttl]
        for k in expired:
            self._dict.pop(k, None)
            self._timestamps.pop(k, None)
            self._access_order.pop(k, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries (must be called with lock held)."""
        while len(self._dict) >= self._max_entries and self._access_order:
            oldest_key, _ = self._access_order.popitem(last=False)
            self._dict.pop(oldest_key, None)
            self._timestamps.pop(oldest_key, None)
    
    def get_or_create(self, key: str, factory: callable) -> Any:
        """Get existing value or create new one using factory."""
        with self._lock:
            now = time.time()
            self._cleanup_expired()
            
            if key in self._dict:
                self._timestamps[key] = now
                self._access_order.move_to_end(key)
                return self._dict[key]
            
            self._evict_lru()
            
            value = factory()
            self._dict[key] = value
            self._timestamps[key] = now
            self._access_order[key] = None
            return value
    
    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._dict
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._dict)


# Global instances with bounded memory
_circuit_breakers = BoundedTenantDict(max_entries=500, ttl_seconds=7200)
_rate_limiters = BoundedTenantDict(max_entries=500, ttl_seconds=7200)
cache = TenantAwareCache()
metrics = PerformanceMetrics()

# ✅ FIX: Enhanced in-flight tracking with result storage
# asyncio.Event must be created outside threading.Lock (avoid binding/cross-thread issues).
# threading.Lock protects _inflight_keys for sync end_invalid_* nodes and async dispatcher.
_inflight_keys = {}  # key_id -> {"timestamp": float, "result": Optional[Dict], "event": asyncio.Event}
_inflight_lock = threading.Lock()
_REQUEST_SLA_SECONDS = int(os.getenv("REQUEST_COALESCE_TIMEOUT_SECONDS", "15"))  # reduced from 300s to avoid thread starvation


def get_circuit_breaker(tenant_id: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a tenant (with bounded memory)."""
    return _circuit_breakers.get_or_create(tenant_id, CircuitBreaker)


def get_rate_limiter(tenant_id: str) -> TokenBucket:
    """Get or create a rate limiter for a tenant (with bounded memory)."""
    return _rate_limiters.get_or_create(tenant_id, TokenBucket)

def validate_tenant_id(tenant_id: str) -> bool:
    return bool(TENANT_ID_REGEX.match(tenant_id)) if tenant_id else False

def validate_user_id(user_id: str) -> bool:
    return bool(USER_ID_REGEX.match(user_id)) if user_id else False

def _looks_like_cc(s:str)->bool:
    d = re.sub(r'\D','',s)
    if not (13 <= len(d) <= 19): return False
    tot,alt = 0, False
    for ch in reversed(d):
        n = ord(ch)-48
        if alt: n = n*2 - (9 if n*2>9 else 0)
        tot += n; alt = not alt
    return tot % 10 == 0

def scrub_and_tokenize(text: str) -> str:
    t = text or ""
    for pat in DANGEROUS: t = pat.sub("[SANITIZED]", t)
    # CC first, guarded by Luhn
    for m in re.finditer(r'\b(?:\d[ -]*?){13,19}\b', t):
        if _looks_like_cc(m.group()):
            t = t.replace(m.group(), "[CC_TOKEN]")
    t = re.sub(PII_PATTERNS['ssn'], "[SSN_TOKEN]", t)
    t = re.sub(PII_PATTERNS['email'], "[EMAIL_TOKEN]", t)
    t = re.sub(PII_PATTERNS['phone'], "[PHONE_TOKEN]", t)
    return t

def validate_payload_size(data: Any) -> bool:
    """Validate that serialized data doesn't exceed size limits.
    Uses sys.getsizeof for a fast in-memory estimate to avoid blocking
    json.dumps on large payloads. Falls back to json.dumps for accuracy."""
    try:
        rough_size = sys.getsizeof(str(data)) if not isinstance(data, str) else len(data)
        if rough_size > MAX_PAYLOAD_SIZE * 3:
            return False
        if rough_size < MAX_PAYLOAD_SIZE // 2:
            return True
        return len(json.dumps(data)) <= MAX_PAYLOAD_SIZE
    except Exception:
        return False

def generate_analysis_id() -> str:
    """Generate unique analysis ID for tracking."""
    return hashlib.sha256(f"{time.time()}{os.urandom(16)}".encode()).hexdigest()[:12]

async def with_timeout_and_retry(func, *args, **kwargs):
    """Execute function with timeout and retry logic."""
    for attempt in range(LLM_RETRY_ATTEMPTS):
        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            metrics.record_timeout()
            if attempt == LLM_RETRY_ATTEMPTS - 1:
                raise
            await asyncio.sleep(LLM_BASE_BACKOFF * (2 ** attempt) + (0.1 * attempt))
        except Exception as e:
            if attempt == LLM_RETRY_ATTEMPTS - 1:
                raise
            await asyncio.sleep(LLM_BASE_BACKOFF * (2 ** attempt) + (0.1 * attempt))

class AgentState(TypedDict, total=False):
    """Represents the shared state that flows through the agent graph.
    
    Reducers:
    - take_last: Overwrites with newest value (for atomic fields)
    - merge_dicts: Deep merges dictionaries
    - _bounded_list_reducer (via partial): Dedupes and caps list fields (env STATE_MERGE_MAX_*)
    - merge_lists: Still used where unbounded concat is required
    - merge_names: Handles string/list name merging
    
    Note: Using named functions (not lambdas) enables state serialization
    for checkpointing and distributed execution.
    """
    # Routing field (CRITICAL: must be in TypedDict for LangGraph to preserve it)
    next: Annotated[str, take_last]  # Next node to route to (used by dispatcher_router)
    
    # Input fields
    resume_url: Annotated[str, take_last]
    jd_url: Annotated[str, take_last]
    uid: Annotated[str, take_last]
    tenant_id: Annotated[str, take_last]
    callback_url: Annotated[str, take_last]
    body: Annotated[Dict[str, Any], take_last]
    endpoint_name: Annotated[str, take_last]
    request_type: Annotated[str, take_last]  # e.g. "candidate_job_match" for compare flow
    user_interests: Annotated[List[Dict[str, Any]], take_last]
    is_second_call: Annotated[bool, take_last]
    assessment_plan: Annotated[List[Dict[str, Any]], take_last]
    job_id: Annotated[str, take_last]  # Compare flow: must be preserved through validate_resume -> groq -> interest_filler -> resume_assembler -> job_matcher_preprocessor
    assessment_id: str | None
    question_doc_id: str | None
    assessment_topic: str | None
    assessment_type: str | None

    # Data processing fields
    resume_text: Annotated[str, take_last]
    jd_text: Annotated[str, take_last]
    is_valid_resume: Annotated[bool, take_last]
    is_valid_jd: Annotated[bool, take_last]
    agent_segments: Annotated[Dict[str, str], take_last]
    name: Annotated[List[str], merge_names]
    total_experience_years: Annotated[str, take_last]
    contact_details: Annotated[dict, merge_dicts]
    education: Annotated[List[Any], _merge_education]
    work_experience: Annotated[List[Any], _merge_work_experience]
    skills: Annotated[List[Any], _merge_skills]
    certifications: Annotated[List[Any], _merge_certifications]
    projects: Annotated[List[Any], _merge_projects]
    extras: Annotated[List[Any], _merge_extras]
    structured_resume: Annotated[dict, merge_dicts]
    raw_skill_gap_analysis_output: Annotated[dict, merge_dicts]
    
    # Manual assessment builder fields
    manual_assessment_text: Annotated[str, lambda x, y: y]
    organization_type: Annotated[str, lambda x, y: y]
    structured_manual_assessment: Annotated[Dict[str, Any], lambda x, y: y]
    beautified_preview: Annotated[str, lambda x, y: y]
    # Enhanced role fit (2nd call): take_last — single declaration (duplicate TypedDict keys overwrite silently)
    enhanced_role_fit: Annotated[List[Dict[str, Any]], take_last]
    job_description: Annotated[Dict[str, Any], take_last]
    job_details: Annotated[Dict[str, Any], take_last]
    ranked_candidates: List[Dict[str, Any]]
    job_status: Annotated[Dict[str, Any], take_last]

    # Job Matcher fields (preprocessor populates these for job_matcher)
    job_ids: Annotated[List[str], take_last]
    job_docs: Annotated[List[Dict[str, Any]], take_last]
    optional_reviewer_qns: Annotated[List[Any], take_last]
    cached_results: Annotated[Dict[str, Any], take_last]
    top_matches: Annotated[List[Dict[str, Any]], take_last]
    job_matcher_status: Annotated[str, take_last]
    total_matches_found: Annotated[int, take_last]
    
    # Candidate-Job Matcher fields
    compare_job_only: Annotated[bool, take_last]  # True when /compare-candidate-job had no resume_url/resume_text → skip resume_scorer
    recruiter_questions: Annotated[List[Dict[str, Any]], take_last]
    interview_feedback: Annotated[Any, take_last]
    candidate_job_match_result: Annotated[Dict[str, Any], take_last]  # Parallel branches (career + job_matcher) may both write; take_last resolves conflict
    match_score: Annotated[float, take_last]
    skill_match_percentage: Annotated[float, take_last]
    skill_match_count: Annotated[int, take_last]
    skills_matched: Annotated[List[str], take_last]
    skills_unmatched: Annotated[List[str], take_last]
    total_required_skills: Annotated[int, take_last]
    rationale: Annotated[str, take_last]
    
    # Certificate update fields
    certificates_result: Annotated[Dict[str, Any], take_last]

    # Resume scoring fields
    resumeScore: Annotated[Dict[str, Any], merge_dicts]
    skills_without_supporting_evidence: Annotated[List[str], take_last]
    assessment_enriched_skills: Annotated[List[str], take_last]
    assessment_boost_applied: Annotated[bool, take_last]
    assessment_boost_details: Annotated[Dict[str, Any], merge_dicts]
    calculation_insights: Annotated[Dict[str, Any], merge_dicts]

    # Fields for feedback loop
    submission: Dict[str, Any]
    assessment_results: Annotated[Dict[str, Any], merge_dicts]
    generated_questions: Annotated[Dict[str, Any], merge_dicts]
    report: Annotated[Dict[str, Any], merge_dicts]

    # Corporate JD flow: prescreening questions (from prescreening_questions agent)
    prescreening_questions: Annotated[List[Dict[str, Any]], take_last]
    
    # Production fields
    confidence_score: Annotated[float, take_last]
    confidence_level: Annotated[str, take_last]
    processing_time_seconds: Annotated[float, take_last]
    
    # Error handling
    error: Annotated[str, take_last]
    validation_error: Annotated[str, take_last]
    security_error: Annotated[str, take_last]
    rate_limit_error: Annotated[str, take_last]
    
    # Session management fields
    session_id: Annotated[str, take_last]
    
    # Domain metadata (for domain-based retrieval in ranker)
    candidate_domains: Annotated[List[str], take_last]

    # Resume content generator fields
    resume_content: Annotated[str, take_last]
    context_used: Annotated[Dict[str, Any], merge_dicts]

    # Internal routing flags
    _job_matcher_parallel: Annotated[bool, take_last]  # True when job_matcher runs in parallel with career chain (Flow 1 / 2nd call)
    _compare_flow_multi_job_run: Annotated[bool, take_last]  # Legacy: 2nd compare preprocessor run (only used when no resume in body)
    _compare_primary_job_id: Annotated[str, take_last]  # Payload job_id for compare merged run (candidate_job_match_result target)
    _skip_callback: Annotated[bool, take_last]  # True when a node's output should not trigger a callback (fan-in guard no-op)

    # Agentic verification / observability fields
    agent_execution_trace: Annotated[List[Dict[str, Any]], _merge_agent_execution_trace]
    graph_steps_executed: Annotated[int, accumulate_int]
    agents_invoked: Annotated[List[str], _merge_agents_invoked]

async def validate_resume_node(state: AgentState) -> AgentState:
    log.info(f"🔍 VALIDATE_RESUME_DEBUG: validate_resume_node called with state keys: {list(state.keys())}")
    
    # For stored structured_resume flow, skip validation if we have structured_resume but no resume_text
    has_stored_resume = state.get("has_stored_resume", False)
    structured_resume = state.get("structured_resume")
    resume_text = state.get("resume_text")
    resume_url = state.get("resume_url")
    
    if has_stored_resume and structured_resume and not resume_text and not resume_url:
        log.info(f"🔍 VALIDATE_RESUME_DEBUG: Stored structured_resume detected, skipping text validation")
        return {
            "is_valid_resume": True,
            "is_resume": True,
            "resume_confidence": 1.0,
            "status": "completed",
            "node": "is_valid_resume",
            "output": True,
            "validation_metadata": {
                "method": "stored_resume_skip",
                "confidence": 1.0,
                "processing_time": 0.0,
                "reasons": ["✅ Stored structured_resume detected, validation skipped"]
            }
        }
    
    # If resume_url is provided but resume_text is not, extract using download_resume_text (same as JDs)
    # This has fallbacks (Mammoth, python-docx) for DOCX files, just like JDs
    if resume_url and not resume_text:
        log.info(f"🔍 VALIDATE_RESUME_DEBUG: Extracting resume text from URL: {resume_url}")
        try:
            # Use the same method as JDs - has fallbacks for DOCX files
            from utils.resume_utils import download_resume_text_async, _normalize_text
            import asyncio
            
            # Extract text using the same method as job descriptions (with DOCX fallbacks)
            log.info(f"📄 Extracting resume text with fallback support for DOCX/PDF/DOC files (same as JDs)")
            resume_text = await download_resume_text_async(resume_url, file_type="resume")
            
            if not resume_text or not resume_text.strip():
                log.error(f"❌ Resume text extraction failed - no text extracted from {resume_url}")
                return {
                    "is_valid_resume": False,
                    "validation_error": "Resume text extraction failed - no text extracted",
                    "error": "Resume text extraction failed"
                }
            
            # Normalize and quality check
            resume_text = _normalize_text(resume_text)
            quality = _quality_diag(resume_text)
            is_weak = is_text_weak(resume_text)
            
            log.info(f"✅ Resume extraction SUCCESS: len={quality['len']} words={quality['words']} "
                    f"alpha_ratio={quality['alpha_ratio']} weak={is_weak}")
            
            if is_weak:
                log.warning(f"⚠️ Resume text quality is weak. "
                          f"Length: {quality['len']} chars, Words: {quality['words']}, "
                          f"Alpha ratio: {quality['alpha_ratio']}")
            
            log.info(f"🔍 VALIDATE_RESUME_DEBUG: Successfully extracted resume text ({len(resume_text)} characters)")
            # Update state with extracted resume_text and preserve job_id
            state = {**state, "resume_text": resume_text}
            # Ensure job_id is preserved in state
            if "job_id" not in state or not state.get("job_id"):
                job_id = state.get("body", {}).get("job_id")
                if job_id:
                    state["job_id"] = job_id
        except Exception as e:
            log.error(f"🔍 VALIDATE_RESUME_DEBUG: Exception extracting resume: {e}")
            import traceback
            log.error(f"🔍 VALIDATE_RESUME_DEBUG: Traceback: {traceback.format_exc()}")
            return {
                "is_valid_resume": False,
                "validation_error": f"Gemini extraction failed: {str(e)}",
                "error": f"Gemini extraction failed: {str(e)}"
            }
    
    try:
        res = await validate_resume_agent(state)
        # Only return agent output + any new fields (resume_text if extracted, job_id if missing)
        result = dict(res) if isinstance(res, dict) else {}
        if resume_text and resume_text != state.get("resume_text"):
            result["resume_text"] = resume_text
        job_id = result.get("job_id") or state.get("job_id") or state.get("body", {}).get("job_id")
        if job_id:
            result["job_id"] = job_id
        return result
    except Exception as e:
        log.error(f"🔍 VALIDATE_RESUME_DEBUG: Exception in validate_resume_node: {e}")
        import traceback
        log.error(f"🔍 VALIDATE_RESUME_DEBUG: Traceback: {traceback.format_exc()}")
        return {
            "is_valid_resume": False,
            "validation_error": str(e),
            "error": f"Validate resume node error: {str(e)}"
        }

def _run_async_in_thread(async_func, state):
    """Run async function in a separate thread with its own event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_func(state))
    finally:
        loop.close()


# Retained for backward-compat (app.py / tests import it for shutdown).
# create_sync_wrapper is no longer called -- keep pool minimal to avoid wasting OS threads.
_sync_wrapper_max_workers = int(os.getenv("SYNC_WRAPPER_MAX_WORKERS", "2"))
_sync_wrapper_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_sync_wrapper_max_workers, thread_name_prefix="sync_wrapper"
)
_SYNC_WRAPPER_TIMEOUT = int(os.getenv("SYNC_WRAPPER_TIMEOUT_SECONDS", "300"))

# Issue 6.4: Track orphaned tasks for monitoring
_orphaned_task_count = 0
_orphaned_task_lock = threading.Lock()


def _cleanup_sync_wrapper_executor():
    """Gracefully shutdown the executor on application exit."""
    log.info("Shutting down sync_wrapper_executor...")
    try:
        _sync_wrapper_executor.shutdown(wait=True, cancel_futures=True)
        log.info("sync_wrapper_executor shutdown complete")
    except Exception as e:
        log.error(f"Error during sync_wrapper_executor shutdown: {e}")


# Issue 6.4: Register atexit hook for graceful shutdown
atexit.register(_cleanup_sync_wrapper_executor)


def create_sync_wrapper(async_agent_func):
    """Create a synchronous wrapper for any async agent function with adaptive learning."""
    def sync_wrapper(state: AgentState) -> AgentState:
        global _orphaned_task_count
        agent_name = async_agent_func.__name__.replace("_agent", "")
        input_state = state.copy()
        
        try:
            start_time = time.time()
            future = _sync_wrapper_executor.submit(_run_async_in_thread, async_agent_func, state)
            result = future.result(timeout=_SYNC_WRAPPER_TIMEOUT)
            
            end_time = time.time()
            timing_seconds = round(max(0.0, end_time - start_time), 2)
            
            # Return only the agent's output (changed fields).  Do NOT merge
            # {**state, **result} — that re-applies merge_lists reducers and doubles
            # education/work_experience/skills/etc.
            learn_from_agent_execution(agent_name, input_state, result)
            
            log.warning(f"{agent_name} completed in {timing_seconds} seconds")
            
            return result
        except concurrent.futures.TimeoutError:
            # Issue 6.4: Attempt to cancel the future and track orphaned tasks
            cancelled = future.cancel()
            with _orphaned_task_lock:
                _orphaned_task_count += 1
                orphan_count = _orphaned_task_count
            
            log.error(
                f"Agent {agent_name} timed out after {_SYNC_WRAPPER_TIMEOUT}s "
                f"(cancel attempted: {cancelled}, total orphaned: {orphan_count})"
            )
            error_state = {"error": f"Agent timed out after {_SYNC_WRAPPER_TIMEOUT} seconds"}
            learn_from_agent_execution(agent_name, input_state, error_state)
            return error_state
        except Exception as e:
            log.error(f"Error in {async_agent_func.__name__}: {e}")
            error_state = {"error": str(e)}
            learn_from_agent_execution(agent_name, input_state, error_state)
            return error_state
    return sync_wrapper

# Sync wrappers removed: all agents are async and registered via apply_middleware
# which handles them natively. The sync wrappers were unused in the graph and
# consumed threads from _sync_wrapper_executor unnecessarily at import time.

def _merge_resume_content_context_data(
    uid: str,
    session_id: Optional[str],
    tenant_id: str,
    resume_doc: dict,
    found_session_id_hint: Any,
    assessments_doc: dict,
    gap_doc: dict,
    user_profile: Any,
) -> dict:
    """Build context_data dict for resume_content_generator hashing (sync; may call Chroma)."""
    from chroma import get_chat_session, fetch_structured_resume, find_session_by_uid, get_resume_score_doc

    context_data: dict = {}
    structured_resume: dict = {}

    try:
        structured_resume = (resume_doc or {}).get("structured_resume") or {}
        if not structured_resume or not structured_resume.get("Name"):
            if session_id:
                structured_resume = fetch_structured_resume(uid, None, session_id) or {}
        if not structured_resume or not structured_resume.get("Name"):
            found_session_id = found_session_id_hint or find_session_by_uid(uid)
            if found_session_id:
                structured_resume = fetch_structured_resume(uid, None, found_session_id) or {}
    except Exception:
        pass

    if structured_resume and structured_resume.get("Name"):
        context_data["structured_resume"] = {
            "Name": structured_resume.get("Name"),
            "ContactDetails": structured_resume.get("ContactDetails"),
            "education": structured_resume.get("education"),
            "experience": structured_resume.get("experience"),
            "skills": structured_resume.get("skills"),
            "certifications": structured_resume.get("certifications"),
            "projects": structured_resume.get("projects"),
            "total_experience_years": structured_resume.get("total_experience_years"),
        }
        user_interests_summary = structured_resume.get("user_interests_summary")
        if user_interests_summary and isinstance(user_interests_summary, str) and user_interests_summary.strip():
            interests_list = [
                {"answer": item.strip()}
                for item in user_interests_summary.split(";")
                if item.strip()
            ]
            if interests_list:
                context_data["user_interests"] = interests_list

    session_data = None
    if session_id:
        try:
            session_data = get_chat_session(session_id, uid)
        except Exception:
            pass

    if not session_data:
        try:
            fsid = found_session_id_hint or find_session_by_uid(uid)
            if fsid:
                session_data = get_chat_session(fsid, uid)
        except Exception:
            pass

    if session_data:
        interest_filler = session_data.get("interest_filler", {})
        if interest_filler:
            detailed_interests = interest_filler.get("detailed_interests", [])
            if detailed_interests and isinstance(detailed_interests, list) and len(detailed_interests) > 0:
                if not context_data.get("user_interests"):
                    context_data["user_interests"] = detailed_interests
            else:
                user_interests = interest_filler.get("user_interests", [])
                if user_interests and isinstance(user_interests, list) and len(user_interests) > 0:
                    if not context_data.get("user_interests"):
                        if isinstance(user_interests[0], str):
                            user_interests = [{"answer": item} for item in user_interests if item]
                        context_data["user_interests"] = user_interests

        if "career_advisor" in session_data:
            career_advisor = session_data.get("career_advisor", {})
            if career_advisor:
                context_data["career_advisor"] = career_advisor

        if "market_and_course_recommender" in session_data:
            market_course = session_data.get("market_and_course_recommender", {})
            if market_course:
                context_data["market_and_course_recommender"] = market_course

        if "resume_score" in session_data:
            resume_score = session_data.get("resume_score", {})
            if resume_score:
                context_data["resume_score"] = resume_score

        if not context_data.get("resume_score") and uid:
            try:
                rs_doc = get_resume_score_doc(uid)
                if rs_doc and isinstance(rs_doc, dict) and rs_doc:
                    context_data["resume_score"] = rs_doc
            except Exception:
                pass

        if "resume_analysis" in session_data:
            resume_analysis = session_data.get("resume_analysis", {})
            if resume_analysis:
                context_data["resume_analysis"] = resume_analysis

        if "report_generator" in session_data:
            report_generator_data = session_data.get("report_generator", {})
            if report_generator_data and isinstance(report_generator_data, dict) and len(report_generator_data) > 0:
                report = report_generator_data.get("report")
                if report and isinstance(report, dict) and len(report) > 0:
                    context_data["report"] = report

        if "assessment_evaluator" in session_data:
            assessment_evaluator_data = session_data.get("assessment_evaluator", {})
            if assessment_evaluator_data:
                context_data["assessment_evaluator"] = assessment_evaluator_data

    adoc = assessments_doc or {}
    if adoc:
        assessment_history = adoc.get("assessment_history", [])
        if assessment_history and isinstance(assessment_history, list) and len(assessment_history) > 0:
            latest_assessment = assessment_history[-1] if assessment_history else {}
            if latest_assessment and isinstance(latest_assessment, dict):
                if not context_data.get("assessment_evaluator"):
                    result = latest_assessment.get("result", {})
                    if result and isinstance(result, dict):
                        context_data["assessment_evaluator"] = {"assessment_results": result}

    gdoc = gap_doc or {}
    if gdoc:
        if not context_data.get("career_advisor") and gdoc.get("career_advisor"):
            context_data["career_advisor"] = gdoc.get("career_advisor", {})
        if not context_data.get("market_and_course_recommender") and gdoc.get("market_and_course_recommender"):
            context_data["market_and_course_recommender"] = gdoc.get("market_and_course_recommender", {})
        if not context_data.get("resume_score") and gdoc.get("resume_score"):
            context_data["resume_score"] = gdoc.get("resume_score", {})

    if user_profile:
        if not context_data.get("career_advisor") and user_profile.get("career_advisor"):
            context_data["career_advisor"] = user_profile.get("career_advisor", {})
        if not context_data.get("market_and_course_recommender") and user_profile.get("market_and_course_recommender"):
            context_data["market_and_course_recommender"] = user_profile.get("market_and_course_recommender", {})
        if not context_data.get("resume_score") and user_profile.get("resume_score"):
            context_data["resume_score"] = user_profile.get("resume_score", {})

    return context_data


def _hash_resume_context_data(context_data: dict, uid: str, session_id: Optional[str]) -> str:
    import hashlib as _hashlib
    import json as _json

    try:
        context_json = _json.dumps(context_data, sort_keys=True, default=str)
        return _hashlib.sha256(context_json.encode()).hexdigest()
    except Exception as e:
        log.warning(f"Error computing context hash for resume_content_generator: {e}")
        return _hashlib.sha256(f"error_{uid}_{session_id}_{time.time()}".encode()).hexdigest()


_resume_context_hash_cache: Dict[Tuple[str, Optional[str], str], Tuple[float, str]] = {}
_resume_context_hash_cache_lock = threading.Lock()


def _compute_resume_content_context_hash(uid: str, session_id: Optional[str] = None, tenant_id: str = "default_tenant") -> str:
    """
    Compute a hash of all context data used by resume_content_generator (sync; sequential prefetch).
    """
    try:
        from chroma import get_resume_doc, find_session_by_uid, get_assessments_doc, get_gap_doc, get_user_profile

        resume_doc = get_resume_doc(uid) or {}
        found_sid = find_session_by_uid(uid)
        assessments_doc = get_assessments_doc(uid) or {}
        gap_doc = get_gap_doc(uid) or {}
        user_profile = get_user_profile(uid)
        ctx = _merge_resume_content_context_data(
            uid, session_id, tenant_id, resume_doc, found_sid, assessments_doc, gap_doc, user_profile
        )
        return _hash_resume_context_data(ctx, uid, session_id)
    except Exception as e:
        log.warning(f"Error computing context hash for resume_content_generator: {e}")
        return hashlib.sha256(f"error_{uid}_{session_id}_{time.time()}".encode()).hexdigest()


async def _compute_resume_content_context_hash_async(
    uid: str, session_id: Optional[str] = None, tenant_id: str = "default_tenant"
) -> str:
    """Parallel Chroma prefetch + merge/hash; optional TTL cache via RESUME_CONTEXT_HASH_CACHE_TTL_SEC."""
    ttl = 0
    try:
        ttl = max(0, int(os.getenv("RESUME_CONTEXT_HASH_CACHE_TTL_SEC", "0")))
    except ValueError:
        ttl = 0
    cache_key = (uid, session_id, tenant_id)
    if ttl > 0:
        now = time.time()
        with _resume_context_hash_cache_lock:
            ent = _resume_context_hash_cache.get(cache_key)
            if ent and now - ent[0] < ttl:
                return ent[1]

    try:
        from chroma import get_resume_doc, find_session_by_uid, get_assessments_doc, get_gap_doc, get_user_profile

        rd, fs, ad, gd, up = await asyncio.gather(
            asyncio.to_thread(lambda: get_resume_doc(uid) or {}),
            asyncio.to_thread(find_session_by_uid, uid),
            asyncio.to_thread(lambda: get_assessments_doc(uid) or {}),
            asyncio.to_thread(lambda: get_gap_doc(uid) or {}),
            asyncio.to_thread(get_user_profile, uid),
            return_exceptions=True,
        )

        def _norm(val: Any, default):
            if isinstance(val, Exception):
                log.debug(f"resume context hash prefetch skipped: {val}")
                return default
            return val if val is not None else default

        resume_doc = _norm(rd, {})
        found_sid = None if isinstance(fs, Exception) else fs
        assessments_doc = _norm(ad, {})
        gap_doc = _norm(gd, {})
        user_profile = None if isinstance(up, Exception) else up

        ctx = await asyncio.to_thread(
            _merge_resume_content_context_data,
            uid,
            session_id,
            tenant_id,
            resume_doc,
            found_sid,
            assessments_doc,
            gap_doc,
            user_profile,
        )
        h = _hash_resume_context_data(ctx, uid, session_id)
        if ttl > 0:
            with _resume_context_hash_cache_lock:
                _resume_context_hash_cache[cache_key] = (time.time(), h)
                while len(_resume_context_hash_cache) > 2000:
                    _resume_context_hash_cache.pop(next(iter(_resume_context_hash_cache)), None)
        return h
    except Exception as e:
        log.warning(f"Error computing context hash (async) for resume_content_generator: {e}")
        return hashlib.sha256(f"error_{uid}_{session_id}_{time.time()}".encode()).hexdigest()

def _extract_resume_data_from_session(session_id: str, uid: str) -> Tuple[Optional[Dict], list]:
    """
    Extracts structured_resume and user_interests from session data.
    """
    structured_resume = None
    user_interests = []

    if not session_id or not uid:
        return structured_resume, user_interests

    try:
        from chroma import get_chat_session, fetch_structured_resume
        
        structured_resume = fetch_structured_resume(uid, current_session_id=session_id)

        session_data = get_chat_session(session_id)
        if session_data:
            interest_data = session_data.get("interest_filler", {})
            user_interests = interest_data.get("user_interests", [])

    except Exception as e:
        log.error(f"❌ Error extracting resume data from session {session_id} for uid {uid}: {e}")

    return structured_resume, user_interests


def _resume_has_no_skills_hydration(sr: Any) -> bool:
    """True if resume lacks usable skills (career flow may only have user_interests_summary)."""
    if not sr or not isinstance(sr, dict):
        return True
    try:
        from agents.skill_utils import extract_primary_skills
        return not extract_primary_skills(sr)
    except Exception:
        return False


def _hydration_session_patch(
    state: AgentState,
    session_data: dict,
    *,
    structured_resume_fetched: Any,
    assessments_doc: Optional[dict],
    force_hydration: bool,
    session_id: str,
) -> Dict[str, Any]:
    """Build hydration fields from session blob and optional parallel-fetched docs. Pure (no I/O)."""
    hydrated_data: Dict[str, Any] = {}
    uid = state.get("uid")

    if structured_resume_fetched is not None:
        sr = structured_resume_fetched
        if sr and (force_hydration or not _resume_has_no_skills_hydration(sr)):
            hydrated_data["structured_resume"] = sr
            log.info(f"✅ Hydrated structured_resume from session {session_id}")

    if force_hydration or not state.get("user_interests"):
        interest_data = session_data.get("interest_filler", {})
        user_interests = interest_data.get("user_interests", [])
        if user_interests:
            hydrated_data["user_interests"] = user_interests
            log.info(f"✅ Hydrated {len(user_interests)} user_interests from session")

    if force_hydration or not state.get("raw_skill_gap_analysis_output"):
        prior_gap = (
            session_data.get("career_advisor", {}).get("raw_skill_gap_analysis_output", {})
            or session_data.get("raw_skill_gap_analysis_output", {})
        )
        if prior_gap:
            hydrated_data["raw_skill_gap_analysis_output"] = prior_gap
            log.info("✅ Hydrated raw_skill_gap_analysis_output from session")

    if force_hydration or not state.get("assessment_results"):
        assessment_results = session_data.get("assessment_evaluator", {}).get("assessment_results")
        if assessment_results:
            hydrated_data["assessment_results"] = assessment_results
            log.info("✅ Hydrated assessment_results from session")

    if force_hydration or not state.get("report"):
        report = session_data.get("report_generator", {}).get("report")
        if report:
            hydrated_data["report"] = report
            log.info("✅ Hydrated report from session")

    if force_hydration or not state.get("generated_questions"):
        generated_questions = session_data.get("assessment_question_generator", {}).get("generated_questions")
        if generated_questions:
            hydrated_data["generated_questions"] = generated_questions
            log.info("✅ Hydrated generated_questions from session")

    if force_hydration or not state.get("prior_assessment_plan"):
        assessment_recommender_data = session_data.get("assessment_recommender", {})
        if assessment_recommender_data:
            hydrated_data["prior_assessment_plan"] = assessment_recommender_data.get("assessment_plan", [])
            hydrated_data["prior_assessment_needs"] = assessment_recommender_data.get("assessment_needs", {})
            log.info("✅ Hydrated prior_assessment_plan from session")

    if assessments_doc is not None and uid:
        try:
            assessment_history = assessments_doc.get("assessment_history", [])
            if assessment_history:
                hydrated_data["assessment_history"] = assessment_history
                hydrated_data["consecutive_passes"] = state.get("consecutive_passes", 0)
                hydrated_data["consecutive_fails"] = state.get("consecutive_fails", 0)
                log.info(f"✅ Hydrated assessment_history from uid_assessments ({len(assessment_history)} entries)")
        except Exception as e:
            log.warning(f"⚠️ Could not hydrate assessment_history: {e}")

    if force_hydration or not state.get("top_matches"):
        job_matcher_data = session_data.get("job_matcher", {})
        if job_matcher_data:
            hydrated_data["top_matches"] = job_matcher_data.get("top_matches", [])
            hydrated_data["job_matcher_status"] = job_matcher_data.get("job_matcher_status", "")
            log.info("✅ Hydrated job_matcher data from session")

    return hydrated_data


async def _hydrate_state_from_session_async(
    state: AgentState, force_hydration: bool = False
) -> AgentState:
    """
    Hydrate state from session without mutating the input mapping.
    Loads session first, then runs independent Chroma reads in parallel via asyncio.to_thread.
    """
    session_id = state.get("session_id")
    uid = state.get("uid")
    if not session_id or not uid:
        return dict(state)

    base = dict(state)
    try:
        from chroma import get_chat_session, fetch_structured_resume, get_assessments_doc

        session_data = await asyncio.to_thread(get_chat_session, session_id)
        session_data = session_data or {}
        if not session_data:
            return base

        need_sr = (
            force_hydration
            or not base.get("structured_resume")
            or _resume_has_no_skills_hydration(base.get("structured_resume"))
        )
        need_ad = force_hydration or not base.get("assessment_history")

        async def _fetch_sr():
            return await asyncio.to_thread(
                fetch_structured_resume, uid, current_session_id=session_id
            )

        async def _fetch_ad():
            return await asyncio.to_thread(get_assessments_doc, uid)

        sr_task = _fetch_sr() if need_sr else asyncio.sleep(0, result=None)
        ad_task = _fetch_ad() if (need_ad and uid) else asyncio.sleep(0, result=None)
        sr_res, ad_res = await asyncio.gather(sr_task, ad_task, return_exceptions=True)

        structured_resume_fetched = None
        if need_sr:
            if isinstance(sr_res, Exception):
                log.warning(f"⚠️ fetch_structured_resume failed during hydration: {sr_res}")
            else:
                structured_resume_fetched = sr_res

        assessments_doc: Optional[dict] = None
        if need_ad and uid:
            if isinstance(ad_res, Exception):
                log.warning(f"⚠️ get_assessments_doc failed during hydration: {ad_res}")
                assessments_doc = {}
            else:
                assessments_doc = ad_res or {}

        hydrated_data = _hydration_session_patch(
            base,
            session_data,
            structured_resume_fetched=structured_resume_fetched,
            assessments_doc=assessments_doc,
            force_hydration=force_hydration,
            session_id=session_id,
        )
        if not hydrated_data:
            return base
        hydrated_data.pop("next", None)
        log.info(f"✅ State hydration complete: added {len(hydrated_data)} fields")
        return {**base, **hydrated_data}
    except Exception as e:
        log.error(f"❌ Error hydrating state from session {session_id}: {e}")
        import traceback
        log.error(traceback.format_exc())
        return base


def _sync_hydrate_state_via_async(st: AgentState, force_hydration: bool = False) -> AgentState:
    """Run async hydrator from sync `graph.invoke` (no running loop, or isolated thread if loop exists)."""
    base = dict(st)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_hydrate_state_from_session_async(base, force_hydration))
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(
            asyncio.run, _hydrate_state_from_session_async(base, force_hydration)
        ).result(timeout=300)


def _hydrate_state_from_session(state: AgentState, force_hydration: bool = False) -> AgentState:
    """
    Sync hydration (backward compatible). Prefer _hydrate_state_from_session_async on the event loop.
    Returns a new state dict; does not mutate the input.
    """
    session_id = state.get("session_id")
    uid = state.get("uid")
    if not session_id or not uid:
        return dict(state)

    base = dict(state)
    try:
        from chroma import get_chat_session, fetch_structured_resume, get_assessments_doc

        session_data = get_chat_session(session_id) or {}
        if not session_data:
            return base

        need_sr = (
            force_hydration
            or not base.get("structured_resume")
            or _resume_has_no_skills_hydration(base.get("structured_resume"))
        )
        structured_resume_fetched = (
            fetch_structured_resume(uid, current_session_id=session_id) if need_sr else None
        )

        need_ad = force_hydration or not base.get("assessment_history")
        assessments_doc: Optional[dict] = None
        if need_ad and uid:
            try:
                assessments_doc = get_assessments_doc(uid) or {}
            except Exception as e:
                log.warning(f"⚠️ Could not load assessments_doc: {e}")
                assessments_doc = {}

        hydrated_data = _hydration_session_patch(
            base,
            session_data,
            structured_resume_fetched=structured_resume_fetched,
            assessments_doc=assessments_doc,
            force_hydration=force_hydration,
            session_id=session_id,
        )
        if not hydrated_data:
            return base
        hydrated_data.pop("next", None)
        log.info(f"✅ State hydration complete: added {len(hydrated_data)} fields")
        return {**base, **hydrated_data}
    except Exception as e:
        log.error(f"❌ Error hydrating state from session {session_id}: {e}")
        import traceback
        log.error(traceback.format_exc())
        return base

@traceable(name="supervisor_dispatcher")
async def dispatcher(state: AgentState) -> dict:
    """
    Production-ready dispatcher with proper coalescing support.
    Now async so coalescing waits yield to the event loop instead of blocking threads.
    """
    from core.utils import run_blocking_io as _rbi
    global _inflight_keys
    start_time = time.time()
    analysis_id = generate_analysis_id()
    
    # REMOVED: Don't hydrate here - it's already done in production_invoke/ainvoke
    # Hydration happens BEFORE graph execution, not during graph execution
    # Mutating state during graph execution can cause conflicts with 'next' field
    # State is already hydrated when it reaches dispatcher
    
    # Step 1: Security Guards & Input Validation
    body = state.get("body", {})
    company = body.get("company", "")
    new_session = None
    resume_text = ""
    uid = body.get("uid", "")
    tenant_id = body.get("tenant_id") or body.get("uid", "default-tenant")
    
    # Early detection of job matching requests (needed for cache skip logic)
    # Used to skip caching for real-time job matching requests
    is_candidate_job_match = (
        body.get("request_type") == "candidate_job_match" or 
        (body.get("uid") and body.get("job_id") and not body.get("resume_url") and not body.get("jd_url") and not body.get("resume_text") and not body.get("job_details"), not body.get("company"), not body.get("job_status"))
    )
    
    # Validate tenant and user IDs
    if not validate_tenant_id(tenant_id):
        log.error(f"Invalid tenant_id format: {tenant_id}")
        return {
            "next": "end",
            "error": "Invalid tenant ID format",
            "security_error": "Invalid tenant ID format",
            "analysis_id": analysis_id
        }
    
    # Allow uid to be empty/None for stored resume flow (structured_resume in body with job_id)
    # Otherwise, validate uid format
    allow_empty_uid = (
        not uid and 
        body.get("job_id") and 
        body.get("structured_resume") and 
        not body.get("resume_url") and 
        not body.get("resume_text")
    )
    
    if uid and not validate_user_id(uid):
        log.error(f"Invalid user_id format: {uid}")
        return {
            "next": "end", 
            "error": "Invalid user ID format",
            "security_error": "Invalid user ID format",
            "analysis_id": analysis_id
        }
    
    if not uid and not allow_empty_uid:
        # uid is required unless it's the stored resume flow
        log.warning(f"Missing uid and not a stored resume flow")
        # Don't return error here - let the flow continue and handle it downstream
    
    # Validate payload size
    if not validate_payload_size(body):
        log.error(f"Payload size exceeds limit for tenant: {tenant_id}")
        return {
            "next": "end",
            "error": "Payload size exceeds limit",
            "security_error": "Payload size exceeds limit",
            "analysis_id": analysis_id
        }
    
    # Step 2: Rate Limiting (per-tenant)
    rate_limiter = get_rate_limiter(tenant_id)
    if not rate_limiter.consume():
        log.warning(f"Rate limit exceeded for tenant: {tenant_id}")
        metrics.record_method("rate_limited")
        return {
            "next": "end",
            "error": "Rate limit exceeded",
            "rate_limit_error": "Rate limit exceeded",
            "analysis_id": analysis_id
        }
    
    # Step 3: Early check for resume_content_generator requests (before cache)
    # Context-aware caching: Only use cache if context hasn't changed
    # Context includes: structured_resume, user_interests, career_advisor, 
    # market_and_course_recommender, assessment_recommender, resume_score, resume_analysis
    resume_url_early = body.get("resume_url", "")
    resume_text_early = body.get("resume_text", "")
    job_id_early = body.get("job_id")
    jd_url_early = body.get("jd_url", "")
    
    is_resume_content_generator_request = (
        uid and 
        not resume_url_early and 
        not resume_text_early and 
        not job_id_early and 
        not jd_url_early
    )
    
    # Compute context hash for resume_content_generator (for context-aware caching)
    resume_content_context_hash = None
    if is_resume_content_generator_request:
        session_id_for_hash = body.get("session_id") or state.get("session_id")
        try:
            resume_content_context_hash = await _compute_resume_content_context_hash_async(
                uid, session_id_for_hash, tenant_id
            )
            log.info(f"Computed context hash for resume_content_generator: {resume_content_context_hash[:16]}...")
        except Exception as e:
            log.warning(f"Failed to compute context hash, will skip cache: {e}")
            resume_content_context_hash = None  # Skip cache on error
    
    # Step 4: Build cache key
    cache_key_data = {"uid": uid}
    request_type = "resume_analysis"
    
    # Add user_interests hash to cache key
    user_interests = body.get("user_interests")
    if user_interests and isinstance(user_interests, list) and len(user_interests) > 0:
        try:
            import hashlib, json as _json
            if isinstance(user_interests[0], dict) and "answer" in user_interests[0]:
                user_interests_for_hash = [item.get("answer", "") for item in user_interests if item.get("answer")]
            else:
                user_interests_for_hash = user_interests
            interests_hash = hashlib.sha256(_json.dumps(user_interests_for_hash, sort_keys=True).encode()).hexdigest()
            cache_key_data["user_interests_hash"] = interests_hash
        except Exception as e:
            log.warning(f"Failed to hash user_interests for cache key: {e}")
            cache_key_data["user_interests_present"] = True

    # Add job_details hash to cache key
    if body.get("job_details"):
        request_type = "job_ranking"
        cache_key_data["request_type"] = request_type
        try:
            import hashlib, json as _json
            job_details = body.get("job_details", {})
            job_details_str = _json.dumps(job_details, sort_keys=True)
            job_hash = hashlib.sha256(job_details_str.encode()).hexdigest()
            cache_key_data["job_details_hash"] = job_hash
            if body.get("job_id"):
                cache_key_data["job_id"] = body.get("job_id")
        except Exception as e:
            log.warning(f"Failed to hash job_details: {e}")
            cache_key_data["job_details_present"] = True
    
    elif body.get("plan"):
        request_type = "assessment_generation"
        cache_key_data["request_type"] = request_type
        try:
            import hashlib, json as _json
            plan_hash = hashlib.sha256(_json.dumps(body.get("plan"), sort_keys=True).encode()).hexdigest()
            cache_key_data["assessment_plan_hash"] = plan_hash
        except Exception:
            cache_key_data["assessment_plan"] = body.get("plan")
    elif body.get("submission"):
        request_type = "assessment_evaluation"
        cache_key_data["request_type"] = request_type
        try:
            import hashlib, json as _json
            sub_hash = hashlib.sha256(_json.dumps(body.get("submission"), sort_keys=True).encode()).hexdigest()
            cache_key_data["submission_hash"] = sub_hash
        except Exception:
            cache_key_data["submission_present"] = True
    elif body.get("jd_url"):
        request_type = "job_description"
        cache_key_data["request_type"] = request_type
        try:
            import hashlib, json as _json
            jd_url = body.get("jd_url", "")
            jd_hash = hashlib.sha256(jd_url.encode()).hexdigest()
            cache_key_data["jd_url_hash"] = jd_hash
            if body.get("job_id"):
                cache_key_data["job_id"] = body.get("job_id")
        except Exception as e:
            log.warning(f"Failed to hash JD URL for cache key: {e}")
            cache_key_data["jd_url_present"] = True

    else:
        cache_key_data["request_type"] = request_type
        resume_url = body.get("resume_url", "")
        if resume_url:
            try:
                import hashlib
                resume_hash = hashlib.sha256(resume_url.encode()).hexdigest()
                cache_key_data["resume_url_hash"] = resume_hash
            except Exception as e:
                log.warning(f"Failed to hash resume URL for cache key: {e}")
                cache_key_data["resume_url_present"] = True

    # Generate cache key fingerprint
    try:
        import json as _json, hashlib as _hashlib
        key_preview_source = {k: cache_key_data.get(k) for k in sorted(cache_key_data.keys()) if k in {"request_type","uid","jd_url_hash","assessment_plan_hash","submission_hash","resume_url_hash","user_interests_hash","job_details_hash", "job_id"}}
        key_fingerprint = _hashlib.sha256(_json.dumps(key_preview_source, sort_keys=True).encode()).hexdigest()[:12]
    except Exception:
        key_fingerprint = "unknown"

    # ✅ FIX: Enhanced request coalescing with result sharing
    key_id = f"{tenant_id}:{key_fingerprint}"
    current_time = time.time()
    
    # Quick check with minimal lock time - separate coalesce decision from lock
    should_wait = False
    wait_event = None
    
    try:
        with _inflight_lock:
            # Defensive check: ensure _inflight_keys is a dict
            if not isinstance(_inflight_keys, dict):
                log.error(f"⚠️ _inflight_keys is not a dict! Type: {type(_inflight_keys)}, value: {_inflight_keys}")
                # Reinitialize as dict if it got corrupted
                _inflight_keys = {}
            
            if key_id in _inflight_keys:
                inflight_data = _inflight_keys[key_id]
                original_start = inflight_data["timestamp"]
                
                # Skip in-flight coalescing for job_matcher and resume_content_generator (always fresh)
                if is_candidate_job_match:
                    log.info("Skipping in-flight coalescing for job_matcher request")
                    should_wait = False
                elif is_resume_content_generator_request:
                    log.info("Skipping in-flight coalescing for resume_content_generator request (always generate fresh)")
                    should_wait = False
                # Check if original request exceeded SLA (stale request)
                elif current_time - original_start > _REQUEST_SLA_SECONDS:
                    # Remove stale request and allow new one
                    del _inflight_keys[key_id]
                    log.warning(f"Removed stale in-flight request: {key_id} (age: {current_time - original_start:.1f}s)")
                else:
                    # Check if result is already available
                    # Skip for job_matcher and resume_content_generator (always fresh)
                    if inflight_data["result"] is not None and not is_candidate_job_match and not is_resume_content_generator_request:
                        cached_result = inflight_data["result"]
                        log.debug(f"COALESCE_RESULT_READY pid={os.getpid()} tenant_id={tenant_id} uid={uid} request_type={request_type} key_fp={key_fingerprint}")
                        
                        # Determine next node based on current request state (not cached state)
                        # Check if user_interests are present in current request
                        user_interests = body.get("user_interests")
                        user_interests_normalized = None
                        if user_interests and isinstance(user_interests, list) and len(user_interests) > 0:
                            if isinstance(user_interests[0], dict) and "answer" in user_interests[0]:
                                user_interests_normalized = [item.get("answer", "") for item in user_interests if item.get("answer")]
                            else:
                                user_interests_normalized = user_interests
                        
                        # Route based on current request's user_interests, not cached result
                        if user_interests_normalized and len(user_interests_normalized) > 0:
                            next_node = "groq_resume_parser"
                        else:
                            # Use cached next or default to validate_resume for resume processing
                            next_node = cached_result.get("next", "validate_resume")
                        
                        metrics.record_cache_hit()
                        metrics.record_method("cached")
                        return {
                            **cached_result,
                            "analysis_id": analysis_id,
                            "analysis_method": "cached",
                            "processing_time_seconds": time.time() - start_time,
                            "next": next_node,
                            # Ensure user_interests are included if present in current request
                            "user_interests": user_interests_normalized if user_interests_normalized else cached_result.get("user_interests")
                        }
                    elif (is_candidate_job_match or is_resume_content_generator_request) and inflight_data["result"] is not None:
                        if is_candidate_job_match:
                            log.info("Skipping in-flight cached result for job_matcher request (real-time comparison)")
                        else:
                            log.info("Skipping in-flight cached result for resume_content_generator request (context-aware caching uses main cache)")
                        should_wait = False
                    else:
                        # Wait for the original request to complete
                        should_wait = True
                        wait_event = inflight_data["event"]
                        log.debug(f"COALESCE_WAIT pid={os.getpid()} tenant_id={tenant_id} uid={uid} request_type={request_type} key_fp={key_fingerprint}")
            
            # Don't set up tracking here - will be done after wait logic below
            # This prevents duplicate tracking setup
    except (NameError, AttributeError, TypeError) as e:
        log.warning(f"⚠️ Error in in-flight tracking system: {e}, continuing without coalescing")
        # If tracking fails, allow the request to proceed normally
        should_wait = False
        wait_event = None
    except Exception as e:
        log.error(f"⚠️ Unexpected error in in-flight tracking: {e}, continuing without coalescing")
        # If tracking fails, allow the request to proceed normally
        should_wait = False
        wait_event = None
    
    # ✅ If we should wait, wait for the original request to complete
    # Skip in-flight coalescing for job_matcher and resume_content_generator (always fresh)
    if should_wait and wait_event and not is_candidate_job_match and not is_resume_content_generator_request:
        log.debug(f"COALESCE_WAITING pid={os.getpid()} tenant_id={tenant_id} uid={uid} request_type={request_type} key_fp={key_fingerprint}")
        # Async wait -- yields to event loop instead of blocking a thread
        try:
            await asyncio.wait_for(wait_event.wait(), timeout=_REQUEST_SLA_SECONDS)
            wait_success = True
        except asyncio.TimeoutError:
            wait_success = False
        
        if wait_success:
            # TOCTOU: inflight entry may have been popped after wait; missing key/result => treat as miss below.
            with _inflight_lock:
                if key_id in _inflight_keys and _inflight_keys[key_id]["result"] is not None:
                    cached_result = _inflight_keys[key_id]["result"]
                    log.debug(f"COALESCE_SUCCESS pid={os.getpid()} tenant_id={tenant_id} uid={uid} request_type={request_type} key_fp={key_fingerprint}")
                    
                    # Determine next node based on current request state (not cached state)
                    # Check if user_interests are present in current request
                    user_interests = body.get("user_interests")
                    user_interests_normalized = None
                    if user_interests and isinstance(user_interests, list) and len(user_interests) > 0:
                        if isinstance(user_interests[0], dict) and "answer" in user_interests[0]:
                            user_interests_normalized = [item.get("answer", "") for item in user_interests if item.get("answer")]
                        else:
                            user_interests_normalized = user_interests
                    
                    # Route based on current request's user_interests, not cached result
                    if user_interests_normalized and len(user_interests_normalized) > 0:
                        next_node = "groq_resume_parser"
                    else:
                        # Use cached next or default to validate_resume for resume processing
                        next_node = cached_result.get("next", "validate_resume")
                    
                    metrics.record_cache_hit()
                    metrics.record_method("cached")
                    return {
                        **cached_result,
                        "analysis_id": analysis_id,
                        "analysis_method": "cached",
                        "processing_time_seconds": time.time() - start_time,
                        "next": next_node,
                        # Ensure user_interests are included if present in current request
                        "user_interests": user_interests_normalized if user_interests_normalized else cached_result.get("user_interests")
                    }
        
        # If wait timed out or no result, proceed with new request
        log.warning(
            "COALESCE_TIMEOUT "
            f"pid={os.getpid()} tenant_id={tenant_id} uid={uid} "
            f"request_type={request_type} key_fp={key_fingerprint}"
        )
        # Reset should_wait since we're proceeding with a new request after timeout
        should_wait = False
    
    # Set up in-flight tracking for new requests (including job_matcher)
    # This ensures we track the request even if we're not waiting for results
    # Set up tracking if we're not waiting (should_wait is False) or if it's a candidate_job_match
    # This covers: new requests, candidate_job_match requests, and requests after timeout
    if not should_wait or is_candidate_job_match:
        if is_candidate_job_match:
            log.info("Setting up in-flight tracking for job_matcher (no coalescing, real-time comparison)")
        # Create Event on the running event loop thread before taking threading.Lock (V-01).
        new_coalesce_event = asyncio.Event()
        with _inflight_lock:
            # Only set up tracking if it doesn't already exist (discard new_coalesce_event if race lost)
            if key_id not in _inflight_keys:
                _inflight_keys[key_id] = {
                    "timestamp": current_time,
                    "result": None,
                    "event": new_coalesce_event,
                }
                log.debug(
                    "COALESCE_TRACK "
                    f"pid={os.getpid()} tenant_id={tenant_id} uid={uid} "
                    f"request_type={request_type} key_fp={key_fingerprint}"
                )

    # Early routing check for job_matcher (skip cache entirely)
    # For resume_content_generator, use context-aware caching (only skip if context changed)
    if is_candidate_job_match:
        log.info("Skipping cache for job_matcher request (real-time comparison)")
        cached_result = None
    elif is_resume_content_generator_request:
        # Context-aware caching: Check cache with context hash
        if resume_content_context_hash:
            cache_key_data_resume_content = {
                "uid": uid,
                "request_type": "resume_content_generation",
                "context_hash": resume_content_context_hash
            }
            cached_result = cache.get(tenant_id, **cache_key_data_resume_content)
            if cached_result:
                log.info(f"✅ CACHE_HIT for resume_content_generator (context unchanged: {resume_content_context_hash[:16]}...)")
            else:
                log.info(f"✅ CACHE_MISS for resume_content_generator (context changed or first generation: {resume_content_context_hash[:16]}...)")
        else:
            # If context hash computation failed, skip cache to be safe
            log.info("Skipping cache for resume_content_generator request (context hash computation failed)")
        cached_result = None
    else:
        # Check cache (now only for long-term cache, not in-flight)
        log.debug(
            "CACHE_LOOKUP "
            f"pid={os.getpid()} tenant_id={tenant_id} uid={uid} "
            f"request_type={request_type} key_fp={key_fingerprint}"
        )

        cached_result = cache.get(tenant_id, **cache_key_data)
    
    # Process cached result only if we have one (skip for job_matcher)
    if cached_result:
        log.info(
            "CACHE_HIT "
            f"pid={os.getpid()} tenant_id={tenant_id} uid={uid} "
            f"request_type={request_type} analysis_id={analysis_id}"
        )
        
        # Store in in-flight cache and signal waiting threads
        with _inflight_lock:
            # Defensive check: ensure _inflight_keys is a dict
            if not isinstance(_inflight_keys, dict):
                log.error(f"⚠️ _inflight_keys is not a dict! Type: {type(_inflight_keys)}, value: {_inflight_keys}")
                # Reinitialize as dict if it got corrupted
                _inflight_keys = {}
            if key_id in _inflight_keys:
                _inflight_keys[key_id]["result"] = cached_result
                _inflight_keys[key_id]["event"].set()
        
        # Special handling for resume_content_generator cached results
        if is_resume_content_generator_request:
            # resume_content_generator always ends the flow
            result = {
                **cached_result,
                "analysis_id": analysis_id,
                "analysis_method": "cached",
                "processing_time_seconds": time.time() - start_time,
                "next": "end",
                "uid": uid,
                "tenant_id": tenant_id,
                "callback_url": callback_url
            }
            metrics.record_cache_hit()
            metrics.record_method("cached")
            log.info(f"✅ CACHE_HIT for resume_content_generator: Returning cached result (context unchanged)")
            return result
        
        # Determine next node based on current request state (not cached state)
        # Check if user_interests are present in current request
        user_interests = body.get("user_interests")
        user_interests_normalized = None
        if user_interests and isinstance(user_interests, list) and len(user_interests) > 0:
            if isinstance(user_interests[0], dict) and "answer" in user_interests[0]:
                user_interests_normalized = [item.get("answer", "") for item in user_interests if item.get("answer")]
            else:
                user_interests_normalized = user_interests
        
        # Route based on current request's user_interests, not cached result
        cached_next = cached_result.get("next", "validate_resume")
        
        # CRITICAL FIX: When user_interests are present, ALWAYS skip cache and process normally
        # This ensures fresh analysis with user's interests, avoiding completion signals from cached results
        if user_interests_normalized and len(user_interests_normalized) > 0:
            log.info(f"✅ CACHE_SKIP FLOW 2: user_interests present, skipping cache to ensure fresh analysis with interests")
            # Don't return cached result - fall through to normal processing
            # This will be handled after the cache check
        else:
            # Flow 1: No user_interests - use cached result
            next_node = cached_next
            log.info(f"✅ CACHE_HIT FLOW 1: No user_interests, using cached next='{next_node}'")
        
        # Flow 1: No user_interests - return cached result
        if not (user_interests_normalized and len(user_interests_normalized) > 0):
            # Normal cached result return for Flow 1
            result = {
                **cached_result,
                "analysis_id": analysis_id,
                "analysis_method": "cached",
                "processing_time_seconds": time.time() - start_time,
                "next": next_node,
                "user_interests": cached_result.get("user_interests")
            }
            metrics.record_cache_hit()
            metrics.record_method("cached")
            log.info(f"✅ CACHE_HIT FLOW 1: Returning result with next='{next_node}'")
            return result
        else:
            # Flow 2: user_interests present - skip cache and continue with normal processing
            log.info(f"✅ CACHE_SKIP FLOW 2: Skipped cache due to user_interests present, continuing with normal processing")
    
    # Skip cache miss logging and DB backfill for job_matcher (already handled above)
    if not is_candidate_job_match:
        log.info(
            "CACHE_MISS "
            f"pid={os.getpid()} tenant_id={tenant_id} uid={uid} "
            f"request_type={request_type} analysis_id={analysis_id}"
        )
        metrics.record_cache_miss()
    
    # DB backfill: try session store for prior identical result
    # CRITICAL: Skip DB backfill when user_interests are present OR for job_matcher (same as cache skip logic)
    user_interests_for_backfill = body.get("user_interests")
    user_interests_normalized_for_backfill = None
    if user_interests_for_backfill and isinstance(user_interests_for_backfill, list) and len(user_interests_for_backfill) > 0:
        if isinstance(user_interests_for_backfill[0], dict) and "answer" in user_interests_for_backfill[0]:
            user_interests_normalized_for_backfill = [item.get("answer", "") for item in user_interests_for_backfill if item.get("answer")]
        else:
            user_interests_normalized_for_backfill = user_interests_for_backfill
    
    # Skip DB backfill when user_interests are present OR for job_matcher OR resume_content_generator
    # resume_content_generator must always generate fresh content to incorporate latest assessment/interview/career updates
    if not is_candidate_job_match and not is_resume_content_generator_request and not (user_interests_normalized_for_backfill and len(user_interests_normalized_for_backfill) > 0):
        try:
            sess = await _rbi(session_manager.get_session_by_owner, uid, "candidate_pipeline")
            session_id = (sess.session_id if sess else state.get("session_id", "")) or ""
            if session_id:
                from chroma import get_chat_session
                sess_data = await _rbi(get_chat_session, session_id) or {}
                cached_bucket = (sess_data.get("cached_results") or {})

                envelope = cached_bucket.get(key_fingerprint)
                if envelope:
                    cache_key_data["request_type"] = request_type
                    cache.set(tenant_id, envelope, **cache_key_data)
                    log.info(
                        "DB_BACKFILL_HIT "
                        f"pid={os.getpid()} tenant_id={tenant_id} uid={uid} "
                        f"request_type={request_type} key_fp={key_fingerprint}"
                    )
                    
                    # Store in in-flight cache and signal waiting coroutines
                    with _inflight_lock:
                        if key_id in _inflight_keys:
                            _inflight_keys[key_id]["result"] = envelope
                            _inflight_keys[key_id]["event"].set()
                    
                    # Use cached next or default to validate_resume for resume processing
                    next_node = envelope.get("next", "validate_resume")
                    
                    return {
                        **envelope,
                        "analysis_id": analysis_id,
                        "analysis_method": "cached",
                        "processing_time_seconds": time.time() - start_time,
                        "next": next_node,
                        "user_interests": envelope.get("user_interests")
                    }
        except Exception:
            pass
    else:
        if is_candidate_job_match:
            log.info(f"✅ DB_BACKFILL_SKIP: job_matcher request, skipping DB backfill for real-time comparison")
        elif is_resume_content_generator_request:
            log.info(f"✅ DB_BACKFILL_SKIP: resume_content_generator request, skipping DB backfill to always generate fresh content (reflects latest assessment/interview/career updates)")
        else:
            log.info(f"✅ DB_BACKFILL_SKIP FLOW 2: user_interests present, skipping DB backfill to ensure fresh analysis")
    
    # Step 4: Route based on request type
    log.info(
        f"Processing request for tenant: {tenant_id}, user: {uid}, "
        f"analysis_id: {analysis_id}"
    )
    
    # Sanitize URLs
    resume_url = await sanitize_url_async(body.get("resume_url", "") or "")
    jd_url = await sanitize_url_async(body.get("jd_url", "") or "")
    callback_url = await sanitize_url_async(body.get("callback_url", "") or "")
    
    # Redact PII from logs
    redacted_body = {k: scrub_and_tokenize(str(v)) if isinstance(v, str) else v for k, v in body.items()}
    log.info(f"Processing request: {redacted_body}")
    
    # Manual assessment builder: route directly when raw assessment text is present in body
    raw_assessment_text = (
        body.get("raw_assessment_text")
        or body.get("rawAssessmentText")
    )
    if raw_assessment_text:
        log.info("Routing to assessment_builder_agent (manual assessment builder flow: raw_assessment_text present)")
        org_type = body.get("organization_type") or body.get("organizationType") or ""
        return {
            "next": "assessment_builder_agent",
            "uid": uid,
            "tenant_id": tenant_id,
            "callback_url": callback_url,
            "body": body,
            "manual_assessment_text": raw_assessment_text,
            "organization_type": org_type,
            "analysis_id": analysis_id,
            "analysis_method": "deterministic",
        }
    
    # Route based on request type
        # Check for notification flow: event field OR email/user_mail
    event = body.get("event")
    if event or body.get("user_mail") or body.get("email"):
        log.info(f"Routing to notification_agent (event={event})")
        return {
            "next": "notification_agent",
            "analysis_id": analysis_id,
            "analysis_method": "deterministic",
            "tenant_id": tenant_id,
            "event": event,
            **body
        }

    # Check for job_matcher flow
    # Both scenarios use job_matcher, but if uid has no stored resume, do validation/parsing first
    job_id = body.get("job_id")
    uid_has_resume = False
    stored_structured_resume = None
    resume_url = body.get("resume_url")
    resume_text = body.get("resume_text")
    has_job_status = body.get("job_status") and body.get("job_status").lower() in ["closed", "reopen", "live"] 
    
    # Check if uid has stored resume (only if no resume_url or resume_text provided)
    if uid and job_id and not resume_url and not resume_text and not has_job_status:
        try:
            resume_doc = await _rbi(get_resume_doc, uid)
            if resume_doc and isinstance(resume_doc, dict):
                stored_structured_resume = resume_doc.get("structured_resume")
                if stored_structured_resume and isinstance(stored_structured_resume, dict):
                    uid_has_resume = True
                    log.info(f"Found stored structured_resume for UID {uid} - routing directly to job_matcher")
        except Exception as e:
            log.warning(f"Error checking for stored resume: {e}")
    
    # CRITICAL: Check for candidate-job comparison FIRST (before job_matcher)
    # This is for comparing ONE candidate with ONE job (View Details flow)
    # Route for candidate-job comparison (View Details flow) - existing account with job_id
    # ⚠️ NOTE: This routing ONLY applies to /compare-candidate-job endpoint (endpoint_name="compare_candidate_job")
    # ⚠️ /analyze-resume-callback endpoint REJECTS request_type="candidate_job_match" and will NEVER route to job_matcher directly
    # ✅ CRITICAL FIX: Check endpoint_name to ensure only /compare-candidate-job can route to job_matcher for comparison
    endpoint_name = state.get("endpoint_name") or body.get("endpoint_name")
    is_compare_candidate_job_endpoint = endpoint_name == "compare_candidate_job"
    
    if body.get("request_type") == "candidate_job_match" and is_compare_candidate_job_endpoint and (body.get("uid") and body.get("job_id") and not body.get("jd_url") and not state.get("jd_url") and not body.get("job_details") and not state.get("job_details") and not body.get("company")):
        # If resume_url or resume_text is provided, route to validate_resume first
        if resume_url or resume_text:
            log.info(f"candidate_job_match with resume_url/resume_text: routing to validate_resume (resume_url={bool(resume_url)}, resume_text={bool(resume_text)})")
            return {
                "resume_url": resume_url,
                "resume_text": resume_text,
                "next": "validate_resume",
                "job_id": body.get("job_id"),
                "endpoint_name": "compare_candidate_job",
                "request_type": "candidate_job_match",
                "analysis_id": analysis_id,
                "analysis_method": "deterministic",
                "tenant_id": tenant_id,
                "uid": uid,
                "callback_url": callback_url,
                "body": body,
                "recruiter_questions": body.get("recruiter_questions") or body.get("recruiterQuestions") or body.get("questions"),
                "interview_feedback": body.get("interview_feedback") or body.get("interviewFeedback"),
            }
        # If no resume_url/resume_text, route directly to job_matcher (graph maps "job_matcher" -> job_matcher_preprocessor)
        log.info("Routing to job_matcher (compare flow via job_matcher_preprocessor)")
        return {
            "next": "job_matcher",
            "compare_job_only": True,
            "endpoint_name": "compare_candidate_job",
            "request_type": "candidate_job_match",
            "analysis_id": analysis_id,
            "analysis_method": "deterministic",
            "tenant_id": tenant_id,
            "uid": uid,
            "job_id": body.get("job_id"),
            "callback_url": callback_url,
            "body": body,
            "recruiter_questions": body.get("recruiter_questions") or body.get("recruiterQuestions") or body.get("questions"),
            "interview_feedback": body.get("interview_feedback") or body.get("interviewFeedback"),
        }
    
    # --- Conditional: certificate update subgraph ---
    is_certificates_endpoint = endpoint_name in ("skill_proficiency_certificates", "upload_certificates")
    certificates_list = body.get("certificates") or body.get("certificates_list") or []
    has_valid_certificates_payload = (
        isinstance(certificates_list, list)
        and 1 <= len(certificates_list) <= 50
    )
    if is_certificates_endpoint and has_valid_certificates_payload:
        log.info(f"Routing to update_certificates node: endpoint={endpoint_name}, certs={len(certificates_list)}")
        return {
            "next": "update_certificates",
            "uid": uid,
            "tenant_id": tenant_id,
            "callback_url": callback_url,
            "body": body,
        }

    # Enhance JD only: return enhanced text to user so they can send again for validation (before parsing)
    if body.get("enhance_jd_only"):
        jd_url_for_enhance = jd_url or body.get("jd_url") or state.get("jd_url")
        job_details_for_enhance = body.get("job_details", {})
        full_job_desc = (
            job_details_for_enhance.get("full_job_description")
            or job_details_for_enhance.get("fullJobDescription")
            or job_details_for_enhance.get("structured_job_description")
        )
        if jd_url_for_enhance:
            try:
                jd_text_enhance = (
                    await download_resume_text_async(
                        jd_url_for_enhance, file_type="job_description"
                    )
                    or ""
                )
            except Exception as e:
                log.error(f"Failed to extract JD for enhance: {e}")
                return {
                    "next": "end",
                    "error": f"Failed to extract job description: {str(e)}",
                    "analysis_id": analysis_id,
                    "analysis_method": "error",
                }
        elif full_job_desc:
            jd_text_enhance = full_job_desc
        else:
            jd_text_parts = []
            job_title = job_details_for_enhance.get("job_title") or job_details_for_enhance.get("jobTitle")
            location = job_details_for_enhance.get("location")
            job_type = job_details_for_enhance.get("job_type") or job_details_for_enhance.get("jobType")
            experience = job_details_for_enhance.get("experience")
            salary_range = job_details_for_enhance.get("salary_range") or job_details_for_enhance.get("salaryRange")
            department = job_details_for_enhance.get("department")
            required_skills = job_details_for_enhance.get("required_skills") or job_details_for_enhance.get("requiredSkills")
            if job_title:
                jd_text_parts.append(f"Job Title: {job_title}")
            if location:
                jd_text_parts.append(f"Location: {location}")
            if job_type:
                jd_text_parts.append(f"Job Type: {job_type}")
            if experience:
                jd_text_parts.append(f"Experience: {experience}")
            if salary_range:
                jd_text_parts.append(f"Salary Range: {salary_range}")
            if department:
                jd_text_parts.append(f"Department: {department}")
            if required_skills:
                jd_text_parts.append(f"Required Skills: {', '.join(required_skills)}" if isinstance(required_skills, list) else f"Required Skills: {required_skills}")
            jd_text_enhance = "\n".join(jd_text_parts) if jd_text_parts else ""
        if not (jd_text_enhance and jd_text_enhance.strip()):
            log.warning("enhance_jd_only set but no JD content from jd_url or job_details")
            return {
                "next": "end",
                "error": "No job description text provided for enhancement (provide jd_url or job_details with full_job_description or fields)",
                "analysis_id": analysis_id,
                "analysis_method": "error",
            }
        log.info("Routing to enhance_jd (enhance only; user can send again for validation)")
        return {
            "jd_text": jd_text_enhance,
            "jd_url": jd_url_for_enhance if jd_url_for_enhance else None,
            "next": "enhance_jd",
            "job_id": body.get("job_id"),
            "company": company,
            "body": body,
            "uid": uid,
            "tenant_id": tenant_id,
            "callback_url": callback_url,
            "analysis_id": analysis_id,
            "analysis_method": "deterministic",
        }

    # CRITICAL: Check for jd_url - if present, route to validate_jd (job_description_parser -> ranker flow)
    # This must come before job_matcher check to ensure proper routing
    jd_url_to_use = jd_url or body.get("jd_url") or state.get("jd_url")
    if jd_url_to_use:
        log.info(f"Routing to Job Description processing flow (jd_url present: {bool(jd_url_to_use)})")
        circuit_breaker = get_circuit_breaker(tenant_id)
        if circuit_breaker.is_open():
            log.warning(f"Circuit breaker open for tenant: {tenant_id}")
            metrics.record_circuit_breaker_event("circuit_open")
            return {
                "next": "end",
                "error": "Service temporarily unavailable",
                "analysis_id": analysis_id,
                "analysis_method": "error"
            }
        
        try:
            # Async JD download (sync download_resume_text returns a Future in async context)
            jd_text = (
                await download_resume_text_async(jd_url_to_use, file_type="job_description")
                or ""
            )
            circuit_breaker.record_success()
        except Exception as e:
            circuit_breaker.record_failure()
            log.error(f"Failed to download/extract job description: {e}")
            import traceback
            log.error(traceback.format_exc())
            return {
                "next": "end",
                "error": f"Failed to extract job description: {str(e)}",
                "analysis_id": analysis_id,
                "analysis_method": "error"
            }
        
        return {
            "jd_text": jd_text,
            "jd_url": jd_url_to_use,
            "next": "validate_jd",
            "job_id": body.get("job_id"),
            "company": company,
            "body": body,
            "uid": uid,
            "tenant_id": tenant_id,
            "callback_url": callback_url,
            "analysis_id": analysis_id,
            "analysis_method": "deterministic"
        }
    
    # If uid has stored resume, go directly to job_matcher (skip validation/parsing)
    # This matches ONE candidate against ALL jobs (Job Recommendations flow)
    if uid_has_resume and job_id:
        log.info(f"Routing directly to job_matcher for UID {uid} with job_id={job_id}")
        return {
            "next": "job_matcher",
            "analysis_id": analysis_id,
            "analysis_method": "deterministic",
            "tenant_id": tenant_id,
            "uid": uid,
            "job_id": job_id,
            "callback_url": callback_url,
            **body
        }
    
    # If uid has no stored resume, check if resume_url or structured_resume is provided - do validation/parsing first
    if not uid_has_resume and job_id and (resume_url or body.get("structured_resume") or resume_text):
        log.info(f"UID {uid} has no stored resume, but resume_url/structured_resume provided - doing validation/parsing before job_matcher")
        
        # Create session for this flow
        new_session = await _rbi(session_manager.get_or_reuse_session,
            owner_id=uid if uid else f"job_{job_id}",
            kind="candidate_pipeline",
            owner_type="candidate",
            initial_step="validate_resume",
            initial_data={
                "uid": uid,
                "job_id": job_id,
                "callback_url": callback_url,
                "resume_url": resume_url,
                "structured_resume": body.get("structured_resume"),
                "has_stored_resume": True
            }
        )
        
        return {
            "resume_url": resume_url,
            "resume_text": resume_text,
            "structured_resume": body.get("structured_resume"),
            "has_stored_resume": True,
            "next": "validate_resume",
            "job_id": job_id,  # Explicitly set job_id in state
            "analysis_id": analysis_id,
            "analysis_method": "deterministic",
            "tenant_id": tenant_id,
            "uid": uid,
            "callback_url": callback_url,
            "session_id": new_session.session_id if new_session else (uid or f"job_{job_id}"),
            "body": body  # Also preserve in body for fallback
        }

    if body.get("job_details"):
        log.info("Routing to validate_jd with manual job details (validate_jd → job_description_parser → ranker)")
        
        job_details = body.get("job_details", {})
        
        # Handle both naming conventions (snake_case and camelCase)
        # Prefer full_job_description/fullJobDescription if available (usually contains everything)
        # full_job_desc = job_details.get("full_job_description") or job_details.get("fullJobDescription")
        # Normalize JD field names → always output as full_Job_Description
        full_job_desc = (
            job_details.get("full_job_description")
            or job_details.get("fullJobDescription")
            or job_details.get("structured_job_description")
        )
        
        # if full_job_desc:
        #     # Use full_job_description directly since it usually contains complete JD content
        #     jd_text = full_job_desc
        #     log.debug("Using full_job_description directly for jd_text")
        if full_job_desc:
            jd_text = full_job_desc
            log.debug("Using normalized full_Job_Description")

            # 🔒 Override job_details to return only one key
            # job_details = {
            #     "full_Job_Description": full_job_desc
            # }
            # ✅ Add normalized key WITHOUT deleting other fields
            job_details["full_job_description"] = full_job_desc
            # 🧹 Remove duplicate JD fields (optional but recommended)

            # 🧹 Remove duplicate JD fields (optional but recommended)
            # 🧹 Remove duplicate JD fields (optional but recommended)
            job_details.pop("fullJobDescription", None)
            job_details.pop("structured_job_description", None)
        
        else:
            # Fallback: Construct jd_text from individual fields if full_job_description not available
            jd_text_parts = []
            
            # Handle both naming conventions for all fields
            job_title = job_details.get("job_title") or job_details.get("jobTitle")
            location = job_details.get("location")
            job_type = job_details.get("job_type") or job_details.get("jobType")
            experience = job_details.get("experience")
            salary_range = job_details.get("salary_range") or job_details.get("salaryRange")
            department = job_details.get("department")
            required_skills = job_details.get("required_skills") or job_details.get("requiredSkills")
            
            if job_title:
                jd_text_parts.append(f"Job Title: {job_title}")
            if location:
                jd_text_parts.append(f"Location: {location}")
            if job_type:
                jd_text_parts.append(f"Job Type: {job_type}")
            if experience:
                jd_text_parts.append(f"Experience: {experience}")
            if salary_range:
                jd_text_parts.append(f"Salary Range: {salary_range}")
            if department:
                jd_text_parts.append(f"Department: {department}")
            if required_skills:
                if isinstance(required_skills, list):
                    jd_text_parts.append(f"Required Skills: {', '.join(required_skills)}")
                else:
                    jd_text_parts.append(f"Required Skills: {required_skills}")
            
            jd_text = "\n".join(jd_text_parts) if jd_text_parts else ""
            log.debug("Constructed jd_text from individual fields")
        
        # Check for status in job_details (for edit operations)
        job_status_from_details = job_details.get("status") or job_details.get("job_status")
        if job_status_from_details and job_status_from_details.lower() in ["closed", "reopen", "live"]:
            log.info(f"🔄 Job status detected in job_details: {job_status_from_details}")
            # Normalize "live" to "reopen"
            if job_status_from_details.lower() == "live":
                job_status_from_details = "reopen"
        
        result = {
            "jd_text": jd_text,
            "next": "validate_jd",
            "job_id": body.get("job_id"),
            "uid": uid,
            "company": company,
            "tenant_id": tenant_id,
            "callback_url": callback_url,
            "analysis_id": analysis_id,
            "analysis_method": "deterministic",
            "job_details": job_details  # Pass original job_details for reference
        }
        
        # Add job_status to state if present in job_details (will be handled after parsing)
        if job_status_from_details:
            result["job_status"] = job_status_from_details.lower()
            result["job_status_from_details"] = True  # Flag to indicate status came from job_details
        
        return result


    # ✅ Enhanced job status handler with optional JD update support
    # Supports: "closed", "reopen", "live" (live = reopen)
    # If jd_url or jd_text is provided, routes to validate_jd -> job_description_parser -> ranker flow
    job_status_value = body.get("job_status")
    if job_status_value and job_status_value.lower() in ["closed", "reopen", "live"] and body.get("job_id"):
        log.info("🔄 Job status management request detected")
        
        # Normalize "live" to "reopen"
        if job_status_value.lower() == "live":
            job_status_value = "reopen"
            log.info("   Normalized 'live' status to 'reopen'")
        
        job_status = job_status_value.lower()
        job_ids = body.get("job_id")
        
        log.info(f"   Status: {job_status}")
        log.info(f"   Job IDs: {job_ids}")
        log.info(f"   Job IDs Type: {type(job_ids)}")

        # Normalize job_ids to list
        if isinstance(job_ids, str):
            job_ids = [job_ids]
        elif not isinstance(job_ids, list):
            log.error(f"❌ Invalid job_id format: {type(job_ids)}")
            return {
                "next": "end",
                "error": "Invalid job_id format - must be string or list",
                "analysis_id": analysis_id,
                "analysis_method": "error",
                "processing_time_seconds": time.time() - start_time
            }
        
        # Check if JD URL or text is provided for update
        jd_url_provided = body.get("jd_url") or jd_url
        jd_text_provided = body.get("jd_text")
        
        # If JD update is requested, only allow single job_id
        if (jd_url_provided or jd_text_provided) and len(job_ids) > 1:
            log.error("❌ Batch job status updates with JD URL/text are not supported")
            return {
                "next": "end",
                "error": "JD URL/text updates can only be applied to a single job at a time",
                "analysis_id": analysis_id,
                "analysis_method": "error",
                "processing_time_seconds": time.time() - start_time
            }
        
        # Determine if single or batch operation
        is_batch = len(job_ids) > 1
        
        if job_status == "closed":
            # Close jobs - move to job_closed collection
            log.info(f"🔒 Closing {len(job_ids)} job(s)")
            
            if is_batch:
                result = await _rbi(batch_move_jobs_to_closed, job_ids)
            else:
                success = await _rbi(move_job_to_closed,
                    job_id=job_ids[0],
                    closed_by=body.get("uid")
                )
                result = {
                    "total": 1,
                    "success": 1 if success else 0,
                    "failed": 0 if success else 1,
                    "details": [{
                        "job_id": job_ids[0],
                        "status": "success" if success else "failed",
                        "action": "closed"
                    }]
                }
            
            log.info(f"✅ Close complete: {result['success']}/{result['total']} successful")
            
            # If JD URL/text provided, route to validation flow (though closing with JD update is unusual)
            if jd_url_provided or jd_text_provided:
                log.info("📄 JD URL/text provided with close operation - routing to JD validation flow")
                # Extract JD text if URL provided
                jd_text_to_use = jd_text_provided
                jd_url_to_use = jd_url_provided
                
                if jd_url_to_use and not jd_text_to_use:
                    try:
                        circuit_breaker = get_circuit_breaker(tenant_id)
                        if circuit_breaker.is_open():
                            log.warning(f"Circuit breaker open for tenant: {tenant_id}")
                            return {
                                "next": "end",
                                "error": "Service temporarily unavailable",
                                "analysis_id": analysis_id,
                                "analysis_method": "error"
                            }
                        jd_text_to_use = (
                            await download_resume_text_async(
                                jd_url_to_use, file_type="job_description"
                            )
                            or ""
                        )
                        circuit_breaker.record_success()
                    except Exception as e:
                        log.error(f"Failed to download/extract job description: {e}")
                        return {
                            "next": "end",
                            "error": f"Failed to extract job description: {str(e)}",
                            "analysis_id": analysis_id,
                            "analysis_method": "error"
                        }
                
                return {
                    "jd_text": jd_text_to_use,
                    "jd_url": jd_url_to_use,
                    "next": "validate_jd",
                    "job_id": job_ids[0],
                    "company": company,
                    "body": body,
                    "uid": uid,
                    "tenant_id": tenant_id,
                    "callback_url": callback_url,
                    "analysis_id": analysis_id,
                    "analysis_method": "deterministic",
                    "job_status_updated": True,
                    "job_status_operation": "closed"
                }
            
            return {
                "next": "end",
                "analysis_id": analysis_id,
                "analysis_method": "job_status_update",
                "job_status_operation": "closed",
                "result": result,
                "message": f"Closed {result['success']}/{result['total']} jobs",
                "processing_time_seconds": time.time() - start_time,
                "ok": True
            }
        
        elif job_status == "reopen":
            # Reopen jobs - move back to job_descriptions collection
            log.info(f"🔓 Reopening {len(job_ids)} job(s)")
            
            if is_batch:
                result = batch_move_jobs_to_active(job_ids)
            else:
                success = move_job_to_active(job_ids[0])
                result = {
                    "total": 1,
                    "success": 1 if success else 0,
                    "failed": 0 if success else 1,
                    "details": [{
                        "job_id": job_ids[0],
                        "status": "success" if success else "failed",
                        "action": "reopened"
                    }]
                }
            
            log.info(f"✅ Reopen complete: {result['success']}/{result['total']} successful")
            
            # If JD URL/text provided, route to validation/parsing/ranking flow
            if jd_url_provided or jd_text_provided:
                log.info("📄 JD URL/text provided with reopen operation - routing to JD validation flow")
                # Extract JD text if URL provided
                jd_text_to_use = jd_text_provided
                jd_url_to_use = jd_url_provided
                
                if jd_url_to_use and not jd_text_to_use:
                    try:
                        circuit_breaker = get_circuit_breaker(tenant_id)
                        if circuit_breaker.is_open():
                            log.warning(f"Circuit breaker open for tenant: {tenant_id}")
                            return {
                                "next": "end",
                                "error": "Service temporarily unavailable",
                                "analysis_id": analysis_id,
                                "analysis_method": "error"
                            }
                        jd_text_to_use = (
                            await download_resume_text_async(
                                jd_url_to_use, file_type="job_description"
                            )
                            or ""
                        )
                        circuit_breaker.record_success()
                    except Exception as e:
                        log.error(f"Failed to download/extract job description: {e}")
                        import traceback
                        log.error(traceback.format_exc())
                        return {
                            "next": "end",
                            "error": f"Failed to extract job description: {str(e)}",
                            "analysis_id": analysis_id,
                            "analysis_method": "error"
                        }
                
                return {
                    "jd_text": jd_text_to_use,
                    "jd_url": jd_url_to_use,
                    "next": "validate_jd",
                    "job_id": job_ids[0],
                    "company": company,
                    "body": body,
                    "uid": uid,
                    "tenant_id": tenant_id,
                    "callback_url": callback_url,
                    "analysis_id": analysis_id,
                    "analysis_method": "deterministic",
                    "job_status_updated": True,
                    "job_status_operation": "reopen"
                }
            
            return {
                "next": "end",
                "analysis_id": analysis_id,
                "analysis_method": "job_status_update",
                "job_status_operation": "reopen",
                "result": result,
                "message": f"Reopened {result['success']}/{result['total']} jobs",
                "processing_time_seconds": time.time() - start_time,
                "ok": True
            }


    # Check for 2nd call scenario: user_interests provided (answers to 10 questions) without resume_url
    # This is the career advisor flow that uses stored context from 1st call + new user_interests
    user_interests = body.get("user_interests")
    user_interests_normalized = None
    if user_interests and isinstance(user_interests, list) and len(user_interests) > 0:
        if isinstance(user_interests[0], dict) and "answer" in user_interests[0]:
            user_interests_normalized = [item.get("answer", "") for item in user_interests if item.get("answer")]
        else:
            user_interests_normalized = user_interests
    
    # Get resume_text_from_body early to avoid undefined variable error
    resume_text_from_body = body.get("resume_text", "")
    
    is_second_call = (
        not resume_url and 
        not resume_text_from_body and 
        uid and
        user_interests_normalized and len(user_interests_normalized) > 0 and  # user_interests answers provided
        not body.get("plan") and
        not body.get("submission") and
        not body.get("job_id") and
        not body.get("jd_url") and
        not body.get("job_details") and 
        not body.get("company") and 
        not body.get("job_status")
    )
    
    if is_second_call:
        # Check if structured_resume exists in chroma (from 1st call)
        try:
            from chroma import fetch_structured_resume
            stored_resume = await _rbi(fetch_structured_resume, uid, tenant_id=tenant_id)
            if stored_resume and isinstance(stored_resume, dict):
                log.info(f"✅ 2ND CALL DETECTED: user_interests provided ({len(user_interests_normalized) if user_interests_normalized else 0} answers), structured_resume found in chroma for UID {uid}")
                log.info(f"✅ 2ND CALL: Flow will be: career_advisor -> market_and_course_recommender -> assessment_recommender -> job_matcher -> assessment_validator (no resume_scorer)")
                log.info(f"✅ 2ND CALL: user_interests will be used as context by all agents in the flow")
                
                # Get session to retrieve stored data
                sess = await _rbi(session_manager.get_session_by_owner, uid, "candidate_pipeline")
                session_id = (sess.session_id if sess else state.get("session_id", "")) or ""
                
                # Hydrate state from session to get all stored data
                hydrated_state = await _hydrate_state_from_session_async(dict(state), force_hydration=True)
                
                # Ensure user_interests from 2nd call are included (these are the answers to the 10 questions)
                # Format: user_interests should be a list of dicts with "question" and "answer" keys
                user_interests_formatted = body.get("user_interests", [])
                if not isinstance(user_interests_formatted, list):
                    user_interests_formatted = []
                
                # Map user_interests to user_interest_answers for assessment_recommender
                # user_interest_answers format: [{"question": "...", "answer": "..."}, ...]
                user_interest_answers = []
                for item in user_interests_formatted:
                    if isinstance(item, dict):
                        # If already in correct format, use as-is
                        if "question" in item and "answer" in item:
                            user_interest_answers.append({
                                "question": item.get("question", ""),
                                "answer": item.get("answer", "")
                            })
                        # If only answer is provided, try to infer question or use placeholder
                        elif "answer" in item:
                            user_interest_answers.append({
                                "question": item.get("question", "User interest question"),
                                "answer": item.get("answer", "")
                            })
                
                log.info(f"✅ 2ND CALL: Mapped {len(user_interest_answers)} user_interests to user_interest_answers for assessment_recommender")
                
                return {
                    **hydrated_state,
                    "structured_resume": stored_resume,  # Ensure structured_resume is present
                    "user_interests": user_interests_formatted,  # Use user_interests from 2nd call
                    "user_interest_answers": user_interest_answers,  # Also provide as user_interest_answers for assessment_recommender
                    "next": "career_advisor",  # Start 2nd call flow
                    "body": body,
                    "uid": uid,
                    "tenant_id": tenant_id,
                    "callback_url": callback_url,
                    "analysis_id": analysis_id,
                    "analysis_method": "deterministic",
                    "session_id": session_id,
                    "is_second_call": True  # Flag to indicate this is 2nd call
                }
        except Exception as e:
            log.warning(f"Error checking for stored resume in 2nd call: {e}, falling through to normal flow")
            # Fall through to normal resume processing if check fails
    
    # Check for resume processing: either resume_text or resume_url
    resume_text_from_body = body.get("resume_text", "")
    if (resume_text_from_body and resume_text_from_body.strip()) or resume_url:
        log.debug("🔍 Processing Resume flow")
        
        user_interests = body.get("user_interests")
        
        # Normalize user_interests format
        if user_interests and isinstance(user_interests, list) and len(user_interests) > 0:
            if isinstance(user_interests[0], dict) and "answer" in user_interests[0]:
                user_interests_normalized = [item.get("answer", "") for item in user_interests if item.get("answer")]
            else:
                user_interests_normalized = user_interests
        else:
            user_interests_normalized = None
        
        # Check circuit breaker
        circuit_breaker = get_circuit_breaker(tenant_id)
        if circuit_breaker.is_open():
            log.warning(f"Circuit breaker open for tenant: {tenant_id}")
            metrics.record_circuit_breaker_event("circuit_open")
            return {
                "next": "end",
                "error": "Service temporarily unavailable",
                "analysis_id": analysis_id,
                "analysis_method": "error"
            }
        
        # If user_interests provided, route to validate_resume first (will extract with Gemini), then groq_resume_parser
        if user_interests_normalized and len(user_interests_normalized) > 0:
            log.info("✅ user_interests in payload - full flow")
            log.info(f"✅ FLOW 2: user_interests detected ({len(user_interests_normalized)} items) - routing to validate_resume first")
            log.info(f"✅ FLOW 2: user_interests values: {user_interests_normalized[:3]}...")  # Log first 3 for debugging
            
            # Don't extract text here - let validator do it with Gemini ONLY
            # Only use resume_text if it was provided in body
            if resume_text_from_body and resume_text_from_body.strip():
                resume_text = resume_text_from_body.strip()
                log.info(f"✅ FLOW 2: Using resume_text from request body ({len(resume_text)} characters)")
            else:
                # Don't extract text here - let validator extract with Gemini ONLY
                resume_text = ""
                log.info(f"✅ FLOW 2: Passing resume_url to validator - will extract using Gemini ONLY")
            
            new_session = await _rbi(session_manager.get_or_reuse_session,
                owner_id=uid,
                kind="candidate_pipeline",
                owner_type="candidate",
                initial_step="validate_resume",
                initial_data={
                    "resume_url": resume_url,
                    "uid": uid,
                    "callback_url": callback_url,
                    "user_interests": user_interests_normalized
                }
            )
            
            result = {
                "resume_text": resume_text,  # Empty if resume_url provided (validator will extract)
                "resume_url": resume_url,  # Pass URL to validator
                "user_interests": user_interests_normalized,
                "next": "validate_resume",  # Route to validator first (will extract, then validate)
                "body": body,
                "uid": uid,
                "tenant_id": tenant_id,
                "callback_url": callback_url,
                "analysis_id": analysis_id,
                "analysis_method": "deterministic",
                "session_id": new_session.session_id if new_session else uid,
                "resume_from": False,
                "session_data": {}
            }
            log.info(f"✅ FLOW 2: Dispatcher returning with next='validate_resume', resume_text_len={len(resume_text) if resume_text else 0}, user_interests_count={len(user_interests_normalized) if user_interests_normalized else 0}")
            log.debug(
                "✅ FLOW 2 DEBUG: Returning result with next="
                f"'{result.get('next')}', has_user_interests="
                f"{bool(result.get('user_interests'))}"
            )
            return result
        
        # No user_interests -> validate -> interest_filler -> END
        # For Flow 1, validator will extract text using Gemini ONLY
        else:
            log.info("✅ No user_interests in payload - short flow")
            log.info("✅ FLOW 1: No user_interests detected - routing to validate_resume")
            
            # Get resume text: prioritize resume_text from body, otherwise pass resume_url to validator
            if resume_text_from_body and resume_text_from_body.strip():
                # Use resume_text directly from body
                resume_text = resume_text_from_body.strip()
                log.info(f"✅ Using resume_text from request body ({len(resume_text)} characters)")
                
                # Quality check: ensure text is usable
                quality = _quality_diag(resume_text)
                is_weak = is_text_weak(resume_text)
                
                log.info(f"📊 Resume text quality: len={quality['len']} words={quality['words']} "
                        f"alpha_ratio={quality['alpha_ratio']} weak={is_weak}")
                
                if is_weak:
                    log.warning(f"⚠️ Resume text quality is weak. "
                              f"Length: {quality['len']} chars, Words: {quality['words']}, "
                              f"Alpha ratio: {quality['alpha_ratio']}")
            elif resume_url:
                # Don't extract here - let validator do it with Gemini ONLY
                resume_text = ""  # Empty, validator will extract using Gemini
                log.info(f"✅ FLOW 1: Passing resume_url to validator - will extract using Gemini ONLY")
            else:
                # Neither resume_text nor resume_url provided
                log.error("Neither resume_text nor resume_url provided in request")
                return {
                    "next": "end",
                    "error": "Either resume_text or resume_url must be provided",
                    "analysis_id": analysis_id,
                    "analysis_method": "error"
                }
            
            new_session = await _rbi(session_manager.get_or_reuse_session,
                owner_id=uid,
                kind="candidate_pipeline",
                owner_type="candidate",
                initial_step="validate_resume",
                initial_data={
                    "resume_url": resume_url,
                    "uid": uid,
                    "callback_url": callback_url
                }
            )
            
            result = {
                "resume_text": resume_text,  # Empty if resume_url provided (validator will extract)
                "resume_url": resume_url,  # Pass URL to validator
                "next": "validate_resume",
                "body": body,
                "uid": uid,
                "tenant_id": tenant_id,
                "callback_url": callback_url,
                "analysis_id": analysis_id,
                "analysis_method": "deterministic",
                "session_id": new_session.session_id if new_session else uid,
                "resume_from": False,
                "session_data": {}
            }
            log.info(f"✅ FLOW 1: Dispatcher returning with next='validate_resume', resume_text_len={len(resume_text) if resume_text else 0}, resume_url={'provided' if resume_url else 'none'}")
            return result

    if body.get("submission"):
        log.info("Routing to Assessment evaluation flow")
        
        current_session_id = state.get("session_id")
        uid = state.get("uid")
        previous_agent_data = {}

        structured_resume, user_interests = await _rbi(_extract_resume_data_from_session, current_session_id, uid)

        if structured_resume:
            previous_agent_data["structured_resume"] = structured_resume

        if user_interests:
            previous_agent_data["user_interests"] = user_interests
        
        if current_session_id:
            try:
                from chroma import get_chat_session
                session_data = await _rbi(get_chat_session, current_session_id)
                if session_data:
                    previous_agent_data.update({
                        "career_advisor": session_data.get("career_advisor", {}),
                        "market_and_course_recommender": session_data.get("market_and_course_recommender", {}),
                        "assessment_recommender": session_data.get("assessment_recommender", {}),
                    })
                    
                    assessment_recommender_data = session_data.get("assessment_recommender", {})
                    if assessment_recommender_data:
                        previous_agent_data["prior_assessment_plan"] = assessment_recommender_data.get("assessment_plan", [])
                        previous_agent_data["prior_assessment_needs"] = assessment_recommender_data.get("assessment_needs", {})
            except Exception as e:
                log.error(f"❌ Error fetching previous agent data: {e}")

        assessment_topic = body.get("assessment_topic") or state.get("assessment_topic")
        
        if current_session_id:
            try:
                await _rbi(session_manager.update_step,
                    session_id=current_session_id,
                    step="dispatcher",
                    data={"routing": "assessment_evaluation", "analysis_id": analysis_id},
                    progress=0.1
                )
            except Exception as e:
                log.warning(f"Failed to update dispatcher session step: {e}")
        
        return {
            **previous_agent_data,
            "submission": body.get("submission"),
            "next": "assessment_evaluator",
            "question_doc_id": body.get("question_doc_id") or state.get("question_doc_id"),
            "assessment_id": body.get("assessment_id") or state.get("assessment_id"),
            "assessment_topic": assessment_topic,
            "body": body,
            "uid": uid,
            "tenant_id": tenant_id,
            "callback_url": callback_url,
            "analysis_id": analysis_id,
            "analysis_method": "deterministic",
            "session_id": current_session_id
        }
    
    if body.get("plan"):
        log.info("Routing to Assessment Question Generation flow")
        current_session_id = state.get("session_id")
        
        if current_session_id:
            try:
                await _rbi(session_manager.update_step,
                    session_id=current_session_id,
                    step="dispatcher",
                    data={"routing": "assessment_question_generation", "analysis_id": analysis_id},
                    progress=0.1
                )
            except Exception as e:
                log.warning(f"Failed to update dispatcher session step: {e}")
        
        assessment_plan = body.get("plan") or {}
        
        return {
            "assessment_plan": assessment_plan,
            "next": "assessment_question_generator",
            "assessment_id": body.get("assessment_id") or state.get("assessment_id"),
            "body": body,
            "uid": uid,
            "tenant_id": tenant_id,
            "callback_url": callback_url,
            "analysis_id": analysis_id,
            "analysis_method": "deterministic",
            "session_id": current_session_id
        }

    # Default to resume processing (should not reach here if resume_url or resume_text was provided)
    # This is a fallback for edge cases
    log.info("Routing to Resume processing flow (default)")
    
    # Check for user_interests even in default flow
    user_interests = body.get("user_interests")
    user_interests_normalized = None
    if user_interests and isinstance(user_interests, list) and len(user_interests) > 0:
        if isinstance(user_interests[0], dict) and "answer" in user_interests[0]:
            user_interests_normalized = [item.get("answer", "") for item in user_interests if item.get("answer")]
        else:
            user_interests_normalized = user_interests
    
    session_id = new_session.session_id if new_session else uid
    if session_id:
        try:
            await _rbi(session_manager.update_step,
                session_id=session_id,
                step="dispatcher",
                data={"routing": "resume_processing", "analysis_id": analysis_id},
                progress=0.1
            )
        except Exception as e:
            log.warning(f"Failed to update dispatcher session step: {e}")
    
    # Route based on user_interests
    if user_interests_normalized and len(user_interests_normalized) > 0:
        # User interests provided -> go directly to resume parser
        log.info("✅ Default flow: user_interests provided -> groq_resume_parser")
        resume_text_value = resume_text if 'resume_text' in locals() else ""
        result = {
            "resume_text": resume_text_value,
            "user_interests": user_interests_normalized,
            "next": "groq_resume_parser",
            "body": body,
            "uid": uid,
            "tenant_id": tenant_id,
            "callback_url": callback_url,
            "analysis_id": analysis_id,
            "analysis_method": "deterministic",
            "session_id": session_id,
            "resume_from": False,
            "session_data": {}
        }
        log.info(
            "✅ DEFAULT FLOW 2: Returning with next='groq_resume_parser', "
            f"resume_text_len={len(resume_text_value) if resume_text_value else 0}, "
            f"user_interests_count={len(user_interests_normalized) if user_interests_normalized else 0}"
        )
        log.debug(
            "✅ DEFAULT FLOW 2 DEBUG: Returning result with next="
            f"'{result.get('next')}', has_user_interests="
            f"{bool(result.get('user_interests'))}"
        )
        return result
    else:
        # No user_interests -> check if resume_url or resume_text is present before routing to validate_resume
        # If not present, check if this is a resume content generation request
        resume_url_present = bool(resume_url)
        resume_text_present = bool(resume_text)
        
        if not resume_url_present and not resume_text_present:
            # Check if this is a career chatbot request (has 'message' field)
            # Career chatbot requests should go to /coach/chat/ask endpoint, not through graph
            if body.get("message") and not body.get("job_id") and not body.get("jd_url"):
                log.warning(
                    f"⚠️ Career chatbot request detected in dispatcher. "
                    f"This should use /coach/chat/ask endpoint, not /analyze-resume-callback. "
                    f"Rejecting request to prevent incorrect routing."
                )
                return {
                    "next": "end",
                    "error": "Career chatbot requests must use /coach/chat/ask endpoint, not /analyze-resume-callback",
                    "validation_error": "Invalid endpoint for career chatbot request",
                    "body": body,
                    "uid": uid,
                    "tenant_id": tenant_id,
                    "callback_url": callback_url,
                    "analysis_id": analysis_id,
                    "analysis_method": "deterministic",
                    "session_id": session_id
                }
            
            # Check if this is a resume content generation request (has uid, no job_id, no jd_url, no message)
            # This is for the resume content generator endpoint
            has_job_id = bool(body.get("job_id"))
            has_jd_url = bool(jd_url)
            
            if uid and not has_job_id and not has_jd_url:
                log.info("✅ Routing to resume_content_generator (uid present, no resume_url/resume_text, no job_id/jd_url)")
                return {
                    "next": "resume_content_generator",
                    "body": body,
                    "uid": uid,
                    "tenant_id": tenant_id,
                    "callback_url": callback_url,
                    "analysis_id": analysis_id,
                    "analysis_method": "deterministic",
                    "session_id": session_id,
                    "resume_from": False,
                    "session_data": {}
                }
            
            log.warning("⚠️ No resume_url or resume_text provided, and no user_interests. Cannot proceed with resume flow.")
            return {
                "next": "end",
                "error": "Missing required fields: resume_url or resume_text is required for resume processing",
                "validation_error": "Missing resume_url or resume_text",
                "body": body,
                "uid": uid,
                "tenant_id": tenant_id,
                "callback_url": callback_url,
                "analysis_id": analysis_id,
                "analysis_method": "deterministic",
                "session_id": session_id
            }
        
        # No user_interests -> go to validate_resume -> interest_filler -> end
        log.info("✅ Default flow: No user_interests -> validate_resume (resume_url or resume_text present)")
        return {
            "resume_text": resume_text if 'resume_text' in locals() else "",
            "resume_url": resume_url if resume_url else "",
            "next": "validate_resume",
            "body": body,
            "uid": uid,
            "tenant_id": tenant_id,
            "callback_url": callback_url,
            "analysis_id": analysis_id,
            "analysis_method": "deterministic",
            "session_id": session_id,
            "resume_from": False,
            "session_data": {}
        }

def store_job_details_to_chroma(job_id: str, job_details: dict, uid: str = None, company: str = None) -> bool:
    """
    Store job_details from payload into ChromaDB job_descriptions collection.
    
    Args:
        job_id: Unique job identifier
        job_details: Raw job details from payload
        uid: Optional user ID for metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from chroma import insert_job_description
        from agents.job_matcher import classify_jd_domains

        # Extract must_have_skills from payload (recruiter-specified priority skills)
        must_have = job_details.get("must_have_skills") or job_details.get("mustHaveSkills") or []
        if isinstance(must_have, str) and must_have.strip():
            must_have = [s.strip() for s in must_have.split(",") if s.strip()]
        elif not isinstance(must_have, list):
            must_have = []

        # Extract good_to_have / preferred skills
        preferred = job_details.get("good_to_have_skills") or job_details.get("goodToHaveSkills") or job_details.get("preferredSkills") or []
        if isinstance(preferred, str) and preferred.strip():
            preferred = [s.strip() for s in preferred.split(",") if s.strip()]
        elif not isinstance(preferred, list):
            preferred = []

        # Transform job_details to match the structured format
        job_description = {
            "jobTitle": job_details.get("job_title", ""),
            "location": job_details.get("location", ""),
            "jobType": job_details.get("job_type", ""),
            "experience": job_details.get("experience", ""),
            "salaryRange": job_details.get("salary_range", ""),
            "status": job_details.get("status", "Live"),
            "requiredSkills": job_details.get("required_skills", []),
            "preferredSkills": preferred,
            "_must_have_skills": must_have,
            "educationRequired": job_details.get("education_qualification") or job_details.get("educationRequired") or "",
            "workMode": job_details.get("work_mode") or job_details.get("workMode") or "",
            "fullJobDescription": job_details.get("full_job_description", ""),
            "department": job_details.get("department", ""),
            "company": company or ""
        }
        
        # Classify domains so Chroma can filter by job_domain_primary
        detected_domains = classify_jd_domains(job_description)
        job_description["jobDomains"] = detected_domains
        log.info(f"🎯 Auto-classified JD domains for job_id={job_id}: {detected_domains}")
        
        # Prepare metadata
        metadata = {
            "job_id": job_id,
            "source": "job_details_payload",
            "stored_at": datetime.utcnow().isoformat()
        }
        
        if uid:
            metadata["uid"] = uid

        if company:
            metadata["company"] = company
        
        # Store in ChromaDB
        insert_job_description(job_id, job_description, metadata)
        
        log.info(f"✅ Stored job_details for job_id={job_id} in ChromaDB")
        return True
        
    except Exception as e:
        log.error(f"❌ Failed to store job_details for job_id={job_id}: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

# Removed async dispatcher wrapper - using simple dispatcher instead

def resume_router(state: AgentState) -> str:
    """
    Routes the workflow after resume validation. If the resume is invalid, it terminates the flow.
    If valid, it checks if job_id is present - if so, skip interest_filler and go to groq_resume_parser.
    For stored structured_resume flow (no account users), routes directly to groq_resume_parser.
    """
    try:
        log.debug("🔍 ROUTER_DEBUG: resume_router called!")
        
        if not is_valid_resume(state):
            log.debug("🔍 ROUTER_DEBUG: Resume invalid, returning end_invalid_resume")
            return "end_invalid_resume"

        # ✅ NEW: Check if this is a candidate_job_match request (from /compare-candidate-job endpoint)
        # ⚠️ NOTE: /analyze-resume-callback endpoint REJECTS request_type="candidate_job_match" and will NEVER route here
        body = state.get("body") or {}
        request_type = state.get("request_type") or body.get("request_type")
        if request_type == "candidate_job_match":
            # candidate_job_match flow with resume_url/resume_text: go to groq_resume_parser
            # (job_matcher will be added to parallel execution after groq_resume_parser)
            log.info("candidate_job_match flow with resume: routing to groq_resume_parser")
            return "groq_resume_parser"

        # Check if job_id is present - if so, skip interest_filler (no personal insights needed)
        job_id = state.get("job_id")
        has_stored_resume = state.get("has_stored_resume", False)
        
        if job_id or has_stored_resume:
            # Job matching flow - skip interest_filler, go directly to groq_resume_parser
            log.debug(
                "🔍 ROUTER_DEBUG: Job_id present or stored resume flow detected "
                f"(job_id={job_id}) -> groq_resume_parser"
            )
            log.info(
                f"Routing to groq_resume_parser (job_id={job_id}, "
                f"has_stored_resume={has_stored_resume}) - skipping interest_filler"
            )
            return "groq_resume_parser"

        # After validation, go to groq_resume_parser first (for normal flow without job_id)
        # Then interest_filler will use the structured output from groq_resume_parser
        log.debug("🔍 ROUTER_DEBUG: Resume valid -> groq_resume_parser (normal flow, will go to interest_filler after)")
        return "groq_resume_parser"
    
    except Exception as e:
        log.error(f"❌ resume_router error: {e}, falling back to 'end_invalid_resume'")
        return "end_invalid_resume"
    
    # Check if user_interests exists and is not empty
    # user_interests = state.get("body", {}).get("user_interests")
    log.debug(f"🔍 ROUTER_DEBUG: user_interests value: {user_interests}")
    log.debug(f"🔍 ROUTER_DEBUG: user_interests type: {type(user_interests)}")
    log.debug(f"🔍 ROUTER_DEBUG: user_interests is None: {user_interests is None}")
    log.debug(
        "🔍 ROUTER_DEBUG: user_interests length: "
        f"{len(user_interests) if isinstance(user_interests, list) else 'N/A'}"
    )
    
def create_standard_response_envelope(state: AgentState, analysis_method: str = "deterministic", 
                                    method_explain: Dict[str, Any] = None, skip_filtering: bool = False) -> Dict[str, Any]:
    """
    Create standard response envelope with intelligent field filtering based on active agents.
    
    Args:
        state: The agent state
        analysis_method: Method used for analysis
        method_explain: Explanation of the method
        skip_filtering: If True, include all state fields without filtering (for end node)
    """
    analysis_id = state.get("analysis_id", generate_analysis_id())
    confidence_score = state.get("confidence_score", 0.0)
    processing_time = state.get("processing_time_seconds", 0.0)
    
    # Get workflow summary to understand what agents are active
    workflow_summary = get_workflow_summary(state)
    active_agents = workflow_summary.get("active_agents", [])
    workflow_context = workflow_summary.get("workflow_context", "resume_processing")
    
    # Determine confidence level
    if confidence_score >= 0.8:
        confidence_level = "high"
    elif confidence_score >= 0.6:
        confidence_level = "medium"
    else:
        confidence_level = "low"
    
    # Determine analysis context based on workflow
    analysis_context = workflow_context
    if workflow_context == "resume_processing":
        analysis_context = "career_analysis"
    elif workflow_context == "job_description_processing":
        analysis_context = "job_matching"
    elif workflow_context == "assessment_evaluation":
        analysis_context = "assessment_review"
    
    # Create base response envelope
    response = {
        "ok": True,
        "analysis_method": analysis_method,
        "method_explain": method_explain or {
            "det_conf": confidence_score,
            "det_coverage": state.get("skill_breakdown", {}).get("base_coverage", 0.0),
            "thresholds": {"confidence": CONFIDENCE_THRESHOLD, "quality": QUALITY_THRESHOLD},
            "llm_timeout": False,
            "circuit_open": False
        },
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "analysis_context": analysis_context,
        "processing_time_seconds": processing_time,
        "user_context": {
            "user_id": state.get("uid", ""),
            "tenant_id": state.get("tenant_id", ""),
            "analysis_timestamp": datetime.utcnow().isoformat()
        },
        "workflow_summary": workflow_summary
    }
    
    # If skip_filtering is True, include all state fields (for end node)
    if skip_filtering:
        # Add all state fields, but let response envelope fields take precedence
        system_fields = {
            "ok", "analysis_method", "method_explain", "confidence_score", "confidence_level",
            "analysis_context", "processing_time_seconds", "user_context", "workflow_summary",
            "uid", "tenant_id", "callback_url", "session_id", "analysis_id"
        }
        
        for key, value in state.items():
            if key not in system_fields:
                response[key] = value
    else:
        # Use intelligent field filtering to get only relevant fields
        filtered_state = filter_agent_output(state)
        
        # Add filtered fields to response, but exclude system fields that are already in response
        system_fields = {
            "ok", "analysis_method", "method_explain", "confidence_score", "confidence_level",
            "analysis_context", "processing_time_seconds", "user_context", "workflow_summary",
            "uid", "tenant_id", "callback_url", "session_id", "analysis_id"
        }
        
        for key, value in filtered_state.items():
            if key not in system_fields:
                response[key] = value
    
    # For candidate-job matching workflow, explicitly include match results
    if state.get("candidate_job_match_result") or state.get("match_score") is not None:
        # Include candidate-job match result
        if state.get("candidate_job_match_result"):
            response["candidate_job_match_result"] = state["candidate_job_match_result"]
        
        # Include all match fields explicitly
        if state.get("match_score") is not None:
            response["match_score"] = state["match_score"]
        if state.get("skill_match_percentage") is not None:
            response["skill_match_percentage"] = state["skill_match_percentage"]
        if state.get("skill_match_count") is not None:
            response["skill_match_count"] = state["skill_match_count"]
        if state.get("skills_matched"):
            response["skills_matched"] = state["skills_matched"]
        if state.get("skills_unmatched"):
            response["skills_unmatched"] = state["skills_unmatched"]
        if state.get("total_required_skills") is not None:
            response["total_required_skills"] = state["total_required_skills"]
        if state.get("rationale"):
            response["rationale"] = state["rationale"]
        if state.get("skill_matching_diagnostics"):
            response["skill_matching_diagnostics"] = state["skill_matching_diagnostics"]
    
    # Include enhanced role fit when present (2nd call career flow)
    if state.get("enhanced_role_fit") is not None:
        response["enhanced_role_fit"] = state["enhanced_role_fit"]
    
    # Include skills with no supporting evidence when present (from skill_proficiency_analyzer)
    if state.get("skills_without_supporting_evidence") is not None:
        response["skills_without_supporting_evidence"] = state["skills_without_supporting_evidence"]
    
    # For assessment evaluation workflow, explicitly include agent outputs that might be filtered out
    if analysis_context == "assessment_review":
        # Include career advisor output
        if state.get("raw_skill_gap_analysis_output"):
            response["raw_skill_gap_analysis_output"] = state["raw_skill_gap_analysis_output"]
            
            # Add career advice field for frontend
            career_data = state["raw_skill_gap_analysis_output"]
            response["careerAdvice"] = career_data.get("career_advice", [])
        
        # Include assessment results
        if state.get("assessment_results"):
            response["assessment_results"] = state["assessment_results"]
        
        # Include generated questions
        if state.get("generated_questions"):
            response["generated_questions"] = state["generated_questions"]
        
        # Include report
        if state.get("report"):
            response["report"] = state["report"]
    
    # Add error fields if present
    if state.get("error"):
        response["error"] = state.get("error")
    if state.get("validation_error"):
        response["validation_error"] = state.get("validation_error")
    if state.get("security_error"):
        response["security_error"] = state.get("security_error")
    if state.get("rate_limit_error"):
        response["rate_limit_error"] = state.get("rate_limit_error")
    
    return response

def create_agent_specific_response(state: AgentState, agent_name: str) -> Dict[str, Any]:
    """
    Create a response filtered specifically for a given agent's output.
    This ensures only relevant fields are displayed for each agent.
    """
    # Get the filtered state for this specific agent
    filtered_state = filter_agent_output(state, agent_name)
    
    # Create a minimal response envelope
    response = {
        "ok": True,
        "agent_name": agent_name,
        "analysis_method": state.get("analysis_method", "deterministic"),
        "confidence_score": state.get("confidence_score", 0.0),
        "processing_time_seconds": state.get("processing_time_seconds", 0.0),
        "user_context": {
            "user_id": state.get("uid", ""),
            "tenant_id": state.get("tenant_id", ""),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    }
    
    # Add agent-specific fields
    response.update(filtered_state)
    
    # Add workflow context information
    workflow_summary = get_workflow_summary(state)
    response["workflow_context"] = workflow_summary.get("workflow_context", "unknown")
    response["agent_status"] = "completed"
    
    return response

async def cache_and_record_metrics(state: AgentState, result: Dict[str, Any]):
    """
    Cache result and record metrics for production monitoring.
    CRITICAL: Cache key MUST match dispatcher logic exactly!
    """
    from core.utils import run_blocking_io as _rbi
    global _inflight_keys
    tenant_id = state.get("tenant_id", "default-tenant")
    uid = state.get("uid", "")
    
    # Cache the result
    # Build the same stable key on store as on lookup (MUST match dispatcher logic exactly)
    body = state.get("body", {})
    cache_key_data = {"uid": uid}
    request_type = "resume_analysis"
    
    # ✅ Add user_interests hash to cache key (mirror dispatcher logic)
    user_interests = body.get("user_interests")
    if user_interests and isinstance(user_interests, list) and len(user_interests) > 0:
        try:
            import hashlib, json as _json
            if isinstance(user_interests[0], dict) and "answer" in user_interests[0]:
                user_interests_for_hash = [item.get("answer", "") for item in user_interests if item.get("answer")]
            else:
                user_interests_for_hash = user_interests
            interests_hash = hashlib.sha256(_json.dumps(user_interests_for_hash, sort_keys=True).encode()).hexdigest()
            cache_key_data["user_interests_hash"] = interests_hash
        except Exception as e:
            log.warning(f"Failed to hash user_interests for cache key: {e}")
            cache_key_data["user_interests_present"] = True

    # ✅ CRITICAL FIX: Add job_details hash to cache key (mirror dispatcher logic)
    if body.get("job_details"):
        request_type = "job_ranking"
        cache_key_data["request_type"] = request_type
        try:
            import hashlib, json as _json
            job_details = body.get("job_details", {})
            job_details_str = _json.dumps(job_details, sort_keys=True)
            job_hash = hashlib.sha256(job_details_str.encode()).hexdigest()
            cache_key_data["job_details_hash"] = job_hash
            if body.get("job_id"):
                cache_key_data["job_id"] = body.get("job_id")
        except Exception as e:
            log.warning(f"Failed to hash job_details: {e}")
            cache_key_data["job_details_present"] = True
    
    elif body.get("plan"):
        request_type = "assessment_generation"
        cache_key_data["request_type"] = request_type
        try:
            import hashlib, json as _json
            plan_hash = hashlib.sha256(_json.dumps(body.get("plan"), sort_keys=True).encode()).hexdigest()
            cache_key_data["assessment_plan_hash"] = plan_hash
        except Exception:
            cache_key_data["assessment_plan"] = body.get("plan")
    elif body.get("submission"):
        request_type = "assessment_evaluation"
        cache_key_data["request_type"] = request_type
        try:
            import hashlib, json as _json
            sub_hash = hashlib.sha256(_json.dumps(body.get("submission"), sort_keys=True).encode()).hexdigest()
            cache_key_data["submission_hash"] = sub_hash
        except Exception:
            cache_key_data["submission_present"] = True
    elif body.get("jd_url"):
        request_type = "job_description"
        cache_key_data["request_type"] = request_type
        try:
            import hashlib, json as _json
            jd_url = body.get("jd_url", "")
            jd_hash = hashlib.sha256(jd_url.encode()).hexdigest()
            cache_key_data["jd_url_hash"] = jd_hash
            if body.get("job_id"):
                cache_key_data["job_id"] = body.get("job_id")
        except Exception as e:
            log.warning(f"Failed to hash JD URL for cache key: {e}")
            cache_key_data["jd_url_present"] = True
    else:
        cache_key_data["request_type"] = request_type
        resume_url = body.get("resume_url", "")
        if resume_url:
            try:
                import hashlib
                resume_hash = hashlib.sha256(resume_url.encode()).hexdigest()
                cache_key_data["resume_url_hash"] = resume_hash
            except Exception as e:
                log.warning(f"Failed to hash resume URL for cache key: {e}")
                cache_key_data["resume_url_present"] = True
    
    # ✅ Handle resume_content_generator: Add context hash to cache key
    # Check if this is a resume_content_generator result (has resume_content field)
    if result.get("resume_content") and not body.get("resume_url") and not body.get("resume_text") and not body.get("job_id") and not body.get("jd_url"):
        try:
            session_id_for_hash = body.get("session_id") or state.get("session_id")
            context_hash = await _compute_resume_content_context_hash_async(uid, session_id_for_hash, tenant_id)
            cache_key_data["request_type"] = "resume_content_generation"
            cache_key_data["context_hash"] = context_hash
            log.info(f"Storing resume_content_generator result with context hash: {context_hash[:16]}...")
        except Exception as e:
            log.warning(f"Failed to compute context hash for cache storage: {e}")
            # Don't cache if context hash computation fails (to avoid stale cache)

    cache.set(tenant_id, result, **cache_key_data)

    # Breadcrumb: cache store (MUST match dispatcher fingerprint calculation exactly)
    try:
        import json as _json, hashlib as _hashlib
        key_preview_source = {k: cache_key_data.get(k) for k in sorted(cache_key_data.keys()) if k in {"request_type","uid","jd_url_hash","assessment_plan_hash","submission_hash","resume_url_hash","user_interests_hash","job_details_hash", "job_id", "context_hash"}}
        key_fingerprint = _hashlib.sha256(_json.dumps(key_preview_source, sort_keys=True).encode()).hexdigest()[:12]
    except Exception:
        key_fingerprint = "unknown"
    log.info(
        "CACHE_STORE "
        f"pid={os.getpid()} tenant_id={tenant_id} uid={uid} "
        f"request_type={request_type} key_fp={key_fingerprint}"
    )
    
    # Store result in in-flight cache and signal waiting coroutines before cleanup
    key_id = f"{tenant_id}:{key_fingerprint}"
    try:
        with _inflight_lock:
            # Defensive check: ensure _inflight_keys is a dict
            if not isinstance(_inflight_keys, dict):
                log.error(f"⚠️ _inflight_keys is not a dict! Type: {type(_inflight_keys)}, value: {_inflight_keys}")
                # Reinitialize as dict if it got corrupted
                _inflight_keys = {}
            # Store result in in-flight cache and signal waiting coroutines
            if key_id in _inflight_keys:
                _inflight_keys[key_id]["result"] = result
                # Now running on event loop -- direct .set() is safe
                _inflight_keys[key_id]["event"].set()
                # Remove the key from in-flight tracking after signaling
                _inflight_keys.pop(key_id, None)
            else:
                # Key not in in-flight tracking, nothing to clean up
                pass
    except NameError as e:
        log.warning(f"⚠️ In-flight tracking variables not available: {e}, skipping cleanup for key: {key_id}")
    except Exception as e:
        log.warning(f"⚠️ Error during in-flight tracking cleanup: {e}, skipping cleanup for key: {key_id}")

    # Persist final envelope to session store for future DB backfill
    try:
        session_id = state.get("session_id", "")
        if session_id:
            from chroma import get_chat_session
            existing = await _rbi(get_chat_session, session_id) or {}
            bucket = existing.get("cached_results") or {}
            bucket[key_fingerprint] = result
            await _rbi(update_session_step,
                session_id, "cached_results", {"cached_results": bucket}, None
            )
            log.info(
                "DB_BACKFILL_STORE "
                f"pid={os.getpid()} tenant_id={tenant_id} uid={uid} "
                f"key_fp={key_fingerprint}"
            )
    except Exception:
        pass
    
    # Record metrics
    processing_time = state.get("processing_time_seconds", 0.0)
    metrics.record_response_time(processing_time)
    metrics.record_confidence(state.get("confidence_score", 0.0))
    metrics.record_method(state.get("analysis_method", "deterministic"))
    
    # Log structured data
    log.info(
        "Analysis completed - tenant: %s, method: %s, confidence: %.2f, time: %.2fs",
        tenant_id,
        state.get("analysis_method", "deterministic"),
        state.get("confidence_score", 0.0),
        processing_time,
    )


_middleware_manager = MiddlewareManager()

# Section 6 Issue 1: shared executor for sync agents (avoid per-call executor churn)
# Configurable so dispatcher and other sync nodes don't queue when many requests start at once
_sync_agent_max_workers = int(os.getenv("SYNC_AGENT_MAX_WORKERS", "20"))
_sync_agent_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_sync_agent_max_workers, thread_name_prefix="sync_agent"
)


def _record_agent_execution_trace(
    state: AgentState,
    agent_name: str,
    decision: str | None = None,
    tools_used: list[str] | None = None,
) -> AgentState:
    """
    Append a single step to the agent_execution_trace structure used for
    agentic verification. This is designed to be lightweight and universal
    across all workflows.
    """
    try:
        trace_entry = {
            "agent": agent_name,
            "input": sorted(list(state.keys())),
            "decision": decision,
            "tools_used": tools_used or [],
        }

        # Return only trace-related additions.  merge_lists / accumulate_int
        # reducers handle accumulation — no need to read+re-emit the full list.
        return {
            **state,
            "agent_execution_trace": [trace_entry],
            "graph_steps_executed": 1,
            "agents_invoked": [agent_name],
        }
    except Exception:
        return state


def apply_middleware(agent_fn, name):
    # Check if the function is async
    import asyncio

    if asyncio.iscoroutinefunction(agent_fn):

        async def traced_agent(state: AgentState) -> AgentState:
            result = await agent_fn(state)
            return _record_agent_execution_trace(result if isinstance(result, dict) else state, name)

        return _middleware_manager.apply(
            name,
            traced_agent,
            evaluation_fn=evaluate_agent_output_for_pii,
        )
    else:
        # For sync functions, run in shared thread pool (Section 6 Issue 1)

        async def async_wrapper(state: AgentState) -> AgentState:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(_sync_agent_executor, agent_fn, state)
            result_dict = result if isinstance(result, dict) else state
            return _record_agent_execution_trace(result_dict, name)

        return _middleware_manager.apply(
            name,
            async_wrapper,
            evaluation_fn=evaluate_agent_output_for_pii,
        )

# def jd_router(state: AgentState) -> str:
#     """
#     Routes the workflow after job description validation.
#     """
#     if is_valid_jd(state):
#         return "job_description_parser"
#     else:
#         return "end_invalid_jd"

def jd_router(state: AgentState) -> str:
    """
    Routes the workflow after job description validation.
    Checks the is_valid_jd field in state (not by calling async function).
    """
    try:
        # Get the validation result from state (already set by validate_jd_agent)
        is_valid = state.get("is_valid_jd", False)
        
        log.info(f"🔍 JD_ROUTER_DEBUG: is_valid_jd value from state: {is_valid}")
        log.info(f"🔍 JD_ROUTER_DEBUG: is_valid_jd type: {type(is_valid)}")
        
        if is_valid:
            log.info("🔍 JD_ROUTER_DEBUG: Routing to job_description_parser")
            return "job_description_parser"
        else:
            log.info("🔍 JD_ROUTER_DEBUG: Routing to end_invalid_jd")
            return "end_invalid_jd"
    
    except Exception as e:
        log.error(f"❌ jd_router error: {e}, falling back to 'end_invalid_jd'")
        return "end_invalid_jd"

def interest_router(state: AgentState) -> str:
    """
    Routes after interest filler completion.
    """
    return "start_parsing"

def parsing_router(state: AgentState) -> str:
    """
    Routes to the first parsing agent (personal_info_parser).
    """
    return "personal_info_parser"

def merge_job_details_with_parsed(parsed_jd: Dict[str, Any], job_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge job_details from payload with parsed JD results.
    When payload has job_title, experience, salary, etc., those replace the parsed values.
    Prioritizes job_details fields if they are not empty, otherwise uses parsed values.
    
    For skills: Combines must_have_skills (priority) with parsed requiredSkills; or uses
    job_details.required_skills/requiredSkills when provided.
    
    Args:
        parsed_jd: Parsed JD dictionary from parser
        job_details: Raw job_details from payload (jobTitle, experience, salary, etc.)
        
    Returns:
        Merged JD dictionary with job_details replacing parsed where provided
    """
    merged = {**parsed_jd}  # Start with parsed results
    
    def is_not_empty(value):
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        if isinstance(value, list):
            return len(value) > 0
        return True
    
    # job_title / jobTitle → replace parsed jobTitle
    job_title = job_details.get("job_title") or job_details.get("jobTitle")
    if is_not_empty(job_title):
        merged["jobTitle"] = job_title.strip() if isinstance(job_title, str) else str(job_title)
        log.debug(f"[MERGE] Using job_details.jobTitle: {merged['jobTitle']}")
    
    # ✅ SKILLS MERGING: Use job_details.required_skills/requiredSkills when provided (string or list); else combine must_have + parsed
    required_from_payload = job_details.get("required_skills") or job_details.get("requiredSkills")
    if isinstance(required_from_payload, str) and required_from_payload.strip():
        required_from_payload = [s.strip() for s in required_from_payload.split(",") if s and s.strip()]
    elif not isinstance(required_from_payload, list):
        required_from_payload = []
    must_have = job_details.get("must_have_skills") or job_details.get("mustHaveSkills")
    parsed_required_skills = merged.get("requiredSkills", []) or []

    if is_not_empty(required_from_payload):
        merged["requiredSkills"] = required_from_payload
        merged["_must_have_skills"] = must_have if isinstance(must_have, list) else ([must_have] if must_have else [])
        log.info(f"[MERGE] Using job_details.requiredSkills: {len(required_from_payload)} skills")
    elif is_not_empty(must_have):
        # Convert to list if needed
        must_have_list = must_have if isinstance(must_have, list) else [must_have]
        combined_skills = []
        combined_normalized = set()
        for skill in must_have_list:
            if skill and str(skill).strip():
                combined_skills.append(str(skill).strip())
                combined_normalized.add(str(skill).strip().lower())
        for skill in parsed_required_skills:
            if skill and str(skill).strip():
                skill_lower = str(skill).strip().lower()
                if skill_lower not in combined_normalized:
                    combined_skills.append(str(skill).strip())
                    combined_normalized.add(skill_lower)
        merged["requiredSkills"] = combined_skills
        merged["_must_have_skills"] = must_have_list
        log.info(f"[MERGE] Combined must_have + parsed requiredSkills: {len(combined_skills)} total")
    else:
        merged["requiredSkills"] = parsed_required_skills
        merged["_must_have_skills"] = []
        log.debug(f"[MERGE] Using parsed requiredSkills: {len(parsed_required_skills)} skills")
    
    # good_to_have_skills → preferredSkills
    good_to_have = job_details.get("good_to_have_skills") or job_details.get("goodToHaveSkills")
    if is_not_empty(good_to_have):
        merged["preferredSkills"] = good_to_have if isinstance(good_to_have, list) else [good_to_have]
        log.debug(f"[MERGE] Using job_details.preferredSkills: {merged['preferredSkills']}")
    elif not merged.get("preferredSkills"):
        # If no good_to_have in job_details and no parsed preferredSkills, set empty list
        merged["preferredSkills"] = []
    
    # educationRequired / education_qualification → educationRequired (replace parsed when provided)
    education = (
        job_details.get("educationRequired")
        or job_details.get("education_qualification")
        or job_details.get("educationQualification")
    )
    if is_not_empty(education):
        merged["educationRequired"] = education.strip() if isinstance(education, str) else str(education)
        log.debug(f"[MERGE] Using job_details.educationRequired: {merged['educationRequired']}")
    
    # companyName → company
    company_name = job_details.get("companyName") or job_details.get("company_name")
    if is_not_empty(company_name):
        merged["company"] = company_name
        log.debug(f"[MERGE] Using job_details.company: {company_name}")
    
    # work_location → workMode
    work_location = job_details.get("work_location") or job_details.get("workLocation")
    if is_not_empty(work_location):
        merged["workMode"] = work_location
        log.debug(f"[MERGE] Using job_details.workMode: {work_location}")
    
    # employment_type → jobType
    employment_type = job_details.get("employment_type") or job_details.get("employmentType")
    if is_not_empty(employment_type):
        merged["jobType"] = employment_type
        log.debug(f"[MERGE] Using job_details.jobType: {employment_type}")
    
    # experience → experience
    experience = job_details.get("experience")
    if is_not_empty(experience):
        merged["experience"] = experience
        log.debug(f"[MERGE] Using job_details.experience: {experience}")
    
    # salary (direct) or salary_range → salary (replace parsed with payload when provided)
    salary_val = job_details.get("salary") or job_details.get("salary_range") or job_details.get("salaryRange")
    if is_not_empty(salary_val):
        merged["salary"] = salary_val.strip() if isinstance(salary_val, str) else str(salary_val)
        log.debug(f"[MERGE] Using job_details.salary: {merged['salary']}")
    
    # location → location
    location = job_details.get("location")
    if is_not_empty(location):
        merged["location"] = location.strip() if isinstance(location, str) else str(location)
        log.debug(f"[MERGE] Using job_details.location: {merged['location']}")

    # full_job_description → fullJobDescription (use payload text when provided)
    full_desc = (
        job_details.get("full_job_description")
        or job_details.get("fullJobDescription")
        or job_details.get("structured_job_description")
    )
    if is_not_empty(full_desc):
        merged["fullJobDescription"] = full_desc.strip() if isinstance(full_desc, str) else str(full_desc)
        log.debug(f"[MERGE] Using job_details.fullJobDescription ({len(merged['fullJobDescription'])} chars)")
    
    log.info(f"[MERGE] Merged job_details with parsed JD (prioritized job_details where provided)")
    return merged


async def job_description_parser_node(state: AgentState) -> AgentState:
    """
    Job description parser node - wraps groq_jd_parser_agent.
    CRITICAL: Removes body and jd_text from output before returning.
    Does NOT modify is_valid_jd - that's already set by validate_jd_agent.
    """
    jd_text = state.get('jd_text', '')
    job_id = state.get('job_id')  # ✅ Get job_id from state
    uid = state.get('uid')        # ✅ Get uid from state
    company = state.get('company') or state.get('body', {}).get('company')  # ✅ ADD: Extract company
    job_details = state.get('job_details', {})  # ✅ Get job_details for merging
    
    if not jd_text:
        log.error("[JD_PARSER_NODE] No JD text provided")
        return {
            "job_description": None,
            "error": "No JD text provided"
        }
    
    # ✅ Log what we're passing
    log.info(f"[JD_PARSER_NODE] Starting parse for JD text ({len(jd_text)} chars)")
    log.info(f"[JD_PARSER_NODE] job_id={job_id}, uid={uid}")
    
    try:
        # Import the parser
        from agents.groq_jd_parser import groq_jd_parser_agent
        from chroma import update_job_description
        
        # ✅ CRITICAL FIX: Pass job_id and uid parameters!
        # skip_rerank_on_update=True: graph runs ranker after prescreening_questions; avoid duplicate ranker
        parsed_jd = await groq_jd_parser_agent(
            jd_text=jd_text,
            job_id=job_id,  # ✅ This enables storage!
            uid=uid,         # ✅ Pass uid for metadata
            company=company,
            skip_rerank_on_update=True,
        )
        
        if not parsed_jd:
            log.error("[JD_PARSER_NODE] Parser returned None")
            return {
                "job_description": None,
                "error": "Parser returned None"
            }
        
        log.info(f"[JD_PARSER_NODE] SUCCESS: Parsed job")
        
        # ✅ NEW: Merge job_details with parsed results (prioritize job_details if not empty)
        if job_details:
            log.info(f"[JD_PARSER_NODE] Merging job_details with parsed results")
            merged_jd = merge_job_details_with_parsed(parsed_jd, job_details)
            
            # Update ChromaDB with merged data
            if job_id:
                try:
                    from core.utils import run_blocking_io as _rbi
                    await _rbi(update_job_description, job_id, merged_jd)
                    log.info(f"[JD_PARSER_NODE] Updated ChromaDB with merged job_details")
                except Exception as update_err:
                    log.error(f"[JD_PARSER_NODE] Failed to update ChromaDB: {update_err}")
            
            parsed_jd = merged_jd
        
        log.info(f"[JD_PARSER_NODE] Should have stored to ChromaDB with job_id={job_id}")
        
        # ✅ Handle job status update if present (for edit operations)
        job_status = state.get("job_status")
        job_status_from_details = state.get("job_status_from_details", False)
        status_update_success = False
        
        if job_status and job_id and job_status_from_details:
            log.info(f"🔄 [JD_PARSER_NODE] Handling job status update: {job_status} for job_id={job_id}")
            try:
                from core.utils import run_blocking_io as _rbi
                if job_status == "closed":
                    success = await _rbi(move_job_to_closed,
                        job_id=job_id,
                        closed_by=uid
                    )
                    if success:
                        log.info(f"✅ [JD_PARSER_NODE] Job {job_id} closed successfully")
                        status_update_success = True
                    else:
                        log.warning(f"⚠️ [JD_PARSER_NODE] Failed to close job {job_id}")
                elif job_status == "reopen":
                    success = move_job_to_active(job_id)
                    if success:
                        log.info(f"✅ [JD_PARSER_NODE] Job {job_id} reopened successfully")
                        status_update_success = True
                    else:
                        log.warning(f"⚠️ [JD_PARSER_NODE] Failed to reopen job {job_id}")
            except Exception as status_error:
                log.error(f"❌ [JD_PARSER_NODE] Error updating job status: {status_error}")
                # Don't fail the entire operation if status update fails
                import traceback
                log.debug(f"Status update error traceback: {traceback.format_exc()}")
        
        # ✅ CRITICAL: Build clean result WITHOUT body and jd_text; preserve jd_url for router (end vs ranker)
        result = {
            **state,
            "job_description": parsed_jd,
            "company": company
        }
        if state.get("jd_url") is not None:
            result["jd_url"] = state["jd_url"]
        
        # ✅ Add job_status to result if it was updated
        if job_status_from_details and status_update_success:
            result["job_status"] = job_status
            result["job_status_updated"] = True
            log.info(f"✅ [JD_PARSER_NODE] Added job_status={job_status} to result")
        
        # ✅ Preserve prescreening params for downstream prescreening_questions agent
        body_for_prescreen = state.get("body") or {}
        prescreen_num = body_for_prescreen.get("prescreening_num_questions") or body_for_prescreen.get("prescreeningNumQuestions")
        if prescreen_num is not None:
            result["prescreening_num_questions"] = prescreen_num
        prescreen_types = body_for_prescreen.get("prescreening_question_types") or body_for_prescreen.get("prescreeningQuestionTypes")
        if prescreen_types:
            result["prescreening_question_types"] = prescreen_types

        # ✅ Remove body and jd_text from result
        result.pop('body', None)
        result.pop('jd_text', None)

        log.info(f"✅ [JD_PARSER_NODE] Removed body and jd_text from output")
        log.info(f"📋 [JD_PARSER_NODE] Final result keys: {list(result.keys())}")
        
        return result

    except TypeError as e:
        if "'company'" in str(e):
            # Handle case where groq_jd_parser_agent doesn't accept company parameter yet
            log.error(f"[JD_PARSER_NODE] groq_jd_parser_agent doesn't accept 'company' parameter yet")
            log.error(f"[JD_PARSER_NODE] Please update groq_jd_parser_agent to accept company parameter")
            log.error(f"[JD_PARSER_NODE] Falling back to adding company after parsing...")
            
            # Fallback: call without company and add it afterward
            parsed_jd = await groq_jd_parser_agent(
                jd_text=jd_text,
                job_id=job_id,
                uid=uid,
                skip_rerank_on_update=True,
            )
            
            if parsed_jd:
                # Add company to parsed result
                parsed_jd['company'] = company
                log.info(f"[JD_PARSER_NODE] Added company='{company}' to parsed result (fallback)")
                
                # Update in ChromaDB
                from chroma import update_job_description
                from core.utils import run_blocking_io as _rbi
                try:
                    await _rbi(update_job_description, job_id, parsed_jd)
                    log.info(f"[JD_PARSER_NODE] Updated ChromaDB with company='{company}'")
                except Exception as update_err:
                    log.error(f"[JD_PARSER_NODE] Failed to update ChromaDB with company: {update_err}")
            
            result = {
                **state,
                "job_description": parsed_jd,
                "company": company
            }
            result.pop('body', None)
            result.pop('jd_text', None)
            return result
        else:
            raise
        
    except Exception as e:
        log.error(f"[JD_PARSER_NODE] Exception: {type(e).__name__}: {str(e)}")
        import traceback
        log.error(traceback.format_exc())
        
        return {
            "job_description": None,
            "error": str(e)
        }

async def update_certificates_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: merge incoming certificates into the candidate's resume,
    derive skills from certificate names, optionally run skill proficiency
    analyzer, persist to ChromaDB, and return the result in state.
    """
    from chroma import fetch_structured_resume, get_resume_doc, upsert_resume_doc
    from core.utils import run_blocking_io

    uid = state.get("uid")
    tenant_id = state.get("tenant_id") or uid
    body = state.get("body") or {}
    callback_url = state.get("callback_url") or body.get("callback_url")
    certificates_list = body.get("certificates") or body.get("certificates_list") or []
    run_proficiency = body.get("run_proficiency_analysis", True)
    if isinstance(run_proficiency, str):
        run_proficiency = run_proficiency.lower() in ("true", "1", "yes")

    log.info(f"update_certificates_node: uid={uid}, certs={len(certificates_list)}, run_proficiency={run_proficiency}")

    structured_resume = await run_blocking_io(fetch_structured_resume, uid, tenant_id)
    if not structured_resume or not isinstance(structured_resume, dict):
        log.warning(f"update_certificates_node: structured_resume not found for uid={uid}")
        return {"next": "end", "error": f"Structured resume not found for UID '{uid}'"}

    # --- Deduplicate and merge certificates ---
    def _cert_dedup_alternate_key(key: str) -> str:
        if not key or not key.strip():
            return key
        parts = key.split()
        if not parts:
            return key
        last = parts[-1]
        if len(last) > 1 and last.endswith("s"):
            return " ".join(parts[:-1] + [last[:-1]])
        return " ".join(parts[:-1] + [last + "s"])

    existing_certs = list(structured_resume.get("certifications") or [])
    seen_cert_names = {
        _normalize_skill_name(c.get("certification_name") or c.get("name") or "").lower()
        for c in existing_certs if isinstance(c, dict)
    }
    for cert in certificates_list:
        if not isinstance(cert, dict):
            continue
        name = (cert.get("certification_name") or cert.get("name") or cert.get("title") or "").strip()
        if not name:
            continue
        key = _normalize_skill_name(name).lower()
        alt = _cert_dedup_alternate_key(key)
        if key in seen_cert_names or alt in seen_cert_names:
            continue
        seen_cert_names.add(key)
        existing_certs.append({
            "certification_name": name,
            "issuing_organization": cert.get("issuing_organization") or cert.get("issuer") or "",
            "year": cert.get("year"),
        })
    structured_resume["certifications"] = existing_certs

    # --- Derive and merge skills from certificate names ---
    derived_skills = extract_skills_from_certificate_names(certificates_list)
    existing_skills = list(structured_resume.get("skills") or [])
    skill_map = {}
    for s in existing_skills:
        if isinstance(s, dict):
            sn = (s.get("SkillName") or s.get("skill") or s.get("name") or "").strip()
            if sn:
                norm = _normalize_skill_name(sn).lower()
                if norm not in skill_map:
                    skill_map[norm] = dict(s)
    for display_name in derived_skills:
        norm = _normalize_skill_name(display_name).lower()
        if norm in skill_map:
            continue
        skill_map[norm] = {
            "SkillName": display_name,
            "Proficiency": "",
            "PositiveRationale": "",
            "NegativeRationale": "",
            "HowToImprove": "",
        }
    structured_resume["skills"] = list(skill_map.values())

    # --- Optionally run skill proficiency analyzer ---
    skills_without_supporting_evidence = []
    if run_proficiency:
        analyzer_state = {
            "uid": uid,
            "tenant_id": tenant_id,
            "structured_resume": structured_resume,
            "skills": structured_resume["skills"],
        }
        analyzer_result = await skill_proficiency_analyzer_agent(analyzer_state)
        updated_skills = analyzer_result.get("skills") or []
        skills_without_supporting_evidence = analyzer_result.get("skills_without_supporting_evidence") or []
        structured_resume["skills"] = updated_skills
    else:
        updated_skills = structured_resume["skills"]
        skills_without_supporting_evidence = structured_resume.get("skills_without_supporting_evidence") or []

    # --- Persist to ChromaDB ---
    doc = await run_blocking_io(get_resume_doc, uid) or {}
    doc["structured_resume"] = structured_resume
    await run_blocking_io(upsert_resume_doc, uid, doc)

    certificates_result = {
        "uid": uid,
        **structured_resume,
        "skills": updated_skills,
        "skills_without_supporting_evidence": skills_without_supporting_evidence,
    }

    log.info(f"update_certificates_node: done for uid={uid}, skills_count={len(updated_skills)}")

    return {
        "structured_resume": structured_resume,
        "skills": updated_skills,
        "skills_without_supporting_evidence": skills_without_supporting_evidence,
        "certificates_result": certificates_result,
        "next": "end",
    }


def create_graph():
    """Builds the complete agent workflow with a feedback loop."""
    log.info("Creating graph workflow...")
    builder = StateGraph(AgentState)

    # Add all nodes
    builder.add_node("dispatcher", apply_middleware(dispatcher, "dispatcher"))
    builder.add_node("validate_resume", apply_middleware(validate_resume_node, "validate_resume"))
    builder.add_node("validate_jd", apply_middleware(validate_jd_agent, "validate_jd"))

    builder.add_node("interest_filler", apply_middleware(interest_filler_agent, "interest_filler"))
    # DEPRECATED: resume_preprocessor replaced by single groq_resume_parser
    # builder.add_node("resume_preprocessor", apply_middleware(resume_preprocessor_agent, "resume_preprocessor"))
    # DEPRECATED: Old parsing agents replaced by single groq_resume_parser
    # builder.add_node("personal_info_parser", apply_middleware(personal_info_parser_agent, "personal_info_parser"))
    # builder.add_node("experience_parser", apply_middleware(experience_parser_agent, "experience_parser"))
    # builder.add_node("education_parser", apply_middleware(education_parser_agent, "education_parser"))
    # builder.add_node("skills_parser", apply_middleware(skills_parser_agent, "skills_parser"))
    builder.add_node("groq_resume_parser", apply_middleware(groq_resume_parser_agent, "groq_resume_parser"))
    builder.add_node("skill_proficiency_analyzer", apply_middleware(skill_proficiency_analyzer_agent, "skill_proficiency_analyzer"))
    builder.add_node("resume_assembler", apply_middleware(resume_assembler_agent, "resume_assembler"))
    builder.add_node("resume_summary", apply_middleware(resume_summary_agent, "resume_summary"))
    builder.add_node("resume_scorer", apply_middleware(resume_scorer_agent, "resume_scorer"))
    builder.add_node("update_certificates", apply_middleware(update_certificates_node, "update_certificates"))
    
    # Create wrapper for resume_content_generator to match standard agent signature
    async def resume_content_generator_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for resume content generator to integrate with supervisor workflow"""
        from core.utils import run_blocking_io

        uid = state.get("uid")
        tenant_id = state.get("tenant_id") or uid
        session_id = state.get("session_id")
        
        if not uid:
            return {
                "error": "Missing uid",
                "next": "end"
            }

        # Persist optional certificates to Chroma when provided in body/state (same as /generate-resume-content)
        certificates = state.get("certificates") or (state.get("body") or {}).get("certificates")
        if certificates and isinstance(certificates, list) and len(certificates) > 0 and len(certificates) <= 50:
            try:
                doc = await run_blocking_io(get_resume_doc, uid) or {}
                doc["uploaded_certificates"] = [c for c in certificates if isinstance(c, dict)][:50]
                await run_blocking_io(upsert_resume_doc, uid, doc)
                log.info(f"resume_content_generator: Persisted {len(doc['uploaded_certificates'])} certificate(s) to resume_doc for uid={uid}")
            except Exception as e:
                log.warning(f"resume_content_generator: Failed to persist certificates to Chroma (non-fatal): {e}")
        
        log.info(f"resume_content_generator: Generating resume content for uid={uid}")
        
        result = await resume_content_generator_agent(
            uid=uid,
            tenant_id=tenant_id,
            session_id=session_id
        )
        
        return {
            **result,
            "next": "end",
        }
    
    builder.add_node("resume_content_generator", apply_middleware(resume_content_generator_wrapper, "resume_content_generator"))

    async def job_matcher_wrapper(state: AgentState) -> Dict[str, Any]:
        from agents.skill_utils import extract_primary_skills
        from core.utils import run_blocking_io as _rbi

        uid = state.get("uid") or state.get("body", {}).get("uid")
        session_id = state.get("session_id") or state.get("body", {}).get("session_id")
        structured_resume = state.get("structured_resume")

        # Career flow can pass a minimal resume (e.g. only user_interests_summary). Hydrate from Chroma if no skills.
        if uid and structured_resume and isinstance(structured_resume, dict):
            skills = extract_primary_skills(structured_resume)
            if not skills:
                full_resume = None
                try:
                    doc = await _rbi(get_resume_doc, uid)
                    if doc and isinstance(doc, dict) and doc.get("structured_resume"):
                        cand = doc["structured_resume"]
                        if isinstance(cand, dict) and extract_primary_skills(cand):
                            full_resume = cand
                            log.info("✅ job_matcher: Hydrated structured_resume from get_resume_doc (had no skills)")
                except Exception as e:
                    log.debug(f"job_matcher get_resume_doc: {e}")
                if not full_resume:
                    try:
                        cand = await _rbi(fetch_structured_resume, uid, current_session_id=session_id)
                        if cand and isinstance(cand, dict) and extract_primary_skills(cand):
                            full_resume = cand
                            log.info("✅ job_matcher: Hydrated structured_resume from fetch_structured_resume (had no skills)")
                    except Exception as e:
                        log.debug(f"job_matcher fetch_structured_resume: {e}")
                if full_resume:
                    # Preserve user_interests_summary from minimal resume if full resume lacks it
                    if structured_resume.get("user_interests_summary") and not full_resume.get("user_interests_summary"):
                        full_resume = dict(full_resume)
                        full_resume["user_interests_summary"] = structured_resume["user_interests_summary"]
                    state = {**state, "structured_resume": full_resume}
                    structured_resume = full_resume

        current_hash = None
        if structured_resume and isinstance(structured_resume, dict):
            current_hash = structured_resume.get("_resume_hash")
            if not current_hash:
                try:
                    from agents.prompt_generator import _calculate_resume_hash
                    current_hash = _calculate_resume_hash(structured_resume)
                except Exception:
                    current_hash = None

        if uid:
            cached = await _rbi(get_job_matcher_ranking, uid)
            if cached is not None and current_hash is not None and cached.get("_input_hash") == current_hash:
                return {**state, **cached}


        # Disable LangSmith tracing for job_matcher to avoid rate limit (429) when tenant exceeds usage
        try:
            from langsmith.run_helpers import tracing_context
            with tracing_context(enabled=False):
                result = await job_matcher_agent(state)
        except ImportError:
            result = await job_matcher_agent(state)

        if uid and not result.get("error") and result.get("status") != "error":
            await _rbi(upsert_job_matcher_ranking, uid, result, input_hash=current_hash)
        return {**state, **result}
    builder.add_node("job_matcher", apply_middleware(job_matcher_wrapper, "job_matcher"))

    # ✅ ADD CANDIDATE JOB MATCHER NODE
    async def candidate_job_matcher_agent(state: AgentState) -> Dict[str, Any]:
        """
        Compare one candidate with one job. Run analysis when: no ranking stored for (uid, job_id),
        or when recruiter_questions / interview_feedback are present. Otherwise use cached ranking.
        """
        uid = state.get("uid") or state.get("body", {}).get("uid")
        job_id = state.get("job_id") or state.get("body", {}).get("job_id")
        
        # Check both state and body for recruiter_questions (multiple possible locations)
        recruiter_questions = (
            state.get("recruiter_questions") or 
            state.get("body", {}).get("recruiter_questions") or
            state.get("body", {}).get("recruiterQuestions") or
            state.get("body", {}).get("questions")
        )
        
        # Optional post-interview feedback for hire recommendation
        interview_feedback = (
            state.get("interview_feedback")
            or state.get("body", {}).get("interview_feedback")
            or state.get("body", {}).get("interviewFeedback")
        )
        
        if not uid or not job_id:
            return {
                "error": "Missing uid or job_id",
                "candidate_job_match_result": {}
            }
        
        # Use cache only when no recruiter Q&A and no interview feedback; otherwise always run analysis
        from core.utils import run_blocking_io as _rbi
        use_cache = not (recruiter_questions or interview_feedback)
        if use_cache:
            cached = await _rbi(get_candidate_job_ranking, uid, job_id)
            if cached is not None:
                return {
                    "candidate_job_match_result": cached,
                    "match_score": cached.get("match_score", 0.0),
                    "skill_match_percentage": cached.get("skill_match_percentage", 0.0),
                    "skill_match_count": cached.get("skill_match_count", 0),
                    "skills_matched": cached.get("skills_matched", []),
                    "skills_unmatched": cached.get("skills_unmatched", []),
                    "hire_recommendation": cached.get("hire_recommendation"),
                    "hire_rationale": cached.get("hire_rationale"),
                }
        
        if recruiter_questions:
            log.info(f"✅ Found {len(recruiter_questions) if isinstance(recruiter_questions, list) else 1} recruiter Q&A in state")
        else:
            log.debug("⚠️ No recruiter questions found in state or body - Q&A alignment will be skipped")
        
        if interview_feedback:
            log.info(f"📋 Found interview feedback in state — will include in candidate-job analysis")
        
        result = await compare_candidate_with_job(
            uid, job_id,
            recruiter_questions=recruiter_questions,
            interview_feedback=interview_feedback
        )
        
        if not result.get("error"):
            await _rbi(upsert_candidate_job_ranking, uid, job_id, result)
        
        return {
            "candidate_job_match_result": result,
            "match_score": result.get("match_score", 0.0),
            "skill_match_percentage": result.get("skill_match_percentage", 0.0),
            "skill_match_count": result.get("skill_match_count", 0),
            "skills_matched": result.get("skills_matched", []),
            "skills_unmatched": result.get("skills_unmatched", []),
            "hire_recommendation": result.get("hire_recommendation"),
            "hire_rationale": result.get("hire_rationale"),
        }
    
    # ✅ Job matcher preprocessor: resolves job_ids, fetches job docs, checks cache, builds optional_reviewer_qns
    # Disable LangSmith tracing for preprocessor too (part of job_matcher flow, avoids rate limit)
    async def job_matcher_preprocessor_wrapper(state: AgentState) -> Dict[str, Any]:
        try:
            from langsmith.run_helpers import tracing_context
            with tracing_context(enabled=False):
                return await job_matcher_preprocessor(state)
        except ImportError:
            return await job_matcher_preprocessor(state)
    builder.add_node("job_matcher_preprocessor", apply_middleware(job_matcher_preprocessor_wrapper, "job_matcher_preprocessor"))

    builder.add_edge("job_matcher_preprocessor", "job_matcher")

    def career_advisor_entry(state: AgentState) -> Dict:
        """Entry point for career advisor flow - prepares data for career analysis.
        Sets _job_matcher_parallel=True so assessment_recommender routes to career_chain_sink."""
        return {
            "career_advisor_entry_status": "completed",
            "message": "Career advisor flow initiated",
            "next_step": "career_advisor",
            "_job_matcher_parallel": True,
        }
        
    def start_parsing(state: AgentState) -> Dict:
        # Pass through the state without modification to preserve agent_segments
        return state

    builder.add_node("start_parsing", apply_middleware(start_parsing, "start_parsing"))
    builder.add_node("career_advisor_entry", apply_middleware(career_advisor_entry, "career_advisor_entry"))
    builder.add_node("enhanced_role_fit", apply_middleware(enhanced_role_fit_agent, "enhanced_role_fit"))

    def career_flow_fan_out(state: AgentState) -> Dict:
        """Passthrough for career flow - fans out to career_advisor and job_matcher_preprocessor (parallel).
        Compare+resume never reaches here (enhanced_role_fit_router sends it straight to career_advisor).
        Compare without resume in body: legacy second-pass multi-job Chroma retrieval.
        IMPORTANT: Only return CHANGED fields — returning dict(state) re-applies merge_lists reducers and doubles
        education/work_experience/skills/etc."""
        is_compare = (state.get("endpoint_name") or (state.get("body") or {}).get("endpoint_name")) == "compare_candidate_job"
        if is_compare:
            log.info("career_flow_fan_out: Compare (no resume in body) — legacy multi-job Chroma retrieval")
            return {
                "_compare_flow_multi_job_run": True,
                "_job_matcher_parallel": True,
                "job_id": None,
                "job_ids": [],
                "next": "",
            }
        return {}

    builder.add_node("career_flow_fan_out", apply_middleware(career_flow_fan_out, "career_flow_fan_out"))
    builder.add_edge("career_flow_fan_out", "career_advisor")
    builder.add_edge("career_flow_fan_out", "job_matcher_preprocessor")

    builder.add_node("career_advisor", apply_middleware(skill_and_career_advisor_agent, "career_advisor"))
    builder.add_node("market_and_course_recommender", apply_middleware(market_and_course_recommender_agent, "market_and_course_recommender"))
    builder.add_node("assessment_recommender", apply_middleware(assessment_recommender_agent, "assessment_recommender"))
    builder.add_node("assessment_question_generator", apply_middleware(assessment_question_generator_agent, "assessment_question_generator"))
    builder.add_node("assessment_builder_agent", apply_middleware(assessment_builder_agent, "assessment_builder_agent"))
    # Create wrapper for assessment evaluator to match standard agent signature
    async def assessment_evaluator_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        return await assessment_evaluator_agent(state)
    
    builder.add_node("assessment_evaluator", apply_middleware(assessment_evaluator_wrapper, "assessment_evaluator"))
    builder.add_node("report_generator", apply_middleware(report_generator_agent, "report_generator"))

    def _career_recommender_outputs_present(state: Dict[str, Any]) -> bool:
        """True after assessment_recommender merged plan/needs into state (even if plan is [])."""
        if "assessment_plan" in state or "assessment_needs" in state:
            return True
        nested = state.get("assessment_recommender")
        if isinstance(nested, dict) and ("assessment_plan" in nested or "assessment_needs" in nested):
            return True
        return False
    
    # Fan-in guard (parallel career + job_matcher):
    # - Never run assessment_validator_agent until assessment_recommender has written assessment_plan / assessment_needs
    #   (avoids job_matcher-only early invocations).
    # - Then require both job_matcher_status and career outputs before real validation.
    # On skip, route to parallel_join_wait (dead end) instead of end so the graph does not terminate
    # before the other branch can trigger a second join attempt.
    async def assessment_validator_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("_job_matcher_parallel"):
            has_job_data = bool(state.get("job_matcher_status"))
            has_career_data = _career_recommender_outputs_present(state)
            # Legacy compare 2nd job_matcher may reach here with unusual state; do not block on career outputs.
            if not state.get("_compare_flow_multi_job_run"):
                if not has_career_data:
                    log.info(
                        "⏳ assessment_validator: skipping — assessment_recommender not finished "
                        "(no assessment_plan / assessment_needs in state)"
                    )
                    return {"_skip_callback": True}
            if not (has_job_data and has_career_data):
                arrived = "job_matcher" if has_job_data else "career chain"
                log.info(
                    f"⏳ assessment_validator: only {arrived} branch ready — skipping join until both complete"
                )
                return {"_skip_callback": True}
        out = await assessment_validator_agent(state)
        # Clear compare fan-out flags after successful join so session/checkpoint cannot re-trigger routing on replay
        if isinstance(out, dict):
            if state.get("_compare_flow_multi_job_run"):
                out = {**out, "_compare_flow_multi_job_run": False}
            out = {**out, "_skip_callback": False}
        return out

    async def parallel_join_wait(state: Dict[str, Any]) -> Dict[str, Any]:
        """Dead-end after a skipped assessment_validator join (other branch still running).

        Hot-path safe: no I/O, no CPU-heavy work — only clears _skip_callback for the next join.
        Registered without full middleware to avoid sync logging / rate-limit / PII passes on every skip.
        """
        if log.isEnabledFor(logging.DEBUG):
            log.debug("parallel_join_wait: fan-in skip consumed; clearing _skip_callback for next join attempt")
        return {"_skip_callback": False}

    async def parallel_join_wait_traced(state: AgentState) -> AgentState:
        """Lightweight trace only (no MiddlewareManager) — internal glue node."""
        out = await parallel_join_wait(state)
        result = out if isinstance(out, dict) else state
        return _record_agent_execution_trace(result, "parallel_join_wait")

    async def parallel_job_matcher_done(state: Dict[str, Any]) -> Dict[str, Any]:
        """Dead-end when parallel job_matcher finishes before assessment_recommender merges plan/needs."""
        log.info(
            "parallel_job_matcher_done: parallel job_matcher branch finished; "
            "assessment_validator is reached only from assessment_recommender after AR completes"
        )
        return {}

    async def parallel_job_matcher_done_traced(state: AgentState) -> AgentState:
        """Lightweight trace only (no MiddlewareManager) — internal glue node."""
        out = await parallel_job_matcher_done(state)
        result = out if isinstance(out, dict) else state
        return _record_agent_execution_trace(result, "parallel_job_matcher_done")

    builder.add_node("parallel_join_wait", parallel_join_wait_traced)
    builder.add_node("parallel_job_matcher_done", parallel_job_matcher_done_traced)
    builder.add_node("assessment_validator", apply_middleware(assessment_validator_wrapper, "assessment_validator"))
    
    builder.add_node("job_description_parser", apply_middleware(job_description_parser_node, "job_description_parser"))
    builder.add_node("prescreening_questions", apply_middleware(prescreening_questions_agent, "prescreening_questions"))
    builder.add_node("ranker", apply_middleware(ranker_agent, "ranker"))
    builder.add_node("notification_agent", apply_middleware(notification_agent_http, "notification_agent"))
    
    async def end_flow(state: AgentState) -> AgentState:
        from core.utils import run_blocking_io as _rbi_end
        log.info("Pipeline finished successfully.")

        # Best-effort: write agent_execution_trace to file (offloaded to thread)
        try:
            trace = state.get("agent_execution_trace") or []
            if isinstance(trace, list):
                import json as _json
                from pathlib import Path as _Path

                def _write_trace():
                    logs_dir = _Path("logs")
                    logs_dir.mkdir(parents=True, exist_ok=True)
                    trace_path = logs_dir / "agentic_trace.json"
                    payload = {
                        "agent_execution_trace": trace,
                        "graph_steps_executed": state.get("graph_steps_executed"),
                        "agents_invoked": state.get("agents_invoked"),
                    }
                    with trace_path.open("w", encoding="utf-8") as f:
                        _json.dump(payload, f, indent=2, ensure_ascii=False)

                await _rbi_end(_write_trace)
        except Exception as e:
            log.debug(f"Failed to write agentic_trace.json (non-fatal): {e}")

        # Debug: Log results if they exist
        if "candidate_job_match_result" in state:
            log.info(f"✅ Candidate job match result found in state: {state.get('candidate_job_match_result', {}).get('match_score', 'N/A')}")
            log.info(f"   Match score: {state.get('match_score', 'N/A')}")
            log.info(f"   Skills matched: {len(state.get('skills_matched', []))}")
        elif "assessment_results" in state:
            log.info(f"Assessment results found in state: {state['assessment_results']}")
        elif "generated_questions" in state:
            log.info(f"Generated questions found in state: {state['generated_questions']}")
        elif "matched_jobs" in state:
            log.info(f"Matched jobs found in state: {len(state['matched_jobs'])} jobs")
        else:
            log.warning("No candidate_job_match_result, assessment_results, or generated_questions found in final state")
        
        # Complete session if it exists
        session_id = state.get("session_id", "")
        if session_id:
            try:
                await _rbi_end(session_manager.complete_session, session_id)
                log.info(f"✅ Session completed: {session_id}")
            except Exception as e:
                log.warning(f"Failed to complete session {session_id}: {e}")
        
        # Build response envelope by READING state (no {**state} spread — that doubles merge_lists fields).
        response = create_standard_response_envelope(
            state,
            analysis_method=state.get("analysis_method", "deterministic"),
            method_explain=state.get("method_explain", {}),
            skip_filtering=False,  # Use intelligent filtering — don't dump all state fields
        )

        # Cache result and record metrics (side-effect only — reads state)
        await cache_and_record_metrics(state, response)

        # Build a MINIMAL return dict with only envelope metadata + fields not already in graph state.
        # LangGraph already holds education/skills/work_experience/etc. from prior nodes; re-returning
        # them here would cause merge_lists reducers to double those lists.
        final_result = {
            "confidence_score": response.get("confidence_score", 0.0),
            "confidence_level": response.get("confidence_level", "low"),
            "processing_time_seconds": response.get("processing_time_seconds", 0.0),
        }

        # Include candidate-job comparison fields (scalar / take_last — safe to return)
        if state.get("candidate_job_match_result"):
            final_result["candidate_job_match_result"] = state["candidate_job_match_result"]
            log.info("✅ Explicitly included candidate_job_match_result in final output")
        if state.get("top_matches"):
            final_result["top_matches"] = state["top_matches"]
            log.info(f"✅ END NODE: top_matches included ({len(state['top_matches'])} jobs)")
        candidate_job_fields = [
            "match_score", "skill_match_percentage", "skill_match_count",
            "skills_matched", "skills_unmatched", "total_required_skills",
            "rationale", "skill_matching_diagnostics",
            "hire_recommendation", "hire_rationale",
        ]
        for field in candidate_job_fields:
            val = state.get(field)
            if val is not None:
                final_result[field] = val
        if final_result.get("candidate_job_match_result"):
            match_result = final_result["candidate_job_match_result"]
            log.info(f"✅ END NODE: candidate_job_match_result included in final output")
            log.info(f"   Match score: {match_result.get('match_score', 'N/A')}")
            log.info(f"   Skill match %: {match_result.get('skill_match_percentage', 'N/A')}")
            log.info(f"   Skills matched: {len(match_result.get('skills_matched', []))}")
            log.info(f"   Skills unmatched: {len(match_result.get('skills_unmatched', []))}")
            if match_result.get("hire_recommendation"):
                log.info(f"   Hire recommendation: {match_result.get('hire_recommendation')}")
                log.info(f"   Hire rationale: {match_result.get('hire_rationale', 'N/A')}")

        # Manual assessment builder: preserve analysis_type and ok flag
        if state.get("analysis_type") == "manual_assessment_builder":
            final_result["analysis_type"] = "manual_assessment_builder"
            if state.get("ok") is False:
                final_result["ok"] = False

        # Prescreening questions (corporate flow)
        if state.get("prescreening_questions"):
            final_result["prescreening_questions"] = state["prescreening_questions"]
            log.info(f"✅ END NODE: prescreening_questions included ({len(state['prescreening_questions'])} questions)")

        return final_result

    def end_invalid_resume_flow(state: AgentState) -> AgentState:
        global _inflight_keys
        log.error("Pipeline terminated due to invalid resume.")
        error_message = state.get("validation_error", "The provided resume is invalid.")
        # Coalescing cleanup (no cache write in invalid flow)
        try:
            tenant_id = state.get("tenant_id", "default-tenant")
            uid = state.get("uid", "")
            body = state.get("body", {})
            request_type = "resume_analysis"
            if body.get("plan"): request_type = "assessment_generation"
            elif body.get("submission"): request_type = "assessment_evaluation"
            elif body.get("jd_url"): request_type = "job_description"
            import json as _json, hashlib as _hashlib
            key_preview_source = {k: v for k, v in {"request_type": request_type, "uid": uid}.items()}
            key_fingerprint = _hashlib.sha256(_json.dumps(key_preview_source, sort_keys=True).encode()).hexdigest()[:12]
            key_id = f"{tenant_id}:{key_fingerprint}"
            with _inflight_lock:
                # Defensive check: ensure _inflight_keys is a dict
                if not isinstance(_inflight_keys, dict):
                    log.error(f"⚠️ _inflight_keys is not a dict! Type: {type(_inflight_keys)}, value: {_inflight_keys}")
                    # Reinitialize as dict if it got corrupted
                    _inflight_keys = {}
                _inflight_keys.pop(key_id, None)
        except Exception:
            pass

        # Complete session if it exists (even on error - force completion for error cases)
        session_id = state.get("session_id", "")
        if session_id:
            try:
                session_manager.complete_session(session_id, force=True)
                log.info(f"✅ Session completed (error): {session_id}")
            except Exception as e:
                log.warning(f"Failed to complete session {session_id}: {e}")
        
        # Create error response envelope
        response = {
            "ok": False,
            "analysis_method": "error",
            "method_explain": {"error_type": "validation_error", "error_message": error_message},
            "confidence_score": 0.0,
            "confidence_level": "low",
            "analysis_context": "general",
            "skill_gap_analysis": {
                "marketAnalysis": [],
                "missingSkills": [],
                "careerAdvice": [],
                "recommendation": [],
                "industry_context": "general",
                "skill_breakdown": {
                    "base_skills_missing": [],
                    "industry_skills_missing": [],
                    "base_coverage": 0.0,
                    "industry_coverage": 0.0
                },
                "analysis_id": state.get("analysis_id", generate_analysis_id())
            },
            "processing_time_seconds": state.get("processing_time_seconds", 0.0),
            "user_context": {
                "user_id": state.get("uid", ""),
                "tenant_id": state.get("tenant_id", ""),
                "analysis_timestamp": datetime.utcnow().isoformat()
            },
            "error": error_message,
            "validation_error": error_message
        }
        
        # Record error metrics
        metrics.record_method("error")
        metrics.record_response_time(state.get("processing_time_seconds", 0.0))
        
        return response

    def end_invalid_jd_flow(state: AgentState) -> AgentState:
        global _inflight_keys
        log.error("Pipeline terminated due to invalid job description.")
        error_message = state.get("validation_error", "The provided job description is invalid.")
        # Coalescing cleanup (no cache write in invalid flow)
        try:
            tenant_id = state.get("tenant_id", "default-tenant")
            uid = state.get("uid", "")
            body = state.get("body", {})
            request_type = "resume_analysis"
            if body.get("plan"): request_type = "assessment_generation"
            elif body.get("submission"): request_type = "assessment_evaluation"
            elif body.get("jd_url"): request_type = "job_description"
            import json as _json, hashlib as _hashlib
            key_preview_source = {k: v for k, v in {"request_type": request_type, "uid": uid}.items()}
            key_fingerprint = _hashlib.sha256(_json.dumps(key_preview_source, sort_keys=True).encode()).hexdigest()[:12]
            key_id = f"{tenant_id}:{key_fingerprint}"
            with _inflight_lock:
                # Defensive check: ensure _inflight_keys is a dict
                if not isinstance(_inflight_keys, dict):
                    log.error(f"⚠️ _inflight_keys is not a dict! Type: {type(_inflight_keys)}, value: {_inflight_keys}")
                    # Reinitialize as dict if it got corrupted
                    _inflight_keys = {}
                _inflight_keys.pop(key_id, None)
        except Exception:
            pass

        # Complete session if it exists (even on error - force completion for error cases)
        session_id = state.get("session_id", "")
        if session_id:
            try:
                session_manager.complete_session(session_id, force=True)
                log.info(f"✅ Session completed (error): {session_id}")
            except Exception as e:
                log.warning(f"Failed to complete session {session_id}: {e}")
        
        # Create error response envelope
        response = {
            "ok": False,
            "analysis_method": "error",
            "method_explain": {"error_type": "validation_error", "error_message": error_message},
            "confidence_score": 0.0,
            "confidence_level": "low",
            "analysis_context": "general",
            "skill_gap_analysis": {
                "marketAnalysis": [],
                "missingSkills": [],
                "careerAdvice": [],
                "recommendation": [],
                "industry_context": "general",
                "skill_breakdown": {
                    "base_skills_missing": [],
                    "industry_skills_missing": [],
                    "base_coverage": 0.0,
                    "industry_coverage": 0.0
                },
                "analysis_id": state.get("analysis_id", generate_analysis_id())
            },
            "processing_time_seconds": state.get("processing_time_seconds", 0.0),
            "user_context": {
                "user_id": state.get("uid", ""),
                "tenant_id": state.get("tenant_id", ""),
                "analysis_timestamp": datetime.utcnow().isoformat()
            },
            "error": error_message,
            "validation_error": error_message
        }
        
        # Record error metrics
        metrics.record_method("error")
        metrics.record_response_time(state.get("processing_time_seconds", 0.0))
        
        return response

    def end_enhanced_jd_flow(state: AgentState) -> AgentState:
        """Success flow: JD was enhanced only; return enhanced text to user for re-submission to validation."""
        enhanced = state.get("enhanced_jd_text") or ""
        response = {
            "ok": True,
            "analysis_method": "enhance_jd_only",
            "method_explain": {"enhance_jd_only": True, "message": "Job description enhanced; send again for validation."},
            "enhanced_jd_text": enhanced,
            "jd_text": state.get("jd_text", ""),
            "processing_time_seconds": state.get("processing_time_seconds", 0.0),
            "user_context": {
                "user_id": state.get("uid", ""),
                "tenant_id": state.get("tenant_id", ""),
                "analysis_timestamp": datetime.utcnow().isoformat(),
            },
        }
        return response

    async def enhance_jd_node(state: AgentState) -> AgentState:
        """Enhance JD text only; do not validate or parse. User receives enhanced text and can send again for validation."""
        from agents.jd_enhancer import enhance_jd_text
        import time
        jd_text = state.get("jd_text", "")
        if not jd_text or not str(jd_text).strip():
            return {
                "enhanced_jd_text": None,
                "error": "No JD text provided for enhancement",
            }
        start = time.time()
        enhanced = await enhance_jd_text(jd_text)
        elapsed = time.time() - start
        return {
            "enhanced_jd_text": enhanced or "",
            "processing_time_seconds": elapsed,
        }

    builder.add_node("end", apply_middleware(end_flow, "end"))
    builder.add_node("end_invalid_resume", apply_middleware(end_invalid_resume_flow, "end_invalid_resume"))
    builder.add_node("end_invalid_jd", apply_middleware(end_invalid_jd_flow, "end_invalid_jd"))
    builder.add_node("end_enhanced_jd", apply_middleware(end_enhanced_jd_flow, "end_enhanced_jd"))
    builder.add_node("enhance_jd", apply_middleware(enhance_jd_node, "enhance_jd"))


    # --- Define Workflow Edges ---
    builder.set_entry_point("dispatcher")

    def dispatcher_router(state: AgentState) -> str:
        """Router function for dispatcher that handles None values safely."""
        next_node = state.get("next")
        user_interests = state.get("user_interests")
        
        log.info(
            f"🔀 dispatcher_router: next_node={next_node}, "
            f"has_user_interests={bool(user_interests)}"
        )
        log.debug(
            f"🔀 dispatcher_router STATE KEYS: {list(state.keys())[:10]}..."
        )  # Show first 10 keys

        # if next_node == "candidate_job_matcher":
        #     log.warning("⚠️ Old node name 'candidate_job_matcher' detected, routing to 'ranker'")
        #     next_node = "ranker"
        
        if next_node is None:
            log.error("❌ dispatcher_router: next_node is None! Routing to end")
            return "end"
        
        # Valid routing options
        valid_routes = {
            "validate_resume",
            "groq_resume_parser",
            "validate_jd",
            "enhance_jd",
            "ranker",
            "notification_agent",
            "assessment_evaluator",
            "assessment_question_generator",
            "assessment_builder_agent",
            "job_matcher",
            "job_matcher_preprocessor",
            "career_advisor",
            "enhanced_role_fit",
            "resume_content_generator",
            "update_certificates",
            "end"
        }
        
        if next_node not in valid_routes:
            log.error(
                f"❌ dispatcher_router: Invalid next_node '{next_node}', routing to end"
            )
            return "end"
        
        # Double-check: if user_interests are present but routing to validate_resume, fix it
        if user_interests and isinstance(user_interests, list) and len(user_interests) > 0:
            if next_node == "validate_resume":
                log.warning(
                    "⚠️ FLOW 2 FIX: user_interests present but routing to "
                    "validate_resume, correcting to groq_resume_parser"
                )
                return "groq_resume_parser"
        
        log.info(f"✅ dispatcher_router: Routing to {next_node}")
        return next_node

    builder.add_conditional_edges(
        "dispatcher",
        dispatcher_router,
        {
            "validate_resume": "validate_resume",  # Re-enabled validation
            "groq_resume_parser": "groq_resume_parser",
            "validate_jd": "validate_jd",
            "enhance_jd": "enhance_jd",  # JD enhance only; user gets enhanced text and can send again for validation
            "ranker": "ranker",# ✅ NEW: Direct route to ranker
            "notification_agent": "notification_agent",
            "assessment_evaluator": "assessment_evaluator",
            "assessment_question_generator": "assessment_question_generator",
            "assessment_builder_agent": "assessment_builder_agent",
            "job_matcher": "job_matcher_preprocessor",  # ✅ Preprocessor → job_matcher
            "career_advisor": "career_advisor_entry",  # ✅ NEW: Route for 2nd call flow
            "resume_content_generator": "resume_content_generator",  # ✅ NEW: Route for resume content generation
            "update_certificates": "update_certificates",
            # "interest_filler": "interest_filler",
            "end": "end"
        }
    )

    builder.add_edge("update_certificates", "end")

    builder.add_conditional_edges(
        "validate_resume",
        resume_router,
        {
            "interest_filler": "interest_filler",
            "groq_resume_parser": "groq_resume_parser",  # For stored structured_resume flow
            "end_invalid_resume": "end_invalid_resume"
        }
    )

    # ✅ OPTIMIZED FLOW: Parallel processing for Stage 1, sequential for Stage 2
    # Stage 1: After groq_resume_parser, run interest_filler, resume_summary, and skill_proficiency_analyzer in parallel
    # Stage 2: skill_proficiency_analyzer -> resume_scorer -> resume_assembler (resume_scorer runs BEFORE assembler)
    #   - Normal flow: resume_assembler -> career_advisor_entry or end
    #   - /compare-candidate-job: resume_assembler -> job_matcher_preprocessor -> job_matcher -> ...
    # Flow: validate_resume -> groq_resume_parser -> [interest_filler + resume_summary + skill_proficiency_analyzer (parallel)] -> resume_scorer -> resume_assembler -> [job_matcher (compare) OR career/end (normal)]
    
    # Router after groq_resume_parser - always routes to interest_filler (parallel execution)
    # job_matcher will run after resume_assembler (when data is saved to ChromaDB)
    def groq_resume_parser_router(state: AgentState) -> str:
        """Routes after groq_resume_parser to interest_filler (triggers parallel nodes)."""
        try:
            body = state.get("body") or {}
            request_type = state.get("request_type") or body.get("request_type")
            log.info(f"🔍 groq_resume_parser_router: routing to interest_filler (request_type={request_type})")
            return "interest_filler"
        except Exception as e:
            log.error(f"❌ groq_resume_parser_router error: {e}, falling back to 'interest_filler'")
            return "interest_filler"
    
    # ✅ MODIFIED: Always route to interest_filler (both normal flow and candidate_job_match flow)
    # job_matcher will run after resume_assembler saves data to ChromaDB
    builder.add_conditional_edges(
        "groq_resume_parser",
        groq_resume_parser_router,
        {
            "interest_filler": "interest_filler"  # Both normal flow and candidate_job_match flow
        }
    )
    
    # ✅ Add edges from interest_filler to trigger parallel execution
    builder.add_edge("interest_filler", "resume_summary")  # Parallel execution
    builder.add_edge("interest_filler", "skill_proficiency_analyzer")  # Parallel execution
    
    # DEPRECATED: Old parallel parsing flow removed
    # After preprocessor, start parallel parsing
    # builder.add_edge("resume_preprocessor", "start_parsing")
    # Start parallel parsing from the dummy node
    # builder.add_edge("start_parsing", "personal_info_parser")
    # builder.add_edge("start_parsing", "experience_parser")
    # builder.add_edge("start_parsing", "education_parser")
    # builder.add_edge("start_parsing", "skills_parser")


    # builder.add_conditional_edges(
    #     "validate_jd", is_valid_jd,
    #     {True: "job_description_parser", False: "end_invalid_jd"}
    # )

    builder.add_conditional_edges(
       "validate_jd", 
       jd_router,
       {
           "job_description_parser": "job_description_parser",
           "end_invalid_jd": "end_invalid_jd"
       }
    )

    def notification_router(state: AgentState) -> str:
        """Router function for notification_agent that handles None values safely."""
        try:
            next_node = state.get("next", "end")
            if next_node is None or next_node == "notification_agent":
                log.warning(f"Notification agent returned invalid next node '{next_node}', routing to end")
                return "end"
            if next_node not in ["end"]:
                log.warning(f"Notification agent returned unexpected next node '{next_node}', routing to end")
                return "end"
            return next_node
        except Exception as e:
            log.error(f"❌ notification_router error: {e}, falling back to 'end'")
            return "end"
    
    builder.add_conditional_edges(
        "notification_agent",
        notification_router,
        {
            "end": "end"
        }
    )


    # DEPRECATED: Old synchronization edges removed
    # Synchronization and Initial Analysis
    # builder.add_edge("personal_info_parser", "resume_assembler")
    # builder.add_edge("experience_parser", "resume_assembler")
    # builder.add_edge("education_parser", "resume_assembler")
    # builder.add_edge("skills_parser", "resume_assembler")
    
    # New: Groq parser → Skill proficiency analyzer → resume_scorer → Resume assembler (normal/compare) OR assessment_recommender (re-run only)
    # resume_scorer runs before resume_assembler in both normal and compare flow
    def skill_proficiency_analyzer_router(state: AgentState) -> str:
        """Route after skill_proficiency_analyzer: re-run (submission in payload) → assessment_recommender; else → resume_scorer (before assembler)."""
        try:
            body = state.get("body") or {}
            has_submission = bool(body.get("submission"))
            if has_submission:
                log.info("Re-run subgraph: skill_proficiency_analyzer → assessment_recommender (submission in payload)")
                return "assessment_recommender"
            log.info("Normal/compare flow: skill_proficiency_analyzer → resume_scorer (before resume_assembler)")
            return "resume_scorer"
        except Exception as e:
            log.error(f"❌ skill_proficiency_analyzer_router error: {e}, falling back to 'resume_scorer'")
            return "resume_scorer"

    builder.add_conditional_edges(
        "skill_proficiency_analyzer",
        skill_proficiency_analyzer_router,
        {
            "resume_scorer": "resume_scorer",
            "assessment_recommender": "assessment_recommender",
        }
    )

    # Router after resume_assembler to handle candidate_job_match vs normal flow
    # Note: resume_scorer runs BEFORE resume_assembler (skill_proficiency_analyzer -> resume_scorer -> resume_assembler)
    def resume_assembler_router(state: AgentState) -> str:
        """Routes after resume_assembler based on request type and endpoint."""
        try:
            # ⚠️ NOTE: /analyze-resume-callback endpoint REJECTS request_type="candidate_job_match" and will NEVER route to job_matcher compare flow
            body = state.get("body") or {}
            request_type = state.get("request_type") or body.get("request_type")
            endpoint_name = state.get("endpoint_name") or body.get("endpoint_name")
            job_id = state.get("job_id") or body.get("job_id")
            auto_second_call = state.get("auto_second_call", False)
            
            # ✅ CRITICAL FIX: Only route to job_matcher compare flow if called from /compare-candidate-job endpoint
            # /analyze-resume-callback endpoint (endpoint_name="analyze_resume_callback") should NEVER route to this compare flow
            is_compare_candidate_job_endpoint = endpoint_name == "compare_candidate_job"
            
            log.info(f"🔍 RESUME_ASSEMBLER_ROUTER: request_type={request_type}, endpoint_name={endpoint_name}, job_id={job_id}")
            
            if request_type == "candidate_job_match" and is_compare_candidate_job_endpoint:
                # ✅ /compare-candidate-job endpoint: route to preprocessor → job_matcher
                log.info(f"✅ RESUME_ASSEMBLER_ROUTER: candidate_job_match flow - routing to job_matcher_preprocessor")
                return "job_matcher_preprocessor"

            # resume_scorer already ran before resume_assembler; route to career flow or end
            original_has_resume = bool(body.get("resume_url") or (body.get("resume_text") or "").strip())
            original_has_user_interests = bool(body.get("user_interests"))
            if auto_second_call or (original_has_resume and not original_has_user_interests):
                if auto_second_call:
                    log.info("✅ RESUME_ASSEMBLER_ROUTER: auto_second_call - routing to career_advisor_entry")
                else:
                    log.info("✅ RESUME_ASSEMBLER_ROUTER: Flow 1 (resume, no user_interests) - routing to career_advisor_entry")
                return "career_advisor_entry"
            log.info("✅ RESUME_ASSEMBLER_ROUTER: Normal flow - routing to end")
            return "end"
        
        except Exception as e:
            log.error(f"❌ resume_assembler_router error: {e}, falling back to 'end'")
            return "end"
    
    # ✅ After resume assembly, route based on request type
    log.info("Routing: resume_assembler -> resume_assembler_router")
    builder.add_conditional_edges(
        "resume_assembler",
        resume_assembler_router,
        {
            "job_matcher_preprocessor": "job_matcher_preprocessor",  # ✅ /compare-candidate-job → preprocessor → job_matcher
            "career_advisor_entry": "career_advisor_entry",  # ✅ Normal flow (Flow 1)
            "end": "end"  # ✅ Normal flow (1st call)
        }
    )
    

    # ✅ REMOVED: resume_summary -> resume_scorer edge (resume_summary is only used in parallel execution now)

    # Router after resume_scorer (runs before resume_assembler): persist via assembler, then assembler routes career/end/compare
    # - Re-run (submission) -> skill_proficiency_analyzer
    # - Else -> resume_assembler (Chroma save; then resume_assembler_router -> career_advisor_entry | end | job_matcher_preprocessor)
    def resume_scorer_router(state: AgentState) -> str:
        """Routes after resume_scorer: submission → skill_proficiency_analyzer; else → resume_assembler."""
        try:
            body = state.get("body") or {}
            has_submission = bool(body.get("submission"))

            if has_submission:
                # Re-run: payload has user's assessment answers → skill_proficiency_analyzer → assessment_recommender
                log.info("Re-run flow (assessment submission in payload): routing to skill_proficiency_analyzer")
                return "skill_proficiency_analyzer"
            log.info("Resume flow: resume_scorer → resume_assembler")
            return "resume_assembler"
        except Exception as e:
            log.error(f"❌ resume_scorer_router error: {e}, falling back to 'resume_assembler'")
            return "resume_assembler"

    log.info("Routing: resume_scorer -> resume_scorer_router")
    builder.add_conditional_edges(
        "resume_scorer",
        resume_scorer_router,
        {
            "skill_proficiency_analyzer": "skill_proficiency_analyzer",
            "resume_assembler": "resume_assembler",
        }
    )
    
    # ✅ REMOVED: legacy candidate_job_matcher node - job_matcher now continues through normal flow
    # job_matcher compare flow continues to parallel nodes (interest_filler, resume_summary, skill_proficiency_analyzer)
    # which then continue to resume_assembler -> resume_scorer -> end
    # job_matcher sends its callback before continuing
    
    log.info("Routing: career_advisor_entry -> enhanced_role_fit -> career_flow_fan_out (career) | career_advisor (compare+resume)")
    builder.add_edge("career_advisor_entry", "enhanced_role_fit")

    def enhanced_role_fit_router(state: AgentState) -> str:
        """Compare+resume: job_matcher already ran (merged); go straight to career_advisor (no parallel branch).
        Career / other compare: fan out to career_advisor + job_matcher_preprocessor (parallel)."""
        is_compare = (state.get("endpoint_name") or (state.get("body") or {}).get("endpoint_name")) == "compare_candidate_job"
        body = state.get("body") or {}
        has_resume = bool(body.get("resume_url") or (body.get("resume_text") or "").strip())
        if is_compare and has_resume:
            log.info("enhanced_role_fit_router: Compare+resume — merged job_matcher done; routing directly to career_advisor (no parallel)")
            return "career_advisor"
        if is_compare:
            log.info("enhanced_role_fit_router: Compare flow (no resume) - routing to career_flow_fan_out")
        else:
            log.info("enhanced_role_fit_router: Career flow - routing to career_flow_fan_out (parallel)")
        return "career_flow_fan_out"

    builder.add_conditional_edges("enhanced_role_fit", enhanced_role_fit_router, {
        "career_flow_fan_out": "career_flow_fan_out",
        "career_advisor": "career_advisor",
    })
    log.info("Routing: career_advisor -> market_and_course_recommender")
    builder.add_edge("career_advisor", "market_and_course_recommender")

    # Router after market_and_course_recommender: Career/compare flow -> assessment_recommender; rerun -> resume_scorer
    def market_and_course_recommender_router(state: AgentState) -> str:
        """Career flow / compare flow -> assessment_recommender; rerun -> resume_scorer.

        Flow 1 (resume, no user_interests) sets _job_matcher_parallel via career_advisor_entry /
        career_flow_fan_out so job_matcher runs in parallel with the career chain. We must route to
        assessment_recommender (not resume_scorer) or the graph loops resume_scorer -> assembler ->
        career_advisor_entry forever and assessment_validator never sees assessment_plan/assessment_needs.
        """
        try:
            is_second_call = state.get("is_second_call", False)
            auto_second_call = state.get("auto_second_call", False)
            parallel_career_job = bool(state.get("_job_matcher_parallel"))
            body = state.get("body") or {}
            has_user_interests = bool(body.get("user_interests"))
            has_resume = bool(body.get("resume_url") or (body.get("resume_text") or "").strip())
            is_compare = (state.get("endpoint_name") or body.get("endpoint_name")) == "compare_candidate_job"
            if (
                parallel_career_job
                or is_compare
                or is_second_call
                or auto_second_call
                or (has_user_interests and not has_resume)
            ):
                if parallel_career_job and not (is_compare or is_second_call or auto_second_call or (has_user_interests and not has_resume)):
                    log.info(
                        "Parallel career+job_matcher flow: market_and_course_recommender -> assessment_recommender"
                    )
                else:
                    log.info("Career/compare flow: market_and_course_recommender -> assessment_recommender")
                return "assessment_recommender"
            log.info("Rerun flow: market_and_course_recommender -> resume_scorer")
            return "resume_scorer"
        except Exception as e:
            log.error(f"❌ market_and_course_recommender_router error: {e}, falling back to 'assessment_recommender'")
            return "assessment_recommender"

    builder.add_conditional_edges(
        "market_and_course_recommender",
        market_and_course_recommender_router,
        {
            "assessment_recommender": "assessment_recommender",
            "resume_scorer": "resume_scorer",
        }
    )
    # skill_proficiency_analyzer → assessment_recommender only via skill_proficiency_analyzer_router when payload has submission (re-run subgraph)

    # Conditional routing after assessment_recommender:
    # - Career flow (Flow 1 / 2nd call): job_matcher is already running in parallel
    #   from enhanced_role_fit fan-out, so route to assessment_validator (guard handles join).
    # - Re-run / evaluation flows: job_matcher hasn't started, so route to it sequentially.
    def assessment_recommender_router(state: AgentState) -> str:
        """Routes after assessment_recommender: parallel flow -> assessment_validator; sequential -> job_matcher_preprocessor."""
        try:
            is_compare = (state.get("endpoint_name") or (state.get("body") or {}).get("endpoint_name")) == "compare_candidate_job"
            has_candidate_match = bool(state.get("candidate_job_match_result"))
            if is_compare and has_candidate_match:
                log.info("assessment_recommender: Compare flow (job already matched), routing to assessment_validator")
                return "assessment_validator"
            if state.get("_job_matcher_parallel"):
                log.info("assessment_recommender: parallel flow, routing to assessment_validator")
                return "assessment_validator"
            log.info("assessment_recommender: sequential flow, routing to job_matcher_preprocessor")
            return "job_matcher_preprocessor"
        except Exception as e:
            log.error(f"❌ assessment_recommender_router error: {e}, falling back to 'job_matcher_preprocessor'")
            return "job_matcher_preprocessor"

    builder.add_conditional_edges(
        "assessment_recommender",
        assessment_recommender_router,
        {
            "assessment_validator": "assessment_validator",
            "job_matcher_preprocessor": "job_matcher_preprocessor",
        }
    )

    # Router after job_matcher: compare flow → career_advisor_entry or assessment_validator; career flow → assessment_validator
    def job_matcher_router(state: AgentState) -> str:
        """Routes after job_matcher.

        Compare + resume: ONE merged job_matcher (payload job_id + Chroma jobs) runs before career_advisor_entry.
        After enhanced_role_fit, career chain runs sequentially (no parallel job_matcher).
        Legacy compare without resume: _compare_flow_multi_job_run → assessment_validator.
        Career flow (parallel): parallel_job_matcher_done until AR merges plan/needs; then assessment_validator.
        Career flow (sequential after AR): assessment_validator.
        """
        try:
            body = state.get("body") or {}
            is_compare_endpoint = (state.get("endpoint_name") or body.get("endpoint_name")) == "compare_candidate_job"
            has_resume_in_payload = bool(body.get("resume_url") or (body.get("resume_text") or "").strip())

            if is_compare_endpoint:
                if state.get("compare_job_only") or not has_resume_in_payload:
                    log.info("job_matcher: Compare flow (no resume), routing to end")
                    return "end"
                # Legacy second matcher run (compare without resume in body)
                if state.get("_compare_flow_multi_job_run"):
                    log.info(
                        "job_matcher: Compare flow (post fan-out / 2nd job_matcher), routing to assessment_validator"
                    )
                    return "assessment_validator"
                log.info("job_matcher: Compare flow (with resume), routing to career_advisor_entry (unified flow)")
                return "career_advisor_entry"

            # Career flow: sequential (after AR → preprocessor → job_matcher) → assessment_validator.
            # Parallel fan-out: job_matcher often finishes before AR — do NOT route to assessment_validator
            # until assessment_recommender has merged assessment_plan/assessment_needs (sink otherwise).
            original_has_resume = bool(body.get("resume_url") or (body.get("resume_text") or "").strip())
            original_has_user_interests = bool(body.get("user_interests"))
            is_second_call = state.get("is_second_call", False)
            is_career_flow = is_second_call or (original_has_resume and not original_has_user_interests)
            if is_career_flow:
                if state.get("_job_matcher_parallel") and not _career_recommender_outputs_present(state):
                    log.info(
                        "job_matcher: parallel career flow — assessment_recommender not finished; "
                        "sink (parallel_job_matcher_done). Validator runs only after AR."
                    )
                    return "parallel_job_matcher_done"
                log.info("job_matcher: Career flow, routing to assessment_validator")
                return "assessment_validator"

            log.info("job_matcher: Direct call, routing to end")
            return "end"
        except Exception as e:
            log.error(f"❌ job_matcher_router error: {e}, falling back to 'end'")
            return "end"

    log.info("Routing: job_matcher -> job_matcher_router")
    builder.add_conditional_edges(
        "job_matcher",
        job_matcher_router,
        {
            "assessment_validator": "assessment_validator",
            "career_advisor_entry": "career_advisor_entry",
            "end": "end",
            "parallel_job_matcher_done": "parallel_job_matcher_done",
        }
    )

    # assessment_validator -> end OR parallel_join_wait (skip must not hit end — other branch still runs)
    def after_assessment_validator_router(state: AgentState) -> str:
        if state.get("_skip_callback"):
            return "parallel_join_wait"
        return "end"

    log.info("Routing: assessment_validator -> end | parallel_join_wait (when fan-in skip)")
    builder.add_conditional_edges(
        "assessment_validator",
        after_assessment_validator_router,
        {
            "parallel_join_wait": "parallel_join_wait",
            "end": "end",
        },
    )
    
    # Separate flow for question generation
    builder.add_edge("assessment_question_generator", "end")
    
    # Manual assessment builder flow: standalone -> end
    builder.add_edge("assessment_builder_agent", "end")

    # Assessment Evaluation Flow - Re-run career flow after report generation
    log.info("Routing: assessment_evaluator -> report_generator")
    builder.add_edge("assessment_evaluator", "report_generator")
    
    # Add data preparation step for career advisor re-run
    async def evaluation_career_entry(state: AgentState) -> Dict:
        """Prepare data for career advisor re-run after evaluation.

        With centralized hydration, state is already loaded from session.
        We only need to ensure assessment_results and report are present.
        """
        from core.utils import run_blocking_io as _rbi_eval
        log.info(f"🔍 EVALUATION_CAREER_ENTRY: Starting data preparation")
        
        # State is already hydrated from session in production_invoke/dispatcher
        # Just validate required fields are present
        assessment_results = state.get("assessment_results")
        report = state.get("report")
        structured_resume = state.get("structured_resume")
        user_interests = state.get("user_interests")
        uid = state.get("uid")
        prior_gap = state.get("raw_skill_gap_analysis_output", {})
        
        log.info(f"🔍 EVALUATION_CAREER_ENTRY: assessment_results={bool(assessment_results)}, report={bool(report)}")
        log.info(f"🔍 EVALUATION_CAREER_ENTRY: structured_resume={bool(structured_resume)}, user_interests={len(user_interests) if user_interests else 0}")

        # If still missing, try one more hydration (defensive)
        if not structured_resume or not user_interests:
            session_id = state.get("session_id")
            if session_id and uid:
                log.warning(f"⚠️ EVALUATION_CAREER_ENTRY: Missing data, attempting hydration")
                temp_state = dict(state)
                temp_state = await _hydrate_state_from_session_async(dict(temp_state), force_hydration=True)
                assessment_results = temp_state.get("assessment_results") or assessment_results
                report = temp_state.get("report") or report
                structured_resume = temp_state.get("structured_resume") or structured_resume
                user_interests = temp_state.get("user_interests") or user_interests
                prior_gap = temp_state.get("raw_skill_gap_analysis_output", {}) or prior_gap
        
        # Ensure resume_summary is loaded from cache for reruns
        if structured_resume and uid and not structured_resume.get("resume_summary"):
            try:
                from chroma import get_resume_doc
                rdoc = await _rbi_eval(get_resume_doc, uid) or {}
                cached_summary_data = rdoc.get("resume_summary_cache", {})
                resume_summary = cached_summary_data.get("summary")
                
                if resume_summary:
                    if not isinstance(structured_resume, dict):
                        structured_resume = dict(structured_resume) if structured_resume else {}
                    structured_resume["resume_summary"] = resume_summary
                    log.info(f"✅ EVALUATION_CAREER_ENTRY: Loaded resume_summary from cache for UID {uid} ({len(resume_summary)} chars)")
                else:
                    log.info(f"ℹ️ EVALUATION_CAREER_ENTRY: No cached resume_summary found for UID {uid}, agents will generate on-demand")
            except Exception as e:
                log.warning(f"⚠️ EVALUATION_CAREER_ENTRY: Failed to load resume_summary from cache: {e}")
        
        # Return state updates (LangGraph will merge automatically)
        # CRITICAL: Never include 'next' in return - it's for routing only
        return {
            "evaluation_career_entry_status": "completed",
            # Data is already in state from hydration, just confirm it's there
            "assessment_results": assessment_results,
            "report": report,
            "structured_resume": structured_resume,
            "user_interests": user_interests,
            "raw_skill_gap_analysis_output": prior_gap
            # Explicitly NOT including 'next' here - routing is handled by edges
        }
    
    builder.add_node("evaluation_career_entry", evaluation_career_entry)
    builder.add_edge("report_generator", "evaluation_career_entry")
    # Evaluation / re-run flow now reuses the main re-run chain:
    # evaluation_career_entry -> career_advisor -> market_and_course_recommender
    # -> resume_scorer -> skill_proficiency_analyzer -> assessment_recommender
    # -> job_matcher -> assessment_validator -> end
    builder.add_edge("evaluation_career_entry", "career_advisor")
    builder.add_edge("career_advisor", "market_and_course_recommender")
    # market_and_course_recommender uses conditional_edges (routes to assessment_recommender or resume_scorer)

    # Job Description Workflow: validate_jd -> job_description_parser -> prescreening_questions -> ranker -> end
    builder.add_edge("job_description_parser", "prescreening_questions")
    builder.add_edge("prescreening_questions", "ranker")
    builder.add_edge("ranker", "end")
    
    # Resume Content Generator Workflow (independent - goes directly to end)
    builder.add_edge("resume_content_generator", "end")

    # JD enhance-only workflow (user gets enhanced text, can send again for validation)
    builder.add_edge("enhance_jd", "end_enhanced_jd")

    builder.set_finish_point("end")
    builder.set_finish_point("end_invalid_resume")
    builder.set_finish_point("end_invalid_jd")
    builder.set_finish_point("end_enhanced_jd")
    log.debug("🔄 Compiling graph...")
    compiled_graph = builder.compile(checkpointer=None, debug=False)
    log.info("✅ Graph compiled successfully")
    return compiled_graph

# Production wrapper with memory management
class AnalysisMemory:
    """Bounded memory management for learning and adaptation."""
    def __init__(self, max_entries: int = MAX_TENANT_MEMORY_ENTRIES):
        self.memories = {}  # tenant_id:user_id -> memory data
        self.max_entries = max_entries
        self._lock = threading.Lock()
    
    def get_memory(self, tenant_id: str, user_id: str) -> Dict[str, Any]:
        """Get user's analysis memory (returns a deep copy to prevent mutation of shared state)."""
        key = f"{tenant_id}:{user_id}"
        with self._lock:
            default = {
                "analysis_history": [],
                "feedback_scores": [],
                "successful_patterns": [],
                "adaptation_metadata": {"ema_success": 0.5}
            }
            return copy.deepcopy(self.memories.get(key, default))
    
    def update_memory(self, tenant_id: str, user_id: str, analysis_data: Dict[str, Any]):
        """Update user's analysis memory with bounded size."""
        key = f"{tenant_id}:{user_id}"
        with self._lock:
            if key not in self.memories:
                self.memories[key] = {
                    "analysis_history": [],
                    "feedback_scores": [],
                    "successful_patterns": [],
                    "adaptation_metadata": {"ema_success": 0.5}
                }
            
            memory = self.memories[key]
            
            # Add to analysis history (keep last 50)
            memory["analysis_history"].append({
                "timestamp": time.time(),
                "analysis_id": analysis_data.get("analysis_id"),
                "method": analysis_data.get("analysis_method"),
                "confidence": analysis_data.get("confidence_score", 0.0)
            })
            memory["analysis_history"] = memory["analysis_history"][-50:]
            
            # Update EMA for success rate
            alpha = 0.15
            success = 1.0 if analysis_data.get("ok", False) else 0.0
            current_ema = memory["adaptation_metadata"]["ema_success"]
            memory["adaptation_metadata"]["ema_success"] = alpha * success + (1 - alpha) * current_ema
            
            # Enforce memory bounds
            if len(self.memories) > self.max_entries:
                # Remove oldest entry
                oldest_key = min(self.memories.keys(), 
                               key=lambda k: self.memories[k]["analysis_history"][0]["timestamp"] 
                               if self.memories[k]["analysis_history"] else 0)
                del self.memories[oldest_key]

# Global memory instance
analysis_memory = AnalysisMemory()


def _pipeline_flow_has_user_interests(body: Dict[str, Any], state: AgentState) -> bool:
    for src in (body.get("user_interests"), state.get("user_interests")):
        if not src or not isinstance(src, list) or len(src) == 0:
            continue
        if isinstance(src[0], dict) and "answer" in src[0]:
            if any((item.get("answer") or "").strip() for item in src if isinstance(item, dict)):
                return True
        else:
            return True
    return False


def _pipeline_flow_is_second_call(body: Dict[str, Any], state: AgentState, resume_url: str) -> bool:
    if state.get("is_second_call"):
        return True
    resume_text = (body.get("resume_text") or state.get("resume_text") or "").strip()
    uid = body.get("uid") or state.get("uid") or ""
    if resume_url or resume_text or not uid:
        return False
    if not _pipeline_flow_has_user_interests(body, state):
        return False
    if body.get("plan") or body.get("submission"):
        return False
    if body.get("job_id") or body.get("jd_url") or body.get("job_details"):
        return False
    if body.get("company") or body.get("job_status"):
        return False
    return True


def classify_pipeline_flow(state: AgentState) -> Tuple[str, str]:
    """
    Label the high-level product flow for timing logs (flows 1–6 from dispatcher semantics).

    Returns:
        (flow_id, flow_label) e.g. ("flow_1", "resume_no_interests").
    """
    body = state.get("body") if isinstance(state.get("body"), dict) else {}
    request_type = (body.get("request_type") or state.get("request_type") or "") or ""
    endpoint_name = (body.get("endpoint_name") or state.get("endpoint_name") or "") or ""
    resume_url = (body.get("resume_url") or state.get("resume_url") or "") or ""
    resume_text = (body.get("resume_text") or state.get("resume_text") or "").strip()
    has_resume = bool(
        resume_url
        or resume_text
        or body.get("structured_resume")
        or state.get("structured_resume")
    )

    if body.get("submission"):
        return ("flow_6", "assessment_evaluation")

    if body.get("plan"):
        return ("flow_assessment_gen", "assessment_plan_generation")

    if request_type == "candidate_job_match" or endpoint_name == "compare_candidate_job":
        return ("flow_5", "candidate_job_compare")

    if body.get("jd_url") or body.get("jd_text") or state.get("jd_url") or state.get("jd_text"):
        return ("flow_4", "job_description")

    if body.get("job_details"):
        if has_resume:
            return ("flow_other", "job_ranking_or_resume_with_job_context")
        return ("flow_4", "job_description")

    if _pipeline_flow_is_second_call(body, state, resume_url):
        return ("flow_3", "resume_second_call_interests_only")

    uid = body.get("uid") or state.get("uid") or ""
    if has_resume:
        if _pipeline_flow_has_user_interests(body, state):
            return ("flow_2", "resume_with_user_interests")
        return ("flow_1", "resume_no_interests")

    if (
        uid
        and not resume_url
        and not resume_text
        and not body.get("job_id")
        and not body.get("jd_url")
        and not body.get("job_details")
    ):
        return ("flow_resume_content", "resume_content_generation")

    return ("flow_unknown", "unknown_or_misc")


def log_pipeline_flow_complete(
    flow_id: str,
    flow_label: str,
    duration_sec: float,
    *,
    uid: str,
    tenant_id: str,
    analysis_id: str,
    ok: bool,
    extra: str = "",
) -> None:
    """Structured INFO log: wall-clock time from production wrapper entry to graph completion."""
    suffix = f" {extra}" if extra else ""
    log.info(
        "PIPELINE_FLOW_COMPLETE flow_id=%s flow_label=%s duration_sec=%.3f uid=%s tenant_id=%s "
        "analysis_id=%s ok=%s%s",
        flow_id,
        flow_label,
        duration_sec,
        uid or "-",
        tenant_id or "-",
        analysis_id or "-",
        ok,
        suffix,
    )


def create_production_graph():
    """Create the production-ready graph with all enhancements."""
    log.info("🔄 Creating base graph...")
    graph = create_graph()
    log.info("✅ Base graph created, adding production wrappers...")
    
    # Add memory management wrapper for async execution
    original_ainvoke = getattr(graph, 'ainvoke', None)
    
    async def production_ainvoke(state: AgentState, config: Dict[str, Any] = None) -> AgentState:
        """Async production wrapper with memory management and monitoring."""
        from core.utils import run_blocking_io as _rbi_prod
        start_time = time.time()
        tenant_id = state.get("tenant_id", "default-tenant")
        uid = state.get("uid", "")
        session_id = state.get("session_id", "")
        
        # Create task metrics
        task_metrics = create_task_metrics("graph_execution", state.get("analysis_id", ""))
        
        # Get user memory for personalization (in-memory dict, no I/O)
        user_memory = analysis_memory.get_memory(tenant_id, uid)
        
        # Hydrate on the event loop: parallel Chroma reads via _hydrate_state_from_session_async
        state = await _hydrate_state_from_session_async(dict(state), force_hydration=False)
        
        # Update session heartbeat if session exists (ChromaDB I/O -- offload)
        if session_id:
            try:
                await _rbi_prod(session_manager.update_heartbeat, session_id)
            except Exception as e:
                log.warning(f"Failed to update session heartbeat: {e}")
        
        # Add memory context to state
        enhanced_state = {
            **state,
            "user_memory": user_memory,
            "processing_start_time": start_time
        }
        flow_id, flow_label = classify_pipeline_flow(enhanced_state)
        analysis_id_for_log = enhanced_state.get("analysis_id", "") or ""
        
        # Add max_concurrency and recursion_limit to config if not present
        if config is None:
            config = {}
        if "max_concurrency" not in config:
            config["max_concurrency"] = get_max_concurrency()
        if "recursion_limit" not in config:
            config["recursion_limit"] = GRAPH_RECURSION_LIMIT

        # Merge Langfuse tracing callbacks + session/user metadata for Langfuse Sessions
        from settings import settings as _settings
        config = merge_langfuse_into_config(
            config, _settings,
            session_id=enhanced_state.get("session_id") or "",
            user_id=enhanced_state.get("uid") or "",
        )
        
        try:
            # Check for short-circuit opportunity
            if can_short_circuit(state, confidence_threshold=0.8):
                log.info(f"Short-circuiting graph execution due to high confidence for uid={uid}")
                task_metrics.success = True
                task_metrics.cache_hit = True
                log_task_metrics(task_metrics)
                _sc_dur = time.time() - start_time
                log_pipeline_flow_complete(
                    flow_id,
                    flow_label,
                    _sc_dur,
                    uid=uid,
                    tenant_id=tenant_id,
                    analysis_id=analysis_id_for_log,
                    ok=True,
                    extra="reason=short_circuit",
                )
                return enhanced_state
            
            # Execute the graph asynchronously
            if original_ainvoke:
                result = await original_ainvoke(enhanced_state, config)
            else:
                # Fallback to sync execution via run_blocking_io (avoids blocking event loop)
                result = await _rbi_prod(graph.invoke, enhanced_state, config)
            
            # Update memory with results
            analysis_memory.update_memory(tenant_id, uid, result)
            
            # Record final metrics
            processing_time = time.time() - start_time
            result["processing_time_seconds"] = processing_time
            metrics.record_response_time(processing_time)
            
            # Update task metrics
            task_metrics.success = result.get("ok", True)
            task_metrics.tokens_in = result.get("tokens_in", 0)
            task_metrics.tokens_out = result.get("tokens_out", 0)
            log_task_metrics(task_metrics)
            
            log_pipeline_flow_complete(
                flow_id,
                flow_label,
                processing_time,
                uid=uid,
                tenant_id=tenant_id,
                analysis_id=result.get("analysis_id", analysis_id_for_log),
                ok=bool(result.get("ok", True)),
            )
            return result
        
        except GraphRecursionError as e:
            log.error(f"Graph exceeded recursion limit ({config.get('recursion_limit', 'unknown')} steps) for uid={uid}: {e}")
            metrics.record_method("recursion_limit_exceeded")
            
            # Update task metrics
            task_metrics.success = False
            task_metrics.error_code = "GraphRecursionError"
            log_task_metrics(task_metrics)
            
            _err_dur = time.time() - start_time
            log_pipeline_flow_complete(
                flow_id,
                flow_label,
                _err_dur,
                uid=uid,
                tenant_id=tenant_id,
                analysis_id=analysis_id_for_log,
                ok=False,
                extra="reason=recursion_limit",
            )
            # Return structured error response
            return {
                **state,
                "ok": False,
                "analysis_method": "recursion_limit_exceeded",
                "error": "Graph execution exceeded maximum iterations. This may indicate a routing loop.",
                "processing_time_seconds": _err_dur,
                "analysis_id": state.get("analysis_id", generate_analysis_id())
            }
            
        except Exception as e:
            log.error(f"Async graph execution failed: {e}")
            metrics.record_method("async_error")
            
            # Update task metrics
            task_metrics.success = False
            task_metrics.error_code = type(e).__name__
            log_task_metrics(task_metrics)
            
            _err_dur = time.time() - start_time
            log_pipeline_flow_complete(
                flow_id,
                flow_label,
                _err_dur,
                uid=uid,
                tenant_id=tenant_id,
                analysis_id=analysis_id_for_log,
                ok=False,
                extra=f"reason=async_error exc={type(e).__name__}",
            )
            # Return error response
            return {
                **state,
                "ok": False,
                "analysis_method": "async_error",
                "error": str(e),
                "processing_time_seconds": _err_dur,
                "analysis_id": state.get("analysis_id", generate_analysis_id())
            }
    
    # Add async memory management wrapper
    original_invoke = graph.invoke
    
    def production_invoke(state: AgentState, config: Dict[str, Any] = None) -> AgentState:
        """Sync production wrapper with memory management and monitoring."""
        start_time = time.time()
        tenant_id = state.get("tenant_id", "default-tenant")
        uid = state.get("uid", "")
        session_id = state.get("session_id", "")
        
        # Get user memory for personalization
        user_memory = analysis_memory.get_memory(tenant_id, uid)
        
        # Hydrate via async path (parallel I/O); safe when called with or without a running event loop
        state = _sync_hydrate_state_via_async(state, force_hydration=False)
        
        # Update session heartbeat if session exists
        if session_id:
            try:
                session_manager.update_heartbeat(session_id)
            except Exception as e:
                log.warning(f"Failed to update session heartbeat: {e}")
        
        # Add memory context to state
        enhanced_state = {
            **state,
            "user_memory": user_memory,
            "processing_start_time": start_time
        }
        flow_id_pi, flow_label_pi = classify_pipeline_flow(enhanced_state)
        analysis_id_pi = enhanced_state.get("analysis_id", "") or ""
        
        # Add max_concurrency and recursion_limit to config if not present
        if config is None:
            config = {}
        if "max_concurrency" not in config:
            config["max_concurrency"] = get_max_concurrency()
        if "recursion_limit" not in config:
            config["recursion_limit"] = GRAPH_RECURSION_LIMIT

        # Merge Langfuse tracing callbacks + session/user metadata for Langfuse Sessions
        from settings import settings as _settings
        config = merge_langfuse_into_config(
            config, _settings,
            session_id=enhanced_state.get("session_id") or "",
            user_id=enhanced_state.get("uid") or "",
        )
        
        try:
            # Execute the graph
            result = original_invoke(enhanced_state, config)
            
            # Update memory with results
            analysis_memory.update_memory(tenant_id, uid, result)
            
            # Record final metrics
            processing_time = time.time() - start_time
            result["processing_time_seconds"] = processing_time
            metrics.record_response_time(processing_time)
            
            log_pipeline_flow_complete(
                flow_id_pi,
                flow_label_pi,
                processing_time,
                uid=uid,
                tenant_id=tenant_id,
                analysis_id=result.get("analysis_id", analysis_id_pi),
                ok=bool(result.get("ok", True)),
            )
            return result
        
        except GraphRecursionError as e:
            log.error(f"Graph exceeded recursion limit ({config.get('recursion_limit', 'unknown')} steps) for uid={uid}: {e}")
            metrics.record_method("recursion_limit_exceeded")
            
            _edur = time.time() - start_time
            log_pipeline_flow_complete(
                flow_id_pi,
                flow_label_pi,
                _edur,
                uid=uid,
                tenant_id=tenant_id,
                analysis_id=analysis_id_pi,
                ok=False,
                extra="reason=recursion_limit",
            )
            # Return structured error response
            return {
                **state,
                "ok": False,
                "analysis_method": "recursion_limit_exceeded",
                "error": "Graph execution exceeded maximum iterations. This may indicate a routing loop.",
                "processing_time_seconds": _edur,
                "analysis_id": state.get("analysis_id", generate_analysis_id())
            }
            
        except Exception as e:
            log.error(f"Graph execution failed: {e}")
            metrics.record_method("error")
            
            _edur = time.time() - start_time
            log_pipeline_flow_complete(
                flow_id_pi,
                flow_label_pi,
                _edur,
                uid=uid,
                tenant_id=tenant_id,
                analysis_id=analysis_id_pi,
                ok=False,
                extra=f"reason=sync_invoke_error exc={type(e).__name__}",
            )
            # Return error response
            return {
                **state,
                "ok": False,
                "analysis_method": "error",
                "error": str(e),
                "processing_time_seconds": _edur,
                "analysis_id": state.get("analysis_id", generate_analysis_id())
            }
    
    # Replace both invoke methods
    graph.invoke = production_invoke
    graph.ainvoke = production_ainvoke
    return graph

def get_production_metrics() -> Dict[str, Any]:
    """Get current production metrics for monitoring (Section 8 Issue 4: include LLM cost/token)."""
    stats = dict(metrics.get_stats())
    try:
        from core.utils import get_io_executor_stats
        stats["io_executor"] = get_io_executor_stats()
    except Exception:
        pass
    try:
        from core.observability import get_task_stats
        task_stats = get_task_stats()
        stats["llm_total_tokens_in"] = task_stats.get("total_tokens_in", 0)
        stats["llm_total_tokens_out"] = task_stats.get("total_tokens_out", 0)
        stats["llm_cache_hit_rate"] = task_stats.get("cache_hit_rate", 0.0)
        stats["llm_estimated_cost"] = task_stats.get("total_estimated_cost", 0.0)
    except Exception:
        pass
    return stats

def get_circuit_breaker_status(tenant_id: str) -> Dict[str, Any]:
    """Get circuit breaker status for a tenant."""
    breaker = get_circuit_breaker(tenant_id)
    return {
        "state": breaker.state.value,
        "failure_count": breaker.failure_count,
        "last_failure_time": breaker.last_failure_time,
        "is_open": breaker.is_open()
    }

def get_rate_limiter_status(tenant_id: str) -> Dict[str, Any]:
    """Get rate limiter status for a tenant."""
    limiter = get_rate_limiter(tenant_id)
    return {
        "tokens_available": limiter.tokens,
        "max_tokens": limiter.max_tokens,
        "refill_rate": limiter.refill_rate
    }

def get_session_status(uid: str) -> Dict[str, Any]:
    """Get session status for a user."""
    try:
        session = session_manager.get_session_by_owner(uid, "candidate_pipeline")
        if session:
            return {
                "session_id": session.session_id,
                "status": session.status,
                "step": session.state.step,
                "progress": session.state.progress,
                "expires_at": session.expires_at,
                "last_heartbeat": session.last_heartbeat_at
            }
        return {"error": "No active session found"}
    except Exception as e:
        return {"error": f"Failed to get session status: {str(e)}"}

def get_memory_summary(uid: str) -> Dict[str, Any]:
    """Get memory summary for a user."""
    try:
        return memory_manager.get_session_summary(uid)
    except Exception as e:
        return {"error": f"Failed to get memory summary: {str(e)}"}

def update_session_step(session_id: str, step: str, data: Dict[str, Any] = None, progress: float = None) -> bool:
    """Update session step and data."""
    try:
        return session_manager.update_step(session_id, step, data, progress)
    except Exception as e:
        log.error(f"Failed to update session step: {e}")
        return False

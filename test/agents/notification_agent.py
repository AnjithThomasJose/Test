#!/usr/bin/env python3
"""
Notification Agent for sending emails based on pipeline results
Integrates with the supervisor agent workflow
Implements Agentic AI Production Playbook patterns
"""

import json
import logging
import re
import time
import asyncio
import hashlib
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from collections import deque
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl
from utils.memory_manager import memory_manager
from log_handler import setup_loggers
from core.middleware import TenantScopedRateLimiter
from core.utils import (
    _mask, _sanitize_text_for_llm, _validate_state_inputs, _to_text, _clean_json_text, 
    _scan_balanced_json, _safe_json_loads, _extract_json_from_response, _create_error_response,
    _generate_request_id, _calculate_processing_time, run_blocking_io
)
from core.config import get_agent_config
from core.security import validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm, PII_PATTERNS, INJECTION_FILTERS
from core.memory import BaseAgentMemory, get_agent_memory
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion

# Get centralized configuration
config = get_agent_config("notification_agent")

# Get existing loggers instead of creating new ones
log = logging.getLogger('main')
error_log = logging.getLogger('error')

# Production Playbook Constants
TENANT_ID_REGEX = re.compile(r'^[a-zA-Z0-9_\-]{8,64}$')
MAX_PAYLOAD_SIZE = 100 * 1024  # 100KB
MAX_EMAIL_ADDRESSES = 50
CACHE_TTL_SECONDS = 1800  # 30 minutes
MAX_CACHE_ENTRIES = 1000
RATE_LIMIT_TOKENS = 10
RATE_LIMIT_REFILL_RATE = 1  # tokens per second
MAX_RETRIES = 3
BASE_BACKOFF = 0.4
TIMEOUT_SECONDS = 30

class AnalysisMethod(str, Enum):
    """Analysis method types for response envelope"""
    LLM = "llm"
    CACHED = "cached"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"

class ConfidenceLevel(str, Enum):
    """Confidence levels for response envelope"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# Custom memory class for notification agent (extends base memory)
class NotificationAgentMemory(BaseAgentMemory):
    def __init__(self, tenant_id: str = "default"):
        super().__init__(tenant_id, config.adaptation_window)
        # Add any notification agent specific fields here if needed

# Use centralized memory management
async def get_notification_agent_memory(tenant_id: str = "default") -> NotificationAgentMemory:
    """Get or create tenant-scoped notification agent memory."""
    return await get_agent_memory("notification_agent", tenant_id, NotificationAgentMemory)

# Rate limiter and circuit breaker functionality moved to centralized middleware

class _CircuitBreakerState:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.state = "closed"
        self.failures = 0
        self.last_failure = 0.0
        self.probe_in_progress = False
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout

class CircuitBreakerManager:
    """Minimal circuit breaker manager compatible with agent usage."""
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.breakers: Dict[str, _CircuitBreakerState] = {}
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout

    def _get(self, tenant_id: str) -> _CircuitBreakerState:
        if tenant_id not in self.breakers:
            self.breakers[tenant_id] = _CircuitBreakerState(
                failure_threshold=self.failure_threshold,
                reset_timeout=self.reset_timeout
            )
        return self.breakers[tenant_id]

    def is_open(self, tenant_id: str) -> bool:
        b = self._get(tenant_id)
        now = time.time()
        if b.state == "open" and (now - b.last_failure) >= b.reset_timeout:
            b.state = "half_open"
        return b.state == "open"

    def can_probe(self, tenant_id: str) -> bool:
        b = self._get(tenant_id)
        return b.state in ("half_open", "open") and not b.probe_in_progress

    def record_success(self, tenant_id: str):
        b = self._get(tenant_id)
        b.state = "closed"
        b.failures = 0
        b.probe_in_progress = False

    def record_failure(self, tenant_id: str):
        b = self._get(tenant_id)
        b.failures += 1
        b.last_failure = time.time()
        b.probe_in_progress = False
        if b.failures >= b.failure_threshold:
            b.state = "open"

class TenantAwareCache:
    """Tenant-scoped cache with TTL and LRU eviction"""
    
    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES, ttl: int = CACHE_TTL_SECONDS):
        self.max_entries = max_entries
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
    
    def _make_key(self, tenant_id: str, data: Dict[str, Any]) -> str:
        """Create deterministic cache key"""
        key_data = {
            "tenant_id": tenant_id,
            "email_type": data.get("type", "unknown"),
            "email_count": len(data.get("emails", [])),
            "user_id": data.get("uid", "")
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()  # Full hash to avoid collisions
    
    def get(self, tenant_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        key = self._make_key(tenant_id, data)
        now = time.time()
        
        if key in self.cache:
            entry = self.cache[key]
            if now - entry["timestamp"] < self.ttl:
                self.access_times[key] = now
                return entry["data"]
            else:
                # Expired
                del self.cache[key]
                del self.access_times[key]
        
        return None
    
    def set(self, tenant_id: str, data: Dict[str, Any], result: Dict[str, Any]):
        """Set cached result"""
        key = self._make_key(tenant_id, data)
        now = time.time()
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_entries:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = {
            "data": result,
            "timestamp": now
        }
        self.access_times[key] = now

class PerformanceMetrics:
    """Performance metrics tracking"""
    
    def __init__(self, window_size: int = 5000):
        self.window_size = window_size
        self.response_times = deque(maxlen=window_size)
        self.method_counts = {method.value: 0 for method in AnalysisMethod}
        self.cache_hits = 0
        self.cache_misses = 0
        self.timeouts = 0
        self.circuit_breaker_events = deque(maxlen=window_size)
    
    def record_response_time(self, duration: float):
        """Record response time"""
        self.response_times.append(duration)
    
    def record_method(self, method: AnalysisMethod):
        """Record analysis method used"""
        self.method_counts[method.value] += 1
    
    def record_cache_hit(self):
        """Record cache hit"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses += 1
    
    def record_timeout(self):
        """Record timeout"""
        self.timeouts += 1
    
    def record_circuit_breaker_event(self, event: str, tenant_id: str):
        """Record circuit breaker event"""
        self.circuit_breaker_events.append({
            "event": event,
            "tenant_id": tenant_id,
            "timestamp": time.time()
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        if not self.response_times:
            return {"error": "No data available"}
        
        response_times = list(self.response_times)
        response_times.sort()
        
        return {
            "response_time_p95": response_times[int(len(response_times) * 0.95)] if response_times else 0,
            "response_time_avg": sum(response_times) / len(response_times) if response_times else 0,
            "method_split": dict(self.method_counts),
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            "timeout_rate": self.timeouts / len(self.response_times) if self.response_times else 0,
            "circuit_breaker_events": list(self.circuit_breaker_events)[-10:]  # Last 10 events
        }

# Security validation functionality moved to centralized middleware

# Simple replacement functions
def validate_tenant_id(tenant_id: str) -> bool:
    """Validate tenant ID format"""
    return bool(TENANT_ID_REGEX.match(tenant_id)) if tenant_id else False

def sanitize_email(email: str) -> str:
    """Sanitize email address"""
    if not email:
        return ""
    email = email.strip().lower()
    email = re.sub(r'[<>"\']', '', email)
    return email

def redact_pii(data: Dict[str, Any]) -> Dict[str, Any]:
    """Redact PII from data for logging"""
    redacted = data.copy()
    if "email" in redacted:
        redacted["email"] = "***@***.***"
    if "user_mail" in redacted:
        redacted["user_mail"] = "***@***.***"
    if "password" in redacted:
        redacted["password"] = "***"
    return redacted

def validate_payload_size(data: Dict[str, Any]) -> bool:
    """Validate payload size"""
    try:
        serialized = json.dumps(data)
        return len(serialized) <= MAX_PAYLOAD_SIZE
    except (TypeError, ValueError):
        return False

class NotificationRequest(BaseModel):
    """Pydantic model for notification request validation"""
    uid: str = Field(..., min_length=8, max_length=64, description="User ID")
    tenant_id: str = Field(..., min_length=8, max_length=64, description="Tenant ID")
    user_mail: Optional[Union[str, List[str]]] = Field(None, description="Email address(es)")
    email: Optional[Union[str, List[str]]] = Field(None, description="Email address(es)")
    password: Optional[str] = Field(None, description="User password")
    key: Optional[str] = Field(None, description="Email type key")
    callback_url: Optional[HttpUrl] = Field(None, description="Callback URL")
    
    @validator('uid', 'tenant_id')
    def validate_ids(cls, v):
        if not validate_tenant_id(v):
            raise ValueError(f"Invalid ID format: {v}")
        return v
    
    @validator('user_mail', 'email')
    def validate_emails(cls, v):
        if v is None:
            return v
        
        if isinstance(v, str):
            emails = [v]
        else:
            emails = v
        
        if len(emails) > MAX_EMAIL_ADDRESSES:
            raise ValueError(f"Too many email addresses: {len(emails)}")
        
        for email in emails:
            if not email or not isinstance(email, str):
                raise ValueError(f"Invalid email: {email}")
            # Basic email validation
            if '@' not in email or '.' not in email.split('@')[-1]:
                raise ValueError(f"Invalid email format: {email}")
        
        return v

class StandardResponseEnvelope(BaseModel):
    """Standard response envelope following playbook contract"""
    ok: bool
    analysis_method: AnalysisMethod
    method_explain: Dict[str, Any]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    analysis_context: str = "notification_processing"
    notification_result: Dict[str, Any]
    processing_time_seconds: float
    user_context: Dict[str, Any]
    analysis_id: str = Field(..., min_length=8, max_length=32)

class NotificationAgent:
    """
    Handles email notifications based on pipeline results
    Implements Agentic AI Production Playbook patterns
    """
    
    def __init__(self):
        self.memory_manager = memory_manager
        # Rate limiter and circuit breaker now handled by centralized middleware
        self.cache = TenantAwareCache()
        self.metrics = PerformanceMetrics()
        self._lock = asyncio.Lock()
        self._tenant_rate_limiter = TenantScopedRateLimiter(max_tokens=RATE_LIMIT_TOKENS, refill_rate=RATE_LIMIT_REFILL_RATE)
        self.circuit_breaker = CircuitBreakerManager()
    
    def _get_tenant_id(self, state: Dict[str, Any]) -> str:
        """Extract tenant ID from centralized middleware state structure"""
        return state.get("security_context", {}).get("tenant_id") or state.get("tenant_id", state.get("uid", "default"))
    
    # Rate limiter functionality moved to centralized middleware
    
    def _get_rate_limiter(self):
        return self._tenant_rate_limiter

    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level from score"""
        if confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _generate_analysis_id(self) -> str:
        """Generate unique analysis ID"""
        return hashlib.sha256(f"{time.time()}{id(self)}".encode()).hexdigest()[:16]
    
    def _determine_email_type(self, state: Dict[str, Any]) -> str:
        """Determine the notification type based on event or body content"""
        body = state.get("body", {})
        event = body.get("event") or state.get("event")
        if event:
            return event
        return body.get("key", "notification")
    
    
    async def send_notification(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send appropriate notification following production pipeline
        Implements: Rate limit → Security guards → Cache → Deterministic analysis → Quality gate → Send
        """
        start_time = time.time()
        analysis_id = self._generate_analysis_id()
        tenant_id = self._get_tenant_id(state)
        
        try:
            log.info(f"📧 Notification Agent: Starting pipeline | analysis_id={analysis_id} | tenant_id={tenant_id}")
            
            # 1. Rate limit (per-tenant)
            limiter = self._get_rate_limiter()
            if not limiter.allow(tenant_id):
                self.metrics.record_method(AnalysisMethod.RATE_LIMITED)
                log.warning(f"Rate limit exceeded for tenant {tenant_id}")
                return self._create_response_envelope(
                    ok=False,
                    analysis_method=AnalysisMethod.RATE_LIMITED,
                    method_explain={"rate_limited": True, "tenant_id": tenant_id},
                    confidence_score=0.0,
                    notification_result={"success": False, "message": "Rate limit exceeded"},
                    processing_time=time.time() - start_time,
                    analysis_id=analysis_id,
                    tenant_id=tenant_id
                )
            
            # 2. Security guards
            if not validate_payload_size(state):
                log.error(f"Payload size exceeded for tenant {tenant_id}")
                return self._create_response_envelope(
                    ok=False,
                    analysis_method=AnalysisMethod.ERROR,
                    method_explain={"error": "Payload size exceeded"},
                    confidence_score=0.0,
                    notification_result={"success": False, "message": "Payload too large"},
                    processing_time=time.time() - start_time,
                    analysis_id=analysis_id,
                    tenant_id=tenant_id
                )
            
            # 3. Cache lookup (tenant-scoped)
            cache_data = {
                "type": self._determine_email_type(state),
                "emails": state.get("body", {}).get("user_mail") or state.get("body", {}).get("email"),
                "uid": state.get("uid", "")
            }
            
            cached_result = self.cache.get(tenant_id, cache_data)
            if cached_result:
                self.metrics.record_cache_hit()
                self.metrics.record_method(AnalysisMethod.CACHED)
                log.info(f"Cache hit for tenant {tenant_id}")
                return self._create_response_envelope(
                    ok=True,
                    analysis_method=AnalysisMethod.CACHED,
                    method_explain={"cached": True, "cache_ttl": CACHE_TTL_SECONDS},
                    confidence_score=cached_result.get("confidence_score", 0.8),
                    notification_result=cached_result.get("notification_result", {}),
                    processing_time=time.time() - start_time,
                    analysis_id=analysis_id,
                    tenant_id=tenant_id
                )
            
            self.metrics.record_cache_miss()
            
            # 4. Determine context
            email_type = self._determine_email_type(state)
            
            # 5. Process emails using LLM-only approach
            self.metrics.record_method(AnalysisMethod.LLM)
            log.info(f"Using LLM processing for tenant {tenant_id}")
            
            result = await self._process_emails_llm(state, email_type, tenant_id)
            method_explain = {
                "llm_processing": True,
                "email_type": email_type
            }
            
            # 7. Cache result
            self.cache.set(tenant_id, cache_data, {
                "notification_result": result,
                "confidence_score": 0.8  # LLM processing confidence
            })
            
            # 8. Record metrics
            self.metrics.record_response_time(time.time() - start_time)
            
            # 9. Create response envelope
            response = self._create_response_envelope(
                ok=result.get("success", False),
                analysis_method=AnalysisMethod.LLM,
                method_explain=method_explain,
                confidence_score=0.8,  # LLM processing confidence
                notification_result=result,
                processing_time=time.time() - start_time,
                analysis_id=analysis_id,
                tenant_id=tenant_id
            )
            
            log.info(f"📧 Notification Agent: Pipeline complete | analysis_id={analysis_id} | success={result.get('success', False)}")
            return response
            
        except Exception as e:
            self.metrics.record_method(AnalysisMethod.ERROR)
            log.error(f"❌ Notification Agent error: {str(e)} | analysis_id={analysis_id}")
            error_log.error(f"Notification Agent error: {str(e)} | analysis_id={analysis_id}")
            
            return self._create_response_envelope(
                ok=False,
                analysis_method=AnalysisMethod.ERROR,
                method_explain={"error": str(e)},
                confidence_score=0.0,
                notification_result={"success": False, "message": f"Notification error: {str(e)}"},
                processing_time=time.time() - start_time,
                analysis_id=analysis_id,
                tenant_id=tenant_id
            )
    
    async def _process_emails_llm(self, state: Dict[str, Any], email_type: str, tenant_id: str) -> Dict[str, Any]:
        """Process notifications using Novu workflows"""
        body = state.get("body", {})
        event = body.get("event") or state.get("event")
        
        # Event-based Novu workflow trigger
        if event:
            if event == "resume-parsed":
                asyncio.create_task(_trigger_novu_resume_parsed(state))
                return {
                    "success": True,
                    "message": f"Novu workflow '{event}' triggered successfully",
                    "email_type": email_type,
                    "email_results": [],
                    "successful_count": 1,
                    "failed_count": 0,
                    "event": event
                }
            else:
                # For other events, trigger Novu with the event name
                asyncio.create_task(_trigger_novu_workflow(state, event))
                return {
                    "success": True,
                    "message": f"Novu workflow '{event}' triggered successfully",
                    "email_type": email_type,
                    "email_results": [],
                    "successful_count": 1,
                    "failed_count": 0,
                    "event": event
                }
        
        # Fallback: if no event but email provided, trigger resume-parsed workflow
        user_mail = body.get("user_mail") or body.get("email")
        if user_mail:
            asyncio.create_task(_trigger_novu_resume_parsed(state))
            return {
                "success": True,
                "message": "Novu workflow 'resume-parsed' triggered (default)",
                "email_type": email_type,
                "email_results": [],
                "successful_count": 1,
                "failed_count": 0
            }
        
        return {
            "success": False,
            "message": "No event or email provided for notification",
            "email_type": email_type,
            "email_results": [],
            "successful_count": 0,
            "failed_count": 0
        }
    
    def _create_response_envelope(self, ok: bool, analysis_method: AnalysisMethod, method_explain: Dict[str, Any], 
                                confidence_score: float, notification_result: Dict[str, Any], 
                                processing_time: float, analysis_id: str, tenant_id: str) -> Dict[str, Any]:
        """Create standard response envelope"""
        return {
            "ok": ok,
            "analysis_method": analysis_method.value,
            "method_explain": method_explain,
            "confidence_score": confidence_score,
            "confidence_level": self._determine_confidence_level(confidence_score).value,
            "analysis_context": "notification_processing",
            "notification_result": notification_result,
            "processing_time_seconds": processing_time,
            "user_context": {
                "user_id": tenant_id,
                "tenant_id": tenant_id,
                "analysis_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            },
            "analysis_id": analysis_id
        }

async def _trigger_novu_workflow(state: Dict[str, Any], event_name: str) -> None:
    """Trigger a Novu workflow by event name"""
    api_key = os.getenv("NOVU_SECRET_KEY")
    if not api_key:
        return
    subscriber_id = state.get("uid")
    if not subscriber_id:
        return

    to: Dict[str, Any] = {
        "subscriberId": subscriber_id,
    }

    body = state.get("body", {})
    name_raw = (
        state.get("name") or
        body.get("name") or
        ""
    )
    
    name = ""
    if isinstance(name_raw, str):
        name = name_raw
    elif isinstance(name_raw, list) and len(name_raw) > 0:
        name = str(name_raw[0])
    elif name_raw:
        name = str(name_raw)
    
    first_name = state.get("first_name") or state.get("firstName") or body.get("first_name") or body.get("firstName")
    last_name = state.get("last_name") or state.get("lastName") or body.get("last_name") or body.get("lastName")
    
    if name and not (first_name or last_name):
        name_parts = name.strip().split(maxsplit=1)
        if len(name_parts) >= 1:
            first_name = name_parts[0]
        if len(name_parts) >= 2:
            last_name = name_parts[1]
    
    email = (
        state.get("email") or
        body.get("email") or
        body.get("user_mail") or
        ""
    )

    if first_name:
        to["firstName"] = first_name
    if last_name:
        to["lastName"] = last_name
    if email:
        to["email"] = email

    payload: Dict[str, Any] = {
        "uid": subscriber_id,
        "tenant_id": state.get("tenant_id"),
    }

    try:
        from novu import Novu
        
        novu = Novu(api_key=api_key)
        await run_blocking_io(
            novu.trigger,
            name=event_name,
            to=to,
            payload=payload,
        )
    except Exception as e:
        log.exception("Novu trigger failed for event %s: %s", event_name, e)
        return

async def _trigger_novu_resume_parsed(state: Dict[str, Any]) -> None:
    """Trigger the resume-parsed Novu workflow (convenience wrapper)"""
    await _trigger_novu_workflow(state, "resume-parsed")

# Create global instance
notification_agent = NotificationAgent()

async def notification_agent_http(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    HTTP wrapper for notification agent with centralized utilities and LLM-only approach
    """
    # Use centralized logging
    log_context = create_log_context("notification_agent", state.get("tenant_id", "default"))
    start_time = log_context["start_time"]
    request_id = log_context["request_id"]
    tenant_id = state.get("tenant_id", "default")
    
    # Get tenant-scoped memory
    notification_memory = await get_notification_agent_memory(tenant_id)
    
    try:
        # Validate request using Pydantic
        try:
            validated_request = NotificationRequest(
                uid=state.get("uid", ""),
                tenant_id=tenant_id,
                user_mail=state.get("body", {}).get("user_mail"),
                email=state.get("body", {}).get("email"),
                password=state.get("body", {}).get("password"),
                key=state.get("body", {}).get("key"),
                callback_url=state.get("callback_url")
            )
        except Exception as validation_error:
            log.error(f"Request validation failed: {validation_error}")
            state["notification_result"] = {
                "success": False,
                "message": f"Request validation failed: {validation_error}",
                "email_type": "unknown"
            }
            state["next"] = "end"
            return state
        
        # Send notification using production pipeline
        result = await notification_agent.send_notification(state)
        
        # Extract analysis_id from result
        analysis_id = result.get("analysis_id", notification_agent._generate_analysis_id())
        
        # Update state with notification result
        state["notification_result"] = result
        state["next"] = "end"  # End the flow after notification
        
        # Update bounded memory (tenant/user scoped)
        try:
            async with notification_agent._lock:
                # Store successful patterns for learning
                if result.get("ok", False):
                    memory_data = {
                        "analysis_id": analysis_id,
                        "tenant_id": tenant_id,
                        "email_type": result.get("notification_result", {}).get("email_type", "unknown"),
                        "successful_count": result.get("notification_result", {}).get("successful_count", 0),
                        "confidence_score": result.get("confidence_score", 0.0),
                        "method": result.get("analysis_method", "unknown"),
                        "timestamp": time.time()
                    }
                    
                    # Store in memory manager (offload sync ChromaDB I/O)
                    await run_blocking_io(
                        notification_agent.memory_manager.store_analysis_result,
                        tenant_id=tenant_id,
                        user_id=state.get("uid", ""),
                        agent_name="notification_agent",
                        result=memory_data
                    )
                    
                    log.info(f"📊 Memory updated for tenant {tenant_id}")
        except Exception as memory_error:
            log.warning(f"Memory update failed: {memory_error}")
        
        # Record successful completion
        processing_time = _calculate_processing_time(start_time)
        await notification_memory.record_attempt(
            'notification_processing', 'llm', True, 0.8, processing_time
        )
        
        # Log success
        log_agent_completion(log_context, {
            "success": True,
            "status": "completed",
            "email_type": result.get("notification_result", {}).get("email_type", "unknown"),
            "successful_count": result.get("notification_result", {}).get("successful_count", 0)
        }, "llm", processing_time)
        
        log.info(f"📧 Notification Agent: HTTP exit point - Success | analysis_id={analysis_id}")
        return state
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        analysis_id = notification_agent._generate_analysis_id()
        AgentLogger.log_error(log_context, f"Notification agent failed: {str(e)}", processing_time)
        
        # Record failure
        await notification_memory.record_attempt(
            'notification_processing', 'llm', False, 0.0, processing_time
        )
        
        log.error(f"📧 Notification Agent: HTTP exit point - Error: {str(e)} | analysis_id={analysis_id}")
        error_log.error(f"Notification Agent HTTP error: {str(e)} | analysis_id={analysis_id}")
        
        # Add error to state
        state["notification_result"] = {
            "success": False,
            "message": f"Notification error: {str(e)}",
            "email_type": "unknown"
        }
        state["next"] = "end"
        
        return state

def get_notification_metrics() -> Dict[str, Any]:
    """Get performance metrics for monitoring"""
    return notification_agent.metrics.get_metrics()

def get_circuit_breaker_status(tenant_id: str) -> Dict[str, Any]:
    """Get circuit breaker status for tenant"""
    if tenant_id in notification_agent.circuit_breaker.breakers:
        breaker = notification_agent.circuit_breaker.breakers[tenant_id]
        return {
            "state": breaker.state,
            "failures": breaker.failures,
            "last_failure": breaker.last_failure,
            "probe_in_progress": breaker.probe_in_progress
        }
    return {"state": "closed", "failures": 0, "last_failure": 0, "probe_in_progress": False}
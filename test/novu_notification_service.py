"""
Production-ready Novu notification service for Python AI service.
Handles both Python-initiated and Node-initiated notification events.
"""

import os
import logging
import asyncio
import hashlib
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# Configure structured logging
class ContextFilter(logging.Filter):
    """Filter that injects default values for structured logging fields"""

    def filter(self, record):
        for field in ("event_id", "subscriber_id", "source", "workflow_id"):
            if not hasattr(record, field):
                setattr(record, field, "N/A")
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - "
           "[event_id=%(event_id)s] [subscriber_id=%(subscriber_id)s] "
           "[source=%(source)s] [workflow_id=%(workflow_id)s] - %(message)s",
)

# Attach the context filter to the root logger and all its handlers
_context_filter = ContextFilter()
_root_logger = logging.getLogger()
_root_logger.addFilter(_context_filter)
for _handler in _root_logger.handlers:
    _handler.addFilter(_context_filter)

logger = logging.getLogger(__name__)


class EventSource(str, Enum):
    """Event source types"""
    PYTHON_ONLY = "python_only"
    REQUIRES_NODE = "requires_node_payload"
    PYTHON_AND_NODE = "python_and_node"  # Triggerable from Python or Node (e.g. resume_parse_failed)


@dataclass
class EventConfig:
    """Configuration for a notification event"""
    name: str
    source: EventSource
    novu_workflow_id: str
    description: str = ""
    required_payload_fields: List[str] = field(default_factory=list)


class EventRegistry:
    """Central registry for all notification events"""
    
    _events: Dict[str, EventConfig] = {}
    
    @classmethod
    def register(cls, config: EventConfig) -> None:
        """Register an event configuration"""
        cls._events[config.name] = config
        logger.info(f"Registered event: {config.name} ({config.source.value})")
    
    @classmethod
    def get(cls, event_name: str) -> Optional[EventConfig]:
        """Get event configuration by name"""
        return cls._events.get(event_name)
    
    @classmethod
    def list_events(cls, node_accessible_only: bool = False) -> Dict[str, Dict[str, Any]]:
        """List all registered events"""
        events = cls._events
        if node_accessible_only:
            events = {
                name: config for name, config in events.items()
                if config.source in (EventSource.REQUIRES_NODE, EventSource.PYTHON_AND_NODE)
            }
        return {
            name: {
                "source": config.source.value,
                "novu_workflow_id": config.novu_workflow_id,
                "description": config.description,
                "required_payload_fields": config.required_payload_fields
            }
            for name, config in events.items()
        }
    
    @classmethod
    def is_python_only(cls, event_name: str) -> bool:
        """Check if event is Python-only"""
        config = cls.get(event_name)
        return config is not None and config.source == EventSource.PYTHON_ONLY
    
    @classmethod
    def requires_node_payload(cls, event_name: str) -> bool:
        """Check if event requires Node payload (or is triggerable by Node)"""
        config = cls.get(event_name)
        return config is not None and config.source in (EventSource.REQUIRES_NODE, EventSource.PYTHON_AND_NODE)


# Register events
EventRegistry.register(EventConfig(
    name="resume_parsed",
    source=EventSource.PYTHON_AND_NODE,
    novu_workflow_id="resume-parsed",
    description="Triggered when resume is parsed (can be triggered by Python agents or Node payload)",
    required_payload_fields=["candidate_name"]
))

EventRegistry.register(EventConfig(
    name="user_action_completed",
    source=EventSource.REQUIRES_NODE,
    novu_workflow_id="user-action-completed",
    description="Triggered when user action completes in backend",
    required_payload_fields=["action_type"]
))

# Candidate & profile lifecycle (Node-initiated)
EventRegistry.register(EventConfig(
    name="candidate_registered",
    source=EventSource.REQUIRES_NODE,
    novu_workflow_id="candidate-registered",
    description="Triggered when a candidate completes registration",
    required_payload_fields=[]
))

EventRegistry.register(EventConfig(
    name="email_verified",
    source=EventSource.REQUIRES_NODE,
    novu_workflow_id="email-verified",
    description="Triggered when candidate email is verified",
    required_payload_fields=[]
))

EventRegistry.register(EventConfig(
    name="profile_incomplete",
    source=EventSource.REQUIRES_NODE,
    novu_workflow_id="profile-incomplete",
    description="Triggered when profile is incomplete (e.g. reminder)",
    required_payload_fields=[]
))

EventRegistry.register(EventConfig(
    name="profile_completed",
    source=EventSource.REQUIRES_NODE,
    novu_workflow_id="profile-completed",
    description="Triggered when candidate profile is completed",
    required_payload_fields=[]
))

EventRegistry.register(EventConfig(
    name="resume_uploaded",
    source=EventSource.REQUIRES_NODE,
    novu_workflow_id="resume-uploaded",
    description="Triggered when resume file is uploaded",
    required_payload_fields=[]
))

EventRegistry.register(EventConfig(
    name="resume_parse_failed",
    source=EventSource.PYTHON_AND_NODE,
    novu_workflow_id="resume-parse-failed",
    description="Triggered when resume parsing fails (Python on parse error, or Node)",
    required_payload_fields=[]
))

EventRegistry.register(EventConfig(
    name="job.recommended",
    source=EventSource.REQUIRES_NODE,
    novu_workflow_id="job-recommended",
    description="Triggered when a job is recommended to a candidate",
    required_payload_fields=["candidate_name", "headline", "job_url"]
))


class IdempotencyCache:
    """In-memory cache for idempotency keys (TTL-based, thread-safe)"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.cache: OrderedDict[str, datetime] = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
    
    def _cleanup_expired(self):
        """Remove expired entries (must be called with lock held)"""
        now = datetime.utcnow()
        expired_keys = [
            key for key, timestamp in self.cache.items()
            if (now - timestamp).total_seconds() > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
    
    async def is_seen(self, key: str) -> bool:
        """Check if idempotency key has been seen (thread-safe)"""
        async with self._lock:
            self._cleanup_expired()
            return key in self.cache
    
    async def mark_seen(self, key: str) -> bool:
        """
        Mark idempotency key as seen (thread-safe).
        Returns True if key was newly added, False if it already existed.
        """
        async with self._lock:
            self._cleanup_expired()
            
            # Check again after acquiring lock (double-check pattern)
            if key in self.cache:
                return False
            
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[key] = datetime.utcnow()
            return True


class Settings(BaseSettings):
    """Application settings"""
    # Use Field with validation_alias to read from environment variables
    novu_api_key: str = Field(..., validation_alias="NOVU_SECRET_KEY")
    novu_api_url: str = Field(default="https://api.novu.co/v1", validation_alias="NOVU_API_URL")
    app_name: str = Field(default="Novu Notification Service", validation_alias="APP_NAME")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    idempotency_cache_ttl: int = Field(default=3600, validation_alias="IDEMPOTENCY_CACHE_TTL")
    
    model_config = {
        "case_sensitive": False,
        # Don't load from .env file - read from os.environ (already loaded by main app)
        # The main app loads .env.development, .env.prod, etc. based on APP_ENV
        "env_file": None,
        "env_file_encoding": "utf-8",
        "populate_by_name": True,  # Allow both field name and alias
    }


# Initialize settings - will raise if NOVU_SECRET_KEY is missing
# This is intentional - the service requires this to function
try:
    settings = Settings()
except Exception as e:
    # Log but don't fail import - allows graceful degradation
    import logging
    _temp_logger = logging.getLogger(__name__)
    _temp_logger.warning(f"Novu settings initialization failed: {e}. Service will not be available.")
    settings = None


class NovuClient:
    """Client for Novu API interactions (shared singleton)"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.client: Optional[httpx.AsyncClient] = None
    
    async def initialize(self):
        """Initialize the HTTP client"""
        if self.client is None:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"ApiKey {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
            )
            logger.info("Novu client initialized with connection pooling")
    
    async def close(self):
        """Close the HTTP client"""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info("Novu client closed")
    
    async def trigger_workflow(
        self,
        workflow_id: str,
        subscriber_id: str,
        email: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Trigger a Novu workflow"""
        if not self.client:
            raise RuntimeError("NovuClient not initialized. Call initialize() first.")
        
        to_data: Dict[str, Any] = {"subscriberId": subscriber_id}
        if email:
            to_data["email"] = email
        if first_name:
            to_data["firstName"] = first_name
        if last_name:
            to_data["lastName"] = last_name
        
        request_body: Dict[str, Any] = {
            "name": workflow_id,
            "to": to_data,
            "payload": payload or {}
        }
        
        if event_id:
            request_body["transactionId"] = event_id
        
        try:
            response = await self.client.post("/events/trigger", json=request_body)
            response.raise_for_status()
            result = response.json()
            
            logger.info(
                "Novu workflow triggered successfully",
                extra={
                    "event_id": event_id or "unknown",
                    "subscriber_id": subscriber_id,
                    "source": "novu_api",
                    "workflow_id": workflow_id
                }
            )
            return result
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Novu API error: {e.response.status_code}",
                extra={
                    "event_id": event_id or "unknown",
                    "subscriber_id": subscriber_id,
                    "source": "novu_api",
                    "workflow_id": workflow_id,
                    "status_code": e.response.status_code,
                    "error": e.response.text[:200]
                }
            )
            raise
        except Exception as e:
            logger.error(
                f"Failed to trigger Novu workflow: {str(e)}",
                extra={
                    "event_id": event_id or "unknown",
                    "subscriber_id": subscriber_id,
                    "source": "novu_api",
                    "workflow_id": workflow_id,
                    "error": str(e)
                },
                exc_info=True
            )
            raise


class NotificationService:
    """Service for handling notification events"""
    
    def __init__(self, novu_client: NovuClient, idempotency_cache: IdempotencyCache):
        self.novu_client = novu_client
        self.idempotency_cache = idempotency_cache

    # ------------------------------------------------------------------
    # Payload normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_indexed_char_dict(value: Any) -> Any:
        """
        Normalize JS-style '{\"0\": \"t\", \"1\": \"h\", ...}' char maps into strings.

        Some callers accidentally spread a string into an object on the JS/Node side
        (e.g. `{ ...response }`), which results in payloads like:

            { "response": { "0": "t", "1": "h", "2": "a", ... } }

        This helper detects that pattern and converts it back to a string:

            { "response": "thank you" }

        The detection is conservative:
        - keys must be digit-only strings
        - values must all be strings
        """
        if not isinstance(value, dict) or not value:
            return value

        # All keys are digit strings and all values are strings
        if all(isinstance(k, str) and k.isdigit() for k in value.keys()) and \
           all(isinstance(v, str) for v in value.values()):
            try:
                # Rebuild string in index order
                chars = [value[str(i)] for i in sorted((int(k) for k in value.keys()))]
                return "".join(chars)
            except Exception:
                # If anything goes wrong, fall back to original value
                return value

        return value

    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        """
        Recursively normalize payload data.

        - Detects and fixes char-map objects created by spreading strings in JS
        - Traverses dicts and lists to apply normalization deeply
        """
        # First, see if this dict itself is a char map
        normalized = cls._normalize_indexed_char_dict(data)
        if normalized is not data:
            return normalized

        if isinstance(data, dict):
            return {k: cls._normalize_payload(v) for k, v in data.items()}
        if isinstance(data, list):
            return [cls._normalize_payload(v) for v in data]
        return data
    
    def _generate_event_id(self, event_name: str, subscriber_id: str, idempotency_key: Optional[str] = None) -> str:
        """Generate unique event ID"""
        if idempotency_key:
            return hashlib.sha256(f"{event_name}:{subscriber_id}:{idempotency_key}".encode()).hexdigest()[:16]
        return hashlib.sha256(f"{event_name}:{subscriber_id}:{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
    
    async def trigger_event(
        self,
        event_name: str,
        subscriber_id: str,
        email: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        source: str = "python",
        idempotency_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Trigger a notification event.
        
        Args:
            event_name: Name of the event to trigger
            subscriber_id: Novu subscriber ID
            email: Subscriber email (optional)
            first_name: Subscriber first name (optional)
            last_name: Subscriber last name (optional)
            payload: Event payload data
            source: Source of the trigger ("python" or "node")
            idempotency_key: Optional idempotency key for deduplication
        
        Returns:
            Dict with trigger result
        
        Raises:
            ValueError: If event validation fails
            RuntimeError: If Novu trigger fails
        """
        event_config = EventRegistry.get(event_name)
        if not event_config:
            raise ValueError(f"Unknown event: {event_name}")
        
        # Validate event source
        if source == "python" and event_config.source == EventSource.REQUIRES_NODE:
            raise ValueError(
                f"Event '{event_name}' requires Node payload and cannot be triggered internally"
            )
        if source == "node" and event_config.source == EventSource.PYTHON_ONLY:
            raise ValueError(
                f"Event '{event_name}' is Python-only and cannot be triggered via HTTP"
            )
        
        # Check idempotency for Node-initiated events
        if source == "node" and idempotency_key:
            # Atomic check-and-mark operation
            was_new = await self.idempotency_cache.mark_seen(idempotency_key)
            if not was_new:
                logger.warning(
                    f"Duplicate idempotency key detected: {idempotency_key}",
                    extra={
                        "event_id": "duplicate",
                        "subscriber_id": subscriber_id,
                        "source": source,
                        "workflow_id": event_config.novu_workflow_id,
                        "idempotency_key": idempotency_key
                    }
                )
                raise ValueError(f"Duplicate request detected (idempotency_key: {idempotency_key})")
        
        # Normalize payload to fix JS char-map objects (e.g. {\"0\": \"t\", \"1\": \"h\", ...})
        if payload is not None:
            payload = self._normalize_payload(payload)

        # Validate required payload fields
        if event_config.required_payload_fields:
            payload = payload or {}
            missing_fields = [
                field for field in event_config.required_payload_fields
                if field not in payload
            ]
            if missing_fields:
                raise ValueError(
                    f"Missing required payload fields: {', '.join(missing_fields)}"
                )
        
        # Generate event ID
        event_id = self._generate_event_id(event_name, subscriber_id, idempotency_key)
        
        # Trigger Novu workflow
        result = await self.novu_client.trigger_workflow(
            workflow_id=event_config.novu_workflow_id,
            subscriber_id=subscriber_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            payload=payload,
            event_id=event_id
        )
        
        logger.info(
            f"Event '{event_name}' triggered successfully",
            extra={
                "event_id": event_id,
                "subscriber_id": subscriber_id,
                "source": source,
                "workflow_id": event_config.novu_workflow_id,
                "event_name": event_name
            }
        )
        
        return {
            "success": True,
            "event": event_name,
            "workflow_id": event_config.novu_workflow_id,
            "subscriber_id": subscriber_id,
            "event_id": event_id,
            "timestamp": datetime.utcnow().isoformat(),
            "novu_response": result
        }


# FastAPI models
class NodeTriggerRequest(BaseModel):
    """
    Request model for Node-initiated events.
    
    Contract: Node → Python Notification Service
    - event: Notification event name (must exist in EventRegistry)
    - subscriber_id: Unique Novu subscriber ID (required)
    - email: Required if subscriber may not already exist
    - payload: Event-specific payload matching Novu workflow requirements (required)
    - idempotency_key: Prevents duplicate notifications (required for Node requests)
    """
    event: str = Field(..., description="Notification event name (must exist in EventRegistry)")
    subscriber_id: str = Field(..., description="Unique Novu subscriber ID")
    email: Optional[str] = Field(None, description="Subscriber email (required if subscriber may not exist)")
    first_name: Optional[str] = Field(None, description="Subscriber first name")
    last_name: Optional[str] = Field(None, description="Subscriber last name")
    payload: Dict[str, Any] = Field(..., description="Event-specific payload matching Novu workflow requirements")
    idempotency_key: str = Field(..., description="Idempotency key for preventing duplicate notifications (mandatory)")
    
    @validator("event")
    def validate_event(cls, v):
        if not EventRegistry.get(v):
            raise ValueError(f"Unknown event: {v}")
        return v
    
    @validator("idempotency_key")
    def validate_idempotency_key(cls, v):
        if not v or not v.strip():
            raise ValueError("idempotency_key is required and cannot be empty")
        return v.strip()


class EventTriggerResponse(BaseModel):
    """Response model for event triggers"""
    success: bool
    event: str
    workflow_id: str
    subscriber_id: str
    event_id: str
    timestamp: str
    message: Optional[str] = None


# FastAPI app
app = FastAPI(title=settings.app_name if settings else "Novu Notification Service", version="1.0.0")

# Global singletons
novu_client: Optional[NovuClient] = None
notification_service: Optional[NotificationService] = None
idempotency_cache: Optional[IdempotencyCache] = None


@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    global novu_client, notification_service, idempotency_cache
    
    if not settings:
        logger.error("Cannot initialize: NOVU_SECRET_KEY environment variable is missing")
        return
    
    novu_client = NovuClient(settings.novu_api_key, settings.novu_api_url)
    await novu_client.initialize()
    
    idempotency_cache = IdempotencyCache(ttl_seconds=settings.idempotency_cache_ttl)
    notification_service = NotificationService(novu_client, idempotency_cache)
    
    logger.info(
        f"{settings.app_name} started. Registered {len(EventRegistry.list_events())} events.",
        extra={
            "event_id": "startup",
            "subscriber_id": "system",
            "source": "system",
            "workflow_id": "none"
        }
    )


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global novu_client
    if novu_client:
        await novu_client.close()
    logger.info(
        f"{settings.app_name if settings else 'Novu Notification Service'} shutdown.",
        extra={
            "event_id": "shutdown",
            "subscriber_id": "system",
            "source": "system",
            "workflow_id": "none"
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": settings.app_name if settings else "Novu Notification Service"}


@app.get("/events")
async def list_events():
    """List all Node-accessible events (Python-only events are hidden)"""
    return {
        "events": EventRegistry.list_events(node_accessible_only=True),
        "count": len(EventRegistry.list_events(node_accessible_only=True))
    }


@app.post("/notifications", response_model=EventTriggerResponse)
async def trigger_node_event(request: NodeTriggerRequest):
    """
    HTTP endpoint for Node-initiated events.
    
    Node.js backend calls this endpoint to trigger notifications via Novu.
    This is an internal service-to-service API, not exposed publicly.
    
    Request Requirements:
    - event: Must be a Node-accessible event (REQUIRES_NODE source)
    - subscriber_id: Unique Novu subscriber ID
    - email: Required if subscriber may not already exist
    - payload: Event-specific payload matching Novu workflow requirements
    - idempotency_key: Mandatory for preventing duplicate notifications
    
    Returns:
    - 200: Event triggered successfully
    - 400: Validation error (missing fields, unknown event)
    - 403: Event is Python-only and cannot be triggered via HTTP
    - 500: Internal server error
    """
    
    try:
        event_config = EventRegistry.get(request.event)
        if not event_config:
            raise HTTPException(status_code=400, detail=f"Unknown event: {request.event}")
        
        # Event ownership validation: Node can trigger REQUIRES_NODE or PYTHON_AND_NODE only
        if event_config.source == EventSource.PYTHON_ONLY:
            raise HTTPException(
                status_code=403,
                detail=f"Event '{request.event}' is Python-only and cannot be triggered via HTTP"
            )
        
        if not notification_service:
            raise HTTPException(status_code=503, detail="Notification service not initialized")
        
        # Trigger event with idempotency protection
        result = await notification_service.trigger_event(
            event_name=request.event,
            subscriber_id=request.subscriber_id,
            email=request.email,
            first_name=request.first_name,
            last_name=request.last_name,
            payload=request.payload,
            source="node",
            idempotency_key=request.idempotency_key
        )
        
        return EventTriggerResponse(
            success=True,
            event=result["event"],
            workflow_id=result["workflow_id"],
            subscriber_id=result["subscriber_id"],
            event_id=result["event_id"],
            timestamp=result["timestamp"],
            message="Event triggered successfully"
        )
    
    except ValueError as e:
        logger.error(
            f"Validation error: {str(e)}",
            extra={
                "event_id": "validation_error",
                "subscriber_id": request.subscriber_id,
                "source": "node",
                "workflow_id": event_config.novu_workflow_id if event_config else "unknown"
            }
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f"Error triggering event: {str(e)}",
            extra={
                "event_id": "trigger_error",
                "subscriber_id": request.subscriber_id,
                "source": "node",
                "workflow_id": event_config.novu_workflow_id if event_config else "unknown"
            },
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to trigger event: {str(e)}")


async def trigger_python_event(
    event_name: str,
    subscriber_id: str,
    email: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Internal function for Python-initiated events.
    Call this from your Python service code.
    """
    if not notification_service:
        raise RuntimeError("Notification service not initialized")
    
    return await notification_service.trigger_event(
        event_name=event_name,
        subscriber_id=subscriber_id,
        email=email,
        first_name=first_name,
        last_name=last_name,
        payload=payload,
        source="python"
    )


async def trigger_resume_parse_failed(
    state: Dict[str, Any],
    error_reason: str,
    additional_payload: Optional[Dict[str, Any]] = None
) -> None:
    """
    Trigger resume_parse_failed Novu workflow from Python when resume parsing fails.
    Safe no-op if notification service is not initialized.
    """
    if not notification_service:
        return
    try:
        body = state.get("body") or {}
        subscriber_id = str(state.get("uid") or body.get("uid") or "unknown")
        email = body.get("user_mail") or body.get("email") or ""
        name_raw = body.get("name") or state.get("name") or ""
        name = (name_raw.strip() if isinstance(name_raw, str) else " ".join(str(x) for x in name_raw).strip()) if name_raw else ""
        first_name, _, last_name = (name.partition(" ") if name else ("", "", ""))
        if not last_name:
            last_name = first_name
            first_name = ""
        payload = {"error_reason": error_reason, **(additional_payload or {})}
        await trigger_python_event(
            event_name="resume_parse_failed",
            subscriber_id=subscriber_id,
            email=email or None,
            first_name=first_name or None,
            last_name=last_name or None,
            payload=payload
        )
    except Exception:
        pass


async def trigger_resume_parsed(
    state: Dict[str, Any],
    additional_payload: Optional[Dict[str, Any]] = None
) -> None:
    """
    Trigger resume_parsed Novu workflow from Python when resume scoring/parsing completes.
    Safe no-op if notification service is not initialized.
    """
    if not notification_service:
        return
    try:
        body = state.get("body") or {}
        subscriber_id = str(state.get("uid") or body.get("uid") or "unknown")
        email = body.get("user_mail") or body.get("email") or ""
        
        structured_resume = state.get("structured_resume") or {}
        if isinstance(structured_resume, dict):
            # Use email from parsed resume (contact_details) if not in request body
            if not email:
                contact = structured_resume.get("contact_details") or structured_resume.get("ContactDetails") or {}
                if isinstance(contact, dict):
                    email = contact.get("Email") or contact.get("email") or ""
            
            name_raw = (
                structured_resume.get("name")
                or structured_resume.get("Name")
                or body.get("name")
                or state.get("name")
                or ""
            )
        else:
            name_raw = body.get("name") or state.get("name") or ""
        
        if isinstance(name_raw, str):
            name = name_raw.strip()
        elif isinstance(name_raw, list):
            name = " ".join(str(x) for x in name_raw).strip()
        else:
            name = str(name_raw).strip() if name_raw else ""
        
        first_name, _, last_name = (name.partition(" ") if name else ("", "", ""))
        candidate_name = name or (first_name or last_name or "Candidate")
        
        payload: Dict[str, Any] = {"candidate_name": candidate_name}
        if additional_payload:
            payload.update(additional_payload)
        
        await trigger_python_event(
            event_name="resume_parsed",
            subscriber_id=subscriber_id,
            email=email or None,
            first_name=first_name or None,
            last_name=last_name or None,
            payload=payload
        )
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

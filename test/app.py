import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# CRITICAL: Disable progress bars globally to reduce log noise and improve performance
# This must be set before importing any sentence_transformers modules
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'  # Only show errors from transformers

# Disable tqdm progress bars globally
os.environ['TQDM_DISABLE'] = '1'  # Disable tqdm progress bars

# Monkey-patch tqdm to disable all progress bars
try:
    import tqdm
    # Create a no-op tqdm class that does nothing
    class NoOpTqdm:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, *args, **kwargs):
            pass
        def close(self):
            pass
        def __iter__(self):
            return iter([])
        def __call__(self, *args, **kwargs):
            return self
    
    # Replace tqdm.tqdm with our no-op version
    tqdm.tqdm = NoOpTqdm
    # Also replace the main tqdm function
    import builtins
    builtins.tqdm = NoOpTqdm
    # Disable tqdm auto module
    if hasattr(tqdm, 'auto'):
        tqdm.auto.tqdm = NoOpTqdm
except (ImportError, AttributeError):
    # tqdm not installed or doesn't have expected attributes, nothing to patch
    pass

from fastapi import FastAPI, Request, HTTPException, Header, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from core.guardrail import GuardrailMiddleware
from core.security import verify_request_token, validate_callback_url
from core.config import GRAPH_RECURSION_LIMIT
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator, validator
import aiohttp
import httpx
from core.http_client import get_http_client, close_http_client
from core.concurrency import get_max_concurrency
from core.timeout_utils import retry_with_backoff
import asyncio
import json
import traceback
import uuid

from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from google.auth import impersonated_credentials, default
from google.auth.transport.requests import Request as GoogleAuthRequest
from log_handler import setup_loggers, shutdown_loggers, get_logging_metrics, get_log_queue_size, set_correlation_id, get_correlation_id, log_security_event, log_business_event, performance_timer
import time
import logging

# Initialize loggers early so they're available for import logging
log, error_log = setup_loggers()
# Log level is set by setup_loggers() based on APP_ENV environment variable
# - development/dev: DEBUG (shows all logs)
# - qa/production: WARNING (shows only WARNING, ERROR, CRITICAL)
# Do not override here - respect environment-based configuration

# Reduce verbosity of Google Generative Language client logs:
# only show WARNING and above from these internal libraries.

_google_genai_logger_base = "google.ai.generativelanguage_v1beta"
for _logger_name in (
    _google_genai_logger_base,
    f"{_google_genai_logger_base}.services.generative_service",
    f"{_google_genai_logger_base}.services.generative_service.client",
    f"{_google_genai_logger_base}.services.generative_service.async_client",
    f"{_google_genai_logger_base}.services.generative_service.transports.grpc_asyncio",
):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

log.info("🔄 About to import supervisor_agent...")
from core.supervisor_agent import (
    create_production_graph,
    AgentState,
    get_production_metrics,
    get_circuit_breaker_status,
    get_rate_limiter_status,
    classify_pipeline_flow,
    log_pipeline_flow_complete,
)
log.info("✅ First supervisor_agent import complete")
from core.supervisor_agent import get_session_status, get_memory_summary, update_session_step, create_agent_specific_response
log.info("✅ Second supervisor_agent import complete")
from core.field_filter import get_workflow_summary, filter_agent_output, get_adaptive_insights, export_learned_patterns
from agents.interview_agent import ai_interview_agent_intelligent
from agents.career_coach import (
    career_chatbot_agent,
    career_chatbot_agent_stream,
    CareerChatRequest,
    CareerChatResponse
)
from settings import settings
from utils.session_manager import session_manager
from utils.memory_manager import memory_manager
from core.utils import run_blocking_io
from core.background_tasks import schedule_background_task
from core.langfuse_tracing import (
    get_langfuse_failure_reason,
    merge_langfuse_into_config,
    get_trace_id_from_handler,
)
from core.langfuse_scores import run_model_based_eval
from datetime import datetime
from pathlib import Path
from utils import callback_validator, callback_storage
from mock_responses import mock_generator

# Helper to surface node outputs in QA where log level is WARNING
# Cap serialized node payload in stream logs (event-loop CPU + log volume)
try:
    _STREAM_NODE_LOG_MAX_CHARS = max(1024, int(os.getenv("STREAM_NODE_LOG_MAX_CHARS", "16384")))
except ValueError:
    _STREAM_NODE_LOG_MAX_CHARS = 16384


def _log_node_visible(message: str):
    """
    Use WARNING in QA so node output appears even when INFO is suppressed.
    Keep INFO for other environments to avoid noisy production logs.
    """
    app_env = os.getenv("APP_ENV", "development").lower()
    if app_env == "dev":
        app_env = "development"
    logger_fn = log.warning if app_env == "qa" else log.info
    logger_fn(message)

log.info(
    "Langfuse settings at startup: ENABLED=%s, PUBLIC_KEY=%r, SECRET_KEY_set=%s, HOST=%r, BASE_URL=%r",
    getattr(settings, "LANGFUSE_ENABLED", None),
    (getattr(settings, "LANGFUSE_PUBLIC_KEY", None) or "")[:10] + "...",
    bool(getattr(settings, "LANGFUSE_SECRET_KEY", None)),
    getattr(settings, "LANGFUSE_HOST", None),
    getattr(settings, "LANGFUSE_BASE_URL", None),
)

# Dedicated executor for session I/O so it does not block the default pool (Section 2 Issue 3).
# Under many concurrent /analyze-resume-callback calls, 4 workers became a bottleneck (queued session ops).
import concurrent.futures
_SESSION_IO_MAX_WORKERS = max(4, int(os.getenv("SESSION_IO_MAX_WORKERS", "12")))
_session_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_SESSION_IO_MAX_WORKERS, thread_name_prefix="session_io"
)

# Global callback failure tracking with thread-safe access (Section 2 Issue 2)
_callback_failures: Dict[str, int] = {}
_callback_failures_lock = asyncio.Lock()
_CALLBACK_DISABLE_THRESHOLD = 5  # Disable callbacks after 5 consecutive failures

async def reset_callback_failures(callback_url: str = None):
    """Reset callback failure tracking for a specific URL or all URLs (thread-safe)."""
    async with _callback_failures_lock:
        if callback_url:
            _callback_failures[callback_url] = 0
            log.info(f"Reset callback failure count for {callback_url}")
        else:
            _callback_failures.clear()
            log.info("Reset all callback failure counts")

async def get_callback_failure_status() -> Dict[str, int]:
    """Get current callback failure status (thread-safe)."""
    async with _callback_failures_lock:
        return dict(_callback_failures)

async def increment_callback_failure(callback_url: str) -> int:
    """Increment failure count for a callback URL (thread-safe). Returns new count."""
    async with _callback_failures_lock:
        _callback_failures[callback_url] = _callback_failures.get(callback_url, 0) + 1
        return _callback_failures[callback_url]

async def reset_callback_failure_count(callback_url: str):
    """Reset failure count for a specific URL on success (thread-safe)."""
    async with _callback_failures_lock:
        _callback_failures[callback_url] = 0

async def is_callback_disabled(callback_url: str) -> bool:
    """Check if callbacks are disabled for a URL (thread-safe)."""
    async with _callback_failures_lock:
        return _callback_failures.get(callback_url, 0) >= _CALLBACK_DISABLE_THRESHOLD

def write_callback_to_file(uid: str, callback_url: str, payload: dict, status: str = "sent"):
    """Write callback data to a text file for debugging and monitoring."""
    try:
        # Check if callback file logging is enabled
        if not os.getenv('ENABLE_CALLBACK_FILE_LOGGING', 'true').lower() == 'true':
            return
            
        # Create callback logs directory
        callback_logs_dir = Path("logs/callbacks")
        callback_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean up old files (keep only last 100 files per UID)
        cleanup_old_callback_files(callback_logs_dir, uid)
        
        # Create filename with timestamp and UID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        filename = f"callback_{uid}_{timestamp}.txt"
        filepath = callback_logs_dir / filename
        
        # Format the callback data
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"CALLBACK DATA - {status.upper()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"UID: {uid}\n")
            f.write(f"Callback URL: {callback_url}\n")
            f.write(f"Status: {status}\n")
            f.write("-" * 80 + "\n")
            f.write("PAYLOAD:\n")
            f.write("-" * 80 + "\n")
            f.write(json.dumps(payload, indent=2, ensure_ascii=False))
            f.write("\n" + "=" * 80 + "\n")
        
        log.info(f"Callback data written to file: {filepath}")
        
    except Exception as e:
        log.error(f"Failed to write callback data to file: {e}")

def cleanup_old_callback_files(callback_logs_dir: Path, uid: str, max_files: int = 100):
    """Clean up old callback files to prevent disk space issues."""
    try:
        # Get all callback files for this UID
        callback_files = list(callback_logs_dir.glob(f"callback_{uid}_*.txt"))
        
        if len(callback_files) > max_files:
            # Sort by modification time (oldest first)
            callback_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest files
            files_to_remove = callback_files[:-max_files]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    log.debug(f"Removed old callback file: {file_path}")
                except Exception as e:
                    log.warning(f"Failed to remove old callback file {file_path}: {e}")
                    
    except Exception as e:
        log.warning(f"Failed to cleanup old callback files: {e}")

load_dotenv()
log.info("Environment variables loaded successfully")
log.info(f"API Key loaded: {'GOOGLE_API_KEY' in os.environ}")
log.info(f"API Key length: {len(os.getenv('GOOGLE_API_KEY', ''))}")

# Create the graph
log.info("Starting graph creation...")
graph = create_production_graph()
log.info("Graph created successfully")

# Create FastAPI app
app = FastAPI(title="Resume Analyzer API", version="1.0")
log.info("FastAPI app initialized")

# Debug: Log LLM model config at startup (helps verify prod uses intended model)
log.info(f"🔧 GEMINI_MODEL from settings: {settings.GEMINI_MODEL}")
log.info(f"🔧 DISABLE_QUOTA_CHECK: {getattr(settings, 'DISABLE_QUOTA_CHECK', False)}, GEMINI_TIER: {getattr(settings, 'GEMINI_TIER', 1)}")
model_registry_cfg = os.getenv("MODEL_REGISTRY_CONFIG")
log.info(f"🔧 MODEL_REGISTRY_CONFIG: {(model_registry_cfg[:200] + '...') if model_registry_cfg and len(model_registry_cfg) > 200 else (model_registry_cfg or 'not set')}")
try:
    from core.model_registry import model_registry
    reg_models = {k: v.display_name for k, v in model_registry.models.items()}
    log.info(f"🔧 Model registry models: {reg_models}")
except Exception as e:
    log.warning(f"🔧 Could not log model registry: {e}")

# Add Guardrail (edge security) middleware first
app.add_middleware(GuardrailMiddleware)

# Add CORS middleware - restrict to Jobsify domains only
ALLOWED_ORIGINS = [
    "https://dev.jobsify.ai",
    "https://qa.jobsify.ai",
    "https://www.jobsify.ai",
    "https://demo.jobsify.ai",
    # Allow localhost for development (only in development mode)
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# In production/qa, only allow Jobsify domains
if settings.APP_ENV in ["production", "qa"]:
    ALLOWED_ORIGINS = [
        "https://dev.jobsify.ai",
        "https://qa.jobsify.ai",
        "https://www.jobsify.ai",
        "https://demo.jobsify.ai",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-API-Key"],
    expose_headers=["Content-Type", "X-Request-ID"],
)
log.info(f"CORS middleware added with {len(ALLOWED_ORIGINS)} allowed origins")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return consistent JSON and log for unhandled exceptions (Section 2 Issue 4)."""
    if isinstance(exc, HTTPException):
        raise exc
    log.exception("Unhandled exception in %s %s: %s", request.method, request.url.path, exc)
    error_log.error(traceback.format_exc())
    from core.security import sanitize_error_message
    message = sanitize_error_message(exc, include_details=False)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "error": message, "type": type(exc).__name__},
    )


# Production monitoring endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint for production monitoring."""
    queue_size = get_log_queue_size()
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "logging": {
            "queue_size": queue_size,
            "status": "healthy" if queue_size < 1000 else "warning"
        }
    }
    
    # Warn if queue is backing up
    if queue_size > 5000:
        health_status["status"] = "degraded"
        health_status["logging"]["status"] = "critical"
    
    return health_status

@app.get("/metrics")
async def get_metrics():
    """Get production metrics for monitoring."""
    metrics = get_production_metrics()
    # Add logging metrics
    metrics["logging"] = get_logging_metrics()
    return metrics

@app.get("/circuit-breaker/{tenant_id}")
async def get_circuit_breaker_info(tenant_id: str):
    """Get circuit breaker status for a tenant."""
    return get_circuit_breaker_status(tenant_id)

@app.get("/rate-limiter/{tenant_id}")
async def get_rate_limiter_info(tenant_id: str):
    """Get rate limiter status for a tenant."""
    return get_rate_limiter_status(tenant_id)

# ============================================================================
# Ranker and Job Matcher Direct API Endpoints
# ============================================================================

class RankerRequest(BaseModel):
    """Request model for ranker endpoint."""
    job_id: str = Field(..., description="ID of the job description to rank candidates for")
    uid: Optional[str] = Field(None, description="Optional candidate UID for single-candidate re-ranking")
    callback_url: Optional[str] = Field(None, description="Optional callback URL to receive results")
    tenant_id: Optional[str] = Field(None, description="Optional Tenant ID")
    recruiter_questions: Optional[List[Dict[str, str]]] = Field(None, description="Optional recruiter Q&A for candidate matching")
    job_description: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional structured job description override (used for this ranker call only)",
    )
    job_details: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional additional job details/overrides to merge into the job description for this ranker call",
    )
    
    @field_validator('callback_url')
    @classmethod
    def validate_callback_url(cls, v):
        if v:
            is_valid, error_msg = validate_callback_url(v)
            if not is_valid:
                raise ValueError(error_msg)
        return v


class JobMatcherRequest(BaseModel):
    """Request model for job_matcher endpoint."""
    uid: str = Field(..., description="UID of the candidate to match jobs for")
    callback_url: Optional[str] = Field(None, description="Optional callback URL to receive results")
    tenant_id: Optional[str] = Field(None, description="Optional tenant ID")
    structured_resume: Optional[Dict[str, Any]] = Field(None, description="Optional structured resume (if not provided, will be fetched from DB)")
    recruiter_questions: Optional[List[Dict[str, str]]] = Field(None, description="Optional list of recruiter questions and candidate answers. Format: [{'question': '...', 'answer': '...'}]")
    
    @field_validator('callback_url')
    @classmethod
    def validate_callback_url(cls, v):
        if v:
            is_valid, error_msg = validate_callback_url(v)
            if not is_valid:
                raise ValueError(error_msg)
        return v


class SkillProficiencyCertificatesRequest(BaseModel):
    """Request model for skill proficiency re-analysis when certificates are added."""
    uid: str = Field(..., description="UID of the candidate")
    certificates: List[Dict[str, Any]] = Field(..., description="List of certificates. Each: certification_name (required), optionally issuing_organization, year")
    callback_url: Optional[str] = Field(None, description="Optional callback URL to receive the updated resume (strict JSON)")
    run_proficiency_analysis: Optional[bool] = Field(True, description="If true, run skill proficiency analyzer after merging; if false, only merge certs and skills")

    @field_validator('callback_url')
    @classmethod
    def validate_callback_url(cls, v):
        if v:
            is_valid, error_msg = validate_callback_url(v)
            if not is_valid:
                raise ValueError(error_msg)
        return v


class MockQuestionsRequest(BaseModel):
    """Request model for PATCH /mock-questions."""
    job_id: str = Field(..., description="ID of the job description")
    uid: str = Field(..., description="UID of the candidate")
    num_questions: Optional[int] = Field(6, description="Number of mock questions to generate")
    callback_url: Optional[str] = Field(None, description="Optional callback URL to receive results")

    @field_validator('callback_url')
    @classmethod
    def validate_callback_url(cls, v):
        if v:
            is_valid, error_msg = validate_callback_url(v)
            if not is_valid:
                raise ValueError(error_msg)
        return v


class EnhanceJobDescriptionRequest(BaseModel):
    """Request model for POST /enhance-job-description."""
    callback_url: str = Field(..., description="Callback URL to receive the enhanced JD")
    job_id: Optional[str] = Field(None, description="Optional job ID for reference")
    jd_url: Optional[str] = Field(None, description="URL to job description file (PDF/DOC/DOCX) to download and enhance")
    jd_text: Optional[str] = Field(None, description="Raw JD text to enhance (use when not providing jd_url)")
    uid: Optional[str] = Field(None, description="Optional UID for callback tracking (defaults to job_id or 'enhance_jd')")

    @field_validator('callback_url')
    @classmethod
    def validate_callback_url(cls, v):
        if v:
            is_valid, error_msg = validate_callback_url(v)
            if not is_valid:
                raise ValueError(error_msg)
        return v

    @model_validator(mode='after')
    def require_jd_source(self):
        if not (self.jd_url and str(self.jd_url).strip()) and not (self.jd_text and str(self.jd_text).strip()):
            raise ValueError("At least one of jd_url or jd_text must be provided and non-empty")
        return self


class InterviewTranscriptEvaluatorRequest(BaseModel):
    """Request model for POST /evaluate-interview-transcript."""
    job_id: str = Field(..., description="Job ID to fetch job description from Chroma")
    uid: str = Field(..., description="Candidate UID for identification")
    job_description: Optional[str] = Field(
        None,
        description="Optional JD text override (if provided, skips Chroma fetch for job_id)"
    )
    transcript: Optional[Any] = Field(
        None,
        description="Transcript as list of {speaker, text} or raw string with Interviewer:/Candidate: prefixes"
    )
    interviewer_questions: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Optional list of {question, answer} - use instead of transcript when Q&A is pre-structured"
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional interview session ID - fetch transcript from interview_chroma"
    )
    include_interviewer_feedback: bool = Field(
        True,
        description="Whether to include feedback for the interviewer (JD coverage, question quality)"
    )
    callback_url: Optional[str] = Field(
        None,
        description="If set: HTTP 202 immediately; evaluation runs in background and result POSTed here when done",
    )

    @field_validator("callback_url")
    @classmethod
    def validate_transcript_callback_url(cls, v):
        if v:
            is_valid, error_msg = validate_callback_url(v)
            if not is_valid:
                raise ValueError(error_msg)
        return v

    @model_validator(mode='after')
    def require_transcript_source(self):
        has_transcript = self.transcript is not None and (
            (isinstance(self.transcript, str) and self.transcript.strip()) or
            (isinstance(self.transcript, list) and len(self.transcript) > 0)
        )
        has_questions = self.interviewer_questions is not None and len(self.interviewer_questions) > 0
        has_session = self.session_id is not None and str(self.session_id).strip()
        if not has_transcript and not has_questions and not has_session:
            raise ValueError("At least one of transcript, interviewer_questions, or session_id must be provided and non-empty")
        return self


@app.patch("/ranker")
async def ranker_endpoint(request: Request):
    """
    Direct API endpoint to run the ranker agent for a specific job description.
    
    Two modes:
    1. **Bulk Ranking** (default): Provide only job_id - ranks all candidates
    2. **Single Candidate Re-ranking**: Provide job_id + uid - ranks specific candidate
    
    Request Body (Bulk Ranking):
    {
        "job_id": "job_123",  // REQUIRED: ID of the job description
        "callback_url": "https://example.com/callback",  // Optional
        "tenant_id": "optional_tenant_id"  // Optional
    }
    
    Request Body (Single Candidate Re-ranking):
    {
        "job_id": "job_123",  // REQUIRED: ID of the job description
        "uid": "candidate_uid",  // REQUIRED for single-candidate mode
        "callback_url": "https://example.com/callback",  // Optional
        "tenant_id": "optional_tenant_id",  // Optional
        "recruiter_questions": [...]  // Optional Q&A for matching
    }
    
    When **callback_url** is provided: returns **202 Accepted** immediately with
    ``status: processing``; full results are sent to ``callback_url`` when done
    (avoids client/proxy timeouts on long runs).

    When **callback_url** is omitted: returns **200** with the full payload after
    the ranker finishes (synchronous).

    Returns (Bulk Ranking, no callback):
    {
        "status": "success",
        "job_id": "job_123",
        "ranked_candidates": [...],
        ...
    }

    Returns (with callback_url): HTTP **202** + ``{"status":"processing", ...}``.
    """
    log.info("Entry point /ranker")
    
    try:
        body = await request.json()
        log.info(f"Received ranker request: job_id={body.get('job_id')}, uid={body.get('uid')}")
        
        # Verify GenAI token
        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            log.warning("Invalid or missing GenAI token for /ranker")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication token"
            )
        
        # Parse and validate request
        try:
            ranker_request = RankerRequest(**body)
        except Exception as e:
            log.error(f"Invalid request format: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
        
        job_id = ranker_request.job_id
        uid = ranker_request.uid
        callback_url = ranker_request.callback_url
        tenant_id = ranker_request.tenant_id or "default_tenant"
        recruiter_questions = ranker_request.recruiter_questions
        job_description_override = ranker_request.job_description
        job_details_override = ranker_request.job_details
        
        _ranker_bg_timeout = float(getattr(settings, "PIPELINE_TIMEOUT_JOB_MATCHING_SECONDS", 600)) + 60
        
        # ✅ NEW: Check if single-candidate re-ranking mode
        if uid:
            # Single candidate re-ranking mode - use job_matcher compare
            log.info(f"🎯 Single candidate re-ranking mode: candidate {uid} for job {job_id}")
            
            from agents.job_matcher import compare_candidate_with_job
            from chroma import get_job_description
            
            # Get job description from ChromaDB
            jd_dict = await run_blocking_io(get_job_description, job_id)
            if not jd_dict:
                raise HTTPException(
                    status_code=404,
                    detail=f"Job description with ID '{job_id}' not found"
                )
            
            # With callback: return 202 immediately; ranking runs in background (avoids client/proxy timeouts).
            if callback_url:
                async def _single_rerank_bg():
                    from core.pipeline_concurrency import ranker_api_execution_slot

                    async def _work():
                        from core.security import sanitize_error_message
                        _st = time.time()
                        try:
                            res = await compare_candidate_with_job(
                                candidate_id=uid,
                                job_id=job_id,
                                job_description=jd_dict,
                                recruiter_questions=recruiter_questions,
                            )
                            if "error" in res:
                                err = res.get("error", "Unknown error")
                                await send_to_callback(
                                    callback_url,
                                    uid,
                                    {
                                        "status": "error",
                                        "node": "ranker",
                                        "mode": "single_candidate_rerank",
                                        "uid": uid,
                                        "job_id": job_id,
                                        "error": err,
                                        "output": {},
                                    },
                                )
                                return
                            processing_time = time.time() - _st
                            response = {
                                "status": "success",
                                "job_id": job_id,
                                "uid": uid,
                                "match_score": res.get("match_score", 0.0),
                                "skill_match_percentage": res.get("skill_match_percentage", 0.0),
                                "skill_match_count": res.get("skill_match_count", 0),
                                "total_required_skills": res.get("total_required_skills", 0),
                                "skills_matched": res.get("skills_matched", []),
                                "skills_unmatched": res.get("skills_unmatched", []),
                                "rationale": res.get("rationale", ""),
                                "semantic_similarity": res.get("semantic_similarity", res.get("vector_similarity", 0.0)),
                                "qa_alignment_score": res.get("qa_alignment_score"),
                                "processing_time_seconds": round(processing_time, 2),
                                "processing_method": res.get("processing_method", "single_candidate_rerank"),
                                "mode": "single_candidate_rerank",
                            }
                            log.info(
                                f"✅ Single candidate re-ranking complete (async): uid={uid}, job_id={job_id}, "
                                f"match_score={res.get('match_score', 0):.2%} in {processing_time:.2f}s"
                            )
                            await send_to_callback(
                                callback_url,
                                uid,
                                {
                                    "status": "completed",
                                    "node": "ranker",
                                    "mode": "single_candidate_rerank",
                                    "uid": uid,
                                    "job_id": job_id,
                                    "output": response,
                                },
                            )
                            log.info(f"📤 Sent callback for single candidate re-rank to {callback_url}")
                        except Exception as e:
                            log.error(f"❌ Background single-candidate ranker failed: {e}", exc_info=True)
                            error_log.error(traceback.format_exc())
                            try:
                                msg = sanitize_error_message(e, include_details=False)
                                await send_to_callback(
                                    callback_url,
                                    uid,
                                    {
                                        "status": "error",
                                        "node": "ranker",
                                        "mode": "single_candidate_rerank",
                                        "uid": uid,
                                        "job_id": job_id,
                                        "error": msg,
                                        "output": {},
                                    },
                                )
                            except Exception as cb_e:
                                log.error(f"❌ Failed to send ranker error callback: {cb_e}")

                    async with ranker_api_execution_slot():
                        await _work()

                schedule_background_task(
                    _single_rerank_bg(),
                    "ranker_single_candidate",
                    timeout_seconds=_ranker_bg_timeout,
                )
                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "processing",
                        "message": "Re-ranking started; results will be sent to callback_url",
                        "job_id": job_id,
                        "uid": uid,
                        "callback_url": callback_url,
                        "mode": "single_candidate_rerank",
                    },
                )

            start_time = time.time()
            result = await compare_candidate_with_job(
                candidate_id=uid,
                job_id=job_id,
                job_description=jd_dict,
                recruiter_questions=recruiter_questions
            )
            
            # Check for errors
            if "error" in result:
                error_msg = result.get("error", "Unknown error")
                raise HTTPException(
                    status_code=404,
                    detail=error_msg
                )
            
            processing_time = time.time() - start_time
            
            # Prepare response for single candidate
            response = {
                "status": "success",
                "job_id": job_id,
                "uid": uid,
                "match_score": result.get("match_score", 0.0),
                "skill_match_percentage": result.get("skill_match_percentage", 0.0),
                "skill_match_count": result.get("skill_match_count", 0),
                "total_required_skills": result.get("total_required_skills", 0),
                "skills_matched": result.get("skills_matched", []),
                "skills_unmatched": result.get("skills_unmatched", []),
                "rationale": result.get("rationale", ""),
                "semantic_similarity": result.get("semantic_similarity", result.get("vector_similarity", 0.0)),
                "qa_alignment_score": result.get("qa_alignment_score"),
                "processing_time_seconds": round(processing_time, 2),
                "processing_method": result.get("processing_method", "single_candidate_rerank"),
                "mode": "single_candidate_rerank"
            }
            
            # ✅ Log output to terminal
            log.info(
                f"✅ Single candidate re-ranking complete: uid={uid}, job_id={job_id}, "
                f"match_score={result.get('match_score', 0):.2%} in {processing_time:.2f}s"
            )
            log.info(
                f"📊 Match Results: score={response['match_score']:.2%}, "
                f"skills={response['skill_match_count']}/{response['total_required_skills']} "
                f"({response['skill_match_percentage']:.1f}%), "
                f"semantic_similarity={response['semantic_similarity']:.2%}"
            )
            if response.get("qa_alignment_score") is not None:
                log.info(f"📝 Q&A Alignment Score: {response['qa_alignment_score']:.2%}")
            log.info(f"✅ Skills Matched ({len(response['skills_matched'])}): {', '.join(response['skills_matched'][:10])}")
            if len(response['skills_matched']) > 10:
                log.info(f"   ... and {len(response['skills_matched']) - 10} more")
            log.info(f"❌ Skills Unmatched ({len(response['skills_unmatched'])}): {', '.join(response['skills_unmatched'][:10])}")
            if len(response['skills_unmatched']) > 10:
                log.info(f"   ... and {len(response['skills_unmatched']) - 10} more")
            log.info(f"💬 Rationale: {response['rationale'][:200]}..." if len(response['rationale']) > 200 else f"💬 Rationale: {response['rationale']}")
            
            response["callback_sent"] = False
            return JSONResponse(content=response)
        
        # ✅ EXISTING: Bulk ranking mode (all candidates for job)
        log.info(f"📊 Bulk ranking mode: ranking all candidates for job_id: {job_id}")
        
        # Import ranker agent
        from agents.ranker import ranker_agent
        from chroma import get_job_description
        
        def _normalize_ranker_job_details(details: dict) -> dict:
            """
            Normalize incoming job_details (recruiter-style fields) into the canonical
            job_description fields expected by ranker.
            """
            if not isinstance(details, dict):
                return {}

            normalized: Dict[str, Any] = {}

            # Common aliases -> canonical keys used across JD storage/ranker
            if details.get("jobTitle") or details.get("job_title"):
                normalized["jobTitle"] = details.get("jobTitle") or details.get("job_title")
            if details.get("company") or details.get("companyName") or details.get("company_name"):
                normalized["company"] = details.get("company") or details.get("companyName") or details.get("company_name")
            if details.get("location"):
                normalized["location"] = details.get("location")

            # Work mode / job type
            if details.get("workMode") or details.get("work_location") or details.get("workLocation"):
                normalized["workMode"] = details.get("workMode") or details.get("work_location") or details.get("workLocation")
            if details.get("jobType") or details.get("employment_type") or details.get("employmentType"):
                normalized["jobType"] = details.get("jobType") or details.get("employment_type") or details.get("employmentType")

            # Experience / education / salary
            if details.get("experience"):
                normalized["experience"] = details.get("experience")
            if details.get("educationRequired") or details.get("education_qualification") or details.get("educationQualification"):
                normalized["educationRequired"] = (
                    details.get("educationRequired")
                    or details.get("education_qualification")
                    or details.get("educationQualification")
                )
            if details.get("salary") or details.get("salary_range") or details.get("salaryRange"):
                normalized["salary"] = details.get("salary") or details.get("salary_range") or details.get("salaryRange")

            # Skills
            required = details.get("requiredSkills") or details.get("must_have_skills") or details.get("mustHaveSkills")
            preferred = details.get("preferredSkills") or details.get("good_to_have_skills") or details.get("goodToHaveSkills")

            if isinstance(required, list):
                normalized["requiredSkills"] = [str(s).strip() for s in required if str(s).strip()]
            if isinstance(preferred, list):
                normalized["preferredSkills"] = [str(s).strip() for s in preferred if str(s).strip()]

            # Optional free-text JD content (if present)
            if details.get("fullJobDescription") or details.get("full_job_description") or details.get("jd_text"):
                normalized["fullJobDescription"] = (
                    details.get("fullJobDescription")
                    or details.get("full_job_description")
                    or details.get("jd_text")
                )

            return normalized

        # Build effective job description:
        # - start with stored JD for job_id (if exists)
        # - overlay job_description (structured) if provided
        # - overlay normalized job_details (extra fields) if provided
        jd_dict = await run_blocking_io(get_job_description, job_id) or {}

        # If caller provides a structured JD override, prefer it (but keep job_id context)
        if isinstance(job_description_override, dict) and job_description_override:
            log.info("🧩 Using job_description override for this /ranker call")
            jd_dict = {**jd_dict, **job_description_override}

        # Merge any additional job_details into the JD for this run
        if isinstance(job_details_override, dict) and job_details_override:
            normalized_details = _normalize_ranker_job_details(job_details_override)
            log.info(
                "🧩 Merging normalized job_details overrides into job_description for this /ranker call "
                f"(keys={list(normalized_details.keys())})"
            )
            jd_dict = {**jd_dict, **normalized_details}

        if not jd_dict:
            raise HTTPException(
                status_code=404,
                detail=f"Job description with ID '{job_id}' not found and no job_description/job_details overrides provided"
            )
        
        # Prepare state for ranker_agent
        state = {
            "job_description": jd_dict,
            "job_id": job_id,
            "jd_id": job_id,
            "tenant_id": tenant_id,
            "uid": tenant_id,
            "rerank_trigger": "api_call"
        }
        if callback_url:
            state["callback_url"] = callback_url
        if recruiter_questions:
            state["recruiter_questions"] = recruiter_questions
            log.info(f"✅ Added {len(recruiter_questions)} recruiter Q&A to ranker state")
        
        # With callback: return 202 immediately; bulk ranking runs in background.
        if callback_url:
            async def _bulk_ranker_bg():
                from core.pipeline_concurrency import ranker_api_execution_slot

                async def _work():
                    from core.security import sanitize_error_message
                    _st = time.time()
                    try:
                        res = await ranker_agent(state)
                        ranked_candidates = res.get("ranked_candidates", [])
                        total_candidates = len(ranked_candidates)
                        processing_time = time.time() - _st
                        job_title = jd_dict.get("jobTitle") or jd_dict.get("title") or ""
                        company = jd_dict.get("company") or ""
                        log.info(
                            f"✅ Ranker completed (async): {total_candidates} candidates ranked "
                            f"for job_id {job_id} in {processing_time:.2f}s"
                        )
                        callback_payload = {
                            "status": "completed",
                            "node": "ranker",
                            "mode": "bulk_ranking",
                            "uid": tenant_id,
                            "job_id": job_id,
                            "job_title": job_title,
                            "company": company,
                            "output": {
                                "job_id": job_id,
                                "job_title": job_title,
                                "company": company,
                                "ranked_candidates": ranked_candidates,
                                "total_candidates": total_candidates,
                                "processing_time_seconds": round(processing_time, 2),
                                "api_call": True,
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        }
                        await send_to_callback(callback_url, tenant_id, callback_payload)
                        log.info(f"📤 Sent callback for ranker API call to {callback_url}")
                    except Exception as e:
                        log.error(f"❌ Background bulk ranker failed: {e}", exc_info=True)
                        error_log.error(traceback.format_exc())
                        try:
                            msg = sanitize_error_message(e, include_details=False)
                            await send_to_callback(
                                callback_url,
                                tenant_id,
                                {
                                    "status": "error",
                                    "node": "ranker",
                                    "mode": "bulk_ranking",
                                    "uid": tenant_id,
                                    "job_id": job_id,
                                    "error": msg,
                                    "output": {},
                                },
                            )
                        except Exception as cb_e:
                            log.error(f"❌ Failed to send ranker error callback: {cb_e}")

                async with ranker_api_execution_slot():
                    await _work()

            schedule_background_task(
                _bulk_ranker_bg(),
                "ranker_bulk",
                timeout_seconds=_ranker_bg_timeout,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "status": "processing",
                    "message": "Ranking started; results will be sent to callback_url",
                    "job_id": job_id,
                    "callback_url": callback_url,
                    "mode": "bulk_ranking",
                },
            )

        start_time = time.time()
        result = await ranker_agent(state)
        
        # Extract ranked candidates
        ranked_candidates = result.get("ranked_candidates", [])
        total_candidates = len(ranked_candidates)
        
        processing_time = time.time() - start_time
        
        log.info(
            f"✅ Ranker completed: {total_candidates} candidates ranked "
            f"for job_id {job_id} in {processing_time:.2f}s"
        )
        
        # Job context for recruiter UI (avoid mixing up rankings for multiple jobs)
        job_title = jd_dict.get("jobTitle") or jd_dict.get("title") or ""
        company = jd_dict.get("company") or ""
        
        response = {
            "status": "success",
            "job_id": job_id,
            "job_title": job_title,
            "company": company,
            "ranked_candidates": ranked_candidates,
            "total_candidates": total_candidates,
            "processing_time_seconds": round(processing_time, 2),
            "callback_sent": False,
            "mode": "bulk_ranking"
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"❌ Ranker endpoint failed: {e}", exc_info=True)
        error_log.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "type": type(e).__name__
            }
        )


@app.post("/evaluate-interview-transcript")
async def evaluate_interview_transcript_endpoint(request: Request):
    """
    Evaluate an interview transcript against a job description.

    Provides per-answer feedback (relevance, depth, evidence, JD alignment),
    overall candidate summary vs JD, and optional interviewer feedback
    (JD coverage gaps, question quality, suggested follow-ups).

    Request Body:
    {
        "job_id": "job_123",           // REQUIRED: Job ID (fetches JD from Chroma)
        "uid": "candidate_456",        // REQUIRED: Candidate UID
        "job_description": "...",      // Optional: JD text override (skips Chroma fetch)
        "transcript": [...],           // OR interviewer_questions OR session_id
        "interviewer_questions": [{"question": "...", "answer": "..."}],
        "session_id": "session_xyz",   // Optional: fetch transcript from interview_chroma
        "include_interviewer_feedback": true
    }

    Returns:
    {
        "job_id": "job_123",
        "uid": "candidate_456",
        "per_answer_evaluations": [...],
        "overall_summary": {...},
        "interviewer_feedback": {...},
        "processing_time_seconds": 12.5
    }
    """
    try:
        body = await request.json()

        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            log.warning("Invalid or missing GenAI token for /evaluate-interview-transcript")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication token"
            )

        try:
            req = InterviewTranscriptEvaluatorRequest(**body)
        except Exception as e:
            log.error(f"Invalid request format: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")

        from agents.interview_transcript_evaluator import evaluate_interview_transcript

        _transcript_bg_timeout = float(getattr(settings, "PIPELINE_TIMEOUT_SECONDS", 300)) + 120

        if req.callback_url:
            async def _transcript_eval_bg():
                from core.security import sanitize_error_message

                try:
                    result = await evaluate_interview_transcript(
                        job_id=req.job_id,
                        uid=req.uid,
                        job_description=req.job_description,
                        transcript=req.transcript,
                        interviewer_questions=req.interviewer_questions,
                        session_id=req.session_id,
                        include_interviewer_feedback=req.include_interviewer_feedback,
                    )
                    if result.get("error"):
                        await send_to_callback(
                            req.callback_url,
                            req.uid,
                            {
                                "status": "error",
                                "node": "evaluate_interview_transcript",
                                "error": result.get("error", "evaluation_failed"),
                                "output": result,
                            },
                        )
                        return
                    await send_to_callback(
                        req.callback_url,
                        req.uid,
                        {
                            "status": "completed",
                            "node": "evaluate_interview_transcript",
                            "output": {
                                "job_id": result.get("job_id"),
                                "uid": result.get("uid"),
                                "per_answer_evaluations": result.get("per_answer_evaluations", []),
                                "overall_summary": result.get("overall_summary", {}),
                                "interviewer_feedback": result.get("interviewer_feedback", {}),
                                "processing_time_seconds": result.get("processing_time_seconds", 0),
                                "total_qa_pairs": result.get("total_qa_pairs", 0),
                            },
                        },
                    )
                    log.info(f"📤 Sent interview transcript evaluation callback to {req.callback_url}")
                except Exception as e:
                    log.error(f"Background interview transcript evaluation failed: {e}", exc_info=True)
                    try:
                        msg = sanitize_error_message(e, include_details=False)
                        await send_to_callback(
                            req.callback_url,
                            req.uid,
                            {
                                "status": "error",
                                "node": "evaluate_interview_transcript",
                                "error": msg,
                                "output": {},
                            },
                        )
                    except Exception as cb_e:
                        log.error(f"Failed to send transcript eval error callback: {cb_e}")

            schedule_background_task(
                _transcript_eval_bg(),
                "evaluate_interview_transcript",
                timeout_seconds=_transcript_bg_timeout,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "status": "processing",
                    "message": "Interview transcript evaluation started; results will be sent to callback_url",
                    "job_id": req.job_id,
                    "uid": req.uid,
                    "callback_url": req.callback_url,
                    "node": "evaluate_interview_transcript",
                },
            )

        result = await evaluate_interview_transcript(
            job_id=req.job_id,
            uid=req.uid,
            job_description=req.job_description,
            transcript=req.transcript,
            interviewer_questions=req.interviewer_questions,
            session_id=req.session_id,
            include_interviewer_feedback=req.include_interviewer_feedback,
        )

        if result.get("error"):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": result["error"], **result}
            )

        return JSONResponse(content={
            "status": "success",
            "job_id": result.get("job_id"),
            "uid": result.get("uid"),
            "per_answer_evaluations": result.get("per_answer_evaluations", []),
            "overall_summary": result.get("overall_summary", {}),
            "interviewer_feedback": result.get("interviewer_feedback", {}),
            "processing_time_seconds": result.get("processing_time_seconds", 0),
            "total_qa_pairs": result.get("total_qa_pairs", 0),
        })
    except Exception as e:
        log.error(f"Interview transcript evaluation failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )


@app.post("/api/job-matcher")
@app.patch("/api/job-matcher")
async def job_matcher_endpoint(request: Request):
    """
    Direct API endpoint to run the job_matcher agent for a specific candidate.
    
    This endpoint allows you to trigger job matching for a specific candidate
    on-demand, independent of the normal flow or scheduled jobs.
    
    Request Body:
    {
        "uid": "candidate_123",  // REQUIRED: UID of the candidate
        "callback_url": "https://example.com/callback",  // Optional: URL to receive results
        "tenant_id": "optional_tenant_id",  // Optional: Tenant ID
        "structured_resume": {...}  // Optional: Structured resume (if not provided, will be fetched from DB)
    }
    
    When **callback_url** is provided: **202 Accepted** + ``processing``; results
    on callback. Without callback: **200** + full match payload (synchronous).
    """
    log.info("Entry point /api/job-matcher")
    
    try:
        body = await request.json()
        log.info(f"Received job_matcher request: uid={body.get('uid')}")
        
        # Verify GenAI token
        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            log.warning("Invalid or missing GenAI token for /api/job-matcher")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication token"
            )
        
        # Parse and validate request
        try:
            matcher_request = JobMatcherRequest(**body)
        except Exception as e:
            log.error(f"Invalid request format: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
        
        uid = matcher_request.uid
        callback_url = matcher_request.callback_url
        tenant_id = matcher_request.tenant_id or uid
        structured_resume = matcher_request.structured_resume
        recruiter_questions = matcher_request.recruiter_questions
        
        log.info(f"Matching jobs for candidate: {uid}")
        
        # Import job_matcher agent
        from agents.job_matcher import job_matcher_agent
        from chroma import fetch_structured_resume
        
        # Get structured resume if not provided
        if not structured_resume:
            log.info(f"Fetching structured resume for UID: {uid}")
            structured_resume = await run_blocking_io(fetch_structured_resume, uid, tenant_id)
            
            if not structured_resume:
                raise HTTPException(
                    status_code=404,
                    detail=f"Structured resume not found for UID '{uid}'. Please provide structured_resume in request or ensure candidate has uploaded a resume."
                )
        
        # Prepare state for job_matcher_agent
        state = {
            "uid": uid,
            "structured_resume": structured_resume,
            "tenant_id": tenant_id,
            "rematch_trigger": "api_call"  # Mark as API call, not scheduled
        }
        
        if callback_url:
            state["callback_url"] = callback_url
        
        if recruiter_questions:
            state["recruiter_questions"] = recruiter_questions
            log.info(f"✅ Added {len(recruiter_questions) if isinstance(recruiter_questions, list) else 1} recruiter Q&A to state")
        
        _jm_timeout = float(getattr(settings, "PIPELINE_TIMEOUT_JOB_MATCHING_SECONDS", 600)) + 60

        if callback_url:
            async def _job_matcher_bg():
                from core.pipeline_concurrency import ranker_api_execution_slot

                async def _work():
                    from core.security import sanitize_error_message
                    _st = time.time()
                    try:
                        result = await job_matcher_agent(state)
                        top_matches = result.get("top_matches", [])
                        total_matches = result.get("total_matches_found", len(top_matches))
                        matching_method = result.get("matching_method", "skill_based_matching")
                        matcher_status = result.get("job_matcher_status", "success")
                        processing_time = time.time() - _st
                        log.info(
                            f"✅ Job matcher completed (async): {total_matches} matches for {uid} in {processing_time:.2f}s"
                        )
                        out = {
                            "top_matches": top_matches,
                            "total_matches_found": total_matches,
                            "matching_method": matching_method,
                            "job_matcher_status": matcher_status,
                            "api_call": True,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        if result.get("error"):
                            out["warning"] = result.get("error")
                        await send_to_callback(
                            callback_url,
                            uid,
                            {"status": "completed", "node": "job_matcher", "uid": uid, "output": out},
                        )
                        log.info(f"📤 Sent callback for job_matcher API call to {callback_url}")
                    except Exception as e:
                        log.error(f"❌ Background job_matcher failed: {e}", exc_info=True)
                        error_log.error(traceback.format_exc())
                        try:
                            msg = sanitize_error_message(e, include_details=False)
                            await send_to_callback(
                                callback_url,
                                uid,
                                {
                                    "status": "error",
                                    "node": "job_matcher",
                                    "uid": uid,
                                    "error": msg,
                                    "output": {},
                                },
                            )
                        except Exception as cb_e:
                            log.error(f"❌ Failed to send job_matcher error callback: {cb_e}")

                async with ranker_api_execution_slot():
                    await _work()

            schedule_background_task(
                _job_matcher_bg(),
                "job_matcher_api",
                timeout_seconds=_jm_timeout,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "status": "processing",
                    "message": "Job matching started; results will be sent to callback_url",
                    "uid": uid,
                    "callback_url": callback_url,
                },
            )

        start_time = time.time()
        result = await job_matcher_agent(state)
        
        top_matches = result.get("top_matches", [])
        total_matches = result.get("total_matches_found", len(top_matches))
        matching_method = result.get("matching_method", "skill_based_matching")
        matcher_status = result.get("job_matcher_status", "success")
        
        processing_time = time.time() - start_time
        
        log.info(
            f"✅ Job matcher completed: {total_matches} matches found "
            f"for candidate {uid} in {processing_time:.2f}s"
        )
        
        response = {
            "status": "success",
            "uid": uid,
            "top_matches": top_matches,
            "total_matches_found": total_matches,
            "matching_method": matching_method,
            "job_matcher_status": matcher_status,
            "processing_time_seconds": round(processing_time, 2),
            "callback_sent": False,
        }
        
        if result.get("error"):
            response["warning"] = result.get("error")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"❌ Job matcher endpoint failed: {e}", exc_info=True)
        error_log.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "type": type(e).__name__
            }
        )


def _cert_dedup_alternate_key(key: str) -> str:
    """
    Return singular/plural variant of the last word for cert deduplication.
    E.g. 'attention to details' -> 'attention to detail'; 'attention to detail' -> 'attention to details'.
    """
    if not key or not key.strip():
        return key
    parts = key.split()
    if not parts:
        return key
    last = parts[-1]
    if len(last) > 1 and last.endswith("s"):
        alt = " ".join(parts[:-1] + [last[:-1]])
    else:
        alt = " ".join(parts[:-1] + [last + "s"])
    return alt


async def _process_certificates_and_skills(
    uid: str,
    certificates_list: List[Dict[str, Any]],
    run_proficiency_analysis: bool = True,
    callback_url: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Shared flow: fetch resume by UID, merge certificates and derived skills, optionally run
    skill proficiency analyzer, persist, return strict JSON (updated skills + rest of resume).
    Returns None if structured resume not found (caller should return 404).
    """
    from chroma import fetch_structured_resume, get_resume_doc, upsert_resume_doc
    from agents.skill_proficiency_analyzer import (
        skill_proficiency_analyzer_agent,
        extract_skills_from_certificate_names,
        _normalize_skill_name,
    )
    tenant_id = tenant_id or uid
    structured_resume = await run_blocking_io(fetch_structured_resume, uid, tenant_id)
    if not structured_resume or not isinstance(structured_resume, dict):
        return None
    # Normalize and merge certificates (dedupe by key or singular/plural variant of last word)
    existing_certs = list(structured_resume.get("certifications") or [])
    seen_cert_names = {_normalize_skill_name(c.get("certification_name") or c.get("name") or "").lower() for c in existing_certs if isinstance(c, dict)}
    for cert in certificates_list or []:
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
    # Extract skills from the new certs (certification names) and merge into resume skills
    derived_skills = extract_skills_from_certificate_names(certificates_list or [])
    existing_skills = list(structured_resume.get("skills") or [])
    skill_map = {}
    for s in existing_skills:
        if isinstance(s, dict):
            sn = (s.get("SkillName") or s.get("skill") or s.get("name") or "").strip()
            if sn:
                norm = _normalize_skill_name(sn).lower()
                if norm not in skill_map:
                    skill_map[norm] = dict(s) if isinstance(s, dict) else {"SkillName": sn, "Proficiency": "", "PositiveRationale": "", "NegativeRationale": "", "HowToImprove": ""}
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
    if run_proficiency_analysis:
        state = {
            "uid": uid,
            "tenant_id": tenant_id,
            "structured_resume": structured_resume,
            "skills": structured_resume["skills"],
        }
        result = await skill_proficiency_analyzer_agent(state)
        updated_skills = result.get("skills") or []
        skills_without_supporting_evidence = result.get("skills_without_supporting_evidence") or []
        structured_resume["skills"] = updated_skills
    else:
        updated_skills = structured_resume["skills"]
        skills_without_supporting_evidence = structured_resume.get("skills_without_supporting_evidence") or []
    doc = await run_blocking_io(get_resume_doc, uid) or {}
    doc["structured_resume"] = structured_resume
    await run_blocking_io(upsert_resume_doc, uid, doc)
    response_body = {"uid": uid, **structured_resume, "skills": updated_skills, "skills_without_supporting_evidence": skills_without_supporting_evidence}
    return response_body


@app.post("/skill-proficiency-certificates")
@app.patch("/skill-proficiency-certificates")
async def skill_proficiency_certificates_endpoint(request: Request):
    """
    Add certificates and merge derived skills into the candidate's resume via LangGraph;
    optionally run skill proficiency analyzer (so skills needing proof are updated).
    Returns strict JSON: uid, skills, skills_without_supporting_evidence, and rest of resume.
    """
    log.info("Entry point /skill-proficiency-certificates")
    try:
        body = await request.json()
        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
        req = SkillProficiencyCertificatesRequest(**body)
        uid = req.uid
        certificates_list = req.certificates or []
        run_proficiency_analysis = req.run_proficiency_analysis if req.run_proficiency_analysis is not None else True
        callback_url = req.callback_url
        tenant_id = uid
        if not certificates_list:
            raise HTTPException(status_code=400, detail="certificates array is required and must not be empty")

        state = {
            "uid": uid,
            "tenant_id": tenant_id,
            "callback_url": callback_url,
            "endpoint_name": "skill_proficiency_certificates",
            "body": {
                **body,
                "certificates": [dict(c) for c in certificates_list],
                "run_proficiency_analysis": run_proficiency_analysis,
            },
        }

        _cert_bg_timeout = float(getattr(settings, "PIPELINE_TIMEOUT_SECONDS", 300)) + 120

        if callback_url:
            async def _skill_cert_bg():
                from core.security import sanitize_error_message

                try:
                    graph_result = await graph.ainvoke(
                        state, config={"recursion_limit": GRAPH_RECURSION_LIMIT}
                    )
                    result = graph_result.get("certificates_result")
                    if result is None:
                        err = graph_result.get("error") or f"Structured resume not found for UID '{uid}'"
                        await send_to_callback(
                            callback_url,
                            uid,
                            {
                                "status": "error",
                                "node": "skill_proficiency_certificates",
                                "error": err,
                                "output": {},
                            },
                        )
                        return
                    await send_to_callback(callback_url, uid, result)
                    log.info(f"📤 Sent skill-proficiency-certificates callback to {callback_url}")
                except Exception as e:
                    log.error(f"Background skill-proficiency-certificates failed: {e}", exc_info=True)
                    try:
                        msg = sanitize_error_message(e, include_details=False)
                        await send_to_callback(
                            callback_url,
                            uid,
                            {
                                "status": "error",
                                "node": "skill_proficiency_certificates",
                                "error": msg,
                                "output": {},
                            },
                        )
                    except Exception as cb_e:
                        log.error(f"skill-proficiency-certificates error callback failed: {cb_e}")

            schedule_background_task(
                _skill_cert_bg(),
                "skill_proficiency_certificates",
                timeout_seconds=_cert_bg_timeout,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "status": "processing",
                    "message": "Certificate merge started; results will be sent to callback_url",
                    "uid": uid,
                    "callback_url": callback_url,
                    "node": "skill_proficiency_certificates",
                },
            )

        graph_result = await graph.ainvoke(state, config={"recursion_limit": GRAPH_RECURSION_LIMIT})
        result = graph_result.get("certificates_result")
        if result is None:
            error_msg = graph_result.get("error") or f"Structured resume not found for UID '{uid}'"
            raise HTTPException(status_code=404, detail=error_msg)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"skill-proficiency-certificates endpoint failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


MAX_CERTIFICATE_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CERTIFICATE_EXTENSIONS = (".pdf", ".doc", ".docx")


@app.post("/upload-certificates")
async def certificates_upload_endpoint(
    request: Request,
    uid: str = Form(...),
    file: UploadFile = File(...),
    callback_url: Optional[str] = Form(None),
    run_proficiency_analysis: Optional[str] = Form("true"),
):
    """
    Upload a certificate (PDF, DOC, or DOCX); extract certification name from document text,
    route through LangGraph update_certificates node to merge cert and derived skills,
    optionally run skill proficiency analyzer.
    Returns strict JSON: uid, skills, skills_without_supporting_evidence, rest of resume.
    """
    log.info("Entry point /upload-certificates")
    try:
        is_valid_token = await verify_request_token(request, None)
        if not is_valid_token:
            raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
        if not file.filename:
            raise HTTPException(status_code=400, detail="file is required")
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_CERTIFICATE_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Allowed file types: {', '.join(ALLOWED_CERTIFICATE_EXTENSIONS)}")
        content = await file.read()
        if len(content) > MAX_CERTIFICATE_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_CERTIFICATE_UPLOAD_BYTES // (1024*1024)} MB limit")
        # Offload sync PDF/DOC/DOCX + possible sync LLM extraction so the event loop stays responsive
        if ext == ".pdf":
            text = await run_blocking_io(_extract_text_from_pdf, content)
        elif ext == ".docx":
            text = await run_blocking_io(_extract_text_from_docx, content)
        else:
            from utils.resume_utils import _extract_text_from_doc

            text = await run_blocking_io(_extract_text_from_doc, content)
        if not text or len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from the certificate document")
        cert_dict = _parse_certificate_from_text(text)
        run_prof = run_proficiency_analysis is None or str(run_proficiency_analysis).lower() in ("true", "1", "yes")

        if callback_url:
            is_valid_url, cb_err = validate_callback_url(callback_url)
            if not is_valid_url:
                raise HTTPException(status_code=400, detail=cb_err or "Invalid callback_url")

        state = {
            "uid": uid,
            "tenant_id": uid,
            "callback_url": callback_url,
            "endpoint_name": "upload_certificates",
            "body": {
                "certificates": [cert_dict],
                "run_proficiency_analysis": run_prof,
            },
        }

        _upload_cert_timeout = float(getattr(settings, "PIPELINE_TIMEOUT_SECONDS", 300)) + 120

        if callback_url:
            async def _upload_cert_bg():
                from core.security import sanitize_error_message

                try:
                    graph_result = await graph.ainvoke(
                        state, config={"recursion_limit": GRAPH_RECURSION_LIMIT}
                    )
                    result = graph_result.get("certificates_result")
                    if result is None:
                        err = graph_result.get("error") or f"Structured resume not found for UID '{uid}'"
                        await send_to_callback(
                            callback_url,
                            uid,
                            {
                                "status": "error",
                                "node": "upload_certificates",
                                "error": err,
                                "output": {},
                            },
                        )
                        return
                    await send_to_callback(callback_url, uid, result)
                    log.info(f"📤 Sent upload-certificates callback to {callback_url}")
                except Exception as e:
                    log.error(f"Background upload-certificates failed: {e}", exc_info=True)
                    try:
                        msg = sanitize_error_message(e, include_details=False)
                        await send_to_callback(
                            callback_url,
                            uid,
                            {
                                "status": "error",
                                "node": "upload_certificates",
                                "error": msg,
                                "output": {},
                            },
                        )
                    except Exception as cb_e:
                        log.error(f"upload-certificates error callback failed: {cb_e}")

            schedule_background_task(
                _upload_cert_bg(),
                "upload_certificates",
                timeout_seconds=_upload_cert_timeout,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "status": "processing",
                    "message": "Certificate upload processing started; results will be sent to callback_url",
                    "uid": uid,
                    "callback_url": callback_url,
                    "node": "upload_certificates",
                },
            )

        graph_result = await graph.ainvoke(state, config={"recursion_limit": GRAPH_RECURSION_LIMIT})
        result = graph_result.get("certificates_result")
        if result is None:
            error_msg = graph_result.get("error") or f"Structured resume not found for UID '{uid}'"
            raise HTTPException(status_code=404, detail=error_msg)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/upload-certificates endpoint failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


@app.patch("/mock-questions")
async def mock_questions_endpoint(request: Request):
    """
    Generate mock interview questions for a candidate and job.
    Fetches JD and candidate from DB, generates Q&A based on the job.
    Independent call (not part of graph).
    """
    log.info("Entry point PATCH /mock-questions")
    try:
        body = await request.json()
        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            log.warning("Invalid or missing GenAI token for /mock-questions")
            raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
        try:
            req = MockQuestionsRequest(**body)
        except Exception as e:
            log.error(f"Invalid request format: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        job_id = req.job_id
        uid = req.uid
        num_questions = req.num_questions or 6
        callback_url = req.callback_url
        from agents.mock_interview_prep_agent import generate_mock_interview_qa
        if callback_url:
            async def run_and_callback():
                try:
                    result = await generate_mock_interview_qa(job_id, uid, num_questions)
                    payload = {
                        "status": "completed",
                        "node": "mock_questions",
                        "output": result,
                    }
                    await send_to_callback(callback_url, uid, payload)
                    log.info(f"📤 Sent mock-questions callback to {callback_url}")
                except Exception as e:
                    log.error(f"Mock-questions callback failed: {e}", exc_info=True)
            schedule_background_task(run_and_callback(), "run_and_callback")
            return JSONResponse(
                status_code=202,
                content={
                    "status": "processing",
                    "job_id": job_id,
                    "uid": uid,
                    "message": "Mock questions generation started; results will be sent to callback_url",
                    "callback_url": callback_url,
                },
            )
        result = await generate_mock_interview_qa(job_id, uid, num_questions)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Mock-questions endpoint failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e), "type": type(e).__name__})


@app.post("/enhance-job-description")
@app.patch("/enhance-job-description")
async def enhance_job_description_endpoint(request: Request):
    """
    Enhance job description text only. Downloads from jd_url or uses jd_text,
    runs enhancement, and sends the result to callback_url. Client can then
    send the enhanced text in the normal flow (validate → parse → rank).
    """
    log.info("Entry point POST /enhance-job-description")
    try:
        body = await request.json()
        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
        try:
            req = EnhanceJobDescriptionRequest(**body)
        except Exception as e:
            log.error(f"Invalid request format: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        callback_url = req.callback_url
        job_id = req.job_id
        uid = req.uid or job_id or "enhance_jd"

        # Resolve JD text: prefer jd_text if non-empty, else download from jd_url
        jd_text = (req.jd_text or "").strip()
        if not jd_text and req.jd_url:
            from utils.resume_utils import download_resume_text_async
            from core.security import sanitize_url_async
            jd_url = await sanitize_url_async(str(req.jd_url))
            if not jd_url:
                raise HTTPException(status_code=400, detail="Invalid or unsafe jd_url")
            try:
                jd_text = await download_resume_text_async(jd_url, file_type="job_description") or ""
                jd_text = (jd_text or "").strip()
            except Exception as e:
                log.error(f"Failed to download JD from URL: {e}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Failed to extract job description from URL: {str(e)}")

        if not jd_text:
            raise HTTPException(
                status_code=400,
                detail="No JD text available. Provide non-empty jd_text or a jd_url that returns extractable text."
            )

        async def do_enhance_and_callback():
            try:
                from agents.jd_enhancer import enhance_jd_text
                start = time.time()
                enhanced = await enhance_jd_text(jd_text)
                elapsed = time.time() - start
                output = {
                    "enhanced_jd_text": enhanced if enhanced else "",
                    "jd_text": jd_text,
                    "processing_time_seconds": round(elapsed, 2),
                }
                if job_id:
                    output["job_id"] = job_id
                payload = {
                    "status": "completed",
                    "node": "enhance_job_description",
                    "output": output,
                }
                await send_to_callback(callback_url, uid, payload)
                log.info(f"📤 Sent enhance-job-description callback to {callback_url}")
            except Exception as e:
                log.error(f"Enhance-job-description failed: {e}", exc_info=True)
                try:
                    await send_to_callback(callback_url, uid, {
                        "status": "error",
                        "node": "enhance_job_description",
                        "error": str(e),
                        "output": {},
                    })
                except Exception as cb_err:
                    log.error(f"Failed to send error callback: {cb_err}")

        schedule_background_task(do_enhance_and_callback(), "do_enhance_and_callback")
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "message": "JD enhancement started; results will be sent to callback_url",
                "callback_url": callback_url,
                "job_id": job_id,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Enhance-job-description endpoint failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e), "type": type(e).__name__},
        )


@app.post("/search-candidate")
@app.patch("/search-candidate")
async def recruitment_search_candidates(request: Request):
    """
    Search for candidates based on natural language query.
    
    This endpoint:
    1. Parses the recruiter's query to extract structured entities
    2. Normalizes locations, skills, and seniority levels
    3. Executes vector search with metadata filtering
    4. Returns ranked candidates with match explanations
    5. Optionally sends results to callback_url if provided
    
    Request Body:
    {
        "query": "Senior Python Developer in Dubai",  // REQUIRED: Natural language query
        "top_k": 50,  // Optional: Number of candidates to return (configurable via RECRUITMENT_DEFAULT_TOP_K setting)
        "callback_url": "https://apis-buh3qzwapq-uc.a.run.app/apis/user/aiResponse/..."  // Optional: Callback URL
    }
    
    Returns:
    {
        "query": "Senior Python Developer in Dubai",
        "parsed_query": {...},
        "matches": [...],
        "total_found": 25,
        "processing_time_ms": 450,
        "search_strategy": "vector_similarity_with_filters",
        "filters_applied": {...},
        "avg_match_score": 0.72,
        "top_match_score": 0.87,
        "callback_url": "..."  // If provided
    }
    """
    log.info("Entry point /search-candidate")
    
    try:
        body = await request.json()
        log.info(f"Received recruitment search request: query={body.get('query', '')[:50]}...")
        
        # Verify GenAI token
        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            log.warning("Invalid or missing GenAI token for /search-candidate")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication token"
            )
        
        # Parse and validate request
        try:
            from api.recruitment_api import CandidateSearchRequest
            search_request = CandidateSearchRequest(**body)
        except Exception as e:
            log.error(f"Invalid request format: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
        
        if search_request.callback_url:
            log.info(f"Callback URL provided: {search_request.callback_url}")

        from api.recruitment_api import get_adapter, CandidateSearchResponse, CandidateMatchResponse
        adapter = get_adapter()

        async def _run_recruitment_search() -> CandidateSearchResponse:
            result = await adapter.search_candidates(
                raw_query=search_request.query,
                top_k=search_request.top_k,
            )
            return CandidateSearchResponse(
                query=result.query.raw_query,
                parsed_query=result.query.to_dict(),
                matches=[
                    CandidateMatchResponse(
                        candidate_id=match.candidate_id,
                        candidate_name=str(match.candidate_name) if match.candidate_name else "N/A",
                        match_score=match.match_score,
                        candidate_skills=match.candidate_skills,
                        total_experience_years=match.total_experience_years,
                        current_location=match.current_location,
                        seniority_level=match.seniority_level,
                        education_level=match.education_level,
                    )
                    for match in result.matches
                ],
                total_found=result.total_found,
                processing_time_ms=result.processing_time_ms,
                search_strategy=result.search_strategy,
                filters_applied=result.filters_applied,
                avg_match_score=result.avg_match_score,
                top_match_score=result.top_match_score,
                callback_url=str(search_request.callback_url) if search_request.callback_url else None,
            )

        if search_request.callback_url:
            async def _search_candidate_bg():
                from core.security import sanitize_error_message
                import uuid
                import re

                try:
                    response = await _run_recruitment_search()
                    log.info(
                        f"Search complete (async): {response.total_found} candidates, {response.processing_time_ms}ms"
                    )
                    callback_url_str = str(search_request.callback_url)
                    uid_match = re.search(r"/apis/user/aiResponse/([a-zA-Z0-9_-]+)", callback_url_str)
                    if uid_match:
                        extracted_uid = uid_match.group(1)
                        log.info(f"Extracted UID from callback URL: {extracted_uid}")
                    else:
                        uid_match = re.search(r"/([a-zA-Z0-9]{20,})$", callback_url_str)
                        if uid_match:
                            extracted_uid = uid_match.group(1)
                            log.info(f"Extracted UID from callback URL (fallback): {extracted_uid}")
                        else:
                            extracted_uid = "recruitment_search"
                            log.warning(
                                f"Could not extract UID from callback URL, using placeholder: {extracted_uid}"
                            )

                    search_id = str(uuid.uuid4())
                    callback_payload = {
                        "node": "recruitment_search",
                        "status": "completed",
                        "output": {
                            "search_id": search_id,
                            "query": response.query,
                            "parsed_query": response.parsed_query,
                            "total_found": response.total_found,
                            "processing_time_ms": response.processing_time_ms,
                            "search_strategy": response.search_strategy,
                            "filters_applied": response.filters_applied,
                            "avg_match_score": response.avg_match_score,
                            "top_match_score": response.top_match_score,
                            "matches": [
                                {
                                    "candidate_id": match.candidate_id,
                                    "candidate_name": match.candidate_name,
                                    "match_score": match.match_score,
                                    "candidate_skills": match.candidate_skills,
                                    "total_experience_years": match.total_experience_years,
                                    "current_location": match.current_location,
                                    "seniority_level": match.seniority_level,
                                    "education_level": match.education_level,
                                }
                                for match in response.matches
                            ],
                            "timestamp": datetime.now().isoformat(),
                        },
                    }
                    await send_to_callback(callback_url_str, extracted_uid, callback_payload)
                    log.info(f"📤 Sent callback for search to {callback_url_str}")
                except Exception as e:
                    log.error(f"Background recruitment search failed: {e}", exc_info=True)
                    try:
                        msg = sanitize_error_message(e, include_details=False)
                        await send_to_callback(
                            str(search_request.callback_url),
                            "recruitment_search",
                            {
                                "node": "recruitment_search",
                                "status": "error",
                                "error": msg,
                                "output": {},
                            },
                        )
                    except Exception as cb_e:
                        log.error(f"Failed to send recruitment search error callback: {cb_e}")

            _search_bg_timeout = float(getattr(settings, "PIPELINE_TIMEOUT_SECONDS", 300)) + 120
            schedule_background_task(
                _search_candidate_bg(),
                "recruitment_search",
                timeout_seconds=_search_bg_timeout,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "status": "processing",
                    "message": "Candidate search started; results will be sent to callback_url",
                    "query": search_request.query,
                    "callback_url": str(search_request.callback_url),
                    "node": "recruitment_search",
                },
            )

        response = await _run_recruitment_search()

        log.info(f"Search complete: {response.total_found} candidates, {response.processing_time_ms}ms")

        callback_sent = False
        
        # Return response as dict (matching pattern from other endpoints)
        response_dict = {
            "query": response.query,
            "parsed_query": response.parsed_query,
            "matches": [
                {
                    "candidate_id": match.candidate_id,
                    "candidate_name": match.candidate_name,
                    "match_score": match.match_score,
                    "candidate_skills": match.candidate_skills,
                    "total_experience_years": match.total_experience_years,
                    "current_location": match.current_location,
                    "seniority_level": match.seniority_level,
                    "education_level": match.education_level
                }
                for match in response.matches
            ],
            "total_found": response.total_found,
            "processing_time_ms": response.processing_time_ms,
            "search_strategy": response.search_strategy,
            "filters_applied": response.filters_applied,
            "avg_match_score": response.avg_match_score,
            "top_match_score": response.top_match_score,
            "callback_sent": callback_sent
        }
        
        if response.callback_url:
            response_dict["callback_url"] = response.callback_url
        
        return JSONResponse(content=response_dict)
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"❌ Recruitment search endpoint failed: {e}", exc_info=True)
        error_log.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "type": type(e).__name__
            }
        )


# Session Management Endpoints
@app.get("/session-status/{uid}")
async def get_user_session_status(uid: str):
    """Get session status for a user."""
    return await asyncio.to_thread(get_session_status, uid)

@app.get("/memory-summary/{uid}")
async def get_user_memory_summary(uid: str):
    """Get memory summary for a user."""
    return get_memory_summary(uid)

# Field Filtering Endpoints
@app.get("/workflow-summary/{uid}")
async def get_workflow_summary_endpoint(uid: str):
    """Get workflow summary showing active agents and progress."""
    try:
        # Get session data to understand current state
        # Run blocking I/O in thread pool to avoid blocking event loop
        session = await run_blocking_io(session_manager.get_session_by_owner, uid, "candidate_pipeline")
        if not session:
            return {"error": "No active session found for user"}
        
        # Get memory data to reconstruct state
        memory_data = memory_manager.get_session_data_for_resume(uid, "all")
        
        # Create a mock state for analysis
        mock_state = {
            "uid": uid,
            "tenant_id": session.state.data.get("tenant_id", "default-tenant"),
            "session_id": session.session_id,
            **memory_data
        }
        
        summary = get_workflow_summary(mock_state)
        return summary
    except Exception as e:
        log.error(f"Failed to get workflow summary: {e}")
        return {"error": f"Failed to get workflow summary: {str(e)}"}

@app.get("/agent-output/{uid}/{agent_name}")
async def get_agent_specific_output(uid: str, agent_name: str):
    """Get filtered output for a specific agent."""
    try:
        # Get session data
        # Run blocking I/O in thread pool to avoid blocking event loop
        session = await run_blocking_io(session_manager.get_session_by_owner, uid, "candidate_pipeline")
        if not session:
            return {"error": "No active session found for user"}
        
        # Get memory data
        memory_data = memory_manager.get_session_data_for_resume(uid, "all")
        
        # Create mock state
        mock_state = {
            "uid": uid,
            "tenant_id": session.state.data.get("tenant_id", "default-tenant"),
            "session_id": session.session_id,
            **memory_data
        }
        
        # Get agent-specific response
        response = create_agent_specific_response(mock_state, agent_name)
        return response
    except Exception as e:
        log.error(f"Failed to get agent output: {e}")
        return {"error": f"Failed to get agent output: {str(e)}"}

@app.get("/filtered-output/{uid}")
async def get_filtered_output(uid: str):
    """Get intelligently filtered output based on active agents."""
    try:
        # Get session data
        # Run blocking I/O in thread pool to avoid blocking event loop
        session = await run_blocking_io(session_manager.get_session_by_owner, uid, "candidate_pipeline")
        if not session:
            return {"error": "No active session found for user"}
        
        # Get memory data
        memory_data = memory_manager.get_session_data_for_resume(uid, "all")
        
        # Create mock state
        mock_state = {
            "uid": uid,
            "tenant_id": session.state.data.get("tenant_id", "default-tenant"),
            "session_id": session.session_id,
            **memory_data
        }
        
        # Get filtered output
        filtered_output = filter_agent_output(mock_state)
        return filtered_output
    except Exception as e:
        log.error(f"Failed to get filtered output: {e}")
        return {"error": f"Failed to get filtered output: {str(e)}"}

# Adaptive Learning Endpoints
@app.get("/adaptive-insights")
async def get_adaptive_insights_endpoint():
    """Get insights from the adaptive learning system."""
    try:
        insights = get_adaptive_insights()
        return insights
    except Exception as e:
        log.error(f"Failed to get adaptive insights: {e}")
        return {"error": f"Failed to get adaptive insights: {str(e)}"}

@app.get("/learned-patterns")
async def get_learned_patterns_endpoint():
    """Export all learned patterns for analysis."""
    try:
        patterns = export_learned_patterns()
        return patterns
    except Exception as e:
        log.error(f"Failed to export learned patterns: {e}")
        return {"error": f"Failed to export learned patterns: {str(e)}"}

@app.get("/agent-predictions/{agent_name}")
async def get_agent_predictions(agent_name: str):
    """Get predictions for what fields an agent will produce."""
    try:
        # Create a mock input state for prediction
        mock_input = {
            "resume_text": "Sample resume text",
            "structured_resume": {},
            "user_interests": []
        }
        
        from core.field_filter import field_filter_manager
        predicted_fields = field_filter_manager.predict_agent_output(agent_name, mock_input)
        
        return {
            "agent_name": agent_name,
            "predicted_fields": list(predicted_fields),
            "prediction_confidence": field_filter_manager.agent_schemas.get(agent_name, {}).confidence if hasattr(field_filter_manager, 'agent_schemas') else 0.0
        }
    except Exception as e:
        log.error(f"Failed to get agent predictions: {e}")
        return {"error": f"Failed to get agent predictions: {str(e)}"}

@app.post("/update-session-step")
async def update_user_session_step(request: Request):
    """Update session step and data."""
    try:
        body = await request.json()
        session_id = body.get("session_id")
        step = body.get("step")
        data = body.get("data", {})
        progress = body.get("progress")
        
        if not session_id or not step:
            raise HTTPException(status_code=400, detail="session_id and step are required")
        
        success = await asyncio.to_thread(update_session_step, session_id, step, data, progress)
        
        if success:
            return {"status": "success", "message": "Session step updated"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update session step")
            
    except Exception as e:
        log.error(f"Failed to update session step: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/callback-status")
async def get_callback_status():
    """Get current callback failure status"""
    try:
        failures = await get_callback_failure_status()
        return {
            "callback_failures": failures,
            "disable_threshold": _CALLBACK_DISABLE_THRESHOLD,
            "disabled_urls": [url for url, count in failures.items() if count >= _CALLBACK_DISABLE_THRESHOLD],
            "callback_file_logging_enabled": os.getenv('ENABLE_CALLBACK_FILE_LOGGING', 'true').lower() == 'true'
        }
    except Exception as e:
        log.error(f"Error getting callback status: {e}")
        return {"error": str(e)}

@app.get("/callback-files/{uid}")
async def get_callback_files(uid: str, limit: int = 10):
    """Get list of callback files for a specific UID"""
    try:
        callback_logs_dir = Path("logs/callbacks")
        if not callback_logs_dir.exists():
            return {"files": [], "message": "No callback files directory found"}
        
        # Get all callback files for this UID
        callback_files = list(callback_logs_dir.glob(f"callback_{uid}_*.txt"))
        
        # Sort by modification time (newest first)
        callback_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Limit results
        callback_files = callback_files[:limit]
        
        files_info = []
        for file_path in callback_files:
            stat = file_path.stat()
            files_info.append({
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "path": str(file_path)
            })
        
        return {
            "uid": uid,
            "files": files_info,
            "total_files": len(list(callback_logs_dir.glob(f"callback_{uid}_*.txt"))),
            "showing": len(files_info)
        }
        
    except Exception as e:
        log.error(f"Error getting callback files: {e}")
        return {"error": str(e)}

@app.get("/callback-files/{uid}/{filename}")
async def get_callback_file_content(uid: str, filename: str):
    """Get content of a specific callback file"""
    try:
        callback_logs_dir = Path("logs/callbacks")
        file_path = callback_logs_dir / filename
        
        # Security check - ensure the file belongs to the UID
        if not filename.startswith(f"callback_{uid}_"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "filename": filename,
            "uid": uid,
            "content": content
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error reading callback file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/callback-reset")
async def reset_callback_status_endpoint(callback_url: str = None):
    """Reset callback failure tracking"""
    try:
        await reset_callback_failures(callback_url)
        return {
            "message": f"Reset callback failures for {'all URLs' if callback_url is None else callback_url}",
            "current_status": await get_callback_failure_status()
        }
    except Exception as e:
        log.error(f"Error resetting callback status: {e}")
        return {"error": str(e)}

# Production Pydantic validation schemas
class ProductionResumeInput(BaseModel):
    """Production-ready resume input validation."""
    model_config = {"arbitrary_types_allowed": True}
    
    uid: str = Field(..., description="User ID", min_length=8, max_length=64, pattern=r'^[a-zA-Z0-9_\-]{8,64}$')
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID", min_length=8, max_length=64, pattern=r'^[a-zA-Z0-9_\-]{8,64}$')
    callback_url: HttpUrl = Field(..., description="Callback URL")
    resume_url: Optional[HttpUrl] = Field(None, description="URL to the resume PDF (optional for 2nd call - uses stored context)")
    jd_url: Optional[HttpUrl] = Field(None, description="URL to the job description PDF")
    user_interests: Optional[List[Dict[str, Any]]] = Field(None, description="User interests (optional for 2nd call - uses stored context)")
    assessment_plan: Optional[List[Dict[str, Any]]] = Field(None, description="Assessment plan")
    job_id: Optional[str] = Field(None, description="Job ID")
    assessment_id: Optional[str] = Field(None, description="Assessment ID")
    question_doc_id: Optional[str] = Field(None, description="Question document ID")
    assessment_type: Optional[str] = Field(None, description="Assessment type")
    submission: Optional[Dict[str, Any]] = Field(None, description="Assessment submission")
    user_mail: Optional[str] = Field(None, description="User email")
    email: Optional[str] = Field(None, description="User email")
    enhance_jd_only: Optional[bool] = Field(default=False, description="If true, only enhance JD text and return it; user can send again for validation (provide jd_url or job_details)")
    
    @field_validator('resume_url', 'jd_url')
    @classmethod
    def validate_urls(cls, v):
        if v is not None:
            from core.security import sanitize_url
            sanitized = sanitize_url(str(v))
            if not sanitized:
                raise ValueError("Invalid or unsafe URL")
        return v
    
    @field_validator('user_mail', 'email')
    @classmethod
    def validate_email(cls, v):
        if v is not None:
            import re
            email_pattern = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
            if not email_pattern.match(v):
                raise ValueError("Invalid email format")
        return v

class ProductionResponse(BaseModel):
    """Production response envelope validation."""
    ok: bool
    analysis_method: str
    method_explain: Dict[str, Any]
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_level: str = Field(pattern=r'^(high|medium|low)$')
    analysis_context: str
    skill_gap_analysis: Dict[str, Any]
    processing_time_seconds: float = Field(ge=0.0)
    user_context: Dict[str, Any]
    analysis_id: str
    error: Optional[str] = None
    validation_error: Optional[str] = None
    security_error: Optional[str] = None
    rate_limit_error: Optional[str] = None

#     @field_validator('resume_url')
#     @classmethod
#     # def validate_resume_url(cls, v): 
#     def validate_resume_url(cls, v):
#         if not v.startswith(('http://', 'https://')):
#             raise ValueError("URL must start with http:// or https://")
#         if not v.lower().split('?')[0].endswith('.pdf'):
#             raise ValueError("URL must point to a PDF file (ending with .pdf)")
#         return v

class ResumeURLInputWithCallback(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    resume_url: str = Field(..., description="URL to the resume file")
    callback_url: HttpUrl = Field(..., description="Callback URL")
    uid: str = Field(..., description="User ID or session identifier")

    @field_validator('resume_url')
    @classmethod
    def validate_resume_url(cls, v: str) -> str:
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        
        # Allow .pdf, .doc, and .docx extensions
        allowed_extensions = ('.pdf', '.doc', '.docx')
        file_path = v.split('?')[0]
        if not file_path.lower().endswith(allowed_extensions):
            raise ValueError("URL must point to a PDF, DOC, or DOCX file.")
        return v


class StructuredResumeInput(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    structured_resume: dict = Field(
        ..., description="Structured resume data in JSON format"
    )

from typing import Union

class ResumeContentGeneratorInput(BaseModel):
    """Input validation for resume content generator endpoint."""
    model_config = {"arbitrary_types_allowed": True}
    
    uid: str = Field(..., description="User ID", min_length=8, max_length=64, pattern=r'^[a-zA-Z0-9_\-]{8,64}$')
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID", min_length=8, max_length=64, pattern=r'^[a-zA-Z0-9_\-]{8,64}$')
    callback_url: Optional[HttpUrl] = Field(default=None, description="Callback URL (optional, for consistency with resume flow)")
    session_id: Optional[str] = Field(default=None, description="Session ID for retrieving additional context")
    certificates: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of certificate objects to include in ATS resume (e.g. name, issuer, date); max 50 items",
    )
    
    @field_validator("certificates")
    @classmethod
    def validate_certificates(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("certificates must be a list")
        if len(v) > 50:
            raise ValueError("certificates list must not exceed 50 items")
        for i, item in enumerate(v):
            if not isinstance(item, dict):
                raise ValueError(f"certificates[{i}] must be a dict")
        return v
    
    @field_validator('callback_url')
    @classmethod
    def validate_callback_url(cls, v):
        if v is not None:
            from core.security import sanitize_url
            sanitized = sanitize_url(str(v))
            if not sanitized:
                raise ValueError("Invalid or unsafe callback URL")
        return v

class AssessmentInput(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    uid: str = Field(..., description="User ID")
    type: str = Field(..., description="Assessment type, e.g. coding/mcq")
    topic: str = Field(..., description="Assessment topic")
    difficulty: str = Field(..., description="Difficulty level")
    num_questions: int | dict = Field(
        ..., description="Number of questions. Can be an integer or dict (e.g. {'MCQ': 5, 'Short': 2})"
    )
    timer_per_question: int | None = Field(
        None, description="Time per question in seconds (only for coding assessments)"
    )
    assessment_time_minutes: int | None = Field(
        None, description="Total assessment time in minutes (only for multi-type assessments)"
    )
    callback_url: HttpUrl = Field(..., description="Callback URL")

# Token Cache (per UID, 30 min TTL) with thread-safe access (Section 2 Issue 1)
token_cache: Dict[str, Dict] = {}
_token_cache_lock = asyncio.Lock()
TOKEN_TTL = 1800  # 30 minutes in seconds

# Token cleanup logic (thread-safe)
async def cleanup_tokens():
    while True:
        now = time.time()
        async with _token_cache_lock:
            expired_uids = [uid for uid, entry in token_cache.items() if now >= entry["expires_at"]]
            for uid in expired_uids:
                del token_cache[uid]
        await asyncio.sleep(300)  # clean every 5 min


@app.on_event("startup")
async def startup_event():
    from core.concurrency import MAX_CONCURRENCY
    from core.pipeline_concurrency import log_pipeline_concurrency_at_startup

    log_pipeline_concurrency_at_startup()
    log.info(
        "Concurrency: MAX_CONCURRENCY=%s (LangGraph parallel branches per run)",
        MAX_CONCURRENCY,
    )
    log.info(
        "Concurrency: SESSION_IO_MAX_WORKERS=%s (thread pool for session get_or_reuse)",
        _SESSION_IO_MAX_WORKERS,
    )

    # Start the job scheduler for background tasks (including daily re-ranking)
    try:
        from utils.job_scheduler import job_scheduler
        await job_scheduler.start_scheduler()
        log.info("✅ Job scheduler started successfully")
    except Exception as e:
        log.error(f"❌ Failed to start job scheduler: {e}", exc_info=True)
    
    # Issue 4.4: Use schedule_background_task for proper exception handling
    schedule_background_task(cleanup_tokens(), "cleanup_tokens")
    # Initialize shared HTTP client
    await get_http_client()
    
    # OPTIMIZATION: Pre-warm LLM connection to reduce first-request latency
    try:
        from agents.interview_agent.llm_utils import warmup_llm_connection
        # Issue 4.4: Use schedule_background_task for proper exception handling
        schedule_background_task(warmup_llm_connection(), "warmup_llm_connection")
        log.info("✅ LLM warmup task started")
    except Exception as e:
        log.warning(f"⚠️ LLM warmup failed (non-critical): {e}")
    # Start quota manager background tasks now that event loop is running
    from core.quota_manager import quota_manager
    quota_manager._start_background_tasks()
    
    # Initialize Novu notification service
    global _novu_notification_service, _novu_client, _idempotency_cache
    try:
        # Verify environment variable is available before importing
        novu_secret_key = os.getenv("NOVU_SECRET_KEY")
        if not novu_secret_key:
            log.warning("⚠️ NOVU_SECRET_KEY not found in environment. Notifications will be disabled.")
            log.debug(f"Available env vars with NOVU: {[k for k in os.environ.keys() if 'NOVU' in k.upper()]}")
        else:
            log.debug(f"NOVU_SECRET_KEY found (length: {len(novu_secret_key)})")
        
        from novu_notification_service import (
            EventRegistry,
            NovuClient,
            NotificationService,
            IdempotencyCache,
            Settings as NovuSettings
        )
        novu_settings = NovuSettings()
        _novu_client = NovuClient(novu_settings.novu_api_key, novu_settings.novu_api_url)
        await _novu_client.initialize()
        
        _idempotency_cache = IdempotencyCache(ttl_seconds=novu_settings.idempotency_cache_ttl)
        _novu_notification_service = NotificationService(_novu_client, _idempotency_cache)
        # Wire into novu_notification_service module so Python-triggered events (e.g. resume_parse_failed) use this instance
        import novu_notification_service as _nns
        _nns.notification_service = _novu_notification_service
        _nns.novu_client = _novu_client
        _nns.idempotency_cache = _idempotency_cache
        
        log.info(f"✅ Novu notification service initialized. Registered {len(EventRegistry.list_events())} events.")
    except Exception as e:
        log.error(f"❌ Failed to initialize Novu notification service: {e}", exc_info=True)
        log.warning("⚠️ Notifications will be disabled. Check NOVU_SECRET_KEY environment variable.")
        log.debug(f"Environment check - NOVU_SECRET_KEY present: {bool(os.getenv('NOVU_SECRET_KEY'))}")

@app.on_event("shutdown")
async def shutdown_event():
    # Stop the job scheduler
    try:
        from utils.job_scheduler import job_scheduler
        await job_scheduler.stop_scheduler()
        log.info("✅ Job scheduler stopped successfully")
    except Exception as e:
        log.error(f"❌ Failed to stop job scheduler: {e}", exc_info=True)
    
    # Shutdown ThreadPoolExecutors used by agent sync wrappers
    try:
        from core.supervisor_agent import _sync_wrapper_executor
        _sync_wrapper_executor.shutdown(wait=True, cancel_futures=False)
        log.info("✅ Agent sync wrapper executor shutdown successfully")
    except Exception as e:
        log.warning(f"⚠️ Error shutting down sync wrapper executor: {e}")
    
    # Cleanup Novu notification service
    global _novu_client
    if _novu_client:
        try:
            await _novu_client.close()
            log.info("✅ Novu notification service shutdown.")
        except Exception as e:
            log.warning(f"⚠️ Error shutting down Novu service: {e}")
    
    # Close shared HTTP client
    await close_http_client()
    # Gracefully shutdown logging system (ensures all queued logs are written)
    shutdown_loggers()


async def get_cached_token(uid: str) -> Optional[str]:
    """Get UID-specific cached token if valid, else refresh automatically.
    
    Thread-safe: Uses asyncio.Lock to prevent race conditions (Section 2 Issue 1).
    """
    now = time.time()
    
    # Check cache with lock (read)
    async with _token_cache_lock:
        user_entry = token_cache.get(uid)
        if user_entry and now < user_entry["expires_at"]:
            return user_entry["token"]

    # Fetch new token (outside lock to avoid blocking other requests)
    new_token = await get_access_token(uid)
    
    # Update cache with lock (write)
    if new_token:
        async with _token_cache_lock:
            token_cache[uid] = {
                "token": new_token,
                "expires_at": now + TOKEN_TTL
            }
        return new_token
    return None

@app.get("/")
def read_root():
    log.info("Entry point / root endpoint")
    return {"message": "Resume Analyzer API is running!"}

# Token Generation Logic

# def get_service_account_token():
#     try:
#         source_credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])




#         target_service_account = "firebase-adminsdk-fbsvc@jobsify-5d910.iam.gserviceaccount.com"
#         target_scopes = [
#             "https://www.googleapis.com/auth/cloud-platform",
#             "https://www.googleapis.com/auth/userinfo.email"
#         ]

#         impersonated_creds = impersonated_credentials.Credentials(
#             source_credentials=source_credentials,
#             target_principal=target_service_account,
#             target_scopes=target_scopes,
#             lifetime=3600
#         )

#         # impersonated_creds.refresh(GoogleRequest())
#         impersonated_creds.refresh(GoogleAuthRequest())

#         return impersonated_creds.token

#     except Exception as e:
#         raise Exception(f"Token generation failed: {str(e)}")

async def get_access_token(uid: str) -> Optional[str]:
    """
    Get Firebase ID token for a user via internal service authentication.
    
    Args:
        uid: Firebase user ID
        
    Returns:
        Firebase ID token or None if failed
    """
    log.info(f"Entry point get_access_token | uid={uid}")
    
    try:
        # Use shared HTTP client for better connection pooling
        client = await get_http_client()
        
        # Call service endpoint with API key
        url = settings.TOKEN
        
        log.info(f"Requesting ID token | uid={uid} | url={url}")
        
        response = await retry_with_backoff(
            client.post,
            max_retries=3,
            url=url,
            json={"uid": uid},
            headers={
                "x-api-key": settings.INTERNAL_API_KEY,
                "Content-Type": "application/json"
            }
        )
        
        log.info(f"Response status | status={response.status_code} | uid={uid}")
        
        if response.status_code != 200:
            error_text = response.text
            # Downgrade 404 (User not found) to WARNING level as this may be expected
            if response.status_code == 404:
                log.warning(
                    f"User not found when fetching ID token | "
                    f"status={response.status_code} | "
                    f"uid={uid} | "
                    f"error={error_text}"
                )
            else:
                log.error(
                    f"Failed to fetch ID token | "
                    f"status={response.status_code} | "
                    f"uid={uid} | "
                    f"error={error_text}"
                )
            return None
        
        data = response.json()
        id_token = data.get("idToken")
        
        if not id_token:
            log.error(f"ID token missing in response | uid={uid}")
            return None
        
        log.info(f"Successfully obtained ID token | uid={uid}")
        return id_token
                
    except httpx.HTTPError as e:
        log.error(f"Network error in get_access_token | uid={uid} | error={str(e)}")
        return None
    except Exception as e:
        log.error(f"Unexpected error in get_access_token | uid={uid} | error={str(e)}")
        return None
    


# @app.get("/get-token")
# async def get_token():
#     log.info("Entry point /get-token")
#     try:
#         token = await get_access_token()
#         log.info("Exit point #1 /get-token | Success")
#         return JSONResponse(content={"accessToken": token}, status_code=200)
#     except Exception as e:
#         log.error(f"Exit point #2 /get-token | Error {str(e)}")
#         error_log.error(traceback.format_exc())
#         return JSONResponse(
#             content={"message": "Unable to fetch token", "error": str(e)},
#             status_code=500
 
#        )

@app.get("/get-token/{uid}")
async def get_token(uid: str):
    log.info(f"Entry point /get-token | uid={uid}")
    try:
        token = await get_cached_token(uid)
        if not token:
            raise Exception("Failed to get token")
        log.info("Exit point #1 /get-token | Success")
        return JSONResponse(content={"accessToken": token}, status_code=200)
    except Exception as e:
        log.error(f"Exit point #2 /get-token | Error {str(e)}")
        error_log.error(traceback.format_exc())
        return JSONResponse(
            content={"message": "Unable to fetch token", "error": str(e)},
            status_code=500
        )



async def send_to_callback(callback_url: str, uid: str, payload: dict):
    log.info(f"Entry point send_to_callback | uid={uid}, url={callback_url}, payload={payload}")

    # Determine agent name for schema validation (node field)
    agent_name = payload.get("node")
    # Only validate and store for agents up to assessment_recommender
    AGENTS_TO_VALIDATE = {
        "is_valid_resume",  # Node name from validate_resume.py
        "validate_resume",  # Alternative node name that might be used
        "is_valid_jd",      # Node name from validate_jd.py
        "validate_jd",      # Alternative node name that might be used
        "groq_resume_parser",  # Replaces personal_info_parser, education_parser, experience_parser, skills_parser
        "resume_scorer",
        "interest_filler",
        "career_advisor",
        "market_and_course_recommender",
        "assessment_recommender",
        "assessment_question_generator",
        "assessment_evaluator",
        "report_generator"
    }
    try:
        if agent_name in AGENTS_TO_VALIDATE:
            # Remove confidence_score from output if present
            if "output" in payload and isinstance(payload["output"], dict) and "confidence_score" in payload["output"]:
                del payload["output"]["confidence_score"]
            
            # Special handling for personal_info_parser to ensure correct key capitalization
            if agent_name == "personal_info_parser" and "output" in payload and isinstance(payload["output"], dict) and "contact_details" in payload["output"]:
                contact_details = payload["output"]["contact_details"]
                if isinstance(contact_details, dict):
                    formatted_contacts = {}
                    for k, v in contact_details.items():
                        if k.lower() == "email":
                            formatted_contacts["Email"] = v
                        elif k.lower() == "phone":
                            formatted_contacts["Phone"] = v
                        elif k.lower() == "location":
                            formatted_contacts["Location"] = v
                        else:
                            formatted_contacts[k] = v
                    payload["output"]["contact_details"] = formatted_contacts
            
            # Validate callback data
            callback_validator.validate_callback(agent_name, payload)
            # Store locally in strict JSON format
            await asyncio.to_thread(callback_storage.store_callback, agent_name, uid, payload)
        # else: skip validation/storage for other agents
    except Exception as e:
        log.error(f"Callback validation/storage failed for {agent_name}: {e}")
        return  # Do not send/store invalid callback

    # Write callback data to file for debugging (offloaded to thread — sync FS I/O)
    asyncio.get_event_loop().run_in_executor(None, write_callback_to_file, uid, callback_url, payload, "attempting")
    
    # Validate callback URL format and whitelist
    is_valid, error_msg = validate_callback_url(callback_url)
    if not is_valid:
        log.error(f"Invalid callback URL: {error_msg} | url={callback_url} | uid={uid}")
        asyncio.get_event_loop().run_in_executor(None, write_callback_to_file, uid, callback_url, payload, "invalid_url")
        return
    
    # Check for potential UID typos in callback URL
    if uid in callback_url and len(uid) > 10:  # Only check for longer UIDs
        # Extract UID from callback URL if present
        import re
        url_uid_match = re.search(r'/([a-zA-Z0-9]{20,})/', callback_url)
        if url_uid_match:
            url_uid = url_uid_match.group(1)
            if url_uid != uid:
                log.warning(f"UID mismatch: request_uid={uid}, callback_url_uid={url_uid} | url={callback_url}")
    
    # Check if callbacks are disabled for this URL due to repeated failures (thread-safe)
    if await is_callback_disabled(callback_url):
        log.warning(f"Callbacks disabled for {callback_url} due to repeated failures | uid={uid}")
        return
    
    try:
        token = await get_cached_token(uid)   # ✅ use cached token
        if not token:
            # Still send callback without auth so receivers that don't require auth get the payload
            log.warning(f"No token available for callback | uid={uid} | sending without Authorization header")
        headers = {
            "Content-Type": "application/json"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async def fire_and_forget():
            log.info(f"Entry point fire_and_forget | uid={uid}")
            
            # Retry configuration
            max_retries = 3
            retry_delay = 2.0
            
            # Use shared HTTP client for better connection pooling
            client = await get_http_client()
            
            for attempt in range(max_retries):
                try:
                    response = await client.patch(callback_url, json=payload, headers=headers)
                    log.info(f"Callback response status: {response.status_code} | uid={uid} | attempt={attempt + 1}")
                    
                    if response.status_code < 500:  # Success or client error (don't retry)
                        log.info(f"Exit point #1 fire_and_forget | Callback sent successfully | uid={uid}")
                        # Reset failure count on success (thread-safe)
                        await reset_callback_failure_count(callback_url)
                        await asyncio.to_thread(write_callback_to_file, uid, callback_url, payload, f"success_{response.status_code}")
                        return
                    else:
                        log.warning(f"Server error {response.status_code}, will retry | uid={uid} | attempt={attempt + 1}")
                        
                except httpx.ConnectTimeout as e:
                    log.warning(f"Connection timeout for callback | uid={uid}, url={callback_url} | attempt={attempt + 1}")
                    if attempt == max_retries - 1:
                        error_log.warning(f"ConnectTimeout after {max_retries} attempts: {str(e)}")
                except httpx.TimeoutException as e:
                    log.warning(f"Request timeout for callback | uid={uid}, url={callback_url} | attempt={attempt + 1}")
                    if attempt == max_retries - 1:
                        error_log.warning(f"TimeoutException after {max_retries} attempts: {str(e)}")
                except httpx.ConnectError as e:
                    log.warning(f"Connection error for callback | uid={uid}, url={callback_url} | attempt={attempt + 1}")
                    if attempt == max_retries - 1:
                        error_log.warning(f"ConnectError after {max_retries} attempts: {str(e)}")
                except Exception as e:
                    log.error(f"Unexpected error in callback | uid={uid} | attempt={attempt + 1} | error={str(e)}")
                    if attempt == max_retries - 1:
                        error_log.error(traceback.format_exc())
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
            
            log.error(f"Failed to send callback after {max_retries} attempts | uid={uid}")
            # Track failure (thread-safe)
            new_count = await increment_callback_failure(callback_url)
            log.warning(f"Callback failure count for {callback_url}: {new_count}")
            await asyncio.to_thread(write_callback_to_file, uid, callback_url, payload, f"failed_after_{max_retries}_attempts")

        schedule_background_task(fire_and_forget(), "send_to_callback_fire_and_forget")
        log.info(f"Exit point #1 send_to_callback | Task created | uid={uid}")

    except Exception as e:
        log.error(f"Exit point #2 send_to_callback | Exception | uid={uid}, error={str(e)}")
        error_log.error(traceback.format_exc())
        await asyncio.to_thread(write_callback_to_file, uid, callback_url, payload, f"exception_{str(e)}")



from fastapi import FastAPI, Request, HTTPException

# Mock streaming for development environment
def _extract_text_from_pdf(binary_content: bytes) -> str:
    """Extract text from PDF binary content using Gemini."""
    try:
        from utils.resume_utils import _extract_text_with_gemini
        md_text = _extract_text_with_gemini(binary_content, file_type="pdf")
        if md_text and len(md_text.strip()) > 0:
            return md_text.strip()
        log.warning("⚠️ Gemini PDF extraction returned empty content")
        return ""
    except Exception as e:
        log.error(f"❌ Gemini PDF extraction failed: {e}")
        return ""

def _extract_text_from_docx(binary_content: bytes) -> str:
    """Extract text from DOCX binary content using Gemini."""
    try:
        from utils.resume_utils import _extract_text_with_gemini
        md_text = _extract_text_with_gemini(binary_content, file_type="docx")
        if md_text and len(md_text.strip()) > 0:
            return md_text.strip()
        log.warning("⚠️ Gemini DOCX extraction returned empty content")
        return ""
    except Exception as e:
        log.error(f"❌ Gemini DOCX extraction failed: {e}")
        return ""


def _parse_certificate_from_text(text: str) -> Dict[str, Any]:
    """Parse certification_name, issuing_organization, year from certificate document text."""
    import re
    if not text or not text.strip():
        return {"certification_name": "Unknown Certification", "issuing_organization": "", "year": None}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    name = (lines[0][:300] if lines else text[:300].strip()) or "Unknown Certification"
    year = None
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if year_match:
        year = year_match.group(1)
    issuer = (lines[1][:200] if len(lines) > 1 else "") or ""
    return {"certification_name": name, "issuing_organization": issuer, "year": year}


def _extract_text_from_rtf(binary_content: bytes) -> str:
    """Extract text from RTF binary content."""
    try:
        import docx2txt
        import io
        
        rtf_file = io.BytesIO(binary_content)
        text = docx2txt.process(rtf_file)
        
        return text.strip()
    except Exception as e:
        log.warning(f"Error extracting text from RTF: {e}")
        return ""

def _extract_text_with_encodings(binary_content: bytes) -> str:
    """Try to extract text using different encodings."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            text = binary_content.decode(encoding)
            if text and len(text.strip()) > 10:  # Basic validation
                return text.strip()
        except UnicodeDecodeError:
            continue
    
    return ""

async def _extract_resume_text_from_url(resume_url: str) -> str:
    """Extract actual resume text from URL for validation in development mode."""
    import aiohttp
    import asyncio
    from urllib.parse import urlparse
    
    try:
        # Check if it's a valid URL
        parsed_url = urlparse(resume_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return ""
        
        # For development, we'll try to fetch the content
        # In production, this would use proper document parsing
        async with aiohttp.ClientSession() as session:
            async with session.get(resume_url, timeout=10) as response:
                if response.status == 200:
                    # Get content type to determine how to decode
                    content_type = response.headers.get('content-type', '').lower()
                    
                    if 'text/' in content_type or 'application/json' in content_type:
                        # Try to decode as text
                        try:
                            content = await response.text()
                            return content
                        except UnicodeDecodeError:
                            log.warning(f"Failed to decode text content from URL: {resume_url}")
                            return ""
                    else:
                        # For binary content (PDF, DOC, etc.), extract actual text
                        log.info(f"Binary content detected ({content_type}), extracting text from document")
                        try:
                            # Get the binary content
                            binary_content = await response.read()
                            
                            # Extract text based on content type and URL extension
                            url_without_query = resume_url.split('?')[0].lower()
                            if content_type == 'application/pdf' or url_without_query.endswith('.pdf'):
                                text = _extract_text_from_pdf(binary_content)
                            elif 'msword' in content_type or 'wordprocessingml' in content_type or url_without_query.endswith('.docx'):
                                text = _extract_text_from_docx(binary_content)
                            elif url_without_query.endswith('.doc'):
                                # Handle .doc files using the same logic as resume processing
                                from utils.resume_utils import _extract_text_from_doc
                                text = _extract_text_from_doc(binary_content)
                            elif 'text/rtf' in content_type:
                                text = _extract_text_from_rtf(binary_content)
                            else:
                                # Try to decode as text with different encodings
                                text = _extract_text_with_encodings(binary_content)
                            
                            if text and len(text.strip()) > 50:  # Ensure we got meaningful text
                                log.info(f"Successfully extracted {len(text)} characters from document")
                                return text
                            else:
                                log.warning(f"Failed to extract meaningful text from document")
                                return ""
                                
                        except Exception as e:
                            log.warning(f"Error extracting text from binary content: {e}")
                            return ""
                else:
                    log.warning(f"Failed to fetch resume from URL: {response.status}")
                    return ""
    except Exception as e:
        log.warning(f"Error extracting resume text from URL: {e}")
        return ""


async def mock_stream_and_callback_events(state: dict, send, config: dict | None = None):
    """Hybrid streaming events for development environment - real validation, mock everything else."""
    uid = state.get("uid", "unknown")
    tenant_id = state.get("tenant_id", "unknown")
    body = state.get("body", {})
    
    log.info(f"🔧 DEVELOPMENT MODE: Using real validation agents and mock responses for other agents")
    
    try:
        # First, run real validation agents and check results
        resume_valid = True
        jd_valid = True
        
        # Check resume validation if resume_url is provided
        if body.get("resume_url"):
            log.info(f"🔧 DEVELOPMENT: Running real validate_resume agent")
            try:
                from agents.validate_resume import validate_resume_agent
                
                # Extract resume text from URL for real validation
                resume_url = body.get("resume_url", "")
                resume_text = await _extract_resume_text_from_url(resume_url)
                state_with_text = {**state, "resume_text": resume_text}
                
                real_output = await validate_resume_agent(state_with_text)
                # Extract boolean value from validation response
                if isinstance(real_output, dict):
                    resume_valid = real_output.get("is_valid_resume", False)
                
                # Send real validation result
                await send({
                    "status": "completed",
                    "node": "is_valid_resume",
                    "output": {
                        "is_valid_resume": resume_valid
                    }
                })
                
                log.info(f"🔧 DEVELOPMENT: Resume validation result: {resume_valid}")
                
            except Exception as e:
                log.error(f"Real validate_resume agent failed: {e}")
                resume_valid = False
                await send({
                    "status": "completed",
                    "node": "is_valid_resume",
                    "output": {
                        "is_valid_resume": False
                    }
                })
        
        # Check JD validation if jd_url is provided
        if body.get("jd_url"):
            log.info(f"🔧 DEVELOPMENT: Running real validate_jd agent")
            try:
                from agents.validate_jd import validate_jd_agent
                
                # Extract JD text from URL for real validation
                jd_url = body.get("jd_url", "")
                jd_text = await _extract_resume_text_from_url(jd_url)  # Reuse the same function
                state_with_text = {**state, "jd_text": jd_text}
                
                real_output = await validate_jd_agent(state_with_text)
                # Extract boolean value from validation response
                if isinstance(real_output, dict):
                    jd_valid = real_output.get("is_valid_jd", False)
                
                # Get job_id using the same logic as mock system
                job_id = state.get("job_id", str(uuid.uuid4()))
                
                # Send real validation result
                await send({
                    "status": "completed",
                    "node": "is_valid_jd",
                    "output": {
                        "is_valid_jd": jd_valid,
                        "job_id": job_id
                    }
                })
                
                log.info(f"🔧 DEVELOPMENT: JD validation result: {jd_valid}")
                
            except Exception as e:
                log.error(f"Real validate_jd agent failed: {e}")
                jd_valid = False
                # Get job_id using the same logic as mock system
                job_id = state.get("job_id", str(uuid.uuid4()))
                
                await send({
                    "status": "completed",
                    "node": "is_valid_jd",
                    "output": {
                        "is_valid_jd": False,
                        "job_id": job_id
                    }
                })
        
        # Now determine which agents to run based on REAL validation results
        agents_to_run = []
        
        # Resume flow - only if resume is valid
        if body.get("resume_url") and resume_valid:
            agents_to_run.extend([
                "groq_resume_parser",  # Replaces all four parsing agents
                "resume_scorer",
                "interest_filler",
                "career_advisor",
                "market_and_course_recommender",
                "assessment_recommender"
            ])
        elif body.get("resume_url") and not resume_valid:
            log.info(f"🔧 DEVELOPMENT: Resume validation failed, skipping parsing agents")
        
        # JD flow - only if JD is valid (includes job_details corporate flow)
        if (body.get("jd_url") or body.get("job_details")) and jd_valid:
            agents_to_run.extend([
                "job_description_parser",
                "prescreening_questions",
                "ranker"
            ])
        elif body.get("jd_url") and not jd_valid:
            log.info(f"🔧 DEVELOPMENT: JD validation failed, skipping parsing agents")
        
        # Assessment question generation
        if body.get("plan"):
            agents_to_run.append("assessment_question_generator")
        
        # Assessment evaluation
        if body.get("submission"):
            agents_to_run.extend(["assessment_evaluator", "report_generator"])
        
        # Notification
        if body.get("user_mail") or body.get("email"):
            agents_to_run.append("notification_agent")
        
        # Run mock agents for the determined list
        for node_name in agents_to_run:
            if node_name in mock_generator.responses:
                log.info(f"🔧 MOCK: Simulating {node_name} completion")
                mock_output = mock_generator._customize_response(node_name, mock_generator.responses[node_name], state)
                
                await send({
                    "status": "completed",
                    "node": node_name,
                    "output": mock_output
                })
                
                # Update session step if session exists
                session_id = state.get("session_id", "")
                if session_id:
                    try:
                        progress = get_node_progress(node_name)
                        await asyncio.to_thread(update_session_step, session_id, node_name, {"status": "completed"}, progress)
                    except Exception as e:
                        log.warning(f"Failed to update session step for {node_name}: {e}")
        
        # Send final completion
        await send({"status": "completed", "node": "pipeline", "message": "Development processing completed"})
        
    except Exception as e:
        log.error(f"Development streaming failed: {e}")
        await send({
            "status": "error",
            "message": f"Development processing failed: {str(e)}"
        })

# Helpers for streaming & events
async def stream_and_callback_events(state: dict, send, config: dict | None = None):
    """Stream events from the graph and forward them to callback."""
    config = config or {"recursion_limit": GRAPH_RECURSION_LIMIT}
    corr_id = get_correlation_id()
    uid = state.get("uid", "unknown")
    tenant_id = state.get("tenant_id", "unknown")
    body = state.get("body", {})
    has_resume_url = bool(body.get("resume_url"))
    has_plan = bool(body.get("plan"))
    has_submission = bool(body.get("submission"))
    has_jd = bool(body.get("jd_url"))

    def _filter_output(output: dict) -> dict:
        if not isinstance(output, dict):
            return output
        # Apply filtering if body contains resume_url, plan, submission, or jd_url
        should_filter = has_resume_url or has_plan or has_submission or has_jd
        if not should_filter:
            return output
        # Remove JD-related fields unless the request explicitly provided jd_url
        # Ensure drop_keys is always a set, never a dict (defensive programming)
        drop_keys = set()
        if not has_jd:
            drop_keys.update({"jd_url", "jd_text", "is_valid_jd"})
        # Remove resume-related fields unless the request explicitly provided resume_url
        if not has_resume_url:
            # In evaluation flow (has_submission), keep raw_skill_gap_analysis_output
            resume_related = {"resume_url", "resume_text", "is_valid_resume", "structured_resume", "user_interests"}
            if not has_submission:
                resume_related.add("raw_skill_gap_analysis_output")
            drop_keys.update(resume_related)
        # Only show assessment-related fields if answers (submission) are provided
        if not has_submission:
            drop_keys.update({"generated_questions", "report"})
        # Don't filter assessment-related fields when plan is provided (for question generation)
        if has_plan:
            # Ensure drop_keys is still a set before calling discard (defensive check)
            if not isinstance(drop_keys, set):
                # If somehow drop_keys is not a set, recreate it as a set
                drop_keys = set(drop_keys) if drop_keys else set()
            drop_keys.discard("assessment_plan")
            drop_keys.discard("updated_plan")
            drop_keys.discard("generation_meta")
            drop_keys.discard("assessment_id")
            drop_keys.discard("assessment_topic")
            drop_keys.discard("generated_questions")  # Keep generated_questions for question generation
        # Always include assessment_plan in evaluation flow (has_submission) - updated plan should be sent to callback
        if has_submission:
            # Ensure drop_keys is still a set before calling discard (defensive check)
            if not isinstance(drop_keys, set):
                # If somehow drop_keys is not a set, recreate it as a set
                drop_keys = set(drop_keys) if drop_keys else set()
            drop_keys.discard("assessment_plan")
            drop_keys.discard("assessment_needs")
        # Final safety check before using drop_keys in comprehension
        if not isinstance(drop_keys, set):
            drop_keys = set(drop_keys) if drop_keys else set()
        return {k: v for k, v in output.items() if k not in drop_keys}

    _pipeline_stream_start = None
    try:
        # Use graph streaming to capture real node execution events
        result = None
        # Wall-clock for PIPELINE_FLOW_COMPLETE (astream bypasses production_ainvoke wrapper)
        _pipeline_stream_start = time.time()
        session_id = state.get("session_id", "")
        
        # ✅ FIX: Track which nodes have already sent callbacks to avoid duplicates
        nodes_that_sent_callbacks = set()
        
        # 🔍 DEBUG: Graph execution start
        log.debug(f"🔍 GRAPH_DEBUG: Starting graph execution with state keys: {list(state.keys())}")
        log.debug(f"🔍 GRAPH_DEBUG: State session_id: {state.get('session_id')}")
        log.debug(f"🔍 GRAPH_DEBUG: State next: {state.get('next')}")
        
        # Stream the graph execution with max_concurrency config + Langfuse tracing
        # handler_ref captures the CallbackHandler so we can read trace_id for scoring/callbacks
        handler_ref = []
        graph_config = {"max_concurrency": get_max_concurrency()}
        graph_config = merge_langfuse_into_config(
            graph_config,
            settings,
            session_id=state.get("session_id") or "",
            user_id=state.get("uid") or "",
            handler_out=handler_ref,
        )
        cb_count = len(graph_config.get("callbacks") or [])
        log.info("Langfuse callbacks in graph_config: %d", cb_count)
        if cb_count == 0:
            reason = get_langfuse_failure_reason(settings)
            log.warning(
                "Langfuse traces disabled: %s. Run uvicorn with venv Python: ./venv/bin/uvicorn app:app --reload --port 8000",
                reason or "unknown",
            )

        def _payload_with_trace_id(payload: dict) -> dict:
            tid = get_trace_id_from_handler(handler_ref)
            if tid:
                return {**payload, "langfuse_trace_id": tid}
            return payload

        async for event in graph.astream(state, config=graph_config):
            # 🔍 DEBUG: Event received
            log.debug(f"🔍 GRAPH_DEBUG: Received event with keys: {list(event.keys())}")
            
            # Process each node execution event
            for node_name, node_output in event.items():
                if node_name == "__start__" or node_name == "__end__":
                    log.debug(f"🔍 GRAPH_DEBUG: Skipping {node_name} event")
                    continue
                
                # ✅ OPTIMIZATION: Check if node wants to override callback node name
                # This allows parallel nodes to report as a different node name in callbacks
                # MUST be done early, before any logging or callback sending
                original_node_name = node_name
                callback_node_name_override = None
                if isinstance(node_output, dict):
                    # Check for override field in node output (directly in the dict)
                    callback_node_name_override = node_output.get("_callback_node_name")
                    if callback_node_name_override:
                        log.info(f"🔄 Found _callback_node_name override: '{callback_node_name_override}' for node '{node_name}'")
                
                if callback_node_name_override and callback_node_name_override != node_name:
                    log.info(f"🔄 Overriding callback node name from '{node_name}' to '{callback_node_name_override}'")
                    node_name = callback_node_name_override
                
                # 🔍 DEBUG: Node execution
                log.debug(f"🔍 GRAPH_DEBUG: Processing node: {node_name}")
                log.debug(f"🔍 GRAPH_DEBUG: Node output type: {type(node_output)}")
                if isinstance(node_output, dict):
                    log.debug(f"🔍 GRAPH_DEBUG: Node output keys: {list(node_output.keys())}")
                    log.debug(f"🔍 GRAPH_DEBUG: Node output next: {node_output.get('next')}")
                    log.debug(f"🔍 GRAPH_DEBUG: Node output session_id: {node_output.get('session_id')}")
                    
                # Log business event for node completion
                log_business_event(
                    event="node_completed",
                    agent=node_name,
                    node=node_name,
                    uid=uid,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    progress=get_node_progress(node_name),
                    status="completed",
                    correlation_id=corr_id
                )

                # Log node output to terminal for real-time visibility
                # OPTIMIZATION: Skip printing for internal context nodes (resume_assembler, resume_summary)
                if original_node_name not in ["resume_assembler", "resume_summary"]:
                    try:
                        serializable = make_serializable(node_output)
                        # Skip filtering for "end" and "dispatcher" nodes to show all fields (including "next")
                        if node_name not in ["end", "dispatcher"]:
                            # Special handling for resume_content_generator - only show relevant fields
                            if node_name == "resume_content_generator":
                                clean_serializable = {}
                                if isinstance(serializable, dict):
                                    # Include only the essential fields for logging
                                    for key in ["resume_content", "success", "processing_time_seconds", "context_used", "uid", "tenant_id", "session_id"]:
                                        if key in serializable:
                                            clean_serializable[key] = serializable[key]
                                serializable = clean_serializable
                            elif node_name == "enhanced_role_fit":
                                # Log only role fit output, not full state
                                clean_serializable = {}
                                if isinstance(serializable, dict):
                                    clean_serializable["enhanced_role_fit"] = serializable.get("enhanced_role_fit", [])
                                    clean_serializable["roles_count"] = len(serializable.get("enhanced_role_fit") or [])
                                serializable = clean_serializable
                            elif node_name == "prescreening_questions":
                                # Log only prescreening output, not full state
                                clean_serializable = {}
                                if isinstance(serializable, dict):
                                    qs = serializable.get("prescreening_questions") or serializable.get("generated_questions")
                                    clean_serializable["prescreening_questions_count"] = len(qs) if isinstance(qs, list) else 0
                                    for key in ("job_id", "processing_time_seconds", "next", "uid", "tenant_id", "session_id"):
                                        if key in serializable:
                                            clean_serializable[key] = serializable[key]
                                serializable = clean_serializable
                            elif node_name == "job_matcher":
                                # Log only callback-needed fields, not full state (resume, skills, assessment_plan, etc.)
                                clean_serializable = {}
                                if isinstance(serializable, dict):
                                    for key in (
                                        "job_matcher_status", "total_matches_found", "top_matches",
                                        "candidate_job_match_result", "compare_job_only", "processing_time_seconds",
                                        "partial_timeout", "jobs_analyzed", "jobs_total",
                                        "next", "uid", "tenant_id", "session_id"
                                    ):
                                        if key in serializable:
                                            clean_serializable[key] = serializable[key]
                                serializable = clean_serializable
                            elif node_name == "job_matcher_preprocessor":
                                # Log only job_ids and count, not full job_docs (JD contents)
                                clean_serializable = {}
                                if isinstance(serializable, dict):
                                    job_ids = serializable.get("job_ids", [])
                                    clean_serializable["job_ids_count"] = len(job_ids)
                                    clean_serializable["job_ids"] = job_ids
                                    for key in ("next", "uid", "tenant_id", "session_id"):
                                        if key in serializable:
                                            clean_serializable[key] = serializable[key]
                                serializable = clean_serializable
                            elif node_name in ("career_advisor_entry", "career_flow_fan_out"):
                                # Passthrough nodes return full state — log only a minimal summary
                                clean_serializable = {}
                                if isinstance(serializable, dict):
                                    for key in ("_compare_flow_multi_job_run", "_job_matcher_parallel", "next", "uid", "tenant_id", "session_id"):
                                        if key in serializable:
                                            clean_serializable[key] = serializable[key]
                                    if node_name == "career_advisor_entry":
                                        clean_serializable["_summary"] = "passthrough (routes to enhanced_role_fit)"
                                    else:
                                        clean_serializable["_summary"] = "passthrough (fan-out to career_advisor + job_matcher_preprocessor)"
                                serializable = clean_serializable
                            elif node_name == "assessment_validator":
                                # Agent returns **state — log only validation fields, not full resume/skills/education
                                clean_serializable = {}
                                if isinstance(serializable, dict):
                                    for key in (
                                        "status", "message", "validation_results", "overall_success",
                                        "processing_time", "processing_time_seconds",
                                        "next", "uid", "tenant_id", "session_id",
                                    ):
                                        if key in serializable:
                                            clean_serializable[key] = serializable[key]
                                serializable = clean_serializable
                            else:
                                serializable = _filter_output(serializable)
                        compact = json.dumps(serializable, ensure_ascii=False)
                        if len(compact) > _STREAM_NODE_LOG_MAX_CHARS:
                            pretty = (
                                compact[:_STREAM_NODE_LOG_MAX_CHARS]
                                + f"\n... [STREAM_NODE_LOG truncated: {len(compact)} chars total; set STREAM_NODE_LOG_MAX_CHARS to raise limit]"
                            )
                        else:
                            pretty = json.dumps(serializable, indent=2, ensure_ascii=False)
                        _log_node_visible(f"\n===== NODE COMPLETED: {node_name} =====")
                        _log_node_visible(pretty)
                        _log_node_visible(f"===== END NODE: {node_name} =====\n")
                        # Log generic processing time for any node if provided by middleware/agent
                        try:
                            if isinstance(serializable, dict):
                                timing = serializable.get("_node_timing")
                                if isinstance(timing, dict) and "processing_time_seconds" in timing:
                                    _log_node_visible(f"Time taken ({node_name}): {round(float(timing.get('processing_time_seconds')), 2)}s")
                                else:
                                    # Fallback: some agents may surface it at top-level
                                    pt = serializable.get("processing_time_seconds")
                                    if pt is not None:
                                        _log_node_visible(f"Time taken ({node_name}): {round(float(pt), 2)}s")
                        except Exception:
                            pass
                    except Exception as _e:
                        log.error(f"\n===== NODE COMPLETED: {node_name} (non-serializable output) =====")
                        log.error(str(node_output))
                        log.error(f"===== END NODE: {node_name} =====\n")
                        # Best-effort timing log if middleware injected it and output is dict-like
                        try:
                            if isinstance(node_output, dict):
                                timing = node_output.get("_node_timing")
                                if isinstance(timing, dict) and "processing_time_seconds" in timing:
                                    _log_node_visible(f"Time taken ({node_name}): {round(float(timing.get('processing_time_seconds')), 2)}s")
                                else:
                                    pt = node_output.get("processing_time_seconds")
                                    if pt is not None:
                                        _log_node_visible(f"Time taken ({node_name}): {round(float(pt), 2)}s")
                        except Exception:
                            pass

                

                # Fan-in guard: skip callback if this is a no-op invocation
                # (e.g. assessment_validator triggered by only one of two parallel branches)
                if isinstance(node_output, dict) and node_output.get("_skip_callback"):
                    log.info(f"⏳ Skipping callback for {node_name} (fan-in guard — waiting for other branch)")
                    result = node_output
                    continue

                # Send completion status for this node with structured output
                # ✅ FIX: Add error handling around serialization to ensure callbacks are always sent
                try:
                    filtered_output = _filter_output(make_serializable(node_output))
                except Exception as e:
                    log.error(f"❌ Failed to serialize output for {node_name}: {e}", exc_info=True)
                    # Fallback: send minimal output with error indication
                    filtered_output = {
                        "error": "Failed to serialize output",
                        "node": node_name,
                        "message": f"Serialization error: {str(e)}"
                    }
                
                # ✅ OPTIMIZATION: Remove internal callback node name override field from output
                if isinstance(filtered_output, dict) and "_callback_node_name" in filtered_output:
                    filtered_output = {k: v for k, v in filtered_output.items() if k != "_callback_node_name"}
                
                # For assessment_question_generator, only keep generated_questions and assessment_id in output
                if node_name == "assessment_question_generator":
                    # Extract only generated_questions from the filtered output
                    clean_output = {}
                    if isinstance(filtered_output, dict) and "generated_questions" in filtered_output:
                        clean_output["generated_questions"] = filtered_output["generated_questions"]
                    # Try to include assessment_id from node output or request body
                    try:
                        assessment_id = None
                        if isinstance(filtered_output, dict):
                            assessment_id = filtered_output.get("assessment_id")
                        if not assessment_id:
                            assessment_id = body.get("assessment_id")
                        if assessment_id:
                            clean_output["assessment_id"] = assessment_id
                    except Exception:
                        pass
                    
                    clean_payload = {
                        "status": "completed",
                        "node": node_name,
                        "output": clean_output
                    }
                    await send(clean_payload)
                    # Track that this node sent a callback
                    nodes_that_sent_callbacks.add(node_name)
                elif node_name == "experience_parser":
                    # Normalize projects/extras to objects for schema compliance
                    clean_output = {}
                    if isinstance(filtered_output, dict):
                        if "work_experience" in filtered_output:
                            clean_output["work_experience"] = filtered_output["work_experience"]
                        if "certifications" in filtered_output:
                            clean_output["certifications"] = filtered_output["certifications"]
                        if "projects" in filtered_output:
                            pv = filtered_output["projects"]
                            if isinstance(pv, list):
                                clean_output["projects"] = {"items": pv}
                            elif isinstance(pv, dict):
                                clean_output["projects"] = pv
                        if "extras" in filtered_output:
                            ev = filtered_output["extras"]
                            if isinstance(ev, list):
                                clean_output["extras"] = {"other": ev}
                            elif isinstance(ev, dict):
                                clean_output["extras"] = ev
                    await send({
                        "status": "completed",
                        "node": node_name,
                        "output": clean_output
                    })
                
                elif node_name == "prescreening_questions":
                    # Corporate JD flow: send prescreening_questions to callback
                    clean_output = {}
                    if isinstance(filtered_output, dict):
                        qs = filtered_output.get("prescreening_questions")
                        if qs is None and "job_description" in filtered_output:
                            jd = filtered_output["job_description"]
                            if isinstance(jd, dict):
                                qs = jd.get("prescreening_questions")
                        if qs is not None:
                            clean_output["prescreening_questions"] = qs
                        if "job_id" in filtered_output:
                            clean_output["job_id"] = filtered_output["job_id"]
                        if "job_description" in filtered_output:
                            jd = filtered_output["job_description"]
                            clean_output["job_description"] = (
                                {"jobTitle": jd.get("jobTitle"), "company": jd.get("company")}
                                if isinstance(jd, dict) else jd
                            )
                        if "processing_time_seconds" in filtered_output:
                            clean_output["processing_time_seconds"] = filtered_output["processing_time_seconds"]
                    await send({
                        "status": "completed",
                        "node": node_name,
                        "output": clean_output
                    })
                    nodes_that_sent_callbacks.add(node_name)
                elif node_name == "report_generator":
                    # Ensure assessment_id and question_doc_id are present in output if available
                    clean_output = {}
                    if isinstance(filtered_output, dict):
                        clean_output.update(filtered_output)
                    try:
                        if isinstance(filtered_output, dict):
                            aid = filtered_output.get("assessment_id")
                            qid = filtered_output.get("question_doc_id")
                        else:
                            aid = None
                            qid = None
                        if not aid:
                            aid = body.get("assessment_id")
                        if not qid:
                            qid = body.get("question_doc_id")
                        if aid is not None:
                            clean_output["assessment_id"] = aid
                        if qid is not None:
                            clean_output["question_doc_id"] = qid
                    except Exception:
                        pass
                    await send({
                        "status": "completed",
                        "node": node_name,
                        "output": clean_output
                    })
                    # Track that this node sent a callback
                    nodes_that_sent_callbacks.add(node_name)
                elif node_name == "resume_content_generator":
                    # Only send relevant fields for resume content generator
                    clean_output = {}
                    if isinstance(filtered_output, dict):
                        # Include only the essential fields
                        if "resume_content" in filtered_output:
                            clean_output["resume_content"] = filtered_output["resume_content"]
                        if "success" in filtered_output:
                            clean_output["success"] = filtered_output["success"]
                        if "processing_time_seconds" in filtered_output:
                            clean_output["processing_time_seconds"] = filtered_output["processing_time_seconds"]
                        if "context_used" in filtered_output:
                            clean_output["context_used"] = filtered_output["context_used"]
                        # Include metadata fields if present
                        if "uid" in filtered_output:
                            clean_output["uid"] = filtered_output["uid"]
                        if "tenant_id" in filtered_output:
                            clean_output["tenant_id"] = filtered_output["tenant_id"]
                        if "session_id" in filtered_output:
                            clean_output["session_id"] = filtered_output["session_id"]
                    await send({
                        "status": "completed",
                        "node": node_name,
                        "output": clean_output
                    })
                    # Track that this node sent a callback
                    nodes_that_sent_callbacks.add(node_name)
                elif node_name == "enhanced_role_fit":
                    # Send clean payload with only enhanced_role_fit so clients see role fit clearly
                    clean_output = {}
                    if isinstance(filtered_output, dict) and "enhanced_role_fit" in filtered_output:
                        clean_output["enhanced_role_fit"] = filtered_output["enhanced_role_fit"]
                    else:
                        clean_output["enhanced_role_fit"] = []
                    await send({
                        "status": "completed",
                        "node": node_name,
                        "output": clean_output
                    })
                    nodes_that_sent_callbacks.add(node_name)
                elif node_name == "job_matcher":
                    # OPTIMIZATION: Send only job_matcher output — NOT full state (avoids duplicate education,
                    # work_experience, skills from merge_lists, and bloat from body/job_ids/structured_resume)
                    log.info(f"📤 Sending job_matcher callback for node: {node_name}")
                    clean_output = {}
                    if isinstance(filtered_output, dict):
                        for key in (
                            "matched_jobs", "top_matches", "candidate_job_match_result",
                            "total_matches_found", "job_matcher_status", "processing_time_seconds",
                            "compare_job_only", "partial_timeout", "jobs_analyzed", "jobs_total",
                            "next", "uid", "tenant_id", "session_id", "resume_url",
                            "endpoint_name", "request_type",
                        ):
                            if key in filtered_output:
                                clean_output[key] = filtered_output[key]
                    await send({
                        "status": "completed",
                        "node": node_name,
                        "output": clean_output
                    })
                    nodes_that_sent_callbacks.add(node_name)
                elif node_name == "job_matcher_preprocessor":
                    # Preprocessor: send only job_ids and metadata, never full job_docs (JD contents)
                    clean_output = {}
                    if isinstance(filtered_output, dict):
                        job_ids = filtered_output.get("job_ids", [])
                        clean_output["job_ids"] = job_ids
                        clean_output["job_ids_count"] = len(job_ids)
                        for key in ("next", "uid", "tenant_id", "session_id"):
                            if key in filtered_output:
                                clean_output[key] = filtered_output[key]
                    await send({
                        "status": "completed",
                        "node": node_name,
                        "output": clean_output
                    })
                    nodes_that_sent_callbacks.add(node_name)
                elif node_name == "assessment_validator":
                    # Agent merges **state into output — send only validation payload (avoid duplicate resume/skills/body)
                    log.info(f"📤 Sending assessment_validator callback (minimal output)")
                    clean_output = {}
                    if isinstance(filtered_output, dict):
                        for key in (
                            "status", "message", "validation_results", "overall_success",
                            "processing_time", "processing_time_seconds",
                            "next", "uid", "tenant_id", "session_id",
                        ):
                            if key in filtered_output:
                                clean_output[key] = filtered_output[key]
                    try:
                        await send({
                            "status": "completed",
                            "node": node_name,
                            "output": clean_output,
                        })
                        nodes_that_sent_callbacks.add(node_name)
                        log.info("✅ Successfully sent assessment_validator callback")
                    except Exception as e:
                        log.error(f"❌ assessment_validator callback failed: {e}", exc_info=True)
                elif node_name in [
                    "resume_assembler",
                    "resume_summary",
                    "career_advisor_entry",
                    "career_flow_fan_out",
                    "parallel_join_wait",
                    "parallel_job_matcher_done",
                ]:
                    # OPTIMIZATION: Passthrough/internal — no client callback (job_matcher_preprocessor handled above)
                    # parallel_join_wait: fan-in dead end after skipped assessment_validator join (other branch still running)
                    # parallel_job_matcher_done: parallel job_matcher finished before AR; validator only after assessment_recommender
                    # Do not send callbacks - they run silently in the background
                    if log.isEnabledFor(logging.DEBUG):
                        log.debug(f"Skipping callback for {node_name} (passthrough/internal, no client output)")
                    # No callback sent - node runs silently in background
                else:
                    # ✅ FIX: Add explicit logging for resume_scorer and groq_resume_parser callbacks
                    if node_name == "resume_scorer":
                        log.info(f"📤 Sending resume_scorer callback for node: {node_name}")
                    elif node_name == "groq_resume_parser":
                        log.info(f"📤 Sending groq_resume_parser callback for node: {node_name}")
                    
                    # ✅ FIX: Add error handling around callback sending to ensure errors are logged
                    try:
                        # Include full job_description (with jobTitle) and add top-level job_title for job_description_parser callbacks
                        callback_output = filtered_output if isinstance(filtered_output, dict) else {}
                        if isinstance(callback_output, dict) and "job_description" in callback_output:
                            job_desc = callback_output.get("job_description")
                            if isinstance(job_desc, dict):
                                job_title = job_desc.get("jobTitle") or job_desc.get("job_title") or job_desc.get("title") or ""
                                if job_title:
                                    callback_output = {**callback_output, "job_title": job_title}
                        
                        await send({
                            "status": "completed",
                            "node": node_name,
                            "output": callback_output
                        })
                        # Track that this node sent a callback
                        nodes_that_sent_callbacks.add(node_name)
                        if node_name == "resume_scorer":
                            log.info(f"✅ Successfully sent resume_scorer callback")
                        elif node_name == "groq_resume_parser":
                            log.info(f"✅ Successfully sent groq_resume_parser callback")
                    except Exception as e:
                        log.error(f"❌ Failed to send callback for {node_name}: {e}", exc_info=True)
                        # Log additional context for resume_scorer, job_matcher, and groq_resume_parser failures
                        if node_name == "resume_scorer":
                            log.error(f"❌ CRITICAL: resume_scorer callback failed - output may not be delivered to client")
                            log.error(f"   Output keys: {list(filtered_output.keys()) if isinstance(filtered_output, dict) else 'Not a dict'}")
                            log.error(f"   Output type: {type(filtered_output)}")
                        elif node_name == "groq_resume_parser":
                            log.error(f"❌ CRITICAL: groq_resume_parser callback failed - output may not be delivered to client")
                            log.error(f"   Output keys: {list(filtered_output.keys()) if isinstance(filtered_output, dict) else 'Not a dict'}")
                            log.error(f"   Output type: {type(filtered_output)}")
                        # Note: We don't re-raise - the graph execution should continue
                
                # Update session step if session exists (offloaded — sync Chroma I/O)
                if session_id:
                    try:
                        progress = get_node_progress(node_name)
                        step_data = {"status": "completed"}
                        if node_name == "prescreening_questions" and isinstance(node_output, dict) and "prescreening_questions" in node_output:
                            step_data["prescreening_questions"] = node_output["prescreening_questions"]
                        await asyncio.to_thread(update_session_step, session_id, node_name, step_data, progress)
                    except Exception as e:
                        log.warning(f"Failed to update session step for {node_name}: {e}")
                
                # Store the final result (last event contains the final state)
                result = node_output

        _stream_dur = time.time() - _pipeline_stream_start
        _flow_id, _flow_label = classify_pipeline_flow(state)
        _analysis_id_stream = ""
        if isinstance(result, dict):
            _analysis_id_stream = result.get("analysis_id") or ""
        if not _analysis_id_stream:
            _analysis_id_stream = state.get("analysis_id") or ""
        _ok_stream = result is not None and (
            not isinstance(result, dict) or bool(result.get("ok", True))
        )
        log_pipeline_flow_complete(
            _flow_id,
            _flow_label,
            _stream_dur,
            uid=uid,
            tenant_id=tenant_id,
            analysis_id=_analysis_id_stream,
            ok=_ok_stream,
            extra="path=astream",
        )

        # Debug logging to see what's in the final result
        log.info(f"🔍 Final Result Debug:")
        log.info(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        if isinstance(result, dict) and "job_description" in result:
            jd = result["job_description"]
            log.info(f"   Job Description found:")
            log.info(f"     Job Title: {jd.get('jobTitle', 'N/A')}")
            log.info(f"     Company: {jd.get('company', 'N/A')}")
            log.info(f"     Location: {jd.get('location', 'N/A')}")
            log.info(f"     Required Skills: {jd.get('requiredSkills', [])}")
        else:
            log.info(f"   No job_description found in result")
        
        # Send final result with appropriate node name based on flow type
        body = state.get("body", {})
        final_node = "analysis"  # default
        is_second_call = result.get("is_second_call", False) or (
            bool(body.get("user_interests")) and
            not body.get("resume_url") and
            not (body.get("resume_text") or "").strip()
        )

        # 2nd call (user_interests only): flow is career_advisor -> market -> assessment_recommender -> job_matcher -> assessment_validator
        if is_second_call:
            final_node = "assessment_validator"
        # JD enhance-only flow: user gets enhanced text and can send again for validation
        elif result.get("enhanced_jd_text") is not None:
            final_node = "end_enhanced_jd"
        elif body.get("job_details"):
            # Manual job description upload flow - always goes to ranker
            final_node = "ranker"
        elif body.get("jd_url"):
            # JD flow - check JD validation
            # If request provides jd_url, we end after job_description_parser (ranker is separate API call).
            if result.get("is_valid_jd") is False:
                # end_invalid_jd node event already emitted; skip duplicate final callback
                final_node = None
            else:
                final_node = "ranker"
        elif body.get("resume_url"):
            # Resume flow - check resume validation
            if result.get("is_valid_resume") is False:
                final_node = "end_invalid_resume"
            else:
                # ✅ FIX: If resume_scorer already sent a callback during graph execution, 
                # don't send duplicate final callback - the graph ended at "end" node
                if "resume_scorer" in nodes_that_sent_callbacks:
                    log.info(f"🔄 resume_scorer already sent callback during execution, skipping duplicate final callback")
                    # Set to None to indicate we should skip the final callback entirely
                    final_node = None
                else:
                    final_node = "resume_scorer"
        # elif body.get("resume_url"):
        #     final_node = "resume_scorer"
        # elif body.get("jd_url"):
        #     final_node = "ranker"
        elif body.get("submission"):
            final_node = "assessment_evaluator"
        elif body.get("plan"):
            final_node = "assessment_question_generator"
        elif body.get("user_mail") or body.get("email"):
            final_node = "notification_agent"
        # Check if this was a resume_content_generator flow (has resume_content but no resume_url/resume_text)
        elif result.get("resume_content") and not body.get("resume_url") and not body.get("resume_text"):
            final_node = "resume_content_generator"
        
        # ✅ FIX: Skip final callback entirely if final_node is None (indicates duplicate was already sent)
        if final_node is None:
            log.info(f"🔄 Skipping final callback entirely (duplicate already sent during execution)")
        else:
            # Skip filtering for end nodes to include all fields
            serializable_result = make_serializable(result)
            if final_node in ["end", "end_invalid_resume", "end_invalid_jd", "end_enhanced_jd"]:
                # For end nodes, check if resume_content_generator was the actual flow
                if result.get("resume_content") and not body.get("resume_url") and not body.get("resume_text"):
                    final_node = "resume_content_generator"
                filtered_result = serializable_result
            else:
                filtered_result = _filter_output(serializable_result)
            
            # Ensure proper schema compliance for all final results
            if final_node == "assessment_question_generator":
                # For assessment_question_generator, only keep generated_questions and assessment_id in output
                clean_output = {}
                if isinstance(filtered_result, dict) and "generated_questions" in filtered_result:
                    clean_output["generated_questions"] = filtered_result["generated_questions"]
                # Try to include assessment_id from result or request body
                try:
                    assessment_id = None
                    if isinstance(filtered_result, dict):
                        assessment_id = filtered_result.get("assessment_id")
                    if not assessment_id:
                        assessment_id = body.get("assessment_id")
                    if assessment_id:
                        clean_output["assessment_id"] = assessment_id
                except Exception:
                    pass

                # ✅ FIX: Skip if already sent during execution
                if final_node in nodes_that_sent_callbacks:
                    log.info(f"🔄 Skipping duplicate final callback for {final_node} (already sent during execution)")
                else:
                    await send(_payload_with_trace_id({
                        "status": "completed",
                        "node": final_node,
                        "output": clean_output,
                    }))
            elif final_node == "resume_content_generator":
                # Only send relevant fields for resume content generator
                clean_output = {}
                if isinstance(filtered_result, dict):
                    # Include only the essential fields
                    if "resume_content" in filtered_result:
                        clean_output["resume_content"] = filtered_result["resume_content"]
                    if "success" in filtered_result:
                        clean_output["success"] = filtered_result["success"]
                    if "processing_time_seconds" in filtered_result:
                        clean_output["processing_time_seconds"] = filtered_result["processing_time_seconds"]
                    if "context_used" in filtered_result:
                        clean_output["context_used"] = filtered_result["context_used"]
                    # Include metadata fields if present
                    if "uid" in filtered_result:
                        clean_output["uid"] = filtered_result["uid"]
                    if "tenant_id" in filtered_result:
                        clean_output["tenant_id"] = filtered_result["tenant_id"]
                    if "session_id" in filtered_result:
                        clean_output["session_id"] = filtered_result["session_id"]
                # ✅ FIX: Skip if already sent during execution
                if final_node in nodes_that_sent_callbacks:
                    log.info(f"🔄 Skipping duplicate final callback for {final_node} (already sent during execution)")
                else:
                    await send(_payload_with_trace_id({
                        "status": "completed",
                        "node": final_node,
                        "output": clean_output,
                    }))
            elif final_node == "end_enhanced_jd":
                # JD enhance-only: return enhanced text so user can send again for validation
                clean_output = {}
                if isinstance(filtered_result, dict):
                    if "enhanced_jd_text" in filtered_result:
                        clean_output["enhanced_jd_text"] = filtered_result["enhanced_jd_text"]
                    if "jd_text" in filtered_result:
                        clean_output["jd_text"] = filtered_result["jd_text"]
                    if "processing_time_seconds" in filtered_result:
                        clean_output["processing_time_seconds"] = filtered_result["processing_time_seconds"]
                    if "ok" in filtered_result:
                        clean_output["ok"] = filtered_result["ok"]
                    if "method_explain" in filtered_result:
                        clean_output["method_explain"] = filtered_result["method_explain"]
                if final_node in nodes_that_sent_callbacks:
                    log.info(f"🔄 Skipping duplicate final callback for {final_node} (already sent during execution)")
                else:
                    await send(_payload_with_trace_id({
                        "status": "completed",
                        "node": final_node,
                        "output": clean_output,
                    }))
            else:
                # ✅ FIX: Skip sending final callback if resume_scorer or assessment_validator already sent one during execution
                if final_node == "resume_scorer" and "resume_scorer" in nodes_that_sent_callbacks:
                    log.info(f"🔄 Skipping duplicate final callback for resume_scorer (already sent during execution)")
                elif final_node == "assessment_validator" and "assessment_validator" in nodes_that_sent_callbacks:
                    log.info(f"🔄 Skipping duplicate final callback for assessment_validator (already sent during execution)")
                else:
                    # Include full job_description (with jobTitle) and add top-level job_title for JD flow callbacks
                    callback_output = filtered_result if isinstance(filtered_result, dict) else {}
                    if isinstance(callback_output, dict) and "job_description" in callback_output:
                        job_desc = callback_output.get("job_description")
                        if isinstance(job_desc, dict):
                            job_title = job_desc.get("jobTitle") or job_desc.get("job_title") or job_desc.get("title") or ""
                            if job_title:
                                callback_output = {**callback_output, "job_title": job_title}
                    
                    await send(_payload_with_trace_id({
                        "status": "completed",
                        "node": final_node,
                        "output": callback_output,
                    }))

            # Run model-based eval (LLM-as-judge) in background if enabled
            trace_id = get_trace_id_from_handler(handler_ref)
            if trace_id and final_node:
                run_model_based_eval(trace_id, result, final_node, settings)
        
        # Defer session completion to graph terminal nodes only (end/end_invalid_*)
        # Do not complete session here to avoid premature termination mid-flow
        
    except Exception as e:
        log.error(f"Graph execution failed: {e}")
        if _pipeline_stream_start is not None:
            try:
                _fid, _flab = classify_pipeline_flow(state)
                log_pipeline_flow_complete(
                    _fid,
                    _flab,
                    time.time() - _pipeline_stream_start,
                    uid=uid,
                    tenant_id=tenant_id,
                    analysis_id=state.get("analysis_id") or "",
                    ok=False,
                    extra="path=astream reason=exception",
                )
            except Exception:
                pass

        # Complete session on error if it exists (force completion for error cases)
        session_id = state.get("session_id", "")
        if session_id:
            try:
                session_manager.complete_session(session_id, force=True)
                log.info(f"✅ Session completed (error): {session_id}")
            except Exception as session_error:
                log.warning(f"Failed to complete session {session_id} on error: {session_error}")
        
        await send(_payload_with_trace_id({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }))

    # Send a final "completed" marker for the whole pipeline (includes trace_id for scoring)
    await send(_payload_with_trace_id({"status": "completed", "node": "pipeline", "message": "Processing completed"}))

def get_node_progress(node_name: str) -> float:
    """Calculate progress percentage based on node name."""
    node_progress_map = {
        "dispatcher": 0.1,
        "validate_resume": 0.2,
        "validate_jd": 0.2,
        "groq_resume_parser": 0.35,  # Single parsing agent
        "job_description_parser": 0.4,
        "prescreening_questions": 0.5,
        "resume_assembler": 0.5,
        "resume_scorer": 0.6,
        "ranker": 0.6,
        "career_advisor": 0.7,
        "market_and_course_recommender": 0.8,
        "assessment_recommender": 0.9,
        "assessment_evaluator": 0.6,
        "assessment_question_generator": 0.6,
        "notification_agent": 0.6,
        "report_generator": 0.8,
    }
    return node_progress_map.get(node_name, 0.5)

def make_serializable(obj):
    """Best-effort convert to JSON-serializable with proper structure."""
    if hasattr(obj, "content"):
        content = obj.content
        if isinstance(content, str) and "```json" in content:
            return extract_json_from_ai_message(obj)
        return content
    if isinstance(obj, dict):
        # Ensure proper nesting and clean structure
        cleaned_dict = {}
        for k, v in obj.items():
            if v is not None:  # Remove None values for cleaner structure
                cleaned_dict[k] = make_serializable(v)
        return cleaned_dict
    if isinstance(obj, list):
        # Clean list items and remove None values
        return [make_serializable(i) for i in obj if i is not None]
    return obj

def extract_json_from_ai_message(message):
    content = getattr(message, "content", None)
    if isinstance(content, str) and "```json" in content:
        json_start = content.find("```json\n") + 8
        json_end = content.rfind("```")
        json_str = content[json_start:json_end].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON from message"}
    return content

def process_event_to_dict(event) -> dict | None:
    """Convert streaming event to dictionary understood by your callback consumer."""
    if not isinstance(event, dict):
        return None

    ev_type = event.get("event")
    name = event.get("name", "unknown")

    if ev_type == "on_chain_start":
        return {
            "status": "started",
            "node": name,
            "message": f"Starting {name}..."
        }

    if ev_type == "on_chain_end":
        output = event.get("data", {}).get("output", {})
        serializable_output = make_serializable(output)

        if name == "resume_parser":
            return {
                "status": "completed",
                "node": name,
                "structured_resume": serializable_output.get("structured_resume", {})
            }
        if name == "gap_analyzer":
            return {
                "status": "completed",
                "node": name,
                "skill_gap_analysis": serializable_output.get("raw_skill_gap_analysis_output", "")
            }
        # Generic
        return {"status": "completed", "node": name, "output": serializable_output}

    return None

def extract_json_from_ai_message(message):
    content = getattr(message, "content", None)
    if isinstance(content, str) and "```json" in content:
        json_start = content.find("```json\n") + 8
        json_end = content.rfind("```")
        json_str = content[json_start:json_end].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON from message"}
    return content

@performance_timer("run_pipeline")
async def run_pipeline(state: dict):
    """Kick off the pipeline and stream events to the callback."""
    uid = state.get("uid")
    callback_url = state.get("callback_url")

    if not uid or not callback_url:
        # No callback possible; nothing to do
        return

    from core.pipeline_concurrency import pipeline_execution_slot

    async with pipeline_execution_slot():
        log.info(f"Pipeline started | uid={uid}, callback={callback_url}")

        async def send(event: dict):
            await send_to_callback(callback_url, uid, event)

        # Optional: a generic "started" message (after acquiring a pipeline slot when capped)
        await send({"status": "started", "message": "Processing started"})

        pipeline_timeout = float(
            state.get("_pipeline_timeout_seconds") or getattr(settings, "PIPELINE_TIMEOUT_SECONDS", 300)
        )

        # Check if we should use mock mode (explicit flag only)
        if getattr(settings, "USE_MOCKS", False):
            log.info(f"🔧 MOCK MODE: Using mock responses instead of LLM calls")
            try:
                await asyncio.wait_for(
                    mock_stream_and_callback_events(state, send, {"recursion_limit": GRAPH_RECURSION_LIMIT}),
                    timeout=pipeline_timeout,
                )
            except asyncio.TimeoutError:
                log.error(f"Mock pipeline timeout | uid={uid}")
                await send(
                    {
                        "status": "error",
                        "message": f"Mock processing timed out after {int(pipeline_timeout)} seconds.",
                    }
                )
            except Exception as e:
                log.error(f"Mock pipeline exception | uid={uid}, error={str(e)}")
                error_log.error(traceback.format_exc())
                # Sanitize error message before sending to callback
                from core.security import sanitize_error_message

                sanitized_message = sanitize_error_message(e, include_details=False)
                await send({"status": "error", "message": sanitized_message})
        else:
            # Production mode - use real LLM calls
            log.info(f"🚀 PRODUCTION MODE: Using real LLM calls (timeout={pipeline_timeout}s)")
            try:
                await asyncio.wait_for(
                    stream_and_callback_events(state, send, {"recursion_limit": GRAPH_RECURSION_LIMIT}),
                    timeout=pipeline_timeout,
                )
            except asyncio.TimeoutError:
                log.error(f"Pipeline timeout | uid={uid}")
                await send(
                    {
                        "status": "error",
                        "message": f"Processing timed out after {int(pipeline_timeout)} seconds.",
                    }
                )
            except Exception as e:
                log.error(f"Pipeline exception | uid={uid}, error={str(e)}")
                error_log.error(traceback.format_exc())
                from core.security import sanitize_error_message

                sanitized_message = sanitize_error_message(e, include_details=False)
                await send({"status": "error", "message": sanitized_message})


# Single PATCH endpoint for all bodies
# Returns 202 Accepted: pipeline runs in background; results via callback_url (same as compare-candidate-job).
@app.patch("/analyze-resume-callback")
async def analyze_resume_callback(request: Request):
    # Set correlation ID for request tracing
    corr_id = set_correlation_id()
    
    # Log security event for API access
    client_ip = request.client.host if request.client else "unknown"
    log_security_event(
        event="api_access",
        severity="INFO",
        source_ip=client_ip,
        action="analyze_resume_callback",
        result="started",
        correlation_id=corr_id
    )
    
    log.info("Entry point /analyze-resume-callback")
    """
    Production-ready entrypoint for resume parsing, skill gap, assessment, etc.
    Body must include at least: uid, callback_url, tenant_id
    Other fields can vary; dispatcher in the graph will route appropriately.

    Returns **202 Accepted** immediately; pipeline runs in background and reports
    via ``callback_url`` (same pattern as ``/compare-candidate-job``).
    """
    try:
        body = await request.json()
        log.info(f"Exit point #1 /analyze-resume-callback | Body parsed | uid={body.get('uid')}")
    except Exception:
        log.error("Exit point #2 /analyze-resume-callback | Invalid JSON body")
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    # Reject candidate_job_match requests - these should use /compare-candidate-job endpoint
    if body.get("request_type") == "candidate_job_match":
        log.warning(f"Invalid request_type 'candidate_job_match' received on /analyze-resume-callback endpoint | uid={body.get('uid')}")
        raise HTTPException(
            status_code=400,
            detail="Invalid request_type for this endpoint. Use /compare-candidate-job endpoint for candidate_job_match requests."
        )
    
    # Verify GenAI token (after parsing body so we can check body for token if needed)
    is_valid_token = await verify_request_token(request, body)
    if not is_valid_token:
        log.warning("Invalid or missing GenAI token for /analyze-resume-callback")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token"
        )

    # Pydantic validation with production schemas
    try:
        validated_input = ProductionResumeInput(**body)
        log.info(f"Input validation passed | uid={validated_input.uid} | tenant_id={validated_input.tenant_id}")
    except Exception as e:
        log.error(f"Input validation failed: {e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    
    # Validate callback URL whitelist
    if validated_input.callback_url:
        is_valid_url, error_msg = validate_callback_url(str(validated_input.callback_url))
        if not is_valid_url:
            log.error(f"Callback URL validation failed: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

    # Additional production validation
    from core.supervisor_agent import validate_payload_size
    if not validate_payload_size(body):
        tenant_id_for_log = validated_input.tenant_id or "default-tenant"
        log.error(f"Payload size exceeds limit for tenant: {tenant_id_for_log}")
        raise HTTPException(status_code=413, detail="Payload size exceeds limit")

    await get_access_token(validated_input.uid)

    # Session Management Integration
    uid = validated_input.uid
    tenant_id = validated_input.tenant_id or uid
    
    # Always use get_or_reuse_session to ensure we reuse existing sessions
    # Run in dedicated thread pool with timeout so event loop is not blocked (Section 2 Issue 3)
    log.info(f"🔄 Getting or creating session for UID={uid}")
    loop = asyncio.get_event_loop()
    session = None
    try:
        session = await asyncio.wait_for(
            loop.run_in_executor(
                _session_executor,
                lambda: session_manager.get_or_reuse_session(
                    owner_id=uid,
                    kind="candidate_pipeline",
                    owner_type="candidate",
                    initial_step="start",
                    initial_data={
                        "uid": uid,
                        "tenant_id": tenant_id,
                        "callback_url": str(validated_input.callback_url)
                    }
                )
            ),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        log.warning(f"Session get_or_reuse_session timed out for UID={uid}; continuing without session")
    except Exception as e:
        log.warning(f"Session get_or_reuse_session failed for UID={uid}: {e}; continuing without session")

    if session:
        log.info(f"✅ Using session: {session.session_id}")
        log.info(f"🔄 Current step: {session.state.step}")
        log.info(f"🔄 Progress: {session.state.progress:.1%}")
        
        # Update heartbeat
        session_manager.update_heartbeat(session.session_id)
        
        # Add session data to state
        session_data = {
            "session_id": session.session_id,
            "resume_from": True,  # Always true since we're reusing sessions
            "current_step": session.state.step,
            "progress": session.state.progress
        }
    else:
        log.error(f"❌ Session creation failed for UID={uid}")
        session_data = {
            "session_id": "",
            "resume_from": False,
            "current_step": "start",
            "progress": 0.0
        }

    # Build initial state with production fields and session data
    state = {}
    state.update(body)                 
    state["body"] = body               
    state["uid"] = uid
    state["tenant_id"] = tenant_id
    state["callback_url"] = str(validated_input.callback_url)
    state.update(session_data)  # Add session data
    state["endpoint_name"] = "analyze_resume_callback"  # ✅ Track which endpoint initiated the request

    # Pipeline timeout: job-matching (career flow with resume_url) needs longer due to many LLM calls
    is_job_matching_flow = bool(body.get("resume_url"))
    pipeline_timeout = (
        getattr(settings, "PIPELINE_TIMEOUT_JOB_MATCHING_SECONDS", 600)
        if is_job_matching_flow
        else getattr(settings, "PIPELINE_TIMEOUT_SECONDS", 300)
    )
    state["_pipeline_started_at"] = time.time()
    state["_pipeline_timeout_seconds"] = pipeline_timeout

    # Fire-and-forget: outer timeout slightly higher than inner so inner can send partial/error first
    schedule_background_task(run_pipeline(state), "run_pipeline", timeout_seconds=float(pipeline_timeout) + 30)
    log.info(f"Exit point #4 /analyze-resume-callback | Processing started | uid={validated_input.uid} | tenant_id={validated_input.tenant_id}")

    return JSONResponse(
        status_code=202,
        content={
            "status": "processing",
            "uid": validated_input.uid,
            "tenant_id": validated_input.tenant_id,
            "callback_url": str(validated_input.callback_url),
            "message": "Processing started; results will be sent to callback_url",
            "session_id": session_data.get("session_id", ""),
            "resume_from": session_data.get("resume_from", False),
            "current_step": session_data.get("current_step", "start"),
            "progress": session_data.get("progress", 0.0),
        },
    )

@app.get("/debug")
def debug_langgraph():
    import langgraph
    import inspect

    version = getattr(langgraph, "__version__", "Unknown")
    graph_methods = [method for method in dir(graph) if not method.startswith("_")]
    streaming_methods = {}
    for method_name in ["astream", "astream_events", "astream_log"]:
        if hasattr(graph, method_name):
            method = getattr(graph, method_name)
            try:
                signature = str(inspect.signature(method))
                streaming_methods[method_name] = signature
            except:
                streaming_methods[method_name] = "Error getting signature"

    return {
        "langgraph_version": version,
        "graph_type": type(graph).__name__,
        "available_methods": graph_methods,
        "streaming_methods": streaming_methods,
        "environment": {
            "APP_ENV": settings.APP_ENV,
            "USE_MOCKS": getattr(settings, "USE_MOCKS", False),
            "mock_mode": getattr(settings, "USE_MOCKS", False),
            "description": "Mock responses enabled" if getattr(settings, "USE_MOCKS", False) else "Real LLM calls enabled"
        }
    }


@app.post("/compare-candidate-job")
@app.patch("/compare-candidate-job")
async def compare_candidate_job_endpoint(request: Request):
    """
    Compare a candidate's structured resume with a job description.
    Routes through the graph for consistent processing and callbacks.
    NON-BLOCKING: Returns **HTTP 202 Accepted** immediately; results sent via callback.
    
    Request Body:
    - uid (str, required): UID of the candidate
    - job_id (str, required): ID of the job (job description will be fetched from DB)
    - callback_url (str, required): URL to send results via callback (required for non-blocking)
    - tenant_id (str, optional): Tenant ID
    
    Returns:
    - status: "processing"
    - uid: Candidate UID
    - job_id: Job ID
    - callback_url: Callback URL where results will be sent
    - message: Processing status message
    """
    log.info("Entry point /compare-candidate-job")
    
    try:
        body = await request.json()
        log.info(f"Received request body: {list(body.keys())}")
        
        # Verify GenAI token (after parsing body so we can check body for token if needed)
        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            log.warning("Invalid or missing GenAI token for /compare-candidate-job")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication token"
            )
        
        uid = body.get("uid")
        job_id = body.get("job_id")
        callback_url = body.get("callback_url")
        tenant_id = body.get("tenant_id", uid)
        
        log.info(f"Extracted: uid={uid}, job_id={job_id}, callback_url={callback_url}, tenant_id={tenant_id}")
        
        # Validate required fields
        if not uid or not job_id:
            log.error(f"Missing required fields: uid={uid}, job_id={job_id}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Missing required fields",
                    "message": "Both uid and job_id are required",
                    "received": {
                        "uid": uid,
                        "job_id": job_id
                    }
                }
            )
        
        # Validate callback URL (required for non-blocking operation)
        if not callback_url:
            log.error("Missing callback_url - required for non-blocking operation")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Missing callback_url",
                    "message": "callback_url is required for non-blocking operation. Results will be sent to the callback URL."
                }
            )
        
        # Validate callback URL format
        is_valid_url, error_msg = validate_callback_url(callback_url)
        if not is_valid_url:
            log.error(f"Callback URL validation failed: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        log.info(f"Routing candidate-job comparison through graph: candidate {uid} with job {job_id}")
        
        # Extract resume_url and resume_text from body if present
        resume_url = body.get("resume_url")
        resume_text = body.get("resume_text")
        recruiter_questions = body.get("recruiter_questions") or body.get("recruiterQuestions") or body.get("questions")
        interview_feedback = body.get("interview_feedback") or body.get("interviewFeedback")
        
        # Prepare state for graph routing
        # Set request_type and endpoint_name to ensure proper routing to job_matcher
        state = {
            "uid": uid,
            "job_id": job_id,
            "request_type": "candidate_job_match",
            "tenant_id": tenant_id,
            "callback_url": callback_url,
            "resume_url": resume_url,
            "resume_text": resume_text,
            "endpoint_name": "compare_candidate_job",
            "body": {
                **body,
                "request_type": "candidate_job_match",
                "endpoint_name": "compare_candidate_job",
                "job_id": job_id,
            }
        }
        
        if recruiter_questions:
            state["recruiter_questions"] = recruiter_questions
            log.info(f"✅ Added {len(recruiter_questions) if isinstance(recruiter_questions, list) else 1} recruiter Q&A to state")
        
        if interview_feedback:
            state["interview_feedback"] = interview_feedback
            log.info(f"📋 Added interview feedback to state for post-interview analysis")

        # Pipeline timeout: compare flow always uses job-matching timeout (many LLM calls)
        pipeline_timeout = getattr(settings, "PIPELINE_TIMEOUT_JOB_MATCHING_SECONDS", 600)
        state["_pipeline_started_at"] = time.time()
        state["_pipeline_timeout_seconds"] = pipeline_timeout
        
        # NON-BLOCKING: Fire-and-forget graph processing
        # This ensures the endpoint returns immediately without blocking
        log.info(f"Starting non-blocking graph processing for candidate-job comparison")
        schedule_background_task(run_pipeline(state), "run_pipeline_compare_candidate_job", timeout_seconds=float(pipeline_timeout) + 30)
        
        log.info(f"✅ Non-blocking graph processing started for candidate {uid} with job {job_id}")
        
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "uid": uid,
                "job_id": job_id,
                "callback_url": callback_url,
                "message": "Comparison started, results will be sent to callback URL",
                "node": "job_matcher"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error in compare_candidate_job_endpoint: {e}", exc_info=True)
        error_log.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "message": "Failed to start candidate-job comparison",
                "type": type(e).__name__
            }
        )


@app.post("/interview")
async def interview(request: Request):
    """
    Unified conversational agent endpoint - routes to interview or career guidance workflows.
    
    Routes based on payload structure:
    - Interview payload: Has "interview_topic", "answer", or "question_count" fields
    - Career guidance payload: Has "message" field
    
    Interview Payload Example:
    {
        "uid": "user123",
        "interview_topic": "Python",
        "structured_resume": {...},
        "answer": "user's answer",
        "question_count": 1,
        "stream": true  # Optional: If true, returns SSE streaming response
    }
    
    Career Guidance Payload Example:
    {
        "uid": "user123",
        "message": "What courses should I take?",
        "session_id": "optional",
        "callback_url": "optional",
        "stream": true
    }
    """
    log.info("Entry point /interview")
    
    try:
        # Lightweight per-UID rate limiting on interview starts (only for interview workflow)
        body = await request.json()
        
        # Verify GenAI token (after parsing body so we can check body for token if needed)
        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            log.warning("Invalid or missing GenAI token for /interview")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication token"
            )
        
        # Check if streaming is requested for interview workflow
        stream = body.get("stream", False)
        
        # For interview workflow, check if streaming is requested
        if stream and ("interview_topic" in body or "answer" in body or "question_count" in body):
            # Use streaming interview agent
            from agents.interview_agent.interview_agent import ai_interview_agent_intelligent_stream
            return await ai_interview_agent_intelligent_stream(request)
        
        # Rate limiting only applies to interview workflow starts
        uid = (body or {}).get("uid")
        session_id = (body or {}).get("session_id")
        question_count = (body or {}).get("question_count")
        has_message = "message" in body and body.get("message")
        
        # Only apply rate limiting to interview workflow (not career guidance)
        if uid and not session_id and not question_count and not has_message:
            # allow 3 starts per 60 seconds
            from time import time
            now = time()
            rl = getattr(app.state, "_interview_rl", {})
            bucket = rl.get(uid, {"ts": now, "count": 0})
            # reset window if older than 60s
            if now - bucket["ts"] > 60:
                bucket = {"ts": now, "count": 0}
            bucket["count"] += 1
            rl[uid] = bucket
            app.state._interview_rl = rl
            if bucket["count"] > 3:
                log.warning(f"Rate limit exceeded for uid={uid}")
                raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait and try again.")

        # Route to appropriate workflow based on payload
        from agents.interview_agent.workflow_router import route_workflow
        response = await route_workflow(request)
        
        # Handle response serialization for interview workflow (backward compatibility)
        # Career guidance workflow returns StreamingResponse or CareerChatResponse directly
        if hasattr(response, 'dict') and hasattr(response, 'status'):
            # Interview response - serialize with backward-compatible shape
            log.info("Exit point #1, /interview | Success (Interview)")
            resp_dict = response.dict()
            resp_dict.update(
                {
                    "status": response.status,
                    "session_id": response.session_id,
                    "question": response.question,
                    "question_count": response.question_count,
                    "conversation_history": response.conversation_history,
                    "evaluation_summary": response.evaluation_summary,
                    # Include structured report (UI-formatted) - this is what Node.js expects
                    "structured_report": response.final_summary,
                    # Alias for UI: final_report mirrors evaluation_summary (raw evaluator output)
                    "final_report": response.evaluation_summary,
                    # Alias for UI: narrative_report mirrors detailed_summary (LLM text)
                    "narrative_report": response.detailed_summary,
                }
            )
            return resp_dict
        else:
            # Career guidance response (StreamingResponse or CareerChatResponse)
            log.info("Exit point #1, /interview | Success (Career Guidance)")
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        #  error details to terminal for debugging
        log.error(f"\n{'='*80}")        
        log.error(f"🚨 APP.PY INTERVIEW ENDPOINT ERROR 🚨")
        log.error(f"{'='*80}")
        log.error(f"Error Type: {type(e).__name__}")
        log.error(f"Error Message: {str(e)}")
        log.error(f"\nFull Traceback:")
        log.error(f"{traceback.format_exc()}")
        log.error(f"{'='*80}\n")
        
        log.error(f"Exit point #2, /interview | Error: {str(e)}")
        error_log.error(traceback.format_exc())
        raise


@app.post("/videointerview")
async def video_interview(request: Request):
    """
    Dedicated video interview endpoint with streaming support and avatar generation.
    
    This endpoint is optimized for video interviews with:
    - Real-time question streaming (SSE)
    - Avatar video generation (HeyGen Streaming API or Video Generation API)
    - Parallel processing for low latency
    - WebRTC streaming support
    
    Request Body:
    {
        "uid": "user123",                    // REQUIRED
        "interview_topic": "Python",         // REQUIRED
        "structured_resume": {...},          // Optional
        "answer": "user's answer",           // Optional (for subsequent questions)
        "question_count": 1,                 // Optional
        "generate_avatar": true,             // Optional: Enable avatar video generation
        "stream": true,                      // Optional: Enable SSE streaming (default: true)
        "session_id": "optional-session-id" // Optional
    }
    
    Response (Streaming - SSE):
    - type: "start" - Stream started
    - type: "progress" - Progress update
    - type: "chunk" - Question text chunk
    - type: "avatar_status" - Avatar generation status
    - type: "avatar_video_ready" - Avatar video ready (WebRTC or URL)
    - type: "done" - Stream complete
    
    Response (Non-streaming):
    {
        "question": "What is your experience with Python?",
        "session_id": "session_123",
        "question_count": 1,
        "conversation_history": [...]
    }
    """
    log.info("Entry point /videointerview")
    
    try:
        body = await request.json()
        
        # Verify GenAI token
        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            log.warning("Invalid or missing GenAI token for /videointerview")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication token"
            )
        
        # Validate required fields
        if not body.get("uid"):
            raise HTTPException(status_code=400, detail="uid is required")
        if not body.get("interview_topic"):
            raise HTTPException(status_code=400, detail="interview_topic is required")
        
        # Check if streaming is requested (default: true for video interviews)
        stream = body.get("stream", True)
        
        if stream:
            # Use streaming interview agent
            from agents.interview_agent.interview_agent import ai_interview_agent_intelligent_stream
            return await ai_interview_agent_intelligent_stream(request)
        else:
            # Non-streaming mode
            from agents.interview_agent.interview_agent import ai_interview_agent_intelligent
            response = await ai_interview_agent_intelligent(request)
            
            # Serialize response
            if hasattr(response, 'dict'):
                resp_dict = response.dict()
                resp_dict.update({
                    "status": response.status,
                    "session_id": response.session_id,
                    "question": response.question,
                    "question_count": response.question_count,
                    "conversation_history": response.conversation_history,
                    "evaluation_summary": response.evaluation_summary,
                    # Include structured report (UI-formatted) - this is what Node.js expects
                    "structured_report": response.final_summary,
                    # Alias for UI: final_report mirrors evaluation_summary (raw evaluator output)
                    "final_report": response.evaluation_summary,
                    "narrative_report": response.detailed_summary,
                })
                return resp_dict
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"Error in /videointerview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/videosdk/meeting")
async def create_videosdk_meeting():
    """
    Create a VideoSDK meeting room for video interviews.
    
    Returns:
        Dict with roomId and token, or error dict
    """
    try:
        from agents.interview_agent.video_utils import create_videosdk_meeting
        result = await create_videosdk_meeting()
        return result
    except Exception as e:
        log.exception(f"Error creating VideoSDK meeting: {e}")
        return {
            "error": f"Error creating meeting: {str(e)}"
        }


@app.get("/health/avatar")
async def health_avatar():
    """
    Health check endpoint for avatar video generation system.
    Checks HeyGen API configuration and connectivity.
    
    Returns:
        Dict with health status and configuration info
    """
    from agents.interview_agent.video_utils import HEYGEN_API_KEY
    from settings import settings
    
    heygen_configured = bool(HEYGEN_API_KEY)
    avatar_enabled = getattr(settings, "AVATAR_ENABLED", True)
    
    status = "healthy" if (heygen_configured and avatar_enabled) else "unhealthy"
    
    return {
        "status": status,
        "heygen_configured": heygen_configured,
        "avatar_enabled": avatar_enabled,
        "mock_mode": getattr(settings, "HEYGEN_MOCK_MODE", False)
    }


@app.post("/coach/chat/ask")
@app.post("/career-coach/chat")
async def coach_chat_ask(request: Request):
    """
    Career chatbot endpoint with streaming support.
    
    CONCURRENCY: This endpoint is fully thread-safe and supports multiple concurrent users.
    Each request is isolated by uid and session_id, ensuring no data conflicts.
    
    Supports both streaming and non-streaming modes based on request parameter.
    
    Request Body:
    {
        "uid": "user123",  // REQUIRED: User ID for session isolation
        "message": "What should I focus on to advance my career?",
        "session_id": "optional-session-id",  // If not provided, auto-generated
        "callback_url": "optional-callback-url",
        "stream": true  // Default: true (enables streaming)
    }
    
    Streaming Response (SSE format):
    - type: "start" - Stream started
    - type: "chunk" - Content chunk
    - type: "done" - Stream complete with metadata
    - type: "error" - Error occurred
    
    Non-streaming Response:
    {
        "response": "Full response text",
        "session_id": "session_id",
        "conversation_history": [...],
        "suggested_topics": [...],
        "action_items": [...],
        "confidence_score": 0.8
    }
    """
    log.info("Entry point /coach/chat/ask")
    
    try:
        body = await request.json()
        
        # Log request details
        uid = body.get("uid", "unknown")
        message = body.get("message", "")
        stream = body.get("stream", True)
        log.info(f"📥 Career Chat Request | uid={uid} | stream={stream} | message_length={len(message)}")
        log.info(f"📝 User Message: {message[:200]}{'...' if len(message) > 200 else ''}")
        
        # Verify GenAI token
        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            log.warning("Invalid or missing GenAI token for /coach/chat/ask")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication token"
            )
        
        # Validate request
        chat_request = CareerChatRequest(**body)
        
        # Check if streaming is requested (default: True)
        if chat_request.stream:
            # Return streaming response (callback_url ignored for 202; use SSE for long responses)
            return await career_chatbot_agent_stream(chat_request)
        else:
            _coach_timeout = float(getattr(settings, "PIPELINE_TIMEOUT_SECONDS", 300)) + 120

            if chat_request.callback_url:
                async def _coach_chat_bg():
                    from core.security import sanitize_error_message

                    try:
                        response = await career_chatbot_agent(chat_request)
                        callback_payload = {
                            "node": "career_chatbot",
                            "status": "completed",
                            "output": response.dict(),
                        }
                        await send_to_callback(
                            str(chat_request.callback_url),
                            chat_request.uid,
                            callback_payload,
                        )
                        log.info(f"📤 Sent career coach callback to {chat_request.callback_url}")
                    except Exception as e:
                        log.error(f"Background career coach chat failed: {e}", exc_info=True)
                        try:
                            msg = sanitize_error_message(e, include_details=False)
                            await send_to_callback(
                                str(chat_request.callback_url),
                                chat_request.uid,
                                {
                                    "node": "career_chatbot",
                                    "status": "error",
                                    "error": msg,
                                    "output": {},
                                },
                            )
                        except Exception as cb_e:
                            log.error(f"career coach error callback failed: {cb_e}")

                schedule_background_task(
                    _coach_chat_bg(),
                    "career_coach_chat",
                    timeout_seconds=_coach_timeout,
                )
                return JSONResponse(
                    status_code=202,
                    content={
                        "status": "processing",
                        "message": "Career chat processing started; results will be sent to callback_url",
                        "uid": chat_request.uid,
                        "callback_url": str(chat_request.callback_url),
                        "node": "career_chatbot",
                    },
                )

            response = await career_chatbot_agent(chat_request)
            log.info("Exit point #1 /coach/chat/ask | Success")
            return response
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Exit point #2 /coach/chat/ask | Error: {str(e)}")
        error_log.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Career chatbot error: {str(e)}"
        )


# ============================================================================
# ASSESSMENTS API — create plan from spec (topic, num_questions, difficulty, time_limit)
# ============================================================================

from agents.assessment_recommender import create_plan_from_spec as assessment_create_plan_from_spec


@app.post("/assessments")
async def assessments_endpoint(request: Request):
    """
    Create assessment plan from spec. Single action: create_from_spec.

    Body: uid, action="create_from_spec", topic, num_questions={mcq, short, long, coding}, difficulty, time_limit_minutes (optional)
    """
    try:
        body = await request.json()
        uid = (body.get("uid") or "").strip()
        action = (body.get("action") or "").strip().lower()

        if action != "create_from_spec":
            raise HTTPException(
                status_code=400,
                detail="Invalid action. Use action: 'create_from_spec' with topic, num_questions, difficulty, and optional time_limit_minutes."
            )
        if not uid:
            raise HTTPException(status_code=400, detail="uid is required")

        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            raise HTTPException(status_code=401, detail="Invalid or missing authentication token")

        topic = (body.get("topic") or "").strip()
        if not topic:
            raise HTTPException(status_code=400, detail="topic is required for create_from_spec")

        num_questions = body.get("num_questions")
        if not isinstance(num_questions, dict):
            num_questions = {}
        difficulty = (body.get("difficulty") or "medium").strip()
        time_limit_minutes = body.get("time_limit_minutes")
        if time_limit_minutes is not None:
            try:
                time_limit_minutes = int(time_limit_minutes)
            except (TypeError, ValueError):
                time_limit_minutes = None
        session_id = body.get("session_id")
        callback_url = (body.get("callback_url") or "").strip() or None
        if callback_url:
            is_valid_url, err_msg = validate_callback_url(callback_url)
            if not is_valid_url:
                raise HTTPException(status_code=400, detail=err_msg or "Invalid callback_url")

        _assess_timeout = float(getattr(settings, "PIPELINE_TIMEOUT_SECONDS", 300)) + 120

        if callback_url:
            async def _assessments_bg():
                from core.security import sanitize_error_message

                try:
                    result = await assessment_create_plan_from_spec(
                        topic=topic,
                        num_questions=num_questions,
                        difficulty=difficulty,
                        time_limit_minutes=time_limit_minutes,
                        uid=uid,
                        session_id=session_id,
                    )
                    if not result.get("success"):
                        await send_to_callback(
                            callback_url,
                            uid,
                            {
                                "status": "error",
                                "node": "assessments",
                                "error": result.get("error", "Assessment plan creation failed"),
                                "output": result,
                            },
                        )
                        return
                    await send_to_callback(
                        callback_url,
                        uid,
                        {"status": "completed", "node": "assessments", "output": result},
                    )
                    log.info(f"📤 Sent assessments callback to {callback_url}")
                except Exception as e:
                    log.error(f"Background /assessments failed: {e}", exc_info=True)
                    try:
                        msg = sanitize_error_message(e, include_details=False)
                        await send_to_callback(
                            callback_url,
                            uid,
                            {
                                "status": "error",
                                "node": "assessments",
                                "error": msg,
                                "output": {},
                            },
                        )
                    except Exception as cb_e:
                        log.error(f"/assessments error callback failed: {cb_e}")

            schedule_background_task(
                _assessments_bg(),
                "assessments_create_plan",
                timeout_seconds=_assess_timeout,
            )
            return JSONResponse(
                status_code=202,
                content={
                    "status": "processing",
                    "message": "Assessment plan creation started; results will be sent to callback_url",
                    "uid": uid,
                    "callback_url": callback_url,
                    "node": "assessments",
                },
            )

        result = await assessment_create_plan_from_spec(
            topic=topic,
            num_questions=num_questions,
            difficulty=difficulty,
            time_limit_minutes=time_limit_minutes,
            uid=uid,
            session_id=session_id,
        )
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Assessment plan creation failed"))
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error in /assessments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-resume-content", response_model=Dict[str, Any])
async def generate_resume_content(request: Request):
    """
    Generate resume content in formatted Markdown from structured resume and context.
    Works with minimal data - only requires structured resume. All other context is optional.
    
    Request Body (distinct from resume flow to avoid routing conflicts):
    {
      "uid": "user123",
      "tenant_id": "tenant_id" (optional, defaults to uid),
      "session_id": "session_id" (optional),
      "callback_url": "https://..." (optional - output will be POSTed here on completion),
      "certificates": [ {"name": "...", "issuer": "...", "date": "..."}, ... ] (optional, max 50 items)
    }
    
    Note: Does NOT accept resume_url or resume_text to avoid being routed to the resume analysis flow.
    
    Returns:
    - resume_content: Resume content (Markdown format)
    - success: Boolean indicating success
    - processing_time_seconds: Processing time
    - context_used: Dictionary showing which context was available
    """
    # Set correlation ID for request tracing
    corr_id = set_correlation_id()
    
    # Log security event for API access
    client_ip = request.client.host if request.client else "unknown"
    log_security_event(
        event="api_access",
        severity="INFO",
        source_ip=client_ip,
        action="generate_resume_content",
        result="started",
        correlation_id=corr_id
    )
    
    log.info("Entry point /generate-resume-content")
    
    try:
        body = await request.json()
        log.info(f"Exit point #1 /generate-resume-content | Body parsed | uid={body.get('uid')}")
    except Exception:
        log.error("Exit point #2 /generate-resume-content | Invalid JSON body")
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    
    # Verify GenAI token (after parsing body so we can check body for token if needed)
    is_valid_token = await verify_request_token(request, body)
    if not is_valid_token:
        log.warning("Invalid or missing GenAI token for /generate-resume-content")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token"
        )
    
    # Pydantic validation with separate schema (not ProductionResumeInput)
    try:
        validated_input = ResumeContentGeneratorInput(**body)
        log.info(f"Input validation passed | uid={validated_input.uid} | tenant_id={validated_input.tenant_id}")
    except Exception as e:
        log.error(f"Input validation failed: {e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    
    # Extract validated fields
    uid = validated_input.uid
    # tenant_id defaults to uid if not provided
    tenant_id = validated_input.tenant_id or uid
    # session_id is optional - will be found automatically if not provided
    session_id = validated_input.session_id
    callback_url = str(validated_input.callback_url) if validated_input.callback_url else None
    
    log.info(f"Generating resume content for uid={uid}, tenant_id={tenant_id}, session_id={session_id}, callback_url={bool(callback_url)}")
    
    # Persist optional certificates to Chroma (resume_doc) so resume_content_generator can include them
    if validated_input.certificates is not None and len(validated_input.certificates) > 0:
        try:
            from chroma import get_resume_doc, upsert_resume_doc
            doc = await run_blocking_io(get_resume_doc, uid) or {}
            doc["uploaded_certificates"] = validated_input.certificates
            await run_blocking_io(upsert_resume_doc, uid, doc)
            log.info(f"Persisted {len(validated_input.certificates)} certificate(s) to resume_doc for uid={uid}")
        except Exception as e:
            log.warning(f"Failed to persist certificates to Chroma (non-fatal): {e}")
    
    try:
        from agents.resume_content_generator import resume_content_generator_agent

        _resume_content_timeout = float(getattr(settings, "PIPELINE_TIMEOUT_SECONDS", 300)) + 120

        if callback_url:
            async def _resume_content_bg():
                from core.security import sanitize_error_message

                try:
                    result = await resume_content_generator_agent(
                        uid=uid,
                        tenant_id=tenant_id,
                        session_id=session_id,
                    )
                    callback_payload = {
                        "status": "completed",
                        "node": "resume_content_generator",
                        "output": {
                            "resume_content": result.get("resume_content", ""),
                            "success": result.get("success", True),
                            "processing_time_seconds": result.get("processing_time_seconds", 0),
                            "context_used": result.get("context_used", {}),
                            "uid": uid,
                            "tenant_id": tenant_id,
                        },
                    }
                    await send_to_callback(callback_url, uid, callback_payload)
                    log.info(f"📤 Sent resume content callback to {callback_url}")
                except Exception as e:
                    log.error(f"Background generate-resume-content failed: {e}", exc_info=True)
                    try:
                        msg = sanitize_error_message(e, include_details=False)
                        await send_to_callback(
                            callback_url,
                            uid,
                            {
                                "status": "error",
                                "node": "resume_content_generator",
                                "error": msg,
                                "output": {},
                            },
                        )
                    except Exception as cb_e:
                        log.error(f"generate-resume-content error callback failed: {cb_e}")

            schedule_background_task(
                _resume_content_bg(),
                "generate_resume_content",
                timeout_seconds=_resume_content_timeout,
            )
            log.info(f"Exit point #1 /generate-resume-content | 202 Accepted | uid={uid}")
            return JSONResponse(
                status_code=202,
                content={
                    "status": "processing",
                    "message": "Resume content generation started; results will be sent to callback_url",
                    "uid": uid,
                    "tenant_id": tenant_id,
                    "callback_url": callback_url,
                    "node": "resume_content_generator",
                },
            )

        result = await resume_content_generator_agent(
            uid=uid,
            tenant_id=tenant_id,
            session_id=session_id,
        )

        log.info(f"Exit point #1 /generate-resume-content | Success | uid={uid}")
        log_security_event(
            event="api_access",
            severity="INFO",
            source_ip=client_ip,
            action="generate_resume_content",
            result="success",
            correlation_id=corr_id,
        )

        return JSONResponse(content=dict(result))
        
    except Exception as e:
        log.error(f"Exit point #2 /generate-resume-content | Error: {str(e)}")
        error_log.error(traceback.format_exc())
        log_security_event(
            event="api_access",
            severity="ERROR",
            source_ip=client_ip,
            action="generate_resume_content",
            result="failed",
            correlation_id=corr_id
        )
        
        # Return error response in the same format as the agent
        if isinstance(e, HTTPException):
            raise
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "processing_time_seconds": 0.0
            }
        )


# Novu Notification Service Integration
_novu_notification_service: Optional[Any] = None
_novu_client: Optional[Any] = None
_idempotency_cache: Optional[Any] = None


@app.get("/events")
async def list_notification_events():
    """List all Node-accessible notification events"""
    try:
        from novu_notification_service import EventRegistry
        return {
            "events": EventRegistry.list_events(node_accessible_only=True),
            "count": len(EventRegistry.list_events(node_accessible_only=True))
        }
    except Exception as e:
        log.error(f"Failed to list events: {e}")
        return {"events": {}, "count": 0, "error": "Notification service not available"}


@app.get("/notifications/status")
async def notification_service_status():
    """Check notification service initialization status"""
    import os
    return {
        "service_initialized": _novu_notification_service is not None,
        "novu_api_key_set": bool(os.getenv("NOVU_SECRET_KEY")),
        "novu_api_key_length": len(os.getenv("NOVU_SECRET_KEY", "")),
        "client_initialized": _novu_client is not None,
        "cache_initialized": _idempotency_cache is not None
    }


@app.post("/notifications")
async def trigger_notification(request: Request):
    """
    HTTP endpoint for Node-initiated notification events.
    
    Node.js backend calls this endpoint to trigger notifications via Novu.
    This is an internal service-to-service API, not exposed publicly.
    """
    if not _novu_notification_service:
        raise HTTPException(status_code=503, detail="Notification service not initialized")
    
    try:
        from novu_notification_service import (
            EventRegistry,
            EventSource,
            NodeTriggerRequest,
            EventTriggerResponse
        )
        
        body = await request.json()
        node_request = NodeTriggerRequest(**body)
        
        event_config = EventRegistry.get(node_request.event)
        if not event_config:
            raise HTTPException(status_code=400, detail=f"Unknown event: {node_request.event}")
        
        # Event ownership validation (CRITICAL)
        if event_config.source == EventSource.PYTHON_ONLY:
            raise HTTPException(
                status_code=403,
                detail=f"Event '{node_request.event}' is Python-only and cannot be triggered via HTTP"
            )
        
        # Trigger event with idempotency protection
        result = await _novu_notification_service.trigger_event(
            event_name=node_request.event,
            subscriber_id=node_request.subscriber_id,
            email=node_request.email,
            first_name=node_request.first_name,
            last_name=node_request.last_name,
            payload=node_request.payload,
            source="node",
            idempotency_key=node_request.idempotency_key
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
        log.error(f"Notification validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error triggering notification: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to trigger event: {str(e)}")

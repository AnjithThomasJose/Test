"""
Groq Resume Parser (relaxed JSON parser)

Uses a relaxed Pydantic schema + JsonOutputParser for robust extraction,
then coerces into the strict StructuredResumeOutput for downstream compatibility.
"""

from __future__ import annotations

import json
import logging
import re
import time
import hashlib
import asyncio
import uuid
import os
from typing import Any, Dict, List, Optional
from collections import deque
from datetime import datetime

from langsmith.run_helpers import traceable

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Local project imports
from settings import settings
from agents.resume_schema import StructuredResumeOutput
from agents.resume_schema import RelaxedResumeOutput, coerce_relaxed_to_structured
from utils.resume_utils import download_resume_text, download_resume_text_async
from utils.normalize_text import normalize
from core.utils import _calculate_processing_time, _create_error_response, run_blocking_io


def _strip_proficiency_from_skills(resume_data: dict) -> dict:
    """Return a copy of resume_data with skills stripped to just SkillName."""
    data = dict(resume_data)
    skills = data.get("skills")
    if isinstance(skills, list):
        data["skills"] = [
            {"SkillName": s.get("SkillName") or s.get("name") or s.get("skill") or ""}
            for s in skills if isinstance(s, dict)
        ]
    return data


async def _insert_resume_after_parse(state: Dict[str, Any], result: Dict[str, Any], tenant_id: str) -> None:
    """Insert resume into ChromaDB immediately after Groq parse (pure Groq output, no proficiency data)."""
    uid = state.get("uid") or (state.get("body") or {}).get("uid")
    if not uid:
        log.debug("insert_resume skipped: no uid in state or body")
        return
    resume_data = result.get("structured_resume") or result
    if not isinstance(resume_data, dict):
        return
    try:
        from chroma import insert_resume
        clean_data = _strip_proficiency_from_skills(resume_data)
        await run_blocking_io(insert_resume, uid, clean_data, {"uid": uid}, tenant_id=tenant_id)
        log.debug(f"✅ ChromaDB insert_resume completed for {uid} (from groq_resume_parser)")
    except Exception as e:
        log.warning(f"⚠️ insert_resume after groq parse failed for {uid}: {e}")


# Keys that belong to the parsed resume object (flat Groq output). Downstream agents
# (interest_filler, resume_summary, etc.) read state["structured_resume"] — LangGraph
# merge_dicts only updates that key when we emit it explicitly.
_STRUCTURED_RESUME_KEYS = (
    "name",
    "professional_summary",
    "contact_details",
    "education",
    "work_experience",
    "skills",
    "certifications",
    "projects",
    "extras",
    "professional_affiliations",
    "total_experience_years",
    "candidate_domains",
)


def _flat_resume_fields_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract resume-shaped fields from a parser result dict (excluding meta keys)."""
    return {k: result[k] for k in _STRUCTURED_RESUME_KEYS if k in result and result[k] is not None}


def _flat_resume_has_useful_content(flat: Dict[str, Any]) -> bool:
    if not flat:
        return False
    return bool(
        flat.get("name")
        or flat.get("Name")
        or flat.get("work_experience")
        or flat.get("experience")
        or flat.get("education")
        or flat.get("skills")
        or flat.get("professional_summary")
        or flat.get("projects")
    )


def _ensure_structured_resume_on_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Attach nested structured_resume when Groq only merged flat keys into graph state."""
    existing = result.get("structured_resume")
    if isinstance(existing, dict) and existing and _flat_resume_has_useful_content(existing):
        return result
    flat = _flat_resume_fields_from_result(result)
    if not _flat_resume_has_useful_content(flat):
        return result
    return {**result, "structured_resume": flat}


def _notify_resume_parse_failed(state: Dict[str, Any], error_reason: str) -> None:
    """Fire resume_parse_failed Novu workflow (non-blocking). No-op if notification service unavailable.
    
    Issue 4.4: Uses schedule_background_task for proper exception logging.
    """
    try:
        from novu_notification_service import trigger_resume_parse_failed
        from core.background_tasks import schedule_background_task
        schedule_background_task(
            trigger_resume_parse_failed(state, error_reason),
            "notify_resume_parse_failed"
        )
    except Exception:
        pass  # Non-critical: don't fail if notification service unavailable
from core.security import validate_tenant_id, sanitize_text_for_llm
from core.logging_helpers import create_log_context, AgentLogger, log_agent_completion
from core.config import get_agent_config
from core.observability import TaskMetrics, log_task_metrics
from utils.experience_utils import calculate_total_experience_from_models
# from evaluation.integration import evaluate_agent

# TEMPORARY: Disabled agent_evaluator
# from evaluation.agent_evaluator import evaluate_agent


log = logging.getLogger(__name__)


# ============================================================================
# PERFORMANCE OPTIMIZATION: Caching System
# ============================================================================

class ResumeParserCache:
    """Thread-safe cache for parsed resume results with TTL and LRU eviction."""
    
    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, tuple[Dict, float]] = {}
        self.access_times: Dict[str, float] = {}
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
    
    def _generate_cache_key(self, resume_text: str, tenant_id: str) -> str:
        """Generate stable hash from resume content and tenant."""
        # Use first 500 chars + length + hash of full content for stability
        content_hash = hashlib.sha256(resume_text.encode()).hexdigest()[:16]
        key_data = f"{tenant_id}:{len(resume_text)}:{content_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get(self, resume_text: str, tenant_id: str) -> Optional[Dict]:
        """Get cached result if available and fresh (thread-safe)."""
        cache_key = self._generate_cache_key(resume_text, tenant_id)
        
        async with self._lock:
            if cache_key in self.cache:
                result, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.ttl_seconds:
                    # Update access time for LRU
                    self.access_times[cache_key] = time.time()
                    self.hits += 1
                    return result
                else:
                    # Expired, remove it
                    del self.cache[cache_key]
                    self.access_times.pop(cache_key, None)
            
            self.misses += 1
            return None
    
    async def set(self, resume_text: str, tenant_id: str, result: Dict):
        """Store parsed result (thread-safe with LRU eviction)."""
        cache_key = self._generate_cache_key(resume_text, tenant_id)
        
        async with self._lock:
            # LRU eviction if cache full
            if len(self.cache) >= self.max_entries:
                # Remove oldest entry
                if self.access_times:
                    oldest_key = min(self.access_times.keys(), 
                                   key=lambda k: self.access_times[k])
                    del self.cache[oldest_key]
                    del self.access_times[oldest_key]
            
            self.cache[cache_key] = (result, time.time())
            self.access_times[cache_key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache),
            "max_size": self.max_entries
        }

# Global cache instance (singleton pattern)
_parser_cache = ResumeParserCache(max_entries=1000, ttl_seconds=3600)


# ============================================================================
# PERFORMANCE OPTIMIZATION: Latency Tracker
# ============================================================================

class LatencyTracker:
    """Track latency metrics for production monitoring."""
    
    def __init__(self, window_size: int = 1000):
        self.metrics: deque = deque(maxlen=window_size)
        self._lock = asyncio.Lock()
    
    async def record(self, method: str, total_time: float, 
                    timing_breakdown: Optional[Dict[str, float]] = None):
        """Record latency metrics."""
        async with self._lock:
            self.metrics.append({
                "method": method,
                "total_time": total_time,
                "timing_breakdown": timing_breakdown or {},
                "timestamp": time.time()
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate percentile statistics."""
        if not self.metrics:
            return {}
        
        with self._lock:
            metrics_list = list(self.metrics)
        
        total_times = [m["total_time"] for m in metrics_list]
        total_times.sort()
        
        n = len(total_times)
        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        
        # Method distribution
        method_dist = {}
        for m in metrics_list:
            method = m["method"]
            method_dist[method] = method_dist.get(method, 0) + 1
        
        # Stage breakdown (if available)
        stage_times = {}
        for m in metrics_list:
            breakdown = m.get("timing_breakdown", {})
            for stage, duration in breakdown.items():
                if stage not in stage_times:
                    stage_times[stage] = []
                stage_times[stage].append(duration)
        
        stage_stats = {}
        for stage, times in stage_times.items():
            times.sort()
            n_stage = len(times)
            stage_stats[stage] = {
                "p50": times[int(n_stage * 0.50)] if n_stage > 0 else 0,
                "p95": times[int(n_stage * 0.95)] if n_stage > 0 else 0,
                "p99": times[int(n_stage * 0.99)] if n_stage > 0 else 0,
            }
        
        return {
            "count": n,
            "total_latency": {
                "p50": total_times[p50_idx] if p50_idx < n else 0,
                "p95": total_times[p95_idx] if p95_idx < n else 0,
                "p99": total_times[p99_idx] if p99_idx < n else 0,
                "max": max(total_times) if total_times else 0,
                "min": min(total_times) if total_times else 0,
            },
            "method_distribution": method_dist,
            "stage_breakdown": stage_stats
        }

# Global latency tracker
_latency_tracker = LatencyTracker(window_size=1000)


# Config with safe defaults (preserve existing behavior)
config = get_agent_config("groq_resume_parser")
GROQ_TIMEOUT_SECONDS = getattr(config, "timeout_seconds", 120)
GROQ_MAX_RETRIES = getattr(config, "llm_retry_attempts", 3)
GROQ_MAX_COMPLETION_TOKENS = getattr(config, "max_completion_tokens", 8000)
RESUME_LLM_INPUT_MAX_CHARS = getattr(config, "max_prompt_chars", 20000)
MIN_CLEAN_TEXT_CHARS = getattr(config, "min_clean_text_chars", 800)
TRIM_HEAD_CHARS = getattr(config, "trim_head_chars", 2000)
TRIM_TAIL_CHARS = getattr(config, "trim_tail_chars", 13000)


def _create_groq_model() -> ChatGroq:
    """Create Groq chat model with safe defaults.
    
    Attempts to enable JSON-mode if supported by the selected model.
    """
    model_kwargs = {
        "top_p": 0.0,
        "max_completion_tokens": GROQ_MAX_COMPLETION_TOKENS,
    }
    
    # Opt-in JSON response format where available
    try:
        model_kwargs["response_format"] = {"type": "json_object"}
    except Exception:
        # Some backends/models may not accept this; ignore gracefully
        pass
    
    return ChatGroq(
        model=settings.GROQ_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0.0,
        max_retries=GROQ_MAX_RETRIES,
        timeout=GROQ_TIMEOUT_SECONDS,
        model_kwargs=model_kwargs,
    )


def clean_for_llm(raw_text: str) -> str:
    """Normalize text for the LLM, preserving bullets.
    
    - Remove emojis/control chars.
    - Normalize whitespace.
    - Convert diverse bullets to a single "- " marker.
    - Deduplicate consecutive identical lines.
    """
    s = normalize(raw_text or "")
    
    # Remove emojis (most BMP+extended ranges) and nulls; keep bullets
    s = re.sub(r"[\U0001F300-\U0001F9FF]", "", s)
    s = re.sub(r"[\U0001FA00-\U0001FAFF]", "", s)
    s = re.sub(r"[\u0000]", "", s)
    
    # Drop obvious null-like tokens but keep content
    null_tokens = {"null", "n/a", "na", "none"}
    cleaned_lines: List[str] = []
    for line in s.splitlines():
        words = [w for w in line.split() if w.lower() not in null_tokens]
        cleaned_lines.append(" ".join(words))
    s = "\n".join(cleaned_lines).strip()
    
    # Normalize bullets (do not remove)
    bullet_variants = ("• ", "▸ ", "► ", "· ")
    lines = []
    for line in s.splitlines():
        for b in bullet_variants:
            if line.startswith(b):
                line = "- " + line[len(b):]
                break
        lines.append(line)
    s = "\n".join(lines)
    
    # Deduplicate consecutive identical lines
    deduped: List[str] = []
    last_line = None
    for line in s.splitlines():
        if line != last_line:
            deduped.append(line)
        last_line = line
    s = "\n".join(deduped)
    
    return s[:RESUME_LLM_INPUT_MAX_CHARS]


# ---------------- Text cleaning & probing ----------------

def quick_probe(text: str) -> Dict[str, Any]:
    """Cheap signals to adapt prompt size/flow without heavy parsing."""
    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.search(r"(?:\+?\d[\d\s\-]{7,}\d)", text)
    headings = len(re.findall(
        r"(?im)^(experience|work experience|education|skills|summary|professional summary)\b",
        text
    ))
    return {
        "has_email": bool(email),
        "has_phone": bool(phone),
        "headings": headings,
        "chars": len(text)
    }


# ============================================================================
# FALLBACK DOMAIN CLASSIFIER
# Keyword-based domain classification when LLM fails to return candidate_domains
# ============================================================================

_DOMAIN_KEYWORDS = {
    "software_engineering": [
        "software", "developer", "programmer", "full stack", "fullstack", "frontend",
        "backend", "web developer", "mobile developer", "react", "angular", "vue",
        "python", "java", "javascript", "typescript", "c++", "c#", ".net", "node",
        "django", "flask", "spring", "microservices", "api", "rest", "graphql",
        "software engineer", "sde", "swe", "devops", "ci/cd", "git",
    ],
    "data_analytics": [
        "data analyst", "business analyst", "analytics", "tableau", "power bi",
        "excel", "sql", "reporting", "dashboards", "visualization", "bi ",
        "business intelligence", "data warehouse", "etl",
    ],
    "data_science": [
        "data scientist", "data science", "machine learning", "deep learning",
        "statistics", "r programming", "pandas", "numpy", "scipy", "jupyter",
        "predictive model", "regression", "classification", "clustering",
    ],
    "ai_ml": [
        "artificial intelligence", " ai ", "machine learning", " ml ", "nlp",
        "natural language", "computer vision", "neural network", "tensorflow",
        "pytorch", "llm", "generative ai", "chatbot", "reinforcement learning",
    ],
    "technical_infrastructure": [
        "infrastructure", "sysadmin", "system admin", "network", "cloud",
        "aws", "azure", "gcp", "linux", "windows server", "virtualization",
        "docker", "kubernetes", "terraform", "ansible", "it operations",
        "database admin", "dba", "oracle", "data center", "server",
    ],
    "traditional_engineering": [
        "mechanical engineer", "civil engineer", "electrical engineer",
        "chemical engineer", "structural", "cad", "autocad", "solidworks",
        "manufacturing engineer", "process engineer", "quality engineer",
    ],
    "management": [
        "manager", "management", "director", "vp ", "vice president", "head of",
        "team lead", "lead", "supervisor", "coordinator", "program manager",
        "project manager", "pmp", "scrum master", "agile", "stakeholder",
    ],
    "business_strategic": [
        "strategy", "strategic", "business development", "growth", "partnerships",
        "corporate development", "mba", "business planning", "market analysis",
    ],
    "sales_business": [
        "sales", "account executive", "business development", "revenue",
        "quota", "pipeline", "crm", "salesforce", "hubspot", "cold call",
        "lead generation", "client acquisition",
    ],
    "marketing": [
        "marketing", "seo", "sem", "social media", "content", "brand",
        "digital marketing", "campaign", "advertising", "google ads",
        "email marketing", "copywriting", "market research",
    ],
    "hr": [
        "human resources", " hr ", "recruitment", "talent acquisition",
        "employee relations", "payroll", "benefits", "onboarding",
        "performance management", "hris", "workday", "people operations",
    ],
    "finance": [
        "finance", "financial", "accounting", "cpa", "cfa", "audit",
        "budgeting", "forecasting", "investment", "banking", "tax",
        "revenue", "profit", "loss", "bookkeeping", "accounts payable",
    ],
    "legal": [
        "legal", "attorney", "lawyer", "paralegal", "compliance",
        "contract", "litigation", "regulatory", "intellectual property",
    ],
    "healthcare": [
        "healthcare", "medical", "nursing", "nurse", "physician", "doctor",
        "clinical", "patient", "hospital", "pharmaceutical", "pharmacy",
        "health", "ehr", "hipaa", "diagnosis",
    ],
    "ui_ux_design": [
        "ui design", "ux design", "user experience", "user interface",
        "figma", "sketch", "wireframe", "prototype", "usability",
        "interaction design", "design system",
    ],
    "creative_design": [
        "graphic design", "creative", "photoshop", "illustrator",
        "indesign", "animation", "video editing", "photography",
        "branding", "visual design",
    ],
    "operations": [
        "operations", "process improvement", "lean", "six sigma",
        "kaizen", "operational excellence", "sla", "kpi", "service delivery",
        "itil", "incident management", "change management",
    ],
    "customer_support": [
        "customer support", "customer service", "help desk", "technical support",
        "troubleshooting", "ticketing", "service desk",
    ],
    "customer_success": [
        "customer success", "account management", "client management",
        "retention", "renewal", "upsell", "csm",
    ],
    "consulting": [
        "consultant", "consulting", "advisory", "professional services",
        "solution architect", "implementation",
    ],
    "education": [
        "teacher", "professor", "instructor", "tutor", "curriculum",
        "teaching", "academic", "university", "school", "edtech",
    ],
    "construction": [
        "construction", "building", "site manager", "foreman", "contractor",
        "blueprint", "concrete", "plumbing", "electrical work",
    ],
    "hospitality": [
        "hotel", "hospitality", "restaurant", "food service", "chef",
        "catering", "front desk", "concierge", "tourism",
    ],
    "retail": [
        "retail", "store manager", "merchandising", "inventory",
        "point of sale", "pos", "cashier", "e-commerce", "shopify",
    ],
    "manufacturing": [
        "manufacturing", "production", "assembly", "factory",
        "quality control", "inspection", "cnc", "machining",
    ],
    "supply_chain": [
        "supply chain", "logistics", "procurement", "warehouse",
        "inventory management", "shipping", "distribution", "sourcing",
    ],
    "security": [
        "cybersecurity", "information security", "infosec", "soc",
        "penetration testing", "vulnerability", "firewall", "siem",
        "security engineer", "security analyst",
    ],
    "admin": [
        "administrative", "admin assistant", "office manager", "receptionist",
        "scheduling", "filing", "data entry", "clerical",
    ],
    "science": [
        "research scientist", "laboratory", "lab ", "phd", "postdoc",
        "biology", "chemistry", "physics", "biotech", "genomics",
    ],
}


def _classify_domains_fallback(parsed_data: Dict[str, Any]) -> List[str]:
    """Keyword-based domain classification fallback when LLM doesn't return candidate_domains.
    
    Scores each domain by counting keyword hits across job titles, skills, and summary,
    then returns the top 3.
    """
    text_parts = []
    
    for exp in (parsed_data.get("work_experience") or parsed_data.get("experience") or []):
        if isinstance(exp, dict):
            text_parts.append(exp.get("job_title", ""))
            for r in (exp.get("responsibilities") or []):
                text_parts.append(str(r))
    
    for skill in (parsed_data.get("skills") or []):
        if isinstance(skill, dict):
            text_parts.append(skill.get("SkillName", "") or skill.get("name", ""))
        elif isinstance(skill, str):
            text_parts.append(skill)
    
    text_parts.append(parsed_data.get("professional_summary") or "")
    
    for cert in (parsed_data.get("certifications") or []):
        if isinstance(cert, dict):
            text_parts.append(cert.get("certification_name", ""))
    
    combined = " ".join(text_parts).lower()
    
    if len(combined.strip()) < 10:
        return ["other", "other", "other"]
    
    scores: Dict[str, float] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0.0
        for kw in keywords:
            count = combined.count(kw.lower())
            if count > 0:
                score += count * (len(kw.split()) ** 0.5)
            
        if score > 0:
            scores[domain] = score
    
    if not scores:
        return ["other", "other", "other"]
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top3 = [d for d, _ in ranked[:3]]
    
    while len(top3) < 3:
        top3.append("other")
    
    return top3


# Preserve original entrypoint name for callers
@traceable
async def groq_resume_parser_agent(state: Dict[str, Any], tenant_id: str = "default_tenant") -> Dict[str, Any]:
    # Default to relaxed JsonOutputParser-based flow for higher robustness.
    return await groq_resume_parser_agent_relaxed(state, tenant_id)


# --- Relaxed JSON-output agent using JsonOutputParser ---
# Load prompt from shared file (single source of truth for Promptfoo + groq_resume_parser)
def _load_resume_parse_system_prompt() -> str:
    _prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "resume_parse_system_prompt.txt")
    with open(_prompt_path, "r", encoding="utf-8") as f:
        return f.read()


RELAXED_SYSTEM_PROMPT = _load_resume_parse_system_prompt()



_relaxed_prompt = ChatPromptTemplate.from_messages([
    ("system", RELAXED_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")),
    ("human", "{resume_text}")
])


async def groq_resume_parser_agent_relaxed(state: Dict[str, Any], tenant_id: str = "default_tenant") -> Dict[str, Any]:
    actual_tenant_id = state.get("tenant_id", tenant_id) or "default_tenant"
    if actual_tenant_id == "default":
        actual_tenant_id = "default_tenant"

    start_time = time.time()
    log_ctx = create_log_context("groq_resume_parser_relaxed", actual_tenant_id)
    
    # PERFORMANCE: Initialize timing breakdown (zero overhead)
    timing_breakdown = {
        "text_extraction": 0.0,
        "text_cleaning": 0.0,
        "llm_call": 0.0,
        "json_parsing": 0.0,
        "total": 0.0
    }

    try:
        if not validate_tenant_id(actual_tenant_id):
            _notify_resume_parse_failed(state, "Invalid tenant identification")
            return _create_error_response("Invalid tenant identification", "groq_resume_parser_relaxed")

        # Check if we already have structured_resume (for stored resume flow)
        existing_structured_resume = state.get("structured_resume")
        has_stored_resume = state.get("has_stored_resume", False)
        
        if has_stored_resume and existing_structured_resume and isinstance(existing_structured_resume, dict):
            # We already have structured_resume, just pass it through
            log.debug(f"✅ Using existing structured_resume from stored resume flow")
            log.info("Using existing structured_resume from stored resume flow")
            processing_time = _calculate_processing_time(start_time)
            timing_breakdown["total"] = processing_time
            
            result = {
                "structured_resume": existing_structured_resume,
                "name": existing_structured_resume.get("Name") or existing_structured_resume.get("name", ""),
                "contact_details": existing_structured_resume.get("ContactDetails") or existing_structured_resume.get("contact_details", {}),
                "education": existing_structured_resume.get("education", []),
                "work_experience": existing_structured_resume.get("experience") or existing_structured_resume.get("work_experience", []),
                "skills": existing_structured_resume.get("skills", []),
                "certifications": existing_structured_resume.get("certifications", []),
                "projects": existing_structured_resume.get("projects", []),
                "extras": existing_structured_resume.get("extras", []),
                "professional_affiliations": existing_structured_resume.get("professional_affiliations", []),
                "total_experience_years": existing_structured_resume.get("total_experience_years", 0.0),
                "professional_summary": existing_structured_resume.get("professional_summary", ""),
                "candidate_domains": existing_structured_resume.get("candidate_domains", [])[:3],
                "processing_time_seconds": processing_time,
                "_timing_breakdown": timing_breakdown
            }
            
            log_agent_completion(
                log_ctx,
                {
                    "success": True,
                    "name": result.get("name"),
                    "education_count": len(result.get("education", [])),
                    "experience_count": len(result.get("work_experience", [])),
                    "skills_count": len(result.get("skills", [])),
                    "method": "stored_resume_passthrough",
                },
                "groq_resume_parser_relaxed",
                processing_time,
            )
            
            # PERFORMANCE: Record metrics
            await _latency_tracker.record("stored_resume_passthrough", processing_time, timing_breakdown)
            
            # PERFORMANCE: Log to observability system
            metrics = TaskMetrics(
                task_id=str(uuid.uuid4()),
                agent_name="groq_resume_parser",
                start_time=start_time,
                latency_ms=processing_time * 1000,
                cache_hit=False,
                success=True,
                metadata={"method": "stored_resume_passthrough", "timing": timing_breakdown}
            )
            log_task_metrics(metrics)
            # TEMPORARY: Disabled agent_evaluator
            # await evaluate_agent("groq_resume_parser", state, result, actual_tenant_id)  # ← ADD THIS
            # Preserve job_id and body for compare-candidate-job flow
            body = state.get("body") or {}
            if body.get("request_type") == "candidate_job_match" or body.get("job_id"):
                result["body"] = body
                job_id = state.get("job_id") or body.get("job_id")
                if job_id:
                    result["job_id"] = job_id
            # Insert into ChromaDB immediately after parse (pure Groq output)
            await _insert_resume_after_parse(state, result, actual_tenant_id)
            return result
        
        # Check for resume_text first (prioritize direct text input)
        resume_text_from_state = state.get("resume_text", "")
        resume_url = state.get("resume_url")
        

        
        # FALLBACK: Use existing text extraction + Groq parsing flow
        # Track whether we're using pre-extracted text or extracting now
        using_pre_extracted_text = False
        
        if resume_text_from_state and resume_text_from_state.strip():
            # Use resume_text directly from state (already extracted in dispatcher)
            log.debug(f"📄 Using resume_text from state ({len(resume_text_from_state)} characters) - skipping re-extraction")
            raw_text = resume_text_from_state.strip()
            timing_breakdown["text_extraction"] = 0.0  # No extraction needed
            using_pre_extracted_text = True
        elif resume_url:
            # PERFORMANCE: Track text extraction time
            extraction_start = time.time()
            log.info(f"📄 Downloading and extracting resume from: {resume_url}")
            raw_text = await download_resume_text_async(resume_url, "resume")
            timing_breakdown["text_extraction"] = time.time() - extraction_start
            
            if not raw_text or len(raw_text.strip()) < 50:
                _notify_resume_parse_failed(state, "Failed to extract meaningful text from resume")
                return _create_error_response("Failed to extract meaningful text from resume", "groq_resume_parser_relaxed")
        else:
            # Neither resume_text nor resume_url provided
            _notify_resume_parse_failed(state, "No resume URL or resume_text provided")
            return _create_error_response("No resume URL or resume_text provided", "groq_resume_parser_relaxed")
        
        # PERFORMANCE: Track text cleaning time
        cleaning_start = time.time()
        cleaned = clean_for_llm(raw_text)
        timing_breakdown["text_cleaning"] = time.time() - cleaning_start
        
        # PERFORMANCE: Check cache before expensive LLM call
        cached_result = await _parser_cache.get(cleaned, actual_tenant_id)
        if cached_result:
            processing_time = _calculate_processing_time(start_time)
            timing_breakdown["total"] = processing_time
            log.info("✅ Cache hit - returning cached parsed resume")
            log.debug(f"✅ Cache hit - returning cached result (saved ~{timing_breakdown.get('llm_call', 5):.1f}s)")
            
            result = {
                **cached_result,
                "processing_time": processing_time,
                "processing_time_seconds": processing_time,
                "method": "groq_relaxed_json_parser_cached",
                "_timing_breakdown": timing_breakdown,
                "_meta": {
                    "status": "success",
                    "method": "groq_relaxed_json_parser_cached",
                    "processing_time": processing_time,
                    "cache_hit": True
                }
            }
            
            log_agent_completion(
                log_ctx,
                {
                    "success": True,
                    "name": result.get("name"),
                    "education_count": len(result.get("education", [])),
                    "experience_count": len(result.get("work_experience", [])),
                    "skills_count": len(result.get("skills", [])),
                    "method": "groq_relaxed_json_parser_cached",
                },
                "groq_resume_parser_relaxed",
                processing_time,
            )
            
            # PERFORMANCE: Record metrics
            await _latency_tracker.record("cached", processing_time, timing_breakdown)
            
            # PERFORMANCE: Log to observability system
            metrics = TaskMetrics(
                task_id=str(uuid.uuid4()),
                agent_name="groq_resume_parser",
                start_time=start_time,
                latency_ms=processing_time * 1000,
                cache_hit=True,
                success=True,
                metadata={"method": "cached", "timing": timing_breakdown}
            )
            log_task_metrics(metrics)
            # TEMPORARY: Disabled agent_evaluator
            # await evaluate_agent("groq_resume_parser", state, result, actual_tenant_id)  # ← ADD THIS
            # Preserve job_id and body for compare-candidate-job flow
            body = state.get("body") or {}
            if body.get("request_type") == "candidate_job_match" or body.get("job_id"):
                result["body"] = body
                job_id = state.get("job_id") or body.get("job_id")
                if job_id:
                    result["job_id"] = job_id
            # Insert into ChromaDB immediately after parse (pure Groq output)
            result = _ensure_structured_resume_on_result(result)
            await _insert_resume_after_parse(state, result, actual_tenant_id)
            return result
        
        # Guardrail: Check if resume is valid before processing
        # If resume is valid, proceed regardless of character count
        is_valid_resume = state.get("is_valid_resume", False)
        
        if len(cleaned) < MIN_CLEAN_TEXT_CHARS:
            if not is_valid_resume:
                # Only reject if resume is invalid AND text is too short
                _notify_resume_parse_failed(state, "Upstream extraction too short; route to Reflow/OCR path")
                return _create_error_response(
                    "Upstream extraction too short; route to Reflow/OCR path",
                    "groq_resume_parser_relaxed",
                )
            else:
                # Resume is valid but text is short - log warning but proceed
                log.warning(
                    f"⚠️ Resume text is short ({len(cleaned)} chars < {MIN_CLEAN_TEXT_CHARS} min), "
                    f"but resume is valid - proceeding with parsing"
                )
        
        # Optional probe for diagnostics (but don't trim - we want all sections)
        probe = quick_probe(cleaned)
        sanitized = sanitize_text_for_llm(cleaned, redact_pii_flag=False, filter_injection_flag=True)
        
        # Don't trim text - we need all sections for complete extraction
        # The LLM can handle longer inputs, and we want to capture everything
        # Only log if text is very long for monitoring purposes
        if probe["chars"] > RESUME_LLM_INPUT_MAX_CHARS:
            log.warning(f"⚠️ Resume text is long ({probe['chars']} chars) but keeping all content for complete extraction")
        
        # More accurate log message based on whether we used pre-extracted text
        if using_pre_extracted_text:
            log.debug(f"✅ Using pre-extracted text ({len(raw_text)} chars, cleaned={len(cleaned)}) - ready for parsing")
        else:
            log.info(f"✅ Extracted {len(raw_text)} characters from resume (clean={len(cleaned)})")

        # Parse JSON manually for better control and error handling
        # JsonOutputParser with pydantic_object can sometimes return unexpected types (like float)
        model = _create_groq_model()
        
        # PERFORMANCE: Optimize retry timeouts - faster failure detection
        # Try up to 2 times with progressively stricter prompts
        max_attempts = 2
        json_str = None
        raw_text_out = None
        data = None
        
        for attempt in range(max_attempts):
            try:
                # PERFORMANCE: Use shorter timeout for retries (faster failure)
                timeout_duration = GROQ_TIMEOUT_SECONDS if attempt == 0 else 30
                
                if attempt == 0:
                    # First attempt: use the standard prompt with full timeout
                    chain = _relaxed_prompt | model
                    llm_start = time.time()
                    raw_response = await asyncio.wait_for(
                        chain.ainvoke({"resume_text": sanitized}),
                        timeout=timeout_duration
                    )
                    timing_breakdown["llm_call"] = time.time() - llm_start
                else:
                    # Second attempt: use an even stricter prompt with shorter timeout
                    strict_human_template = (
                        "Extract the resume information below into JSON format.\n\n"
                        "RESUME TEXT:\n{resume_text}\n\n"
                        "OUTPUT INSTRUCTIONS:\n"
                        "- You MUST respond with ONLY a valid JSON object.\n"
                        "- Start your response with {{ and end with }}.\n"
                        "- DO NOT include any text, explanations, markdown, or code blocks.\n"
                        "- DO NOT repeat any text from the resume.\n"
                        "- Your ENTIRE response must be parseable as JSON.\n\n"
                        "The JSON schema is defined in the system message above."
                    )
                    strict_chain = ChatPromptTemplate.from_messages([
                        ("system", RELAXED_SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")),
                        ("human", strict_human_template)
                    ]) | model
                    llm_start = time.time()
                    raw_response = await asyncio.wait_for(
                        strict_chain.ainvoke({"resume_text": sanitized}),
                        timeout=timeout_duration
                    )
                    timing_breakdown["llm_call"] += time.time() - llm_start
                
                raw_text_out = getattr(raw_response, "content", str(raw_response))
                
                # Safer logging: avoid dumping raw PII-heavy LLM content
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        f"Attempt {attempt + 1} - LLM response length: {len(raw_text_out or '')}"
                    )
                
                # Log warning if response doesn't start with JSON
                trimmed_response = raw_text_out.strip()
                if not trimmed_response.startswith("{"):
                    log.warning(
                        f"Attempt {attempt + 1} - Response does not start with JSON. "
                        f"First 200 chars: {trimmed_response[:200]}"
                    )
                    # Check if it's clearly not JSON (no JSON structure at all)
                    if "{" not in trimmed_response:
                        log.warning(f"Attempt {attempt + 1} - No JSON structure found in response")
                        if attempt < max_attempts - 1:
                            continue  # Try again with stricter prompt
                        # Otherwise, still try to extract in case JSON is buried in the response
                
                # PERFORMANCE: Track JSON parsing time
                parsing_start = time.time()
                
                # Try direct extract (handles code-fenced blocks as well)
                json_candidate = _extract_json_from_response(raw_text_out)
                
                if not json_candidate:
                    # Possibly the whole content is raw JSON (no fences)
                    trimmed = (raw_text_out or "").strip()
                    if trimmed.startswith("```json"):
                        trimmed = trimmed[7:]
                    elif trimmed.startswith("```"):
                        trimmed = trimmed[3:]
                    if trimmed.endswith("```"):
                        trimmed = trimmed[:-3]
                    trimmed = trimmed.strip()
                    json_candidate = trimmed
                
                # Validate/repair JSON
                data = _json_try_repair(json_candidate) if json_candidate else None
                
                timing_breakdown["json_parsing"] = time.time() - parsing_start
                if data is not None:
                    log.info(f"Successfully extracted JSON on attempt {attempt + 1}")
                    json_str = json_candidate
                    break
                else:
                    # No JSON found, try next attempt if available
                    if attempt < max_attempts - 1:
                        log.warning(f"Attempt {attempt + 1} failed to extract JSON, retrying with stricter prompt...")
                        continue
                    else:
                        # Last attempt failed
                        error_msg = (
                            f"Failed to extract JSON from LLM response after {max_attempts} attempts. "
                            f"Raw response (first 1000 chars): {raw_text_out[:1000]}"
                        )
                        log.error(error_msg)
                        raise ValueError(error_msg)
                        
            except asyncio.TimeoutError:
                # PERFORMANCE: Faster timeout detection for retries
                if attempt < max_attempts - 1:
                    log.warning(f"Attempt {attempt + 1} timed out after {timeout_duration}s, retrying with shorter timeout...")
                    continue
                else:
                    raise
            except Exception as attempt_err:
                # If this isn't the last attempt, continue to next try
                if attempt < max_attempts - 1:
                    log.warning(f"Attempt {attempt + 1} raised exception: {attempt_err}, retrying...")
                    continue
                else:
                    # Last attempt, re-raise the exception
                    raise
        
        # If we reach here without json_str and data, something went wrong
        if not json_str or data is None:
            error_msg = (
                f"Failed to extract valid JSON from LLM response after {max_attempts} attempts. "
                f"Last raw response length: {len(raw_text_out) if raw_text_out else 0}"
            )
            log.error(error_msg)
            raise ValueError(error_msg)
        
        # Ensure data is a dictionary before unpacking
        if not isinstance(data, dict):
            error_msg = (
                f"Parsed JSON is not a dictionary: {type(data)}. "
                f"Extracted JSON string length: {len(json_str) if json_str else 0}"
            )
            log.error(error_msg)
            raise ValueError(error_msg)

        # # ---------------- NORMALIZE EXPERIENCE KEY ----------------
        if "experience" in data and "work_experience" not in data:
            log.debug("Normalizing 'experience' key to 'work_experience'")
            data["work_experience"] = data.pop("experience")

        # ---------------- NORMALIZE TOTAL EXPERIENCE TYPE ----------------
        if "total_experience_years" in data:
            if isinstance(data["total_experience_years"], (int, float)):
                data["total_experience_years"] = "0 months"

        # ---------------- FALLBACK DOMAIN CLASSIFICATION ----------------
        raw_domains = data.get("candidate_domains")
        if not raw_domains or not isinstance(raw_domains, list) or len(raw_domains) == 0:
            fallback_domains = _classify_domains_fallback(data)
            data["candidate_domains"] = fallback_domains
            log.info(f"🔄 candidate_domains missing from LLM output — fallback classified: {fallback_domains}")
        else:
            log.info(f"✅ LLM returned candidate_domains: {raw_domains[:3]}")

        # ---------------- CREATE RELAXED MODEL ----------------
        relaxed = RelaxedResumeOutput(**data)
        output = relaxed.model_dump()

        # ---------------- EXPERIENCE CALCULATION ----------------
        work_exp = relaxed.work_experience or []

        # Ensure work_exp is always a list
        if not isinstance(work_exp, list):
            log.warning(f"work_experience is not a list: {type(work_exp)}, converting to empty list")
            work_exp = []

        # Calculate total experience (RETURNS STRING)
        total_experience = calculate_total_experience_from_models(work_exp)

        # SINGLE SOURCE OF TRUTH (STRING, NOT FLOAT)
        relaxed.total_experience_years = total_experience

        log.info(f"Calculated total experience: {total_experience}")

        # ---------------- CONVERT TO STRUCTURED OUTPUT ----------------
        structured = coerce_relaxed_to_structured(relaxed)

        output = structured.model_dump(exclude_none=False)

        # DEFENSIVE SET (string-safe)
        output["total_experience_years"] = total_experience

        # DEFENSIVE SET: Ensure candidate_domains survives model_dump
        output["candidate_domains"] = data.get("candidate_domains", [])[:3]

        processing_time = _calculate_processing_time(start_time)
        timing_breakdown["total"] = processing_time
        
        # PERFORMANCE: Cache the result for future use
        await _parser_cache.set(cleaned, actual_tenant_id, output)
        
        log_agent_completion(
            log_ctx,
            {
                "success": True,
                "name": output.get("name"),
                "education_count": len(output.get("education", [])),
                "experience_count": len(output.get("work_experience", [])),
                "skills_count": len(output.get("skills", [])),
                "method": "groq_relaxed_json_parser",
            },
            "llm",
            processing_time,
        )

        log.info(
            f"✅ Groq resume parsing (relaxed) completed in {processing_time:.3f}s — "
            f"{output.get('name')} | edu={len(output.get('education', []))} | "
            f"exp={len(output.get('work_experience', []))} | skills={len(output.get('skills', []))}"
        )
        
        # PERFORMANCE: Record metrics
        await _latency_tracker.record("groq_relaxed_json_parser", processing_time, timing_breakdown)
        
        # PERFORMANCE: Log to observability system
        metrics = TaskMetrics(
            task_id=str(uuid.uuid4()),
            agent_name="groq_resume_parser",
            start_time=start_time,
            latency_ms=processing_time * 1000,
            cache_hit=False,
            success=True,
            metadata={
                "method": "groq_relaxed_json_parser",
                "timing": timing_breakdown,
                "text_extraction_ms": timing_breakdown["text_extraction"] * 1000,
                "text_cleaning_ms": timing_breakdown["text_cleaning"] * 1000,
                "llm_call_ms": timing_breakdown["llm_call"] * 1000,
                "json_parsing_ms": timing_breakdown["json_parsing"] * 1000,
            }
        )
        log_task_metrics(metrics)

        # return {
        #     **output,
        #     "_meta": {
        #         "status": "success",
        #         "method": "groq_relaxed_json_parser",
        #         "processing_time": processing_time,
        #     },
        #     "_timing_breakdown": timing_breakdown
        # }

        final_result = {
            **output,
            "_meta": {
                "status": "success",
                "method": "groq_relaxed_json_parser",
                "processing_time": processing_time,
            },
            "_timing_breakdown": timing_breakdown,
        }
        final_result = _ensure_structured_resume_on_result(final_result)
        # Preserve job_id and body for compare-candidate-job flow
        body = state.get("body") or {}
        if body.get("request_type") == "candidate_job_match" or body.get("job_id"):
            final_result["body"] = body
            job_id = state.get("job_id") or body.get("job_id")
            if job_id:
                final_result["job_id"] = job_id

        # Insert into ChromaDB immediately after parse (pure Groq output)
        await _insert_resume_after_parse(state, final_result, actual_tenant_id)

        return final_result


        # await evaluate_agent("groq_resume_parser", state, result, actual_tenant_id)
    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        timing_breakdown["total"] = processing_time
        err_str = str(e)
        log.error(f"❌ groq_resume_parser_agent_relaxed error: {err_str}")
        log.debug(f"Traceback: {__import__('traceback').format_exc()}")
        AgentLogger.log_error(log_ctx, f"Groq resume parser relaxed failed: {err_str}", processing_time)
        
        # PERFORMANCE: Record error metrics
        metrics = TaskMetrics(
            task_id=str(uuid.uuid4()),
            agent_name="groq_resume_parser",
            start_time=start_time,
            latency_ms=processing_time * 1000,
            cache_hit=False,
            success=False,
            error_code=type(e).__name__,
            metadata={"method": "error", "timing": timing_breakdown, "error": err_str[:200]}
        )
        log_task_metrics(metrics)
        _notify_resume_parse_failed(state, f"Groq resume parser relaxed error: {err_str}")
        return _create_error_response(f"Groq resume parser relaxed error: {err_str}", "groq_resume_parser_relaxed")


def _extract_json_from_response(response_text: str) -> Optional[str]:
    """Extract the first balanced JSON object from a string.
    
    Handles code fences and nested braces.
    """
    if not response_text:
        return None
    text = response_text.strip()
    fence_pos = text.find("```")
    
    if fence_pos != -1:
        text = text[fence_pos + 3 :]
        second = text.find("```")
        if second != -1:
            text = text[:second].strip()
    
    start = text.find("{")
    if start == -1:
        return None
    
    brace_count = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start : i + 1].strip()
    
    last = text.rfind('}')
    if last > start:
        return text[start : last + 1].strip()
    return None


def _json_try_repair(s: str) -> Optional[dict]:
    """Attempt to parse JSON; apply minimal repairs on failure.
    
    NOTE: Keep repairs conservative to avoid accidental data shifts.
    """
    if not s:
        return None
    
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    
    s2 = s.replace("\n", " ").replace("\t", " ")
    # Drop trailing commas before closing braces/brackets
    s2 = re.sub(r",\s*([}\]])", r"\1", s2)
    # Normalize Python-style booleans/nulls
    s2 = re.sub(r"\bTrue\b", "true", s2)
    s2 = re.sub(r"\bFalse\b", "false", s2)
    s2 = re.sub(r"\bNone\b", "null", s2)
    
    try:
        return json.loads(s2)
    except json.JSONDecodeError:
        return None


# ============================================================================
# PERFORMANCE MONITORING: Helper functions for production metrics
# ============================================================================

def get_resume_parser_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring."""
    return _parser_cache.get_stats()


def get_resume_parser_latency_stats() -> Dict[str, Any]:
    """Get latency statistics for monitoring."""
    return _latency_tracker.get_stats()


def get_resume_parser_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary for production monitoring."""
    return {
        "cache": get_resume_parser_cache_stats(),
        "latency": get_resume_parser_latency_stats()
    }


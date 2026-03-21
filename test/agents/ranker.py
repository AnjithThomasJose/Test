import json
import logging
import asyncio
import re
import time
import hashlib
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
logging.getLogger("langsmith.client").setLevel(logging.CRITICAL)
logging.getLogger("langsmith").setLevel(logging.CRITICAL)

# ✅ Skill extraction utility (moved from skill_matcher to avoid semantic matching dependency)
from agents.skill_utils import extract_primary_skills
from agents.matching_utils import validate_skills_against_candidate
from agents.matching_prompts import (
    ORDERED_ANALYSIS_STEPS_UNIFIED,
    SCORING_GUIDELINES_0_100,
    CRITICAL_RULES,
    SKILLS_MATCH_INSTRUCTIONS,
    normalize_match_score_0_100_to_0_1,
)

from chroma import match_job_description, get_resume_doc, get_resume, parse_resume_from_metadata
from core.model_registry import TaskType
from models.llm_invoker import invoke_llm, invoke_structured_llm
from pydantic import BaseModel, Field
from settings import settings as _settings

from core.gemini_embedding_cache import cached_embed_texts
from core.utils import (
    _mask, _sanitize_text_for_llm, _to_text, _clean_json_text, _scan_balanced_json,
    _safe_json_loads, _extract_json_from_response, _create_error_response, _validate_state_inputs,
    _generate_request_id, _calculate_processing_time
)
from core.config import get_agent_config
from core.security import (
    validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm,
    PII_PATTERNS, INJECTION_FILTERS
)
from core.memory import BaseAgentMemory, get_agent_memory
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion
from utils.callback_storage import store_callback

# Helper function to run CPU-intensive operations in thread pool
async def run_cpu_intensive(func, *args, **kwargs):
    """Run CPU-intensive operations in thread pool to avoid blocking event loop."""
    return await asyncio.to_thread(func, *args, **kwargs)

class Config:
    """Configuration constants for the ranker agent."""
    TOP_K_CANDIDATES = 5
    FUZZY_MATCH_THRESHOLD = 0.75
    # UPDATED: Prioritize skill match, include experience matching
    SKILL_MATCH_WEIGHT = 0.80  # 80% weight to skill match
    SIMILARITY_WEIGHT = 0.10   # 10% weight to vector similarity (tiebreaker)
    EXPERIENCE_WEIGHT = 0.10   # 10% weight to experience match
    LLM_TIMEOUT_SECONDS = 90  
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY_SECONDS = 2  
    RATIONALE_TIMEOUT = 45
    # OPTIMIZATION: Parallel processing and early filtering
    PARALLEL_BATCH_SIZE = 10
    EARLY_FILTER_THRESHOLD = 0.30  # Minimum match score to process (30%)
    LLM_RATIONALE_MIN_MATCH = 0.30
    LLM_RATIONALE_MAX_MATCH = 0.85
    # HIGH PRIORITY: Caching and performance
    SKILL_EXTRACTION_CACHE_TTL = 3600  # 1 hour
    SKILL_EXTRACTION_CACHE_SIZE = 5000
    SKILL_MATCHING_CACHE_TTL = 1800  # 30 minutes
    SKILL_MATCHING_CACHE_SIZE = 10000
    JD_CACHE_TTL = 7200  # 2 hours
    JD_CACHE_SIZE = 100
    # HIGH PRIORITY: Circuit breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
    CIRCUIT_BREAKER_TIMEOUT = 60
    # MEDIUM PRIORITY: Adaptive batching
    ADAPTIVE_BATCH_ENABLED = True
    MIN_BATCH_SIZE = 3
    MAX_BATCH_SIZE = 15
    # FULL DATABASE RANKING
    RANK_FROM_ALL_DATABASE = True
    MAX_DATABASE_CANDIDATES = 10000
    FINAL_TOP_K = 10
    RATIONALE_TOP_K = 10
    MAX_CANDIDATES_TO_PROCESS = 30  # Hard cap on candidates sent to LLM
    # VECTOR SEARCH OPTIMIZATION
    VECTOR_SIMILARITY_THRESHOLD = 0.30
    MAX_CANDIDATES_TO_RETRIEVE = 300  # Capped at ChromaDB cloud quota limit (300 per query)
    # EARLY SKILL FILTER: Minimum skill overlap to include candidate
    EARLY_SKILL_FILTER_THRESHOLD = 0.10
    # MINIMUM SKILL MATCH: contextual quality bar
    MIN_SKILL_MATCH_PERCENTAGE = 50
    # OPTIMISTIC SCALING: Only scale down when LLM score exceeds skill match by this margin (avoids double penalty)
    LLM_OPTIMISTIC_THRESHOLD = 0.15
    # TIER-BASED RANKING: Only send Tiers 1-3 to LLM (skip Tier 4-5)
    TIER_LLM_CUTOFF = 3
    # SCALE: Batch sizes for the LLM analysis stage
    LLM_BATCH_SIZE = 4  # Candidates per LLM prompt (keeps output under model limit to prevent truncation)
    LLM_PARALLEL_BATCHES = 5  # Concurrent LLM calls per round (raised from 3)
    # SCALE: Tier classification batch size
    TIER_BATCH_SIZE = 25  # Candidates per async tier batch (raised from 10)
    # SCALE: Resume fetching concurrency (lowered from 50 to avoid ChromaDB 429 rate limits)
    RESUME_FETCH_WORKERS = 12
    RESUME_FETCH_TIMEOUT = 10  # Per-resume fetch timeout in seconds


# Shared domain taxonomy for both JD and resume domain classification
# Used by: ranker (retrieval), groq_jd_parser (JD parsing), groq_resume_parser (resume parsing), chroma (metadata)
DOMAIN_TAXONOMY = [
    "software_engineering", "data_analytics", "data_science", "ai_ml",
    "technical_infrastructure", "traditional_engineering", "business_strategic",
    "sales_business", "marketing", "hr", "finance", "legal", "management",
    "healthcare", "ui_ux_design", "creative_design", "manufacturing",
    "supply_chain", "operations", "customer_support", "customer_success",
    "admin", "education", "science", "construction", "consulting",
    "hospitality", "retail", "trades", "real_estate", "fitness", "beauty",
    "aviation", "transportation", "military", "nonprofit", "public_sector",
    "arts", "sports", "environmental", "security", "cleaning_janitorial",
    "personal_care", "childcare", "other"
]

# Domain descriptions for semantic matching (used for domain-relevant experience calculation)
DOMAIN_DESCRIPTIONS = {
    "software_engineering": "Software development, programming, web development, mobile apps, frontend, backend, full-stack development, coding, computer science, software engineering",
    "data_analytics": "Business intelligence, data analysis, reporting, dashboards, data visualization, analytics tools like Tableau and Power BI, SQL queries and reporting",
    "data_science": "Data science, machine learning model development, statistical analysis, data mining, big data processing with Python, R, Spark, Hadoop",
    "ai_ml": "Artificial intelligence, machine learning engineering, deep learning, neural networks, LLMs, AI model development with TensorFlow, PyTorch, transformers",
    "technical_infrastructure": "DevOps, cloud engineering, system administration, SRE, platform engineering, infrastructure automation, CI/CD pipelines, containerization",
    "traditional_engineering": "Mechanical engineering, electrical engineering, civil engineering, chemical engineering, CAD design, circuit design, manufacturing processes",
    "business_strategic": "Product management, business analysis, strategy consulting, project management, business planning, roadmap development",
    "sales_business": "Sales, account management, business development, client relations, revenue generation, deal closing, sales strategy",
    "marketing": "Digital marketing, SEO, SEM, content marketing, brand management, social media marketing, advertising campaigns",
    "hr": "Human resources, talent acquisition, recruitment, employee relations, HR management, hiring, onboarding",
    "finance": "Finance, accounting, auditing, financial analysis, tax preparation, investment banking, financial planning",
    "legal": "Legal services, attorney, lawyer, compliance, corporate law, litigation, legal counsel",
    "management": "Executive leadership, general management, operations management, strategic management, C-suite roles",
    "healthcare": "Medical services, healthcare, nursing, clinical care, patient care, hospital administration, medical practice",
    "ui_ux_design": "User interface design, user experience design, UX research, prototyping, design systems, Figma, Sketch",
    "creative_design": "Graphic design, video editing, animation, photography, art direction, creative services",
    "manufacturing": "Production, assembly line, machining, quality control, lean manufacturing, factory operations",
    "supply_chain": "Supply chain management, logistics, procurement, warehouse operations, distribution, inventory management",
    "operations": "Business operations, operational excellence, process improvement, workflow optimization, operations management",
    "customer_support": "Customer service, help desk, technical support, customer care, support ticketing systems",
    "customer_success": "Customer success management, client retention, product adoption, account health, renewals",
    "admin": "Administrative assistant, office management, executive assistant, receptionist, clerical work",
    "education": "Teaching, educational services, training, academic instruction, curriculum development",
    "science": "Research scientist, laboratory work, R&D, biology, chemistry, biotechnology, pharmaceutical research",
    "construction": "Construction management, site supervision, civil engineering projects, facilities management, safety",
    "consulting": "Management consulting, strategy consulting, business advisory, transformation consulting",
    "hospitality": "Hotel management, restaurant service, culinary arts, food service, event planning, tourism, bartending, housekeeping",
    "retail": "Retail sales, store management, merchandising, e-commerce, customer service in retail",
    "trades": "Skilled trades, plumbing, electrical work, carpentry, welding, HVAC technician",
    "real_estate": "Real estate agent, broker, property management, leasing, real estate sales",
    "fitness": "Personal training, fitness coaching, nutrition, wellness coaching, exercise physiology",
    "beauty": "Cosmetology, esthetics, hairstyling, makeup artistry, skincare, salon services",
    "aviation": "Pilot, flight attendant, air traffic control, aircraft maintenance, aviation operations",
    "transportation": "Truck driving, delivery driver, logistics driving, fleet management, transportation",
    "military": "Military service, armed forces, defense, veterans, security clearance",
    "nonprofit": "Nonprofit organization, NGO, fundraising, grant writing, volunteer coordination",
    "public_sector": "Government, public administration, policy development, regulatory compliance",
    "arts": "Performing arts, theater, dance, music performance, fine arts, museum curation",
    "sports": "Athletic coaching, sports training, sports management, sports medicine",
    "environmental": "Environmental science, conservation, ecology, sustainability, wildlife management",
    "security": "Security guard, security officer, loss prevention, surveillance, access control, patrol services",
    "cleaning_janitorial": "Janitorial services, cleaning, custodial work, housekeeping, sanitation, facility cleaning",
    "personal_care": "Personal care aide, caregiver, home health aide, elderly care, disability support, assisted living",
    "childcare": "Childcare, daycare, nanny, babysitting, early childhood education, preschool teaching",
    "other": "General professional experience, miscellaneous work, various industries"
}


# ============================================================================
# HIGH PRIORITY: Caching Classes
# ============================================================================

class SkillExtractionCache:
    """Cache for extracted candidate skills to avoid redundant extraction."""
    
    def __init__(self, max_entries: int = 5000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[set, float]] = {}
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, candidate_id: str, resume_hash: str) -> str:
        """Generate cache key from candidate ID and resume content hash."""
        key_data = f"{candidate_id}:{resume_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get(self, candidate_id: str, resume_data: Dict) -> Optional[set]:
        """Get cached skills if available."""
        resume_str = json.dumps(resume_data, sort_keys=True)
        resume_hash = hashlib.sha256(resume_str.encode()).hexdigest()[:16]
        key = self._generate_key(candidate_id, resume_hash)
        
        async with self._lock:
            if key in self.cache:
                skills, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    self.hits += 1
                    return skills
                else:
                    del self.cache[key]
            self.misses += 1
        return None
    
    async def set(self, candidate_id: str, resume_data: Dict, skills: set):
        """Cache extracted skills."""
        resume_str = json.dumps(resume_data, sort_keys=True)
        resume_hash = hashlib.sha256(resume_str.encode()).hexdigest()[:16]
        key = self._generate_key(candidate_id, resume_hash)
        
        async with self._lock:
            # LRU eviction if at capacity
            if len(self.cache) >= self.max_entries:
                oldest_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k][1]
                )
                del self.cache[oldest_key]
            
            self.cache[key] = (skills, time.time())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "size": len(self.cache),
            "max_size": self.max_entries
        }


class SkillMatchingCache:
    """Cache skill matching results for candidate+JD pairs."""
    
    def __init__(self, max_entries: int = 10000, ttl_seconds: int = 1800):
        self.cache: Dict[str, Tuple[List[str], List[str], float, float]] = {}
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, candidate_id: str, jd_hash: str) -> str:
        """Generate cache key."""
        key_data = f"{candidate_id}:{jd_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get(self, candidate_id: str, jd_dict: Dict) -> Optional[Tuple[List[str], List[str], float]]:
        """Get cached matching results if available."""
        jd_str = json.dumps(jd_dict, sort_keys=True)
        jd_hash = hashlib.sha256(jd_str.encode()).hexdigest()[:16]
        key = self._generate_key(candidate_id, jd_hash)
        
        async with self._lock:
            if key in self.cache:
                matched, unmatched, match_pct, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    self.hits += 1
                    return (matched, unmatched, match_pct)
                else:
                    del self.cache[key]
            self.misses += 1
        return None
    
    async def set(self, candidate_id: str, jd_dict: Dict, matched: List[str], 
                  unmatched: List[str], match_pct: float):
        """Cache matching results."""
        jd_str = json.dumps(jd_dict, sort_keys=True)
        jd_hash = hashlib.sha256(jd_str.encode()).hexdigest()[:16]
        key = self._generate_key(candidate_id, jd_hash)
        
        async with self._lock:
            if len(self.cache) >= self.max_entries:
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][3])
                del self.cache[oldest_key]
            
            self.cache[key] = (matched, unmatched, match_pct, time.time())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "size": len(self.cache),
            "max_size": self.max_entries
        }


class JobDescriptionCache:
    """Cache parsed job descriptions."""
    
    def __init__(self, max_entries: int = 100, ttl_seconds: int = 7200):
        self.cache: Dict[str, Tuple[Any, float]] = {}  # Any = JobDescription
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, jd_dict: Dict) -> str:
        """Generate cache key from JD content."""
        key_data = json.dumps(jd_dict, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, jd_dict: Dict) -> Optional[Any]:
        """Get cached JD if available."""
        key = self._generate_key(jd_dict)
        
        if key in self.cache:
            jd, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.hits += 1
                return jd
            else:
                del self.cache[key]
        self.misses += 1
        return None
    
    def set(self, jd_dict: Dict, jd: Any):
        """Cache parsed JD."""
        key = self._generate_key(jd_dict)
        
        if len(self.cache) >= self.max_entries:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (jd, time.time())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "size": len(self.cache),
            "max_size": self.max_entries
        }


# ============================================================================
# HIGH PRIORITY: Circuit Breaker
# ============================================================================

class LLMCircuitBreaker:
    """Circuit breaker pattern for LLM calls to prevent cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self._lock = asyncio.Lock()
        self.total_calls = 0
        self.blocked_calls = 0
    
    async def can_proceed(self) -> bool:
        """Check if LLM calls can proceed."""
        async with self._lock:
            self.total_calls += 1
            if self.state == "closed":
                return True
            elif self.state == "open":
                # Check if timeout has passed
                if self.last_failure_time and \
                   time.time() - self.last_failure_time > self.timeout:
                    self.state = "half_open"
                    log.info("🔄 Circuit breaker: Moving to half-open state")
                    return True
                self.blocked_calls += 1
                return False
            else:  # half_open
                return True
    
    async def record_success(self):
        """Record successful LLM call."""
        async with self._lock:
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
                log.info("✅ Circuit breaker: Closed (recovered)")
            elif self.state == "closed":
                # Reset failure count on success
                self.failure_count = 0
    
    async def record_failure(self):
        """Record failed LLM call."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                log.warning(
                    f"⚠️ Circuit breaker: OPEN (too many failures: {self.failure_count})"
                )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "total_calls": self.total_calls,
            "blocked_calls": self.blocked_calls,
            "block_rate": round((self.blocked_calls / self.total_calls * 100) if self.total_calls > 0 else 0.0, 2)
        }


# ============================================================================
# HIGH PRIORITY: Performance Metrics
# ============================================================================

class RankerMetrics:
    """Track performance metrics for optimization."""
    
    def __init__(self):
        self.metrics = {
            "total_rankings": 0,
            "total_candidates_processed": 0,
            "avg_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_calls": 0,
            "llm_skips": 0,
            "early_filtered": 0,
            "errors": 0,
            "circuit_breaker_blocked": 0,
        }
        self._lock = asyncio.Lock()
    
    async def record_ranking(
        self,
        candidates_processed: int,
        processing_time: float,
        cache_hits: int,
        cache_misses: int,
        llm_calls: int,
        llm_skips: int,
        early_filtered: int,
        errors: int,
        circuit_breaker_blocked: int = 0
    ):
        """Record metrics for a ranking operation."""
        async with self._lock:
            self.metrics["total_rankings"] += 1
            self.metrics["total_candidates_processed"] += candidates_processed
            self.metrics["cache_hits"] += cache_hits
            self.metrics["cache_misses"] += cache_misses
            self.metrics["llm_calls"] += llm_calls
            self.metrics["llm_skips"] += llm_skips
            self.metrics["early_filtered"] += early_filtered
            self.metrics["errors"] += errors
            self.metrics["circuit_breaker_blocked"] += circuit_breaker_blocked
            
            # Update rolling average
            total = self.metrics["total_rankings"]
            current_avg = self.metrics["avg_processing_time"]
            self.metrics["avg_processing_time"] = (
                (current_avg * (total - 1) + processing_time) / total
            )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        metrics = self.metrics.copy()
        if metrics["total_rankings"] > 0:
            total_cache_ops = metrics["cache_hits"] + metrics["cache_misses"]
            metrics["cache_hit_rate"] = (
                metrics["cache_hits"] / total_cache_ops * 100
                if total_cache_ops > 0 else 0.0
            )
            total_llm_ops = metrics["llm_calls"] + metrics["llm_skips"]
            metrics["llm_skip_rate"] = (
                metrics["llm_skips"] / total_llm_ops * 100
                if total_llm_ops > 0 else 0.0
            )
            metrics["avg_candidates_per_ranking"] = (
                metrics["total_candidates_processed"] / metrics["total_rankings"]
            )
        return metrics


# ============================================================================
# MEDIUM PRIORITY: Adaptive Batch Sizing
# ============================================================================

class AdaptiveBatchSizer:
    """Dynamically adjust batch size based on system performance."""
    
    def __init__(self):
        self.current_batch_size = Config.PARALLEL_BATCH_SIZE
        self.recent_times: deque = deque(maxlen=10)
        self.recent_errors: int = 0
        self.max_history = 10
    
    def adjust_batch_size(self, batch_time: float, error_count: int):
        """Adjust batch size based on performance metrics."""
        self.recent_times.append(batch_time)
        self.recent_errors += error_count
        
        if len(self.recent_times) > self.max_history:
            self.recent_times.popleft()
        
        avg_time = sum(self.recent_times) / len(self.recent_times) if self.recent_times else batch_time
        
        # If too slow or too many errors, reduce batch size
        if avg_time > 10.0 or self.recent_errors > 3:
            self.current_batch_size = max(
                Config.MIN_BATCH_SIZE,
                self.current_batch_size - 1
            )
            log.info(f"📉 Reduced batch size to {self.current_batch_size} (slow/errors)")
        # If fast and no errors, increase batch size
        elif avg_time < 3.0 and self.recent_errors == 0:
            self.current_batch_size = min(
                Config.MAX_BATCH_SIZE,
                self.current_batch_size + 1
            )
            log.info(f"📈 Increased batch size to {self.current_batch_size} (fast)")
        
        # Reset error count periodically
        if len(self.recent_times) >= self.max_history:
            self.recent_errors = 0
    
    def get_batch_size(self) -> int:
        """Get current optimal batch size."""
        return self.current_batch_size


# ============================================================================
# Global Instances
# ============================================================================

_skill_extraction_cache = SkillExtractionCache(
    max_entries=Config.SKILL_EXTRACTION_CACHE_SIZE,
    ttl_seconds=Config.SKILL_EXTRACTION_CACHE_TTL
)

_skill_matching_cache = SkillMatchingCache(
    max_entries=Config.SKILL_MATCHING_CACHE_SIZE,
    ttl_seconds=Config.SKILL_MATCHING_CACHE_TTL
)

_jd_cache = JobDescriptionCache(
    max_entries=Config.JD_CACHE_SIZE,
    ttl_seconds=Config.JD_CACHE_TTL
)

_llm_circuit_breaker = LLMCircuitBreaker(
    failure_threshold=Config.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    timeout=Config.CIRCUIT_BREAKER_TIMEOUT
)

_ranker_metrics = RankerMetrics()

_batch_sizer = AdaptiveBatchSizer() if Config.ADAPTIVE_BATCH_ENABLED else None


class ErrorCode(Enum):
    """Error codes for better error tracking."""
    NO_JOB_DESCRIPTION = "NO_JOB_DESCRIPTION"
    NO_CANDIDATES_FOUND = "NO_CANDIDATES_FOUND"
    CHROMA_FETCH_ERROR = "CHROMA_FETCH_ERROR"
    LLM_INVOCATION_ERROR = "LLM_INVOCATION_ERROR"
    JSON_PARSE_ERROR = "JSON_PARSE_ERROR"
    INVALID_INPUT = "INVALID_INPUT"


log = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging with structured format.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@dataclass
class CandidateScore:
    """Data model for candidate scoring information."""
    candidate_id: str
    name: str
    email: str
    phone: str
    overall_score: float
    skills_matched: List[str]
    skills_unmatched: List[str]
    rationale: str
    rank: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class JobDescription:
    """Data model for job description."""
    job_title: str
    department: str
    description: str
    required_skills: List[str]
    experience: str

    @classmethod
    def from_dict(cls, jd: Dict[str, Any]) -> 'JobDescription':
        """Create JobDescription from dictionary."""
        return cls(
            job_title=jd.get('jobTitle', jd.get('title', '')),
            department=jd.get('department', ''),
            description=jd.get('fullJobDescription', jd.get('description', '')),
            required_skills=jd.get('requiredSkills', []),
            experience=jd.get('experience', '')
        )

    def to_text(self) -> str:
        """Convert to text representation for processing."""
        # Note: Domain context is now added in async _enhance_query_with_domain_context_async
        return (
            f"Title: {self.job_title}\n"
            f"Department: {self.department}\n"
            f"Description: {self.description}\n"
            f"Required Skills: {', '.join(self.required_skills)}\n"
            f"Experience: {self.experience}\n"
        )
    
    async def _get_domain_context_async(self) -> str:
        """
        Semantic embedding-based domain detection using Google Gemini.
        Replaces deterministic keyword matching with pure semantic similarity.
        
        Returns:
            Domain string (lowercase)
        """
        domain, confidence = await _classify_jd_domain_with_embeddings(self)
        
        # Use confidence threshold
        CONFIDENCE_THRESHOLD = 0.5
        if confidence < CONFIDENCE_THRESHOLD:
            log.warning(f"⚠️ Low confidence ({confidence:.2f}) for domain '{domain}', defaulting to OTHER")
            return "other"
        
        log.debug(f"✅ Semantic domain classification: '{domain}' (confidence: {confidence:.2f})")
        return domain
    
    def _get_domain_context(self) -> Tuple[str, float]:
        """
        DEPRECATED: Deterministic keyword-based domain detection.
        Kept for reference only. Use _get_domain_context_async() instead.
        
        Returns:
            Tuple of (domain, confidence) where confidence is 0.0-1.0
        """
        # Map job titles and skills to domains
        title_lower = self.job_title.lower()
        skills_lower = [s.lower() for s in self.required_skills]
        
        # Domain keywords mapping - Comprehensive coverage for all industries
        domain_keywords = {
            # ==========================================
            # 1. SOFTWARE & TECHNOLOGY (Renamed from "engineering")
            # ==========================================
            "software_engineering": [
                # Core
                "software engineer", "developer", "programmer", "coding", "computer science",
                "full stack", "backend", "frontend", "sdlc", "agile", "scrum",
                # Languages
                "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "php", "ruby", "swift", "kotlin",
                # Web
                "react", "angular", "vue", "node.js", "django", "spring boot", ".net", "html", "css",
                # Cloud/DevOps
                "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ci/cd", "devops",
                # Mobile
                "ios", "android", "react native", "flutter",
                # Database
                "sql", "nosql", "postgresql", "mongodb", "redis"
            ],

            "data_analytics": [
                "data analyst", "business intelligence", "bi", "reporting", "dashboards",
                "tableau", "power bi", "qlik", "looker", "excel", "sql", "etl",
                "data visualization", "metrics", "kpi", "analytics", "insights", "reporting analyst"
            ],

            "data_science": [
                "data scientist", "statistics", "statistical analysis", "hypothesis testing",
                "pandas", "numpy", "scipy", "r", "jupyter", "data analysis", "exploratory data analysis",
                "eda", "regression", "classification", "clustering", "data mining", "big data", "spark", "hadoop"
            ],

            "ai_ml": [
                "machine learning", "ml", "artificial intelligence", "ai", "deep learning",
                "neural networks", "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost",
                "nlp", "natural language processing", "computer vision", "cv", "opencv",
                "llm", "large language models", "gpt", "transformer", "bert", "llama",
                "reinforcement learning", "rl", "mlops", "model deployment", "model training",
                "feature engineering", "model evaluation", "ai engineer", "ml engineer",
                "prompt engineering", "langchain", "hugging face", "stable diffusion", "midjourney",
                "generative ai", "genai", "chatgpt", "claude", "anthropic"
            ],

            "product": [
                "product manager", "product owner", "pm", "product strategy", "roadmap", 
                "user stories", "backlog", "jira", "confluence", "prd", "go-to-market", "kpi",
                "a/b testing", "product lifecycle", "stakeholder management"
            ],

            # ==========================================
            # 2. TRADITIONAL ENGINEERING (Physical World)
            # ==========================================
            "mechanical": [
                "mechanical engineer", "mechanical design", "mechatronics", "robotics", 
                "cad", "solidworks", "catia", "autocad", "creo", "ansys", "pro-e",
                "thermodynamics", "fluid mechanics", "hydraulics", "pneumatics", "hvac",
                "fea", "finite element analysis", "cfd", "gd&t", "geometric dimensioning",
                "manufacturing engineering", "prototyping", "3d printing", "automotive", "aerospace"
            ],

            "electrical": [
                "electrical engineer", "electronics", "circuit design", "pcb", "schematic",
                "altium", "orcad", "eagle", "fpga", "verilog", "vhdl", "microcontrollers",
                "embedded systems", "plc", "scada", "power systems", "high voltage",
                "control systems", "instrumentation", "rf", "signal processing", "iot"
            ],

            "civil": [
                "civil engineer", "structural engineer", "geotechnical", "environmental engineering",
                "construction management", "infrastructure", "autocad civil 3d", "revit", "bim", 
                "staad", "etabs", "surveying", "land development", "hydrology", 
                "urban planning", "concrete", "steel structures", "bridge design", "transportation"
            ],

            "chemical": [
                "chemical engineer", "process engineer", "process simulation", "aspen hysys",
                "aspen plus", "matlab", "polymers", "material science", "metallurgy",
                "petrochemical", "refinery", "distillation", "reaction engineering",
                "mass transfer", "heat transfer", "chromatography", "quality control"
            ],

            # ==========================================
            # 3. CORPORATE FUNCTIONS
            # ==========================================
            "sales": [
                "sales", "business development", "account executive", "sdr", "bdr",
                "account management", "enterprise sales", "saas sales", "revenue", "quota",
                "prospecting", "lead generation", "closing", "negotiation", "crm", "salesforce",
                "cold calling", "pipeline management", "forecasting"
            ],

            "marketing": [
                "marketing", "digital marketing", "seo", "sem", "content marketing",
                "social media", "brand management", "email marketing", "growth hacking",
                "google ads", "analytics", "copywriting", "public relations", "pr",
                "campaign management", "market research"
            ],

            "finance": [
                "finance", "accounting", "auditing", "financial analyst", "cpa", "tax",
                "bookkeeping", "payroll", "fp&a", "forecasting", "budgeting", "compliance",
                "gaap", "ifrs", "investment banking", "private equity", "excel"
            ],

            "hr": [
                "human resources", "hr", "recruiter", "talent acquisition", "hrbp",
                "employee relations", "benefits", "compensation", "onboarding", 
                "performance management", "learning and development", "hris", "workday"
            ],

            "legal": [
                "legal", "attorney", "lawyer", "paralegal", "general counsel", "litigation",
                "corporate law", "contracts", "compliance", "regulatory", "intellectual property",
                "gdpr", "privacy", "employment law", "mergers and acquisitions"
            ],

            "management": [
                "ceo", "cto", "cfo", "coo", "vice president", "vp", "director", "head of",
                "general manager", "operations manager", "strategy", "leadership", 
                "p&l", "budget management", "change management", "organizational development"
            ],

            # ==========================================
            # 4. SPECIALIZED DOMAINS
            # ==========================================
            "healthcare": [
                "healthcare", "medical", "nursing", "rn", "doctor", "physician", 
                "clinical", "patient care", "hospital", "pharmacy", "public health",
                "medical records", "emr", "ehr", "health administration"
            ],

            "ui_ux_design": [
                "ui/ux", "user interface", "user experience", "ux design", "ui design",
                "interaction design", "ux research", "user research", "usability testing",
                "wireframing", "prototyping", "figma", "sketch", "adobe xd", "invision",
                "design systems", "information architecture", "user flows", "personas"
            ],

            "creative_design": [
                "graphic design", "visual design", "web design", "video editing", "animation",
                "motion graphics", "photography", "videography", "art director", 
                "creative director", "adobe creative suite", "photoshop", "illustrator",
                "premiere pro", "after effects", "indesign", "branding", "print design"
            ],
            
            "manufacturing": [
                "manufacturing", "production", "plant manager", "assembly", "machining",
                "lean manufacturing", "six sigma", "kaizen", "quality assurance", "qa",
                "production planning", "manufacturing engineer", "production supervisor",
                "cnc", "fabrication", "welding", "machining", "assembly line"
            ],
            
            "supply_chain": [
                "supply chain", "logistics", "procurement", "warehouse",
                "inventory", "distribution", "sourcing", "vendor management",
                "purchasing", "supply chain management", "logistics coordinator"
            ],

            "operations": [
                "operations", "operations manager", "business operations", "operational excellence",
                "process improvement", "operations analyst", "business process", "workflow"
            ],
            
            "customer_support": [
                "customer service", "support", "help desk", "technical support", 
                "troubleshooting", "ticketing", "zendesk", "intercom", "freshdesk",
                "customer support", "support engineer", "support specialist", "it support",
                "call center", "contact center", "live chat", "email support"
            ],

            "customer_success": [
                "customer success", "csm", "customer success manager", "onboarding", 
                "implementation", "retention", "churn reduction", "client success", 
                "account health", "qbr", "quarterly business review", "adoption", "renewals",
                "account management", "customer advocacy", "expansion", "upsell"
            ],
            
            "admin": [
                "administrative", "admin", "office manager", "executive assistant", "ea", 
                "receptionist", "data entry", "clerical", "secretary", "scheduling", 
                "travel arrangements", "office support", "calendar management"
            ],
            
            "education": [
                "education", "teacher", "teaching", "tutor", "curriculum", "instructional design",
                "training", "l&d", "learning and development", "faculty", "academic", 
                "professor", "k-12", "higher education"
            ],
            
            "science": [
                "science", "research", "biology", "chemistry", "laboratory", "lab", 
                "scientist", "physics", "biotech", "r&d", "clinical trials", "genomics",
                "microbiology", "pharmaceutical", "research and development"
            ],
            
            "construction": [
                "construction", "civil engineer", "site manager", "project manager", 
                "estimator", "surveyor", "architecture", "real estate", "property manager",
                "facilities", "blueprint", "safety", "osha"
            ],
            
            "consulting": [
                "consultant", "consulting", "management consulting", "strategy", "advisor",
                "due diligence", "mergers and acquisitions", "m&a", "transformation",
                "business process", "frameworks", "risk management", "advisory",
                "client delivery", "engagement manager", "principal"
            ],
            
            # ==========================================
            # 5. ADDITIONAL DOMAINS
            # ==========================================
            "hospitality": [
                "hospitality", "hotel", "restaurant", "culinary", "chef", "food service",
                "event planning", "catering", "banquet", "front desk", "concierge",
                "hospitality management", "tourism", "travel",
                # Food & Beverage Service
                "tea maker", "coffee maker", "beverage service", "beverage preparation",
                "pantry", "pantry assistant", "pantry helper", "kitchen helper",
                "kitchen assistant", "food preparation", "food service assistant",
                "cafeteria", "canteen", "dining service", "wait staff", "server",
                "waiter", "waitress", "barista", "bartender", "food handler", 
                "hygiene", "cleanliness", "host", "hostess", "busser", "busboy",
                "dishwasher", "line cook", "prep cook", "sous chef", "pastry chef",
                "food runner", "room service", "banquet server", "catering staff",
                # Hotel Services
                "housekeeping", "housekeeper", "laundry", "laundry attendant",
                "maintenance", "hotel maintenance", "bellhop", "valet", "parking attendant",
                "guest services", "reservations", "front office", "night auditor"
            ],
            
            "security": [
                "security guard", "security officer", "security personnel", "security staff",
                "loss prevention", "surveillance", "access control", "patrol", "patrolling",
                "security management", "security supervisor", "security coordinator",
                "armed guard", "unarmed guard", "event security", "retail security",
                "corporate security", "facility security", "gate guard", "watchman",
                "security system", "cctv", "monitoring", "security operations"
            ],
            
            "cleaning_janitorial": [
                "janitor", "janitorial", "cleaner", "cleaning", "custodial", "custodian",
                "housekeeping", "sanitation", "sanitation worker", "cleaning staff",
                "office cleaner", "building cleaner", "floor care", "window cleaning",
                "carpet cleaning", "deep cleaning", "maintenance cleaning", "industrial cleaning",
                "commercial cleaning", "residential cleaning", "maid", "house cleaner"
            ],
            
            "personal_care": [
                "personal care", "personal care aide", "caregiver", "home care", "home health aide",
                "elderly care", "senior care", "elder care", "disability support", "disability care",
                "companion care", "respite care", "assisted living", "nursing aide", "cna",
                "certified nursing assistant", "patient care", "personal support worker"
            ],
            
            "childcare": [
                "childcare", "child care", "daycare", "day care", "nanny", "babysitter",
                "babysitting", "early childhood", "preschool teacher", "daycare teacher",
                "childcare provider", "childcare worker", "after school care", "summer camp",
                "childcare assistant", "toddler care", "infant care", "child development"
            ],
            
            "retail": [
                "retail", "store manager", "merchandising", "inventory", "point of sale",
                "pos", "customer service", "sales associate", "visual merchandising",
                "retail management", "e-commerce", "omnichannel"
            ],
            
            "trades": [
                "plumber", "electrician", "carpenter", "welder", "mason", "roofer",
                "hvac technician", "journeyman", "apprentice", "master tradesman",
                "construction trades", "skilled trades", "trade certification"
            ],
            
            "real_estate": [
                "real estate", "real estate agent", "broker", "property management",
                "commercial real estate", "residential real estate", "leasing",
                "property development", "real estate investment", "mls"
            ],
            
            "fitness": [
                "fitness", "personal trainer", "nutritionist", "yoga instructor",
                "strength and conditioning", "wellness", "health coach",
                "exercise physiology", "group fitness", "fitness training"
            ],
            
            "beauty": [
                "cosmetology", "esthetics", "barber", "hairstylist", "makeup artist",
                "nail technician", "spa", "salon", "beauty therapy", "skincare"
            ],
            
            "aviation": [
                "pilot", "flight attendant", "air traffic controller", "aviation",
                "aircraft", "faa", "commercial pilot", "private pilot", "airline",
                "aviation maintenance", "airport operations"
            ],
            
            "transportation": [
                "truck driver", "delivery driver", "logistics driver", "cdl",
                "transportation", "fleet management", "dispatch", "route planning",
                "shipping", "freight", "warehouse driver"
            ],
            
            "military": [
                "military", "veteran", "armed forces", "navy", "army", "air force",
                "marines", "coast guard", "military service", "veteran affairs",
                "defense", "security clearance"
            ],
            
            "nonprofit": [
                "nonprofit", "ngo", "fundraising", "grant writing", "donor relations",
                "volunteer coordination", "nonprofit management", "advocacy",
                "community outreach", "social work", "philanthropy"
            ],
            
            "public_sector": [
                "government", "public administration", "policy", "regulatory compliance",
                "public service", "federal", "state", "local government", "civil service"
            ],
            
            "arts": [
                "performing arts", "theater", "dance", "music", "acting", "directing",
                "fine arts", "sculpture", "painting", "art history", "art curator",
                "museum", "gallery", "arts administration"
            ],
            
            "sports": [
                "coaching", "athletic training", "sports", "sports management",
                "strength and conditioning", "sports medicine", "athletic director",
                "player development", "sports analytics"
            ],
            
            "environmental": [
                "environmental", "conservation", "ecology", "sustainability",
                "wildlife", "forestry", "park ranger", "environmental science",
                "renewable energy", "climate", "environmental policy"
            ],
        }
        
        # Determine domain from title with confidence scoring
        detected_domains = []
        domain_scores = {}  # Track keyword match scores per domain
        
        for domain, keywords in domain_keywords.items():
            score = 0
            # Check job title (worth more)
            if any(keyword in title_lower for keyword in keywords):
                detected_domains.append(domain)
                score += 2  # Title match is worth more
            
            # Check skills (need at least 2 matching keywords)
            matching_keywords = [kw for kw in keywords if any(kw in skill for skill in skills_lower)]
            if len(matching_keywords) >= 2:
                detected_domains.append(domain)
                score += len(matching_keywords)  # Add keyword match count
            
            if score > 0:
                domain_scores[domain] = score
        
        # Return primary domain with confidence score
        if detected_domains:
            from collections import Counter
            domain_counts = Counter(detected_domains)
            primary_domain, count = domain_counts.most_common(1)[0]
            
            # Calculate confidence: how dominant is the primary domain?
            total_detections = len(detected_domains)
            confidence = count / total_detections if total_detections > 0 else 0.0
            
            # Boost confidence if domain has high keyword match score
            if primary_domain in domain_scores:
                score_boost = min(domain_scores[primary_domain] / 10.0, 0.3)  # Max 0.3 boost
                confidence = min(confidence + score_boost, 1.0)
            
            return primary_domain, confidence
        else:
            return "general", 0.0


def _normalize_llm_domain(domain: str) -> str:
    """
    Normalize LLM-returned domain (uppercase with underscores) to lowercase format.
    
    Args:
        domain: Domain string from LLM (e.g., "SALES_BUSINESS", "SOFTWARE_ENGINEERING")
        
    Returns:
        Normalized lowercase domain (e.g., "sales", "software_engineering")
    """
    # Mapping from LLM uppercase format to lowercase format
    domain_mapping = {
        "SOFTWARE_ENGINEERING": "software_engineering",
        "DATA_ANALYTICS": "data_analytics",
        "DATA_SCIENCE": "data_science",
        "AI_ML": "ai_ml",
        "TECHNICAL_INFRASTRUCTURE": "technical_infrastructure",
        "TRADITIONAL_ENGINEERING": "mechanical",  # Map to primary traditional eng
        "BUSINESS_STRATEGIC": "management",
        "SALES_BUSINESS": "sales",
        "MARKETING": "marketing",
        "HR": "hr",
        "FINANCE": "finance",
        "LEGAL": "legal",
        "MANAGEMENT": "management",
        "HEALTHCARE": "healthcare",
        "UI_UX_DESIGN": "ui_ux_design",
        "CREATIVE_DESIGN": "creative_design",
        "MANUFACTURING": "manufacturing",
        "SUPPLY_CHAIN": "supply_chain",
        "OPERATIONS": "operations",
        "CUSTOMER_SUPPORT": "customer_support",
        "CUSTOMER_SUCCESS": "customer_success",
        "ADMIN": "admin",
        "EDUCATION": "education",
        "SCIENCE": "science",
        "CONSTRUCTION": "construction",
        "CONSULTING": "consulting",
        "HOSPITALITY": "hospitality",
        "RETAIL": "retail",
        "TRADES": "trades",
        "REAL_ESTATE": "real_estate",
        "FITNESS": "fitness",
        "BEAUTY": "beauty",
        "AVIATION": "aviation",
        "TRANSPORTATION": "transportation",
        "MILITARY": "military",
        "NONPROFIT": "nonprofit",
        "PUBLIC_SECTOR": "public_sector",
        "ARTS": "arts",
        "SPORTS": "sports",
        "ENVIRONMENTAL": "environmental",
        "SECURITY": "security",
        "CLEANING_JANITORIAL": "cleaning_janitorial",
        "PERSONAL_CARE": "personal_care",
        "CHILDCARE": "childcare",
        "OTHER": "other",
    }
    
    domain_upper = domain.upper()
    return domain_mapping.get(domain_upper, domain.lower())


# Cache for JD domain classification by JD text hash (reduces embedding API calls)
_JD_DOMAIN_CACHE_MAX = 500
_jd_domain_cache: Dict[str, Tuple[str, float]] = {}


def _evict_jd_domain_cache_if_needed() -> None:
    """Evict oldest half of JD domain cache when over max size."""
    if len(_jd_domain_cache) >= _JD_DOMAIN_CACHE_MAX:
        keys_to_remove = list(_jd_domain_cache.keys())[: _JD_DOMAIN_CACHE_MAX // 2]
        for k in keys_to_remove:
            _jd_domain_cache.pop(k, None)
        log.debug(f"JD domain cache evicted {len(keys_to_remove)} entries")


async def _classify_jd_domain_with_embeddings(jd: JobDescription) -> Tuple[str, float]:
    """
    Semantic embedding-based domain classification for job descriptions using Google Gemini.
    
    Creates embeddings for domain descriptions and job description, then finds best match
    using cosine similarity. Results are cached by JD text hash to reduce API calls.
    
    Args:
        jd: JobDescription object
        
    Returns:
        Tuple of (domain, confidence) where domain is lowercase and confidence is 0.0-1.0
    """
    try:
        from google import genai
        from settings import settings
        import numpy as np
        
        # Build JD text for cache key and embedding
        jd_text = jd.to_text()
        if not jd_text.strip():
            log.warning("No meaningful text in job description for domain classification")
            return "other", 0.0

        # Check cache before embedding
        cache_key = hashlib.sha256(jd_text.encode("utf-8")).hexdigest()
        if cache_key in _jd_domain_cache:
            cached = _jd_domain_cache[cache_key]
            log.debug(f"JD domain cache HIT: {cached[0]} (confidence: {cached[1]:.3f})")
            return cached[0], cached[1]
        
        # Get API key
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            log.warning("Google API key not found, falling back to OTHER domain")
            return "other", 0.0
        
        client = genai.Client(api_key=api_key)
        
        # Define domain descriptions (semantic, not keywords) - SAME AS job_matcher.py
        domain_descriptions = {
            "software_engineering": "Software development, programming, web development, mobile apps, frontend, backend, full-stack development, coding, computer science, software engineering",
            "data_analytics": "Business intelligence, data analysis, reporting, dashboards, data visualization, analytics tools like Tableau and Power BI, SQL queries and reporting",
            "data_science": "Data science, machine learning model development, statistical analysis, data mining, big data processing with Python, R, Spark, Hadoop",
            "ai_ml": "Artificial intelligence, machine learning engineering, deep learning, neural networks, LLMs, AI model development with TensorFlow, PyTorch, transformers",
            "technical_infrastructure": "DevOps, cloud engineering, system administration, SRE, platform engineering, infrastructure automation, CI/CD pipelines, containerization",
            "traditional_engineering": "Mechanical engineering, electrical engineering, civil engineering, chemical engineering, CAD design, circuit design, manufacturing processes",
            "business_strategic": "Product management, business analysis, strategy consulting, project management, business planning, roadmap development",
            "sales_business": "Sales, account management, business development, client relations, revenue generation, deal closing, sales strategy",
            "marketing": "Digital marketing, SEO, SEM, content marketing, brand management, social media marketing, advertising campaigns",
            "hr": "Human resources, talent acquisition, recruitment, employee relations, HR management, hiring, onboarding",
            "finance": "Finance, accounting, auditing, financial analysis, tax preparation, investment banking, financial planning",
            "legal": "Legal services, attorney, lawyer, compliance, corporate law, litigation, legal counsel",
            "management": "Executive leadership, general management, operations management, strategic management, C-suite roles",
            "healthcare": "Medical services, healthcare, nursing, clinical care, patient care, hospital administration, medical practice",
            "ui_ux_design": "User interface design, user experience design, UX research, prototyping, design systems, Figma, Sketch",
            "creative_design": "Graphic design, video editing, animation, photography, art direction, creative services",
            "manufacturing": "Production, assembly line, machining, quality control, lean manufacturing, factory operations",
            "supply_chain": "Supply chain management, logistics, procurement, warehouse operations, distribution, inventory management",
            "operations": "Business operations, operational excellence, process improvement, workflow optimization, operations management",
            "customer_support": "Customer service, help desk, technical support, customer care, support ticketing systems",
            "customer_success": "Customer success management, client retention, product adoption, account health, renewals",
            "admin": "Administrative assistant, office management, executive assistant, receptionist, clerical work",
            "education": "Teaching, educational services, training, academic instruction, curriculum development",
            "science": "Research scientist, laboratory work, R&D, biology, chemistry, biotechnology, pharmaceutical research",
            "construction": "Construction management, site supervision, civil engineering projects, facilities management, safety",
            "consulting": "Management consulting, strategy consulting, business advisory, transformation consulting",
            "hospitality": "Hotel management, restaurant service, culinary arts, food service, event planning, tourism, bartending, housekeeping",
            "retail": "Retail sales, store management, merchandising, e-commerce, customer service in retail",
            "trades": "Skilled trades, plumbing, electrical work, carpentry, welding, HVAC technician",
            "real_estate": "Real estate agent, broker, property management, leasing, real estate sales",
            "fitness": "Personal training, fitness coaching, nutrition, wellness coaching, exercise physiology",
            "beauty": "Cosmetology, esthetics, hairstyling, makeup artistry, skincare, salon services",
            "aviation": "Pilot, flight attendant, air traffic control, aircraft maintenance, aviation operations",
            "transportation": "Truck driving, delivery driver, logistics driving, fleet management, transportation",
            "military": "Military service, armed forces, defense, veterans, security clearance",
            "nonprofit": "Nonprofit organization, NGO, fundraising, grant writing, volunteer coordination",
            "public_sector": "Government, public administration, policy development, regulatory compliance",
            "arts": "Performing arts, theater, dance, music performance, fine arts, museum curation",
            "sports": "Athletic coaching, sports training, sports management, sports medicine",
            "environmental": "Environmental science, conservation, ecology, sustainability, wildlife management",
            "security": "Security guard, security officer, loss prevention, surveillance, access control, patrol services",
            "cleaning_janitorial": "Janitorial services, cleaning, custodial work, housekeeping, sanitation, facility cleaning",
            "personal_care": "Personal care aide, caregiver, home health aide, elderly care, disability support, assisted living",
            "childcare": "Childcare, daycare, nanny, babysitting, early childhood education, preschool teaching"
        }
        
        # Prepare texts for embedding: JD + all domain descriptions (jd_text from cache key above)
        all_texts = [jd_text] + list(domain_descriptions.values())
        
        # Batch embed all texts (cached across restarts via gemini_embedding_cache)
        # Issue 4.1: Wrap synchronous Gemini call in asyncio.to_thread to avoid blocking event loop
        log.debug(f"Embedding JD + {len(domain_descriptions)} domain descriptions...")
        import asyncio
        
        def _sync_embed():
            return cached_embed_texts(client, "models/gemini-embedding-001", all_texts)
        
        # Run in thread pool to avoid blocking the event loop
        try:
            loop = asyncio.get_running_loop()
            embeddings = await asyncio.to_thread(_sync_embed)
        except RuntimeError:
            # No running event loop, call directly (sync context)
            embeddings = _sync_embed()
        
        if not embeddings:
            log.error("Unexpected embedding response format")
            return "other", 0.0
        
        if len(embeddings) != len(all_texts):
            log.error(f"Embedding count mismatch: expected {len(all_texts)}, got {len(embeddings)}")
            return "other", 0.0
        
        # Convert to numpy arrays
        jd_embedding = np.array(embeddings[0])
        domain_embeddings = np.array(embeddings[1:])
        
        # Normalize
        jd_norm = np.linalg.norm(jd_embedding)
        if jd_norm > 0:
            jd_embedding = jd_embedding / jd_norm
        
        domain_norms = np.linalg.norm(domain_embeddings, axis=1, keepdims=True)
        domain_embeddings = np.where(domain_norms > 0, domain_embeddings / domain_norms, domain_embeddings)
        
        # Calculate cosine similarities
        similarities = np.dot(domain_embeddings, jd_embedding)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = float(similarities[best_idx])
        domain_names = list(domain_descriptions.keys())
        best_domain = domain_names[best_idx]
        
        log.info(f"✅ Semantic JD domain classification: {best_domain} (confidence: {best_score:.3f})")
        log.debug(f"   Top 3 domains: {[(domain_names[i], float(similarities[i])) for i in np.argsort(similarities)[-3:][::-1]]}")
        
        # Store in cache for future requests with same JD
        _evict_jd_domain_cache_if_needed()
        _jd_domain_cache[cache_key] = (best_domain, best_score)
        
        return best_domain, best_score
        
    except ImportError:
        log.error("Google GenAI package not installed")
        return "other", 0.0
    except Exception as e:
        log.error(f"❌ Semantic JD domain classification failed: {e}")
        import traceback
        log.error(f"Traceback: {traceback.format_exc()}")
        return "other", 0.0


async def _classify_domain_with_llm(jd: JobDescription) -> str:
    """
    DEPRECATED: Fallback LLM-based domain classification for ambiguous cases.
    Kept for reference only. Use _classify_jd_domain_with_embeddings() instead.
    
    Args:
        jd: JobDescription object
        
    Returns:
        Domain string (lowercase)
    """
    try:
        jd_text = jd.to_text()
        prompt = f"""Analyze this job description and classify its primary professional domain.

JOB DESCRIPTION:
{jd_text}

Return ONLY a JSON object with this structure:
{{
    "primary_domain": "SOFTWARE_ENGINEERING|DATA_ANALYTICS|DATA_SCIENCE|AI_ML|TECHNICAL_INFRASTRUCTURE|TRADITIONAL_ENGINEERING|BUSINESS_STRATEGIC|SALES_BUSINESS|MARKETING|HR|FINANCE|LEGAL|MANAGEMENT|HEALTHCARE|UI_UX_DESIGN|CREATIVE_DESIGN|MANUFACTURING|SUPPLY_CHAIN|OPERATIONS|CUSTOMER_SUPPORT|CUSTOMER_SUCCESS|ADMIN|EDUCATION|SCIENCE|CONSTRUCTION|CONSULTING|HOSPITALITY|RETAIL|TRADES|REAL_ESTATE|FITNESS|BEAUTY|AVIATION|TRANSPORTATION|MILITARY|NONPROFIT|PUBLIC_SECTOR|ARTS|SPORTS|ENVIRONMENTAL|SECURITY|CLEANING_JANITORIAL|PERSONAL_CARE|CHILDCARE|OTHER",
    "confidence": 0.95,
    "reasoning": "Brief explanation"
}}

Domain Definitions:
- SOFTWARE_ENGINEERING: Software Development, Programming, Frontend/Backend Development, Mobile Development, Full-Stack Development
- DATA_ANALYTICS: Business Intelligence, Reporting, Dashboards, Data Visualization, Analytics (Tableau, Power BI, SQL)
- DATA_SCIENCE: Data Science, Statistical Analysis, Hypothesis Testing, Data Mining, Big Data (Python, R, Spark, Hadoop)
- AI_ML: Machine Learning, Artificial Intelligence, Deep Learning, Neural Networks, LLMs, AI Engineering (TensorFlow, PyTorch, GPT, LangChain)
- TECHNICAL_INFRASTRUCTURE: DevOps, Cloud Engineering, System Administration, SRE, Platform Engineering, Infrastructure
- TRADITIONAL_ENGINEERING: Mechanical, Electrical, Civil, Chemical Engineering (CAD, SolidWorks, PCB design, etc.)
- BUSINESS_STRATEGIC: Product Management, Business Analysis, Strategy, Project Management
- SALES_BUSINESS: Sales, Account Management, Business Development, Client Relations
- MARKETING: Digital Marketing, SEO, SEM, Content Marketing, Brand Management, Advertising
- HR: Human Resources, Talent Acquisition, Recruitment, Employee Relations
- FINANCE: Finance, Accounting, Auditing, Financial Analysis, Tax, Investment Banking
- LEGAL: Legal, Attorney, Lawyer, Compliance, Corporate Law, Litigation
- MANAGEMENT: Executive Leadership, General Management, Operations Management, Strategy
- HEALTHCARE: Medical, Healthcare, Nursing, Clinical, Patient Care, Hospital Administration
- UI_UX_DESIGN: User Interface/Experience Design, UX Research, Prototyping, Design Systems (Figma, Sketch)
- CREATIVE_DESIGN: Graphic Design, Video Editing, Animation, Photography, Art Direction
- MANUFACTURING: Production, Assembly, Machining, Quality Control, Lean Manufacturing
- SUPPLY_CHAIN: Supply Chain, Logistics, Procurement, Warehouse, Distribution, Sourcing
- OPERATIONS: Business Operations, Operational Excellence, Process Improvement, Workflow
- CUSTOMER_SUPPORT: Customer Service, Help Desk, Technical Support, Ticketing (Zendesk, Intercom)
- CUSTOMER_SUCCESS: Customer Success Management, Retention, Adoption, Renewals, Account Health
- ADMIN: Administrative, Office Management, Executive Assistant, Receptionist, Clerical
- EDUCATION: Teaching, Educational Services, Training, Academic, Instructional Design
- SCIENCE: Research, Laboratory, R&D, Biology, Chemistry, Biotech, Pharmaceutical
- CONSTRUCTION: Construction, Site Management, Civil Engineering, Facilities, Safety
- CONSULTING: Management Consulting, Strategy, Advisory, Business Transformation
- HOSPITALITY: Hotel, Restaurant, Culinary, Food Service, Event Planning, Tourism, Tea Maker, Coffee Maker, Beverage Service, Pantry, Kitchen Helper, Server, Waiter, Waitress, Barista, Bartender, Host, Hostess, Busser, Dishwasher, Line Cook, Prep Cook, Sous Chef, Pastry Chef, Food Runner, Room Service, Housekeeping, Laundry, Bellhop, Valet, Parking Attendant, Guest Services, Reservations, Front Office, Night Auditor
- RETAIL: Retail, Store Management, Merchandising, E-commerce, Visual Merchandising
- TRADES: Plumbing, Electrical Work, Carpentry, Welding, HVAC, Skilled Trades
- REAL_ESTATE: Real Estate Agent, Broker, Property Management, Leasing, MLS
- FITNESS: Personal Training, Nutrition, Wellness, Health Coaching, Exercise Physiology
- BEAUTY: Cosmetology, Esthetics, Hairstyling, Makeup, Skincare, Salon
- AVIATION: Pilot, Flight Attendant, Air Traffic Control, Aircraft, Aviation Maintenance
- TRANSPORTATION: Truck Driver, Delivery Driver, Logistics Driver, Fleet Management, CDL
- MILITARY: Military, Veteran, Armed Forces, Defense, Security Clearance
- NONPROFIT: Nonprofit, NGO, Fundraising, Grant Writing, Volunteer Coordination
- PUBLIC_SECTOR: Government, Public Administration, Policy, Regulatory Compliance
- ARTS: Performing Arts, Theater, Dance, Music, Fine Arts, Museum, Gallery
- SPORTS: Coaching, Athletic Training, Sports Management, Sports Medicine
- ENVIRONMENTAL: Environmental, Conservation, Ecology, Sustainability, Wildlife, Forestry
- SECURITY: Security Guard, Security Officer, Loss Prevention, Surveillance, Access Control, Patrol, Armed Guard, Unarmed Guard, Event Security, Retail Security, Corporate Security, Facility Security, Gate Guard, Watchman, CCTV Monitoring, Security Operations
- CLEANING_JANITORIAL: Janitor, Janitorial, Cleaner, Cleaning, Custodial, Custodian, Housekeeping, Sanitation, Sanitation Worker, Cleaning Staff, Office Cleaner, Building Cleaner, Floor Care, Window Cleaning, Carpet Cleaning, Deep Cleaning, Maintenance Cleaning, Industrial Cleaning, Commercial Cleaning, Residential Cleaning, Maid, House Cleaner
- PERSONAL_CARE: Personal Care, Personal Care Aide, Caregiver, Home Care, Home Health Aide, Elderly Care, Senior Care, Elder Care, Disability Support, Disability Care, Companion Care, Respite Care, Assisted Living, Nursing Aide, CNA, Certified Nursing Assistant, Patient Care, Personal Support Worker
- CHILDCARE: Childcare, Child Care, Daycare, Day Care, Nanny, Babysitter, Babysitting, Early Childhood, Preschool Teacher, Daycare Teacher, Childcare Provider, Childcare Worker, After School Care, Summer Camp, Childcare Assistant, Toddler Care, Infant Care, Child Development
- OTHER: Any other domain not listed above

Return ONLY the JSON object."""
        
        response = await invoke_llm(
            prompt=prompt,
            task_type="classification",
            agent_name="domain_classifier",
            max_retries=2
        )
        
        if response:
            try:
                result = json.loads(response)
                domain = result.get("primary_domain", "OTHER")
                log.info(f"🤖 LLM classified domain: {domain}")
                return _normalize_llm_domain(domain)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                match = re.search(r'"primary_domain"\s*:\s*"([^"]+)"', response)
                if match:
                    domain = match.group(1)
                    log.info(f"🤖 LLM classified domain (regex): {domain}")
                    return _normalize_llm_domain(domain)
        
        log.warning("⚠️ LLM domain classification failed, using 'general'")
        return "general"
    except Exception as e:
        log.warning(f"⚠️ LLM domain classification error: {e}")
        return "general"


class RankerAgentError(Exception):
    """Base exception for ranker agent errors."""
    def __init__(self, message: str, error_code: ErrorCode):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ChromaFetchError(RankerAgentError):
    """Exception raised when Chroma fetch fails."""
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.CHROMA_FETCH_ERROR)


class LLMInvocationError(RankerAgentError):
    """Exception raised when LLM invocation fails."""
    def __init__(self, message: str):
        super().__init__(message, ErrorCode.LLM_INVOCATION_ERROR)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from text that may contain additional content.
    
    Args:
        text: Raw text potentially containing JSON
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If JSON cannot be extracted or parsed
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")
    
    try:
        # First, try direct parsing
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                log.error(f"Failed to parse extracted JSON: {e}")
                raise ValueError(f"Invalid JSON format: {e}")
        raise ValueError("No JSON object found in text")


def _sanitize_json_string_values(s: str) -> str:
    """
    Fix unescaped control characters and unescaped double quotes inside JSON string values.
    Walks the string tracking whether we're inside a quoted value and:
    - Escapes raw newlines/tabs/control chars that break json.loads
    - Escapes unescaped double quotes inside strings (e.g. "Python" -> \\"Python\\")
      which cause "Expecting ',' delimiter" when Gemini outputs e.g. "Strong in "Python" skills"
    """
    out = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_string:
            out.append(c)
            if i + 1 < len(s):
                i += 1
                out.append(s[i])
            i += 1
            continue
        if c == '"':
            if in_string:
                # Peek ahead: is this the closing quote or an unescaped inner quote?
                j = i + 1
                while j < len(s) and s[j] in ' \t\r\n':
                    j += 1
                next_char = s[j] if j < len(s) else ''
                # Closing quote is followed by , } ] : or end
                if next_char in ',}]:' or next_char == '':
                    in_string = False
                    out.append(c)
                elif next_char == '"':
                    # "foo "bar" baz" vs "a" "b": peek past the next " to distinguish.
                    # "inner" " -> after next " we see , } ] : = inner quote, escape.
                    # "a" "b" -> after next " we see letter = closing quote, don't escape.
                    k = j + 1
                    while k < len(s) and s[k] in ' \t\r\n':
                        k += 1
                    after_next = s[k] if k < len(s) else ''
                    if after_next in ',}]:' or after_next == '':
                        # "inner" " , } ] - current " is inner, escape it
                        out.append('\\"')
                    else:
                        # "a" "b" - current " closes first string
                        in_string = False
                        out.append(c)
                else:
                    # Unescaped inner quote - escape it
                    out.append('\\"')
            else:
                in_string = True
                out.append(c)
            i += 1
            continue
        if in_string:
            if c == '\n':
                out.append('\\n')
            elif c == '\r':
                out.append('\\r')
            elif c == '\t':
                out.append('\\t')
            elif ord(c) < 0x20:
                out.append(f'\\u{ord(c):04x}')
            else:
                out.append(c)
        else:
            out.append(c)
        i += 1
    return ''.join(out)


def _try_parse_llm_ranking_array(json_str: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse JSON array from LLM response with multi-stage repair.
    Returns list of candidate dicts or None if parsing fails.
    """
    if not json_str or not json_str.strip():
        return None

    last_err = None

    # Stage 1: direct parse
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError as e:
        last_err = e
        pos = e.pos or 0
        snippet = json_str[max(0, pos - 40):pos + 40]
        log.debug(f"JSON repair stage 1 (direct): {e} | around error: ...{snippet!r}...")

    # Stage 2: sanitize control chars inside string values
    s2 = _sanitize_json_string_values(json_str)
    try:
        parsed = json.loads(s2)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError as e:
        last_err = e
        log.debug(f"JSON repair stage 2 (sanitize strings): {e}")

    # Stage 3: fix missing commas + trailing commas + Python literals
    s3 = s2
    # Normalize line endings first (Gemini may return \r\n)
    s3 = s3.replace('\r\n', '\n').replace('\r', '\n')
    # Insert missing commas – use [ \t]* (horizontal whitespace only) before \n
    # to prevent greedy \s* from crossing lines and eating brackets/braces.
    # Process brackets/braces FIRST, then string-to-string.
    s3 = re.sub(r'(\])[ \t]*\r?\n(\s*")', r'\1,\n\2', s3)    # ]\n"key" → ],\n"key"
    s3 = re.sub(r'(\})[ \t]*\r?\n(\s*\{)', r'\1,\n\2', s3)   # }\n{ → },\n{
    s3 = re.sub(r'(true|false|null|\d)[ \t]*\r?\n(\s*")', r'\1,\n\2', s3)
    s3 = re.sub(r'"[ \t]*\r?\n(\s*")', r'",\n\1', s3)        # "val"\n"key" → "val",\n"key"
    s3 = re.sub(r'"[ \t]+"(?=[a-zA-Z_])', '", "', s3)     # "val" "next" → "val", "next"
    s3 = re.sub(r'""(?=[a-zA-Z_])', '", "', s3)            # "val""key" → "val", "key" (Gemini omits comma)
    # Remove trailing commas
    s3 = re.sub(r',(\s*[}\]])', r'\1', s3)
    s3 = re.sub(r'\bTrue\b', 'true', s3)
    s3 = re.sub(r'\bFalse\b', 'false', s3)
    s3 = re.sub(r'\bNone\b', 'null', s3)
    try:
        parsed = json.loads(s3)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError as e:
        last_err = e
        log.debug(f"JSON repair stage 3 (missing/trailing commas, literals): {e}")

    # Stage 4: single-quote keys/values to double-quote
    try:
        s4 = re.sub(r"'([^']*)':", r'"\1":', s3)
        s4 = re.sub(r":\s*'([^']*)'", r': "\1"', s4)
        parsed = json.loads(s4)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError as e:
        last_err = e
        log.debug(f"JSON repair stage 4 (single quotes): {e}")

    # Stage 4b: targeted fix at error position – try inserting comma where parser choked
    if last_err and hasattr(last_err, 'pos') and 'delimiter' in str(last_err):
        pos = last_err.pos
        s4b = s3[:pos] + ',' + s3[pos:]
        try:
            parsed = json.loads(s4b)
            if isinstance(parsed, list):
                log.debug(f"JSON repair stage 4b (insert comma at pos {pos}): success")
                return parsed
        except json.JSONDecodeError:
            pass

    # Stage 5: partial parse – extract top-level {...} objects via brace matching
    #   (skips braces inside string literals to avoid confusion)
    source = s3
    objects = []
    in_str = False
    depth = 0
    start = -1
    i = 0
    while i < len(source):
        c = source[i]
        if c == '\\' and in_str:
            i += 2
            continue
        if c == '"':
            in_str = not in_str
        elif not in_str:
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    objects.append(source[start : i + 1])
                    start = -1
        i += 1

    if objects:
        parsed = []
        for obj_str in objects:
            for attempt_str in (obj_str, _sanitize_json_string_values(obj_str)):
                try:
                    p = json.loads(attempt_str)
                    if isinstance(p, dict) and p.get("candidate_id"):
                        parsed.append(p)
                        break
                except json.JSONDecodeError:
                    continue
        if parsed:
            log.warning(f"JSON partial parse recovered {len(parsed)}/{len(objects)} candidate objects")
            return parsed

    if last_err:
        log.warning(f"JSON repair failed at all stages. Last error: {last_err}")
    return None


def fuzzy_match(skill: str, skill_set: set, threshold: float = None) -> str:
    """
    Find exact or very close matches only - much stricter matching.
    Uses Config.FUZZY_MATCH_THRESHOLD (0.92) to prevent false positives.
    
    Args:
        skill: Target skill to match (required skill from JD)
        skill_set: Set of candidate skills (from resume)
        threshold: Minimum similarity threshold (defaults to Config.FUZZY_MATCH_THRESHOLD)
        
    Returns:
        Best matching skill or empty string if no match above threshold
    """
    if threshold is None:
        threshold = Config.FUZZY_MATCH_THRESHOLD
    if not skill or not skill_set:
        return ""
    
    # Normalize the target skill (required skill from JD)
    skill_lower = skill.lower().strip()
    
    best_match = ""
    best_score = 0.0
    
    for candidate_skill in skill_set:
        candidate_lower = candidate_skill.lower().strip()
        
        # 1. EXACT MATCH (highest priority)
        if skill_lower == candidate_lower:
            return candidate_skill
        
        # 2. STRICT SUBSTRING MATCHING
        # Only match if:
        # - Both skills are substantial (> 4 chars)
        # - The shorter one is at least 80% of the longer one
        # - One is a complete word boundary in the other
        if len(skill_lower) > 4 and len(candidate_lower) > 4:
            shorter = skill_lower if len(skill_lower) < len(candidate_lower) else candidate_lower
            longer = candidate_lower if len(skill_lower) < len(candidate_lower) else skill_lower
            
            # Calculate length ratio
            length_ratio = len(shorter) / len(longer)
            
            # Only allow substring matches if very similar length (80%+)
            if length_ratio >= 0.8:
                # Check if shorter is in longer with word boundaries
                import re
                pattern = r'\b' + re.escape(shorter) + r'\b'
                if re.search(pattern, longer):
                    score = 0.95  # High score for word-boundary substring match
                else:
                    # Use sequence matcher for close variants
                    score = SequenceMatcher(None, skill_lower, candidate_lower).ratio()
            else:
                # For very different lengths, require higher similarity
                score = SequenceMatcher(None, skill_lower, candidate_lower).ratio()
        else:
            # For short skills (≤4 chars), require very high similarity
            score = SequenceMatcher(None, skill_lower, candidate_lower).ratio()
            # Increase threshold for short skills to avoid false matches
            if len(skill_lower) <= 4 or len(candidate_lower) <= 4:
                score = score * 0.9  # Penalize short skill matches
        
        if score > best_score:
            best_match = candidate_skill
            best_score = score
    
    # Return match only if above STRICTER threshold
    return best_match if best_score >= threshold else ""


async def _enhance_query_with_domain_context_async(jd_text: str, jd: JobDescription) -> Tuple[str, str]:
    """
    Enhance job description query with domain-specific context using hybrid detection.
    
    This prevents false matches where different domains share common words (e.g., 
    "analytics" in marketing vs engineering).
    
    Strategy:
    1. Extract domain using hybrid approach (deterministic + LLM fallback)
    2. Add domain-specific keywords to emphasize domain context
    3. Repeat key domain terms to increase their weight in embeddings
    
    Args:
        jd_text: Original job description text
        jd: JobDescription object
        
    Returns:
        Tuple of (enhanced query text, domain string) for retrieval and LLM context
    """
    domain = await jd._get_domain_context_async()
    
    # Domain-specific emphasis keywords (repeated to increase weight in embeddings)
    domain_emphasis = {
        "software_engineering": (
            "software engineering development programming full-stack backend frontend "
            "cloud devops distributed systems algorithms data structures architecture "
            "web mobile security api microservices scalability"
        ),

        "data_analytics": (
            "data analytics business intelligence reporting dashboards visualization "
            "metrics kpi sql etl tableau power bi insights data analysis reporting"
        ),

        "data_science": (
            "data science statistics statistical analysis hypothesis testing "
            "exploratory data analysis regression classification clustering "
            "pandas numpy scipy r jupyter data mining big data spark hadoop"
        ),

        "ai_ml": (
            "artificial intelligence machine learning deep learning neural networks "
            "tensorflow pytorch keras scikit-learn xgboost natural language processing "
            "computer vision llm large language models transformer bert gpt "
            "reinforcement learning mlops model deployment training feature engineering "
            "ai engineer ml engineer prompt engineering langchain hugging face "
            "generative ai genai chatgpt claude anthropic stable diffusion"
        ),

        "product": (
            "product management strategy roadmap user experience agile scrum backlog "
            "prioritization market analysis customer needs launch lifecycle metrics"
        ),

        "mechanical": (
            "mechanical engineering design cad solidworks manufacturing thermodynamics "
            "fluid dynamics machinery robotics mechatronics automotive aerospace "
            "materials prototyping structural analysis"
        ),

        "electrical": (
            "electrical engineering electronics circuit design power systems pcb "
            "embedded systems control systems instrumentation signal processing "
            "telecommunications hardware microcontrollers"
        ),

        "civil": (
            "civil engineering structural design infrastructure construction bim "
            "geotechnical transportation environmental land development concrete "
            "steel urban planning surveying"
        ),

        "chemical": (
            "chemical engineering process design material science thermodynamics "
            "petrochemical polymers reaction engineering refinery safety "
            "production optimization"
        ),

        "sales": (
            "sales business development revenue generation client acquisition "
            "account management negotiation relationship building target quotas "
            "solution selling pipeline management"
        ),

        "marketing": (
            "marketing strategy digital advertising brand awareness social media "
            "content creation market research customer segmentation campaigns "
            "analytics lead generation"
        ),

        "finance": (
            "financial analysis accounting auditing budget forecasting reporting "
            "tax compliance risk management investment valuation cash flow "
            "corporate finance"
        ),

        "hr": (
            "human resources talent acquisition recruitment employee engagement "
            "workforce planning compensation benefits performance management "
            "training development labor relations"
        ),

        "legal": (
            "legal counsel compliance corporate law litigation contract negotiation "
            "regulatory affairs risk assessment intellectual property governance "
            "dispute resolution"
        ),

        "management": (
            "executive leadership strategic planning operational excellence "
            "team management business growth profit and loss decision making "
            "change management stakeholder relations"
        ),

        "healthcare": (
            "healthcare medical services patient care clinical operations "
            "hospital administration nursing public health diagnosis treatment "
            "regulatory compliance"
        ),

        "ui_ux_design": (
            "user interface user experience ux design ui design interaction design "
            "ux research user research usability testing wireframing prototyping "
            "figma sketch adobe xd design systems information architecture user flows"
        ),

        "creative_design": (
            "creative design visual arts multimedia video production graphic design "
            "branding storytelling animation aesthetic content production "
            "photography videography art direction print design motion graphics"
        ),
        
        "manufacturing": (
            "manufacturing operations production planning quality control lean six sigma "
            "process improvement assembly machining fabrication cnc production supervisor "
            "manufacturing engineer plant manager"
        ),
        
        "supply_chain": (
            "supply chain logistics procurement warehouse distribution "
            "inventory management sourcing vendor management purchasing "
            "supply chain management logistics coordinator"
        ),

        "operations": (
            "operations business operations operational excellence process improvement "
            "operations analyst business process workflow operations manager"
        ),
        
        "customer_support": (
            "customer service support help desk technical support troubleshooting "
            "ticketing zendesk intercom freshdesk call center contact center "
            "live chat email support it support"
        ),

        "customer_success": (
            "customer success csm customer success manager onboarding retention "
            "churn renewal implementation client satisfaction relationship management "
            "adoption account health qbr quarterly business review account management "
            "customer advocacy expansion upsell"
        ),
        
        "admin": (
            "administrative support office management executive assistant scheduling "
            "clerical organization data entry receptionist"
        ),
        
        "education": (
            "education teaching training instructional design curriculum learning "
            "development academic faculty k-12 higher education"
        ),
        
        "science": (
            "science research laboratory r&d biology chemistry biotech clinical scientist "
            "genomics microbiology pharmaceutical research and development"
        ),
        
        "construction": (
            "construction site management civil engineering facilities real estate "
            "safety project estimation blueprint"
        ),
        
        "consulting": (
            "management consulting strategy advisory business transformation "
            "mergers acquisitions due diligence problem solving framework "
            "client engagement analysis"
        ),
        
        "hospitality": (
            "hospitality hotel restaurant culinary food service event planning "
            "catering banquet front desk concierge tourism travel tea maker coffee maker "
            "beverage service pantry kitchen helper server waiter waitress barista bartender "
            "host hostess busser dishwasher line cook prep cook sous chef pastry chef "
            "food runner room service housekeeping laundry bellhop valet parking attendant "
            "guest services reservations front office night auditor"
        ),
        
        "security": (
            "security guard security officer security personnel loss prevention surveillance "
            "access control patrol patrolling security management security supervisor "
            "armed guard unarmed guard event security retail security corporate security "
            "facility security gate guard watchman cctv monitoring security operations"
        ),
        
        "cleaning_janitorial": (
            "janitor janitorial cleaner cleaning custodial custodian housekeeping sanitation "
            "sanitation worker cleaning staff office cleaner building cleaner floor care "
            "window cleaning carpet cleaning deep cleaning maintenance cleaning industrial cleaning "
            "commercial cleaning residential cleaning maid house cleaner"
        ),
        
        "personal_care": (
            "personal care personal care aide caregiver home care home health aide "
            "elderly care senior care elder care disability support disability care "
            "companion care respite care assisted living nursing aide cna "
            "certified nursing assistant patient care personal support worker"
        ),
        
        "childcare": (
            "childcare child care daycare day care nanny babysitter babysitting "
            "early childhood preschool teacher daycare teacher childcare provider "
            "childcare worker after school care summer camp childcare assistant "
            "toddler care infant care child development"
        ),
        
        "retail": (
            "retail store management merchandising inventory point of sale "
            "customer service sales associate visual merchandising e-commerce"
        ),
        
        "trades": (
            "trades skilled trades construction trades plumbing electrical "
            "carpentry welding hvac journeyman master tradesman certification"
        ),
        
        "real_estate": (
            "real estate property management commercial residential leasing "
            "brokerage property development investment mls"
        ),
        
        "fitness": (
            "fitness personal training nutrition wellness health coaching "
            "strength conditioning exercise physiology group fitness"
        ),
        
        "beauty": (
            "beauty cosmetology esthetics hairstyling makeup skincare salon "
            "spa nail technician barber"
        ),
        
        "aviation": (
            "aviation pilot flight attendant air traffic control aircraft "
            "faa commercial aviation maintenance airport operations"
        ),
        
        "transportation": (
            "transportation logistics trucking delivery cdl fleet management "
            "dispatch route planning shipping freight"
        ),
        
        "military": (
            "military veteran armed forces defense security clearance "
            "military service leadership discipline"
        ),
        
        "nonprofit": (
            "nonprofit ngo fundraising grant writing donor relations "
            "volunteer coordination advocacy community outreach philanthropy"
        ),
        
        "public_sector": (
            "government public administration policy regulatory compliance "
            "public service federal state local civil service"
        ),
        
        "arts": (
            "arts performing arts theater dance music acting directing "
            "fine arts sculpture painting museum gallery curation"
        ),
        
        "sports": (
            "sports coaching athletic training sports management "
            "strength conditioning sports medicine player development"
        ),
        
        "environmental": (
            "environmental conservation ecology sustainability wildlife "
            "forestry environmental science renewable energy climate policy"
        ),
    }
    
    # Get emphasis keywords for detected domain
    emphasis_keywords = domain_emphasis.get(domain, "")
    
    # Enhance query: Add domain context at the beginning (most important for embeddings)
    enhanced_text = f"DOMAIN: {domain.upper()}\n"
    if emphasis_keywords:
        enhanced_text += f"Domain Keywords: {emphasis_keywords}\n"
    enhanced_text += jd_text
    
    # Repeat key skills to increase their weight (helps with skill-focused matching)
    if jd.required_skills:
        # Repeat top 5 skills to emphasize them
        top_skills = jd.required_skills[:5]
        enhanced_text += f"\nKey Skills Emphasis: {' '.join(top_skills * 2)}"  # Repeat once
    
    return enhanced_text, domain


def extract_personal_info(parsed_resume: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extract personal information from parsed resume.
    FIXED: Handles cases where name/email/phone might be lists.
    Supports both old format (personalInformation) and new format (ContactDetails).
    
    Args:
        parsed_resume: Parsed resume dictionary
        
    Returns:
        Tuple of (name, email, phone) - all as strings
    """
    personal_info = parsed_resume.get("personalInformation", {})
    contact_details = parsed_resume.get("ContactDetails", {}) or parsed_resume.get("contact_details", {})
    
    # FIXED: Extract name - handle both Name and name (groq format)
    name = (
        personal_info.get("name") or
        parsed_resume.get("Name") or
        parsed_resume.get("name") or
        "Unknown"
    )
    # CRITICAL FIX: Convert list to string if needed
    if isinstance(name, list):
        name = name[0] if name else "Unknown"
    
    # FIXED: Extract email and handle list case
    # Check ContactDetails first (new format), then fall back to old format
    email = (
        contact_details.get("Email") or
        personal_info.get("email") or
        parsed_resume.get("Email") or
        ""
    )
    if isinstance(email, list):
        email = email[0] if email else ""
    
    # FIXED: Extract phone and handle list case
    # Check ContactDetails first (new format), then fall back to old format
    phone = (
        contact_details.get("Phone") or
        personal_info.get("phone") or
        parsed_resume.get("Phone") or
        personal_info.get("contactNumber") or
        ""
    )
    if isinstance(phone, list):
        phone = phone[0] if phone else ""
    
    return str(name), str(email), str(phone)


def _extract_skills_from_resume(parsed_resume: Dict[str, Any]) -> set:
    """
    Extract candidate skill names from structured resume.
    Handles multiple formats: Skills, skills, SkillName, skillName, name, Skill, etc.
    Returns a set of lowercase strings for overlap checks.
    """
    out = set()
    
    # Try multiple possible skill field locations
    raw = (
        parsed_resume.get("Skills") or 
        parsed_resume.get("skills") or 
        parsed_resume.get("Skill") or
        parsed_resume.get("skill") or
        parsed_resume.get("technical_skills") or
        parsed_resume.get("technicalSkills") or
        []
    )
    
    if not isinstance(raw, list):
        # Handle case where skills is a string (comma-separated)
        if isinstance(raw, str) and raw.strip():
            for s in raw.split(","):
                if s.strip():
                    out.add(s.strip().lower())
        return out
    
    for s in raw:
        if isinstance(s, str) and s.strip():
            out.add(s.strip().lower())
        elif isinstance(s, dict):
            # Try multiple possible keys for skill name
            name = (
                s.get("SkillName") or 
                s.get("skillName") or 
                s.get("name") or 
                s.get("Name") or
                s.get("skill") or
                s.get("Skill") or
                s.get("title") or
                s.get("Title") or
                ""
            )
            if name and isinstance(name, str) and name.strip():
                out.add(name.strip().lower())
    
    return out


def _skill_overlap_ratio(required_skills: List[str], candidate_skills_set: set) -> float:
    """
    Compute overlap as fraction of required_skills that appear in candidate skills (exact match, case-insensitive).
    Returns 0.0–1.0. If required_skills is empty, returns 1.0 (no filter).
    """
    if not required_skills:
        return 1.0
    req_lower = [r.strip().lower() for r in required_skills if r and isinstance(r, str)]
    if not req_lower:
        return 1.0
    matched = sum(1 for r in req_lower if r in candidate_skills_set)
    return matched / len(req_lower)


# ============================================================================
# TIER-BASED RANKING: Helper Functions
# ============================================================================

# Cache for domain embeddings (computed once per domain)
_domain_embedding_cache: Dict[str, List[float]] = {}


# Max texts per batch for Gemini embed_content (API may have limits; 20 is conservative)
_EMBED_BATCH_SIZE = 20


async def _get_text_embedding(text: str) -> List[float]:
    """Get embedding using Gemini embedding model."""
    results = await _get_text_embeddings_batch([text])
    return results[0] if results else []


async def _get_text_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Batch embed multiple texts. Uses gemini_embedding_cache for persistence across
    restarts. Reduces embedding requests when processing many work experiences
    (e.g. in _calculate_domain_relevant_experience).
    """
    if not texts:
        return []
    try:
        from google import genai

        client = genai.Client(api_key=_settings.GOOGLE_API_KEY)
        # Truncate each text to 2000 chars
        truncated = [t[:2000] for t in texts]
        all_embeddings: List[List[float]] = []

        # Process in batches to respect API limits (cached via gemini_embedding_cache)
        for i in range(0, len(truncated), _EMBED_BATCH_SIZE):
            batch = truncated[i : i + _EMBED_BATCH_SIZE]
            vectors = await asyncio.to_thread(
                lambda b=batch: cached_embed_texts(
                    client, "models/gemini-embedding-001", b
                )
            )
            if vectors and len(vectors) == len(batch):
                all_embeddings.extend(vectors)
            else:
                # Pad with empty for failed batch
                all_embeddings.extend([[]] * len(batch))
        return all_embeddings
    except Exception as e:
        log.warning(f"⚠️ Batch embedding failed: {e}")
        return [[]] * len(texts)


async def _get_domain_embedding(domain_description: str) -> List[float]:
    """Get or compute cached embedding for domain description."""
    if domain_description in _domain_embedding_cache:
        return _domain_embedding_cache[domain_description]
    
    embedding = await _get_text_embedding(domain_description)
    if embedding:
        _domain_embedding_cache[domain_description] = embedding
    return embedding


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np
    if not vec1 or not vec2:
        return 0.0
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _calculate_tenure_years(exp: Dict[str, Any]) -> float:
    """
    Calculate years worked at a position from start/end dates.
    Returns 0.0 if dates cannot be parsed.
    """
    from dateutil import parser as date_parser
    from datetime import datetime
    
    # Try various date field names
    start_str = (
        exp.get("start_date") or exp.get("startDate") or 
        exp.get("StartDate") or exp.get("from") or exp.get("From") or ""
    )
    end_str = (
        exp.get("end_date") or exp.get("endDate") or 
        exp.get("EndDate") or exp.get("to") or exp.get("To") or ""
    )
    
    if not start_str:
        return 0.0
    
    try:
        # Parse start date
        start_date = date_parser.parse(str(start_str), fuzzy=True)
        
        # Parse end date (or use current date for "Present")
        end_str_lower = str(end_str).lower().strip()
        if not end_str or end_str_lower in ("present", "current", "now", "ongoing", ""):
            end_date = datetime.now()
        else:
            end_date = date_parser.parse(str(end_str), fuzzy=True)
        
        # Calculate difference in years
        delta = end_date - start_date
        years = delta.days / 365.25
        return max(0.0, round(years, 2))
    except Exception:
        return 0.0


async def _calculate_domain_relevant_experience(
    work_experience: List[Dict[str, Any]],
    jd_domain_description: str,
    domain_embedding: Optional[List[float]] = None
) -> float:
    """
    Calculate years of experience relevant to the JD domain using embeddings.
    
    Args:
        work_experience: List of work experience entries from resume
        jd_domain_description: Full description of domain for embedding comparison
        domain_embedding: Pre-computed domain embedding (optional, for efficiency)
        
    Returns:
        Total domain-relevant years (float)
    """
    if not work_experience:
        return 0.0
    
    # Get JD domain embedding
    if domain_embedding is None:
        domain_embedding = await _get_domain_embedding(jd_domain_description)
    
    if not domain_embedding:
        # Fallback: return total experience if embedding fails
        return sum(_calculate_tenure_years(exp) for exp in work_experience)
    
    domain_relevant_years = 0.0

    # Build experience texts for batch embedding
    exp_texts = []
    for exp in work_experience:
        title = exp.get("title") or exp.get("job_title") or exp.get("JobTitle") or ""
        company = exp.get("company") or exp.get("Company") or ""
        description = exp.get("description") or exp.get("responsibilities") or exp.get("Description") or ""
        exp_text = f"{title} at {company}. {description}"[:500]
        exp_texts.append(exp_text)

    # Batch embed all experiences in one (or few) API call(s)
    exp_embeddings = await _get_text_embeddings_batch(exp_texts)

    for i, exp in enumerate(work_experience):
        try:
            exp_embedding = exp_embeddings[i] if i < len(exp_embeddings) else []
            if not exp_embedding:
                continue

            # Calculate similarity
            similarity = _cosine_similarity(domain_embedding, exp_embedding)

            # Calculate tenure
            years = _calculate_tenure_years(exp)

            # Only count if similarity >= threshold
            if similarity >= 0.7:
                domain_relevant_years += years  # 100% credit for high similarity
            elif similarity >= 0.5:
                domain_relevant_years += years * 0.5  # 50% credit for moderate similarity
            # < 0.5: no credit (not relevant)

        except Exception as e:
            log.debug(f"⚠️ Error processing experience: {e}")
            continue

    return round(domain_relevant_years, 2)


def _parse_experience_range(experience_str: str) -> Tuple[float, float]:
    """
    Parse experience requirement string into (min_years, max_years).
    
    Examples:
        "4-10 years" -> (4.0, 10.0)
        "5+ years" -> (5.0, 99.0)
        "3 years" -> (3.0, 3.0)
        "Senior (8+ years)" -> (8.0, 99.0)
    """
    if not experience_str:
        return (0.0, 99.0)  # No requirement = any experience OK
    
    exp_lower = experience_str.lower()
    
    # Try to find range pattern: "4-10 years"
    range_match = re.search(r'(\d+)\s*[-–to]+\s*(\d+)', exp_lower)
    if range_match:
        return (float(range_match.group(1)), float(range_match.group(2)))
    
    # Try to find "X+ years" pattern
    plus_match = re.search(r'(\d+)\s*\+', exp_lower)
    if plus_match:
        return (float(plus_match.group(1)), 99.0)
    
    # Try to find single number
    single_match = re.search(r'(\d+)', exp_lower)
    if single_match:
        years = float(single_match.group(1))
        return (years, years + 2)  # Allow some flexibility
    
    return (0.0, 99.0)


# ============================================================================
# TIER CRITERIA EVALUATION FUNCTIONS
# ============================================================================

# Tier thresholds (configurable) - VERY RELAXED for better tier distribution
TIER_MUST_HAVE_THRESHOLD = 0.20  # Require 20% of must-have skills (very relaxed)
TIER_REQUIRED_SKILLS_THRESHOLD = 0.15  # Require 15% of required skills (very relaxed)
TIER_EXPERIENCE_FLEXIBILITY = 5  # Allow ±5 years from requirement (very relaxed)
TIER_MIN_MUST_HAVE_COUNT = 2  # Minimum absolute count of must-have skills to match (lowered from 3)
TIER_MIN_REQUIRED_COUNT = 2  # Minimum absolute count of required skills to match


def _fuzzy_skill_match(skill1: str, skill2: str) -> bool:
    """
    Check if two skills match (exact or fuzzy).
    Handles variations like 'python' vs 'python3', 'javascript' vs 'js', etc.
    """
    s1 = skill1.lower().strip()
    s2 = skill2.lower().strip()
    
    # Exact match
    if s1 == s2:
        return True
    
    # One contains the other
    if s1 in s2 or s2 in s1:
        return True
    
    # Common aliases
    aliases = {
        "javascript": ["js", "javascript", "ecmascript"],
        "typescript": ["ts", "typescript"],
        "python": ["python", "python3", "py"],
        "react": ["react", "reactjs", "react.js"],
        "node": ["node", "nodejs", "node.js"],
        "vue": ["vue", "vuejs", "vue.js"],
        "angular": ["angular", "angularjs", "angular.js"],
        "c++": ["c++", "cpp", "cplusplus"],
        "c#": ["c#", "csharp", "c sharp"],
        "sql": ["sql", "mysql", "postgresql", "postgres", "mssql", "sqlite"],
        "aws": ["aws", "amazon web services"],
        "gcp": ["gcp", "google cloud", "google cloud platform"],
        "azure": ["azure", "microsoft azure"],
        "docker": ["docker", "containerization"],
        "kubernetes": ["kubernetes", "k8s"],
        "machine learning": ["ml", "machine learning", "machinelearning"],
        "artificial intelligence": ["ai", "artificial intelligence"],
        "api": ["api", "rest", "restful", "rest api"],
        "git": ["git", "github", "gitlab", "bitbucket"],
    }
    
    for base_skill, alias_list in aliases.items():
        if s1 in alias_list and s2 in alias_list:
            return True
        if any(alias in s1 for alias in alias_list) and any(alias in s2 for alias in alias_list):
            return True
    
    return False


def _count_matching_skills(required_skills: List[str], candidate_skills: set) -> int:
    """Count skills that match (exact or fuzzy)."""
    matched = 0
    for req_skill in required_skills:
        req_lower = req_skill.strip().lower()
        # Check exact match first
        if req_lower in candidate_skills:
            matched += 1
        # Then check fuzzy match
        elif any(_fuzzy_skill_match(req_lower, cand) for cand in candidate_skills):
            matched += 1
    return matched


def _evaluate_skills_match(
    candidate_skills: set,
    required_skills: List[str],
    must_have_skills: List[str]
) -> Tuple[bool, float, float, int, int]:
    """
    Criterion A: Skills Match (VERY RELAXED thresholds with fuzzy matching)
    
    Logic:
    - If must_have_skills specified: require TIER_MUST_HAVE_THRESHOLD (20%) of them
      OR at least TIER_MIN_MUST_HAVE_COUNT (2) skills matched
    - Plus at least TIER_REQUIRED_SKILLS_THRESHOLD (15%) of required_skills
      OR at least TIER_MIN_REQUIRED_COUNT (2) skills matched
    
    Returns:
        (is_match: bool, match_percentage: float, must_have_percentage: float, 
         must_have_matched: int, must_have_total: int)
    """
    if not required_skills:
        return True, 100.0, 100.0, 0, 0
    
    must_have_percentage = 100.0
    must_have_matched = 0
    must_have_total = 0
    
    # Check must-have skills (very relaxed - require 20% OR minimum count)
    if must_have_skills:
        must_have_lower = [s.strip().lower() for s in must_have_skills if s]
        must_have_total = len(must_have_lower)
        must_have_matched = _count_matching_skills(must_have_lower, candidate_skills)
        must_have_percentage = (must_have_matched / must_have_total * 100) if must_have_total > 0 else 100.0
        
        # Pass if meets EITHER percentage threshold OR minimum absolute count
        meets_percentage = must_have_percentage >= (TIER_MUST_HAVE_THRESHOLD * 100)
        meets_min_count = must_have_matched >= TIER_MIN_MUST_HAVE_COUNT
        
        if not (meets_percentage or meets_min_count):
            # Still fail must-have, calculate overall overlap for info
            req_matched = _count_matching_skills(required_skills, candidate_skills)
            overlap = (req_matched / len(required_skills)) if required_skills else 1.0
            return False, round(overlap * 100, 1), round(must_have_percentage, 1), must_have_matched, must_have_total
    
    # Check overall skill overlap (with fuzzy matching)
    req_matched = _count_matching_skills(required_skills, candidate_skills)
    overlap = (req_matched / len(required_skills)) if required_skills else 1.0
    
    # Pass if meets EITHER percentage threshold OR minimum absolute count
    meets_pct = overlap >= TIER_REQUIRED_SKILLS_THRESHOLD
    meets_count = req_matched >= TIER_MIN_REQUIRED_COUNT
    is_match = meets_pct or meets_count
    
    return is_match, round(overlap * 100, 1), round(must_have_percentage, 1), must_have_matched, must_have_total


async def _evaluate_experience_match(
    candidate_resume: Dict[str, Any],
    jd_domain_description: str,
    experience_requirement: str,
    domain_embedding: Optional[List[float]] = None
) -> Tuple[bool, float, float, float, float]:
    """
    Criterion B: Experience Match (SIMPLIFIED - uses total experience, not domain-specific embeddings)
    
    Returns:
        (is_match: bool, total_years: float, total_years: float, min_years: float, max_years: float)
        
    Note: domain_years and total_years return the same value now (total experience).
          Domain-specific embedding calculation removed for reliability and performance.
    """
    min_years, max_years = _parse_experience_range(experience_requirement)
    
    # Get work experience from resume - try MANY possible field names
    work_experience = (
        candidate_resume.get("work_experience") or 
        candidate_resume.get("WorkExperience") or 
        candidate_resume.get("experience") or
        candidate_resume.get("Experience") or
        candidate_resume.get("workExperience") or
        candidate_resume.get("employment") or
        candidate_resume.get("Employment") or
        candidate_resume.get("professional_experience") or
        candidate_resume.get("professionalExperience") or
        []
    )
    
    if not isinstance(work_experience, list):
        work_experience = []
    
    # Check for total_experience_years field first (most reliable) - try MANY variations
    total_exp_field = (
        candidate_resume.get("total_experience_years") or
        candidate_resume.get("totalExperienceYears") or
        candidate_resume.get("TotalExperienceYears") or
        candidate_resume.get("years_of_experience") or
        candidate_resume.get("yearsOfExperience") or
        candidate_resume.get("YearsOfExperience") or
        candidate_resume.get("experience_years") or
        candidate_resume.get("experienceYears") or
        candidate_resume.get("total_years") or
        candidate_resume.get("totalYears") or
        0
    )
    
    # Parse the total_experience_years field (could be "10 years", "5.5", etc.)
    total_years_from_field = 0.0
    if total_exp_field:
        try:
            if isinstance(total_exp_field, (int, float)):
                total_years_from_field = float(total_exp_field)
            elif isinstance(total_exp_field, str):
                # Parse strings like "10 years", "5.5 years", "2 years 3 months"
                import re
                # Try to extract number
                years_match = re.search(r'(\d+\.?\d*)\s*(?:years?|yrs?)?', total_exp_field.lower())
                if years_match:
                    total_years_from_field = float(years_match.group(1))
                # Add months if present
                months_match = re.search(r'(\d+)\s*months?', total_exp_field.lower())
                if months_match:
                    total_years_from_field += float(months_match.group(1)) / 12
        except (ValueError, TypeError):
            total_years_from_field = 0.0
    
    # Calculate total years from work experience list as backup
    total_years_from_list = sum(_calculate_tenure_years(exp) for exp in work_experience)
    total_years = max(total_years_from_field, total_years_from_list)
    
    # SIMPLIFIED: Use total experience directly (no domain-specific embedding calculation)
    # The LLM already provides domain_relevant_experience_years in its analysis
    # This avoids embedding API calls and improves reliability
    domain_years = total_years
    
    # Check if within range (allow flexibility based on TIER_EXPERIENCE_FLEXIBILITY)
    # More lenient matching criteria:
    is_match = (
        # Within flexible range
        (min_years - TIER_EXPERIENCE_FLEXIBILITY) <= total_years <= (max_years + TIER_EXPERIENCE_FLEXIBILITY)
        # Junior roles: any experience counts
        or (total_years > 0 and min_years <= 3)
        # Has at least minimum required experience
        or (total_years >= min_years)
        # Substantial experience (5+ years) should generally pass
        or (total_years >= 5)
        # Has substantial experience (5+ years) even if above max
        or (total_years >= 5)
    )
    
    return is_match, domain_years, total_years, min_years, max_years


def _evaluate_location_match(
    candidate_location: str,
    jd_location: str,
    work_mode: str
) -> Tuple[bool, str]:
    """
    Criterion C: Location Match (LENIENT - errs on the side of inclusion)
    
    Returns:
        (is_match: bool, reason: str)
    """
    work_mode_lower = (work_mode or "").lower()
    
    # Remote/hybrid = always match
    if "remote" in work_mode_lower or "hybrid" in work_mode_lower or "wfh" in work_mode_lower:
        return True, "Remote/Hybrid position"
    
    # Flexible/any location = always match
    if not jd_location or jd_location.lower() in ("n/a", "not specified", "any", "", "flexible", "various", "multiple"):
        return True, "No location requirement"
    
    # If candidate location unknown, be LENIENT and pass (can verify later)
    if not candidate_location or candidate_location.lower() in ("n/a", "not specified", "", "unknown"):
        return True, "Candidate location unknown (lenient pass)"
    
    # Simple location matching (city/state/country)
    jd_loc_lower = jd_location.lower()
    cand_loc_lower = candidate_location.lower()
    
    # Check for common location words
    jd_parts = set(re.split(r'[,\s/\-\.]+', jd_loc_lower))
    cand_parts = set(re.split(r'[,\s/\-\.]+', cand_loc_lower))
    
    # Remove common stop words
    stop_words = {"the", "of", "in", "at", ""}
    jd_parts = jd_parts - stop_words
    cand_parts = cand_parts - stop_words
    
    # Check overlap (any match counts)
    if jd_parts & cand_parts:
        return True, f"Location match: {jd_parts & cand_parts}"
    
    # Same country matching (be lenient with country-level matches)
    country_aliases = {
        "india": ["india", "in", "ind"],
        "usa": ["usa", "us", "united states", "america"],
        "uk": ["uk", "united kingdom", "england", "britain", "gb"],
        "canada": ["canada", "ca"],
        "australia": ["australia", "au"],
        "germany": ["germany", "de"],
    }
    
    for country, aliases in country_aliases.items():
        jd_has_country = any(alias in jd_loc_lower for alias in aliases)
        cand_has_country = any(alias in cand_loc_lower for alias in aliases)
        if jd_has_country and cand_has_country:
            return True, f"Same country: {country}"
    
    # If we can't determine, be lenient and pass
    if len(jd_parts) <= 1 or len(cand_parts) <= 1:
        return True, "Insufficient location data (lenient pass)"
    
    return False, f"Location mismatch: {candidate_location} vs {jd_location}"


def _evaluate_education_match(
    candidate_education: List[Dict[str, Any]],
    required_education: str
) -> Tuple[bool, str]:
    """
    Criterion D: Education Match (LENIENT - experience can often substitute for education)
    
    Returns:
        (is_match: bool, reason: str)
    """
    if not required_education or required_education.lower() in ("n/a", "not specified", "any", "", "preferred", "or equivalent"):
        return True, "No strict education requirement"
    
    # If no education listed and JD has a specific requirement, fail the check
    # This allows Tier 2 (Strong Match) to capture skilled+experienced candidates without verified education
    if not candidate_education:
        return False, f"No education listed (JD requires: {required_education})"
    
    req_lower = required_education.lower()
    
    # Education level hierarchy
    education_levels = {
        "phd": 5, "doctorate": 5, "doctoral": 5, "ph.d": 5,
        "master": 4, "mba": 4, "ms": 4, "ma": 4, "msc": 4, "mtech": 4, "m.tech": 4, "m.s": 4, "m.a": 4,
        "bachelor": 3, "bs": 3, "ba": 3, "bsc": 3, "btech": 3, "b.tech": 3, "be": 3, "b.e": 3, "bcom": 3, "b.com": 3, "b.s": 3, "b.a": 3, "degree": 3, "graduate": 3, "graduation": 3,
        "associate": 2, "diploma": 2,
        "high school": 1, "secondary": 1, "12th": 1, "hsc": 1
    }
    
    # Find required level
    req_level = 0
    for edu_name, level in education_levels.items():
        if edu_name in req_lower:
            req_level = max(req_level, level)
    
    # If requirement is vague (e.g., "relevant degree"), assume bachelor's level (3)
    if req_level == 0 and ("degree" in req_lower or "graduate" in req_lower or "education" in req_lower):
        req_level = 3
    
    # Find candidate's highest level
    cand_level = 0
    cand_degree = "Unknown"
    for edu in candidate_education:
        if isinstance(edu, dict):
            # Try multiple keys for degree field
            degree = (
                edu.get("degree") or edu.get("Degree") or 
                edu.get("qualification") or edu.get("Qualification") or
                edu.get("course") or edu.get("Course") or
                edu.get("name") or edu.get("Name") or
                ""
            ).lower()
            
            # Also check institution name as backup
            institution = (edu.get("institution") or edu.get("Institution") or edu.get("school") or "").lower()
            combined = f"{degree} {institution}"
            
            for edu_name, level in education_levels.items():
                if edu_name in combined:
                    if level > cand_level:
                        cand_level = level
                        cand_degree = edu.get("degree") or edu.get("Degree") or degree
        elif isinstance(edu, str):
            # Handle case where education is a list of strings
            edu_lower = edu.lower()
            for edu_name, level in education_levels.items():
                if edu_name in edu_lower:
                    if level > cand_level:
                        cand_level = level
                        cand_degree = edu
    
    # Check if candidate meets or exceeds requirement
    if cand_level >= req_level:
        return True, f"Education met: {cand_degree}"
    
    # Allow one level below the requirement (e.g., diploma when bachelor required)
    if req_level > 0 and cand_level >= req_level - 1 and cand_level >= 2:
        return True, f"Close education match: {cand_degree} (one level below, accepted)"
    
    # Below requirement — fail so Tier 2/4 can be reached
    if cand_level > 0:
        return False, f"Education below requirement: {cand_degree} vs required {required_education}"
    
    return False, f"Education not verified vs required {required_education}"


def classify_candidate_tier(
    skills_match: bool,
    experience_match: bool,
    location_match: bool,
    education_match: bool
) -> Tuple[int, str, str]:
    """
    Classify candidate into tier based on boolean criteria.
    Location (C) is stored but NOT used as a gate — only a tiebreaker.
    
    A = Skills Match
    B = Experience Match (domain-relevant)
    C = Location Match (tiebreaker only)
    D = Education Match
    
    Returns:
        (tier_number, tier_name, tier_action)
    """
    A, B, D = skills_match, experience_match, education_match
    
    if A and B and D:
        return (1, "Unicorn", "Immediate interview")
    elif A and B:
        return (2, "Strong Match", "High priority - verify education fit")
    elif A and (B or D):
        return (3, "High Potential", "Good for junior/associate version of role")
    elif B and D:
        return (4, "Pivoter", "Consider if skills are trainable")
    else:
        return (5, "No Match", "Skip")


async def compute_candidate_tier(
    candidate: Dict[str, Any],
    required_skills: List[str],
    must_have_skills: List[str],
    jd_domain: str,
    jd_experience: str,
    jd_location: str,
    work_mode: str,
    jd_education: str,
    domain_embedding: Optional[List[float]] = None,
    debug_logging: bool = True  # Default to True for better visibility
) -> Dict[str, Any]:
    """
    Compute tier classification for a single candidate.
    
    Returns candidate dict with tier info added:
    - tier: int (1-5)
    - tier_name: str
    - tier_action: str
    - criteria: dict with A, B, C, D details
    """
    candidate_id = candidate.get("candidate_id", "unknown")
    
    # ROBUST RESUME EXTRACTION - try multiple paths
    resume = candidate.get("candidate", {}).get("structuredResume", {})
    if not resume:
        resume = candidate.get("candidate", {})
    if not resume:
        resume = candidate.get("structuredResume", {})
    if not resume:
        resume = candidate.get("resume", {})
    if not resume:
        resume = candidate  # Last resort: use candidate dict directly
    
    # Extract candidate skills with logging
    candidate_skills = _extract_skills_from_resume(resume)
    
    # ALWAYS log candidate skills found for debugging tier issues
    log.info(f"   📋 {candidate_id}: Found {len(candidate_skills)} skills")
    if len(candidate_skills) <= 15:
        log.info(f"      Skills: {sorted(candidate_skills)}")
    else:
        log.info(f"      Skills (first 15): {sorted(list(candidate_skills)[:15])}")
    
    # Criterion A: Skills Match (now returns 5 values)
    skills_match, skill_pct, must_have_pct, must_have_matched, must_have_total = _evaluate_skills_match(
        candidate_skills, required_skills, must_have_skills
    )
    
    # ALWAYS log skill matching results
    log.info(
        f"   🔹 {candidate_id} Criterion A (Skills): match={skills_match}, "
        f"skill_pct={skill_pct}%, must_have={must_have_matched}/{must_have_total} ({must_have_pct}%)"
    )
    
    # Criterion B: Domain-Relevant Experience Match (now returns 5 values)
    domain_description = DOMAIN_DESCRIPTIONS.get(jd_domain, DOMAIN_DESCRIPTIONS.get("other", ""))
    exp_match, domain_years, total_years, min_years, max_years = await _evaluate_experience_match(
        resume, domain_description, jd_experience, domain_embedding
    )
    
    # ALWAYS log experience matching results
    log.info(
        f"   🔹 {candidate_id} Criterion B (Experience): match={exp_match}, "
        f"total_years={total_years}, required={min_years}-{max_years}"
    )
    
    # Criterion C: Location Match - try multiple location fields
    candidate_location = (
        resume.get("location") or resume.get("Location") or
        resume.get("current_location") or resume.get("CurrentLocation") or
        resume.get("city") or resume.get("City") or ""
    )
    if not candidate_location:
        personal_info = resume.get("personalInformation") or resume.get("personal_info") or resume.get("PersonalInformation") or {}
        if isinstance(personal_info, dict):
            candidate_location = (
                personal_info.get("location") or personal_info.get("Location") or
                personal_info.get("city") or personal_info.get("City") or
                personal_info.get("state") or personal_info.get("State") or
                personal_info.get("country") or personal_info.get("Country") or ""
            )
        # Also try contact details
        contact = resume.get("ContactDetails") or resume.get("contactDetails") or resume.get("contact") or {}
        if isinstance(contact, dict) and not candidate_location:
            candidate_location = (
                contact.get("location") or contact.get("Location") or
                contact.get("city") or contact.get("City") or ""
            )
    loc_match, loc_reason = _evaluate_location_match(candidate_location, jd_location, work_mode)
    
    # ALWAYS log location matching results
    log.info(f"   🔹 {candidate_id} Criterion C (Location): match={loc_match}, reason={loc_reason}")
    
    # Criterion D: Education Match - try multiple education fields
    candidate_education = (
        resume.get("education") or resume.get("Education") or
        resume.get("educations") or resume.get("Educations") or
        resume.get("academic") or resume.get("Academic") or
        []
    )
    if not isinstance(candidate_education, list):
        if isinstance(candidate_education, dict):
            candidate_education = [candidate_education]
        elif isinstance(candidate_education, str):
            candidate_education = [{"degree": candidate_education}]
        else:
            candidate_education = []
    edu_match, edu_reason = _evaluate_education_match(candidate_education, jd_education)
    
    # ALWAYS log education matching results
    log.info(f"   🔹 {candidate_id} Criterion D (Education): match={edu_match}, reason={edu_reason}")
    
    # Classify tier
    tier, tier_name, tier_action = classify_candidate_tier(
        skills_match, exp_match, loc_match, edu_match
    )
    
    # ALWAYS log final tier classification
    log.info(f"   🏷️ {candidate_id} → Tier {tier} ({tier_name}): A={skills_match}, B={exp_match}, C={loc_match}, D={edu_match}")
    
    # Add tier info to candidate
    candidate["tier"] = tier
    candidate["tier_name"] = tier_name
    candidate["tier_action"] = tier_action
    candidate["criteria"] = {
        "skills_match": skills_match,
        "skills_match_percentage": skill_pct,
        "must_have_percentage": must_have_pct,
        "must_have_matched": must_have_matched,
        "must_have_total": must_have_total,
        "experience_match": exp_match,
        "domain_relevant_years": domain_years,
        "total_experience_years": total_years,
        "experience_range": f"{min_years}-{max_years} years",
        "location_match": loc_match,
        "location_reason": loc_reason,
        "education_match": edu_match,
        "education_reason": edu_reason,
    }
    
    return candidate


async def compute_tiers_for_candidates(
    candidates: List[Dict[str, Any]],
    required_skills: List[str],
    must_have_skills: List[str],
    jd_domain: str,
    jd_experience: str,
    jd_location: str,
    work_mode: str,
    jd_education: str
) -> List[Dict[str, Any]]:
    """
    Compute tier classification for all candidates in parallel.
    
    Returns candidates with tier info added, sorted by tier (ascending).
    """
    if not candidates:
        return []
    
    # Pre-compute domain embedding for efficiency
    domain_description = DOMAIN_DESCRIPTIONS.get(jd_domain, DOMAIN_DESCRIPTIONS.get("other", ""))
    domain_embedding = await _get_domain_embedding(domain_description)
    
    log.info(f"🏷️ Computing tiers for {len(candidates)} candidates...")
    log.info(f"   Thresholds: must_have={TIER_MUST_HAVE_THRESHOLD*100}% (or min {TIER_MIN_MUST_HAVE_COUNT} skills), required_skills={TIER_REQUIRED_SKILLS_THRESHOLD*100}%")
    log.info(f"   Experience flexibility: ±{TIER_EXPERIENCE_FLEXIBILITY} years")
    log.info(f"   JD has {len(must_have_skills)} must-have skills, {len(required_skills)} required skills")
    
    tier_batch_size = Config.TIER_BATCH_SIZE
    tiered_candidates = []
    tier_start = time.time()
    
    for i in range(0, len(candidates), tier_batch_size):
        batch = candidates[i:i + tier_batch_size]
        debug_first_batch = (i == 0)
        
        if debug_first_batch:
            log.info(f"🔍 Debug logging enabled for first batch of {len(batch)} candidates:")
        
        tasks = [
            compute_candidate_tier(
                c, required_skills, must_have_skills,
                jd_domain, jd_experience, jd_location, work_mode, jd_education,
                domain_embedding,
                debug_logging=debug_first_batch
            )
            for c in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                log.warning(f"⚠️ Tier computation failed for candidate: {result}")
                # Assign Tier 4 as fallback (not Tier 5, to give them a chance)
                batch[idx]["tier"] = 4
                batch[idx]["tier_name"] = "Review Needed"
                batch[idx]["tier_action"] = "Manual review required"
                batch[idx]["criteria"] = {"error": str(result)}
                tiered_candidates.append(batch[idx])
            else:
                tiered_candidates.append(result)
    
    tiered_candidates.sort(key=lambda c: c.get("tier", 5))
    
    tier_counts = {}
    for c in tiered_candidates:
        t = c.get("tier", 5)
        tier_counts[t] = tier_counts.get(t, 0) + 1
    
    tier_elapsed = time.time() - tier_start
    log.info(f"📊 Tier distribution: {tier_counts} ({tier_elapsed:.2f}s for {len(candidates)} candidates)")
    
    return tiered_candidates


def _candidate_matches_jd_domain(
    candidate_domains_str: str,
    jd_domains: List[str],
) -> bool:
    """Check if a candidate's stored domains overlap with the JD domains.

    candidate_domains_str is the comma-separated string already stored in
    Chroma metadata (e.g. "software_engineering,data_analytics").
    jd_domains is the list of domains from the parsed job description.

    Returns True when there is at least one overlapping domain, or when
    either side has no domain information (unclassified → include).
    """
    if not jd_domains:
        return True
    if not candidate_domains_str or not candidate_domains_str.strip():
        return True  # unclassified candidates are always included

    cand_set = {d.strip().lower() for d in candidate_domains_str.split(",") if d.strip()}
    jd_set = {d.strip().lower() for d in jd_domains if d and isinstance(d, str)}

    if not cand_set or not jd_set:
        return True

    return bool(cand_set & jd_set)


def fetch_top_resumes_from_chroma(
    jd_text: str,
    top_k: int = Config.TOP_K_CANDIDATES,
    where_clause: dict = None,
    similarity_threshold: float = Config.VECTOR_SIMILARITY_THRESHOLD,
    required_skills: List[str] = None,
    job_domain: Optional[str] = None,
    job_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch top matching resumes from Chroma vector database (OPTIMIZED).
    
    Two-pass retrieval strategy:
    1. If job_domains available, first try domain-filtered vector search for higher precision.
    2. If domain-filtered results are insufficient (< MIN_DOMAIN_RESULTS) or job_domains not
       available, fall back to full vector search.
    
    Filters by: (1) domain metadata (optional), (2) vector similarity threshold,
    (3) early skill overlap (if required_skills provided).
    
    Args:
        jd_text: Job description text
        top_k: Number of top candidates to retrieve
        where_clause: Optional metadata filter
        similarity_threshold: Minimum similarity to include (0-1)
        required_skills: JD required skills for early filtering
        job_domain: Optional job domain string
        job_domains: Optional list of job domains from JD parsing (for domain pre-filter)
        
    Returns:
        List of candidate dictionaries with metadata (similarity + skill-filtered)
        
    Raises:
        ChromaFetchError: If fetching from Chroma fails
    """
    MIN_DOMAIN_RESULTS = 20

    if not jd_text or not isinstance(jd_text, str):
        raise ChromaFetchError("Job description text must be a non-empty string")
    
    if top_k <= 0:
        raise ChromaFetchError(f"top_k must be positive, got {top_k}")
    
    try:
        log.info(f"🔍 Fetching top {top_k} resumes from Chroma with vector search...")
        
        effective_top_k = top_k

        # Vector search (single pass — domain filtering done in-memory below)
        results = None
        for attempt_k in [effective_top_k, 200, 100]:
            try:
                results = match_job_description(jd_text, top_k=attempt_k, where_clause=where_clause)
                break
            except Exception as quota_err:
                if "Quota exceeded" in str(quota_err) or "NumResults" in str(quota_err):
                    log.warning(f"⚠️ ChromaDB quota hit at top_k={attempt_k}, retrying with lower limit...")
                    continue
                raise
        
        if not results or "ids" not in results or not results["ids"] or not results["ids"][0]:
            log.warning("No results returned from Chroma")
            return []
        
        top_resumes = []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        ids = results["ids"][0] if results.get("ids") else []
        distances = results["distances"][0] if results.get("distances") else []
        skill_filter_skipped = 0
        domain_filter_skipped = 0
        no_metadata_skipped = 0
        apply_skill_filter = bool(required_skills and len(required_skills) > 0)
        threshold = Config.EARLY_SKILL_FILTER_THRESHOLD

        # Build the combined JD domain list for in-memory filtering
        jd_domain_list: List[str] = []
        if job_domains:
            jd_domain_list = [d.strip().lower() for d in job_domains if d and isinstance(d, str) and d.strip()]
        if not jd_domain_list and job_domain and str(job_domain).strip().lower() not in ("", "other", "general"):
            jd_domain_list = [str(job_domain).strip().lower()]
        apply_domain_filter = len(jd_domain_list) > 0

        if apply_domain_filter:
            log.info(f"🎯 Will apply in-memory domain filter: {jd_domain_list[:5]}")

        # Pre-filter by similarity threshold
        candidates_above_threshold = []
        for doc_id, meta, dist in zip(ids, metadatas, distances):
            similarity = 1.0 / (1.0 + dist)
            if similarity >= similarity_threshold:
                candidates_above_threshold.append((doc_id, meta, similarity))

        log.info(f"📊 Vector search returned {len(ids)} candidates, {len(candidates_above_threshold)} above similarity threshold")

        # Resolve resume from Chroma metadata first, then fall back to chat_sessions/get_resume
        doc_id_to_resume = {}
        for doc_id, meta, _ in candidates_above_threshold:
            parsed = parse_resume_from_metadata(meta)
            if parsed:
                doc_id_to_resume[doc_id] = parsed

        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

        need_fetch = [c for c in candidates_above_threshold if c[0] not in doc_id_to_resume]

        def fetch_single_resume(doc_id: str):
            try:
                parsed = get_resume(doc_id)
                return doc_id, parsed
            except Exception as e:
                log.warning(f"⚠️ Error fetching resume {doc_id}: {e}")
                return doc_id, None

        fetch_timeout = Config.RESUME_FETCH_TIMEOUT
        if need_fetch:
            with ThreadPoolExecutor(max_workers=Config.RESUME_FETCH_WORKERS) as executor:
                futures = {executor.submit(fetch_single_resume, c[0]): c[0] for c in need_fetch}
                for future in as_completed(futures, timeout=fetch_timeout * len(need_fetch) / Config.RESUME_FETCH_WORKERS + 30):
                    try:
                        doc_id, parsed = future.result(timeout=fetch_timeout)
                        doc_id_to_resume[doc_id] = parsed
                    except (TimeoutError, FuturesTimeoutError):
                        doc_id = futures[future]
                        log.warning(f"⚠️ Resume fetch timed out for {doc_id} ({fetch_timeout}s)")
                        doc_id_to_resume[doc_id] = None
                    except Exception as e:
                        doc_id = futures[future]
                        log.warning(f"⚠️ Resume fetch failed for {doc_id}: {e}")
                        doc_id_to_resume[doc_id] = None

        # Build results using resolved resumes (from metadata or fetch)
        for doc_id, meta, similarity in candidates_above_threshold:
            parsed_resume = doc_id_to_resume.get(doc_id)
            if not parsed_resume:
                no_metadata_skipped += 1
                continue

            # Domain filter: read candidate_domains from the parsed structured_resume_json
            if apply_domain_filter:
                resume_domains = parsed_resume.get("candidate_domains") or parsed_resume.get("candidateDomains") or []
                cand_domains_str = ",".join(str(d) for d in resume_domains if d) if isinstance(resume_domains, list) else str(resume_domains)
                if not _candidate_matches_jd_domain(cand_domains_str, jd_domain_list):
                    domain_filter_skipped += 1
                    continue

            # Early skill filter: only include candidates with at least threshold overlap
            if apply_skill_filter:
                candidate_skills = _extract_skills_from_resume(parsed_resume)
                req_lower = [r.strip().lower() for r in required_skills if r and isinstance(r, str)]
                matched_count = _count_matching_skills(req_lower, candidate_skills)
                overlap = matched_count / len(req_lower) if req_lower else 1.0
                if overlap < threshold:
                    skill_filter_skipped += 1
                    continue

            name, email, phone = extract_personal_info(parsed_resume)

            top_resumes.append({
                "candidate_id": doc_id,
                "candidate": {"structuredResume": parsed_resume},
                "name": name,
                "email": email,
                "phone": phone,
                "vector_similarity": round(similarity, 3),
                "_vector_similarity": similarity,
                "overall_score": round(similarity, 3),
                "skills_matched_local": [],
                "skills_unmatched_local": [],
                "rationale": "",
            })

        if no_metadata_skipped > 0:
            log.warning(f"⚠️ Skipped {no_metadata_skipped} candidates (no resume in metadata or chat_sessions)")

        # Domain filter fallback: if domain filtering excluded too many and we
        # have fewer than MIN_DOMAIN_RESULTS, re-run without domain filter
        if apply_domain_filter and domain_filter_skipped > 0 and len(top_resumes) < MIN_DOMAIN_RESULTS:
            log.warning(
                f"⚠️ Domain filter left only {len(top_resumes)} candidates (skipped {domain_filter_skipped}), "
                f"below minimum {MIN_DOMAIN_RESULTS}. Re-including cross-domain candidates."
            )
            top_resumes = []
            skill_filter_skipped = 0
            domain_filter_skipped = 0
            for doc_id, meta, similarity in candidates_above_threshold:
                parsed_resume = parse_resume_from_metadata(meta)
                if not parsed_resume:
                    continue
                if apply_skill_filter:
                    candidate_skills = _extract_skills_from_resume(parsed_resume)
                    req_lower = [r.strip().lower() for r in required_skills if r and isinstance(r, str)]
                    matched_count = _count_matching_skills(req_lower, candidate_skills)
                    overlap = matched_count / len(req_lower) if req_lower else 1.0
                    if overlap < threshold:
                        skill_filter_skipped += 1
                        continue
                name, email, phone = extract_personal_info(parsed_resume)
                top_resumes.append({
                    "candidate_id": doc_id,
                    "candidate": {"structuredResume": parsed_resume},
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "vector_similarity": round(similarity, 3),
                    "_vector_similarity": similarity,
                    "overall_score": round(similarity, 3),
                    "skills_matched_local": [],
                    "skills_unmatched_local": [],
                    "rationale": "",
                })
            apply_domain_filter = False  # mark that we fell back

        # Fallback: when early skill filter yields 0 candidates, retry with threshold=0
        used_relaxed_fallback = False
        if apply_skill_filter and skill_filter_skipped > 0 and len(top_resumes) == 0:
            log.warning(
                f"⚠️ Early skill filter excluded all {skill_filter_skipped} candidates; "
                "retrying with relaxed threshold (0%) so tier classification can evaluate"
            )
            used_relaxed_fallback = True
            skill_filter_skipped = 0
            for doc_id, meta, similarity in candidates_above_threshold:
                parsed_resume = doc_id_to_resume.get(doc_id)
                if not parsed_resume:
                    continue
                name, email, phone = extract_personal_info(parsed_resume)
                top_resumes.append({
                    "candidate_id": doc_id,
                    "candidate": {"structuredResume": parsed_resume},
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "vector_similarity": round(similarity, 3),
                    "_vector_similarity": similarity,
                    "overall_score": round(similarity, 3),
                    "skills_matched_local": [],
                    "skills_unmatched_local": [],
                    "rationale": "",
                })
        
        domain_note = f" [domain-filtered: {','.join(jd_domain_list[:3])}]" if apply_domain_filter and domain_filter_skipped > 0 else ""
        relaxed_note = " [relaxed skill filter fallback]" if used_relaxed_fallback else ""
        if domain_filter_skipped > 0:
            log.info(f"🎯 Domain filter: skipped {domain_filter_skipped} out-of-domain candidates")
        if apply_skill_filter and skill_filter_skipped > 0:
            log.info(
                f"✅ Filtered to {len(top_resumes)} candidates{domain_note}{relaxed_note} (similarity ≥{similarity_threshold:.0%}, "
                f"skill overlap ≥{threshold:.0%}); skipped {skill_filter_skipped} by early skill filter"
            )
        else:
            log.info(
                f"✅ Filtered to {len(top_resumes)} candidates{domain_note}{relaxed_note} above similarity threshold ({similarity_threshold})"
                + (f" and early skill filter ({threshold:.1%} skill overlap)" if apply_skill_filter else "")
            )
        return top_resumes
    
    except Exception as e:
        error_msg = f"Error fetching resumes from Chroma: {str(e)}"
        log.error(error_msg, exc_info=True)
        raise ChromaFetchError(error_msg)


def build_llm_prompt(jd_text: str, resume_text: str) -> str:
    """
    Build the prompt for LLM analysis with strict matching rules.
    """
    return f"""You are a strict JSON-only expert job-matching assistant.
Compare the Job Description and Candidate Resume.

**CRITICAL RULES:**
1. Only mark a skill as "matched" if it is EXPLICITLY mentioned in the resume
2. Do NOT infer or assume skills based on related technologies
3. A skill must have a DIRECT mention or exact synonym in the resume
4. If a skill is not explicitly present, it goes to "skills_unmatched"
5. Example: If "ReactJS" is required but only "React Native" is in resume, it's UNMATCHED

Job Description:
{jd_text}

Candidate Resume:
{resume_text}

Return ONLY a JSON object with these keys:

{{
    "skills_matched": [list of required skills EXPLICITLY found in resume],
    "skills_unmatched": [list of required skills NOT explicitly found],
    "rationale": "Provide a concise 2–3 sentence explanation including candidate strengths, weaknesses, and overall fit for the role."
}}

Example rationale: "This candidate is a poor match for the Junior Product Creator role. While he possesses strong communication and problem-solving skills, his background is heavily focused on IT management, infrastructure, and traditional project management. He lacks the required technical skills in JavaScript, React, Next.js, and Python, which are essential for a front-end development role."

Do NOT include any text outside the JSON.
Do NOT mark a skill as matched unless it's explicitly in the resume.
"""


async def invoke_llm_with_retry(
    prompt: str,
    max_retries: int = Config.MAX_RETRY_ATTEMPTS
) -> str:
    """
    Invoke LLM with retry logic.
    
    Args:
        prompt: Prompt to send to LLM
        max_retries: Maximum number of retry attempts
        
    Returns:
        LLM response text
        
    Raises:
        LLMInvocationError: If all retry attempts fail
    """
    for attempt in range(max_retries):
        try:
            log.debug(f"LLM invocation attempt {attempt + 1}/{max_retries}")
            response = await asyncio.wait_for(
                invoke_llm(prompt),
                timeout=Config.LLM_TIMEOUT_SECONDS
            )
            return response
        except asyncio.TimeoutError:
            log.warning(f"LLM timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(Config.RETRY_DELAY_SECONDS)
            continue
        except Exception as e:
            log.error(f"LLM invocation error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(Config.RETRY_DELAY_SECONDS)
            continue
    
    raise LLMInvocationError(f"Failed to invoke LLM after {max_retries} attempts")


def calculate_experience_match_score(
    candidate_years: float,
    required_years: float,
    max_years: Optional[float] = None
) -> float:
    """
    Calculate experience match score (0-1) with support for ranges and over-qualification handling.
    
    Args:
        candidate_years: Total years of experience from candidate's work history
        required_years: Minimum required years (from range or single value)
        max_years: Maximum preferred years (from range like "3-5 years"), optional
        
    Returns:
        Experience match score (0-1):
        - 1.0 if candidate is in ideal range (min <= candidate <= max) or meets single requirement
        - 0.0 if candidate has 0 years
        - Linear scale below minimum
        - Slight penalty if way above maximum (for ranges) or way over-qualified (2x+ requirement)
    """
    if required_years <= 0:
        return 1.0  # No requirement = full score
    
    if candidate_years <= 0:
        return 0.0  # No experience = 0 score
    
    # If range is specified (e.g., "3-5 years")
    if max_years and max_years > required_years:
        # Ideal range: candidate is within min-max
        if required_years <= candidate_years <= max_years:
            return 1.0
        
        # Below minimum: linear scale
        if candidate_years < required_years:
            return candidate_years / required_years
        
        # Above maximum: slight penalty for over-qualification
        # Formula: 1.0 - (excess / (max * 2))
        # Example: 10 years vs 3-5 range → excess=5, penalty=5/(5*2)=0.5, score=0.5
        # But we cap penalty at 30% and minimum score at 70%
        excess = candidate_years - max_years
        penalty = min(0.3, excess / (max_years * 2))  # Max 30% penalty
        return max(0.7, 1.0 - penalty)  # Minimum 70% score even if over-qualified
    
    # Single value requirement (no range)
    if candidate_years >= required_years:
        # Check if way over-qualified (more than 2x requirement)
        if candidate_years > required_years * 2:
            # Slight penalty for being way over-qualified
            # Example: 10 years vs 3 years required → excess_ratio = (10-3)/3 = 2.33
            # penalty = min(0.2, 2.33 * 0.1) = 0.2, score = 0.8
            excess_ratio = (candidate_years - required_years) / required_years
            penalty = min(0.2, excess_ratio * 0.1)  # Max 20% penalty
            return max(0.8, 1.0 - penalty)  # Minimum 80% score
        return 1.0  # Meets requirement, not over-qualified
    
    # Below requirement: linear scale
    return candidate_years / required_years


def compute_skill_match_score(
    matched_count: int,
    required_count: int,
    similarity: float = 0.0,
    experience_match: float = 1.0
) -> Tuple[float, float]:
    """
    Compute scores with skill match, vector similarity, and experience matching.
    
    PRIMARY SCORE: Skill match percentage (0-1)
    SECONDARY SCORES: Vector similarity (0-1) and Experience match (0-1)
    
    Args:
        matched_count: Number of matched skills
        required_count: Total number of required skills
        similarity: Vector similarity score (tiebreaker)
        experience_match: Experience match score (0-1), defaults to 1.0 if not provided
        
    Returns:
        Tuple of (skill_match_percentage, final_score)
        - skill_match_percentage: Exact match ratio (e.g., 4/8 = 0.50)
        - final_score: Weighted score for ranking (80% skill + 10% similarity + 10% experience)
    """
    # Calculate exact skill match percentage
    skill_match_percentage = matched_count / max(1, required_count)
    
    # Final score: Weighted combination
    # 80% skill match + 10% vector similarity + 10% experience match
    final_score = (
        (skill_match_percentage * Config.SKILL_MATCH_WEIGHT) + 
        (similarity * Config.SIMILARITY_WEIGHT) + 
        (experience_match * Config.EXPERIENCE_WEIGHT)
    )
    
    return round(skill_match_percentage, 4), round(final_score, 4)


def normalize_skill_match_from_required(
    required_skills: List[str],
    llm_skills_matched: List[str],
) -> Tuple[List[str], List[str], float]:
    """
    Unified skill match logic for job_matcher (and any consumers).
    Uses JD required_skills as source of truth and case-insensitive matching
    so all agents get the same skills_matched, skills_unmatched, and percentage.

    Args:
        required_skills: Required skills from the job description (JD).
        llm_skills_matched: Skills the LLM reported as matched (any casing).

    Returns:
        Tuple of (skills_matched, skills_unmatched, skill_match_percentage).
        - skills_matched: JD skills that have a match in llm_skills_matched (JD casing preserved).
        - skills_unmatched: JD skills that have no match (JD casing preserved).
        - skill_match_percentage: len(skills_matched) / len(required_skills) * 100, rounded to 2 decimals.
    """
    if not required_skills:
        return [], [], 0.0
    required = [s for s in required_skills if s is not None and str(s).strip()]
    if not required:
        return [], [], 0.0
    matched_lower = {str(s).strip().lower() for s in (llm_skills_matched or []) if s is not None and str(s).strip()}
    skills_matched = [s for s in required if (s or "").strip().lower() in matched_lower]
    skills_unmatched = [s for s in required if (s or "").strip().lower() not in matched_lower]
    pct = round(len(skills_matched) / len(required) * 100, 2)
    return skills_matched, skills_unmatched, pct


def sanitize_resume_for_llm(resume_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove personal identifiable information from resume before sending to LLM.
    
    Args:
        resume_data: Original resume dictionary
        
    Returns:
        Sanitized resume dictionary without PII
    """
    sanitized = resume_data.copy()
    
    # Remove personal information section
    if "personalInformation" in sanitized:
        personal_info = sanitized["personalInformation"].copy()
        # Remove PII fields
        personal_info.pop("name", None)
        personal_info.pop("email", None)
        personal_info.pop("phone", None)
        personal_info.pop("contactNumber", None)
        personal_info.pop("address", None)
        personal_info.pop("dateOfBirth", None)
        personal_info.pop("nationality", None)
        sanitized["personalInformation"] = personal_info
    
    # Remove top-level PII fields (alternative formats)
    sanitized.pop("Name", None)
    sanitized.pop("Email", None)
    sanitized.pop("Phone", None)
    sanitized.pop("Address", None)
    
    return sanitized


    """
    Validate and fix incomplete rationales that end abruptly.
    
    Args:
        rationale: The rationale text to validate
        
    Returns:
        Fixed rationale (or original if already complete)
    """
    if not rationale or len(rationale.strip()) < 20:
        return rationale
    
    rationale = rationale.strip()
    
    # Check for incomplete endings (ends mid-sentence)
    incomplete_patterns = [
        rationale.endswith('the candidate'),
        rationale.endswith('candidate'),
        rationale.endswith('they'),
        rationale.endswith('their'),
        rationale.endswith('while'),
        rationale.endswith('however'),
        rationale.endswith('although'),
        rationale.endswith(','),
        # Check if ends without proper punctuation
        not rationale.rstrip().endswith(('.', '!', '?')) and len(rationale) > 100
    ]
    
    is_incomplete = any(incomplete_patterns[:8]) or (not rationale.rstrip().endswith(('.', '!', '?')) and len(rationale.split('.')) > 1)
    
    if is_incomplete:
        # Check if we have at least one complete sentence
        sentences = rationale.split('.')
        if len(sentences) >= 2 and sentences[-1].strip() and len(sentences[-1].strip()) > 20:
            # Last "sentence" is too long and incomplete - likely cut off
            # Remove the incomplete last sentence if it's short
            if len(sentences[-1].strip()) < 50:
                # Remove incomplete last sentence
                rationale = '.'.join(sentences[:-1]).strip()
                if not rationale.endswith('.'):
                    rationale += '.'
                log.debug(f"⚠️ Removed incomplete last sentence from rationale")
            else:
                # Try to complete it
                if not rationale.rstrip().endswith('.'):
                    rationale = rationale.rstrip() + '.'
                    log.debug(f"⚠️ Added period to incomplete rationale")
        elif len(sentences) >= 2:
            # We have complete sentences, just add period if missing
            if not rationale.rstrip().endswith('.'):
                rationale = rationale.rstrip() + '.'
        else:
            # Only one sentence or no sentences - check if it's complete
            if rationale and not rationale.rstrip().endswith(('.', '!', '?')):
                # Check if it's a reasonable ending (ends with a word)
                last_word = rationale.split()[-1] if rationale.split() else ''
                if len(last_word) > 3:  # Likely a complete word
                    rationale = rationale.rstrip() + '.'
                    log.debug(f"⚠️ Added period to single-sentence rationale")
    
    return rationale


def generate_fallback_rationale_with_percentage(
    matched_skills: List[str],
    unmatched_skills: List[str],
    required_skills_count: int,
    match_score: float
) -> str:
    """
    Generate rationale using ACTUAL skill match percentage, not vector similarity.
    
    Args:
        matched_skills: List of matched skills
        unmatched_skills: List of unmatched skills
        required_skills_count: Total required skills
        match_score: Final ranking score (skill-weighted)
        
    Returns:
        Human-readable rationale with accurate percentage
    """
    matched_count = len(matched_skills)
    unmatched_count = len(unmatched_skills)
    
    # Calculate ACTUAL skill match percentage
    skill_percentage = int((matched_count / max(1, required_skills_count)) * 100)
    
    # Determine match quality based on SKILL PERCENTAGE (not overall score)
    if skill_percentage >= 70:
        quality = "strong"
        fit = "excellent fit"
    elif skill_percentage >= 50:
        quality = "moderate"
        fit = "reasonable fit"
    elif skill_percentage >= 30:
        quality = "fair"
        fit = "partial fit"
    else:
        quality = "weak"
        fit = "poor fit"
    
    # Build rationale
    parts = []
    
    # Opening with SKILL-BASED percentage
    parts.append(
        f"This candidate shows a {quality} match for the role "
        f"({skill_percentage}% skill overlap - {matched_count}/{required_skills_count} required skills)."
    )
    
    # Strengths
    if matched_skills:
        if matched_count <= 3:
            skills_str = ", ".join(matched_skills)
            parts.append(f"They possess required skills in {skills_str}.")
        else:
            top_skills = ", ".join(matched_skills[:3])
            remaining = matched_count - 3
            parts.append(
                f"They possess required skills including {top_skills} "
                f"and {remaining} others."
            )
    
    # Weaknesses
    if unmatched_skills:
        if unmatched_count <= 3:
            gaps_str = ", ".join(unmatched_skills)
            parts.append(
                f"However, they lack experience in {gaps_str}, "
                f"which are required for this position."
            )
        else:
            top_gaps = ", ".join(unmatched_skills[:3])
            remaining_gaps = unmatched_count - 3
            parts.append(
                f"However, they lack experience in {top_gaps} "
                f"and {remaining_gaps} other required skills."
            )
    
    # Overall assessment
    if skill_percentage >= 70:
        parts.append(
            f"Overall, this candidate represents an {fit} with strong "
            f"alignment to the role requirements."
        )
    elif skill_percentage >= 50:
        parts.append(
            f"Overall, this candidate represents a {fit}, though some skill "
            f"gaps would need to be addressed."
        )
    else:
        parts.append(
            f"Overall, this candidate represents a {fit} due to significant "
            f"skill gaps."
        )
    
    return " ".join(parts)


async def generate_llm_rationale(
    jd_text: str,
    sanitized_resume: Dict[str, Any],
    matched_skills: List[str],
    unmatched_skills: List[str],
    match_score: float = 0.0,
    required_skills_count: int = 0,
    recruiter_questions: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generate rationale using LLM with robust fallback mechanisms.
    
    OPTIMIZED: Uses circuit breaker to prevent cascading failures.
    
    ✅ ENHANCED: Now includes recruiter questions and candidate answers in rationale.
    
    Args:
        jd_text: Job description text
        sanitized_resume: Sanitized resume data
        matched_skills: List of matched skills
        unmatched_skills: List of unmatched skills
        match_score: Overall match score (0.0-1.0)
        required_skills_count: Total number of required skills
        recruiter_questions: Optional list of Q&A dicts with 'question' and 'answer' keys
    
    Returns:
        Rationale text (never empty - uses fallback if LLM fails)
    """
    # Check circuit breaker before proceeding
    can_proceed = await _llm_circuit_breaker.can_proceed()
    if not can_proceed:
        log.warning("🚫 Circuit breaker OPEN: Skipping LLM call, using fallback")
        return generate_fallback_rationale_with_percentage(
            matched_skills,
            unmatched_skills,
            required_skills_count or len(matched_skills) + len(unmatched_skills),
            match_score
        )
    
    resume_text = json.dumps(sanitized_resume, indent=2, ensure_ascii=False)
    
    # ✅ FIXED: Send full resume - no truncation for better analysis
    # Modern LLMs (Gemini) can handle large contexts, so we send complete data
    
    # ✅ NEW: Format recruiter questions and answers
    recruiter_qa_section = ""
    if recruiter_questions:
        qa_pairs = []
        for qa in recruiter_questions[:5]:  # Limit to top 5 Q&A pairs
            if isinstance(qa, dict):
                question = qa.get("question") or qa.get("Question", "")
                answer = qa.get("answer") or qa.get("Answer", "")
                if question and answer:
                    qa_pairs.append(f"Q: {question}\nA: {answer}")
        if qa_pairs:
            recruiter_qa_section = f"""

Application Questionnaire Responses:
{chr(10).join(qa_pairs)}

IMPORTANT: Consider these Q&A responses when generating the rationale. 
If the candidate's answers contradict information in their resume (e.g., resume shows 
experience with a technology but Q&A says "no" or "nothing"), you MUST mention this 
discrepancy as it raises concerns about accuracy or understanding of requirements.
"""
            log.debug(f"✅ Including {len(qa_pairs)} recruiter Q&A pairs in rationale")
    
    prompt = f"""You are an expert job-matching assistant.

Job Description:
{jd_text}  

Candidate Resume Summary:
{resume_text}

Skills Analysis (already computed):
- Matched Skills ({len(matched_skills)}): {', '.join(matched_skills) if matched_skills else 'None'}
- Missing Skills ({len(unmatched_skills)}): {', '.join(unmatched_skills) if unmatched_skills else 'None'}
{recruiter_qa_section}

Provide a concise 2-3 sentence rationale explaining:
1. The candidate's overall fit for the role
2. Key strengths that align with the position (use skills, experience, education, certifications, projects from the Candidate Resume Summary above)
3. Critical gaps or areas for improvement
4. Any discrepancies between resume and Q&A responses (if applicable)

CRITICAL: Use the full Candidate Resume Summary above (it includes skills, education, experience, certifications, projects). Do NOT state that the candidate has "no certifications" or "no relevant certifications" if the resume JSON contains a non-empty "certifications" array. Ensure your rationale is COMPLETE - it must end with proper punctuation (period, exclamation, or question mark). Do NOT cut off mid-sentence.

Return ONLY a JSON object:
{{
    "rationale": "Your COMPLETE 2-3 sentence professional explanation here. Ensure each sentence ends properly."
}}

Example: {{"rationale": "This candidate demonstrates strong technical alignment with 8 of 10 required skills including Python, AWS, and Docker. Their background in cloud architecture and DevOps practices directly matches the role requirements. However, they lack explicit experience with Kubernetes and Terraform, which are essential for this position."}}

Be specific and professional. Do NOT include text outside the JSON.
"""
    
    # Strategy 1: Try structured output with timeout (centralized invoke_structured_llm)
    try:
        class _Rationale(BaseModel):
            rationale: str

        log.debug("Attempting structured LLM rationale generation")
        structured: _Rationale = await asyncio.wait_for(
            invoke_structured_llm(
                prompt,
                _Rationale,
                task_type=TaskType.TEXT_GENERATION,
                preferred_model=_settings.GEMINI_MODEL,
                agent_name="ranker",
                max_output_tokens=2000,
                temperature=0.2,
                timeout=float(Config.RATIONALE_TIMEOUT),
                raise_on_fallback=False,
            ),
            timeout=Config.RATIONALE_TIMEOUT,
        )
        rationale = structured.model_dump().get("rationale", "").strip()
        
        if rationale:
            # Check if still too incomplete after fix attempt
            if rationale and (rationale.endswith('the candidate') or rationale.endswith('candidate')):
                log.warning(f"⚠️ Rationale still incomplete after fix attempt (ends with: '{rationale[-30:]}')")
                # If it's very incomplete, try raw LLM fallback
                if len(rationale.split('.')) < 2:
                    log.warning("⚠️ Rationale too incomplete, will try raw LLM fallback")
                    raise ValueError("Incomplete structured response")
        
        if rationale and len(rationale) > 20:
            log.debug(f"✅ Structured LLM rationale generated successfully (length: {len(rationale)})")
            await _llm_circuit_breaker.record_success()
            return rationale
        else:
            log.warning(f"⚠️ Structured LLM returned empty/short rationale: length={len(rationale) if rationale else 0}")
            raise ValueError("Empty structured response")
            
    except asyncio.TimeoutError:
        log.warning("⏱️ Structured LLM rationale timed out")
        await _llm_circuit_breaker.record_failure()
    except Exception as e:
        log.warning(f"⚠️ Structured LLM rationale failed: {str(e)[:100]}")
        await _llm_circuit_breaker.record_failure()
    
    # Strategy 2: Try raw LLM with JSON parsing
    try:
        log.debug("Attempting raw LLM rationale generation")
        raw_response = await asyncio.wait_for(
            invoke_llm(prompt),
            timeout=Config.RATIONALE_TIMEOUT
        )
        
        # Try to extract JSON
        if raw_response:
            try:
                response_data = json.loads(raw_response)
                rationale = response_data.get("rationale", "").strip()
                
                if rationale and len(rationale) > 20:
                    log.debug(f"✅ Raw LLM rationale extracted successfully (length: {len(rationale)})")
                    await _llm_circuit_breaker.record_success()
                    return rationale
                else:
                    log.warning(f"⚠️ JSON rationale too short: {len(rationale) if rationale else 0} chars")
            except json.JSONDecodeError as e:
                log.debug(f"⚠️ JSON decode error: {str(e)[:100]}, trying regex extraction")
                # Try improved regex extraction that handles escaped quotes and multi-line
                # Match: "rationale": "..." (handles escaped quotes and newlines)
                match = re.search(r'"rationale"\s*:\s*"((?:[^"\\]|\\.|\\n)*)"', raw_response, re.DOTALL)
                if match:
                    rationale = match.group(1).strip()
                    # Unescape common escape sequences
                    rationale = rationale.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')
                    
                    if len(rationale) > 20:
                        log.debug(f"✅ Raw LLM rationale extracted via regex (length: {len(rationale)})")
                        await _llm_circuit_breaker.record_success()
                        return rationale
                else:
                    # Try alternative: find JSON object and extract rationale value more flexibly
                    # Look for rationale field value even if JSON is malformed
                    alt_match = re.search(r'"rationale"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', raw_response, re.DOTALL)
                    if alt_match:
                        rationale = alt_match.group(1).strip()
                        rationale = rationale.replace('\\"', '"').replace('\\n', ' ').replace('\\t', ' ')
                        
                        if len(rationale) > 20:
                            log.debug(f"✅ Raw LLM rationale extracted via alternative regex (length: {len(rationale)})")
                            await _llm_circuit_breaker.record_success()
                            return rationale
            except Exception as e:
                log.warning(f"⚠️ JSON parsing exception: {str(e)[:100]}")
        
        log.warning("⚠️ Raw LLM response parsing failed - no valid rationale extracted")
    except asyncio.TimeoutError:
        log.warning("⏱️ Raw LLM rationale timed out")
        await _llm_circuit_breaker.record_failure()
    except Exception as e:
        log.warning(f"⚠️ Raw LLM rationale failed: {str(e)[:100]}")
        await _llm_circuit_breaker.record_failure()
    
    # Strategy 3: ALWAYS return fallback rationale
    log.info("📝 Generating fallback rationale (LLM unavailable)")
    return generate_fallback_rationale_with_percentage(
        matched_skills,
        unmatched_skills,
        required_skills_count or len(matched_skills) + len(unmatched_skills),
        match_score
    )


    """
    Remove temporary state keys that are no longer needed.
    
    Args:
        state: State dictionary
        
    Returns:
        Cleaned state dictionary
    """
    keys_to_remove = [
        "resume_url",
        "jd_text",
        "user_interests",
        "assessment_plan",
        "resume_text",
        "is_valid_resume",
        "structured_resume",
        "raw_skill_gap_analysis_output",
        "assessment_results",
        "generated_questions",
        "report",
        "body",           # ✅ ALWAYS remove body (contains job_details)
        "job_details",    # ✅ ALWAYS remove job_details (duplicate of job_description)
        "jd_url",         # ✅ Remove URL (not needed in output)
    ]
    
    # Preserve body if it contains job_details
    # body = state.get("body", {})
    # if not body.get("job_details"):
    #     keys_to_remove.append("body")
    
    for key in keys_to_remove:
        state.pop(key, None)
    log.info(f"✅ Cleaned up state: removed {len(keys_to_remove)} temporary fields")
    
    return state


def extract_base_uid(candidate_id: str) -> str:
    """
    Extract the base UID from a candidate_id that might include session suffix.
    
    Examples:
        - "EH9AJzQCRDbIXhEbYo6L8wnCNAx1" -> "EH9AJzQCRDbIXhEbYo6L8wnCNAx1"
        - "EH9AJzQCRDbIXhEbYo6L8wnCNAx1_session_123" -> "EH9AJzQCRDbIXhEbYo6L8wnCNAx1"
    
    Args:
        candidate_id: Full candidate ID from ChromaDB
        
    Returns:
        Base UID without session suffix
    """
    if not candidate_id:
        return candidate_id
    
    parts = candidate_id.split('_')
    
    if len(parts) == 1:
        return candidate_id
    
    # Check if last part looks like a session ID
    if parts[-1].isdigit() or (parts[-2] == 'session' and len(parts) > 2):
        if parts[-2] == 'session':
            return '_'.join(parts[:-2])
        else:
            if len(parts) > 2 and parts[-2].isdigit():
                return '_'.join(parts[:-1])
            return '_'.join(parts[:-1])
    
    return candidate_id


def deduplicate_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate candidates based on base UID.
    Keeps the candidate with the highest vector_similarity for each unique UID.
    
    Args:
        candidates: List of candidate dictionaries
        
    Returns:
        Deduplicated list of candidates with unique base UIDs
    """
    seen = {}
    uid_mapping = {}
    
    for candidate in candidates:
        full_id = candidate.get("candidate_id")
        if not full_id:
            continue
        
        base_uid = extract_base_uid(full_id)
        current_score = candidate.get("vector_similarity", 0)
        
        if base_uid not in seen:
            seen[base_uid] = candidate
            uid_mapping[base_uid] = [full_id]
        else:
            uid_mapping[base_uid].append(full_id)
            
            existing_score = seen[base_uid].get("vector_similarity", 0)
            if current_score > existing_score:
                log.info(
                    f"Duplicate UID detected: {base_uid} "
                    f"(from {full_id} with score {current_score:.4f} > "
                    f"{seen[base_uid]['candidate_id']} with score {existing_score:.4f})"
                )
                seen[base_uid] = candidate
            else:
                log.info(
                    f"Duplicate UID detected: {base_uid} "
                    f"(keeping {seen[base_uid]['candidate_id']} with score {existing_score:.4f} > "
                    f"{full_id} with score {current_score:.4f})"
                )
    
    deduplicated = list(seen.values())
    
    for candidate in deduplicated:
        full_id = candidate.get("candidate_id")
        base_uid = extract_base_uid(full_id)
        candidate["candidate_id"] = base_uid
        candidate["original_id"] = full_id
    
    if len(deduplicated) < len(candidates):
        duplicates_removed = len(candidates) - len(deduplicated)
        log.warning(
            f"⚠️ Deduplication: {len(candidates)} candidates -> "
            f"{len(deduplicated)} unique candidates "
            f"({duplicates_removed} duplicates removed)"
        )
        
        for uid, ids in uid_mapping.items():
            if len(ids) > 1:
                log.warning(f"   - UID {uid} had {len(ids)} entries: {ids}")
    
    return deduplicated


def format_ranked_output(
    candidates: List[Dict[str, Any]],
    required_skills_count: int
) -> List[Dict[str, Any]]:
    """
    Format candidates for final output with accurate skill match percentages and experience info.
    
    Args:
        candidates: Ranked and scored candidates
        required_skills_count: Total number of required skills
        
    Returns:
        List of formatted candidate dictionaries
    """
    ranked_output = []
    
    for candidate in candidates:
        matched_count = len(candidate.get("skills_matched_local", []))
        skill_percentage = (matched_count / max(1, required_skills_count)) * 100
        
        # Extract experience information
        candidate_years = candidate.get("candidate_experience_years", 0.0)
        required_years = candidate.get("required_experience_years", 0.0)
        max_years = candidate.get("max_experience_years")  # For ranges
        experience_match_score = candidate.get("experience_match_score", 1.0)
        
        # Format experience match status
        experience_match_status = ""
        if required_years > 0:
            if max_years and max_years > required_years:
                # Range specified (e.g., "3-5 years")
                if required_years <= candidate_years <= max_years:
                    experience_match_status = f"{candidate_years:.1f} years (within ideal range of {required_years:.1f}-{max_years:.1f} years)"
                elif candidate_years < required_years:
                    experience_match_status = f"{candidate_years:.1f} years (below minimum requirement of {required_years:.1f} years, range: {required_years:.1f}-{max_years:.1f})"
                else:
                    experience_match_status = f"{candidate_years:.1f} years (above preferred range of {required_years:.1f}-{max_years:.1f} years)"
            else:
                # Single value requirement
                if candidate_years >= required_years:
                    if candidate_years > required_years * 2:
                        experience_match_status = f"{candidate_years:.1f} years (exceeds requirement of {required_years:.1f} years - over-qualified)"
                    else:
                        experience_match_status = f"{candidate_years:.1f} years (meets requirement of {required_years:.1f} years)"
                else:
                    experience_match_status = f"{candidate_years:.1f} years (below requirement of {required_years:.1f} years)"
        elif candidate_years > 0:
            experience_match_status = f"{candidate_years:.1f} years (no requirement specified)"
        else:
            experience_match_status = "0 years (no experience listed)"
        
        overall = candidate.get("overall_score", 0.0)
        ranked_output.append({
            "rank": candidate.get("rank", 0),
            "candidate_id": candidate.get("candidate_id"),
            "name": candidate.get("name", "Unknown"),
            "email": candidate.get("email", ""),
            "phone": candidate.get("phone", ""),
            "match_score": overall,
            "match_band": _compute_match_band(overall),
            "score_explanation": candidate.get("score_explanation", ""),
            "skill_match_percentage": round(skill_percentage, 1),
            "skill_match_count": matched_count,
            "skills_matched": candidate.get("skills_matched_local", []),
            "skills_unmatched": candidate.get("skills_unmatched_local", []),
            "rationale": candidate.get("rationale", ""),
            "positive_rationale": candidate.get("positive_rationale", ""),
            "negative_rationale": candidate.get("negative_rationale", ""),
            "vector_similarity": candidate.get("vector_similarity", 0.0),
            "preferred_skills_matched": candidate.get("preferred_skills_matched", []),
            "current_role": candidate.get("current_role", ""),
            "education_summary": candidate.get("education_summary", ""),
            "current_location": candidate.get("current_location", ""),
            "education_match_status": candidate.get("education_match_status", ""),
            "must_have_skills_matched": candidate.get("must_have_skills_matched", []),
            "must_have_count": candidate.get("must_have_count", 0),
            "must_have_matched_count": candidate.get("must_have_matched_count", 0),
            "summary": candidate.get("summary", ""),
            "concerns": candidate.get("concerns", []) or [],
            "recruiter_qa_summary": candidate.get("recruiter_qa_summary", ""),
            # Experience information
            "candidate_experience_years": round(candidate_years, 1),
            "required_experience_years": round(required_years, 1) if required_years > 0 else None,
            "max_experience_years": round(max_years, 1) if max_years else None,  # For ranges
            "experience_match_status": experience_match_status,
            "experience_match_score": round(experience_match_score, 3)
        })
    
    return ranked_output


def _compute_education_match_status(jd_education: str, candidate_education_summary: str) -> str:
    """
    Compare JD education requirement to candidate's education summary.
    Returns: "meets", "above", "below", or "not_specified".
    """
    if not jd_education or str(jd_education).strip().lower() in ("not specified", "n/a", ""):
        return "not_specified"
    if not candidate_education_summary or str(candidate_education_summary).strip().lower() in ("n/a", ""):
        return "not_specified"
    jd_lower = jd_education.lower().strip()
    cand_lower = candidate_education_summary.lower().strip()
    # Simple heuristic: degree levels
    levels = ["phd", "doctorate", "mba", "m.", "ms ", "m.sc", "mtech", "m.tech", "btech", "b.tech", "b.e", "b.sc", "b.com", "b.a", "bachelor", "master", "masters"]
    jd_level = next((i for i, l in enumerate(levels) if l in jd_lower), len(levels))
    cand_level = next((i for i, l in enumerate(levels) if l in cand_lower), len(levels))
    if cand_level < jd_level:
        return "above"  # candidate has higher degree
    if cand_level > jd_level:
        return "below"
    return "meets"


def _compute_match_band(match_score: float) -> str:
    """
    Map match_score to a recruiter-friendly band for filters/badges.
    """
    if not isinstance(match_score, (int, float)):
        return "unknown"
    s = float(match_score)
    if s >= 0.9:
        return "strong_match"
    if s >= 0.7:
        return "good_match"
    if s >= 0.5:
        return "moderate_match"
    if s >= 0.3:
        return "weak_match"
    return "poor_match"


def _align_score_explanation_to_final_score(
    score_explanation: str, final_score: float, was_scaled: bool
) -> str:
    """
    When the match score was scaled down post-LLM, update score_explanation so the
    percentage mentioned matches the final displayed score (avoids mismatch e.g. "78%" vs 48.8%).
    """
    if not score_explanation or not was_scaled:
        return score_explanation
    final_pct = int(round(final_score * 100))
    # Replace first percentage pattern (e.g. "This 78%" or "78%" or "75% reflects")
    match = re.search(r"\b(\d{1,3})%", score_explanation)
    if match and int(match.group(1)) != final_pct:
        return re.sub(r"\b\d{1,3}%", f"{final_pct}%", score_explanation, count=1)
    return score_explanation


async def analyze_candidate_batch_with_llm(
    jd_title: str,
    jd_skills: List[str],  # Combined: must_have + parsed
    jd_must_have_skills: List[str],  # ✅ Priority skills from payload
    jd_preferred_skills: List[str],  # ✅ NEW
    jd_experience: str,
    jd_responsibilities: str,
    jd_location: str,
    jd_education: str,
    jd_work_mode: str,  # ✅ NEW
    candidates_batch: List[Dict[str, Any]],
    batch_idx: int = 0,
    recruiter_questions: Optional[List[Dict[str, Any]]] = None,
    job_domain: Optional[str] = None,  # ✅ Job domain for contextual analysis
) -> List[Dict[str, Any]]:
    """
    Analyze a batch of candidates using LLM (REDESIGNED APPROACH).
    
    Returns list of dicts with:
    - candidate_id
    - skills_matched: List[str]
    - match_score: float (0.0-1.0)
    - rationale: str
    
    Args:
        jd_title: Job title
        jd_skills: List of required skills (combined: must_have + parsed)
        jd_must_have_skills: List of priority must-have skills from payload
        jd_preferred_skills: List of preferred/good-to-have skills
        jd_experience: Experience requirement
        jd_responsibilities: Key responsibilities
        jd_location: Job location
        jd_education: Education requirement
        jd_work_mode: Work mode (remote, hybrid, etc.)
        candidates_batch: List of candidate dicts with resume_data
        batch_idx: Batch index for logging
        
    Returns:
        List of analyzed candidates with skills_matched, match_score, rationale
    """
    
    # Pre-validate skills for each candidate BEFORE LLM (skill dropping first, then rationale)
    pre_validated_by_candidate: Dict[str, List[str]] = {}
    for candidate in candidates_batch:
        resume = candidate.get("candidate", {}).get("structuredResume", {})
        cand_skills_set = extract_primary_skills(resume) if resume else set()
        validated = validate_skills_against_candidate(jd_skills, cand_skills_set, jd_skills)
        pre_validated_by_candidate[candidate["candidate_id"]] = validated

    # Format candidates for LLM
    candidates_text = ""
    for i, candidate in enumerate(candidates_batch, start=1):
        # ✅ FIXED: Access resume data from correct ChromaDB structure
        resume = candidate.get("candidate", {}).get("structuredResume", {})
        pre_validated = pre_validated_by_candidate.get(candidate["candidate_id"], [])

        # Extract key resume info
        candidate_skills = []
        skills_raw = resume.get("Skills") or resume.get("skills", [])
        if isinstance(skills_raw, list):
            # ✅ FIXED: Send ALL skills to LLM for comprehensive analysis
            for skill in skills_raw:  # All skills, no truncation
                if isinstance(skill, str):
                    candidate_skills.append(skill)
                elif isinstance(skill, dict):
                    # Extract skill name from dict (actual format uses "SkillName")
                    skill_name = skill.get("SkillName") or skill.get("skillName") or skill.get("name", "")
                    if skill_name:
                        candidate_skills.append(skill_name)
        
        # Name is stored as an array in actual data format
        name_raw = resume.get("Name", [])
        candidate_name = name_raw[0] if isinstance(name_raw, list) and len(name_raw) > 0 else "N/A"
        
        # Extract experience info
        total_exp = resume.get("total_experience_years", 0)
        if total_exp == 0:
            # Calculate from experience array if not set
            exp_list = resume.get("experience", [])
            total_exp = f"{len(exp_list)} roles" if exp_list else "N/A"
        
        # Get recent role from experience array
        exp_list = resume.get("experience", [])
        recent_role = exp_list[0].get("job_title", "N/A") if exp_list else "N/A"
        
        # Get education (full details)
        edu_list = resume.get("education", []) or resume.get("Education", [])
        if edu_list and isinstance(edu_list, list) and len(edu_list) > 0:
            edu_item = edu_list[0]
            if isinstance(edu_item, dict):
                degree = edu_item.get("degree") or edu_item.get("Degree", "")
                major = edu_item.get("major") or edu_item.get("Major") or edu_item.get("field", "")
                education = f"{degree} {major}".strip() if degree or major else "N/A"
            else:
                education = str(edu_item) if edu_item else "N/A"
        else:
            education = "N/A"

        # Location (best-effort)
        candidate_location = (
            resume.get("location")
            or resume.get("Location")
            or resume.get("current_location")
            or resume.get("CurrentLocation")
        )
        if not candidate_location:
            personal_info = resume.get("personalInformation") or resume.get("personal_info") or {}
            if isinstance(personal_info, dict):
                candidate_location = (
                    personal_info.get("location")
                    or personal_info.get("city")
                    or personal_info.get("state")
                    or personal_info.get("country")
                )
        candidate_location = candidate_location or "N/A"

        # Extract certifications (support certification_name, name)
        certs_raw = resume.get("certifications") or resume.get("Certifications") or []
        cert_names = []
        if isinstance(certs_raw, list):
            for c in certs_raw:
                if isinstance(c, dict):
                    name = c.get("certification_name") or c.get("name") or c.get("certification") or ""
                    if name and isinstance(name, str):
                        cert_names.append(name.strip())
                elif c:
                    cert_names.append(str(c).strip())
        certifications_str = ", ".join(cert_names) if cert_names else "None listed"

        # Extract projects (brief summary for context)
        proj_raw = resume.get("projects") or resume.get("Projects") or []
        proj_summaries = []
        if isinstance(proj_raw, list):
            for p in proj_raw[:5]:  # Top 5 projects
                if isinstance(p, dict):
                    title = p.get("title") or p.get("Title") or p.get("name") or ""
                    desc = (p.get("description") or p.get("Description") or "")[:200]
                    if title:
                        proj_summaries.append(f"{title}" + (f": {desc}..." if desc else ""))
                elif p:
                    proj_summaries.append(str(p)[:150])
        projects_str = "; ".join(proj_summaries) if proj_summaries else "None listed"
        
        verified_skills_line = f"VERIFIED skills_matched (use ONLY these): {', '.join(pre_validated) if pre_validated else 'None'}"
        candidates_text += f"""
--- CANDIDATE {i} ---
ID: {candidate['candidate_id']}
Name: {candidate_name}
{verified_skills_line}
Skills: {', '.join(candidate_skills) if candidate_skills else 'Not listed'}
Experience: {total_exp} years
Recent Role: {recent_role}
Education: {education}
Certifications: {certifications_str}
Projects: {projects_str}
Location: {candidate_location}
"""
    
    # ✅ Build skills section with priority indication
    if jd_must_have_skills:
        # Has priority must_have_skills from payload
        must_have_text = f"Must-Have Skills (PRIORITY - {len(jd_must_have_skills)}): {', '.join(jd_must_have_skills)}"
        other_skills = [s for s in jd_skills if s not in jd_must_have_skills]
        if other_skills:
            other_skills_text = f"\nAdditional Required Skills (from JD parsing - {len(other_skills)}): {', '.join(other_skills)}"
        else:
            other_skills_text = ""
        skills_section = must_have_text + other_skills_text
    else:
        # No must_have_skills, use all parsed skills
        skills_section = f"Required Skills ({len(jd_skills)}): {', '.join(jd_skills)}"
    
    recruiter_qa_section = ""
    if recruiter_questions:
        qa_pairs = []
        for qa in recruiter_questions[:5]:
            if isinstance(qa, dict):
                q = qa.get("question") or qa.get("Question", "")
                a = qa.get("answer") or qa.get("Answer", "")
                if q and a:
                    qa_pairs.append(f"Q: {q}\nA: {a}")
        if qa_pairs:
            recruiter_qa_section = f"""
11. **recruiter_qa_summary**: When considering the Application Questionnaire below, 1-2 sentences on how well the candidate's answers align with the role and any discrepancies with the resume. If not applicable, return "".
Application Questionnaire Responses:
{chr(10).join(qa_pairs)}
"""
    
    job_domain_section = f"""
**JOB DOMAIN**: {job_domain or "general"}
Use this domain for contextual analysis. Compare each candidate's career domain/role with this job domain.
""" if job_domain else ""

    # Static system instruction for Gemini context caching (~1,700 tokens cached per session)
    # JD details + static prompts are same for all batches; prompt = candidates only
    system_instruction = f"""You are an expert job matching AI. Perform CONTEXTUAL analysis (not generic) for each candidate.
Use the SAME matching logic as job_matcher: ordered analysis steps and 0-100 scoring scale.
{job_domain_section}

JOB DETAILS:
Title: {jd_title}
Location: {jd_location}
Work Mode: {jd_work_mode}
{skills_section}
Preferred Skills ({len(jd_preferred_skills)}): {', '.join(jd_preferred_skills) if jd_preferred_skills else 'None'}
Experience Required: {jd_experience}
Education Required: {jd_education}
Full Job Description & Responsibilities: {jd_responsibilities}

CRITICAL - Skill validation happens BEFORE your analysis: Each candidate has "VERIFIED skills_matched" (deterministic match from resume). You MUST use ONLY those skills for skills_matched. Do NOT add any skills not in the verified list. Generate match_score and rationale based on the verified skills.

For EACH candidate, following the ordered steps above, analyze and return:
{SKILLS_MATCH_INSTRUCTIONS}
3. **match_score**: Overall match score from 0-100 (integer). Use ORDERED CONTEXTUAL ANALYSIS (same as job_matcher):
{ORDERED_ANALYSIS_STEPS_UNIFIED}
{SCORING_GUIDELINES_0_100}
4. **score_explanation**: 1-2 sentences that explicitly tie the match_score to the fit (e.g. "This 75% reflects strong AI/ML and education alignment, reduced by experience below 4y and missing TypeScript/React stack."). Must reference the score or percentage.
5. **domain_relevant_experience_years**: Estimated years of experience RELEVANT to this job's domain/role (e.g. TypeScript, AI/ML, RAG). Infer from work history and role titles; if unclear use total experience. Return a number (e.g. 4.5) or null.
6. **rationale**: Overall fit as 2-4 bullet points. Format: each point on its own line starting with "- " (e.g. "- Strong in Python and AWS.\\n- B.Tech in CS.\\n- Below required experience.").
7. **positive_rationale**: Why the score is as high as it is, as 2-4 bullet points. Format: each line "- point" (strengths, skills matched, education). Be specific.
8. **negative_rationale**: Why it's not higher, as 2-4 bullet points. Format: each line "- point" (gaps, missing skills, shortfalls). Be specific.
9. **summary**: 1-2 sentences (e.g. "5y backend, Python/AWS, fintech") - a one-line pitch for the candidate.
10. **concerns**: Optional list of red flags or concerns, e.g. ["No Python in last 3 years", "Employment gap 2022"]. Return empty list [] if none.
{recruiter_qa_section}

{CRITICAL_RULES}
- Rationale fields must be bullet points (each line starting with "- ")
- CRITICAL JSON: In all string fields (rationale, score_explanation, summary, etc.), escape any double quotes inside the string with a backslash (e.g. "Python" → \\"Python\\"). Never output unescaped quotes that would break JSON parsing.
"""

    # Prompt = dynamic content only (candidates) - enables Gemini to cache system_instruction
    prompt = f"""CANDIDATES (each has Skills, Experience, Recent Role, Education, Certifications, Projects):
{candidates_text}

Return ONLY a JSON array (no markdown, no extra text):
[
  {{
    "candidate_id": "exact_id_from_above",
    "skills_matched": ["skill1", "skill2", ...],
    "preferred_skills_matched": ["skill3", ...],
    "match_score": 75,
    "score_explanation": "This 75% reflects strong X and Y, reduced by Z.",
    "domain_relevant_experience_years": 4.5,
    "rationale": "- Point one.\\n- Point two.\\n- Point three.",
    "positive_rationale": "- Strength one.\\n- Strength two.",
    "negative_rationale": "- Gap one.\\n- Gap two.",
    "summary": "1-2 sentence one-line pitch e.g. 5y backend, Python/AWS, fintech",
    "concerns": ["red flag 1", "red flag 2"] or [],
    "recruiter_qa_summary": "Alignment with questionnaire or empty string"
  }},
  ...
]
"""
    
    try:
        # Use invoke_llm with response_mime_type="application/json" (same as job_matcher)
        # so Gemini returns valid JSON instead of plain text with missing commas.
        tokens_per_candidate = 2000
        estimated_tokens = max(8000, len(candidates_batch) * tokens_per_candidate)

        max_parse_attempts = 3
        results = None
        for attempt in range(max_parse_attempts):
            try:
                content = await asyncio.wait_for(
                    invoke_llm(
                        prompt=prompt,
                        task_type="classification",
                        agent_name="ranker",
                        max_output_tokens=estimated_tokens,
                        response_mime_type="application/json",
                        system_instruction=system_instruction,
                    ),
                    timeout=60.0  # 60 second timeout per batch
                )
            except asyncio.TimeoutError:
                log.error(f"❌ Batch {batch_idx + 1}: LLM call timed out (attempt {attempt + 1}/{max_parse_attempts})")
                if attempt < max_parse_attempts - 1:
                    continue
                return []

            content = (content or "").strip()

            # Extract JSON from response (handle markdown and other formatting)
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            
            # Try to find JSON array (with or without closing bracket for truncated output)
            json_match = re.search(r'\[[\s\S]*\]', content)
            if not json_match:
                # LLM output may have been truncated before closing ']'
                json_match = re.search(r'\[[\s\S]+', content)
                if json_match:
                    log.warning(f"⚠️ Batch {batch_idx + 1}: LLM output appears truncated (no closing ']'), attempting partial recovery")

            if json_match:
                json_str = json_match.group(0)
                results = _try_parse_llm_ranking_array(json_str)
                if results is not None:
                    break  # Success
                if attempt < max_parse_attempts - 1:
                    log.warning(f"⚠️ Batch {batch_idx + 1}: JSON parsing failed (attempt {attempt + 1}), retrying")
                else:
                    log.error(f"❌ Batch {batch_idx + 1}: JSON parsing failed after {max_parse_attempts} attempts (len={len(json_str)})")
                    log.debug(f"JSON start: {json_str[:500]}")
                    log.debug(f"JSON end:   ...{json_str[-500:]}")
                    return []
            else:
                if attempt < max_parse_attempts - 1:
                    log.warning(f"⚠️ Batch {batch_idx + 1}: No JSON found in LLM response (attempt {attempt + 1}), retrying")
                else:
                    log.error(f"❌ Batch {batch_idx + 1}: No JSON found in LLM response after {max_parse_attempts} attempts")
                    return []
        
        if results is None:
            return []
        
        # Validate and enrich results
        enriched_results = []
        for result in results:
            # Add vector similarity from original candidate data
            candidate_id = result.get("candidate_id")
            original_candidate = next(
                (c for c in candidates_batch if c["candidate_id"] == candidate_id),
                None
            )
            if original_candidate:
                result["vector_similarity"] = original_candidate.get("vector_similarity", 0.0)
                result["candidate"] = original_candidate.get("candidate", {})
                # Copy pre-extracted personal info for quick access
                result["name"] = original_candidate.get("name", "Unknown")
                result["email"] = original_candidate.get("email", "")
                result["phone"] = original_candidate.get("phone", "")
                
                # Override skills_matched with pre-validated list (skill dropping done before LLM)
                result["skills_matched"] = pre_validated_by_candidate.get(candidate_id, result.get("skills_matched") or [])
                if not isinstance(result["skills_matched"], list):
                    result["skills_matched"] = []
                
                # Ensure preferred_skills_matched is a list
                if not isinstance(result.get("preferred_skills_matched"), list):
                    result["preferred_skills_matched"] = []
                
                # Ensure match_score is valid; convert 0-100 to 0.0-1.0 (unified with job_matcher)
                if not isinstance(result.get("match_score"), (int, float)):
                    result["match_score"] = 0.0
                else:
                    result["match_score"] = normalize_match_score_0_100_to_0_1(float(result["match_score"]))
                
                # Ensure rationale exists
                if not result.get("rationale"):
                    result["rationale"] = "Analysis unavailable."
                
                # Positive/negative rationale, summary, concerns, recruiter_qa_summary
                if not result.get("positive_rationale"):
                    result["positive_rationale"] = ""
                if not result.get("negative_rationale"):
                    result["negative_rationale"] = ""
                if not result.get("summary"):
                    result["summary"] = ""
                if not isinstance(result.get("concerns"), list):
                    result["concerns"] = []
                if "recruiter_qa_summary" not in result:
                    result["recruiter_qa_summary"] = ""
                if "score_explanation" not in result or not result.get("score_explanation"):
                    result["score_explanation"] = ""
                
                # Normalize domain_relevant_experience_years (optional LLM output)
                dr_years = result.get("domain_relevant_experience_years")
                if dr_years is None:
                    pass  # keep None, will use total_experience_years
                elif isinstance(dr_years, (int, float)) and dr_years >= 0:
                    result["domain_relevant_experience_years"] = float(dr_years)
                else:
                    try:
                        result["domain_relevant_experience_years"] = float(dr_years) if dr_years is not None else None
                    except (ValueError, TypeError):
                        result["domain_relevant_experience_years"] = None
                
                # Current/recent role, education_summary, current_location from resume
                resume = result.get("candidate", {}).get("structuredResume", {})
                exp_list = resume.get("experience", []) or resume.get("Experience", [])
                if exp_list and isinstance(exp_list[0], dict):
                    first_exp = exp_list[0]
                    job_title = first_exp.get("job_title") or first_exp.get("jobTitle") or first_exp.get("title", "N/A")
                    company = first_exp.get("company") or first_exp.get("company_name") or first_exp.get("Company", "")
                    result["current_role"] = f"{job_title} at {company}".strip().rstrip(" at") if company else (job_title or "N/A")
                else:
                    result["current_role"] = "N/A"
                edu_list = resume.get("education", []) or resume.get("Education", [])
                if edu_list and isinstance(edu_list[0], dict):
                    edu_item = edu_list[0]
                    degree = edu_item.get("degree") or edu_item.get("Degree", "")
                    major = edu_item.get("major") or edu_item.get("Major") or edu_item.get("field", "")
                    result["education_summary"] = f"{degree} {major}".strip() if degree or major else "N/A"
                else:
                    result["education_summary"] = "N/A"
                loc = (
                    resume.get("location") or resume.get("Location")
                    or resume.get("current_location") or resume.get("CurrentLocation")
                )
                if not loc:
                    personal_info = resume.get("personalInformation") or resume.get("personal_info") or {}
                    if isinstance(personal_info, dict):
                        loc = (
                            personal_info.get("location") or personal_info.get("city")
                            or personal_info.get("state") or personal_info.get("country")
                        )
                result["current_location"] = loc or "N/A"
                
                enriched_results.append(result)
        
        log.info(f"✅ Batch {batch_idx + 1}: Analyzed {len(enriched_results)}/{len(candidates_batch)} candidates")
        return enriched_results

    except Exception as e:
        log.error(f"❌ Batch {batch_idx + 1}: LLM analysis failed: {e}")
        return []


def compute_tier_from_llm_results(
    candidate: Dict[str, Any],
    jd_skills: List[str],
    jd_must_have_skills: List[str],
    jd_experience: str,
    jd_location: str,
    jd_education: str,
    work_mode: str = ""
) -> Dict[str, Any]:
    """
    Compute tier classification from LLM analysis results.
    
    Uses LLM's skills_matched, match_score, and domain_relevant_experience_years
    to determine the tier, rather than re-evaluating from raw resume data.
    
    Tier System (Location de-prioritized - used as tiebreaker only):
    - Tier 1 (Unicorn): A ∧ B ∧ D - Skills + Experience + Education
    - Tier 2 (Strong Match): A ∧ B - Skills + Experience
    - Tier 3 (High Potential): A ∧ (B ∨ D) - Skills + either Experience or Education
    - Tier 4 (Pivoter): B ∧ D - Experience + Education, lacking Skills
    - Tier 5 (No Match): Does not fit above tiers
    
    Location (C) is evaluated and stored but only used as a minor tiebreaker
    within each tier (candidates with matching location rank slightly higher).
    
    Args:
        candidate: Candidate dict with LLM analysis results
        jd_skills: Required skills from JD
        jd_must_have_skills: Must-have skills from JD
        jd_experience: Experience requirement string
        jd_location: JD location
        jd_education: Education requirement
        work_mode: Work mode (remote/hybrid/onsite)
        
    Returns:
        Dict with tier, tier_name, tier_action, and criteria details
    """
    import re
    
    # ========== Criterion A: Skills Match ==========
    skills_matched = candidate.get("skills_matched", [])
    match_score = candidate.get("match_score", 0.0)
    
    # If must-have skills specified, check coverage
    if jd_must_have_skills:
        must_have_matched = [s for s in jd_must_have_skills if s.lower() in [m.lower() for m in skills_matched]]
        must_have_pct = (len(must_have_matched) / len(jd_must_have_skills)) * 100 if jd_must_have_skills else 100
    else:
        must_have_matched = []
        must_have_pct = 100  # No must-haves = pass
    
    # Overall skill coverage
    if jd_skills:
        skill_pct = (len(skills_matched) / len(jd_skills)) * 100
    else:
        skill_pct = 100
    
    # Skills match: Either ≥30% must-have OR ≥20% overall OR match_score ≥ 0.5
    skills_match = (
        must_have_pct >= 30 or  # At least 30% of must-haves
        skill_pct >= 20 or  # At least 20% of required skills
        match_score >= 0.5  # LLM gave decent score
    )
    
    # ========== Criterion B: Experience Match ==========
    domain_years = candidate.get("domain_relevant_experience_years")
    
    # Get total experience from resume as fallback
    resume = candidate.get("candidate", {}).get("structuredResume", {})
    total_exp = resume.get("total_experience_years") or resume.get("TotalExperienceYears") or 0
    if isinstance(total_exp, str):
        try:
            total_exp = float(re.search(r'[\d.]+', total_exp).group()) if re.search(r'[\d.]+', total_exp) else 0
        except:
            total_exp = 0
    
    # Use domain experience if available, else total
    candidate_years = float(domain_years) if isinstance(domain_years, (int, float)) and domain_years >= 0 else float(total_exp or 0)
    
    # Parse required experience
    min_years = 0
    max_years = 99
    if jd_experience and jd_experience.lower() not in ("not specified", "n/a", ""):
        exp_match = re.search(r'(\d+)(?:\s*[-–]\s*(\d+))?', jd_experience)
        if exp_match:
            min_years = float(exp_match.group(1))
            if exp_match.group(2):
                max_years = float(exp_match.group(2))
            else:
                max_years = min_years + 10  # Assume range
    
    # Experience match: scaled flexibility (at most 50% below min, generous above max)
    lower_flex = max(1, min_years * 0.5)  # e.g. 4yr req → allow 2yr+, 8yr req → allow 4yr+
    upper_flex = 5
    floor_years = max(0, min_years - lower_flex)

    if min_years == 0:
        experience_match = True  # No requirement specified
    elif min_years <= 2 and candidate_years > 0:
        experience_match = True  # Junior role, any experience counts
    else:
        experience_match = (
            candidate_years >= floor_years and candidate_years <= (max_years + upper_flex)
        )
    
    # ========== Criterion C: Location Match ==========
    candidate_location = candidate.get("current_location", "") or ""
    
    # Remote/hybrid = always pass
    if work_mode and work_mode.lower() in ("remote", "hybrid", "work from home", "wfh"):
        location_match = True
        location_reason = f"Remote/hybrid work mode: {work_mode}"
    elif not jd_location or jd_location.lower() in ("n/a", "not specified", "anywhere", "remote", ""):
        location_match = True
        location_reason = "No location requirement"
    elif not candidate_location or candidate_location.lower() in ("n/a", "not specified", "unknown", ""):
        location_match = True  # Be lenient
        location_reason = "Candidate location unknown (lenient)"
    else:
        # Simple location matching
        jd_loc_lower = jd_location.lower()
        cand_loc_lower = candidate_location.lower()
        
        # Check for common elements (city, state, country)
        jd_parts = set(re.split(r'[,\s/]+', jd_loc_lower))
        cand_parts = set(re.split(r'[,\s/]+', cand_loc_lower))
        
        location_match = bool(jd_parts & cand_parts)  # Any overlap
        location_reason = f"{candidate_location} vs {jd_location}"
    
    # ========== Criterion D: Education Match ==========
    education_summary = candidate.get("education_summary", "") or ""
    
    if not jd_education or jd_education.lower() in ("n/a", "not specified", "any", "", "preferred"):
        education_match = True
        education_reason = "No strict education requirement"
    elif not education_summary or education_summary == "N/A":
        education_match = False
        education_reason = f"Education not listed (JD requires: {jd_education})"
    else:
        education_levels = {
            "phd": 5, "doctorate": 5, "ph.d": 5,
            "master": 4, "mba": 4, "ms": 4, "msc": 4, "mtech": 4,
            "bachelor": 3, "bs": 3, "ba": 3, "bsc": 3, "btech": 3, "be": 3, "bcom": 3, "degree": 3,
            "associate": 2, "diploma": 2,
            "high school": 1, "hsc": 1
        }
        
        req_level = 0
        for edu, level in education_levels.items():
            if edu in jd_education.lower():
                req_level = max(req_level, level)
        
        cand_level = 0
        for edu, level in education_levels.items():
            if edu in education_summary.lower():
                cand_level = max(cand_level, level)
        
        if cand_level >= req_level:
            education_match = True
        elif req_level > 0 and cand_level >= req_level - 1 and cand_level >= 2:
            education_match = True  # One level below is acceptable (e.g., diploma for bachelor)
        else:
            education_match = False
        education_reason = f"{education_summary} vs required: {jd_education}"
    
    # ========== Tier Classification (Location NOT a gate, only a tiebreaker) ==========
    A, B, C, D = skills_match, experience_match, location_match, education_match
    
    if A and B and D:
        tier = 1
        tier_name = "Unicorn"
        tier_action = "Immediate interview"
    elif A and B:
        tier = 2
        tier_name = "Strong Match"
        tier_action = "High priority - verify education fit"
    elif A and (B or D):
        tier = 3
        tier_name = "High Potential"
        tier_action = "Good for junior/associate version of role"
    elif B and D:
        tier = 4
        tier_name = "Pivoter"
        tier_action = "Consider if skills are trainable"
    else:
        tier = 5
        tier_name = "No Match"
        tier_action = "Review manually or skip"
    
    # Option A: Score-based tier adjustment so tier and match_score align for recruiters
    score_val = float(candidate.get("match_score", 0.0))
    if tier == 1 and score_val < 0.5:
        tier = 2
        tier_name = "Strong Match"
        tier_action = "High priority - verify education fit"
    elif tier == 5 and score_val >= 0.75:
        tier = 3
        tier_name = "High Potential"
        tier_action = "Good for junior/associate version of role"
    
    # Location bonus: small score boost for location match (tiebreaker within tier)
    location_bonus = 0.02 if C else 0.0
    
    return {
        "tier": tier,
        "tier_name": tier_name,
        "tier_action": tier_action,
        "location_bonus": location_bonus,
        "criteria": {
            "A_skills_match": A,
            "B_experience_match": B,
            "C_location_match": C,
            "D_education_match": D,
            "skill_match_pct": round(skill_pct, 1),
            "must_have_match_pct": round(must_have_pct, 1),
            "candidate_years": candidate_years,
            "required_years": f"{min_years}-{max_years}",
            "location_reason": location_reason,
            "education_reason": education_reason
        }
    }


def cleanup_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up state dictionary by removing internal temporary fields.
    
    Args:
        state: State dictionary to clean
        
    Returns:
        Cleaned state dictionary
    """
    # Remove any temporary/internal fields that shouldn't be in final output
    state.pop("_temp", None)
    state.pop("_internal", None)
    
    return state


async def ranker_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main ranker agent: Semantic filtering (30%) + Batch LLM analysis.
    
    NO SKILL_MATCHER DEPENDENCY - Pure semantic + LLM approach.
    
    Flow:
    1. Get JD details and required skills
    2. Query ChromaDB for candidates with semantic score ≥ 30%
    3. Batch process candidates through LLM (5 per batch)
    4. LLM returns: skills_matched, match_score, rationale for each
    5. Sort by match_score and return top candidates
    
    Args:
        state: State dictionary containing job_description
        
    Returns:
        Updated state with ranked_candidates (ranked by LLM match_score)
    """
    log.info("="*70)
    log.info("🚀 REDESIGNED RANKER V2: Semantic Filter + Batch LLM")
    log.info("="*70)
    _ranker_start = time.time()
    
    # Validate input
    jd_dict = state.get("job_description")
    
    if not jd_dict:
        log.error("No job description provided in state")
        raise RankerAgentError(
            "No job description provided",
            ErrorCode.NO_JOB_DESCRIPTION
        )
    
    try:
        # ============================================================
        # STEP 1: Parse JD and extract details
        # ============================================================
        jd = _jd_cache.get(jd_dict)
        if not jd:
            jd = JobDescription.from_dict(jd_dict)
            _jd_cache.set(jd_dict, jd)
            log.debug("💾 Cached parsed job description")
        else:
            log.debug("✅ Using cached job description")
        
        jd_title = jd.job_title
        required_skills = jd.required_skills
        preferred_skills = jd_dict.get("preferredSkills") or []  # ✅ NEW
        must_have_skills = jd_dict.get("_must_have_skills") or []  # ✅ Priority skills from payload
        jd_experience = jd.experience or "Not specified"
        # ✅ FIXED: Use fullJobDescription if available, include responsibilities - NO TRUNCATION
        jd_responsibilities = jd_dict.get("fullJobDescription") or jd.description or "Not specified"
        # Also get responsibilities list if available
        jd_responsibilities_list = jd_dict.get("responsibilities") or jd_dict.get("keyResponsibilities") or []
        if isinstance(jd_responsibilities_list, list) and jd_responsibilities_list:
            jd_responsibilities = jd_responsibilities + "\n\nKey Responsibilities:\n" + "\n".join([f"- {r}" for r in jd_responsibilities_list])
        # Extract required education from JD
        jd_education = jd_dict.get("educationRequired") or jd_dict.get("education") or jd_dict.get("requiredEducation") or "Not specified"
        work_mode = jd_dict.get("workMode") or ""
        job_domains = jd_dict.get("jobDomains") or jd_dict.get("job_domains") or []
        
        log.info(
            f"📋 JD: {jd_title} | Required skills: {len(required_skills)} | "
            f"Experience: {jd_experience} | Domains: {job_domains[:3] if job_domains else 'N/A'}"
        )
        
        # ============================================================
        # STEP 2: Semantic filtering (ChromaDB with 30% threshold)
        # ============================================================
        jd_text = jd.to_text()
        
        # Get job domain for domain-aware filtering
        job_domain, _ = jd._get_domain_context()
        log.info(f"🔍 Job domain classified as: {job_domain}")
        
        # Enhance query with domain context (returns enhanced text and semantic domain)
        enhanced_jd_text, semantic_domain = await _enhance_query_with_domain_context_async(jd_text, jd)
        job_domain_for_llm = semantic_domain or job_domain  # Prefer semantic domain for LLM
        
        SEMANTIC_THRESHOLD = 0.30
        fetch_limit = Config.MAX_CANDIDATES_TO_RETRIEVE
        
        log.info(f"🔎 Querying ChromaDB for candidates (threshold: {SEMANTIC_THRESHOLD}, limit: {fetch_limit})")
        _step2_start = time.time()

        # Run vector search in thread pool so it doesn't block the event loop
        chroma_candidates = await run_cpu_intensive(
            fetch_top_resumes_from_chroma,
            enhanced_jd_text,
            top_k=fetch_limit,
            where_clause=None,
            similarity_threshold=SEMANTIC_THRESHOLD,
            required_skills=required_skills,
            job_domain=job_domain,
            job_domains=job_domains,
        )
        
        if not chroma_candidates:
            log.warning("No candidates retrieved from ChromaDB")
            state["ranked_candidates"] = []
            state["message"] = f"No candidates found with ≥{SEMANTIC_THRESHOLD*100}% semantic similarity"
            return cleanup_state(state)
        
        _step2_elapsed = time.time() - _step2_start
        log.info(f"✅ ChromaDB returned {len(chroma_candidates)} candidates ({_step2_elapsed:.2f}s)")
        
        # Deduplicate candidates
        chroma_candidates = deduplicate_candidates(chroma_candidates)
        
        # Filter by semantic threshold
        filtered_candidates = [
            c for c in chroma_candidates
            if c.get("vector_similarity", 0.0) >= SEMANTIC_THRESHOLD
        ]
        
        log.info(
            f"✅ Semantic filtering: {len(chroma_candidates)} → "
            f"{len(filtered_candidates)} candidates (≥{SEMANTIC_THRESHOLD*100}% similarity)"
        )
        
        if not filtered_candidates:
            state["ranked_candidates"] = []
            state["message"] = f"No candidates passed {SEMANTIC_THRESHOLD*100}% semantic threshold"
            return cleanup_state(state)
        
        # ============================================================
        # STEP 3: TIER-BASED CLASSIFICATION
        # ============================================================
        _step3_start = time.time()
        jd_location = jd_dict.get("location") or getattr(jd, "location", None) or "N/A"
        
        tiered_candidates = await compute_tiers_for_candidates(
            filtered_candidates,
            required_skills=required_skills,
            must_have_skills=must_have_skills,
            jd_domain=job_domain_for_llm,
            jd_experience=jd_experience,
            jd_location=jd_location,
            work_mode=work_mode,
            jd_education=jd_education
        )
        
        _step3_elapsed = time.time() - _step3_start
        log.info(f"⏱️ Tier classification took {_step3_elapsed:.2f}s for {len(tiered_candidates)} candidates")
        
        candidates_for_llm = [c for c in tiered_candidates if c.get("tier", 5) <= Config.TIER_LLM_CUTOFF]
        candidates_no_llm = [c for c in tiered_candidates if c.get("tier", 5) > Config.TIER_LLM_CUTOFF]
        
        log.info(
            f"🏷️ Tier-based filtering: {len(candidates_for_llm)} candidates for LLM (Tiers 1-{Config.TIER_LLM_CUTOFF}), "
            f"{len(candidates_no_llm)} skipped (Tiers {Config.TIER_LLM_CUTOFF + 1}-5)"
        )
        
        tier_breakdown = {}
        for c in candidates_for_llm:
            t = c.get("tier", 5)
            tier_breakdown[t] = tier_breakdown.get(t, 0) + 1
        log.info(f"   - Tier breakdown for LLM: {tier_breakdown}")
        
        if not candidates_for_llm:
            log.warning("⚠️ No candidates in Tiers 1-3, including Tier 4 candidates as fallback")
            candidates_for_llm = [c for c in tiered_candidates if c.get("tier", 5) <= 4][:20]
            candidates_no_llm = [c for c in tiered_candidates if c not in candidates_for_llm]
        
        if not candidates_for_llm:
            log.warning("⚠️ ALL candidates are Tier 5. Falling back to top candidates by vector similarity.")
            tiered_candidates.sort(key=lambda c: c.get("vector_similarity", 0.0), reverse=True)
            fallback_count = max(10, len(tiered_candidates) // 2)
            candidates_for_llm = tiered_candidates[:fallback_count]
            candidates_no_llm = tiered_candidates[fallback_count:]
            log.info(f"   Fallback: sending top {len(candidates_for_llm)} candidates to LLM by vector similarity")
        
        if not candidates_for_llm:
            state["ranked_candidates"] = []
            state["message"] = "No candidates available for analysis"
            return cleanup_state(state)
        
        # Enforce hard cap — send only the best candidates to the expensive LLM stage
        if len(candidates_for_llm) > Config.MAX_CANDIDATES_TO_PROCESS:
            candidates_for_llm.sort(key=lambda c: (c.get("tier", 5), -c.get("vector_similarity", 0.0)))
            overflow = candidates_for_llm[Config.MAX_CANDIDATES_TO_PROCESS:]
            candidates_for_llm = candidates_for_llm[:Config.MAX_CANDIDATES_TO_PROCESS]
            candidates_no_llm.extend(overflow)
            log.info(
                f"⚡ Hard cap applied: {len(candidates_for_llm)} candidates sent to LLM "
                f"(MAX_CANDIDATES_TO_PROCESS={Config.MAX_CANDIDATES_TO_PROCESS}), "
                f"{len(overflow)} overflow moved to no-LLM pool"
            )
        
        # ============================================================
        # STEP 4: Batch LLM analysis (Tiers 1-3 only)
        # ============================================================
        _step4_start = time.time()
        batch_size = Config.LLM_BATCH_SIZE
        batches = [
            candidates_for_llm[i:i + batch_size]
            for i in range(0, len(candidates_for_llm), batch_size)
        ]
        
        parallel_batch_size = Config.LLM_PARALLEL_BATCHES
        total_rounds = (len(batches) + parallel_batch_size - 1) // parallel_batch_size
        log.info(
            f"🤖 Processing {len(candidates_for_llm)} candidates in "
            f"{len(batches)} batches (size={batch_size}), "
            f"{total_rounds} rounds (parallelism={parallel_batch_size})"
        )
        
        analyzed_candidates = []
        
        for batch_group_idx in range(0, len(batches), parallel_batch_size):
            batch_group = batches[batch_group_idx:batch_group_idx + parallel_batch_size]
            current_round = batch_group_idx // parallel_batch_size + 1
            
            log.info(
                f"📦 Round {current_round}/{total_rounds}: "
                f"{len(batch_group)} batches in parallel "
                f"({sum(len(b) for b in batch_group)} candidates)"
            )
            
            recruiter_questions = state.get("recruiter_questions") or []
            # Process this group of batches in parallel
            tasks = [
                analyze_candidate_batch_with_llm(
                    jd_title=jd_title,
                    jd_skills=required_skills,  # Combined: must_have + parsed
                    jd_must_have_skills=must_have_skills,  # ✅ Priority skills only
                    jd_preferred_skills=preferred_skills,  # ✅ NEW
                    jd_experience=jd_experience,
                    jd_responsibilities=jd_responsibilities,
                    jd_education=jd_education,
                    jd_location=jd_dict.get("location") or getattr(jd, "location", None) or "N/A",
                    jd_work_mode=work_mode,  # ✅ NEW
                    candidates_batch=batch,
                    batch_idx=batch_group_idx + i,
                    recruiter_questions=recruiter_questions if recruiter_questions else None,
                    job_domain=job_domain_for_llm,  # ✅ Semantic domain for contextual analysis
                )
                for i, batch in enumerate(batch_group)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Issue 6.2: Track failures for retry, alerting, and circuit breaker integration
            failed_batches = []
            successful_results = []
            
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    log.error(f"Batch {batch_group_idx + idx} processing failed: {result}")
                    failed_batches.append((batch_group_idx + idx, result))
                    # Record failure for circuit breaker
                    await _llm_circuit_breaker.record_failure()
                    continue
                if result:
                    successful_results.extend(result)
                # Record success for circuit breaker
                await _llm_circuit_breaker.record_success()
            
            # Alert if failure rate exceeds threshold
            failure_rate = len(failed_batches) / len(batch_results) if batch_results else 0
            if failure_rate > 0.3:  # 30% failure threshold
                log.error(
                    f"HIGH_FAILURE_RATE in batch group {batch_group_idx}: "
                    f"{failure_rate:.1%} of batches failed ({len(failed_batches)}/{len(batch_results)})"
                )
            
            analyzed_candidates.extend(successful_results)
        
        _step4_elapsed = time.time() - _step4_start
        log.info(
            f"✅ LLM analysis complete: {len(analyzed_candidates)}/{len(candidates_for_llm)} candidates "
            f"in {_step4_elapsed:.2f}s ({_step4_elapsed / max(1, len(analyzed_candidates)):.2f}s/candidate)"
        )
        
        if not analyzed_candidates:
            state["ranked_candidates"] = []
            state["message"] = "LLM analysis failed for all candidates"
            return cleanup_state(state)
        
        # ============================================================
        # STEP 5: Filter and sort by match_score
        # ============================================================
        # ✅ EXPLICIT: Only LLM-analyzed candidates are included in final results
        log.info(
            f"📊 Processing {len(analyzed_candidates)} LLM-analyzed candidates "
            f"(bottom {len(candidates_no_llm)} candidates excluded - no LLM analysis)"
        )
        
        # ✅ VALIDATION: Ensure all candidates have LLM analysis fields
        llm_validated_candidates = []
        for candidate in analyzed_candidates:
            if "match_score" not in candidate or "rationale" not in candidate:
                log.warning(f"⚠️ Skipping candidate {candidate.get('candidate_id', 'unknown')} - missing LLM analysis fields")
                continue
            llm_validated_candidates.append(candidate)
        
        if len(llm_validated_candidates) < len(analyzed_candidates):
            log.warning(f"⚠️ Filtered out {len(analyzed_candidates) - len(llm_validated_candidates)} candidates missing LLM fields")
        
        analyzed_candidates = llm_validated_candidates
        analyzed_candidates.sort(
            key=lambda c: c.get("match_score", 0.0),
            reverse=True
        )
        
        # ============================================================
        # STEP 5b: POST-LLM TIER COMPUTATION
        # ============================================================
        _step5_start = time.time()
        log.info(f"🏷️ Computing post-LLM tiers for {len(analyzed_candidates)} candidates...")
        
        # Format output - ONLY LLM-analyzed candidates (no filter by skill %; Option B scales score when low)
        ranked_output = []
        adjusted_count = 0
        for rank, candidate in enumerate(analyzed_candidates, start=1):
            # ✅ DOUBLE-CHECK: Verify this candidate was LLM-analyzed
            if "match_score" not in candidate or "rationale" not in candidate:
                log.error(f"❌ CRITICAL: Candidate {candidate.get('candidate_id')} missing LLM fields - skipping")
                continue
            # Skills already pre-validated before LLM (skill dropping first, then rationale)
            skills_matched = candidate.get("skills_matched", [])
            skills_unmatched = [s for s in required_skills if s not in skills_matched]
            skill_match_percentage = round(len(skills_matched) / len(required_skills) * 100, 2) if required_skills else 0.0
            rationale = candidate.get("rationale", "")
            
            # Option B: when skill match is low and LLM is "optimistic", scale down match_score
            llm_match_score = candidate.get("match_score", 0.0)
            skill_implied_score = skill_match_percentage / 100.0
            llm_optimistic = llm_match_score > skill_implied_score + Config.LLM_OPTIMISTIC_THRESHOLD
            if (
                skill_match_percentage < Config.MIN_SKILL_MATCH_PERCENTAGE
                and llm_optimistic
            ):
                effective_match_score = llm_match_score * (
                    skill_match_percentage / Config.MIN_SKILL_MATCH_PERCENTAGE
                )
                adjusted_count += 1
                score_was_scaled = True
            else:
                effective_match_score = llm_match_score
                score_was_scaled = False
            
            # Extract candidate details - use pre-extracted values from ChromaDB
            candidate_name = candidate.get("name", "Unknown")
            candidate_email = candidate.get("email", "")
            candidate_phone = candidate.get("phone", "")
            
            # Get resume for experience extraction
            resume = candidate.get("candidate", {}).get("structuredResume", {})
            
            # Prefer domain-relevant experience (LLM estimate for this job's domain) over total experience
            domain_years = candidate.get("domain_relevant_experience_years")
            if isinstance(domain_years, (int, float)) and domain_years >= 0:
                candidate_years = float(domain_years)
            else:
                candidate_exp = resume.get("total_experience_years", 0)
                try:
                    candidate_years = float(candidate_exp) if candidate_exp else 0.0
                except (ValueError, TypeError):
                    candidate_years = 0.0
            
            # Parse required experience from JD
            required_years = 0.0
            max_years = None
            experience_match_status = "unknown"
            experience_match_score = 0.0
            
            if jd_experience and jd_experience != "Not specified":
                import re
                exp_match = re.search(r'(\d+)(?:\s*-\s*(\d+))?', jd_experience)
                if exp_match:
                    required_years = float(exp_match.group(1))
                    if exp_match.group(2):
                        max_years = float(exp_match.group(2))
                    
                    # Calculate experience match
                    if max_years:
                        if required_years <= candidate_years <= max_years:
                            experience_match_status = "exact_match"
                            experience_match_score = 1.0
                        elif candidate_years < required_years:
                            experience_match_status = "under_qualified"
                            experience_match_score = max(0.0, candidate_years / required_years)
                        else:
                            experience_match_status = "over_qualified"
                            experience_match_score = 0.9
                    else:
                        if candidate_years >= required_years:
                            experience_match_status = "qualified"
                            experience_match_score = min(1.0, candidate_years / required_years)
                        else:
                            experience_match_status = "under_qualified"
                            experience_match_score = max(0.0, candidate_years / required_years)
            
            # Education match status (meets / above / below / not_specified)
            education_match_status = _compute_education_match_status(
                jd_education, candidate.get("education_summary", "") or ""
            )
            # Must-have coverage when JD has must-have skills
            must_have_skills_matched = []
            must_have_count = 0
            must_have_matched_count = 0
            if must_have_skills:
                must_have_count = len(must_have_skills)
                must_have_skills_matched = [s for s in must_have_skills if s in skills_matched]
                must_have_matched_count = len(must_have_skills_matched)
            
            # ============================================================
            # POST-LLM TIER COMPUTATION (uses LLM results for accurate tier)
            # ============================================================
            # Compute tier from LLM analysis results (skills_matched, domain_relevant_experience_years, etc.)
            # This is more reliable than pre-LLM tier calculation which may have data extraction issues
            tier_result = compute_tier_from_llm_results(
                candidate=candidate,
                jd_skills=required_skills,
                jd_must_have_skills=must_have_skills,
                jd_experience=jd_experience,
                jd_location=jd_location,
                jd_education=jd_education,
                work_mode=work_mode
            )
            tier = tier_result["tier"]
            tier_name = tier_result["tier_name"]
            tier_action = tier_result["tier_action"]
            criteria = tier_result["criteria"]
            location_bonus = tier_result.get("location_bonus", 0.0)
            
            # Apply location bonus to match score (minor tiebreaker within tier)
            match_score_val = round(effective_match_score + location_bonus, 4)
            
            # Log tier computation for debugging
            loc_tag = " 📍" if location_bonus > 0 else ""
            log.debug(
                f"   📊 {candidate['candidate_id']}: Tier {tier} ({tier_name}){loc_tag} - "
                f"A={criteria.get('A_skills_match')}, B={criteria.get('B_experience_match')}, "
                f"C={criteria.get('C_location_match')} (tiebreaker), D={criteria.get('D_education_match')}"
            )
            
            ranked_output.append({
                "rank": rank,
                "candidate_id": candidate["candidate_id"],
                "name": candidate_name,
                "email": candidate_email,
                "phone": candidate_phone,
                # Tier information (computed post-LLM)
                "tier": tier,
                "tier_name": tier_name,
                "tier_action": tier_action,
                "tier_criteria": {
                    "skills_match": criteria.get("A_skills_match", False),
                    "experience_match": criteria.get("B_experience_match", False),
                    "location_match": criteria.get("C_location_match", False),
                    "location_match_note": "tiebreaker only, not used for tier classification",
                    "education_match": criteria.get("D_education_match", False),
                    "skill_match_pct": criteria.get("skill_match_pct", 0.0),
                    "must_have_match_pct": criteria.get("must_have_match_pct", 0.0),
                    "candidate_years": criteria.get("candidate_years", 0.0),
                    "required_years": criteria.get("required_years", ""),
                    "location_reason": criteria.get("location_reason", ""),
                    "location_bonus_applied": location_bonus,
                },
                "domain_relevant_years": criteria.get("candidate_years", 0.0),
                "match_score": match_score_val,
                "match_band": _compute_match_band(effective_match_score),
                "score_explanation": _align_score_explanation_to_final_score(
                    candidate.get("score_explanation", ""),
                    match_score_val,
                    score_was_scaled,
                ),
                "skill_match_percentage": round(skill_match_percentage, 1),
                "skill_match_count": len(skills_matched),
                "skills_matched": skills_matched,
                "skills_unmatched": skills_unmatched,
                "rationale": rationale,
                "positive_rationale": candidate.get("positive_rationale", ""),
                "negative_rationale": candidate.get("negative_rationale", ""),
                "vector_similarity": candidate.get("vector_similarity", 0.0),
                # Preferred / nice-to-have skills
                "preferred_skills_matched": candidate.get("preferred_skills_matched", []),
                # Current role, education, location (recruiter quick scan)
                "current_role": candidate.get("current_role", ""),
                "education_summary": candidate.get("education_summary", ""),
                "current_location": candidate.get("current_location", ""),
                "education_match_status": education_match_status,
                # Must-have coverage
                "must_have_skills_matched": must_have_skills_matched,
                "must_have_count": must_have_count,
                "must_have_matched_count": must_have_matched_count,
                # Short summary and red flags
                "summary": candidate.get("summary", ""),
                "concerns": candidate.get("concerns", []) or [],
                "recruiter_qa_summary": candidate.get("recruiter_qa_summary", ""),
                # Experience information (domain-relevant when LLM provided it)
                "candidate_experience_years": round(candidate_years, 1),
                "domain_relevant_experience_years": candidate.get("domain_relevant_experience_years"),
                "required_experience_years": round(required_years, 1) if required_years > 0 else None,
                "max_experience_years": round(max_years, 1) if max_years else None,
                "experience_match_status": experience_match_status,
                "experience_match_score": round(experience_match_score, 3)
            })
        
        # Sort by match_score descending so rank 1 = highest score
        ranked_output.sort(key=lambda c: -c.get("match_score", 0.0))
        for idx, c in enumerate(ranked_output, start=1):
            c["rank"] = idx
        
        # Log tier breakdown in final output
        final_tier_counts = {}
        for c in ranked_output:
            t = c.get("tier", 5)
            final_tier_counts[t] = final_tier_counts.get(t, 0) + 1
        log.info(f"📊 Final output tier breakdown: {final_tier_counts}")
        
        if adjusted_count > 0:
            log.info(
                f"📉 Adjusted score for {adjusted_count} candidates with skill match below {Config.MIN_SKILL_MATCH_PERCENTAGE}% "
                f"(fit reflected in lower score; {len(ranked_output)} candidates ranked)"
            )
        
        if not ranked_output:
            state["ranked_candidates"] = []
            state["message"] = "No candidates with matching skills found"
            return cleanup_state(state)
        
        log.info("="*70)
        log.info("🎯 FINAL RANKING (LLM Match Score)")
        log.info("="*70)
        for candidate in ranked_output[:10]:  # Show top 10
            log.info(
                f"Rank {candidate['rank']}: {candidate['candidate_id']} | "
                f"Match: {candidate['match_score']:.3f} | "
                f"Skills: {candidate['skill_match_count']}/{len(required_skills)} "
                f"({candidate['skill_match_percentage']:.1f}%) | "
                f"Vector: {candidate['vector_similarity']:.3f}"
            )
        log.info("="*70)
        
        # ✅ FINAL VALIDATION: Ensure all returned candidates were LLM-analyzed
        for candidate in ranked_output:
            if "rationale" not in candidate or candidate.get("match_score") is None:
                log.error(f"❌ CRITICAL: Candidate {candidate.get('candidate_id')} in results but missing LLM analysis!")
                raise ValueError("Non-LLM analyzed candidate found in final results")
        
        log.info(f"✅ VALIDATED: All {len(ranked_output)} returned candidates were LLM-analyzed")
        
        _step5_elapsed = time.time() - _step5_start
        processing_time = time.time() - _ranker_start
        
        log.info("="*70)
        log.info("⏱️ PIPELINE TIMING BREAKDOWN")
        log.info(f"   Step 2 (ChromaDB fetch):      {_step2_elapsed:>7.2f}s")
        log.info(f"   Step 3 (Tier classification):  {_step3_elapsed:>7.2f}s")
        log.info(f"   Step 4 (LLM analysis):         {_step4_elapsed:>7.2f}s")
        log.info(f"   Step 5 (Post-LLM tier + sort): {_step5_elapsed:>7.2f}s")
        log.info(f"   TOTAL:                         {processing_time:>7.2f}s")
        log.info(f"   Candidates: {fetch_limit} fetched → {len(filtered_candidates)} filtered → {len(candidates_for_llm)} LLM'd → {len(ranked_output)} ranked")
        log.info("="*70)
        confidence_score = (
            sum(c["match_score"] for c in ranked_output) / len(ranked_output)
            if ranked_output else 0.0
        )
        confidence_level = (
            "high" if confidence_score >= 0.7 
            else ("medium" if confidence_score >= 0.5 else "low")
        )
        
        state["ranked_candidates"] = ranked_output
        state["confidence_score"] = round(confidence_score, 4)
        state["confidence_level"] = confidence_level
        state["processing_time_seconds"] = round(processing_time, 3)
        state["total_unique_candidates"] = len(ranked_output)
        state["semantic_filter_threshold"] = SEMANTIC_THRESHOLD
        state["processing_method"] = "semantic_filter_batch_llm_v2"
        state["batches_processed"] = len(batches)
        state["pipeline_stats"] = {
            "chromadb_fetch_limit": fetch_limit,
            "candidates_after_semantic_filter": len(filtered_candidates),
            "candidates_sent_to_llm": len(candidates_for_llm),
            "candidates_skipped_llm": len(candidates_no_llm),
            "candidates_ranked": len(ranked_output),
            "llm_batch_size": batch_size,
            "llm_parallel_batches": parallel_batch_size,
            "timing": {
                "chromadb_fetch_s": round(_step2_elapsed, 2),
                "tier_classification_s": round(_step3_elapsed, 2),
                "llm_analysis_s": round(_step4_elapsed, 2),
                "post_llm_tier_sort_s": round(_step5_elapsed, 2),
                "total_s": round(processing_time, 2),
            },
        }
        
        log.info(f"✅ Ranker V2 complete: {len(ranked_output)} candidates ranked in {processing_time:.2f}s")
        log.info(f"📊 Confidence: {confidence_score:.3f} ({confidence_level})")
        
        # Store callback
        try:
            import time as _t
            uid = str(
                state.get("uid") or state.get("request_id") 
                or int(_t.time() * 1000)
            )
            job_id_for_cb = state.get("jd_id") or state.get("job_id") or ""
            
            job_title_cb = jd_dict.get("jobTitle") or jd_dict.get("title") or ""
            company_cb = jd_dict.get("company") or ""
            callback_payload = {
                "status": "completed",
                "node": "ranker_v2",
                "job_id": job_id_for_cb,
                "job_title": job_title_cb,
                "company": company_cb,
                "output": {
                    "job_id": job_id_for_cb,
                    "job_title": job_title_cb,
                    "company": company_cb,
                    "ranked_candidates": ranked_output,
                    "confidence_score": round(confidence_score, 4),
                    "confidence_level": confidence_level,
                    "processing_time_seconds": round(processing_time, 3),
                    "total_unique_candidates": len(ranked_output),
                    "ranking_method": "semantic_filter_batch_llm_v2",
                    "semantic_threshold": SEMANTIC_THRESHOLD,
                    "batches_processed": len(batches)
                }
            }
            
            store_callback("ranker_v2", uid, callback_payload)
        except Exception:
            pass
        
        return cleanup_state(state)
    
    except RankerAgentError:
        raise
    except Exception as e:
        log.error(f"Unexpected error in ranker_agent_v2: {e}", exc_info=True)
        raise RankerAgentError(
            f"Unexpected error: {str(e)}",
            ErrorCode.INVALID_INPUT
        )
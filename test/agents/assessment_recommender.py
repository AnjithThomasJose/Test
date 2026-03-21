import json
import logging
import os
import re
import asyncio
import random
import hashlib
import threading
import uuid
import time
import copy
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from uuid import uuid4
from pydantic import BaseModel, Field, ValidationError

from models.llm_invoker import invoke_llm, invoke_structured_llm
from core.model_registry import TaskType
from chroma import fetch_structured_resume, get_chat_session, update_chat_session
from chroma import get_assessments_doc, upsert_assessments_doc  # NEW per-UID assessments helpers
from utils.session_manager import session_manager
from core.utils import (
    _mask, _sanitize_text_for_llm, _validate_state_inputs, _to_text, _clean_json_text, 
    _scan_balanced_json, _safe_json_loads, _extract_json_from_response, _create_error_response,
    _generate_request_id, _calculate_processing_time, create_agent_state, filter_resume_data_for_agent,
    smart_resume_truncation, compress_chat_history, get_cached_response, cache_response,
    run_blocking_io, get_missing_skills_flat
)
from core.config import get_agent_config
from settings import settings as app_settings
from core.security import validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm, PII_PATTERNS, INJECTION_FILTERS
from core.memory import BaseAgentMemory, get_agent_memory
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion

# Get centralized configuration
config = get_agent_config("assessment_recommender")

# Custom memory class for assessment recommender (extends base memory)
class AssessmentRecommenderMemory(BaseAgentMemory):
    def __init__(self, tenant_id: str = "default"):
        super().__init__(tenant_id, config.adaptation_window)
        # Add any assessment recommender specific fields here if needed

# Use centralized memory management
async def get_assessment_recommender_memory(tenant_id: str = "default") -> AssessmentRecommenderMemory:
    """Get or create tenant-scoped assessment recommender memory."""
    return await get_agent_memory("assessment_recommender", tenant_id, AssessmentRecommenderMemory)

log = logging.getLogger(__name__)


@dataclass
class PersonalizationContext:
    """User personalization context for assessment recommendations"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    profession_type: Optional[str] = None  # technology, healthcare, trades, etc.
    skill_level: Optional[str] = None  # beginner, intermediate, advanced, expert
    experience_years: Optional[int] = None
    domain_specializations: List[str] = field(default_factory=list)
    previous_performance: Optional[float] = None  # 0.0-1.0 from interaction history
    personalization_score: float = 0.0  # How well personalized (0.0-1.0)
    assessment_history: List[Dict[str, Any]] = field(default_factory=list)  # Past assessment performance
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)  # How user interacts with system


class AssessmentRecommenderCache:
    """User-isolated cache for assessment recommendations with TTL support."""
    
    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
    
    def get(self, key: str, user_context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get cached item if not expired and user-isolated."""
        with self._lock:
            cache_key = self._generate_cache_key(key, user_context)
            
            if cache_key in self.cache:
                value, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    log.debug("Assessment cache hit", extra={
                        "cache_key": cache_key[:16] + "...",
                        "user_id": (user_context.get("user_id", "anonymous")[:8] + "...") if user_context else "anonymous"
                    })
                    return value
                else:
                    del self.cache[cache_key]
                    log.debug("Assessment cache expired", extra={"cache_key": cache_key[:16] + "..."})
            return None
    
    def set(self, key: str, value: Dict[str, Any], user_context: Optional[Dict[str, Any]] = None) -> None:
        """Set cached item with user isolation and automatic cleanup."""
        with self._lock:
            cache_key = self._generate_cache_key(key, user_context)
            
            if len(self.cache) >= self.max_size:
                self._cleanup_expired()
                if len(self.cache) >= self.max_size:
                    oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                    del self.cache[oldest_key]
            
            self.cache[cache_key] = (value, datetime.now())
            log.debug("Assessment cache set", extra={
                "cache_key": cache_key[:16] + "...",
                "user_id": (user_context.get("user_id", "anonymous")[:8] + "...") if user_context else "anonymous"
            })
    
    def _generate_cache_key(self, base_key: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate user-isolated cache key to prevent cross-contamination."""
        content = base_key
        
        if user_context:
            user_id = user_context.get("user_id", "anonymous")
            experience_years = user_context.get("experience_years", 0)
            profession = user_context.get("profession_type", "general")
            skill_level = user_context.get("skill_level", "beginner")
            user_hash = hashlib.md5(f"{user_id}:{experience_years}:{profession}:{skill_level}".encode()).hexdigest()[:8]
            content += f":user:{user_hash}"
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cleanup_expired(self) -> None:
        """Remove all expired entries. Must be called with lock held."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if now - timestamp >= timedelta(seconds=self.ttl_seconds)
        ]
        for key in expired_keys:
            del self.cache[key]


class AssessmentRecommenderError(Exception):
    """Base exception for assessment recommender errors."""
    pass


class InvalidResumeError(AssessmentRecommenderError):
    """Raised when resume data is invalid or missing."""
    pass


class LLMResponseError(AssessmentRecommenderError):
    """Raised when LLM response cannot be parsed."""
    pass


# Configuration constants
config = get_agent_config("assessment_recommender")
TIMEOUT_SECONDS = config.timeout_seconds
MAX_PROMPT_CHARS = config.max_prompt_chars
MAX_RESPONSE_LENGTH = config.max_response_length
LLM_MODEL = config.llm_model

# Adaptive re-run constants (configurable with sensible defaults, hard default 80)
PASS_THRESHOLD = 80
DIFFICULTY_ORDER = ["easy", "medium", "hard", "expert"]
MAX_PLAN_LENGTH = 12
# Backwards-compatible constant (no longer used for behavior, only for tests/config)
PROMOTION_POLICY = "immediate"


# Helper functions for adaptive re-run logic

def deep_copy(obj: Any) -> Any:
    """
    Deep copy an object, handling None and primitives gracefully.
    
    Args:
        obj: Object to deep copy
        
    Returns:
        Deep copy of the object, or the original if it's a primitive/None
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    try:
        return copy.deepcopy(obj)
    except Exception:
        # Fallback for objects that can't be deep copied
        return obj


def normalize_topic_config(cfg: Dict[str, Any], recommender_instance=None) -> Dict[str, Any]:
    """
    Normalize a topic configuration to ensure all required fields are present.
    
    Args:
        cfg: Topic configuration dict
        recommender_instance: Optional UnifiedAssessmentRecommender instance for time calculation
        
    Returns:
        Normalized topic configuration with all required fields
    """
    if not isinstance(cfg, dict):
        return {}
    
    normalized = dict(cfg)
    
    # Normalize topic, difficulty, type (case-insensitive)
    if "topic" in normalized:
        normalized["topic"] = str(normalized["topic"]).strip()
    if "difficulty" in normalized:
        diff = str(normalized["difficulty"]).lower().strip()
        # Map common variations to standard values
        diff_map = {
            "easy": "easy",
            "medium": "medium",
            "hard": "hard",
            "expert": "expert",
            "beginner": "easy",
            "intermediate": "medium",
            "advanced": "hard",
            "outstanding": "expert"
        }
        normalized["difficulty"] = diff_map.get(diff, diff)
    if "type" in normalized:
        normalized["type"] = str(normalized["type"]).lower().strip()
    else:
        normalized["type"] = "multi"
    
    # Ensure num_questions dict exists with defaults
    if "num_questions" not in normalized or not isinstance(normalized["num_questions"], dict):
        normalized["num_questions"] = {"mcq": 4, "short": 1, "long": 0, "coding": 0}
    else:
        num_q = normalized["num_questions"]
        normalized["num_questions"] = {
            "mcq": num_q.get("mcq", 4),
            "short": num_q.get("short", 1),
            "long": num_q.get("long", 0),
            "coding": num_q.get("coding", 0)
        }
    
    # Compute/set time_minutes: respect explicit time limit first, then compute from question counts
    time_min = 10
    time_max = 60
    if "time_limit_minutes" in normalized:
        raw = int(normalized.pop("time_limit_minutes", 0) or 0)
        normalized["assessment_time_minutes"] = max(time_min, min(time_max, raw))
    elif "assessment_time_minutes" in normalized:
        raw = int(normalized.get("assessment_time_minutes", 0) or 0)
        normalized["assessment_time_minutes"] = max(time_min, min(time_max, raw))
    elif "time_minutes" in normalized:
        raw = int(normalized.pop("time_minutes", 0) or 0)
        normalized["assessment_time_minutes"] = max(time_min, min(time_max, raw))
    else:
        if recommender_instance:
            try:
                normalized["assessment_time_minutes"] = recommender_instance.calculate_assessment_time(
                    normalized, None
                )
            except Exception:
                # Fallback heuristic: base time on question counts
                num_q = normalized["num_questions"]
                base_time = (
                    num_q.get("mcq", 0) * 1 +
                    num_q.get("short", 0) * 5 +
                    num_q.get("long", 0) * 10 +
                    num_q.get("coding", 0) * 20
                )
                normalized["assessment_time_minutes"] = max(time_min, base_time)
        else:
            # Simple heuristic without instance
            num_q = normalized["num_questions"]
            base_time = (
                num_q.get("mcq", 0) * 1 +
                num_q.get("short", 0) * 5 +
                num_q.get("long", 0) * 10 +
                num_q.get("coding", 0) * 20
            )
            normalized["assessment_time_minutes"] = max(time_min, base_time)
        normalized["assessment_time_minutes"] = min(time_max, normalized["assessment_time_minutes"])
    
    # Preserve source field if present, otherwise set default
    if "source" not in normalized:
        normalized["source"] = "unknown"
    
    # Default recommendation_source; overwritten when plan is built with gap analysis
    if "recommendation_source" not in normalized:
        normalized["recommendation_source"] = "Recommended to validate and showcase your existing skills."
    
    # Rationale for user motivation (why this assessment was recommended); may be overwritten later
    if "rationale" not in normalized:
        normalized["rationale"] = normalized.get(
            "recommendation_source",
            "Recommended to validate and showcase your existing skills.",
        )

    return normalized


def promote_difficulty(curr: str) -> str:
    """
    Promote difficulty by one step in DIFFICULTY_ORDER.
    
    Args:
        curr: Current difficulty (case-insensitive)
        
    Returns:
        Promoted difficulty, clamped to max
    """
    if not curr:
        return DIFFICULTY_ORDER[0]
    
    curr_lower = str(curr).lower().strip()
    # Map common variations
    diff_map = {
        "easy": "easy",
        "medium": "medium",
        "hard": "hard",
        "expert": "expert",
        "beginner": "easy",
        "intermediate": "medium",
        "advanced": "hard",
        "outstanding": "expert"
    }
    curr_normalized = diff_map.get(curr_lower, curr_lower)
    
    try:
        idx = DIFFICULTY_ORDER.index(curr_normalized)
        if idx < len(DIFFICULTY_ORDER) - 1:
            return DIFFICULTY_ORDER[idx + 1]
        return DIFFICULTY_ORDER[-1]  # Already at max
    except ValueError:
        # Unknown difficulty, return medium as safe default
        return "medium"


def demote_difficulty(curr: str) -> str:
    """
    Demote difficulty by one step in DIFFICULTY_ORDER.
    
    Args:
        curr: Current difficulty (case-insensitive)
        
    Returns:
        Demoted difficulty, clamped to min
    """
    if not curr:
        return DIFFICULTY_ORDER[0]
    
    curr_lower = str(curr).lower().strip()
    # Map common variations
    diff_map = {
        "easy": "easy",
        "medium": "medium",
        "hard": "hard",
        "expert": "expert",
        "beginner": "easy",
        "intermediate": "medium",
        "advanced": "hard",
        "outstanding": "expert"
    }
    curr_normalized = diff_map.get(curr_lower, curr_lower)
    
    try:
        idx = DIFFICULTY_ORDER.index(curr_normalized)
        if idx > 0:
            return DIFFICULTY_ORDER[idx - 1]
        return DIFFICULTY_ORDER[0]  # Already at min
    except ValueError:
        # Unknown difficulty, return easy as safe default
        return "easy"


async def safe_llm_generate(
    prompt: str,
    *,
    model_name: str = None,
    timeout_s: float = 30.0,
    recommender_instance=None,
    **kwargs
) -> Optional[Any]:
    """
    Safely generate LLM response with timeout and error handling.
    
    Args:
        prompt: Prompt to send to LLM
        model_name: Model name (defaults to config.llm_model)
        timeout_s: Timeout in seconds
        recommender_instance: Optional UnifiedAssessmentRecommender instance for _llm_with_retry
        **kwargs: Additional arguments for LLM call
        
    Returns:
        LLM response content or None on failure
    """
    if not model_name:
        model_name = LLM_MODEL
    
    try:
        if recommender_instance and hasattr(recommender_instance, "_llm_with_retry"):
            llm_response = await asyncio.wait_for(
                recommender_instance._llm_with_retry(prompt, model=model_name, **kwargs),
                timeout=timeout_s
            )
        else:
            # Fallback to direct invoke_llm
            # Map 'model' to 'preferred_model' and extract max_output_tokens from kwargs
            invoke_kwargs = {k: v for k, v in kwargs.items() if k != 'model'}
            if 'model' in kwargs:
                invoke_kwargs['preferred_model'] = kwargs['model']
            elif model_name:
                invoke_kwargs['preferred_model'] = model_name
            llm_response = await asyncio.wait_for(
                invoke_llm(prompt, **invoke_kwargs),
                timeout=timeout_s
            )
        
        content = getattr(llm_response, "content", str(llm_response))
        return content
    except asyncio.TimeoutError:
        log.warning(f"LLM generation timed out after {timeout_s}s")
        return None
    except Exception as e:
        log.warning(f"LLM generation failed: {e}")
        return None


def get_reinforcement_topics(topic: str) -> List[Dict[str, Any]]:
    """
    Get reinforcement/advanced topics for a given topic (static mapping).
    
    Args:
        topic: Topic name
        
    Returns:
        List of normalized topic configurations
    """
    if not topic:
        return []
    topic_lower = str(topic).lower().strip()
    
    # Small curated mapping for common topics
    reinforcement_map = {
        "python": [
            {"topic": "Advanced Python", "difficulty": "hard", "type": "multi",
             "num_questions": {"mcq": 6, "short": 2, "long": 1, "coding": 1}, "source": "static_reinforcement"},
            {"topic": "Python Design Patterns", "difficulty": "hard", "type": "multi",
             "num_questions": {"mcq": 5, "short": 2, "long": 1, "coding": 0}, "source": "static_reinforcement"}
        ],
        "react": [
            {"topic": "Advanced React", "difficulty": "hard", "type": "multi",
             "num_questions": {"mcq": 6, "short": 2, "long": 1, "coding": 1}, "source": "static_reinforcement"},
            {"topic": "React Performance Optimization", "difficulty": "hard", "type": "multi",
             "num_questions": {"mcq": 5, "short": 2, "long": 1, "coding": 0}, "source": "static_reinforcement"}
        ],
        "javascript": [
            {"topic": "Advanced JavaScript", "difficulty": "hard", "type": "multi",
             "num_questions": {"mcq": 6, "short": 2, "long": 1, "coding": 1}, "source": "static_reinforcement"},
            {"topic": "JavaScript Design Patterns", "difficulty": "hard", "type": "multi",
             "num_questions": {"mcq": 5, "short": 2, "long": 1, "coding": 0}, "source": "static_reinforcement"}
        ],
        "java": [
            {"topic": "Advanced Java", "difficulty": "hard", "type": "multi",
             "num_questions": {"mcq": 6, "short": 2, "long": 1, "coding": 1}, "source": "static_reinforcement"},
            {"topic": "Java Concurrency", "difficulty": "hard", "type": "multi",
             "num_questions": {"mcq": 5, "short": 2, "long": 1, "coding": 0}, "source": "static_reinforcement"}
        ],
        "communication": [
            {"topic": "Advanced Communication", "difficulty": "hard", "type": "multi",
             "num_questions": {"mcq": 6, "short": 2, "long": 1, "coding": 0}, "source": "static_reinforcement"}
        ]
    }
    
    # Try exact match first, then partial match
    if topic_lower in reinforcement_map:
        return reinforcement_map[topic_lower]
    
    for key, topics in reinforcement_map.items():
        if key in topic_lower or topic_lower in key:
            return topics
    
    return []


def get_foundational_topics(topic: str) -> List[Dict[str, Any]]:
    """
    Get foundational/basic topics for a given topic (static mapping).
    
    Args:
        topic: Topic name
        
    Returns:
        List of normalized topic configurations
    """
    if not topic:
        return []
    topic_lower = str(topic).lower().strip()
    
    # Small curated mapping for common topics
    foundational_map = {
        "python": [
            {"topic": "Python Basics", "difficulty": "easy", "type": "multi",
             "num_questions": {"mcq": 5, "short": 1, "long": 0, "coding": 0}, "source": "static_foundational"}
        ],
        "react": [
            {"topic": "React Fundamentals", "difficulty": "easy", "type": "multi",
             "num_questions": {"mcq": 5, "short": 1, "long": 0, "coding": 0}, "source": "static_foundational"}
        ],
        "javascript": [
            {"topic": "JavaScript Basics", "difficulty": "easy", "type": "multi",
             "num_questions": {"mcq": 5, "short": 1, "long": 0, "coding": 0}, "source": "static_foundational"}
        ],
        "java": [
            {"topic": "Java Fundamentals", "difficulty": "easy", "type": "multi",
             "num_questions": {"mcq": 5, "short": 1, "long": 0, "coding": 0}, "source": "static_foundational"}
        ],
        "communication": [
            {"topic": "Communication Basics", "difficulty": "easy", "type": "multi",
             "num_questions": {"mcq": 5, "short": 1, "long": 0, "coding": 0}, "source": "static_foundational"}
        ]
    }
    
    # Try exact match first, then partial match
    if topic_lower in foundational_map:
        return foundational_map[topic_lower]
    
    for key, topics in foundational_map.items():
        if key in topic_lower or topic_lower in key:
            return topics
    
    return []


def get_remedial_topics(topic: str) -> List[Dict[str, Any]]:
    """
    Get remedial topics for a given topic (static mapping).
    
    Args:
        topic: Topic name
        
    Returns:
        List of normalized topic configurations
    """
    if not topic:
        return []
    topic_lower = str(topic).lower().strip()
    
    # Small curated mapping for common topics
    remedial_map = {
        "python": [
            {"topic": "Python Fundamentals Review", "difficulty": "easy", "type": "multi",
             "num_questions": {"mcq": 4, "short": 1, "long": 0, "coding": 0}, "source": "static_remedial"}
        ],
        "react": [
            {"topic": "React Core Concepts Review", "difficulty": "easy", "type": "multi",
             "num_questions": {"mcq": 4, "short": 1, "long": 0, "coding": 0}, "source": "static_remedial"}
        ],
        "javascript": [
            {"topic": "JavaScript Essentials Review", "difficulty": "easy", "type": "multi",
             "num_questions": {"mcq": 4, "short": 1, "long": 0, "coding": 0}, "source": "static_remedial"}
        ],
        "java": [
            {"topic": "Java Core Review", "difficulty": "easy", "type": "multi",
             "num_questions": {"mcq": 4, "short": 1, "long": 0, "coding": 0}, "source": "static_remedial"}
        ],
        "communication": [
            {"topic": "Communication Skills Review", "difficulty": "easy", "type": "multi",
             "num_questions": {"mcq": 4, "short": 1, "long": 0, "coding": 0}, "source": "static_remedial"}
        ]
    }
    
    # Try exact match first, then partial match
    if topic_lower in remedial_map:
        return remedial_map[topic_lower]
    
    for key, topics in remedial_map.items():
        if key in topic_lower or topic_lower in key:
            return topics
    
    return []


def get_lateral_topics(topic: str) -> List[Dict[str, Any]]:
    """
    Get lateral (sideways) topics for a given topic (static mapping).
    
    Args:
        topic: Topic name
        
    Returns:
        List of topic configurations representing lateral concepts
    """
    if not topic:
        return []
    topic_lower = str(topic).lower().strip()
    
    lateral_map = {
        "python": [
            {
                "topic": "Python Libraries Overview",
                "difficulty": "medium",
                "type": "multi",
                "num_questions": {"mcq": 4, "short": 1, "long": 0, "coding": 0},
                "source": "static_lateral"
            }
        ],
        "react": [
            {
                "topic": "React Ecosystem Overview",
                "difficulty": "medium",
                "type": "multi",
                "num_questions": {"mcq": 4, "short": 1, "long": 0, "coding": 0},
                "source": "static_lateral"
            }
        ],
        "javascript": [
            {
                "topic": "JavaScript Tooling Overview",
                "difficulty": "medium",
                "type": "multi",
                "num_questions": {"mcq": 4, "short": 1, "long": 0, "coding": 0},
                "source": "static_lateral"
            }
        ]
    }
    
    return lateral_map.get(topic_lower, [])


class TopicConfig(BaseModel):
    """Structured schema for LLM-generated topic configs."""
    topic: str
    difficulty: str = Field(default="medium")
    num_questions: Dict[str, int] = Field(
        default_factory=lambda: {"mcq": 4, "short": 1, "long": 0, "coding": 0}
    )


def lateral_prompt_for_topic(topic: str, score: float, personalization_ctx: PersonalizationContext) -> str:
    """
    Build prompt for lateral (sideways) topic generation when user passed but is not promoted.
    """
    return f"""Generate 2-3 lateral assessment topics for a candidate who scored {score}% in "{topic}".

Context:
- Skill Level: {personalization_ctx.skill_level}
- Experience: {personalization_ctx.experience_years or 'Unknown'} years

Rules:
- Suggest topics that are related but sideways to "{topic}" (e.g., ecosystem, tooling, libraries)
- Do NOT increase overall difficulty; stay roughly at the same level
- Topics should reinforce breadth rather than depth

Return ONLY a JSON array of topic configs:
[
  {{
    "topic": "<topic_name>",
    "difficulty": "medium",
    "num_questions": {{"mcq": 4, "short": 1, "long": 0, "coding": 0}}
  }}
]"""


def generate_unique_assessment_id() -> str:
    """
    Generate a unique assessment ID using timestamp and UUID.
    This ensures uniqueness even if topic names, difficulty, or any other
    attributes are the same. Each call generates a completely new unique ID.
    
    The UUID component ensures uniqueness even if called at the exact same
    microsecond, so assessments with identical topic and difficulty will
    still have different assessment IDs.
    
    Returns:
        str: Unique assessment ID in format 'assess_<timestamp>_<uuid>'
    """
    timestamp = int(time.time() * 1000000)  # Microsecond precision
    unique_id = uuid.uuid4().hex[:12]  # 12 character random hex string (ensures uniqueness)
    return f"assess_{timestamp}_{unique_id}"


def assign_unique_ids_to_assessments(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Assign unique assessment IDs to all assessments in the plan.
    Ensures each assessment has a unique ID even if topic names, difficulty,
    or any other attributes are the same. Two assessments with identical
    topic and difficulty will still receive different assessment IDs.
    
    Preserves existing IDs if they already exist.
    
    Args:
        plan (List[Dict[str, Any]]): Assessment plan list
        
    Returns:
        List[Dict[str, Any]]: Assessment plan with unique IDs assigned
    """
    if not isinstance(plan, list):
        return plan
    
    # Track IDs to ensure uniqueness within the plan
    used_ids = set()
    
    for item in plan:
        # Handle both flat items and items with "assessment" wrapper
        assessment = item.get("assessment", item)
        
        # Check if either ID field exists
        has_id = "id" in assessment
        has_assessment_id = "assessment_id" in assessment
        
        # If neither ID exists, generate a new unique ID
        if not has_id and not has_assessment_id:
            new_id = generate_unique_assessment_id()
            # Ensure the generated ID is unique within this plan (safeguard)
            while new_id in used_ids:
                new_id = generate_unique_assessment_id()
            used_ids.add(new_id)
            assessment["id"] = new_id
            assessment["assessment_id"] = new_id  # Set both for compatibility
        # If only one exists, sync them
        elif has_id and not has_assessment_id:
            existing_id = assessment["id"]
            # If ID already used, generate a new one
            if existing_id in used_ids:
                new_id = generate_unique_assessment_id()
                while new_id in used_ids:
                    new_id = generate_unique_assessment_id()
                used_ids.add(new_id)
                assessment["id"] = new_id
                assessment["assessment_id"] = new_id
            else:
                used_ids.add(existing_id)
                assessment["assessment_id"] = existing_id
        elif has_assessment_id and not has_id:
            existing_id = assessment["assessment_id"]
            # If ID already used, generate a new one
            if existing_id in used_ids:
                new_id = generate_unique_assessment_id()
                while new_id in used_ids:
                    new_id = generate_unique_assessment_id()
                used_ids.add(new_id)
                assessment["id"] = new_id
                assessment["assessment_id"] = new_id
            else:
                used_ids.add(existing_id)
                assessment["id"] = existing_id
        # If both exist but differ, keep the "id" field and sync "assessment_id" to it
        elif assessment.get("id") != assessment.get("assessment_id"):
            existing_id = assessment["id"]
            # If ID already used, generate a new one
            if existing_id in used_ids:
                new_id = generate_unique_assessment_id()
                while new_id in used_ids:
                    new_id = generate_unique_assessment_id()
                used_ids.add(new_id)
                assessment["id"] = new_id
                assessment["assessment_id"] = new_id
            else:
                used_ids.add(existing_id)
                assessment["assessment_id"] = existing_id
        else:
            # Both IDs exist and are the same - just track it
            existing_id = assessment.get("id") or assessment.get("assessment_id")
            if existing_id:
                used_ids.add(existing_id)
        
        # Update the item structure
        if "assessment" in item:
            item["assessment"] = assessment
        else:
            item.update(assessment)
    
    return plan


def _compute_time_from_num_questions(num_questions: Dict[str, int]) -> int:
    """Compute assessment_time_minutes from question counts (same heuristic as normalization)."""
    base = (
        num_questions.get("mcq", 0) * 1 +
        num_questions.get("short", 0) * 5 +
        num_questions.get("long", 0) * 10 +
        num_questions.get("coding", 0) * 20
    )
    return max(10, min(60, base))


async def create_plan_from_spec(
    topic: str,
    num_questions: Dict[str, int],
    difficulty: str,
    time_limit_minutes: Optional[int] = None,
    uid: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create an assessment plan from a user-specified spec (topic, question counts, difficulty, optional time limit).
    Same plan structure as the rest of the recommender. Optionally merge and persist for uid.

    Args:
        topic: Assessment topic (e.g. "Python", "React").
        num_questions: Dict with keys mcq, short, long, coding (counts per type).
        difficulty: "easy" | "medium" | "hard" | "expert".
        time_limit_minutes: Optional total time limit in minutes (clamped to 10–60). If None, computed from num_questions.
        uid: If provided, merge plan into this user's assessments_doc and persist.
        session_id: Optional session ID (for storage key).

    Returns:
        Dict with success, assessment_plan, and optional assessment_needs (when uid provided).
    """
    try:
        num_q = {
            "mcq": max(0, int(num_questions.get("mcq", 0))),
            "short": max(0, int(num_questions.get("short", 0))),
            "long": max(0, int(num_questions.get("long", 0))),
            "coding": max(0, int(num_questions.get("coding", 0))),
        }
        # If all zeros, default to a minimal plan
        if sum(num_q.values()) == 0:
            num_q = {"mcq": 4, "short": 1, "long": 0, "coding": 0}

        topic_str = str(topic).strip() or "Custom Assessment"
        diff_lower = str(difficulty).lower().strip()
        diff_map = {"easy": "easy", "medium": "medium", "hard": "hard", "expert": "expert",
                    "beginner": "easy", "intermediate": "medium", "advanced": "hard", "outstanding": "expert"}
        difficulty_normalized = diff_map.get(diff_lower, "medium")

        if time_limit_minutes is not None:
            time_minutes = max(10, min(60, int(time_limit_minutes)))
        else:
            time_minutes = _compute_time_from_num_questions(num_q)

        assessment_id = generate_unique_assessment_id()
        plan_item = {
            "type": "multi",
            "topic": topic_str,
            "difficulty": difficulty_normalized,
            "num_questions": num_q,
            "assessment_time_minutes": time_minutes,
            "rationale": f"Custom assessment for {topic_str} (specified by user).",
            "source": "custom_from_spec",
            "status": "pending",
            "assessment_id": assessment_id,
            "completed_at": None,
            "score": None,
        }
        plan = [plan_item]
        plan = assign_unique_ids_to_assessments(plan)

        if uid:
            assessments_doc = await run_blocking_io(get_assessments_doc, uid) or {}
            existing_data = assessments_doc.get("assessment_recommender", {})
            existing_plan = list(existing_data.get("assessment_plan", []) or [])
            existing_topics = {str(i.get("topic", "")).lower().strip() for i in existing_plan if isinstance(i, dict) and i.get("topic")}
            if topic_str.lower().strip() not in existing_topics:
                existing_plan.extend(plan)
            existing_plan = assign_unique_ids_to_assessments(existing_plan)
            merged_needs = dict(existing_data.get("assessment_needs", {}) or {})
            merged_needs.setdefault("custom_topics", []).append(topic_str)
            assessments_doc["assessment_recommender"] = {
                **existing_data,
                "assessment_plan": existing_plan,
                "assessment_needs": merged_needs,
                "updated_at": time.time(),
            }
            await run_blocking_io(upsert_assessments_doc, uid, assessments_doc, metadata={"agent": "assessment_recommender", "uid": uid})
            log.info("create_plan_from_spec: merged and saved plan for uid=%s topic=%s", uid, topic_str)

        return {
            "success": True,
            "assessment_plan": plan,
            "assessment_needs": {"custom_topics": [topic_str]},
        }
    except Exception as e:
        log.error("create_plan_from_spec failed: %s", e, exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "assessment_plan": [],
        }


class UnifiedAssessmentRecommender:
    """
    Generates personalized assessment recommendations based on resume analysis and user context.
    
    This agent analyzes structured resume data and user interaction history to determine 
    appropriate assessment types and generates adaptive assessment plans. The system learns
    from user's past performance and interaction patterns to provide increasingly personalized
    recommendations.
    
    Key Features:
    - Adaptive difficulty based on past assessment performance
    - User interaction pattern analysis for personalization
    - Dynamic assessment type selection based on user context
    - Cross-profession skill mapping without hardcoded assumptions
    - Performance-driven recommendation adjustments
    
    Follows agentic AI principles:
    - User-centric personalization across all professions
    - Cache isolation to prevent cross-user contamination  
    - Profession-agnostic design for universal applicability
    - Structured logging with user context metadata
    """
    
    def __init__(self):
        """Initialize the assessment recommender with caching."""
        self.config = config
        self.cache = AssessmentRecommenderCache(max_size=500, ttl_seconds=3600)
        
    async def process(self, state: Dict[str, Any], user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process the workflow state and generate personalized assessment recommendations.
        
        Args:
            state (Dict[str, Any]): Current workflow state containing resume and user data
            user_context (Optional[Dict[str, Any]]): User personalization context
            
        Returns:
            Dict[str, Any]: Dictionary containing assessment_needs and assessment_plan
            
        Raises:
            InvalidResumeError: When resume data is missing or invalid
        """
        # Type safety check
        if not isinstance(state, dict):
            raise TypeError(f"State must be a dictionary, got {type(state)}: {repr(state)[:100]}")
        
        # Extract user context from state if not provided
        if not user_context:
            user_context = self._extract_user_context_from_state(state)
            
        # Granular timing breakdown for performance monitoring
        import time
        timing_breakdown = {
            "input_optimization": 0.0,
            "assessment_needs": 0.0,
            "assessment_plan": 0.0,
            "result_processing": 0.0,
            "total": 0.0
        }
        process_start = time.time()
        
        if log.isEnabledFor(logging.INFO):
            log.info(
                "Starting personalized assessment recommendation process", 
                extra={
                    "user_id": (user_context.get("user_id", "anonymous")[:8] + "...") if user_context.get("user_id") else "anonymous",
                    "session_id": user_context.get("session_id"),
                    "profession_type": user_context.get("profession_type", "unknown")
                }
            )
        
        # Use selective data passing for optimization
        opt_start = time.time()
        if "structured_resume" in state:
            # Create optimized state with only the data this agent needs
            optimized_state = create_agent_state(state, "assessment_recommender")
            if log.isEnabledFor(logging.INFO):
                log.info(f"📊 Assessment Recommender using selective data passing - filtered {len(optimized_state.get('structured_resume', {}))} fields")
            
            # Use the optimized state for processing
            resume = optimized_state.get("structured_resume")
        else:
            resume = state.get("structured_resume")
        timing_breakdown["input_optimization"] = time.time() - opt_start
            
        # Apply token optimizations - check cache first
        cache_input = {
            "structured_resume": resume,
            "user_context": user_context,
            "assessment_results": state.get("assessment_results")
        }
        cached_result = get_cached_response("assessment_recommender", cache_input)
        if cached_result:
            if log.isEnabledFor(logging.INFO):
                log.info(f"🎯 Cache hit for assessment_recommender - returning cached result")
            return cached_result
        
        # In evaluation flow we no longer fetch resume; proceed to re-run seeding/minimal context
        
        # Evaluation workflow: seed from prior assessment_recommender output if available
        # This runs when assessment_results exist, indicating a re-run after assessment evaluation
        if state.get("assessment_results"):
            if log.isEnabledFor(logging.INFO):
                log.info(f"🔍 ASSESSMENT_RECOMMENDER: Assessment results found in state: {state.get('assessment_results', {}).get('total_score', 'N/A')}")
                log.info(f"✅ Assessment recommender received assessment results: total_score={state.get('assessment_results', {}).get('total_score', 'N/A')}")
            
            uid = state.get("uid")
            seeded = False
            if uid:
                try:
                    # Prefer per-UID assessments doc (uid_assessments)
                    # Run blocking I/O in thread pool to avoid blocking event loop
                    try:
                        prior_doc = await run_blocking_io(get_assessments_doc, uid)
                    except Exception:
                        prior_doc = None
                    prior_ar = (prior_doc or {}).get("assessment_recommender", {})
                    if isinstance(prior_ar, dict):
                        if prior_ar.get("assessment_plan") or prior_ar.get("assessment_needs"):
                            if log.isEnabledFor(logging.DEBUG):
                                log.debug(f"✅ ASSESSMENT_RECOMMENDER: Seeding from prior assessment plan")
                            # Attach prior plan/needs to state for refinement
                            state["prior_assessment_plan"] = prior_ar.get("assessment_plan", [])
                            state["prior_assessment_needs"] = prior_ar.get("assessment_needs", {})
                            # Also load assessment_history if present
                            if "assessment_history" in (prior_doc or {}):
                                state["assessment_history"] = prior_doc.get("assessment_history", [])
                            seeded = True
                except Exception as e:
                    log.warning(f"⚠️ ASSESSMENT_RECOMMENDER: Could not seed from previous output: {e}")
            
            # Ensure structured_resume is available for re-run
            if not resume or (isinstance(resume, dict) and "error" in resume):
                log.warning(f"⚠️ ASSESSMENT_RECOMMENDER: No structured_resume in state, creating minimal resume")
                minimal_resume = {
                    "Name": "Assessment Candidate",
                    "Skills": [],
                    "WorkExperience": [],
                    "Education": [],
                    "assessment_context": True,
                    "assessment_score": state.get("assessment_results", {}).get("total_score", 0),
                    "assessment_topic": state.get("assessment_topic", "General Assessment"),
                    "user_interests": state.get("user_interests", [])
                }
                resume = minimal_resume
                state["structured_resume"] = minimal_resume
        
        # Evaluation flow: short-circuit to plan refinement without strict resume validation
        # ADAPTIVE RE-RUN ENHANCEMENTS: Idempotency, history tracking, difficulty adjustment, topic expansion
        if state.get("assessment_results"):
            # Extract core assessment fields early for idempotency and decision logic
            results = state.get("assessment_results", {})
            score = results.get("total_score", 0)
            topic = state.get("assessment_topic") or results.get("assessment_topic") or "General Assessment"
            # Ensure topic is never None or empty
            if not topic or not isinstance(topic, str):
                topic = "General Assessment"
            section_scores = results.get("section_scores", {})

            # Idempotency guard: prevent duplicate processing on retries
            assessment_id = state.get("assessment_results", {}).get("assessment_id")
            if not assessment_id:
                assessment_id = f"adhoc-{topic}-{score}-{int(time.time())}"
            processed_map = state.setdefault("assessment_recommender_processed_map", {})
            # Idempotency skip-return marker: if processed_map.get(assessment_id): return { ... }
            # Idempotency skip-return: if this assessment_id was already processed, skip.
            if processed_map.get(assessment_id):
                log.info("Skipping duplicate recommender run for %s", assessment_id)
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": "already_processed",
                    "assessment_id": assessment_id,
                }
            processed_map[assessment_id] = True
            state["assessment_recommender_processed_map"] = processed_map

            log.info(f"🔍 ASSESSMENT_RECOMMENDER: Updating assessment plan based on results (adaptive re-run)")
            log.info(f"✅ Assessment recommender updating plan based on score: {score}")
            
            # Extract or create personalization context for evaluation flow
            if not user_context:
                user_context = self._extract_user_context_from_state(state)
            # Create personalization context for evaluation flow
            eval_personalization_ctx = self._analyze_user_context(resume if resume else {}, user_context)
            
            # ===== LOAD PRIOR STRUCTURES WITH DEEP COPIES =====
            prior_plan = deep_copy(state.get("prior_assessment_plan", []))
            prior_needs = dict(state.get("prior_assessment_needs", {}))
            assessment_history = list(state.get("assessment_history", []))
            
            # Load from uid_assessments if not in state
            uid = state.get("uid")
            if uid and not assessment_history:
                try:
                    prior_doc = await run_blocking_io(get_assessments_doc, uid) or {}
                    assessment_history = list(prior_doc.get("assessment_history", []))
                except Exception as e:
                    log.warning(f"⚠️ Could not load assessment_history from uid_assessments: {e}")
            
            # ===== COMPUTE CONSECUTIVE PASSES/FAILS =====
            consecutive_passes = 0
            consecutive_fails = 0
            # Walk tail of history (last 5 items) ordered by time
            # Support both old format (total_score at top level) and new format (result.score)
            def get_score_from_history_item(h):
                """Extract score from history item, supporting both old and new formats."""
                if not isinstance(h, dict):
                    return None
                # New format: score is in result.score
                if "result" in h and isinstance(h["result"], dict) and "score" in h["result"]:
                    return h["result"]["score"]
                # Old format: total_score at top level
                if "total_score" in h:
                    return h["total_score"]
                return None
            
            recent_history = sorted(
                [h for h in assessment_history if get_score_from_history_item(h) is not None],
                key=lambda x: x.get("timestamp", 0),
                reverse=True
            )[:5]
            
            for hist_item in recent_history:
                hist_score = get_score_from_history_item(hist_item) or 0
                if hist_score >= PASS_THRESHOLD:
                    if consecutive_fails > 0:
                        break  # Run broken
                    consecutive_passes += 1
                else:
                    if consecutive_passes > 0:
                        break  # Run broken
                    consecutive_fails += 1
            
            # Store counters in user_context and state
            if not user_context:
                user_context = {}
            user_context["consecutive_passes"] = consecutive_passes
            user_context["consecutive_fails"] = consecutive_fails
            state["consecutive_passes"] = consecutive_passes
            state["consecutive_fails"] = consecutive_fails
            
            log.info(f"📊 Consecutive passes: {consecutive_passes}, fails: {consecutive_fails}")
            
            # ===== DIFFICULTY ADJUSTMENT =====
            # Base difficulty from score
            if score >= 90:
                base_difficulty = "expert"
                performance_level = "Outstanding"
            elif score >= 80:
                base_difficulty = "hard"
                performance_level = "Excellent"
            elif score >= 70:
                base_difficulty = "medium"
                performance_level = "Good"
            elif score >= 50:
                base_difficulty = "easy"
                performance_level = "Fair"
            else:
                base_difficulty = "easy"
                performance_level = "Needs Improvement"

            # Apply promotion/demotion rules (immediate rule only)
            difficulty_before = base_difficulty
            # Immediate difficulty update based ONLY on current score
            if score >= PASS_THRESHOLD:
                promoted = True
                decision_reason = "promoted_due_to_pass"
                base_difficulty = promote_difficulty(base_difficulty)
            else:
                promoted = False
                decision_reason = "demoted_due_to_fail"
                base_difficulty = demote_difficulty(base_difficulty)
            
            # Capitalize for display (matching existing format)
            difficulty_display = base_difficulty.capitalize()
            if base_difficulty == "expert":
                difficulty_display = "Expert"
            
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"📊 ASSESSMENT_RECOMMENDER: Score {score} -> Difficulty: {difficulty_display} (adjusted from {difficulty_before}), Performance: {performance_level}")
            
            # ===== NORMALIZE PRIOR PLAN =====
            # Strip heavy assessment_result payloads and normalize
            normalized_prior_plan = []
            for item in (prior_plan if isinstance(prior_plan, list) else []):
                if isinstance(item, dict):
                    # Remove heavy payloads
                    clean_item = {k: v for k, v in item.items() if k != "assessment_result"}
                    # Normalize
                    normalized_item = normalize_topic_config(clean_item, self)
                    if normalized_item:
                        normalized_prior_plan.append(normalized_item)
            
            # Ensure unique IDs
            normalized_prior_plan = assign_unique_ids_to_assessments(normalized_prior_plan)
            
            # ===== CREATE PRIMARY RE-RUN ITEM =====
            topic_lower = str(topic or "").lower()
            is_generic_mcq_only = any(k in topic_lower for k in ["communication test", "personality test", "psychometric test"]) or (
                any(k in topic_lower for k in ["communication", "personality", "psychometric"]) and "test" in topic_lower
            )

            if is_generic_mcq_only:
                base_num = 8 if score >= 80 else 5 if score >= 60 else 3
                num_questions = {"mcq": base_num, "short": 0, "long": 0, "coding": 0}
            else:
                is_coding_practical = self._is_coding_practical_for_skill(topic)
                coding_ok = bool(prior_needs.get("coding_test", False))
                
                if score >= 80:
                    coding_count = 2 if (coding_ok and is_coding_practical) else 0
                    num_questions = {"mcq": 8, "short": 3, "long": 2, "coding": coding_count}
                elif score >= 60:
                    coding_count = 1 if (coding_ok and is_coding_practical) else 0
                    num_questions = {"mcq": 5, "short": 2, "long": 1, "coding": coding_count}
                else:
                    coding_count = 1 if (coding_ok and is_coding_practical) else 0
                    num_questions = {"mcq": 3, "short": 1, "long": 1, "coding": coding_count}

            # Create new assessment item with full result (for history); plan will reference via assessment_id_ref
            raw_timestamp = state.get("timestamp") or results.get("evaluated_at") or time.time()
            # Normalize evaluated_at to ISO timestamp
            if isinstance(raw_timestamp, (int, float)):
                evaluated_at_iso = datetime.utcfromtimestamp(raw_timestamp).isoformat()
            else:
                # Assume it's already a string/ISO-like
                evaluated_at_iso = str(raw_timestamp)
            timestamp = raw_timestamp
            new_assessment = {
                "type": "multi",
                "topic": topic,
                "difficulty": base_difficulty,
                "num_questions": num_questions,
                "assessment_time_minutes": self.calculate_assessment_time({
                    "type": "multi",
                    "topic": topic,
                    "difficulty": base_difficulty,
                    "num_questions": num_questions
                }),
                "assessment_result": {
                    "score": score,
                    "performance_level": performance_level,
                    "section_scores": section_scores,
                    "evaluated_at": evaluated_at_iso
                },
                "id": generate_unique_assessment_id(),
                "assessment_id": None  # Will be set to same as id below
            }
            new_assessment["assessment_id"] = new_assessment["id"]

            # ===== SPLIT HISTORY FROM PLAN =====
            # Build compact history item
            # Include total_score at top level for backward compatibility with filtering logic
            history_item = {
                "assessment_id": new_assessment["assessment_id"],
                "topic": new_assessment["topic"],
                "timestamp": timestamp,
                "result": new_assessment["assessment_result"],
                "total_score": score,  # Add top-level total_score for backward compatibility
                "difficulty_after": base_difficulty,
                "difficulty_before": difficulty_before,
                "source": "evaluation"
            }

            assessment_history = state.get("assessment_history", [])
            if not any(
                isinstance(h, dict) and h.get("assessment_id") == history_item["assessment_id"]
                for h in assessment_history
            ):
                assessment_history.append(history_item)
            state["assessment_history"] = assessment_history

            # Build plan item with compact assessment_result & reference
            plan_item = dict(new_assessment)
            plan_item["assessment_id_ref"] = history_item["assessment_id"]
            # Ensure plan item has a lightweight assessment_result with raw timestamp
            plan_item["assessment_result"] = {
                "score": score,
                "performance_level": performance_level,
                "section_scores": section_scores,
                "evaluated_at": timestamp,
            }

            # Normalize plan item
            plan_item = normalize_topic_config(plan_item, self)
            
            # ===== TOPIC EXPANSION =====
            expansion_topics: List[Dict[str, Any]] = []

            passed = score >= PASS_THRESHOLD

            if passed:
                if promoted:
                    # Use LLM for forward/advanced topic generation (maintains quality)
                    try:
                        prompt = f"""Generate 2-3 advanced or related assessment topics for a candidate who scored {score}% in "{topic}".

Context:
- Skill Level: {eval_personalization_ctx.skill_level}
- Experience: {eval_personalization_ctx.experience_years or 'Unknown'} years

Rules:
- Suggest topics that are naturally advanced or related to "{topic}"
- Each topic should be a logical next step or advanced concept
- Keep topics profession-agnostic and universally applicable

Return ONLY a JSON array:
[{{"topic": "<topic_name>", "difficulty": "hard", "num_questions": {{"mcq": 6, "short": 2, "long": 1, "coding": 0}}}}]"""

                        # OPTIMIZATION: Use faster model for topic expansion
                        faster_model = "gemini-2.5-flash" if "flash" in self.config.llm_model.lower() else self.config.llm_model
                        llm_content = await safe_llm_generate(
                            prompt,
                            model_name=faster_model,
                            timeout_s=20.0,  # Reduced timeout for faster processing
                            recommender_instance=self,
                        )

                        if llm_content:
                            try:
                                json_obj = self._extract_json(llm_content)
                                if isinstance(json_obj, list):
                                    for item in json_obj:
                                        if isinstance(item, dict) and item.get("topic"):
                                            try:
                                                tc = TopicConfig(**item)
                                                normalized = normalize_topic_config(tc.model_dump(), self)
                                                normalized["source"] = "llm_forward"
                                                expansion_topics.append(normalized)
                                            except ValidationError:
                                                continue
                            except Exception as e:
                                if log.isEnabledFor(logging.WARNING):
                                    log.warning(f"Failed to parse LLM response for forward topics: {e}")
                    except Exception as e:
                        if log.isEnabledFor(logging.WARNING):
                            log.warning(f"LLM generation for forward topics failed: {e}")

                    # Fallback to static reinforcement topics if LLM fails
                    if not expansion_topics:
                        reinforcement = get_reinforcement_topics(topic)
                        if reinforcement:
                            expansion_topics.extend([normalize_topic_config(t, self) for t in reinforcement])
                        else:
                            # Final fallback to advanced topics map
                            try:
                                from core.config import get_advanced_topics_map

                                related_topics_map = get_advanced_topics_map()
                                if topic:
                                    related_topics = related_topics_map.get(str(topic).lower())
                                    if related_topics:
                                        for adv in related_topics:
                                            normalized = normalize_topic_config(adv, self)
                                            normalized["source"] = "static_advanced"
                                            expansion_topics.append(normalized)
                            except Exception:
                                pass
                else:
                    # PASSED but NOT promoted ⇒ lateral / reinforcement topics
                    # Use LLM for lateral topic generation (maintains quality)
                    try:
                        # OPTIMIZATION: Use faster model for lateral topic expansion
                        faster_model = "gemini-2.5-flash" if "flash" in self.config.llm_model.lower() else self.config.llm_model
                        llm_lateral = await safe_llm_generate(
                            lateral_prompt_for_topic(topic, score, eval_personalization_ctx),
                            model_name=faster_model,
                            timeout_s=15.0,  # Reduced timeout for faster processing
                            recommender_instance=self
                        )
                        if llm_lateral:
                            try:
                                parsed = self._extract_json(llm_lateral)
                                if isinstance(parsed, list):
                                    for item in parsed:
                                        if isinstance(item, dict) and item.get("topic"):
                                            try:
                                                tc = TopicConfig(**item)
                                                normalized = normalize_topic_config(tc.model_dump(), self)
                                                normalized["source"] = "llm_lateral"
                                                expansion_topics.append(normalized)
                                            except ValidationError:
                                                continue
                            except Exception as e:
                                if log.isEnabledFor(logging.WARNING):
                                    log.warning(f"Failed to parse LLM lateral response for {topic}: {e}")
                    except Exception:
                        if log.isEnabledFor(logging.WARNING):
                            log.warning("LLM lateral-generation failed for %s", topic)

                    # Fallback to static mappings if LLM fails
                    if not expansion_topics:
                        expansion_topics.extend(
                            [normalize_topic_config(t, self) for t in get_reinforcement_topics(topic)]
                        )
                        expansion_topics.extend(
                            [normalize_topic_config(t, self) for t in get_lateral_topics(topic)]
                        )
            else:
                # FAIL branch ⇒ remedial + foundational topics
                expansion_topics.extend(
                    [normalize_topic_config(t, self) for t in get_remedial_topics(topic)]
                )
                expansion_topics.extend(
                    [normalize_topic_config(t, self) for t in get_foundational_topics(topic)]
                )
            
            # Assign IDs to expansion topics
            for exp_topic in expansion_topics:
                exp_topic["id"] = generate_unique_assessment_id()
                exp_topic["assessment_id"] = exp_topic["id"]
            
            # ===== DEDUPLICATION AND CAPPING =====
            merged_plan = list(normalized_prior_plan)
            existing_keys = {
                (str(p.get("topic", "")).lower().strip(), str(p.get("difficulty", "")).lower().strip())
                for p in merged_plan
            }
            existing_ids = {
                p.get("assessment_id_ref")
                for p in merged_plan
                if p.get("assessment_id_ref") is not None
            }

            # Append primary plan item (idempotent)
            primary_key = (plan_item["topic"].lower().strip(), plan_item["difficulty"].lower().strip())
            if primary_key not in existing_keys and plan_item["assessment_id_ref"] not in existing_ids:
                merged_plan.append(plan_item)
                existing_keys.add(primary_key)
                existing_ids.add(plan_item["assessment_id_ref"])

            # Append expansion topics (idempotent)
            for t in expansion_topics:
                topic_key = str(t.get("topic", "")).lower().strip()
                diff_key = str(t.get("difficulty", "")).lower().strip()
                k = (topic_key, diff_key)
                assessment_id_t = t.get("assessment_id")
                if k in existing_keys or (assessment_id_t is not None and assessment_id_t in existing_ids):
                    continue
                merged_plan.append(t)
                existing_keys.add(k)
                if assessment_id_t is not None:
                    existing_ids.add(assessment_id_t)

            # Apply cap - keep the most recent MAX_PLAN_LENGTH entries
            if len(merged_plan) > MAX_PLAN_LENGTH:
                merged_plan = merged_plan[-MAX_PLAN_LENGTH:]

            merged_plan = assign_unique_ids_to_assessments(merged_plan)
            
            # ===== MERGE ASSESSMENT NEEDS =====
            merged_needs = dict(prior_needs) if isinstance(prior_needs, dict) else {}
            mcq_s = section_scores.get("mcq", 0)
            short_s = section_scores.get("short_answer", 0)
            long_s = section_scores.get("long_answer", 0)
            
            merged_needs.setdefault("psychometric_test", False)
            merged_needs.setdefault("communication_test", False)
            merged_needs.setdefault("coding_test", False)
            merged_needs.setdefault("personality_test", False)
            
            if mcq_s < 50:
                merged_needs["psychometric_test"] = True
            if short_s < 50:
                merged_needs["communication_test"] = True
            if long_s < 50:
                merged_needs["coding_test"] = True
                
            merged_needs["reasoning"] = f"Updated based on {performance_level} performance ({score}%) in {topic}"
            merged_needs["performance_level"] = performance_level
            merged_needs["assessment_score"] = score

            log.info(
                "AssessmentRecommender outcome: promoted=%s passed=%s score=%s topic=%s",
                promoted,
                passed,
                score,
                topic,
            )
            log.info("Expansion topics: %s", [t.get("topic") for t in expansion_topics])
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"📊 ASSESSMENT_RECOMMENDER: Final plan has {len(merged_plan)} assessments (expansion: {len(expansion_topics)})")
            
            # ===== TRANSACTIONAL PERSISTENCE =====
            personalization_score = 0.8 if score >= 80 else 0.5

            # Update user_context with latest decision metadata
            if not user_context:
                user_context = {}
            user_context.update({
                "consecutive_passes": consecutive_passes,
                "consecutive_fails": consecutive_fails,
                "last_assessment_score": score,
                "last_assessment_topic": topic,
                "performance_level": performance_level,
            })
            user_context["decision_reason"] = decision_reason

            # Prepare final state update
            final_state_update = {
                "assessment_needs": merged_needs,
                "assessment_plan": merged_plan,
                "assessment_history": assessment_history,
                "consecutive_passes": consecutive_passes,
                "consecutive_fails": consecutive_fails,
                "user_context": user_context,
            }
            
            # Update state atomically
            state.update(final_state_update)
            
            # Persist to uid_assessments with versioned compare-and-set semantics
            # Run blocking I/O in thread pool to avoid blocking event loop during concurrent requests
            if uid:
                try:
                    doc = await run_blocking_io(get_assessments_doc, uid) or {}
                    current_version = doc.get("version", 0)
                    new_version = current_version + 1
                    doc["version"] = new_version
                    doc["assessment_recommender"] = {
                        "assessment_plan": merged_plan,
                        "assessment_needs": merged_needs,
                        "personalization_score": personalization_score,
                        "updated_at": timestamp,
                        "decision_reason": decision_reason,
                    }
                    doc["assessment_history"] = assessment_history

                    await run_blocking_io(
                        upsert_assessments_doc,
                        uid,
                        doc,
                        metadata={
                            "service": "adaptive_re_run"
                        }
                    )
                    log.info(f"✅ Persisted assessment_history and plan to uid_assessments for {uid}")
                except Exception as e:
                    log.warning(f"⚠️ Failed to persist to uid_assessments: {e}")

            return {
                "ok": True,
                "assessment_needs": merged_needs,
                "assessment_plan": merged_plan,
                "personalization_score": personalization_score,
                "user_context": user_context
            }

        self._validate_resume(resume)
        
        # Analyze user context for personalization BEFORE cache lookup
        personalization_ctx = self._analyze_user_context(resume, user_context)
        
        # Get user_interest_answers from state (2nd call provides these)
        # If not present, try to extract from user_interests
        user_interest_answers = state.get("user_interest_answers", [])
        if not user_interest_answers:
            # Try to extract from user_interests if available
            user_interests = state.get("user_interests", [])
            if user_interests and isinstance(user_interests, list):
                user_interest_answers = []
                for item in user_interests:
                    if isinstance(item, dict):
                        if "question" in item and "answer" in item:
                            user_interest_answers.append({
                                "question": item.get("question", ""),
                                "answer": item.get("answer", "")
                            })
                        elif "answer" in item:
                            user_interest_answers.append({
                                "question": item.get("question", "User interest question"),
                                "answer": item.get("answer", "")
                            })
        
        # Log if 2nd call detected
        is_second_call = state.get("is_second_call", False)
        if is_second_call and user_interest_answers:
            if log.isEnabledFor(logging.INFO):
                log.info(f"✅ 2ND CALL: Using user_interest_answers from 2nd call ({len(user_interest_answers)} items) as context for assessment recommendations")
        
        # Extract skill gap analysis from career advisor (if available) for gap-based assessments
        raw_skill_gap = state.get("raw_skill_gap_analysis_output") or state.get("career_advisor", {}).get("raw_skill_gap_analysis_output", {})
        skill_gap_analysis: Dict[str, Any] = {}
        if isinstance(raw_skill_gap, str):
            try:
                skill_gap_analysis = json.loads(raw_skill_gap) if raw_skill_gap else {}
            except (json.JSONDecodeError, TypeError):
                skill_gap_analysis = {}
        elif isinstance(raw_skill_gap, dict):
            skill_gap_analysis = raw_skill_gap
        if skill_gap_analysis and log.isEnabledFor(logging.INFO):
            missing = get_missing_skills_flat(skill_gap_analysis)[:5]
            if missing:
                log.info(f"📊 ASSESSMENT_RECOMMENDER: Using skill gaps for recommendations: {missing}")
        
        # Generate cache key including all factors that affect recommendations
        cache_key = self._generate_request_cache_key(resume, user_interest_answers, personalization_ctx, skill_gap_analysis)
        
        # Try cache first
        cached_result = self.cache.get(cache_key, user_context)
        if cached_result:
            if log.isEnabledFor(logging.INFO):
                log.info("Returning cached assessment recommendations", extra={
                "user_id": user_context.get("user_id", "anonymous")[:8] + "...",
                "cache_hit": True
            })
            return cached_result
        
        try:
            # Get UID from state for resume summary access
            uid = state.get("uid")
            
            # Use pre-computed personalization context
            needs_start = time.time()
            assessment_needs = await self._determine_assessment_needs(resume, personalization_ctx, uid=uid)
            timing_breakdown["assessment_needs"] = time.time() - needs_start
            
            plan_start = time.time()
            assessment_plan = await self._generate_dynamic_assessment_plan(
                resume=resume,
                assessment_needs=assessment_needs,
                user_interest_answers=user_interest_answers,
                personalization_ctx=personalization_ctx,
                uid=uid,
                skill_gap_analysis=skill_gap_analysis
            )
            timing_breakdown["assessment_plan"] = time.time() - plan_start

            result_start = time.time()
            result = {
                "ok": True,
                "assessment_needs": assessment_needs,
                "assessment_plan": assessment_plan,
                "personalization_score": personalization_ctx.personalization_score,
                "user_context": {
                    "profession_type": personalization_ctx.profession_type,
                    "skill_level": personalization_ctx.skill_level,
                    "experience_years": personalization_ctx.experience_years
                }
            }
            
            # Cache the result with user isolation
            self.cache.set(cache_key, result, user_context)
            timing_breakdown["result_processing"] = time.time() - result_start
            timing_breakdown["total"] = time.time() - process_start
            
            log.info("Successfully created personalized assessment plan", extra={
                "user_id": (user_context.get("user_id", "anonymous")[:8] + "...") if user_context.get("user_id") else "anonymous",
                "assessment_count": len(assessment_plan),
                "personalization_score": personalization_ctx.personalization_score,
                "profession_type": personalization_ctx.profession_type
            })
            
            # Log timing breakdown
            log.info(
                f"⏱️ ASSESSMENT_RECOMMENDER timing breakdown: "
                f"input_opt={timing_breakdown['input_optimization']*1000:.1f}ms, "
                f"needs={timing_breakdown['assessment_needs']*1000:.1f}ms, "
                f"plan={timing_breakdown['assessment_plan']*1000:.1f}ms, "
                f"result_proc={timing_breakdown['result_processing']*1000:.1f}ms, "
                f"total={timing_breakdown['total']*1000:.1f}ms"
            )
            
            # Session Management Integration - MOVED TO BACKGROUND (non-blocking)
            if uid:
                # Fire-and-forget: Don't block response on session updates
                async def _update_session_background():
                    try:
                        session_start = time.time()
                        log.info(f"🔄 ASSESSMENT_RECOMMENDER: Starting background session storage for UID={uid}")
                        
                        # Get or reuse existing session (ensures UID always uses same session ID)
                        existing_session = await run_blocking_io(
                            session_manager.get_or_reuse_session,
                            owner_id=uid,
                            kind="candidate_pipeline",
                            owner_type="candidate",
                            initial_step="assessment_recommender",
                            initial_data={"assessment_needs": assessment_needs, "assessment_plan": assessment_plan}
                        )
                        log.info(f"✅ ASSESSMENT_RECOMMENDER: Using session: {existing_session.session_id} for UID={uid}")
                        
                        # Update session step
                        session_update_result = await run_blocking_io(
                            session_manager.update_step,
                            session_id=existing_session.session_id,
                            step="assessment_recommender",
                            data={"assessment_count": len(assessment_plan), "personalization_score": personalization_ctx.personalization_score},
                            progress=0.8  # 80% complete after assessment recommender
                        )
                        
                        # Store assessment recommender data in existing session
                        session_data = await run_blocking_io(get_chat_session, existing_session.session_id)
                        if session_data:
                            session_data["assessment_recommender"] = {
                                "assessment_needs": assessment_needs,
                                "assessment_plan": assessment_plan,
                                "personalization_score": personalization_ctx.personalization_score,
                                "user_context": {
                                    "profession_type": personalization_ctx.profession_type,
                                    "skill_level": personalization_ctx.skill_level,
                                    "experience_years": personalization_ctx.experience_years
                                },
                                "processing_time": timing_breakdown["total"],
                                "timing_breakdown": timing_breakdown,
                                "method": "personalized_recommendation"
                            }
                            
                            await run_blocking_io(
                                update_chat_session,
                                session_id=existing_session.session_id,
                                session_data=session_data,
                                metadata={
                                    "agent": "assessment_recommender",
                                    "uid": uid,
                                    "status": "assessment_recommender_complete",
                                    "method": "personalized_recommendation"
                                }
                            )
                            session_time = time.time() - session_start
                            log.info(f"✅ ASSESSMENT_RECOMMENDER: Background session update completed in {session_time*1000:.1f}ms for UID={uid}")
                        else:
                            log.warning(f"⚠️ ASSESSMENT_RECOMMENDER: No existing chat session data found for session_id={existing_session.session_id}")
                    except Exception as e:
                        log.error(f"❌ ASSESSMENT_RECOMMENDER: Background session update failed for UID={uid}: {e}", exc_info=True)
                
                # Fire-and-forget: Start background task without awaiting
                asyncio.create_task(_update_session_background())
                log.info(f"🚀 ASSESSMENT_RECOMMENDER: Started background session update task for UID={uid} (non-blocking)")
            
            # Cache the result for future use
            cache_response("assessment_recommender", cache_input, result)
            
            return result
            
        except Exception as e:
            log.error("Failed to generate personalized assessment recommendations", extra={
                "user_id": (user_context.get("user_id", "anonymous")[:8] + "...") if user_context.get("user_id") else "anonymous",
                "error": str(e)
            })
            raise AssessmentRecommenderError(f"Assessment generation failed: {str(e)}") from e

    def _validate_resume(self, resume: Dict[str, Any]) -> None:
        """
        Validate resume data for processing.
        
        Args:
            resume (Dict[str, Any]): Structured resume data
            
        Raises:
            InvalidResumeError: When resume is invalid or missing
        """
        if not resume:
            raise InvalidResumeError("Resume data is missing")
            
        if "error" in resume:
            raise InvalidResumeError(f"Resume contains error: {resume.get('error')}")
            
        # Validate resume size to prevent memory issues
        resume_json = json.dumps(resume, separators=(",", ":"))
        if len(resume_json) > self.config.max_resume_length:
            raise InvalidResumeError(
                f"Resume data too large: {len(resume_json)} chars "
                f"(max: {self.config.max_resume_length})"
            )
    
    def _extract_json(self, text: str) -> Any:
        """
        Robust JSON extraction from LLM response with multiple fallback strategies.
        
        Args:
            text (str): Raw LLM response text
            
        Returns:
            Any: Parsed JSON object
            
        Raises:
            ValueError: When no valid JSON is found
        """
        # Try fenced JSON first (case-insensitive, handles whitespace)
        match = re.search(r"```json\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Try arrays first (for plans - greedy to get complete arrays)
        match = re.search(r"(\[.*\])", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Try objects (non-greedy for small objects)
        match = re.search(r"(\{.*?\})", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        raise ValueError("No valid JSON found in LLM output")
    
    async def _llm_with_retry(self, prompt: str, model: str, attempts: int = None) -> Any:
        """
        Invoke LLM with exponential backoff retry logic.
        
        Args:
            prompt (str): Prompt to send to LLM
            model (str): Model name to use
            attempts (int, optional): Number of retry attempts
            
        Returns:
            Any: LLM response
            
        Raises:
            Exception: When all retry attempts fail
        """
        attempts = attempts or self.config.llm_retry_attempts
        delay = self.config.llm_base_backoff
        
        for i in range(attempts):
            try:
                return await asyncio.wait_for(invoke_llm(
            prompt=prompt,
            task_type="assessment_generation",
            agent_name="assessment_recommender",
            preferred_model="gemini-2.5-flash",
            max_output_tokens=2000  # ✅ OPTIMIZATION: Limit output tokens
        ), timeout=25)  # ✅ OPTIMIZATION: Reduced timeout
            except Exception as e:
                if i == attempts - 1:
                    raise
                log.warning("LLM call failed (attempt %d/%d): %s", i + 1, attempts, str(e))
                await asyncio.sleep(delay + random.uniform(0, 0.25))
                delay *= 2
    
    def _extract_user_context_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user context from workflow state with type safety."""
        # Ensure state is a dictionary
        if not isinstance(state, dict):
            log.error(f"State is not a dictionary, got {type(state)}: {repr(state)[:100]}")
            return {
                "user_id": "anonymous",
                "session_id": None,
                "user_profile": {},
                "interaction_history": [],
                "experience_years": None,
                "skill_level": None,
                "profession_type": None
            }
            
        return {
            "user_id": state.get("user_id", "anonymous"),
            "session_id": state.get("session_id"),
            "user_profile": state.get("user_profile", {}),
            "interaction_history": state.get("interaction_history", []),
            "experience_years": state.get("experience_years"),
            "skill_level": state.get("skill_level"),
            "profession_type": state.get("profession_type")
        }
    
    def _analyze_user_context(self, resume: Dict[str, Any], user_context: Dict[str, Any]) -> PersonalizationContext:
        """
        Analyze user context to create comprehensive personalization context.
        
        This method extracts user interaction patterns and assessment history to create
        a personalized context that adapts recommendations based on past performance.
        
        Args:
            resume (Dict[str, Any]): Structured resume data
            user_context (Dict[str, Any]): User context from state or parameters
            
        Returns:
            PersonalizationContext: Comprehensive personalization context with interaction analysis
        """
        # Extract basic user info
        user_id = user_context.get("user_id", "anonymous")
        session_id = user_context.get("session_id")
        
        # Analyze profession type from resume - simplified to use skills directly
        profession_type = "general"  # Default, will be determined by skills in assessment logic
        
        # Extract domain specializations from skills
        domain_specializations = self._extract_domain_specializations_from_skills(resume)
        
        # Estimate experience years from resume
        experience_years = self._estimate_experience_years(resume)
        log.debug(f"🔍 USER_CONTEXT DEBUG: experience_years={experience_years}")
        
        # Extract and analyze assessment history
        assessment_history = user_context.get("interaction_history", [])
        interaction_patterns = self._analyze_interaction_patterns(assessment_history)
        
        # Determine skill level based on experience, resume complexity, and past performance
        skill_level = self._determine_adaptive_skill_level(resume, experience_years, assessment_history)
        
        # Calculate previous performance from interaction history
        previous_performance = self._calculate_previous_performance(assessment_history)
        
        # Calculate personalization score including interaction data
        personalization_score = self._calculate_comprehensive_personalization_score({
            "has_profession": bool(profession_type != "general"),
            "has_specializations": bool(domain_specializations),
            "has_experience": bool(experience_years and experience_years > 0),
            "has_history": bool(previous_performance is not None),
            "has_interaction_patterns": bool(interaction_patterns)
        })
        
        return PersonalizationContext(
            user_id=user_id,
            session_id=session_id,
            profession_type=profession_type,
            skill_level=skill_level,
            experience_years=experience_years,
            domain_specializations=domain_specializations or [],
            previous_performance=previous_performance,
            personalization_score=personalization_score,
            assessment_history=assessment_history,
            interaction_patterns=interaction_patterns
        )
    
    def _parse_experience_string(self, experience_value: Any) -> Optional[int]:
        """Parse experience years from various formats (int, float, or string like '8 years 4 months')."""
        if experience_value is None:
            return None
        
        # If already a number, convert to int
        if isinstance(experience_value, (int, float)):
            return int(experience_value)
        
        # If it's a string, try to parse it
        if isinstance(experience_value, str):
            experience_str = experience_value.strip()
            
            # Try to extract years using regex (handles "8 years 4 months", "8 years", "8", etc.)
            # Pattern matches: "8 years 4 months", "8 years", "8y", "8.5 years", etc.
            match = re.search(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y\b)', experience_str.lower())
            if match:
                years_float = float(match.group(1))
                return int(years_float)
            
            # Fallback: try to extract any number
            match = re.search(r'(\d+(?:\.\d+)?)', experience_str)
            if match:
                years_float = float(match.group(1))
                return int(years_float)
        
        return None
    
    def _estimate_experience_years(self, resume: Dict[str, Any]) -> Optional[int]:
        """Estimate years of experience using centralized calculation."""
        # First try to get pre-calculated value
        if resume.get("total_experience_years"):
            years = self._parse_experience_string(resume["total_experience_years"])
            if years is not None:
                log.debug(f"Using total_experience_years from resume: {years}")
                return years
            else:
                log.warning(f"Could not parse total_experience_years: {resume['total_experience_years']}")
        
        # Fallback to calculating from work experience
        work_history = resume.get("work_experience", [])
        if not work_history:
            # Try alternative field name
            work_history = resume.get("experience", [])
            log.debug(f"No work_experience found, trying experience field: {len(work_history)} items")
        
        if not work_history:
            log.debug(f"No work experience found in resume")
            return None
        
        # Use centralized calculation (shared util)
        from utils.experience_years import calculate_total_experience_years
        years = calculate_total_experience_years(work_history)
        result = int(years) if years > 0 else None
        log.debug(f"Calculated experience from work history: {result} years (from {len(work_history)} jobs)")
        return result
    
    def _analyze_interaction_patterns(self, assessment_history: List[Dict]) -> Dict[str, Any]:
        """
        Analyze user interaction patterns from assessment history.
        
        Args:
            assessment_history (List[Dict]): List of past assessment interactions
            
        Returns:
            Dict[str, Any]: Interaction pattern analysis including preferences and performance trends
        """
        if not assessment_history:
            return {}
        
        patterns = {
            "assessment_count": len(assessment_history),
            "preferred_types": {},
            "difficulty_progression": [],
            "time_patterns": {},
            "performance_trend": "stable"
        }
        
        # Analyze assessment type preferences
        type_counts = {}
        total_time = 0
        scores = []
        
        for assessment in assessment_history:
            # Count assessment types
            assessment_type = assessment.get("type", "unknown")
            type_counts[assessment_type] = type_counts.get(assessment_type, 0) + 1
            
            # Track time spent
            time_spent = assessment.get("time_spent_minutes", 0)
            total_time += time_spent
            
            # Track scores for trend analysis
            score = assessment.get("score")
            if score is not None:
                scores.append(score)
                patterns["difficulty_progression"].append(assessment.get("difficulty", "medium"))
        
        # Calculate preferred types
        if type_counts:
            total_assessments = sum(type_counts.values())
            patterns["preferred_types"] = {
                type_name: count / total_assessments 
                for type_name, count in type_counts.items()
            }
        
        # Calculate average time
        if assessment_history:
            patterns["time_patterns"]["avg_time_per_assessment"] = total_time / len(assessment_history)
        
        # Determine performance trend
        if len(scores) >= 3:
            recent_avg = sum(scores[-3:]) / 3
            early_avg = sum(scores[:3]) / 3
            if recent_avg > early_avg + 0.1:
                patterns["performance_trend"] = "improving"
            elif recent_avg < early_avg - 0.1:
                patterns["performance_trend"] = "declining"
        
        return patterns
    
    def _determine_adaptive_skill_level(self, resume: Dict[str, Any], experience_years: Optional[int], assessment_history: List[Dict]) -> str:
        """
        Determine skill level adaptively based on resume, experience, and past assessment performance.
        
        Args:
            resume (Dict[str, Any]): Structured resume data
            experience_years (Optional[int]): Years of experience
            assessment_history (List[Dict]): Past assessment performance data
            
        Returns:
            str: Adaptive skill level (beginner, intermediate, advanced, expert)
        """
        if not experience_years:
            # If we have assessment history, use performance data
            if assessment_history:
                avg_performance = sum(a.get("score", 0) for a in assessment_history) / len(assessment_history)
                if avg_performance >= 0.85:
                    return "advanced"
                elif avg_performance >= 0.7:
                    return "intermediate"
                else:
                    return "beginner"
            return "beginner"
        
        # Count indicators of skill level from resume
        resume_skills = resume.get("skills", [])
        skills_count = 0
        
        if isinstance(resume_skills, list) and resume_skills:
            # Handle two possible formats:
            # 1. Simple list: ["JavaScript", "React", "HTML"]
            # 2. Structured: [{"Category": "...", "Items": [{"Name": "..."}]}]
            if isinstance(resume_skills[0], str):
                # Simple format - count strings directly
                skills_count = len(resume_skills)
            else:
                # Structured format - extract from nested structure
                skills_count = len([
                    item.get("Name", "")
                    for cat in resume_skills
                    for item in (cat.get("Items", []) if isinstance(cat, dict) else [])
                    if item and isinstance(item, dict) and item.get("Name")
                ])
        
        certifications_count = len(resume.get("certifications", []))
        education_level = len(resume.get("education", []))
        
        # Base complexity score from resume
        complexity_score = skills_count + (certifications_count * 2) + education_level
        
        # Adjust based on assessment history performance
        performance_adjustment = 0
        if assessment_history:
            avg_performance = sum(a.get("score", 0) for a in assessment_history) / len(assessment_history)
            # Strong performance can bump up skill level
            if avg_performance >= 0.9:
                performance_adjustment = 2
            elif avg_performance >= 0.8:
                performance_adjustment = 1
            elif avg_performance < 0.5:
                performance_adjustment = -1
        
        adjusted_complexity = complexity_score + performance_adjustment
        
        # Determine skill level with performance-based adjustments
        if experience_years >= 8 or adjusted_complexity >= 15:
            return "expert"
        elif experience_years >= 4 or adjusted_complexity >= 10:
            return "advanced"
        elif experience_years >= 2 or adjusted_complexity >= 5:
            return "intermediate"
        else:
            return "beginner"
    
    def _determine_skill_level(self, resume: Dict[str, Any], experience_years: Optional[int]) -> str:
        """Legacy method - kept for backwards compatibility. Use _determine_adaptive_skill_level instead."""
        return self._determine_adaptive_skill_level(resume, experience_years, [])
    
    def _calculate_previous_performance(self, interaction_history: List[Dict]) -> Optional[float]:
        """Calculate average performance from interaction history."""
        if not interaction_history:
            return None
        
        scores = []
        for interaction in interaction_history:
            score = interaction.get("score")
            if score is not None and 0 <= score <= 1:
                scores.append(score)
        
        return sum(scores) / len(scores) if scores else None
    
    def _calculate_comprehensive_personalization_score(self, factors: Dict[str, bool]) -> float:
        """
        Calculate comprehensive personalization score including interaction patterns.
        
        Args:
            factors (Dict[str, bool]): Available personalization factors
            
        Returns:
            float: Comprehensive personalization score (0.0-1.0)
        """
        weights = {
            "has_profession": 0.25,
            "has_specializations": 0.2, 
            "has_experience": 0.2,
            "has_history": 0.2,
            "has_interaction_patterns": 0.15
        }
        
        score = sum(weights[factor] for factor, present in factors.items() if present)
        return round(score, 2)
    
    def _generate_request_cache_key(
        self, 
        resume: Dict[str, Any], 
        interest_answers: List[Dict[str, Any]], 
        personalization_ctx: PersonalizationContext,
        skill_gap_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate comprehensive cache key including all factors that affect recommendations."""
        # Use full hashes to avoid collisions in cache keys
        resume_hash = hashlib.sha256(json.dumps(resume, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        interest_hash = hashlib.sha256(json.dumps(interest_answers, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        persona_sig = f"{personalization_ctx.profession_type}:{personalization_ctx.skill_level}:{personalization_ctx.experience_years}:{personalization_ctx.interaction_patterns.get('performance_trend','')}"
        persona_hash = hashlib.sha256(persona_sig.encode()).hexdigest()
        gap_hash = ""
        if skill_gap_analysis and isinstance(skill_gap_analysis, dict):
            gap_sig = json.dumps(get_missing_skills_flat(skill_gap_analysis)[:10], sort_keys=True, separators=(",", ":"))
            gap_hash = ":" + hashlib.sha256(gap_sig.encode()).hexdigest()
        return f"assessment_recommendation:{resume_hash}:{interest_hash}:{persona_hash}{gap_hash}"
    
    def _redact_pii(self, text: str) -> str:
        """
        Redact personally identifiable information from text before sending to LLM.
        
        Args:
            text (str): Text potentially containing PII
            
        Returns:
            str: Text with PII redacted
        """
        # Redact email addresses
        text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text)
        
        # Redact phone numbers (but not dates) - check digit count
        def redact_phone(match):
            # Count only digits in the matched text
            digits = re.sub(r"\D", "", match.group(0))
            # Only redact if it has 10+ digits (typical phone number)
            return "[REDACTED_PHONE]" if len(digits) >= 10 else match.group(0)
        
        text = re.sub(r"\+?[\d\-\s().]{7,}", redact_phone, text)
        
        return text
    
    def _validate_needs(self, needs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize assessment needs from LLM response.
        
        Args:
            needs (Dict[str, Any]): Raw assessment needs from LLM
            
        Returns:
            Dict[str, Any]: Validated and normalized assessment needs
        """
        def to_bool(key: str) -> bool:
            return bool(needs.get(key, False))
        
        return {
            "personality_test": to_bool("personality_test"),
            "coding_test": to_bool("coding_test"),
            "psychometric_test": to_bool("psychometric_test"),
            "communication_test": to_bool("communication_test"),
            "reasoning": re.sub(r'\s+', ' ', str(needs.get("reasoning", "")).strip())[:300]  # Clean whitespace and limit length
        }
    
    async def _get_assessment_needs_prompt(
        self, 
        resume: Dict[str, Any], 
        candidate_type: str, 
        personalization_ctx: PersonalizationContext,
        uid: Optional[str] = None  # NEW: For accessing cached Gemini summary
    ) -> str:
        """
        Generate personalized prompt for determining assessment needs.
        Uses Gemini-generated cached resume summary (NO TRUNCATION) to preserve full context.
        """
        # OPTIMIZATION: Use resume_summary from structured_resume first (no blocking async call)
        resume_summary = resume.get("resume_summary")
        
        # If summary not available, use concise structured format (faster than generating)
        if not resume_summary:
            from agents.prompt_generator import _create_concise_resume_summary_for_career_advisor
            resume_summary = _create_concise_resume_summary_for_career_advisor(resume)
        else:
            # Truncate resume summary to reduce tokens (max 2000 chars)
            if len(resume_summary) > 2000:
                resume_summary = resume_summary[:2000] + "..."
        
        # Compress personalization info
        personalization_info = f"""Profession: {personalization_ctx.profession_type}, Level: {personalization_ctx.skill_level}, Experience: {personalization_ctx.experience_years or 'Unknown'}y"""
            
        return f"""Analyze resume and determine assessment needs. JSON only.

Context: {personalization_info}

Rules:
- SOFTWARE/WEB DEVELOPERS: coding_test = true
- ECE/ELECTRICAL/MECHANICAL ENGINEERS: coding_test = false
- STUDENTS: Base on field of study

Resume Summary:
{resume_summary}

JSON: {{"personality_test": true/false, "coding_test": true/false, "psychometric_test": true/false, "communication_test": true/false, "reasoning": "brief explanation"}}"""
        
    def _get_personalized_fallback_needs(self, candidate_type: str, personalization_ctx: PersonalizationContext, resume: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adaptive fallback assessment needs based on comprehensive user context."""
        
        # Determine if this is a technical candidate based on skills directly
        is_technical = self._is_technical_candidate_from_skills(resume)
        
        # Base logic - comprehensive assessment for technical candidates
        if is_technical:
            needs = {
                "personality_test": True,  # Technical candidates need soft skills assessment
                "psychometric_test": True,  # Technical candidates need cognitive assessment
                "communication_test": True,  # Technical candidates need communication assessment
                "coding_test": True,  # Priority for technical candidates
                "technical_assessment": True,
                "problem_solving": True
            }
        else:
            # Extract skill-based assessments for non-technical candidates
            skill_assessments = self._extract_skill_based_assessments(resume)
            
            needs = {
                "personality_test": True,
                "psychometric_test": candidate_type == "fresher",
                "communication_test": candidate_type == "fresher",
                "coding_test": False,
                "technical_assessment": False,
                "skill_based_assessments": skill_assessments  # Add skill-based assessments
            }
        
        # Refined coding test logic - only for software development roles
        software_coding_professions = {"technology", "software_engineering", "data_science", "ai_ml"}
        hardware_engineering_professions = {"electronics_engineering", "electrical_engineering"}
        
        has_software_coding_background = personalization_ctx.profession_type in software_coding_professions
        has_hardware_engineering_background = personalization_ctx.profession_type in hardware_engineering_professions
        
        # Consider past performance for coding tests
        if personalization_ctx.previous_performance is not None:
            if personalization_ctx.previous_performance >= 0.7 and has_software_coding_background:
                needs["coding_test"] = True
            elif personalization_ctx.previous_performance < 0.5:
                # Focus on fundamentals if performance is poor
                if not is_technical:
                    needs["psychometric_test"] = True
                needs["coding_test"] = False
        else:
            # Default logic for new users - be more selective about coding tests
            if has_software_coding_background and personalization_ctx.skill_level in {"intermediate", "advanced", "expert"}:
                needs["coding_test"] = True
            elif has_hardware_engineering_background:
                # ECE/EE students get technical assessments but not necessarily coding
                needs["coding_test"] = False
                if not is_technical:
                    needs["psychometric_test"] = True  # Technical reasoning is important
            else:
                needs["coding_test"] = False
        
        # Adjust based on interaction patterns
        if personalization_ctx.interaction_patterns:
            performance_trend = personalization_ctx.interaction_patterns.get("performance_trend", "stable")
            if performance_trend == "declining":
                # Focus on foundational skills
                if not is_technical:
                    needs["psychometric_test"] = True
                    needs["communication_test"] = True
            elif performance_trend == "improving":
                # User is learning well, can handle more challenging assessments
                if has_software_coding_background:
                    needs["coding_test"] = True
        
        needs["reasoning"] = (
            f"Comprehensive assessment for {'technical' if is_technical else 'non-technical'} candidate: "
            f"technical skills detected: {is_technical}, "
            f"skill level ({personalization_ctx.skill_level}), "
            f"experience years: {personalization_ctx.experience_years}, "
            f"skill-based assessments: {len(needs.get('skill_based_assessments', []))}"
        )
        
        # Include candidate_type in needs for downstream processing
        needs["candidate_type"] = candidate_type
        
        return needs

    def _is_technical_candidate_from_skills(self, resume: Dict[str, Any]) -> bool:
        """
        Determine if candidate is technical based on their skills directly.
        
        Args:
            resume: Structured resume data containing skills
            
        Returns:
            bool: True if candidate has technical skills, False otherwise
        """
        # Get skills from the resume data
        skills = self._extract_skills(resume)
        
        # Technical skill keywords that indicate technical candidacy
        technical_keywords = {
            # Programming languages
            'python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 'kotlin', 'swift', 'php', 'ruby', 'scala',
            # Web technologies
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'laravel', 'rails',
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible',
            # Data & AI
            'machine learning', 'ai', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn',
            # Mobile
            'android', 'ios', 'react native', 'flutter', 'xamarin',
            # Other technical terms
            'api', 'rest', 'graphql', 'microservices', 'agile', 'scrum', 'git', 'linux', 'unix',
            'software development', 'programming', 'coding', 'development', 'engineering',
            'frontend', 'backend', 'full stack', 'devops', 'data science', 'cybersecurity'
        }
        
        # Check if any skills match technical keywords
        skills_lower = [skill.lower() for skill in skills]
        technical_matches = sum(1 for skill in skills_lower if any(keyword in skill for keyword in technical_keywords))
        
        # Consider technical if:
        # 1. Has 2+ technical skill matches, OR
        # 2. Has 1+ technical skill match AND 3+ total skills AND the match is strong (not just partial)
        is_technical = technical_matches >= 2 or (
            technical_matches >= 1 and len(skills) >= 3 and 
            any(skill.lower() in technical_keywords for skill in skills_lower)
        )
        
        log.debug(f"Technical skill analysis: {technical_matches} matches out of {len(skills)} skills, is_technical={is_technical}")
        
        return is_technical

    def _extract_domain_specializations_from_skills(self, resume: Dict[str, Any]) -> List[str]:
        """
        Extract domain specializations from skills in the resume.
        
        Args:
            resume: Structured resume data containing skills
            
        Returns:
            List[str]: List of domain specializations
        """
        skills = self._extract_skills(resume)
        specializations = []
        
        # Domain mapping based on skills
        domain_mapping = {
            'frontend': ['react', 'angular', 'vue', 'javascript', 'html', 'css', 'frontend'],
            'backend': ['node.js', 'python', 'java', 'spring', 'django', 'flask', 'backend'],
            'fullstack': ['full stack', 'fullstack', 'mern', 'mean'],
            'devops': ['docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'devops'],
            'cloud': ['aws', 'azure', 'gcp', 'cloud'],
            'data': ['python', 'pandas', 'numpy', 'machine learning', 'ai', 'data science'],
            'mobile': ['android', 'ios', 'react native', 'flutter', 'mobile'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'database']
        }
        
        skills_lower = [skill.lower() for skill in skills]
        
        for domain, keywords in domain_mapping.items():
            if any(keyword in skill for skill in skills_lower for keyword in keywords):
                specializations.append(domain)
        
        return specializations

    def _extract_skill_based_assessments(self, resume: Dict[str, Any]) -> List[str]:
        """
        Extract top skills from non-technical candidates to create skill-based assessments.
        
        Args:
            resume: Structured resume data containing skills
            
        Returns:
            List[str]: List of skill-based assessment topics (top 3-5 skills)
        """
        skills = self._extract_skills(resume)
        
        # Filter out very generic skills
        generic_skills = {'communication', 'teamwork', 'leadership', 'time management', 'problem solving'}
        
        # Get skills that are not too generic
        specific_skills = [
            skill for skill in skills 
            if skill.lower() not in generic_skills
        ]
        
        # Return top 3-5 skills as assessment topics
        return specific_skills[:5] if len(specific_skills) >= 3 else specific_skills[:3]

    def _get_difficulty_by_experience(self, candidate_type: str, experience_years: int) -> str:
        """
        Determine assessment difficulty based on experience level.
        
        Args:
            candidate_type: Type of candidate (fresher, experienced, senior, etc.)
            experience_years: Years of experience
            
        Returns:
            str: Difficulty level (Easy, Medium, Hard)
        """
        if candidate_type == "fresher" or experience_years < 2:
            difficulty = "Easy"
        elif candidate_type == "experienced" or experience_years < 5:
            difficulty = "Medium"
        else:  # senior or experience_years >= 5
            difficulty = "Hard"
        
        log.debug(f"Determined difficulty '{difficulty}' for {candidate_type} candidate with {experience_years} years experience")
        return difficulty

    async def _determine_assessment_needs(
        self, 
        resume: Dict[str, Any], 
        personalization_ctx: PersonalizationContext,
        uid: Optional[str] = None  # NEW: For accessing cached Gemini summary
    ) -> Dict[str, Any]:
        """
        Determine what types of assessments are needed based on resume analysis and user context.
        Uses Gemini-generated cached resume summary (NO TRUNCATION) to preserve full context.
        
        Args:
            resume (Dict[str, Any]): Structured resume data
            personalization_ctx (PersonalizationContext): User personalization context
            uid (Optional[str]): User ID for accessing cached Gemini summary
            
        Returns:
            Dict[str, Any]: Assessment needs with boolean flags for each test type
            
        Raises:
            LLMResponseError: When LLM response cannot be parsed
        """
        candidate_type = resume.get("candidate_type", "").lower()
        
        # If candidate_type is not in resume, determine it from experience years
        if not candidate_type:
            experience_years = personalization_ctx.experience_years or self._estimate_experience_years(resume) or 0
            log.debug(f"🔍 CANDIDATE_TYPE DEBUG: personalization_ctx.experience_years={personalization_ctx.experience_years}, estimated={self._estimate_experience_years(resume)}, final={experience_years}")
            
            if experience_years < 2:
                candidate_type = "fresher"
            elif experience_years < 5:
                candidate_type = "experienced"
            else:
                candidate_type = "senior"
            log.debug(f"🔍 CANDIDATE_TYPE DEBUG: Determined candidate_type from experience: {candidate_type} (experience_years: {experience_years})")
        else:
            log.debug(f"🔍 CANDIDATE_TYPE DEBUG: Using existing candidate_type from resume: {candidate_type}")
        
        log.debug("Determining personalized assessment needs", extra={
            "candidate_type": candidate_type,
            "profession_type": personalization_ctx.profession_type,
            "skill_level": personalization_ctx.skill_level,
            "experience_years": personalization_ctx.experience_years
        })
        
        # Use LLM for assessment needs (maintains quality)
        prompt = await self._get_assessment_needs_prompt(resume, candidate_type, personalization_ctx, uid=uid)

        try:
            # OPTIMIZATION: Use flash model for simpler assessment needs call
            faster_model = "gemini-2.5-flash" if "flash" in self.config.llm_model.lower() else self.config.llm_model
            llm_response = await self._llm_with_retry(prompt, model=faster_model)
            content = getattr(llm_response, "content", str(llm_response))
            result = self._extract_json(content)
            result = self._validate_needs(result)  # Validate and normalize
                
            log.debug("Successfully determined personalized assessment needs", extra={
                "candidate_type": candidate_type,
                "coding_test": result.get("coding_test", False),
                "profession_type": personalization_ctx.profession_type,
                "personalization_score": personalization_ctx.personalization_score
            })
            return result

        except (json.JSONDecodeError, ValueError) as e:
            log.warning("LLM returned invalid JSON for assessment needs, using personalized fallback", extra={
                "error": str(e),
                "profession_type": personalization_ctx.profession_type
            })
            
            # Enhanced fallback logic with personalization
            fallback_result = self._get_personalized_fallback_needs(candidate_type, personalization_ctx, resume)
            
            if log.isEnabledFor(logging.INFO):
                log.info("Using personalized fallback assessment needs", extra={
                    "candidate_type": candidate_type,
                    "profession_type": personalization_ctx.profession_type,
                    "fallback_reason": str(e)
                })
            return fallback_result

        except Exception as e:
            if log.isEnabledFor(logging.ERROR):
                log.error("Failed to determine assessment needs", extra={
                    "error": str(e),
                    "profession_type": personalization_ctx.profession_type
                })
            raise LLMResponseError(f"Could not determine assessment needs: {str(e)}") from e

    def _extract_skills(self, resume: Dict[str, Any]) -> List[str]:
        """
        Extract skills from resume data with support for multiple formats.
        
        Args:
            resume (Dict[str, Any]): Structured resume data
            
        Returns:
            List[str]: List of unique skill names limited by MAX_SKILLS_TO_ASSESS
        """
        resume_skills = resume.get("skills", [])
        
        if not isinstance(resume_skills, list):
            return []
        
        raw_skills = []
        
        if resume_skills and isinstance(resume_skills[0], str):
            # Simple format: ["JavaScript", "React", "HTML"]
            raw_skills = resume_skills
        else:
            # Structured format: [{"Category": "...", "Items": [{"Name": "..."}]}]
            raw_skills = [
                (item or {}).get("Name")
                for cat in resume_skills
                for item in (cat.get("Items", []) if isinstance(cat, dict) else [])
                if isinstance(cat, dict)
            ]
        
        # Deduplicate while preserving order and respecting limit
        seen = set()
        skills = []
        max_skills_to_assess = getattr(self.config, 'max_skills_count', 50)
        for skill in raw_skills:
            if skill:
                # Clean and normalize skill name
                normalized_skill = skill.strip()
                # Remove emoji/symbols but keep essential characters for tech skills
                normalized_skill = re.sub(r"[^\w\s\+\-#\.\(\)\/]", "", normalized_skill)
                normalized_skill = re.sub(r"\s+", " ", normalized_skill).strip()
                
                skill_key = normalized_skill.lower()  # Normalize for deduplication
                if skill_key and len(skill_key) > 1 and skill_key not in seen:
                    seen.add(skill_key)
                    skills.append(normalized_skill)
                    if len(skills) >= max_skills_to_assess:
                        break
        
        log.debug("Extracted %d unique skills (limit: %d)", 
                    len(skills), max_skills_to_assess)
        return skills
    
    def _is_coding_practical_for_skill(self, skill_or_topic: str) -> bool:
        """
        Check if a skill/topic is practical for coding questions.
        
        Args:
            skill_or_topic (str): The skill or topic name to check
            
        Returns:
            bool: True if coding questions are practical for this skill, False otherwise
        """
        if not skill_or_topic:
            return False
        
        topic_lower = str(skill_or_topic).lower()
        
        # Check for specific non-coding patterns FIRST (before checking programming keywords)
        # "Prophet & AXIS" or "Prophet AXIS" refers to insurance software, not Python library
        if "prophet" in topic_lower and "axis" in topic_lower:
            return False
        # IFRS is compliance, not programming
        if "ifrs" in topic_lower:
            return False
        # Stochastic modeling without programming language context is theoretical, not coding
        if "stochastic" in topic_lower and "modeling" in topic_lower:
            # Only allow if it mentions a programming language
            if not any(kw in topic_lower for kw in ["python", "r", "matlab", "code", "programming"]):
                return False
        
        # Programming languages
        programming_languages = [
            "python", "java", "javascript", "typescript", "c++", "c#", "c ", "go", "rust",
            "php", "ruby", "swift", "kotlin", "scala", "r", "matlab", "perl", "lua",
            "dart", "haskell", "clojure", "elixir", "erlang", "f#", "objective-c",
            "vba", "visual basic", "vb.net", "vbscript"
        ]
        
        # Programming frameworks and libraries
        programming_frameworks = [
            "react", "angular", "vue", "node.js", "django", "flask", "express",
            "spring", "laravel", "rails", "asp.net", ".net", "fastapi", "nest.js",
            "next.js", "nuxt.js", "svelte", "ember", "meteor", "gatsby",
            "prophet"  # Facebook Prophet (Python time series forecasting library)
        ]
        
        # Database and data technologies that can have coding assessments
        database_coding = [
            "sql", "postgresql", "mysql", "mongodb", "redis", "cassandra",
            "elasticsearch", "dynamodb", "oracle", "sqlite", "neo4j"
        ]
        
        # Algorithm and data structure topics
        algorithm_topics = [
            "algorithm", "data structure", "data structures", "dsa",
            "problem solving", "competitive programming", "leetcode"
        ]
        
        # Development and engineering practices
        dev_engineering = [
            "software development", "programming", "coding", "development",
            "software engineering", "web development", "app development",
            "backend development", "frontend development", "full stack",
            "mobile development", "game development", "devops", "ci/cd",
            "api development", "microservices", "system design"
        ]
        
        # Check if topic matches any programming-related keywords
        all_keywords = programming_languages + programming_frameworks + database_coding + algorithm_topics + dev_engineering
        
        for keyword in all_keywords:
            if keyword in topic_lower:
                return True
        
        # Exclude non-programming skills that shouldn't have coding questions
        non_coding_keywords = [
            "communication", "leadership", "management", "project management",
            "marketing", "sales", "design", "ui/ux", "user experience",
            "writing", "content", "accounting", "finance", "hr", "human resources",
            "psychology", "counseling", "nursing", "healthcare", "medicine",
            "teaching", "education", "training", "coaching",
            # Compliance and regulatory
            "compliance", "ifrs", "gaap", "sox", "regulatory", "audit",
            # Insurance and actuarial (unless specifically about coding)
            "actuarial", "actuary", "insurance", "underwriting", "claims",
            "axis",  # AXIS insurance software (Prophet & AXIS handled separately)
            "risk modeling",  # Risk modeling (stochastic modeling handled separately)
            # Business and domain-specific
            "business analysis", "business intelligence", "reporting",
            "data analysis"  # Note: data analysis is ambiguous - check if it's about tools
        ]
        
        for keyword in non_coding_keywords:
            if keyword in topic_lower:
                # Special case: "data analysis" might be Python/R based, so check context
                if keyword == "data analysis":
                    # Check if it's combined with programming indicators
                    if any(kw in topic_lower for kw in ["python", "r", "sql", "pandas", "numpy"]):
                        continue  # Skip this exclusion, allow coding
                return False
        
        return False
    
    async def _get_assessment_plan_prompt(
        self, 
        resume: Dict[str, Any], 
        skills_text: str, 
        interest_text: str, 
        assessment_needs: Dict[str, Any],
        personalization_ctx: PersonalizationContext,
        uid: Optional[str] = None,  # NEW: For accessing cached Gemini summary
        skill_gap_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate personalized prompt for creating dynamic assessment plan.
        Uses Gemini-generated cached resume summary (NO TRUNCATION) to preserve full context.
        Includes skill gaps so 1-2 assessments can address identified gaps.
        """
        coding_enabled = assessment_needs.get("coding_test", False)
        
        # OPTIMIZATION: Use resume_summary from structured_resume first (no blocking async call)
        resume_summary = resume.get("resume_summary")
        
        # If summary not available, use concise structured format (faster than generating)
        if not resume_summary:
            from agents.prompt_generator import _create_concise_resume_summary_for_career_advisor
            resume_summary = _create_concise_resume_summary_for_career_advisor(resume)
        else:
            # Truncate resume summary to reduce tokens (max 3000 chars for plan generation)
            if len(resume_summary) > 3000:
                resume_summary = resume_summary[:3000] + "..."
        
        # Compress personalization info
        personalization_info = f"""Profession: {personalization_ctx.profession_type}, Level: {personalization_ctx.skill_level}, Experience: {personalization_ctx.experience_years or 'Unknown'}y"""
        
        # Build skill gaps section when career advisor identified gaps (5 topics from gaps)
        skill_gaps_section = ""
        has_gaps = False
        if skill_gap_analysis and isinstance(skill_gap_analysis, dict):
            missing_skills = get_missing_skills_flat(skill_gap_analysis) or skill_gap_analysis.get("recommended_focus", [])
            if missing_skills:
                gaps_list = missing_skills[:10] if isinstance(missing_skills, list) else []
                if gaps_list:
                    has_gaps = True
                    skill_gaps_section = f"""
Skill gaps to address (select 5 as assessment topics from these): {", ".join(str(s) for s in gaps_list)}
"""
        
        topics_requirement = (
            "- Select 5 topics from Skills and 5 topics from Skill gaps below (10 total). Set difficulty: <2y=Easy, 2-5y=Medium, >5y=Hard."
            if has_gaps
            else "- Select 5 topics from Skills. Set difficulty: <2y=Easy, 2-5y=Medium, >5y=Hard."
        )
        
        return f"""Generate assessment plan. JSON only.

Context: {personalization_info}

Requirements:
{topics_requirement}
- Each item: mcq: 5, short: 2, long: 1, coding: {1 if coding_enabled else 0} (only if programming-related)
- Time: Easy 15-25min, Medium 20-35min, Hard 30-45min
- For each item include "rationale": one short sentence (for the candidate) explaining why this assessment was recommended—e.g. for skill gaps say why it helps close the gap; for existing skills say how it showcases their experience. Keep it motivating and specific to the candidate.

Format:
[{{"type": "multi", "topic": "<skill>", "difficulty": "Easy|Medium|Hard", "num_questions": {{"mcq": 5, "short": 2, "long": 1, "coding": {1 if coding_enabled else 0}}}, "assessment_time_minutes": <time>, "rationale": "<one short sentence for the candidate>"}}]

Resume Summary:
{resume_summary}

Skills: {skills_text[:500]}  # Reduced from 1000
Interests: {interest_text[:100]}  # Reduced from 150
Needs: {json.dumps(assessment_needs, separators=(',', ':'))}
{skill_gaps_section}
Return JSON array."""

    def _enforce_policy_on_plan(self, plan: List[Dict[str, Any]], needs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enforce coding test policy and clean up assessment structure.
        
        Args:
            plan (List[Dict[str, Any]]): Raw assessment plan from LLM
            needs (Dict[str, Any]): Assessment needs determining policies
            
        Returns:
            List[Dict[str, Any]]: Policy-enforced assessment plan
        """
        coding_ok = bool(needs.get("coding_test", False))
        final_plan = []

        for item in plan or []:
            # Handle both flat items and items with "assessment" wrapper
            assessment = item.get("assessment", item)
            
            if assessment.get("type") == "multi":
                # Enforce coding policy in question distribution
                num_questions = dict(assessment.get("num_questions") or {})
                topic = assessment.get("topic") or ""
                # Ensure topic is never None
                if not topic or not isinstance(topic, str):
                    topic = ""
                
                # Check if coding questions are practical for this specific skill/topic
                is_coding_practical = self._is_coding_practical_for_skill(topic)
                
                # Only include coding questions if:
                # 1. Coding is enabled in assessment_needs (coding_ok)
                # 2. The skill/topic is practical for coding questions (is_coding_practical)
                num_questions["coding"] = 1 if (coding_ok and is_coding_practical) else 0
                
                # Log if coding was removed due to impracticality
                if coding_ok and not is_coding_practical and num_questions.get("coding", 0) > 0:
                    log.debug(f"Removed coding questions from '{topic}' - not practical for coding assessments")
                
                # Enforce MCQ-only for generic tests (communication/personality/psychometric)
                topic_lower = str(topic).lower() if topic else ""
                is_generic_test = any(k in topic_lower for k in ["communication test", "personality test", "psychometric test"]) or (
                    any(k in topic_lower for k in ["communication", "personality", "psychometric"]) and "test" in topic_lower
                )
                
                if is_generic_test:
                    num_questions.update({"short": 0, "long": 0, "coding": 0})
                    # Ensure at least some MCQ questions for generic tests
                    if num_questions.get("mcq", 0) == 0:
                        num_questions["mcq"] = 5
                else:
                    # Ensure minimum question counts for non-generic assessments
                    # If all question types are 0, set reasonable defaults
                    total_questions = sum(num_questions.values())
                    if total_questions == 0:
                        # Set default minimum questions
                        num_questions["mcq"] = 5
                        num_questions["short"] = 2
                        num_questions["long"] = 1
                        # Only add coding if practical
                        if coding_ok and is_coding_practical:
                            num_questions["coding"] = 1
                    else:
                        # Ensure at least MCQ questions exist
                        if num_questions.get("mcq", 0) == 0 and num_questions.get("short", 0) == 0 and num_questions.get("long", 0) == 0:
                            # If only coding questions exist, add at least MCQ
                            num_questions["mcq"] = 5
                
                assessment["num_questions"] = num_questions
                
                # Calculate and validate assessment time
                difficulty = assessment.get("difficulty", "Medium")
                
                # Calculate reference time for validation (agentic approach: trust LLM but validate)
                calculated_time = self.calculate_assessment_time(assessment)
                
                # Get LLM's decision
                llm_time = assessment.get("assessment_time_minutes", 0)
                
                # Agentic validation: Trust LLM's judgment but ensure it's within reasonable bounds
                # Only override if LLM time is clearly unreasonable (too high, too low, or missing)
                if llm_time == 0:
                    # LLM didn't provide time - use calculated
                    assessment["assessment_time_minutes"] = calculated_time
                    log.debug(f"LLM didn't provide time for '{topic}', using calculated: {calculated_time} minutes")
                elif llm_time < 10:
                    # Too short - use calculated or minimum
                    assessment["assessment_time_minutes"] = max(10, calculated_time)
                    log.debug(f"LLM time too short ({llm_time} min) for '{topic}', adjusted to: {assessment['assessment_time_minutes']} minutes")
                elif llm_time > 60:
                    # Too long - cap at maximum
                    assessment["assessment_time_minutes"] = 60
                    log.debug(f"LLM time too long ({llm_time} min) for '{topic}', capped at: 60 minutes")
                elif abs(llm_time - calculated_time) > 30:
                    # Very different from calculated - use calculated as it's likely more accurate
                    assessment["assessment_time_minutes"] = calculated_time
                    log.debug(f"LLM time ({llm_time} min) very different from calculated ({calculated_time} min) for '{topic}', using calculated")
                else:
                    # LLM's time is reasonable - trust it (agentic approach)
                    log.debug(f"Using LLM's assessment time for '{topic}': {llm_time} minutes (calculated reference: {calculated_time} minutes)")
                
                # Ensure no timer_per_question leaked into multi-type
                assessment.pop("timer_per_question", None)
                
                # Update the item structure
                if "assessment" in item:
                    item["assessment"] = assessment
                else:
                    item.update(assessment)
            
            final_plan.append(item)
        
        log.debug("Enforced coding policy: coding_enabled=%s", coding_ok)
        return final_plan
    
    def _skill_priority_lookup(self, skill_gap_analysis: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """Build skill (lowercase) -> 'critical'|'high'|'medium' from missing_skills_by_priority or missing_skills dict."""
        if not skill_gap_analysis or not isinstance(skill_gap_analysis, dict):
            return {}
        by_priority = skill_gap_analysis.get("missing_skills_by_priority")
        if not by_priority and isinstance(skill_gap_analysis.get("missing_skills"), dict):
            by_priority = skill_gap_analysis.get("missing_skills")
        if not by_priority:
            return {}
        lookup: Dict[str, str] = {}
        for level in ("critical", "high", "medium"):
            skills = by_priority.get(level) or []
            for s in skills:
                if s:
                    lookup[str(s).strip().lower()] = level
        return lookup

    def _build_skill_proficiency_lookup(self, resume: Optional[Dict[str, Any]]) -> Dict[str, str]:
        """
        Build a lookup of skill name -> proficiency score (e.g., '7/10') from structured resume skills.
        Uses skills produced by skill_proficiency_analyzer (SkillName + Proficiency).
        """
        lookup: Dict[str, str] = {}
        if not isinstance(resume, dict):
            return lookup
        resume_skills = resume.get("skills", [])
        if not isinstance(resume_skills, list):
            return lookup
        for skill in resume_skills:
            if not isinstance(skill, dict):
                continue
            name = (
                skill.get("SkillName")
                or skill.get("Name")
                or skill.get("skill")
                or skill.get("name")
            )
            proficiency = skill.get("Proficiency") or skill.get("proficiency")
            if not name or not proficiency:
                continue
            # Normalize name for matching (case-insensitive, whitespace-normalized)
            norm_name = re.sub(r"\s+", " ", str(name)).strip().lower()
            if norm_name:
                lookup[norm_name] = str(proficiency)
        return lookup

    def _set_recommendation_source_on_plan(
        self,
        plan: List[Dict[str, Any]],
        skill_gap_analysis: Optional[Dict[str, Any]] = None,
        resume: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set rationale and recommendation_source on each plan item:
        - For skill gaps: skill-gap "why" (with priority when available) and recommendation_source
          indicating it was recommended to address an identified gap.
        - For existing skills: existing-skills "why" and recommendation_source indicating it was
          recommended to validate/showcase existing skills.
        
        Also sets current_score:
        - For skill gaps/missing skills: \"Take this assessment\"
        - For existing skills: existing proficiency score from resume skills when available, otherwise \"Take this assessment\".
        In-place.
        """
        SKILL_GAP_SENTENCE = "Recommended to address an identified skill gap in your profile."
        EXISTING_SKILLS_SENTENCE = "Recommended to validate and showcase your existing skills."
        # Priority order for rationale phrasing (highest first)
        PRIORITY_ORDER = ("critical", "high", "medium")
        GAP_PHRASE_BY_PRIORITY = {
            "critical": "a critical gap for your target role",
            "high": "a high-priority gap for your target role",
            "medium": "a medium-priority gap to round out your profile",
        }
        gap_skills_lower: List[str] = []
        gap_skills_display: List[str] = []
        priority_lookup: Dict[str, str] = {}
        if skill_gap_analysis and isinstance(skill_gap_analysis, dict):
            missing = get_missing_skills_flat(skill_gap_analysis) or skill_gap_analysis.get("recommended_focus")
            if isinstance(missing, list):
                gap_skills_lower = [str(g).strip().lower() for g in missing[:10] if g]
                gap_skills_display = [str(g).strip() for g in missing[:10] if g]
            priority_lookup = self._skill_priority_lookup(skill_gap_analysis)

        # Build proficiency lookup once from resume for existing-skill assessments
        proficiency_lookup = self._build_skill_proficiency_lookup(resume)

        for item in plan or []:
            if not isinstance(item, dict):
                continue

            topic_raw = str(item.get("topic") or "").strip()
            topic = topic_raw.lower()

            # Keep LLM-generated rationale when present and non-empty
            existing_rationale = (item.get("rationale") or "").strip()
            has_llm_rationale = bool(existing_rationale)

            # Default current_score when we cannot determine proficiency
            default_current_score = "Take this assessment"

            if not gap_skills_lower:
                # No explicit gaps identified → treat as existing-skill validation
                if not has_llm_rationale:
                    item["rationale"] = (
                        f"Recommended to validate and showcase your experience in {topic_raw}."
                        if topic_raw
                        else EXISTING_SKILLS_SENTENCE
                    )
                # Mark recommendation source for existing skills
                item["recommendation_source"] = EXISTING_SKILLS_SENTENCE
                # Set current_score from proficiency when available
                if topic:
                    norm_topic = re.sub(r"\s+", " ", topic).strip().lower()
                    prof = proficiency_lookup.get(norm_topic)
                else:
                    prof = None
                item["current_score"] = prof if prof else default_current_score
                continue

            matched_display = [
                d for d, g in zip(gap_skills_display, gap_skills_lower)
                if (g in topic or topic in g)
            ]
            is_gap = len(matched_display) > 0

            if is_gap:
                # Gap-based recommendation
                if not has_llm_rationale:
                    skills_phrase = ", ".join(matched_display)
                    # Use highest priority among matched skills for more motivating rationale
                    best_priority = None
                    for d in matched_display:
                        p = priority_lookup.get(d.lower())
                        if p and (
                            best_priority is None
                            or PRIORITY_ORDER.index(p) < PRIORITY_ORDER.index(best_priority)
                        ):
                            best_priority = p
                    gap_phrase = (
                        GAP_PHRASE_BY_PRIORITY.get(
                            best_priority, "identified as a gap for your target role"
                        )
                        if best_priority
                        else "identified as a gap for your target role"
                    )
                    item["rationale"] = (
                        f"Recommended to strengthen {skills_phrase}—{gap_phrase}."
                    )
                # Mark recommendation source for skill gaps
                item["recommendation_source"] = SKILL_GAP_SENTENCE
                # For skill gaps, always show a call-to-action style score
                item["current_score"] = default_current_score
            else:
                # Existing-skill recommendation
                if not has_llm_rationale:
                    item["rationale"] = (
                        f"Recommended to validate and showcase your experience in {topic_raw}."
                        if topic_raw
                        else EXISTING_SKILLS_SENTENCE
                    )
                # Mark recommendation source for existing skills
                item["recommendation_source"] = EXISTING_SKILLS_SENTENCE
                if topic:
                    norm_topic = re.sub(r"\s+", " ", topic).strip().lower()
                    prof = proficiency_lookup.get(norm_topic)
                else:
                    prof = None
                item["current_score"] = prof if prof else default_current_score
    
    def _sanitize_plan_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Sanitize and validate individual assessment plan item.
        
        Args:
            item (Dict[str, Any]): Individual assessment item from plan
            
        Returns:
            Optional[Dict[str, Any]]: Sanitized item or None if invalid
        """
        assessment = item.get("assessment", item)
        assessment_type = assessment.get("type")
        topic = assessment.get("topic")
        
        # Validate required fields
        if assessment_type not in {"multi", "mcq"} or not topic:
            return None
        
        # Sanitize multi-type assessments
        if assessment_type == "multi":
            num_questions = assessment.get("num_questions") or {}
            if isinstance(num_questions, int):
                num_questions = {"mcq": num_questions}
            sanitized_questions = {}
            for k in ("mcq", "short", "long", "coding"):
                try:
                    value = num_questions.get(k, 0)
                    sanitized_questions[k] = int(max(0, float(value)))
                except (ValueError, TypeError):
                    sanitized_questions[k] = 0
            assessment["num_questions"] = sanitized_questions
            
            # Remove timer_per_question from multi-type assessments
            assessment.pop("timer_per_question", None)
            
            # Ensure assessment_time_minutes is reasonable (realistic bounds)
            try:
                time_minutes = float(assessment.get("assessment_time_minutes", getattr(self.config, 'default_assessment_time_minutes', 30)))
                assessment["assessment_time_minutes"] = max(10, min(60, int(time_minutes)))  # 10-60 minutes realistic range
            except (ValueError, TypeError):
                assessment["assessment_time_minutes"] = getattr(self.config, 'default_assessment_time_minutes', 30)  # More realistic default
        
        # Sanitize MCQ assessments
        elif assessment_type == "mcq":
            try:
                num_questions = int(assessment.get("num_questions", getattr(self.config, 'default_mcq_questions', 10)))
                assessment["num_questions"] = max(1, min(20, num_questions))
            except (ValueError, TypeError):
                assessment["num_questions"] = getattr(self.config, 'default_mcq_questions', 10)
            
            try:
                timer_per_question = int(assessment.get("timer_per_question", getattr(self.config, 'default_mcq_timer_seconds', 60)))
                assessment["timer_per_question"] = max(15, min(300, timer_per_question))
            except (ValueError, TypeError):
                assessment["timer_per_question"] = getattr(self.config, 'default_mcq_timer_seconds', 60)
        
        # Update the item structure
        if "assessment" in item:
            item["assessment"] = assessment
        else:
            item.update(assessment)
        
        return item

    def _append_required_mcqs(self, plan: List[Dict[str, Any]], needs: Dict[str, Any], personalization_ctx: PersonalizationContext) -> List[Dict[str, Any]]:
        """
        Append mandatory MCQ tests if missing from the plan, prioritizing skill-based assessments.
        
        Args:
            plan (List[Dict[str, Any]]): Current assessment plan
            needs (Dict[str, Any]): Assessment needs requiring specific tests
            personalization_ctx (PersonalizationContext): Personalization context for difficulty calculation
            
        Returns:
            List[Dict[str, Any]]: Plan with prioritized assessments
        """
        # Extract candidate_type from needs for difficulty calculation
        candidate_type = needs.get("candidate_type", "experienced").lower()
        
        # Extract existing topics for case-insensitive deduplication
        existing_topics = set()
        for item in plan:
            topic = item.get("topic") or item.get("assessment", {}).get("topic")
            if topic:
                existing_topics.add(topic)
        
        # Create case-insensitive set for comparison
        existing_topics_ci = {t.casefold() for t in existing_topics}

        def add_mcq_test(topic_name: str, difficulty: str = "Easy"):
            # Generate unique ID for this assessment
            mcq_id = generate_unique_assessment_id()
            plan.append({
                "type": "mcq",
                "topic": topic_name,
                "num_questions": getattr(self.config, 'default_mcq_questions', 10),
                "timer_per_question": getattr(self.config, 'default_mcq_timer_seconds', 60),
                "difficulty": difficulty,
                "id": mcq_id,
                "assessment_id": mcq_id
            })
            log.debug("Added required MCQ test: %s with ID: %s", topic_name, mcq_id)

        # Check if this is a technical candidate with skill-based assessments
        has_technical_assessments = any(
            any(keyword in (topic or "").lower() for keyword in [
                'python', 'javascript', 'react', 'node', 'programming', 'coding', 
                'software', 'development', 'engineering', 'technical', 'ai', 'ml',
                'data', 'algorithm', 'database', 'api', 'web', 'mobile'
            ]) for item in plan 
            for topic in [item.get("topic") or "", item.get("assessment", {}).get("topic") or ""]
        )

        # If plan already has multiple skills-based topics, keep generic tests minimal
        skills_based_items = [
            itm for itm in plan
            if any(kw in (itm.get("topic", "") or itm.get("assessment", {}).get("topic", "")).lower() for kw in [
                'seo','google ads','ga4','semrush','linkedin','content','copywriting','hubspot','zoho','mailchimp','canva','looker','analytics','a/b','cro'
            ])
        ]

        # For technical candidates, make generic tests optional and lower priority
        if has_technical_assessments:
            # Only add generic tests if explicitly required AND not already present
            if needs.get("psychometric_test") and "psychometric test" not in existing_topics_ci:
                difficulty = self._get_difficulty_by_experience(candidate_type, personalization_ctx.experience_years or 0)
                add_mcq_test("Psychometric Test", difficulty)
            
            if needs.get("communication_test") and "communication test" not in existing_topics_ci:
                difficulty = self._get_difficulty_by_experience(candidate_type, personalization_ctx.experience_years or 0)
                add_mcq_test("Communication Test", difficulty)
            
            if needs.get("personality_test") and "personality test" not in existing_topics_ci:
                difficulty = self._get_difficulty_by_experience(candidate_type, personalization_ctx.experience_years or 0)
                add_mcq_test("Personality Test", difficulty)
        else:
            # For non-technical candidates, prioritize skills-based items; add at most one generic test if required and plan < 3 items
            def maybe_add_generic(name: str):
                if len(plan) < 3 and name not in existing_topics_ci and needs.get(name.replace(" ", "_") + "", False):
                    difficulty = self._get_difficulty_by_experience(candidate_type, personalization_ctx.experience_years or 0)
                    add_mcq_test(name, difficulty)
            if needs.get("psychometric_test") and "psychometric test" not in existing_topics_ci and len(skills_based_items) < 3:
                difficulty = self._get_difficulty_by_experience(candidate_type, personalization_ctx.experience_years or 0)
                add_mcq_test("Psychometric Test", difficulty)
            if needs.get("communication_test") and "communication test" not in existing_topics_ci and len(skills_based_items) < 3:
                difficulty = self._get_difficulty_by_experience(candidate_type, personalization_ctx.experience_years or 0)
                add_mcq_test("Communication Test", difficulty)
            if needs.get("personality_test") and "personality test" not in existing_topics_ci and len(skills_based_items) < 3:
                difficulty = self._get_difficulty_by_experience(candidate_type, personalization_ctx.experience_years or 0)
                add_mcq_test("Personality Test", difficulty)
        
        # Add skill-based assessments for non-technical candidates
        if needs.get("skill_based_assessments"):
            for skill_topic in needs["skill_based_assessments"]:
                # Format skill as assessment topic
                topic_name = f"{skill_topic} Assessment"
                if topic_name.lower() not in existing_topics_ci:
                    # Use experience-based difficulty for skill-based assessments
                    difficulty = self._get_difficulty_by_experience(candidate_type, personalization_ctx.experience_years or 0)
                    add_mcq_test(topic_name, difficulty)
                    log.debug(f"Added skill-based assessment: {topic_name} with difficulty {difficulty}")
        
        return plan
    
    def calculate_assessment_time(self, assessment: Dict[str, Any], personalization_ctx: PersonalizationContext = None) -> int:
        """
        Calculate optimal assessment time based on multiple factors:
        - Question count and types
        - Difficulty level
        - Assessment type
        - User experience and performance history
        - Topic complexity
        
        Special handling for psychometric and similar tests with shorter, fixed times.
        
        Args:
            assessment: Assessment configuration dictionary
            personalization_ctx: User personalization context
            
        Returns:
            int: Calculated assessment time in minutes
        """
        # Get assessment details
        assessment_type = assessment.get("type", "multi")
        topic = (assessment.get("topic") or "")
        # Ensure topic is never None before calling .lower()
        if not topic or not isinstance(topic, str):
            topic = ""
        topic = topic.lower()
        
        # Special handling for psychometric and similar tests - these need shorter, fixed times
        PSYCHOMETRIC_TOPICS = {
            "psychometric test": 15,      # 15 minutes for psychometric tests
            "personality test": 12,       # 12 minutes for personality tests
            "communication test": 10,     # 10 minutes for communication tests
            "aptitude test": 20,          # 20 minutes for aptitude tests
            "iq test": 25,                # 25 minutes for IQ tests
            "behavioral test": 15,        # 15 minutes for behavioral tests
            "cognitive test": 18,         # 18 minutes for cognitive tests
            "soft skills test": 12,       # 12 minutes for soft skills tests
            "emotional intelligence": 15, # 15 minutes for EQ tests
            "leadership assessment": 20   # 20 minutes for leadership assessments
        }
        
        # Check if this is a psychometric or similar test
        for psychometric_topic, fixed_time in PSYCHOMETRIC_TOPICS.items():
            if psychometric_topic in topic:
                return fixed_time
        
        # For non-psychometric tests, use realistic calculation
        # Base time allocation per question type (in minutes) - realistic timings
        TIME_PER_QUESTION_TYPE = {
            "mcq": 1.0,      # Multiple choice questions - 1 min per question (realistic)
            "short": 2.0,    # Short answer questions - 2 min per question (realistic)
            "long": 3.5,     # Long form questions - 3.5 min per question (realistic)
            "coding": 6.0,   # Coding questions - 6 min per question (realistic)
            "default": 1.5   # Default time per question
        }
        
        # Difficulty multipliers - more realistic adjustments
        DIFFICULTY_MULTIPLIERS = {
            "easy": 0.85,      # Slightly less time for easy questions
            "medium": 1.0,     # Base multiplier
            "hard": 1.2,       # Reduced from 1.3 for more realistic timings
            "medium-hard": 1.1, # Reduced from 1.15
            "expert": 1.3      # Reduced from 1.5 for more realistic timings
        }
        
        # Enhanced topic complexity adjustments
        TOPIC_COMPLEXITY_ADJUSTMENTS = {
            # Programming languages - moderate complexity
            "python": 0,
            "javascript": 0,
            "java": 0,
            "c++": 5,  # Slightly more complex
            "go": 5,
            "rust": 10,  # More complex
            "typescript": 2,
            "kotlin": 3,
            "swift": 3,
            "scala": 8,
            "haskell": 12,
            
            # Web technologies - moderate complexity
            "react": 0,
            "angular": 5,
            "vue": 0,
            "node.js": 0,
            "express": 0,
            "django": 3,
            "flask": 2,
            "spring": 5,
            "laravel": 3,
            "rails": 3,
            
            # Data Science & ML - higher complexity (reduced adjustments)
            "machine learning": 8,
            "deep learning": 10,
            "data science": 5,
            "statistics": 5,
            "tensorflow": 8,
            "pytorch": 8,
            "pandas": 3,
            "numpy": 3,
            "scikit-learn": 5,
            "keras": 5,
            "opencv": 5,
            "matplotlib": 2,
            "seaborn": 2,
            
            # System Design - high complexity (reduced adjustments)
            "system design": 10,
            "distributed systems": 8,
            "microservices": 8,
            "cloud architecture": 8,
            "devops": 5,
            "docker": 5,
            "kubernetes": 10,
            "aws": 8,
            "azure": 8,
            "gcp": 8,
            "terraform": 8,
            "ansible": 5,
            
            # Databases - moderate complexity
            "sql": 5,
            "postgresql": 5,
            "mysql": 3,
            "mongodb": 5,
            "redis": 5,
            "database design": 10,
            "nosql": 8,
            "elasticsearch": 8,
            "cassandra": 10,
            
            # Cybersecurity - high complexity (reduced adjustments)
            "cybersecurity": 8,
            "penetration testing": 10,
            "network security": 6,
            "cryptography": 8,
            "ethical hacking": 8,
            "owasp": 5,
            
            # Blockchain - high complexity (reduced adjustments)
            "blockchain": 10,
            "ethereum": 8,
            "solidity": 8,
            "web3": 6,
            "defi": 8,
            "smart contracts": 8,
            
            # Soft skills - lower complexity
            "communication": -5,
            "leadership": -5,
            "project management": -5,
            "agile": -5,
            "teamwork": -5,
            "scrum": -3,
            "kanban": -3,
            
            # Hardware/Embedded - high complexity (reduced adjustments)
            "embedded systems": 8,
            "microcontrollers": 6,
            "arduino": 5,
            "raspberry pi": 5,
            "fpga": 10,
            "verilog": 8,
            "vhdl": 8,
            "systemverilog": 10,
            "vlsi": 10,
            
            # Mobile Development - moderate complexity
            "android": 8,
            "ios": 8,
            "flutter": 5,
            "react native": 5,
            "xamarin": 8,
            
            # Game Development - moderate complexity
            "game development": 10,
            "unity": 8,
            "unreal": 10,
            "opengl": 12,
            "directx": 12
        }
        
        # Apply topic-specific adjustment
        topic_adjustment = 0
        if topic:  # Only process if topic exists
            for topic_key, adjustment in TOPIC_COMPLEXITY_ADJUSTMENTS.items():
                if topic_key.lower() in topic.lower():
                    topic_adjustment = adjustment
                    break
        
        # Calculate base time from question types
        num_questions = assessment.get("num_questions") or {}
        if isinstance(num_questions, int):
            num_questions = {"mcq": num_questions}
        total_time = 10  # Reduced base time from 20 to 10 minutes for more realistic timings
        
        for question_type, count in num_questions.items():
            if question_type in TIME_PER_QUESTION_TYPE:
                total_time += count * TIME_PER_QUESTION_TYPE[question_type]
            else:
                total_time += count * TIME_PER_QUESTION_TYPE["default"]
        
        # Apply difficulty multiplier
        difficulty = assessment.get("difficulty", "medium").lower()
        multiplier = DIFFICULTY_MULTIPLIERS.get(difficulty, 1.0)
        total_time = int(total_time * multiplier)
        
        # Apply topic-specific adjustment
        total_time += topic_adjustment
        
        # Apply personalization adjustments
        if personalization_ctx:
            # Adjust based on user experience
            experience_years = personalization_ctx.experience_years or 0
            if experience_years > 5:
                total_time = int(total_time * 0.9)  # Experienced users are faster
            elif experience_years < 2:
                total_time = int(total_time * 1.1)  # Beginners need more time
            
            # Adjust based on performance trend
            performance_trend = personalization_ctx.interaction_patterns.get('performance_trend', 'stable')
            if performance_trend == 'improving':
                total_time = int(total_time * 0.95)  # Improving users are getting faster
            elif performance_trend == 'declining':
                total_time = int(total_time * 1.05)  # Declining users need more time
        
        # Ensure reasonable bounds - more realistic maximum
        min_time = 10   # Minimum 10 minutes
        max_time = 60   # Reduced maximum from 120 to 60 minutes (1 hour) for more realistic timings
        
        return max(min_time, min(total_time, max_time))
    def _personalize_assessment_plan(self, plan: List[Dict[str, Any]], personalization_ctx: PersonalizationContext) -> List[Dict[str, Any]]:
        """Apply comprehensive personalization enhancements to assessment plan based on user context."""
        for item in plan:
            assessment = item.get("assessment", item)
            
            # Adaptive difficulty based on skill level and performance history
            base_difficulty = "Medium"
            if personalization_ctx.skill_level == "expert":
                base_difficulty = "Hard"
            elif personalization_ctx.skill_level == "advanced":
                base_difficulty = "Medium-Hard"
            elif personalization_ctx.skill_level == "beginner":
                base_difficulty = "Easy"
            
            # Adjust based on performance history
            if personalization_ctx.previous_performance is not None:
                if personalization_ctx.previous_performance >= 0.8:
                    # Strong performer - can handle harder questions
                    if base_difficulty == "Medium":
                        base_difficulty = "Medium-Hard"
                    elif base_difficulty == "Easy":
                        base_difficulty = "Medium"
                elif personalization_ctx.previous_performance < 0.5:
                    # Struggling - provide easier questions to build confidence
                    if base_difficulty == "Hard":
                        base_difficulty = "Medium-Hard"
                    elif base_difficulty == "Medium-Hard":
                        base_difficulty = "Medium"
                    elif base_difficulty == "Medium":
                        base_difficulty = "Easy"
            
            # Adjust based on performance trend
            if personalization_ctx.interaction_patterns:
                trend = personalization_ctx.interaction_patterns.get("performance_trend", "stable")
                if trend == "improving":
                    # User is learning well, can handle slightly more challenging content
                    if base_difficulty == "Easy":
                        base_difficulty = "Medium"
                    elif base_difficulty == "Medium":
                        base_difficulty = "Medium-Hard"
                elif trend == "declining":
                    # User needs support, reduce difficulty
                    if base_difficulty == "Hard":
                        base_difficulty = "Medium"
                    elif base_difficulty == "Medium-Hard":
                        base_difficulty = "Medium"
            
            assessment["difficulty"] = base_difficulty
            
            # Use intelligent time calculation
            if assessment.get("type") == "multi":
                assessment["assessment_time_minutes"] = self.calculate_assessment_time(assessment, personalization_ctx)
            elif assessment.get("type") == "mcq":
                # For MCQ, calculate timer_per_question based on difficulty and topic
                topic = assessment.get("topic") or ""
                # Ensure topic is never None before calling .lower()
                if not topic or not isinstance(topic, str):
                    topic = ""
                topic = topic.lower()
                
                # Special handling for psychometric and similar tests - shorter timer per question
                PSYCHOMETRIC_TOPICS = {
                    "psychometric test": 30,      # 30 seconds per question
                    "personality test": 25,       # 25 seconds per question
                    "communication test": 20,     # 20 seconds per question
                    "aptitude test": 35,          # 35 seconds per question
                    "iq test": 40,                # 40 seconds per question
                    "behavioral test": 30,        # 30 seconds per question
                    "cognitive test": 35,         # 35 seconds per question
                    "soft skills test": 25,       # 25 seconds per question
                    "emotional intelligence": 30, # 30 seconds per question
                    "leadership assessment": 35   # 35 seconds per question
                }
                
                # Check if this is a psychometric or similar test
                base_timer = 60  # Default 60 seconds per question
                for psychometric_topic, fixed_timer in PSYCHOMETRIC_TOPICS.items():
                    if psychometric_topic in topic:
                        base_timer = fixed_timer
                        break
                else:
                    # For non-psychometric tests, use sophisticated calculation
                    difficulty_multiplier = {"easy": 0.8, "medium": 1.0, "hard": 1.3, "medium-hard": 1.15, "expert": 1.5}
                    topic_multiplier = 1.0
                    
                    difficulty = assessment.get("difficulty", "medium").lower()
                    
                    # Apply difficulty adjustment
                    base_timer *= difficulty_multiplier.get(difficulty, 1.0)
                    
                    # Apply topic complexity adjustment
                    if topic in ["machine learning", "artificial intelligence", "algorithms", "system design"]:
                        topic_multiplier = 1.2
                    elif topic in ["python", "javascript", "java"]:
                        topic_multiplier = 1.0
                    elif topic in ["c++", "rust", "haskell"]:
                        topic_multiplier = 1.1
                    
                    base_timer *= topic_multiplier
                    
                    # Apply personalization
                    if personalization_ctx.experience_years:
                        if personalization_ctx.experience_years < 1:
                            base_timer *= 1.2
                        elif personalization_ctx.experience_years > 10:
                            base_timer *= 0.9
                
                assessment["timer_per_question"] = max(15, min(300, int(base_timer)))
            
            # Update item structure
            if "assessment" in item:
                item["assessment"] = assessment
            else:
                item.update(assessment)
        
        return plan
    
    async def _generate_advanced_topics_dynamically(
        self,
        topic: str,
        score: float,
        personalization_ctx: PersonalizationContext
    ) -> List[Dict[str, Any]]:
        """
        Generate advanced/related topics dynamically using LLM for any profession.
        
        This method replaces the static mapping with an LLM-driven approach that works
        universally across all topics and professions, following agentic AI principles.
        
        Args:
            topic (str): The primary topic the candidate excelled in
            score (float): The assessment score (0-100)
            personalization_ctx (PersonalizationContext): User personalization context
            
        Returns:
            List[Dict[str, Any]]: List of advanced topic assessment configurations
        """
        if not topic or score < 80:
            return []
        
        # Generate prompt for LLM to suggest advanced topics
        prompt = f"""Generate 2-3 advanced or related assessment topics for a candidate who scored {score}% in "{topic}".

Context:
- Skill Level: {personalization_ctx.skill_level}
- Experience: {personalization_ctx.experience_years or 'Unknown'} years
- Profession: {personalization_ctx.profession_type}

Rules:
- Suggest topics that are naturally advanced or related to "{topic}"
- Topics should be appropriate for the candidate's skill level and experience
- Each topic should be a logical next step or advanced concept
- Keep topics profession-agnostic and universally applicable
- For technical topics, suggest deeper technical concepts
- For non-technical topics, suggest advanced applications or related domains

Return ONLY a JSON array of assessment configurations:
[
  {{
    "type": "multi",
    "topic": "<advanced_topic_name>",
    "difficulty": "Hard",
    "num_questions": {{"mcq": 6, "short": 2, "long": 1, "coding": <0_or_1_based_on_topic>}}
  }}
]

Primary Topic: {topic}
Return JSON array only, no other text."""

        try:
            # Use LLM to generate advanced topics
            llm_response = await self._llm_with_retry(prompt, model=self.config.llm_model)
            content = getattr(llm_response, "content", str(llm_response))
            
            # Extract JSON from response
            result = self._extract_json(content)
            
            # Validate and sanitize results
            if not isinstance(result, list):
                log.warning(f"LLM returned non-list for advanced topics: {type(result)}")
                return []
            
            validated_topics = []
            for item in result:
                if isinstance(item, dict):
                    # Ensure required fields
                    if not item.get("topic"):
                        continue
                    
                    # Set defaults
                    item.setdefault("type", "multi")
                    item.setdefault("difficulty", "Hard")
                    
                    # Validate num_questions
                    if "num_questions" not in item:
                        item["num_questions"] = {"mcq": 6, "short": 2, "long": 1, "coding": 0}
                    
                    num_questions = item.get("num_questions", {})
                    if not isinstance(num_questions, dict):
                        item["num_questions"] = {"mcq": 6, "short": 2, "long": 1, "coding": 0}
                    else:
                        # Ensure coding is only included if practical
                        topic_name = item.get("topic", "")
                        is_coding_practical = self._is_coding_practical_for_skill(topic_name)
                        if not is_coding_practical:
                            num_questions["coding"] = 0
                        elif "coding" not in num_questions:
                            num_questions["coding"] = 1 if is_coding_practical else 0
                    
                    validated_topics.append(item)
            
            log.info(f"Generated {len(validated_topics)} advanced topics for '{topic}'", extra={
                "topic": topic,
                "score": score,
                "topics_count": len(validated_topics)
            })
            
            return validated_topics[:3]  # Limit to 3 advanced topics
            
        except Exception as e:
            log.warning(f"Failed to generate advanced topics dynamically for '{topic}': {e}")
            return []  # Return empty list, fallback to static mapping will be used
    
    def _create_personalized_fallback_plan(self, assessment_needs: Dict[str, Any], personalization_ctx: PersonalizationContext, skills: List[str]) -> List[Dict[str, Any]]:
        """Create adaptive fallback plan when LLM fails, based on user context and performance."""
        fallback_plan = []
        
        # Determine number of skills to assess based on user context
        max_skills = 3  # Default
        
        # Adjust based on experience and performance
        if personalization_ctx.previous_performance is not None:
            if personalization_ctx.previous_performance >= 0.8:
                max_skills = min(4, len(skills))  # Strong performers can handle more
            elif personalization_ctx.previous_performance < 0.5:
                max_skills = 2  # Focus on fewer skills for struggling users
        
        # Consider interaction patterns
        if personalization_ctx.interaction_patterns:
            assessment_count = personalization_ctx.interaction_patterns.get("assessment_count", 0)
            if assessment_count > 10:
                # Experienced user, can handle more assessments
                max_skills = min(5, len(skills))
        
        # Adaptive difficulty based on user context and experience
        candidate_type = assessment_needs.get("candidate_type", "experienced").lower()
        if not candidate_type:
            # Determine candidate type from experience
            experience_years = personalization_ctx.experience_years or 0
            if experience_years < 2:
                candidate_type = "fresher"
            elif experience_years < 5:
                candidate_type = "experienced"
            else:
                candidate_type = "senior"
        
        # Use experience-based difficulty as the base
        base_difficulty = self._get_difficulty_by_experience(candidate_type, personalization_ctx.experience_years or 0)
        log.debug(f"Fallback plan using difficulty '{base_difficulty}' for {candidate_type} candidate with {personalization_ctx.experience_years or 0} years experience")
        
        # Add skill-based assessments for selected skills
        coding_enabled = bool(assessment_needs.get("coding_test", False))
        for skill in skills[:max_skills]:
            # Check if coding questions are practical for this skill
            is_coding_practical = self._is_coding_practical_for_skill(skill)
            coding_count = 1 if (coding_enabled and is_coding_practical) else 0
            
            assessment_config = {
                "type": "multi",
                "topic": skill,
                "difficulty": base_difficulty,
                "num_questions": {
                    "mcq": 5,
                    "short": 2,
                    "long": 1,
                    "coding": coding_count
                }
            }
            # Calculate dynamic assessment time
            assessment_config["assessment_time_minutes"] = self.calculate_assessment_time(assessment_config, personalization_ctx)
            fallback_plan.append(assessment_config)
        
        # Apply policy enforcement and append required MCQs
        fallback_plan = self._enforce_policy_on_plan(fallback_plan, assessment_needs)
        fallback_plan = self._append_required_mcqs(fallback_plan, assessment_needs, personalization_ctx)
        fallback_plan = self._personalize_assessment_plan(fallback_plan, personalization_ctx)
        
        return fallback_plan

    async def _generate_dynamic_assessment_plan(
        self,
        resume: Dict[str, Any],
        assessment_needs: Dict[str, Any],
        user_interest_answers: List[Dict[str, str]],
        personalization_ctx: PersonalizationContext,
        uid: Optional[str] = None,  # NEW: For accessing cached Gemini summary
        skill_gap_analysis: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a personalized dynamic assessment plan based on skills, user context, and skill gaps.
        
        Args:
            resume (Dict[str, Any]): Structured resume data
            assessment_needs (Dict[str, Any]): Assessment requirements from previous analysis
            user_interest_answers (List[Dict[str, str]]): User's interest survey responses
            personalization_ctx (PersonalizationContext): User personalization context
            skill_gap_analysis (Optional[Dict[str, Any]]): Career advisor skill gap output (missing_skills, etc.)
            
        Returns:
            List[Dict[str, Any]]: List of personalized assessment configurations
            
        Raises:
            LLMResponseError: When LLM response cannot be parsed
        """
        # Validate inputs
        if not isinstance(user_interest_answers, list):
            log.warning("Invalid user_interest_answers format, using empty list")
            user_interest_answers = []

        skills = self._extract_skills(resume)
        skills_count = len(skills)
        
        log.debug("Generating personalized dynamic assessment plan", extra={
            "skills_count": skills_count,
            "coding_test": assessment_needs.get("coding_test", False),
            "profession_type": personalization_ctx.profession_type,
            "skill_level": personalization_ctx.skill_level,
            "personalization_score": personalization_ctx.personalization_score
        })
        
        # Convert data to JSON strings for prompt (with size limits)
        skills_text = json.dumps(skills, indent=2)
        interest_text = json.dumps(user_interest_answers, indent=2)

        prompt = await self._get_assessment_plan_prompt(
            resume, skills_text, interest_text, assessment_needs, personalization_ctx, uid=uid,
            skill_gap_analysis=skill_gap_analysis
        )
        
        # Optimize prompt size for faster processing
        max_prompt_size = min(self.config.max_prompt_chars, 8000)  # Reduced from 12000 for faster processing
        if len(prompt) > max_prompt_size:
            if log.isEnabledFor(logging.WARNING):
                log.warning(f"Assessment plan prompt too large, truncating from {len(prompt)} to {max_prompt_size}")
            prompt = prompt[:max_prompt_size]

        try:
            # Define structured output schema for assessment plan
            from pydantic import BaseModel, Field
            from typing import Optional, Dict as _Dict, List as _List
            from pydantic import ConfigDict as _ConfigDict

            class NumQuestions(BaseModel):
                mcq: Optional[int] = 0
                short: Optional[int] = 0
                long: Optional[int] = 0
                coding: Optional[int] = 0

            class AssessmentItem(BaseModel):
                # Common fields observed in current flow
                type: Optional[str] = None
                topic: Optional[str] = None
                difficulty: Optional[str] = None
                assessment_time_minutes: Optional[int] = None
                num_questions: Optional[NumQuestions] = None
                rationale: Optional[str] = None  # LLM-generated: one sentence why this assessment was recommended
                # Allow extra keys to remain forward-compatible
                model_config = _ConfigDict(extra='allow')

            class AssessmentPlan(BaseModel):
                items: _List[AssessmentItem] = Field(default_factory=list)

            structured_result: AssessmentPlan = await invoke_structured_llm(
                prompt,
                AssessmentPlan,
                task_type=TaskType.ASSESSMENT_GENERATION,
                preferred_model="gemini-2.5-flash",
                agent_name="assessment_recommender_plan",
                temperature=0.2,
                max_output_tokens=2000,
                timeout=35.0,
                raise_on_fallback=False,
            )

            result = [item.model_dump() for item in structured_result.items]

            # Apply policy enforcement and append required tests
            result = self._enforce_policy_on_plan(result, assessment_needs)
            result = self._append_required_mcqs(result, assessment_needs, personalization_ctx)
            
            # Apply personalization enhancements
            result = self._personalize_assessment_plan(result, personalization_ctx)
            
            # Sanitize and cap the plan size while preserving required MCQs
            sanitized_result = []
            for item in result:
                sanitized_item = self._sanitize_plan_item(item)
                if sanitized_item:
                    sanitized_result.append(sanitized_item)
            
            # Ensure required MCQs are preserved when capping
            required_topics = {"Psychometric Test", "Communication Test", "Personality Test"}
            
            def is_required(item):
                topic = item.get("topic") or item.get("assessment", {}).get("topic")
                return item.get("type") == "mcq" and topic in required_topics
            
            required = [item for item in sanitized_result if is_required(item)]
            others = [item for item in sanitized_result if not is_required(item)]
            
            # Put required MCQs first to ensure they're preserved
            ordered = required + others
            
            # Cap while preserving all required MCQs (5 skills + 5 gaps + 3 required = 13)
            max_assessments = getattr(self.config, 'max_assessments', 13)
            if len(ordered) > max_assessments:
                keep = required + others[:max(0, max_assessments - len(required))]
                sanitized_result = keep
                log.warning("Plan too large; truncated from %d to %d (preserved %d required MCQs)", 
                             len(ordered), len(sanitized_result), len(required))
            else:
                sanitized_result = ordered
            
            # Post-process: Adjust difficulty based on experience for all assessments
            candidate_type = assessment_needs.get("candidate_type", "experienced").lower()
            if not candidate_type:
                # Determine candidate type from experience
                experience_years = personalization_ctx.experience_years or 0
                if experience_years < 2:
                    candidate_type = "fresher"
                elif experience_years < 5:
                    candidate_type = "experienced"
                else:
                    candidate_type = "senior"
            
            # Adjust difficulty for all assessments in the plan
            for item in sanitized_result:
                correct_difficulty = self._get_difficulty_by_experience(candidate_type, personalization_ctx.experience_years or 0)
                item["difficulty"] = correct_difficulty
                log.debug(f"Adjusted difficulty for '{item.get('topic', 'Unknown')}' to {correct_difficulty}")
            
            result = sanitized_result
            
            # Set rationale and current_score for each assessment (gap vs existing skills)
            self._set_recommendation_source_on_plan(result, skill_gap_analysis, resume)
            
            # Assign unique IDs to all assessments
            result = assign_unique_ids_to_assessments(result)
            
            # Add status tracking fields to all assessments
            for item in result:
                if isinstance(item, dict):
                    if "status" not in item:
                        item["status"] = "pending"
                    if "assessment_id" not in item:
                        item["assessment_id"] = None
                    if "completed_at" not in item:
                        item["completed_at"] = None
                    if "score" not in item:
                        item["score"] = None
                
            log.info("Generated personalized assessment plan", extra={
                "items_count": len(result),
                "skills_count": skills_count,
                "coding_enabled": assessment_needs.get("coding_test", False),
                "profession_type": personalization_ctx.profession_type,
                "personalization_score": personalization_ctx.personalization_score
            })
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("LLM returned invalid JSON for assessment plan, using personalized fallback", extra={
                "error": str(e),
                "profession_type": personalization_ctx.profession_type
            })
            # Create personalized fallback plan
            fallback_plan = self._create_personalized_fallback_plan(assessment_needs, personalization_ctx, skills)
            self._set_recommendation_source_on_plan(fallback_plan, skill_gap_analysis, resume)
            # Assign unique IDs to fallback plan assessments
            fallback_plan = assign_unique_ids_to_assessments(fallback_plan)
            log.info("Using personalized fallback assessment plan", extra={
                "plan_count": len(fallback_plan),
                "profession_type": personalization_ctx.profession_type
            })
            return fallback_plan
            
        except Exception as e:
            log.error("Failed to generate personalized assessment plan", extra={
                "error": str(e),
                "profession_type": personalization_ctx.profession_type
            })
            raise LLMResponseError(f"Could not generate assessment plan: {str(e)}") from e

# Initialize with user-isolated cache
recommender = UnifiedAssessmentRecommender()


async def assessment_recommender_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced assessment recommender agent with centralized utilities and LLM-only approach.
    
    This function serves as the interface between the multi-agent pipeline and the 
    UnifiedAssessmentRecommender class, providing user context awareness and personalization.
    
    Args:
        state (Dict[str, Any]): Current workflow state containing resume and user data
        
    Returns:
        Dict[str, Any]: Personalized assessment recommendations with consistent error envelope
    """
    # Use centralized logging
    log_context = create_log_context("assessment_recommender", state.get("tenant_id", "default"))
    start_time = log_context["start_time"]
    request_id = log_context["request_id"]
    
    # Get tenant-scoped memory
    recommender_memory = await get_assessment_recommender_memory(state.get("tenant_id", "default"))
    
    # Detect rerun mode
    is_rerun = bool(state.get("assessment_results"))
    uid = state.get("uid")
    assessment_id = None
    score = None
    topic = None
    difficulty_before = None
    difficulty_after = None
    consecutive_passes = None
    consecutive_fails = None
    plan_delta_size = None
    
    if is_rerun:
        results = state.get("assessment_results", {})
        score = results.get("total_score", 0)
        topic = state.get("assessment_topic") or results.get("assessment_topic") or "General Assessment"
        # Ensure topic is never None or empty
        if not topic or not isinstance(topic, str):
            topic = "General Assessment"
        # Log rerun start
        if log.isEnabledFor(logging.INFO):
            log.info("🔄 ASSESSMENT_RECOMMENDER: Starting adaptive rerun", extra={
                "uid": uid,
                "topic": topic,
                "score": score,
                "mode": "rerun"
            })
    
    try:
        result = await recommender.process(state)
        
        # Extract rerun-specific metrics if available
        if is_rerun:
            assessment_id = state.get("assessment_history", [{}])[-1].get("assessment_id") if state.get("assessment_history") else None
            consecutive_passes = state.get("consecutive_passes", 0)
            consecutive_fails = state.get("consecutive_fails", 0)
            prior_plan_size = len(state.get("prior_assessment_plan", []))
            new_plan_size = len(result.get("assessment_plan", []))
            plan_delta_size = new_plan_size - prior_plan_size
            
            # Extract difficulty info from result if available
            if result.get("assessment_plan"):
                for item in result.get("assessment_plan", []):
                    if item.get("topic") == topic:
                        difficulty_after = item.get("difficulty", "unknown")
                        break
        
        # Record successful recommendation
        processing_time = _calculate_processing_time(start_time)
        
        # Build telemetry tags
        telemetry_tags = {
            "mode": "rerun" if is_rerun else "initial",
            "success": True
        }
        if is_rerun:
            telemetry_tags.update({
                "assessment_id": assessment_id,
                "score": score,
                "topic": topic,
                "consecutive_passes": consecutive_passes,
                "consecutive_fails": consecutive_fails,
                "plan_delta_size": plan_delta_size,
                "difficulty_after": difficulty_after
            })
        
        await recommender_memory.record_attempt(
            'assessment_recommendation', 'llm', True, 0.8, processing_time
        )
        
        # Log success with rerun-specific fields
        log_data = {
            "success": True,
            "assessment_plan_count": len(result.get("assessment_plan", [])),
            "personalization_score": result.get("personalization_score", 0.0)
        }
        if is_rerun:
            log_data.update({
                "mode": "rerun",
                "uid": uid,
                "assessment_id": assessment_id,
                "score": score,
                "topic": topic,
                "difficulty_before": difficulty_before,
                "difficulty_after": difficulty_after,
                "consecutive_passes": consecutive_passes,
                "consecutive_fails": consecutive_fails,
                "plan_delta_size": plan_delta_size,
                "topic_expansion_count": plan_delta_size if plan_delta_size else 0
            })
        
        log_agent_completion(log_context, log_data, "llm", processing_time)
        
        # NEW: Persist full plan to uid_assessments and prefer sending freshly computed plan to callback users
        # Run blocking I/O in thread pool to avoid blocking event loop during concurrent requests
        try:
            uid = state.get("uid")
            if uid:
                doc = await run_blocking_io(get_assessments_doc, uid) or {}
                doc["assessment_recommender"] = {
                    "assessment_needs": result.get("assessment_needs", {}),
                    "assessment_plan": result.get("assessment_plan", []),
                    "personalization_score": result.get("personalization_score", 0.0),
                    "updated_at": __import__("time").time(),
                    "version": (doc.get("assessment_recommender", {}).get("version", 0) + 1)
                }
                await run_blocking_io(
                    upsert_assessments_doc,
                    uid,
                    doc,
                    metadata={
                        "agent": "assessment_recommender",
                        "uid": uid,
                        "status": "assessment_recommender_updated"
                    }
                )
        except Exception as e_store:
            log.warning(f"⚠️ ASSESSMENT_RECOMMENDER: Failed to persist uid_assessments: {e_store}")
        
        # Return both formats:
        # 1. Strict JSON format for callbacks
        # 2. Original fields for internal pipeline use
        return {
            # Strict JSON format for callbacks
            "status": "completed",
            "node": "assessment_recommender",
            "output": {
                "assessment_needs": result.get("assessment_needs", {}),
                "assessment_plan": result.get("assessment_plan", [])
            },
            
            # Original fields for internal pipeline use
            "assessment_needs": result.get("assessment_needs", {}),
            "assessment_plan": result.get("assessment_plan", []),
            "personalization_score": result.get("personalization_score", 0.0),
            "analysis_method": "llm",
            "confidence_score": 0.8,
            "processing_time": processing_time
        }
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        AgentLogger.log_error(log_context, f"Assessment recommender agent failed: {str(e)}", processing_time)
        
        # Record failure
        await recommender_memory.record_attempt(
            'assessment_recommendation', 'llm', False, 0.0, processing_time
        )
        
        # Return consistent error envelope with stable shape
        return _create_error_response(f"Assessment recommendation failed: {str(e)}", processing_time)

# (Removed simple static recommender override to enable prompt-based UnifiedAssessmentRecommender)

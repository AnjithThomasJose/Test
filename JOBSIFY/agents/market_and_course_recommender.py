import json
import os
import re
import asyncio
import logging
import time
import hashlib
import uuid
import aiohttp
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Tuple, Set, Literal

from core.model_registry import TaskType
from models.llm_invoker import invoke_llm, is_fallback_response, invoke_structured_llm
from langsmith.run_helpers import traceable
from agents.prompt_generator import generate_market_and_course_recommendation_prompt
from chroma import fetch_structured_resume, get_chat_session, update_chat_session
from chroma import get_gap_doc, upsert_gap_doc, get_assessments_doc  # NEW per-UID gap helpers
from utils.session_manager import session_manager
from core.utils import (
    _mask, _sanitize_text_for_llm, _validate_state_inputs, _to_text, _clean_json_text, 
    _scan_balanced_json, _safe_json_loads, _extract_json_from_response, _create_error_response,
    _generate_request_id, _calculate_processing_time, create_agent_state, filter_resume_data_for_agent,
    run_blocking_io, get_missing_skills_flat
)
from core.config import get_agent_config
from core.security import validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm, PII_PATTERNS, INJECTION_FILTERS
from core.memory import BaseAgentMemory, get_agent_memory
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion
from agents.course_knowledge_base import CourseKnowledgeBase
try:
    # Optional: SerpAPI search via langchain if key available
    from langchain_community.utilities import SerpAPIWrapper  # type: ignore
except Exception:
    SerpAPIWrapper = None  # type: ignore

log = logging.getLogger(__name__)


def _env_bounded_int(name: str, default: int, upper: int) -> int:
    try:
        v = int(os.getenv(name, str(default)).strip())
        return max(1, min(v, upper))
    except ValueError:
        return default


# Parallel LLM course-URL checks (fallback path); global LLM semaphore still applies
_COURSE_LLM_VALIDATION_MAX_PARALLEL = _env_bounded_int("COURSE_LLM_VALIDATION_MAX_PARALLEL", 6, 32)
# HTTP HEAD/GET accessibility checks in batch URL validation
_COURSE_URL_HTTP_VALIDATION_MAX_PARALLEL = _env_bounded_int("COURSE_URL_HTTP_VALIDATION_MAX_PARALLEL", 15, 64)

# LangChain imports for hybrid RAG approach
try:
    from langchain_community.vectorstores import Chroma  # type: ignore
    from langchain_community.embeddings import SentenceTransformerEmbeddings  # type: ignore
    from langchain_core.documents import Document  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    SystemMessage = None  # type: ignore
    HumanMessage = None  # type: ignore
    log.warning("LangChain not available, falling back to manual retrieval")

# Static system instructions for Gemini context caching
_SEARCH_QUERIES_SYSTEM_INSTRUCTION = """You are a course recommendation assistant. Your job is to generate search queries 
to find relevant courses in a knowledge base. You will NOT create course titles or URLs - you will only generate 
search queries that will be used to search an existing course database.

Generate 5-8 specific search queries that would find the most relevant courses in a semantic search database.
Each query should:
1. Be specific and focused (not too broad or generic)
2. Include relevant technical terms and skill names
3. Consider the candidate's experience level
4. Target the missing skills they need to learn
5. Align with their target roles and career paths
6. Use natural language that would match course titles, descriptions, and skills

CRITICAL - Intelligent Synonym & Terminology Expansion (APPLIES TO ALL DOMAINS):
For EVERY skill, role, or domain mentioned, you MUST automatically:

1. **Identify Synonyms**: Think about alternative names for the same concept
   - Example: "QA" = "Quality Assurance" = "Software Testing" = "Test Engineering"
   - Example: "Data Science" = "Data Analytics" = "Data Analysis" = "Analytics"
   - Example: "Machine Learning" = "ML" = "AI" = "Artificial Intelligence" = "Deep Learning"

2. **Include Acronyms & Full Terms**: Always search for both
   - "API" and "Application Programming Interface"
   - "ML" and "Machine Learning"
   - "AWS" and "Amazon Web Services"
   - "SQL" and "Structured Query Language"

3. **Consider Course Provider Terminology**: Course providers may use different terms than job titles
   - Job title: "QA Engineer" → Course might say "Software Testing" or "Test Automation"
   - Job title: "Data Scientist" → Course might say "Data Analytics" or "Data Analysis"
   - Job title: "DevOps Engineer" → Course might say "CI/CD" or "Infrastructure as Code"

4. **Include Related Domain Concepts**: Think about broader and narrower terms
   - For "Web Development": include "Frontend", "Backend", "Full Stack", "React", "Vue", "Angular", "Node.js"
   - For "Cloud Computing": include "AWS", "Azure", "GCP", "Infrastructure", "DevOps", "Containerization"
   - For "Testing": include "Test Automation", "Selenium", "Cypress", "Jest", "Manual Testing", "QA"

5. **Include Tool/Framework Names**: Add relevant tools and frameworks
   - For "Python": include "Django", "Flask", "Pandas", "NumPy"
   - For "JavaScript": include "React", "Node.js", "Vue", "Angular", "TypeScript"
   - For "Testing": include "Selenium", "Cypress", "Jest", "TestNG", "JUnit"

6. **Think About Industry Variations**: Consider how different industries/regions name things
   - "Software Engineering" vs "Software Development" vs "Programming"
   - "Data Engineering" vs "ETL" vs "Data Pipeline"
   - "Product Management" vs "Product Owner" vs "Product Strategy"

7. **Include Related Skills**: Add skills that are commonly associated
   - For "Frontend": also include "UI/UX", "Design", "CSS", "HTML"
   - For "Backend": also include "API", "Database", "Server", "Microservices"
   - For "Data Science": also include "Statistics", "Python", "SQL", "Visualization"

IMPORTANT: Apply this synonym expansion intelligently to ALL domains mentioned in the profile, not just the examples above. 
Think creatively about how the same concept might be expressed differently in course titles and descriptions.

Examples of intelligent queries with comprehensive synonym expansion:
- "QA Quality Assurance software testing test automation Selenium courses"
- "Data science data analytics machine learning Python SQL courses"
- "React.js frontend web development JavaScript TypeScript UI courses"
- "AWS cloud computing infrastructure DevOps CI/CD courses"
- "Product management product owner strategy agile courses"

Return a JSON object with search_queries (list of strings) and reasoning (string)."""

_RAG_SELECTION_SYSTEM_INSTRUCTION = """You are a learning material recommendation expert. Select exactly 5 courses and 5 other materials (books, papers, videos, tutorials) and provide personalized explanations.

TASK: Select exactly 5 courses and 5 other materials. For each, provide:
- personalized_explanation (2-3 sentences max)
- target_skill (one missing skill it addresses)
- career_path_alignment (brief)

Return JSON with selected_courses and selected_materials arrays."""

# Pydantic imports for structured LLM output (Issue 5.1)
from pydantic import BaseModel, Field, field_validator
from settings import settings as _settings


# =============================================================================
# Pydantic Models for Structured LLM Output (Issue 5.1)
# =============================================================================
# These models ensure type-safe, validated responses from LLM calls.

def _clamp(v: float, min_val: float, max_val: float) -> float:
    """Clamp a value to a range."""
    return max(min_val, min(max_val, v))

def _sanitize_str(v: Any, max_length: int = 500, default: str = "") -> str:
    """Sanitize a string value."""
    if not v or not isinstance(v, str):
        return default
    return v.strip()[:max_length]

def _sanitize_str_list(v: Any, max_items: int = 10, max_item_length: int = 200) -> List[str]:
    """Sanitize a list of strings."""
    if not v or not isinstance(v, list):
        return []
    result = []
    for item in v[:max_items]:
        if isinstance(item, str) and item.strip():
            result.append(item.strip()[:max_item_length])
    return result


class SearchQueriesResult(BaseModel):
    """Structured output for LLM-generated search queries."""
    search_queries: List[str] = Field(default_factory=list, description="List of search queries with synonyms")
    reasoning: str = Field(default="", description="Explanation of query choices")
    
    @field_validator('search_queries', mode='before')
    @classmethod
    def validate_queries(cls, v):
        return _sanitize_str_list(v, max_items=10, max_item_length=200)
    
    @field_validator('reasoning', mode='before')
    @classmethod
    def validate_reasoning(cls, v):
        return _sanitize_str(v, max_length=500, default="")


class SelectedCourseItem(BaseModel):
    """A course selected by RAG."""
    course_id: str = Field(default="", description="Course identifier")
    personalized_explanation: str = Field(default="", description="Why this course is recommended")
    target_skill: str = Field(default="", description="Skill this course addresses")
    career_path_alignment: str = Field(default="", description="How it aligns with career goals")
    
    @field_validator('course_id', 'target_skill', 'career_path_alignment', mode='before')
    @classmethod
    def validate_short_str(cls, v):
        return _sanitize_str(v, max_length=100, default="")
    
    @field_validator('personalized_explanation', mode='before')
    @classmethod
    def validate_explanation(cls, v):
        return _sanitize_str(v, max_length=500, default="")


class SelectedMaterialItem(BaseModel):
    """A material selected by RAG."""
    material_id: str = Field(default="", description="Material identifier")
    personalized_explanation: str = Field(default="", description="Why this material is recommended")
    target_skill: str = Field(default="", description="Skill this material addresses")
    career_path_alignment: str = Field(default="", description="How it aligns with career goals")
    
    @field_validator('material_id', 'target_skill', 'career_path_alignment', mode='before')
    @classmethod
    def validate_short_str(cls, v):
        return _sanitize_str(v, max_length=100, default="")
    
    @field_validator('personalized_explanation', mode='before')
    @classmethod
    def validate_explanation(cls, v):
        return _sanitize_str(v, max_length=500, default="")


class RAGSelectionResult(BaseModel):
    """Structured output for RAG course/material selection."""
    selected_courses: List[SelectedCourseItem] = Field(default_factory=list, description="Selected courses")
    selected_materials: List[SelectedMaterialItem] = Field(default_factory=list, description="Selected materials")
    
    @field_validator('selected_courses', mode='before')
    @classmethod
    def validate_courses(cls, v):
        if not v or not isinstance(v, list):
            return []
        return v[:10]  # Limit to 10 courses
    
    @field_validator('selected_materials', mode='before')
    @classmethod
    def validate_materials(cls, v):
        if not v or not isinstance(v, list):
            return []
        return v[:10]  # Limit to 10 materials


class ValidationItem(BaseModel):
    """A single course validation result."""
    index: int = Field(default=-1, description="Index of the course in the batch")
    is_valid: bool = Field(default=False, description="Whether the course is valid")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(default="No reason provided", description="Explanation")
    
    @field_validator('index', mode='before')
    @classmethod
    def validate_index(cls, v):
        if isinstance(v, (int, float)):
            return max(-1, int(v))
        return -1
    
    @field_validator('confidence', mode='before')
    @classmethod
    def validate_confidence(cls, v):
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.0
    
    @field_validator('reason', mode='before')
    @classmethod
    def validate_reason(cls, v):
        return _sanitize_str(v, max_length=300, default="No reason provided")


class BatchValidationResult(BaseModel):
    """Structured output for batch course validation."""
    validations: List[ValidationItem] = Field(default_factory=list, description="Validation results")
    
    @field_validator('validations', mode='before')
    @classmethod
    def validate_items(cls, v):
        if not v or not isinstance(v, list):
            return []
        return v[:50]  # Limit to 50 validations per batch


class SingleValidationResult(BaseModel):
    """Structured output for single course validation."""
    is_valid: bool = Field(default=False, description="Whether the course is valid")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    reason: str = Field(default="No reason provided", description="Explanation")
    
    @field_validator('confidence', mode='before')
    @classmethod
    def validate_confidence(cls, v):
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.0
    
    @field_validator('reason', mode='before')
    @classmethod
    def validate_reason(cls, v):
        return _sanitize_str(v, max_length=300, default="No reason provided")


# --- Security and Safety Limits (externalize to env) ---
# Remove all local config constants and use centralized config
config = get_agent_config("market_and_course_recommender")
TIMEOUT_SECONDS = config.timeout_seconds
MAX_PROMPT_CHARS = config.max_prompt_chars
MAX_RESPONSE_LENGTH = config.max_response_length
MAX_INTERESTS_COUNT = config.max_interests_count
MAX_RECOMMENDATIONS_COUNT = config.max_recommendations_count
LLM_MODEL = config.llm_model

# Tenant isolation patterns
TENANT_ID_RX = re.compile(r"^[a-zA-Z0-9_\-]{8,64}$")

# Type-safe analysis methods (LLM-only approach)
AnalysisMethod = Literal["llm", "error", "timeout", "llm_timeout"]

# Custom memory class for market and course recommender (extends base memory)
class MarketCourseRecommenderMemory(BaseAgentMemory):
    def __init__(self, tenant_id: str = "default"):
        super().__init__(tenant_id, config.adaptation_window)
        # Add any market recommender specific fields here if needed

# Use centralized memory management
async def get_market_course_recommender_memory(tenant_id: str = "default") -> MarketCourseRecommenderMemory:
    """Get or create tenant-scoped market and course recommender memory."""
    return await get_agent_memory("market_and_course_recommender", tenant_id, MarketCourseRecommenderMemory)

# Split allow-lists for clarity and fewer edge case checks
ALLOWED_ETLD1 = {
    "coursera.org", "edx.org", "udacity.com", "udemy.com", "pluralsight.com",
    "datacamp.com", "khanacademy.org", "deeplearning.ai", "mozilla.org",
    "codecademy.com", "lynda.com", "futurelearn.com", "brilliant.org",
    "freecodecamp.org", "skillshare.com", "youtube.com", "youtube.com",
    "linkedin.com", "mit.edu", "stanford.edu", "harvard.edu", "cambridge.org",
    "oxford.ac.uk", "berkeley.edu", "caltech.edu", "mit.edu", "cmu.edu",
    "github.com", "stackoverflow.com", "w3schools.com", "mdn.mozilla.org",
    "developer.mozilla.org", "docs.python.org", "reactjs.org", "nodejs.org",
    "angular.io", "vuejs.org", "svelte.dev", "nextjs.org", "nuxtjs.org"
}
ALLOWED_HOSTS = {"developers.google.com", "cloud.google.com", "grow.google", "ai.google"}

# Resource allow-list (books, papers, videos, blogs, webpages) in addition to courses
ALLOWED_RESOURCE_ETLD1 = ALLOWED_ETLD1.union({
    "arxiv.org", "ieee.org", "ieeexplore.ieee.org", "dl.acm.org",
    "springer.com", "nature.com", "sciencedirect.com", "doaj.org",
    "oreilly.com", "packtpub.com", "leanpub.com",
    # Blog platforms and educational websites
    "medium.com", "dev.to", "hashnode.com", "freecodecamp.org",
    "css-tricks.com", "smashingmagazine.com", "alistapart.com",
    "sitepoint.com", "tutsplus.com", "web.dev", "developers.google.com",
    "aws.amazon.com", "azure.microsoft.com", "cloud.google.com",
    "realpython.com", "python.org", "javascript.info", "react.dev",
    "vuejs.org", "angular.io", "nextjs.org", "nodejs.org", "mongodb.com",
    "postgresql.org", "mysql.com", "redis.io", "docker.com", "kubernetes.io",
    "terraform.io", "ansible.com", "git-scm.com", "atlassian.com",
    "stackoverflow.blog", "github.blog", "netlify.com", "vercel.com",
    # Documentation and technical sites
    "kubeflow.org", "mlflow.org", "martinfowler.com", "kubeflow.org"
})

# --- Agentic AI Constants ---
CONFIDENCE_THRESHOLD = 0.75   # Minimum confidence for recommendations
ADAPTATION_WINDOW = 50        # Number of recent analyses to consider
MIN_MARKET_INSIGHTS = 3       # Minimum market insights required
MIN_COURSE_RECOMMENDATIONS = 5 # Minimum course recommendations

# Allowed difficulty levels for course validation
DIFFICULTY_ALLOWED = {"Beginner", "Intermediate", "Advanced", "All Levels"}

# PII patterns for redaction (precompiled for performance)
PII_PATTERNS = PII_PATTERNS

INJECTION_FILTERS = INJECTION_FILTERS

# Memory management now handled by centralized system

def _calculate_salary_modifiers_from_assessments(assessment_history: List[Dict[str, Any]], current_assessment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Calculate bounded salary modifiers based on assessment history.
    
    NOTE: This function is kept for logging/reference purposes only.
    The actual salary adjustments are now determined by the LLM based on raw assessment data.
    This provides a reference point for debugging and monitoring.
    
    Args:
        assessment_history: List of past assessment results
        current_assessment: Current assessment result (if any)
    
    Returns:
        Dict with modifiers, total_adjustment_percent, and reasons (for reference only)
    """
    modifiers = []
    total_adjustment = 0.0
    reasons = []
    
    # Combine history with current assessment
    all_assessments = list(assessment_history) if assessment_history else []
    if current_assessment:
        all_assessments.append(current_assessment)
    
    if not all_assessments:
        return {
            "modifiers": [],
            "total_adjustment_percent": 0.0,
            "reasons": ["No assessment history available. Salary based on resume and experience only."],
            "baseline_computation": "resume_and_experience_only"
        }
    
    # Track assessment types and scores
    communication_scores = []
    technical_scores = []
    topic_scores = []
    
    for assessment in all_assessments:
        if not isinstance(assessment, dict):
            continue
            
        topic = str(assessment.get("assessment_topic", "")).lower()
        assessment_type = str(assessment.get("assessment_type", "")).lower()
        total_score = assessment.get("total_score", 0)
        max_score = assessment.get("max_score", 100)
        
        if max_score == 0:
            continue
            
        percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        # Categorize assessments
        is_communication = "communication" in topic or "communication" in assessment_type
        is_technical = any(kw in topic for kw in ["technical", "coding", "programming", "dsa", "algorithm"])
        
        if is_communication:
            communication_scores.append(percentage)
        elif is_technical:
            technical_scores.append(percentage)
        else:
            topic_scores.append(percentage)
    
    # Calculate modifiers: only non-negative adjustments (0 or positive). No reductions.
    # Labels avoid "Fair" with negative %; below-85 bands get "No adjustment" / nudge to improve.
    # Communication skills: 0% to +3% (soft skills; no penalty for low scores)
    if communication_scores:
        avg_comm = sum(communication_scores) / len(communication_scores)
        if avg_comm >= 85:
            adjustment = +3.0
            label = "Excellent"
            reasons.append(f"Communication assessment average: {avg_comm:.1f}% ({label}). Applied +3% soft-skill bonus.")
        elif avg_comm >= 70:
            adjustment = 0.0
            reasons.append(f"Communication assessment average: {avg_comm:.1f}% (Good). No salary adjustment applied.")
        elif avg_comm >= 50:
            adjustment = 0.0
            reasons.append(f"Communication assessment average: {avg_comm:.1f}% (Satisfactory). No salary adjustment. Consider retaking or courses to qualify for a bonus.")
        else:
            adjustment = 0.0
            reasons.append(f"Communication assessment average: {avg_comm:.1f}% (Needs Improvement). No salary adjustment. Consider assessments and courses to improve.")
        
        if adjustment != 0:
            modifiers.append({
                "source": "communication_assessment",
                "score": avg_comm,
                "adjustment_percent": adjustment,
                "label": label,
                "note": "Communication skills assessment",
            })
            total_adjustment += adjustment
    
    # Technical skills: 0% to +5% (technical; no penalty for low scores)
    if technical_scores:
        avg_tech = sum(technical_scores) / len(technical_scores)
        if avg_tech >= 85:
            adjustment = +5.0
            label = "Excellent"
            reasons.append(f"Technical assessment average: {avg_tech:.1f}% ({label}). Applied +5% technical-skill bonus.")
        elif avg_tech >= 70:
            adjustment = 0.0
            reasons.append(f"Technical assessment average: {avg_tech:.1f}% (Good). No salary adjustment applied.")
        elif avg_tech >= 50:
            adjustment = 0.0
            reasons.append(f"Technical assessment average: {avg_tech:.1f}% (Satisfactory). No salary adjustment. Consider retaking or courses to qualify for a bonus.")
        else:
            adjustment = 0.0
            reasons.append(f"Technical assessment average: {avg_tech:.1f}% (Needs Improvement). No salary adjustment. Consider assessments and courses to improve.")
        
        if adjustment != 0:
            modifiers.append({
                "source": "technical_assessment",
                "score": avg_tech,
                "adjustment_percent": adjustment,
                "label": label,
                "note": "Technical skills assessment",
            })
            total_adjustment += adjustment
    
    # Topic-based assessments: 0% to +3% (domain knowledge; no penalty)
    if topic_scores:
        avg_topic = sum(topic_scores) / len(topic_scores)
        if avg_topic >= 85:
            adjustment = +3.0
            label = "Excellent"
            reasons.append(f"Topic assessment average: {avg_topic:.1f}% ({label}). Applied +3% domain-knowledge bonus.")
        elif avg_topic >= 70:
            adjustment = 0.0
            reasons.append(f"Topic assessment average: {avg_topic:.1f}% (Good). No salary adjustment applied.")
        elif avg_topic >= 50:
            adjustment = 0.0
            reasons.append(f"Topic assessment average: {avg_topic:.1f}% (Satisfactory). No salary adjustment. Consider retaking or courses to qualify for a bonus.")
        else:
            adjustment = 0.0
            reasons.append(f"Topic assessment average: {avg_topic:.1f}% (Needs Improvement). No salary adjustment. Consider assessments and courses to improve.")
        
        if adjustment != 0:
            modifiers.append({
                "source": "topic_assessment",
                "score": avg_topic,
                "adjustment_percent": adjustment,
                "label": label,
                "note": "Topic-based assessment",
            })
            total_adjustment += adjustment
    
    # Cap total adjustment to non-negative bounds (0% to +10%)
    total_adjustment = max(0.0, min(10.0, total_adjustment))
    
    if not reasons:
        reasons.append("All assessments show satisfactory performance. Salary based on resume and experience with no adjustments.")
    
    return {
        "modifiers": modifiers,
        "total_adjustment_percent": total_adjustment,
        "reasons": reasons,
        "baseline_computation": "resume_experience_with_assessment_modifiers",
        "assessment_count": len(all_assessments)
    }

def _aggregate_assessment_history(uid: str, current_assessment: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Aggregate assessment history from stored data and current assessment.
    Returns a unified list of all assessments for salary calculation.
    """
    history = []
    
    try:
        # Get stored assessment history - run blocking I/O in thread pool
        # Note: This is a sync function, but it's called from async context
        # For now, we'll keep it sync since it's a helper function
        # If called from async, the caller should wrap it
        assessments_doc = get_assessments_doc(uid) or {}
        
        # Check assessment_evaluator for historical results
        assessment_evaluator_data = assessments_doc.get("assessment_evaluator", {})
        if assessment_evaluator_data:
            # Extract historical assessment results
            historical_results = assessment_evaluator_data.get("assessment_results")
            if historical_results and isinstance(historical_results, dict):
                # If it's a single result, add it
                if "assessment_topic" in historical_results or "total_score" in historical_results:
                    history.append(historical_results)
            elif isinstance(historical_results, list):
                history.extend(historical_results)
        
        # Check for multiple assessment results stored over time
        assessment_history_list = assessments_doc.get("assessment_history", [])
        if isinstance(assessment_history_list, list):
            history.extend(assessment_history_list)
        
    except Exception as e:
        log.warning(f"Failed to retrieve assessment history for UID={uid}: {e}")
    
    # Add current assessment if provided
    if current_assessment and isinstance(current_assessment, dict):
        history.append(current_assessment)
    
    return history

def _calculate_confidence_score(recommendations: Dict[str, Any], method: str) -> float:
    """Calculate confidence score for market analysis and recommendations."""
    if not recommendations:
        return 0.0
    
    score = 0.3  # Base score
    
    # Market insights scoring
    market_insights = recommendations.get('market_insights', [])
    if isinstance(market_insights, list) and len(market_insights) >= MIN_MARKET_INSIGHTS:
        score += 0.2
    
    # Course recommendations scoring
    course_recs = recommendations.get('course_recommendations', [])
    if isinstance(course_recs, list) and len(course_recs) >= MIN_COURSE_RECOMMENDATIONS:
        score += 0.2
    
    # Quality indicators
    if recommendations.get('salary_trends'):
        score += 0.1
    if recommendations.get('career_paths'):
        score += 0.1
    if recommendations.get('skill_demand_analysis'):
        score += 0.1
    
    # Method-specific adjustments
    if method == 'llm':
        score += 0.05  # Slight bonus for LLM comprehensiveness
    elif method == 'hybrid':
        score += 0.1   # Bonus for hybrid approach
    
    return max(0.0, min(score, 1.0))

def _assess_recommendation_quality(recommendations: Dict[str, Any]) -> float:
    """Assess quality of market analysis and course recommendations."""
    if not recommendations:
        return 0.0
    
    quality_score = 0.0
    
    # Structure validation
    expected_keys = ['market_insights', 'course_recommendations', 'salary_trends', 'career_paths']
    present_keys = sum(1 for key in expected_keys if key in recommendations)
    if present_keys > 0:
        quality_score += (present_keys / len(expected_keys)) * 0.4
    
    # Content quality assessment
    market_insights = recommendations.get('market_insights', [])
    if isinstance(market_insights, list):
        for insight in market_insights:
            if isinstance(insight, dict) and insight.get('insight') and insight.get('relevance_score'):
                quality_score += 0.05
    
    course_recs = recommendations.get('course_recommendations', [])
    if isinstance(course_recs, list):
        for course in course_recs:
            if isinstance(course, dict) and course.get('title') and course.get('provider'):
                quality_score += 0.03
    
    return min(quality_score, 1.0)

def _normalize_course_url(url: str) -> str:
    """Normalize course URL to remove query/fragment parameters."""
    if url.startswith("www."):
        url = "https://" + url
    u = urlparse(url)
    # Keep scheme/host/path only
    path = u.path or "/"
    return f"{u.scheme}://{u.hostname}{path}"


def _is_allowed_course_domain(url: str) -> bool:
    """Check if URL domain is from an allowed course provider."""
    try:
        if url.startswith("www."):
            url = "https://" + url
        u = urlparse(url)
        host = (u.hostname or "").lower()
        parts = host.split(".")
        etld1 = ".".join(parts[-2:]) if len(parts) >= 2 else host
        
        if host in ALLOWED_HOSTS or etld1 in ALLOWED_ETLD1:
            return True
        
        # LinkedIn Learning only
        if etld1 == "linkedin.com" and (u.path.startswith("/learning") or host.startswith("learning.")):
            return True
        
        return False
    except Exception:
        return False

def _is_allowed_resource_domain(url: str) -> bool:
    """Allow domains for books, papers, videos, blogs, and webpages."""
    try:
        if url.startswith("www."):
            url = "https://" + url
        u = urlparse(url)
        host = (u.hostname or "").lower()
        parts = host.split(".")
        etld1 = ".".join(parts[-2:]) if len(parts) >= 2 else host
        
        # Check against allowed hosts and domains
        if host in ALLOWED_HOSTS or etld1 in ALLOWED_RESOURCE_ETLD1:
            return True
        
        # YouTube specific
        if etld1 == "youtube.com" and (u.path.startswith("/watch") or u.path.startswith("/playlist")):
            return True
        
        # Additional blog and educational website patterns
        # Allow subdomains of known educational platforms
        if any(domain in host for domain in ["medium.com", "dev.to", "hashnode.com", "freecodecamp.org"]):
            return True
        
        # Allow documentation and developer sites
        if any(domain in host for domain in [
            "developers.google.com", "web.dev", "realpython.com", 
            "javascript.info", "react.dev", "vuejs.org", "angular.io",
            "mongodb.com", "postgresql.org", "redis.io", "docker.com"
        ]):
            return True
        
        return False
    except Exception:
        return False

def _redact_urls_keep_providers(text: str) -> str:
    """Redact all URLs except allow-listed course providers."""
    def repl(m):
        url = m.group(0).rstrip('.,);]!?\'"')
        if _is_allowed_course_domain(url):
            return _normalize_course_url(url)
        return "[REDACTED_URL]"
    return re.sub(r'https?://[^\s)]+|\bwww\.[^\s)]+\b', repl, text)

def _norm_duration(s: str) -> str:
    """Normalize course duration string."""
    return s.strip()[:100]

def _dedupe_courses(courses: List[dict]) -> List[dict]:
    """Remove near-duplicate course recommendations based on title and type (no URL)."""
    seen: Set[Tuple[str, str]] = set()
    out = []
    for c in courses:
        k = ((c.get('title') or '').strip().lower(), (c.get('type') or 'course').strip().lower())
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out

def _fallback_courses_for_skills(missing_skills: List[str]) -> List[Dict[str, Any]]:
    """Deterministic, provider-safe course suggestions mapped from missing skills.
    Only uses allow-listed providers with valid course URL patterns.
    """
    if not isinstance(missing_skills, list):
        missing_skills = []

    skill_map: Dict[str, List[Dict[str, Any]]] = {
        "instructional design": [
            {
                "title": "Instructional Design Foundations and Applications",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/instructional-design-foundations-applications",
                "difficulty": "Intermediate",
                "duration": "4 weeks",
                "relevance_score": 0.9,
                "description": "Core models and methods for designing effective learning."
            },
            {
                "title": "Learning How to Learn",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/learning-how-to-learn",
                "difficulty": "Beginner",
                "duration": "4 weeks",
                "relevance_score": 0.85,
                "description": "Evidence-based techniques to design and sequence learning."
            }
        ],
        "adult learning": [
            {
                "title": "Adult Learning: Theories and Principles",
                "provider": "edX",
                "url": "https://www.edx.org/course/adult-learning-theories-and-principles",
                "difficulty": "Intermediate",
                "duration": "4-6 weeks",
                "relevance_score": 0.85,
                "description": "Apply andragogy to course and training design."
            },
            {
                "title": "Creating Engaging Online Courses",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/creating-engaging-online-courses-2",
                "difficulty": "All Levels",
                "duration": "2-3 hours",
                "relevance_score": 0.8,
                "description": "Practical strategies for adult learner engagement online."
            }
        ],
        "articulate storyline": [
            {
                "title": "Articulate Storyline 360: The Complete Beginner's Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/articulate-storyline-360-the-complete-beginners-course/",
                "difficulty": "Beginner",
                "duration": "8-10 hours",
                "relevance_score": 0.85,
                "description": "Hands-on authoring with Storyline for interactive e-learning."
            }
        ],
        "adobe captivate": [
            {
                "title": "Adobe Captivate: The Complete Course",
                "provider": "Udemy",
                "url": "https://www.udemy.com/course/adobe-captivate-the-complete-course/",
                "difficulty": "Intermediate",
                "duration": "8-10 hours",
                "relevance_score": 0.8,
                "description": "Build professional e-learning with Captivate."
            }
        ],
        "lms": [
            {
                "title": "Learning Management Systems (LMS) Fundamentals",
                "provider": "edX",
                "url": "https://www.edx.org/learn/learning-management-systems/university-of-michigan-learning-management-systems-fundamentals",
                "difficulty": "Beginner",
                "duration": "3-4 weeks",
                "relevance_score": 0.8,
                "description": "Plan, configure, and operate LMS for delivery and tracking."
            }
        ],
        "blended learning": [
            {
                "title": "Blended Learning Design",
                "provider": "edX",
                "url": "https://www.edx.org/learn/blended-learning/university-of-pennsylvania-blended-learning-design",
                "difficulty": "Intermediate",
                "duration": "4 weeks",
                "relevance_score": 0.75,
                "description": "Design effective hybrid instructional experiences."
            }
        ],
        "gamification": [
            {
                "title": "Gamification for Learning",
                "provider": "LinkedIn Learning",
                "url": "https://www.linkedin.com/learning/gamification-for-learning",
                "difficulty": "All Levels",
                "duration": "1-2 hours",
                "relevance_score": 0.7,
                "description": "Use game mechanics to drive motivation and outcomes."
            }
        ],
        "curriculum development": [
            {
                "title": "Curriculum Design and Development",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/curriculum-design-development",
                "difficulty": "Intermediate",
                "duration": "4 weeks",
                "relevance_score": 0.9,
                "description": "Systematic design aligned to outcomes and assessments."
            }
        ],
        "educational technology": [
            {
                "title": "Emerging Trends & Technologies in the Virtual K-12 Classroom",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/virtual-k12-classroom",
                "difficulty": "Intermediate",
                "duration": "4 weeks",
                "relevance_score": 0.75,
                "description": "Modern tools and practices for tech-enabled learning."
            }
        ],
        "assessment design": [
            {
                "title": "Assessment and Teaching of 21st Century Skills",
                "provider": "Coursera",
                "url": "https://www.coursera.org/learn/assessment-teaching-21st-century-skills",
                "difficulty": "Intermediate",
                "duration": "4 weeks",
                "relevance_score": 0.75,
                "description": "Design valid and reliable assessments for learning."
            }
        ]
    }

    collected: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    for raw_skill in missing_skills[:6]:  # cap breadth
        key = str(raw_skill or "").strip().lower()
        # fuzzy contains matching
        for k, items in skill_map.items():
            if k in key:
                for it in items:
                    sig = (it["url"].lower(), it["title"].strip().lower())
                    if sig in seen:
                        continue
                    # Ensure strict URL validator will pass later
                    if _validate_course_url_strict(it["url"]):
                        collected.append(dict(it))
                        seen.add(sig)
        if len(collected) >= 8:
            break

    # If nothing matched, provide a general instructional design starter set
    if not collected:
        defaults = skill_map.get("instructional design", []) + skill_map.get("adult learning", [])
        for it in defaults:
            if _validate_course_url_strict(it["url"]):
                collected.append(dict(it))
            if len(collected) >= 6:
                break

    return collected[:MAX_RECOMMENDATIONS_COUNT]

async def _search_courses_for_skills(skills: List[str], max_results: int, log_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Use web search (SerpAPI) to find real course URLs for given skills and return only accessible links.
    Requires SERPAPI_API_KEY. If unavailable or errors, returns empty list.
    
    NOTE: SerpAPI is paid. This function is disabled by default to avoid costs.
    Only works if SERPAPI_API_KEY is explicitly set (user has paid account).
    """
    # Check if SerpAPI is available and configured
    if not SerpAPIWrapper or not isinstance(skills, list) or not skills:
        return []
    
    # Only use if API key is explicitly set (user has paid account)
    if not os.getenv("SERPAPI_API_KEY"):
        AgentLogger.log_info(log_context, "SerpAPI not configured - skipping web search for courses (to avoid costs)")
        return []
    
    try:
        serp = SerpAPIWrapper()
    except Exception as e:
        AgentLogger.log_warning(log_context, f"SerpAPI unavailable: {e}")
        return []
    queries: List[str] = []
    providers = [
        ("site:coursera.org/learn", "Coursera"),
        ("site:edx.org/course", "edX"),
        ("site:udemy.com/course", "Udemy"),
        ("site:linkedin.com/learning", "LinkedIn Learning"),
    ]
    for skill in skills[:6]:
        s = str(skill).strip()
        if not s:
            continue
        for site, _ in providers:
            queries.append(f"{site} {s}")
    results: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for q in queries:
        try:
            raw = serp.results(q)
            organic = (raw or {}).get("organic_results") or []
            for item in organic:
                url = (item.get("link") or "").strip()
                title = (item.get("title") or "").strip()
                if not url or not title:
                    continue
                if url in seen:
                    continue
                if not _validate_course_url_strict(url):
                    continue
                # Infer provider from host
                host = urlparse(url).hostname or ""
                provider = ""
                if "coursera.org" in host:
                    provider = "Coursera"
                elif "edx.org" in host:
                    provider = "edX"
                elif "udemy.com" in host:
                    provider = "Udemy"
                elif "linkedin.com" in host:
                    provider = "LinkedIn Learning"
                else:
                    continue
                # Build course object
                course = {
                    "title": title[:200],
                    "provider": provider,
                    "url": _normalize_course_url(url),
                    "difficulty": "All Levels",
                    "duration": "",
                    "relevance_score": 0.7,
                    "description": ""
                }
                results.append(course)
                seen.add(url)
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break
        except Exception as e:
            AgentLogger.log_warning(log_context, f"SerpAPI query failed for '{q}': {e}")
            continue
    # Validate accessibility and keep only working links
    if results:
        validated = await _validate_urls_batch_async(results, log_context)
        # Keep only those marked valid+accessible
        out: List[Dict[str, Any]] = []
        for c in validated:
            uv = c.get("url_validation", {})
            if uv.get("status") == "valid" and uv.get("accessible") is True:
                out.append(c)
        return out
    return []

def _validate_tenant_isolation(tenant_id: str) -> bool:
    """Validate tenant isolation requirements"""
    return bool(tenant_id and TENANT_ID_RX.fullmatch(tenant_id))

def _validate_recommendation_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize recommendation data."""
    if not isinstance(data, dict):
        return {}
    
    validated_data = {}
    
    # Validate market insights
    market_insights = data.get('market_insights', [])
    if isinstance(market_insights, list):
        validated_insights = []
        for insight in market_insights[:15]:  # Cap insights
            if isinstance(insight, dict):
                validated_insight = {}
                if insight.get('insight') and isinstance(insight['insight'], str):
                    validated_insight['insight'] = insight['insight'][:500]  # Cap length
                if insight.get('relevance_score') is not None and isinstance(insight['relevance_score'], (int, float)):
                    validated_insight['relevance_score'] = max(0.0, min(float(insight['relevance_score']), 1.0))
                if insight.get('trend'):
                    validated_insight['trend'] = str(insight['trend'])[:100]
                if validated_insight:
                    validated_insights.append(validated_insight)
        validated_data['market_insights'] = validated_insights
    
    # Validate course recommendations
    course_recs = data.get('course_recommendations', [])
    if isinstance(course_recs, list):
        validated_courses = []
        for course in course_recs[:MAX_RECOMMENDATIONS_COUNT]:
            if isinstance(course, dict):
                validated_course = {}
                
                # Require both title and provider for course validity
                has_title = course.get('title') and isinstance(course['title'], str)
                has_provider = course.get('provider') and isinstance(course['provider'], str)
                
                if has_title:
                    validated_course['title'] = course['title'][:200]
                if has_provider:
                    validated_course['provider'] = course['provider'][:100]
                if course.get('url') and isinstance(course['url'], str):
                    # Safe URL validation - only allow trusted course providers
                    url = course['url'][:500]
                    if url.startswith(('http://', 'https://')) and _is_allowed_course_domain(url):
                        validated_course['url'] = _normalize_course_url(url)
                
                # No fallback URLs - only accept valid URLs
                if course.get('difficulty') and isinstance(course['difficulty'], str):
                    # Validate and normalize difficulty levels
                    d = str(course['difficulty']).strip().title()
                    if d in DIFFICULTY_ALLOWED:
                        validated_course['difficulty'] = d
                if course.get('duration') and isinstance(course['duration'], str):
                    validated_course['duration'] = _norm_duration(course['duration'])
                if course.get('relevance_score') is not None and isinstance(course['relevance_score'], (int, float)):
                    validated_course['relevance_score'] = max(0.0, min(float(course['relevance_score']), 1.0))
                
                # Only keep courses that have at least title+provider
                if has_title and has_provider:
                    validated_courses.append(validated_course)
        validated_data['course_recommendations'] = validated_courses
    
    # Validate other fields
    for field in ['salary_trends', 'career_paths', 'skill_demand_analysis']:
        if field in data and data[field]:
            if isinstance(data[field], (str, dict, list)):
                validated_data[field] = data[field]
    
    return validated_data

def _map_courses_for_output(courses: List[dict]) -> List[dict]:
    """Map to required keys: course (title), platform (provider), type, difficulty, duration, relevance_score, description, rationale, target_skill, author, url."""
    if not isinstance(courses, list):
        return []
    mapped: List[dict] = []
    for c in courses:
        if not isinstance(c, dict):
            continue
        course_name = c.get('course') or c.get('title')
        platform = c.get('platform') or c.get('provider')
        difficulty = c.get('difficulty')
        duration = c.get('duration')
        relevance = c.get('relevance_score')
        description = c.get('description')
        rationale = (c.get('rationale') or c.get('personalized_explanation') or '').strip()
        target_skill = c.get('target_skill')
        rtype = c.get('type')
        author = c.get('author')
        url = c.get('url') or c.get('url_validation', {}).get('url', '')
        
        # Build output (include URL)
        ordered = {}
        if course_name is not None:
            ordered['course'] = course_name
        if platform is not None:
            ordered['platform'] = platform
        if rtype is not None:
            ordered['type'] = rtype
        if difficulty is not None:
            ordered['difficulty'] = difficulty
        if duration is not None:
            ordered['duration'] = duration
        if relevance is not None:
            ordered['relevance_score'] = relevance
        if description is not None:
            ordered['description'] = description
        if rationale:
            ordered['rationale'] = rationale
        if target_skill is not None:
            ordered['target_skill'] = target_skill
        if author is not None:
            ordered['author'] = author
        if url:
            ordered['url'] = url
        mapped.append(ordered)
    return mapped

def _filter_career_advice_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out career advice data to avoid duplication in market analysis output."""
    if not isinstance(data, dict):
        return data
    
    # Fields that belong to career advice and should be filtered out
    career_advice_fields = {
        'skill_categories', 'career_paths', 'missing_skills', 
        'improvement_recommendations', 'skill_proficiency_assessment',
        # Also filter camelCase variants that may come from upstream
        'missingSkills', 'careerAdvice'
    }
    
    filtered_data = {}
    for key, value in data.items():
        if key not in career_advice_fields:
            filtered_data[key] = value
    
    return filtered_data

def _generate_enhanced_course_recommendation_prompt(
    structured_resume: Dict[str, Any],
    user_interests: List[Dict[str, Any]],
    skill_gap_analysis: Dict[str, Any]
) -> str:
    """ULTRA-OPTIMIZED: Course recommendation prompt with minimal tokens for faster processing."""
    
    # Extract essential data with minimal context for token optimization
    # Ensure skill_gap_analysis is a dictionary
    if isinstance(skill_gap_analysis, str):
        try:
            skill_gap_analysis = json.loads(skill_gap_analysis)
        except (json.JSONDecodeError, TypeError):
            skill_gap_analysis = {}
    elif not isinstance(skill_gap_analysis, dict):
        skill_gap_analysis = {}
    
    missing_skills = get_missing_skills_flat(skill_gap_analysis)[:3]  # Top 3 only (reduced from 4)
    career_advice = skill_gap_analysis.get('career_paths', [])[:1]  # Top 1 only (reduced from 2)

    # Derive performance-aware context from embedded assessment_performance (if available)
    assessment_perf = skill_gap_analysis.get('assessment_performance', {}) if isinstance(skill_gap_analysis, dict) else {}
    score = None
    performance_level = None
    weak_sections: Dict[str, Any] = {}
    strong_sections: Dict[str, Any] = {}
    last_topic = None

    if isinstance(assessment_perf, dict):
        score = assessment_perf.get('total_score')
        section_scores = assessment_perf.get('section_scores', {}) or {}
        weak_sections = {k: v for k, v in section_scores.items() if isinstance(v, (int, float)) and v < 50}
        strong_sections = {k: v for k, v in section_scores.items() if isinstance(v, (int, float)) and v >= 80}
        last_topic = assessment_perf.get('assessment_topic')

        if isinstance(score, (int, float)):
            if score >= 90:
                performance_level = "Outstanding"
            elif score >= 80:
                performance_level = "Excellent"
            elif score >= 70:
                performance_level = "Good"
            elif score >= 50:
                performance_level = "Fair"
            else:
                performance_level = "Needs Improvement"

    performance_context = {
        "score": score,
        "performance_level": performance_level,
        "weak_sections": weak_sections,
        "strong_sections": strong_sections,
        "last_topic": last_topic,
    }
    
    # Compress resume to minimal key fields
    # Ensure structured_resume is a dictionary
    if isinstance(structured_resume, str):
        try:
            structured_resume = json.loads(structured_resume)
        except (json.JSONDecodeError, TypeError):
            structured_resume = {}
    elif not isinstance(structured_resume, dict):
        structured_resume = {}
    
    # Prefer missing skills for targeting; fallback to top resume skills
    resume_skills = structured_resume.get("skills", [])
    target_skills = missing_skills if missing_skills else resume_skills[:6]

    key_resume = {
        "skills": resume_skills[:6],  # Top 6 skills only
        "experience_years": len(structured_resume.get("work_experience", [])),
        "target_roles": structured_resume.get("target_roles", [])[:1],  # Top 1 role only
        "education": structured_resume.get("education", [])[:1]  # Top 1 education only
    }
    
    base_prompt = f"""Generate learning recommendations for skill development.

**Profile:**
{json.dumps(key_resume, indent=1)}

**Skill Targets:**
- Missing/Target Skills: {json.dumps(target_skills, indent=1)}
- Career Paths: {json.dumps(career_advice, indent=1)}

**Requirements:**
1. Recommend 8-12 real online learning items (mix of courses, books, papers, videos, blogs, and webpages) addressing the candidate's missing/target skills (works for any job profile). Include at least 2 books (title + author) and at least 2-3 blogs or webpages with educational content.
2. **NO URLs REQUIRED**: Do NOT include URLs for any resources. Only provide:
   - "type": one of "course", "book", "paper", "video", "blog", or "webpage"
   - "title": a descriptive title
   - "provider": platform name (REQUIRED for courses, optional for blogs/webpages)
   - "author": author name (REQUIRED for books, optional for blogs/webpages)
3. Use difficulty labels "Easy", "Medium", and "Hard". Adjust course difficulty based on the candidate's assessment score and performance_level.
4. Include hands-on projects wherever possible.
5. Set "target_skill" to the most relevant missing/target skill.
6. For high performers (Excellent/Outstanding), emphasize specialization and advanced/Hard tracks. For Fair/Needs Improvement, emphasize Easy/Medium foundational and remedial content.

**Platforms (Allowed)**:
  - Courses: Coursera, Udemy, edX, LinkedIn Learning
  - Papers: arXiv, IEEE Xplore, ACM Digital Library
  - Books: O'Reilly, Packt, Leanpub
  - Videos: YouTube
  - Blogs: Medium, Dev.to, Hashnode, FreeCodeCamp, CSS-Tricks, Smashing Magazine, A List Apart, SitePoint, Real Python, JavaScript.info, and other educational blogs
  - Webpages: Official documentation sites (React.dev, Vue.js, Angular.io, Node.js, MongoDB, PostgreSQL, etc.), developer blogs (Google Developers, AWS, Azure, GitHub Blog, Stack Overflow Blog), and educational resource sites (MDN, W3Schools, etc.)

**IMPORTANT - Response Format:**
You MUST return a valid JSON object with a "course_recommendations" array. Each item in the array must have at minimum:
- "type": one of "course", "book", "paper", "video", "blog", or "webpage"
- "title": a descriptive title
- "provider": platform name (REQUIRED for courses, optional for blogs/webpages)
- "author": author name (REQUIRED for books, optional for blogs/webpages)
- DO NOT include "url" field - URLs are not required

**JSON Format (CRITICAL - Follow exactly):**
{{
    "course_recommendations": [
        {{
            "type": "course",
            "title": "Complete Python Bootcamp",
            "provider": "Udemy",
            "difficulty": "Medium",
            "duration": "6 weeks",
            "relevance_score": 0.9,
            "target_skill": "Python",
            "description": "Learn Python from scratch"
        }},
        {{
            "type": "book",
            "title": "Clean Code",
            "author": "Robert C. Martin",
            "difficulty": "Medium",
            "relevance_score": 0.85,
            "target_skill": "Software Engineering",
            "description": "A handbook of agile software craftsmanship"
        }},
        {{
            "type": "blog",
            "title": "Understanding React Hooks",
            "provider": "Medium",
            "author": "John Doe",
            "difficulty": "Easy",
            "relevance_score": 0.8,
            "target_skill": "React",
            "description": "A comprehensive guide to React Hooks"
        }}
    ]
}}

Return ONLY valid JSON. Do not include any text before or after the JSON object."""

    # Append assessment-performance-specific guidance when available
    if performance_level:
        performance_block = f"""

Assessment Performance Context: {json.dumps(performance_context, default=str)}

Assessment Performance Insights:
- Topic Assessed: {last_topic}
- Score: {score}
- Performance Level: {performance_level}
- Weak Sections: {weak_sections}
- Strong Sections: {strong_sections}

Adjust all course recommendations using the performance insight:
- Excellent/Outstanding → Recommend primarily advanced/specialization courses (difficulty "Hard") with some "Medium".
- Good → Recommend mostly "Medium" difficulty structured learning paths with a few "Hard" items.
- Fair → Recommend practice-focused intermediate ("Medium") courses with some "Easy" reinforcement.
- Needs Improvement → Recommend remedial and foundational ("Easy") courses before anything "Medium".
- If MCQ sections are weak → include analytical/psychometric and reasoning-focused courses.
- If short_answer sections are weak → include communication and technical-writing courses.
- If long_answer sections are weak → include coding, algorithms, and system-design courses.
- Focus course topics around the last assessed skill/topic: {last_topic}.
"""
        return base_prompt + performance_block

    return base_prompt

def _extract_queries_from_llm_response(llm_response: Any, log_context: Dict[str, Any]) -> List[str]:
    """
    Convert LLM response to list of query strings.
    Handles any format: dict, list, string, etc.
    
    Args:
        llm_response: LLM response in any format
        log_context: Logging context
        
    Returns:
        List of query strings (empty list if conversion fails)
    """
    queries = []
    original_string = llm_response if isinstance(llm_response, str) else None
    
    # Step 1: Parse JSON if it's a string
    if isinstance(llm_response, str):
        AgentLogger.log_debug(log_context, f"LLM response is string (length: {len(llm_response)})")
        # Try direct JSON parse first
        try:
            parsed = json.loads(llm_response)
            AgentLogger.log_debug(log_context, f"✅ Direct JSON parse successful: {type(parsed).__name__}")
            llm_response = parsed
        except json.JSONDecodeError as e:
            AgentLogger.log_debug(log_context, f"Direct JSON parse failed at pos {e.pos}: {e.msg}")
            # Try safe_json_loads
            try:
                parsed = _safe_json_loads(llm_response)
                if parsed:
                    AgentLogger.log_debug(log_context, "✅ Safe JSON parse successful")
                    llm_response = parsed
                else:
                    # Try regex to extract JSON object
                    import re
                    json_match = re.search(r'(\{.*\})', llm_response, re.DOTALL)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group(1))
                            AgentLogger.log_debug(log_context, "✅ Regex JSON extraction successful")
                            llm_response = parsed
                        except Exception as e2:
                            AgentLogger.log_debug(log_context, f"Regex JSON extraction failed: {e2}")
            except Exception as e3:
                AgentLogger.log_debug(log_context, f"Safe JSON parse exception: {e3}")
    
    # Step 2: Extract queries from any format
    if isinstance(llm_response, list):
        # Direct list - extract strings from it
        for item in llm_response:
            if isinstance(item, str) and item.strip():
                queries.append(item.strip())
            elif isinstance(item, dict):
                # Try common keys
                query = item.get("query") or item.get("text") or item.get("search_query") or item.get("q")
                if query and isinstance(query, str):
                    queries.append(query.strip())
            elif item:
                queries.append(str(item).strip())
    
    elif isinstance(llm_response, dict):
        # Dict - look for queries in common keys
        queries_list = llm_response.get("search_queries") or llm_response.get("queries") or llm_response.get("query_list")
        
        if queries_list and isinstance(queries_list, list):
            for item in queries_list:
                if isinstance(item, str) and item.strip():
                    queries.append(item.strip())
                elif isinstance(item, dict):
                    query = item.get("query") or item.get("text") or item.get("search_query")
                    if query and isinstance(query, str):
                        queries.append(query.strip())
                elif item:
                    queries.append(str(item).strip())
        elif queries_list and isinstance(queries_list, str):
            queries.append(queries_list.strip())
    
    # Step 3: If still no queries, try aggressive extraction from original string
    if not queries and original_string:
        AgentLogger.log_debug(log_context, f"No queries found in parsed response, trying aggressive extraction from original string (length: {len(original_string)})...")
        import re
        
        # Method 1: Extract individual query strings directly - works even with truncated JSON
        # Look for "search_queries": [ and extract all quoted strings that follow (even if array is not closed)
        # Pattern: "search_queries": [ "query1", "query2", ...
        array_start_match = re.search(r'"(?:search_queries|queries)"\s*:\s*\[', original_string, re.IGNORECASE)
        if array_start_match:
            # Find the position where the array starts
            array_start_pos = array_start_match.end()
            # Extract everything after the opening bracket
            array_content = original_string[array_start_pos:]
            # Extract all quoted strings from the array content (works even if JSON is truncated)
            string_matches = re.findall(r'"([^"]+)"', array_content)
            if string_matches:
                queries = [q.strip() for q in string_matches if q.strip()]
                if queries:
                    AgentLogger.log_info(log_context, f"🤖 Extracted {len(queries)} queries using direct string extraction (handles truncated JSON)")
        
        # Method 2: If that failed, try to find and parse complete array
        if not queries:
            array_match = re.search(r'"(?:search_queries|queries)"\s*:\s*(\[.*?\])', original_string, re.DOTALL)
            if array_match:
                try:
                    queries_array = json.loads(array_match.group(1))
                    if isinstance(queries_array, list):
                        queries = [str(q).strip() for q in queries_array if q and str(q).strip()]
                        if queries:
                            AgentLogger.log_info(log_context, f"🤖 Extracted {len(queries)} queries using array JSON parsing")
                except Exception as e:
                    AgentLogger.log_debug(log_context, f"Array JSON parsing failed: {e}")
    
    # Step 4: Log result
    if queries:
        AgentLogger.log_info(log_context, f"🤖 Extracted {len(queries)} queries from LLM response")
        AgentLogger.log_debug(log_context, f"Queries: {queries[:3]}")
    else:
        AgentLogger.log_warning(log_context, f"⚠️ Could not extract queries from LLM response (type: {type(llm_response).__name__})")
        AgentLogger.log_debug(log_context, f"LLM response preview: {str(llm_response)[:500]}")
    
    return queries

# =============================
# Auto-Discovery Functions
# =============================

def _evaluate_material_availability(
    results: List[Dict[str, Any]],
    min_count: int = 5,
    min_avg_similarity: float = 0.4
) -> bool:
    """
    Check if available materials are sufficient.
    
    Args:
        results: List of material results from KB search
        min_count: Minimum number of results required
        min_avg_similarity: Minimum average similarity score required
        
    Returns:
        True if materials are sufficient, False otherwise
    """
    if len(results) < min_count:
        return False
    
    if results:
        similarities = [r.get("similarity_score", 0) for r in results if r.get("similarity_score")]
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            if avg_similarity < min_avg_similarity:
                return False
    
    return True


def _infer_provider_from_url(url: str, material_type: str) -> str:
    """Infer provider/platform from URL"""
    url_lower = url.lower()
    
    provider_map = {
        "coursera.org": "Coursera",
        "udemy.com": "Udemy",
        "edx.org": "edX",
        "linkedin.com": "LinkedIn Learning",
        "oreilly.com": "O'Reilly",
        "packtpub.com": "Packt",
        "amazon.com": "Amazon",
        "arxiv.org": "arXiv",
        "ieee.org": "IEEE",
        "dl.acm.org": "ACM",
        "youtube.com": "YouTube",
        "vimeo.com": "Vimeo",
        "developer.mozilla.org": "MDN",
        "w3schools.com": "W3Schools",
        "realpython.com": "Real Python",
    }
    
    for domain, provider in provider_map.items():
        if domain in url_lower:
            return provider
    
    return "Other"


def _extract_skills_from_text(text: str) -> List[str]:
    """Extract skills/keywords from text (basic implementation)"""
    # Common tech skills to look for
    common_skills = [
        "python", "javascript", "react", "node.js", "java", "sql", "aws",
        "docker", "kubernetes", "machine learning", "data science", "ai",
        "typescript", "angular", "vue", "spring", "django", "flask",
        "mongodb", "postgresql", "mysql", "redis", "elasticsearch",
        "terraform", "ansible", "jenkins", "git", "ci/cd", "devops",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy"
    ]
    
    found_skills = []
    text_lower = text.lower()
    for skill in common_skills:
        if skill in text_lower:
            found_skills.append(skill.title())
    
    return found_skills[:5]  # Limit to 5


def _infer_difficulty_from_text(text: str) -> str:
    """Infer difficulty level from text"""
    text_lower = text.lower()
    
    beginner_keywords = ["beginner", "intro", "basics", "getting started", "fundamentals", "101"]
    advanced_keywords = ["advanced", "expert", "master", "deep dive", "production", "enterprise"]
    
    if any(word in text_lower for word in beginner_keywords):
        return "Beginner"
    elif any(word in text_lower for word in advanced_keywords):
        return "Advanced"
    else:
        return "Intermediate"


def _validate_material(material: Dict[str, Any]) -> bool:
    """Validate material before adding to KB"""
    # Must have title and URL
    if not material.get("title") or not material.get("url"):
        return False
    
    # URL must be valid
    url = material.get("url", "")
    if not url.startswith(("http://", "https://")):
        return False
    
    # Title should not be too short
    if len(material.get("title", "").strip()) < 5:
        return False
    
    return True


def _extract_material_from_search_result(
    search_result: Dict[str, Any],
    material_type: str,
    topic: str
) -> Optional[Dict[str, Any]]:
    """
    Extract structured material data from search result.
    
    Args:
        search_result: Raw search result from SerpAPI
        material_type: Type of material (course/book/paper/video/tutorial)
        topic: Topic/skill being searched for
        
    Returns:
        Structured material dict or None if invalid
    """
    url = search_result.get("link", "").strip()
    title = search_result.get("title", "").strip()
    snippet = search_result.get("snippet", "").strip()
    
    if not url or not title:
        return None
    
    # Infer provider from URL
    provider = _infer_provider_from_url(url, material_type)
    
    # Extract skills (use topic as primary skill)
    skills = [topic] + _extract_skills_from_text(f"{title} {snippet}")
    
    # Infer difficulty
    difficulty = _infer_difficulty_from_text(f"{title} {snippet}")
    
    return {
        "title": title[:200],
        "provider": provider,
        "url": url,
        "type": material_type,
        "description": snippet[:500] if snippet else f"Learn {topic} with this {material_type}",
        "skills": skills[:10],
        "difficulty": difficulty,
        "duration": "",  # Will be empty for non-courses
        "source": "auto_discovered"
    }


async def _search_online_materials(
    topic: str,
    material_types: List[str],
    max_results_per_type: int = 5,
    log_context: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Search online for materials using multiple sources.
    
    Now includes:
    - Academic APIs (Google Scholar, arXiv, IEEE, PubMed) for papers
    - Partner APIs (Coursera, Udemy, edX) for courses
    - Web search (SerpAPI) for all material types
    
    Args:
        topic: Topic/skill to search for
        material_types: List of material types to search (course/book/paper/video/tutorial)
        max_results_per_type: Maximum results per material type
        log_context: Logging context
        
    Returns:
        List of discovered materials
    """
    all_materials = []
    
    # NEW: Use Academic APIs for papers (priority)
    if "paper" in material_types:
        try:
            from agents.academic_api_adapters import AcademicAPIManager
            
            academic_manager = AcademicAPIManager()
            AgentLogger.log_info(
                log_context,
                f"📚 Searching academic APIs for papers on '{topic}'..."
            )
            
            # Search all academic sources (free APIs only - no SerpAPI required)
            # Use arXiv, PubMed, and Crossref (all free)
            free_sources = ["arxiv", "pubmed", "crossref"]
            papers = await academic_manager.search_all_sources(
                query=topic,
                sources=free_sources,
                max_results_per_source=max_results_per_type // len(free_sources)  # Split across sources
            )
            
            # Convert papers to material format
            for paper in papers[:max_results_per_type]:
                material = {
                    "title": paper.get("title", ""),
                    "url": paper.get("url", ""),
                    "type": "paper",
                    "provider": paper.get("source", "Academic"),
                    "author": ", ".join(paper.get("authors", [])),
                    "description": paper.get("abstract", ""),
                    "difficulty": "Advanced",  # Papers are typically advanced
                    "skills": [topic]
                }
                if _validate_material(material):
                    all_materials.append(material)
            
            if papers:
                AgentLogger.log_info(
                    log_context,
                    f"✅ Found {len(papers)} papers from academic APIs"
                )
        except Exception as e:
            AgentLogger.log_warning(
                log_context,
                f"Academic API search failed: {e}. Falling back to web search."
            )
    
    # NEW: Use Book APIs for books (priority)
    if "book" in material_types:
        try:
            from agents.academic_api_adapters import AcademicAPIManager
            
            academic_manager = AcademicAPIManager()
            AgentLogger.log_info(
                log_context,
                f"📖 Searching book APIs for books on '{topic}'..."
            )
            
            # Search book sources
            books = await academic_manager.search_books(
                query=topic,
                sources=["google_books", "open_library", "wikibooks", "crossref"],
                max_results_per_source=max_results_per_type // 2  # Split across sources
            )
            
            # Convert books to material format
            for book in books[:max_results_per_type]:
                material = {
                    "title": book.get("title", ""),
                    "url": book.get("url", ""),
                    "type": "book",
                    "provider": book.get("source", "Book API"),
                    "author": ", ".join(book.get("authors", [])),
                    "description": book.get("description", ""),
                    "difficulty": "Intermediate",  # Books are typically intermediate
                    "skills": [topic],
                    "isbn": book.get("isbn"),
                    "publisher": book.get("publisher", ""),
                    "year": book.get("year")
                }
                if _validate_material(material):
                    all_materials.append(material)
            
            if books:
                AgentLogger.log_info(
                    log_context,
                    f"✅ Found {len(books)} books from book APIs"
                )
        except Exception as e:
            AgentLogger.log_warning(
                log_context,
                f"Book API search failed: {e}. Falling back to web search."
            )
    
    # Search queries for each material type (fallback/web search)
    search_queries = {
        "course": [
            f"{topic} course site:coursera.org",
            f"{topic} course site:udemy.com",
            f"{topic} course site:edx.org",
            f"{topic} course site:linkedin.com/learning"
        ],
        "book": [
            f"{topic} book site:oreilly.com",
            f"{topic} book site:packtpub.com",
            f"{topic} book site:amazon.com"
        ],
        "paper": [
            f"{topic} site:arxiv.org",
            f"{topic} site:ieee.org",
            f"{topic} site:dl.acm.org"
        ],
        "video": [
            f"{topic} tutorial site:youtube.com",
            f"{topic} tutorial site:vimeo.com"
        ],
        "tutorial": [
            f"{topic} tutorial site:developer.mozilla.org",
            f"{topic} tutorial site:w3schools.com",
            f"{topic} tutorial site:realpython.com"
        ]
    }
    
    # Web search fallback (optional - only if SerpAPI is available and free)
    # Note: SerpAPI is paid, so we skip web search to avoid costs
    # The free APIs (arXiv, PubMed, Crossref, Google Books, Open Library, Wikibooks) should be sufficient
    if SerpAPIWrapper and os.getenv("SERPAPI_API_KEY"):
        # Only use SerpAPI if explicitly configured (user has paid account)
        try:
            serp = SerpAPIWrapper()
            
            for material_type in material_types:
                # Skip papers if we already got them from academic APIs
                if material_type == "paper" and any(m.get("type") == "paper" for m in all_materials):
                    AgentLogger.log_info(
                        log_context,
                        "Skipping web search for papers (already found via academic APIs)"
                    )
                    continue
                
                # Skip books if we already got them from book APIs
                if material_type == "book" and any(m.get("type") == "book" for m in all_materials):
                    AgentLogger.log_info(
                        log_context,
                        "Skipping web search for books (already found via book APIs)"
                    )
                    continue
                
                if material_type not in search_queries:
                    continue
                
                type_materials = []
                queries_to_try = search_queries[material_type][:2]  # Limit queries per type
                
                for query in queries_to_try:
                    try:
                        raw = serp.results(query)
                        organic = (raw or {}).get("organic_results", [])
                        
                        for item in organic[:max_results_per_type]:
                            material = _extract_material_from_search_result(
                                item, material_type, topic
                            )
                            if material and _validate_material(material):
                                type_materials.append(material)
                                if len(type_materials) >= max_results_per_type:
                                    break
                    except Exception as e:
                        AgentLogger.log_warning(
                            log_context, 
                            f"Search failed for query '{query}': {e}"
                        )
                        continue
                
                all_materials.extend(type_materials)
                if type_materials:
                    AgentLogger.log_info(
                        log_context,
                        f"🔍 Found {len(type_materials)} {material_type} materials for '{topic}'"
                    )
                
        except Exception as e:
            AgentLogger.log_warning(
                log_context, 
                f"SerpAPI unavailable or error: {e}"
            )
    else:
        # SerpAPI not available - this is fine, we use free APIs
        AgentLogger.log_info(
            log_context,
            "Using free APIs only (arXiv, PubMed, Crossref, Google Books, Open Library, Wikibooks). Web search disabled to avoid SerpAPI costs."
        )
    
    return all_materials


async def _check_and_discover_materials(
    topic: str,
    material_types: List[str] = None,
    min_results_required: int = 5,
    min_similarity_threshold: float = 0.4,
    max_results_per_type: int = 5,
    log_context: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Check KB availability for a topic, search online if insufficient.
    
    Flow:
    1. Search KB for topic across all material types
    2. Evaluate if results are sufficient PER MATERIAL TYPE
    3. For each insufficient type, search online and add to KB
    4. Return all available materials
    
    Args:
        topic: Topic/skill to check (e.g., "React.js", "Machine Learning", "Hairstyling")
        material_types: List of material types to check (default: ["course", "book", "paper", "video", "tutorial"])
        min_results_required: Minimum number of results required PER TYPE
        min_similarity_threshold: Minimum average similarity score required
        max_results_per_type: Maximum results to fetch per material type from online search
        log_context: Logging context
        
    Returns:
        List of available materials from KB (after potential online discovery)
    """
    if material_types is None:
        material_types = ["course", "book", "paper", "video", "tutorial"]
    
    kb = CourseKnowledgeBase()
    
    # Step 1: Check KB availability PER MATERIAL TYPE
    kb_results_by_type = {}
    all_kb_results = []
    
    for material_type in material_types:
        try:
            # Search with type filter if not "course" (courses are default)
            filters = {"type": material_type} if material_type != "course" else {"type": "course"}
            results = kb.search_courses(
                query=f"{topic} {material_type}",
                top_k=10,
                filters=filters
            )
            kb_results_by_type[material_type] = results
            all_kb_results.extend(results)
        except Exception as e:
            AgentLogger.log_warning(
                log_context,
                f"KB search failed for {material_type}: {e}"
            )
            kb_results_by_type[material_type] = []
    
    # Step 2: Evaluate sufficiency PER MATERIAL TYPE
    # Determine which material types need discovery
    types_needing_discovery = []
    min_per_type = max(2, min_results_required // len(material_types))  # At least 2 per type, or distribute min_results
    
    for material_type in material_types:
        type_results = kb_results_by_type.get(material_type, [])
        
        # Check if this type has sufficient results
        is_sufficient = _evaluate_material_availability(
            type_results,
            min_per_type,  # Minimum per type
            min_similarity_threshold
        )
        
        if not is_sufficient:
            types_needing_discovery.append(material_type)
            AgentLogger.log_info(
                log_context,
                f"⚠️ {material_type}: Only found {len(type_results)} materials (need {min_per_type}+). Will discover more."
            )
        else:
            AgentLogger.log_info(
                log_context,
                f"✅ {material_type}: Found {len(type_results)} materials (sufficient)"
            )
    
    # Step 3: Search online for types that need discovery
    if types_needing_discovery:
        discovered_materials = await _search_online_materials(
            topic=topic,
            material_types=types_needing_discovery,  # Only search for missing types
            max_results_per_type=max_results_per_type,
            log_context=log_context
        )
    
    # Step 4: Add discovered materials to KB
    if discovered_materials:
        added_count = 0
        for material in discovered_materials:
            try:
                course_id = kb._generate_course_id(material)
                if kb.add_course(course_id, material):
                    added_count += 1
            except Exception as e:
                AgentLogger.log_warning(
                    log_context, 
                    f"Failed to add material '{material.get('title', 'unknown')}': {e}"
                )
            
            # Re-search KB with newly added materials
            updated_results = []
            for material_type in material_types:
                try:
                    filters = {"type": material_type} if material_type != "course" else {"type": "course"}
                    results = kb.search_courses(
                        query=f"{topic} {material_type}",
                        top_k=10,
                        filters=filters
                    )
                    updated_results.extend(results)
                except Exception as e:
                    AgentLogger.log_warning(
                        log_context,
                        f"Re-search failed for {material_type}: {e}"
                    )
            
            return updated_results
    # All material types are sufficient, return existing KB results
    
    return all_kb_results  # Return what we have


def _collect_retriever_docs(retriever: Any, queries: List[str]) -> List[Any]:
    """Run LangChain retriever for each query; dedupe by course_id / id / content hash."""
    all_docs: List[Any] = []
    seen: Set[str] = set()
    for q in queries:
        if not q or not str(q).strip():
            continue
        try:
            batch = retriever.get_relevant_documents(str(q))
        except Exception:
            continue
        for doc in batch:
            meta = getattr(doc, "metadata", None) or {}
            cid = meta.get("course_id") or meta.get("id")
            key = str(cid) if cid is not None else None
            if key is None:
                content = getattr(doc, "page_content", None) or ""
                key = hashlib.sha256(str(content)[:500].encode()).hexdigest()[:32]
            if key in seen:
                continue
            seen.add(key)
            all_docs.append(doc)
    return all_docs


async def _retrieve_with_langchain(
    kb: CourseKnowledgeBase,
    user_query: str,
    difficulty: Optional[str],
    allowed_difficulties: List[str],
    log_context: Dict[str, Any],
    max_retries: int = 3
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve courses and materials via LangChain vectorstore: LLM generates sub-queries
    (invoke_structured_llm) then retriever runs per query with deduplication.

    Returns:
        Tuple of (courses, other_materials)
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available")
    
    embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-small-en-v1.5")  # Better quality, same 384 dimensions
    
    # Create LangChain Chroma wrapper (reuses existing collection)
    vectorstore = Chroma(
        client=kb.client,
        collection_name="courses_knowledge_base",
        embedding_function=embeddings
    )
    
    all_courses = []
    all_other_materials = []
    
    # Retry logic with exponential backoff
    last_exception = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                AgentLogger.log_info(log_context, f"🔄 Retrying LangChain retrieval (attempt {attempt + 1}/{max_retries}) after {wait_time}s...")
                await asyncio.sleep(wait_time)
            
            # Build filter for difficulty (flexible - allow adjacent levels)
            search_kwargs = {"k": 20, "filter": {"type": "course"}}
            if difficulty and allowed_difficulties:
                if len(allowed_difficulties) == 1:
                    search_kwargs["filter"] = {
                        "$and": [
                            {"difficulty": allowed_difficulties[0]},
                            {"type": "course"}
                        ]
                    }
                else:
                    difficulty_conditions = [{"difficulty": d} for d in allowed_difficulties]
                    search_kwargs["filter"] = {
                        "$and": [
                            {"$or": difficulty_conditions},
                            {"type": "course"}
                        ]
                    }
            
            # Create base retriever
            base_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )

            mq_input = (
                [SystemMessage(content=_SEARCH_QUERIES_SYSTEM_INSTRUCTION), HumanMessage(content=user_query)]
                if SystemMessage and HumanMessage
                else _SEARCH_QUERIES_SYSTEM_INSTRUCTION + "\n\n" + user_query
            )
            try:
                mq_struct = await invoke_structured_llm(
                    mq_input,
                    SearchQueriesResult,
                    task_type=TaskType.TEXT_GENERATION,
                    preferred_model="gemini-2.5-flash",
                    agent_name="market_and_course_langchain_mq",
                    max_output_tokens=1500,
                    temperature=0.7,
                    timeout=45.0,
                    raise_on_fallback=False,
                    skip_cache=True,
                )
                sub_queries = [q for q in (mq_struct.search_queries or []) if q and str(q).strip()]
            except Exception as mq_err:
                AgentLogger.log_warning(
                    log_context, f"LangChain sub-query generation failed, using raw user_query: {mq_err}"
                )
                sub_queries = []
            if not sub_queries:
                sub_queries = [user_query[:2000]]

            docs = await asyncio.to_thread(_collect_retriever_docs, base_retriever, sub_queries)
            
            AgentLogger.log_info(log_context, f"✅ LangChain retrieved {len(docs)} course documents")
            
            # Convert LangChain documents to course dicts
            for doc in docs:
                metadata = doc.metadata
                course_id = metadata.get("course_id") or metadata.get("id")
                
                if course_id:
                    course = {
                        "course_id": course_id,
                        "title": metadata.get("title", ""),
                        "provider": metadata.get("provider", ""),
                        "url": metadata.get("url", ""),
                        "difficulty": metadata.get("difficulty", "All Levels"),
                        "duration": metadata.get("duration", ""),
                        "type": metadata.get("type", "course"),
                        "author": metadata.get("author", ""),
                        "similarity_score": 1.0 - (doc.metadata.get("distance", 0.0) if "distance" in doc.metadata else 0.0),
                        "description": metadata.get("description", ""),
                        "skills": metadata.get("skills", "").split(",") if isinstance(metadata.get("skills"), str) else metadata.get("skills", [])
                    }
                    all_courses.append(course)
            
            # Also search for other materials (books, papers, videos, tutorials)
            for material_type in ["book", "paper", "video", "tutorial"]:
                try:
                    material_search_kwargs = {"k": 10, "filter": {"type": material_type}}
                    material_retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs=material_search_kwargs
                    )
                    material_docs = await asyncio.to_thread(
                        _collect_retriever_docs, material_retriever, sub_queries
                    )
                    
                    for doc in material_docs:
                        metadata = doc.metadata
                        material_id = metadata.get("course_id") or metadata.get("id")
                        if material_id:
                            material = {
                                "course_id": material_id,
                                "title": metadata.get("title", ""),
                                "provider": metadata.get("provider", ""),
                                "url": metadata.get("url", ""),
                                "type": metadata.get("type", material_type),
                                "author": metadata.get("author", ""),
                                "similarity_score": 1.0 - (doc.metadata.get("distance", 0.0) if "distance" in doc.metadata else 0.0),
                                "description": metadata.get("description", ""),
                                "year": metadata.get("year", "")
                            }
                            all_other_materials.append(material)
                except Exception as e:
                    AgentLogger.log_debug(log_context, f"Material search failed for {material_type}: {e}")
            
            # Fallback if not enough courses, try without difficulty filter
            if len(all_courses) < 12:
                AgentLogger.log_info(log_context, f"⚠️ Only found {len(all_courses)} courses with difficulty filter, trying without filter...")
                search_kwargs_no_filter = {"k": 20, "filter": {"type": "course"}}
                base_retriever_no_filter = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs=search_kwargs_no_filter
                )
                docs_no_filter = await asyncio.to_thread(
                    _collect_retriever_docs, base_retriever_no_filter, sub_queries
                )
                
                for doc in docs_no_filter:
                    metadata = doc.metadata
                    course_id = metadata.get("course_id") or metadata.get("id")
                    if course_id and not any(c.get("course_id") == course_id for c in all_courses):
                        course = {
                            "course_id": course_id,
                            "title": metadata.get("title", ""),
                            "provider": metadata.get("provider", ""),
                            "url": metadata.get("url", ""),
                            "difficulty": metadata.get("difficulty", "All Levels"),
                            "duration": metadata.get("duration", ""),
                            "type": metadata.get("type", "course"),
                            "author": metadata.get("author", ""),
                            "similarity_score": 1.0 - (doc.metadata.get("distance", 0.0) if "distance" in doc.metadata else 0.0),
                            "description": metadata.get("description", ""),
                            "skills": metadata.get("skills", "").split(",") if isinstance(metadata.get("skills"), str) else metadata.get("skills", [])
                        }
                        all_courses.append(course)
            
            # Success - return results
            return all_courses, all_other_materials
            
        except Exception as e:
            last_exception = e
            AgentLogger.log_warning(log_context, f"LangChain retrieval attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Last attempt failed, raise the exception
                raise
    
    # Should not reach here, but just in case
    raise last_exception or Exception("LangChain retrieval failed after all retries")

async def _generate_course_recommendations_from_knowledge_base(
    structured_resume: Dict[str, Any],
    user_interests: List[Dict[str, Any]],
    skill_gap_analysis: Dict[str, Any],
    log_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate course recommendations using RAG (Retrieval-Augmented Generation) flow.
    NO semantic search fallback - strictly RAG only.
    
    RAG Flow:
    1. LLM analyzes user context and generates optimized search queries
    2. Search knowledge base with LLM-generated queries ONLY
    3. LLM selects and personalizes courses from KB results (required step)
    4. Return personalized recommendations with explanations
    
    If any step fails, returns empty results (no fallback to semantic search).
    
    Args:
        structured_resume: Parsed resume data with skills, experience, education, target_roles
        user_interests: User's interests
        skill_gap_analysis: Contains missing_skills, career_paths, assessment_performance
        log_context: Logging context
        
    Returns:
        Dict with "course_recommendations" list (all courses from knowledge base)
    """
    start_time = log_context.get("start_time", time.time())
    
    try:
        # Initialize knowledge base
        kb = CourseKnowledgeBase()
        
        # Extract skills and context from inputs
        resume_skills = structured_resume.get("skills", []) if structured_resume else []
        target_roles = structured_resume.get("target_roles", []) if structured_resume else []
        experience_years = len(structured_resume.get("work_experience", [])) if structured_resume else 0
        
        # Extract from skill gap analysis
        missing_skills = []
        career_paths = []
        assessment_performance = {}
        
        if isinstance(skill_gap_analysis, dict):
            missing_skills = get_missing_skills_flat(skill_gap_analysis)
            career_paths = skill_gap_analysis.get("career_paths", [])
            assessment_performance = skill_gap_analysis.get("assessment_performance", {})
        
        # Track missing topics for background population (don't discover now - background job will handle it)
        if missing_skills:
            AgentLogger.log_info(
                log_context,
                f"📝 Tracking {len(missing_skills)} missing skills for background KB population..."
            )
            
            # Helper function to extract skill name from string or dict
            def extract_skill_name(skill):
                """Extract skill name from string or dict"""
                if isinstance(skill, str):
                    return skill.strip()
                elif isinstance(skill, dict):
                    return skill.get("name", "") or skill.get("skill", "") or skill.get("title", "") or str(skill.get("value", ""))
                else:
                    return str(skill).strip()
            
            # Track top missing skills for background population
            try:
                from agents.topic_tracker import get_topic_tracker
                tracker = get_topic_tracker()
                
                skills_to_track = [
                    extract_skill_name(s) 
                    for s in missing_skills[:10]  # Track top 10 missing skills
                    if extract_skill_name(s)
                ]
                
                # Add topics to queue (background job will process them)
                added = tracker.add_topics_batch(
                    topics=skills_to_track,
                    source="skill_gap",
                    priority=1  # High priority for missing skills
                )
                
                AgentLogger.log_info(
                    log_context,
                    f"✅ Tracked {added} topics for background KB population (will be processed by background job)"
                )
            except Exception as e:
                AgentLogger.log_debug(log_context, f"Could not track topics for background population: {e}")
        
        # Extract interests
        interest_skills = []
        if isinstance(user_interests, list):
            for interest in user_interests:
                if isinstance(interest, dict):
                    interest_skills.extend(interest.get("skills", []))
        
        # Determine difficulty based on experience and assessment
        difficulty = None
        if experience_years < 2:
            difficulty = "Beginner"
        elif experience_years < 5:
            difficulty = "Intermediate"
        else:
            difficulty = "Advanced"
        
        # Adjust difficulty based on assessment performance
        if isinstance(assessment_performance, dict):
            performance_level = assessment_performance.get("performance_level", "").lower()
            if performance_level in ["needs improvement", "fair"]:
                # Downgrade difficulty for weaker performers
                if difficulty == "Advanced":
                    difficulty = "Intermediate"
                elif difficulty == "Intermediate":
                    difficulty = "Beginner"
            elif performance_level in ["excellent", "outstanding"]:
                # Upgrade difficulty for strong performers
                if difficulty == "Beginner":
                    difficulty = "Intermediate"
                elif difficulty == "Intermediate":
                    difficulty = "Advanced"
        
        # Helper function to get allowed difficulty levels (flexible filtering)
        def get_allowed_difficulties(target_difficulty: Optional[str]) -> List[str]:
            """
            Get allowed difficulty levels including adjacent levels for flexible filtering.
            This allows experienced professionals to access intermediate courses in new domains,
            and beginners to access slightly advanced content.
            
            Args:
                target_difficulty: Primary difficulty level (Beginner/Intermediate/Advanced)
                
            Returns:
                List of allowed difficulty levels
            """
            if not target_difficulty:
                return ["Beginner", "Intermediate", "Advanced", "All Levels"]
            
            target = target_difficulty.lower()
            
            # Map difficulty levels
            if target == "beginner":
                # Beginners can access Beginner and Intermediate (for growth)
                return ["Beginner", "Intermediate", "All Levels"]
            elif target == "intermediate":
                # Intermediate users can access all levels (they might need basics in new domains or advanced in their domain)
                return ["Beginner", "Intermediate", "Advanced", "All Levels"]
            elif target == "advanced":
                # Advanced users can access Intermediate and Advanced (they might need intermediate courses in new domains)
                return ["Intermediate", "Advanced", "All Levels"]
            else:
                # Unknown difficulty - allow all
                return ["Beginner", "Intermediate", "Advanced", "All Levels"]
        
        # Get allowed difficulty levels for flexible filtering
        allowed_difficulties = get_allowed_difficulties(difficulty)
        AgentLogger.log_info(log_context, f"📊 Using flexible difficulty filtering: {difficulty} (allowing: {', '.join(allowed_difficulties)})")
        
        # Helper function to extract skill name from string or dict
        def extract_skill_name(skill):
            """Extract skill name from string or dict"""
            if isinstance(skill, str):
                return skill.strip()
            elif isinstance(skill, dict):
                # Try common keys for skill name
                return skill.get("name", "") or skill.get("skill", "") or skill.get("title", "") or str(skill.get("value", ""))
            else:
                return str(skill).strip()
        
        # Prepare context strings for LLM
        missing_skills_str = ', '.join([extract_skill_name(s) for s in missing_skills[:8] if extract_skill_name(s)]) or 'None specified'
        target_roles_str = ', '.join([r if isinstance(r, str) else str(r) for r in target_roles[:3]]) or 'None specified'
        career_paths_str = ', '.join([
            p.get('title', '') if isinstance(p, dict) else str(p) 
            for p in career_paths[:2]
        ]) or 'None specified'
        resume_skills_str = ', '.join([extract_skill_name(s) for s in resume_skills[:10] if extract_skill_name(s)]) or 'None specified'
        
        # STEP 1-2: Multi-query retrieval via invoke_structured_llm + vector retriever (no MultiQueryRetriever)
        all_courses = []
        all_other_materials = []
        
        # Build user query from context with intelligent synonym awareness
        user_query = f"""I need courses for:
- Experience: {experience_years} years ({difficulty} level)
- Current Skills: {resume_skills_str}
- Missing Skills: {missing_skills_str}
- Target Roles: {target_roles_str}
- Career Paths: {career_paths_str}

Find relevant courses that address my missing skills and align with my career goals.

CRITICAL INSTRUCTION - Intelligent Synonym & Terminology Expansion:
You must automatically identify and include synonyms, related terms, acronyms, and alternative terminology for ALL domains and skills mentioned. 

For EACH skill, role, or career path mentioned:
1. Identify common synonyms (e.g., "QA" = "Quality Assurance" = "Software Testing")
2. Include acronyms AND full terms (e.g., "API" and "Application Programming Interface", "ML" and "Machine Learning")
3. Think about how course providers might name courses (they may use different terminology than job titles)
4. Include related domain concepts (e.g., for "Data Science" also consider "Data Analytics", "Data Analysis", "Analytics", "Statistics")
5. Consider industry variations (e.g., "DevOps" might be called "CI/CD", "Infrastructure as Code", "Site Reliability Engineering")
6. Include tool/framework names related to the skill (e.g., for "Testing" include "Selenium", "Jest", "Cypress", "Test Automation")
7. Think about broader and narrower terms (e.g., "Web Development" includes "Frontend", "Backend", "Full Stack", "React", "Node.js")

Generate queries that will match courses even if they use different terminology than what's explicitly mentioned in the profile."""
        
        # Try LangChain Path A with retry logic
        if LANGCHAIN_AVAILABLE:
            try:
                AgentLogger.log_info(log_context, "🔗 Using LangChain hybrid RAG retrieval (structured sub-queries + retriever)...")
                all_courses, all_other_materials = await _retrieve_with_langchain(
                    kb=kb,
                    user_query=user_query,
                    difficulty=difficulty,
                    allowed_difficulties=allowed_difficulties,
                    log_context=log_context,
                    max_retries=3
                )
                AgentLogger.log_info(log_context, f"✅ LangChain retrieval successful: {len(all_courses)} courses, {len(all_other_materials)} other materials")
            except ImportError:
                # LangChain truly not available - fall through to Path B
                AgentLogger.log_warning(log_context, "LangChain not available, using manual retrieval")
                all_courses = []
                all_other_materials = []
            except Exception as e:
                # Path A failed after retries - fall back to Path B
                AgentLogger.log_warning(log_context, f"LangChain retrieval failed after retries: {e}. Falling back to manual approach.")
                all_courses = []
                all_other_materials = []
        
        # Use Path B if LangChain is unavailable OR if Path A failed after retries
        use_path_b = False
        if not LANGCHAIN_AVAILABLE:
            use_path_b = True
            AgentLogger.log_warning(log_context, "LangChain not available, using manual retrieval")
        elif len(all_courses) == 0 and len(all_other_materials) == 0:
            use_path_b = True
            AgentLogger.log_warning(log_context, "LangChain retrieval failed, using manual fallback")
        
        if use_path_b:
            # Path B: Manual approach (only if LangChain unavailable or Path A completely failed)
            # STEP 1: Use LLM to generate intelligent search queries (original approach)
            llm_queries = []
            try:
                AgentLogger.log_info(log_context, "🤖 Generating intelligent search queries with LLM...")
                
                # Prompt = dynamic content only; system_instruction cached for Gemini context caching
                llm_query_prompt = f"""Candidate Profile:
- Experience Level: {experience_years} years ({difficulty} level)
- Current Skills: {resume_skills_str}
- Missing Skills (Priority): {missing_skills_str}
- Target Roles: {target_roles_str}
- Career Paths: {career_paths_str}
"""
                llm_query_input = (
                    [SystemMessage(content=_SEARCH_QUERIES_SYSTEM_INSTRUCTION), HumanMessage(content=llm_query_prompt)]
                    if SystemMessage and HumanMessage
                    else _SEARCH_QUERIES_SYSTEM_INSTRUCTION + "\n\n" + llm_query_prompt
                )

                # Issue 5.1: Use structured output for type-safe parsing (centralized invoke_structured_llm)
                try:
                    result: SearchQueriesResult = await asyncio.wait_for(
                        invoke_structured_llm(
                            llm_query_input,
                            SearchQueriesResult,
                            task_type=TaskType.TEXT_GENERATION,
                            preferred_model=_settings.GEMINI_MODEL,
                            agent_name="course_recommender",
                            max_output_tokens=2000,
                            temperature=0.3,
                            timeout=30.0,
                            raise_on_fallback=False,
                            skip_cache=True,
                        ),
                        timeout=30,
                    )
                    llm_queries = result.search_queries
                    AgentLogger.log_debug(log_context, f"Structured output reasoning: {result.reasoning}")
                except asyncio.TimeoutError:
                    AgentLogger.log_warning(log_context, "Structured LLM call timed out, falling back to manual parsing")
                    llm_response = await invoke_llm(
                        prompt=llm_query_prompt,
                        task_type="text_generation",
                        agent_name="course_recommender",
                        max_output_tokens=2000,
                        response_mime_type="application/json",
                        system_instruction=_SEARCH_QUERIES_SYSTEM_INSTRUCTION
                    )
                    llm_queries = _extract_queries_from_llm_response(llm_response, log_context)
                except Exception as structured_err:
                    AgentLogger.log_warning(log_context, f"Structured output failed ({structured_err}), falling back to manual parsing")
                    llm_response = await invoke_llm(
                        prompt=llm_query_prompt,
                        task_type="text_generation",
                        agent_name="course_recommender",
                        max_output_tokens=2000,
                        response_mime_type="application/json",
                        system_instruction=_SEARCH_QUERIES_SYSTEM_INSTRUCTION
                    )
                    llm_queries = _extract_queries_from_llm_response(llm_response, log_context)
                
                # Validate queries - RAG flow requires LLM queries
                if not llm_queries or len(llm_queries) == 0:
                    AgentLogger.log_warning(log_context, "LLM did not return valid search queries. Using fallback queries...")
                    # Fallback: generate simple queries from missing skills
                    llm_queries = []
                    for skill in missing_skills[:5]:
                        skill_name = extract_skill_name(skill)
                        if skill_name:
                            llm_queries.append(f"{skill_name} course")
                    if not llm_queries:
                        # Last resort: generic queries
                        llm_queries = ["programming", "development", "software engineering", "technology", "skills"]
                    AgentLogger.log_info(log_context, f"Using {len(llm_queries)} fallback queries: {llm_queries[:3]}...")
                    
            except Exception as e2:
                try:
                    processing_time = _calculate_processing_time(start_time)
                except Exception as e3:
                    AgentLogger.log_warning(log_context, f"Processing time calculation failed: {e3}, using default time")
                    processing_time = "N/A"
                AgentLogger.log_error(log_context, f"LLM query generation failed: {e2}. Using fallback queries...", processing_time)
                # Fallback: generate simple queries from missing skills
                llm_queries = []
                for skill in missing_skills[:5]:
                    skill_name = extract_skill_name(skill)
                    if skill_name:
                        llm_queries.append(f"{skill_name} course")
                if not llm_queries:
                    # Last resort: generic queries
                    llm_queries = ["programming", "development", "software engineering", "technology", "skills"]
                AgentLogger.log_info(log_context, f"Using {len(llm_queries)} fallback queries: {llm_queries[:3]}...")
            
            # STEP 2: Search knowledge base with LLM-generated queries for courses AND other materials
            # Use LLM-generated queries to search KB
            if not all_courses:  # Initialize if not already set
                all_courses = []
            all_other_materials = []  # Books, papers, videos, tutorials
            
            for query in llm_queries[:10]:  # Use up to 10 queries
                try:
                    # Search for courses with flexible difficulty filter
                    if difficulty and allowed_difficulties:
                        if len(allowed_difficulties) == 1:
                            filters = {"$and": [{"difficulty": allowed_difficulties[0]}, {"type": "course"}]}
                        else:
                            difficulty_conditions = [{"difficulty": d} for d in allowed_difficulties]
                            filters = {
                                "$and": [
                                    {"$or": difficulty_conditions},
                                    {"type": "course"}
                                ]
                            }
                    else:
                        filters = {"type": "course"}
                    courses = kb.search_courses(query=query, top_k=8, filters=filters)
                    all_courses.extend(courses)
                    AgentLogger.log_info(log_context, f"✅ Found {len(courses)} courses from LLM query: '{query[:60]}...'")
                    
                    # Also search for other materials (books, papers, videos, tutorials)
                    for material_type in ["book", "paper", "video", "tutorial"]:
                        try:
                            material_filters = {"type": material_type}
                            materials = kb.search_courses(query=query, top_k=3, filters=material_filters)
                            all_other_materials.extend(materials)
                            if materials:
                                AgentLogger.log_info(log_context, f"✅ Found {len(materials)} {material_type}s from query: '{query[:60]}...'")
                        except Exception as e3:
                            AgentLogger.log_debug(log_context, f"Search failed for {material_type}: {e3}")
                            
                except Exception as e2:
                    AgentLogger.log_warning(log_context, f"Search failed for LLM query '{query}': {e2}")
            
            # If not enough courses, try without difficulty filter
            if len(all_courses) < 8:
                AgentLogger.log_info(log_context, f"⚠️ Only found {len(all_courses)} courses with difficulty filter, trying without filter...")
                for query in llm_queries[:8]:
                    try:
                        courses = kb.search_courses(query=query, top_k=6, filters={"type": "course"})
                        all_courses.extend(courses)
                        AgentLogger.log_info(log_context, f"✅ Found {len(courses)} courses from query without difficulty filter: '{query[:60]}...'")
                    except Exception as e2:
                        AgentLogger.log_warning(log_context, f"Search failed for query '{query}': {e2}")
        
        # Log warning if still not enough, but proceed with what we have (RAG will handle it)
        if len(all_courses) < 5:
            AgentLogger.log_warning(log_context, f"⚠️ Only found {len(all_courses)} courses from queries. Proceeding with RAG enhancement.")
        
        # Deduplicate courses by course_id
        seen_course_ids = set()
        unique_courses = []
        for course in all_courses:
            course_id = course.get("course_id")
            if course_id and course_id not in seen_course_ids:
                seen_course_ids.add(course_id)
                unique_courses.append(course)
        
        # Deduplicate other materials by URL or ID
        seen_material_ids = set()
        unique_other_materials = []
        for material in all_other_materials:
            material_id = material.get("course_id") or material.get("url", "")
            if material_id and material_id not in seen_material_ids:
                seen_material_ids.add(material_id)
                unique_other_materials.append(material)
        
        AgentLogger.log_info(log_context, f"📚 Found {len(unique_courses)} unique courses and {len(unique_other_materials)} other materials from knowledge base")
        
        # STEP 2.5: If insufficient materials, query APIs directly (for RAG access)
        # This allows RAG to access materials from APIs even if not in KB yet
        if len(unique_courses) < 5 or len(unique_other_materials) < 5:
            AgentLogger.log_info(log_context, f"📡 KB has insufficient materials. Querying APIs directly for RAG access...")
            
            # Extract main topics from LLM queries for API search
            api_search_topics = []
            for query in llm_queries[:3]:  # Use top 3 queries
                # Extract key terms from query (simple extraction)
                words = query.lower().split()
                # Filter out common words and get meaningful terms
                key_terms = [w for w in words if len(w) > 3 and w not in ["course", "courses", "learn", "learning", "tutorial", "tutorials"]]
                if key_terms:
                    api_search_topics.append(" ".join(key_terms[:3]))  # Take top 3 terms
            
            # If no good topics extracted, use missing skills
            if not api_search_topics and missing_skills:
                api_search_topics = [extract_skill_name(s) for s in missing_skills[:3] if extract_skill_name(s)]
            
            # Query APIs for materials
            for topic in api_search_topics[:2]:  # Limit to 2 topics to avoid too many API calls
                try:
                    # Query book APIs
                    if len(unique_other_materials) < 5:
                        try:
                            from agents.academic_api_adapters import AcademicAPIManager
                            academic_manager = AcademicAPIManager()
                            
                            # Search for books
                            api_books = await academic_manager.search_books(
                                query=topic,
                                sources=["google_books", "open_library"],
                                max_results_per_source=3
                            )
                            
                            # Convert to KB format
                            for book in api_books[:5]:
                                material = {
                                    "title": book.get("title", ""),
                                    "url": book.get("url", ""),
                                    "type": "book",
                                    "provider": book.get("source", "Book API"),
                                    "author": ", ".join(book.get("authors", [])),
                                    "description": book.get("description", ""),
                                    "difficulty": "Intermediate",
                                    "skills": [topic],
                                    "similarity_score": 0.7,  # Default score for API results
                                    "course_id": book.get("url", "")  # Use URL as ID
                                }
                                # Check if already in unique_other_materials
                                if material["url"] not in {m.get("url", "") for m in unique_other_materials}:
                                    unique_other_materials.append(material)
                                    AgentLogger.log_info(log_context, f"📖 Added book from API: {book.get('title', '')[:50]}")
                        except Exception as e:
                            AgentLogger.log_debug(log_context, f"Book API search failed: {e}")
                    
                    # Query paper APIs
                    if len(unique_other_materials) < 5:
                        try:
                            from agents.academic_api_adapters import AcademicAPIManager
                            academic_manager = AcademicAPIManager()
                            
                            # Search for papers
                            api_papers = await academic_manager.search_all_sources(
                                query=topic,
                                sources=["arxiv", "pubmed", "crossref"],
                                max_results_per_source=2,
                                material_type="paper"
                            )
                            
                            # Convert to KB format
                            for paper in api_papers[:5]:
                                material = {
                                    "title": paper.get("title", ""),
                                    "url": paper.get("url", ""),
                                    "type": "paper",
                                    "provider": paper.get("source", "Academic API"),
                                    "author": ", ".join(paper.get("authors", [])),
                                    "description": paper.get("abstract", ""),
                                    "difficulty": "Advanced",
                                    "skills": [topic],
                                    "similarity_score": 0.7,
                                    "course_id": paper.get("url", "")
                                }
                                # Check if already in unique_other_materials
                                if material["url"] not in {m.get("url", "") for m in unique_other_materials}:
                                    unique_other_materials.append(material)
                                    AgentLogger.log_info(log_context, f"📄 Added paper from API: {paper.get('title', '')[:50]}")
                        except Exception as e:
                            AgentLogger.log_debug(log_context, f"Paper API search failed: {e}")
                            
                except Exception as e:
                    AgentLogger.log_warning(log_context, f"API query failed for topic '{topic}': {e}")
            
            AgentLogger.log_info(log_context, f"📡 After API queries: {len(unique_courses)} courses, {len(unique_other_materials)} other materials")
        
        # Sort by similarity score
        unique_courses.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        unique_other_materials.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        # STEP 3: RAG Enhancement - Use LLM to generate personalized recommendations
        # Select top candidates: 8 courses (for 5 final) and 8 other materials (for 5 final)
        top_course_candidates = unique_courses[:8]
        top_material_candidates = unique_other_materials[:8]
        
        # Use RAG approach: Pass full course and material details to LLM for intelligent selection
        formatted_courses = []
        formatted_other_materials = []
        try:
            AgentLogger.log_info(log_context, "🤖 Using RAG: LLM analyzing courses and other materials from knowledge base...")
            
            # Build compact course data
            optimized_courses = []
            for c in top_course_candidates:
                course_data = {
                    "course_id": c.get("course_id"),
                    "title": c.get("title"),
                    "provider": c.get("provider"),
                    "description": (c.get("description", "") or "")[:100],
                    "skills": (c.get("skills", []) or [])[:5],
                    "difficulty": c.get("difficulty", "All Levels"),
                    "similarity_score": round(c.get("similarity_score", 0), 2)
                }
                if c.get("duration"):
                    course_data["duration"] = c.get("duration")
                optimized_courses.append(course_data)
            
            # Build compact material data
            optimized_materials = []
            for m in top_material_candidates:
                material_data = {
                    "material_id": m.get("course_id") or m.get("url", ""),
                    "title": m.get("title"),
                    "type": m.get("type", "unknown"),
                    "provider": m.get("provider", ""),
                    "description": (m.get("description", "") or "")[:100],
                    "url": m.get("url", ""),
                    "similarity_score": round(m.get("similarity_score", 0), 2)
                }
                if m.get("author"):
                    material_data["author"] = m.get("author")
                if m.get("year"):
                    material_data["year"] = m.get("year")
                optimized_materials.append(material_data)
            
            # Prompt = dynamic content only; system_instruction cached for Gemini context caching
            rag_prompt = f"""CANDIDATE:
- Experience: {experience_years} years ({difficulty})
- Skills: {resume_skills_str}
- Missing: {missing_skills_str}
- Roles: {target_roles_str}
- Paths: {career_paths_str}

COURSES ({len(optimized_courses)} available):
{json.dumps(optimized_courses, indent=1)}

OTHER MATERIALS ({len(optimized_materials)} available):
{json.dumps(optimized_materials, indent=1)}
"""
            rag_input = (
                [SystemMessage(content=_RAG_SELECTION_SYSTEM_INSTRUCTION), HumanMessage(content=rag_prompt)]
                if SystemMessage and HumanMessage
                else _RAG_SELECTION_SYSTEM_INSTRUCTION + "\n\n" + rag_prompt
            )

            # Issue 5.1: Use structured output for type-safe parsing
            rag_data = None
            try:
                result: RAGSelectionResult = await asyncio.wait_for(
                    invoke_structured_llm(
                        rag_input,
                        RAGSelectionResult,
                        task_type=TaskType.TEXT_GENERATION,
                        preferred_model="gemini-2.5-flash",
                        agent_name="market_and_course_recommender_rag",
                        max_output_tokens=5000,
                        temperature=0.3,
                        timeout=60.0,
                        raise_on_fallback=False,
                        skip_cache=True,
                    ),
                    timeout=60,
                )
                # Convert Pydantic models to dicts for downstream compatibility
                rag_data = {
                    "selected_courses": [c.model_dump() for c in result.selected_courses],
                    "selected_materials": [m.model_dump() for m in result.selected_materials]
                }
                AgentLogger.log_debug(log_context, f"Structured RAG output: {len(result.selected_courses)} courses, {len(result.selected_materials)} materials")
            except asyncio.TimeoutError:
                AgentLogger.log_warning(log_context, "Structured RAG call timed out, falling back to manual parsing")
            except Exception as structured_err:
                AgentLogger.log_warning(log_context, f"Structured RAG output failed ({structured_err}), falling back to manual parsing")
            
            # Fallback to manual parsing if structured output failed
            if not rag_data:
                rag_response = await invoke_llm(
                    prompt=rag_prompt,
                    task_type="text_generation",
                    agent_name="course_recommender",
                    preferred_model="gemini-2.5-flash",
                    max_output_tokens=5000,
                    response_mime_type="application/json",
                    system_instruction=_RAG_SELECTION_SYSTEM_INSTRUCTION
                )
                
                AgentLogger.log_debug(log_context, f"RAG response type: {type(rag_response)}")
                
                if isinstance(rag_response, (dict, list)):
                    rag_data = rag_response
                    AgentLogger.log_debug(log_context, f"RAG response is already-parsed JSON: {type(rag_data)}")
                elif isinstance(rag_response, str):
                    AgentLogger.log_debug(log_context, f"RAG response length: {len(rag_response)}, preview: {rag_response[:500]}")
                    try:
                        rag_data = json.loads(rag_response)
                    except json.JSONDecodeError as e:
                        AgentLogger.log_debug(log_context, f"JSON parse error: {e}, trying safe parse...")
                        try:
                            if "Unterminated string" in str(e) or "Expecting" in str(e):
                                json_match = re.search(r'(\{.*\})', rag_response, re.DOTALL)
                                if json_match:
                                    try:
                                        json_str = json_match.group(1)
                                        open_braces = json_str.count('{')
                                        close_braces = json_str.count('}')
                                        if open_braces > close_braces:
                                            json_str += '}' * (open_braces - close_braces)
                                        rag_data = json.loads(json_str)
                                        AgentLogger.log_debug(log_context, "Fixed truncated JSON")
                                    except:
                                        pass
                        except:
                            pass
                        
                        if not rag_data:
                            rag_data = _safe_json_loads(rag_response)
                            if not rag_data:
                                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', rag_response, re.DOTALL)
                                if json_match:
                                    try:
                                        rag_data = json.loads(json_match.group(1))
                                        AgentLogger.log_debug(log_context, "Extracted JSON from markdown code block")
                                    except Exception as e2:
                                        AgentLogger.log_debug(log_context, f"Failed to parse extracted JSON: {e2}")
                else:
                    rag_data = rag_response
                    AgentLogger.log_debug(log_context, f"RAG response is unexpected type, using as-is: {type(rag_data)}")
            
            if rag_data:
                AgentLogger.log_debug(log_context, f"RAG data type: {type(rag_data)}, keys: {list(rag_data.keys()) if isinstance(rag_data, dict) else 'N/A'}")
            else:
                AgentLogger.log_warning(log_context, f"RAG response parsing resulted in None/empty data")
            
            # Handle different RAG response formats
            selected_courses_list = None
            selected_materials_list = None
            if isinstance(rag_data, dict):
                selected_courses_list = rag_data.get("selected_courses", [])
                selected_materials_list = rag_data.get("selected_materials", [])
                # Also check for alternative formats
                if not selected_courses_list:
                    selected_courses_list = rag_data.get("courses", [])
                if not selected_materials_list:
                    selected_materials_list = rag_data.get("materials", [])
            
            # Process selected courses
            if selected_courses_list and isinstance(selected_courses_list, list) and len(selected_courses_list) > 0:
                # Create course map for quick lookup
                course_map = {c.get("course_id"): c for c in top_course_candidates}
                
                # Process LLM-selected courses with explanations (limit to 5)
                for selected in selected_courses_list[:5]:
                    # Handle both dict format and direct course_id string
                    if isinstance(selected, dict):
                        course_id = selected.get("course_id")
                    elif isinstance(selected, str):
                        course_id = selected
                    else:
                        continue
                    
                    if course_id and course_id in course_map:
                        course = course_map[course_id]
                        formatted_course = {
                            "course": course.get("title", ""),  # Use "course" to match output format
                            "platform": course.get("provider", ""),
                            "url": course.get("url", ""),
                            "difficulty": course.get("difficulty", "All Levels"),
                            "duration": course.get("duration", ""),
                            "relevance_score": course.get("similarity_score", 0.7),
                            "description": course.get("description", ""),
                            "skills": course.get("skills", []),
                            "type": "course"
                        }
                        
                        # Add RAG-generated fields if available
                        if isinstance(selected, dict):
                            formatted_course["personalized_explanation"] = selected.get("personalized_explanation", "")
                            formatted_course["target_skill"] = selected.get("target_skill", "")
                            formatted_course["career_path_alignment"] = selected.get("career_path_alignment", "")
                        
                        formatted_courses.append(formatted_course)
                
                # Fill remaining course slots if LLM didn't select 5
                if len(formatted_courses) < 5:
                    remaining_ids = {c.get("course_id") for c in formatted_courses}
                    for course in top_course_candidates:
                        if len(formatted_courses) >= 5:
                            break
                        if course.get("course_id") not in remaining_ids:
                            formatted_course = {
                                "course": course.get("title", ""),  # Use "course" to match output format
                                "platform": course.get("provider", ""),
                                "url": course.get("url", ""),
                                "difficulty": course.get("difficulty", "All Levels"),
                                "duration": course.get("duration", ""),
                                "relevance_score": course.get("similarity_score", 0.7),
                                "description": course.get("description", ""),
                                "skills": course.get("skills", []),
                                "type": "course"
                            }
                            # Add target_skill using traditional method
                            if missing_skills and course.get("skills"):
                                course_skills_lower = [str(s).lower() for s in course.get("skills", []) if s]
                                for missing_skill in missing_skills:
                                    skill_name = extract_skill_name(missing_skill)
                                    if skill_name and (skill_name.lower() in course_skills_lower or any(cs in skill_name.lower() for cs in course_skills_lower)):
                                        formatted_course["target_skill"] = skill_name
                                        break
                            formatted_courses.append(formatted_course)
            
            # Process selected other materials
            if selected_materials_list and isinstance(selected_materials_list, list) and len(selected_materials_list) > 0:
                # Create material map for quick lookup by course_id OR url (LLM may return either as material_id)
                material_map = {}
                for m in top_material_candidates:
                    cid = m.get("course_id")
                    url = m.get("url", "")
                    if cid:
                        material_map[cid] = m
                    if url:
                        material_map[url] = m
                
                # Process LLM-selected materials with explanations (limit to 5)
                for selected in selected_materials_list[:5]:
                    if isinstance(selected, dict):
                        material_id = selected.get("material_id")
                    elif isinstance(selected, str):
                        material_id = selected
                    else:
                        continue
                    
                    if material_id and material_id in material_map:
                        material = material_map[material_id]
                        # Format material to match course structure (will be combined into course_recommendations)
                        formatted_material = {
                            "course": material.get("title", ""),  # Use "course" to match output format
                            "platform": material.get("provider", ""),
                            "type": material.get("type", "unknown"),  # book, paper, video, tutorial
                            "difficulty": material.get("difficulty", "All Levels"),
                            "duration": material.get("duration", "Self-paced"),
                            "relevance_score": material.get("similarity_score", 0.7),
                            "description": material.get("description", ""),
                            "url": material.get("url", "")
                        }
                        
                        # Add optional fields
                        if material.get("author"):
                            formatted_material["author"] = material.get("author")
                        if material.get("year"):
                            formatted_material["year"] = material.get("year")
                        if material.get("publisher"):
                            formatted_material["publisher"] = material.get("publisher")
                        
                        # Add RAG-generated fields if available (rationale and target_skill for output)
                        if isinstance(selected, dict):
                            formatted_material["target_skill"] = selected.get("target_skill", "")
                            if selected.get("personalized_explanation"):
                                formatted_material["personalized_explanation"] = selected.get("personalized_explanation", "")
                        
                        formatted_other_materials.append(formatted_material)
                
                # Fill remaining material slots if LLM didn't select 5 (no rationale/target_skill from RAG)
                if len(formatted_other_materials) < 5:
                    remaining_ids = {(m.get("material_id") or m.get("url", "")) for m in formatted_other_materials}
                    for material in top_material_candidates:
                        if len(formatted_other_materials) >= 5:
                            break
                        material_id = material.get("material_id") or material.get("url", "")
                        if material_id and material_id not in remaining_ids:
                            # Format material to match course structure (will be combined into course_recommendations)
                            formatted_material = {
                                "course": material.get("title", ""),  # Use "course" to match output format
                                "platform": material.get("provider", ""),
                                "type": material.get("type", "unknown"),  # book, paper, video, tutorial
                                "difficulty": material.get("difficulty", "All Levels"),
                                "duration": material.get("duration", "Self-paced"),
                                "relevance_score": material.get("similarity_score", 0.7),
                                "description": material.get("description", ""),
                                "url": material.get("url", "")
                            }
                            if material.get("author"):
                                formatted_material["author"] = material.get("author")
                            if material.get("year"):
                                formatted_material["year"] = material.get("year")
                            formatted_other_materials.append(formatted_material)
                
                AgentLogger.log_info(log_context, f"✅ RAG: LLM selected {len(formatted_courses)} courses and {len(formatted_other_materials)} other materials")
            elif selected_courses_list:
                # Only courses selected, no materials
                AgentLogger.log_info(log_context, f"✅ RAG: LLM selected {len(formatted_courses)} courses (no materials selected)")
            else:
                # Log what we got for debugging
                if rag_data:
                    AgentLogger.log_warning(log_context, f"RAG response format issue: type={type(rag_data)}, keys={list(rag_data.keys()) if isinstance(rag_data, dict) else 'N/A'}")
                    AgentLogger.log_debug(log_context, f"RAG response preview: {str(rag_data)[:300]}")
                else:
                    AgentLogger.log_warning(log_context, f"RAG response is None or empty")
                raise ValueError("RAG response invalid - RAG flow requires valid LLM response")
                
        except Exception as e:
            processing_time = _calculate_processing_time(start_time)
            AgentLogger.log_error(log_context, f"RAG enhancement failed: {e}. RAG flow requires LLM to generate personalized recommendations.", processing_time)
            # RAG flow failed - return what we have (fallback to top results)
            # Note: unique_courses and unique_other_materials are defined earlier in the function
            if not formatted_courses:
                # Check if we have unique_courses available
                if 'unique_courses' in locals() and unique_courses:
                    AgentLogger.log_info(log_context, f"🔄 Fallback: Using top {min(5, len(unique_courses))} courses by similarity score")
                    # Fallback: use top 5 courses by similarity
                    for course in unique_courses[:5]:
                        formatted_course = {
                            "course": course.get("title", ""),  # Use "course" to match output format
                            "platform": course.get("provider", ""),
                            "url": course.get("url", ""),
                            "difficulty": course.get("difficulty", "All Levels"),
                            "duration": course.get("duration", ""),
                            "relevance_score": course.get("similarity_score", 0.7),
                            "description": course.get("description", ""),
                            "skills": course.get("skills", []),
                            "type": "course"
                        }
                        formatted_courses.append(formatted_course)
                else:
                    # Last resort: try to search KB with a generic query
                    AgentLogger.log_warning(log_context, "⚠️ No courses found in search results. Attempting generic KB search...")
                    try:
                        kb = CourseKnowledgeBase()
                        # Try a very broad search
                        generic_courses = kb.search_courses(query="programming development", top_k=5, filters={"type": "course"})
                        for course in generic_courses[:5]:
                            formatted_course = {
                                "course": course.get("title", ""),  # Use "course" to match output format
                                "platform": course.get("provider", ""),
                                "url": course.get("url", ""),
                                "difficulty": course.get("difficulty", "All Levels"),
                                "duration": course.get("duration", ""),
                                "relevance_score": course.get("similarity_score", 0.5),
                                "description": course.get("description", ""),
                                "skills": course.get("skills", []),
                                "type": "course"
                            }
                            formatted_courses.append(formatted_course)
                        if formatted_courses:
                            AgentLogger.log_info(log_context, f"✅ Found {len(formatted_courses)} courses via generic search")
                    except Exception as e2:
                        AgentLogger.log_error(log_context, f"Generic KB search also failed: {e2}")
            
            if not formatted_other_materials:
                # Fallback when RAG didn't return selected_materials (no rationale/target_skill)
                if 'unique_other_materials' in locals() and unique_other_materials:
                    AgentLogger.log_info(log_context, f"🔄 Fallback: Using top {min(5, len(unique_other_materials))} materials by similarity score")
                    # Fallback: use top 5 materials by similarity
                    for material in unique_other_materials[:5]:
                        # Format material to match course structure (will be combined into course_recommendations)
                        formatted_material = {
                            "course": material.get("title", ""),  # Use "course" to match output format
                            "platform": material.get("provider", ""),
                            "type": material.get("type", "unknown"),  # book, paper, video, tutorial
                            "difficulty": material.get("difficulty", "All Levels"),
                            "duration": material.get("duration", "Self-paced"),
                            "relevance_score": material.get("similarity_score", 0.7),
                            "description": material.get("description", ""),
                            "url": material.get("url", "")
                        }
                        if material.get("author"):
                            formatted_material["author"] = material.get("author")
                        formatted_other_materials.append(formatted_material)
                else:
                    # Last resort: generic KB search (no rationale/target_skill)
                    AgentLogger.log_warning(log_context, "⚠️ No other materials found in search results. Attempting generic KB search...")
                    try:
                        kb = CourseKnowledgeBase()
                        # Try searching for books
                        for material_type in ["book", "paper", "video", "tutorial"]:
                            if len(formatted_other_materials) >= 5:
                                break
                            try:
                                materials = kb.search_courses(query="learning education", top_k=2, filters={"type": material_type})
                                for material in materials:
                                    if len(formatted_other_materials) >= 5:
                                        break
                                    # Format material to match course structure (will be combined into course_recommendations)
                                    formatted_material = {
                                        "course": material.get("title", ""),  # Use "course" to match output format
                                        "platform": material.get("provider", ""),
                                        "type": material.get("type", material_type),  # book, paper, video, tutorial
                                        "difficulty": material.get("difficulty", "All Levels"),
                                        "duration": material.get("duration", "Self-paced"),
                                        "relevance_score": material.get("similarity_score", 0.5),
                                        "description": material.get("description", ""),
                                        "url": material.get("url", "")
                                    }
                                    if material.get("author"):
                                        formatted_material["author"] = material.get("author")
                                    formatted_other_materials.append(formatted_material)
                            except Exception:
                                continue
                        if formatted_other_materials:
                            AgentLogger.log_info(log_context, f"✅ Found {len(formatted_other_materials)} materials via generic search")
                    except Exception as e2:
                        AgentLogger.log_error(log_context, f"Generic KB search for materials also failed: {e2}")
        
        # Combine all materials (courses, books, papers, videos, tutorials) into course_recommendations
        # Materials are already formatted to match course structure, so just combine them
        all_recommendations = formatted_courses[:5]
        
        # Add other materials (books, papers, videos, tutorials) to course_recommendations
        # Materials are already formatted with "course", "platform", "type", etc. fields
        for material in formatted_other_materials[:5]:
            all_recommendations.append(material)
        
        AgentLogger.log_info(log_context, f"✅ Generated {len(all_recommendations)} total recommendations ({len(formatted_courses)} courses + {len(formatted_other_materials)} other materials)")
        
        return {
            "course_recommendations": all_recommendations
        }
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        AgentLogger.log_error(log_context, f"Error in LLM-powered KB search: {e}", processing_time)
        import traceback
        AgentLogger.log_error(log_context, f"Traceback: {traceback.format_exc()}", processing_time)
        return {"course_recommendations": []}

def _validate_course_url_strict(url: str) -> bool:
    """Comprehensive URL validation for course recommendations."""
    if not url or not isinstance(url, str):
        return False
    
    # Basic URL format validation
    url = url.strip()
    if not url.startswith('https://'):
        return False
    
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        path = parsed.path.lower()
        query = parsed.query.lower()
        
        if not hostname:
            return False
        
        # Check against allowed domains
        if hostname in ALLOWED_HOSTS:
            return _validate_course_path_structure(hostname, path, query)
            
        # Check eTLD+1 domains
        parts = hostname.split('.')
        if len(parts) >= 2:
            etld1 = '.'.join(parts[-2:])
            if etld1 in ALLOWED_ETLD1:
                return _validate_course_path_structure(hostname, path, query)
            
            # Special case for LinkedIn Learning
            if etld1 == "linkedin.com" and ("learning" in hostname or path.startswith("/learning")):
                return _validate_course_path_structure(hostname, path, query)
            
        return False
        
    except Exception:
        return False

def _validate_course_path_structure(hostname: str, path: str, query: str) -> bool:
    """Validate that the URL structure indicates a specific course, not a search or generic page."""
    
    # Invalid patterns that indicate search pages or generic content
    invalid_patterns = [
        '/search', '/browse', '/courses', '/catalog', '/explore',
        '/category', '/subject', '/topic', '/tag', '/filter',
        '/results', '/find', '/discover', '/all-courses',
        '/free-courses', '/popular', '/featured', '/trending',
        '/list', '/directory', '/index', '/home', '/main'
    ]
    
    # Check for invalid patterns in path
    for pattern in invalid_patterns:
        if pattern in path:
            return False
    
    # Check for search parameters in query
    search_params = ['q=', 'query=', 'search=', 'keyword=', 'term=', 'keywords=', 'search_query=', 'search_query=']
    for param in search_params:
        if param in query:
            return False
    
    # Platform-specific validation
    if 'coursera.org' in hostname:
        return _validate_coursera_url(path)
    elif 'udemy.com' in hostname:
        return _validate_udemy_url(path)
    elif 'edx.org' in hostname:
        return _validate_edx_url(path)
    elif 'linkedin.com' in hostname:
        return _validate_linkedin_learning_url(path)
    elif 'youtube.com' in hostname:
        # YouTube URLs use query parameters, check for valid patterns
        youtube_patterns = ['v=', 'list=']
        return any(pattern in query for pattern in youtube_patterns) and _validate_youtube_url(path)
    elif 'udacity.com' in hostname:
        return _validate_udacity_url(path)
    elif 'pluralsight.com' in hostname:
        return _validate_pluralsight_url(path)
    elif 'datacamp.com' in hostname:
        return _validate_datacamp_url(path)
    elif 'khanacademy.org' in hostname:
        return _validate_khan_academy_url(path)
    elif 'freecodecamp.org' in hostname:
        return _validate_freecodecamp_url(path)
    elif any(uni in hostname for uni in ['mit.edu', 'stanford.edu', 'harvard.edu', 'berkeley.edu']):
        return _validate_university_url(path)
    else:
        # For other platforms, basic validation
        return len(path) > 1 and not any(pattern in path for pattern in invalid_patterns)

def _validate_coursera_url(path: str) -> bool:
    """Validate Coursera course URLs with stricter checks."""
    # Valid patterns: /learn/course-name, /specializations/course-name, /professional-certificates/course-name
    valid_patterns = ['/learn/', '/specializations/', '/professional-certificates/']
    if not any(pattern in path for pattern in valid_patterns):
        return False
    
    # Must have at least 3 path segments: /learn/course-name
    path_parts = [p for p in path.split('/') if p]  # Remove empty strings
    if len(path_parts) < 2:
        return False
    
    # Course name should be after the pattern (e.g., /learn/course-name)
    # Check that course name exists and is not empty
    for pattern in valid_patterns:
        if pattern in path:
            # Get the part after the pattern
            after_pattern = path.split(pattern, 1)[1] if pattern in path else ''
            # Remove trailing slashes and query params
            course_slug = after_pattern.split('/')[0].split('?')[0].split('#')[0]
            # Course slug should be non-empty and look like a valid slug (kebab-case, lowercase, alphanumeric with hyphens)
            if not course_slug or len(course_slug) < 3:
                return False
            # Should be mostly lowercase alphanumeric with hyphens/underscores (typical course slug format)
            if not all(c.isalnum() or c in ['-', '_'] for c in course_slug):
                return False
            
            # Reject common placeholder patterns
            invalid_patterns = [
                'test', 'example', 'demo', 'sample', 'placeholder', 'course-name', 'new-course',
                'tutorial', 'learn', 'training', 'class', 'lesson', 'module', 'course', 'intro',
                'introduction', 'basics', 'beginner', 'advanced', 'intermediate', 'guide',
                'how-to', 'getting-started', 'overview', 'summary', 'review'
            ]
            # Check if slug is just a generic word (not a real course)
            if course_slug.lower() in invalid_patterns:
                return False
            # Check if slug contains placeholder patterns
            if any(pattern in course_slug.lower() for pattern in invalid_patterns):
                # But allow if it's part of a longer, meaningful slug (e.g., "machine-learning-for-beginners" is OK)
                if len(course_slug) < 10:  # Very short slugs with placeholder words are suspicious
                    return False
            
            # Course slug should be meaningful (at least 5 characters for real courses)
            if len(course_slug) < 5:
                return False
    
    return True

def _validate_udemy_url(path: str) -> bool:
    """Validate Udemy course URLs with stricter checks."""
    # Valid pattern: /course/course-name/ or /course/course-name
    if not path.startswith('/course/'):
        return False
    
    # Must have course name after /course/
    path_parts = [p for p in path.split('/') if p]  # Remove empty strings
    if len(path_parts) < 2:  # At least ['course', 'course-name']
        return False
    
    # Extract course slug (the part after /course/)
    course_slug = path_parts[1] if len(path_parts) > 1 else ''
    # Remove query params and fragments
    course_slug = course_slug.split('?')[0].split('#')[0]
    
    # Course slug validation:
    # - Should be non-empty
    # - Should be at least 3 characters (real Udemy courses have meaningful slugs)
    # - Should be lowercase (Udemy URLs are lowercase)
    # - Should contain alphanumeric characters and hyphens/underscores
    if not course_slug or len(course_slug) < 3:
        return False
    
    # Check if it's lowercase (Udemy URLs are always lowercase)
    if course_slug != course_slug.lower():
        return False
    
    # Should be valid slug format (alphanumeric, hyphens, underscores)
    if not all(c.isalnum() or c in ['-', '_'] for c in course_slug):
        return False
    
    # Reject common placeholder patterns - expanded list
    invalid_patterns = [
        'test', 'example', 'demo', 'sample', 'placeholder', 'course-name', 'new-course',
        'tutorial', 'learn', 'training', 'class', 'lesson', 'module', 'course', 'intro',
        'introduction', 'basics', 'beginner', 'advanced', 'intermediate', 'guide',
        'how-to', 'getting-started', 'overview', 'summary', 'review'
    ]
    # Check if slug is just a generic word (not a real course)
    if course_slug.lower() in invalid_patterns:
        return False
    # Check if slug contains placeholder patterns
    if any(pattern in course_slug.lower() for pattern in invalid_patterns):
        # But allow if it's part of a longer, meaningful slug (e.g., "python-for-beginners" is OK)
        if len(course_slug) < 10:  # Very short slugs with placeholder words are suspicious
            return False
    
    # Course slug should be meaningful (at least 5 characters for real courses)
    if len(course_slug) < 5:
        return False
    
    return True

def _validate_edx_url(path: str) -> bool:
    """Validate edX course URLs."""
    # Valid patterns: /course/course-name, /learn/course-name, /micromasters/course-name
    valid_patterns = ['/course/', '/learn/', '/micromasters/']
    return any(pattern in path for pattern in valid_patterns) and len(path.split('/')) >= 3

def _validate_linkedin_learning_url(path: str) -> bool:
    """Validate LinkedIn Learning course URLs."""
    # Valid pattern: /learning/course-name or /learning/course-name/
    # Must have at least 3 path segments: /learning/course-name
    return path.startswith('/learning/') and len(path.split('/')) >= 3 and path != '/learning/'

def _validate_youtube_url(path: str) -> bool:
    """Validate YouTube educational content URLs."""
    # Valid patterns: /watch?v=, /playlist?list=, /channel/
    # Note: YouTube uses query parameters, not path parameters
    return True  # YouTube URLs are handled by query parameter validation

def _validate_udacity_url(path: str) -> bool:
    """Validate Udacity course URLs."""
    # Valid patterns: /course/course-name, /nanodegree/course-name
    valid_patterns = ['/course/', '/nanodegree/']
    return any(pattern in path for pattern in valid_patterns) and len(path.split('/')) >= 3

def _validate_pluralsight_url(path: str) -> bool:
    """Validate Pluralsight course URLs."""
    # Valid pattern: /courses/course-name
    return path.startswith('/courses/') and len(path.split('/')) >= 3

def _validate_datacamp_url(path: str) -> bool:
    """Validate DataCamp course URLs."""
    # Valid patterns: /courses/course-name, /tracks/course-name
    valid_patterns = ['/courses/', '/tracks/']
    return any(pattern in path for pattern in valid_patterns) and len(path.split('/')) >= 3

def _validate_khan_academy_url(path: str) -> bool:
    """Validate Khan Academy course URLs."""
    # Valid patterns: /courses/course-name, /learn/course-name
    valid_patterns = ['/courses/', '/learn/']
    return any(pattern in path for pattern in valid_patterns) and len(path.split('/')) >= 3

def _validate_freecodecamp_url(path: str) -> bool:
    """Validate FreeCodeCamp course URLs."""
    # Valid patterns: /learn/course-name, /certifications/course-name
    valid_patterns = ['/learn/', '/certifications/']
    return any(pattern in path for pattern in valid_patterns) and len(path.split('/')) >= 3

def _validate_university_url(path: str) -> bool:
    """Validate university course URLs."""
    # Valid patterns: /courses/course-name, /learn/course-name, /course/course-name
    valid_patterns = ['/courses/', '/learn/', '/course/']
    return any(pattern in path for pattern in valid_patterns) and len(path.split('/')) >= 3

async def _validate_and_filter_courses(courses: List[Dict[str, Any]], log_context: Dict[str, Any], timing_breakdown: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Validate and filter course recommendations - only validates title, type, and platform (no URL validation)."""
    start_time = log_context.get("start_time", time.time())
    
    if not isinstance(courses, list):
        return []
    
    valid_courses = []
    invalid_count = 0
    invalid_reasons = []
    
    AgentLogger.log_info(log_context, f"Starting validation of {len(courses)} course recommendations (no URL validation)")
    if len(courses) > 0:
        AgentLogger.log_info(log_context, f"Sample course structure: {courses[0] if courses else 'N/A'}")
    
    for i, course in enumerate(courses):
        if not isinstance(course, dict):
            invalid_count += 1
            invalid_reasons.append(f"Course {i+1}: Not a dictionary")
            continue
        
        # Check required fields (no URL required)
        title = course.get('title', '').strip()
        provider = course.get('provider', '').strip()
        author = course.get('author', '') if isinstance(course.get('author', ''), str) else ''
        author = author.strip()
        ctype = str(course.get('type', 'course')).strip().lower()
        
        # Validate based on type
        if ctype == 'book':
            # Books require title and author
            if not title or not author:
                invalid_count += 1
                invalid_reasons.append(f"Resource {i+1}: Book missing title or author")
                continue
        elif ctype == 'course':
            # Courses require title and provider
            if not title:
                invalid_count += 1
                invalid_reasons.append(f"Course {i+1}: Missing title")
                continue
            if not provider:
                invalid_count += 1
                invalid_reasons.append(f"Course {i+1}: Missing provider")
                continue
        elif ctype in ['blog', 'webpage']:
            # Blogs and webpages require title (provider optional)
            if not title:
                invalid_count += 1
                invalid_reasons.append(f"Resource {i+1}: {ctype.capitalize()} missing title")
                continue
        else:
            # Other types (paper, video) require at least title
            if not title:
                invalid_count += 1
                invalid_reasons.append(f"Resource {i+1}: Missing title")
                continue
        
        # Build validated course (no URL field)
        validated_course = {
            'title': title[:200],
            'type': ctype,
            'difficulty': course.get('difficulty', 'All Levels'),
            'duration': _norm_duration(course.get('duration', '')),
            'relevance_score': max(0.0, min(float(course.get('relevance_score') or 0.8), 1.0)),
            'target_skill': course.get('target_skill', '')[:100],
            'description': course.get('description', '')[:300]
        }
        
        # Add provider for courses (required)
        if ctype == 'course' and provider:
            validated_course['provider'] = provider[:100]
        elif provider:
            # Optional provider for other types
            validated_course['provider'] = provider[:100]
        
        # Add author for books (required) or optional for other types
        if ctype == 'book' and author:
            validated_course['author'] = author[:120]
        elif author:
            validated_course['author'] = author[:120]
        
        # Normalize difficulty
        difficulty = validated_course['difficulty'].strip().title()
        if difficulty in DIFFICULTY_ALLOWED:
            validated_course['difficulty'] = difficulty
        else:
            validated_course['difficulty'] = 'All Levels'
        
        valid_courses.append(validated_course)
        AgentLogger.log_info(log_context, f"✅ ACCEPTED: '{title}' (type: {ctype})")
    
    # Log detailed validation results
    if invalid_count > 0:
        AgentLogger.log_warning(log_context, f"🚫 FILTERED OUT {invalid_count} invalid course recommendations")
        for reason in invalid_reasons[:10]:  # Log first 10 reasons
            AgentLogger.log_warning(log_context, f"  - {reason}")
        if len(invalid_reasons) > 10:
            AgentLogger.log_warning(log_context, f"  - ... and {len(invalid_reasons) - 10} more")
    
    # Remove duplicates based on title and type
    original_count = len(valid_courses)
    seen: Set[Tuple[str, str]] = set()
    deduped_courses = []
    for c in valid_courses:
        key = ((c.get('title') or '').lower().strip(), (c.get('type') or '').lower().strip())
        if key not in seen:
            seen.add(key)
            deduped_courses.append(c)
    valid_courses = deduped_courses
    duplicates_removed = original_count - len(valid_courses)
    
    if duplicates_removed > 0:
        AgentLogger.log_info(log_context, f"🔄 Removed {duplicates_removed} duplicate courses")
    
    # Limit to maximum recommendations
    if len(valid_courses) > MAX_RECOMMENDATIONS_COUNT:
        AgentLogger.log_info(log_context, f"📊 Limited to {MAX_RECOMMENDATIONS_COUNT} courses (had {len(valid_courses)})")
        valid_courses = valid_courses[:MAX_RECOMMENDATIONS_COUNT]
    
    AgentLogger.log_info(log_context, f"✅ FINAL RESULT: {len(valid_courses)} valid course recommendations (no URL validation performed)")
    
    return valid_courses

async def _validate_and_filter_additional_resources(resources: List[Dict[str, Any]], log_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate and filter books, research papers, videos, blogs, and webpages. Keep only allowed domains and accessible URLs."""
    if not isinstance(resources, list):
        return []
    valid: List[Dict[str, Any]] = []
    for i, r in enumerate(resources):
        if not isinstance(r, dict):
            continue
        title = str(r.get('title', '')).strip()
        url = str(r.get('url', '')).strip()
        rtype = str(r.get('type', '')).strip().lower()
        source = str(r.get('source', '')).strip()
        target_skill = str(r.get('target_skill', '')).strip()
        if not title or not rtype:
            continue
        # Books may not have URLs, but other types must have URLs
        if rtype != 'book' and not url:
            continue
        if url and not url.startswith('https://'):
            continue
        if url and not _is_allowed_resource_domain(url):
            continue
        # Type-specific basic checks
        if url:
            u = urlparse(url)
            host = (u.hostname or '').lower()
            path = (u.path or '').lower()
            if rtype == 'video':
                if 'youtube.com' not in host or not (path.startswith('/watch') or path.startswith('/playlist')):
                    continue
            elif rtype == 'paper':
                if 'arxiv.org' in host and not (path.startswith('/abs/') or path.startswith('/pdf/')):
                    continue
                if 'ieeexplore.ieee.org' in host and '/document/' not in path:
                    continue
                if 'dl.acm.org' in host and '/doi/' not in path:
                    continue
            elif rtype == 'book':
                if url:  # Books may have optional URLs
                    if 'oreilly.com' in host and '/library/' not in path:
                        continue
                    if 'packtpub.com' in host and '/product/' not in path:
                        continue
            elif rtype in ['blog', 'webpage']:
                # Blogs and webpages should have valid paths (not just homepage)
                invalid_patterns = ['/search', '/browse', '/category', '/tag', '/archive', '/page/', '/?page=']
                if any(pattern in path or pattern in u.query.lower() for pattern in invalid_patterns):
                    continue
                # Must have a meaningful path (not just root)
                if len(path) <= 1 and not u.query:
                    continue
        # Accessibility check (skip for books without URLs)
        if rtype == 'book' and not url:
            # Books without URLs are automatically valid
            item = {
                'type': rtype,
                'title': title[:200],
                'source': source[:120] if source else '',
                'relevance_score': max(0.0, min(float(r.get('relevance_score') or 0.7), 1.0)),
                'target_skill': target_skill[:120],
                'description': str(r.get('description', ''))[:300]
            }
            if r.get('author'):
                item['author'] = str(r.get('author'))[:120]
            valid.append(item)
        elif url:
            url_validation = await _generate_url_validation_info_async(url, title, source or rtype, rtype)
            if url_validation.get('status') == 'valid' and url_validation.get('accessible') is True:
                item = {
                    'type': rtype,
                    'title': title[:200],
                    'source': source[:120] if source else '',
                    'url': _normalize_course_url(url),
                    'relevance_score': max(0.0, min(float(r.get('relevance_score') or 0.7), 1.0)),
                    'target_skill': target_skill[:120],
                    'description': str(r.get('description', ''))[:300]
                }
                if rtype == 'video' and r.get('duration'):
                    item['duration'] = str(r.get('duration'))[:60]
                if rtype in ['blog', 'webpage', 'book'] and r.get('author'):
                    item['author'] = str(r.get('author'))[:120]
                item['url_validation'] = url_validation
                valid.append(item)
    # Dedupe by URL+title
    valid = _dedupe_courses(valid)
    # Cap results
    if len(valid) > MAX_RECOMMENDATIONS_COUNT:
        valid = valid[:MAX_RECOMMENDATIONS_COUNT]
    return valid

def _is_course_url_high_quality(url: str, title: str, provider: str) -> bool:
    """Additional quality checks for course URLs."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        path = parsed.path.lower()
        
        # Check if URL looks like a real course (not generic)
        if len(path.split('/')) < 3:
            return False
        
        # Extract course slug for validation
        course_slug = ''
        if 'udemy.com' in hostname and '/course/' in path:
            # Extract slug after /course/
            parts = path.split('/course/')
            if len(parts) > 1:
                course_slug = parts[1].split('/')[0].split('?')[0]
        elif 'coursera.org' in hostname:
            # Extract slug after /learn/, /specializations/, or /professional-certificates/
            for pattern in ['/learn/', '/specializations/', '/professional-certificates/']:
                if pattern in path:
                    course_slug = path.split(pattern, 1)[1].split('/')[0].split('?')[0]
                    break
        
        # Check for placeholder/invalid course slugs
        if course_slug:
            invalid_slugs = [
                'course-name', 'test-course', 'example-course', 'new-course',
                'course', 'test', 'example', 'demo', 'sample', 'placeholder',
                'tutorial', 'learn', 'training', 'class'
            ]
            if course_slug.lower() in invalid_slugs or any(inv in course_slug.lower() for inv in ['-course', '-test', '-example']):
                return False
            
            # Course slug should be meaningful (at least 5 characters for real courses)
            if len(course_slug) < 5:
                return False
        
        # Check for suspicious patterns in full URL
        suspicious_patterns = [
            'test', 'demo', 'sample', 'example', 'placeholder',
            'coming-soon', 'under-construction', 'maintenance'
        ]
        
        url_lower = url.lower()
        for pattern in suspicious_patterns:
            if pattern in url_lower:
                return False
        
        # Check if title and provider match the URL domain
        provider_lower = provider.lower()
        if 'coursera' in provider_lower and 'coursera.org' not in hostname:
            return False
        if 'udemy' in provider_lower and 'udemy.com' not in hostname:
            return False
        if 'edx' in provider_lower and 'edx.org' not in hostname:
            return False
        if 'linkedin' in provider_lower and 'linkedin.com' not in hostname:
            return False
        
        return True
        
    except Exception:
        return False

async def _check_url_accessibility(url: str, timeout: int = 5) -> Tuple[bool, str, Optional[int]]:
    """Check URL accessibility using a lightweight GET with a real User-Agent.
    Accept 200 and standard redirects. Do not download full content.
    For courses, this is critical - courses must return 200 or redirect to be valid.
    Returns: (is_accessible, message, status_code)
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout), headers=headers) as session:
            async with session.get(url, allow_redirects=True) as response:
                # Read a tiny portion to establish connection, then release
                try:
                    await response.content.readany()
                except Exception:
                    pass
                status = response.status
                if status == 200:
                    return True, f"URL accessible (HTTP {status})", status
                if status in [301, 302, 303, 307, 308]:
                    # Follow redirect and check final status
                    final_url = str(response.url)
                    return True, f"URL accessible with redirect (HTTP {status}) to {final_url}", status
                # Return status code so we can distinguish between 403 (bot blocking) and 404 (not found)
                return False, f"URL not accessible (HTTP {status})", status
    except aiohttp.ClientTimeout:
        return False, "URL check timed out", None
    except aiohttp.ClientError as e:
        return False, f"URL not accessible: {str(e)}", None
    except Exception as e:
        return False, f"URL validation error: {str(e)}", None

def _generate_url_validation_info(url: str, title: str, provider: str) -> Dict[str, Any]:
    """Generate URL validation information for the course output."""
    try:
        # Basic URL validation
        is_valid = _validate_course_url_strict(url)
        
        if not is_valid:
            return {
                "status": "invalid",
                "accessible": False,
                "message": "URL failed validation checks"
            }
        
        # For now, return pending status - will be updated by actual HTTP check
        return {
            "status": "pending",
            "accessible": None,
            "message": "URL validation in progress"
        }
            
    except Exception as e:
        return {
            "status": "error",
            "accessible": False,
            "message": f"Validation error: {str(e)}"
        }

async def _generate_url_validation_info_async(url: str, title: str, provider: str, resource_type: str = "course") -> Dict[str, Any]:
    """Generate URL validation information with actual HTTP checking."""
    try:
        # FAST PATH: Early rejection of obviously invalid URLs before HTTP check
        # This saves time by avoiding HTTP requests for clearly invalid URLs
        if resource_type == "course":
            # For courses, do strict validation first
            is_valid = _validate_course_url_strict(url)
            if not is_valid:
                return {
                    "status": "invalid",
                    "accessible": False,
                    "message": f"Course URL failed structure validation"
                }
            # Additional quality check before HTTP request
            if not _is_course_url_high_quality(url, title, provider):
                return {
                    "status": "invalid",
                    "accessible": False,
                    "message": f"Course URL failed quality checks (likely placeholder or invalid slug)"
                }
        elif resource_type in ["blog", "webpage"]:
            # For blogs/webpages, use resource domain validation
            is_valid = _is_allowed_resource_domain(url)
            if not is_valid:
                return {
                    "status": "invalid",
                    "accessible": False,
                    "message": f"Blog/webpage URL failed domain validation"
                }
            # Additional path validation for blogs/webpages
            try:
                parsed = urlparse(url)
                path = (parsed.path or '').lower()
                query = (parsed.query or '').lower()
                # Exclude invalid patterns
                invalid_patterns = ['/search', '/browse', '/category', '/tag', '/archive']
                if any(pattern in path or pattern in query for pattern in invalid_patterns):
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Blog/webpage URL appears to be a search/category page"
                    }
                elif '/page/' in path or '?page=' in query:
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Blog/webpage URL appears to be a pagination page"
                    }
                elif len(path) <= 1 and not query:
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Blog/webpage URL appears to be a homepage"
                    }
            except Exception:
                return {
                    "status": "invalid",
                    "accessible": False,
                    "message": f"Blog/webpage URL failed path validation"
                }
        else:
            # For other types (paper, video), use resource domain validation
            is_valid = _is_allowed_resource_domain(url)
            if not is_valid:
                return {
                    "status": "invalid",
                    "accessible": False,
                    "message": f"Resource URL failed domain validation for {resource_type}"
                }
        
        # Check actual accessibility (only for URLs that passed structure validation)
        is_accessible, message, status_code = await _check_url_accessibility(url)
        
        # Determine platform-specific message and check if it's a trusted platform
        parsed = urlparse(url)
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        
        # List of trusted educational platforms - accept even if HTTP check fails (might be bot blocking)
        trusted_platforms = {
            'youtube.com': 'YouTube',
            'coursera.org': 'Coursera',
            'udemy.com': 'Udemy',
            'edx.org': 'edX',
            'linkedin.com': 'LinkedIn Learning',
            'pluralsight.com': 'Pluralsight',
            'freecodecamp.org': 'FreeCodeCamp',
            'medium.com': 'Medium',
            'dev.to': 'Dev.to',
            'hashnode.com': 'Hashnode',
            'github.com': 'GitHub',
            'react.dev': 'React',
            'vuejs.org': 'Vue.js',
            'angular.io': 'Angular',
            'nodejs.org': 'Node.js',
            'mongodb.com': 'MongoDB',
            'postgresql.org': 'PostgreSQL',
            'docker.com': 'Docker',
            'kubernetes.io': 'Kubernetes',
            'terraform.io': 'Terraform',
            'azure.microsoft.com': 'Azure',
            'aws.amazon.com': 'AWS',
            'cloud.google.com': 'Google Cloud',
            'developers.google.com': 'Google Developers'
        }
        
        is_trusted = any(domain in hostname for domain in trusted_platforms.keys())
        platform_name = next((name for domain, name in trusted_platforms.items() if domain in hostname), "URL")
        
        # For COURSES from trusted platforms, accept if URL structure is valid even if HTTP check fails
        # BUT: Reject 404s (course doesn't exist) - only accept 403s (bot blocking) or timeouts/errors
        # Course platforms (Coursera, Udemy, edX) often block bots but URLs are valid for users
        # For other resource types (blogs, webpages), be more lenient as they might block bots
        if resource_type == "course":
            # Check if it's a trusted course platform
            course_platforms = ['coursera.org', 'udemy.com', 'edx.org', 'linkedin.com', 'pluralsight.com']
            is_course_platform = any(domain in hostname for domain in course_platforms)
            
            if is_accessible:
                # Only accept if HTTP check actually succeeded (200 or valid redirect)
                platform_msg = f"{platform_name} course verified accessible"
                return {
                    "status": "valid",
                    "accessible": True,
                    "message": f"{platform_msg} - {message}"
                }
            else:
                # HTTP check failed - reject the course
                # Even from trusted platforms, we need actual HTTP success (200/redirect) to accept
                if status_code == 404:
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Course URL not found (HTTP 404) - course does not exist on {platform_name}"
                    }
                elif status_code == 403:
                    # 403 could be bot blocking, but we can't verify the course exists
                    # Reject to ensure only working links are returned
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Course URL blocked or inaccessible (HTTP 403) - cannot verify course exists on {platform_name}"
                    }
                elif status_code is None:
                    # Timeout or connection error - reject to ensure reliability
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Course URL validation timed out or connection failed - cannot verify course exists on {platform_name}"
                    }
                else:
                    # Other error status codes - reject
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Course URL not accessible (HTTP {status_code}) - {message}"
                    }
        else:
            # For blogs, webpages, videos, papers - require actual HTTP success
            if is_accessible:
                # Only accept if HTTP check actually succeeded (200 or valid redirect)
                platform_msg = f"{platform_name} resource verified accessible"
                return {
                    "status": "valid",
                    "accessible": True,
                    "message": f"{platform_msg} - {message}"
                }
            else:
                # HTTP check failed - reject the resource
                if status_code == 404:
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Resource URL not found (HTTP 404) - resource does not exist on {platform_name}"
                    }
                elif status_code == 403:
                    # 403 could be bot blocking, but we can't verify the resource exists
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Resource URL blocked or inaccessible (HTTP 403) - cannot verify resource exists on {platform_name}"
                    }
                elif status_code is None:
                    # Timeout or connection error - reject to ensure reliability
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Resource URL validation timed out or connection failed - cannot verify resource exists on {platform_name}"
                    }
                else:
                    # Other error status codes - reject
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"URL not accessible (HTTP {status_code}) - {message}"
                    }
            
    except Exception as e:
        return {
            "status": "error",
            "accessible": False,
            "message": f"Validation error: {str(e)}"
        }

async def _validate_urls_batch_async(courses: List[Dict[str, Any]], log_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Validate multiple URLs concurrently and return only those verified accessible.
    Items without URLs are only passed through if type == 'book'.
    """
    if not courses:
        return []
    
    if log_context is None:
        log_context = {"tenant_id": "default", "user_id": "unknown"}
    
    # Prepare inputs for validation
    validation_inputs: List[Tuple[Dict[str, Any], str, str, str]] = []
    accessible_courses: List[Dict[str, Any]] = []
    for course in courses:
        if not isinstance(course, dict):
            continue
        url = course.get('url')
        ctype = str(course.get('type', 'course')).strip().lower()
        if url:
            validation_inputs.append((
                course,
                url,
                course.get('course', course.get('title', '')),
                course.get('platform', course.get('provider', ''))
            ))
        else:
            # Pass-through for books without URL
            if ctype == 'book' and course.get('title') and (course.get('author') or course.get('provider')):
                course['url_validation'] = {
                    'status': 'not_applicable',
                    'accessible': None,
                    'message': 'URL not required for books'
                }
                accessible_courses.append(course)
    
    # Concurrency limiter to avoid overwhelming network (tunable via COURSE_URL_HTTP_VALIDATION_MAX_PARALLEL)
    semaphore = asyncio.Semaphore(_COURSE_URL_HTTP_VALIDATION_MAX_PARALLEL)
    
    async def validate_one(c: Dict[str, Any], url: str, title: str, provider: str) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
        async with semaphore:
            try:
                # Get resource type from course dict
                resource_type = str(c.get('type', 'course')).strip().lower()
                url_validation = await _generate_url_validation_info_async(url, title, provider, resource_type)
                return c, url_validation, None
            except Exception as e:
                return c, {"status": "error", "accessible": False, "message": f"Validation error: {str(e)}"}, str(e)
    
    tasks = [asyncio.create_task(validate_one(c, u, t, p)) for (c, u, t, p) in validation_inputs]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    inaccessible_count = 0
    inaccessible_reasons: List[str] = []
    
    for course, url_validation, err in results:
        course['url_validation'] = url_validation
        if url_validation.get('status') == 'valid' and url_validation.get('accessible') is True:
            accessible_courses.append(course)
            if log_context:
                AgentLogger.log_info(log_context, f"✅ URL ACCESSIBLE: {course.get('course', course.get('title', 'Unknown'))} - {url_validation.get('message', '')}")
        else:
            inaccessible_count += 1
            reason = f"Course '{course.get('course', course.get('title', 'Unknown'))}': {url_validation.get('message', 'Unknown error')}"
            inaccessible_reasons.append(reason)
            if log_context:
                AgentLogger.log_warning(log_context, f"⚠️ URL NOT VERIFIED: {reason}")
    
    # Log results (filtering applied)
    if log_context and inaccessible_count > 0:
        AgentLogger.log_warning(log_context, f"ℹ️ {inaccessible_count} course URLs removed (not verified accessible)")
        for reason in inaccessible_reasons[:5]:
            AgentLogger.log_warning(log_context, f"  - {reason}")
        if len(inaccessible_reasons) > 5:
            AgentLogger.log_warning(log_context, f"  - ... and {len(inaccessible_reasons) - 5} more")
    
    if log_context:
        AgentLogger.log_info(log_context, f"✅ Returning {len(accessible_courses)} verified-accessible courses")
    
    return accessible_courses

async def _llm_validate_course_url_batch(courses: List[Dict[str, Any]], log_context: Dict[str, Any]) -> List[bool]:
    """Use LLM to validate multiple course URLs in a single call for efficiency."""
    try:
        # Prepare course information for batch validation
        course_info = []
        for i, course in enumerate(courses):
            course_info.append({
                "index": i,
                "type": course.get('type', 'course'),
                "title": course.get('title', ''),
                "provider": course.get('provider', ''),
                "url": course.get('url', ''),
                "author": course.get('author', '')  # For books
            })
        
        validation_prompt = f"""
You are a learning resource URL validation expert. Analyze the following learning resources and determine which URLs are legitimate, direct links to real educational content.

**Resource Information:**
{json.dumps(course_info, indent=2)}

**Resource Types to Validate:**
- **course**: Structured online courses (Coursera, Udemy, edX, LinkedIn Learning)
- **book**: Books (may not have URLs - if missing URL but has author, accept it)
- **blog**: Educational blog posts/articles (Medium, Dev.to, Hashnode, FreeCodeCamp, etc.)
- **webpage**: Documentation pages, developer guides, official docs (React.dev, Vue.js, MDN, etc.)
- **video**: YouTube videos or playlists
- **paper**: Research papers (arXiv, IEEE, ACM)

**Validation Criteria for each resource:**
1. **For courses (CRITICAL)**: 
   - Is this a direct link to a SPECIFIC, REAL course that actually exists (not a search/browse page)?
   - Does the course slug look legitimate (proper kebab-case format, not placeholder text)?
   - For Udemy: Course slug should be lowercase, alphanumeric with hyphens (e.g., "complete-python-bootcamp", NOT "course-name" or "test-course")
   - For Coursera: Course slug should follow proper format (e.g., "machine-learning" not "course" or "test")
   - REJECT if URL looks like a placeholder, example, or hypothetical course
2. **For books**: If URL is missing but author is provided, ACCEPT it. If URL is provided, validate it.
3. **For blogs/webpages**: Is this a direct link to a specific article/page (not a category/listing page)?
4. **For videos**: Is this a direct link to a YouTube video or playlist?
5. **For papers**: Is this a direct link to a research paper?
6. Does the URL structure match the expected pattern for the provider/type?
7. Does the URL appear to point to real, existing educational content (not hypothetical/fake)?
8. Is the provider correctly identified (if provided)?
9. Does the URL look legitimate and not like a placeholder, test link, or example URL?

**Common Invalid Patterns to REJECT:**
- Placeholder slugs: "course-name", "test-course", "example-course", "new-course"
- Generic patterns: "course", "learn", "tutorial" as standalone slugs
- URLs that look generated or hypothetical rather than real courses

**Provider-Specific Patterns:**
- **Courses**: 
  - Coursera: /learn/, /specializations/, /professional-certificates/
  - Udemy: /course/course-name/
  - edX: /course/, /learn/, /micromasters/
  - LinkedIn Learning: /learning/course-name
- **Blogs**: Medium, Dev.to, Hashnode, FreeCodeCamp articles (not category/search pages)
- **Webpages**: Official docs, developer guides (React.dev, Vue.js, MDN, etc.)
- **Videos**: YouTube /watch?v= or /playlist?list=
- **Papers**: arXiv /abs/ or /pdf/, IEEE /document/, ACM /doi/

**CRITICAL RULES:**
- **For COURSES (Udemy, Coursera, edX, LinkedIn Learning)**: 
  - URL structure MUST be valid and match platform patterns
  - If HTTP check succeeds (200 or redirect), accept it
  - If HTTP check fails BUT URL structure is valid and from trusted platform, ACCEPT it (platforms may block bots)
  - REJECT only if URL structure is invalid or from unknown platform
- **Books without URLs but with authors are VALID** - accept them
- **Blogs and webpages are VALID learning resources** - accept educational articles/docs
- **YouTube videos are VALID** - accept individual videos and playlists
- **Documentation pages are VALID** - accept official docs and guides
- **GitHub repositories can be VALID** if they're educational resources (tutorials, guides)
- Reject only: search pages, category pages, listing pages, marketing pages, homepages, inaccessible courses

**IMPORTANT: You must return ONLY valid JSON. No additional text, explanations, or formatting outside the JSON object.**

**Response Format (return exactly this structure):**
{{
    "validations": [
        {{
            "index": 0,
            "is_valid": true,
            "confidence": 0.85,
            "reason": "Valid course URL with proper structure"
        }},
        {{
            "index": 1,
            "is_valid": false,
            "confidence": 0.2,
            "reason": "Appears to be search page"
        }}
    ]
}}

Analyze all courses and return a JSON object with validations array."""

        # Issue 5.1: Use structured output for type-safe parsing
        validation_result = None
        try:
            result: BatchValidationResult = await asyncio.wait_for(
                invoke_structured_llm(
                    validation_prompt,
                    BatchValidationResult,
                    task_type=TaskType.COURSE_VALIDATION,
                    preferred_model="gemini-2.5-flash",
                    agent_name="market_and_course_recommender",
                    max_output_tokens=2000,
                    temperature=0.1,
                    timeout=10.0,
                    raise_on_fallback=False,
                    skip_cache=True,
                ),
                timeout=10,
            )
            validation_result = {"validations": [v.model_dump() for v in result.validations]}
            AgentLogger.log_debug(log_context, f"Structured validation output: {len(result.validations)} validations")
        except asyncio.TimeoutError:
            AgentLogger.log_warning(log_context, "Structured validation timed out, falling back to manual parsing")
        except Exception as structured_err:
            AgentLogger.log_warning(log_context, f"Structured validation failed ({structured_err}), falling back to manual parsing")
        
        # Fallback to manual parsing if structured output failed
        if not validation_result:
            llm_response = await asyncio.wait_for(
                invoke_llm(
                    prompt=validation_prompt, 
                    task_type="course_validation",
                    agent_name="market_and_course_recommender",
                    preferred_model="gemini-2.5-flash"
                ), timeout=10
            )
            response_text = _to_text(llm_response)
            AgentLogger.log_info(log_context, f"LLM Response: {response_text[:500]}...")
            validation_result = _extract_json_from_response(response_text, 2000)
        
        if isinstance(validation_result, dict) and 'validations' in validation_result:
            validations = validation_result['validations']
            results = [False] * len(courses)
            
            for validation in validations:
                if isinstance(validation, dict):
                    index = validation.get('index', -1)
                    is_valid = validation.get('is_valid', False)
                    confidence = validation.get('confidence', 0.0)
                    reason = validation.get('reason', 'No reason provided')
                    
                    if 0 <= index < len(courses):
                        if is_valid and confidence > 0.6:
                            results[index] = True
                            AgentLogger.log_info(log_context, f"✅ LLM VALIDATED: '{courses[index].get('title', '')}' - {reason} (confidence: {confidence})")
                        else:
                            AgentLogger.log_warning(log_context, f"❌ LLM REJECTED: '{courses[index].get('title', '')}' - {reason} (confidence: {confidence})")
            
            return results
        else:
            AgentLogger.log_warning(log_context, f"❌ LLM BATCH VALIDATION FAILED - Invalid response format: {validation_result}")
            return await _fallback_individual_validation(courses, log_context)
            
    except asyncio.TimeoutError:
        AgentLogger.log_warning(log_context, "⏰ LLM BATCH VALIDATION TIMEOUT - Using individual validation")
        return [False] * len(courses)
    except Exception as e:
        AgentLogger.log_warning(log_context, f"❌ LLM BATCH VALIDATION ERROR - {str(e)} - Using individual validation")
        return [False] * len(courses)

async def _fallback_individual_validation(courses: List[Dict[str, Any]], log_context: Dict[str, Any]) -> List[bool]:
    """Fallback to individual validation when batch validation fails. Bounded parallel LLM calls."""

    _sem = asyncio.Semaphore(_COURSE_LLM_VALIDATION_MAX_PARALLEL)

    async def _validate_one(course: Dict[str, Any]) -> bool:
        title = course.get('title', '')
        provider = course.get('provider', '')
        url = course.get('url', '')
        async with _sem:
            try:
                return await _llm_validate_course_url(url, title, provider, log_context)
            except Exception as e:
                AgentLogger.log_warning(log_context, f"❌ Individual validation failed for '{title}': {str(e)}")
                return False

    results = await asyncio.gather(*[_validate_one(course) for course in courses])
    return list(results)

async def _llm_validate_course_url(url: str, title: str, provider: str, log_context: Dict[str, Any]) -> bool:
    """Use LLM to validate if a course URL is legitimate and points to a real course."""
    try:
        validation_prompt = f"""
You are a course URL validation expert. Analyze the following course information and determine if the URL is a legitimate, direct link to a real course.

**Course Information:**
- Title: {title}
- Provider: {provider}
- URL: {url}

**Validation Criteria:**
1. Is this a direct link to a specific course (not a search/browse page)?
2. Does the URL structure match the expected pattern for the provider?
3. Does the URL appear to point to a real, existing course?
4. Is the provider correctly identified?
5. Does the URL look legitimate and not like a placeholder or test link?

**Provider-Specific Patterns:**
- Coursera: Should have /learn/, /specializations/, or /professional-certificates/
- Udemy: Should have /course/course-name/
- edX: Should have /course/, /learn/, or /micromasters/
- LinkedIn Learning: Should have /learning/course-name
- YouTube: Should have /watch?v= or /playlist?list=
- Pluralsight: Should have /courses/course-name

**IMPORTANT: You must return ONLY valid JSON. No additional text, explanations, or formatting outside the JSON object.**

**Response Format (return exactly this structure):**
{{
    "is_valid": true,
    "confidence": 0.85,
    "reason": "Valid course URL with proper structure"
}}

Analyze the URL and return a JSON object with is_valid, confidence, and reason."""

        # Issue 5.1: Use structured output for type-safe parsing
        validation_result = None
        try:
            result: SingleValidationResult = await asyncio.wait_for(
                invoke_structured_llm(
                    validation_prompt,
                    SingleValidationResult,
                    task_type=TaskType.COURSE_VALIDATION,
                    preferred_model="gemini-2.5-flash",
                    agent_name="market_and_course_recommender",
                    max_output_tokens=500,
                    temperature=0.1,
                    timeout=8.0,
                    raise_on_fallback=False,
                    skip_cache=True,
                ),
                timeout=8,
            )
            validation_result = result.model_dump()
        except asyncio.TimeoutError:
            AgentLogger.log_warning(log_context, f"Structured validation timed out for '{title}', falling back")
        except Exception as structured_err:
            AgentLogger.log_warning(log_context, f"Structured validation failed for '{title}' ({structured_err}), falling back")
        
        # Fallback to manual parsing if structured output failed
        if not validation_result:
            llm_response = await asyncio.wait_for(
                invoke_llm(
                    prompt=validation_prompt, 
                    task_type="course_validation",
                    agent_name="market_and_course_recommender",
                    preferred_model="gemini-2.5-flash"
                ), timeout=8
            )
            response_text = _to_text(llm_response)
            AgentLogger.log_info(log_context, f"Individual LLM Response for '{title}': {response_text[:200]}...")
            validation_result = _extract_json_from_response(response_text, 500)
        
        if isinstance(validation_result, dict):
            is_valid = validation_result.get('is_valid', False)
            confidence = validation_result.get('confidence', 0.0)
            reason = validation_result.get('reason', 'No reason provided')
            
            if is_valid and confidence > 0.6:
                AgentLogger.log_info(log_context, f"✅ LLM VALIDATED: '{title}' - {reason} (confidence: {confidence})")
                return True
            else:
                AgentLogger.log_warning(log_context, f"❌ LLM REJECTED: '{title}' - {reason} (confidence: {confidence})")
                return False
        else:
            AgentLogger.log_warning(log_context, f"❌ LLM VALIDATION FAILED: '{title}' - Invalid response format: {validation_result}")
            return False
            
    except asyncio.TimeoutError:
        AgentLogger.log_warning(log_context, f"⏰ LLM VALIDATION TIMEOUT: '{title}' - Using fallback validation")
        return _is_course_url_high_quality(url, title, provider)
    except Exception as e:
        AgentLogger.log_warning(log_context, f"❌ LLM VALIDATION ERROR: '{title}' - {str(e)} - Using fallback validation")
        return _is_course_url_high_quality(url, title, provider)


# ---------- Deterministic functions removed - using LLM-only approach ----------

@traceable(name="market_and_course_recommender_agent")
async def market_and_course_recommender_agent(state: Dict[str, Any], tenant_id: str = "default_tenant") -> Dict[str, Any]:
    """
    Enhanced market and course recommender with LLM-only approach.
    """
    # Use centralized logging
    log_context = create_log_context("market_and_course_recommender", tenant_id)
    start_time = log_context["start_time"]
    request_id = log_context["request_id"]
    
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"--- Entering Market and Course Recommender Agent for tenant: {tenant_id} ---")
    
    # Get tenant-scoped memory
    market_memory = await get_market_course_recommender_memory(tenant_id)
    
    # Use selective data passing for optimization
    if "structured_resume" in state:
        # Create optimized state with only the data this agent needs
        optimized_state = create_agent_state(state, "market_and_course_recommender")
        if log.isEnabledFor(logging.INFO):
            log.info(f"📊 Market & Course Recommender using selective data passing - filtered {len(optimized_state.get('structured_resume', {}))} fields")
        
        # Use the optimized state for processing
        structured_resume = optimized_state.get("structured_resume", {})
    else:
        structured_resume = state.get("structured_resume", {})
    
    user_interests = state.get("user_interests", [])
    
    # Check if this is a 2nd call (user_interests from 2nd call should be prioritized)
    is_second_call = state.get("is_second_call", False)
    if is_second_call and user_interests:
        if log.isEnabledFor(logging.INFO):
            log.info(
                f"✅ 2ND CALL: Using user_interests from 2nd call ({len(user_interests) if isinstance(user_interests, list) else 'N/A'} items) as context"
            )
            AgentLogger.log_info(log_context, "2nd call detected - using user_interests from 2nd call as context")
    
    raw_skill_gap_analysis_output = state.get("raw_skill_gap_analysis_output", {})
    
    # Ensure raw_skill_gap_analysis_output is a dictionary
    if isinstance(raw_skill_gap_analysis_output, str):
        try:
            raw_skill_gap_analysis_output = json.loads(raw_skill_gap_analysis_output)
            if log.isEnabledFor(logging.DEBUG):
                log.debug("Successfully parsed raw_skill_gap_analysis_output as JSON")
        except (json.JSONDecodeError, TypeError):
            if log.isEnabledFor(logging.WARNING):
                AgentLogger.log_warning(
                    log_context, "Failed to parse raw_skill_gap_analysis_output as JSON, using empty dict"
                )
            raw_skill_gap_analysis_output = {}
    elif not isinstance(raw_skill_gap_analysis_output, dict):
        if log.isEnabledFor(logging.WARNING):
            AgentLogger.log_warning(
                log_context,
                f"raw_skill_gap_analysis_output is not a dict or string, using empty dict"
            )
        raw_skill_gap_analysis_output = {}

    assessment_results = state.get("assessment_results")
    report = state.get("report")
    session_id = state.get("session_id")
    
    # Only hydrate from session if critical data is missing
    needs_hydration = (not structured_resume or not raw_skill_gap_analysis_output) and session_id
    if needs_hydration:
        try:
            # Run blocking I/O in thread pool to avoid blocking event loop
            session_data = await run_blocking_io(get_chat_session, session_id)

            if not structured_resume:
                structured_resume = session_data.get("structured_resume", {})

            # Only get user_interests from session if not already present in state (2nd call has them in state)
            if not user_interests and not is_second_call:
                interest_data = session_data.get("interest_filler", {})
                user_interests = interest_data.get("user_interests", [])
            elif is_second_call:
                if log.isEnabledFor(logging.DEBUG):
                    log.debug("✅ 2ND CALL: Keeping user_interests from state (2nd call), not overriding from session")

            if not raw_skill_gap_analysis_output:
                skill_gap_data = (
                    session_data.get("career_advisor", {}).get("raw_skill_gap_analysis_output", {})
                    or session_data.get("raw_skill_gap_analysis_output", {})
                )

                # Ensure skill_gap_data is a dictionary
                if isinstance(skill_gap_data, str):
                    try:
                        raw_skill_gap_analysis_output = json.loads(skill_gap_data)
                        if log.isEnabledFor(logging.DEBUG):
                            log.debug("Successfully parsed skill_gap_data as JSON")
                    except (json.JSONDecodeError, TypeError):
                        if log.isEnabledFor(logging.WARNING):
                            AgentLogger.log_warning(
                                log_context,
                                "Failed to parse skill_gap_data from session as JSON, using empty dict"
                            )
                        raw_skill_gap_analysis_output = {}
                elif isinstance(skill_gap_data, dict):
                    raw_skill_gap_analysis_output = skill_gap_data
                else:
                    if log.isEnabledFor(logging.WARNING):
                        AgentLogger.log_warning(
                            log_context,
                            f"skill_gap_data from session is not a dict or string, using empty dict"
                        )
                    raw_skill_gap_analysis_output = {}

            if not assessment_results:
                assessment_results = session_data.get("assessment_evaluator", {}).get("assessment_results")

            if not report:
                report = session_data.get("report_generator", {}).get("report")
        except Exception as e:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"Exception during session hydration: {str(e)}")
            pass

    # Section 8 Issue 1: agent-level cache
    from core.utils import get_cached_response, cache_response
    cache_input = {
        "structured_resume": structured_resume,
        "user_interests": user_interests,
        "raw_skill_gap_analysis_output": raw_skill_gap_analysis_output,
        "assessment_results": assessment_results,
        "report": report,
    }
    cached_result = get_cached_response("market_and_course_recommender", cache_input)
    if cached_result:
        log.info("Cache hit for market_and_course_recommender - returning cached result")
        return cached_result
    
    # Security validation
    if len(str(structured_resume)) > MAX_PROMPT_CHARS:
        AgentLogger.log_warning(log_context, f"Structured resume too large, truncating")
        structured_resume = str(structured_resume)[:MAX_PROMPT_CHARS]
    
    # Aggregate assessment history for stable salary calculation
    uid = state.get("uid")
    assessment_history = []
    salary_modifiers = None
    
    if uid:
        # Parse current assessment if available
        current_assessment = None
        if assessment_results:
            if isinstance(assessment_results, str):
                try:
                    current_assessment = json.loads(assessment_results)
                except (json.JSONDecodeError, TypeError):
                    current_assessment = None
            elif isinstance(assessment_results, dict):
                current_assessment = assessment_results
        
        # Aggregate all assessment history (wrap blocking I/O)
        from core.utils import run_blocking_io
        assessment_history = await run_blocking_io(_aggregate_assessment_history, uid, current_assessment)
        
        # Calculate salary modifiers from aggregated history (for logging/reference only)
        # LLM will determine actual adjustments based on raw assessment data
        salary_modifiers = _calculate_salary_modifiers_from_assessments(assessment_history, current_assessment)
        log.info(f"🔍 MARKET_COURSE_RECOMMENDER: Calculated reference salary modifiers: {salary_modifiers.get('total_adjustment_percent', 0):.1f}% (LLM will determine actual adjustment)")
    
    # Log if assessment results are found
    log.debug("Checking if assessment_results exist")
    if assessment_results:
        log.debug(f"Assessment results found, type: {type(assessment_results)}")
        log.info(f"🔍 MARKET_COURSE_RECOMMENDER: Assessment results found, integrating into analysis")
        
        # Ensure assessment_results is a dictionary before calling .get()
        if isinstance(assessment_results, str):
            log.debug("assessment_results is string, attempting JSON parse")
            try:
                assessment_results = json.loads(assessment_results)
                log.debug("Successfully parsed assessment_results as JSON")
            except (json.JSONDecodeError, TypeError):
                log.debug("Failed to parse assessment_results as JSON")
                AgentLogger.log_warning(log_context, f"Failed to parse assessment_results as JSON, skipping integration")
                assessment_results = None
        
        if isinstance(assessment_results, dict):
            log.debug("assessment_results is dict, getting total_score")
            log.info(f"🔍 MARKET_COURSE_RECOMMENDER: Assessment score: {assessment_results.get('total_score', 'N/A')}")
            
            # Integrate assessment results into skill gap analysis if available
            log.debug("Checking if raw_skill_gap_analysis_output is dict for integration")
            if isinstance(raw_skill_gap_analysis_output, dict):
                log.debug("raw_skill_gap_analysis_output is dict, proceeding with integration")
                # Ensure report is a dictionary before calling .get()
                report_summary = ""
                if report:
                    log.debug(f"Report exists, type: {type(report)}")
                    if isinstance(report, str):
                        log.debug("report is string, attempting JSON parse")
                        try:
                            report = json.loads(report)
                            log.debug("Successfully parsed report as JSON")
                        except (json.JSONDecodeError, TypeError):
                            log.debug("Failed to parse report as JSON")
                            AgentLogger.log_warning(log_context, f"Failed to parse report as JSON, using empty string")
                            report = None
                    
                    if isinstance(report, dict):
                        log.debug("report is dict, getting summary")
                        report_summary = report.get("summary", "")
                
                log.debug("Creating assessment_performance integration")
                # Store comprehensive assessment data for LLM-driven salary adjustment
                # Include all assessment details so LLM can make intelligent decisions
                raw_skill_gap_analysis_output["assessment_performance"] = {
                    "total_score": assessment_results.get("total_score", 0),
                    "max_score": assessment_results.get("max_score", 100),
                    "section_scores": assessment_results.get("section_scores", {}),
                    "assessment_topic": assessment_results.get("assessment_topic", ""),
                    "assessment_type": assessment_results.get("assessment_type", ""),
                    "report": report_summary,
                    "assessment_history_count": len(assessment_history),
                    "assessment_history": assessment_history,  # Include full history for context
                    "salary_modifiers": salary_modifiers  # Keep for logging/reference, but LLM determines adjustment
                }
                log.debug("Successfully integrated assessment_performance")
            else:
                log.debug(f"raw_skill_gap_analysis_output is not dict, type: {type(raw_skill_gap_analysis_output)}")
    else:
        log.debug("No assessment_results found")
        # Explicitly mark that no assessments have been taken
        if isinstance(raw_skill_gap_analysis_output, dict):
            if "assessment_performance" not in raw_skill_gap_analysis_output:
                raw_skill_gap_analysis_output["assessment_performance"] = {}
            # CRITICAL: Set assessment_history_count to 0 to indicate NO assessments taken
            # This distinguishes "no assessments" from "0% score from assessment"
            raw_skill_gap_analysis_output["assessment_performance"]["assessment_history_count"] = 0
            raw_skill_gap_analysis_output["assessment_performance"]["has_assessments"] = False
            if salary_modifiers:
                raw_skill_gap_analysis_output["assessment_performance"]["salary_modifiers"] = salary_modifiers
            log.debug("Set assessment_history_count=0 to indicate no assessments taken")
    
    # Derive performance-aware context from latest assessment
    log.debug("Deriving performance-aware assessment context")
    last_topic = state.get("assessment_topic")
    performance_level: Optional[str] = None
    score: Optional[float] = None
    weak_sections: Dict[str, Any] = {}
    strong_sections: Dict[str, Any] = {}
    
    if isinstance(assessment_results, dict):
        score = assessment_results.get("total_score")
        section_scores = assessment_results.get("section_scores", {}) or {}
        weak_sections = {k: v for k, v in section_scores.items() if isinstance(v, (int, float)) and v < 50}
        strong_sections = {k: v for k, v in section_scores.items() if isinstance(v, (int, float)) and v >= 80}
        
        if isinstance(score, (int, float)):
            if score >= 90:
                performance_level = "Outstanding"
            elif score >= 80:
                performance_level = "Excellent"
            elif score >= 70:
                performance_level = "Good"
            elif score >= 50:
                performance_level = "Fair"
            else:
                performance_level = "Needs Improvement"
        
        if not last_topic:
            last_topic = assessment_results.get("assessment_topic")
    
    performance_context = {
        "score": score,
        "performance_level": performance_level,
        "weak_sections": weak_sections,
        "strong_sections": strong_sections,
        "last_topic": last_topic,
    }

    # Normalize the gap analysis payload so downstream prompt builders can rely on
    # consistent keys even when upstream agents omitted them.
    if isinstance(raw_skill_gap_analysis_output, dict):
        if assessment_results and isinstance(assessment_results, dict):
            raw_skill_gap_analysis_output.setdefault("assessment_performance", assessment_results)
        raw_skill_gap_analysis_output.setdefault("performance_context", performance_context)
    
    # Initialize variables to avoid scope issues
    log.debug("Initializing variables")
    final_valid_courses = []
    valid_courses = []
    recommendations = {}
    market_insights = {}  # Initialize market_insights outside try block
    # Ensure salary_modifiers is initialized even if uid is None
    if 'salary_modifiers' not in locals():
        salary_modifiers = None
    log.debug("Variables initialized successfully")
    
    # Granular timing breakdown for performance monitoring
    timing_breakdown = {
        "prompt_generation": 0.0,
        "market_insights_llm": 0.0,
        "course_validation": 0.0,
        "result_processing": 0.0,
        "total": 0.0
    }
    
    log.debug("Starting main processing section")
    try:
        # Step 1: Generate market insights using original prompt
        # Note: We pass salary_modifiers for backward compatibility/logging, but LLM determines actual adjustments
        log.debug("Generating market insights prompt")
        # Extract salary modifiers from skill gap analysis if available (for reference only)
        modifiers_for_prompt = None
        if isinstance(raw_skill_gap_analysis_output, dict):
            assessment_perf = raw_skill_gap_analysis_output.get("assessment_performance", {})
            if isinstance(assessment_perf, dict):
                # Pass modifiers for logging/reference, but LLM will use raw assessment data to determine adjustments
                modifiers_for_prompt = assessment_perf.get("salary_modifiers")
        
        # If not found in skill gap, use the calculated modifiers (for reference only)
        if not modifiers_for_prompt and salary_modifiers:
            modifiers_for_prompt = salary_modifiers
        
        # Detect if this is a rerun (has assessment results or previous outputs)
        is_rerun = bool(
            assessment_results or 
            (salary_modifiers and salary_modifiers.get('assessment_count', 0) > 0) or
            report or
            state.get("rerun_mode", False)
        )
        
        # NEW: Await async prompt generation with Gemini summary
        prompt_start = time.time()
        uid = state.get("uid")
        market_prompt = await generate_market_and_course_recommendation_prompt(
            structured_resume, user_interests, raw_skill_gap_analysis_output, modifiers_for_prompt, is_rerun=is_rerun, uid=uid
        )
        timing_breakdown["prompt_generation"] = time.time() - prompt_start
        # Optionally append high-level performance context for market analysis and salary guidance
        if performance_level:
            market_prompt += f"""

Assessment Performance Context: {json.dumps(performance_context, default=str)}

Assessment Performance Insights:
- Topic Assessed: {performance_context.get('last_topic')}
- Score: {performance_context.get('score')}
- Performance Level: {performance_level}
- Weak Sections: {performance_context.get('weak_sections')}
- Strong Sections: {performance_context.get('strong_sections')}

Use these signals to slightly tune any salary and market commentary, but keep final salary ranges consistent with the role, location, and experience.
"""
        log.debug("Generated market insights prompt successfully")
        
        # Validate market prompt size - use smarter truncation that preserves JSON schema
        max_prompt_size = min(MAX_PROMPT_CHARS, 15000)  # Increased from 10000 to prevent truncation of important data
        if len(market_prompt) > max_prompt_size:
            AgentLogger.log_warning(log_context, f"Market prompt too large, truncating from {len(market_prompt)} to {max_prompt_size}")
            # Smart truncation: preserve the JSON schema at the end which is critical for parsing
            json_schema_marker = "**JSON:**"
            json_schema_start = market_prompt.rfind(json_schema_marker)
            if json_schema_start > 0:
                # Keep instructions + JSON schema, truncate middle content
                instructions = market_prompt[:json_schema_start + len(json_schema_marker)]
                json_schema = market_prompt[json_schema_start:]
                # Calculate available space for instructions
                available_instruction_space = max_prompt_size - len(json_schema)
                if available_instruction_space > 1000:  # Only truncate if we have reasonable space
                    truncated_instructions = instructions[:available_instruction_space]
                    market_prompt = truncated_instructions + json_schema
                else:
                    # Not enough space, use simple truncation but preserve last 2000 chars (likely contains schema)
                    market_prompt = market_prompt[:max_prompt_size - 2000] + market_prompt[-2000:]
            else:
                # Fallback: simple truncation if schema marker not found
                market_prompt = market_prompt[:max_prompt_size]
        
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Calling LLM for market insights")
        if log.isEnabledFor(logging.INFO):
            log.info("📊 Calling LLM for market insights and course recommendations in parallel...")
        try:
            market_llm_start = time.time()

            # Run market analysis and course search in parallel (same inputs, independent outputs)
            async def _market_llm_task():
                return await asyncio.wait_for(
                    invoke_llm(
                        prompt=market_prompt,
                        task_type="market_analysis",
                        agent_name="market_and_course_recommender",
                        preferred_model="gemini-2.5-flash",
                        max_output_tokens=4000,
                        response_mime_type="application/json",
                        skip_cache=True
                    ), timeout=30
                )

            async def _course_task():
                return await asyncio.wait_for(
                    _generate_course_recommendations_from_knowledge_base(
                        structured_resume, user_interests, raw_skill_gap_analysis_output, log_context
                    ), timeout=15
                )

            market_result, course_result = await asyncio.gather(
                _market_llm_task(), _course_task(), return_exceptions=True
            )

            timing_breakdown["market_insights_llm"] = time.time() - market_llm_start
            if log.isEnabledFor(logging.INFO):
                log.info(f"⏱️ MARKET_AND_COURSE_RECOMMENDER: Parallel market+course took {timing_breakdown['market_insights_llm']*1000:.1f}ms")

            # Handle market result
            if isinstance(market_result, Exception):
                if isinstance(market_result, asyncio.TimeoutError):
                    raise market_result
                log.warning(f"Market analysis failed: {market_result}")
                market_llm_response = None
            else:
                market_llm_response = market_result

            # Handle course result
            if isinstance(course_result, Exception):
                log.warning(f"Course recommendations failed: {course_result}")
                course_recommendations_data = {"course_recommendations": []}
            else:
                course_recommendations_data = course_result

            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"Got market LLM response, type: {type(market_llm_response)}")
                if market_llm_response:
                    log.debug(f"Raw LLM response content: {market_llm_response}")
                    log.debug(f"Raw LLM response length: {len(market_llm_response) if market_llm_response else 0}")

            # Parse LLM response with improved extraction for full JSON structure
            log.debug("Parsing LLM response")
            
            # Helper function to extract full JSON structure from markdown-wrapped responses
            def _extract_full_json_structure(text: str) -> Dict[str, Any]:
                """Extract the complete JSON structure, handling markdown text before/after JSON."""
                if not text:
                    return {}
                # LLM fallback (quota/timeout) returns non-JSON - use minimal skeleton so flow continues
                if is_fallback_response(text):
                    log.warning("Market insights LLM returned fallback response (quota/timeout) - using empty structure")
                    return {
                        "market_insights": [], "salary_trends": {}, "career_paths": [],
                        "skill_demand_analysis": {}, "course_recommendations": []
                    }
                # Normalize camelCase keys to snake_case for matching
                def _normalize_keys(d: Dict) -> Dict:
                    key_map = {"marketinsights": "market_insights", "salarytrends": "salary_trends",
                               "careerpaths": "career_paths", "skilldemandanalysis": "skill_demand_analysis",
                               "courserecommendations": "course_recommendations"}
                    out = {}
                    for k, v in d.items():
                        nk = key_map.get(k.lower().replace("_", ""), k)
                        out[nk] = v
                    return out
                # Expected keys that must be present in a valid market insights response
                expected_keys = ['market_insights', 'salary_trends', 'career_paths', 'skill_demand_analysis', 'course_recommendations']
                
                # Step 0: Clean text - remove markdown headers and extra whitespace that might interfere
                # But preserve JSON code blocks
                cleaned_text = text
                
                # Step 1: Try to extract from markdown code blocks
                # Use content up to the *last* ``` so embedded ``` in strings don't truncate (same as core.utils)
                json_start = re.search(r"```json\s*", text, re.IGNORECASE)
                if json_start:
                    start_pos = json_start.end()
                    last_fence = text.rfind("```")
                    if last_fence > start_pos:
                        json_text = text[start_pos:last_fence].strip()
                    else:
                        # Unclosed block: take from ```json to end (handles truncated responses)
                        json_text = text[start_pos:].strip()
                    if json_text:
                        try:
                            parsed = json.loads(json_text)
                            if isinstance(parsed, dict):
                                norm = _normalize_keys(parsed)
                                if any(key in norm for key in expected_keys):
                                    log.debug(f"Successfully parsed JSON from markdown code block with keys: {list(norm.keys())}")
                                    return norm
                        except json.JSONDecodeError as e:
                            log.debug(f"Failed to parse JSON from markdown block: {e}")
                            try:
                                json_text_fixed = re.sub(r',(\s*[}\]])', r'\1', json_text)
                                parsed = json.loads(json_text_fixed)
                                if isinstance(parsed, dict):
                                    norm = _normalize_keys(parsed)
                                    if any(key in norm for key in expected_keys):
                                        log.debug(f"Successfully parsed JSON after fixing trailing commas")
                                        return norm
                            except Exception:
                                pass
                
                # Step 1b: Try to find JSON after markdown headers (common LLM pattern)
                # Look for JSON that starts after markdown headers like "## Market Insights"
                json_after_markdown = re.search(r'(?:^|\n)(?:##[^\n]*\n)*\s*(\{[\s\S]*\})', text, re.MULTILINE)
                if json_after_markdown:
                    json_text = json_after_markdown.group(1).strip()
                    try:
                        parsed = json.loads(json_text)
                        if isinstance(parsed, dict):
                            norm = _normalize_keys(parsed)
                            if any(key in norm for key in expected_keys):
                                log.debug(f"Successfully parsed JSON after markdown headers with keys: {list(norm.keys())}")
                                return norm
                    except json.JSONDecodeError:
                        pass
                
                # Step 2: Try direct parse (only accept if it has expected keys)
                try:
                    parsed = json.loads(text.strip())
                    if isinstance(parsed, dict):
                        norm = _normalize_keys(parsed)
                        if any(key in norm for key in expected_keys):
                            log.debug(f"Successfully parsed JSON directly with keys: {list(norm.keys())}")
                            return norm
                except json.JSONDecodeError:
                    pass
                
                # Step 3: Find the OUTERMOST balanced JSON object (not just the first one)
                # Look for the largest balanced object that contains the expected keys
                start_positions = []
                for i, char in enumerate(text):
                    if char == '{':
                        start_positions.append(i)
                
                # Try each starting position, but prioritize ones that might contain the full structure
                for start in reversed(start_positions):  # Start from the end to find outermost
                    depth = 0
                    in_str = False
                    esc = False
                    for i in range(start, len(text)):
                        ch = text[i]
                        if in_str:
                            if esc:
                                esc = False
                            elif ch == '\\':
                                esc = True
                            elif ch == '"':
                                in_str = False
                        else:
                            if ch == '"':
                                in_str = True
                            elif ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    candidate = text[start:i+1]
                                    try:
                                        parsed = json.loads(candidate)
                                        if isinstance(parsed, dict):
                                            norm = _normalize_keys(parsed)
                                            if any(key in norm for key in expected_keys):
                                                log.debug(f"Successfully parsed JSON from balanced scan with keys: {list(norm.keys())}")
                                                return norm
                                    except json.JSONDecodeError:
                                        pass
                                    break
                
                # Step 4: Fallback to existing _extract_json_from_response with increased max length
                result = _extract_json_from_response(text, max_response_length=50000)
                if result and isinstance(result, dict):
                    norm = _normalize_keys(result)
                    if any(key in norm for key in expected_keys):
                        log.debug(f"Successfully parsed JSON from fallback function with keys: {list(norm.keys())}")
                        return norm
                
                # Step 5: Last resort - try to extract partial JSON if response was truncated
                # Look for any JSON object that has at least one expected key (even if incomplete)
                log.warning(f"All JSON parsing methods failed. Text length: {len(text)}, preview: {text[:500]}")
                return {}
            
            # Convert LLM response to text first (handles various response formats)
            market_llm_response_text = _to_text(market_llm_response)
            log.debug(f"LLM response text length: {len(market_llm_response_text) if market_llm_response_text else 0}")
            if market_llm_response_text:
                log.debug(f"LLM response preview (first 500 chars): {market_llm_response_text[:500]}")
            
            market_insights = _extract_full_json_structure(market_llm_response_text)
            log.debug(f"Parsed market_insights, type: {type(market_insights)}")
            log.debug(f"market_insights keys: {list(market_insights.keys()) if isinstance(market_insights, dict) else 'N/A'}")
            
            if not isinstance(market_insights, dict):
                log.debug("market_insights is not dict, creating empty dict")
                processing_time = _calculate_processing_time(start_time)
                AgentLogger.log_error(log_context, f"market_insights is not a dict (type: {type(market_insights)}), resetting to empty dict", processing_time)
                # ✅ FIX: Log raw response for debugging truncation issues
                if market_llm_response_text:
                    response_preview = market_llm_response_text[:1000] if len(market_llm_response_text) > 1000 else market_llm_response_text
                    AgentLogger.log_warning(log_context, f"Raw LLM response preview (first 1000 chars): {response_preview}")
                    AgentLogger.log_warning(log_context, f"Raw LLM response length: {len(market_llm_response_text)} chars")
                market_insights = {}
            else:
                log.debug(f"market_insights keys: {list(market_insights.keys())}")
                # Check if market_insights dict is empty or missing expected fields
                expected_keys = ['market_insights', 'salary_trends', 'career_paths', 'skill_demand_analysis']
                if not market_insights or market_insights == {}:
                    processing_time = _calculate_processing_time(start_time)
                    AgentLogger.log_error(log_context, "⚠️ CRITICAL: market_insights dict is empty after parsing. All market fields will be empty.", processing_time)
                    # ✅ FIX: Log raw response for debugging truncation issues
                    if market_llm_response_text:
                        response_preview = market_llm_response_text[:2000] if len(market_llm_response_text) > 2000 else market_llm_response_text
                        AgentLogger.log_warning(log_context, f"Raw LLM response when empty (first 2000 chars): {response_preview}")
                        AgentLogger.log_warning(log_context, f"Raw LLM response length: {len(market_llm_response_text)} chars")
                        # Log last 500 chars to see if JSON was cut off
                        if len(market_llm_response_text) > 500:
                            AgentLogger.log_warning(log_context, f"Response ending (last 500 chars): {market_llm_response_text[-500:]}")
                elif not any(key in market_insights for key in expected_keys):
                    processing_time = _calculate_processing_time(start_time)
                    AgentLogger.log_warning(log_context, f"⚠️ WARNING: market_insights dict missing expected fields. Keys found: {list(market_insights.keys())}", processing_time)
                    # ✅ FIX: Log raw response and response length for debugging truncation issues
                    if market_llm_response_text:
                        response_length = len(market_llm_response_text)
                        response_preview = market_llm_response_text[:2000] if response_length > 2000 else market_llm_response_text
                        AgentLogger.log_warning(log_context, f"Raw LLM response length: {response_length} chars. Preview (first 2000 chars): {response_preview}")
                        # Check if response might be truncated
                        if response_length > 3000:  # If response is long but missing keys, might be truncated
                            AgentLogger.log_warning(log_context, f"⚠️ Response is {response_length} chars but missing expected keys - possible truncation issue")
                        # Log last 500 chars to see if JSON was cut off
                        if response_length > 500:
                            AgentLogger.log_warning(log_context, f"Response ending (last 500 chars): {market_llm_response_text[-500:]}")
            
            log.debug("Market insights processing completed successfully")
            
        except asyncio.TimeoutError as _timeout_err:
            log.debug(f"Market analysis timed out: {str(_timeout_err)}")
            processing_time = _calculate_processing_time(start_time)
            AgentLogger.log_error(log_context, "Market analysis timed out after 40 seconds", processing_time)
            await market_memory.record_attempt(
                'market_analysis', 'llm_timeout', False, 0.0, processing_time
            )
            return _create_error_response(
                "Market analysis failed: Model timeout after 40 seconds",
                processing_time
            )
        except Exception as e:
            log.debug(f"Exception during market analysis: {str(e)}")
            log.debug(f"Exception type: {type(e)}")
            import traceback
            traceback_str = traceback.format_exc()
            log.debug(f"Market analysis traceback: {traceback_str}")
            processing_time = _calculate_processing_time(start_time)
            AgentLogger.log_error(log_context, f"❌ EXCEPTION during market analysis LLM call: {str(e)}. Type: {type(e).__name__}", processing_time)
            AgentLogger.log_error(log_context, f"Traceback: {traceback_str[:500]}", processing_time)  # First 500 chars of traceback
            # Continue with empty market_insights instead of failing completely
            market_insights = {}
            AgentLogger.log_warning(log_context, "⚠️ Continuing with empty market_insights due to exception - all market fields will be empty")
        
        # Step 2: course_recommendations_data already obtained from parallel gather above
        # Step 3: Courses and materials from knowledge base are already validated
        log.debug("Courses and materials from knowledge base are pre-validated")
        # Ensure course_recommendations_data is a dictionary
        if isinstance(course_recommendations_data, str):
            log.debug("course_recommendations_data is string, attempting JSON parse")
            try:
                course_recommendations_data = json.loads(course_recommendations_data)
                log.debug("Successfully parsed course_recommendations_data as JSON")
            except (json.JSONDecodeError, TypeError):
                log.debug("Failed to parse course_recommendations_data as JSON")
                AgentLogger.log_warning(log_context, f"Failed to parse course_recommendations_data as JSON, using empty dict")
                course_recommendations_data = {}
        elif not isinstance(course_recommendations_data, dict):
            log.debug(f"course_recommendations_data is not dict or string, type: {type(course_recommendations_data)}")
            AgentLogger.log_warning(log_context, f"course_recommendations_data is not a dict or string, using empty dict")
            course_recommendations_data = {}
        
        raw_courses = course_recommendations_data.get('course_recommendations', [])
        # All materials (courses, books, papers, videos, tutorials) are now in course_recommendations
        log.debug(f"Got {len(raw_courses) if isinstance(raw_courses, list) else 0} total recommendations from knowledge base")
        
        # Knowledge base recommendations are already validated, so use them directly
        final_valid_courses = raw_courses if isinstance(raw_courses, list) else []
        AgentLogger.log_info(log_context, f"✅ Using {len(final_valid_courses)} validated recommendations from knowledge base")

        # No search supplement needed since we're not using URLs
        
        # Step 5: Combine market insights with validated course recommendations
        log.debug("Building recommendations dictionary")
        
        # Map courses from knowledge base to output format
        # Knowledge base courses use 'title' and 'provider', but output needs 'course' and 'platform'
        final_course_recommendations = _map_courses_for_output(final_valid_courses) if final_valid_courses else []
        AgentLogger.log_info(log_context, f"✅ Mapped {len(final_course_recommendations)} courses to output format")
        
        # ✅ FIX: Extract from parsed market_insights dict structure
        # The parsed dict should have keys: market_insights, salary_trends, career_paths, skill_demand_analysis
        parsed_market_insights = market_insights.get('market_insights', []) if isinstance(market_insights, dict) else []
        parsed_salary_trends = market_insights.get('salary_trends', {}) if isinstance(market_insights, dict) else {}
        parsed_career_paths = market_insights.get('career_paths', []) if isinstance(market_insights, dict) else []
        parsed_skill_demand = market_insights.get('skill_demand_analysis', {}) if isinstance(market_insights, dict) else {}
        
        # ✅ FIX: Use fallback data from raw_skill_gap_analysis_output if parsing failed
        # This ensures we don't lose data from previous runs or cached results in re-run scenarios
        fallback_market_insights = raw_skill_gap_analysis_output.get('market_insights', []) if isinstance(raw_skill_gap_analysis_output, dict) else []
        fallback_salary_trends = raw_skill_gap_analysis_output.get('salary_trends', {}) if isinstance(raw_skill_gap_analysis_output, dict) else {}
        fallback_career_paths = raw_skill_gap_analysis_output.get('career_paths', []) if isinstance(raw_skill_gap_analysis_output, dict) else []
        fallback_skill_demand = raw_skill_gap_analysis_output.get('skill_demand_analysis', {}) if isinstance(raw_skill_gap_analysis_output, dict) else {}
        
        # Use parsed data if available and non-empty, otherwise use fallback
        # This handles re-run scenarios where previous data exists but new parsing might fail
        final_market_insights = parsed_market_insights if (parsed_market_insights and len(parsed_market_insights) > 0) else fallback_market_insights
        final_salary_trends = parsed_salary_trends if (parsed_salary_trends and isinstance(parsed_salary_trends, dict) and len(parsed_salary_trends) > 0) else fallback_salary_trends
        final_career_paths = parsed_career_paths if (parsed_career_paths and len(parsed_career_paths) > 0) else fallback_career_paths
        final_skill_demand = parsed_skill_demand if (parsed_skill_demand and isinstance(parsed_skill_demand, dict) and len(parsed_skill_demand) > 0) else fallback_skill_demand
        
        # When both parsed and fallback salary_trends are empty, synthesize so users always get something
        # Prefer assessment-based context when available; otherwise use a minimal placeholder
        if not final_salary_trends:
            if salary_modifiers:
                mod_reasons = salary_modifiers.get("reasons", [])
                adj_pct = salary_modifiers.get("total_adjustment_percent", 0)
                final_salary_trends = {
                    "current_level": "Based on resume and experience" + (f" (+{adj_pct:.1f}% assessment adjustment)" if adj_pct else ""),
                    "current_level_rationale": mod_reasons if mod_reasons else ["No assessment history. Salary based on resume and experience only."],
                    "assessment_modifiers": salary_modifiers,
                    "note": "Market insights were unavailable. Complete salary trends will appear when market analysis succeeds."
                }
                AgentLogger.log_warning(log_context, "⚠️ Synthesized salary_trends from assessment modifiers (market insights LLM parsing failed)")
            else:
                final_salary_trends = {
                    "current_level": "Based on resume and experience",
                    "current_level_rationale": ["Market analysis temporarily unavailable. Salary insights will appear when the market analysis completes successfully."],
                    "note": "Market insights were unavailable. Complete salary trends will appear when market analysis succeeds."
                }
                AgentLogger.log_warning(log_context, "⚠️ Synthesized minimal salary_trends (market insights LLM parsing failed, no assessment data)")
        
        # Log if we're using fallback data (important for debugging re-run scenarios)
        if not parsed_market_insights and fallback_market_insights:
            AgentLogger.log_warning(log_context, f"⚠️ Using fallback market_insights from raw_skill_gap_analysis_output ({len(fallback_market_insights)} insights) - parsing may have failed or this is a re-run")
        if not parsed_salary_trends and fallback_salary_trends:
            AgentLogger.log_warning(log_context, f"⚠️ Using fallback salary_trends from raw_skill_gap_analysis_output - parsing may have failed or this is a re-run")
        
        recommendations = {
            'market_insights': final_market_insights,
            'course_recommendations': final_course_recommendations,
            'salary_trends': final_salary_trends,
            'career_paths': final_career_paths,
            'skill_demand_analysis': final_skill_demand,
            # Expose performance context so downstream consumers can surface reasoning
            'performance_context': performance_context,
        }
        log.debug(f"Built recommendations, keys: {list(recommendations.keys())}")
        
        # Diagnostic check: Warn if critical fields are empty
        empty_fields = []
        if not recommendations.get('market_insights'):
            empty_fields.append('market_insights')
        if not recommendations.get('salary_trends'):
            empty_fields.append('salary_trends')
        if not recommendations.get('career_paths'):
            empty_fields.append('career_paths')
        if not recommendations.get('skill_demand_analysis'):
            empty_fields.append('skill_demand_analysis')
        if not recommendations.get('course_recommendations'):
            empty_fields.append('course_recommendations')
        
        if empty_fields:
            AgentLogger.log_warning(log_context, f"⚠️ WARNING: The following fields are empty: {', '.join(empty_fields)}. This may indicate LLM parsing failure or validation issues.")
            if 'market_insights' in empty_fields or 'salary_trends' in empty_fields or 'career_paths' in empty_fields or 'skill_demand_analysis' in empty_fields:
                AgentLogger.log_warning(log_context, f"⚠️ Market insights dict keys: {list(market_insights.keys()) if isinstance(market_insights, dict) else 'Not a dict'}")
            if 'course_recommendations' in empty_fields:
                AgentLogger.log_warning(log_context, f"⚠️ Course recommendations: {len(final_valid_courses)} courses after validation")
        
        processing_time = _calculate_processing_time(start_time)
        timing_breakdown["total"] = processing_time
        
        # Log timing breakdown
        log.info(
            f"⏱️ MARKET_AND_COURSE_RECOMMENDER timing breakdown: "
            f"prompt_gen={timing_breakdown['prompt_generation']*1000:.1f}ms, "
            f"market_llm={timing_breakdown['market_insights_llm']*1000:.1f}ms, "
            f"course_validation={timing_breakdown['course_validation']*1000:.1f}ms, "
            f"total={timing_breakdown['total']*1000:.1f}ms"
        )
        
        # Record successful analysis
        await market_memory.record_attempt(
            'market_analysis', 'llm', True, 0.8, processing_time
        )
        
        # Log success with enhanced course flow details
        log_agent_completion(log_context, {
            "success": True,
            "market_insights_count": len(recommendations.get('market_insights', [])),
            "course_recommendations_count": len(final_valid_courses),
            "enhanced_course_flow": True,
            "validated_courses": len(valid_courses),
            "final_valid_courses": len(final_valid_courses)
        }, "llm", processing_time)
        
        # Filter out career advice data and update with market insights only
        filtered_skill_gap = _filter_career_advice_data(raw_skill_gap_analysis_output)
        
        # Add market insights and other data (excluding course_recommendations for now)
        log.debug("Building market_data dictionary")
        # ✅ FIX: Only update with non-empty data to preserve existing data from previous runs
        market_data = {}
        
        # Only add market_insights if we have new data
        if recommendations.get('market_insights'):
            market_data['market_insights'] = recommendations.get('market_insights', [])
        elif filtered_skill_gap.get('market_insights'):
            # Preserve existing data if new parsing failed
            market_data['market_insights'] = filtered_skill_gap.get('market_insights', [])
            AgentLogger.log_warning(log_context, f"⚠️ Preserving existing market_insights ({len(market_data['market_insights'])} insights) as new parsing returned empty")
        
        # Only add salary_trends if we have new data
        if recommendations.get('salary_trends'):
            market_data['salary_trends'] = recommendations.get('salary_trends', {})
        elif filtered_skill_gap.get('salary_trends'):
            # Preserve existing data if new parsing failed
            market_data['salary_trends'] = filtered_skill_gap.get('salary_trends', {})
            AgentLogger.log_warning(log_context, f"⚠️ Preserving existing salary_trends as new parsing returned empty")
        
        # Only add career_paths if we have new data
        if recommendations.get('career_paths'):
            market_data['career_paths'] = recommendations.get('career_paths', [])
        elif filtered_skill_gap.get('career_paths'):
            # Preserve existing data if new parsing failed
            market_data['career_paths'] = filtered_skill_gap.get('career_paths', [])
        
        # Only add skill_demand_analysis if we have new data
        if recommendations.get('skill_demand_analysis'):
            market_data['skill_demand_analysis'] = recommendations.get('skill_demand_analysis', {})
        elif filtered_skill_gap.get('skill_demand_analysis'):
            # Preserve existing data if new parsing failed
            market_data['skill_demand_analysis'] = filtered_skill_gap.get('skill_demand_analysis', {})
        
        log.debug(f"Built market_data, keys: {list(market_data.keys())}")
        log.debug("Updating filtered_skill_gap with market_data")
        filtered_skill_gap.update(market_data)
        log.debug("Updated filtered_skill_gap successfully")
        
        # Format course recommendations for output (no URL validation)
        # All materials (courses, books, papers, videos, tutorials) are now in course_recommendations
        log.debug("Formatting course recommendations for output")
        formatted_courses = _map_courses_for_output(recommendations.get('course_recommendations', []))
        log.debug(f"Got {len(formatted_courses) if isinstance(formatted_courses, list) else 0} total recommendations")
        
        # No URL validation - use formatted courses directly
        final_valid_courses = formatted_courses
        log.debug(f"Final formatting complete, {len(final_valid_courses)} recommendations ready")
        
        # Add course recommendations to the filtered data (includes all materials)
        filtered_skill_gap['course_recommendations'] = final_valid_courses
        
        # Safety check: ensure final_valid_courses is defined
        log.debug("Performing safety check for final_valid_courses")
        if 'final_valid_courses' not in locals():
            log.debug("final_valid_courses not in locals, creating empty list")
            final_valid_courses = []
        
        # Calculate final processing time
        processing_time = _calculate_processing_time(start_time)
        
        # Update state
        log.debug("Updating state with results")
        state["raw_skill_gap_analysis_output"] = filtered_skill_gap
        state["analysis_status"] = "success"
        state["analysis_method"] = "llm"
        state["confidence_score"] = 0.8
        state["processing_time"] = processing_time
        state["request_id"] = request_id
        log.debug("State updated successfully")
        
        # Store market and course recommender data in chat_sessions per-UID doc - MOVED TO BACKGROUND (non-blocking)
        uid = state.get("uid")
        if uid and (recommendations.get("market_insights") or final_valid_courses):
            # Fire-and-forget: Don't block response on storage operations
            async def _store_data_background():
                try:
                    storage_start = time.time()
                    log.info(f"🔄 MARKET_AND_COURSE_RECOMMENDER: Starting background storage for UID={uid}")
                    
                    # In evaluation flow (assessment_results present), REPLACE instead of merge
                    is_evaluation_flow = bool(state.get("assessment_results"))
                    existing = await run_blocking_io(get_gap_doc, uid) or {}
                    
                    if is_evaluation_flow:
                        # Evaluation flow: MERGE instead of REPLACE to preserve assessment history
                        # This ensures salary modifiers are calculated from aggregated history
                        merged = {**existing}
                        merged["market_and_course_recommender"] = {
                            "raw_skill_gap_analysis_output": filtered_skill_gap,
                            "market_insights": recommendations.get("market_insights", []),
                            "course_recommendations": final_valid_courses,
                            "analysis_status": "success",
                            "analysis_method": "enhanced_llm",
                            "confidence_score": 0.8,
                            "processing_time": processing_time,
                            "request_id": request_id,
                            "enhanced_course_flow": True,
                            "validated_courses_count": len(valid_courses),
                            "evaluation_flow": True,
                            "assessment_history_aggregated": True,
                            "salary_modifiers_applied": bool(salary_modifiers)
                        }
                        # Preserve career_advisor if present
                        if "skill_and_career_advisor" in existing:
                            merged["skill_and_career_advisor"] = existing["skill_and_career_advisor"]
                        log.info(f"✅ Evaluation flow: Merged market_and_course_recommender data with assessment history for UID={uid}")
                    else:
                        # Normal flow: Merge with existing
                        merged = {**existing}
                        merged["market_and_course_recommender"] = {
                            "raw_skill_gap_analysis_output": filtered_skill_gap,
                            "market_insights": recommendations.get("market_insights", []),
                            "course_recommendations": final_valid_courses,
                            "analysis_status": "success",
                            "analysis_method": "enhanced_llm",
                            "confidence_score": 0.8,
                            "processing_time": processing_time,
                            "request_id": request_id,
                            "enhanced_course_flow": True,
                            "validated_courses_count": len(valid_courses)
                        }
                    
                    await run_blocking_io(
                        upsert_gap_doc,
                        uid,
                        merged,
                        metadata={
                            "agent": "market_and_course_recommender",
                            "uid": uid,
                            "status": "market_and_course_recommender_complete",
                            "evaluation_flow": is_evaluation_flow,
                            "method": "enhanced_llm"
                        }
                    )
                    storage_time = time.time() - storage_start
                    log.info(f"✅ MARKET_AND_COURSE_RECOMMENDER: Background storage completed in {storage_time*1000:.1f}ms for UID={uid}")
                except Exception as e:
                    log.error(f"❌ MARKET_AND_COURSE_RECOMMENDER: Background storage failed for UID={uid}: {e}", exc_info=True)
            
            # Fire-and-forget: Start background task without awaiting
            asyncio.create_task(_store_data_background())
            log.info(f"🚀 MARKET_AND_COURSE_RECOMMENDER: Started background storage task for UID={uid} (non-blocking)")
        
        # Return both formats:
        # 1. Strict JSON format for callbacks
        # 2. Original fields for internal pipeline use
        log.debug("Building final return dictionary")
        return_dict = {
            # Strict JSON format for callbacks
            "status": "completed",
            "node": "market_and_course_recommender",
            "output": {
                "raw_skill_gap_analysis_output": filtered_skill_gap,
                "market_insights": recommendations.get("market_insights", []),
                "salary_trends": recommendations.get("salary_trends", {}),
                "career_paths": recommendations.get("career_paths", []),
                "skill_demand_analysis": recommendations.get("skill_demand_analysis", {}),
                "course_recommendations": final_valid_courses
            },
            
            # Original fields for internal pipeline use
            "recommendations": recommendations,
            "raw_skill_gap_analysis_output": filtered_skill_gap,
            "analysis_status": "success",
            "analysis_method": "enhanced_llm",
            "confidence_score": 0.8,
            "processing_time": processing_time,
            "request_id": request_id,
            "enhanced_course_flow": True,
            "validated_courses_count": len(valid_courses)
        }
        log.debug("Built return dictionary successfully")
        log.debug("Returning successful result")
        cache_response("market_and_course_recommender", cache_input, return_dict)
        return return_dict
        
    except Exception as e:
        log.debug(f"Exception caught in market_and_course_recommender: {str(e)}")
        log.debug(f"Exception type: {type(e)}")
        import traceback
        log.debug(f"Traceback: {traceback.format_exc()}")
        processing_time = _calculate_processing_time(start_time)
        AgentLogger.log_error(log_context, f"Market and course recommendation failed: {str(e)}", processing_time)
        
        # Record failure
        await market_memory.record_attempt(
            'market_analysis', 'llm', False, 0.0, processing_time
        )
        
        # Return error response with empty course recommendations
        return {
            "status": "error",
            "node": "market_and_course_recommender",
            "output": {
                "raw_skill_gap_analysis_output": raw_skill_gap_analysis_output,
                "market_insights": [],
                "course_recommendations": []
            },
            "error": f"Market and course recommendation failed: {str(e)}",
            "processing_time": processing_time,
            "request_id": request_id
        }

        for reason in invalid_reasons[:10]:  # Log first 10 reasons
            AgentLogger.log_warning(log_context, f"  - {reason}")
        if len(invalid_reasons) > 10:
            AgentLogger.log_warning(log_context, f"  - ... and {len(invalid_reasons) - 10} more")
    
    # Remove duplicates based on title and type
    original_count = len(valid_courses)
    seen: Set[Tuple[str, str]] = set()
    deduped_courses = []
    for c in valid_courses:
        key = ((c.get('title') or '').lower().strip(), (c.get('type') or '').lower().strip())
        if key not in seen:
            seen.add(key)
            deduped_courses.append(c)
    valid_courses = deduped_courses
    duplicates_removed = original_count - len(valid_courses)
    
    if duplicates_removed > 0:
        AgentLogger.log_info(log_context, f"🔄 Removed {duplicates_removed} duplicate courses")
    
    # Limit to maximum recommendations
    if len(valid_courses) > MAX_RECOMMENDATIONS_COUNT:
        AgentLogger.log_info(log_context, f"📊 Limited to {MAX_RECOMMENDATIONS_COUNT} courses (had {len(valid_courses)})")
        valid_courses = valid_courses[:MAX_RECOMMENDATIONS_COUNT]
    
    AgentLogger.log_info(log_context, f"✅ FINAL RESULT: {len(valid_courses)} valid course recommendations (no URL validation performed)")
    
    return valid_courses

async def _validate_and_filter_additional_resources(resources: List[Dict[str, Any]], log_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate and filter books, research papers, videos, blogs, and webpages. Keep only allowed domains and accessible URLs."""
    if not isinstance(resources, list):
        return []
    valid: List[Dict[str, Any]] = []
    for i, r in enumerate(resources):
        if not isinstance(r, dict):
            continue
        title = str(r.get('title', '')).strip()
        url = str(r.get('url', '')).strip()
        rtype = str(r.get('type', '')).strip().lower()
        source = str(r.get('source', '')).strip()
        target_skill = str(r.get('target_skill', '')).strip()
        if not title or not rtype:
            continue
        # Books may not have URLs, but other types must have URLs
        if rtype != 'book' and not url:
            continue
        if url and not url.startswith('https://'):
            continue
        if url and not _is_allowed_resource_domain(url):
            continue
        # Type-specific basic checks
        if url:
            u = urlparse(url)
            host = (u.hostname or '').lower()
            path = (u.path or '').lower()
            if rtype == 'video':
                if 'youtube.com' not in host or not (path.startswith('/watch') or path.startswith('/playlist')):
                    continue
            elif rtype == 'paper':
                if 'arxiv.org' in host and not (path.startswith('/abs/') or path.startswith('/pdf/')):
                    continue
                if 'ieeexplore.ieee.org' in host and '/document/' not in path:
                    continue
                if 'dl.acm.org' in host and '/doi/' not in path:
                    continue
            elif rtype == 'book':
                if url:  # Books may have optional URLs
                    if 'oreilly.com' in host and '/library/' not in path:
                        continue
                    if 'packtpub.com' in host and '/product/' not in path:
                        continue
            elif rtype in ['blog', 'webpage']:
                # Blogs and webpages should have valid paths (not just homepage)
                invalid_patterns = ['/search', '/browse', '/category', '/tag', '/archive', '/page/', '/?page=']
                if any(pattern in path or pattern in u.query.lower() for pattern in invalid_patterns):
                    continue
                # Must have a meaningful path (not just root)
                if len(path) <= 1 and not u.query:
                    continue
        # Accessibility check (skip for books without URLs)
        if rtype == 'book' and not url:
            # Books without URLs are automatically valid
            item = {
                'type': rtype,
                'title': title[:200],
                'source': source[:120] if source else '',
                'relevance_score': max(0.0, min(float(r.get('relevance_score') or 0.7), 1.0)),
                'target_skill': target_skill[:120],
                'description': str(r.get('description', ''))[:300]
            }
            if r.get('author'):
                item['author'] = str(r.get('author'))[:120]
            valid.append(item)
        elif url:
            url_validation = await _generate_url_validation_info_async(url, title, source or rtype, rtype)
            if url_validation.get('status') == 'valid' and url_validation.get('accessible') is True:
                item = {
                    'type': rtype,
                    'title': title[:200],
                    'source': source[:120] if source else '',
                    'url': _normalize_course_url(url),
                    'relevance_score': max(0.0, min(float(r.get('relevance_score') or 0.7), 1.0)),
                    'target_skill': target_skill[:120],
                    'description': str(r.get('description', ''))[:300]
                }
                if rtype == 'video' and r.get('duration'):
                    item['duration'] = str(r.get('duration'))[:60]
                if rtype in ['blog', 'webpage', 'book'] and r.get('author'):
                    item['author'] = str(r.get('author'))[:120]
                item['url_validation'] = url_validation
                valid.append(item)
    # Dedupe by URL+title
    valid = _dedupe_courses(valid)
    # Cap results
    if len(valid) > MAX_RECOMMENDATIONS_COUNT:
        valid = valid[:MAX_RECOMMENDATIONS_COUNT]
    return valid

def _is_course_url_high_quality(url: str, title: str, provider: str) -> bool:
    """Additional quality checks for course URLs."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        path = parsed.path.lower()
        
        # Check if URL looks like a real course (not generic)
        if len(path.split('/')) < 3:
            return False
        
        # Extract course slug for validation
        course_slug = ''
        if 'udemy.com' in hostname and '/course/' in path:
            # Extract slug after /course/
            parts = path.split('/course/')
            if len(parts) > 1:
                course_slug = parts[1].split('/')[0].split('?')[0]
        elif 'coursera.org' in hostname:
            # Extract slug after /learn/, /specializations/, or /professional-certificates/
            for pattern in ['/learn/', '/specializations/', '/professional-certificates/']:
                if pattern in path:
                    course_slug = path.split(pattern, 1)[1].split('/')[0].split('?')[0]
                    break
        
        # Check for placeholder/invalid course slugs
        if course_slug:
            invalid_slugs = [
                'course-name', 'test-course', 'example-course', 'new-course',
                'course', 'test', 'example', 'demo', 'sample', 'placeholder',
                'tutorial', 'learn', 'training', 'class'
            ]
            if course_slug.lower() in invalid_slugs or any(inv in course_slug.lower() for inv in ['-course', '-test', '-example']):
                return False
            
            # Course slug should be meaningful (at least 5 characters for real courses)
            if len(course_slug) < 5:
                return False
        
        # Check for suspicious patterns in full URL
        suspicious_patterns = [
            'test', 'demo', 'sample', 'example', 'placeholder',
            'coming-soon', 'under-construction', 'maintenance'
        ]
        
        url_lower = url.lower()
        for pattern in suspicious_patterns:
            if pattern in url_lower:
                return False
        
        # Check if title and provider match the URL domain
        provider_lower = provider.lower()
        if 'coursera' in provider_lower and 'coursera.org' not in hostname:
            return False
        if 'udemy' in provider_lower and 'udemy.com' not in hostname:
            return False
        if 'edx' in provider_lower and 'edx.org' not in hostname:
            return False
        if 'linkedin' in provider_lower and 'linkedin.com' not in hostname:
            return False
        
        return True
        
    except Exception:
        return False

async def _check_url_accessibility(url: str, timeout: int = 5) -> Tuple[bool, str, Optional[int]]:
    """Check URL accessibility using a lightweight GET with a real User-Agent.
    Accept 200 and standard redirects. Do not download full content.
    For courses, this is critical - courses must return 200 or redirect to be valid.
    Returns: (is_accessible, message, status_code)
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout), headers=headers) as session:
            async with session.get(url, allow_redirects=True) as response:
                # Read a tiny portion to establish connection, then release
                try:
                    await response.content.readany()
                except Exception:
                    pass
                status = response.status
                if status == 200:
                    return True, f"URL accessible (HTTP {status})", status
                if status in [301, 302, 303, 307, 308]:
                    # Follow redirect and check final status
                    final_url = str(response.url)
                    return True, f"URL accessible with redirect (HTTP {status}) to {final_url}", status
                # Return status code so we can distinguish between 403 (bot blocking) and 404 (not found)
                return False, f"URL not accessible (HTTP {status})", status
    except aiohttp.ClientTimeout:
        return False, "URL check timed out", None
    except aiohttp.ClientError as e:
        return False, f"URL not accessible: {str(e)}", None
    except Exception as e:
        return False, f"URL validation error: {str(e)}", None

def _generate_url_validation_info(url: str, title: str, provider: str) -> Dict[str, Any]:
    """Generate URL validation information for the course output."""
    try:
        # Basic URL validation
        is_valid = _validate_course_url_strict(url)
        
        if not is_valid:
            return {
                "status": "invalid",
                "accessible": False,
                "message": "URL failed validation checks"
            }
        
        # For now, return pending status - will be updated by actual HTTP check
        return {
            "status": "pending",
            "accessible": None,
            "message": "URL validation in progress"
        }
            
    except Exception as e:
        return {
            "status": "error",
            "accessible": False,
            "message": f"Validation error: {str(e)}"
        }

async def _generate_url_validation_info_async(url: str, title: str, provider: str, resource_type: str = "course") -> Dict[str, Any]:
    """Generate URL validation information with actual HTTP checking."""
    try:
        # FAST PATH: Early rejection of obviously invalid URLs before HTTP check
        # This saves time by avoiding HTTP requests for clearly invalid URLs
        if resource_type == "course":
            # For courses, do strict validation first
            is_valid = _validate_course_url_strict(url)
            if not is_valid:
                return {
                    "status": "invalid",
                    "accessible": False,
                    "message": f"Course URL failed structure validation"
                }
            # Additional quality check before HTTP request
            if not _is_course_url_high_quality(url, title, provider):
                return {
                    "status": "invalid",
                    "accessible": False,
                    "message": f"Course URL failed quality checks (likely placeholder or invalid slug)"
                }
        elif resource_type in ["blog", "webpage"]:
            # For blogs/webpages, use resource domain validation
            is_valid = _is_allowed_resource_domain(url)
            if not is_valid:
                return {
                    "status": "invalid",
                    "accessible": False,
                    "message": f"Blog/webpage URL failed domain validation"
                }
            # Additional path validation for blogs/webpages
            try:
                parsed = urlparse(url)
                path = (parsed.path or '').lower()
                query = (parsed.query or '').lower()
                # Exclude invalid patterns
                invalid_patterns = ['/search', '/browse', '/category', '/tag', '/archive']
                if any(pattern in path or pattern in query for pattern in invalid_patterns):
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Blog/webpage URL appears to be a search/category page"
                    }
                elif '/page/' in path or '?page=' in query:
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Blog/webpage URL appears to be a pagination page"
                    }
                elif len(path) <= 1 and not query:
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Blog/webpage URL appears to be a homepage"
                    }
            except Exception:
                return {
                    "status": "invalid",
                    "accessible": False,
                    "message": f"Blog/webpage URL failed path validation"
                }
        else:
            # For other types (paper, video), use resource domain validation
            is_valid = _is_allowed_resource_domain(url)
            if not is_valid:
                return {
                    "status": "invalid",
                    "accessible": False,
                    "message": f"Resource URL failed domain validation for {resource_type}"
                }
        
        # Check actual accessibility (only for URLs that passed structure validation)
        is_accessible, message, status_code = await _check_url_accessibility(url)
        
        # Determine platform-specific message and check if it's a trusted platform
        parsed = urlparse(url)
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        
        # List of trusted educational platforms - accept even if HTTP check fails (might be bot blocking)
        trusted_platforms = {
            'youtube.com': 'YouTube',
            'coursera.org': 'Coursera',
            'udemy.com': 'Udemy',
            'edx.org': 'edX',
            'linkedin.com': 'LinkedIn Learning',
            'pluralsight.com': 'Pluralsight',
            'freecodecamp.org': 'FreeCodeCamp',
            'medium.com': 'Medium',
            'dev.to': 'Dev.to',
            'hashnode.com': 'Hashnode',
            'github.com': 'GitHub',
            'react.dev': 'React',
            'vuejs.org': 'Vue.js',
            'angular.io': 'Angular',
            'nodejs.org': 'Node.js',
            'mongodb.com': 'MongoDB',
            'postgresql.org': 'PostgreSQL',
            'docker.com': 'Docker',
            'kubernetes.io': 'Kubernetes',
            'terraform.io': 'Terraform',
            'azure.microsoft.com': 'Azure',
            'aws.amazon.com': 'AWS',
            'cloud.google.com': 'Google Cloud',
            'developers.google.com': 'Google Developers'
        }
        
        is_trusted = any(domain in hostname for domain in trusted_platforms.keys())
        platform_name = next((name for domain, name in trusted_platforms.items() if domain in hostname), "URL")
        
        # For COURSES from trusted platforms, accept if URL structure is valid even if HTTP check fails
        # BUT: Reject 404s (course doesn't exist) - only accept 403s (bot blocking) or timeouts/errors
        # Course platforms (Coursera, Udemy, edX) often block bots but URLs are valid for users
        # For other resource types (blogs, webpages), be more lenient as they might block bots
        if resource_type == "course":
            # Check if it's a trusted course platform
            course_platforms = ['coursera.org', 'udemy.com', 'edx.org', 'linkedin.com', 'pluralsight.com']
            is_course_platform = any(domain in hostname for domain in course_platforms)
            
            if is_accessible:
                # Only accept if HTTP check actually succeeded (200 or valid redirect)
                platform_msg = f"{platform_name} course verified accessible"
                return {
                    "status": "valid",
                    "accessible": True,
                    "message": f"{platform_msg} - {message}"
                }
            else:
                # HTTP check failed - reject the course
                # Even from trusted platforms, we need actual HTTP success (200/redirect) to accept
                if status_code == 404:
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Course URL not found (HTTP 404) - course does not exist on {platform_name}"
                    }
                elif status_code == 403:
                    # 403 could be bot blocking, but we can't verify the course exists
                    # Reject to ensure only working links are returned
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Course URL blocked or inaccessible (HTTP 403) - cannot verify course exists on {platform_name}"
                    }
                elif status_code is None:
                    # Timeout or connection error - reject to ensure reliability
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Course URL validation timed out or connection failed - cannot verify course exists on {platform_name}"
                    }
                else:
                    # Other error status codes - reject
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Course URL not accessible (HTTP {status_code}) - {message}"
                    }
        else:
            # For blogs, webpages, videos, papers - require actual HTTP success
            if is_accessible:
                # Only accept if HTTP check actually succeeded (200 or valid redirect)
                platform_msg = f"{platform_name} resource verified accessible"
                return {
                    "status": "valid",
                    "accessible": True,
                    "message": f"{platform_msg} - {message}"
                }
            else:
                # HTTP check failed - reject the resource
                if status_code == 404:
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Resource URL not found (HTTP 404) - resource does not exist on {platform_name}"
                    }
                elif status_code == 403:
                    # 403 could be bot blocking, but we can't verify the resource exists
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Resource URL blocked or inaccessible (HTTP 403) - cannot verify resource exists on {platform_name}"
                    }
                elif status_code is None:
                    # Timeout or connection error - reject to ensure reliability
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"Resource URL validation timed out or connection failed - cannot verify resource exists on {platform_name}"
                    }
                else:
                    # Other error status codes - reject
                    return {
                        "status": "invalid",
                        "accessible": False,
                        "message": f"URL not accessible (HTTP {status_code}) - {message}"
                    }
            
    except Exception as e:
        return {
            "status": "error",
            "accessible": False,
            "message": f"Validation error: {str(e)}"
        }

async def _validate_urls_batch_async(courses: List[Dict[str, Any]], log_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Validate multiple URLs concurrently and return only those verified accessible.
    Items without URLs are only passed through if type == 'book'.
    """
    if not courses:
        return []
    
    if log_context is None:
        log_context = {"tenant_id": "default", "user_id": "unknown"}
    
    # Prepare inputs for validation
    validation_inputs: List[Tuple[Dict[str, Any], str, str, str]] = []
    accessible_courses: List[Dict[str, Any]] = []
    for course in courses:
        if not isinstance(course, dict):
            continue
        url = course.get('url')
        ctype = str(course.get('type', 'course')).strip().lower()
        if url:
            validation_inputs.append((
                course,
                url,
                course.get('course', course.get('title', '')),
                course.get('platform', course.get('provider', ''))
            ))
        else:
            # Pass-through for books without URL
            if ctype == 'book' and course.get('title') and (course.get('author') or course.get('provider')):
                course['url_validation'] = {
                    'status': 'not_applicable',
                    'accessible': None,
                    'message': 'URL not required for books'
                }
                accessible_courses.append(course)
    
    # Concurrency limiter to avoid overwhelming network (tunable via COURSE_URL_HTTP_VALIDATION_MAX_PARALLEL)
    semaphore = asyncio.Semaphore(_COURSE_URL_HTTP_VALIDATION_MAX_PARALLEL)
    
    async def validate_one(c: Dict[str, Any], url: str, title: str, provider: str) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str]]:
        async with semaphore:
            try:
                # Get resource type from course dict
                resource_type = str(c.get('type', 'course')).strip().lower()
                url_validation = await _generate_url_validation_info_async(url, title, provider, resource_type)
                return c, url_validation, None
            except Exception as e:
                return c, {"status": "error", "accessible": False, "message": f"Validation error: {str(e)}"}, str(e)
    
    tasks = [asyncio.create_task(validate_one(c, u, t, p)) for (c, u, t, p) in validation_inputs]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    
    inaccessible_count = 0
    inaccessible_reasons: List[str] = []
    
    for course, url_validation, err in results:
        course['url_validation'] = url_validation
        if url_validation.get('status') == 'valid' and url_validation.get('accessible') is True:
            accessible_courses.append(course)
            if log_context:
                AgentLogger.log_info(log_context, f"✅ URL ACCESSIBLE: {course.get('course', course.get('title', 'Unknown'))} - {url_validation.get('message', '')}")
        else:
            inaccessible_count += 1
            reason = f"Course '{course.get('course', course.get('title', 'Unknown'))}': {url_validation.get('message', 'Unknown error')}"
            inaccessible_reasons.append(reason)
            if log_context:
                AgentLogger.log_warning(log_context, f"⚠️ URL NOT VERIFIED: {reason}")
    
    # Log results (filtering applied)
    if log_context and inaccessible_count > 0:
        AgentLogger.log_warning(log_context, f"ℹ️ {inaccessible_count} course URLs removed (not verified accessible)")
        for reason in inaccessible_reasons[:5]:
            AgentLogger.log_warning(log_context, f"  - {reason}")
        if len(inaccessible_reasons) > 5:
            AgentLogger.log_warning(log_context, f"  - ... and {len(inaccessible_reasons) - 5} more")
    
    if log_context:
        AgentLogger.log_info(log_context, f"✅ Returning {len(accessible_courses)} verified-accessible courses")
    
    return accessible_courses

async def _llm_validate_course_url_batch(courses: List[Dict[str, Any]], log_context: Dict[str, Any]) -> List[bool]:
    """Use LLM to validate multiple course URLs in a single call for efficiency."""
    try:
        # Prepare course information for batch validation
        course_info = []
        for i, course in enumerate(courses):
            course_info.append({
                "index": i,
                "type": course.get('type', 'course'),
                "title": course.get('title', ''),
                "provider": course.get('provider', ''),
                "url": course.get('url', ''),
                "author": course.get('author', '')  # For books
            })
        
        validation_prompt = f"""
You are a learning resource URL validation expert. Analyze the following learning resources and determine which URLs are legitimate, direct links to real educational content.

**Resource Information:**
{json.dumps(course_info, indent=2)}

**Resource Types to Validate:**
- **course**: Structured online courses (Coursera, Udemy, edX, LinkedIn Learning)
- **book**: Books (may not have URLs - if missing URL but has author, accept it)
- **blog**: Educational blog posts/articles (Medium, Dev.to, Hashnode, FreeCodeCamp, etc.)
- **webpage**: Documentation pages, developer guides, official docs (React.dev, Vue.js, MDN, etc.)
- **video**: YouTube videos or playlists
- **paper**: Research papers (arXiv, IEEE, ACM)

**Validation Criteria for each resource:**
1. **For courses (CRITICAL)**: 
   - Is this a direct link to a SPECIFIC, REAL course that actually exists (not a search/browse page)?
   - Does the course slug look legitimate (proper kebab-case format, not placeholder text)?
   - For Udemy: Course slug should be lowercase, alphanumeric with hyphens (e.g., "complete-python-bootcamp", NOT "course-name" or "test-course")
   - For Coursera: Course slug should follow proper format (e.g., "machine-learning" not "course" or "test")
   - REJECT if URL looks like a placeholder, example, or hypothetical course
2. **For books**: If URL is missing but author is provided, ACCEPT it. If URL is provided, validate it.
3. **For blogs/webpages**: Is this a direct link to a specific article/page (not a category/listing page)?
4. **For videos**: Is this a direct link to a YouTube video or playlist?
5. **For papers**: Is this a direct link to a research paper?
6. Does the URL structure match the expected pattern for the provider/type?
7. Does the URL appear to point to real, existing educational content (not hypothetical/fake)?
8. Is the provider correctly identified (if provided)?
9. Does the URL look legitimate and not like a placeholder, test link, or example URL?

**Common Invalid Patterns to REJECT:**
- Placeholder slugs: "course-name", "test-course", "example-course", "new-course"
- Generic patterns: "course", "learn", "tutorial" as standalone slugs
- URLs that look generated or hypothetical rather than real courses

**Provider-Specific Patterns:**
- **Courses**: 
  - Coursera: /learn/, /specializations/, /professional-certificates/
  - Udemy: /course/course-name/
  - edX: /course/, /learn/, /micromasters/
  - LinkedIn Learning: /learning/course-name
- **Blogs**: Medium, Dev.to, Hashnode, FreeCodeCamp articles (not category/search pages)
- **Webpages**: Official docs, developer guides (React.dev, Vue.js, MDN, etc.)
- **Videos**: YouTube /watch?v= or /playlist?list=
- **Papers**: arXiv /abs/ or /pdf/, IEEE /document/, ACM /doi/

**CRITICAL RULES:**
- **For COURSES (Udemy, Coursera, edX, LinkedIn Learning)**: 
  - URL structure MUST be valid and match platform patterns
  - If HTTP check succeeds (200 or redirect), accept it
  - If HTTP check fails BUT URL structure is valid and from trusted platform, ACCEPT it (platforms may block bots)
  - REJECT only if URL structure is invalid or from unknown platform
- **Books without URLs but with authors are VALID** - accept them
- **Blogs and webpages are VALID learning resources** - accept educational articles/docs
- **YouTube videos are VALID** - accept individual videos and playlists
- **Documentation pages are VALID** - accept official docs and guides
- **GitHub repositories can be VALID** if they're educational resources (tutorials, guides)
- Reject only: search pages, category pages, listing pages, marketing pages, homepages, inaccessible courses

**IMPORTANT: You must return ONLY valid JSON. No additional text, explanations, or formatting outside the JSON object.**

**Response Format (return exactly this structure):**
{{
    "validations": [
        {{
            "index": 0,
            "is_valid": true,
            "confidence": 0.85,
            "reason": "Valid course URL with proper structure"
        }},
        {{
            "index": 1,
            "is_valid": false,
            "confidence": 0.2,
            "reason": "Appears to be search page"
        }}
    ]
}}

Analyze all courses and return a JSON object with validations array."""

        # Issue 5.1: Use structured output for type-safe parsing
        validation_result = None
        try:
            result: BatchValidationResult = await asyncio.wait_for(
                invoke_structured_llm(
                    validation_prompt,
                    BatchValidationResult,
                    task_type=TaskType.COURSE_VALIDATION,
                    preferred_model="gemini-2.5-flash",
                    agent_name="market_and_course_recommender",
                    max_output_tokens=2000,
                    temperature=0.1,
                    timeout=10.0,
                    raise_on_fallback=False,
                    skip_cache=True,
                ),
                timeout=10,
            )
            validation_result = {"validations": [v.model_dump() for v in result.validations]}
            AgentLogger.log_debug(log_context, f"Structured validation output: {len(result.validations)} validations")
        except asyncio.TimeoutError:
            AgentLogger.log_warning(log_context, "Structured validation timed out, falling back to manual parsing")
        except Exception as structured_err:
            AgentLogger.log_warning(log_context, f"Structured validation failed ({structured_err}), falling back to manual parsing")
        
        # Fallback to manual parsing if structured output failed
        if not validation_result:
            llm_response = await asyncio.wait_for(
                invoke_llm(
                    prompt=validation_prompt, 
                    task_type="course_validation",
                    agent_name="market_and_course_recommender",
                    preferred_model="gemini-2.5-flash"
                ), timeout=10
            )
            response_text = _to_text(llm_response)
            AgentLogger.log_info(log_context, f"LLM Response: {response_text[:500]}...")
            validation_result = _extract_json_from_response(response_text, 2000)
        
        if isinstance(validation_result, dict) and 'validations' in validation_result:
            validations = validation_result['validations']
            results = [False] * len(courses)
            
            for validation in validations:
                if isinstance(validation, dict):
                    index = validation.get('index', -1)
                    is_valid = validation.get('is_valid', False)
                    confidence = validation.get('confidence', 0.0)
                    reason = validation.get('reason', 'No reason provided')
                    
                    if 0 <= index < len(courses):
                        if is_valid and confidence > 0.6:
                            results[index] = True
                            AgentLogger.log_info(log_context, f"✅ LLM VALIDATED: '{courses[index].get('title', '')}' - {reason} (confidence: {confidence})")
                        else:
                            AgentLogger.log_warning(log_context, f"❌ LLM REJECTED: '{courses[index].get('title', '')}' - {reason} (confidence: {confidence})")
            
            return results
        else:
            AgentLogger.log_warning(log_context, f"❌ LLM BATCH VALIDATION FAILED - Invalid response format: {validation_result}")
            return await _fallback_individual_validation(courses, log_context)
            
    except asyncio.TimeoutError:
        AgentLogger.log_warning(log_context, "⏰ LLM BATCH VALIDATION TIMEOUT - Using individual validation")
        return [False] * len(courses)
    except Exception as e:
        AgentLogger.log_warning(log_context, f"❌ LLM BATCH VALIDATION ERROR - {str(e)} - Using individual validation")
        return [False] * len(courses)

async def _fallback_individual_validation(courses: List[Dict[str, Any]], log_context: Dict[str, Any]) -> List[bool]:
    """Fallback to individual validation when batch validation fails. Bounded parallel LLM calls."""

    _sem = asyncio.Semaphore(_COURSE_LLM_VALIDATION_MAX_PARALLEL)

    async def _validate_one(course: Dict[str, Any]) -> bool:
        title = course.get('title', '')
        provider = course.get('provider', '')
        url = course.get('url', '')
        async with _sem:
            try:
                return await _llm_validate_course_url(url, title, provider, log_context)
            except Exception as e:
                AgentLogger.log_warning(log_context, f"❌ Individual validation failed for '{title}': {str(e)}")
                return False

    results = await asyncio.gather(*[_validate_one(course) for course in courses])
    return list(results)

async def _llm_validate_course_url(url: str, title: str, provider: str, log_context: Dict[str, Any]) -> bool:
    """Use LLM to validate if a course URL is legitimate and points to a real course."""
    try:
        validation_prompt = f"""
You are a course URL validation expert. Analyze the following course information and determine if the URL is a legitimate, direct link to a real course.

**Course Information:**
- Title: {title}
- Provider: {provider}
- URL: {url}

**Validation Criteria:**
1. Is this a direct link to a specific course (not a search/browse page)?
2. Does the URL structure match the expected pattern for the provider?
3. Does the URL appear to point to a real, existing course?
4. Is the provider correctly identified?
5. Does the URL look legitimate and not like a placeholder or test link?

**Provider-Specific Patterns:**
- Coursera: Should have /learn/, /specializations/, or /professional-certificates/
- Udemy: Should have /course/course-name/
- edX: Should have /course/, /learn/, or /micromasters/
- LinkedIn Learning: Should have /learning/course-name
- YouTube: Should have /watch?v= or /playlist?list=
- Pluralsight: Should have /courses/course-name

**IMPORTANT: You must return ONLY valid JSON. No additional text, explanations, or formatting outside the JSON object.**

**Response Format (return exactly this structure):**
{{
    "is_valid": true,
    "confidence": 0.85,
    "reason": "Valid course URL with proper structure"
}}

Analyze the URL and return a JSON object with is_valid, confidence, and reason."""

        # Issue 5.1: Use structured output for type-safe parsing
        validation_result = None
        try:
            result: SingleValidationResult = await asyncio.wait_for(
                invoke_structured_llm(
                    validation_prompt,
                    SingleValidationResult,
                    task_type=TaskType.COURSE_VALIDATION,
                    preferred_model="gemini-2.5-flash",
                    agent_name="market_and_course_recommender",
                    max_output_tokens=500,
                    temperature=0.1,
                    timeout=8.0,
                    raise_on_fallback=False,
                    skip_cache=True,
                ),
                timeout=8,
            )
            validation_result = result.model_dump()
        except asyncio.TimeoutError:
            AgentLogger.log_warning(log_context, f"Structured validation timed out for '{title}', falling back")
        except Exception as structured_err:
            AgentLogger.log_warning(log_context, f"Structured validation failed for '{title}' ({structured_err}), falling back")
        
        # Fallback to manual parsing if structured output failed
        if not validation_result:
            llm_response = await asyncio.wait_for(
                invoke_llm(
                    prompt=validation_prompt, 
                    task_type="course_validation",
                    agent_name="market_and_course_recommender",
                    preferred_model="gemini-2.5-flash"
                ), timeout=8
            )
            response_text = _to_text(llm_response)
            AgentLogger.log_info(log_context, f"Individual LLM Response for '{title}': {response_text[:200]}...")
            validation_result = _extract_json_from_response(response_text, 500)
        
        if isinstance(validation_result, dict):
            is_valid = validation_result.get('is_valid', False)
            confidence = validation_result.get('confidence', 0.0)
            reason = validation_result.get('reason', 'No reason provided')
            
            if is_valid and confidence > 0.6:
                AgentLogger.log_info(log_context, f"✅ LLM VALIDATED: '{title}' - {reason} (confidence: {confidence})")
                return True
            else:
                AgentLogger.log_warning(log_context, f"❌ LLM REJECTED: '{title}' - {reason} (confidence: {confidence})")
                return False
        else:
            AgentLogger.log_warning(log_context, f"❌ LLM VALIDATION FAILED: '{title}' - Invalid response format: {validation_result}")
            return False
            
    except asyncio.TimeoutError:
        AgentLogger.log_warning(log_context, f"⏰ LLM VALIDATION TIMEOUT: '{title}' - Using fallback validation")
        return _is_course_url_high_quality(url, title, provider)
    except Exception as e:
        AgentLogger.log_warning(log_context, f"❌ LLM VALIDATION ERROR: '{title}' - {str(e)} - Using fallback validation")
        return _is_course_url_high_quality(url, title, provider)


# ---------- Deterministic functions removed - using LLM-only approach ----------

@traceable(name="market_and_course_recommender_agent")
async def market_and_course_recommender_agent(state: Dict[str, Any], tenant_id: str = "default_tenant") -> Dict[str, Any]:
    """
    Enhanced market and course recommender with LLM-only approach.
    """
    # Use centralized logging
    log_context = create_log_context("market_and_course_recommender", tenant_id)
    start_time = log_context["start_time"]
    request_id = log_context["request_id"]
    
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"--- Entering Market and Course Recommender Agent for tenant: {tenant_id} ---")
    
    # Get tenant-scoped memory
    market_memory = await get_market_course_recommender_memory(tenant_id)
    
    # Use selective data passing for optimization
    if "structured_resume" in state:
        # Create optimized state with only the data this agent needs
        optimized_state = create_agent_state(state, "market_and_course_recommender")
        if log.isEnabledFor(logging.INFO):
            log.info(f"📊 Market & Course Recommender using selective data passing - filtered {len(optimized_state.get('structured_resume', {}))} fields")
        
        # Use the optimized state for processing
        structured_resume = optimized_state.get("structured_resume", {})
    else:
        structured_resume = state.get("structured_resume", {})
    
    user_interests = state.get("user_interests", [])
    
    # Check if this is a 2nd call (user_interests from 2nd call should be prioritized)
    is_second_call = state.get("is_second_call", False)
    if is_second_call and user_interests:
        if log.isEnabledFor(logging.INFO):
            log.info(
                f"✅ 2ND CALL: Using user_interests from 2nd call ({len(user_interests) if isinstance(user_interests, list) else 'N/A'} items) as context"
            )
            AgentLogger.log_info(log_context, "2nd call detected - using user_interests from 2nd call as context")
    
    raw_skill_gap_analysis_output = state.get("raw_skill_gap_analysis_output", {})
    
    # Ensure raw_skill_gap_analysis_output is a dictionary
    if isinstance(raw_skill_gap_analysis_output, str):
        try:
            raw_skill_gap_analysis_output = json.loads(raw_skill_gap_analysis_output)
            if log.isEnabledFor(logging.DEBUG):
                log.debug("Successfully parsed raw_skill_gap_analysis_output as JSON")
        except (json.JSONDecodeError, TypeError):
            if log.isEnabledFor(logging.WARNING):
                AgentLogger.log_warning(
                    log_context, "Failed to parse raw_skill_gap_analysis_output as JSON, using empty dict"
                )
            raw_skill_gap_analysis_output = {}
    elif not isinstance(raw_skill_gap_analysis_output, dict):
        if log.isEnabledFor(logging.WARNING):
            AgentLogger.log_warning(
                log_context,
                f"raw_skill_gap_analysis_output is not a dict or string, using empty dict"
            )
        raw_skill_gap_analysis_output = {}

    assessment_results = state.get("assessment_results")
    report = state.get("report")
    session_id = state.get("session_id")
    
    # Only hydrate from session if critical data is missing
    needs_hydration = (not structured_resume or not raw_skill_gap_analysis_output) and session_id
    if needs_hydration:
        try:
            # Run blocking I/O in thread pool to avoid blocking event loop
            session_data = await run_blocking_io(get_chat_session, session_id)

            if not structured_resume:
                structured_resume = session_data.get("structured_resume", {})

            # Only get user_interests from session if not already present in state (2nd call has them in state)
            if not user_interests and not is_second_call:
                interest_data = session_data.get("interest_filler", {})
                user_interests = interest_data.get("user_interests", [])
            elif is_second_call:
                if log.isEnabledFor(logging.DEBUG):
                    log.debug("✅ 2ND CALL: Keeping user_interests from state (2nd call), not overriding from session")

            if not raw_skill_gap_analysis_output:
                skill_gap_data = (
                    session_data.get("career_advisor", {}).get("raw_skill_gap_analysis_output", {})
                    or session_data.get("raw_skill_gap_analysis_output", {})
                )

                # Ensure skill_gap_data is a dictionary
                if isinstance(skill_gap_data, str):
                    try:
                        raw_skill_gap_analysis_output = json.loads(skill_gap_data)
                        if log.isEnabledFor(logging.DEBUG):
                            log.debug("Successfully parsed skill_gap_data as JSON")
                    except (json.JSONDecodeError, TypeError):
                        if log.isEnabledFor(logging.WARNING):
                            AgentLogger.log_warning(
                                log_context,
                                "Failed to parse skill_gap_data from session as JSON, using empty dict"
                            )
                        raw_skill_gap_analysis_output = {}
                elif isinstance(skill_gap_data, dict):
                    raw_skill_gap_analysis_output = skill_gap_data
                else:
                    if log.isEnabledFor(logging.WARNING):
                        AgentLogger.log_warning(
                            log_context,
                            f"skill_gap_data from session is not a dict or string, using empty dict"
                        )
                    raw_skill_gap_analysis_output = {}

            if not assessment_results:
                assessment_results = session_data.get("assessment_evaluator", {}).get("assessment_results")

            if not report:
                report = session_data.get("report_generator", {}).get("report")
        except Exception as e:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"Exception during session hydration: {str(e)}")
            pass

    # Section 8 Issue 1: agent-level cache
    from core.utils import get_cached_response, cache_response
    cache_input = {
        "structured_resume": structured_resume,
        "user_interests": user_interests,
        "raw_skill_gap_analysis_output": raw_skill_gap_analysis_output,
        "assessment_results": assessment_results,
        "report": report,
    }
    cached_result = get_cached_response("market_and_course_recommender", cache_input)
    if cached_result:
        log.info("Cache hit for market_and_course_recommender - returning cached result")
        return cached_result
    
    # Security validation
    if len(str(structured_resume)) > MAX_PROMPT_CHARS:
        AgentLogger.log_warning(log_context, f"Structured resume too large, truncating")
        structured_resume = str(structured_resume)[:MAX_PROMPT_CHARS]
    
    # Aggregate assessment history for stable salary calculation
    uid = state.get("uid")
    assessment_history = []
    salary_modifiers = None
    
    if uid:
        # Parse current assessment if available
        current_assessment = None
        if assessment_results:
            if isinstance(assessment_results, str):
                try:
                    current_assessment = json.loads(assessment_results)
                except (json.JSONDecodeError, TypeError):
                    current_assessment = None
            elif isinstance(assessment_results, dict):
                current_assessment = assessment_results
        
        # Aggregate all assessment history (wrap blocking I/O)
        from core.utils import run_blocking_io
        assessment_history = await run_blocking_io(_aggregate_assessment_history, uid, current_assessment)
        
        # Calculate salary modifiers from aggregated history (for logging/reference only)
        # LLM will determine actual adjustments based on raw assessment data
        salary_modifiers = _calculate_salary_modifiers_from_assessments(assessment_history, current_assessment)
        log.info(f"🔍 MARKET_COURSE_RECOMMENDER: Calculated reference salary modifiers: {salary_modifiers.get('total_adjustment_percent', 0):.1f}% (LLM will determine actual adjustment)")
    
    # Log if assessment results are found
    log.debug("Checking if assessment_results exist")
    if assessment_results:
        log.debug(f"Assessment results found, type: {type(assessment_results)}")
        log.info(f"🔍 MARKET_COURSE_RECOMMENDER: Assessment results found, integrating into analysis")
        
        # Ensure assessment_results is a dictionary before calling .get()
        if isinstance(assessment_results, str):
            log.debug("assessment_results is string, attempting JSON parse")
            try:
                assessment_results = json.loads(assessment_results)
                log.debug("Successfully parsed assessment_results as JSON")
            except (json.JSONDecodeError, TypeError):
                log.debug("Failed to parse assessment_results as JSON")
                AgentLogger.log_warning(log_context, f"Failed to parse assessment_results as JSON, skipping integration")
                assessment_results = None
        
        if isinstance(assessment_results, dict):
            log.debug("assessment_results is dict, getting total_score")
            log.info(f"🔍 MARKET_COURSE_RECOMMENDER: Assessment score: {assessment_results.get('total_score', 'N/A')}")
            
            # Integrate assessment results into skill gap analysis if available
            log.debug("Checking if raw_skill_gap_analysis_output is dict for integration")
            if isinstance(raw_skill_gap_analysis_output, dict):
                log.debug("raw_skill_gap_analysis_output is dict, proceeding with integration")
                # Ensure report is a dictionary before calling .get()
                report_summary = ""
                if report:
                    log.debug(f"Report exists, type: {type(report)}")
                    if isinstance(report, str):
                        log.debug("report is string, attempting JSON parse")
                        try:
                            report = json.loads(report)
                            log.debug("Successfully parsed report as JSON")
                        except (json.JSONDecodeError, TypeError):
                            log.debug("Failed to parse report as JSON")
                            AgentLogger.log_warning(log_context, f"Failed to parse report as JSON, using empty string")
                            report = None
                    
                    if isinstance(report, dict):
                        log.debug("report is dict, getting summary")
                        report_summary = report.get("summary", "")
                
                log.debug("Creating assessment_performance integration")
                # Store comprehensive assessment data for LLM-driven salary adjustment
                # Include all assessment details so LLM can make intelligent decisions
                raw_skill_gap_analysis_output["assessment_performance"] = {
                    "total_score": assessment_results.get("total_score", 0),
                    "max_score": assessment_results.get("max_score", 100),
                    "section_scores": assessment_results.get("section_scores", {}),
                    "assessment_topic": assessment_results.get("assessment_topic", ""),
                    "assessment_type": assessment_results.get("assessment_type", ""),
                    "report": report_summary,
                    "assessment_history_count": len(assessment_history),
                    "assessment_history": assessment_history,  # Include full history for context
                    "salary_modifiers": salary_modifiers  # Keep for logging/reference, but LLM determines adjustment
                }
                log.debug("Successfully integrated assessment_performance")
            else:
                log.debug(f"raw_skill_gap_analysis_output is not dict, type: {type(raw_skill_gap_analysis_output)}")
    else:
        log.debug("No assessment_results found")
        # Explicitly mark that no assessments have been taken
        if isinstance(raw_skill_gap_analysis_output, dict):
            if "assessment_performance" not in raw_skill_gap_analysis_output:
                raw_skill_gap_analysis_output["assessment_performance"] = {}
            # CRITICAL: Set assessment_history_count to 0 to indicate NO assessments taken
            # This distinguishes "no assessments" from "0% score from assessment"
            raw_skill_gap_analysis_output["assessment_performance"]["assessment_history_count"] = 0
            raw_skill_gap_analysis_output["assessment_performance"]["has_assessments"] = False
            if salary_modifiers:
                raw_skill_gap_analysis_output["assessment_performance"]["salary_modifiers"] = salary_modifiers
            log.debug("Set assessment_history_count=0 to indicate no assessments taken")
    
    # Derive performance-aware context from latest assessment
    log.debug("Deriving performance-aware assessment context")
    last_topic = state.get("assessment_topic")
    performance_level: Optional[str] = None
    score: Optional[float] = None
    weak_sections: Dict[str, Any] = {}
    strong_sections: Dict[str, Any] = {}
    
    if isinstance(assessment_results, dict):
        score = assessment_results.get("total_score")
        section_scores = assessment_results.get("section_scores", {}) or {}
        weak_sections = {k: v for k, v in section_scores.items() if isinstance(v, (int, float)) and v < 50}
        strong_sections = {k: v for k, v in section_scores.items() if isinstance(v, (int, float)) and v >= 80}
        
        if isinstance(score, (int, float)):
            if score >= 90:
                performance_level = "Outstanding"
            elif score >= 80:
                performance_level = "Excellent"
            elif score >= 70:
                performance_level = "Good"
            elif score >= 50:
                performance_level = "Fair"
            else:
                performance_level = "Needs Improvement"
        
        if not last_topic:
            last_topic = assessment_results.get("assessment_topic")
    
    performance_context = {
        "score": score,
        "performance_level": performance_level,
        "weak_sections": weak_sections,
        "strong_sections": strong_sections,
        "last_topic": last_topic,
    }

    # Normalize the gap analysis payload so downstream prompt builders can rely on
    # consistent keys even when upstream agents omitted them.
    if isinstance(raw_skill_gap_analysis_output, dict):
        if assessment_results and isinstance(assessment_results, dict):
            raw_skill_gap_analysis_output.setdefault("assessment_performance", assessment_results)
        raw_skill_gap_analysis_output.setdefault("performance_context", performance_context)
    
    # Initialize variables to avoid scope issues
    log.debug("Initializing variables")
    final_valid_courses = []
    valid_courses = []
    recommendations = {}
    market_insights = {}  # Initialize market_insights outside try block
    # Ensure salary_modifiers is initialized even if uid is None
    if 'salary_modifiers' not in locals():
        salary_modifiers = None
    log.debug("Variables initialized successfully")
    
    # Granular timing breakdown for performance monitoring
    timing_breakdown = {
        "prompt_generation": 0.0,
        "market_insights_llm": 0.0,
        "course_validation": 0.0,
        "result_processing": 0.0,
        "total": 0.0
    }
    
    log.debug("Starting main processing section")
    try:
        # Step 1: Generate market insights using original prompt
        # Note: We pass salary_modifiers for backward compatibility/logging, but LLM determines actual adjustments
        log.debug("Generating market insights prompt")
        # Extract salary modifiers from skill gap analysis if available (for reference only)
        modifiers_for_prompt = None
        if isinstance(raw_skill_gap_analysis_output, dict):
            assessment_perf = raw_skill_gap_analysis_output.get("assessment_performance", {})
            if isinstance(assessment_perf, dict):
                # Pass modifiers for logging/reference, but LLM will use raw assessment data to determine adjustments
                modifiers_for_prompt = assessment_perf.get("salary_modifiers")
        
        # If not found in skill gap, use the calculated modifiers (for reference only)
        if not modifiers_for_prompt and salary_modifiers:
            modifiers_for_prompt = salary_modifiers
        
        # Detect if this is a rerun (has assessment results or previous outputs)
        is_rerun = bool(
            assessment_results or 
            (salary_modifiers and salary_modifiers.get('assessment_count', 0) > 0) or
            report or
            state.get("rerun_mode", False)
        )
        
        # NEW: Await async prompt generation with Gemini summary
        prompt_start = time.time()
        uid = state.get("uid")
        market_prompt = await generate_market_and_course_recommendation_prompt(
            structured_resume, user_interests, raw_skill_gap_analysis_output, modifiers_for_prompt, is_rerun=is_rerun, uid=uid
        )
        timing_breakdown["prompt_generation"] = time.time() - prompt_start
        # Optionally append high-level performance context for market analysis and salary guidance
        if performance_level:
            market_prompt += f"""

Assessment Performance Context: {json.dumps(performance_context, default=str)}

Assessment Performance Insights:
- Topic Assessed: {performance_context.get('last_topic')}
- Score: {performance_context.get('score')}
- Performance Level: {performance_level}
- Weak Sections: {performance_context.get('weak_sections')}
- Strong Sections: {performance_context.get('strong_sections')}

Use these signals to slightly tune any salary and market commentary, but keep final salary ranges consistent with the role, location, and experience.
"""
        log.debug("Generated market insights prompt successfully")
        
        # Validate market prompt size - use smarter truncation that preserves JSON schema
        max_prompt_size = min(MAX_PROMPT_CHARS, 15000)  # Increased from 10000 to prevent truncation of important data
        if len(market_prompt) > max_prompt_size:
            AgentLogger.log_warning(log_context, f"Market prompt too large, truncating from {len(market_prompt)} to {max_prompt_size}")
            # Smart truncation: preserve the JSON schema at the end which is critical for parsing
            json_schema_marker = "**JSON:**"
            json_schema_start = market_prompt.rfind(json_schema_marker)
            if json_schema_start > 0:
                # Keep instructions + JSON schema, truncate middle content
                instructions = market_prompt[:json_schema_start + len(json_schema_marker)]
                json_schema = market_prompt[json_schema_start:]
                # Calculate available space for instructions
                available_instruction_space = max_prompt_size - len(json_schema)
                if available_instruction_space > 1000:  # Only truncate if we have reasonable space
                    truncated_instructions = instructions[:available_instruction_space]
                    market_prompt = truncated_instructions + json_schema
                else:
                    # Not enough space, use simple truncation but preserve last 2000 chars (likely contains schema)
                    market_prompt = market_prompt[:max_prompt_size - 2000] + market_prompt[-2000:]
            else:
                # Fallback: simple truncation if schema marker not found
                market_prompt = market_prompt[:max_prompt_size]
        
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Calling LLM for market insights")
        if log.isEnabledFor(logging.INFO):
            log.info("📊 Calling LLM for market insights and course recommendations in parallel...")
        try:
            market_llm_start = time.time()

            # Run market analysis and course search in parallel (same inputs, independent outputs)
            async def _market_llm_task():
                return await asyncio.wait_for(
                    invoke_llm(
                        prompt=market_prompt,
                        task_type="market_analysis",
                        agent_name="market_and_course_recommender",
                        preferred_model="gemini-2.5-flash",
                        max_output_tokens=4000,
                        response_mime_type="application/json",
                        skip_cache=True
                    ), timeout=30
                )

            async def _course_task():
                return await asyncio.wait_for(
                    _generate_course_recommendations_from_knowledge_base(
                        structured_resume, user_interests, raw_skill_gap_analysis_output, log_context
                    ), timeout=15
                )

            market_result, course_result = await asyncio.gather(
                _market_llm_task(), _course_task(), return_exceptions=True
            )

            timing_breakdown["market_insights_llm"] = time.time() - market_llm_start
            if log.isEnabledFor(logging.INFO):
                log.info(f"⏱️ MARKET_AND_COURSE_RECOMMENDER: Parallel market+course took {timing_breakdown['market_insights_llm']*1000:.1f}ms")

            # Handle market result
            if isinstance(market_result, Exception):
                if isinstance(market_result, asyncio.TimeoutError):
                    raise market_result
                log.warning(f"Market analysis failed: {market_result}")
                market_llm_response = None
            else:
                market_llm_response = market_result

            # Handle course result
            if isinstance(course_result, Exception):
                log.warning(f"Course recommendations failed: {course_result}")
                course_recommendations_data = {"course_recommendations": []}
            else:
                course_recommendations_data = course_result

            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"Got market LLM response, type: {type(market_llm_response)}")
                if market_llm_response:
                    log.debug(f"Raw LLM response content: {market_llm_response}")
                    log.debug(f"Raw LLM response length: {len(market_llm_response) if market_llm_response else 0}")

            # Parse LLM response with improved extraction for full JSON structure
            log.debug("Parsing LLM response")
            
            # Helper function to extract full JSON structure from markdown-wrapped responses
            def _extract_full_json_structure(text: str) -> Dict[str, Any]:
                """Extract the complete JSON structure, handling markdown text before/after JSON."""
                if not text:
                    return {}
                # LLM fallback (quota/timeout) returns non-JSON - use minimal skeleton so flow continues
                if is_fallback_response(text):
                    log.warning("Market insights LLM returned fallback response (quota/timeout) - using empty structure")
                    return {
                        "market_insights": [], "salary_trends": {}, "career_paths": [],
                        "skill_demand_analysis": {}, "course_recommendations": []
                    }
                # Normalize camelCase keys to snake_case for matching
                def _normalize_keys(d: Dict) -> Dict:
                    key_map = {"marketinsights": "market_insights", "salarytrends": "salary_trends",
                               "careerpaths": "career_paths", "skilldemandanalysis": "skill_demand_analysis",
                               "courserecommendations": "course_recommendations"}
                    out = {}
                    for k, v in d.items():
                        nk = key_map.get(k.lower().replace("_", ""), k)
                        out[nk] = v
                    return out
                # Expected keys that must be present in a valid market insights response
                expected_keys = ['market_insights', 'salary_trends', 'career_paths', 'skill_demand_analysis', 'course_recommendations']
                
                # Step 0: Clean text - remove markdown headers and extra whitespace that might interfere
                # But preserve JSON code blocks
                cleaned_text = text
                
                # Step 1: Try to extract from markdown code blocks
                # Use content up to the *last* ``` so embedded ``` in strings don't truncate (same as core.utils)
                json_start = re.search(r"```json\s*", text, re.IGNORECASE)
                if json_start:
                    start_pos = json_start.end()
                    last_fence = text.rfind("```")
                    if last_fence > start_pos:
                        json_text = text[start_pos:last_fence].strip()
                    else:
                        # Unclosed block: take from ```json to end (handles truncated responses)
                        json_text = text[start_pos:].strip()
                    if json_text:
                        try:
                            parsed = json.loads(json_text)
                            if isinstance(parsed, dict):
                                norm = _normalize_keys(parsed)
                                if any(key in norm for key in expected_keys):
                                    log.debug(f"Successfully parsed JSON from markdown code block with keys: {list(norm.keys())}")
                                    return norm
                        except json.JSONDecodeError as e:
                            log.debug(f"Failed to parse JSON from markdown block: {e}")
                            try:
                                json_text_fixed = re.sub(r',(\s*[}\]])', r'\1', json_text)
                                parsed = json.loads(json_text_fixed)
                                if isinstance(parsed, dict):
                                    norm = _normalize_keys(parsed)
                                    if any(key in norm for key in expected_keys):
                                        log.debug(f"Successfully parsed JSON after fixing trailing commas")
                                        return norm
                            except Exception:
                                pass
                
                # Step 1b: Try to find JSON after markdown headers (common LLM pattern)
                # Look for JSON that starts after markdown headers like "## Market Insights"
                json_after_markdown = re.search(r'(?:^|\n)(?:##[^\n]*\n)*\s*(\{[\s\S]*\})', text, re.MULTILINE)
                if json_after_markdown:
                    json_text = json_after_markdown.group(1).strip()
                    try:
                        parsed = json.loads(json_text)
                        if isinstance(parsed, dict):
                            norm = _normalize_keys(parsed)
                            if any(key in norm for key in expected_keys):
                                log.debug(f"Successfully parsed JSON after markdown headers with keys: {list(norm.keys())}")
                                return norm
                    except json.JSONDecodeError:
                        pass
                
                # Step 2: Try direct parse (only accept if it has expected keys)
                try:
                    parsed = json.loads(text.strip())
                    if isinstance(parsed, dict):
                        norm = _normalize_keys(parsed)
                        if any(key in norm for key in expected_keys):
                            log.debug(f"Successfully parsed JSON directly with keys: {list(norm.keys())}")
                            return norm
                except json.JSONDecodeError:
                    pass
                
                # Step 3: Find the OUTERMOST balanced JSON object (not just the first one)
                # Look for the largest balanced object that contains the expected keys
                start_positions = []
                for i, char in enumerate(text):
                    if char == '{':
                        start_positions.append(i)
                
                # Try each starting position, but prioritize ones that might contain the full structure
                for start in reversed(start_positions):  # Start from the end to find outermost
                    depth = 0
                    in_str = False
                    esc = False
                    for i in range(start, len(text)):
                        ch = text[i]
                        if in_str:
                            if esc:
                                esc = False
                            elif ch == '\\':
                                esc = True
                            elif ch == '"':
                                in_str = False
                        else:
                            if ch == '"':
                                in_str = True
                            elif ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    candidate = text[start:i+1]
                                    try:
                                        parsed = json.loads(candidate)
                                        if isinstance(parsed, dict):
                                            norm = _normalize_keys(parsed)
                                            if any(key in norm for key in expected_keys):
                                                log.debug(f"Successfully parsed JSON from balanced scan with keys: {list(norm.keys())}")
                                                return norm
                                    except json.JSONDecodeError:
                                        pass
                                    break
                
                # Step 4: Fallback to existing _extract_json_from_response with increased max length
                result = _extract_json_from_response(text, max_response_length=50000)
                if result and isinstance(result, dict):
                    norm = _normalize_keys(result)
                    if any(key in norm for key in expected_keys):
                        log.debug(f"Successfully parsed JSON from fallback function with keys: {list(norm.keys())}")
                        return norm
                
                # Step 5: Last resort - try to extract partial JSON if response was truncated
                # Look for any JSON object that has at least one expected key (even if incomplete)
                log.warning(f"All JSON parsing methods failed. Text length: {len(text)}, preview: {text[:500]}")
                return {}
            
            # Convert LLM response to text first (handles various response formats)
            market_llm_response_text = _to_text(market_llm_response)
            log.debug(f"LLM response text length: {len(market_llm_response_text) if market_llm_response_text else 0}")
            if market_llm_response_text:
                log.debug(f"LLM response preview (first 500 chars): {market_llm_response_text[:500]}")
            
            market_insights = _extract_full_json_structure(market_llm_response_text)
            log.debug(f"Parsed market_insights, type: {type(market_insights)}")
            log.debug(f"market_insights keys: {list(market_insights.keys()) if isinstance(market_insights, dict) else 'N/A'}")
            
            if not isinstance(market_insights, dict):
                log.debug("market_insights is not dict, creating empty dict")
                processing_time = _calculate_processing_time(start_time)
                AgentLogger.log_error(log_context, f"market_insights is not a dict (type: {type(market_insights)}), resetting to empty dict", processing_time)
                # ✅ FIX: Log raw response for debugging truncation issues
                if market_llm_response_text:
                    response_preview = market_llm_response_text[:1000] if len(market_llm_response_text) > 1000 else market_llm_response_text
                    AgentLogger.log_warning(log_context, f"Raw LLM response preview (first 1000 chars): {response_preview}")
                    AgentLogger.log_warning(log_context, f"Raw LLM response length: {len(market_llm_response_text)} chars")
                market_insights = {}
            else:
                log.debug(f"market_insights keys: {list(market_insights.keys())}")
                # Check if market_insights dict is empty or missing expected fields
                expected_keys = ['market_insights', 'salary_trends', 'career_paths', 'skill_demand_analysis']
                if not market_insights or market_insights == {}:
                    processing_time = _calculate_processing_time(start_time)
                    AgentLogger.log_error(log_context, "⚠️ CRITICAL: market_insights dict is empty after parsing. All market fields will be empty.", processing_time)
                    # ✅ FIX: Log raw response for debugging truncation issues
                    if market_llm_response_text:
                        response_preview = market_llm_response_text[:2000] if len(market_llm_response_text) > 2000 else market_llm_response_text
                        AgentLogger.log_warning(log_context, f"Raw LLM response when empty (first 2000 chars): {response_preview}")
                        AgentLogger.log_warning(log_context, f"Raw LLM response length: {len(market_llm_response_text)} chars")
                        # Log last 500 chars to see if JSON was cut off
                        if len(market_llm_response_text) > 500:
                            AgentLogger.log_warning(log_context, f"Response ending (last 500 chars): {market_llm_response_text[-500:]}")
                elif not any(key in market_insights for key in expected_keys):
                    processing_time = _calculate_processing_time(start_time)
                    AgentLogger.log_warning(log_context, f"⚠️ WARNING: market_insights dict missing expected fields. Keys found: {list(market_insights.keys())}", processing_time)
                    # ✅ FIX: Log raw response and response length for debugging truncation issues
                    if market_llm_response_text:
                        response_length = len(market_llm_response_text)
                        response_preview = market_llm_response_text[:2000] if response_length > 2000 else market_llm_response_text
                        AgentLogger.log_warning(log_context, f"Raw LLM response length: {response_length} chars. Preview (first 2000 chars): {response_preview}")
                        # Check if response might be truncated
                        if response_length > 3000:  # If response is long but missing keys, might be truncated
                            AgentLogger.log_warning(log_context, f"⚠️ Response is {response_length} chars but missing expected keys - possible truncation issue")
                        # Log last 500 chars to see if JSON was cut off
                        if response_length > 500:
                            AgentLogger.log_warning(log_context, f"Response ending (last 500 chars): {market_llm_response_text[-500:]}")
            
            log.debug("Market insights processing completed successfully")
            
        except asyncio.TimeoutError as _timeout_err:
            log.debug(f"Market analysis timed out: {str(_timeout_err)}")
            processing_time = _calculate_processing_time(start_time)
            AgentLogger.log_error(log_context, "Market analysis timed out after 40 seconds", processing_time)
            await market_memory.record_attempt(
                'market_analysis', 'llm_timeout', False, 0.0, processing_time
            )
            return _create_error_response(
                "Market analysis failed: Model timeout after 40 seconds",
                processing_time
            )
        except Exception as e:
            log.debug(f"Exception during market analysis: {str(e)}")
            log.debug(f"Exception type: {type(e)}")
            import traceback
            traceback_str = traceback.format_exc()
            log.debug(f"Market analysis traceback: {traceback_str}")
            processing_time = _calculate_processing_time(start_time)
            AgentLogger.log_error(log_context, f"❌ EXCEPTION during market analysis LLM call: {str(e)}. Type: {type(e).__name__}", processing_time)
            AgentLogger.log_error(log_context, f"Traceback: {traceback_str[:500]}", processing_time)  # First 500 chars of traceback
            # Continue with empty market_insights instead of failing completely
            market_insights = {}
            AgentLogger.log_warning(log_context, "⚠️ Continuing with empty market_insights due to exception - all market fields will be empty")
        
        # Step 2: course_recommendations_data already obtained from parallel gather above
        # Step 3: Courses and materials from knowledge base are already validated
        log.debug("Courses and materials from knowledge base are pre-validated")
        # Ensure course_recommendations_data is a dictionary
        if isinstance(course_recommendations_data, str):
            log.debug("course_recommendations_data is string, attempting JSON parse")
            try:
                course_recommendations_data = json.loads(course_recommendations_data)
                log.debug("Successfully parsed course_recommendations_data as JSON")
            except (json.JSONDecodeError, TypeError):
                log.debug("Failed to parse course_recommendations_data as JSON")
                AgentLogger.log_warning(log_context, f"Failed to parse course_recommendations_data as JSON, using empty dict")
                course_recommendations_data = {}
        elif not isinstance(course_recommendations_data, dict):
            log.debug(f"course_recommendations_data is not dict or string, type: {type(course_recommendations_data)}")
            AgentLogger.log_warning(log_context, f"course_recommendations_data is not a dict or string, using empty dict")
            course_recommendations_data = {}
        
        raw_courses = course_recommendations_data.get('course_recommendations', [])
        # All materials (courses, books, papers, videos, tutorials) are now in course_recommendations
        log.debug(f"Got {len(raw_courses) if isinstance(raw_courses, list) else 0} total recommendations from knowledge base")
        
        # Knowledge base recommendations are already validated, so use them directly
        final_valid_courses = raw_courses if isinstance(raw_courses, list) else []
        AgentLogger.log_info(log_context, f"✅ Using {len(final_valid_courses)} validated recommendations from knowledge base")

        # No search supplement needed since we're not using URLs
        
        # Step 5: Combine market insights with validated course recommendations
        log.debug("Building recommendations dictionary")
        
        # Map courses from knowledge base to output format
        # Knowledge base courses use 'title' and 'provider', but output needs 'course' and 'platform'
        final_course_recommendations = _map_courses_for_output(final_valid_courses) if final_valid_courses else []
        AgentLogger.log_info(log_context, f"✅ Mapped {len(final_course_recommendations)} courses to output format")
        
        # ✅ FIX: Extract from parsed market_insights dict structure
        # The parsed dict should have keys: market_insights, salary_trends, career_paths, skill_demand_analysis
        parsed_market_insights = market_insights.get('market_insights', []) if isinstance(market_insights, dict) else []
        parsed_salary_trends = market_insights.get('salary_trends', {}) if isinstance(market_insights, dict) else {}
        parsed_career_paths = market_insights.get('career_paths', []) if isinstance(market_insights, dict) else []
        parsed_skill_demand = market_insights.get('skill_demand_analysis', {}) if isinstance(market_insights, dict) else {}
        
        # ✅ FIX: Use fallback data from raw_skill_gap_analysis_output if parsing failed
        # This ensures we don't lose data from previous runs or cached results in re-run scenarios
        fallback_market_insights = raw_skill_gap_analysis_output.get('market_insights', []) if isinstance(raw_skill_gap_analysis_output, dict) else []
        fallback_salary_trends = raw_skill_gap_analysis_output.get('salary_trends', {}) if isinstance(raw_skill_gap_analysis_output, dict) else {}
        fallback_career_paths = raw_skill_gap_analysis_output.get('career_paths', []) if isinstance(raw_skill_gap_analysis_output, dict) else []
        fallback_skill_demand = raw_skill_gap_analysis_output.get('skill_demand_analysis', {}) if isinstance(raw_skill_gap_analysis_output, dict) else {}
        
        # Use parsed data if available and non-empty, otherwise use fallback
        # This handles re-run scenarios where previous data exists but new parsing might fail
        final_market_insights = parsed_market_insights if (parsed_market_insights and len(parsed_market_insights) > 0) else fallback_market_insights
        final_salary_trends = parsed_salary_trends if (parsed_salary_trends and isinstance(parsed_salary_trends, dict) and len(parsed_salary_trends) > 0) else fallback_salary_trends
        final_career_paths = parsed_career_paths if (parsed_career_paths and len(parsed_career_paths) > 0) else fallback_career_paths
        final_skill_demand = parsed_skill_demand if (parsed_skill_demand and isinstance(parsed_skill_demand, dict) and len(parsed_skill_demand) > 0) else fallback_skill_demand
        
        # When both parsed and fallback salary_trends are empty, synthesize so users always get something
        # Prefer assessment-based context when available; otherwise use a minimal placeholder
        if not final_salary_trends:
            if salary_modifiers:
                mod_reasons = salary_modifiers.get("reasons", [])
                adj_pct = salary_modifiers.get("total_adjustment_percent", 0)
                final_salary_trends = {
                    "current_level": "Based on resume and experience" + (f" (+{adj_pct:.1f}% assessment adjustment)" if adj_pct else ""),
                    "current_level_rationale": mod_reasons if mod_reasons else ["No assessment history. Salary based on resume and experience only."],
                    "assessment_modifiers": salary_modifiers,
                    "note": "Market insights were unavailable. Complete salary trends will appear when market analysis succeeds."
                }
                AgentLogger.log_warning(log_context, "⚠️ Synthesized salary_trends from assessment modifiers (market insights LLM parsing failed)")
            else:
                final_salary_trends = {
                    "current_level": "Based on resume and experience",
                    "current_level_rationale": ["Market analysis temporarily unavailable. Salary insights will appear when the market analysis completes successfully."],
                    "note": "Market insights were unavailable. Complete salary trends will appear when market analysis succeeds."
                }
                AgentLogger.log_warning(log_context, "⚠️ Synthesized minimal salary_trends (market insights LLM parsing failed, no assessment data)")
        
        # Log if we're using fallback data (important for debugging re-run scenarios)
        if not parsed_market_insights and fallback_market_insights:
            AgentLogger.log_warning(log_context, f"⚠️ Using fallback market_insights from raw_skill_gap_analysis_output ({len(fallback_market_insights)} insights) - parsing may have failed or this is a re-run")
        if not parsed_salary_trends and fallback_salary_trends:
            AgentLogger.log_warning(log_context, f"⚠️ Using fallback salary_trends from raw_skill_gap_analysis_output - parsing may have failed or this is a re-run")
        
        recommendations = {
            'market_insights': final_market_insights,
            'course_recommendations': final_course_recommendations,
            'salary_trends': final_salary_trends,
            'career_paths': final_career_paths,
            'skill_demand_analysis': final_skill_demand,
            # Expose performance context so downstream consumers can surface reasoning
            'performance_context': performance_context,
        }
        log.debug(f"Built recommendations, keys: {list(recommendations.keys())}")
        
        # Diagnostic check: Warn if critical fields are empty
        empty_fields = []
        if not recommendations.get('market_insights'):
            empty_fields.append('market_insights')
        if not recommendations.get('salary_trends'):
            empty_fields.append('salary_trends')
        if not recommendations.get('career_paths'):
            empty_fields.append('career_paths')
        if not recommendations.get('skill_demand_analysis'):
            empty_fields.append('skill_demand_analysis')
        if not recommendations.get('course_recommendations'):
            empty_fields.append('course_recommendations')
        
        if empty_fields:
            AgentLogger.log_warning(log_context, f"⚠️ WARNING: The following fields are empty: {', '.join(empty_fields)}. This may indicate LLM parsing failure or validation issues.")
            if 'market_insights' in empty_fields or 'salary_trends' in empty_fields or 'career_paths' in empty_fields or 'skill_demand_analysis' in empty_fields:
                AgentLogger.log_warning(log_context, f"⚠️ Market insights dict keys: {list(market_insights.keys()) if isinstance(market_insights, dict) else 'Not a dict'}")
            if 'course_recommendations' in empty_fields:
                AgentLogger.log_warning(log_context, f"⚠️ Course recommendations: {len(final_valid_courses)} courses after validation")
        
        processing_time = _calculate_processing_time(start_time)
        timing_breakdown["total"] = processing_time
        
        # Log timing breakdown
        log.info(
            f"⏱️ MARKET_AND_COURSE_RECOMMENDER timing breakdown: "
            f"prompt_gen={timing_breakdown['prompt_generation']*1000:.1f}ms, "
            f"market_llm={timing_breakdown['market_insights_llm']*1000:.1f}ms, "
            f"course_validation={timing_breakdown['course_validation']*1000:.1f}ms, "
            f"total={timing_breakdown['total']*1000:.1f}ms"
        )
        
        # Record successful analysis
        await market_memory.record_attempt(
            'market_analysis', 'llm', True, 0.8, processing_time
        )
        
        # Log success with enhanced course flow details
        log_agent_completion(log_context, {
            "success": True,
            "market_insights_count": len(recommendations.get('market_insights', [])),
            "course_recommendations_count": len(final_valid_courses),
            "enhanced_course_flow": True,
            "validated_courses": len(valid_courses),
            "final_valid_courses": len(final_valid_courses)
        }, "llm", processing_time)
        
        # Filter out career advice data and update with market insights only
        filtered_skill_gap = _filter_career_advice_data(raw_skill_gap_analysis_output)
        
        # Add market insights and other data (excluding course_recommendations for now)
        log.debug("Building market_data dictionary")
        # ✅ FIX: Only update with non-empty data to preserve existing data from previous runs
        market_data = {}
        
        # Only add market_insights if we have new data
        if recommendations.get('market_insights'):
            market_data['market_insights'] = recommendations.get('market_insights', [])
        elif filtered_skill_gap.get('market_insights'):
            # Preserve existing data if new parsing failed
            market_data['market_insights'] = filtered_skill_gap.get('market_insights', [])
            AgentLogger.log_warning(log_context, f"⚠️ Preserving existing market_insights ({len(market_data['market_insights'])} insights) as new parsing returned empty")
        
        # Only add salary_trends if we have new data
        if recommendations.get('salary_trends'):
            market_data['salary_trends'] = recommendations.get('salary_trends', {})
        elif filtered_skill_gap.get('salary_trends'):
            # Preserve existing data if new parsing failed
            market_data['salary_trends'] = filtered_skill_gap.get('salary_trends', {})
            AgentLogger.log_warning(log_context, f"⚠️ Preserving existing salary_trends as new parsing returned empty")
        
        # Only add career_paths if we have new data
        if recommendations.get('career_paths'):
            market_data['career_paths'] = recommendations.get('career_paths', [])
        elif filtered_skill_gap.get('career_paths'):
            # Preserve existing data if new parsing failed
            market_data['career_paths'] = filtered_skill_gap.get('career_paths', [])
        
        # Only add skill_demand_analysis if we have new data
        if recommendations.get('skill_demand_analysis'):
            market_data['skill_demand_analysis'] = recommendations.get('skill_demand_analysis', {})
        elif filtered_skill_gap.get('skill_demand_analysis'):
            # Preserve existing data if new parsing failed
            market_data['skill_demand_analysis'] = filtered_skill_gap.get('skill_demand_analysis', {})
        
        log.debug(f"Built market_data, keys: {list(market_data.keys())}")
        log.debug("Updating filtered_skill_gap with market_data")
        filtered_skill_gap.update(market_data)
        log.debug("Updated filtered_skill_gap successfully")
        
        # Format course recommendations for output (no URL validation)
        # All materials (courses, books, papers, videos, tutorials) are now in course_recommendations
        log.debug("Formatting course recommendations for output")
        formatted_courses = _map_courses_for_output(recommendations.get('course_recommendations', []))
        log.debug(f"Got {len(formatted_courses) if isinstance(formatted_courses, list) else 0} total recommendations")
        
        # No URL validation - use formatted courses directly
        final_valid_courses = formatted_courses
        log.debug(f"Final formatting complete, {len(final_valid_courses)} recommendations ready")
        
        # Add course recommendations to the filtered data (includes all materials)
        filtered_skill_gap['course_recommendations'] = final_valid_courses
        
        # Safety check: ensure final_valid_courses is defined
        log.debug("Performing safety check for final_valid_courses")
        if 'final_valid_courses' not in locals():
            log.debug("final_valid_courses not in locals, creating empty list")
            final_valid_courses = []
        
        # Calculate final processing time
        processing_time = _calculate_processing_time(start_time)
        
        # Update state
        log.debug("Updating state with results")
        state["raw_skill_gap_analysis_output"] = filtered_skill_gap
        state["analysis_status"] = "success"
        state["analysis_method"] = "llm"
        state["confidence_score"] = 0.8
        state["processing_time"] = processing_time
        state["request_id"] = request_id
        log.debug("State updated successfully")
        
        # Store market and course recommender data in chat_sessions per-UID doc - MOVED TO BACKGROUND (non-blocking)
        uid = state.get("uid")
        if uid and (recommendations.get("market_insights") or final_valid_courses):
            # Fire-and-forget: Don't block response on storage operations
            async def _store_data_background():
                try:
                    storage_start = time.time()
                    log.info(f"🔄 MARKET_AND_COURSE_RECOMMENDER: Starting background storage for UID={uid}")
                    
                    # In evaluation flow (assessment_results present), REPLACE instead of merge
                    is_evaluation_flow = bool(state.get("assessment_results"))
                    existing = await run_blocking_io(get_gap_doc, uid) or {}
                    
                    if is_evaluation_flow:
                        # Evaluation flow: MERGE instead of REPLACE to preserve assessment history
                        # This ensures salary modifiers are calculated from aggregated history
                        merged = {**existing}
                        merged["market_and_course_recommender"] = {
                            "raw_skill_gap_analysis_output": filtered_skill_gap,
                            "market_insights": recommendations.get("market_insights", []),
                            "course_recommendations": final_valid_courses,
                            "analysis_status": "success",
                            "analysis_method": "enhanced_llm",
                            "confidence_score": 0.8,
                            "processing_time": processing_time,
                            "request_id": request_id,
                            "enhanced_course_flow": True,
                            "validated_courses_count": len(valid_courses),
                            "evaluation_flow": True,
                            "assessment_history_aggregated": True,
                            "salary_modifiers_applied": bool(salary_modifiers)
                        }
                        # Preserve career_advisor if present
                        if "skill_and_career_advisor" in existing:
                            merged["skill_and_career_advisor"] = existing["skill_and_career_advisor"]
                        log.info(f"✅ Evaluation flow: Merged market_and_course_recommender data with assessment history for UID={uid}")
                    else:
                        # Normal flow: Merge with existing
                        merged = {**existing}
                        merged["market_and_course_recommender"] = {
                            "raw_skill_gap_analysis_output": filtered_skill_gap,
                            "market_insights": recommendations.get("market_insights", []),
                            "course_recommendations": final_valid_courses,
                            "analysis_status": "success",
                            "analysis_method": "enhanced_llm",
                            "confidence_score": 0.8,
                            "processing_time": processing_time,
                            "request_id": request_id,
                            "enhanced_course_flow": True,
                            "validated_courses_count": len(valid_courses)
                        }
                    
                    await run_blocking_io(
                        upsert_gap_doc,
                        uid,
                        merged,
                        metadata={
                            "agent": "market_and_course_recommender",
                            "uid": uid,
                            "status": "market_and_course_recommender_complete",
                            "evaluation_flow": is_evaluation_flow,
                            "method": "enhanced_llm"
                        }
                    )
                    storage_time = time.time() - storage_start
                    log.info(f"✅ MARKET_AND_COURSE_RECOMMENDER: Background storage completed in {storage_time*1000:.1f}ms for UID={uid}")
                except Exception as e:
                    log.error(f"❌ MARKET_AND_COURSE_RECOMMENDER: Background storage failed for UID={uid}: {e}", exc_info=True)
            
            # Fire-and-forget: Start background task without awaiting
            asyncio.create_task(_store_data_background())
            log.info(f"🚀 MARKET_AND_COURSE_RECOMMENDER: Started background storage task for UID={uid} (non-blocking)")
        
        # Return both formats:
        # 1. Strict JSON format for callbacks
        # 2. Original fields for internal pipeline use
        log.debug("Building final return dictionary")
        return_dict = {
            # Strict JSON format for callbacks
            "status": "completed",
            "node": "market_and_course_recommender",
            "output": {
                "raw_skill_gap_analysis_output": filtered_skill_gap,
                "market_insights": recommendations.get("market_insights", []),
                "salary_trends": recommendations.get("salary_trends", {}),
                "career_paths": recommendations.get("career_paths", []),
                "skill_demand_analysis": recommendations.get("skill_demand_analysis", {}),
                "course_recommendations": final_valid_courses
            },
            
            # Original fields for internal pipeline use
            "recommendations": recommendations,
            "raw_skill_gap_analysis_output": filtered_skill_gap,
            "analysis_status": "success",
            "analysis_method": "enhanced_llm",
            "confidence_score": 0.8,
            "processing_time": processing_time,
            "request_id": request_id,
            "enhanced_course_flow": True,
            "validated_courses_count": len(valid_courses)
        }
        log.debug("Built return dictionary successfully")
        log.debug("Returning successful result")
        cache_response("market_and_course_recommender", cache_input, return_dict)
        return return_dict
        
    except Exception as e:
        log.debug(f"Exception caught in market_and_course_recommender: {str(e)}")
        log.debug(f"Exception type: {type(e)}")
        import traceback
        log.debug(f"Traceback: {traceback.format_exc()}")
        processing_time = _calculate_processing_time(start_time)
        AgentLogger.log_error(log_context, f"Market and course recommendation failed: {str(e)}", processing_time)
        
        # Record failure
        await market_memory.record_attempt(
            'market_analysis', 'llm', False, 0.0, processing_time
        )
        
        # Return error response with empty course recommendations
        return {
            "status": "error",
            "node": "market_and_course_recommender",
            "output": {
                "raw_skill_gap_analysis_output": raw_skill_gap_analysis_output,
                "market_insights": [],
                "course_recommendations": []
            },
            "error": f"Market and course recommendation failed: {str(e)}",
            "processing_time": processing_time,
            "request_id": request_id
        }

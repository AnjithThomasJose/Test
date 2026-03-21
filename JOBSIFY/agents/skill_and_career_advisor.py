import json
import re
import asyncio
import logging
import time
import uuid
import os
from typing import Dict, Any, List, Optional, Tuple, Literal
from models.llm_invoker import invoke_llm, invoke_structured_llm
from core.model_registry import TaskType
from langsmith.run_helpers import traceable
# Import the base prompt generator under an alias so we can wrap it with
# performance-context injection while keeping the public name identical for tests.
from agents.prompt_generator import generate_skill_and_career_advice_prompt as _base_generate_skill_and_career_advice_prompt
from chroma import fetch_structured_resume, get_chat_session, update_chat_session
from chroma import get_gap_doc, upsert_gap_doc  # NEW per-UID gap helpers
from core.utils import (
    _mask, _sanitize_text_for_llm, _validate_state_inputs, _to_text, _clean_json_text, 
    _scan_balanced_json, _safe_json_loads, _extract_json_from_response, _create_error_response,
    _generate_request_id, _calculate_processing_time, create_agent_state, filter_resume_data_for_agent,
    run_blocking_io
)
from core.config import get_agent_config
from core.security import validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm, PII_PATTERNS, INJECTION_FILTERS
from core.memory import BaseAgentMemory, get_agent_memory
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion

log = logging.getLogger(__name__)

# Remove all local config constants and use centralized config
config = get_agent_config("skill_and_career_advisor")
TIMEOUT_SECONDS = config.timeout_seconds
MAX_PROMPT_CHARS = config.max_prompt_chars
MAX_RESPONSE_LENGTH = config.max_response_length
MAX_INTERESTS_COUNT = config.max_interests_count
MAX_SKILLS_COUNT = config.max_skills_count
LLM_MODEL = config.llm_model

# Tenant isolation patterns
TENANT_ID_RX = re.compile(r"^[a-zA-Z0-9_\-]{8,64}$")

# Type-safe analysis methods (LLM-only approach)
AnalysisMethod = Literal["llm", "error", "llm_timeout"]

# --- Agentic AI Constants ---
CONFIDENCE_THRESHOLD = 0.70   # Minimum confidence for skill analysis
ADAPTATION_WINDOW = 50        # Number of recent analyses to consider     # Minimum skill categories required
MIN_CAREER_PATHS = 2          # Minimum career paths required

# PII patterns for redaction (precompiled for performance)
PII_PATTERNS = PII_PATTERNS

INJECTION_FILTERS = INJECTION_FILTERS

# Custom memory class for skill and career advisor (extends base memory)
class SkillCareerAdvisorMemory(BaseAgentMemory):
    def __init__(self, tenant_id: str = "default"):
        super().__init__(tenant_id, config.adaptation_window)
        # Add any skill advisor specific fields here if needed

# Local wrapper to inject performance context and enhanced role fit into the prompt before returning.
# This ensures tests that capture the generated prompt see the performance block.
async def generate_skill_and_career_advice_prompt(
    structured_resume: Dict[str, Any],
    user_interests: List[Dict[str, Any]],
    assessment_results: Optional[Dict[str, Any]],
    report: Optional[Dict[str, Any]],
    *,
    performance_context: Optional[Dict[str, Any]] = None,
    uid: Optional[str] = None,
    enhanced_role_fit: Optional[List[Dict[str, Any]]] = None,
):
    base_prompt = await _base_generate_skill_and_career_advice_prompt(
        structured_resume, user_interests, assessment_results, report,
        uid=uid, enhanced_role_fit=enhanced_role_fit
    )

    # If we have a performance signal, append it directly so captured prompts include it.
    if performance_context and performance_context.get("performance_level"):
        base_prompt += f"""

Assessment Performance Context: {json.dumps(performance_context, default=str)}

Assessment Performance Insights:
- Topic Assessed: {performance_context.get('last_topic')}
- Score: {performance_context.get('score')}
- Performance Level: {performance_context.get('performance_level')}
- Strong Sections: {performance_context.get('strong_sections')}
- Weak Sections: {performance_context.get('weak_sections')}

Explicitly tailor career paths, alternates, and recommendations to this performance profile.
"""

    return base_prompt

# Use centralized memory management
async def get_skill_career_advisor_memory(tenant_id: str = "default") -> SkillCareerAdvisorMemory:
    """Get or create tenant-scoped skill and career advisor memory."""
    return await get_agent_memory("skill_and_career_advisor", tenant_id, SkillCareerAdvisorMemory)

def _normalize_insight_alignment(value: Any) -> str:
    """Normalize alternate-path alignment to the supported API values."""
    normalized = str(value or "").strip().lower()
    if "insight" in normalized or "interest" in normalized:
        return "insight based"
    return "similar to current"

def _text_to_bullets(text: str) -> str:
    """Convert paragraph text to bullet points if needed."""
    if not text or not isinstance(text, str):
        return text or ""
    s = text.strip()
    if not s:
        return s
    if (s.startswith("* ") or s.startswith("- ")) and (" * " in s or "\n* " in s or " - " in s or "\n- " in s):
        normalized = re.sub(r"\s*\n\s*", " ", s).strip()
        normalized = re.sub(r"- ", "* ", normalized)
        return normalized
    parts = re.split(r"\.\s+(?=[A-Z])", s)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return s
    bullets = []
    for p in parts:
        if p and not p.endswith("."):
            p = p + "."
        if p:
            bullets.append("* " + p)
    return " ".join(bullets) if bullets else s

def _normalize_alternate_career_paths(paths: Any) -> List[Dict[str, Any]]:
    """Normalize alternate career paths to the API shape expected downstream."""
    normalized_paths: List[Dict[str, Any]] = []
    if not isinstance(paths, list):
        return normalized_paths

    for path in paths:
        if not isinstance(path, dict):
            continue
        normalized_path = dict(path)
        pct = normalized_path.get("match_percentage", 0)
        normalized_path["match_percentage"] = max(
            0, min(100, int(pct) if isinstance(pct, (int, float)) else 0)
        )
        if "lacking_skills" not in normalized_path or not isinstance(normalized_path.get("lacking_skills"), list):
            normalized_path["lacking_skills"] = []
        normalized_path["insight_alignment"] = _normalize_insight_alignment(
            normalized_path.get("insight_alignment", normalized_path.get("interest_alignment"))
        )
        normalized_path.pop("interest_alignment", None)
        for key in ("description", "rationale"):
            val = normalized_path.get(key)
            if isinstance(val, str):
                normalized_path[key] = _text_to_bullets(val)
        normalized_paths.append(normalized_path)

    return normalized_paths

def _build_alternate_career_paths_retry_prompt(base_prompt: str) -> str:
    """Create a focused retry prompt for alternate career paths only."""
    return f"""{base_prompt}

Previous attempt returned no alternate career paths.

Retry task:
- Return ONLY a JSON object with the key `alternate_career_paths`
- Generate at least 1 and at most 3 alternate career paths
- Every item must include: title, description, rationale, match_percentage, lacking_skills, insight_alignment
- Keep using the career-specific insight_alignment rule already defined above
- Do not return any other top-level keys
"""

async def _retry_missing_alternate_career_paths(prompt: str) -> List[Dict[str, Any]]:
    """Retry only alternate career path generation when the main analysis returned none."""
    retry_prompt = _build_alternate_career_paths_retry_prompt(prompt)
    try:
        llm_response = await invoke_llm(
            prompt=retry_prompt,
            task_type="skill_analysis",
            agent_name="skill_and_career_advisor",
            response_mime_type="application/json",
            preferred_model="gemini-2.5-flash",
            max_output_tokens=900,
        )
        retry_analysis = _extract_json_from_response(_to_text(llm_response), MAX_RESPONSE_LENGTH)
        if not isinstance(retry_analysis, dict):
            return []
        return _normalize_alternate_career_paths(retry_analysis.get("alternate_career_paths"))
    except Exception as retry_err:
        log.warning(f"Alternate career path retry failed: {retry_err}")
        return []


def _calculate_confidence_score(analysis: Dict[str, Any], method: str) -> float:
    """Calculate confidence score for skill analysis."""
    if not analysis:
        return 0.0
    
    score = 0.3  # Base score
    
    
    # Career paths scoring
    career_paths = analysis.get('career_paths', [])
    if isinstance(career_paths, list) and len(career_paths) >= MIN_CAREER_PATHS:
        score += 0.2
    
    # Quality indicators
    if analysis.get('skill_proficiency_assessment'):
        score += 0.1
    if analysis.get('missing_skills'):
        score += 0.1
    if analysis.get('improvement_recommendations'):
        score += 0.1
    
    # Method-specific adjustments
    if method == 'llm':
        score += 0.05  # Slight bonus for LLM comprehensiveness
    elif method == 'hybrid':
        score += 0.1   # Bonus for hybrid approach
    
    return max(0.0, min(score, 1.0))

def _assess_analysis_quality(analysis: Dict[str, Any]) -> float:
    """Assess quality of skill analysis."""
    if not analysis:
        return 0.0
    
    quality_score = 0.0
    
    # Structure validation
    expected_keys = ['career_paths', 'skill_proficiency_assessment', 'missing_skills']
    present_keys = sum(1 for key in expected_keys if key in analysis and analysis[key])
    if present_keys > 0:
        quality_score += (present_keys / len(expected_keys)) * 0.5
    
    # Content quality assessment
    
    
    career_paths = analysis.get('career_paths', [])
    if isinstance(career_paths, list):
        for path in career_paths:
            if isinstance(path, dict) and path.get('title') and path.get('description'):
                quality_score += 0.05
    
    return min(quality_score, 1.0)

# Deterministic function removed - using LLM-only approach

@traceable(name="skill_and_career_advisor_agent")
async def skill_and_career_advisor_agent(state: Dict[str, Any], tenant_id: str = "default_tenant") -> Dict[str, Any]:
    """
    Enhanced skill and career advisor with LLM-only approach.
    Note: When invoked from the graph, only state is passed; tenant_id is taken from state.
    """
    # Graph invokes with (state) only; use tenant_id from state when present
    tenant_id = state.get("tenant_id") or tenant_id
    # Use centralized logging
    log_context = create_log_context("skill_and_career_advisor", tenant_id)
    start_time = log_context["start_time"]
    request_id = log_context["request_id"]
    
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            f"--- Entering Skill and Career Advisor Agent for tenant: {tenant_id} ---"
        )
        log.debug(
            f"🔍 SKILL_AND_CAREER_ADVISOR DEBUG: State keys: {list(state.keys())}"
        )
        log.debug(
            f"🔍 SKILL_AND_CAREER_ADVISOR DEBUG: UID: {state.get('uid', 'N/A')}"
        )
        log.debug(
            f"🔍 SKILL_AND_CAREER_ADVISOR DEBUG: Session ID: "
            f"{state.get('session_id', 'N/A')}"
        )
    
    # Get tenant-scoped memory
    skill_memory = await get_skill_career_advisor_memory(tenant_id)
    
    # ✅ PERFORMANCE: Use optimized inputs - only send required data
    # This truncates structured_resume to only fields needed by career advisor
    from core.utils import create_optimized_career_advisor_inputs
    
    optimized_inputs = create_optimized_career_advisor_inputs(state)
    structured_resume = optimized_inputs.get("structured_resume", {})
    user_interests = optimized_inputs.get("user_interests", [])
    current_gap_analysis = optimized_inputs.get("raw_skill_gap_analysis_output", {})
    assessment_results = optimized_inputs.get("assessment_results")
    report = optimized_inputs.get("report")
    session_id = optimized_inputs.get("session_id")
    
    # Check if this is a 2nd call (user_interests from 2nd call should be prioritized)
    is_second_call = state.get("is_second_call", False)
    if is_second_call and user_interests:
        log.info(
            "✅ 2ND CALL: Using user_interests from 2nd call "
            f"({len(user_interests) if isinstance(user_interests, list) else 'N/A'} "
            "items) as context"
        )
        log.info(
            "2nd call detected - using user_interests from 2nd call as context for "
            "career advisor"
        )
    
    # Section 8 Issue 1: agent-level cache for same resume + interests + assessment
    from core.utils import get_cached_response, cache_response
    cache_input = {
        "structured_resume": structured_resume,
        "user_interests": user_interests,
        "assessment_results": assessment_results,
        "report": report,
    }
    cached_result = get_cached_response("skill_and_career_advisor", cache_input)
    if cached_result:
        log.info("Cache hit for skill_and_career_advisor - returning cached result")
        return cached_result

    # Defensive check: if resume is empty, attempt to hydrate from session
    if session_id and (not structured_resume or not structured_resume.get("skills")):
        log.warning(
            "SKILL_AND_CAREER_ADVISOR: structured_resume is missing or empty. "
            "Attempting to hydrate from session."
        )
        from chroma import fetch_structured_resume
        # Run blocking I/O in thread pool to avoid blocking event loop
        hydrated_resume = await run_blocking_io(
            fetch_structured_resume, state.get("uid"), current_session_id=session_id
        )
        if hydrated_resume:
            structured_resume = hydrated_resume
            log.info(
                "✅ SKILL_AND_CAREER_ADVISOR: Successfully hydrated structured_resume "
                "from session."
            )

    # OPTIMIZATION: Only hydrate from session if critical data is missing
    # Skip if we already have structured_resume and user_interests (most common case)
    needs_hydration = (not structured_resume or (not user_interests and not is_second_call)) and session_id
    if needs_hydration:
        try:
            # Run blocking I/O in thread pool to avoid blocking event loop
            session_data = await run_blocking_io(get_chat_session, session_id)
            # Note: structured_resume is no longer stored in chat_sessions by resume_assembler
            # Get it from the state instead
            if not structured_resume:
                # Try to get from state first, then fallback to session
                structured_resume = state.get("structured_resume", session_data.get("structured_resume", {}))
            # Only get user_interests from session if not already present in state (2nd call has them in state)
            if not user_interests and not is_second_call:
                interest_data = session_data.get("interest_filler", {})
                user_interests = interest_data.get("user_interests", [])
            elif is_second_call:
                if log.isEnabledFor(logging.DEBUG):
                    log.debug(
                        "✅ 2ND CALL: Keeping user_interests from state (2nd call), "
                        "not overriding from session"
                    )
            if not current_gap_analysis:
                current_gap_analysis = (
                    session_data.get("career_advisor", {}).get("raw_skill_gap_analysis_output", {})
                    or session_data.get("raw_skill_gap_analysis_output", {})
                )
            if not assessment_results:
                assessment_results = session_data.get("assessment_evaluator", {}).get("assessment_results")
            if not report:
                report = session_data.get("report_generator", {}).get("report")
        except Exception as e:
            if log.isEnabledFor(logging.WARNING):
                log.warning(f"⚠️ SKILL_AND_CAREER_ADVISOR: Error getting data from session: {e}")
            pass
    
    # If still nothing, proceed using assessment/report only (evaluation flow)
    if not isinstance(current_gap_analysis, dict):
        current_gap_analysis = {}
    
    # Debug logging for data availability (only if debug enabled)
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            "🔍 SKILL_AND_CAREER_ADVISOR DEBUG: structured_resume available: "
            f"{'✅ YES' if structured_resume else '❌ NO'}"
        )
        log.debug(
            "🔍 SKILL_AND_CAREER_ADVISOR DEBUG: user_interests available: "
            f"{'✅ YES' if user_interests else '❌ NO'}"
        )
        log.debug(
            "🔍 SKILL_AND_CAREER_ADVISOR DEBUG: user_interests count: "
            f"{len(user_interests) if isinstance(user_interests, list) else 'N/A'}"
        )
        log.debug(
            "🔍 SKILL_AND_CAREER_ADVISOR DEBUG: current_gap_analysis available: "
            f"{'✅ YES' if current_gap_analysis else '❌ NO'}"
        )
        log.debug(
            "🔍 SKILL_AND_CAREER_ADVISOR DEBUG: assessment_results available: "
            f"{'✅ YES' if assessment_results else '❌ NO'}"
        )
        log.debug(
            "🔍 SKILL_AND_CAREER_ADVISOR DEBUG: report available: "
            f"{'✅ YES' if report else '❌ NO'}"
        )
    
    # Security validation
    if len(str(structured_resume)) > MAX_PROMPT_CHARS:
        AgentLogger.log_warning(log_context, f"Structured resume too large, truncating")
        structured_resume = str(structured_resume)[:MAX_PROMPT_CHARS]
    
    # Normalize assessment_results to a dict when possible
    if assessment_results and not isinstance(assessment_results, dict):
        try:
            if isinstance(assessment_results, str):
                assessment_results = json.loads(assessment_results)
        except (json.JSONDecodeError, TypeError):
            log.warning("SKILL_AND_CAREER_ADVISOR: Failed to parse assessment_results as JSON, skipping normalization")
            assessment_results = None
    
    # Extract performance-aware assessment context
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
        
        # Derive performance_level
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
    
    # Log if assessment results are found
    if assessment_results:
        log.info(f"🔍 SKILL_AND_CAREER_ADVISOR: Assessment results found, integrating into analysis")
        log.info(f"🔍 SKILL_AND_CAREER_ADVISOR: Assessment score: {assessment_results.get('total_score', 'N/A')}")
    
    # Granular timing breakdown for performance monitoring
    timing_breakdown = {
        "prompt_generation": 0.0,
        "llm_call": 0.0,
        "result_processing": 0.0,
        "total": 0.0
    }
    
    try:
        # Generate prompt with Gemini summary (NO TRUNCATION)
        prompt_start = time.time()
        uid = state.get("uid")
        enhanced_role_fit = state.get("enhanced_role_fit") or []
        prompt = await generate_skill_and_career_advice_prompt(
            structured_resume,
            user_interests,
            assessment_results,
            report,
            performance_context=performance_context,
            uid=uid,
            enhanced_role_fit=enhanced_role_fit if isinstance(enhanced_role_fit, list) else [],
        )
        timing_breakdown["prompt_generation"] = time.time() - prompt_start
        
        # Validate prompt size - reduce for faster processing (Section 8 Issue 3: centralized truncation)
        max_prompt_size = min(MAX_PROMPT_CHARS, 6000)  # Further reduced for faster processing
        if len(prompt) > max_prompt_size:
            from core.prompt_utils import truncate_prompt
            AgentLogger.log_warning(log_context, f"Prompt too large, truncating from {len(prompt)} to {max_prompt_size}")
            prompt = truncate_prompt(prompt, max_prompt_size, strategy="sentence")
        
        if log.isEnabledFor(logging.DEBUG):
            log.debug("🤖 Calling LLM for skill and career advice...")
        
        # Use structured output for faster processing
        try:
            from pydantic import BaseModel, Field
            from typing import List as _List, Optional as _Optional
            from pydantic import field_validator
            
            # Helper functions for validation (Issue 5.3)
            def _sanitize_str(v, max_len=500, default=""):
                if not v or not isinstance(v, str):
                    return default
                return v.strip()[:max_len]
            
            def _sanitize_str_list(v, max_items=10, max_item_len=100):
                if not isinstance(v, list):
                    return []
                return [item.strip()[:max_item_len] for item in v[:max_items] if item and isinstance(item, str) and item.strip()]

            class CareerPath(BaseModel):
                title: str = Field(default="Career Path")
                description: str = Field(default="")
                required_skills: _List[str] = Field(default_factory=list)
                
                @field_validator('title', mode='before')
                @classmethod
                def validate_title(cls, v):
                    return _sanitize_str(v, max_len=100, default="Career Path")
                
                @field_validator('description', mode='before')
                @classmethod
                def validate_description(cls, v):
                    return _sanitize_str(v, max_len=500, default="")
                
                @field_validator('required_skills', mode='before')
                @classmethod
                def validate_skills(cls, v):
                    return _sanitize_str_list(v, max_items=15, max_item_len=50)

            class AlternateCareerPath(BaseModel):
                title: str = Field(default="Alternate Path")
                description: str = Field(default="", description="What this role involves, as 2-4 bullet points")
                rationale: str = Field(default="", description="Why this path fits the candidate, as 2-4 bullet points")
                match_percentage: int = Field(default=0, ge=0, le=100, description="How well this path fits the candidate (0-100)")
                lacking_skills: _List[str] = Field(default_factory=list, description="Skills the candidate needs to attain for this path")
                insight_alignment: str = Field(
                    default="similar to current",
                    description='Whether this alternate path is "insight based" or "similar to current"',
                )
                
                @field_validator('title', mode='before')
                @classmethod
                def validate_title(cls, v):
                    return _sanitize_str(v, max_len=100, default="Alternate Path")
                
                @field_validator('description', 'rationale', mode='before')
                @classmethod
                def validate_text(cls, v):
                    return _sanitize_str(v, max_len=800, default="")
                
                @field_validator('match_percentage', mode='before')
                @classmethod
                def clamp_percentage(cls, v):
                    if isinstance(v, (int, float)):
                        return int(max(0, min(100, v)))
                    return 0
                
                @field_validator('lacking_skills', mode='before')
                @classmethod
                def validate_skills(cls, v):
                    return _sanitize_str_list(v, max_items=10, max_item_len=50)

                @field_validator('insight_alignment', mode='before')
                @classmethod
                def validate_insight_alignment(cls, v):
                    return _normalize_insight_alignment(v)

            class MissingSkills(BaseModel):
                """Missing skills by priority: critical > high > medium."""
                critical: _List[str] = Field(default_factory=list, description="1-2 most critical gaps to close first")
                high: _List[str] = Field(default_factory=list, description="2-3 high-impact gaps")
                medium: _List[str] = Field(default_factory=list, description="Remaining gaps, medium priority")
                
                @field_validator('critical', mode='before')
                @classmethod
                def validate_critical(cls, v):
                    return _sanitize_str_list(v, max_items=3, max_item_len=50)
                
                @field_validator('high', mode='before')
                @classmethod
                def validate_high(cls, v):
                    return _sanitize_str_list(v, max_items=5, max_item_len=50)
                
                @field_validator('medium', mode='before')
                @classmethod
                def validate_medium(cls, v):
                    return _sanitize_str_list(v, max_items=10, max_item_len=50)

            class SkillAnalysis(BaseModel):
                career_paths: _List[CareerPath] = Field(default_factory=list)
                alternate_career_paths: _List[AlternateCareerPath] = Field(default_factory=list)
                missing_skills: MissingSkills = Field(default_factory=MissingSkills, description="Missing skills by priority tier: critical, high, medium")
                improvement_recommendations: _List[str] = Field(default_factory=list)
                career_advice: _List[str] = Field(default_factory=list)
                
                @field_validator('career_paths', mode='before')
                @classmethod
                def limit_career_paths(cls, v):
                    if isinstance(v, list):
                        return v[:5]  # Max 5 career paths
                    return []
                
                @field_validator('alternate_career_paths', mode='before')
                @classmethod
                def limit_alternate_paths(cls, v):
                    if isinstance(v, list):
                        return v[:5]  # Max 5 alternate paths
                    return []
                
                @field_validator('improvement_recommendations', 'career_advice', mode='before')
                @classmethod
                def validate_advice_lists(cls, v):
                    return _sanitize_str_list(v, max_items=10, max_item_len=300)

            llm_start = time.time()
            structured_result: SkillAnalysis = await invoke_structured_llm(
                prompt,
                SkillAnalysis,
                task_type=TaskType.SKILL_ANALYSIS,
                preferred_model="gemini-2.5-flash",
                agent_name="skill_and_career_advisor",
                temperature=0.2,
                max_output_tokens=1500,
                timeout=40.0,
                raise_on_fallback=False,
            )
            timing_breakdown["llm_call"] = time.time() - llm_start
            # Defensive: Gemini can return None on MALFORMED_FUNCTION_CALL (langchain-google #1207)
            if structured_result is None:
                raise ValueError("LLM returned None for structured output (possible MALFORMED_FUNCTION_CALL)")
            analysis = structured_result.model_dump()

            # --- Ensure alternate career path rationales are unique ---
            # try:
            #     seen = set()
            #     for path in analysis.get("alternate_career_paths", []):
            #         rationale_raw = path.get("rationale")
            #         rationale = (rationale_raw or "").strip().lower()

            #         if not rationale or rationale in seen:
            #             path["rationale"] = (
            #                 "This role is a suitable alternative because it emphasizes a different "
            #                 "set of skills and long-term career trajectory compared to the primary "
            #                 "career paths, aligning with the candidate’s background and growth goals."
            #             )

            #         seen.add(path["rationale"].strip().lower())
            # except Exception as e:
            #     log.warning(f"Alternate career rationale validation failed: {e}")
        except Exception as struct_err:
            # Fallback to regular LLM call
            log.warning(f"Structured output failed, using fallback: {struct_err}")
            llm_start = time.time()
            llm_response = await invoke_llm(
                prompt=prompt, 
                task_type="skill_analysis",
                agent_name="skill_and_career_advisor",
                response_mime_type="application/json",
                preferred_model="gemini-2.5-flash",  # ✅ OPTIMIZATION: Use cheaper model
                max_output_tokens=1500  # ✅ OPTIMIZATION: Limit output tokens
            )
            timing_breakdown["llm_call"] = time.time() - llm_start
            analysis = _extract_json_from_response(_to_text(llm_response), MAX_RESPONSE_LENGTH)
        
        if not isinstance(analysis, dict):
            analysis = {}
        analysis["alternate_career_paths"] = _normalize_alternate_career_paths(
            analysis.get("alternate_career_paths")
        )
        if not analysis["alternate_career_paths"]:
            retry_start = time.time()
            analysis["alternate_career_paths"] = await _retry_missing_alternate_career_paths(
                prompt=prompt,
            )
            timing_breakdown["llm_call"] += time.time() - retry_start

        proc_start = time.time()
        processing_time = _calculate_processing_time(start_time)
        timing_breakdown["result_processing"] = time.time() - proc_start
        timing_breakdown["total"] = processing_time
        
        # Log timing breakdown
        log.info(
            f"⏱️ SKILL_AND_CAREER_ADVISOR timing breakdown: "
            f"prompt_gen={timing_breakdown['prompt_generation']*1000:.1f}ms, "
            f"llm={timing_breakdown['llm_call']*1000:.1f}ms, "
            f"result_proc={timing_breakdown['result_processing']*1000:.1f}ms, "
            f"total={timing_breakdown['total']*1000:.1f}ms"
        )

        # Ensure each alternate career path has match_percentage, lacking_skills, and description/rationale as bullet points
        ap = analysis.get("alternate_career_paths") or []
        if ap:
            analysis["alternate_career_paths"] = _normalize_alternate_career_paths(ap)

        # Record successful analysis
        await skill_memory.record_attempt(
            'skill_analysis', 'llm', True, 0.8, processing_time
        )
        
        # Log success
        log_agent_completion(log_context, {
            "success": True,
            "career_paths_count": len(analysis.get('career_paths', []))
        }, "llm", processing_time)
        
        # --- Universal de-duplication: ensure 'missing_skills' excludes already-extracted skills ---
        try:
            # Build a normalized set of skills already present in the resume
            resume_skills_raw = []
            structured_resume_skills = structured_resume.get("skills", []) if isinstance(structured_resume, dict) else []
            if isinstance(structured_resume_skills, list):
                for item in structured_resume_skills:
                    if isinstance(item, dict):
                        # Common keys across schemas
                        for key in ("SkillName", "skill", "name", "Name"):
                            if key in item and item[key]:
                                resume_skills_raw.append(str(item[key]))
                                break
                    elif isinstance(item, str):
                        resume_skills_raw.append(item)

            def normalize(skill: str) -> str:
                # Take the left side before a colon to handle entries like "Skill: explanation"
                base = str(skill).split(":", 1)[0]
                return base.strip().lower()

            resume_skill_set = {normalize(s) for s in resume_skills_raw if s}

            # Filter LLM-proposed missing skills (by tier) against resume skills
            missing = analysis.get("missing_skills")
            if isinstance(missing, dict):
                for tier in ("critical", "high", "medium"):
                    raw_tier = (missing.get(tier) or []) if isinstance(missing.get(tier), list) else []
                    filtered_tier = [entry for entry in raw_tier if normalize(entry) and normalize(entry) not in resume_skill_set]
                    missing[tier] = filtered_tier
                analysis["missing_skills"] = missing
            elif missing is not None and hasattr(missing, "model_dump"):
                missing = missing.model_dump() if callable(getattr(missing, "model_dump")) else {}
                for tier in ("critical", "high", "medium"):
                    raw_tier = missing.get(tier) or []
                    filtered_tier = [entry for entry in raw_tier if normalize(entry) and normalize(entry) not in resume_skill_set]
                    missing[tier] = filtered_tier
                analysis["missing_skills"] = missing
        except Exception as _filter_err:
            # Non-fatal: continue with original analysis if anything goes wrong
            log.warning(f"Missing skills de-duplication failed: {_filter_err}")

        # Update the skill gap analysis with (possibly filtered) LLM response
        current_gap_analysis.update(analysis)
        # Attach performance-aware reasoning for downstream consumers
        if isinstance(current_gap_analysis, dict):
            if performance_context:
                current_gap_analysis.setdefault("performance_context", performance_context)
            # Also expose raw assessment signals under a conventional key used by downstream agents
            if assessment_results and isinstance(assessment_results, dict):
                current_gap_analysis.setdefault("assessment_performance", assessment_results)
        
        # Keep the original LLM response structure - no need to reformat
        # The LLM returns: career_paths, alternate_career_paths, missing_skills { critical, high, medium }, improvement_recommendations, career_advice
        
        
        # Update state
        state["raw_skill_gap_analysis_output"] = current_gap_analysis
        state["analysis_status"] = "success"
        state["analysis_method"] = "llm"
        state["confidence_score"] = 0.8
        state["processing_time"] = processing_time
        state["request_id"] = request_id
        
        # Store skill and career advisor data in chat_sessions (per-UID doc) - MOVED TO BACKGROUND (non-blocking)
        uid = state.get("uid")
        if uid and current_gap_analysis:
            # Fire-and-forget: Don't block response on storage operations
            async def _store_data_background():
                try:
                    storage_start = time.time()
                    log.info(f"🔄 SKILL_AND_CAREER_ADVISOR: Starting background storage for UID={uid}")
                    
                    # In evaluation flow (assessment_results present), REPLACE instead of merge
                    is_evaluation_flow = bool(state.get("assessment_results"))
                    existing = await run_blocking_io(get_gap_doc, uid) or {}
                    
                    if is_evaluation_flow:
                        # Evaluation flow: REPLACE career_advisor completely
                        merged = {
                            "skill_and_career_advisor": {
                                "raw_skill_gap_analysis_output": current_gap_analysis,
                                "analysis_status": "success",
                                "analysis_method": "llm",
                                "confidence_score": 0.8,
                                "processing_time": processing_time,
                                "timing_breakdown": timing_breakdown,
                                "request_id": request_id,
                                "career_paths_count": len(current_gap_analysis.get('career_paths', [])),
                                "evaluation_flow": True
                            }
                        }
                        # Preserve market_and_course_recommender if present, but replace career_advisor
                        if "market_and_course_recommender" in existing:
                            merged["market_and_course_recommender"] = existing["market_and_course_recommender"]
                        log.info(f"✅ Evaluation flow: Replaced career_advisor data for UID={uid}")
                    else:
                        # Normal flow: Merge with existing
                        merged = {**existing}
                        merged["skill_and_career_advisor"] = {
                            "raw_skill_gap_analysis_output": current_gap_analysis,
                            "analysis_status": "success",
                            "analysis_method": "llm",
                            "confidence_score": 0.8,
                            "processing_time": processing_time,
                            "timing_breakdown": timing_breakdown,
                            "request_id": request_id,
                            "career_paths_count": len(current_gap_analysis.get('career_paths', []))
                        }
                    
                    await run_blocking_io(
                        upsert_gap_doc,
                        uid,
                        merged,
                        metadata={
                            "agent": "skill_and_career_advisor",
                            "uid": uid,
                            "status": "skill_and_career_advisor_complete",
                            "method": "llm",
                            "evaluation_flow": is_evaluation_flow
                        }
                    )
                    storage_time = time.time() - storage_start
                    log.info(f"✅ SKILL_AND_CAREER_ADVISOR: Background storage completed in {storage_time*1000:.1f}ms for UID={uid}")
                except Exception as e:
                    log.error(f"❌ SKILL_AND_CAREER_ADVISOR: Background storage failed for UID={uid}: {e}", exc_info=True)
            
            # Fire-and-forget: Start background task without awaiting
            asyncio.create_task(_store_data_background())
            log.info(f"🚀 SKILL_AND_CAREER_ADVISOR: Started background storage task for UID={uid} (non-blocking)")
        
        # Return both formats:
        # 1. Strict JSON format for callbacks
        # 2. Original fields for internal pipeline use
        result = {
            # Strict JSON format for callbacks
            "status": "completed",
            "node": "career_advisor",
            "output": {
                "raw_skill_gap_analysis_output": current_gap_analysis
            },
            # Original fields for internal pipeline use
            "raw_skill_gap_analysis_output": current_gap_analysis,
            "analysis_status": "success",
            "analysis_method": "llm",
            "confidence_score": 0.8,
            "processing_time": processing_time,
            "request_id": request_id
        }
        cache_response("skill_and_career_advisor", cache_input, result)
        return result
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        AgentLogger.log_error(log_context, f"Skill and career advice failed: {str(e)}", processing_time)
        
        # Record failure
        await skill_memory.record_attempt(
            'skill_analysis', 'llm', False, 0.0, processing_time
        )
        
        return _create_error_response(f"Skill and career advice failed: {str(e)}", processing_time)

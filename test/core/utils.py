"""
Centralized utility functions for all KAFIN agents.

This module provides common functionality used across all agents,
eliminating code duplication and ensuring consistent behavior.
"""

import json
import os
import re
import time
import logging
import threading
import asyncio
import concurrent.futures
from functools import partial
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass

# Re-export schedule_background_task for convenience
from core.background_tasks import schedule_background_task

log = logging.getLogger(__name__)


def get_missing_skills_flat(skill_gap_analysis: Dict[str, Any]) -> List[str]:
    """
    Return a flat list of missing skills from skill gap analysis.
    Supports both formats: missing_skills as object { critical, high, medium } or legacy list.
    """
    if not isinstance(skill_gap_analysis, dict):
        return []
    missing = skill_gap_analysis.get("missing_skills")
    if isinstance(missing, dict):
        return (
            list(missing.get("critical") or [])
            + list(missing.get("high") or [])
            + list(missing.get("medium") or [])
        )
    if isinstance(missing, list):
        return list(missing)
    return []


def _mask(s: str, keep: int = 6) -> str:
    """Mask sensitive strings for logging."""
    return (s or "")[:keep] + "***" if s else ""


def _sanitize_text_for_llm(text: str, pii_patterns: List[re.Pattern] = None, injection_filters: List[re.Pattern] = None) -> str:
    """Sanitize text before sending to LLM."""
    if not isinstance(text, str):
        return ""
    
    # Keep printable chars
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
    
    # Redact PII before LLM processing
    if pii_patterns:
        for pattern in pii_patterns:
            text = pattern.sub("[REDACTED]", text)
    
    # Filter injection attempts
    if injection_filters:
        for pattern in injection_filters:
            text = pattern.sub("[FILTERED]", text)
    
    return text.strip()


def _to_text(resp: Any, max_length: int = 50000) -> str:
    """Convert response to text with length protection."""
    content = resp.content if hasattr(resp, "content") else str(resp)
    if not isinstance(content, str):
        content = str(content)
    if len(content) > max_length:
        content = content[:max_length]
    return content


def _clean_json_text(s: str) -> str:
    """Clean JSON text by removing trailing commas."""
    return re.sub(r",(\s*[}\]])", r"\1", s).strip()


def _scan_balanced_json(text: str) -> Optional[str]:
    """Find first balanced JSON object in text."""
    start = text.find("{")
    while start != -1:
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
                        return text[start:i+1]
        start = text.find("{", start + 1)
    return None


def _scan_balanced_array(text: str) -> Optional[str]:
    """Find first balanced JSON array in text (e.g. root-level [...])."""
    start = text.find("[")
    while start != -1:
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
                elif ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        return text[start:i+1]
        start = text.find("[", start + 1)
    return None


def _safe_json_loads(s: str) -> Dict[str, Any]:
    """Safely load JSON with fallback parsing."""
    if not s:
        return {}
    try:
        return json.loads(_clean_json_text(s))
    except json.JSONDecodeError:
        # Try to convert single quotes to double quotes
        try:
            s2 = re.sub(r"'([^']*)':", r'"\1":', s)
            s2 = re.sub(r":\s*'([^']*)'", r': "\1"', s2)
            return json.loads(_clean_json_text(s2))
        except json.JSONDecodeError:
            return {}


def _extract_json_from_response(response: Any, max_response_length: int = 50000) -> Dict[str, Any]:
    """Robustly extract and validate JSON from LLM response."""
    if not response:
        return {}
    
    text = _to_text(response, max_response_length)
    
    # Prefer fenced ```json blocks. Use content up to the *last* ``` so that if the LLM
    # embeds ``` inside a string (e.g. in rationale), we still capture the full JSON.
    m = re.search(r"```json\s*", text, re.IGNORECASE)
    if m:
        start = m.end()
        last_fence = text.rfind("```")
        if last_fence > start:
            content = text[start:last_fence].strip()
            parsed = _safe_json_loads(content)
            if parsed:
                return parsed
        # Fallback: original non-greedy match in case last ``` is wrong
        m2 = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if m2:
            parsed = _safe_json_loads(m2.group(1))
            if parsed:
                return parsed
    
    # Try direct parse
    parsed = _safe_json_loads(text)
    if parsed:
        return parsed
    
    # Balanced scan: try root-level array first so we don't take only the first { } inside [...]
    scanned = _scan_balanced_array(text)
    if scanned:
        parsed = _safe_json_loads(scanned)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed
    scanned = _scan_balanced_json(text)
    if scanned:
        parsed = _safe_json_loads(scanned)
        if parsed:
            return parsed
    
    return {}


def _coerce_score(v) -> Optional[float]:
    """Coerce numeric strings to float (LLMs often return "92" as string)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _merge_dedupe(a: List[str], b: List[str], limit: int) -> List[str]:
    """Merge and dedupe lists while preserving order (avoid set() reordering)."""
    out, seen = [], set()
    for x in (a or []) + (b or []):
        if not isinstance(x, str):
            continue
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= limit:
            break
    return out


def _normalize_chat_history(chat_history: List[Dict], max_items: int = 50, max_content_length: int = 2000) -> List[Dict]:
    """Normalize and clamp chat history for prompt size & safety."""
    if not chat_history:
        return []
    
    chat_history = chat_history[-max_items:]
    
    def _norm_msg(m):
        if not isinstance(m, dict): 
            return None
        role = str(m.get("role", "user"))[:10]
        content = str(m.get("content", ""))[:max_content_length]
        return {"role": role, "content": content}
    
    return [m for m in (_norm_msg(m) for m in chat_history) if m]


def _create_error_response(error_message: str, agent_type: str = "agent") -> Dict[str, Any]:
    """Create standardized error response."""
    base_response = {
        "success": False,
        "error": error_message,
        "timestamp": time.time(),
        "agent_type": agent_type,
        "confidence_score": 0.0,
        "analysis_method": "error"
    }
    
    # Add agent-specific required fields for callback schema compliance
    if agent_type == "personal_info_parser":
        base_response.update({
            "status": "completed",
            "node": "personal_info_parser",
            "name": "",  # Required by callback schema - use empty string instead of None
            "output": {
                "name": "",
                "contact_details": {}
            }
        })
    elif agent_type == "education_parser":
        base_response.update({
            "status": "completed", 
            "node": "education_parser",
            "education": []  # Required by callback schema
        })
    elif agent_type == "experience_parser":
        base_response.update({
            "status": "completed",
            "node": "experience_parser", 
            "work_experience": [],
            "certifications": [],
            "projects": [],
            "extras": []
        })
    elif agent_type == "skills_parser":
        base_response.update({
            "status": "completed",
            "node": "skills_parser",
            "skills": []  # Required by callback schema
        })
    
    return base_response


def _validate_state_inputs(state: Dict[str, Any], required_fields: List[str] = None) -> Tuple[bool, str]:
    """Validate input state for security and completeness."""
    if not isinstance(state, dict):
        return False, "Invalid state format"
    
    # Check for required fields
    if required_fields:
        for field in required_fields:
            if field not in state:
                return False, f"Missing required field: {field}"
    
    # Check for error states
    for key, value in state.items():
        if isinstance(value, dict) and "error" in value:
            return False, f"Error detected in {key}"
    
    return True, "Valid"


def _generate_request_id() -> str:
    """Generate a short request ID for traceability."""
    import uuid
    return str(uuid.uuid4())[:8]


def _calculate_processing_time(start_time: float) -> float:
    """Calculate processing time from start time."""
    return time.time() - start_time


# =============================================================================
# SELECTIVE DATA PASSING SYSTEM
# =============================================================================

@dataclass
class AgentDataRequirements:
    """Defines what data each agent needs from the structured resume."""
    agent_name: str
    required_fields: Set[str]
    optional_fields: Set[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.optional_fields is None:
            self.optional_fields = set()

# Define data requirements for each agent based on analysis
AGENT_DATA_REQUIREMENTS = {
    "resume_parser": AgentDataRequirements(
        agent_name="resume_parser",
        required_fields={"resume_text"},  # Special case - needs raw text
        optional_fields=set(),
        description="Parses raw resume text into structured format"
    ),
    
    "resume_scorer": AgentDataRequirements(
        agent_name="resume_scorer",
        required_fields={"experience", "skills", "education", "projects", "certifications", "Summary"},
        optional_fields={"Name", "Achievements", "Languages"},
        description="Scores resume based on experience, skills, education, and presentation"
    ),
    
    "interview_agent": AgentDataRequirements(
        agent_name="interview_agent", 
        required_fields={"Name", "skills", "experience"},
        optional_fields={"Summary", "projects", "education"},
        description="Conducts interviews using candidate name, roles, skills, and experience"
    ),
    
    "gap_analyzer": AgentDataRequirements(
        agent_name="gap_analyzer",
        required_fields={"skills", "experience", "education"},
        optional_fields={"projects", "certifications", "Summary"},
        description="Analyzes skill gaps based on experience, skills, and education"
    ),
    
    "assessment_recommender": AgentDataRequirements(
        agent_name="assessment_recommender",
        required_fields={"skills", "experience", "education", "projects"},
        optional_fields={"certifications", "Achievements", "Summary"},
        description="Recommends assessments based on skills, experience, and education"
    ),
    
    "skill_and_career_advisor": AgentDataRequirements(
        agent_name="skill_and_career_advisor",
        required_fields={"skills", "experience", "education"},
        optional_fields={"projects", "Summary", "Achievements"},
        description="Provides career advice based on skills, experience, and education"
    ),
    
    "market_and_course_recommender": AgentDataRequirements(
        agent_name="market_and_course_recommender",
        required_fields={"skills", "experience", "education"},
        optional_fields={"projects", "certifications", "Summary"},
        description="Recommends courses and market insights based on skills and experience"
    ),
    
    "personal_info_parser": AgentDataRequirements(
        agent_name="personal_info_parser",
        required_fields={"resume_text"},  # Special case - needs raw text
        optional_fields=set(),
        description="Extracts personal information from raw resume text"
    ),
    
    "education_parser": AgentDataRequirements(
        agent_name="education_parser",
        required_fields={"resume_text"},  # Special case - needs raw text
        optional_fields=set(),
        description="Extracts education information from raw resume text"
    ),
    
    "experience_parser": AgentDataRequirements(
        agent_name="experience_parser",
        required_fields={"resume_text"},  # Special case - needs raw text
        optional_fields=set(),
        description="Extracts work experience from raw resume text"
    ),
    
    "skills_parser": AgentDataRequirements(
        agent_name="skills_parser",
        required_fields={"resume_text"},  # Special case - needs raw text
        optional_fields=set(),
        description="Extracts skills from raw resume text"
    ),
    
    "job_description_parser": AgentDataRequirements(
        agent_name="job_description_parser",
        required_fields={"jd_text"},  # Special case - needs raw JD text
        optional_fields=set(),
        description="Parses job description text into structured format"
    ),
    
    "validate_resume": AgentDataRequirements(
        agent_name="validate_resume",
        required_fields={"resume_text"},  # Special case - needs raw text
        optional_fields=set(),
        description="Validates resume text for quality and completeness"
    ),
    
    "validate_jd": AgentDataRequirements(
        agent_name="validate_jd",
        required_fields={"jd_text"},  # Special case - needs raw text
        optional_fields=set(),
        description="Validates job description text for quality and completeness"
    ),
    
    "ranker": AgentDataRequirements(
        agent_name="ranker",
        required_fields={"structured_resume", "job_description"},
        optional_fields={"resumeScore"},
        description="Ranks candidates against job requirements"
    ),
    
    "report_generator": AgentDataRequirements(
        agent_name="report_generator",
        required_fields={"structured_resume", "assessment_results"},
        optional_fields={"resumeScore", "generated_questions"},
        description="Generates comprehensive reports from all analysis data"
    ),
    
    "assessment_evaluator": AgentDataRequirements(
        agent_name="assessment_evaluator",
        required_fields={"submission", "assessment_plan"},
        optional_fields={"structured_resume"},
        description="Evaluates assessment submissions"
    ),
    
    "assessment_question_generator": AgentDataRequirements(
        agent_name="assessment_question_generator",
        required_fields={"assessment_plan", "structured_resume"},
        optional_fields={"skills", "experience"},
        description="Generates assessment questions based on candidate profile"
    )
}

def filter_resume_data_for_agent(
    structured_resume: Dict[str, Any], 
    agent_name: str
) -> Dict[str, Any]:
    """
    Extract only the data fields that a specific agent needs.
    
    Args:
        structured_resume: Full structured resume data
        agent_name: Name of the agent requesting data
        
    Returns:
        Filtered resume data containing only required fields
    """
    if agent_name not in AGENT_DATA_REQUIREMENTS:
        log.warning(f"Agent '{agent_name}' not found in requirements, returning minimal data")
        # Return minimal data for unknown agents
        return {
            "Name": structured_resume.get("Name", "Unknown"),
            "skills": structured_resume.get("skills", []),
            "experience": structured_resume.get("experience", [])
        }
    
    requirements = AGENT_DATA_REQUIREMENTS[agent_name]
    filtered_data = {}
    
    # Add required fields
    for field in requirements.required_fields:
        if field in structured_resume:
            filtered_data[field] = structured_resume[field]
        else:
            # Provide default empty values for missing required fields
            if field in ["skills", "experience", "education", "projects", "certifications", "Achievements", "Languages"]:
                filtered_data[field] = []
            elif field in ["Name", "Summary"]:
                filtered_data[field] = ""
            else:
                filtered_data[field] = None
            log.debug(f"Agent '{agent_name}' missing required field '{field}', using default")
    
    # Add optional fields if they exist
    for field in requirements.optional_fields:
        if field in structured_resume:
            filtered_data[field] = structured_resume[field]
    
    log.info(f"Filtered data for '{agent_name}': {len(filtered_data)} fields (required: {len(requirements.required_fields)}, optional: {len(requirements.optional_fields)})")
    return filtered_data

def create_agent_state(
    base_state: Dict[str, Any], 
    agent_name: str,
    additional_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a state dictionary for a specific agent with only the data it needs.
    
    Args:
        base_state: Base state containing structured_resume and system data
        agent_name: Name of the target agent
        additional_data: Any additional data to include
        
    Returns:
        Optimized state for the specific agent
    """
    # Always include system context
    agent_state = {
        "tenant_id": base_state.get("tenant_id", "default_tenant"),
        "uid": base_state.get("uid"),
        "session_id": base_state.get("session_id"),
        "callback_url": base_state.get("callback_url"),
        "chat_history": base_state.get("chat_history", [])
    }
    
    # Handle special cases for agents that need raw text
    if agent_name in ["resume_parser", "personal_info_parser", "education_parser", 
                      "experience_parser", "skills_parser", "validate_resume"]:
        agent_state["resume_text"] = base_state.get("resume_text", "")
    elif agent_name in ["job_description_parser", "validate_jd"]:
        agent_state["jd_text"] = base_state.get("jd_text", "")
    
    # Add filtered resume data for agents that need structured data
    if "structured_resume" in base_state and agent_name not in ["resume_parser", "personal_info_parser", 
                                                               "education_parser", "experience_parser", 
                                                               "skills_parser", "validate_resume"]:
        agent_state["structured_resume"] = truncate_resume_data_for_agent(
            base_state["structured_resume"], 
            agent_name
        )
    
    # Add any additional data
    if additional_data:
        agent_state.update(additional_data)
    
    # Add any existing results from previous agents that might be needed
    for key in ["resumeScore", "assessment_results", "generated_questions", 
                "job_description", "assessment_plan", "submission"]:
        if key in base_state:
            agent_state[key] = base_state[key]
    
    log.info(f"Created optimized state for '{agent_name}' with {len(agent_state)} fields")
    return agent_state

def truncate_resume_data_for_agent(
    structured_resume: Dict[str, Any], 
    agent_name: str
) -> Dict[str, Any]:
    """
    OPTIMIZED: Truncate resume data based on agent needs to reduce token usage.
    """
    if not isinstance(structured_resume, dict):
        return {}
    
    # Agent-specific truncation limits
    truncation_limits = {
        "resume_score": {
            "skills": 6,  # Top 6 skills only
            "work_experience": 2,  # Top 2 jobs
            "education": 1,  # Top 1 degree
            "projects": 1,  # Top 1 project
            "certifications": 3  # Top 3 certifications
        },
        "market_and_course_recommender": {
            "skills": 8,  # Top 8 skills
            "work_experience": 2,  # Top 2 jobs
            "education": 1,  # Top 1 degree
            "projects": 2,  # Top 2 projects
            "certifications": 3  # Top 3 certifications
        },
        "skill_and_career_advisor": {
            "skills": 10,  # Top 10 skills
            "work_experience": 3,  # Top 3 jobs
            "education": 2,  # Top 2 degrees
            "projects": 2,  # Top 2 projects
            "certifications": 5  # Top 5 certifications
        },
        "assessment_recommender": {
            "skills": 6,  # Top 6 skills
            "work_experience": 2,  # Top 2 jobs
            "education": 1,  # Top 1 degree
            "projects": 1,  # Top 1 project
            "certifications": 3  # Top 3 certifications
        },
        "gap_analyzer": {
            "skills": 8,  # Top 8 skills
            "work_experience": 2,  # Top 2 jobs
            "education": 1,  # Top 1 degree
            "projects": 2,  # Top 2 projects
            "certifications": 3  # Top 3 certifications
        },
        "interest_filler": {
            "skills": 5,  # Top 5 skills
            "work_experience": 1,  # Top 1 job
            "education": 1,  # Top 1 degree
            "projects": 1,  # Top 1 project
            "certifications": 2  # Top 2 certifications
        }
    }
    
    # Get limits for this agent
    limits = truncation_limits.get(agent_name, {
        "skills": 5,
        "work_experience": 2,
        "education": 1,
        "projects": 1,
        "certifications": 3
    })
    
    # Create truncated copy
    truncated_data = structured_resume.copy()
    
    # Truncate list fields
    for field, limit in limits.items():
        if field in truncated_data and isinstance(truncated_data[field], list):
            original_length = len(truncated_data[field])
            truncated_data[field] = truncated_data[field][:limit]
            if original_length > limit:
                log.info(f"🔧 Truncated {field} for '{agent_name}': {original_length} → {limit} items")
    
    # Truncate text fields
    text_fields = ["Summary", "Achievements"]
    for field in text_fields:
        if field in truncated_data and isinstance(truncated_data[field], str):
            if len(truncated_data[field]) > 500:  # Limit to 500 chars
                truncated_data[field] = truncated_data[field][:500] + "..."
                log.info(f"🔧 Truncated {field} for '{agent_name}': {len(truncated_data[field])} chars")
    
    # OPTIMIZATION: Preserve cached/internal fields (fields starting with _)
    # These are pre-computed optimizations that should not be truncated
    for key in structured_resume.keys():
        if key.startswith("_"):
            truncated_data[key] = structured_resume[key]
            log.debug(f"🔧 Preserved cached field '{key}' for '{agent_name}'")
    
    return truncated_data


# Groq / parser often merge these at top level while structured_resume stays empty.
_RESUME_SCORER_FLAT_KEYS = (
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


def resume_dict_has_scorable_content(d: Any) -> bool:
    """True if dict has enough resume sections for scoring (nested or flat)."""
    if not isinstance(d, dict) or not d:
        return False
    return bool(
        d.get("name")
        or d.get("Name")
        or d.get("work_experience")
        or d.get("experience")
        or d.get("education")
        or d.get("skills")
        or d.get("professional_summary")
        or d.get("projects")
    )


def synthesize_structured_resume_from_flat_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build a structured_resume-shaped dict from top-level parser fields when nested
    structured_resume is missing or empty (same pattern as interest_filler / groq output).
    """
    out: Dict[str, Any] = {}
    for k in _RESUME_SCORER_FLAT_KEYS:
        if k in state and state[k] is not None:
            out[k] = state[k]
    if not out.get("name") and state.get("Name"):
        out["name"] = state["Name"]
    if not out.get("work_experience") and state.get("experience"):
        out["work_experience"] = state["experience"]
    if not out.get("contact_details") and state.get("ContactDetails"):
        out["contact_details"] = state["ContactDetails"]
    if resume_dict_has_scorable_content(out):
        return out
    return None


def get_effective_structured_resume_for_scorer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer nested structured_resume; fall back to top-level Groq/parser fields."""
    sr = state.get("structured_resume")
    if isinstance(sr, dict) and resume_dict_has_scorable_content(sr):
        return sr
    syn = synthesize_structured_resume_from_flat_state(state)
    if syn is not None:
        return syn
    if isinstance(sr, dict):
        return sr
    return {}


def create_optimized_resume_scorer_inputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create optimized inputs for resume_scorer_agent - only send required data.
    
    Resume Scorer only needs:
    - structured_resume (truncated)
    - chat_history (compressed)
    - tenant_id
    - uid
    - session_id
    """
    # Extract only essential fields
    optimized_inputs = {
        "tenant_id": state.get("tenant_id", "default_tenant"),
        "uid": state.get("uid"),
        "session_id": state.get("session_id"),
        "callback_url": state.get("callback_url")
    }
    
    # Nested structured_resume may be empty while Groq merged flat fields into state
    effective_sr = get_effective_structured_resume_for_scorer(state)
    if effective_sr:
        optimized_inputs["structured_resume"] = truncate_resume_data_for_agent(
            effective_sr, "resume_score"
        )
    
    # Add compressed chat history
    chat_history = state.get("chat_history", [])
    if chat_history:
        optimized_inputs["chat_history"] = compress_chat_history(chat_history, max_messages=5, max_message_length=150)
    
    log.info(f"🔧 Resume Scorer inputs optimized: {len(state)} → {len(optimized_inputs)} fields")
    return optimized_inputs


def map_assessment_score_to_proficiency(
    score: Any,
    difficulty: Optional[str] = None,
    existing_proficiency: Optional[str] = None,
) -> Optional[str]:
    """
    Map an assessment total_score (0–100) + difficulty → proficiency string "X/10".

    Rules (from product spec):
    - <60:        **no change** (keep existing_proficiency)
    - 60–79:      7/10
    - 80–89:      8/10
    - 90–95:      9/10
    - 96–100:     10/10
    - Difficulty nudges the band upwards slightly for harder tests.
    - We NEVER downgrade an existing proficiency value.
    """
    try:
        if score is None:
            return existing_proficiency
        numeric_score = float(score)
    except (TypeError, ValueError):
        return existing_proficiency

    # Below 60 → keep existing proficiency (explicit spec)
    if numeric_score < 60:
        return existing_proficiency

    # Base band from score
    if 60 <= numeric_score <= 79:
        base = 7.0
    elif 80 <= numeric_score <= 89:
        base = 8.0
    elif 90 <= numeric_score <= 95:
        base = 9.0
    else:  # 96–100
        base = 10.0

    # Difficulty-aware bonus (small, clamped later)
    bonus = 0.0
    if isinstance(difficulty, str):
        diff = difficulty.lower().strip()
        if diff in ("hard", "expert"):
            bonus = 1.0
        elif diff in ("medium",):
            bonus = 0.5

    new_value = min(10.0, base + bonus)

    # Never downgrade existing proficiency
    if existing_proficiency:
        try:
            existing_numeric = float(str(existing_proficiency).split("/")[0])
            new_value = max(new_value, existing_numeric)
        except (TypeError, ValueError):
            pass

    # Round to nearest integer for X/10 format
    return f"{int(round(new_value))}/10"


def map_assessment_score_to_skill_score_floor(
    score: Any,
    difficulty: Optional[str] = None,
) -> Optional[float]:
    """
    Map assessment total_score (0–100) + difficulty → a minimum SkillsScore (0–100).

    Used by resume scorer to ensure the SkillsScore component is at least as strong
    as what a good assessment warrants – without hardcoding absolute scores.

    Bands:
    - <60:      no floor (returns None)
    - 60–79:    70
    - 80–89:    80
    - 90–95:    90
    - 96–100:   90 (single assessment does not imply overall skills 100; floor capped for one topic)

    Difficulty nudges the floor up slightly for "hard" / "expert" assessments,
    clamped to 100.
    """
    try:
        if score is None:
            return None
        numeric_score = float(score)
    except (TypeError, ValueError):
        return None

    if numeric_score < 60:
        return None

    if 60 <= numeric_score <= 79:
        base = 70.0
    elif 80 <= numeric_score <= 89:
        base = 80.0
    elif 90 <= numeric_score <= 95:
        base = 90.0
    else:
        # 96–100: use 90 so one strong assessment doesn't force overall SkillsScore to 100
        base = 90.0

    bonus = 0.0
    if isinstance(difficulty, str):
        diff = difficulty.lower().strip()
        if diff in ("hard", "expert"):
            bonus = 5.0
        elif diff == "medium":
            bonus = 2.5

    return min(100.0, base + bonus)

def create_optimized_career_advisor_inputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create optimized inputs for skill_and_career_advisor_agent - only send required data.
    
    Career Advisor needs:
    - structured_resume (truncated)
    - user_interests (compressed)
    - raw_skill_gap_analysis_output (compressed)
    - assessment_results (compressed)
    - report (compressed)
    - tenant_id, uid, session_id
    """
    # Extract only essential fields
    optimized_inputs = {
        "tenant_id": state.get("tenant_id", "default_tenant"),
        "uid": state.get("uid"),
        "session_id": state.get("session_id"),
        "callback_url": state.get("callback_url")
    }
    
    # Add truncated structured resume
    if "structured_resume" in state:
        optimized_inputs["structured_resume"] = truncate_resume_data_for_agent(
            state["structured_resume"], "skill_and_career_advisor"
        )
    
    # Compress user interests
    user_interests = state.get("user_interests", [])
    if user_interests:
        compressed_interests = []
        for i, interest in enumerate(user_interests[:5]):  # Only first 5 interests
            if isinstance(interest, dict):
                answer = interest.get("answer", "")
            else:
                # Handle case where interest is a string
                answer = str(interest)
            compressed_interests.append({
                "question": f"Q{i+1}", 
                "answer": answer[:100]
            })
        optimized_inputs["user_interests"] = compressed_interests
    
    # Compress skill gap analysis
    raw_skill_gap_analysis_output = state.get("raw_skill_gap_analysis_output", {})
    if raw_skill_gap_analysis_output:
        missing = raw_skill_gap_analysis_output.get('missing_skills')
        if isinstance(missing, dict):
            compressed_missing = {
                "critical": (missing.get("critical") or [])[:2],
                "high": (missing.get("high") or [])[:3],
                "medium": (missing.get("medium") or [])[:5]
            }
        else:
            flat = get_missing_skills_flat(raw_skill_gap_analysis_output)
            compressed_missing = {"critical": flat[:2], "high": flat[2:5], "medium": flat[5:8]}
        compressed_gap_analysis = {
            "missing_skills": compressed_missing,
            "career_paths": raw_skill_gap_analysis_output.get('career_paths', [])[:3]
        }
        optimized_inputs["raw_skill_gap_analysis_output"] = compressed_gap_analysis
    
    # Compress assessment results
    assessment_results = state.get("assessment_results")
    if assessment_results:
        # Handle both list and dict formats
        if isinstance(assessment_results, list):
            # Aggregate from list: average score, collect topics
            scores = []
            topics = []
            for ar in assessment_results:
                if isinstance(ar, dict):
                    score = ar.get("total_score") or ar.get("score")
                    if score is not None:
                        scores.append(float(score))
                    topic = ar.get("topic") or ar.get("assessment_topic")
                    if topic and topic not in topics:
                        topics.append(topic)
            avg_score = sum(scores) / len(scores) if scores else 0
            compressed_assessment = {
                "total_score": avg_score,
                "topics": topics[:3]  # Top 3 topics only
            }
        elif isinstance(assessment_results, dict):
            # Original dict format
            compressed_assessment = {
                "total_score": assessment_results.get("total_score", 0),
                "topics": assessment_results.get("topics", [])[:3]  # Top 3 topics only
            }
        else:
            compressed_assessment = {"total_score": 0, "topics": []}
        optimized_inputs["assessment_results"] = compressed_assessment
    
    # Compress report
    report = state.get("report")
    if report:
        raw_summary = report.get("summary", "")
        if isinstance(raw_summary, dict):
            summary_str = json.dumps(raw_summary, default=str)[:200]
        else:
            summary_str = (raw_summary or "")[:200]
        compressed_report = {
            "summary": summary_str,
            "performance_level": report.get("performance_level", "Unknown")
        }
        optimized_inputs["report"] = compressed_report
    
    log.info(f"🔧 Career Advisor inputs optimized: {len(state)} → {len(optimized_inputs)} fields")
    return optimized_inputs

def calculate_data_reduction(agent_name: str, full_resume_size: int) -> Dict[str, Any]:
    """Calculate the data reduction achieved by selective passing."""
    if agent_name not in AGENT_DATA_REQUIREMENTS:
        return {"error": f"Agent '{agent_name}' not found"}
    
    requirements = AGENT_DATA_REQUIREMENTS[agent_name]
    total_fields = len(requirements.required_fields) + len(requirements.optional_fields)
    
    # Estimate reduction (this is approximate)
    estimated_size = (total_fields / 15) * full_resume_size  # Assuming ~15 total fields in full resume
    
    return {
        "agent_name": agent_name,
        "required_fields": len(requirements.required_fields),
        "optional_fields": len(requirements.optional_fields),
        "total_fields": total_fields,
        "estimated_size": estimated_size,
        "reduction_percentage": max(0, (1 - estimated_size / full_resume_size) * 100)
    }


# ============================================================================
# TOKEN OPTIMIZATION UTILITIES
# ============================================================================

def smart_resume_truncation(resume_text: str, agent_name: str, max_chars: int = 2000) -> str:
    """
    Intelligently truncate resume text based on agent needs.
    
    Args:
        resume_text: Full resume text
        agent_name: Name of the agent requesting data
        max_chars: Maximum characters to return
        
    Returns:
        Truncated resume text optimized for the specific agent
    """
    if not resume_text or len(resume_text) <= max_chars:
        return resume_text
    
    log.info(f"🔧 Smart truncation for '{agent_name}': {len(resume_text)} → {max_chars} chars")
    
    # Agent-specific truncation strategies
    if agent_name in ["resume_parser", "personal_info_parser", "education_parser", 
                      "experience_parser", "skills_parser"]:
        # Parsing agents need more content but can be optimized
        # Keep first 70% + last 20% for context preservation
        first_part = resume_text[:int(max_chars * 0.7)]
        last_part = resume_text[-int(max_chars * 0.2):]
        return f"{first_part}...[truncated]...{last_part}"
    
    elif agent_name in ["resume_score", "interview_agent"]:
        # Scoring and interview agents need summary + skills + recent experience
        lines = resume_text.split('\n')
        summary_lines = [line for line in lines if any(keyword in line.lower() 
                       for keyword in ['summary', 'objective', 'profile', 'about'])]
        skill_lines = [line for line in lines if any(keyword in line.lower() 
                      for keyword in ['skill', 'technology', 'programming', 'language', 'tool'])]
        experience_lines = [line for line in lines if any(keyword in line.lower() 
                           for keyword in ['experience', 'work', 'employment', 'job'])]
        
        # Combine and truncate
        relevant_content = '\n'.join(summary_lines[:3] + skill_lines[:5] + experience_lines[:3])
        return relevant_content[:max_chars]
    
    elif agent_name in ["assessment_generator", "assessment_recommender"]:
        # Assessment agents need skills + experience level + education
        lines = resume_text.split('\n')
        skill_lines = [line for line in lines if any(keyword in line.lower() 
                      for keyword in ['skill', 'technology', 'programming', 'language'])]
        education_lines = [line for line in lines if any(keyword in line.lower() 
                          for keyword in ['education', 'degree', 'university', 'college', 'bachelor', 'master'])]
        experience_lines = [line for line in lines if any(keyword in line.lower() 
                           for keyword in ['experience', 'work', 'employment', 'years'])]
        
        relevant_content = '\n'.join(skill_lines[:4] + education_lines[:2] + experience_lines[:2])
        return relevant_content[:max_chars]
    
    elif agent_name in ["market_and_course_recommender", "gap_analyzer", "skill_and_career_advisor"]:
        # Market analysis agents need skills + experience + goals
        lines = resume_text.split('\n')
        skill_lines = [line for line in lines if any(keyword in line.lower() 
                      for keyword in ['skill', 'technology', 'programming', 'language'])]
        goal_lines = [line for line in lines if any(keyword in line.lower() 
                      for keyword in ['goal', 'objective', 'career', 'aspiration'])]
        experience_lines = [line for line in lines if any(keyword in line.lower() 
                           for keyword in ['experience', 'work', 'employment'])]
        
        relevant_content = '\n'.join(skill_lines[:3] + goal_lines[:2] + experience_lines[:2])
        return relevant_content[:max_chars]
    
    else:
        # Default: keep beginning with truncation indicator
        return resume_text[:max_chars] + "...[truncated]"


def compress_chat_history(chat_history: List[Dict], max_messages: int = 5, max_message_length: int = 200) -> List[Dict]:
    """
    Compress chat history while preserving essential context.
    
    Args:
        chat_history: List of chat messages
        max_messages: Maximum number of messages to keep
        max_message_length: Maximum length per message
        
    Returns:
        Compressed chat history
    """
    if not chat_history or len(chat_history) <= max_messages:
        return chat_history
    
    log.info(f"🔧 Chat compression: {len(chat_history)} → {max_messages} messages")
    
    # Strategy: Keep first message (context) + last N-1 messages (recent context)
    compressed = []
    
    # Always keep the first message for context
    if chat_history:
        first_msg = chat_history[0].copy()
        if len(first_msg.get('content', '')) > max_message_length:
            first_msg['content'] = first_msg['content'][:max_message_length] + "...[truncated]"
        compressed.append(first_msg)
    
    # Add recent messages
    recent_messages = chat_history[-(max_messages-1):] if len(chat_history) > 1 else []
    for msg in recent_messages:
        compressed_msg = msg.copy()
        if len(compressed_msg.get('content', '')) > max_message_length:
            compressed_msg['content'] = compressed_msg['content'][:max_message_length] + "...[truncated]"
        compressed.append(compressed_msg)
    
    return compressed


class TokenAwareCache:
    """Cache that tracks token usage and input similarity for optimization."""
    
    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.token_tracker = {}
        self.timestamps = {}
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()  # Thread safety
    
    def _generate_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Generate hash from input data for cache key (full hash to avoid collisions)."""
        import hashlib
        # Create a stable hash from input data - use full hash to avoid collisions
        input_str = json.dumps(input_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(input_str.encode()).hexdigest()  # Full hash, not truncated
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > self.ttl_seconds
    
    def _cleanup_expired(self):
        """Remove expired entries (thread-safe)."""
        current_time = time.time()
        expired_keys = [key for key, ts in self.timestamps.items() 
                       if current_time - ts > self.ttl_seconds]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.token_tracker.pop(key, None)
            self.timestamps.pop(key, None)
    
    def get_cached_response(self, agent_name: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get cached response if available and not expired (thread-safe).
        
        Args:
            agent_name: Name of the agent
            input_data: Input data to check cache for
            
        Returns:
            Cached response if available, None otherwise
        """
        with self._lock:
            input_hash = self._generate_input_hash(input_data)
            cache_key = f"{agent_name}:{input_hash}"
            
            # Cleanup expired entries periodically
            if len(self.cache) > self.max_entries * 0.8:
                self._cleanup_expired()
            
            if cache_key in self.cache:
                if not self._is_expired(self.timestamps.get(cache_key, 0)):
                    log.info(f"🎯 Cache hit for '{agent_name}' - saved tokens")
                    return self.cache[cache_key]
                else:
                    # Remove expired entry
                    self.cache.pop(cache_key, None)
                    self.token_tracker.pop(cache_key, None)
                    self.timestamps.pop(cache_key, None)
            
            return None
    
    def cache_response(self, agent_name: str, input_data: Dict[str, Any], response: Dict[str, Any]):
        """
        Cache response for future use (thread-safe).
        
        Args:
            agent_name: Name of the agent
            input_data: Input data that generated the response
            response: Response to cache
        """
        with self._lock:
            input_hash = self._generate_input_hash(input_data)
            cache_key = f"{agent_name}:{input_hash}"
            
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_entries:
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                self.cache.pop(oldest_key, None)
                self.token_tracker.pop(oldest_key, None)
                self.timestamps.pop(oldest_key, None)
            
            self.cache[cache_key] = response
            self.token_tracker[cache_key] = input_hash
            self.timestamps[cache_key] = time.time()
            
            log.info(f"💾 Cached response for '{agent_name}'")
    
    def clear(self):
        """Clear all cache entries (thread-safe)."""
        with self._lock:
            self.cache.clear()
            self.token_tracker.clear()
            self.timestamps.clear()
            log.info("🧹 TokenAwareCache cleared")
    
    def invalidate(self, agent_name: str, input_data: Dict[str, Any]):
        """Invalidate a specific cache entry (thread-safe)."""
        with self._lock:
            input_hash = self._generate_input_hash(input_data)
            cache_key = f"{agent_name}:{input_hash}"
            self.cache.pop(cache_key, None)
            self.token_tracker.pop(cache_key, None)
            self.timestamps.pop(cache_key, None)
    
    def invalidate_agent(self, agent_name: str):
        """Invalidate all cache entries for a specific agent (thread-safe)."""
        with self._lock:
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"{agent_name}:")]
            for key in keys_to_remove:
                self.cache.pop(key, None)
                self.token_tracker.pop(key, None)
                self.timestamps.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics (thread-safe)."""
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for ts in self.timestamps.values()
                if current_time - ts > self.ttl_seconds
            )
            return {
                "size": len(self.cache),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "expired": expired_count,
                "utilization": len(self.cache) / self.max_entries if self.max_entries > 0 else 0
            }


# Global cache instance
_response_cache = TokenAwareCache()


def get_cached_response(agent_name: str, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get cached response for an agent."""
    return _response_cache.get_cached_response(agent_name, input_data)


def cache_response(agent_name: str, input_data: Dict[str, Any], response: Dict[str, Any]):
    """Cache response for an agent."""
    _response_cache.cache_response(agent_name, input_data, response)


# Global thread pool executor for running blocking I/O operations
# This prevents blocking the event loop during concurrent requests
# ✅ PERFORMANCE: Make thread pool size configurable via environment variable
# Default: 50 workers, can be increased for high-concurrency scenarios
_thread_pool_size = int(os.getenv("THREAD_POOL_MAX_WORKERS", "50"))
_io_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_thread_pool_size,
    thread_name_prefix="agent_io"
)
# Section 4 Issue 4: monitoring and optional backpressure
_io_executor_invocations = 0
_io_executor_invocations_lock = threading.Lock()
_io_executor_max_queue_depth = int(os.getenv("IO_EXECUTOR_MAX_QUEUE_DEPTH", "0"))  # 0 = disabled


def get_io_executor_stats() -> Dict[str, Any]:
    """Return best-effort stats for the global I/O executor (Section 4 Issue 4)."""
    with _io_executor_invocations_lock:
        invocations = _io_executor_invocations
    queue_size = 0
    try:
        # ThreadPoolExecutor has _work_queue; qsize() is best-effort
        if hasattr(_io_executor, "_work_queue"):
            queue_size = _io_executor._work_queue.qsize()
    except Exception:
        pass
    return {
        "max_workers": _thread_pool_size,
        "invocations_total": invocations,
        "queue_size": queue_size,
        "backpressure_max_queue_depth": _io_executor_max_queue_depth,
    }


async def run_blocking_io(func, *args, **kwargs):
    """
    Run a blocking I/O operation in a thread pool to avoid blocking the event loop.
    This is critical for concurrent request handling across all agents.
    
    Args:
        func: The blocking function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        The result of the function call
    """
    global _io_executor_invocations
    with _io_executor_invocations_lock:
        _io_executor_invocations += 1
    if _io_executor_max_queue_depth > 0:
        try:
            queue_size = _io_executor._work_queue.qsize() if hasattr(_io_executor, "_work_queue") else 0
            if queue_size >= _io_executor_max_queue_depth:
                raise RuntimeError(
                    f"IO executor overload: queue depth {queue_size} >= limit {_io_executor_max_queue_depth}"
                )
        except RuntimeError:
            raise
        except Exception:
            pass
    loop = asyncio.get_event_loop()
    # Use lambda instead of partial to avoid issues with unhashable types (like slices)
    return await loop.run_in_executor(_io_executor, lambda: func(*args, **kwargs))
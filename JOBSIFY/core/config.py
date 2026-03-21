"""
Centralized configuration management for all KA agents.

This module provides consistent configuration across all agents,
eliminating duplicate environment variable handling.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


# =============================================================================
# Graph Execution Limits
# =============================================================================
# Maximum number of steps the LangGraph can execute before terminating.
# This prevents infinite routing loops from consuming resources indefinitely.
# Set via GRAPH_RECURSION_LIMIT env var; default 50 provides headroom for
# complex multi-step flows while catching true infinite loops quickly.
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "75"))
GRAPH_RECURSION_WARNING_THRESHOLD = int(os.getenv("GRAPH_RECURSION_WARNING_THRESHOLD", "50"))


# =============================================================================
# Embedding Configuration (Single Source of Truth)
# =============================================================================
# All embedding-related modules should import these constants instead of
# hardcoding model names. This ensures consistency and enables easy model
# upgrades without code changes across multiple files.
#
# Used by: chromadb_manager, chromadb_optimizer, multi_view_embedder, search_gateway
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))


@dataclass
class AgentConfig:
    """Centralized configuration for all agents."""
    timeout_seconds: int = 60
    max_prompt_chars: int = 30000
    max_response_length: int = 50000
    max_chat_history_items: int = 50
    llm_model: str = "gemini"  # Using gemini-2.5-flash
    force_deterministic: bool = False
    confidence_threshold: float = 0.75
    adaptation_window: int = 50
    # Agent-specific fields
    max_interests_count: int = 20
    max_skills_count: int = 15
    max_recommendations_count: int = 30
    min_market_insights: int = 3
    min_course_recommendations: int = 5
    max_resume_length: int = 50000
    max_jd_length: int = 30000
    min_confidence_threshold: float = 0.5
    min_resume_length: int = 100
    max_tenant_memory_entries: int = 1000
    llm_retry_attempts: int = 3
    llm_base_backoff: float = 0.4
    # Groq resume parser specific fields
    max_completion_tokens: int = 8000
    min_clean_text_chars: int = 800
    trim_head_chars: int = 2000
    trim_tail_chars: int = 13000
    # Auto-discovery configuration
    auto_discovery_enabled: bool = True
    auto_discovery_min_results: int = 5
    auto_discovery_min_similarity: float = 0.4
    auto_discovery_max_per_request: int = 20
    auto_discovery_max_per_topic: int = 5
    # Resume scorer: assessment-based skills boost by difficulty
    skills_boost_easy: int = 2
    skills_boost_medium: int = 5
    skills_boost_hard: int = 8
    skills_boost_expert: int = 8
    skills_boost_default: int = 5  # when difficulty is missing/unknown
    # Add more as needed
    
    @classmethod
    def from_env(cls, agent_prefix: str) -> 'AgentConfig':
        """Create config from environment variables."""
        return cls(
            timeout_seconds=int(os.getenv(f"{agent_prefix}_TIMEOUT_SECONDS", "60")),
            max_prompt_chars=int(os.getenv(f"{agent_prefix}_MAX_PROMPT_CHARS", "30000")),
            max_response_length=int(os.getenv(f"{agent_prefix}_MAX_RESPONSE_LENGTH", "50000")),
            max_chat_history_items=int(os.getenv(f"{agent_prefix}_MAX_CHAT_HISTORY", "50")),
            llm_model=os.getenv(f"{agent_prefix}_LLM_MODEL", "gemini"),
            force_deterministic=os.getenv(f"{agent_prefix}_FORCE_DETERMINISTIC", "false").lower() == "true",
            confidence_threshold=float(os.getenv(f"{agent_prefix}_CONFIDENCE_THRESHOLD", "0.75")),
            adaptation_window=int(os.getenv(f"{agent_prefix}_ADAPTATION_WINDOW", "50")),
            max_interests_count=int(os.getenv(f"{agent_prefix}_MAX_INTERESTS_COUNT", "20")),
            max_skills_count=int(os.getenv(f"{agent_prefix}_MAX_SKILLS_COUNT", "15")),
            max_recommendations_count=int(os.getenv(f"{agent_prefix}_MAX_RECOMMENDATIONS_COUNT", "30")),
            min_market_insights=int(os.getenv(f"{agent_prefix}_MIN_MARKET_INSIGHTS", "3")),
            min_course_recommendations=int(os.getenv(f"{agent_prefix}_MIN_COURSE_RECOMMENDATIONS", "5")),
            max_resume_length=int(os.getenv(f"{agent_prefix}_MAX_RESUME_LENGTH", "50000")),
            max_jd_length=int(os.getenv(f"{agent_prefix}_MAX_JD_LENGTH", "30000")),
            min_confidence_threshold=float(os.getenv(f"{agent_prefix}_MIN_CONFIDENCE_THRESHOLD", "0.5")),
            min_resume_length=int(os.getenv(f"{agent_prefix}_MIN_RESUME_LENGTH", "100")),
            max_tenant_memory_entries=int(os.getenv(f"{agent_prefix}_MAX_TENANT_MEMORY_ENTRIES", "1000")),
            llm_retry_attempts=int(os.getenv(f"{agent_prefix}_LLM_RETRY_ATTEMPTS", "3")),
            llm_base_backoff=float(os.getenv(f"{agent_prefix}_LLM_BASE_BACKOFF", "0.4")),
            max_completion_tokens=int(os.getenv(f"{agent_prefix}_MAX_COMPLETION_TOKENS", "8000")),
            min_clean_text_chars=int(os.getenv(f"{agent_prefix}_MIN_CLEAN_TEXT_CHARS", "800")),
            trim_head_chars=int(os.getenv(f"{agent_prefix}_TRIM_HEAD_CHARS", "2000")),
            trim_tail_chars=int(os.getenv(f"{agent_prefix}_TRIM_TAIL_CHARS", "13000")),
            auto_discovery_enabled=os.getenv(f"{agent_prefix}_AUTO_DISCOVERY_ENABLED", "true").lower() == "true",
            auto_discovery_min_results=int(os.getenv(f"{agent_prefix}_AUTO_DISCOVERY_MIN_RESULTS", "5")),
            auto_discovery_min_similarity=float(os.getenv(f"{agent_prefix}_AUTO_DISCOVERY_MIN_SIMILARITY", "0.4")),
            auto_discovery_max_per_request=int(os.getenv(f"{agent_prefix}_AUTO_DISCOVERY_MAX_PER_REQUEST", "20")),
            auto_discovery_max_per_topic=int(os.getenv(f"{agent_prefix}_AUTO_DISCOVERY_MAX_PER_TOPIC", "5")),
            skills_boost_easy=int(os.getenv(f"{agent_prefix}_SKILLS_BOOST_EASY", "2")),
            skills_boost_medium=int(os.getenv(f"{agent_prefix}_SKILLS_BOOST_MEDIUM", "5")),
            skills_boost_hard=int(os.getenv(f"{agent_prefix}_SKILLS_BOOST_HARD", "8")),
            skills_boost_expert=int(os.getenv(f"{agent_prefix}_SKILLS_BOOST_EXPERT", "8")),
            skills_boost_default=int(os.getenv(f"{agent_prefix}_SKILLS_BOOST_DEFAULT", "5")),
        )


@dataclass
class SecurityConfig:
    """Security-related configuration."""
    max_input_length: int = 100000
    max_resume_length: int = 500000
    max_skills_count: int = 15
    max_experience_entries: int = 25
    max_education_entries: int = 20
    
    @classmethod
    def from_env(cls, agent_prefix: str) -> 'SecurityConfig':
        """Create security config from environment variables."""
        return cls(
            max_input_length=int(os.getenv(f"{agent_prefix}_MAX_INPUT_LENGTH", "100000")),
            max_resume_length=int(os.getenv(f"{agent_prefix}_MAX_RESUME_LENGTH", "500000")),
            max_skills_count=int(os.getenv(f"{agent_prefix}_MAX_SKILLS_COUNT", "15")),
            max_experience_entries=int(os.getenv(f"{agent_prefix}_MAX_EXPERIENCE_ENTRIES", "25")),
            max_education_entries=int(os.getenv(f"{agent_prefix}_MAX_EDUCATION_ENTRIES", "20"))
        )


# Pre-configured configs for common agents
RESUME_SCORER_CONFIG = AgentConfig.from_env("RESUME_SCORER")
SKILLS_PARSER_CONFIG = AgentConfig.from_env("SKILLS")
RESUME_ANALYSIS_CONFIG = AgentConfig.from_env("RESUME_ANALYSIS")
VALIDATE_RESUME_CONFIG = AgentConfig.from_env("RESUME_VALIDATION")
PERSONAL_INFO_CONFIG = AgentConfig.from_env("PERSONAL_INFO")
EDUCATION_CONFIG = AgentConfig.from_env("EDUCATION")
EXPERIENCE_CONFIG = AgentConfig.from_env("EXPERIENCE")
JOB_DESCRIPTION_CONFIG = AgentConfig.from_env("JOB_DESCRIPTION")
ASSESSMENT_CONFIG = AgentConfig.from_env("ASSESSMENT")
INTERVIEW_CONFIG = AgentConfig.from_env("INTERVIEW")
NOTIFICATION_CONFIG = AgentConfig.from_env("NOTIFICATION")
RESUME_SECTION_DETECTOR_CONFIG = AgentConfig.from_env("RESUME_SECTION_DETECTOR")
CAREER_MENTOR_CONFIG = AgentConfig.from_env("CAREER_MENTOR")


def get_agent_config(agent_name: str) -> AgentConfig:
    """Get configuration for a specific agent."""
    config_map = {
        "resume_scorer": RESUME_SCORER_CONFIG,
        "skills_parser": SKILLS_PARSER_CONFIG,
        "resume_analysis": RESUME_ANALYSIS_CONFIG,
        "validate_resume": VALIDATE_RESUME_CONFIG,
        "personal_info_parser": PERSONAL_INFO_CONFIG,
        "education_parser": EDUCATION_CONFIG,
        "experience_parser": EXPERIENCE_CONFIG,
        "job_description_parser": JOB_DESCRIPTION_CONFIG,
        "assessment_question_generator": ASSESSMENT_CONFIG,
        "assessment_evaluator": ASSESSMENT_CONFIG,
        "assessment_recommender": ASSESSMENT_CONFIG,
        "interview_agent": INTERVIEW_CONFIG,
        "notification_agent": NOTIFICATION_CONFIG,
        "resume_section_detector": RESUME_SECTION_DETECTOR_CONFIG,
        "groq_resume_parser": AgentConfig.from_env("GROQ_RESUME_PARSER"),
        "market_and_course_recommender": AgentConfig.from_env("MARKET_AND_COURSE_RECOMMENDER"),
        "career_mentor": CAREER_MENTOR_CONFIG,
        "skill_and_career_advisor": AgentConfig.from_env("SKILL_AND_CAREER_ADVISOR"),
        "enhanced_role_fit": AgentConfig.from_env("ENHANCED_ROLE_FIT"),
        "jd_enhancer": AgentConfig.from_env("JD_ENHANCER"),
        "interview_transcript_evaluator": AgentConfig.from_env("INTERVIEW_TRANSCRIPT_EVALUATOR"),
    }
    
    return config_map.get(agent_name, AgentConfig())


def get_security_config(agent_name: str) -> SecurityConfig:
    """Get security configuration for a specific agent."""
    return SecurityConfig.from_env(agent_name.upper().replace("_", "_"))


# =============================
# Domain maps for easy updates
# =============================

# Related advanced topics per primary topic (keys are lowercase)
ADVANCED_TOPICS_MAP: Dict[str, List[Dict[str, Any]]] = {
    "verilog hdl": [
        {
            "type": "multi",
            "topic": "SystemVerilog",
            "difficulty": "Hard",
            "num_questions": {"mcq": 6, "short": 2, "long": 1, "coding": 1},
        },
        {
            "type": "multi",
            "topic": "FPGA Design",
            "difficulty": "Hard",
            "num_questions": {"mcq": 5, "short": 2, "long": 1, "coding": 1},
        },
        {
            "type": "multi",
            "topic": "VLSI Design",
            "difficulty": "Hard",
            "num_questions": {"mcq": 5, "short": 2, "long": 1, "coding": 1},
        },
    ],
}


LEARNING_RESOURCES: Dict[str, Dict[str, Any]] = {
    "verilog": {
        "courses": [
            "Verilog HDL Basics on Coursera",
            "Digital Design with Verilog on Udemy",
            "FPGA Design with Verilog on edX",
        ],
        "books": [
            "Verilog HDL: A Guide to Digital Design and Synthesis by Samir Palnitkar",
            "Digital Design and Computer Architecture by Harris & Harris",
        ],
        "tools": [
            "ModelSim Simulator",
            "Vivado Design Suite",
            "Quartus Prime",
        ],
        "practice_sites": [
            "EDA Playground for Verilog",
            "HDLbits for Verilog practice",
            "Verilog Tutorial on ChipVerify",
        ],
    },
    "python": {
        "courses": [
            "Python for Everybody on Coursera",
            "Complete Python Bootcamp on Udemy",
            "Python Data Structures on Coursera",
        ],
        "books": [
            "Python Crash Course by Eric Matthes",
            "Automate the Boring Stuff with Python by Al Sweigart",
        ],
        "tools": [
            "PyCharm IDE",
            "Jupyter Notebook",
            "VS Code with Python extension",
        ],
        "practice_sites": [
            "LeetCode for Python problems",
            "HackerRank Python challenges",
            "Codewars Python kata",
        ],
    },
    "algorithms": {
        "courses": [
            "Algorithms Specialization on Coursera",
            "Data Structures and Algorithms on Udemy",
            "Algorithm Design and Analysis on edX",
        ],
        "books": [
            "Introduction to Algorithms by Cormen",
            "Algorithm Design Manual by Skiena",
        ],
        "tools": [
            "Visualgo for algorithm visualization",
            "Algorithm Visualizer",
            "Big-O Cheat Sheet",
        ],
        "practice_sites": [
            "LeetCode for algorithm problems",
            "HackerRank algorithm challenges",
            "Codeforces contests",
        ],
    },
}


def get_advanced_topics_map() -> Dict[str, List[Dict[str, Any]]]:
    return ADVANCED_TOPICS_MAP


def get_learning_resources() -> Dict[str, Dict[str, Any]]:
    return LEARNING_RESOURCES


# =============================
# Interview flow configuration
# =============================

_INTERVIEW_CONFIG_CACHE: Dict[str, Any] = {}


def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        if yaml:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        # Fallback: allow JSON if yaml not available
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def get_interview_config() -> Dict[str, Any]:
    global _INTERVIEW_CONFIG_CACHE
    if _INTERVIEW_CONFIG_CACHE:
        return _INTERVIEW_CONFIG_CACHE
    base_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(base_dir, "config", "interview.yml")
    _INTERVIEW_CONFIG_CACHE = _load_yaml(cfg_path)
    return _INTERVIEW_CONFIG_CACHE


def get_interview_flow_version() -> str:
    cfg = get_interview_config()
    return str(cfg.get("version", os.getenv("INTERVIEW_FLOW_VERSION", "v1")))


def get_interview_state_config() -> Dict[str, Any]:
    config = get_interview_config()
    return config.get("states", {})

def get_interview_scoring_config() -> Dict[str, Any]:
    """Loads the scoring configuration from interview.yml."""
    config = get_interview_config()
    return config.get("scoring", {})


def get_intelligent_staging_config() -> Dict[str, Any]:
    """Loads the intelligent staging configuration from interview.yml."""
    config = get_interview_config()
    return config.get("intelligent_staging", {})


def get_role_based_thresholds(role_level: str = None, years_experience: int = None) -> Dict[str, Any]:
    """
    Get role-based thresholds for intelligent staging.
    
    Args:
        role_level: Explicit role level ("junior", "mid", "senior")
        years_experience: Years of experience (used to infer role level if not provided)
    
    Returns:
        Dict with confidence_threshold, word_count_threshold, coverage_gap_weight
    """
    staging_config = get_intelligent_staging_config()
    
    # Determine role level if not provided
    if not role_level and years_experience is not None:
        if years_experience <= 2:
            role_level = "junior"
        elif years_experience <= 5:
            role_level = "mid"
        else:
            role_level = "senior"
    
    # Get role-specific thresholds
    role_thresholds = staging_config.get("role_thresholds", {})
    if role_level and role_level in role_thresholds:
        thresholds = role_thresholds[role_level].copy()
        # Remove description field if present
        thresholds.pop("description", None)
        return thresholds
    
    # Fallback to default thresholds
    return {
        "confidence_threshold": staging_config.get("confidence_threshold", 0.7),
        "word_count_threshold": staging_config.get("word_count_threshold", 50),
        "coverage_gap_weight": staging_config.get("coverage_gap_weight", 0.3)
    }


def get_interview_stage_instructions() -> Dict[str, str]:
    cfg = get_interview_config()
    return cfg.get("stage_instructions", {})


def get_interview_follow_up_guidance() -> Dict[str, str]:
    cfg = get_interview_config()
    return cfg.get("follow_up_guidance", {})


def get_interview_thresholds() -> Dict[str, Any]:
    cfg = get_interview_config()
    return cfg.get("thresholds", {})


def get_interview_feature_flags() -> Dict[str, bool]:
    cfg = get_interview_config()
    flags = cfg.get("feature_flags", {})
    # Env overrides
    for key in ["SAFETY_ENFORCED", "COVERAGE_GATING_ENABLED", "METRICS_ENABLED"]:
        env_val = os.getenv(key)
        if env_val is not None:
            flags[key] = env_val.lower() == "true"
    return flags


# =============================
# Role classification config
# =============================
_ROLES_CFG_CACHE: Dict[str, Any] = {}


def get_roles_config() -> Dict[str, Any]:
    global _ROLES_CFG_CACHE
    if _ROLES_CFG_CACHE:
        return _ROLES_CFG_CACHE
    base_dir = os.path.dirname(__file__)
    cfg_path = os.path.join(base_dir, "config", "roles.yml")
    _ROLES_CFG_CACHE = _load_yaml(cfg_path)
    return _ROLES_CFG_CACHE
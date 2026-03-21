"""
Canonical configuration adapter for interview agent.

Wraps core.config functions and provides typed getters for interview-specific
configuration with validation, defaults, and caching.
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

from core.config import (
    get_interview_config as _get_interview_config,
    get_interview_feature_flags as _get_interview_feature_flags,
    get_role_based_thresholds as _get_role_based_thresholds,
    get_interview_scoring_config as _get_interview_scoring_config,
    get_agent_config,
)

log = logging.getLogger(__name__)

# Cache for configuration
_CONFIG_CACHE: Optional[Dict[str, Any]] = None


@dataclass
class InterviewConfig:
    """Typed configuration for interview agent."""
    # Feature flags
    safety_enforced: bool = False
    coverage_gating_enabled: bool = False
    metrics_enabled: bool = False
    intelligent_flow: bool = True
    
    # Timeouts (in seconds)
    default_timeout: int = 60
    llm_timeout: int = 30
    llm_retry_attempts: int = 3
    llm_retry_delay: float = 1.0
    
    # Circuit breaker
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_cooldown_seconds: int = 60
    
    # Anti-repetition
    anti_repetition_window_size: int = 5
    anti_repetition_similarity_threshold: float = 0.7
    anti_repetition_ngram_size: int = 3
    
    # Question generation
    max_question_length: int = 300
    fallback_non_dup_question: str = "Can you walk me through a specific example from your experience?"
    
    # Negative intent
    negative_intent_consecutive_threshold: int = 2
    negative_intent_confidence_standard: float = 0.7
    negative_intent_confidence_short: float = 0.5
    negative_intent_confidence_very_short: float = 0.3
    negative_intent_loop_breaker_minimum: float = 0.3
    negative_intent_single_high_confidence_trigger: float = 0.6
    
    # Scoring weights (defaults, can be overridden by YAML)
    scoring_weights: Dict[str, Any] = None
    
    # Role-based thresholds (defaults)
    role_thresholds: Dict[str, Dict[str, Any]] = None


def get_interview_config() -> Dict[str, Any]:
    """
    Get interview configuration from core.config.
    
    Returns:
        Dict with interview configuration
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = _get_interview_config()
    return _CONFIG_CACHE


def get_config() -> InterviewConfig:
    """
    Get typed InterviewConfig object.
    Alias for backward compatibility.
    
    Returns:
        InterviewConfig dataclass instance
    """
    config_dict = get_interview_config()
    return InterviewConfig(**config_dict)


def get_interview_feature_flags() -> Dict[str, bool]:
    """
    Get interview feature flags.
    
    Returns:
        Dict with feature flags (all boolean values)
    """
    return _get_interview_feature_flags()


def get_timeouts() -> Dict[str, Any]:
    """
    Get timeout configuration.
    
    Returns:
        Dict with timeout settings:
        - default_timeout: Default timeout in seconds
        - llm_timeout: LLM operation timeout
        - llm_retry_attempts: Number of retry attempts
        - llm_retry_delay: Delay between retries
    """
    config = get_interview_config()
    agent_config = get_agent_config("interview_agent")
    
    return {
        "default_timeout": int(os.getenv("INTERVIEW_TIMEOUT_SECONDS", config.get("timeout_seconds", 60))),
        "llm_timeout": int(os.getenv("INTERVIEW_LLM_TIMEOUT", 30)),
        "llm_retry_attempts": agent_config.llm_retry_attempts,
        "llm_retry_delay": float(os.getenv("INTERVIEW_LLM_RETRY_DELAY", 1.0)),
    }


def get_scoring_weights() -> Dict[str, Any]:
    """
    Get scoring weights configuration.
    
    Returns:
        Dict with scoring weights from YAML config
    """
    return _get_interview_scoring_config()


def get_role_based_thresholds(
    role_level: Optional[str] = None,
    years_experience: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get role-based thresholds for intelligent staging.
    
    Args:
        role_level: Explicit role level ("junior", "mid", "senior")
        years_experience: Years of experience (used to infer role level if not provided)
    
    Returns:
        Dict with confidence_threshold, word_count_threshold, coverage_gap_weight
    """
    return _get_role_based_thresholds(role_level, years_experience)


def get_circuit_breaker_config() -> Dict[str, Any]:
    """
    Get circuit breaker configuration.
    
    Returns:
        Dict with:
        - failure_threshold: Number of failures before opening circuit
        - cooldown_seconds: Cooldown duration in seconds
    """
    config = get_interview_config()
    cb_config = config.get("circuit_breaker", {})
    
    return {
        "failure_threshold": int(os.getenv(
            "INTERVIEW_CB_FAILURE_THRESHOLD",
            cb_config.get("failure_threshold", 3)
        )),
        "cooldown_seconds": int(os.getenv(
            "INTERVIEW_CB_COOLDOWN_SECONDS",
            cb_config.get("cooldown_seconds", 60)
        )),
    }


def get_anti_repetition_config() -> Dict[str, Any]:
    """
    Get anti-repetition configuration.
    
    Returns:
        Dict with window_size, similarity_threshold, ngram_size
    """
    config = get_interview_config()
    ar_config = config.get("anti_repetition", {})
    
    return {
        "window_size": ar_config.get("window_size", 5),
        "similarity_threshold": ar_config.get("similarity_threshold", 0.7),
        "ngram_size": ar_config.get("ngram_size", 3),
    }


def validate_config() -> None:
    """
    Validate configuration and fail fast on missing required keys.
    
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    try:
        config = get_interview_config()
        if not config:
            raise ValueError("Interview configuration is empty")
        
        # Validate feature flags
        feature_flags = get_interview_feature_flags()
        if not isinstance(feature_flags, dict):
            raise ValueError("Feature flags must be a dictionary")
        
        # Validate timeouts
        timeouts = get_timeouts()
        if timeouts["default_timeout"] <= 0:
            raise ValueError("default_timeout must be positive")
        if timeouts["llm_retry_attempts"] < 0:
            raise ValueError("llm_retry_attempts must be non-negative")
        
        # Validate circuit breaker
        cb_config = get_circuit_breaker_config()
        if cb_config["failure_threshold"] <= 0:
            raise ValueError("circuit_breaker.failure_threshold must be positive")
        if cb_config["cooldown_seconds"] <= 0:
            raise ValueError("circuit_breaker.cooldown_seconds must be positive")
        
        # Validate anti-repetition
        ar_config = get_anti_repetition_config()
        if not 0.0 <= ar_config["similarity_threshold"] <= 1.0:
            raise ValueError("anti_repetition.similarity_threshold must be between 0.0 and 1.0")
        if ar_config["window_size"] <= 0:
            raise ValueError("anti_repetition.window_size must be positive")
        
        log.info("Configuration validation passed")
        
    except Exception as e:
        log.error(f"Configuration validation failed: {e}")
        raise ValueError(f"Invalid configuration: {e}") from e


# Validate on import
try:
    validate_config()
except Exception as e:
    log.warning(f"Configuration validation failed on import: {e}. Some defaults may be used.")


# ============================================================================
# YAML Prompt Loading
# ============================================================================

_PROMPTS_CACHE: Optional[Dict[str, Any]] = None


def load_prompts() -> Dict[str, Any]:
    """
    Load all prompts from prompts.yaml.
    
    Returns:
        Dict with all prompt templates
    """
    global _PROMPTS_CACHE
    if _PROMPTS_CACHE is not None:
        return _PROMPTS_CACHE
    
    import yaml
    import os
    
    try:
        base_dir = os.path.dirname(__file__)
        prompts_path = os.path.join(base_dir, "constants", "prompts.yaml")
        
        if not os.path.exists(prompts_path):
            log.warning(f"Prompts file not found: {prompts_path}")
            return {}
        
        with open(prompts_path, 'r', encoding='utf-8') as f:
            _PROMPTS_CACHE = yaml.safe_load(f) or {}
        
        log.debug(f"Loaded {len(_PROMPTS_CACHE)} prompts from YAML")
        return _PROMPTS_CACHE
        
    except Exception as e:
        log.error(f"Failed to load prompts from YAML: {e}")
        return {}


def get_prompt(prompt_name: str, **kwargs) -> str:
    """
    Get a prompt template by name and format it with provided kwargs.
    
    Args:
        prompt_name: Name of the prompt template
        **kwargs: Variables to format into the prompt
        
    Returns:
        Formatted prompt string
    """
    prompts = load_prompts()
    template = prompts.get(prompt_name, "")
    
    if not template:
        log.warning(f"Prompt template '{prompt_name}' not found in prompts.yaml")
        return ""
    
    try:
        return template.format(**kwargs)
    except KeyError as e:
        log.error(f"Missing variable {e} for prompt '{prompt_name}'")
        return template


def get_prompt_template(prompt_name: str) -> str:
    """
    Get a prompt template by name without formatting.
    
    Args:
        prompt_name: Name of the prompt template
        
    Returns:
        Raw prompt template string
    """
    prompts = load_prompts()
    return prompts.get(prompt_name, "")


def get_interview_state_config() -> Optional[Dict[str, Any]]:
    """
    Get interview state configuration.
    Returns a dict with state transition rules and configuration.
    
    Note: This function returns an empty dict by default since state_config
    is not part of InterviewConfig. State configurations should be loaded
    from YAML files or environment variables if needed.
    """
    try:
        # Try to get from InterviewConfig if state_config attribute exists
        config = get_config()
        if hasattr(config, 'state_config') and config.state_config:
            return config.state_config
    except Exception as e:
        log.debug(f"Could not get state config from InterviewConfig: {e}")
    
    # Return empty dict - state_manager will use defaults
    return {}


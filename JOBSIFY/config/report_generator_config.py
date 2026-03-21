"""
Configuration for Report Generator
==================================

This module provides configuration options for the report generator,
allowing flexibility between LLM-based and rule-based approaches.
"""

from typing import Dict, Any


class ReportGeneratorConfig:
    """Configuration for report generation strategies."""
    
    # ========================================================================
    # DOMAIN DETECTION
    # ========================================================================
    
    # Use LLM for intelligent domain detection (vs. keyword matching)
    USE_LLM_DOMAIN_DETECTION = True
    
    # Fallback to keyword matching if LLM fails
    FALLBACK_TO_KEYWORDS = True
    
    # Cache LLM domain detection results to reduce API calls
    CACHE_DOMAIN_DETECTION = True
    CACHE_TTL_SECONDS = 3600  # 1 hour
    
    # ========================================================================
    # PERFORMANCE ASSESSMENT
    # ========================================================================
    
    # Use LLM for adaptive performance level assessment
    USE_LLM_PERFORMANCE_LEVELS = True  # ✅ Implemented
    
    # Hard-coded performance thresholds (current approach)
    PERFORMANCE_THRESHOLDS = {
        "excellent": 80,
        "good": 60,
        "satisfactory": 40,
        "needs_development": 0
    }
    
    # ========================================================================
    # FEEDBACK GENERATION
    # ========================================================================
    
    # Use LLM for personalized feedback generation
    USE_LLM_FEEDBACK = True  # ✅ Implemented
    
    # Use template-based feedback (fallback)
    USE_TEMPLATE_FEEDBACK = False  # Deprecated, LLM is now primary
    
    # ========================================================================
    # EXPERIENCE CLASSIFICATION (Future Enhancement)
    # ========================================================================
    
    # Use LLM to assess experience level (vs. years-based)
    USE_LLM_EXPERIENCE_LEVEL = False  # Not yet implemented
    
    # Years-based experience thresholds (current approach)
    EXPERIENCE_THRESHOLDS = {
        "entry": (0, 2),
        "mid": (2, 5),
        "senior": (5, 100)
    }
    
    # ========================================================================
    # RESOURCE RECOMMENDATIONS (Future Enhancement)
    # ========================================================================
    
    # Use LLM for curated resource recommendations
    USE_LLM_RESOURCES = False  # Not yet implemented
    
    # ========================================================================
    # GENERIC TEST DETECTION
    # ========================================================================
    
    # Keywords to identify generic/behavioral tests
    GENERIC_TEST_KEYWORDS = [
        "personality",
        "psychometric",
        "communication",
        "behavioral"
    ]
    
    # ========================================================================
    # LLM SETTINGS
    # ========================================================================
    
    # Maximum retries for LLM calls
    MAX_LLM_RETRIES = 3
    
    # Timeout for LLM calls (seconds)
    LLM_TIMEOUT_SECONDS = 30
    
    # ========================================================================
    # DEBUGGING & MONITORING
    # ========================================================================
    
    # Enable verbose logging
    VERBOSE_LOGGING = True
    
    # Track LLM usage metrics
    TRACK_LLM_METRICS = True
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        return {
            "domain_detection": {
                "use_llm": cls.USE_LLM_DOMAIN_DETECTION,
                "fallback_to_keywords": cls.FALLBACK_TO_KEYWORDS,
                "cache_enabled": cls.CACHE_DOMAIN_DETECTION,
                "cache_ttl": cls.CACHE_TTL_SECONDS
            },
            "performance_assessment": {
                "use_llm": cls.USE_LLM_PERFORMANCE_LEVELS,
                "thresholds": cls.PERFORMANCE_THRESHOLDS
            },
            "feedback_generation": {
                "use_llm": cls.USE_LLM_FEEDBACK,
                "use_templates": cls.USE_TEMPLATE_FEEDBACK
            },
            "experience_classification": {
                "use_llm": cls.USE_LLM_EXPERIENCE_LEVEL,
                "thresholds": cls.EXPERIENCE_THRESHOLDS
            },
            "resource_recommendations": {
                "use_llm": cls.USE_LLM_RESOURCES
            },
            "generic_test_detection": {
                "keywords": cls.GENERIC_TEST_KEYWORDS
            },
            "llm_settings": {
                "max_retries": cls.MAX_LLM_RETRIES,
                "timeout": cls.LLM_TIMEOUT_SECONDS
            },
            "monitoring": {
                "verbose_logging": cls.VERBOSE_LOGGING,
                "track_metrics": cls.TRACK_LLM_METRICS
            }
        }
    
    @classmethod
    def update_config(cls, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
    
    @classmethod
    def reset_to_defaults(cls):
        """Reset all configuration to default values."""
        cls.USE_LLM_DOMAIN_DETECTION = True
        cls.FALLBACK_TO_KEYWORDS = True
        cls.CACHE_DOMAIN_DETECTION = True
        cls.CACHE_TTL_SECONDS = 3600
        cls.USE_LLM_PERFORMANCE_LEVELS = True  # ✅ Now implemented
        cls.USE_LLM_FEEDBACK = True  # ✅ Now implemented
        cls.USE_TEMPLATE_FEEDBACK = False
        cls.USE_LLM_EXPERIENCE_LEVEL = False
        cls.USE_LLM_RESOURCES = False
        cls.VERBOSE_LOGGING = True
        cls.TRACK_LLM_METRICS = True





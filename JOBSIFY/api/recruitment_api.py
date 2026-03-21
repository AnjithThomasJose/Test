"""
Recruitment API Models and Adapter

This module provides Pydantic models and adapter functions for the recruitment search feature.
The actual endpoints are defined in app.py following the codebase pattern.
"""

import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl, field_validator

from features.recruitment.adapter import RecruitmentAdapter
from core.security import validate_callback_url
from settings import settings

log = logging.getLogger(__name__)

# Initialize adapter (singleton)
_adapter = None


def get_adapter() -> RecruitmentAdapter:
    """Get or create recruitment adapter instance"""
    global _adapter
    if _adapter is None:
        _adapter = RecruitmentAdapter(
            collection_name="resume",  # Changed from "candidates_v1" to "resume"
            model="gemini-2.5-flash"
        )
        log.info("RecruitmentAdapter initialized with collection: resume")
    return _adapter


# ==================== REQUEST/RESPONSE MODELS ====================

class CandidateSearchRequest(BaseModel):
    """Request model for candidate search"""
    query: str = Field(
        ...,
        description="Natural language query from recruiter",
        example="Senior Python Developer in Dubai"
    )
    top_k: int = Field(
        default=None,
        ge=1,
        le=None,  # Will be set dynamically from settings
        description=f"Number of candidates to return (1-{settings.RECRUITMENT_MAX_TOP_K}, default: {settings.RECRUITMENT_DEFAULT_TOP_K})"
    )
    callback_url: Optional[HttpUrl] = Field(
        default=None,
        description="Optional callback URL to receive search results asynchronously"
    )
    
    @validator('top_k', pre=True, always=True)
    def set_default_top_k(cls, v):
        """Set default top_k from settings if not provided"""
        if v is None:
            return settings.RECRUITMENT_DEFAULT_TOP_K
        return v
    
    @validator('top_k')
    def validate_top_k_range(cls, v):
        """Validate top_k is within allowed range"""
        max_k = settings.RECRUITMENT_MAX_TOP_K
        if v < 1:
            raise ValueError(f"top_k must be at least 1, got {v}")
        if v > max_k:
            raise ValueError(f"top_k cannot exceed {max_k}, got {v}")
        return v
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 500:
            raise ValueError("Query too long (max 500 characters)")
        return v.strip()
    
    @field_validator('callback_url')
    @classmethod
    def validate_callback_url_field(cls, v):
        """Validate callback URL if provided"""
        if v:
            is_valid, error_msg = validate_callback_url(str(v))
            if not is_valid:
                raise ValueError(f"Invalid callback URL: {error_msg}")
        return v


class CandidateMatchResponse(BaseModel):
    """Response model for a single candidate match"""
    candidate_id: str
    candidate_name: str
    match_score: float
    candidate_skills: List[str]  # All candidate skills
    total_experience_years: float
    current_location: str
    seniority_level: str
    education_level: str


class CandidateSearchResponse(BaseModel):
    """Response model for candidate search"""
    query: str
    parsed_query: Dict[str, Any]
    matches: List[CandidateMatchResponse]
    total_found: int
    processing_time_ms: int
    search_strategy: str
    filters_applied: Dict[str, Any]
    avg_match_score: float
    top_match_score: float
    callback_url: Optional[str] = Field(
        default=None,
        description="Callback URL if provided (for confirmation)"
    )

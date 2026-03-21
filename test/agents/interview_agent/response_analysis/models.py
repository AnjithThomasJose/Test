"""
Pydantic models for response analysis.

Defines the ResponseAnalysis model with strict validation.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, validator


class ResponseAnalysis(BaseModel):
    """
    Comprehensive response analysis model.
    
    All fields are validated and normalized.
    """
    type: Literal["technical", "behavioral", "brief", "general"] = Field(
        default="general",
        description="Response type classification"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the analysis"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Extracted keywords from the response"
    )
    skills: List[str] = Field(
        default_factory=list,
        description="Extracted skills from the response"
    )
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        default="neutral",
        description="Sentiment classification"
    )
    sentiment_polarity: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Sentiment polarity score (-1 to 1)"
    )
    volatility: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sentiment volatility score"
    )
    engagement_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Engagement score (0 to 1)"
    )
    evidence: str = Field(
        default="",
        description="Evidence or reasoning for the analysis"
    )
    
    @validator("keywords", "skills", pre=True)
    def normalize_lists(cls, v):
        """Normalize lists: lowercase, strip, deduplicate."""
        if not isinstance(v, list):
            return []
        normalized = []
        seen = set()
        for item in v:
            if isinstance(item, str):
                item_lower = item.lower().strip()
                if item_lower and item_lower not in seen:
                    normalized.append(item_lower)
                    seen.add(item_lower)
        return normalized
    
    @validator("evidence", pre=True)
    def normalize_evidence(cls, v):
        """Normalize evidence string."""
        if not isinstance(v, str):
            return ""
        return v.strip()
    
    @validator("type", pre=True)
    def normalize_type(cls, v):
        """Normalize type to lowercase."""
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ["technical", "behavioral", "brief", "general"]:
                return v_lower
        return "general"
    
    @validator("sentiment", pre=True)
    def normalize_sentiment(cls, v):
        """Normalize sentiment to lowercase."""
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ["positive", "neutral", "negative"]:
                return v_lower
        return "neutral"
    
    class Config:
        """Pydantic config."""
        extra = "forbid"  # Reject extra fields
        validate_assignment = True


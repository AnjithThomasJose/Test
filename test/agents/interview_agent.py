# interview.py (LangGraph State Machine Implementation)

import json
import os
import re
import hashlib
import random
import time
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from fastapi import Request, HTTPException
from pydantic import BaseModel, Field, field_validator, ValidationError
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

# Import your existing LLM invoker
from models.llm_invoker import invoke_llm, invoke_structured_llm
from core.model_registry import TaskType
from settings import settings

# Import ChromaDB integration
from .response_analyzer import hybrid_response_analyzer
from .interview_chroma import interview_chroma
from core.utils import run_blocking_io as _rbi

# Import utility functions
from utils.interview_utils import (
    mask_sensitive_data,
    log_event,
    _redact_pii,
    _policy_compliant,
    _rewrite_to_safe,
    extract_and_anonymize_candidate_info,
    anonymize_conversation_history,
    count_questions_in_history,
    get_previous_questions,
    sanitize_history,
    sanitize_job_details,
    get_candidate_name,
    get_job_title,
    get_candidate_skills,
    get_experience_years,
    get_personalized_opener,
    get_topic_focused_opener,
    get_contextual_suggestions,
    get_problem_solving_keywords,
    build_context_from_payload,
)

# -------------------------
# Setup Logging
# -------------------------
# Structured logging for negative intent detection
log = logging.getLogger(__name__)

# Cap hang time on direct Gemini structured-output calls (semaphore / event-loop safety)
_INTERVIEW_LLM_TIMEOUT = float(os.getenv("INTERVIEW_LLM_TIMEOUT_SECONDS", "90"))


# =============================================================================
# Pydantic Models for Structured LLM Output (Issue 5.1 Migration + 5.3 Validation)
# =============================================================================
# These models enable with_structured_output() for reliable JSON parsing.
# Prefer structured output for new LLM calls; avoid ad-hoc json.loads on raw content.
#
# Issue 5.3: Business-rule validation is enforced via:
# - Field constraints (ge, le, min_length, max_length)
# - @field_validator decorators for enum validation and sanitization
# - @model_validator for cross-field consistency checks

# Valid enum values for business rule validation
VALID_RESPONSE_TYPES = {"UNRESPONSIVE", "TECHNICAL", "BEHAVIORAL", "BRIEF", "GENERAL"}
VALID_RESPONSE_TYPES_LOWER = {"unresponsive", "technical", "behavioral", "brief", "general"}
VALID_SENTIMENTS = {"positive", "negative", "neutral"}
VALID_INTENTS = {"LACK_OF_KNOWLEDGE", "EXIT_INTENT", "TOPIC_SWITCH", "NONE"}
VALID_KEYWORD_RESPONSE_TYPES = {"TECHNICAL", "BEHAVIORAL", "GENERAL"}


def _clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value to a range (used when LLM returns out-of-bounds values)."""
    return max(min_val, min(max_val, value))


def _sanitize_string(value: str, max_length: int = 500) -> str:
    """Sanitize and truncate a string value."""
    if not value:
        return ""
    return value.strip()[:max_length]


def _sanitize_list(items: List[str], max_items: int = 20, max_item_length: int = 100) -> List[str]:
    """Sanitize a list of strings: truncate items and limit count."""
    if not items:
        return []
    return [item.strip()[:max_item_length] for item in items[:max_items] if item and item.strip()]


class CoverageDetectionResult(BaseModel):
    """Structured output for interview response coverage detection."""
    technical_skills: bool = Field(default=False, description="Discusses technical tools, technologies, or capabilities")
    experience: bool = Field(default=False, description="Discusses past work experience or projects")
    problem_solving: bool = Field(default=False, description="Discusses problem-solving approaches or challenges")
    collaboration: bool = Field(default=False, description="Discusses teamwork or stakeholder interaction")
    constraints: bool = Field(default=False, description="Discusses limitations, deadlines, or compliance")
    outcomes: bool = Field(default=False, description="Discusses results, metrics, or impact")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence in the detection")
    
    @field_validator('confidence', mode='before')
    @classmethod
    def clamp_confidence(cls, v):
        """Clamp confidence to valid range if LLM returns out-of-bounds value."""
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.7  # Default if invalid type


class PerSkillItem(BaseModel):
    """A single skill evaluation item."""
    skill: str = Field(description="Name of the skill or competency")
    score: int = Field(ge=0, le=10, description="Score for this skill (0-10)")
    evidence: str = Field(default="", description="Evidence from the response")
    
    @field_validator('skill', mode='before')
    @classmethod
    def sanitize_skill(cls, v):
        """Ensure skill name is non-empty and reasonable length."""
        if not v or not isinstance(v, str):
            return "unknown_skill"
        return _sanitize_string(v, max_length=80)
    
    @field_validator('score', mode='before')
    @classmethod
    def clamp_score(cls, v):
        """Clamp score to valid range."""
        if isinstance(v, (int, float)):
            return int(_clamp(float(v), 0, 10))
        return 5  # Default mid-score if invalid
    
    @field_validator('evidence', mode='before')
    @classmethod
    def sanitize_evidence(cls, v):
        """Truncate evidence to reasonable length."""
        return _sanitize_string(v or "", max_length=400)


class TechnicalSkillsEvaluationResult(BaseModel):
    """Structured output for technical skills evaluation."""
    skills: List[PerSkillItem] = Field(default_factory=list, description="List of evaluated skills")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in the evaluation")
    notes: str = Field(default="", description="Additional notes")
    
    @field_validator('confidence', mode='before')
    @classmethod
    def clamp_confidence(cls, v):
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.0
    
    @field_validator('notes', mode='before')
    @classmethod
    def sanitize_notes(cls, v):
        return _sanitize_string(v or "", max_length=400)
    
    @field_validator('skills', mode='before')
    @classmethod
    def limit_skills(cls, v):
        """Limit number of skills to prevent excessive output."""
        if isinstance(v, list):
            return v[:20]  # Max 20 skills
        return []


class ResponseClassificationResult(BaseModel):
    """Structured output for interview response classification."""
    type: str = Field(description="Response type: UNRESPONSIVE, TECHNICAL, BEHAVIORAL, BRIEF, or GENERAL")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in the classification")
    keywords: List[str] = Field(default_factory=list, description="Technical or behavioral keywords mentioned")
    sentiment: str = Field(default="neutral", description="Sentiment: positive, negative, or neutral")
    engagement_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Engagement level")
    reasoning: str = Field(default="", description="Brief explanation of classification")
    
    @field_validator('type', mode='before')
    @classmethod
    def validate_type(cls, v):
        """Ensure type is a valid response type."""
        if not v or not isinstance(v, str):
            return "GENERAL"
        v_upper = v.upper().strip()
        if v_upper in VALID_RESPONSE_TYPES:
            return v_upper
        return "GENERAL"  # Default fallback
    
    @field_validator('confidence', 'engagement_score', mode='before')
    @classmethod
    def clamp_scores(cls, v):
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.5
    
    @field_validator('sentiment', mode='before')
    @classmethod
    def validate_sentiment(cls, v):
        if not v or not isinstance(v, str):
            return "neutral"
        v_lower = v.lower().strip()
        if v_lower in VALID_SENTIMENTS:
            return v_lower
        return "neutral"
    
    @field_validator('keywords', mode='before')
    @classmethod
    def sanitize_keywords(cls, v):
        return _sanitize_list(v or [], max_items=30, max_item_length=50)
    
    @field_validator('reasoning', mode='before')
    @classmethod
    def sanitize_reasoning(cls, v):
        return _sanitize_string(v or "", max_length=300)


class IntelligentResponseAnalysisResult(BaseModel):
    """Structured output for deep response analysis during interviews."""
    type: str = Field(default="general", description="Response type: technical, behavioral, brief, general, or unresponsive")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in the analysis")
    keywords: List[str] = Field(default_factory=list, description="Key technical/behavioral keywords")
    sentiment: str = Field(default="neutral", description="Sentiment: positive, negative, or neutral")
    length_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Length appropriateness score")
    engagement_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Engagement level")
    word_count: int = Field(default=0, ge=0, description="Word count of the response")
    context_understanding: str = Field(default="", description="What the candidate is really saying")
    follow_up_suggestions: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    strengths_mentioned: List[str] = Field(default_factory=list, description="Skills/achievements highlighted")
    areas_to_explore: List[str] = Field(default_factory=list, description="Topics needing deeper investigation")
    
    @field_validator('type', mode='before')
    @classmethod
    def validate_type(cls, v):
        if not v or not isinstance(v, str):
            return "general"
        v_lower = v.lower().strip()
        if v_lower in VALID_RESPONSE_TYPES_LOWER:
            return v_lower
        return "general"
    
    @field_validator('confidence', 'length_score', 'engagement_score', mode='before')
    @classmethod
    def clamp_scores(cls, v):
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.5
    
    @field_validator('sentiment', mode='before')
    @classmethod
    def validate_sentiment(cls, v):
        if not v or not isinstance(v, str):
            return "neutral"
        v_lower = v.lower().strip()
        return v_lower if v_lower in VALID_SENTIMENTS else "neutral"
    
    @field_validator('word_count', mode='before')
    @classmethod
    def validate_word_count(cls, v):
        if isinstance(v, (int, float)):
            return max(0, int(v))
        return 0
    
    @field_validator('keywords', 'follow_up_suggestions', 'strengths_mentioned', 'areas_to_explore', mode='before')
    @classmethod
    def sanitize_lists(cls, v):
        return _sanitize_list(v or [], max_items=15, max_item_length=100)
    
    @field_validator('context_understanding', mode='before')
    @classmethod
    def sanitize_context(cls, v):
        return _sanitize_string(v or "", max_length=500)


class KeywordExtractionResult(BaseModel):
    """Structured output for keyword extraction from interview responses."""
    technical_keywords: List[str] = Field(default_factory=list, description="Technical keywords mentioned")
    action_verbs: List[str] = Field(default_factory=list, description="Action verbs indicating experience")
    response_type: str = Field(default="GENERAL", description="Response type: TECHNICAL, BEHAVIORAL, or GENERAL")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in the extraction")
    engagement_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Engagement level")
    
    @field_validator('response_type', mode='before')
    @classmethod
    def validate_type(cls, v):
        if not v or not isinstance(v, str):
            return "GENERAL"
        v_upper = v.upper().strip()
        return v_upper if v_upper in VALID_KEYWORD_RESPONSE_TYPES else "GENERAL"
    
    @field_validator('confidence', 'engagement_score', mode='before')
    @classmethod
    def clamp_scores(cls, v):
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.5
    
    @field_validator('technical_keywords', 'action_verbs', mode='before')
    @classmethod
    def sanitize_lists(cls, v):
        return _sanitize_list(v or [], max_items=20, max_item_length=50)


class AlternativeRole(BaseModel):
    """An alternative role suggestion."""
    role: str = Field(description="Alternative role name")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence for this alternative")
    
    @field_validator('role', mode='before')
    @classmethod
    def sanitize_role(cls, v):
        if not v or not isinstance(v, str):
            return "Unknown Role"
        return _sanitize_string(v, max_length=100)
    
    @field_validator('confidence', mode='before')
    @classmethod
    def clamp_confidence(cls, v):
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.0


class RoleClassificationLLMResult(BaseModel):
    """Structured output for LLM-driven role classification."""
    role_category: str = Field(default="GENERAL_PROFESSIONAL", description="Specific professional domain")
    confidence_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence in classification")
    reasoning: str = Field(default="", description="Brief explanation of classification")
    matched_criteria: List[str] = Field(default_factory=list, description="Evidence that matched")
    alternative_roles: List[AlternativeRole] = Field(default_factory=list, description="Alternative role suggestions")
    
    @field_validator('role_category', mode='before')
    @classmethod
    def sanitize_category(cls, v):
        if not v or not isinstance(v, str):
            return "GENERAL_PROFESSIONAL"
        return _sanitize_string(v, max_length=100) or "GENERAL_PROFESSIONAL"
    
    @field_validator('confidence_score', mode='before')
    @classmethod
    def clamp_confidence(cls, v):
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.7
    
    @field_validator('reasoning', mode='before')
    @classmethod
    def sanitize_reasoning(cls, v):
        return _sanitize_string(v or "", max_length=500)
    
    @field_validator('matched_criteria', mode='before')
    @classmethod
    def sanitize_criteria(cls, v):
        return _sanitize_list(v or [], max_items=10, max_item_length=200)
    
    @field_validator('alternative_roles', mode='before')
    @classmethod
    def limit_alternatives(cls, v):
        if isinstance(v, list):
            return v[:5]  # Max 5 alternative roles
        return []


class RubricCategory(BaseModel):
    """A single rubric category."""
    weight: float = Field(ge=0.0, le=1.0, description="Weight of this category")
    max_score: int = Field(default=10, ge=1, le=100, description="Maximum score for this category")
    description: str = Field(default="", description="What this category measures")
    
    @field_validator('weight', mode='before')
    @classmethod
    def clamp_weight(cls, v):
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.2  # Default weight
    
    @field_validator('max_score', mode='before')
    @classmethod
    def validate_max_score(cls, v):
        if isinstance(v, (int, float)):
            return int(_clamp(float(v), 1, 100))
        return 10
    
    @field_validator('description', mode='before')
    @classmethod
    def sanitize_description(cls, v):
        return _sanitize_string(v or "", max_length=300)


class EvaluationRubricResult(BaseModel):
    """Structured output for evaluation rubric generation."""
    categories: Dict[str, RubricCategory] = Field(default_factory=dict, description="Evaluation categories")
    keywords: Dict[str, List[str]] = Field(default_factory=dict, description="Keywords per category")
    
    @field_validator('categories', mode='before')
    @classmethod
    def limit_categories(cls, v):
        """Limit number of categories and ensure valid structure."""
        if not isinstance(v, dict):
            return {}
        # Limit to 10 categories max
        return dict(list(v.items())[:10])
    
    @field_validator('keywords', mode='before')
    @classmethod
    def sanitize_keywords(cls, v):
        """Sanitize keyword lists per category."""
        if not isinstance(v, dict):
            return {}
        result = {}
        for key, words in list(v.items())[:10]:  # Max 10 categories
            if isinstance(words, list):
                result[str(key)[:50]] = _sanitize_list(words, max_items=20, max_item_length=50)
        return result


class ResponseAnalysisForQuestion(BaseModel):
    """Response analysis embedded in question generation."""
    type: str = Field(default="general", description="Response type")
    engagement_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Engagement level")
    topics_mentioned: List[str] = Field(default_factory=list, description="Topics mentioned")
    skills_demonstrated: List[str] = Field(default_factory=list, description="Skills demonstrated")
    strengths: List[str] = Field(default_factory=list, description="Strengths identified")
    areas_to_explore: List[str] = Field(default_factory=list, description="Areas to explore")
    
    @field_validator('type', mode='before')
    @classmethod
    def validate_type(cls, v):
        if not v or not isinstance(v, str):
            return "general"
        v_lower = v.lower().strip()
        return v_lower if v_lower in VALID_RESPONSE_TYPES_LOWER else "general"
    
    @field_validator('engagement_score', mode='before')
    @classmethod
    def clamp_score(cls, v):
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.5
    
    @field_validator('topics_mentioned', 'skills_demonstrated', 'strengths', 'areas_to_explore', mode='before')
    @classmethod
    def sanitize_lists(cls, v):
        return _sanitize_list(v or [], max_items=10, max_item_length=100)


class SingleCallQuestionLLMResult(BaseModel):
    """Structured output for single-call question generation."""
    question: str = Field(description="The generated interview question")
    response_analysis: ResponseAnalysisForQuestion = Field(default_factory=ResponseAnalysisForQuestion)
    engagement_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Engagement score")
    suggested_follow_ups: List[str] = Field(default_factory=list, description="Suggested follow-up questions")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence in the question")
    reasoning: str = Field(default="", description="Reasoning for the question")
    topics_mentioned: List[str] = Field(default_factory=list, description="Topics mentioned")
    skills_demonstrated: List[str] = Field(default_factory=list, description="Skills demonstrated")
    strengths: List[str] = Field(default_factory=list, description="Strengths identified")
    areas_to_explore: List[str] = Field(default_factory=list, description="Areas needing exploration")
    
    @field_validator('question', mode='before')
    @classmethod
    def validate_question(cls, v):
        """Ensure question is non-empty and reasonable length."""
        if not v or not isinstance(v, str) or not v.strip():
            return "Can you tell me more about your experience?"
        return _sanitize_string(v, max_length=500)
    
    @field_validator('engagement_score', 'confidence', mode='before')
    @classmethod
    def clamp_scores(cls, v):
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.5
    
    @field_validator('reasoning', mode='before')
    @classmethod
    def sanitize_reasoning(cls, v):
        return _sanitize_string(v or "", max_length=300)
    
    @field_validator('suggested_follow_ups', 'topics_mentioned', 'skills_demonstrated', 'strengths', 'areas_to_explore', mode='before')
    @classmethod
    def sanitize_lists(cls, v):
        return _sanitize_list(v or [], max_items=10, max_item_length=200)


class IntentDetectionResult(BaseModel):
    """Structured output for candidate intent detection during interviews."""
    intent: str = Field(default="NONE", description="Detected intent: LACK_OF_KNOWLEDGE, EXIT_INTENT, TOPIC_SWITCH, or NONE")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in the detection")
    reasoning: str = Field(default="", description="Brief explanation of why this intent was detected")
    new_topic: Optional[str] = Field(default=None, description="New topic if intent is TOPIC_SWITCH")
    
    @field_validator('intent', mode='before')
    @classmethod
    def validate_intent(cls, v):
        """Ensure intent is a valid value."""
        if not v or not isinstance(v, str):
            return "NONE"
        v_upper = v.upper().strip()
        return v_upper if v_upper in VALID_INTENTS else "NONE"
    
    @field_validator('confidence', mode='before')
    @classmethod
    def clamp_confidence(cls, v):
        if isinstance(v, (int, float)):
            return _clamp(float(v), 0.0, 1.0)
        return 0.0
    
    @field_validator('reasoning', mode='before')
    @classmethod
    def sanitize_reasoning(cls, v):
        return _sanitize_string(v or "", max_length=300)
    
    @field_validator('new_topic', mode='before')
    @classmethod
    def sanitize_new_topic(cls, v):
        if not v or not isinstance(v, str):
            return None
        sanitized = _sanitize_string(v, max_length=100)
        return sanitized if sanitized else None

# -------------------------
# Anti-Repetition N-Gram Fingerprinting
# -------------------------
def generate_ngram_fingerprint(question: str, n: int = 3) -> List[str]:
    """
    Generate n-gram fingerprints from a question for similarity detection.
    Normalizes text and creates overlapping n-grams of words.
    
    Args:
        question: The question text to fingerprint
        n: Size of n-grams (default: 3)
    
    Returns:
        List of n-gram strings
    """
    if not question or not isinstance(question, str):
        return []
    
    # Normalize: lowercase, remove punctuation, split into words
    normalized = re.sub(r'[^\w\s]', '', question.lower())
    words = normalized.split()
    
    if len(words) < n:
        # If question is shorter than n, return single gram
        return [' '.join(words)] if words else []
    
    # Generate overlapping n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

def is_semantically_similar(fingerprint1: List[str], fingerprint2: List[str], threshold: float = 0.7) -> bool:
    """
    Check if two fingerprints are semantically similar based on n-gram overlap.
    
    Args:
        fingerprint1: First fingerprint (list of n-grams)
        fingerprint2: Second fingerprint (list of n-grams)
        threshold: Similarity threshold (0.0 to 1.0)
    
    Returns:
        True if similarity exceeds threshold, False otherwise
    """
    if not fingerprint1 or not fingerprint2:
        return False
    
    # Calculate Jaccard similarity (intersection over union)
    set1 = set(fingerprint1)
    set2 = set(fingerprint2)
    
    if not set1 or not set2:
        return False
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return False
    
    similarity = intersection / union
    return similarity >= threshold

def check_question_similarity(question: str, session_id: str, window_size: int = None, similarity_threshold: float = None, ngram_size: int = None) -> bool:
    """
    Check if a question is too similar to recent questions in the session.
    Blocks questions that exceed the similarity threshold within the rolling window.
    
    Args:
        question: The question to check
        session_id: Session identifier
        window_size: Number of recent questions to check (from config if None)
        similarity_threshold: Threshold for blocking (from config if None)
        ngram_size: Size of n-grams (from config if None)
    
    Returns:
        True if question is similar (should be blocked), False if unique enough
    """
    # Load config if parameters not provided
    if window_size is None or similarity_threshold is None or ngram_size is None:
        try:
            config = get_interview_config()
            anti_rep_config = config.get("anti_repetition", {})
            window_size = window_size or anti_rep_config.get("window_size", 5)
            similarity_threshold = similarity_threshold or anti_rep_config.get("similarity_threshold", 0.6)
            ngram_size = ngram_size or anti_rep_config.get("ngram_size", 3)
        except Exception:
            # Use defaults if config loading fails
            window_size = window_size or 5
            similarity_threshold = similarity_threshold or 0.6
            ngram_size = ngram_size or 3
    
    # Generate fingerprint for current question
    current_fingerprint = generate_ngram_fingerprint(question, ngram_size)
    
    if not current_fingerprint:
        return False  # Empty question, don't block
    
    # Get recent fingerprints for this session (recover from ChromaDB if not in memory)
    if session_id not in _QUESTION_FINGERPRINTS:
        # Try to recover from ChromaDB
        try:
            recovered_fingerprints = interview_chroma.get_question_fingerprints(session_id)
            if recovered_fingerprints:
                _QUESTION_FINGERPRINTS[session_id] = recovered_fingerprints
            else:
                _QUESTION_FINGERPRINTS[session_id] = []
        except Exception:
            _QUESTION_FINGERPRINTS[session_id] = []
    
    recent_fingerprints = _QUESTION_FINGERPRINTS[session_id]
    
    # Check similarity against recent questions in window
    for past_fingerprint in recent_fingerprints[-window_size:]:
        if is_semantically_similar(current_fingerprint, past_fingerprint, similarity_threshold):
            log.warning(f"🚫 Anti-Repetition: Question blocked (similarity > {similarity_threshold})")
            return True  # Too similar, block this question
    
    return False  # Unique enough, allow this question

def track_question_fingerprint(question: str, session_id: str, ngram_size: int = None) -> None:
    """
    Track a question's fingerprint for future similarity checks.
    Should be called after a question is generated and approved.
    Persists to ChromaDB for recovery.
    
    Args:
        question: The question to track
        session_id: Session identifier
        ngram_size: Size of n-grams (from config if None)
    """
    # Load ngram_size from config if not provided
    if ngram_size is None:
        try:
            config = get_interview_config()
            ngram_size = config.get("anti_repetition", {}).get("ngram_size", 3)
        except Exception:
            ngram_size = 3
    
    # Generate and store fingerprint
    fingerprint = generate_ngram_fingerprint(question, ngram_size)
    
    if not fingerprint:
        return
    
    if session_id not in _QUESTION_FINGERPRINTS:
        _QUESTION_FINGERPRINTS[session_id] = []
    
    _QUESTION_FINGERPRINTS[session_id].append(fingerprint)
    
    # Keep only last 10 fingerprints to prevent memory buildup
    if len(_QUESTION_FINGERPRINTS[session_id]) > 10:
        _QUESTION_FINGERPRINTS[session_id] = _QUESTION_FINGERPRINTS[session_id][-10:]
    
    # Persist to ChromaDB (non-blocking)
    _persist_question_fingerprints(session_id, _QUESTION_FINGERPRINTS[session_id])

# -------------------------
# Negative Intent Pattern Definitions
# -------------------------

# Centralized negative intent patterns for maintainability
NEGATIVE_PATTERNS = {
    "single_word": [
        "no", "nope", "nah", "won't", "can't", "don't", "refuse", "decline",
        "nothing", "nada", "not", "never", "neither", "none", "noway", "nuh-uh",
        "nay", "negative"
    ],
    # NOTE: Refusal pattern detection is now handled by LLM for semantic understanding.
    # These phrases are kept for reference but are no longer used in regex pre-filtering.
    "refusal_phrases": [
        "i won't", "i can't", "i don't want", "i refuse", "i decline",
        "won't tell", "can't tell", "don't want to", "i don't know",
        "no idea", "not interested", "not willing", "not going to",
        "refuse to", "decline to", "can't answer", "won't answer",
        "don't want to answer", "not gonna", "not going"
    ],
    "exit_phrases": [
        # NOTE: Exit intent detection is now primarily handled by LLM for semantic understanding.
        # These phrases are kept for reference but are no longer used in regex pre-filtering.
        # The LLM can understand exit intent from context and various phrasings.
        "good night", "bye", "exit interview", "end interview",
        "stop interview", "goodbye", "see you", "wrap up", "wrapup",
        "let's wrap up", "we can wrap up", "wrap this up"
    ],
    # NOTE: Uncertainty pattern detection is now handled by LLM for semantic understanding.
    # These phrases are kept for reference but are no longer used in regex pre-filtering.
    "uncertainty_phrases": [
        "i'm not sure", "not my area", "haven't worked with", "not familiar with",
        "not experienced with", "don't have experience", "that's not my area",
        "i haven't worked with that"
    ]
}

# Compiled regex patterns for robust matching
NEGATIVE_PATTERNS_REGEX = {
    "single_word": re.compile(
        r'\b(no|nope|nah|won\'t|can\'t|don\'t|refuse|decline|nothing|nada|not|never|neither|none|noway|nuh-uh|nay|negative)\b',
        re.IGNORECASE
    ),
    # NOTE: Refusal pattern detection is now handled by LLM for semantic understanding.
    # This regex is kept for backward compatibility but is no longer used in the main flow.
    "refusal": re.compile(
        r'\b(i\s+(won\'t|can\'t|don\'t\s+want|refuse|decline)|won\'t\s+(tell|answer)|can\'t\s+(tell|answer)|don\'t\s+want\s+to|i\s+don\'t\s+know|no\s+idea|not\s+(interested|willing|going|gonna)|refuse\s+to|decline\s+to)\b',
        re.IGNORECASE
    ),
    # NOTE: Exit intent detection is now handled by LLM for semantic understanding.
    # This regex is kept for backward compatibility but is no longer used in the main flow.
    "exit": re.compile(
        r'\b(good\s+night|bye|exit\s+interview|end\s+interview|stop\s+interview|goodbye|see\s+you|wrap\s+up|wrapup|let\'?s\s+wrap\s+up|we\s+can\s+wrap\s+up|wrap\s+this\s+up)\b',
        re.IGNORECASE
    ),
    # NOTE: Uncertainty pattern detection is now handled by LLM for semantic understanding.
    # This regex is kept for backward compatibility but is no longer used in the main flow.
    "uncertainty": re.compile(
        r'\b(i\'m\s+not\s+sure|not\s+my\s+area|haven\'t\s+worked\s+with|not\s+familiar\s+with|not\s+experienced\s+with|don\'t\s+have\s+experience|that\'s\s+not\s+my\s+area|i\s+haven\'t\s+worked\s+with\s+that)\b',
        re.IGNORECASE
    )
}

# Frozenset for O(1) lookup performance
NEGATIVE_SINGLE_WORDS = frozenset(NEGATIVE_PATTERNS["single_word"])

# Positive indicators that negate negative intent
POSITIVE_INDICATORS = [
    "but", "however", "willing", "interested", "learn", "try", "explore",
    "happy to", "glad to", "excited", "would like", "want to", "can learn"
]

# -------------------------
# Negative Intent Loop-Breaker
# -------------------------
def should_offer_loop_breaker(session_id: str, consecutive_threshold: int = None) -> bool:
    """
    Check if we should offer loop-breaker options based on consecutive negative intents.
    Enhanced to track uncooperative patterns even with low confidence and trigger faster
    for short negative responses.
    
    Args:
        session_id: Session identifier
        consecutive_threshold: Number of consecutive intents to trigger (from config if None)
    
    Returns:
        True if threshold reached, False otherwise
    """
    # CRITICAL: Always attempt recovery first if session_id not in memory
    # This ensures history is recovered on every call, not just when checking
    if session_id not in _NEGATIVE_INTENT_HISTORY:
        try:
            recovered_intents = interview_chroma.get_negative_intent_history(session_id)
            if recovered_intents:
                _NEGATIVE_INTENT_HISTORY[session_id] = recovered_intents
                log.debug("Recovered negative intent history from ChromaDB", extra={
                    "session_id": session_id,
                    "intent_count": len(recovered_intents)
                })
            else:
                _NEGATIVE_INTENT_HISTORY[session_id] = []
        except Exception as e:
            _NEGATIVE_INTENT_HISTORY[session_id] = []
            log.warning("Failed to recover negative intent history from ChromaDB", extra={
                "session_id": session_id,
                "error": str(e)
            })
    
    if session_id not in _NEGATIVE_INTENT_HISTORY:
        return False
    
    # Load threshold from config if not provided
    if consecutive_threshold is None:
        try:
            config = get_interview_config()
            consecutive_threshold = config.get("negative_intent", {}).get("consecutive_threshold", 2)
        except Exception:
            consecutive_threshold = 2
    
    recent_intents = _NEGATIVE_INTENT_HISTORY[session_id]
    
    # Check if we have enough consecutive negative intents
    if len(recent_intents) < consecutive_threshold:
        return False
    
    # Get last N intents for analysis
    last_n_intents = recent_intents[-consecutive_threshold:]
    
    # Check if the last N intents are all negative (not NONE)
    all_negative = all(intent.get("intent") in ["LACK_OF_KNOWLEDGE", "EXIT_INTENT", "TOPIC_SWITCH", "OFF_TOPIC"] 
                      for intent in last_n_intents)
    
    if all_negative:
        return True
    
    # Enhanced: Check for short negative responses - trigger faster (more aggressive)
    # If we have 1-2 consecutive short negative responses, trigger loop-breaker even if confidence is low
    if len(recent_intents) >= 1:
        # Check last 1-2 intents for negative patterns
        last_intents = recent_intents[-2:] if len(recent_intents) >= 2 else recent_intents[-1:]
        short_negative_count = 0
        for intent in last_intents:
            intent_type = intent.get("intent", "")
            confidence = intent.get("confidence", 0.0)
            reasoning = intent.get("reasoning", "").lower()
            
            # Consider it a short negative if:
            # 1. It's a negative intent type, OR
            # 2. Reasoning mentions "short" or "brief" or "single word" or "obvious negative" or "pre-filter"
            is_negative = intent_type in ["LACK_OF_KNOWLEDGE", "EXIT_INTENT"]
            is_short_negative = (
                "short" in reasoning or 
                "brief" in reasoning or 
                "single word" in reasoning or 
                "obvious negative" in reasoning or
                "pre-filter" in reasoning or
                "refusal pattern" in reasoning
            )
            
            # More aggressive: accept negative intents even with lower confidence
            if is_negative or (is_short_negative and confidence >= 0.3):
                short_negative_count += 1
        
        # Trigger loop-breaker after 1-2 consecutive short negative responses (more aggressive)
        if short_negative_count >= 1 and len(recent_intents) >= 1:
            # If single very clear negative, trigger immediately
            last_intent = recent_intents[-1]
            # Get single high confidence trigger threshold from config
            try:
                config = get_interview_config()
                single_trigger_threshold = config.get("negative_intent", {}).get("confidence_thresholds", {}).get("single_high_confidence_trigger", 0.6)
            except Exception:
                single_trigger_threshold = 0.6
            
            if last_intent.get("intent") in ["LACK_OF_KNOWLEDGE", "EXIT_INTENT"] and last_intent.get("confidence", 0) >= single_trigger_threshold:
                log.info("Loop-breaker triggered by single high-confidence negative response", extra={
                    "session_id": session_id,
                    "intent": last_intent.get("intent"),
                    "confidence": last_intent.get("confidence", 0),
                    "threshold": single_trigger_threshold
                })
                return True
        if short_negative_count >= 2:
            log.info("Loop-breaker triggered by consecutive short negative responses", extra={
                "session_id": session_id,
                "short_negative_count": short_negative_count
            })
            return True
    
    return False

def track_intent(session_id: str, intent: str, confidence: float = 0.0, reasoning: str = "") -> None:
    """
    Track an intent detection for loop-breaker analysis.
    Persists to ChromaDB for recovery.
    
    Args:
        session_id: Session identifier
        intent: Intent type (LACK_OF_KNOWLEDGE, EXIT_INTENT, TOPIC_SWITCH, OFF_TOPIC, NONE)
        confidence: Confidence score
        reasoning: Explanation of the detection
    """
    if session_id not in _NEGATIVE_INTENT_HISTORY:
        _NEGATIVE_INTENT_HISTORY[session_id] = []
    
    _NEGATIVE_INTENT_HISTORY[session_id].append({
        "intent": intent,
        "confidence": confidence,
        "reasoning": reasoning,
        "timestamp": time.time()
    })
    
    # Keep only last 10 intents to prevent memory buildup
    if len(_NEGATIVE_INTENT_HISTORY[session_id]) > 10:
        _NEGATIVE_INTENT_HISTORY[session_id] = _NEGATIVE_INTENT_HISTORY[session_id][-10:]
    
    # Persist to ChromaDB (non-blocking)
    _persist_negative_intent_history(session_id, _NEGATIVE_INTENT_HISTORY[session_id])

def get_loop_breaker_offer(interview_topic: str = None) -> str:
    """
    Generate empathetic loop-breaker offer message.
    
    Args:
        interview_topic: Current interview topic (optional)
    
    Returns:
        Offer message with options
    """
    try:
        config = get_interview_config()
        options = config.get("negative_intent", {}).get("auto_offer_options", ["primer", "switch", "end"])
    except Exception:
        options = ["primer", "switch", "end"]
    
    # More conversational and supportive intro messages
    if interview_topic:
        intro_messages = [
            f"I can see {interview_topic} might not be your strongest area—that's completely okay!",
            f"No worries if {interview_topic} isn't your main expertise.",
            f"It seems like {interview_topic} might be outside your usual focus—no problem at all!"
        ]
        base_message = random.choice(intro_messages) + " Here are some ways we can proceed:"
    else:
        base_message = "I want to make sure we're focusing on areas where you can showcase your strengths. Here are your options:"
    
    # Build friendly options list
    option_lines = []
    if "primer" in options:
        option_lines.append("• Type 'primer' for a quick overview of the fundamentals")
    if "switch" in options:
        option_lines.append("• Type 'switch' or just tell me what topic you'd like to discuss instead")
    if "end" in options:
        option_lines.append("• Type 'end' if you'd like to wrap up the interview")
    
    # Always offer continue as implicit option
    option_lines.append("• Or just continue answering if you'd like to keep going")
    
    options_text = "\n" + "\n".join(option_lines)
    
    return base_message + options_text

def _is_truly_negative(answer_lower: str, word_count: int) -> bool:
    """
    Check if response is truly negative, not just containing negative words.
    NOTE: This function is now primarily used for very short single-word responses.
    For longer responses, the LLM handles context-aware negative intent detection
    which can better understand nuance and positive indicators.
    
    Args:
        answer_lower: Lowercased answer text
        word_count: Number of words in the answer
    
    Returns:
        True if response is truly negative, False if positive indicators suggest otherwise
    """
    # For very short responses, check for obvious positive indicators
    # For longer responses, LLM handles this more accurately
    if word_count <= 3:
        if any(indicator in answer_lower for indicator in POSITIVE_INDICATORS):
            return False
    
    return True

def _should_handle_negative_intent(
    intent: str,
    confidence: float,
    word_count: int,
    min_confidence: float,
    session_id: Optional[str],
    topic: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Determine if negative intent should be handled.
    Consolidates confidence threshold logic for maintainability.
    
    Args:
        intent: Detected intent type
        confidence: Confidence score from detection
        word_count: Number of words in the response
        min_confidence: Base minimum confidence threshold
        session_id: Session identifier (optional)
        topic: Current interview topic (optional)
    
    Returns:
        Tuple of (should_handle, loop_breaker_response)
        - should_handle: True if intent should be handled
        - loop_breaker_response: Response string if loop-breaker should trigger, None otherwise
    """
    is_negative = intent in ["LACK_OF_KNOWLEDGE", "EXIT_INTENT", "TOPIC_SWITCH"]
    
    if not is_negative:
        return (False, None)
    
    # Very short negatives always handled
    if word_count <= 3 and intent in ["LACK_OF_KNOWLEDGE", "EXIT_INTENT"]:
        return (True, None)
    
    # Check loop-breaker first
    if session_id and should_offer_loop_breaker(session_id):
        return (True, get_loop_breaker_offer(topic))
    
    # Standard confidence check
    if confidence >= min_confidence:
        return (True, None)
    
    # Low confidence but still negative - check if loop-breaker should trigger
    # Get loop-breaker minimum threshold from config
    try:
        config = get_interview_config()
        loop_breaker_min = config.get("negative_intent", {}).get("confidence_thresholds", {}).get("loop_breaker_minimum", 0.3)
    except Exception:
        loop_breaker_min = 0.3
    
    if session_id and confidence >= loop_breaker_min:
        if should_offer_loop_breaker(session_id):
            return (True, get_loop_breaker_offer(topic))
    
    return (False, None)

def reset_negative_intent_history(session_id: str) -> None:
    """
    Reset negative intent history after loop-breaker is handled.
    Clears both in-memory and ChromaDB storage.
    
    Args:
        session_id: Session identifier
    """
    if session_id in _NEGATIVE_INTENT_HISTORY:
        _NEGATIVE_INTENT_HISTORY[session_id] = []
    
    # Also clear from ChromaDB
    try:
        interview_chroma.clear_negative_intent_history(session_id)
    except Exception as e:
        log.warning(f"⚠️  WARNING: Failed to clear negative intent history from ChromaDB: {e}")

def handle_loop_breaker_choice(choice: str, conversation_context: "ConversationContext", interview_topic: str = None) -> Optional[str]:
    """
    Process the candidate's loop-breaker choice and return appropriate response.
    
    Args:
        choice: Candidate's choice (primer/switch/end)
        conversation_context: Current conversation context
        interview_topic: Current interview topic (optional)
    
    Returns:
        Response message, or None if choice not recognized
    """
    choice_lower = choice.strip().lower()
    
    if "primer" in choice_lower:
        # Offer a primer on the topic
        if interview_topic:
            return (f"Great! Let me give you a quick primer on {interview_topic}. "
                   f"{interview_topic} is... [brief explanation]. "
                   f"Now, let's start with a fundamental question about {interview_topic}.")
        else:
            return "Great! Let me provide some context and we'll start with the fundamentals."
    
    elif "switch" in choice_lower:
        # Mark topic as switched and ask for new topic
        if hasattr(conversation_context, 'topic_switched'):
            conversation_context.topic_switched = True
        return "Of course! What topic would you like to discuss instead? Feel free to mention any technology, skill, or area you'd like to explore."
    
    elif "end" in choice_lower or "stop" in choice_lower or "wrap up" in choice_lower or "wrapup" in choice_lower:
        # Mark interview for ending
        conversation_context.set_should_end(True)
        return "I understand. Thank you for your time. Let me wrap up with some final thoughts."
    
    return None  # Choice not recognized

# Simple in-memory circuit breaker map: session_id -> {failures:int, until_ts:float}
# These are kept in-memory for fast access, but also persisted to ChromaDB
_CB: Dict[str, Dict[str, Any]] = {}

# Track recent fallback questions to avoid repetition
_RECENT_FALLBACKS: Dict[str, List[str]] = {}

# Track n-gram fingerprints of recent questions for anti-repetition
_QUESTION_FINGERPRINTS: Dict[str, List[List[str]]] = {}

# Track consecutive negative intents for loop-breaking
_NEGATIVE_INTENT_HISTORY: Dict[str, List[Dict[str, Any]]] = {}

# ==================== PERSISTENT STATE MANAGEMENT ====================

def _persist_question_fingerprints(session_id: str, fingerprints: List[List[str]]):
    """Persist question fingerprints to ChromaDB (non-blocking)"""
    try:
        if session_id and fingerprints:
            interview_chroma.store_question_fingerprints(session_id, fingerprints)
    except Exception as e:
        log.warning(f"⚠️  WARNING: Failed to persist question fingerprints: {e}")

def _persist_negative_intent_history(session_id: str, intent_history: List[Dict[str, Any]]):
    """Persist negative intent history to ChromaDB (non-blocking)"""
    try:
        if session_id and intent_history:
            interview_chroma.store_negative_intent_history(session_id, intent_history)
    except Exception as e:
        log.warning(f"⚠️  WARNING: Failed to persist negative intent history: {e}")

def _persist_recent_fallbacks(session_id: str, fallbacks: List[str]):
    """Persist recent fallback questions to ChromaDB (non-blocking)"""
    try:
        if session_id and fallbacks:
            interview_chroma.store_recent_fallbacks(session_id, fallbacks)
    except Exception as e:
        log.warning(f"⚠️  WARNING: Failed to persist recent fallbacks: {e}")

def _persist_circuit_breaker(session_id: str, cb_state: Dict[str, Any]):
    """Persist circuit breaker state to ChromaDB (non-blocking)"""
    try:
        if session_id and cb_state:
            interview_chroma.store_circuit_breaker_state(session_id, cb_state)
    except Exception as e:
        log.warning(f"⚠️  WARNING: Failed to persist circuit breaker state: {e}")

def _persist_conversation_context(session_id: str, context: "ConversationContext"):
    """Persist conversation context to ChromaDB (non-blocking)"""
    try:
        if session_id and context:
            context_dict = {
                "topics_discussed": context.topics_discussed,
                "candidate_strengths": context.candidate_strengths,
                "areas_to_explore": context.areas_to_explore,
                "conversation_depth": context.conversation_depth,
                "engagement_level": context.engagement_level,
                "candidate_questions": context.candidate_questions,
                "interview_phase": context.interview_phase,
                "time_elapsed": context.time_elapsed,
                "quality_score": context.quality_score,
                "coverage": context.coverage,
                "last_bins": context.last_bins,
                "interview_stage": context.interview_stage,
                "stage_progress": context.stage_progress,
                "technical_depth_level": context.technical_depth_level,
                "adaptive_follow_up_type": context.adaptive_follow_up_type,
                "response_quality_history": context.response_quality_history,
                "technical_depth_demonstrated": context.technical_depth_demonstrated,
                "used_scenario_ids": context.used_scenario_ids
            }
            interview_chroma.store_conversation_context(session_id, context_dict)
    except Exception as e:
        log.warning(f"⚠️  WARNING: Failed to persist conversation context: {e}")

def _recover_session_state(session_id: str) -> Dict[str, Any]:
    """Recover session state from ChromaDB and restore to in-memory structures"""
    try:
        if not session_id:
            return {}
        
        recovered = interview_chroma.recover_session_state(session_id)
        
        if not recovered:
            return {}
        
        # Restore to in-memory structures
        if "question_fingerprints" in recovered:
            _QUESTION_FINGERPRINTS[session_id] = recovered["question_fingerprints"]
        
        if "negative_intent_history" in recovered:
            _NEGATIVE_INTENT_HISTORY[session_id] = recovered["negative_intent_history"]
        
        if "recent_fallbacks" in recovered:
            _RECENT_FALLBACKS[session_id] = recovered["recent_fallbacks"]
        
        if "circuit_breaker" in recovered and recovered["circuit_breaker"]:
            _CB[session_id] = recovered["circuit_breaker"]
        
        log.info(f"Recovered session state for: {session_id}")
        return recovered
        
    except Exception as e:
        log.warning(f"Failed to recover session state: {e}")
        return {}

def _recover_conversation_context(session_id: str, context: "ConversationContext") -> bool:
    """Recover conversation context from ChromaDB and restore to ConversationContext object"""
    try:
        if not session_id:
            return False
        
        context_data = interview_chroma.get_conversation_context(session_id)
        
        if not context_data:
            return False
        
        # Restore context attributes
        context.topics_discussed = context_data.get("topics_discussed", [])
        context.candidate_strengths = context_data.get("candidate_strengths", [])
        context.areas_to_explore = context_data.get("areas_to_explore", [])
        context.conversation_depth = context_data.get("conversation_depth", 0)
        context.engagement_level = context_data.get("engagement_level", 0.5)
        context.candidate_questions = context_data.get("candidate_questions", [])
        context.interview_phase = context_data.get("interview_phase", "opening")
        context.time_elapsed = context_data.get("time_elapsed", 0)
        context.quality_score = context_data.get("quality_score", 0.0)
        context.coverage = context_data.get("coverage", {
            "skills": 0, "experience": 0, "problem_solving": 0,
            "collaboration": 0, "constraints": 0, "outcomes": 0
        })
        context.last_bins = context_data.get("last_bins", [])
        context.interview_stage = context_data.get("interview_stage", "warm_up")
        context.stage_progress = context_data.get("stage_progress", 0.0)
        context.technical_depth_level = context_data.get("technical_depth_level", "basic")
        context.adaptive_follow_up_type = context_data.get("adaptive_follow_up_type")
        context.response_quality_history = context_data.get("response_quality_history", [])
        context.technical_depth_demonstrated = context_data.get("technical_depth_demonstrated", 0.0)
        context.used_scenario_ids = context_data.get("used_scenario_ids", [])
        
        log.info(f"Recovered conversation context for: {session_id}")
        return True
        
    except Exception as e:
        log.warning(f"⚠️  WARNING: Failed to recover conversation context: {e}")
        return False

logging.basicConfig(level=logging.INFO)

# -------------------------
# LangGraph State Machine Models
# -------------------------
class InterviewState(str, Enum):
    # Old states (for backward compatibility)
    OPENING_RAPPORT = "opening_rapport"
    STAGE_SETTING = "stage_setting"
    MAIN_QUESTIONS = "main_questions"
    CANDIDATE_QUESTIONS = "candidate_questions"
    FEEDBACK = "feedback"
    CLOSING = "closing"
    COMPLETED = "completed"
    
    # Topic-focused states
    TOPIC_INTRODUCTION = "topic_introduction"
    TOPIC_FUNDAMENTALS = "topic_fundamentals"
    TOPIC_PROBLEM_SOLVING = "topic_problem_solving"
    TOPIC_DEEP_DIVE = "topic_deep_dive"
    TOPIC_FEEDBACK = "topic_feedback"

class ResponseType(str, Enum):
    BRIEF = "brief"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    GENERAL = "general"
    UNRESPONSIVE = "unresponsive"

class PersonaType(str, Enum):
    TECH = "tech"
    HIRING_MANAGER = "hiring_manager"
    HR_CULTURE = "hr_culture"
    CLIENT_READINESS = "client_readiness"

class SingleCallQuestionResult(BaseModel):
    """Structured result from single LLM call for question generation"""
    question: str = Field(description="The generated interview question")
    response_analysis: Dict[str, Any] = Field(description="Analysis of the candidate's response")
    engagement_score: float = Field(ge=0.0, le=1.0, description="Engagement score from 0.0 to 1.0")
    suggested_follow_ups: List[str] = Field(description="List of suggested follow-up questions")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the generated question")
    reasoning: str = Field(description="Brief explanation of why this question was chosen")
    topics_mentioned: List[str] = Field(description="Topics mentioned by the candidate")
    skills_demonstrated: List[str] = Field(description="Skills demonstrated by the candidate")
    strengths: List[str] = Field(description="Strengths identified in the response")
    areas_to_explore: List[str] = Field(description="Areas to explore further")

class InterviewRequest(BaseModel):
    uid: str
    callback_url: Optional[str] = None
    
    # New required fields for topic-focused interview
    interview_topic: Optional[str] = None
    structured_resume: Optional[Dict[str, Any]] = None
    
    # Existing fields, now optional for fallback
    resume: Optional[Dict[str, Any]] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    answer: str = ""
    persona: PersonaType = PersonaType.TECH
    job_details: Optional[Dict[str, Any]] = None
    end_interview: bool = False
    portfolio_links: Optional[List[str]] = Field(default_factory=list)
    session_id: Optional[str] = None
    current_state: Optional[InterviewState] = None
    question_count: int = 0
    start_time: Optional[float] = None
    auth_token: Optional[str] = None
    
    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v):
        if len(v) > 5000:
            raise ValueError("Answer too long")
        return v.strip()
    
    @field_validator('interview_topic')
    @classmethod
    def validate_interview_topic(cls, v):
        if v is not None and not v.strip():
            raise ValueError("interview_topic cannot be empty if provided")
        return v

class InterviewResponse(BaseModel):
    conversation_history: List[Dict[str, str]]
    agent_message: str
    status: str
    current_state: Optional[InterviewState] = None
    question_count: Optional[int] = None
    response_analysis: Optional[Dict[str, Any]] = None
    final_summary: Optional[str] = None
    error: Optional[str] = None
    session_metadata: Optional[Dict[str, Any]] = None


# -------------------------
# Helpers
# -------------------------
def _map_phase_to_state(phase: str) -> InterviewState:
    phase_l = (phase or "").lower()
    if phase_l == "opening":
        return InterviewState.OPENING_RAPPORT
    if phase_l == "exploration":
        return InterviewState.MAIN_QUESTIONS
    if phase_l == "deep_dive":
        return InterviewState.MAIN_QUESTIONS  # deep_dive maps to main_questions
    if phase_l == "candidate_questions":
        return InterviewState.CANDIDATE_QUESTIONS
    if phase_l == "closing":
        return InterviewState.CLOSING
    return InterviewState.OPENING_RAPPORT

# -------------------------
# LangGraph State Machine Configuration (config-driven)
# -------------------------
from core.config import (
    get_interview_state_config,
    get_interview_thresholds,
    get_interview_stage_instructions,
    get_interview_follow_up_guidance,
    get_interview_feature_flags,
    get_interview_config,
    get_roles_config,
    get_role_based_thresholds,
    get_intelligent_staging_config,
)
async def classify_role_hybrid(uid_context: Dict[str, Any] = None, job_details: Dict[str, Any] = None):
    """Hybrid classifier: rules first from roles.yml, then LLM fallback.
    Returns a simple object with role_category, confidence_score, reasoning, matched_criteria, skill_analysis, experience_analysis.
    """
    cfg = get_roles_config() or {}
    rules = cfg.get("rules", {})
    blend = cfg.get("blending", {})
    lock_min = float(blend.get("rule_min_score_for_lock", 0.7))
    lock_conf = float(blend.get("rule_confidence_when_locked", 0.9))
    llm_w = float(blend.get("llm_weight_when_unlocked", 0.4))
    rule_w = float(blend.get("rule_weight_when_unlocked", 0.6))

    text_fields = []
    try:
        if job_details:
            for k, v in job_details.items():
                if isinstance(v, str):
                    text_fields.append(v)
                elif isinstance(v, list):
                    text_fields.extend([str(i) for i in v])
        if uid_context:
            sr = uid_context.get("skills_analysis", {}) or uid_context.get("skills_parser", {}) or {}
            skills = sr.get("skills", [])
            text_fields.extend([str(s.get("SkillName", "")) for s in skills if isinstance(s, dict)])
            # also include resume experience if available
            exp = uid_context.get("experience_data", {}).get("experience", []) if uid_context.get("experience_data") else []
            for item in exp:
                if isinstance(item, dict):
                    text_fields.extend([str(item.get("job_title", "")), str(item.get("company", ""))])
    except Exception:
        pass

    haystack = " ".join([t.lower() for t in text_fields if isinstance(t, str)])
    best_role = None
    best_score = 0.0
    matched = []
    for role, rule in rules.items():
        score = 0.0
        weight = float(rule.get("weight", 1.0))
        for keylist in ["title_keywords", "skill_keywords", "company_keywords"]:
            for kw in rule.get(keylist, []) or []:
                kw_l = str(kw).lower()
                if kw_l and kw_l in haystack:
                    score += 0.15
                    matched.append((role, kw_l))
        score = min(1.0, score) * weight
        if score > best_score:
            best_score = score
            best_role = role

    if best_role and best_score >= lock_min:
        return type("RoleResult", (), {
            "role_category": best_role,
            "confidence_score": lock_conf,
            "reasoning": f"Rule-based match with score {best_score:.2f}",
            "matched_criteria": matched,
            "alternative_roles": [],
            "skill_analysis": {},
            "experience_analysis": {},
        })()

    # Fallback to LLM classification and blend confidence
    # Use LLM-driven classifier for better coverage of all domains
    if uid_context:
        try:
            # Try LLM-driven classification first (handles all domains)
            llm_res = await classify_role_llm_driven(uid_context)
        except Exception as llm_error:
            log.warning(f"⚠️  WARNING: LLM-driven classification failed: {llm_error}. Falling back to rule-based.")
            llm_res = classify_role_enhanced(uid_context)
    else:
        llm_res = type("RoleResult", (), {
        "role_category": best_role or "TECHNICAL_CODING",
        "confidence_score": 0.5,
        "reasoning": "No UID context available - using default classification",
        "matched_criteria": [],
        "alternative_roles": [],
        "skill_analysis": {},
        "experience_analysis": {}
    })()

    final_role = best_role or llm_res.role_category
    final_conf = (rule_w * best_score) + (llm_w * getattr(llm_res, "confidence_score", 0.5))
    final_conf = max(final_conf, 0.55)  # avoid ultra-low confidence when we have some rule/llm signal
    return type("RoleResult", (), {
        "role_category": final_role,
        "confidence_score": final_conf,
        "reasoning": f"Hybrid: rules score {best_score:.2f}, LLM {getattr(llm_res, 'confidence_score', 0.5):.2f}",
        "matched_criteria": matched or getattr(llm_res, "matched_criteria", []),
        "alternative_roles": getattr(llm_res, "alternative_roles", []),
        "skill_analysis": getattr(llm_res, "skill_analysis", {}),
        "experience_analysis": getattr(llm_res, "experience_analysis", {}),
    })()
from core.retrieval import get_job_context, get_company_context, summarize_for_prompt

# ==================== CONSTANTS (Defaults & Thresholds) ====================
DEFAULT_ROLE_TITLE = "Professional"
DEFAULT_COMPANY = "Tech Company"
DEFAULT_SKILLS_TEXT = "General background"
LOW_CONF_THRESHOLD = 0.6
MAX_HISTORY_TURNS = 4  # Reduced from 6 to prevent irrelevant questions in deep conversations
MAX_QUESTION_LEN = 400  # Increased from 260 to allow for more complete questions
SKILL_LIST_MAX = 10
DOMAIN_JOB_KEYWORDS_MAX_CHARS = 600
GENERIC_OPENER_PATTERNS = [
    r"^to start, could you briefly share your background and interests\??$",
    r"^could you briefly share your background and interests\??$",
    r"^let'?s start"
]
FALLBACK_NON_DUP_QUESTION = "Could you expand on a recent project and your specific contributions?"

PERSONAS = {
    PersonaType.TECH: "You are a technical interviewer focusing on coding, architecture, and problem-solving skills. Adapt to the interview state with relevant questions.",
    PersonaType.HIRING_MANAGER: "You are a hiring manager assessing role fit, leadership, and project delivery experience. Pose structured questions per state.",
    PersonaType.HR_CULTURE: "You are an HR interviewer evaluating cultural fit, communication, and team collaboration. Align questions with current state.",
    PersonaType.CLIENT_READINESS: "You assess client-facing skills, requirement gathering, and stakeholder management. Focus questions on state goals."
}

def _build_state_configurations() -> Dict[InterviewState, Dict[str, Any]]:
    raw = get_interview_state_config()
    mapped: Dict[InterviewState, Dict[str, Any]] = {}
    name_to_enum = {
        "opening_rapport": InterviewState.OPENING_RAPPORT,
        "stage_setting": InterviewState.STAGE_SETTING,
        "main_questions": InterviewState.MAIN_QUESTIONS,
        "candidate_questions": InterviewState.CANDIDATE_QUESTIONS,
        "feedback": InterviewState.FEEDBACK,
        "closing": InterviewState.CLOSING,
        "completed": InterviewState.COMPLETED,
    }
    for key, cfg in raw.items():
        state = name_to_enum.get(key)
        if not state:
            continue
        next_states = []
        for nxt in cfg.get("next_states", []):
            nxt_enum = name_to_enum.get(nxt)
            if nxt_enum:
                next_states.append(nxt_enum)
        mapped[state] = {
            "max_questions": cfg.get("max_questions", 1),
            "focus": cfg.get("focus", ""),
            "next_states": next_states,
            "timeout_minutes": cfg.get("timeout_minutes", 5),
    }
    return mapped

STATE_CONFIGURATIONS = _build_state_configurations()

# -------------------------
# Intelligent Context-Aware Interview Flow
# -------------------------
class ConversationContext:
    """Tracks conversation context and topics dynamically"""
    
    def __init__(self):
        self.topics_discussed = []
        self.candidate_strengths = []
        self.areas_to_explore = []
        self.conversation_depth = 0
        self.engagement_level = 0.5
        self.candidate_questions = []
        self.interview_phase = "opening"  # opening, exploration, deep_dive, candidate_questions, closing
        self.time_elapsed = 0
        self.quality_score = 0.0
        self.coverage = {
            "skills": 0,
            "experience": 0,
            "problem_solving": 0,
            "collaboration": 0,
            "constraints": 0,
            "outcomes": 0
        }
        self.last_bins = []  # recent coverage bins
        
        # NEW: Multi-stage tracking
        self.interview_stage = "warm_up"  # warm_up, domain_assessment, problem_solving, behavioral, fitment, wrap_up
        self.stage_progress = 0.0  # 0.0 to 1.0 per stage
        self.technical_depth_level = "basic"  # basic, intermediate, advanced
        self.adaptive_follow_up_type = None  # deeper_probe, corrective, escalation, continuation
        
        # NEW: Response quality tracking
        self.response_quality_history = []  # List of (question_num, quality_score)
        self.technical_depth_demonstrated = 0.0  # 0.0 to 1.0
        self.used_scenario_ids = []  # Track used scenario IDs
        
        # Session ID for persistence
        self.session_id = None
        
    async def _discover_new_skills(self, response_text: str):
        """Discover new skills from interview responses in real-time"""
        try:
            new_skills = await _extract_skills_from_response(response_text)
            for skill in new_skills:
                if skill not in self.coverage:
                    self.coverage[skill] = 0
                    log.debug(f"Discovered new skill: {skill}")
        except Exception as e:
            log.warning(f"Skill discovery failed: {e}")
        
    async def update_context(self, response_analysis: Dict[str, Any], question_count: int):
        """Update context based on response analysis"""
        self.conversation_depth = question_count
        self.engagement_level = response_analysis.get("engagement_score", 0.5)
        
        # Track topics from keywords
        keywords = response_analysis.get("keywords", [])
        for keyword in keywords:
            if keyword not in self.topics_discussed:
                self.topics_discussed.append(keyword)
        # Coverage bins heuristic updates
        ctext = (response_analysis.get("context_understanding", "") or "").lower()
        lkeys = [str(k).lower() for k in keywords]
        def bump(bin_name: str, cond: bool):
            if cond:
                self.coverage[bin_name] = min(self.coverage.get(bin_name, 0) + 1, 99)
        
        # Core competency bins (universal for all roles)
        # LLM-based coverage detection using structured output (Issue 5.1 migration)
        try:
            coverage_detection_prompt = f"""Analyze this interview response and determine which coverage categories are discussed.

RESPONSE TEXT: "{ctext}"
RESPONSE KEYWORDS: {keywords}

Determine if the response discusses any of these categories:
- technical_skills: Technical tools, technologies, software, platforms, systems, or technical capabilities
- experience: Past work experience, projects, accomplishments, or professional history
- problem_solving: Problem-solving approaches, debugging, optimization, challenges, solutions, troubleshooting
- collaboration: Teamwork, collaboration, stakeholder interaction, communication, mentoring, partnerships
- constraints: Constraints, limitations, deadlines, budgets, resources, compliance, regulations
- outcomes: Results, impact, metrics, KPIs, success measures, improvements, business outcomes"""

            coverage_data: CoverageDetectionResult = await invoke_structured_llm(
                coverage_detection_prompt,
                CoverageDetectionResult,
                task_type=TaskType.INTERVIEW,
                preferred_model=settings.GEMINI_MODEL,
                agent_name="interview_coverage_detection",
                temperature=0.1,
                timeout=_INTERVIEW_LLM_TIMEOUT,
                raise_on_fallback=False,
            )
            
            # Only bump categories if confidence is high enough
            if coverage_data.confidence >= 0.6:
                if coverage_data.technical_skills:
                    bump("technical_skills", True)
                if coverage_data.experience:
                    bump("experience", True)
                if coverage_data.problem_solving:
                    bump("problem_solving", True)
                if coverage_data.collaboration:
                    bump("collaboration", True)
                if coverage_data.constraints:
                    bump("constraints", True)
                if coverage_data.outcomes:
                    bump("outcomes", True)
        except Exception as e:
            log.warning(f"LLM-based coverage detection failed: {e}, using fallback")
            # Fallback: Simple keyword matching for critical categories only
            bump("technical_skills", any(k in lkeys for k in ["react", "angular", "python", "java", "sql", "design", "development", "software"]))
            bump("experience", any(w in ctext for w in ["experience", "project", "worked", "built", "developed", "implemented"]))
            bump("problem_solving", any(w in lkeys for w in ["problem", "solve", "challenge", "solution"]))
            bump("collaboration", any(w in ctext for w in ["team", "collaborat", "stakeholder", "communication"]))
        
        # Dynamic skill bins from UID context (domain-agnostic)
        # Check if any of the candidate's specific skills from UID context are mentioned
        for skill_bin in self.coverage.keys():
            if skill_bin not in ["technical_skills", "experience", "problem_solving", "collaboration", "constraints", "outcomes"]:
                # This is a specific skill bin from UID context
                skill_keywords = skill_bin.replace("_", " ").split()
                # Check for exact matches and partial matches
                if any(skill_kw in lkeys for skill_kw in skill_keywords) or any(skill_kw in ctext for skill_kw in skill_keywords):
                    bump(skill_bin, True)
        
        # Real-time skill discovery: Extract new skills from current response
        # This allows the system to discover skills not in the original UID context
        try:
            # Extract skills from the current response using LLM
            response_text = response_analysis.get("context_understanding", "") + " " + " ".join(keywords)
            if response_text.strip():
                # Run skill extraction in background (non-blocking)
                import asyncio
                asyncio.create_task(self._discover_new_skills(response_text))
        except Exception as e:
            log.warning(f"Real-time skill discovery failed: {e}")
        
        # Track most recently bumped bin (simple heuristic: pick min bin after bump)
        try:
            least_bin = min(self.coverage, key=self.coverage.get)
            self.last_bins.append(least_bin)
            if len(self.last_bins) > 3:
                self.last_bins = self.last_bins[-3:]
        except Exception:
            pass
        
        # Determine interview phase dynamically
        if question_count == 0:
            self.interview_phase = "opening"
        elif question_count < 3:
            self.interview_phase = "exploration"
        elif question_count < 8:
            self.interview_phase = "deep_dive"
        elif any("question" in msg.get("content", "").lower() for msg in self.candidate_questions):
            self.interview_phase = "candidate_questions"
        else:
            self.interview_phase = "closing"
        
        # Persist context after update (non-blocking, offloaded to thread pool)
        if self.session_id:
            await _rbi(_persist_conversation_context, self.session_id, self)
    
    def determine_interview_stage(self, question_count: int) -> str:
        """Map question count to interview stage (fallback method)"""
        if question_count <= 2:
            return "warm_up"
        elif question_count <= 5:
            return "domain_assessment"
        elif question_count <= 8:
            return "problem_solving"
        elif question_count <= 10:
            return "behavioral"
        elif question_count <= 12:
            return "fitment"
        else:
            return "wrap_up"
    
    def determine_stage_from_analysis(self, question_count: int, response_analysis: Dict, coverage_gaps: List[str] = None, 
                                     role_level: str = None, years_experience: int = None) -> str:
        """
        Intelligent stage determination using response_analysis metrics.
        Considers confidence, word_count, coverage gaps instead of just question_count.
        Used when INTELLIGENT_FLOW is enabled.
        
        Enhanced with role-based thresholds for adaptive progression.
        
        Args:
            question_count: Current question number
            response_analysis: Analysis of candidate's response
            coverage_gaps: List of coverage gaps identified
            role_level: Candidate role level ("junior", "mid", "senior")
            years_experience: Years of experience (used to infer role level)
        """
        # Load role-based thresholds (enhanced)
        try:
            thresholds = get_role_based_thresholds(role_level, years_experience)
            confidence_threshold = thresholds.get("confidence_threshold", 0.7)
            word_count_threshold = thresholds.get("word_count_threshold", 50)
            coverage_gap_weight = thresholds.get("coverage_gap_weight", 0.3)
        except Exception:
            confidence_threshold = 0.7
            word_count_threshold = 50
            coverage_gap_weight = 0.3
        
        # Extract metrics from response_analysis
        confidence = response_analysis.get("confidence", 0.5)
        word_count = response_analysis.get("word_count", 0)
        engagement_score = response_analysis.get("engagement_score", 0.5)
        
        # Calculate coverage gap score (higher = more gaps)
        coverage_gap_score = 0.0
        if coverage_gaps:
            coverage_gap_score = len(coverage_gaps) / max(len(self.coverage), 1)
        
        # Calculate depth score based on response quality
        depth_score = (confidence + engagement_score) / 2.0
        response_depth = 1.0 if word_count >= word_count_threshold else (word_count / word_count_threshold)
        
        # Combine scores to determine stage
        overall_readiness = (depth_score * 0.5) + (response_depth * 0.3) + ((1 - coverage_gap_score * coverage_gap_weight) * 0.2)
        
        # Stage progression logic based on overall readiness and question count
        if question_count <= 2:
            return "warm_up"
        
        if overall_readiness < 0.4:
            # Low readiness - stay in earlier stages longer
            if question_count <= 6:
                return "domain_assessment"
            elif question_count <= 10:
                return "problem_solving"
            else:
                return "behavioral"
        elif overall_readiness < 0.6:
            # Medium readiness - normal progression
            if question_count <= 5:
                return "domain_assessment"
            elif question_count <= 8:
                return "problem_solving"
            elif question_count <= 11:
                return "behavioral"
            else:
                return "fitment"
        else:
            # High readiness - can progress faster
            if question_count <= 4:
                return "domain_assessment"
            elif question_count <= 7:
                return "problem_solving"
            elif question_count <= 9:
                return "behavioral"
            elif question_count <= 11:
                return "fitment"
            else:
                return "wrap_up"
    
    def determine_follow_up_type(self, response_analysis: Dict) -> str:
        """Determine adaptive follow-up based on response quality"""
        engagement = response_analysis.get("engagement_score", 0.5)
        tech_depth = self.technical_depth_demonstrated
        
        if engagement > 0.8 and tech_depth > 0.7:
            return "deeper_probe"
        elif engagement < 0.4 or tech_depth < 0.3:
            return "corrective"
        elif engagement > 0.6:
            return "escalation"
        return "continuation"
    
    def determine_technical_depth(self, response_analysis: Dict, question_count: int) -> str:
        """Determine technical depth level based on performance"""
        avg_quality = sum(q[1] for q in self.response_quality_history) / len(self.response_quality_history) if self.response_quality_history else 0.5
        
        if avg_quality > 0.75 and question_count > 3:
            return "advanced"
        elif avg_quality > 0.5 or question_count > 5:
            return "intermediate"
        return "basic"
    
    def update_stage(self, question_count: int, response_analysis: Dict, 
                    role_level: str = None, years_experience: int = None):
        """
        Update stage and adaptive flow parameters.
        Uses intelligent staging based on response_analysis when INTELLIGENT_FLOW is enabled.
        Enhanced with role-based thresholds for adaptive progression.
        Persists context after update.
        
        Args:
            question_count: Current question number
            response_analysis: Analysis of candidate's response
            role_level: Candidate role level ("junior", "mid", "senior")
            years_experience: Years of experience (used to infer role level)
        """
        # Check if INTELLIGENT_FLOW is enabled
        intelligent_flow_enabled = False
        try:
            feature_flags = get_interview_feature_flags()
            intelligent_flow_enabled = bool(feature_flags.get("INTELLIGENT_FLOW", False))
        except Exception:
            pass
        
        # Calculate coverage gaps for intelligent staging
        coverage_gaps = []
        try:
            if intelligent_flow_enabled:
                # Identify areas with low coverage
                coverage_gaps = [key for key, value in self.coverage.items() if value < 2]
        except Exception:
            pass
        
        # Use intelligent or traditional staging based on flag
        if intelligent_flow_enabled and response_analysis:
            self.interview_stage = self.determine_stage_from_analysis(
                question_count, response_analysis, coverage_gaps, role_level, years_experience
            )
            log.info(f"🎯 Intelligent Staging: stage={self.interview_stage} (adaptive based on performance, role_level={role_level})")
        else:
            self.interview_stage = self.determine_interview_stage(question_count)
        
        self.adaptive_follow_up_type = self.determine_follow_up_type(response_analysis)
        self.technical_depth_level = self.determine_technical_depth(response_analysis, question_count)
        
        # Persist context after stage update (sync method — call Chroma helper directly;
        # async callers should use ConversationContext.update_context which uses _rbi)
        if self.session_id:
            _persist_conversation_context(self.session_id, self)
        
        # Track response quality
        quality_score = response_analysis.get("engagement_score", 0.5)
        self.response_quality_history.append((question_count, quality_score))
        
        # Update technical depth demonstrated
        if response_analysis.get("type") == "technical":
            keywords_count = len(response_analysis.get("keywords", []))
            self.technical_depth_demonstrated = min(1.0, keywords_count / 10.0)
    
    def get_coverage_score(self) -> float:
        """
        Calculate overall coverage score (0.0-1.0) based on coverage bins.
        
        Returns:
            Coverage score between 0.0 and 1.0
        """
        if not self.coverage:
            return 0.0
        
        # Normalize coverage values (assuming max meaningful value is 5)
        normalized_values = [min(value / 5.0, 1.0) for value in self.coverage.values()]
        return sum(normalized_values) / len(normalized_values) if normalized_values else 0.0
    
    def should_end_interview(self, end_requested: bool = False) -> bool:
        """Intelligently determine if interview should end"""
        if end_requested:
            return True
        
        # Coverage gating (config-driven)
        flags = get_interview_feature_flags()
        if flags.get("COVERAGE_GATING_ENABLED", False):
            quotas = get_interview_config().get("coverage_quotas", {}).get("default", {})
            unmet = []
            for k, min_count in quotas.items():
                if int(self.coverage.get(k, 0)) < int(min_count):
                    unmet.append(k)
            if unmet and self.conversation_depth < 14:
                # Do not end yet; push flow to cover remaining quotas
                return False
        
        # End if we have sufficient depth and engagement (increased threshold)
        if (self.conversation_depth >= 10 and 
            self.engagement_level > 0.7 and 
            len(self.topics_discussed) >= 5):
            return True
        
        # End if conversation is getting stale
        if self.conversation_depth >= 12:
            return True
            
        return False


# -------------------------
# Interview Evaluation System
# -------------------------
class InterviewStatus(str, Enum):
    """Interview outcome status"""
    STRONG_HIRE = "strong_hire"
    HIRE = "hire"
    MAYBE = "maybe"
    NO_HIRE = "no_hire"
    STRONG_NO_HIRE = "strong_no_hire"

class EvaluationCriteria:
    """Role-specific evaluation criteria for interviews"""
    
    def __init__(self, job_details: Dict[str, Any], role_category: str = None):
        self.job_details = job_details
        self.role_category = role_category
        
        # Import role-specific scoring configuration
        from agents.interview_scoring_config import ROLE_SPECIFIC_SCORING, CORE_SCORING_CATEGORIES
        
        # Get role-specific or core categories
        if role_category and role_category in ROLE_SPECIFIC_SCORING:
            scoring_config = ROLE_SPECIFIC_SCORING[role_category]
            self.criteria = {}
            for category, config in scoring_config["categories"].items():
                self.criteria[category] = {
                    "weight": config["weight"],
                    "score": 0.0,
                    "max_score": config["max_score"],
                    "evidence": []
                }
            self.keywords_map = scoring_config["keywords"]
        else:
            self.criteria = {}
            for category, config in CORE_SCORING_CATEGORIES.items():
                self.criteria[category] = {
                    "weight": config["weight"],
                    "score": 0.0,
                    "max_score": config["max_score"],
                    "evidence": []
                }
            self.keywords_map = {}
    
    def update_score(self, category: str, score: float, evidence: str):
        """Update score for a specific category (cumulative across turns)"""
        if category in self.criteria:
            # Add to existing score for cumulative evaluation across turns
            new_score = self.criteria[category]["score"] + score
            # Ensure score doesn't go below 0
            new_score = max(0.0, new_score)
            self.criteria[category]["score"] = min(
                new_score, 
                self.criteria[category]["max_score"]
            )
            self.criteria[category]["evidence"].append(evidence)
    
    def get_overall_score(self) -> float:
        """Calculate weighted overall score"""
        total_score = 0.0
        total_weight = 0.0
        
        for category, data in self.criteria.items():
            weighted_score = (data["score"] / data["max_score"]) * data["weight"]
            total_score += weighted_score
            total_weight += data["weight"]
        
        return (total_score / total_weight) * 10 if total_weight > 0 else 0.0
    
    def get_status(self) -> InterviewStatus:
        """Determine interview status based on overall score"""
        overall_score = self.get_overall_score()
        
        if overall_score >= 8.5:
            return InterviewStatus.STRONG_HIRE
        elif overall_score >= 7.0:
            return InterviewStatus.HIRE
        elif overall_score >= 5.5:
            return InterviewStatus.MAYBE
        elif overall_score >= 4.0:
            return InterviewStatus.NO_HIRE
        else:
            return InterviewStatus.STRONG_NO_HIRE
    
    def update_score_by_keywords(self, response_text: str, category: str):
        """Update score based on keyword detection"""
        if category not in self.keywords_map:
            return
        
        keywords = self.keywords_map[category]
        response_lower = response_text.lower()
        matches = [kw for kw in keywords if kw in response_lower]
        
        if matches:
            score_increment = min(2.0, len(matches) * 0.5)
            self.update_score(category, score_increment, f"Keywords: {', '.join(matches)}")
class InterviewEvaluator:
    """Main interview evaluation engine"""
    
    def __init__(self, job_details: Dict[str, Any], uid_context: Dict[str, Any] = None):
        # Role category will be determined later via async_init or set_role_category
        # We can't await in __init__ so we'll need to call async_init after creation
        role_category = None
        
        self.evaluation_criteria = EvaluationCriteria(job_details, role_category)
        # Persist topic for optional topic-focused weighting (set externally)
        self.interview_topic: Optional[str] = None
        self.interview_notes = []
        self.red_flags = []
        self.green_flags = []
        self.questions_asked = []
        self.candidate_responses = []
        self.latest_technical_skills_list = []
        self._uid_context = uid_context
        self._job_details = job_details
        self._role_initialized = False
        
        # Per-topic evaluation tracking
        # Maps topic_name -> EvaluationCriteria instance
        self.topic_evaluations: Dict[str, EvaluationCriteria] = {}
        # Maps question_index -> topic_name to track which topic each question belongs to
        self.question_topic_map: Dict[int, str] = {}
        # Track topic history (order of topics discussed)
        self.topic_history: List[Tuple[str, int]] = []  # (topic, question_count_when_switched)
    
    async def async_init(self):
        """Async initialization to classify role using LLM-driven classifier"""
        if self._role_initialized:
            return
        
        if self._uid_context:
            try:
                role_result = await classify_role_hybrid(self._uid_context, self._job_details)
                role_category = role_result.role_category
                # Update evaluation criteria with determined role
                self.evaluation_criteria = EvaluationCriteria(self._job_details, role_category)
                log.info(f"Role classification: {role_category} (confidence: {role_result.confidence_score:.2f})")
                self._role_initialized = True
            except Exception as e:
                log.warning(f"Role classification failed: {e}, using default criteria")
                self._role_initialized = True
        self.all_technical_skills = {}  # skill_name -> max_score_seen
        
        # Callback storage for various interview events
        self.callbacks = {
            'on_question_asked': [],
            'on_response_received': [],
            'on_skill_detected': [],
            'on_score_updated': [],
            'on_red_flag_triggered': [],
            'on_green_flag_triggered': [],
            'on_interview_started': [],
            'on_interview_ended': [],
            'on_evaluation_completed': []
        }

    # ------------------------
    # Topic-focused weighting
    # ------------------------
    def apply_topic_weighting(self, topic: str, technical_multiplier: float = None, comm_multiplier: float = None, other_multiplier: float = None) -> None:
        """
        Strengthen topic weighting by boosting technical/problem-solving categories and slightly
        damping generic/behavioral categories. We normalize weights to preserve total weight.
        Works universally across role configs by matching category names.
        Multipliers are now loaded from config (interview.yml) with backward-compatible defaults.
        
        Also initializes per-topic evaluation tracking if this is a new topic.
        """
        try:
            # Load multipliers from config if not provided
            if technical_multiplier is None or comm_multiplier is None or other_multiplier is None:
                try:
                    config = get_interview_config()
                    weighting_config = config.get("topic_weighting", {})
                    technical_multiplier = weighting_config.get("technical_multiplier", 1.6) if technical_multiplier is None else technical_multiplier
                    comm_multiplier = weighting_config.get("comm_multiplier", 1.2) if comm_multiplier is None else comm_multiplier
                    other_multiplier = weighting_config.get("other_multiplier", 0.6) if other_multiplier is None else other_multiplier
                except Exception as e:
                    log.warning(f"Failed to load topic weighting config, using defaults: {e}")
                    technical_multiplier = technical_multiplier or 1.6
                    comm_multiplier = comm_multiplier or 1.2
                    other_multiplier = other_multiplier or 0.6
            
            # Initialize per-topic evaluation if this is a new topic
            if topic and topic not in self.topic_evaluations:
                # Create a new EvaluationCriteria instance for this topic
                role_category = self.evaluation_criteria.role_category if hasattr(self.evaluation_criteria, 'role_category') else None
                topic_criteria = EvaluationCriteria(self._job_details, role_category)
                self.topic_evaluations[topic] = topic_criteria
                log.debug(f"Initialized per-topic evaluation tracking for '{topic}'")
            
            # Update current topic
            previous_topic = self.interview_topic
            self.interview_topic = topic
            
            # Track topic switch in history
            if previous_topic and previous_topic != topic:
                current_question_count = len(self.questions_asked)
                if not self.topic_history or self.topic_history[-1][0] != previous_topic:
                    self.topic_history.append((previous_topic, current_question_count))
                log.info(f"Topic switched from '{previous_topic}' to '{topic}' at question {current_question_count}")
            
            # Apply weighting to the current evaluation criteria (for backward compatibility)
            criteria = self.evaluation_criteria.criteria
            if not criteria:
                return
            
            # Log that topic weighting is being applied
            log.debug(f"Topic weighting applied for '{topic}' (tech={technical_multiplier}x, comm={comm_multiplier}x, other={other_multiplier}x)")

            # Classify categories by simple, robust heuristics (name-based)
            def is_technical(name: str) -> bool:
                n = name.lower()
                return any(k in n for k in [
                    "functional", "problem", "ml", "data", "architecture", "design",
                    "scalability", "reliability", "technical", "evaluation", "pipelines"
                ])

            def is_comm(name: str) -> bool:
                n = name.lower()
                return "communication" in n or "comm" in n

            # Compute boosted weights
            original_sum = sum(v["weight"] for v in criteria.values())
            if original_sum <= 0:
                return

            for cat, v in criteria.items():
                w = v["weight"]
                if is_technical(cat):
                    w *= technical_multiplier
                elif is_comm(cat):
                    w *= comm_multiplier
                else:
                    w *= other_multiplier
                v["weight"] = w

            # Normalize to preserve total
            new_sum = sum(v["weight"] for v in criteria.values())
            if new_sum > 0:
                scale = original_sum / new_sum
                for v in criteria.values():
                    v["weight"] *= scale
            
            # Also apply weighting to topic-specific criteria if it exists
            if topic and topic in self.topic_evaluations:
                topic_criteria = self.topic_evaluations[topic].criteria
                if topic_criteria:
                    topic_original_sum = sum(v["weight"] for v in topic_criteria.values())
                    if topic_original_sum > 0:
                        for cat, v in topic_criteria.items():
                            w = v["weight"]
                            if is_technical(cat):
                                w *= technical_multiplier
                            elif is_comm(cat):
                                w *= comm_multiplier
                            else:
                                w *= other_multiplier
                            v["weight"] = w
                        
                        topic_new_sum = sum(v["weight"] for v in topic_criteria.values())
                        if topic_new_sum > 0:
                            topic_scale = topic_original_sum / topic_new_sum
                            for v in topic_criteria.values():
                                v["weight"] *= topic_scale
        except Exception:
            # Non-fatal: keep default weights
            pass
    
    def _update_score_both(self, category: str, score: float, evidence: str, topic_criteria: EvaluationCriteria = None):
        """
        Helper method to update both overall and topic-specific evaluation scores.
        
        Args:
            category: Category name
            score: Score to add
            evidence: Evidence string
            topic_criteria: Topic-specific criteria (if None, will use current topic's criteria)
        """
        # Update overall criteria
        self.evaluation_criteria.update_score(category, score, evidence)
        
        # Update topic-specific criteria
        if topic_criteria is None:
            current_topic = self.interview_topic or "general"
            if current_topic in self.topic_evaluations:
                topic_criteria = self.topic_evaluations[current_topic]
        
        if topic_criteria and category in topic_criteria.criteria:
            topic_criteria.update_score(category, score, evidence)
    
    def get_topic_scores(self) -> Dict[str, Dict[str, Any]]:
        """
        Get evaluation scores broken down by topic.
        
        Returns:
            Dict mapping topic_name -> {
                "overall_score": float,
                "status": InterviewStatus,
                "category_scores": Dict[str, float],
                "question_count": int
            }
        """
        topic_scores = {}
        
        for topic, criteria in self.topic_evaluations.items():
            overall_score = criteria.get_overall_score()
            status = criteria.get_status()
            category_scores = {
                cat: data["score"] for cat, data in criteria.criteria.items()
            }
            # Count questions for this topic
            question_count = sum(1 for q_idx, t in self.question_topic_map.items() if t == topic)
            
            topic_scores[topic] = {
                "overall_score": overall_score,
                "status": status.value if hasattr(status, 'value') else str(status),
                "category_scores": category_scores,
                "question_count": question_count
            }
        
        return topic_scores
    
    def apply_llm_rubric(self, llm_rubric: Dict[str, Any]) -> None:
        """
        Apply LLM-generated evaluation rubric to the evaluator.
        
        Args:
            llm_rubric: Dictionary containing:
                - categories: Dict[str, Dict] with category definitions
                - keywords: List[str] domain-specific terms
                - focus_areas: List[str] key evaluation areas
        
        Example llm_rubric structure:
        {
            "categories": {
                "creative_vision": {
                    "weight": 0.3,
                    "description": "Artistic creativity and design innovation"
                },
                ...
            },
            "keywords": ["fashion", "illustration", ...],
            "focus_areas": ["Creative Vision", "Technical Execution", ...]
        }
        """
        try:
            if not llm_rubric or "categories" not in llm_rubric:
                log.warning("Invalid LLM rubric structure, skipping application")
                return
            
            # Get existing criteria structure
            criteria = self.evaluation_criteria.criteria
            if not criteria:
                log.warning("No existing criteria found to update")
                return
            
            # Update criteria weights and descriptions from LLM rubric
            llm_categories = llm_rubric.get("categories", {})
            updated_count = 0
            
            for llm_key, llm_data in llm_categories.items():
                # Try to find matching criterion (exact match or fuzzy match)
                matched_key = None
                llm_key_normalized = llm_key.lower().replace("_", " ").replace("-", " ")
                
                for existing_key in criteria.keys():
                    existing_normalized = existing_key.lower().replace("_", " ").replace("-", " ")
                    if llm_key_normalized == existing_normalized or llm_key_normalized in existing_normalized:
                        matched_key = existing_key
                        break
                
                # If no match, add as new criterion
                if not matched_key:
                    criteria[llm_key] = {
                        "weight": llm_data.get("weight", 0.1),
                        "description": llm_data.get("description", ""),
                        "is_llm_generated": True
                    }
                    updated_count += 1
                else:
                    # Update existing criterion
                    criteria[matched_key]["weight"] = llm_data.get("weight", criteria[matched_key].get("weight", 0.1))
                    if "description" in llm_data:
                        criteria[matched_key]["description"] = llm_data["description"]
                    criteria[matched_key]["is_llm_generated"] = True
                    updated_count += 1
            
            # Normalize weights to sum to 1.0
            total_weight = sum(c.get("weight", 0) for c in criteria.values())
            if total_weight > 0:
                for key in criteria:
                    criteria[key]["weight"] = criteria[key].get("weight", 0) / total_weight
            
            # Store LLM keywords for semantic scoring
            if "keywords" in llm_rubric:
                self.evaluation_criteria.llm_keywords = llm_rubric["keywords"]
            
            log.info(f"Applied LLM rubric: Updated {updated_count} criteria from LLM-generated rubric")
            
        except Exception as e:
            log.error(f"Failed to apply LLM rubric: {e}")

    class _PerSkillItem(BaseModel):
        skill: str
        score: int  # 0-10
        evidence: str = ""

    class _TechnicalPerSkillLLMResult(BaseModel):
        skills: List["InterviewEvaluator._PerSkillItem"] = []
        confidence: float = 0.0
        notes: str = ""
    
    # Callback management methods
    def add_callback(self, event_type: str, callback_func):
        """Add a callback function for a specific event type"""
        if event_type not in self.callbacks:
            raise ValueError(f"Unknown event type: {event_type}")
        self.callbacks[event_type].append(callback_func)
    
    def remove_callback(self, event_type: str, callback_func):
        """Remove a specific callback function"""
        if event_type in self.callbacks and callback_func in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback_func)
    
    def clear_callbacks(self, event_type: str = None):
        """Clear all callbacks or callbacks for a specific event type"""
        if event_type:
            if event_type in self.callbacks:
                self.callbacks[event_type].clear()
        else:
            for event_type in self.callbacks:
                self.callbacks[event_type].clear()
    
    async def _trigger_callbacks(self, event_type: str, *args, **kwargs):
        """Trigger all callbacks for a specific event type"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception as e:
                    log.error(f"Error in callback {callback.__name__}: {e}")

    async def _llm_per_skill_technical_eval(self,
                                            response: str,
                                            job_details: Dict[str, Any],
                                            response_analysis: Dict[str, Any]) -> "InterviewEvaluator._TechnicalPerSkillLLMResult":
        """Use LLM to identify which technical skills were assessed in this turn and score each.
        The LLM should infer skills from the answer content and optional job details.
        Returns a structured list of (skill, score[0-10], evidence) and overall confidence.
        """
        jd_skills = []
        try:
            if isinstance(job_details, dict):
                # Support common shapes like {"skills": [..]} or nested
                if "skills" in job_details and isinstance(job_details["skills"], list):
                    jd_skills = [str(s) for s in job_details["skills"]][:30]
        except Exception:
            jd_skills = []

        keywords = response_analysis.get("keywords", []) if isinstance(response_analysis, dict) else []
        detected_domains = response_analysis.get("detected_domains", []) if isinstance(response_analysis, dict) else []

        prompt = (
            "You are an expert technical interviewer. From the candidate's answer, "
            "identify which concrete technical skills OR technical concepts were assessed in THIS TURN and score each from 0 to 10. "
            "Treat frameworks, tools, and concepts as skills when appropriate (e.g., Python, React, SQL, APIs, data structures, system design, async programming, event-driven architecture, background workers, scalability, security). "
            "Focus only on items evidenced in the answer. Do not invent skills. "
            "Use job details and provided keywords/domains only for grounding/disambiguation.\n\n"
            f"JOB_DETAILS_SKILLS (optional): {json.dumps(jd_skills) if jd_skills else '[]'}\n"
            f"RESPONSE_KEYWORDS (optional): {json.dumps(keywords)}\n"
            f"DETECTED_DOMAINS (optional): {json.dumps(detected_domains)}\n\n"
            f"CANDIDATE_ANSWER:\n{response}"
        )

        try:
            data: TechnicalSkillsEvaluationResult = await invoke_structured_llm(
                prompt,
                TechnicalSkillsEvaluationResult,
                task_type=TaskType.INTERVIEW,
                preferred_model=settings.GEMINI_MODEL,
                agent_name="interview_technical_per_skill",
                temperature=0.2,
                timeout=_INTERVIEW_LLM_TIMEOUT,
                raise_on_fallback=False,
            )
            
            # Convert to internal dataclass format
            items = []
            for skill_item in data.skills:
                try:
                    items.append(InterviewEvaluator._PerSkillItem(
                        skill=skill_item.skill[:80],
                        score=skill_item.score,
                        evidence=skill_item.evidence[:400]
                    ))
                except Exception:
                    continue
            result = InterviewEvaluator._TechnicalPerSkillLLMResult(
                skills=items,
                confidence=data.confidence,
                notes=data.notes[:400]
            )
            return result
        except Exception:
            # Fallback: synthesize a minimal per-skill breakdown from keywords/domains
            try:
                normalized = [str(k).strip().lower() for k in keywords][:10]
                # Merge common multi-word concepts
                concept_map = {
                    "event-driven architecture": ["event-driven architecture", "event driven architecture"],
                    "async programming": ["async", "asynchronous", "async processing"],
                    "background workers": ["background workers", "workers", "queue workers"],
                    "apis": ["api", "apis", "rest"],
                    "data flow": ["data flow", "data pipeline"],
                    "scalability": ["scalability", "scale"],
                }
                counts = {}
                for concept, aliases in concept_map.items():
                    counts[concept] = int(any(a in " ".join(normalized) for a in aliases))
                # Add raw keywords that look like technologies (case-insensitive)
                for k in normalized:
                    k_clean = k.strip().lower()
                    if k_clean in ["react", "angular", "vue", "python", "javascript", "typescript", 
                                   "java", "sql", "mongodb", "redis", "docker", "kubernetes", 
                                   "figma", "sketch", "tailwind", "sass", "redux", "graphql"]:
                        counts[k_clean] = 1
                items = [InterviewEvaluator._PerSkillItem(skill=s, score=5, evidence="Detected in keywords/domains") for s, v in counts.items() if v]
                return InterviewEvaluator._TechnicalPerSkillLLMResult(skills=items, confidence=0.3, notes="Fallback keyword-derived breakdown")
            except Exception:
                return InterviewEvaluator._TechnicalPerSkillLLMResult()
        
    async def evaluate_response(self, question: str, response: str, response_analysis: Dict[str, Any], uid_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate a candidate response and update scores"""
        
        # Extract evaluation data from response analysis
        response_type = response_analysis.get("type", ResponseType.GENERAL)
        keywords = response_analysis.get("keywords", [])
        context_understanding = response_analysis.get("context_understanding", "")
        strengths_mentioned = response_analysis.get("strengths_mentioned", [])
        
        # Store interview data
        question_index = len(self.questions_asked)
        self.questions_asked.append(question)
        self.candidate_responses.append(response)
        
        # Track which topic this question belongs to
        current_topic = self.interview_topic or "general"
        self.question_topic_map[question_index] = current_topic
        
        # Get or create topic-specific evaluation criteria
        if current_topic not in self.topic_evaluations:
            role_category = self.evaluation_criteria.role_category if hasattr(self.evaluation_criteria, 'role_category') else None
            self.topic_evaluations[current_topic] = EvaluationCriteria(self._job_details, role_category)
            # Apply topic weighting if this is a specific topic (not "general")
            if current_topic != "general":
                self.apply_topic_weighting(current_topic)
        
        topic_criteria = self.topic_evaluations[current_topic]
        
        # NEW: Apply keyword-based scoring for all categories (both overall and topic-specific)
        for category in self.evaluation_criteria.criteria.keys():
            self.evaluation_criteria.update_score_by_keywords(response, category)
        
        # Also update topic-specific criteria
        for category in topic_criteria.criteria.keys():
            topic_criteria.update_score_by_keywords(response, category)
        
        # Trigger callbacks
        await self._trigger_callbacks('on_question_asked', question)
        await self._trigger_callbacks('on_response_received', response, question)
        
        # Evaluate based on response type and content (cumulative scoring)
        # Use dynamic categories based on role-specific scoring
        evaluation_result = {}
        for category, data in self.evaluation_criteria.criteria.items():
            evaluation_result[f"{category}_score"] = data["score"]
        evaluation_result["flags"] = []
        evaluation_result["notes"] = []
        
        # Technical Skills Evaluation (LLM-driven per-skill; fallback to keywords)
        if response_type == ResponseType.TECHNICAL or response_type == "technical":
            # Primary path: ask LLM to extract and score per-skill
            llm_per_skill = await self._llm_per_skill_technical_eval(
                response=response,
                job_details=self.evaluation_criteria.job_details,
                response_analysis=response_analysis
            )
            technical_skills_breakdown = {it.skill: it.score for it in llm_per_skill.skills if it.skill}
            technical_skills_list = sorted(technical_skills_breakdown.keys())
            self.latest_technical_skills_list = technical_skills_list

            if technical_skills_breakdown:
                # Average per-skill score (already 0-10 scale)
                technical_score = sum(technical_skills_breakdown.values()) / max(1, len(technical_skills_breakdown))
                technical_score = min(10.0, max(0.0, float(technical_score)))
                evidence_note = "Per-skill LLM evaluation | " + \
                    ", ".join([f"{k}:{v}" for k, v in list(technical_skills_breakdown.items())[:8]])
                # Use the first available category that might be technical-related
                technical_category = None
                for cat in self.evaluation_criteria.criteria.keys():
                    if "technical" in cat.lower() or "functional" in cat.lower():
                        technical_category = cat
                        break
                if technical_category:
                    self._update_score_both(technical_category, technical_score, evidence_note, topic_criteria)
                await self._trigger_callbacks('on_score_updated', 'technical_skills', technical_score, evidence_note)
                evaluation_result["technical_score"] = technical_score
                evaluation_result["technical_skills_breakdown"] = technical_skills_breakdown
                evaluation_result["technical_skills_list"] = technical_skills_list
                
                # Aggregate skills across turns (keep max score per skill)
                for skill, score in technical_skills_breakdown.items():
                    if skill not in self.all_technical_skills:
                        self.all_technical_skills[skill] = score
                        await self._trigger_callbacks('on_skill_detected', skill, score, 'new')
                    else:
                        old_score = self.all_technical_skills[skill]
                        self.all_technical_skills[skill] = max(self.all_technical_skills[skill], score)
                        if score > old_score:
                            await self._trigger_callbacks('on_skill_detected', skill, score, 'updated', old_score)
            else:
                # Fallback to lightweight keyword proxy if LLM returned nothing
                technical_keywords = ["python", "javascript", "react", "angular", "figma", "api", "database", "algorithm", "machine learning", "ai", "tensorflow", "pytorch", "ui", "ux", "design", "frontend", "backend", "optimization", "performance", "accessibility", "responsive", "component", "state", "props", "hooks", "animation", "motion", "prototype", "wireframe", "mockup", "user research", "usability", "testing"]
                technical_match = sum(1 for kw in keywords if kw.lower() in technical_keywords)
                technical_score = min(technical_match * 2, 10)
                # Choose an available category that best represents technical capability
                fallback_technical_category = None
                for cat in self.evaluation_criteria.criteria.keys():
                    if "technical" in cat.lower() or "functional" in cat.lower():
                        fallback_technical_category = cat
                        break
                if not fallback_technical_category:
                    # If nothing matches, just take the first available category to avoid KeyError
                    fallback_technical_category = next(iter(self.evaluation_criteria.criteria.keys()), None)
                if fallback_technical_category:
                    self._update_score_both(fallback_technical_category, technical_score, f"Technical response with {technical_match} relevant keywords", topic_criteria)
                await self._trigger_callbacks('on_score_updated', 'technical_skills', technical_score, f"Technical response with {technical_match} relevant keywords")
                evaluation_result["technical_score"] = technical_score
                evaluation_result["technical_skills_breakdown"] = {}
                evaluation_result["technical_skills_list"] = []
            
            if technical_score >= 7:
                self.green_flags.append(f"Strong technical knowledge demonstrated: {', '.join(keywords[:3])}")
                await self._trigger_callbacks('on_green_flag_triggered', f"Strong technical knowledge demonstrated: {', '.join(keywords[:3])}", technical_score)
            elif technical_score < 4:
                self.red_flags.append("Limited technical depth in response")
                await self._trigger_callbacks('on_red_flag_triggered', "Limited technical depth in response", technical_score)
        else:
            # For non-technical responses, still give some technical score based on keywords
            technical_keywords = ["python", "javascript", "react", "angular", "figma", "api", "database", "algorithm", "machine learning", "ai", "tensorflow", "pytorch", "ui", "ux", "design", "frontend", "backend", "optimization", "performance", "accessibility", "responsive", "component", "state", "props", "hooks", "animation", "motion", "prototype", "wireframe", "mockup", "user research", "usability", "testing"]
            technical_match = sum(1 for kw in keywords if kw.lower() in technical_keywords)
            technical_score = min(technical_match * 1.5, 8)  # Lower multiplier for non-technical responses
            # For non-technical answers, we avoid per-skill LLM evaluation to reduce noise
            self.latest_technical_skills_list = []
            evaluation_result["technical_score"] = technical_score
            evaluation_result["technical_skills_breakdown"] = {}
            evaluation_result["technical_skills_list"] = []
        
        # Experience Relevance Evaluation
        # Use enhanced keywords from response analysis if available
        if "experience_keywords" in response_analysis:
            experience_keywords = response_analysis["experience_keywords"]
            experience_match = len(experience_keywords)
        else:
            # Fallback to original logic
            experience_keywords = ["project", "experience", "worked", "built", "developed", "implemented", "managed", "led"]
            experience_match = sum(1 for kw in keywords if kw.lower() in experience_keywords)
        
        experience_score = min(experience_match * 1.5, 10)
        
        self._update_score_both("experience_relevance", experience_score, f"Experience-related response with {experience_match} relevant terms", topic_criteria)
        await self._trigger_callbacks('on_score_updated', 'experience_relevance', experience_score, f"Experience-related response with {experience_match} relevant terms")
        evaluation_result["experience_score"] = experience_score
        
        # Problem Solving Evaluation - use enhanced keywords from response analysis if available
        if "problem_solving_keywords" in response_analysis:
            problem_solving_keywords = response_analysis["problem_solving_keywords"]
            problem_solving_match = len(problem_solving_keywords)
        else:
            # Fallback to original logic
            problem_solving_keywords = get_problem_solving_keywords(uid_context) if uid_context else ["problem", "solve", "challenge", "approach", "solution", "debug", "fix", "optimize"]
            problem_solving_match = sum(1 for kw in keywords if kw.lower() in problem_solving_keywords)
        
        problem_solving_score = min(problem_solving_match * 2, 10)
        
        self._update_score_both("problem_solving", problem_solving_score, f"Problem-solving response with {problem_solving_match} relevant terms", topic_criteria)
        await self._trigger_callbacks('on_score_updated', 'problem_solving', problem_solving_score, f"Problem-solving response with {problem_solving_match} relevant terms")
        evaluation_result["problem_solving_score"] = problem_solving_score
        
        # Communication Evaluation
        word_count = len(response.split())
        communication_score = min(word_count / 5, 10)  # More detailed responses score higher
        
        self._update_score_both("communication", communication_score, f"Response length: {word_count} words", topic_criteria)
        await self._trigger_callbacks('on_score_updated', 'communication', communication_score, f"Response length: {word_count} words")
        evaluation_result["communication_score"] = communication_score
        
        if word_count < 10:
            self.red_flags.append("Very brief response - may indicate poor communication")
            await self._trigger_callbacks('on_red_flag_triggered', "Very brief response - may indicate poor communication", word_count)
        elif word_count > 50:
            self.green_flags.append("Detailed, comprehensive response")
            await self._trigger_callbacks('on_green_flag_triggered', "Detailed, comprehensive response", word_count)
        
        # Cultural Fit Evaluation - use enhanced keywords from response analysis if available
        if "cultural_keywords" in response_analysis:
            cultural_keywords = response_analysis["cultural_keywords"]
            cultural_match = len(cultural_keywords)
        else:
            # Fallback to original logic
            cultural_keywords = ["team", "collaborate", "learn", "grow", "help", "support", "mentor", "culture"]
            cultural_match = sum(1 for kw in keywords if kw.lower() in cultural_keywords)
        
        cultural_score = min(cultural_match * 2, 10)
        
        self._update_score_both("cultural_fit", cultural_score, f"Cultural fit indicators: {cultural_match} relevant terms", topic_criteria)
        await self._trigger_callbacks('on_score_updated', 'cultural_fit', cultural_score, f"Cultural fit indicators: {cultural_match} relevant terms")
        evaluation_result["cultural_fit_score"] = cultural_score
        
        # Add notes
        if context_understanding:
            evaluation_result["notes"].append(f"Context: {context_understanding}")
        
        if strengths_mentioned:
            evaluation_result["notes"].append(f"Strengths mentioned: {', '.join(strengths_mentioned)}")
        
        return evaluation_result
    
    async def evaluate_full_conversation(self, 
                                        conversation_history: List[Dict],
                                        uid_context: Dict[str, Any] = None) -> None:
        """
        Replay all Q&A pairs from conversation history to accumulate scores.
        Extracts (question, answer) pairs and calls evaluate_response for each.
        """
        if conversation_history is None:
            return
        
        # Trigger interview started callback
        await self._trigger_callbacks('on_interview_started', len(conversation_history))
            
        current_question = None
        qa_pairs_evaluated = 0
        
        for i, message in enumerate(conversation_history):
            if message.get("role") == "assistant":
                # Store the question for the next user response
                current_question = message.get("content", "")
            elif message.get("role") == "user" and current_question:
                # We have a Q&A pair, evaluate it
                answer = message.get("content", "")
                if answer.strip():
                    # Generate minimal response analysis for this answer
                    response_analysis = await self._generate_minimal_response_analysis(answer)
                    
                    # Evaluate this Q&A pair
                    await self.evaluate_response(
                        question=current_question,
                        response=answer,
                        response_analysis=response_analysis,
                        uid_context=uid_context
                    )
                    
                    qa_pairs_evaluated += 1
                
                # Reset for next pair
                current_question = None
        
        # Log summary only
        try:
            category_summaries = []
            for cat, data in self.evaluation_criteria.criteria.items():
                category_summaries.append(f"{cat}={data.get('score', 0)}")
            summary_str = ", ".join(category_summaries) if category_summaries else "no categories configured"
            log.info(f"Evaluated {qa_pairs_evaluated} Q&A pairs. Final scores: {summary_str}")
        except Exception:
            log.info(f"Evaluated {qa_pairs_evaluated} Q&A pairs")
        
        # Trigger interview ended and evaluation completed callbacks
        await self._trigger_callbacks('on_interview_ended', len(self.questions_asked), len(self.candidate_responses))
        await self._trigger_callbacks('on_evaluation_completed', self.evaluation_criteria.get_overall_score(), self.evaluation_criteria.get_status().value)
    
    async def _generate_minimal_response_analysis(self, answer: str) -> Dict[str, Any]:
        """Generate lightweight response analysis for conversation replay"""
        words = answer.split()
        word_count = len(words)
        
        # Simple keyword extraction
        answer_lower = answer.lower()
        technical_keywords = []
        
        # Common technical terms (expanded to match improved fallback analysis)
        tech_terms = ["react", "angular", "vue", "javascript", "python", "java", "sql", "api", 
                     "database", "figma", "design", "ui", "ux", "wireframe", "prototype", 
                     "testing", "performance", "optimization", "scalability", "architecture",
                     "aws", "docker", "kubernetes", "terraform", "ansible", "prometheus", 
                     "grafana", "jenkins", "gitlab", "ci/cd", "devops", "infrastructure",
                     "monitoring", "logging", "deployment", "automation", "cloud", "azure",
                     "gcp", "linux", "bash", "shell", "nginx", "apache", "redis", "mongodb",
                     "postgresql", "mysql", "elasticsearch", "kibana", "logstash", "elk",
                     "machine learning", "ml", "deep learning", "ai", "artificial intelligence",
                     "neural network", "cnn", "rnn", "lstm", "transformer", "bert", "gpt",
                     "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "matplotlib",
                     "data science", "data analysis", "statistics", "algorithms", "data structures",
                     "rest", "graphql", "microservices", "serverless", "lambda",
                     "frontend", "backend", "full-stack", "web development", "mobile development",
                     "unit test", "integration test", "tdd", "bdd",
                     "agile", "scrum", "kanban", "sre",
                     "security", "authentication", "authorization", "encryption", "ssl", "https",
                     "orm", "migration", "data augmentation", "preprocessing", "normalization", 
                     "face alignment", "robustness", "computer vision", "image processing", 
                     "nlp", "natural language processing", "langchain", "openai", "hugging face", 
                     "transformers", "fine-tuning"]
        
        for term in tech_terms:
            if term in answer_lower:
                technical_keywords.append(term)
        
        # Experience-related keywords (for experience_relevance scoring)
        experience_keywords = []
        experience_terms = ["project", "experience", "worked", "built", "developed", "implemented", 
                           "managed", "led", "created", "designed", "optimized", "improved", 
                           "delivered", "achieved", "accomplished", "years", "role", "position",
                           "company", "team", "client", "user", "application", "system"]
        
        for term in experience_terms:
            if term in answer_lower:
                experience_keywords.append(term)
        
        # Problem-solving keywords (for problem_solving scoring)
        problem_solving_keywords = []
        problem_terms = ["problem", "solve", "challenge", "approach", "solution", "debug", "fix", 
                        "optimize", "issue", "difficulty", "troubleshoot", "analyze", "investigate",
                        "strategy", "method", "process", "workflow", "iteration", "feedback",
                        "testing", "prototype", "wireframe", "research", "analysis", "breaking",
                        "refine", "improve", "enhance", "restructure", "rework", "simplify",
                        "cluttered", "navigation", "layout", "grid", "flow", "access", "features"]
        
        for term in problem_terms:
            if term in answer_lower:
                problem_solving_keywords.append(term)
        
        # Cultural fit keywords (for cultural_fit scoring)
        cultural_keywords = []
        cultural_terms = ["team", "collaborate", "learn", "grow", "help", "support", "mentor", 
                         "culture", "communication", "feedback", "share", "together", "cooperation",
                         "leadership", "initiative", "passion", "motivation", "values", "goals",
                         "contribute", "explore", "discuss", "opportunities", "collaboration",
                         "approach", "skills", "excited", "challenging", "projects", "impactful"]
        
        for term in cultural_terms:
            if term in answer_lower:
                cultural_keywords.append(term)
        
        # Determine response type
        response_type = "general"
        if technical_keywords:
            response_type = "technical"
        elif any(word in answer_lower for word in ["team", "collaborate", "work", "project"]):
            response_type = "behavioral"
        
        return {
            "type": response_type,
            "keywords": technical_keywords,
            "word_count": word_count,
            "context_understanding": f"Response analysis for conversation replay",
            "strengths_mentioned": technical_keywords[:3],
            "areas_to_explore": [],
            # Add experience and problem-solving keywords for better scoring
            "experience_keywords": experience_keywords,
            "problem_solving_keywords": problem_solving_keywords,
            "cultural_keywords": cultural_keywords
        }
    
    def get_interview_summary(self, question_count: int = None) -> Dict[str, Any]:
        """Generate comprehensive interview summary"""
        overall_score = self.evaluation_criteria.get_overall_score()
        status = self.evaluation_criteria.get_status()
        
        # Get detailed technical skills breakdown
        technical_skills_breakdown = {}
        if hasattr(self, 'all_technical_skills') and self.all_technical_skills:
            technical_skills_breakdown = dict(self.all_technical_skills)
        
        # Get per-topic scores if available
        topic_scores = {}
        if hasattr(self, 'get_topic_scores'):
            try:
                topic_scores = self.get_topic_scores()
            except Exception:
                pass
        
        summary = {
            "overall_score": overall_score,
            "status": status.value,
            "status_description": self._get_status_description(status),
            "criteria_scores": {
                category: {
                    "score": data["score"],
                    "max_score": data["max_score"],
                    "weight": data["weight"],
                    "evidence": data["evidence"][-3:],  # Last 3 pieces of evidence
                    "skills_mentioned": sorted(self.all_technical_skills.keys()) if category == "technical_skills" else []
                }
                for category, data in self.evaluation_criteria.criteria.items()
            },
            "technical_skills_breakdown": technical_skills_breakdown,  # Add detailed breakdown
            "green_flags": self.green_flags,
            "red_flags": self.red_flags,
            "questions_asked": question_count or len(self.questions_asked),
            "recommendation": self._get_recommendation(status, overall_score)
        }
        
        # Include per-topic scores if available and multiple topics were discussed
        if topic_scores and len(topic_scores) > 1:
            summary["topic_scores"] = topic_scores
        
        return summary
    
    def _get_status_description(self, status: InterviewStatus) -> str:
        """Get human-readable status description"""
        descriptions = {
            InterviewStatus.STRONG_HIRE: "Exceptional candidate - Strong hire recommendation",
            InterviewStatus.HIRE: "Good candidate - Hire recommendation",
            InterviewStatus.MAYBE: "Average candidate - Consider for other roles",
            InterviewStatus.NO_HIRE: "Below expectations - No hire",
            InterviewStatus.STRONG_NO_HIRE: "Poor fit - Strong no hire"
        }
        return descriptions.get(status, "Unknown status")
    
    def _get_recommendation(self, status: InterviewStatus, score: float) -> str:
        """Generate detailed recommendation"""
        if status in [InterviewStatus.STRONG_HIRE, InterviewStatus.HIRE]:
            return f"RECOMMENDED FOR HIRE - Score: {score:.1f}/10. Candidate demonstrates strong qualifications and cultural fit."
        elif status == InterviewStatus.MAYBE:
            return f"CONDITIONAL HIRE - Score: {score:.1f}/10. Consider for different role or additional interviews."
        else:
            return f"NOT RECOMMENDED - Score: {score:.1f}/10. Candidate does not meet minimum requirements."

    
    def get_next_focus(self) -> str:
        """Determine what to focus on next based on context"""
        if not self.topics_discussed:
            return "introduction_and_motivation"
        elif len(self.topics_discussed) < 2:
            return "background_and_experience"
        elif "technical" not in self.topics_discussed:
            return "technical_skills"
        elif "teamwork" not in self.topics_discussed:
            return "collaboration_and_leadership"
        elif not self.candidate_questions:
            return "candidate_questions"
        else:
            return "wrap_up_and_next_steps"

# Dynamic interview flow - no rigid state machine
INTERVIEW_FLOW_CONFIG = {
    "min_questions": 3,
    "max_questions": 15,
    "target_engagement": 0.7,
    "min_topics": 3,
    "timeout_minutes": 15
}

# -------------------------
# Enhanced PII Anonymization Layer
# -------------------------
class PIIAnonymizer:
    """Enhanced PII anonymization with hash-based placeholders"""
    
    def __init__(self):
        self.pii_map = {}
        self.reverse_map = {}
        self.session_id = None
    
    def set_session_id(self, session_id: str):
        """Set session ID for consistent anonymization across conversation"""
        self.session_id = session_id
    
    def _generate_hash_placeholder(self, original: str, prefix: str) -> str:
        """Generate consistent hash-based placeholder for PII"""
        # Use session ID + original for consistent hashing across conversation
        hash_input = f"{self.session_id or 'default'}_{original}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        placeholder = f"{prefix}_{hash_value}"
        
        self.pii_map[original] = placeholder
        self.reverse_map[placeholder] = original
        return placeholder
    
    def anonymize_name(self, name: str) -> str:
        """Replace name with CANDIDATE_a3f8 format"""
        if not name or name.lower() in ["candidate", "applicant", "interviewee"]:
            return "CANDIDATE_A"
        if name in self.pii_map:
            return self.pii_map[name]
        return self._generate_hash_placeholder(name, "CANDIDATE")
    
    def anonymize_email(self, email: str) -> str:
        """Replace email with candidate@domain.com format"""
        if not email:
            return ""
        if email in self.pii_map:
            return self.pii_map[email]
        domain = email.split('@')[-1] if '@' in email else "domain.com"
        return f"candidate@{domain}"
    
    def anonymize_phone(self, phone: str) -> str:
        """Replace phone with XXX-XXX-XXXX format"""
        if not phone:
            return ""
        return "XXX-XXX-XXXX"
    
    def anonymize_company(self, company: str) -> str:
        """Replace company name with COMPANY_7b2e format"""
        if not company or len(company) < 2:
            return company
        if company in self.pii_map:
            return self.pii_map[company]
        return self._generate_hash_placeholder(company, "COMPANY")
    
    def anonymize_text(self, text: str, names: List[str], companies: List[str]) -> str:
        """Remove PII from free text with enhanced patterns"""
        if not text:
            return text
            
        anonymized = text
        
        # Replace names with case-insensitive matching
        for name in names:
            if name and len(name) > 1:
                placeholder = self.anonymize_name(name)
                pattern = re.compile(re.escape(name), re.IGNORECASE)
                anonymized = pattern.sub(placeholder, anonymized)
        
        # Replace companies with case-insensitive matching
        for company in companies:
            if company and len(company) > 1:
                placeholder = self.anonymize_company(company)
                pattern = re.compile(re.escape(company), re.IGNORECASE)
                anonymized = pattern.sub(placeholder, anonymized)
        
        # Enhanced email pattern matching
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        anonymized = re.sub(email_pattern, 'EMAIL_REDACTED', anonymized)
        
        # Enhanced phone pattern matching
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
            r'\b\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b',  # International
            r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b'  # Parentheses format
        ]
        for pattern in phone_patterns:
            anonymized = re.sub(pattern, 'PHONE_REDACTED', anonymized)
        
        # Additional PII patterns
        # URLs (generic)
        url_pattern = r'(https?://[^\s)]+)'
        anonymized = re.sub(url_pattern, 'URL_REDACTED', anonymized)
        # Social handles like @username
        handle_pattern = r'(?<!\w)@([A-Za-z0-9_\.\-]{2,30})'
        anonymized = re.sub(handle_pattern, 'HANDLE_REDACTED', anonymized)
        # Organization names with common suffixes (Inc, LLC, Ltd, Corp, GmbH)
        org_suffix_pattern = r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,2}\s(?:Inc|LLC|Ltd|Corporation|Corp|GmbH))\b'
        anonymized = re.sub(org_suffix_pattern, 'COMPANY_GENERIC', anonymized)
        # US SSN
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        anonymized = re.sub(ssn_pattern, 'SSN_REDACTED', anonymized)
        
        return anonymized
    
    def deanonymize_text(self, text: str) -> str:
        """Restore original PII in text for final output"""
        if not text:
            return text
            
        deanonymized = text
        for placeholder, original in self.reverse_map.items():
            deanonymized = deanonymized.replace(placeholder, original)
        return deanonymized
    
    def get_anonymization_summary(self) -> Dict[str, Any]:
        """Get summary of anonymized PII for audit purposes"""
        return {
            "total_anonymized_items": len(self.pii_map),
            "anonymized_names": [k for k in self.pii_map.keys() if k in self.reverse_map.values()],
            "anonymized_companies": [k for k in self.pii_map.keys() if k.startswith("COMPANY_")],
            "session_id": self.session_id
        }
# -------------------------
# Answer Sanitization for Safe Prompt Embedding
# -------------------------
def _sanitize_answer_for_prompt(answer: str) -> str:
    """
    Sanitizes user answer for safe embedding in LLM prompts and JSON parsing.
    
    Handles:
    - Special characters that break JSON parsing (quotes, newlines, backslashes)
    - Unicode normalization
    - Preserves meaningful content while preventing parsing errors
    
    Args:
        answer: Raw user answer string
        
    Returns:
        Sanitized string safe for embedding in prompts and JSON
    """
    if not isinstance(answer, str):
        return ""
    
    # Step 1: Normalize unicode characters
    import unicodedata
    normalized = unicodedata.normalize('NFKC', answer)
    
    # Step 2: Escape special characters that break JSON/string embedding
    # Replace newlines and carriage returns with spaces
    sanitized = normalized.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    
    # Replace tabs with spaces
    sanitized = sanitized.replace('\t', ' ')
    
    # Escape backslashes first (before other escapes)
    sanitized = sanitized.replace('\\', '\\\\')
    
    # Escape double quotes (critical for JSON)
    sanitized = sanitized.replace('"', '\\"')
    
    # Step 3: Remove or replace problematic control characters
    # Keep printable characters and common whitespace
    sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', sanitized)
    
    # Step 4: Normalize whitespace (multiple spaces to single space)
    sanitized = re.sub(r'\s+', ' ', sanitized)
    sanitized = sanitized.strip()
    
    # Step 5: Limit length to prevent prompt injection and excessive tokens
    max_length = 2000  # Reasonable limit for interview responses
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "... [truncated]"
    
    return sanitized

# -------------------------
# Intelligence Layer - Response Analysis & State Management
# -------------------------

# Track consecutive invalid answers per session to avoid infinite loops
_INVALID_ANSWER_COUNT: Dict[str, int] = {}


def is_invalid_answer(text: str) -> bool:
    """Detect invalid or nonsensical answers like random characters, repeated letters, etc.

    This is intentionally conservative to avoid misclassifying short but valid answers.
    """
    if not isinstance(text, str):
        return False

    text = text.strip()
    if len(text) < 2:
        return False

    text_lower = text.lower()
    # Remove spaces for some checks
    text_no_spaces = text_lower.replace(" ", "")

    # Check for repeated single characters (e.g., "qqq", "aaa", "xxx", "q q q")
    if len(text_no_spaces) >= 3 and len(set(text_no_spaces)) <= 2:
        # Allow some valid short words like "no", "ok", "hi"
        valid_single_words = {
            "no",
            "ok",
            "hi",
            "oh",
            "ah",
            "eh",
            "um",
            "uh",
            "yes",
            "yeah",
        }
        if text_lower not in valid_single_words:
            return True

    # Check for random character sequences (e.g., "asdf", "qwerty", "zxcv")
    # If more than 60% of characters are consonants in sequence without vowels
    if len(text_lower) >= 3:
        consonants_only = re.sub(r"[aeiou\s]", "", text_lower)
        if len(consonants_only) > len(text_lower) * 0.6:
            # Allow valid words that happen to have many consonants
            valid_words = {
                "why",
                "try",
                "cry",
                "fly",
                "sky",
                "my",
                "by",
                "dry",
                "gym",
                "lynx",
            }
            if text_lower not in valid_words:
                return True

    # Check for non-alphabetic patterns (mostly numbers, symbols, etc.)
    alpha_chars = sum(1 for c in text_lower if c.isalpha())
    if len(text_lower) >= 3 and alpha_chars < len(text_lower) * 0.5:
        return True

    # For very short (2-char) responses, treat clearly non-word patterns as invalid
    if len(text_lower) == 2:
        common_two_letter_words = {
            "no",
            "ok",
            "hi",
            "oh",
            "ah",
            "eh",
            "um",
            "uh",
            "my",
            "by",
            "we",
            "me",
            "he",
            "she",
            "it",
            "is",
            "am",
            "an",
            "as",
            "at",
            "be",
            "do",
            "go",
            "if",
            "in",
            "on",
            "or",
            "so",
            "to",
            "up",
            "us",
        }
        if text_lower not in common_two_letter_words and not text_lower.isalpha():
            return True

    return False


class ResponseAnalyzer:
    """Analyzes candidate responses for intelligence-driven follow-ups"""
    
    @staticmethod
    async def analyze_response(answer: str, is_first_request: bool = False) -> Dict[str, Any]:
        """Comprehensive response analysis"""
        if not answer or not answer.strip():
            # For first request, empty answer is expected - don't mark as unresponsive
            if is_first_request:
                return {
                    "type": ResponseType.GENERAL,
                    "confidence": 0.5,
                    "keywords": [],
                    "sentiment": "neutral",
                    "length_score": 0.0,
                    "engagement_score": 0.0
                }
            else:
                return {
                    "type": ResponseType.UNRESPONSIVE,
                    "confidence": 1.0,
                    "keywords": [],
                    "sentiment": "neutral",
                    "length_score": 0.0,
                    "engagement_score": 0.0
                }
        
        answer_lower = answer.lower().strip()
        word_count = len(answer.split())

        # Check for invalid answers first
        if is_invalid_answer(answer):
            log.info("Invalid / nonsensical answer detected; prompting for re-answer")
            return {
                "type": ResponseType.UNRESPONSIVE,
                "confidence": 0.95,
                "keywords": [],
                "sentiment": "neutral",
                "length_score": 0.0,
                "engagement_score": 0.0,
                "is_invalid": True,  # Flag for invalid answer
                "requires_reanswer": True  # Flag to prompt for re-answer
            }
        
        # LLM-based unresponsiveness and response type detection (replaces hardcoded patterns)
        # Fast path: Check for very obvious cases first (word count < 3)
        if word_count < 3:
            return {
                "type": ResponseType.UNRESPONSIVE,
                "confidence": 0.9,
                "keywords": [],
                "sentiment": "neutral",
                "length_score": 0.1,
                "engagement_score": 0.1
            }
        
        # Use LLM for intelligent response classification with structured output (Issue 5.1)
        try:
            sanitized_answer = _sanitize_answer_for_prompt(answer)
            classification_prompt = f"""Analyze this interview response and classify it.

RESPONSE: "{sanitized_answer}"

Classify the response into one of these types:
- UNRESPONSIVE: Very brief, uncooperative, non-engaging, or INVALID/NONSENSICAL answers (e.g., "no", "nothing", "good night", "i won't tell", single word answers, random characters like "qqq", "asdf", repeated letters like "aaa", "xxx", nonsensical text)
- TECHNICAL: Mentions technical skills, tools, technologies, implementation details
- BEHAVIORAL: Discusses teamwork, collaboration, leadership, interpersonal situations
- BRIEF: Short but polite responses (e.g., "yes", "maybe", "okay")
- GENERAL: Normal conversational response that doesn't fit above categories

IMPORTANT: If the response is invalid/nonsensical (random characters, repeated letters, meaningless text), classify it as UNRESPONSIVE with high confidence.

Also analyze sentiment, key technical/behavioral keywords, and engagement level."""

            classification_data: ResponseClassificationResult = await invoke_structured_llm(
                classification_prompt,
                ResponseClassificationResult,
                task_type=TaskType.INTERVIEW,
                preferred_model=settings.GEMINI_MODEL,
                agent_name="interview_response_classification",
                temperature=0.1,
                timeout=_INTERVIEW_LLM_TIMEOUT,
                raise_on_fallback=False,
            )
            
            # Map LLM response type to ResponseType enum
            type_mapping = {
                "UNRESPONSIVE": ResponseType.UNRESPONSIVE,
                "TECHNICAL": ResponseType.TECHNICAL,
                "BEHAVIORAL": ResponseType.BEHAVIORAL,
                "BRIEF": ResponseType.BRIEF,
                "GENERAL": ResponseType.GENERAL
            }
            
            detected_type = type_mapping.get(classification_data.type, ResponseType.GENERAL)
            length_score = min(1.0, word_count / 50)
            
            return {
                "type": detected_type,
                "confidence": classification_data.confidence,
                "keywords": classification_data.keywords,
                "sentiment": classification_data.sentiment,
                "length_score": length_score,
                "engagement_score": classification_data.engagement_score,
                "word_count": word_count
            }
        except Exception as e:
            log.warning(f"LLM-based response classification failed: {e}, using fallback")
        
        # Fallback: Simple rule-based classification if LLM fails
        detected_type = ResponseType.GENERAL
        confidence = 0.5
        matched_keywords = []
        sentiment = "neutral"
        length_score = min(1.0, word_count / 50)
        engagement_score = 0.5
        
        return {
            "type": detected_type,
            "confidence": confidence,
            "keywords": matched_keywords,
            "sentiment": sentiment,
            "length_score": length_score,
            "engagement_score": engagement_score,
            "word_count": word_count
        }

class IntelligentResponseAnalyzer:
    """LLM-powered response analysis for deep context understanding"""
    
    @staticmethod
    async def analyze_response_intelligently(answer: str, conversation_history: List[Dict], 
                                          candidate_info: Dict[str, str], job_details: Dict[str, Any],
                                          is_first_request: bool = False, uid_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Use LLM to deeply understand response context and meaning"""
        
        if not answer.strip() and not is_first_request:
            return {
                "type": ResponseType.UNRESPONSIVE,
                "confidence": 1.0,
                "keywords": [],
                "sentiment": "neutral",
                "length_score": 0.0,
                "engagement_score": 0.0,
                "word_count": 0,
                "context_understanding": "No response provided",
                "follow_up_suggestions": [],
                "strengths_mentioned": [],
                "areas_to_explore": []
            }
        
        # Build context for LLM analysis - use compressed history
        compressed_history = _compress_conversation_history(conversation_history, max_recent_turns=4)
        recent_context = compressed_history[-4:] if len(compressed_history) > 4 else compressed_history
        
        # Use aggressive UID context for analysis
        uid_context = uid_context or {}
        candidate_name = get_candidate_name(uid_context, candidate_info, [])
        candidate_roles = get_job_title(uid_context, job_details, candidate_info)
        candidate_skills = get_candidate_skills(uid_context, candidate_info)
        candidate_experience = get_experience_years(uid_context, candidate_info)
        
        # Sanitize answer for safe embedding in prompt
        sanitized_answer = _sanitize_answer_for_prompt(answer)
        
        analysis_prompt = f"""You are an expert interview analyst. Analyze this candidate's response in the context of their interview.

CANDIDATE BACKGROUND:
- Name: {candidate_name}
- Roles: {candidate_roles}
- Experience: {candidate_experience} years
- Skills: {candidate_skills}

JOB REQUIREMENTS:
{json.dumps(job_details, indent=2) if job_details else f"{candidate_roles} position"}

RECENT CONVERSATION CONTEXT:
{json.dumps(recent_context, indent=2) if recent_context else "First response"}

CANDIDATE'S RESPONSE:
"{sanitized_answer}"

Analyze the response type (technical/behavioral/brief/general/unresponsive), confidence, keywords, sentiment, engagement level, context understanding, follow-up suggestions, strengths mentioned, and areas to explore."""

        try:
            analysis_data: IntelligentResponseAnalysisResult = await invoke_structured_llm(
                analysis_prompt,
                IntelligentResponseAnalysisResult,
                task_type=TaskType.INTERVIEW,
                preferred_model=settings.GEMINI_MODEL,
                agent_name="interview_intelligent_analysis",
                temperature=0.2,
                timeout=_INTERVIEW_LLM_TIMEOUT,
                raise_on_fallback=False,
            )
            
            # Map response type
            type_mapping = {
                "technical": ResponseType.TECHNICAL,
                "behavioral": ResponseType.BEHAVIORAL,
                "brief": ResponseType.BRIEF,
                "general": ResponseType.GENERAL,
                "unresponsive": ResponseType.UNRESPONSIVE
            }
            
            return {
                "type": type_mapping.get(analysis_data.type.lower(), ResponseType.GENERAL),
                "confidence": analysis_data.confidence,
                "keywords": analysis_data.keywords,
                "sentiment": analysis_data.sentiment,
                "length_score": analysis_data.length_score,
                "engagement_score": analysis_data.engagement_score,
                "word_count": analysis_data.word_count or len(answer.split()),
                "context_understanding": analysis_data.context_understanding,
                "follow_up_suggestions": analysis_data.follow_up_suggestions,
                "strengths_mentioned": analysis_data.strengths_mentioned,
                "areas_to_explore": analysis_data.areas_to_explore
            }
            
        except Exception as e:
            log.error(f"Error in intelligent response analysis: {e}, using fallback")
            return await IntelligentResponseAnalyzer._fallback_analysis(answer, is_first_request)
    
    @staticmethod
    async def _fallback_analysis(answer: str, is_first_request: bool = False) -> Dict[str, Any]:
        """Fallback analysis when LLM analysis fails"""
        word_count = len(answer.split())
        answer_lower = answer.lower()
        
        # LLM-based keyword and type extraction using structured output (Issue 5.1)
        try:
            sanitized_answer = _sanitize_answer_for_prompt(answer)
            keyword_extraction_prompt = f"""Extract technical keywords, action verbs, and classify this interview response.

RESPONSE: "{sanitized_answer}"

Tasks:
1. Extract all technical keywords mentioned (tools, technologies, frameworks, languages, platforms, etc.)
2. Extract action verbs indicating experience (developed, built, created, implemented, etc.)
3. Determine response type: TECHNICAL, BEHAVIORAL, or GENERAL
4. Assess engagement level (0.0-1.0)"""

            extraction_data: KeywordExtractionResult = await invoke_structured_llm(
                keyword_extraction_prompt,
                KeywordExtractionResult,
                task_type=TaskType.INTERVIEW,
                preferred_model=settings.GEMINI_MODEL,
                agent_name="interview_keyword_extraction",
                temperature=0.1,
                timeout=_INTERVIEW_LLM_TIMEOUT,
                raise_on_fallback=False,
            )
            
            detected_keywords = extraction_data.technical_keywords
            detected_actions = extraction_data.action_verbs
            confidence = extraction_data.confidence
            engagement_score = extraction_data.engagement_score
            
            # Map to ResponseType enum
            type_mapping = {
                "TECHNICAL": ResponseType.TECHNICAL,
                "BEHAVIORAL": ResponseType.BEHAVIORAL,
                "GENERAL": ResponseType.GENERAL
            }
            response_type = type_mapping.get(extraction_data.response_type, ResponseType.GENERAL)
        except Exception as e:
            log.warning(f"LLM-based keyword extraction failed in fallback: {e}, using minimal fallback")
            detected_keywords = []
            detected_actions = []
            response_type = ResponseType.GENERAL
            confidence = 0.3
            engagement_score = 0.3
        
        # Use LLM-extracted engagement score, or calculate fallback if not available
        if 'engagement_score' not in locals() or engagement_score is None:
            engagement_score = min(1.0, 0.3 + (len(detected_keywords) * 0.1) + (len(detected_actions) * 0.05))
        
        # Extract strengths mentioned
        strengths_mentioned = []
        if detected_keywords:
            strengths_mentioned.extend(detected_keywords[:3])  # Top 3 technical skills
        if detected_actions:
            strengths_mentioned.extend(detected_actions[:2])  # Top 2 actions
        
        result = {
            "type": response_type,
            "confidence": confidence,
            "keywords": detected_keywords,
            "sentiment": "positive" if detected_keywords else "neutral",
            "length_score": min(word_count / 50, 1.0),
            "engagement_score": engagement_score,
            "word_count": word_count,
            "context_understanding": f"Technical response with {len(detected_keywords)} keywords identified" if detected_keywords else "Response received, basic analysis",
            "follow_up_suggestions": ["Ask for more technical details"] if detected_keywords else ["Ask for more details"],
            "strengths_mentioned": strengths_mentioned,
            "areas_to_explore": []
        }
        
        # Log fallback analysis results (debug level)
        log.debug(f"Fallback analysis - Type: {response_type}, Confidence: {confidence}, Engagement: {engagement_score}")
        
        return result

class StateManager:
    """Manages LangGraph state transitions and timing with enhanced intelligent staging"""
    
    @staticmethod
    def _get_adaptive_timeout(base_timeout: float, engagement_score: float, 
                             adaptive_config: Dict[str, Any] = None) -> float:
        """
        Calculate adaptive timeout based on candidate engagement.
        
        Args:
            base_timeout: Base timeout in minutes
            engagement_score: Candidate engagement score (0.0-1.0)
            adaptive_config: Adaptive timeout configuration
        
        Returns:
            Adjusted timeout in minutes
        """
        if not adaptive_config or not adaptive_config.get("enabled", False):
            return base_timeout
        
        engagement_threshold = adaptive_config.get("engagement_threshold", 0.7)
        extension_multiplier = adaptive_config.get("extension_multiplier", 1.5)
        max_extension = adaptive_config.get("max_extension", 2.0)
        
        if engagement_score >= engagement_threshold:
            extended_timeout = base_timeout * extension_multiplier
            # Cap at max_extension times base timeout
            return min(extended_timeout, base_timeout * max_extension)
        
        return base_timeout
    
    @staticmethod
    def _check_skip_conditions(current_state: InterviewState, response_analysis: Dict[str, Any],
                               question_count: int, coverage_score: float = 0.0) -> Optional[InterviewState]:
        """
        Check if stage skipping conditions are met.
        
        Args:
            current_state: Current interview state
            response_analysis: Analysis of candidate's response
            question_count: Current question count
            coverage_score: Coverage score (0.0-1.0)
        
        Returns:
            Next state if skip condition met, None otherwise
        """
        try:
            config = get_interview_config()
            staging_config = config.get("intelligent_staging", {})
            
            if not staging_config.get("allow_skip_stage", False):
                return None
            
            skip_conditions = staging_config.get("skip_conditions", [])
            if not skip_conditions:
                return None
            
            # Extract metrics for condition evaluation
            confidence = response_analysis.get("confidence", 0.5)
            word_count = response_analysis.get("word_count", 0)
            engagement_score = response_analysis.get("engagement_score", 0.5)
            
            for condition_config in skip_conditions:
                condition_str = condition_config.get("condition", "")
                action = condition_config.get("action", "")
                
                # Simple condition evaluation (can be enhanced with proper parser)
                # For now, check common patterns
                if "skip_to_deep_dive" in action:
                    if (confidence > 0.9 and word_count > 80 and engagement_score > 0.85):
                        return InterviewState.TOPIC_DEEP_DIVE
                
                if "skip_to_candidate_questions" in action:
                    if coverage_score > 0.9 and question_count >= 5:
                        return InterviewState.CANDIDATE_QUESTIONS
            
        except Exception as e:
            log.warning(f"Error checking skip conditions: {e}")
        
        return None
    
    @staticmethod
    def get_next_state(current_state: InterviewState, question_count: int, 
                      response_analysis: Dict[str, Any], start_time: float,
                      coverage_score: float = 0.0) -> InterviewState:
        """
        Determine next state based on current state, progress, and response analysis.
        Enhanced with adaptive timeouts and stage skipping.
        
        Args:
            current_state: Current interview state
            question_count: Current question number
            response_analysis: Analysis of candidate's response
            start_time: Interview start time (timestamp)
            coverage_score: Coverage score (0.0-1.0) for skip condition evaluation
        """
        config = STATE_CONFIGURATIONS.get(current_state, {})
        max_questions = config.get("max_questions", 1)
        base_timeout = config.get("timeout_minutes", 5)
        next_states = config.get("next_states", [])
        adaptive_timeout_enabled = config.get("adaptive_timeout", False)
        
        # Get adaptive timeout configuration
        adaptive_timeout_config = None
        if adaptive_timeout_enabled:
            try:
                interview_config = get_interview_config()
                staging_config = interview_config.get("intelligent_staging", {})
                adaptive_timeout_config = staging_config.get("adaptive_timeout", {})
            except Exception:
                pass
        
        # Calculate adaptive timeout
        engagement_score = response_analysis.get("engagement_score", 0.5)
        timeout_minutes = StateManager._get_adaptive_timeout(
            base_timeout, engagement_score, adaptive_timeout_config
        )
        
        # Check timeout (using adaptive timeout)
        elapsed_minutes = (time.time() - start_time) / 60
        if elapsed_minutes > timeout_minutes:
            log.info(
                f"State timeout reached for {current_state} after {elapsed_minutes:.1f} minutes "
                f"(base: {base_timeout}, adaptive: {timeout_minutes})"
            )
            return InterviewState.CLOSING
        
        # Check skip conditions (enhanced feature)
        if config.get("allow_skip", False):
            skipped_state = StateManager._check_skip_conditions(
                current_state, response_analysis, question_count, coverage_score
            )
            if skipped_state:
                log.info(
                    f"Stage skip triggered: {current_state} -> {skipped_state} "
                    f"(confidence: {response_analysis.get('confidence', 0):.2f}, "
                    f"engagement: {engagement_score:.2f})"
                )
                return skipped_state
        
        # Check if max questions reached for current state
        if question_count >= max_questions:
            if next_states:
                return next_states[0]  # Move to first available next state
            else:
                return InterviewState.CLOSING
        
        # Special logic for main questions state (config-driven thresholds)
        if current_state == InterviewState.MAIN_QUESTIONS:
            th = get_interview_thresholds()
            min_engagement = th.get("main_to_candidate_engagement", 0.7)
            min_questions = th.get("main_to_candidate_min_questions", 6)
            if response_analysis.get("engagement_score", 0) > min_engagement and question_count >= min_questions:
                return InterviewState.CANDIDATE_QUESTIONS
            # Check for unresponsive behavior
            if response_analysis.get("type") == ResponseType.UNRESPONSIVE:
                return InterviewState.FEEDBACK
        
        # Default: stay in current state
        return current_state
    
    @staticmethod
    def should_end_interview(current_state: InterviewState, question_count: int, 
                           response_analysis: Dict[str, Any], end_requested: bool) -> bool:
        """Determine if interview should end"""
        
        if end_requested:
            return True
        
        if current_state == InterviewState.COMPLETED:
            return True
        
        # Don't end on first request (empty answer is expected)
        if question_count == 0:
            return False
        
        # End if candidate is consistently unresponsive (but not on first request)
        th = get_interview_thresholds()
        unresp_conf = th.get("unresponsive_confidence_to_end", 0.8)
        if (response_analysis.get("type") == ResponseType.UNRESPONSIVE and 
            response_analysis.get("confidence", 0) > unresp_conf and
            question_count > 2):  # Allow a few attempts before giving up
            return True
        
        # End if we've covered all major states
        if current_state in [InterviewState.FEEDBACK, InterviewState.CLOSING]:
            return True
        
        return False

# -------------------------
# Helper Functions
# -------------------------

# Domain-agnostic context extractor
def get_domain_context(uid_context: Dict[str, Any], job_details: Dict[str, Any]) -> Dict[str, Any]:
    skills: List[str] = []
    try:
        if uid_context and isinstance(uid_context, dict) and "skills_parser" in uid_context:
            for s in uid_context["skills_parser"].get("skills", []) or []:
                name = (s or {}).get("SkillName")
                if isinstance(name, str):
                    name = name.strip()
                if name:
                    skills.append(name)
    except Exception:
        pass

    jd_terms: List[str] = []
    try:
        if isinstance(job_details, dict):
            for k in ["title", "requirements", "responsibilities", "technologies", "skills_required", "department"]:
                v = job_details.get(k)
                if isinstance(v, str):
                    if v:
                        jd_terms.append(v)
                elif isinstance(v, list):
                    jd_terms.extend([str(x) for x in v if x])
    except Exception:
        pass

    # De-duplicate while preserving order
    seen = set()
    dedup_skills: List[str] = []
    for s in skills:
        if s not in seen:
            dedup_skills.append(s)
            seen.add(s)

    return {
        "skill_keywords": dedup_skills[:10],
        "job_keywords": " ".join([t for t in jd_terms if t])[:600],
    }

# -------------------------
# Adaptive Question Generation with Intelligence Layer
# -------------------------
async def generate_adaptive_question(
    candidate_info: Dict[str, str],
    persona: PersonaType,
    current_state: InterviewState,
    question_count: int,
    conversation_history: List[Dict],
    job_details: Dict[str, Any],
    response_analysis: Dict[str, Any],
    names: List[str] = None,
    interview_topic: Optional[str] = None,
    structured_resume: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
) -> str:
    """Generate intelligent, adaptive interview questions based on state and response analysis.
    
    If interview_topic is provided, generates topic-focused questions.
    """
    # Optional flag to remove hardcodings and let the agent steer adaptively
    try:
        feature_flags = get_interview_feature_flags()
        intelligent_flow = bool(feature_flags.get("INTELLIGENT_FLOW", False))
    except Exception:
        intelligent_flow = False
    
    state_config = STATE_CONFIGURATIONS.get(current_state, {})
    state_focus = state_config.get("focus", "General interview questions")
    
    # Format state_focus with interview_topic if available
    if interview_topic:
        try:
            state_focus = state_focus.format(interview_topic=interview_topic)
        except (KeyError, ValueError):
            pass  # If formatting fails, use as-is
    
    # Get recent context for better question generation - use compressed history
    compressed_history = _compress_conversation_history(conversation_history, max_recent_turns=MAX_HISTORY_TURNS)
    recent_context = compressed_history[1:] if compressed_history and compressed_history[0].get('role') == 'system' else compressed_history
    previous_questions = get_previous_questions(conversation_history)
    
    # Determine question strategy based on response analysis (with safer defaults like V2)
    if response_analysis is None:
        response_analysis = {}
    
    # Check if the answer is invalid/nonsensical and requires re-answer
    if response_analysis.get("requires_reanswer") or response_analysis.get("is_invalid"):
        # Safely get session id for tracking consecutive invalid answers
        if session_id:
            try:
                _INVALID_ANSWER_COUNT[session_id] = _INVALID_ANSWER_COUNT.get(session_id, 0) + 1
            except Exception:
                # Best-effort tracking only
                pass
        
        # If too many consecutive invalid answers, offer a different path to avoid loops
        if session_id and _INVALID_ANSWER_COUNT.get(session_id, 0) >= 3:
            return (
                "I'm having some trouble understanding your responses. "
                "Would you like to skip this question and move on, or would you prefer to try answering it in a different way?"
            )
        
        # Get the last question asked to reference it in the re-answer prompt
        last_question = previous_questions[-1] if previous_questions else None
        if isinstance(last_question, dict):
            last_question = last_question.get("question_text", last_question.get("content", ""))
        elif not isinstance(last_question, str):
            last_question = ""
        
        # Try to extract the core question sentence
        if last_question:
            # Prefer the part ending with a question mark
            m = re.search(r'([^.!?]*\?)', last_question)
            if m:
                last_question = m.group(1).strip()
            elif "." in last_question:
                # Fallback: use last sentence after a period
                last_question = last_question.split(".")[-1].strip()
        
        if last_question and len(last_question) > 10:
            reanswer_prompt = (
                "I didn't quite understand your response. "
                f"Could you please provide a more detailed answer to: {last_question}"
            )
        else:
            reanswer_prompt = (
                "I didn't quite understand your response. "
                "Could you please provide a more detailed answer to the previous question?"
            )
        return reanswer_prompt
    
    # Last candidate answer (if any) - extract from multiple possible fields
    last_answer = (
        response_analysis.get("raw_text")
        or response_analysis.get("answer")
        or response_analysis.get("original_response")
        or ""
    )
    
    # Parsed analysis fields (with safe defaults)
    response_type = response_analysis.get("type", ResponseType.GENERAL)
    # Handle response_type being either string or enum
    response_type_str = response_type.value if hasattr(response_type, 'value') else str(response_type)
    # Also check response_type field as fallback
    if not response_type_str or response_type_str == "general":
        response_type_str = response_analysis.get("response_type") or response_analysis.get("type") or "general"
    
    sentiment = response_analysis.get("sentiment", "neutral")
    engagement_score = response_analysis.get("engagement_score", 0.5)
    
    # Safely convert engagement_score to float with bounds checking (like V2)
    try:
        engagement_score = float(engagement_score)
        engagement_score = max(0.0, min(1.0, engagement_score))
    except (ValueError, TypeError):
        engagement_score = 0.5
    
    # Derive role / candidate / JD context (improved extraction like V2)
    job_title = (
        (job_details.get("title")
         or job_details.get("role")
         or job_details.get("position")
         or "Software Engineer")
        if job_details else "Software Engineer"
    )
    
    company_name = (
        job_details.get("company_name", "Knowledge Artisans / JobsifyAI client")
        if job_details else "Knowledge Artisans / JobsifyAI client"
    )
    
    candidate_name = (
        candidate_info.get("name")
        or candidate_info.get("full_name")
        or "the candidate"
    )
    
    candidate_roles = candidate_info.get("roles") or candidate_info.get("current_role") or "N/A"
    candidate_skills = candidate_info.get("skills", "N/A")
    years_experience = candidate_info.get("years_experience", "N/A")
    candidate_location = candidate_info.get("location", "N/A")
    work_mode = candidate_info.get("work_mode", "N/A")
    
    # Build adaptive system prompt - topic-focused if interview_topic is provided
    if interview_topic:
        # Extract topic-relevant skills from structured_resume (improved like V2 - handles both dict and list formats)
        topic_skills = []
        topic = (interview_topic or "").strip()
        
        if topic:
            topic_lower = topic.lower()
            topic_words = [w.strip() for w in re.split(r"[,\-/()]", topic_lower) if w.strip()]
            
            # 3a. Try from structured_resume (handles both dict and list formats)
            if structured_resume and isinstance(structured_resume, dict):
                # Look in skills section (can be dict or list)
                skills_section = structured_resume.get("skills") or structured_resume.get("technical_skills") or {}
                
                # Handle dict format (like V2)
                if isinstance(skills_section, dict):
                    for skill_name, detail in skills_section.items():
                        skill_lower = str(skill_name).lower()
                        if (
                            topic_lower in skill_lower
                            or skill_lower in topic_lower
                            or any(word in skill_lower for word in topic_words)
                        ):
                            topic_skills.append(str(skill_name))
                
                # Handle list format (original logic, but improved)
                elif isinstance(skills_section, list):
                    for skill in skills_section:
                        if isinstance(skill, dict) and "SkillName" in skill:
                            skill_name = skill.get("SkillName", "").strip()
                            if skill_name:
                                skill_lower = skill_name.lower()
                                if (
                                    topic_lower in skill_lower
                                    or skill_lower in topic_lower
                                    or any(word in skill_lower for word in topic_words)
                                ):
                                    topic_skills.append(skill_name)
                        elif isinstance(skill, str):
                            skill_lower = skill.lower()
                            if (
                                topic_lower in skill_lower
                                or skill_lower in topic_lower
                                or any(word in skill_lower for word in topic_words)
                            ):
                                topic_skills.append(skill)
                
                # Also scan "experience" / "projects" text (like V2)
                if not topic_skills:
                    for section_key in ("experience", "projects", "summary"):
                        sec = structured_resume.get(section_key)
                        if isinstance(sec, list):
                            for item in sec:
                                text = str(item)
                                txt_lower = text.lower()
                                if topic_lower in txt_lower:
                                    topic_skills.append(topic)
                                    break
        
        # 3b. Fallback to raw candidate_info.skills string
        if not topic_skills and isinstance(candidate_skills, str):
            topic_lower = interview_topic.lower()
            for skill in candidate_skills.split(","):
                s = skill.strip()
                s_lower = s.lower()
                if s and (topic_lower in s_lower or s_lower in topic_lower):
                    topic_skills.append(s)
        
        skills_str = ", ".join(topic_skills[:5]) if topic_skills else (candidate_skills or "N/A")
        topic_lower = topic.lower() if topic else ""
        psychometric_keywords = ["psychometric", "psychological", "behavioral", "aptitude", "personality"]
        communication_keywords = ["communication", "communication test", "communication skills", "speaking", "presentation", "listening"]
        is_psychometric_topic = any(kw in topic_lower for kw in psychometric_keywords)
        is_communication_topic = any(kw in topic_lower for kw in communication_keywords)
        
        if is_psychometric_topic:
            # Psychometric mode: assess traits via scenarios, not meta questions about tests
            system_prompt = f"""You are a behavioral interviewer running a psychometric-style conversation to infer the candidate's traits. Do NOT ask about psychometric tests/tools; ask scenario-based or past-behavior questions that reveal traits.

STATE: {current_state.value} - {state_focus}

CANDIDATE CONTEXT:
- Skills/role context: {skills_str}
- Experience: {candidate_info.get('years_experience', 'N/A')} years

ANALYSIS: {response_type_str} | Engagement: {engagement_score:.2f} | Sentiment: {sentiment}
KEYWORDS: {response_analysis.get('keywords', [])}
PREVIOUS QUESTIONS: {json.dumps(previous_questions[-5:], indent=2) if previous_questions else "None"}

TRAIT AREAS TO SAMPLE:
- Decision making, critical thinking, integrity/ethics
- Stress management, adaptability, time management
- Collaboration, conflict handling, communication clarity
- Motivation/ownership, learning agility

GUIDELINES:
- Ask ONE concise question (≤2 sentences, <300 chars) that elicits behavior in context.
- Target 1–2 traits per question; vary traits across the interview.
- Use scenarios or past-behavior prompts (\"tell me about a time…\", \"how would you handle…\").
- Hard ban: Do NOT ask about definitions or test-taking (e.g., "what is a psychometric test/assessment?", "have you taken a psychometric test?", "what comes to mind when you hear psychometric?").
- Avoid meta questions about psychometric testing or definitions.
- CRITICAL: Build on the candidate's last answer - acknowledge what they said and move to a NEW, DIFFERENT question.
- CRITICAL: Do NOT rephrase the same question - if they answered about "what is X", move to "how do you use X" or "when would you use X", NOT "can you explain X" (which is the same).
- CRITICAL: Check PREVIOUS QUESTIONS above - ensure your question is DIFFERENT from all previous questions.
- CRITICAL: After a valid answer, you MUST progress to a NEW question about a DIFFERENT aspect or go deeper - do NOT ask variations of the same question.
- CRITICAL: If they answered "what does X involve", do NOT ask "what does a typical X involve" or "can you explain X" - these are the SAME question. Move to "how do you use X" or "when would you use X" instead."""
        
        elif is_communication_topic:
            # Communication mode: evaluate clarity, listening, empathy, conflict handling via scenarios
            system_prompt = f"""You are a communication-focused interviewer assessing how the candidate communicates in real situations. Do NOT ask about communication tests/tools; ask scenario-based or past-behavior questions that reveal communication skills.

STATE: {current_state.value} - {state_focus}

CANDIDATE CONTEXT:
- Skills/role context: {skills_str}
- Experience: {candidate_info.get('years_experience', 'N/A')} years

ANALYSIS: {response_type_str} | Engagement: {engagement_score:.2f} | Sentiment: {sentiment}
KEYWORDS: {response_analysis.get('keywords', [])}
PREVIOUS QUESTIONS: {json.dumps(previous_questions[-5:], indent=2) if previous_questions else "None"}

COMMUNICATION DIMENSIONS TO SAMPLE:
- Clarity and structure; adapting message to audience
- Active listening and summarizing; questioning to confirm understanding
- Empathy and tone management, especially under stress/conflict
- Handling misalignment/feedback; influencing and persuading

GUIDELINES:
- Ask ONE concise question (≤2 sentences, <300 chars) that elicits communication behavior.
- Use scenarios or past-behavior prompts tied to team/client/stakeholder situations.
- Hard ban: Do NOT ask about definitions or test-taking (no "what is a communication test?", "have you taken a communication test?", "what comes to mind when you hear communication test?").
- Avoid meta questions about "communication tests" or definitions of communication.
- CRITICAL: Vary dimensions across questions; build on the last answer by acknowledging it and moving to a NEW, DIFFERENT question.
- CRITICAL: Do NOT rephrase the same question - if they answered about "what is X", move to "how do you use X" or "when would you use X", NOT "can you explain X" (which is the same).
- CRITICAL: Check PREVIOUS QUESTIONS above - ensure your question is DIFFERENT from all previous questions.
- CRITICAL: After a valid answer, you MUST progress to a NEW question about a DIFFERENT aspect or go deeper - do NOT ask variations of the same question.
- CRITICAL: If they answered "what does X involve", do NOT ask "what does a typical X involve" or "can you explain X" - these are the SAME question. Move to "how do you use X" or "when would you use X" instead."""
        
        elif intelligent_flow:
            system_prompt = f"""You are an expert interviewer running an analysis-driven, topic-focused interview on {interview_topic}.

STATE: {current_state.value} - {state_focus}

CANDIDATE CONTEXT:
- Skills related to {interview_topic}: {skills_str}
- Experience: {candidate_info.get('years_experience', 'N/A')} years

ANALYSIS: {response_type_str} | Engagement: {engagement_score:.2f} | Sentiment: {sentiment}
KEYWORDS: {response_analysis.get('keywords', [])}
PREVIOUS QUESTIONS: {json.dumps(previous_questions[-5:], indent=2) if previous_questions else "None"}

GUIDELINES:
- Ask ONE question that naturally advances the interview.
- Keep it aligned with {interview_topic} and the current stage ({current_state.value}).
- CRITICAL: Build on the candidate's last answer by acknowledging it and moving to a NEW, DIFFERENT question - do NOT rephrase the same question.
- CRITICAL: Avoid repetition of prior questions - check PREVIOUS QUESTIONS above and ensure your question is DIFFERENT.
- CRITICAL: If they answered about "what is X", move to "how do you use X" or "when would you use X" - NOT "can you explain X" (which is the same).
- CRITICAL: After a valid answer, you MUST progress to a NEW question about a DIFFERENT aspect or go deeper - do NOT ask variations of the same question.
- CRITICAL: If they answered "what does X involve", do NOT ask "what does a typical X involve" or "can you explain X" - these are the SAME question. Move to "how do you use X" or "when would you use X" instead.
- Avoid generic behaviorals unless highly relevant.
- IMPORTANT: Keep questions concise and focused (max 2-3 sentences, under 300 characters)."""
        else:
            system_prompt = f"""You are an expert interviewer conducting a FOCUSED TECHNICAL INTERVIEW EXCLUSIVELY on the topic of {interview_topic}.

THIS IS A TOPIC-FOCUSED INTERVIEW - ALL QUESTIONS MUST BE ABOUT {interview_topic.upper()}

STATE: {current_state.value} - {state_focus}

CANDIDATE CONTEXT:
- Skills related to {interview_topic}: {skills_str}
- Experience: {candidate_info.get('years_experience', 'N/A')} years

ANALYSIS: {response_type_str} | Engagement: {engagement_score:.2f} | Sentiment: {sentiment}
KEYWORDS: {response_analysis.get('keywords', [])}
PREVIOUS QUESTIONS: {json.dumps(previous_questions[-5:], indent=2) if previous_questions else "None"}

CRITICAL GUIDELINES - READ CAREFULLY:
1. MANDATORY: Your question MUST be directly about {interview_topic} - no exceptions
2. MANDATORY: Do NOT ask generic behavioral questions (e.g., "tell me about continuous learning", "describe your approach to skill development", "expand on a project")
3. MANDATORY: Do NOT ask questions about other technologies, frameworks, or topics - ONLY {interview_topic}
4. MANDATORY: If asking about projects, they MUST be specifically about {interview_topic} usage
5. MANDATORY: If asking about experience, it MUST be about {interview_topic} experience
6. Ensure your question aligns with the current interview stage: {current_state.value}
7. Keep questions conversational, clear, and concise (max 2-3 sentences, under 300 characters)
8. CRITICAL: Do NOT repeat or rephrase any of the previous questions - check PREVIOUS QUESTIONS above
9. CRITICAL: If the candidate has already answered a question, you MUST acknowledge their answer and move to a NEW, DIFFERENT question - do NOT ask the same question again or a slight variation
10. CRITICAL: Progress logically - if they answered about "what is X", move to "how do you use X" or "when would you use X" - do NOT ask "what is X" again or "can you explain X" (which is the same)

ANTI-REPETITION RULES:
- If previous question was "What is {interview_topic}?" or "What does {interview_topic} involve?", do NOT ask "What does a typical {interview_topic} involve?" or "Can you explain {interview_topic}?" - these are the SAME question
- If they answered about definitions/basics, move to application, examples, challenges, or advanced topics
- If they answered about one aspect, move to a DIFFERENT aspect or go deeper into a NEW area
- Always check PREVIOUS QUESTIONS list above before generating your question

EXAMPLES OF GOOD QUESTIONS (for {interview_topic}):
- "Can you explain how you handle state management in {interview_topic}?"
- "What challenges have you faced when working with {interview_topic}?"
- "How do you optimize performance in {interview_topic} applications?"
- "Can you walk me through a specific {interview_topic} project you worked on?"

EXAMPLES OF BAD QUESTIONS (DO NOT ASK THESE):
- "Tell me about your approach to continuous learning" ❌ (not about {interview_topic})
- "Can you expand on a recent project?" ❌ (too generic, not {interview_topic}-specific)
- "How do you handle team collaboration?" ❌ (behavioral, not about {interview_topic})
- "What other technologies are you interested in?" ❌ (not about {interview_topic})
- Repeating or rephrasing previous questions ❌ (causes loops)

Generate ONE question that:
1. Is focused EXCLUSIVELY on {interview_topic} - the question must mention or clearly relate to {interview_topic}
2. Acknowledges their last answer (if provided) and moves to a NEW, DIFFERENT question - NOT a rephrase
3. Fits the conversation flow and current stage
4. Is DIFFERENT from all previous questions - check the PREVIOUS QUESTIONS list
5. Progresses logically - if they answered basics, move to application; if they answered application, move to challenges or advanced topics
6. Maintains professional conversational tone
7. Is concise and focused (max 2-3 sentences, under 300 characters)

OUTPUT: Only the question - no greetings or extra text. The question MUST be about {interview_topic} and MUST be different from all previous questions."""
    else:
        # General KA / JobsifyAI interview mode (improved context like V2)
        system_prompt = f"""You are the **AI Interview Agent for JobsifyAI**, an AI-first recruitment platform by Knowledge Artisans.
You are interviewing a candidate on behalf of Harish (Head of India Operations / Founder), who prefers direct, practical, no-nonsense interviews.

ROLE CONTEXT:
- Target role: {job_title}
- Company / Client: {company_name}

CANDIDATE CONTEXT:
- Name: {candidate_name}
- Roles: {candidate_roles}
- Experience: {years_experience} years
- Primary skills: {candidate_skills}
- Location / work mode: {candidate_location} / {work_mode}

JOB DETAILS:
{json.dumps(job_details, indent=2) if job_details else "General IT/engineering role; mix of delivery, hands-on work, and client/stakeholder communication."}

INTERVIEW STATE:
- Phase: {current_state.value} — {state_focus}
- Question number: {question_count + 1}

LATEST ANSWER ANALYSIS:
- Response type: {response_type_str}
- Engagement score (0–1): {engagement_score:.2f}
- Sentiment: {sentiment}
- Extracted keywords: {response_analysis.get('keywords', [])}

RECENT QUESTIONS (avoid repetition):
{json.dumps(previous_questions[-3:], indent=2) if previous_questions else "None"}

GOALS:
- Assess:
  1) Technical depth in relevant stack/tools
  2) Problem solving & design/debugging ability
  3) Communication & clarity
  4) Culture & client readiness (ownership, maturity, working with US/global stakeholders)

STYLE:
- Tone: senior delivery manager who understands both tech and business.
- Be concise, specific, and grounded in real-world project scenarios.

QUESTION RULES:
1. Ask only ONE question in this turn.
2. Build naturally on the candidate's last answer where possible.
3. Avoid generic chit-chat and filler questions (e.g., "tell me about yourself", "strengths/weaknesses").
4. Do not repeat questions you already asked.
5. Max length: 1–2 sentences, under 300 characters.

OUTPUT:
Return ONLY the next question as plain text.
No greetings, no commentary, no bullet points."""

    try:
        response = await invoke_llm(
            prompt=system_prompt,
            agent_name="interview_agent_adaptive_question"
        )
        
        # Clean and validate the generated question
        if not isinstance(response, str):
            response = str(response or "")
        
        question = response.strip()
        
        # Handle multi-line responses (improvement from V2)
        if "\n" in question:
            for line in question.splitlines():
                line = line.strip()
                if line:
                    question = line
                    break
        
        # Remove any remaining patterns that shouldn't be in questions
        cleanup_patterns = [
            r'^(Hi|Hello|Hey|Welcome|Greetings)[,\s]+',
            r'^(Great|Nice|Good|Wonderful)\s+(to\s+)?(meet|hear|connect|talk)[^.!?]*[.!?]\s*',
            r'^Thanks?\s+(for\s+)?(sharing|that|your)[^.!?]*[.!?]\s*',
        ]
        
        for pattern in cleanup_patterns:
            question = re.sub(pattern, '', question, flags=re.IGNORECASE).strip()
        
        # Remove candidate name if present
        if names:
            for name in names:
                if name:
                    question = re.sub(re.escape(name), '', question, flags=re.IGNORECASE).strip()
        
        # Ensure question ends with proper punctuation
        # If it ends with '.', check if it's actually a question (contains question words or should be converted)
        if question.endswith('.'):
            # Check if it's actually a statement that should be converted to a question
            question_lower = question.lower()
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would', 'should', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'tell', 'describe', 'explain', 'walk']
            # If it doesn't contain question words, it's likely a statement - convert to question
            if not any(qw in question_lower for qw in question_words):
                # Convert statement to question by replacing period with question mark
                question = question.rstrip('.') + '?'
            else:
                # Contains question words but ends with period - likely should be a question
                question = question.rstrip('.') + '?'
        elif not question.endswith(('?', '!')):
            question += '?'
        
        # Fallback if question is too short or empty
        # Allow 2-word questions (e.g., "What is React?") but reject single-word or empty
        if not question or len(question.split()) < 2:
            question = get_fallback_question_for_state(current_state, candidate_info, question_count, interview_topic)
        
        # For topic-focused interviews, validate that the question mentions the topic
        # Skip strict enforcement when intelligent_flow is enabled
        if interview_topic and not intelligent_flow:
            topic_lower = interview_topic.lower()
            question_lower = question.lower()
            # Check if topic or significant words from topic are mentioned
            topic_words = [w for w in topic_lower.split() if len(w) > 3]  # Filter short words
            topic_mentioned = (
                topic_lower in question_lower or 
                any(word in question_lower for word in topic_words) or
                any(word in question_lower for word in [topic_lower.replace('.', ''), topic_lower.replace('.js', '')])
            )
            
            if not topic_mentioned:
                # Question doesn't mention the topic - inject it or use fallback
                log.warning(
                    "Generated question doesn't mention interview topic '%s'. Question: %s",
                    interview_topic,
                    question[:100]
                )
                # Try to inject the topic naturally, or use a topic-focused fallback
                if "project" in question_lower or "experience" in question_lower:
                    # Replace generic project/experience question with topic-specific one
                    question = f"Can you tell me about a specific {interview_topic} project you've worked on and the challenges you faced?"
                elif "learn" in question_lower or "skill" in question_lower or "development" in question_lower:
                    # Replace generic learning question with topic-specific one
                    question = f"How did you learn {interview_topic} and what resources did you find most helpful?"
                else:
                    # Use topic-focused default question
                    question = get_default_question(current_state, interview_topic)
                    log.info("Using topic-focused default question: %s", question[:100])
        
        return question
        
    except Exception as e:
        logging.exception("Error generating adaptive question: %s", e)
        return get_fallback_question_for_state(current_state, candidate_info, question_count, interview_topic)

async def generate_intelligent_question(
    candidate_info: Dict[str, str],
    persona: PersonaType,
    conversation_context: ConversationContext,
    conversation_history: List[Dict],
    job_details: Dict[str, Any],
    response_analysis: Dict[str, Any],
    names: List[str] = None,
    uid_context: Dict[str, Any] = None
) -> str:
    """Generate intelligent, context-aware questions using LLM understanding"""
    
    # Get recent conversation context
    # Use compressed history to prevent irrelevant questions
    compressed_history = _compress_conversation_history(conversation_history, max_recent_turns=4)
    recent_context = compressed_history[1:] if compressed_history and compressed_history[0].get('role') == 'system' else compressed_history
    previous_questions = get_previous_questions(conversation_history)
    
    # Extract intelligent analysis data
    context_understanding = response_analysis.get("context_understanding", "")
    follow_up_suggestions = response_analysis.get("follow_up_suggestions", [])
    strengths_mentioned = response_analysis.get("strengths_mentioned", [])
    areas_to_explore = response_analysis.get("areas_to_explore", [])
    
    # Use aggressive UID context for question generation
    uid_context = uid_context or {}
    candidate_name = get_candidate_name(uid_context, candidate_info, [])
    candidate_roles = get_job_title(uid_context, job_details, candidate_info)
    candidate_skills = get_candidate_skills(uid_context, candidate_info)
    candidate_experience = get_experience_years(uid_context, candidate_info)
    
    # Build intelligent system prompt
    system_prompt = f"""You are an expert interviewer conducting a natural conversation.
CANDIDATE: {candidate_name} | {candidate_roles} | {candidate_experience} years | {candidate_skills}
POSITION: {json.dumps(job_details, indent=2) if job_details else f"{candidate_roles} position"}
CONTEXT: Phase: {conversation_context.interview_phase} | Topics: {conversation_context.topics_discussed} | Engagement: {conversation_context.engagement_level:.2f} | Questions: {conversation_context.conversation_depth}
RECENT: {json.dumps(recent_context, indent=2) if recent_context else "Starting conversation"}
ANALYSIS: What they're saying: {context_understanding} | Follow-ups: {follow_up_suggestions} | Strengths: {strengths_mentioned} | Explore: {areas_to_explore}
GUIDELINES: Be natural, build on responses, dig deeper into interesting points, balance technical/behavioral, show genuine interest, avoid repetition.
Generate ONE natural question that:
1. Shows you understood their response
2. Builds on what they shared
3. Feels like natural conversation flow
4. Avoids repetition
5. Demonstrates genuine interest

OUTPUT: Only the question - no greetings or extra text."""

    try:
        response = await invoke_llm(
            prompt=system_prompt
        )
        
        # Clean and validate the generated question
        question = response.strip()
        
        # Remove any remaining patterns that shouldn't be in questions
        cleanup_patterns = [
            r'^(Hi|Hello|Hey|Welcome|Greetings)[,\s]+',
            r'^(Great|Nice|Good|Wonderful)\s+(to\s+)?(meet|hear|connect|talk)[^.!?]*[.!?]\s*',
            r'^Thanks?\s+(for\s+)?(sharing|that|your)[^.!?]*[.!?]\s*',
        ]
        
        for pattern in cleanup_patterns:
            question = re.sub(pattern, '', question, flags=re.IGNORECASE).strip()
        
        # Remove candidate name if present
        if names:
            for name in names:
                if name:
                    question = question.replace(name, "").replace(name.upper(), "").replace(name.lower(), "").strip()
        
        # Ensure question ends with proper punctuation
        if not question.endswith(('?', '!', '.')):
            question += "?"
        
        return question
        
    except Exception as e:
        log.error(f"Error generating intelligent question: {e}")
        return await get_natural_fallback_question(conversation_context, candidate_info, uid_context=uid_context)
async def get_natural_fallback_question(conversation_context: ConversationContext, candidate_info: Dict[str, str], session_id: str = None, uid_context: Dict[str, Any] = None, interview_topic: str = None) -> str:
    """Generate intelligent fallback questions with role-specific context and improved reliability.
    Now respects interview_topic when provided and checks for topic switches in conversation_context.
    """
    
    phase = conversation_context.interview_phase
    topics = conversation_context.topics_discussed
    engagement_level = conversation_context.engagement_level
    question_count = conversation_context.conversation_depth
    
    # Check if topic was switched (avoid referencing old topic)
    topic_switched = hasattr(conversation_context, 'topic_switched') and conversation_context.topic_switched
    
    # Use interview_topic if available and not switched
    active_topic = None if topic_switched else interview_topic
    
    # Config-driven safe defaults per state (use session_id for dedup tracking when available)
    try:
        state = _map_phase_to_state(phase)
        cfg = get_interview_config()
        qmap = cfg.get("safe_default_questions", {})
        state_key = state.value
        if state_key in qmap and qmap[state_key]:
            # Prefer a question not used recently in this session
            options = qmap[state_key]
            # Format with interview_topic if available
            if active_topic:
                options = [opt.format(interview_topic=active_topic) if "{interview_topic}" in opt else opt for opt in options]
            
            # Recover fallbacks from ChromaDB if not in memory
            if session_id and session_id not in _RECENT_FALLBACKS:
                try:
                    recovered_fallbacks = await _rbi(interview_chroma.get_recent_fallbacks, session_id)
                    if recovered_fallbacks:
                        _RECENT_FALLBACKS[session_id] = recovered_fallbacks
                except Exception:
                    pass
            
            if session_id and session_id in _RECENT_FALLBACKS:
                used = set([q.strip().lower() for q in _RECENT_FALLBACKS[session_id]])
                for opt in options:
                    if opt.strip().lower() not in used:
                        return opt
            return options[0]
    except Exception:
        pass
    
    # Determine role category for role-specific questions
    role_category = "TECHNICAL_CODING"  # Default
    if uid_context:
        try:
            # Use hybrid role classification (rules + LLM)
            role_result = await classify_role_hybrid(uid_context, {})
            role_category = role_result.role_category
        except Exception:
            pass
    
    # Extract candidate skills for context-aware questions
    candidate_skills = []
    if uid_context and "skills_parser" in uid_context:
        skills_data = uid_context["skills_parser"].get("skills", [])
        candidate_skills = [skill.get("SkillName", "") for skill in skills_data[:3]]
    
    # Extract and format candidate name properly (handle list format)
    candidate_name = get_candidate_name(uid_context or {}, candidate_info, [])
    
    # Role-specific fallback questions with better variety and context
    role_specific_questions = {
        "TECHNICAL_DESIGN": {
        "opening": [
                f"Hi {candidate_name}! What draws you to user experience design?",
                f"Welcome! I'd love to hear about your design background and what excites you about this UX role.",
                f"Hello! Can you tell me about your experience with design tools and what interests you about this position?",
                f"Great to meet you! What aspects of user-centered design are you most passionate about?",
                f"Welcome! I'm excited to learn about your design journey. What should I know about your background?"
        ],
        "exploration": [
                "That's interesting! Can you walk me through your design process for a recent project?",
                "I'd love to hear more about how you approach user research and validation.",
                "What design tools do you find most effective for your workflow?",
                "How do you typically collaborate with developers and stakeholders?",
                "Can you tell me about a design challenge you recently solved?",
                "What's your approach to creating user personas and journey maps?",
                "How do you handle feedback and iteration in your design process?",
                "What's the most innovative design solution you've implemented?"
        ],
        "deep_dive": [
                "That's a great example. How did you validate that design decision with users?",
                "What usability testing methods do you prefer and why?",
                "Can you elaborate on the accessibility considerations in that project?",
                "How did you measure the success of that design solution?",
                "What design systems or component libraries have you worked with?",
                "How do you approach responsive design and cross-platform consistency?",
                "What's your experience with prototyping and user testing tools?",
                "How do you balance user needs with business requirements?"
            ]
        },
        "TECHNICAL_CODING": {
            "opening": [
                f"Hi {candidate_name}! What programming languages are you most comfortable with?",
                f"Welcome! I'd love to hear about your development background and what excites you about this role.",
                f"Hello! Can you tell me about your coding experience and what interests you about this position?",
                f"Great to meet you! What aspects of software development are you most passionate about?",
                f"Welcome! I'm excited to learn about your technical journey. What should I know about your background?"
            ],
            "exploration": [
                "That's interesting! Can you walk me through a recent coding project you worked on?",
                "I'd love to hear more about your experience with different programming languages.",
                "What development frameworks and tools do you find most effective?",
                "How do you typically approach debugging and problem-solving?",
                "Can you tell me about a technical challenge you recently solved?",
                "What's your experience with version control and collaboration tools?",
                "How do you handle code reviews and maintain code quality?",
                "What's the most complex algorithm or system you've implemented?"
            ],
            "deep_dive": [
                "That's a great example. How did you optimize that solution for performance?",
                "What testing strategies do you use to ensure code reliability?",
                "Can you elaborate on the architecture decisions you made in that project?",
                "How did you handle scalability and deployment challenges?",
                "What design patterns do you find most useful in your work?",
                "How do you approach database design and optimization?",
                "What's your experience with cloud platforms and DevOps practices?",
                "How do you balance code maintainability with development speed?"
            ]
        },
        "BUSINESS_STRATEGIC": {
            "opening": [
                f"Hi {candidate_name}! What draws you to strategic business roles?",
                f"Welcome! I'd love to hear about your business background and what excites you about this position.",
                f"Hello! Can you tell me about your experience with strategy and what interests you about this role?",
                f"Great to meet you! What aspects of business development are you most passionate about?",
                f"Welcome! I'm excited to learn about your strategic journey. What should I know about your background?"
            ],
            "exploration": [
                "That's interesting! Can you walk me through a strategic initiative you led?",
                "I'd love to hear more about your experience with market analysis and planning.",
                "What business tools and methodologies do you find most effective?",
                "How do you typically approach stakeholder management and communication?",
                "Can you tell me about a business challenge you recently solved?",
                "What's your experience with data-driven decision making?",
                "How do you handle cross-functional collaboration and alignment?",
                "What's the most impactful business strategy you've implemented?"
            ],
            "deep_dive": [
                "That's a great example. How did you measure the success of that strategic initiative?",
                "What frameworks do you use for competitive analysis and positioning?",
                "Can you elaborate on the risk assessment and mitigation strategies you employed?",
                "How did you handle change management and organizational alignment?",
                "What's your approach to financial modeling and ROI analysis?",
                "How do you balance short-term results with long-term strategic goals?",
                "What's your experience with partnership development and negotiations?",
                "How do you approach innovation and market opportunity identification?"
            ]
        },
        "OPERATIONS_MANAGEMENT": {
            "opening": [
                f"Hi {candidate_name}! What draws you to operations and leadership roles?",
                f"Welcome! I'd love to hear about your leadership background and what excites you about this position.",
                f"Hello! Can you tell me about your experience with operations management and what interests you about this role?",
                f"Great to meet you! What aspects of team leadership and operational excellence are you most passionate about?",
                f"Welcome! I'm excited to learn about your leadership journey. What should I know about your background?"
            ],
            "exploration": [
                "That's interesting! Can you walk me through a major operational challenge you've led?",
                "I'd love to hear more about your approach to team management and development.",
                "What strategies do you use to drive operational efficiency and process improvement?",
                "How do you typically handle stakeholder management and cross-functional collaboration?",
                "Can you tell me about a time when you had to transform or scale operations?",
                "What's your experience with change management and organizational development?",
                "How do you balance strategic vision with day-to-day operational execution?",
                "What's the most significant operational improvement you've implemented?"
            ],
            "deep_dive": [
                "That's a great example. How did you measure the success of that operational initiative?",
                "What challenges did you face in implementing that change and how did you overcome them?",
                "Can you elaborate on your approach to building and developing high-performing teams?",
                "How do you handle conflict resolution and difficult conversations with team members?",
                "What's your experience with budget management and resource allocation?",
                "How do you ensure operational excellence while maintaining team morale and engagement?",
                "What metrics do you use to track operational performance and team productivity?",
                "How do you approach strategic planning and long-term operational roadmapping?"
            ]
        }
    }
    
    # Get role-specific questions or fall back to generic ones
    role_questions = role_specific_questions.get(role_category, role_specific_questions["TECHNICAL_CODING"])
    phase_questions = role_questions.get(phase, ["Can you tell me more about your background?"])
    
    # Add skill-specific questions if we have candidate skills
    if candidate_skills and phase in ["exploration", "deep_dive"]:
        skill_questions = []
        for skill in candidate_skills[:2]:  # Limit to top 2 skills
            if skill.lower() in ["figma", "adobe xd", "sketch"]:
                skill_questions.extend([
                    f"What's your experience with {skill} and how do you use it in your workflow?",
                    f"How do you leverage {skill} for prototyping and user testing?",
                    f"What advanced features of {skill} do you find most valuable?"
                ])
            elif skill.lower() in ["react", "angular", "vue", "javascript", "python", "java"]:
                skill_questions.extend([
                    f"What's your experience with {skill} and what projects have you built with it?",
                    f"How do you stay current with {skill} best practices and updates?",
                    f"What's the most complex {skill} application you've developed?"
                ])
            elif skill.lower() in ["ui/ux design", "user experience", "user interface"]:
                skill_questions.extend([
                    f"How do you approach user research and validation in your design process?",
                    f"What's your methodology for creating intuitive user interfaces?",
                    f"How do you balance user needs with business requirements in your designs?"
                ])
        
        phase_questions.extend(skill_questions[:3])  # Add up to 3 skill-specific questions
    
    # Enhanced engagement-based questions
    if engagement_level < 0.3:
        low_engagement_questions = [
            "I'd love to hear about a project you're particularly proud of. What made it special?",
            "Can you share an example of when you had to learn something new quickly?",
            "What's the most exciting challenge you've tackled recently?",
            "Tell me about a time when you had to think outside the box to solve a problem.",
            "What aspect of your work gets you most excited and motivated?"
        ]
        phase_questions.extend(low_engagement_questions)
    elif engagement_level > 0.7:
        high_engagement_questions = [
            "That's fascinating! Can you dive deeper into the technical details?",
            "I'm really interested in this. How did you approach the implementation?",
            "That sounds complex. What were the key challenges you overcame?",
            "Excellent! What would you do differently if you tackled this again?",
            "That's impressive! What did you learn from this experience?"
        ]
        phase_questions.extend(high_engagement_questions)
    
    # Topic-based follow-up questions with better context
    if topics and len(topics) > 0:
        recent_topic = topics[-1]
        topic_questions = [
            f"That's great insight about {recent_topic}. Can you elaborate on your experience with that?",
            f"I'm curious about your approach to {recent_topic}. What methodology did you use?",
            f"You mentioned {recent_topic}. What challenges did you face and how did you overcome them?",
            f"Regarding {recent_topic}, what would you say is your strongest area of expertise?",
            f"That's interesting about {recent_topic}. How has your experience with that evolved over time?"
        ]
        phase_questions.extend(topic_questions)
    
    # Question progression based on interview depth
    if question_count > 5 and phase in ["exploration", "deep_dive"]:
        progression_questions = [
            "Based on our conversation so far, what would you say are your key strengths?",
            "What areas are you looking to develop further in your career?",
            "How do you see yourself contributing to our team and projects?",
            "What kind of projects or challenges excite you most?",
            "What's your approach to continuous learning and skill development?"
        ]
        phase_questions.extend(progression_questions)
    
    # Avoid recent repetitions with improved tracking
    if session_id and session_id in _RECENT_FALLBACKS:
        recent_questions = _RECENT_FALLBACKS[session_id]
        # Filter out recently used questions
        available_questions = [q for q in phase_questions if q not in recent_questions]
        if available_questions:
            phase_questions = available_questions
        else:
            # If all questions were used recently, clear the history and start fresh
            _RECENT_FALLBACKS[session_id] = []
    
    # Select a question with weighted randomness (prefer role-specific questions)
    if len(phase_questions) > 0:
        # Weight role-specific questions higher
        weights = [2.0 if any(skill in q.lower() for skill in candidate_skills) else 1.0 for q in phase_questions]
        selected_question = random.choices(phase_questions, weights=weights)[0]
    else:
        selected_question = "Can you tell me more about your background and experience?"
    
    # Track this question for this session
    if session_id:
        if session_id not in _RECENT_FALLBACKS:
            _RECENT_FALLBACKS[session_id] = []
        _RECENT_FALLBACKS[session_id].append(selected_question)
        # Keep only last 5 questions to avoid memory buildup
        if len(_RECENT_FALLBACKS[session_id]) > 5:
            _RECENT_FALLBACKS[session_id] = _RECENT_FALLBACKS[session_id][-5:]
        
        # Persist to ChromaDB (offloaded to thread pool)
        await _rbi(_persist_recent_fallbacks, session_id, _RECENT_FALLBACKS[session_id])
    
    return selected_question

def get_default_question(state: InterviewState, interview_topic: str) -> str:
    """Get topic-focused default question for a given state."""
    default_questions = {
        InterviewState.TOPIC_INTRODUCTION: f"Tell me about your experience with {interview_topic}.",
        InterviewState.TOPIC_FUNDAMENTALS: f"What are the core concepts of {interview_topic} that you find most important?",
        InterviewState.TOPIC_PROBLEM_SOLVING: f"Can you walk me through how you would approach a problem using {interview_topic}?",
        InterviewState.TOPIC_DEEP_DIVE: f"What are some advanced features or concepts in {interview_topic} that you've worked with?",
        InterviewState.TOPIC_FEEDBACK: f"Based on our discussion about {interview_topic}, what would you like to explore further?",
        InterviewState.CANDIDATE_QUESTIONS: f"Do you have any questions about {interview_topic} or how it's used in this role?",
        InterviewState.CLOSING: "Thank you for your time. We'll be in touch with next steps.",
        InterviewState.COMPLETED: "Thank you for your time. We'll be in touch with next steps."
    }
    
    return default_questions.get(state, f"Tell me about your experience with {interview_topic}.")


def get_fallback_question_for_state(state: InterviewState, candidate_info: Dict[str, str], question_count: int = 0, interview_topic: str = None) -> str:
    """State-aware fallback questions that reference interview_topic when available"""
    
    # Build topic-aware fallback questions if interview_topic is provided
    if interview_topic:
        # Extract and format candidate name properly
        name = candidate_info.get('name', 'Candidate')
        if isinstance(name, list):
            name = name[0] if name else 'Candidate'
        name = str(name) if name else 'Candidate'
        
        fallback_questions = {
            InterviewState.OPENING_RAPPORT: [
                f"Hello {name}! What motivated you to apply for this {candidate_info.get('roles', 'role')} position focusing on {interview_topic}?",
                f"Welcome! Can you tell me about your experience with {interview_topic}?"
            ] if question_count == 0 else [
                f"What aspects of {interview_topic} excite you most?",
                f"How do you see yourself using {interview_topic} in this role?"
            ],
            
            InterviewState.STAGE_SETTING: [
                f"Let me explain how our interview will work today. We'll focus on your experience with {interview_topic} and how it relates to the role. Does that sound good?",
                f"I'd like to learn about your experience with {interview_topic} and how it aligns with this position. Are you ready to begin?"
            ],
            
            InterviewState.MAIN_QUESTIONS: [
                f"Can you describe a challenging project you've worked on involving {interview_topic}?",
                f"What was your most impactful contribution using {interview_topic} in your previous role?",
                f"How do you approach solving complex problems with {interview_topic}?",
                f"Tell me about a time you had to learn something new about {interview_topic} quickly."
            ],
            
            InterviewState.CANDIDATE_QUESTIONS: [
                f"What questions do you have about how we use {interview_topic} in this role?",
                f"Is there anything specific you'd like to know about our use of {interview_topic}?",
                f"What aspects of {interview_topic} would you like to learn more about in this position?"
            ],
            
            InterviewState.FEEDBACK: [
                f"Thank you for sharing your experience with {interview_topic}. What are your thoughts on the next steps?",
                f"I appreciate the time you've taken to discuss {interview_topic} with me today. Do you have any final questions?"
            ],
            
            InterviewState.CLOSING: [
                "Thank you for your time today. We'll be in touch soon about next steps.",
                "It was great speaking with you. We'll review your application and get back to you shortly."
            ]
        }
    else:
        # Extract and format candidate name properly
        name = candidate_info.get('name', 'Candidate')
        if isinstance(name, list):
            name = name[0] if name else 'Candidate'
        name = str(name) if name else 'Candidate'
        
        # Generic fallback questions when no topic is specified
        fallback_questions = {
        InterviewState.OPENING_RAPPORT: [
            f"Hello {name}! What motivated you to apply for this {candidate_info.get('roles', 'role')} position?",
            f"Welcome! Can you tell me about yourself and what interests you most about this opportunity?"
        ] if question_count == 0 else [
            "What aspects of this role excite you most?",
            "How do you see yourself contributing to our team?"
        ],
        
        InterviewState.STAGE_SETTING: [
            "Let me explain how our interview will work today. We'll cover your background, technical skills, and discuss the role. Does that sound good?",
            "I'd like to learn about your experience and how it aligns with this position. Are you ready to begin?"
        ],
        
        InterviewState.MAIN_QUESTIONS: [
            "Can you describe a challenging project you've worked on recently?",
            "What was your most impactful contribution in your previous role?",
            "How do you approach solving complex technical problems?",
            "Tell me about a time you had to learn something new quickly."
        ],
        
        InterviewState.CANDIDATE_QUESTIONS: [
            "What questions do you have about the role or our team?",
            "Is there anything specific you'd like to know about our company culture?",
            "What aspects of this position would you like to learn more about?"
        ],
        
        InterviewState.FEEDBACK: [
            "Thank you for sharing your experience with me. Based on our conversation, I can see you have strong skills in [area]. What are your thoughts on the next steps?",
            "I appreciate the time you've taken to speak with me today. Do you have any final questions about the role?"
        ],
        
        InterviewState.CLOSING: [
            "Thank you for your time today. We'll be in touch soon about next steps.",
            "It was great speaking with you. We'll review your application and get back to you shortly."
        ]
    }
    
    options = fallback_questions.get(state, [f"Can you tell me more about your experience with {interview_topic}?" if interview_topic else "Can you tell me more about your background?"])
    return random.choice(options)

# ==================== ROLE CLASSIFICATION ====================

def classify_role_from_chromadb_data(chromadb_data: Dict[str, Any]) -> str:
    """Classify candidate role from ChromaDB data with zero latency"""
    try:
        # Extract job titles from experience parser
        job_titles = []
        if "experience_parser" in chromadb_data:
            work_exp = chromadb_data["experience_parser"].get("work_experience", [])
            job_titles = [exp.get("job_title", "") for exp in work_exp if exp.get("job_title")]
        
        # Extract skills from skills parser
        skills = []
        if "skills_parser" in chromadb_data:
            skills_data = chromadb_data["skills_parser"].get("skills", [])
            skills = [skill.get("SkillName", "") for skill in skills_data if skill.get("SkillName")]
        
        # Extract interests from interest filler
        interests = []
        if "interest_filler" in chromadb_data:
            interests_data = chromadb_data["interest_filler"].get("user_interests", [])
            interests = interests_data if isinstance(interests_data, list) else []
        
        # Combine all text for analysis
        all_text = " ".join(job_titles + skills + interests).lower()
        
        # Classification logic with priority order
        # Technical Coding (highest priority for software roles)
        if any(term in all_text for term in ["engineer", "developer", "programmer", "software", "coding", "programming"]):
            return "TECHNICAL_CODING"
        
        # Technical Design (UI/UX, Frontend)
        elif any(term in all_text for term in ["ui", "ux", "designer", "frontend", "design", "user experience", "user interface"]):
            return "TECHNICAL_DESIGN"
        
        # Technical Data (Data Science, ML, AI)
        elif any(term in all_text for term in ["data scientist", "data analyst", "ml engineer", "ai engineer", "machine learning", "artificial intelligence", "analytics"]):
            return "TECHNICAL_DATA"
        
        # Technical Infrastructure (DevOps, Cloud, System Admin)
        elif any(term in all_text for term in ["devops", "cloud", "system administrator", "infrastructure", "deployment", "kubernetes", "docker"]):
            return "TECHNICAL_INFRASTRUCTURE"
        
        # Business Strategic (Product, Business Analysis)
        elif any(term in all_text for term in ["product manager", "business analyst", "project manager", "strategy", "business"]):
            return "BUSINESS_STRATEGIC"
        
        # Creative Marketing (Marketing, Brand, Content)
        elif any(term in all_text for term in ["marketing", "brand", "content", "creative", "advertising", "social media"]):
            return "CREATIVE_MARKETING"
        
        # Sales Business (Sales, Account Management)
        elif any(term in all_text for term in ["sales", "account manager", "business development", "client", "customer"]):
            return "SALES_BUSINESS"
        
        # Operations Management (HR, Operations, General Management)
        elif any(term in all_text for term in ["hr", "operations", "general manager", "administrative", "management"]):
            return "OPERATIONS_MANAGEMENT"
        
        # Default fallback for technical roles
        return "TECHNICAL_CODING"
        
    except Exception as e:
        log.error(f"Error in role classification: {e}")
        return "TECHNICAL_CODING"

# ==================== ENHANCED ROLE CLASSIFICATION ====================

@dataclass
class RoleClassificationResult:
    """Enhanced role classification result with confidence scoring"""
    role_category: str
    confidence_score: float
    reasoning: str
    matched_criteria: List[str]
    alternative_roles: List[Tuple[str, float]]
    skill_analysis: Dict[str, Any]
    experience_analysis: Dict[str, Any]


# ==================== LLM-DRIVEN ROLE CLASSIFICATION ====================

async def classify_role_llm_driven(chromadb_data: Dict[str, Any]) -> RoleClassificationResult:
    """
    LLM-driven role classification that analyzes structured resume to infer professional domain.
    
    This approach:
    - Handles ANY professional domain (technical, creative, business, etc.)
    - No hardcoded categories needed
    - Provides confidence + reasoning
    - Works for hybrid/emerging roles
    
    Args:
        chromadb_data: Structured resume data from ChromaDB
    
    Returns:
        RoleClassificationResult with domain classification
    """
    try:
        # Extract structured data from resume
        job_titles = []
        skills = []
        responsibilities = []
        interests = []
        education = []
        
        # Extract job titles
        if "experience_parser" in chromadb_data:
            work_exp = chromadb_data["experience_parser"].get("work_experience", [])
            for exp in work_exp:
                if exp.get("job_title"):
                    job_titles.append(exp.get("job_title"))
                # Extract responsibilities
                resp_list = exp.get("responsibilities", [])
                if isinstance(resp_list, list):
                    responsibilities.extend(resp_list[:3])  # Top 3 per job
        
        # Extract skills
        if "skills_parser" in chromadb_data:
            skills_data = chromadb_data["skills_parser"].get("skills", [])
            skills = [skill.get("SkillName", "") for skill in skills_data if skill.get("SkillName")][:15]  # Top 15
        
        # Extract interests
        if "interest_filler" in chromadb_data:
            interests_data = chromadb_data["interest_filler"].get("user_interests", [])
            if isinstance(interests_data, list):
                interests = interests_data[:5]  # Top 5
        
        # Extract education
        if "education_parser" in chromadb_data:
            edu_data = chromadb_data["education_parser"].get("education", [])
            for edu in edu_data:
                degree = edu.get("degree", "")
                field = edu.get("field_of_study", "")
                if degree or field:
                    education.append(f"{degree} in {field}".strip())
        
        # Build compact resume summary for LLM
        resume_summary = {
            "job_titles": job_titles[:5],  # Top 5 most recent
            "top_skills": skills,
            "key_responsibilities": responsibilities[:10],  # Top 10
            "interests": interests,
            "education": education[:3]  # Top 3
        }
        
        # Create LLM prompt for role classification using structured output (Issue 5.1)
        prompt = f"""Analyze this candidate's professional background and classify their primary professional domain.

RESUME SUMMARY:
{json.dumps(resume_summary, indent=2)}

Identify the candidate's PRIMARY professional domain (e.g., Software Engineering, Fashion Design, Data Science, Product Management).
Be specific (e.g., "Fashion Illustration" not just "Design"). Consider ALL evidence: titles, skills, responsibilities, interests, education.
Handle hybrid roles (e.g., "Technical Product Manager", "Creative Technologist")."""
        
        try:
            result_data: RoleClassificationLLMResult = await invoke_structured_llm(
                prompt,
                RoleClassificationLLMResult,
                task_type=TaskType.INTERVIEW,
                preferred_model=settings.GEMINI_MODEL,
                agent_name="interview_role_classification",
                temperature=0.2,
                timeout=_INTERVIEW_LLM_TIMEOUT,
                raise_on_fallback=False,
            )
            
            # Convert alternative roles to tuple format expected by RoleClassificationResult
            alternative_roles = [(alt.role, alt.confidence) for alt in result_data.alternative_roles]
            
            # Analyze skills and experience
            skill_analysis = {
                "total_skills": len(skills),
                "top_skills": skills[:5]
            }
            experience_analysis = {
                "total_positions": len(job_titles),
                "has_leadership": any("lead" in title.lower() or "manager" in title.lower() or "director" in title.lower() for title in job_titles),
                "has_senior_role": any("senior" in title.lower() or "principal" in title.lower() for title in job_titles)
            }
            
            log.debug(f"LLM-driven role classification: {result_data.role_category} (confidence: {result_data.confidence_score:.2f})")
            
            return RoleClassificationResult(
                role_category=result_data.role_category,
                confidence_score=result_data.confidence_score,
                reasoning=result_data.reasoning,
                matched_criteria=result_data.matched_criteria,
                alternative_roles=alternative_roles,
                skill_analysis=skill_analysis,
                experience_analysis=experience_analysis
            )
            
        except Exception as parse_error:
            log.warning(f"Failed to parse LLM classification response: {parse_error}. Using fallback.")
            return enhanced_role_classifier.classify_role_enhanced(chromadb_data)
    
    except Exception as e:
        log.error(f"Error in LLM-driven role classification: {e}")
        # Fallback to enhanced classifier
        return enhanced_role_classifier.classify_role_enhanced(chromadb_data)


# ==================== LLM-DRIVEN EVALUATION RUBRIC GENERATOR ====================

async def generate_evaluation_rubric_llm(
    role_category: str,
    interview_topic: str = None,
    job_description: str = None
) -> Dict[str, Any]:
    """
    LLM-driven evaluation rubric generator that creates custom criteria based on role + topic.
    
    This approach:
    - Generates domain-specific evaluation criteria
    - Adapts to ANY professional field (technical, creative, business, etc.)
    - Considers interview focus/topic
    - Provides weighted categories with descriptions
    
    Args:
        role_category: Professional domain from role classification (e.g., "Fashion Illustration")
        interview_topic: Specific interview focus (e.g., "Fashion Illustration", "Product Management")
        job_description: Optional job description for context
    
    Returns:
        Dict with evaluation rubric structure:
        {
            "categories": {
                "category_name": {"weight": 0.25, "max_score": 10, "description": "..."},
                ...
            },
            "keywords": {
                "category_name": ["keyword1", "keyword2", ...],
                ...
            }
        }
    """
    try:
        # Build context for rubric generation
        context = f"Role: {role_category}"
        if interview_topic:
            context += f"\nInterview Focus: {interview_topic}"
        if job_description:
            context += f"\nJob Description: {job_description[:200]}..."  # First 200 chars
        
        # Create LLM prompt for rubric generation using structured output (Issue 5.1)
        prompt = f"""Generate an evaluation rubric for assessing candidates in this professional domain.

CONTEXT:
{context}

Create 4-5 evaluation categories that are most relevant for assessing candidates in this domain.
Each category should have a descriptive name, weight (0.0-1.0, must sum to 1.0), description, and 10-15 keywords.

Examples by domain:
- Software Engineering: Technical Proficiency, Problem Solving, System Design, Code Quality, Communication
- Fashion Design: Creative Vision, Technical Skills, Industry Knowledge, Portfolio Quality, Communication
- Product Management: Strategic Thinking, Stakeholder Management, Data-Driven Decisions, Execution, Leadership

Tailor categories to the SPECIFIC domain. Weights should reflect importance."""
        
        try:
            rubric_data: EvaluationRubricResult = await invoke_structured_llm(
                prompt,
                EvaluationRubricResult,
                task_type=TaskType.INTERVIEW,
                preferred_model=settings.GEMINI_MODEL,
                agent_name="interview_evaluation_rubric",
                temperature=0.3,
                timeout=_INTERVIEW_LLM_TIMEOUT,
                raise_on_fallback=False,
            )
            
            # Convert Pydantic model to dict format expected by callers
            categories = {}
            for cat_name, cat_data in rubric_data.categories.items():
                categories[cat_name] = {
                    "weight": cat_data.weight,
                    "max_score": cat_data.max_score,
                    "description": cat_data.description
                }
            
            # Normalize weights to sum to 1.0
            total_weight = sum(cat["weight"] for cat in categories.values())
            if total_weight > 0:
                for cat_name in categories:
                    categories[cat_name]["weight"] = categories[cat_name]["weight"] / total_weight
            
            log.info(f"LLM-Generated Rubric for {role_category}: {len(categories)} categories")
            
            return {
                "categories": categories,
                "keywords": rubric_data.keywords
            }
            
        except Exception as parse_error:
            log.warning(f"Failed to parse LLM rubric response: {parse_error}. Using fallback.")
            raise
    
    except Exception as e:
        log.error(f"Error in LLM-driven rubric generation: {e}")
        # Fallback to generic rubric
        return {
            "categories": {
                "domain_expertise": {"weight": 0.30, "max_score": 10, "description": "Domain-specific knowledge and skills"},
                "problem_solving": {"weight": 0.25, "max_score": 10, "description": "Analytical and problem-solving abilities"},
                "communication": {"weight": 0.20, "max_score": 10, "description": "Clear communication and presentation skills"},
                "experience": {"weight": 0.15, "max_score": 10, "description": "Relevant experience and past work"},
                "professionalism": {"weight": 0.10, "max_score": 10, "description": "Professional demeanor and work ethic"}
            },
            "keywords": {
                "domain_expertise": ["experience", "skill", "knowledge", "expertise", "proficiency", "understanding"],
                "problem_solving": ["solve", "analyze", "approach", "challenge", "solution", "strategy", "optimize"],
                "communication": ["explain", "communicate", "present", "describe", "clarify", "articulate"],
                "experience": ["project", "worked", "developed", "built", "created", "implemented", "managed"],
                "professionalism": ["professional", "organized", "reliable", "dedicated", "committed", "thorough"]
            }
        }


class EnhancedRoleClassifier:
    """Enhanced role classification with multi-factor analysis and confidence scoring"""
    
    def __init__(self):
        # Define comprehensive role criteria with weights
        self.role_criteria = {
            "TECHNICAL_CODING": {
                "job_titles": {
                    "software engineer": 10, "software developer": 10, "programmer": 8,
                    "backend developer": 9, "frontend developer": 7, "full stack developer": 9,
                    "mobile developer": 8, "ios developer": 8, "android developer": 8,
                    "web developer": 7, "python developer": 9, "java developer": 9,
                    "javascript developer": 8, "react developer": 7, "angular developer": 7,
                    "node.js developer": 8, "c++ developer": 9, "c# developer": 8,
                    "senior developer": 9, "lead developer": 10, "tech lead": 10,
                    "architect": 10, "software architect": 10, "solution architect": 9,
                    "application developer": 8, "systems developer": 8, "game developer": 7,
                    "embedded developer": 8, "firmware developer": 8, "api developer": 7,
                    "microservices developer": 8, "cloud developer": 8, "devops engineer": 7,
                    "platform engineer": 8, "integration developer": 7, "automation engineer": 7,
                    "qa engineer": 6, "test engineer": 6, "quality engineer": 6,
                    "performance engineer": 7, "security engineer": 7, "reliability engineer": 7
                },
                "skills": {
                    "programming": 10, "coding": 10, "software development": 10,
                    "python": 9, "java": 9, "javascript": 8, "typescript": 8,
                    "react": 7, "angular": 7, "vue": 7, "node.js": 8,
                    "sql": 8, "database": 7, "api": 8, "rest": 7,
                    "git": 7, "version control": 7, "agile": 6, "scrum": 6,
                    "testing": 7, "unit testing": 7, "tdd": 7, "ci/cd": 7,
                    "docker": 7, "kubernetes": 7, "aws": 7, "azure": 7,
                    "algorithms": 8, "data structures": 8, "system design": 8
                },
                "responsibilities": {
                    "develop": 10, "code": 10, "program": 10, "implement": 9,
                    "build": 9, "create": 8, "design": 7, "debug": 8,
                    "test": 7, "deploy": 7, "maintain": 7, "optimize": 7,
                    "refactor": 7, "review": 6, "mentor": 6, "lead": 6
                }
            },
            "TECHNICAL_DESIGN": {
                "job_titles": {
                    "ui designer": 10, "ux designer": 10, "ui/ux designer": 10,
                    "product designer": 9, "interaction designer": 9, "visual designer": 8,
                    "frontend designer": 8, "web designer": 7, "mobile designer": 8,
                    "graphic designer": 6, "designer": 7, "creative designer": 6,
                    "user experience designer": 10, "user interface designer": 10,
                    "design lead": 9, "senior designer": 9, "principal designer": 10,
                    "design manager": 8, "creative director": 9, "art director": 8,
                    "brand designer": 7, "motion designer": 7, "game designer": 7,
                    "industrial designer": 8, "service designer": 8, "design researcher": 7,
                    "design strategist": 8, "design consultant": 7, "design architect": 8,
                    "accessibility designer": 7, "design systems designer": 8, "content designer": 6
                },
                "skills": {
                    "ui design": 10, "ux design": 10, "user experience": 10,
                    "user interface": 10, "figma": 9, "sketch": 8, "adobe xd": 8,
                    "photoshop": 6, "illustrator": 6, "invision": 7, "principle": 7,
                    "wireframing": 9, "prototyping": 9, "user research": 8,
                    "usability testing": 8, "a/b testing": 7, "design system": 8,
                    "responsive design": 8, "mobile design": 8, "web design": 7,
                    "typography": 7, "color theory": 6, "visual design": 7,
                    "interaction design": 9, "information architecture": 8,
                    "accessibility": 7, "wcag": 7, "design thinking": 8
                },
                "responsibilities": {
                    "design": 10, "create": 8, "prototype": 9, "wireframe": 9,
                    "research": 7, "test": 7, "iterate": 7, "collaborate": 6,
                    "present": 6, "document": 6, "maintain": 6, "evolve": 6
                }
            },
            "TECHNICAL_DATA": {
                "job_titles": {
                    "data scientist": 10, "data analyst": 9, "ml engineer": 10,
                    "ai engineer": 10, "machine learning engineer": 10,
                    "artificial intelligence engineer": 10, "data engineer": 9,
                    "analytics engineer": 8, "research scientist": 8,
                    "quantitative analyst": 8, "statistician": 7,
                    "business intelligence": 7, "data architect": 9,
                    "senior data scientist": 10, "lead data scientist": 10,
                    "data science manager": 9, "ai researcher": 9, "ml researcher": 9,
                    "computer vision engineer": 9, "nlp engineer": 9, "deep learning engineer": 9,
                    "data mining engineer": 8, "predictive analytics engineer": 8,
                    "business analyst": 7, "data visualization specialist": 7,
                    "research engineer": 8, "ai consultant": 8, "data consultant": 7,
                    "analytics consultant": 7, "machine learning consultant": 8,
                    "ai product manager": 8, "data product manager": 8
                },
                "skills": {
                    "machine learning": 10, "artificial intelligence": 10,
                    "data science": 10, "data analysis": 9, "statistics": 8,
                    "python": 9, "r": 8, "sql": 8, "pandas": 8, "numpy": 8,
                    "scikit-learn": 9, "tensorflow": 9, "pytorch": 9,
                    "keras": 8, "xgboost": 8, "matplotlib": 7,
                    "seaborn": 7, "jupyter": 7, "deep learning": 9,
                    "neural networks": 9, "nlp": 8, "computer vision": 8,
                    "big data": 8, "hadoop": 7, "spark": 8, "kafka": 7,
                    "aws": 7, "gcp": 7, "azure": 7, "docker": 7,
                    "kubernetes": 7, "mlops": 8, "model deployment": 8
                },
                "responsibilities": {
                    "analyze": 9, "model": 9, "predict": 8, "research": 8,
                    "develop": 7, "implement": 7, "optimize": 7, "evaluate": 7,
                    "deploy": 7, "monitor": 6, "maintain": 6, "improve": 6
                }
            },
            "TECHNICAL_INFRASTRUCTURE": {
                "job_titles": {
                    "devops engineer": 10, "site reliability engineer": 10,
                    "cloud engineer": 9, "infrastructure engineer": 9,
                    "platform engineer": 9, "system administrator": 8,
                    "linux administrator": 8, "network administrator": 7,
                    "security engineer": 8, "automation engineer": 8,
                    "release engineer": 7, "build engineer": 7,
                    "senior devops": 10, "devops lead": 10, "sre": 10,
                    "cloud architect": 9, "infrastructure architect": 9,
                    "systems engineer": 8, "network engineer": 7, "security analyst": 7,
                    "it operations manager": 8, "infrastructure manager": 8,
                    "cloud consultant": 8, "devops consultant": 8,
                    "site reliability manager": 9, "platform manager": 8,
                    "infrastructure specialist": 7, "cloud specialist": 7,
                    "systems architect": 9, "enterprise architect": 9,
                    "technical operations manager": 8, "it manager": 7
                },
                "skills": {
                    "devops": 10, "ci/cd": 9, "docker": 9, "kubernetes": 9,
                    "aws": 8, "azure": 8, "gcp": 8, "terraform": 8,
                    "ansible": 7, "jenkins": 7, "gitlab": 7, "github": 7,
                    "linux": 8, "bash": 7, "python": 7, "go": 7,
                    "monitoring": 8, "logging": 7, "prometheus": 7,
                    "grafana": 7, "elasticsearch": 7, "kibana": 7,
                    "nginx": 7, "apache": 6, "load balancing": 7,
                    "security": 7, "networking": 7, "ssl": 6, "tls": 6
                },
                "responsibilities": {
                    "deploy": 9, "maintain": 8, "monitor": 8, "automate": 8,
                    "scale": 7, "optimize": 7, "secure": 7, "backup": 6,
                    "recover": 6, "troubleshoot": 7, "support": 6, "manage": 6
                }
            },
            "BUSINESS_STRATEGIC": {
                "job_titles": {
                    "product manager": 10, "business analyst": 9,
                    "project manager": 8, "program manager": 8,
                    "strategy consultant": 9, "business consultant": 8,
                    "operations manager": 7, "general manager": 7,
                    "senior product manager": 10, "principal product manager": 10,
                    "product lead": 9, "product director": 10,
                    "product owner": 8, "scrum master": 7, "agile coach": 7,
                    "business development manager": 8, "strategy manager": 8,
                    "management consultant": 9, "strategy analyst": 8,
                    "product strategist": 9, "business strategist": 9,
                    "portfolio manager": 8, "program director": 9,
                    "chief product officer": 10, "vp product": 10,
                    "head of product": 10, "director of product": 10,
                    "senior business analyst": 9, "lead business analyst": 9,
                    "business intelligence analyst": 8, "data analyst": 7,
                    "market research analyst": 7, "competitive analyst": 7,
                    "product marketing manager": 8, "growth manager": 8
                },
                "skills": {
                    "product management": 10, "business analysis": 9,
                    "project management": 8, "strategy": 8, "analytics": 7,
                    "sql": 6, "excel": 6, "power bi": 6, "tableau": 6,
                    "agile": 7, "scrum": 7, "kanban": 6, "jira": 6,
                    "stakeholder management": 8, "communication": 7,
                    "presentation": 6, "negotiation": 6, "leadership": 7,
                    "market research": 7, "competitive analysis": 7,
                    "roadmap": 8, "prioritization": 8, "metrics": 7
                },
                "responsibilities": {
                    "manage": 8, "lead": 7, "coordinate": 7, "plan": 8,
                    "analyze": 7, "strategize": 7, "communicate": 6,
                    "present": 6, "negotiate": 6, "decide": 6, "prioritize": 7
                }
            },
            "CREATIVE_MARKETING": {
                "job_titles": {
                    "marketing manager": 9, "digital marketing": 8,
                    "content manager": 8, "brand manager": 8,
                    "social media manager": 7, "growth hacker": 8,
                    "marketing analyst": 7, "campaign manager": 7,
                    "creative director": 9, "art director": 8,
                    "senior marketing": 9, "marketing lead": 9,
                    "marketing director": 10, "head of marketing": 10,
                    "chief marketing officer": 10, "vp marketing": 10,
                    "content strategist": 8, "brand strategist": 8,
                    "marketing strategist": 8, "digital strategist": 8,
                    "seo specialist": 7, "sem specialist": 7, "ppc specialist": 7,
                    "email marketing specialist": 6, "social media specialist": 6,
                    "content creator": 6, "copywriter": 6, "content writer": 6,
                    "marketing coordinator": 6, "marketing assistant": 5,
                    "community manager": 7, "influencer marketing manager": 7,
                    "affiliate marketing manager": 7, "partnership manager": 7,
                    "event marketing manager": 7, "trade show manager": 6,
                    "public relations manager": 7, "pr specialist": 6,
                    "communications manager": 7, "communications specialist": 6
                },
                "skills": {
                    "marketing": 10, "digital marketing": 9, "content marketing": 8,
                    "social media": 7, "seo": 7, "sem": 7, "ppc": 7,
                    "email marketing": 6, "analytics": 7, "google analytics": 6,
                    "facebook ads": 6, "google ads": 6, "branding": 8,
                    "creative": 7, "copywriting": 7, "design": 6,
                    "campaign": 7, "strategy": 6, "growth": 7
                },
                "responsibilities": {
                    "market": 8, "promote": 7, "create": 7, "manage": 6,
                    "analyze": 6, "optimize": 6, "grow": 6, "engage": 6
                }
            },
            "SALES_BUSINESS": {
                "job_titles": {
                    "sales manager": 9, "account manager": 8,
                    "business development": 9, "sales representative": 7,
                    "account executive": 8, "sales director": 9,
                    "partnership manager": 8, "client manager": 7,
                    "senior sales": 9, "sales lead": 9,
                    "sales director": 9, "head of sales": 10,
                    "chief sales officer": 10, "vp sales": 10,
                    "regional sales manager": 8, "territory manager": 7,
                    "inside sales representative": 6, "outside sales representative": 7,
                    "sales engineer": 8, "technical sales": 8,
                    "business development manager": 8, "business development director": 9,
                    "partnership director": 9, "alliance manager": 8,
                    "channel manager": 7, "channel partner manager": 7,
                    "customer success manager": 7, "account director": 8,
                    "sales operations manager": 7, "sales enablement manager": 7,
                    "revenue operations manager": 7, "sales analyst": 6,
                    "sales coordinator": 5, "sales assistant": 4,
                    "lead generation specialist": 6, "sales development representative": 6,
                    "business development representative": 6, "sales consultant": 7
                },
                "skills": {
                    "sales": 10, "business development": 9, "account management": 8,
                    "negotiation": 8, "relationship building": 7,
                    "communication": 7, "presentation": 6, "crm": 6,
                    "lead generation": 7, "prospecting": 7, "closing": 7,
                    "partnership": 7, "networking": 6, "cold calling": 6
                },
                "responsibilities": {
                    "sell": 9, "develop": 7, "manage": 6, "build": 6,
                    "negotiate": 7, "close": 7, "grow": 6, "maintain": 6
                }
            },
            "OPERATIONS_MANAGEMENT": {
                "job_titles": {
                    "operations manager": 9, "hr manager": 8,
                    "general manager": 8, "administrative manager": 7,
                    "office manager": 6, "facilities manager": 6,
                    "senior manager": 9, "operations lead": 9,
                    "head of operations": 10, "head of india operations": 10,
                    "country head": 10, "regional head": 9,
                    "operations director": 10, "operations head": 10,
                    "founder": 9, "ceo": 10, "cto": 9, "coo": 10,
                    "chief operating officer": 10, "vp operations": 10,
                    "director of operations": 10, "head of hr": 9,
                    "hr director": 9, "chief human resources officer": 10,
                    "people operations manager": 8, "talent manager": 8,
                    "recruitment manager": 7, "hr business partner": 8,
                    "employee relations manager": 7, "compensation manager": 7,
                    "benefits manager": 7, "training manager": 7,
                    "organizational development manager": 8, "culture manager": 7,
                    "workplace manager": 6, "facilities coordinator": 5,
                    "administrative coordinator": 5, "executive assistant": 5,
                    "office coordinator": 5, "receptionist": 4,
                    "administrative assistant": 4, "office administrator": 5,
                    "business operations manager": 8, "process improvement manager": 8,
                    "quality manager": 8, "compliance manager": 7,
                    "risk manager": 7, "vendor manager": 7,
                    "supply chain manager": 8, "logistics manager": 7,
                    "procurement manager": 7, "contract manager": 7
                },
                "skills": {
                    "operations": 9, "management": 8, "hr": 7, "administration": 7,
                    "process improvement": 7, "team management": 7,
                    "budget": 6, "planning": 6, "coordination": 6,
                    "communication": 6, "leadership": 7, "training": 6,
                    "ai strategy": 8, "digital transformation": 7, "strategic planning": 8,
                    "governance": 7, "stakeholder management": 7, "client engagement": 6
                },
                "responsibilities": {
                    "manage": 8, "coordinate": 7, "organize": 6, "plan": 6,
                    "supervise": 6, "support": 6, "improve": 6, "maintain": 6,
                    "lead": 9, "oversee": 8, "stabilize": 7, "transform": 7,
                    "drive": 8, "develop": 7, "engage": 6, "govern": 7
                }
            }
        }
    
    def classify_role_enhanced(self, chromadb_data: Dict[str, Any]) -> RoleClassificationResult:
        """Enhanced role classification with multi-factor analysis and confidence scoring"""
        
        try:
            # Extract data from ChromaDB
            job_titles = self._extract_job_titles(chromadb_data)
            skills = self._extract_skills(chromadb_data)
            responsibilities = self._extract_responsibilities(chromadb_data)
            interests = self._extract_interests(chromadb_data)
            
            # Calculate scores for each role
            role_scores = {}
            matched_criteria = {}
            
            for role, criteria in self.role_criteria.items():
                score = 0
                criteria_matches = []
                
                # Job title matching (highest weight)
                title_score = self._calculate_job_title_score(job_titles, criteria["job_titles"])
                score += title_score * 0.4
                if title_score > 0:
                    criteria_matches.append(f"Job titles: {title_score:.1f}")
                
                # Skills matching (high weight)
                skill_score = self._calculate_skills_score(skills, criteria["skills"])
                score += skill_score * 0.35
                if skill_score > 0:
                    criteria_matches.append(f"Skills: {skill_score:.1f}")
                
                # Responsibilities matching (medium weight)
                resp_score = self._calculate_responsibilities_score(responsibilities, criteria["responsibilities"])
                score += resp_score * 0.15
                if resp_score > 0:
                    criteria_matches.append(f"Responsibilities: {resp_score:.1f}")
                
                # Interests matching (low weight)
                interest_score = self._calculate_interests_score(interests, criteria.get("skills", {}))
                score += interest_score * 0.1
                if interest_score > 0:
                    criteria_matches.append(f"Interests: {interest_score:.1f}")
                
                role_scores[role] = score
                matched_criteria[role] = criteria_matches
            
            # Find best role and calculate confidence
            best_role = max(role_scores, key=role_scores.get)
            best_score = role_scores[best_role]
            
            # Calculate confidence (0-1 scale)
            total_possible = 10.0  # Max score per category
            confidence = min(best_score / total_possible, 1.0)
            
            # Get alternative roles (top 3)
            sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
            alternative_roles = [(role, score) for role, score in sorted_roles[1:4] if score > 0]
            
            # Generate reasoning
            reasoning = self._generate_reasoning(best_role, best_score, matched_criteria[best_role])
            
            # Analyze skills and experience
            skill_analysis = self._analyze_skills(skills)
            experience_analysis = self._analyze_experience(job_titles, responsibilities)
            
            return RoleClassificationResult(
                role_category=best_role,
                confidence_score=confidence,
                reasoning=reasoning,
                matched_criteria=matched_criteria[best_role],
                alternative_roles=alternative_roles,
                skill_analysis=skill_analysis,
                experience_analysis=experience_analysis
            )
            
        except Exception as e:
            log.error(f"Error in enhanced role classification: {e}")
            return RoleClassificationResult(
                role_category="TECHNICAL_CODING",
                confidence_score=0.5,
                reasoning=f"Fallback due to error: {str(e)}",
                matched_criteria=[],
                alternative_roles=[],
                skill_analysis={},
                experience_analysis={}
            )
    
    def _extract_job_titles(self, chromadb_data: Dict[str, Any]) -> List[str]:
        """Extract job titles from experience parser"""
        job_titles = []
        if "experience_parser" in chromadb_data:
            work_exp = chromadb_data["experience_parser"].get("work_experience", [])
            job_titles = [exp.get("job_title", "").lower() for exp in work_exp if exp.get("job_title")]
        return job_titles
    
    def _extract_skills(self, chromadb_data: Dict[str, Any]) -> List[str]:
        """Extract skills from skills parser"""
        skills = []
        if "skills_parser" in chromadb_data:
            skills_data = chromadb_data["skills_parser"].get("skills", [])
            skills = [skill.get("SkillName", "").lower() for skill in skills_data if skill.get("SkillName")]
        return skills
    
    def _extract_responsibilities(self, chromadb_data: Dict[str, Any]) -> List[str]:
        """Extract responsibilities from experience parser"""
        responsibilities = []
        if "experience_parser" in chromadb_data:
            work_exp = chromadb_data["experience_parser"].get("work_experience", [])
            for exp in work_exp:
                resp_list = exp.get("responsibilities", [])
                if isinstance(resp_list, list):
                    responsibilities.extend([resp.lower() for resp in resp_list])
        return responsibilities
    
    def _extract_interests(self, chromadb_data: Dict[str, Any]) -> List[str]:
        """Extract interests from interest filler"""
        interests = []
        if "interest_filler" in chromadb_data:
            interests_data = chromadb_data["interest_filler"].get("user_interests", [])
            if isinstance(interests_data, list):
                interests = [interest.lower() for interest in interests_data]
        return interests
    
    def _calculate_job_title_score(self, job_titles: List[str], title_criteria: Dict[str, int]) -> float:
        """Calculate job title matching score"""
        if not job_titles:
            return 0.0
        
        total_score = 0
        for title in job_titles:
            for criterion, weight in title_criteria.items():
                if criterion in title:
                    total_score += weight
                    break  # Only match first criterion per title
        
        return min(total_score / len(job_titles), 10.0)  # Normalize to max 10
    
    def _calculate_skills_score(self, skills: List[str], skill_criteria: Dict[str, int]) -> float:
        """Calculate skills matching score"""
        if not skills:
            return 0.0
        
        total_score = 0
        matched_skills = 0
        for skill in skills:
            for criterion, weight in skill_criteria.items():
                if criterion in skill:
                    total_score += weight
                    matched_skills += 1
                    break  # Only match first criterion per skill
        
        if matched_skills == 0:
            return 0.0
        
        return min(total_score / matched_skills, 10.0)  # Normalize to max 10
    
    def _calculate_responsibilities_score(self, responsibilities: List[str], resp_criteria: Dict[str, int]) -> float:
        """Calculate responsibilities matching score"""
        if not responsibilities:
            return 0.0
        
        total_score = 0
        matched_resp = 0
        for resp in responsibilities:
            for criterion, weight in resp_criteria.items():
                if criterion in resp:
                    total_score += weight
                    matched_resp += 1
                    break  # Only match first criterion per responsibility
        
        if matched_resp == 0:
            return 0.0
        
        return min(total_score / matched_resp, 10.0)  # Normalize to max 10
    
    def _calculate_interests_score(self, interests: List[str], skill_criteria: Dict[str, int]) -> float:
        """Calculate interests matching score"""
        if not interests:
            return 0.0
        
        total_score = 0
        matched_interests = 0
        for interest in interests:
            for criterion, weight in skill_criteria.items():
                if criterion in interest:
                    total_score += weight
                    matched_interests += 1
                    break  # Only match first criterion per interest
        
        if matched_interests == 0:
            return 0.0
        
        return min(total_score / matched_interests, 10.0)  # Normalize to max 10
    
    def _generate_reasoning(self, role: str, score: float, criteria_matches: List[str]) -> str:
        """Generate human-readable reasoning for role classification"""
        if score >= 8.0:
            confidence_level = "high"
        elif score >= 6.0:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        criteria_text = ", ".join(criteria_matches) if criteria_matches else "limited evidence"
        
        return f"Classified as {role} with {confidence_level} confidence (score: {score:.1f}/10) based on: {criteria_text}"
    
    def _analyze_skills(self, skills: List[str]) -> Dict[str, Any]:
        """Analyze skills for additional insights"""
        if not skills:
            return {"total_skills": 0, "skill_categories": {}}
        
        # Categorize skills
        categories = {
            "programming": [],
            "tools": [],
            "frameworks": [],
            "databases": [],
            "cloud": [],
            "other": []
        }
        
        for skill in skills:
            skill_lower = skill.lower()
            if any(term in skill_lower for term in ["python", "java", "javascript", "typescript", "c++", "c#", "go", "rust"]):
                categories["programming"].append(skill)
            elif any(term in skill_lower for term in ["figma", "sketch", "photoshop", "git", "docker", "kubernetes"]):
                categories["tools"].append(skill)
            elif any(term in skill_lower for term in ["react", "angular", "vue", "node", "django", "flask", "spring"]):
                categories["frameworks"].append(skill)
            elif any(term in skill_lower for term in ["sql", "mysql", "postgresql", "mongodb", "redis"]):
                categories["databases"].append(skill)
            elif any(term in skill_lower for term in ["aws", "azure", "gcp", "cloud"]):
                categories["cloud"].append(skill)
            else:
                categories["other"].append(skill)
        
        return {
            "total_skills": len(skills),
            "skill_categories": {k: v for k, v in categories.items() if v},
            "top_skills": skills[:5]  # Top 5 skills
        }
    
    def _analyze_experience(self, job_titles: List[str], responsibilities: List[str]) -> Dict[str, Any]:
        """Analyze experience for additional insights"""
        return {
            "total_positions": len(job_titles),
            "has_leadership": any("lead" in title or "manager" in title or "director" in title for title in job_titles),
            "has_senior_role": any("senior" in title or "principal" in title or "architect" in title for title in job_titles),
            "responsibility_count": len(responsibilities),
            "has_management_responsibilities": any("manage" in resp or "lead" in resp or "supervise" in resp for resp in responsibilities)
        }

# Global instance
enhanced_role_classifier = EnhancedRoleClassifier()

def classify_role_enhanced(chromadb_data: Dict[str, Any]) -> RoleClassificationResult:
    """Enhanced role classification with confidence scoring and multi-factor analysis"""
    return enhanced_role_classifier.classify_role_enhanced(chromadb_data)

def get_role_specific_phases(role_category: str, question_count: int) -> str:
    """Get role-specific interview phases based on classification"""
    
    if role_category == "TECHNICAL_CODING":
        phases = [
            "TECHNICAL_WARMUP: Coding background and recent projects",
            "FRAMEWORK_DEEP_DIVE: Technologies, frameworks, and development practices", 
            "ARCHITECTURE_THINKING: System design, performance optimization, and scalability",
            "TECHNICAL_LEADERSHIP: Code review, mentoring, and technical decision-making"
        ]
    
    elif role_category == "TECHNICAL_DESIGN":
        phases = [
            "DESIGN_PHILOSOPHY: Design approach and user-centered thinking",
            "PORTFOLIO_DEEP_DIVE: Specific projects, tools, and design processes",
            "USER_RESEARCH: Research methods, usability testing, and user insights",
            "DESIGN_SYSTEMS: Component libraries, design consistency, and collaboration"
        ]
    
    elif role_category == "TECHNICAL_DATA":
        phases = [
            "DATA_FOUNDATION: Data analysis background and statistical knowledge",
            "MACHINE_LEARNING: ML models, algorithms, and data science projects",
            "DATA_ENGINEERING: Data pipelines, ETL processes, and infrastructure",
            "BUSINESS_INSIGHTS: Data-driven decision making and business impact"
        ]
    
    elif role_category == "TECHNICAL_INFRASTRUCTURE":
        phases = [
            "INFRASTRUCTURE_BASICS: System administration and cloud platforms",
            "DEVOPS_PRACTICES: CI/CD, automation, and deployment strategies",
            "SCALABILITY_RELIABILITY: System design, monitoring, and performance",
            "SECURITY_COMPLIANCE: Security practices, compliance, and risk management"
        ]
    
    elif role_category == "BUSINESS_STRATEGIC":
        phases = [
            "STRATEGIC_THINKING: Product vision, market analysis, and business strategy",
            "STAKEHOLDER_MANAGEMENT: Cross-functional collaboration and communication",
            "DECISION_MAKING: Data-driven decisions, trade-offs, and prioritization",
            "LEADERSHIP: Team management, strategic planning, and execution"
        ]
    
    elif role_category == "CREATIVE_MARKETING":
        phases = [
            "MARKETING_STRATEGY: Campaign planning, brand positioning, and market research",
            "CREATIVE_EXECUTION: Content creation, campaign management, and creative processes",
            "ANALYTICS_ROI: Performance measurement, optimization, and business impact",
            "STAKEHOLDER_COLLABORATION: Cross-team coordination and client management"
        ]
    
    elif role_category == "SALES_BUSINESS":
        phases = [
            "SALES_FUNDAMENTALS: Sales process, relationship building, and client management",
            "MARKET_KNOWLEDGE: Industry understanding, competitive positioning, and market trends",
            "NEGOTIATION_CLOSING: Deal negotiation, objection handling, and closing techniques",
            "BUSINESS_DEVELOPMENT: Growth strategies, partnerships, and revenue generation"
        ]
    
    elif role_category == "OPERATIONS_MANAGEMENT":
        phases = [
            "OPERATIONAL_EXCELLENCE: Process optimization, efficiency, and quality management",
            "TEAM_MANAGEMENT: Leadership, HR practices, and organizational development",
            "STRATEGIC_PLANNING: Business planning, resource allocation, and goal setting",
            "STAKEHOLDER_COMMUNICATION: Internal communication, reporting, and relationship management"
        ]
    
    else:
        # Default technical phases
        phases = [
            "TECHNICAL_WARMUP: Background and recent projects",
            "DEEP_EXPLORATION: Specific experiences and technical details",
            "PROBLEM_SOLVING: Challenges, solutions, and decision-making",
            "LEADERSHIP: Collaboration, mentoring, and future goals"
        ]
    
    # Return appropriate phase based on question count
    phase_index = min(question_count // 3, len(phases) - 1)
    return phases[phase_index]

# ==================== SYSTEM MESSAGE BUILDER ====================

def build_contextual_system_message(
    uid_context: Dict[str, Any],
    candidate_name: str,
    job_title: str,
    skills_str: str,
    interests: List[str],
    topics_discussed: List[str],
    engagement_level: float,
    question_count: int,
    role_guidelines: str = "",
    facts_summary: str = ""
) -> str:
    """Build a realistic interview system prompt that mimics natural conversation flow with role-specific phases."""
    interests_short = ", ".join(interests[:2]) if interests else ""
    topics_short = ", ".join(topics_discussed[:3]) if topics_discussed else "none yet"
    engagement_fmt = f"{engagement_level:.2f}" if isinstance(engagement_level, (int, float)) else "0.50"
    
    # Classify role from ChromaDB data (zero latency)
    role_category = classify_role_from_chromadb_data(uid_context)
    
    # Get role-specific phase based on classification and question count
    phase_context = get_role_specific_phases(role_category, question_count)
    
    # Role-specific focus areas (condensed)
    role_focus = {
        "TECHNICAL_CODING": "coding practices, frameworks, system design, problem-solving",
        "TECHNICAL_DESIGN": "user-centered design, design tools, user research, collaboration",
        "BUSINESS_STRATEGIC": "business strategy, stakeholder management, decision-making, leadership",
        "CREATIVE_MARKETING": "campaign strategy, creative execution, analytics, stakeholder collaboration",
        "TECHNICAL_DATA": "data analysis, ML models, data pipelines, business insights",
        "TECHNICAL_INFRASTRUCTURE": "system admin, DevOps, scalability, security",
        "SALES_BUSINESS": "sales process, market knowledge, negotiation, business development",
        "OPERATIONS_MANAGEMENT": "process optimization, team management, strategic planning, communication"
    }.get(role_category, "technical skills, problem-solving, collaboration, outcomes")
    
    parts = [
        f"You are a skilled interviewer conducting a natural conversation with {candidate_name}.",
        f"Background: {skills_str or 'General technical background'} | Interests: {interests_short or 'Various technical areas'}",
        f"Topics discussed: {topics_short or 'Just getting started'} | Phase: {phase_context}",
        f"Role: {role_category} | Questions: {question_count} | Engagement: {engagement_fmt}"
    ]
    
    if facts_summary:
        parts.append(f"Key facts: {facts_summary}")
    
    parts.extend([
        "",
        "STYLE: Natural, conversational, show genuine interest, use follow-ups, balance technical/behavioral",
        f"FOCUS: {role_focus}",
        "",
        "GUIDELINES: One question at a time, build on responses, vary types, avoid repetition, use 'you/your'",
        "COVERAGE: Assess all domains, rotate topics, explore successes/challenges",
        "",
        "OUTPUT: Only the next question, written naturally."
    ])
    
    return "\n".join(parts)

# ==================== SINGLE LLM CALL ARCHITECTURE ====================

async def build_unified_question_prompt(
    candidate_info: Dict[str, str],
    conversation_history: List[Dict],
    job_details: Dict[str, Any],
    uid_context: Dict[str, Any],
    question_count: int,
    response_analysis: Optional[Dict[str, Any]] = None,
    interview_stage: str = "warm_up",  # NEW
    adaptive_follow_up: str = "continuation",  # NEW
    technical_depth: str = "basic"  # NEW
) -> str:
    """Build a single comprehensive prompt that handles analysis, context preparation, and question generation"""
    
    # Check if the answer is invalid/nonsensical and requires re-answer
    if response_analysis and (response_analysis.get("requires_reanswer") or response_analysis.get("is_invalid")):
        from agents.interview_agent.history_utils import get_previous_questions as _get_prev_qs
        previous_questions = _get_prev_qs(conversation_history)
        last_question = previous_questions[-1] if previous_questions else None
        if isinstance(last_question, dict):
            last_question = last_question.get("question_text", last_question.get("content", ""))
        elif not isinstance(last_question, str):
            last_question = ""
        
        if last_question:
            m = re.search(r'([^.!?]*\?)', last_question)
            if m:
                last_question = m.group(1).strip()
            elif "." in last_question:
                last_question = last_question.split(".")[-1].strip()
        
        if last_question and len(last_question) > 10:
            return (
                "I didn't quite understand your response. "
                f"Could you please provide a more detailed answer to: {last_question}"
            )
        return (
            "I didn't quite understand your response. "
            "Could you please provide a more detailed answer to the previous question?"
        )
    
    # Extract context
    candidate_name = candidate_info.get('name', 'Candidate')
    skills = candidate_info.get('skills', 'General technical background')
    
    # Feature flags
    feature_flags = get_interview_feature_flags()
    resume_only_mode = feature_flags.get("resume_only_questioning", False)

    # Initialize defaults for logging consistency
    role_result = None
    role_category = "N/A"
    confidence_score = 0.0
    reasoning = "Classification disabled or not used"
    suppress_when_low_conf = feature_flags.get("suppress_role_phrasing_on_low_conf", True)
    suppress_role_phrasing = True if resume_only_mode else False

    # If not in resume-only mode, optionally compute role classification (kept as a hint)
    if not resume_only_mode:
        role_result = await classify_role_hybrid(uid_context, job_details) if uid_context else RoleClassificationResult(
            role_category="TECHNICAL_CODING",
            confidence_score=0.5,
            reasoning="No UID context available - using default classification",
            matched_criteria=[],
            alternative_roles=[],
            skill_analysis={},
            experience_analysis={}
        )
        role_category = role_result.role_category
        confidence_score = role_result.confidence_score
        reasoning = role_result.reasoning
    suppress_role_phrasing = (suppress_when_low_conf and (confidence_score < LOW_CONF_THRESHOLD))
    
    # Initialize domain keywords before any logging usage
    domain_job_keywords = ""
    
    # Debug: Log role classification result
    log.debug(f"\n{'='*60}")
    log.debug(f"ENHANCED ROLE CLASSIFICATION RESULT")
    log.debug(f"{'='*60}")
    log.debug(f"Role Category: {role_category}")
    log.debug(f"Confidence Score: {confidence_score:.2f} (suppress_role_phrasing={suppress_role_phrasing}, resume_only={resume_only_mode})")
    log.debug(f"Reasoning: {reasoning}")
    if role_result is not None:
        log.debug(f"Matched Criteria: {role_result.matched_criteria}")
    # Log domain job context keywords for visibility
    if domain_job_keywords:
        log.debug(f"Domain Job Keywords: {domain_job_keywords[:200]}{'...' if len(domain_job_keywords)>200 else ''}")
    log.debug(f"{'='*60}\n")
    
    # Get recent conversation context (last 3 exchanges)
    # Use compressed history to prevent irrelevant questions
    compressed_history = _compress_conversation_history(conversation_history, max_recent_turns=4)
    recent_context = compressed_history[1:] if compressed_history and compressed_history[0].get('role') == 'system' else compressed_history
    
    # Build job/company context (Chroma-backed if enabled)
    job_id = (job_details or {}).get('id') if isinstance(job_details, dict) else None
    company_id = (job_details or {}).get('company_id') if isinstance(job_details, dict) else None
    job_ctx = get_job_context(job_id)
    company_ctx = get_company_context(company_id)
    ctx_summary = summarize_for_prompt(job_ctx, company_ctx)
    job_title = ctx_summary.get('job_summary') or (job_details.get('title') if job_details else DEFAULT_ROLE_TITLE)
    job_company = ctx_summary.get('company_summary') or (job_details.get('company') if job_details else DEFAULT_COMPANY)
    
    # Extract candidate skills for context
    candidate_skills = []
    if uid_context and "skills_parser" in uid_context:
        skills_data = uid_context["skills_parser"].get("skills", [])
        candidate_skills = [skill.get("SkillName", "") for skill in skills_data[:SKILL_LIST_MAX]]
    
    # Domain-agnostic context
    domain_ctx = get_domain_context(uid_context, job_details)
    domain_skill_keywords = domain_ctx.get("skill_keywords", [])
    domain_job_keywords = domain_ctx.get("job_keywords", "")
    
    # NEW: Config-driven instructions and guidance
    stage_instructions = get_interview_stage_instructions()
    follow_up_guidance = get_interview_follow_up_guidance()
    
    # Compact analyses for prompt hygiene
    skill_summary = domain_skill_keywords or (list(role_result.skill_analysis.keys()) if (role_result is not None and hasattr(role_result, "skill_analysis")) else [])
    exp_summary = list(role_result.experience_analysis.keys()) if (role_result is not None and hasattr(role_result, "experience_analysis")) else []

    # Safe alternative roles accessor
    alt_str = "None"
    if role_result is not None:
        alt = getattr(role_result, "alternative_roles", []) or []
        alt_str = ", ".join([f"{r} ({s:.1f})" for r, s in alt[:2]]) if alt else "None"

    # If resume-only mode: build and return a simplified, context-driven prompt
    if resume_only_mode:
        # Build a richer resume skill list (from resume_data if present)
        resume_skills: List[str] = []
        try:
            sr = (uid_context or {}).get("resume_data") or (uid_context or {}).get("structured_resume")
            if isinstance(sr, dict):
                for s in sr.get("skills", []) or []:
                    if isinstance(s, dict):
                        n = s.get("SkillName")
                        if isinstance(n, str) and n.strip():
                            resume_skills.append(n.strip())
        except Exception:
            pass
        merged_skills = []
        seen_sk = set()
        for n in (candidate_skills + resume_skills):
            if n and n not in seen_sk:
                merged_skills.append(n)
                seen_sk.add(n)
        # Use compressed history to prevent irrelevant questions in deep conversations
        compressed_history = _compress_conversation_history(conversation_history, max_recent_turns=MAX_HISTORY_TURNS)
        recent_context = compressed_history[1:] if compressed_history and compressed_history[0].get('role') == 'system' else compressed_history
        prompt = f"""You are conducting a {interview_stage} stage interview.

OBJECTIVE: {stage_instructions.get(interview_stage, "")}
ADAPTIVE APPROACH: {follow_up_guidance.get(adaptive_follow_up, "")}
TECHNICAL DEPTH: {technical_depth}

You are an expert interviewer conducting a natural conversation. Analyze the candidate's response and generate the next question in one step. Tailor to the candidate's resume/background and their last answer. Do not rely on any role classification.

CANDIDATE: {candidate_name} | Skills: {', '.join(merged_skills[:SKILL_LIST_MAX]) or skills} | Questions: {question_count}

RECENT CONVERSATION:
{json.dumps(recent_context, indent=2) if recent_context else "Starting conversation"}

TASK: Analyze the candidate's last response and generate the next question.

REQUIREMENTS:
1. Build directly on their last answer by acknowledging it and moving to a NEW, DIFFERENT question - do NOT rephrase the same question.
2. Rotate across experience, problem-solving, collaboration, constraints, outcomes over time.
3. CRITICAL: Avoid repeating earlier questions or generic openers - check conversation history and ensure your question is DIFFERENT.
4. CRITICAL: If they answered about "what is X", move to "how do you use X" or "when would you use X" - NOT "can you explain X" (which is the same).
5. Keep it natural, professional, and concise.
"""
        return prompt

    # Build unified prompt (force context personalization in every turn)
    # If suppression is enabled, hide role as a strong constraint and keep it as a weak hint
    role_hint_line = f"Role hint: {role_category} (Conf: {confidence_score:.2f})" if not suppress_role_phrasing else f"Role hint: n/a (Conf: {confidence_score:.2f})"
    prompt = f"""You are conducting a {interview_stage} stage interview.

STAGE OBJECTIVE: {stage_instructions.get(interview_stage, "")}
ADAPTIVE APPROACH: {follow_up_guidance.get(adaptive_follow_up, "")}
TECHNICAL DEPTH: {technical_depth}

You are an expert interviewer conducting a natural conversation. Analyze the candidate's response and generate the next question in one step. Always tailor to the job/company context and assess both technical, non-technical, and soft skills over the session.

CANDIDATE: {candidate_name} | {role_hint_line} | Skills: {', '.join(candidate_skills) if candidate_skills else skills} | Questions: {question_count}

POSITION: {job_title} at {job_company}
ROLE RESPONSIBILITIES: {ctx_summary.get('job_responsibilities', '')}
ROLE REQUIREMENTS: {ctx_summary.get('job_requirements', '')}
COMPANY PRODUCTS: {ctx_summary.get('company_products', '')}

JOB CONTEXT KEYWORDS: {domain_job_keywords}

ROLE ANALYSIS (hint): {reasoning}
ALTERNATIVE ROLES (hint): {alt_str}
SKILL AREAS: {', '.join(skill_summary[:8])}
EXPERIENCE AREAS: {', '.join(exp_summary[:8])}

RECENT CONVERSATION:
{json.dumps(recent_context, indent=2) if recent_context else "Starting conversation"}

TASK: Analyze the candidate's last response and generate the next question.

ANALYSIS REQUIREMENTS:
1. Determine response type (technical/behavioral/brief/general)
2. Calculate engagement score (0.0-1.0) based on detail and enthusiasm
3. Extract key topics and skills mentioned
4. Identify strengths and areas to explore
5. Generate 2-3 suggested follow-up questions that leverage job/company context

QUESTION GENERATION REQUIREMENTS:
1. Build naturally on their response - acknowledge what they said and move forward
2. Show genuine interest and active listening
3. Use job context keywords and candidate skills/experience to stay relevant
4. CRITICAL: Avoid repetition from previous questions - do NOT rephrase the same question
5. CRITICAL: If they answered a question, move to a NEW, DIFFERENT question - do NOT ask the same thing again with different words
6. CRITICAL: Progress logically - if they answered "what is X", move to "how do you use X" or "when would you use X" - NOT "can you explain X" (which is the same)
7. CRITICAL: After a valid answer, you MUST progress to a NEW question about a DIFFERENT aspect or go deeper - do NOT ask variations of the same question
8. CRITICAL: If they answered "what does X involve", do NOT ask "what does a typical X involve" or "can you explain X" - these are the SAME question. Move to "how do you use X" or "when would you use X" instead
9. Maintain conversational, professional tone
10. Ask ONE focused question only

IMPORTANT: The candidate's role is {role_category}. Generate questions that are specifically relevant to this role type, not generic technical questions.

OUTPUT FORMAT (JSON):
{{
    "question": "Your next question here",
    "response_analysis": {{
        "type": "technical|behavioral|brief|general",
        "engagement_score": 0.0-1.0,
        "topics_mentioned": ["topic1", "topic2"],
        "skills_demonstrated": ["skill1", "skill2"],
        "strengths": ["strength1", "strength2"],
        "areas_to_explore": ["area1", "area2"]
    }},
    "suggested_follow_ups": ["follow-up 1", "follow-up 2"],
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why this question was chosen",
    "topics_mentioned": ["topic1", "topic2"],
    "skills_demonstrated": ["skill1", "skill2"],
    "strengths": ["strength1", "strength2"],
    "areas_to_explore": ["area1", "area2"]
}}

Generate the analysis and question now:"""
    
    return prompt

async def generate_question_single_call(
    candidate_info: Dict[str, str],
    conversation_history: List[Dict],
    job_details: Dict[str, Any],
    uid_context: Dict[str, Any],
    question_count: int,
    response_analysis: Optional[Dict[str, Any]] = None,
    interview_stage: str = "warm_up",
    adaptive_follow_up: str = "continuation",
    technical_depth: str = "basic"
) -> SingleCallQuestionResult:
    """Generate question using single LLM call with structured output (Issue 5.1)"""
    
    # Build unified prompt
    prompt = await build_unified_question_prompt(
        candidate_info, conversation_history, job_details, uid_context, question_count, response_analysis,
        interview_stage, adaptive_follow_up, technical_depth
    )
    
    try:
        start_time = time.time()
        
        llm_result: SingleCallQuestionLLMResult = await invoke_structured_llm(
            prompt,
            SingleCallQuestionLLMResult,
            task_type=TaskType.INTERVIEW,
            preferred_model=settings.GEMINI_MODEL,
            agent_name="interview_single_call_question",
            temperature=0.4,
            timeout=_INTERVIEW_LLM_TIMEOUT,
            raise_on_fallback=False,
        )
        
        processing_time = time.time() - start_time
        log.debug(f"Single call question generation took {processing_time:.3f}s")
        
        # Convert to SingleCallQuestionResult (internal dataclass)
        result = SingleCallQuestionResult(
            question=llm_result.question,
            response_analysis={
                "type": llm_result.response_analysis.type,
                "engagement_score": llm_result.response_analysis.engagement_score,
                "topics_mentioned": llm_result.response_analysis.topics_mentioned,
                "skills_demonstrated": llm_result.response_analysis.skills_demonstrated,
                "strengths": llm_result.response_analysis.strengths,
                "areas_to_explore": llm_result.response_analysis.areas_to_explore
            },
            engagement_score=llm_result.engagement_score,
            suggested_follow_ups=llm_result.suggested_follow_ups,
            confidence=llm_result.confidence,
            reasoning=llm_result.reasoning,
            topics_mentioned=llm_result.topics_mentioned,
            skills_demonstrated=llm_result.skills_demonstrated,
            strengths=llm_result.strengths,
            areas_to_explore=llm_result.areas_to_explore
        )
        
        # Post-generation safety: redact PII and ensure policy compliance
        result.question = _redact_pii(result.question)
        if not _policy_compliant(result.question):
            result.question = _rewrite_to_safe(result.question)
        
        return result
        
    except Exception as e:
        log.error(f"Single call question generation failed: {e}")
        fallback_question = await get_natural_fallback_question(
            ConversationContext(), candidate_info, uid_context=uid_context
        )
        
        return SingleCallQuestionResult(
            question=fallback_question,
            response_analysis={"type": "general", "engagement_score": 0.5},
            engagement_score=0.5,
            suggested_follow_ups=["Can you tell me more about your background?"],
            confidence=0.7,
            reasoning="Fallback due to single call failure",
            topics_mentioned=[],
            skills_demonstrated=[],
            strengths=[],
            areas_to_explore=[]
        )

# ==================== MEMORY HELPERS (ROLLING + FACTS) ====================

def build_rolling_memory(conversation_history: List[Dict], max_turns: int = 3) -> List[Dict[str, str]]:
    """Return last N messages (user/assistant) as rolling memory."""
    if not conversation_history:
        return []
    # Keep non-system last 2*max_turns messages
    msgs = [m for m in conversation_history if m.get("role") in ("user", "assistant")]
    return msgs[-(max_turns * 2):]

def _compress_conversation_history(
    history: List[Dict], 
    max_recent_turns: int = 4,
    max_message_length: int = 300
) -> List[Dict]:
    """
    Compress conversation history for LLM prompts to prevent irrelevant questions.
    
    Strategy:
    - Keep first system message (if exists) for context
    - Keep last N turns (recent context) with full detail
    - Truncate long messages to prevent token bloat
    - Remove excessive system messages
    
    Args:
        history: Full conversation history
        max_recent_turns: Maximum number of recent turns to keep (default: 4)
        max_message_length: Maximum length per message before truncation
        
    Returns:
        Compressed conversation history suitable for LLM prompts
    """
    if not history:
        return []
    
    # If history is short enough, just truncate long messages
    if len(history) <= (max_recent_turns * 2 + 1):  # +1 for system message
        compressed = []
        for msg in history:
            compressed_msg = msg.copy()
            content = compressed_msg.get('content', '')
            if len(content) > max_message_length:
                compressed_msg['content'] = content[:max_message_length] + "...[truncated]"
            compressed.append(compressed_msg)
        return compressed
    
    # Strategy: Keep first system message + last N turns
    compressed = []
    
    # Keep first system message if it exists (usually contains important context)
    if history and history[0].get('role') == 'system':
        first_msg = history[0].copy()
        content = first_msg.get('content', '')
        if len(content) > max_message_length:
            first_msg['content'] = content[:max_message_length] + "...[truncated]"
        compressed.append(first_msg)
    
    # Get last N turns (user + assistant pairs)
    # Calculate how many messages to keep: max_recent_turns * 2 (user + assistant pairs)
    messages_to_keep = max_recent_turns * 2
    recent_messages = history[-messages_to_keep:]
    
    # Truncate long messages in recent context
    for msg in recent_messages:
        compressed_msg = msg.copy()
        content = compressed_msg.get('content', '')
        if len(content) > max_message_length:
            compressed_msg['content'] = content[:max_message_length] + "...[truncated]"
        compressed.append(compressed_msg)
    
    return compressed

def update_facts_memory(
    existing: Dict[str, Any],
    uid_context: Dict[str, Any],
    candidate_info: Dict[str, Any],
    response_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge skills/interests from UID and keywords from response into a compact facts memory."""
    facts = existing.copy() if isinstance(existing, dict) else {}
    # Seed skills
    skills = set()
    for source in [candidate_info.get("skills"),
                   ", ".join([s.get("SkillName", "") for s in (uid_context.get("skills_analysis", {}).get("skills", []) if uid_context and uid_context.get("skills_analysis") else [])])]:
        if isinstance(source, str) and source:
            for s in source.split(","):
                s = s.strip()
                if s:
                    skills.add(s)
    if skills:
        facts["skills"] = sorted(list(set(facts.get("skills", [])) | skills))[:20]
    # Interests
    interests = (uid_context.get("user_interests", {}).get("user_interests", []) if uid_context and uid_context.get("user_interests") else []) or []
    if interests:
        facts["interests"] = interests[:10]
    # Emerging topics from keywords
    kws = response_analysis.get("keywords", []) or []
    if kws:
        topics = set(facts.get("topics", [])) | set([str(k).strip() for k in kws if str(k).strip()])
        facts["topics"] = sorted(list(topics))[:25]
    return facts

def summarize_facts_memory(facts: Dict[str, Any]) -> str:
    if not facts:
        return ""
    parts = []
    if facts.get("skills"):
        parts.append("Skills: " + ", ".join(facts["skills"][:8]))
    if facts.get("topics"):
        parts.append("Topics: " + ", ".join(facts["topics"][:6]))
    if facts.get("interests"):
        parts.append("Interests: " + ", ".join([i[:40] for i in facts["interests"][:2]]))
    return "\n".join(parts)

# ==================== ROLE MICRO-GUIDELINES & GUARDRAILS ====================

def derive_role_micro_guidelines(job_title: str) -> str:
    title = (job_title or "").lower()
    if any(k in title for k in ["web", "designer", "ui", "ux", "frontend"]):
        return "- Prioritize usability, accessibility, information hierarchy\n- Ask for concrete UI decisions, trade-offs, and validation\n- Anchor questions in constraints (performance, responsiveness, a11y, brand)"
    if any(k in title for k in ["embedded", "firmware", "electronics", "iot"]):
        return "- Prioritize constraints, hardware integration, debugging methodology\n- Ask for timing, memory, power trade-offs\n- Probe tooling (oscilloscope, JTAG, RTOS) and test strategy"
    if any(k in title for k in ["data", "ml", "ai", "backend"]):
        return "- Prioritize data quality, reliability, scalability\n- Ask for modeling/architecture choices with metrics\n- Probe failure modes, monitoring, and rollout safety"
    return "- Ask for concrete examples, constraints, and measurable outcomes\n- Explore collaboration, decision rationale, and trade-offs"

async def enforce_question_guardrails(question: str, previous_questions: List[str], conversation_context: ConversationContext, candidate_info: Dict[str, str], uid_context: Dict[str, Any] = None, session_id: str = None) -> str:
    if not question:
        return await get_natural_fallback_question(conversation_context, candidate_info, uid_context=uid_context)
    q = question.strip()
    # Remove greeting prefixes
    q = re.sub(r"^(hello|hi|hey|greetings)[!,\.]?\s+", "", q, flags=re.IGNORECASE)
    # Enforce single question: keep up to first '?'
    if "?" in q:
        first = q.split("?")[0].strip()
        q = first + "?"
    else:
        if not q.endswith("?"):
            q = (q + "?").strip()
    
    # Check for semantic similarity using n-gram fingerprinting (anti-repetition)
    if session_id:
        # Use async version from anti_repetition module for better paraphrase detection
        try:
            from agents.interview_agent.anti_repetition import check_question_similarity as async_check_similarity
            is_similar = await async_check_similarity(q, session_id)
        except (ImportError, AttributeError, Exception):
            # Fallback to sync version if async import fails
            is_similar = await _rbi(check_question_similarity, q, session_id)
        if is_similar:
            log.info(f"Anti-Repetition: Question blocked due to similarity, using fallback")
            # Try to get a different question
            try:
                fallback_q = await get_natural_fallback_question(conversation_context, candidate_info, session_id, uid_context)
                # Check if fallback is also similar
                if not await _rbi(check_question_similarity, fallback_q, session_id):
                    return fallback_q
            except Exception:
                pass
            # If fallback also similar or failed, use generic safe question
            return "Can you tell me more about your experience?"
    
    # Avoid repetition against previous N (default: all provided)
    # Handle both string and dict formats for previous_questions
    previous_texts = []
    for pq in previous_questions:
        if isinstance(pq, dict):
            # Extract text from dict format
            text = pq.get("question_text", pq.get("content", ""))
            if text:
                previous_texts.append(text)
        elif isinstance(pq, str):
            previous_texts.append(pq)
    
    is_generic_opener = any(re.search(pat, q.strip().lower()) for pat in GENERIC_OPENER_PATTERNS)
    if any(q.strip().lower() == pq.strip().lower() for pq in previous_texts) or (previous_texts and is_generic_opener):
        # Avoid repeating the last question: pick a different safe default for current state if available
        try:
            state = _map_phase_to_state(conversation_context.interview_phase)
            cfg = get_interview_config()
            qmap = cfg.get("safe_default_questions", {})
            options = qmap.get(state.value, [])
            alt = next((opt for opt in options if opt.strip().lower() != q.strip().lower() and opt.strip().lower() not in [p.strip().lower() for p in previous_texts[-3:]]), None)
            if alt:
                q = alt
            else:
                # Fall back to role-neutral, non-duplicate variant
                q = FALLBACK_NON_DUP_QUESTION
        except Exception:
            q = await get_natural_fallback_question(conversation_context, candidate_info, uid_context=uid_context)
    # Keep concise (smart truncation at word boundaries)
    if len(q) > MAX_QUESTION_LEN:
        original_len = len(q)
        # Find the last sentence or phrase boundary before MAX_QUESTION_LEN
        truncated = q[:MAX_QUESTION_LEN]
        # Try to break at sentence boundaries first (., !, ?)
        last_sentence = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        if last_sentence > MAX_QUESTION_LEN * 0.7:  # If sentence break is reasonably close
            q = truncated[:last_sentence+1]
            if not q.endswith('?'):
                q = q.rstrip('.!') + '?'
        else:
            # Break at last complete word
            last_space = truncated.rfind(' ')
            if last_space > 0:
                q = truncated[:last_space].rstrip() + '?'
            else:
                q = truncated.rstrip() + '?'
        log.warning(f"Question truncated from {original_len} to {len(q)} chars at word boundary")
    
    # Validate question quality
    # Minimum 8 characters to allow short but valid questions (e.g., "Who are you?" = 11 chars)
    # But reject very short fragments that are likely incomplete
    if len(q) < 8 or not q.endswith("?"):
        log.warning(f"Question validation failed: '{q}' (length: {len(q)}) - using fallback")
        return await get_natural_fallback_question(conversation_context, candidate_info, uid_context=uid_context)
    
    # Check for truncated words (ends with partial word)
    # Only flag if last word is suspiciously short AND question seems incomplete
    if q.endswith("?") and len(q) > 10:
        last_word = q.split()[-1].replace("?", "").strip()
        # Common short words that are valid: "on", "in", "at", "to", "of", "it", "is", "as", "if", "or", "so", "up", "we", "do", "go", "no", "my", "me", "be", "he", "an", "am", "us"
        valid_short_words = {"on", "in", "at", "to", "of", "it", "is", "as", "if", "or", "so", "up", "we", "do", "go", "no", "my", "me", "be", "he", "an", "am", "us", "by", "ok", "hi", "oh"}
        
        # Only flag as truncated if:
        # 1. Last word is very short (< 3 chars) AND
        # 2. It's not a common valid word AND
        # 3. Question doesn't end with common valid patterns
        if len(last_word) < 3 and last_word.lower() not in valid_short_words:
            # Additional check: if question ends with common patterns, it's likely complete
            question_lower = q.rstrip("?").lower()
            common_valid_endings = ["worked on", "focused on", "based on", "relied on", "depended on", "built on", 
                                   "worked in", "used in", "interested in", "experience with", "knowledge of", 
                                   "familiar with", "expertise in", "proficiency in"]
            if not any(question_lower.endswith(ending) for ending in common_valid_endings):
                log.warning(f"Question appears truncated: '{q}' - using fallback")
                return await get_natural_fallback_question(conversation_context, candidate_info, uid_context=uid_context)
    
    # Track question fingerprint for anti-repetition (after approval)
    if session_id:
        track_question_fingerprint(q, session_id)
    
    return q

def process_uid_context(context: Dict[str, Any], interview_req: InterviewRequest) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], List[str]]:
    """Process UID context and merge with interview request data"""
    
    # Use resume from context if not provided in request
    resume_data = interview_req.resume
    if not resume_data and context.get("resume_data"):
        resume_data = context["resume_data"]
    elif not resume_data:
        # Fallback to basic resume structure
        resume_data = {
            "Name": "Candidate",
            "RoleFit": ["Software Developer"],
            "skills": [{"Category": "General", "Items": [{"Name": "Programming"}]}],
            "experience": []
        }
    
    # Use job details from context if not provided in request
    job_details = interview_req.job_details
    if not job_details and context.get("job_details"):
        job_details = context["job_details"]
    elif not job_details:
        # Try to extract from experience data
        if context.get("experience_data") and "work_experience" in context["experience_data"]:
            work_experience = context["experience_data"]["work_experience"]
            if work_experience and len(work_experience) > 0:
                latest_job = work_experience[0]
                job_details = {
                    "title": latest_job.get("job_title", "Professional"),
                    "company": latest_job.get("company", ""),
                    "location": latest_job.get("location", ""),
        "requirements": latest_job.get("responsibilities", []) or []
                }
            else:
                job_details = {"title": "Professional", "requirements": []}
        else:
            # Final fallback - use generic title
            job_details = {"title": "Professional", "requirements": []}
    
    # Extract names and companies for anonymization
    names = []
    companies = []
    
    if resume_data:
        # Check multiple name field variations (matching extract_and_anonymize_candidate_info pattern)
        name = (
            resume_data.get("Name") or 
            resume_data.get("name") or
            resume_data.get("fullName") or
            resume_data.get("full_name")
        )
        
        # Handle nested Personal_Information structure
        if not name and "Personal_Information" in resume_data:
            personal_info = resume_data.get("Personal_Information", {})
            if isinstance(personal_info, dict):
                name = (
                    personal_info.get("Name") or
                    personal_info.get("name") or
                    personal_info.get("FullName") or
                    personal_info.get("fullName") or
                    personal_info.get("full_name")
                )
        
        if name:
            names.append(name)
        
        if "experience" in resume_data:
            for exp in resume_data["experience"]:
                if "company" in exp:
                    companies.append(exp["company"])
    
    return resume_data, job_details, names, companies

def analyze_response_depth(response_analysis: Dict[str, Any]) -> float:
    """Analyze response depth without LLM calls - O(1) complexity"""
    
    word_count = response_analysis.get("word_count", 0)
    engagement_score = response_analysis.get("engagement_score", 0)
    keywords = response_analysis.get("keywords", [])
    context_understanding = response_analysis.get("context_understanding", "")
    
    # Depth scoring algorithm (0-1.0)
    depth_score = 0.0
    
    # Length factor (0-0.3)
    if word_count > 100:
        depth_score += 0.3
    elif word_count > 50:
        depth_score += 0.2
    elif word_count > 20:
        depth_score += 0.1
    
    # Engagement factor (0-0.4)
    depth_score += engagement_score * 0.4
    
    # Keyword diversity factor (0-0.2)
    unique_keywords = len(set(keywords))
    depth_score += min(unique_keywords / 10, 0.2)
    
    # Context richness factor (0-0.1)
    if len(context_understanding) > 50:
        depth_score += 0.1
    
    return min(depth_score, 1.0)

def identify_skill_gaps(response_analysis: Dict[str, Any], candidate_context: Dict[str, Any]) -> List[str]:
    """Identify missing skills efficiently - O(n) complexity"""
    
    mentioned_skills = set([kw.lower() for kw in response_analysis.get("keywords", [])])
    candidate_skills = set([skill.lower() for skill in candidate_context.get("skills", [])])
    job_requirements = set([req.lower() for req in candidate_context.get("job_requirements", [])])
    
    # If no job requirements, use a more intelligent approach
    if not job_requirements:
        # Look for common skill categories that might be missing
        all_mentioned = mentioned_skills | candidate_skills
        
        # Define skill categories to explore
        skill_categories = {
            "technical": ["programming", "coding", "development", "software", "engineering"],
            "frontend": ["react", "javascript", "html", "css", "ui", "ux", "frontend"],
            "backend": ["python", "java", "node", "api", "database", "sql", "backend"],
            "data": ["data", "analysis", "sql", "database", "analytics", "machine learning"],
            "devops": ["deployment", "cloud", "aws", "docker", "ci/cd", "infrastructure"],
            "soft_skills": ["leadership", "communication", "teamwork", "management", "collaboration"],
            "testing": ["testing", "qa", "quality", "automation", "test"]
        }
        
        # Find categories that haven't been mentioned
        missing_categories = []
        for category, skills in skill_categories.items():
            if not any(skill in all_mentioned for skill in skills):
                missing_categories.append(category)
        
        # Return missing categories as skill gaps
        return missing_categories[:3]
    
    # Original logic for when job requirements are available
    skill_gaps = job_requirements - mentioned_skills - candidate_skills
    
    # Prioritize gaps by importance
    priority_gaps = []
    core_skills = ["react", "javascript", "python", "java", "sql", "html", "css"]
    soft_skills = ["leadership", "communication", "teamwork", "problem_solving", "management"]
    
    for gap in skill_gaps:
        if gap in core_skills:
            priority_gaps.append(gap)
        elif gap in soft_skills:
            priority_gaps.append(gap)
        else:
            priority_gaps.append(gap)
    
    return priority_gaps[:3]  # Top 3 gaps

def analyze_conversation_flow(session_context: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze conversation flow patterns - O(n) complexity"""
    
    topics_discussed = session_context.get("topics_discussed", [])
    question_count = session_context.get("question_count", 0)
    coverage_bins = session_context.get("coverage_bins", {})
    
    # Analyze topic distribution
    topic_distribution = {}
    for topic in topics_discussed:
        topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
    
    # Check for topic dominance
    max_topic_count = max(topic_distribution.values()) if topic_distribution else 0
    needs_transition = max_topic_count > question_count * 0.4  # 40% threshold
    
    # Check coverage balance
    min_coverage = min(coverage_bins.values()) if coverage_bins else 0
    max_coverage = max(coverage_bins.values()) if coverage_bins else 0
    coverage_imbalance = max_coverage - min_coverage > 3
    
    # Determine dominant topic
    dominant_topic = max(topic_distribution, key=topic_distribution.get) if topic_distribution else None
    
    return {
        "needs_transition": needs_transition,
        "coverage_imbalance": coverage_imbalance,
        "dominant_topic": dominant_topic,
        "topic_distribution": topic_distribution
    }

def _fallback_skill_extraction(resp_lower: str, conversation_context) -> None:
    """Fallback skill extraction using simple keyword patterns"""
    import re
    
    # Common skill patterns (much smaller set as fallback)
    skill_patterns = [
        r'\b([a-z]+(?:\.js|\.py|\.java|\.net|\.sql))\b',  # File extensions
        r'\b(react|angular|vue|python|java|javascript|sql|mongodb|redis|docker|kubernetes|aws|azure|gcp)\b',  # Common tech
        r'\b(salesforce|hubspot|excel|powerpoint|word|outlook|slack|teams|zoom)\b',  # Common tools
        r'\b(marketing|sales|hr|finance|operations|management|leadership|strategy|analysis|reporting)\b',  # Common roles
        r'\b(agile|scrum|kanban|lean|six sigma|pmp|certification)\b',  # Methodologies
    ]
    
    for pattern in skill_patterns:
        matches = re.findall(pattern, resp_lower)
        for match in matches:
            skill_normalized = match.lower().replace(" ", "_").replace("-", "_").replace(".", "")
            if skill_normalized not in conversation_context.coverage:
                conversation_context.coverage[skill_normalized] = 0

class SkillExtractionResult(BaseModel):
    """Structured output for skill extraction from interview responses."""
    skills: List[str] = Field(default_factory=list, description="List of extracted skills, tools, and competencies")
    
    @field_validator('skills', mode='before')
    @classmethod
    def sanitize_skills(cls, v):
        """Sanitize skill list: limit count and clean values."""
        return _sanitize_list(v or [], max_items=30, max_item_length=50)


async def _extract_skills_from_response(response_text: str) -> List[str]:
    """Extract skills from interview response using LLM with structured output (Issue 5.1)"""
    try:
        skill_extraction_prompt = f"""Extract specific skills, tools, technologies, and competencies mentioned in this interview response.

RESPONSE: "{response_text}"

Examples:
- "I used React and Redux" → skills: ["react", "redux"]
- "Managed a team of 5 developers" → skills: ["team_management", "leadership"]
- "Conducted user research interviews" → skills: ["user_research", "interviewing"]
- "Optimized database queries" → skills: ["database_optimization", "sql"]

Extract ALL relevant skills, tools, and competencies mentioned."""

        result: SkillExtractionResult = await invoke_structured_llm(
            skill_extraction_prompt,
            SkillExtractionResult,
            task_type=TaskType.INTERVIEW,
            preferred_model=settings.GEMINI_MODEL,
            agent_name="interview_skill_extraction",
            temperature=0.1,
            timeout=_INTERVIEW_LLM_TIMEOUT,
            raise_on_fallback=False,
        )
        
        return [skill.lower().replace(" ", "_").replace("-", "_").replace(".", "") for skill in result.skills]
            
    except Exception as e:
        log.warning(f"LLM skill extraction failed: {e}")
        return []

# Sophisticated Follow-up Question Templates
DEPTH_PROBING_TEMPLATES = {
    "technical": [
        "Can you walk me through the technical implementation details of that project?",
        "What specific challenges did you face during the development process?",
        "How did you measure the success and impact of this solution?",
        "What tools and technologies did you use to solve this problem?",
        "Can you explain the architecture or design decisions you made?"
    ],
    "behavioral": [
        "What was your thought process when approaching that situation?",
        "How did you handle the team dynamics and collaboration?",
        "What would you do differently if faced with the same challenge?",
        "How did you communicate your ideas to stakeholders?",
        "What did you learn from this experience?"
    ],
    "problem_solving": [
        "What alternative approaches did you consider before choosing this solution?",
        "How did you validate that your approach was the right one?",
        "What were the trade-offs you had to make?",
        "How did you handle unexpected obstacles?",
        "What metrics did you use to track progress?"
    ]
}

SKILL_EXPLORATION_TEMPLATES = {
    "react": "I notice you haven't mentioned React yet. Can you tell me about your React experience and how you've used it in your projects?",
    "javascript": "I'd love to hear more about your JavaScript expertise. What are some advanced concepts you've worked with?",
    "python": "Can you share your experience with Python? What libraries or frameworks have you used?",
    "leadership": "How have you demonstrated leadership in your previous roles? Can you give me a specific example?",
    "communication": "Tell me about a time when you had to communicate complex technical concepts to non-technical stakeholders.",
    "teamwork": "How do you approach collaboration in team environments? What's your experience working with cross-functional teams?",
    "problem_solving": "Walk me through your problem-solving process. How do you typically approach complex challenges?",
    "management": "What's your experience with project management or team management?",
    "testing": "What's your approach to testing and quality assurance? How do you ensure code quality?",
    "technical": "I'd like to hear more about your technical background. What programming languages and technologies are you most comfortable with?",
    "frontend": "Can you tell me about your frontend development experience? What frameworks or tools have you worked with?",
    "backend": "I'd love to hear about your backend development experience. What technologies and architectures have you worked with?",
    "data": "What's your experience with data analysis or data science? What tools and techniques have you used?",
    "devops": "Can you share your experience with deployment and infrastructure? What DevOps tools and practices are you familiar with?",
    "soft_skills": "How do you approach collaboration and communication in your work? Can you give me an example of a challenging team situation you've handled?"
}

TRANSITION_TEMPLATES = {
    "technical_to_behavioral": "That's great technical insight. Now, let's talk about your experience working with teams. How do you handle collaboration and communication?",
    "behavioral_to_technical": "Excellent leadership example. On the technical side, can you tell me about a challenging technical problem you've solved?",
    "topic_shift": "I'd like to explore a different aspect of your experience. Can you tell me about your approach to continuous learning and skill development?",
    "depth_to_breadth": "That's a deep dive into that area. Let me ask about your broader experience - how do you stay current with industry trends?",
    "breadth_to_depth": "You've mentioned several areas. Let's focus on one - can you go deeper into your experience with [specific skill]?"
}

CONTINUATION_TEMPLATES = {
    "high_engagement": [
        "That's fascinating! Can you tell me more about the impact this had?",
        "Excellent! How did this experience shape your approach to similar challenges?",
        "Great example! What were the key learnings you took away from this?"
    ],
    "medium_engagement": [
        "That's interesting. Can you elaborate on that point?",
        "I see. How did you handle the implementation details?",
        "Good point. What was the outcome of this approach?"
    ],
    "low_engagement": [
        "I'd like to understand this better. Can you provide more specific details?",
        "That's helpful. Can you walk me through a specific example?",
        "I'm curious about this. What made this approach effective?"
    ]
}

from functools import lru_cache

# Circuit breaker for LLM calls
class CircuitBreaker:
    def __init__(self, failure_threshold=3, timeout=30):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self):
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# Global circuit breaker instance
llm_circuit_breaker = CircuitBreaker()

# Cache for frequently used analyses
@lru_cache(maxsize=100)
def analyze_response_depth_cached(response_hash: str, word_count: int, engagement_score: float, keyword_count: int, context_length: int) -> float:
    """Cached version of depth analysis"""
    # Reconstruct response analysis from cached parameters
    response_analysis = {
        "word_count": word_count,
        "engagement_score": engagement_score,
        "keywords": [f"keyword_{i}" for i in range(keyword_count)],
        "context_understanding": "x" * context_length
    }
    return analyze_response_depth(response_analysis)

@lru_cache(maxsize=50)
def identify_skill_gaps_cached(mentioned_skills_hash: str, candidate_skills_hash: str, job_requirements_hash: str) -> tuple:
    """Cached skill gap identification"""
    # For now, disable caching to avoid hardcoded responses
    # In production, you'd implement proper caching with actual skill lists
    return ()  # Return empty to fall back to direct analysis
def get_response_hash(response_analysis: Dict[str, Any]) -> str:
    """Generate hash for response analysis caching"""
    key_data = f"{response_analysis.get('word_count', 0)}_{response_analysis.get('engagement_score', 0)}_{len(response_analysis.get('keywords', []))}_{len(response_analysis.get('context_understanding', ''))}"
    return hashlib.md5(key_data.encode()).hexdigest()
def get_skills_hash(skills_list: List[str]) -> str:
    """Generate hash for skills list caching"""
    return hashlib.md5(",".join(sorted(skills_list)).encode()).hexdigest()

def generate_sophisticated_follow_up(response_analysis: Dict[str, Any], 
                                   candidate_context: Dict[str, Any], 
                                   session_context: Dict[str, Any]) -> str:
    """Generate sophisticated follow-up questions with minimal latency"""
    
    # Analyze response characteristics (with caching for performance)
    try:
        # Use cached analysis if possible
        response_hash = get_response_hash(response_analysis)
        depth_score = analyze_response_depth_cached(
            response_hash,
            response_analysis.get("word_count", 0),
            response_analysis.get("engagement_score", 0),
            len(response_analysis.get("keywords", [])),
            len(response_analysis.get("context_understanding", ""))
        )
    except Exception:
        # Fallback to direct analysis
        depth_score = analyze_response_depth(response_analysis)
    
    # Skill gap analysis
    mentioned_skills = [kw.lower() for kw in response_analysis.get("keywords", [])]
    candidate_skills = candidate_context.get("skills", [])
    job_requirements = candidate_context.get("job_requirements", [])
    
    try:
        # Use cached skill gap analysis if possible
        mentioned_hash = get_skills_hash(mentioned_skills)
        candidate_hash = get_skills_hash(candidate_skills)
        job_hash = get_skills_hash(job_requirements)
        skill_gaps = list(identify_skill_gaps_cached(mentioned_hash, candidate_hash, job_hash))
    except Exception:
        # Fallback to direct analysis
        skill_gaps = identify_skill_gaps(response_analysis, candidate_context)
    
    # Conversation flow analysis (no caching needed - O(n) complexity)
    conversation_flow = analyze_conversation_flow(session_context)
    
    response_type = response_analysis.get("type", "general")
    engagement_score = response_analysis.get("engagement_score", 0.5)
    
    # Determine follow-up strategy (ultra-optimized for speed)
    if depth_score < 0.4:  # Very shallow response - use ultra-fast probing
        return generate_ultra_fast_question(response_analysis)
    
    elif skill_gaps and len(skill_gaps) > 0:  # Missing skills - explore gaps
        return generate_skill_exploration_question(skill_gaps[0])
    
    elif conversation_flow.get("needs_transition", False):  # Topic transition needed
        return generate_transition_question(conversation_flow)
    
    else:  # Continue current thread - use ultra-fast fallback
        return generate_ultra_fast_question(response_analysis)

def generate_depth_probing_question(response_type: str, response_analysis: Dict[str, Any]) -> str:
    """Generate depth-probing questions based on response type"""
    
    templates = DEPTH_PROBING_TEMPLATES.get(response_type, DEPTH_PROBING_TEMPLATES["technical"])
    
    # Select template based on response content
    keywords = response_analysis.get("keywords", [])
    if any(kw in keywords for kw in ["project", "built", "developed", "created"]):
        return templates[0]  # Implementation details
    elif any(kw in keywords for kw in ["challenge", "problem", "issue", "difficult"]):
        return templates[1]  # Challenges faced
    elif any(kw in keywords for kw in ["success", "result", "impact", "improved"]):
        return templates[2]  # Success measurement
    else:
        return templates[0]  # Default to implementation details

def generate_skill_exploration_question(skill: str) -> str:
    """Generate skill exploration questions"""
    
    skill_lower = skill.lower()
    
    # Check for exact matches first
    if skill_lower in SKILL_EXPLORATION_TEMPLATES:
        return SKILL_EXPLORATION_TEMPLATES[skill_lower]
    
    # Check for partial matches
    for template_skill, template in SKILL_EXPLORATION_TEMPLATES.items():
        if template_skill in skill_lower or skill_lower in template_skill:
            return template.replace(template_skill, skill)
    
    # Generic skill exploration
    return f"I'd like to hear more about your experience with {skill}. Can you tell me about a specific project where you used this skill?"

def generate_transition_question(conversation_flow: Dict[str, Any]) -> str:
    """Generate transition questions"""
    
    dominant_topic = conversation_flow.get("dominant_topic")
    
    if dominant_topic:
        # Transition away from dominant topic
        if dominant_topic in ["react", "javascript", "python", "java"]:
            return TRANSITION_TEMPLATES["technical_to_behavioral"]
        elif dominant_topic in ["team", "leadership", "management"]:
            return TRANSITION_TEMPLATES["behavioral_to_technical"]
        else:
            return TRANSITION_TEMPLATES["topic_shift"]
    else:
        return TRANSITION_TEMPLATES["topic_shift"]

def generate_continuation_question(engagement_score: float, response_analysis: Dict[str, Any]) -> str:
    """Generate continuation questions based on engagement level"""
    
    if engagement_score > 0.7:
        templates = CONTINUATION_TEMPLATES["high_engagement"]
    elif engagement_score > 0.4:
        templates = CONTINUATION_TEMPLATES["medium_engagement"]
    else:
        templates = CONTINUATION_TEMPLATES["low_engagement"]
    
    # Select template based on response content
    keywords = response_analysis.get("keywords", [])
    if any(kw in keywords for kw in ["impact", "result", "success", "improved"]):
        return templates[0]  # Impact-focused
    elif any(kw in keywords for kw in ["implement", "built", "developed", "created"]):
        return templates[1]  # Implementation-focused
    else:
        return templates[2]  # Example-focused

def generate_fast_continuation_question(engagement_score: float, response_analysis: Dict[str, Any]) -> str:
    """Generate fast continuation questions without complex analysis"""
    
    # Fast, template-based questions
    fast_templates = [
        "That's interesting. Can you tell me more about that?",
        "I'd like to hear more details. What was your approach?",
        "That sounds like a good example. Can you walk me through what happened?",
        "Interesting perspective. How did you handle that situation?",
        "That's helpful context. What were the key challenges you faced?"
    ]
    
    # Select based on engagement level
    if engagement_score > 0.7:
        return random.choice(fast_templates[:3])  # More engaging questions
    else:
        return random.choice(fast_templates[2:])  # More probing questions

def generate_ultra_fast_question(response_analysis: Dict[str, Any]) -> str:
    """Ultra-fast question generation using pre-computed templates"""
    
    response_type = response_analysis.get("type", "general")
    keywords = response_analysis.get("keywords", [])
    
    # Pre-computed question templates by type
    ultra_fast_templates = {
        "technical": [
            "Can you walk me through the technical implementation?",
            "What challenges did you face with that approach?",
            "How did you ensure the solution was scalable?",
            "What tools or technologies did you use?",
            "Can you describe the architecture you implemented?"
        ],
        "behavioral": [
            "Can you give me a specific example of that?",
            "How did you handle that situation?",
            "What was the outcome of that approach?",
            "What did you learn from that experience?",
            "How did you work with your team on this?"
        ],
        "collaboration": [
            "How did you approach working with your team?",
            "Can you describe a challenging team situation?",
            "How do you handle disagreements in the team?",
            "What's your approach to mentoring others?",
            "How do you ensure good communication?"
        ],
        "general": [
            "That's interesting. Can you tell me more?",
            "What was your thought process there?",
            "How did you approach that problem?",
            "What made that approach effective?",
            "Can you walk me through what happened?"
        ]
    }
    
    # Select template based on response type
    templates = ultra_fast_templates.get(response_type, ultra_fast_templates["general"])
    
    # If we have specific keywords, try to match them
    if any(kw in keywords for kw in ["react", "javascript", "frontend"]):
        return "Can you tell me more about your frontend development experience?"
    elif any(kw in keywords for kw in ["python", "java", "backend", "api"]):
        return "What's your experience with backend development?"
    elif any(kw in keywords for kw in ["database", "sql", "data"]):
        return "Can you describe your database design approach?"
    elif any(kw in keywords for kw in ["team", "collaboration", "mentor"]):
        return "How do you approach team collaboration?"
    else:
        return random.choice(templates)

async def send_callback_notification(callback_url: str, uid: str, session_id: str, 
                                   status: str, evaluation_summary: Dict[str, Any] = None,
                                   detailed_summary: str = None, auth_token: str = None,
                                   conversation_context: Dict[str, Any] = None,
                                   response_analysis: Dict[str, Any] = None,
                                   session_metadata: Dict[str, Any] = None):
    """Send callback notification to the provided URL"""
    try:
        import aiohttp
        
        # Format callback data to match callback_validator.py schema
        callback_data = {
            "node": "interview_agent",
            "status": status,
            "output": {
                "uid": uid,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "evaluation_summary": evaluation_summary,
                "detailed_summary": detailed_summary,
                "conversation_context": conversation_context,
                "response_analysis": response_analysis,
                "session_metadata": session_metadata
            }
        }
        
        # Log callback data
        log.debug(f"Callback data - UID: {uid}, Session: {session_id}, Status: {status}")
        
        if evaluation_summary:
            log.info(f"Evaluation summary - Overall Score: {evaluation_summary.get('overall_score', 'N/A')}, Status: {evaluation_summary.get('status', 'N/A')}, Questions: {evaluation_summary.get('questions_asked', 'N/A')}")
        
        # Prepare headers with auth token if provided
        headers = {
            "Content-Type": "application/json"
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
            # Mask token in logs
            masked_token = mask_sensitive_data({"auth_token": auth_token})["auth_token"]
            log.debug(f"Using auth token for callback: {masked_token}")
        else:
            log.warning("No auth token provided for callback")
        
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                callback_url,
                json=callback_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response_text = await response.text()
                if response.status == 200:
                    log.info(f"Callback sent successfully to {callback_url}")
                else:
                    log.warning(f"Callback failed with status {response.status}: {response_text}")
    
    except Exception as e:
        log.error(f"Error sending callback to {callback_url}: {e}")


# -------------------------
# Helper Functions for Interview Agent Refactoring
# -------------------------

async def _parse_and_validate_request(request: Request) -> InterviewRequest:
    """Handles parsing, validation, and auth token extraction."""
    data = await request.json()

    # Extract auth token from header (preferred) or body (fallback)
    auth_token = request.headers.get("authorization", "").replace("Bearer ", "")
    auth_method = "header" if auth_token else "none"

    if not auth_token and "auth_token" in data:
        auth_token = data.get("auth_token")
        auth_method = "body"

    # Add auth_token to data if found in header
    if auth_token and "auth_token" not in data:
        data["auth_token"] = auth_token

    # Log auth method without exposing token
    log.debug(f"Auth token method: {auth_method}, token present: {bool(auth_token)}")

    # Normalize payload: handle structured_resume sent at top level
    if "structured_resume" in data:
        structured_resume_value = data["structured_resume"]
        # If structured_resume is a JSON string, parse it; if empty string, use None
        if isinstance(structured_resume_value, str):
            if structured_resume_value.strip():
                try:
                    structured_resume_value = json.loads(structured_resume_value)
                except (json.JSONDecodeError, ValueError):
                    # If not valid JSON, treat as empty
                    structured_resume_value = None
            else:
                structured_resume_value = None

        # Keep structured_resume at top level for topic-focused interviews
        if structured_resume_value:
            data["structured_resume"] = structured_resume_value
            # Also wrap in resume object for backward compatibility
            if "resume" not in data:
                data["resume"] = {"structured_resume": structured_resume_value}
        else:
            data["structured_resume"] = None
            if "resume" not in data:
                data["resume"] = None

    # ALWAYS log full input payload to terminal
    try:
        # Mask sensitive data before logging
        masked_data = mask_sensitive_data(data)
        log.debug("=" * 80)
        log.debug("INTERVIEW AGENT INPUT PAYLOAD:")
        log.debug("=" * 80)
        log.debug(json.dumps(masked_data, indent=2, default=str))
        log.debug("=" * 80)
        log.info(f"INTERVIEW REQ: uid={data.get('uid')} | session_id={data.get('session_id')} | end={data.get('end_interview')}")
        log.debug(f"Payload keys: {sorted(list(data.keys()))}")
    except Exception as e:
        log.error(f"Failed to log input payload: {e}")

    interview_req = InterviewRequest(**data)

    # Extract structured_resume from resume if not at top level
    try:
        if not interview_req.structured_resume and interview_req.resume:
            if isinstance(interview_req.resume, dict):
                # If resume contains a "structured_resume" key, use that
                if "structured_resume" in interview_req.resume:
                    interview_req.structured_resume = interview_req.resume["structured_resume"]
                    log.debug("Extracted structured_resume from resume['structured_resume']")
                # Otherwise, if resume is a dict with actual data (has keys like skills, experience, etc.),
                # treat the entire resume dict as the structured_resume (universal for any industry/role)
                elif interview_req.resume and len(interview_req.resume) > 0:
                    # Check if it looks like a structured resume (has typical resume fields - universal across all industries)
                    resume_keys = set(interview_req.resume.keys())
                    # Universal resume fields that appear across all industries and roles
                    typical_resume_keys = {
                        # Core fields (universal)
                        "skills", "experience", "work_experience", "education", "name",
                        # Contact fields (various formats)
                        "Email", "Phone", "Location", "contact_details", "LinkedIn", "GitHub",
                        # Additional fields (universal)
                        "projects", "certifications", "professional_summary", "summary",
                        # Extended fields (may be present)
                        "RoleFit", "resumeScore", "total_experience_years", "extras",
                        "professional_affiliations", "awards", "publications"
                    }
                    if resume_keys.intersection(typical_resume_keys):
                        interview_req.structured_resume = interview_req.resume
                        log.debug(f"Using entire resume dict as structured_resume (has {len(resume_keys)} keys, detected as structured resume)")
    except Exception as e:
        log.warning(f"Error extracting structured_resume from resume: {e}. Continuing without structured_resume.")
        # Don't fail the request, just log the warning
    
    # Log interview topic and structured_resume status for debugging
    if interview_req.interview_topic:
        log.debug(f"Interview topic provided: {interview_req.interview_topic}")
        log.debug(f"Structured resume available: {bool(interview_req.structured_resume)}")
        if interview_req.structured_resume:
            resume_keys = list(interview_req.structured_resume.keys())[:10]
            log.debug(f"Structured resume keys: {resume_keys}")
            # Log if skills are available for topic matching
            if "skills" in interview_req.structured_resume:
                skills_count = len(interview_req.structured_resume.get("skills", []))
                log.debug(f"Structured resume has {skills_count} skills for topic matching")
        else:
            log.warning(f"Interview topic '{interview_req.interview_topic}' provided but structured_resume is None - topic-focused features may not work optimally")

    # Validate critical fields and cap history
    if not interview_req.uid:
        raise HTTPException(status_code=400, detail="uid is required")
    if interview_req.conversation_history and len(interview_req.conversation_history) > 50:
        interview_req.conversation_history = interview_req.conversation_history[-50:]
        log.info("Capped conversation_history to last 50 messages")

    return interview_req


def _build_interview_context_data(interview_req: InterviewRequest) -> Tuple[Dict, Dict, Dict, List[str], List[str], str, float, PIIAnonymizer]:
    """Builds all context data needed for the interview."""
    # Build context from the request payload
    uid_context = build_context_from_payload(interview_req)
    log.debug(f"Built context for UID {interview_req.uid}: {list(uid_context.keys())}")

    # Quick context stats for terminal
    try:
        sr = (uid_context.get("resume_data") or {}) if isinstance(uid_context, dict) else {}
        exp = sr.get("experience") or []
        skills = sr.get("skills") or []
        jt = (interview_req.job_details or {}).get("title") or (
            exp[0].get("job_title") if exp and isinstance(exp[0], dict) else "Professional"
        )
        name = sr.get("Name") or "Candidate"
        log.debug(f"CTX name={name} exp_count={len(exp)} skills_count={len(skills)} job_title={jt}")
    except Exception:
        pass

    # Process UID context and merge with interview request data
    resume_data, job_details, names, companies = process_uid_context(uid_context, interview_req)

    log.debug(f"Processed resume_data: Name='{resume_data.get('Name', 'N/A')}', RoleFit='{resume_data.get('RoleFit', [])}'")

    # Initialize session metadata
    session_id = interview_req.session_id or f"session_{int(time.time())}"
    start_time = interview_req.start_time or time.time()

    # Initialize enhanced anonymizer with session ID
    anonymizer = PIIAnonymizer()
    anonymizer.set_session_id(session_id)

    # Extract and anonymize candidate info
    candidate_info, names, companies = extract_and_anonymize_candidate_info(
        resume_data,
        anonymizer
    )

    # Enhance candidate info with ChromaDB data
    if uid_context and uid_context.get("personal_info") and "name" in uid_context["personal_info"]:
        candidate_info["name"] = uid_context["personal_info"]["name"]

    # Add skills from ChromaDB
    if uid_context and uid_context.get("skills_analysis") and "skills" in uid_context["skills_analysis"]:
        skills_list = []
        for skill in uid_context["skills_analysis"]["skills"]:
            if "SkillName" in skill:
                skills_list.append(skill["SkillName"])
        if skills_list:
            candidate_info["skills"] = ", ".join(skills_list)

    # Sanitize job details
    safe_job_details = sanitize_job_details(job_details)

    return uid_context, candidate_info, safe_job_details, names, companies, session_id, start_time, anonymizer


async def _initialize_conversation_history(
    interview_req: InterviewRequest,
    uid_context: Dict,
    candidate_info: Dict,
    safe_job_details: Dict,
    names: List[str]
) -> None:
    """Initializes conversation history with contextualized system message if empty."""
    if not interview_req.conversation_history:
        candidate_name = get_candidate_name(uid_context, candidate_info, names)
        job_title = get_job_title(uid_context, safe_job_details, candidate_info)
        skills_str = get_candidate_skills(uid_context, candidate_info)
        # Be defensive: some sessions may store `user_interests` as None
        interests_container = (uid_context.get("user_interests") or {}) if uid_context else {}
        interests = interests_container.get("user_interests", [])
        topics_discussed = []
        engagement_level = 0.5
        question_count_for_prompt = 0
        role_guidelines = derive_role_micro_guidelines(job_title)
        # facts memory initial summary (+ optional portfolio digest)
        facts_base = {
            "skills": [s.strip() for s in skills_str.split(",") if s.strip()] if skills_str else [],
            "interests": interests
        }
        if interview_req.portfolio_links:
            try:
                digest = await digest_portfolio_links(interview_req.portfolio_links)
                if digest:
                    facts_base["skills"] = sorted(list(set(facts_base.get("skills", [])) | set(digest.get("tech", []))))
                    # store roles/topics in facts under topics
                    extra_topics = digest.get("topics", [])
                    facts_base["topics"] = sorted(list(set(facts_base.get("topics", [])) | set(extra_topics)))
                    # keep summary in session metadata below
            except Exception:
                pass
        facts_summary = summarize_facts_memory(facts_base)
        system_msg = build_contextual_system_message(
            uid_context,
            candidate_name,
            job_title,
            skills_str,
            interests,
            topics_discussed,
            engagement_level,
            question_count_for_prompt,
            role_guidelines,
            facts_summary
        )
        interview_req.conversation_history = [{"role": "system", "content": system_msg}]
        

def _anonymize_and_prepare_history(
    interview_req: InterviewRequest,
    anonymizer: PIIAnonymizer,
    names: List[str],
    companies: List[str]
) -> List[Dict]:
    """Anonymizes conversation history and adds candidate's answer if present."""
    # Anonymize existing conversation history
    anonymized_history = anonymize_conversation_history(
        interview_req.conversation_history,
        anonymizer,
        names,
        companies
    )

    # Remove greetings from history to prevent LLM from repeating them
    anonymized_history = sanitize_history(anonymized_history, names)

    # Add candidate's answer to history (anonymized)
    if interview_req.answer.strip():
        anonymized_answer = anonymizer.anonymize_text(
            interview_req.answer.strip(),
            names,
            companies
        )
        anonymized_history.append({"role": "user", "content": anonymized_answer})

        # Also add to the original conversation history for response
        interview_req.conversation_history.append({"role": "user", "content": interview_req.answer})

    return anonymized_history


def _calculate_question_count(
    interview_req: InterviewRequest,
    server_question_count: int,
    anonymized_history: List[Dict]
) -> int:
    """Calculates the current question count with proper precedence."""
    # Calculate current progress (server-side precedence)
    # For first request, question_count should be 0
    if not interview_req.answer.strip() and not interview_req.conversation_history:
        question_count = 0
    else:
        question_count = server_question_count or interview_req.question_count or count_questions_in_history(anonymized_history)

    log.debug(f"Question count calculation: server={server_question_count}, req={interview_req.question_count}, history={count_questions_in_history(anonymized_history)}, final={question_count}")
    return question_count
        

def _initialize_conversation_context(uid_context: Dict) -> ConversationContext:
    """Initializes conversation context with comprehensive skill coverage."""
    conversation_context = ConversationContext()

    # Seed coverage bins with ALL skills from UID context (domain-agnostic)
    if uid_context:
        # Extract all skills from skills_parser
        if "skills_parser" in uid_context and isinstance(uid_context["skills_parser"], dict):
            skills_data = uid_context["skills_parser"].get("skills", [])
            for skill in skills_data:
                if isinstance(skill, dict) and "SkillName" in skill:
                    skill_name = skill["SkillName"].lower().replace(" ", "_").replace("-", "_")
                    conversation_context.coverage[skill_name] = 0

        # Extract additional skills from experience_parser (domain-agnostic)
        if "experience_parser" in uid_context and isinstance(uid_context["experience_parser"], dict):
            exp_data = uid_context["experience_parser"]
            if "work_experience" in exp_data:
                for exp in exp_data["work_experience"]:
                    if isinstance(exp, dict) and "responsibilities" in exp:
                        for resp in exp["responsibilities"]:
                            # Extract skills dynamically from responsibilities using intelligent parsing
                                resp_lower = resp.lower()
                                
                                # Smart skill extraction without hardcoding
                                import re
                                
                                # Extract capitalized words (likely proper nouns/technologies)
                                capitalized_skills = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\.[a-zA-Z0-9]+)*\b', resp)
                                for skill in capitalized_skills:
                                    skill_normalized = skill.lower().replace(".", "")
                                    if skill_normalized not in conversation_context.coverage:
                                        conversation_context.coverage[skill_normalized] = 0
                                
                                # Extract common skill patterns
                                skill_patterns = [
                                    r'\b([a-z]+(?:\.js|\.py|\.java|\.net|\.sql|\.ts|\.jsx|\.tsx))\b',  # File extensions
                                    r'\b([a-z]+_[a-z]+(?:_[a-z]+)?)\b',  # Snake_case skills
                                    r'\b([a-z]+-[a-z]+(?:-[a-z]+)?)\b',  # Kebab-case skills
                                    r'\b([a-z]+ [a-z]+(?: [a-z]+)?)\b',  # Multi-word skills
                                ]
                                
                                for pattern in skill_patterns:
                                    matches = re.findall(pattern, resp_lower)
                                    for match in matches:
                                        skill_normalized = match.replace(" ", "_").replace("-", "_").replace(".", "")
                                        if len(skill_normalized) > 2 and skill_normalized not in conversation_context.coverage:
                                            conversation_context.coverage[skill_normalized] = 0
                                
                                # Extract action-based skills (verbs + objects)
                                action_patterns = [
                                    r'\b(managed|led|developed|designed|implemented|optimized|analyzed|created|built|configured|deployed|maintained|monitored|tested|debugged|troubleshot|researched|analyzed|reported|presented|trained|mentored|coached|consulted|strategized|planned|executed|delivered|launched|marketed|sold|negotiated|procured|audited|complied|governed|regulated)\s+([a-z]+(?:\s+[a-z]+)*)\b',
                                ]
                                
                                for pattern in action_patterns:
                                    matches = re.findall(pattern, resp_lower)
                                    for action, target in matches:
                                        skill_normalized = f"{action}_{target}".replace(" ", "_")
                                        if skill_normalized not in conversation_context.coverage:
                                            conversation_context.coverage[skill_normalized] = 0
        
        # Ensure core competency bins exist
        core_bins = ["technical_skills", "problem_solving", "collaboration", "outcomes", "constraints"]
        for bin_name in core_bins:
            if bin_name not in conversation_context.coverage:
                conversation_context.coverage[bin_name] = 0
    
    return conversation_context


async def _setup_interview_evaluator(safe_job_details: Dict, uid_context: Dict) -> InterviewEvaluator:
    """Sets up the interview evaluator with callback handlers."""
    # Initialize interview evaluator with role-based scoring
    interview_evaluator = InterviewEvaluator(safe_job_details, uid_context)
    # Perform async role classification
    await interview_evaluator.async_init()

    # If the current request is topic-focused, strengthen topic weighting on evaluation
    # We detect this later where we have access to the request and call a small shim on the evaluator.
    # The actual application of weighting is handled by InterviewEvaluator.apply_topic_weighting if present.

    # Set up callback handler for writing to local_callbacks
    from utils.interview_callback_handler import InterviewCallbackHandler
    callback_handler = InterviewCallbackHandler()

    # Register callback handlers
    interview_evaluator.add_callback('on_interview_started',
        lambda conversation_length: callback_handler.on_interview_started(conversation_length))
    interview_evaluator.add_callback('on_question_asked',
        lambda question: callback_handler.on_question_asked(question))
    interview_evaluator.add_callback('on_response_received',
        lambda response, question: callback_handler.on_response_received(response, question))
    interview_evaluator.add_callback('on_skill_detected',
        lambda skill, score, context, *args: callback_handler.on_skill_detected(skill, score, context, args[0] if args else None))
    interview_evaluator.add_callback('on_score_updated',
        lambda criteria, score, evidence: callback_handler.on_score_updated(criteria, score, evidence))
    interview_evaluator.add_callback('on_red_flag_triggered',
        lambda flag, context: callback_handler.on_red_flag_triggered(flag, context))
    interview_evaluator.add_callback('on_green_flag_triggered',
        lambda flag, context: callback_handler.on_green_flag_triggered(flag, context))
    interview_evaluator.add_callback('on_interview_ended',
        lambda questions_count, responses_count: callback_handler.on_interview_ended(questions_count, responses_count))
    interview_evaluator.add_callback('on_evaluation_completed',
        lambda overall_score, status: callback_handler.on_evaluation_completed(overall_score, status))

    return interview_evaluator


async def _analyze_response(
    interview_req: InterviewRequest,
    anonymized_history: List[Dict],
    candidate_info: Dict,
    safe_job_details: Dict,
    uid_context: Dict,
    question_count: int
) -> Dict[str, Any]:
    """Analyzes the candidate's response using intelligent response analysis."""
    # Use intelligent response analysis
    is_first_request = question_count == 0 and not interview_req.answer.strip()
    log.debug(f"Debug: question_count={question_count}, answer='{interview_req.answer}', is_first_request={is_first_request}")

    if interview_req.answer.strip():
        return await hybrid_response_analyzer.analyze_response(
            interview_req.answer,
            anonymized_history,
            candidate_info,
            safe_job_details,
            is_first_request,
            uid_context
        )
    else:
        # For first request with no answer, use contextual analysis
        contextual_suggestions = get_contextual_suggestions(uid_context)
        return {
            "type": "general",
            "confidence": 0.5,
            "keywords": [],
            "sentiment": "neutral",
            "length_score": 0.0,
            "engagement_score": 0.0,
            "word_count": 0,
            "context_understanding": "First request - no answer provided",
            "follow_up_suggestions": contextual_suggestions,
            "strengths_mentioned": [],
            "areas_to_explore": []
        }
        

async def _generate_next_question(
    candidate_info: Dict,
    anonymized_history: List[Dict],
    safe_job_details: Dict,
    uid_context: Dict,
    question_count: int,
    conversation_context: ConversationContext,
    interview_req: InterviewRequest
) -> Tuple[str, Dict[str, Any], Dict, List[str]]:
    """Generates the next question using topic-focused or single LLM call architecture."""
    response_analysis = None
    enhanced_candidate_info = None
    previous_questions = []
    next_question = None
    
    # LLM-based intent detection for edge cases (replaces hardcoded patterns)
    async def _detect_and_handle_edge_intents_llm(answer: str, topic: Optional[str], conversation_history: List[Dict]) -> Optional[str]:
        """
        Uses LLM to intelligently detect edge case intents:
        - Lack of knowledge/unknown topic
        - Exit/end interview intent
        - Topic switch request
        Returns appropriate response message or None if no edge case detected.
        """
        try:
            if not isinstance(answer, str) or not answer.strip():
                return None
            
            # Pre-filter: Minimal rule-based detection for very obvious single-word responses
            # This provides a fast path for obvious cases (e.g., "no", "nope") to avoid LLM calls.
            # All other intent detection (refusal, uncertainty, exit, topic switch) is handled by LLM
            # for better semantic understanding and flexibility.
            answer_lower = answer.strip().lower()
            word_count = len(answer.split())
            
            # Only pre-filter very obvious single-word negatives (1-2 words) for performance
            # Longer responses and complex patterns are handled by LLM for semantic understanding
            if word_count <= 2:
                words_set = set(answer_lower.split())
                # Only check for very obvious single-word negatives
                if words_set.intersection(NEGATIVE_SINGLE_WORDS):
                        session_id = conversation_context.session_id if hasattr(conversation_context, 'session_id') else None
                        if session_id:
                            await _rbi(track_intent, session_id, "LACK_OF_KNOWLEDGE", 0.8, "Single word negative detected via pre-filter")
                            if await _rbi(should_offer_loop_breaker, session_id):
                                return get_loop_breaker_offer(topic)
                        # Return appropriate response for obvious negative
                        if topic:
                            return f"No problem! We can handle this a few ways: I can give you a quick primer on {topic}, we can switch to a different topic you're more comfortable with, or we can wrap up the interview. What would you prefer?"
                        return "No problem! We can handle this a few ways: I can give you a quick primer, we can switch to a different topic you're more comfortable with, or we can wrap up the interview. What would you prefer?"
            
            # NOTE: Refusal patterns, uncertainty phrases, and exit intent are now handled by LLM
            # for better semantic understanding. The LLM can understand context, nuance, and
            # various phrasings that regex patterns cannot capture.
            
            # Sanitize answer for safe embedding in prompt
            sanitized_answer = _sanitize_answer_for_prompt(answer)
            
            # Build context from recent conversation - use compressed history
            compressed_history = _compress_conversation_history(conversation_history, max_recent_turns=4)
            recent_context = compressed_history[-3:] if len(compressed_history) > 3 else compressed_history
            context_str = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:200]}"
                for msg in recent_context
            ]) if recent_context else "No previous conversation"
            
            # Create LLM prompt for intent detection using structured output (Issue 5.1)
            intent_detection_prompt = f"""You are analyzing a candidate's response during an interview to detect specific edge case intents.

CURRENT INTERVIEW TOPIC: {topic if topic else "General interview"}

RECENT CONVERSATION CONTEXT:
{context_str}

CANDIDATE'S CURRENT RESPONSE:
"{sanitized_answer}"

Analyze and determine if they are expressing:
1. LACK_OF_KNOWLEDGE: Indicates they don't know about the topic (directly or through refusal/uncertainty). NOT if they show willingness to learn.
2. EXIT_INTENT: Wants to end the interview (not just politeness like "thank you").
3. TOPIC_SWITCH: Wants to change to a different topic.
4. NONE: Normal response.

Be especially sensitive to uncooperative/negative responses. Short negative responses (<5 words) like "no", "won't", "can't" should be LACK_OF_KNOWLEDGE with high confidence (0.8+)."""

            try:
                intent_data: IntentDetectionResult = await invoke_structured_llm(
                    intent_detection_prompt,
                    IntentDetectionResult,
                    task_type=TaskType.INTERVIEW,
                    preferred_model=settings.GEMINI_MODEL,
                    agent_name="interview_intent_detection",
                    temperature=0.1,
                    timeout=_INTERVIEW_LLM_TIMEOUT,
                    raise_on_fallback=False,
                )
                
                intent = intent_data.intent
                confidence = intent_data.confidence
                reasoning = intent_data.reasoning
                new_topic_from_llm = intent_data.new_topic
            except Exception as e:
                log.warning(f"Intent detection failed: {e}")
                return None
            
            # Track intent for loop-breaker (always track, even if low confidence)
            session_id = conversation_context.session_id if hasattr(conversation_context, 'session_id') else None
            if session_id:
                await _rbi(track_intent, session_id, intent, confidence, reasoning)
            
            # Get confidence thresholds from config
            try:
                config = get_interview_config()
                thresholds = config.get("negative_intent", {}).get("confidence_thresholds", {})
                min_confidence = thresholds.get("standard", 0.7)
                if word_count < 5 and intent in ["LACK_OF_KNOWLEDGE", "EXIT_INTENT"]:
                    min_confidence = thresholds.get("short_response", 0.5)
                if word_count <= 3 and intent in ["LACK_OF_KNOWLEDGE", "EXIT_INTENT"]:
                    min_confidence = thresholds.get("very_short_response", 0.3)
            except Exception:
                # Fallback to defaults
                min_confidence = 0.7
                if word_count < 5 and intent in ["LACK_OF_KNOWLEDGE", "EXIT_INTENT"]:
                    min_confidence = 0.5
                if word_count <= 3 and intent in ["LACK_OF_KNOWLEDGE", "EXIT_INTENT"]:
                    min_confidence = 0.3
            
            # Use helper function to determine if negative intent should be handled
            should_handle, loop_breaker_response = await _rbi(
                _should_handle_negative_intent,
                intent, confidence, word_count, min_confidence, session_id, topic
            )
            
            if loop_breaker_response:
                log.debug("Loop-breaker triggered", extra={
                    "intent": intent,
                    "confidence": confidence,
                    "word_count": word_count,
                    "session_id": session_id
                })
                return loop_breaker_response
            
            if not should_handle:
                log.debug("Intent detection confidence too low", extra={
                    "intent": intent,
                    "confidence": confidence,
                    "min_confidence": min_confidence,
                    "word_count": word_count,
                    "session_id": session_id,
                    "reasoning": reasoning
                })
                return None
            
            # Handle detected intents
            if intent == "EXIT_INTENT":
                conversation_context.set_should_end(True)
                log.info("Exit intent detected", extra={
                    "intent": intent,
                    "confidence": confidence,
                    "session_id": session_id,
                    "reasoning": reasoning
                })
                return "Thanks for your time. We can wrap up here—I'll summarize and share next steps."
            
            elif intent == "LACK_OF_KNOWLEDGE":
                log.info("Lack of knowledge detected", extra={
                    "intent": intent,
                    "confidence": confidence,
                    "word_count": word_count,
                    "session_id": session_id,
                    "reasoning": reasoning
                })
                if topic:
                    return f"No problem! We can handle this a few ways: I can give you a quick primer on {topic}, we can switch to a different topic you're more comfortable with, or we can wrap up the interview. What would you prefer?"
                return "No problem! We can handle this a few ways: I can give you a quick primer, we can switch to a different topic you're more comfortable with, or we can wrap up the interview. What would you prefer?"
            
            elif intent == "TOPIC_SWITCH":
                log.info("Topic switch request detected", extra={
                    "intent": intent,
                    "confidence": confidence,
                    "session_id": session_id,
                    "reasoning": reasoning,
                    "new_topic": new_topic_from_llm
                })
                
                # Fast-path: Try regex for explicit "switch to X" format (optional optimization)
                import re
                topic_match = re.search(r'switch\s+to\s+([^\s]+(?:\s+[^\s]+)*)', answer, re.IGNORECASE)
                if topic_match:
                    new_topic = topic_match.group(1).strip()
                    interview_req.interview_topic = new_topic
                    conversation_context.topic_switched = True
                    log.info("Topic switched via regex fast-path", extra={
                        "session_id": session_id,
                        "new_topic": new_topic,
                        "method": "regex"
                    })
                    return f"Great! Let's switch to {new_topic}. Can you tell me about your experience with {new_topic}?"
                
                # Primary path: Use topic extracted from LLM intent detection (single call optimization)
                if new_topic_from_llm and new_topic_from_llm.strip() and new_topic_from_llm.lower() != "null":
                    new_topic = new_topic_from_llm.strip()
                    interview_req.interview_topic = new_topic
                    conversation_context.topic_switched = True
                    log.info("Topic switched via LLM extraction", extra={
                        "session_id": session_id,
                        "new_topic": new_topic,
                        "method": "llm"
                    })
                    return f"Sure! Let's switch to {new_topic}. Can you tell me about your experience with {new_topic}?"
                else:
                    # User wants to switch but hasn't specified the topic
                    conversation_context.topic_switched = True
                    log.info("Topic switch requested but no specific topic mentioned", extra={
                        "session_id": session_id
                    })
                    return "Of course! What topic would you like to discuss instead? Feel free to mention any technology, skill, or area you'd like to explore."
            
            # Intent is NONE or unrecognized
            return None
            
        except Exception as e:
            log.error("Error in LLM-based intent detection", extra={
                "error": str(e)
            }, exc_info=True)
            return None
    
    # Fallback function with basic pattern matching for fast path or LLM failure
    def _detect_and_handle_edge_intents_fallback(answer: str, topic: Optional[str]) -> Optional[str]:
        """
        Fast fallback with basic pattern matching for critical edge cases.
        Used when LLM detection fails or for very short responses.
        Enhanced to detect uncooperative patterns.
        """
        try:
            if not isinstance(answer, str):
                return None
            a = answer.strip().lower()
            if not a:
                return None
            
            word_count = len(a.split())
            
            # NOTE: Exit intent, refusal patterns, and uncertainty detection are handled by LLM
            # for better semantic understanding. Only very obvious single-word negatives are
            # pre-filtered here for performance optimization.
            
            # Only pre-filter very obvious single-word negatives (1-2 words) for performance
            if word_count <= 2:
                words_set = set(a.split())
                if words_set.intersection(NEGATIVE_SINGLE_WORDS):
                    sid = conversation_context.session_id if hasattr(conversation_context, "session_id") else None
                    if sid:
                        track_intent(sid, "LACK_OF_KNOWLEDGE", 0.8, "Single word negative detected in fallback")
                    if sid and should_offer_loop_breaker(sid):
                        return get_loop_breaker_offer(topic)
                    if topic:
                        return f"No problem! We can handle this a few ways: I can give you a quick primer on {topic}, we can switch to a different topic you're more comfortable with, or we can wrap up the interview. What would you prefer?"
                    return "No problem! We can handle this a few ways: I can give you a quick primer, we can switch to a different topic you're more comfortable with, or we can wrap up the interview. What would you prefer?"
            
            return None
        except Exception as e:
            log.error("Error in fallback intent detection", extra={
                "error": str(e)
            }, exc_info=True)
            return None
    
    # Check if this is a topic-focused interview
    if interview_req.interview_topic and interview_req.structured_resume:
        # Use topic-focused question generation
        try:
            # Graceful early handling if user reply suggests exit/switch/unknown (LLM-based)
            if interview_req.answer and interview_req.answer.strip():
                early = await _detect_and_handle_edge_intents_llm(
                    interview_req.answer, 
                    interview_req.interview_topic, 
                    anonymized_history
                )
                # Fallback to pattern matching if LLM detection fails
                if not early:
                    early = _detect_and_handle_edge_intents_fallback(
                        interview_req.answer, 
                        interview_req.interview_topic
                    )
                if early:
                    return early, {"type": "general", "engagement_score": 0.5}, candidate_info, get_previous_questions(anonymized_history)
            # Determine current state based on question count
            if question_count == 0:
                current_state = InterviewState.TOPIC_INTRODUCTION
            elif question_count < 3:
                current_state = InterviewState.TOPIC_FUNDAMENTALS
            elif question_count < 8:
                current_state = InterviewState.TOPIC_PROBLEM_SOLVING
            elif question_count < 12:
                current_state = InterviewState.TOPIC_DEEP_DIVE
            else:
                current_state = InterviewState.TOPIC_FEEDBACK
            
            # Analyze response if available
            if interview_req.answer.strip():
                is_first_request = question_count == 0
                response_analysis = await hybrid_response_analyzer.analyze_response(
                    interview_req.answer,
                    anonymized_history,
                    candidate_info,
                    safe_job_details,
                    is_first_request,
                    uid_context
                )
            else:
                response_analysis = {
                    "type": "general",
                    "confidence": 0.5,
                    "engagement_score": 0.5,
                    "keywords": [],
                    "sentiment": "neutral"
                }
            
            # Generate topic-focused question
            next_question = await generate_adaptive_question(
                candidate_info=candidate_info,
                persona=interview_req.persona,
                current_state=current_state,
                question_count=question_count,
                conversation_history=anonymized_history,
                job_details=safe_job_details,
                response_analysis=response_analysis,
                names=[],  # names will be extracted from anonymizer if needed
                interview_topic=interview_req.interview_topic,
                structured_resume=interview_req.structured_resume,
                session_id=getattr(conversation_context, "session_id", None),
            )
            
            # Prepare enhanced candidate info
            enhanced_candidate_info = {
                **candidate_info,
                "skills_analysis": uid_context.get("skills_analysis", {}) if uid_context else {},
                "assessment_results": uid_context.get("assessment_results", {}) if uid_context else {},
            }
            
            # Get previous questions
            previous_questions = get_previous_questions(anonymized_history)
            
            log.debug(f"Generated topic-focused question: {next_question[:100]}...")
            
        except Exception as e:
            log.error(f"Topic-focused question generation failed: {e}")
            import traceback
            log.debug(traceback.format_exc())
            # Fallback to default question
            next_question = get_default_question(current_state if 'current_state' in locals() else InterviewState.TOPIC_INTRODUCTION, interview_req.interview_topic)
            response_analysis = response_analysis if response_analysis else {"type": "general", "engagement_score": 0.5}
            enhanced_candidate_info = candidate_info
            previous_questions = []
    else:
        # Use old single call architecture for non-topic-focused interviews
        try:
            # Graceful early handling if user reply suggests exit/switch/unknown (LLM-based, for non-topic interviews too)
            if interview_req.answer and interview_req.answer.strip():
                early = await _detect_and_handle_edge_intents_llm(
                    interview_req.answer, 
                    interview_req.interview_topic, 
                    anonymized_history
                )
                # Fallback to pattern matching if LLM detection fails
                if not early:
                    early = _detect_and_handle_edge_intents_fallback(
                        interview_req.answer, 
                        interview_req.interview_topic
                    )
                if early:
                    return early, {"type": "general", "engagement_score": 0.5}, candidate_info, get_previous_questions(anonymized_history)
            
            # Use single call for analysis and question generation
            single_call_result = await asyncio.wait_for(
                generate_question_single_call(
                    candidate_info=candidate_info,
                    conversation_history=anonymized_history,
                    job_details=safe_job_details,
                    uid_context=uid_context,
                    question_count=question_count,
                    response_analysis=None,  # Will be generated in single call
                    interview_stage=conversation_context.interview_stage,
                    adaptive_follow_up=conversation_context.adaptive_follow_up_type,
                    technical_depth=conversation_context.technical_depth_level
                ),
                timeout=8.0  # Much faster timeout
            )
            
            # Extract results from single call
            response_analysis = single_call_result.response_analysis
            next_question = single_call_result.question
            
            # Prepare enhanced candidate info (simplified for single call)
            enhanced_candidate_info = {
                **candidate_info,
                "skills_analysis": uid_context.get("skills_analysis", {}) if uid_context else {},
                "assessment_results": uid_context.get("assessment_results", {}) if uid_context else {},
                "market_insights": uid_context.get("market_and_course_recommender", {}).get("market_insights", []) if uid_context and uid_context.get("market_and_course_recommender") else [],
                "course_recommendations": uid_context.get("market_and_course_recommender", {}).get("course_recommendations", []) if uid_context and uid_context.get("market_and_course_recommender") else []
            }
            
            # Get previous questions from conversation history
            log.debug("Interview: deriving previous_questions from conversation history and/or Chroma snapshot")
            previous_questions = get_previous_questions(anonymized_history)
            
            # Apply guardrails to the generated question to prevent repetition
            session_id = conversation_context.session_id if hasattr(conversation_context, 'session_id') else None
            next_question = await enforce_question_guardrails(
                next_question,
                previous_questions,
                conversation_context,
                candidate_info,
                uid_context,
                session_id=session_id
            )
            
            log.debug(f"Generated question: {next_question[:100]}...")
            
        except asyncio.TimeoutError:
            log.warning("Single call timed out, using improved fallback")
            
            # Use improved fallback analysis
            is_first_request = question_count == 0 and not interview_req.answer.strip()
            response_analysis = await IntelligentResponseAnalyzer._fallback_analysis(
                interview_req.answer, is_first_request
            )
            enhanced_candidate_info = candidate_info
            previous_questions = []
            next_question = await get_natural_fallback_question(
                ConversationContext(), candidate_info, uid_context=uid_context
            )
        
    return next_question, response_analysis, enhanced_candidate_info, previous_questions


async def _update_stage_and_select_scenario(
    conversation_context: ConversationContext,
    question_count: int,
    response_analysis: Dict[str, Any],
    uid_context: Dict,
    safe_job_details: Dict,
    next_question: str
) -> str:
    """Updates interview stage and optionally selects scenario question."""
    # Extract role level and years of experience for role-based thresholds
    role_level = None
    years_experience = None
    try:
        # Try to get from candidate info
        candidate_info = uid_context.get("candidate_info", {}) if uid_context else {}
        years_experience_str = candidate_info.get("years_experience") or uid_context.get("years_experience")
        if years_experience_str:
            try:
                years_experience = int(str(years_experience_str).split()[0]) if isinstance(years_experience_str, str) else int(years_experience_str)
            except (ValueError, AttributeError):
                pass
    except Exception:
        pass
    
    # Update stage now that response_analysis is available (enhanced with role-based thresholds)
    conversation_context.update_stage(question_count, response_analysis, role_level, years_experience)
    scenario_question = None

    if conversation_context.interview_stage in ["problem_solving", "behavioral"]:
        role_result = await classify_role_hybrid(uid_context, safe_job_details) if uid_context else RoleClassificationResult(
            role_category="TECHNICAL_CODING",
            confidence_score=0.5,
            reasoning="No UID context available - using default classification",
            matched_criteria=[],
            alternative_roles=[],
            skill_analysis={},
            experience_analysis={}
        )
        scenario_question = await _rbi(
            interview_chroma.get_scenario_question,
            role_category=role_result.role_category,
            stage=conversation_context.interview_stage,
            difficulty=conversation_context.technical_depth_level,
            previous_scenario_ids=conversation_context.used_scenario_ids
        )
        if scenario_question:
            conversation_context.used_scenario_ids.append(scenario_question["id"])
            next_question = scenario_question["question_text"]

    return next_question


def _rebuild_contextual_system_message(
    interview_req: InterviewRequest,
    uid_context: Dict,
    candidate_info: Dict,
    safe_job_details: Dict,
    names: List[str],
    conversation_context: ConversationContext,
    question_count: int,
    response_analysis: Dict[str, Any],
    session_id: str
) -> None:
    """Rebuilds contextual system message for later turns."""
    try:
        candidate_name_ctx = get_candidate_name(uid_context, candidate_info, names)
        job_title_ctx = get_job_title(uid_context, safe_job_details, candidate_info)
        skills_str_ctx = get_candidate_skills(uid_context, candidate_info)
        interests_ctx = uid_context.get("user_interests", {}).get("user_interests", []) if uid_context and uid_context.get("user_interests") else []
        topics_ctx = conversation_context.topics_discussed
        engagement_ctx = conversation_context.engagement_level
        question_count_ctx = question_count
        # Build/Update facts memory
        existing_snapshot = interview_chroma.get_interview_session(session_id) or {}
        old_facts = existing_snapshot.get("facts_memory", {})
        facts_memory = update_facts_memory(old_facts, uid_context, candidate_info, response_analysis)
        facts_summary_ctx = summarize_facts_memory(facts_memory)
        rebuilt_system = build_contextual_system_message(
            uid_context,
            candidate_name_ctx,
            job_title_ctx,
            skills_str_ctx,
            interests_ctx,
            topics_ctx,
            engagement_ctx,
            question_count_ctx,
            derive_role_micro_guidelines(job_title_ctx),
            facts_summary_ctx
        )
        if interview_req.conversation_history and interview_req.conversation_history[0].get("role") == "system":
            interview_req.conversation_history[0]["content"] = rebuilt_system
        else:
            interview_req.conversation_history.insert(0, {"role": "system", "content": rebuilt_system})
    except Exception as _e:
        # Non-fatal: keep existing system message
        pass


def validate_evaluation_inputs(
    conversation_context: ConversationContext,
    role_category: str,
    interview_topic: str = None,
    question_count: int = 0,
    conversation_history: List[Dict] = None
) -> Tuple[bool, List[str]]:
    """
    Validate inputs before final evaluation to catch mismatches and missing context.
    Context-aware: More lenient for early-stage interviews, stricter for final evaluations.
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    # Determine if this is an early-stage interview (first question or just started)
    is_early_stage = question_count == 0 or (conversation_history and len(conversation_history) <= 2)
    
    # Check 1: Question count validation (only strict if not early stage)
    if question_count == 0 and not is_early_stage:
        issues.append("No questions tracked in interview history")
    elif question_count == 0 and is_early_stage:
        # Early stage - this is expected, just log as info, not an issue
        pass
    
    # Check 2: Conversation history validation
    if not conversation_history or len(conversation_history) < 2:
        if not is_early_stage:
            issues.append("Insufficient conversation history (need at least 1 Q&A pair)")
        # Early stage with just system message is OK
    
    # Check 3: Role-topic mismatch detection (always check, but only warn if not early stage)
    if interview_topic and role_category:
        topic_lower = interview_topic.lower()
        role_lower = role_category.lower()
        
        # Detect obvious mismatches
        if "fashion" in topic_lower and "technical" in role_lower and "coding" in role_lower:
            issues.append(f"Role-Topic Mismatch: Interview topic '{interview_topic}' doesn't match role '{role_category}'")
        elif "illustration" in topic_lower and "technical" in role_lower and "coding" in role_lower:
            issues.append(f"Role-Topic Mismatch: Interview topic '{interview_topic}' doesn't match role '{role_category}'")
        elif "design" in topic_lower and "technical" in role_lower and "coding" in role_lower:
            issues.append(f"Role-Topic Mismatch: Interview topic '{interview_topic}' doesn't match role '{role_category}'")
    
    # Check 4: Topics discussed validation (only strict if not early stage)
    if not conversation_context.topics_discussed or len(conversation_context.topics_discussed) == 0:
        if not is_early_stage and question_count > 2:
            # Only flag if we've asked multiple questions but still no topics
            issues.append("No topics tracked during interview")
        # Early stage - topics will be populated as conversation progresses
    
    is_valid = len(issues) == 0
    return is_valid, issues


async def _generate_llm_interview_report(
    interview_summary: Dict[str, Any],
    candidate_name: str,
    job_title: str,
    topic_scores: Dict[str, Dict[str, Any]] = None,
    interview_topic: Optional[str] = None,
    conversation_context: ConversationContext = None
) -> str:
    """
    Generate an intelligent, contextual interview evaluation report using LLM.
    
    Args:
        interview_summary: Structured interview evaluation data
        candidate_name: Candidate's name
        job_title: Job position title
        topic_scores: Per-topic evaluation scores (if topics were switched)
        interview_topic: Primary interview topic (if topic-focused)
        conversation_context: Conversation context for additional insights
    
    Returns:
        Formatted interview evaluation report as string
    """
    try:
        # Build comprehensive prompt for LLM
        overall_score = interview_summary.get('overall_score', 0.0)
        status = interview_summary.get('status_description', 'Unknown')
        criteria_scores = interview_summary.get('criteria_scores', {})
        green_flags = interview_summary.get('green_flags', [])
        red_flags = interview_summary.get('red_flags', [])
        recommendation = interview_summary.get('recommendation', 'No recommendation available')
        questions_asked = interview_summary.get('questions_asked', 0)
        technical_skills = interview_summary.get('technical_skills_breakdown', {})
        
        # Format criteria scores for prompt
        criteria_text = []
        for cat, data in criteria_scores.items():
            if isinstance(data, dict):
                score = data.get('score', 0.0)
                max_score = data.get('max_score', 10.0)
                cat_name = cat.replace('_', ' ').title()
                criteria_text.append(f"{cat_name}: {score:.1f}/{max_score:.1f}")
        
        criteria_str = '\n'.join(f"- {c}" for c in criteria_text) if criteria_text else "No criteria scores available"
        
        # Format topic scores if available
        topic_analysis = ""
        if topic_scores and len(topic_scores) > 1:
            topic_analysis = "\n\nPER-TOPIC PERFORMANCE:\n"
            for topic, scores in topic_scores.items():
                topic_score = scores.get('overall_score', 0.0)
                topic_status = scores.get('status', 'Unknown')
                topic_q_count = scores.get('question_count', 0)
                topic_analysis += f"- {topic}: {topic_score:.1f}/10 ({topic_status}) - {topic_q_count} questions\n"
        
        # Build context about interview
        interview_context = ""
        if conversation_context:
            topics_discussed = getattr(conversation_context, 'topics_discussed', [])
            engagement = getattr(conversation_context, 'engagement_level', 0.5)
            if topics_discussed:
                interview_context = f"\nTopics discussed: {', '.join(topics_discussed[:5])}\n"
            interview_context += f"Engagement level: {engagement:.2f}/1.0\n"
        
        # Build LLM prompt
        report_prompt = f"""You are an expert HR professional generating a comprehensive interview evaluation report.

CANDIDATE INFORMATION:
- Name: {candidate_name}
- Position: {job_title}
{f"- Primary Interview Topic: {interview_topic}" if interview_topic else ""}

EVALUATION RESULTS:
- Overall Score: {overall_score:.1f}/10
- Status: {status}
- Questions Asked: {questions_asked}

CRITERIA BREAKDOWN:
{criteria_str}
{topic_analysis if topic_analysis else ""}

POSITIVE INDICATORS (Green Flags):
{chr(10).join(f"- {flag}" for flag in green_flags) if green_flags else "- None identified"}

CONCERNS (Red Flags):
{chr(10).join(f"- {flag}" for flag in red_flags) if red_flags else "- None identified"}

TECHNICAL SKILLS ASSESSED:
{chr(10).join(f"- {skill}: {score}/10" for skill, score in list(technical_skills.items())[:10]) if technical_skills else "- No specific technical skills assessed"}
{interview_context}

Generate a professional, comprehensive interview evaluation report that:
1. Provides a clear executive summary of the candidate's performance
2. Highlights key strengths and achievements
3. Addresses areas of concern or improvement
4. Provides context-aware insights (especially if multiple topics were discussed)
5. Includes specific examples from the evaluation data
6. Maintains a professional, constructive tone
7. Is concise but comprehensive (approximately 300-500 words)

Format the report with clear sections:
- EXECUTIVE SUMMARY
- KEY STRENGTHS
- AREAS FOR IMPROVEMENT
- TECHNICAL ASSESSMENT (if applicable)
- RECOMMENDATION

IMPORTANT: 
- Use the actual scores and data provided
- If multiple topics were discussed, provide insights on performance across topics
- Be specific and actionable in feedback
- Maintain objectivity and fairness
"""

        # Call LLM to generate report
        llm_response = await invoke_llm(
            prompt=report_prompt,
            task_type="interview",
            agent_name="interview_report_generator"
        )
        
        # Extract content from response
        report_content = getattr(llm_response, "content", str(llm_response)).strip()
        
        # Add header with key metrics
        header = f"""
{'='*80}
INTERVIEW EVALUATION REPORT
{'='*80}
CANDIDATE: {candidate_name}
POSITION: {job_title}
OVERALL SCORE: {overall_score:.1f}/10 | STATUS: {status}
QUESTIONS ASKED: {questions_asked}
{'='*80}

"""
        
        return header + report_content
        
    except Exception as e:
        log.warning(f"LLM report generation failed: {e}")
        log.info("Falling back to template-based report...")
        
        # Fallback to template-based report (using variables from outer scope)
        try:
            overall_score = interview_summary.get('overall_score', 0.0)
            status = interview_summary.get('status_description', 'Unknown')
            criteria_scores = interview_summary.get('criteria_scores', {})
            green_flags = interview_summary.get('green_flags', [])
            red_flags = interview_summary.get('red_flags', [])
            recommendation = interview_summary.get('recommendation', 'No recommendation available')
            questions_asked = interview_summary.get('questions_asked', 0)
            
            def format_criteria_name(key: str) -> str:
                """Convert snake_case to Title Case"""
                return key.replace('_', ' ').title()
            
            criteria_breakdown = []
            for criteria_key, criteria_data in criteria_scores.items():
                if isinstance(criteria_data, dict) and 'score' in criteria_data:
                    score = criteria_data.get('score', 0.0)
                    criteria_name = format_criteria_name(criteria_key)
                    criteria_breakdown.append(f"- {criteria_name}: {score:.1f}/10")
            
            breakdown_text = '\n'.join(criteria_breakdown) if criteria_breakdown else "- No criteria scores available"
            
            return f"""
INTERVIEW EVALUATION SUMMARY

CANDIDATE: {candidate_name}
POSITION: {job_title}
OVERALL SCORE: {overall_score:.1f}/10
STATUS: {status}
EVALUATION BREAKDOWN:
{breakdown_text}

GREEN FLAGS:
{chr(10).join(f"- {flag}" for flag in green_flags) if green_flags else "- None identified"}

RED FLAGS:
{chr(10).join(f"- {flag}" for flag in red_flags) if red_flags else "- None identified"}

RECOMMENDATION: {recommendation}

QUESTIONS ASKED: {questions_asked}
"""
        except Exception as fallback_error:
            log.error(f"Fallback report generation also failed: {fallback_error}")
            return f"""
INTERVIEW EVALUATION SUMMARY

CANDIDATE: {candidate_name}
POSITION: {job_title}
OVERALL SCORE: {interview_summary.get('overall_score', 0.0):.1f}/10
STATUS: {interview_summary.get('status_description', 'Unknown')}
QUESTIONS ASKED: {interview_summary.get('questions_asked', 0)}

Note: Detailed breakdown unavailable due to report generation error.
"""


async def _conclude_interview(
    interview_req: InterviewRequest,
    interview_evaluator: InterviewEvaluator,
    question_count: int,
    uid_context: Dict,
    candidate_info: Dict,
    safe_job_details: Dict,
    names: List[str],
    conversation_context: ConversationContext,
    response_analysis: Dict[str, Any],
    session_id: str,
    start_time: float,
    anonymizer: PIIAnonymizer
) -> InterviewResponse:
    """Handles interview conclusion, summary generation, and callback."""
    
    # PRE-EVALUATION VALIDATION: Catch role/topic mismatches and missing context
    try:
        # Get role classification for validation
        role_result = await classify_role_hybrid(uid_context, safe_job_details) if uid_context else None
        role_category = role_result.role_category if role_result else "UNKNOWN"
        role_confidence = role_result.confidence_score if role_result else 0.0
        
        # CONFIDENCE THRESHOLD CHECK: Warn if confidence is too low
        CONFIDENCE_THRESHOLD = 0.6  # Configurable threshold
        if role_confidence < CONFIDENCE_THRESHOLD:
            log.warning(f"Low Role Classification Confidence: {role_confidence:.2f} < {CONFIDENCE_THRESHOLD}")
            log.warning(f"   Role: {role_category}")
            log.warning(f"   Reasoning: {role_result.reasoning if role_result else 'N/A'}")
            log.warning(f"   This may indicate an unusual professional domain or insufficient resume data.")
            
            # If confidence is VERY low AND there's an interview topic, regenerate rubric via LLM
            if role_confidence < 0.4 and interview_req.interview_topic:
                log.info(f"Very low confidence ({role_confidence:.2f}) - Using LLM to generate domain-specific rubric for '{interview_req.interview_topic}'")
                try:
                    # Generate custom rubric for the interview topic
                    llm_rubric = await generate_evaluation_rubric_llm(
                        role_category=role_category,
                        interview_topic=interview_req.interview_topic,
                        job_description=safe_job_details.get("job_description", "") if safe_job_details else None
                    )
                    log.info(f"Generated LLM-driven rubric with {len(llm_rubric.get('categories', {}))} categories")
                    # Apply the LLM-generated rubric to the evaluator
                    interview_evaluator.apply_llm_rubric(llm_rubric)
                except Exception as rubric_error:
                    log.error(f"Failed to generate LLM rubric: {rubric_error}")
        
        is_valid, validation_issues = validate_evaluation_inputs(
            conversation_context=conversation_context,
            role_category=role_category,
            interview_topic=interview_req.interview_topic,
            question_count=question_count,
            conversation_history=interview_req.conversation_history
        )
        
        if not is_valid:
            # Determine if this is early stage (less critical warnings)
            is_early_stage = question_count == 0 or (interview_req.conversation_history and len(interview_req.conversation_history) <= 2)
            
            if is_early_stage:
                # Early stage - log as info, not critical warning
                log.info(f"📋 Pre-Evaluation Check (Early Stage):")
                for issue in validation_issues:
                    log.info(f"   - {issue} (expected for first question)")
            else:
                # Later stage - these are actual issues
                log.warning(f"🚨 Pre-Evaluation Validation Failed:")
                for issue in validation_issues:
                    log.warning(f"   - {issue}")
            
            # Log detailed diagnostic info
            if is_early_stage:
                log.info(f"   Diagnostic Info:")
                log.info(f"     - Role Category: {role_category} (confidence: {role_confidence:.2f})")
                log.info(f"     - Interview Topic: {interview_req.interview_topic}")
                log.info(f"     - Question Count: {question_count}")
                log.info(f"     - Conversation History Length: {len(interview_req.conversation_history)}")
                log.info(f"     - Topics Discussed: {conversation_context.topics_discussed}")
            else:
                log.warning(f"   Diagnostic Info:")
                log.warning(f"     - Role Category: {role_category} (confidence: {role_confidence:.2f})")
                log.warning(f"     - Interview Topic: {interview_req.interview_topic}")
                log.warning(f"     - Question Count: {question_count}")
                log.warning(f"     - Conversation History Length: {len(interview_req.conversation_history)}")
                log.warning(f"     - Topics Discussed: {conversation_context.topics_discussed}")
            
            # If role-topic mismatch detected, try to regenerate evaluation with LLM-driven approach
            if any("Role-Topic Mismatch" in issue for issue in validation_issues):
                log.info(f"Attempting to regenerate evaluation with LLM-driven approach...")
                try:
                    # Generate custom rubric for the interview topic
                    llm_rubric = await generate_evaluation_rubric_llm(
                        role_category=interview_req.interview_topic,  # Use topic as role if mismatched
                        interview_topic=interview_req.interview_topic,
                        job_description=safe_job_details.get("job_description", "") if safe_job_details else None
                    )
                    log.info(f"Generated corrective LLM rubric with {len(llm_rubric.get('categories', {}))} categories")
                    # Apply the corrective LLM-generated rubric to fix mismatched evaluation
                    interview_evaluator.apply_llm_rubric(llm_rubric)
                except Exception as rubric_error:
                    log.error(f"Failed to generate corrective LLM rubric: {rubric_error}")
    except Exception as validation_error:
        log.error(f"Error during pre-evaluation validation: {validation_error}")
    
    # Generate comprehensive interview evaluation
    interview_summary = interview_evaluator.get_interview_summary(question_count)

    # Build detailed final summary before any callback usage
    candidate_name = get_candidate_name(uid_context, candidate_info, names)
    job_title = get_job_title(uid_context, safe_job_details, candidate_info)
    
    # Get per-topic scores from summary (already included if multiple topics)
    topic_scores = interview_summary.get('topic_scores', {})
    
    # Generate LLM-based detailed summary
    detailed_summary = await _generate_llm_interview_report(
        interview_summary=interview_summary,
        candidate_name=candidate_name,
        job_title=job_title,
        topic_scores=topic_scores,
        interview_topic=interview_req.interview_topic,
        conversation_context=conversation_context
    )

    # Print completed interview summary to terminal
    log.info(f"\n{'='*80}")
    log.info(f"✅ INTERVIEW COMPLETED ✅")
    log.info(f"{'='*80}")
    log.info(f"Final Summary:")
    log.info(f"{detailed_summary}")
    log.info(f"{'='*80}\n")

    # Send callback notification if callback URL is provided
    if interview_req.callback_url:
        try:
            try:
                log.info(f"CALLBACK url={interview_req.callback_url} status=completed sent_keys={list(interview_summary.keys())}")
            except Exception:
                pass
            await send_callback_notification(
                callback_url=interview_req.callback_url,
                uid=interview_req.uid,
                session_id=session_id,
                status="completed",
                evaluation_summary=interview_summary,
                detailed_summary=detailed_summary,
                auth_token=interview_req.auth_token,
                conversation_context={
                    "topics_discussed": conversation_context.topics_discussed,
                    "engagement_level": conversation_context.engagement_level,
                    "interview_phase": conversation_context.interview_phase
                },
                response_analysis=response_analysis,
                session_metadata={
                    "session_id": session_id,
                    "duration_minutes": (time.time() - start_time) / 60,
                    "anonymization_summary": anonymizer.get_anonymization_summary(),
                }
            )
        except Exception as e:
            log.error(f"Error sending evaluation callback to {interview_req.callback_url}: {e}")
    
    # Build final response
    final_response = InterviewResponse(
        conversation_history=interview_req.conversation_history,
        agent_message="Thank you for your time. We will be in touch with the next steps.",
        status="completed",
        current_state=InterviewState.COMPLETED,
        question_count=question_count
    )
    
    # Log final output response
    log.info(f"Interview completed - Status: {final_response.status}, State: {final_response.current_state}, Questions: {final_response.question_count}")
    
    return final_response


# -------------------------
# Intelligent Free-Flowing Interview Agent
# -------------------------
async def ai_interview_agent_intelligent(request: Request) -> InterviewResponse:
    """Intelligent, context-aware interview agent with natural conversation flow"""
    try:
        # Parse and validate request using helper function
        interview_req = await _parse_and_validate_request(request)
        
        # Build context data using helper function
        uid_context, candidate_info, safe_job_details, names, companies, session_id, start_time, anonymizer = _build_interview_context_data(interview_req)
        
        # Get resume_data for backward compatibility
        resume_data, job_details, names, companies = process_uid_context(uid_context, interview_req)
        
        # Rely on client-provided question_count, or default to 0
        server_question_count = interview_req.question_count or 0
        
        # Initialize conversation history using helper function
        await _initialize_conversation_history(interview_req, uid_context, candidate_info, safe_job_details, names)
        
        # Anonymize and prepare history using helper function
        anonymized_history = _anonymize_and_prepare_history(interview_req, anonymizer, names, companies)
        
        # Calculate question count using helper function
        question_count = _calculate_question_count(interview_req, server_question_count, anonymized_history)
        
        # Initialize conversation context using helper function
        conversation_context = _initialize_conversation_context(uid_context)
        
        # Recover session state from ChromaDB if available (for session continuity)
        if session_id:
            try:
                # Recover in-memory state (fingerprints, intents, fallbacks, circuit breaker)
                await _rbi(_recover_session_state, session_id)
                
                # Recover conversation context if this is a continuing session
                if question_count > 0:
                    await _rbi(_recover_conversation_context, session_id, conversation_context)
                    log.info(f"Recovered state for continuing session: {session_id} (question_count={question_count})")
            except Exception as e:
                log.warning(f"Failed to recover session state: {e}")
        
        # Set session_id in conversation_context for persistence
        if hasattr(conversation_context, 'session_id'):
            conversation_context.session_id = session_id
        
        # Set up interview evaluator using helper function (with async role classification)
        interview_evaluator = await _setup_interview_evaluator(safe_job_details, uid_context)
        # Apply stronger topic weighting if topic-focused interview
        if interview_req.interview_topic:
            try:
                interview_evaluator.apply_topic_weighting(interview_req.interview_topic)
                log.debug(f"Applied stronger topic weighting for topic: {interview_req.interview_topic}")
            except Exception:
                pass
        
        # Get previous questions to check if this is the first question
        previous_questions = get_previous_questions(anonymized_history)
        
        # Check if this is the first question (no previous questions and no answer)
        # If it's the first question, we'll use personalized opener instead of generating via _generate_next_question
        is_first_question = not previous_questions and not interview_req.answer.strip()
        
        if is_first_question:
            # First question - use personalized opener (skip _generate_next_question to avoid duplicate generation)
            # Set default values for response_analysis and enhanced_candidate_info
            response_analysis = {
                "type": "general",
                "confidence": 0.5,
                "engagement_score": 0.5,
                "keywords": [],
                "sentiment": "neutral"
            }
            enhanced_candidate_info = {
                **candidate_info,
                "skills_analysis": uid_context.get("skills_analysis", {}) if uid_context else {},
                "assessment_results": uid_context.get("assessment_results", {}) if uid_context else {},
            }
            # next_question will be set below in the personalized opener section
            next_question = None
        else:
            # Generate next question using helper function (for follow-up questions)
            next_question, response_analysis, enhanced_candidate_info, previous_questions = await _generate_next_question(
                candidate_info,
                anonymized_history,
                safe_job_details,
                uid_context,
                question_count,
                conversation_context,
                interview_req
            )
            
            # Check if exit intent was detected - if so, end interview immediately
            if conversation_context.should_end_interview(interview_req.end_interview):
                # Exit intent detected - conclude interview
                # Store the exit message to preserve it in the response
                exit_message = next_question if next_question else "Thank you for your time. We will be in touch with the next steps."
                
                # Add exit message to conversation history before concluding
                interview_req.conversation_history.append({
                    "role": "assistant",
                    "content": exit_message
                })
                
                # Conclude interview and use the exit message in the response
                final_response = await _conclude_interview(
                    interview_req,
                    interview_evaluator,
                    question_count,
                    uid_context,
                    candidate_info,
                    safe_job_details,
                    names,
                    conversation_context,
                    response_analysis,
                    session_id,
                    start_time,
                    anonymizer
                )
                
                # Override the agent_message with the exit message to preserve user's exit intent response
                final_response.agent_message = exit_message
                return final_response
        
        # Update stage and select scenario using helper function
        next_question = await _update_stage_and_select_scenario(
            conversation_context,
            question_count,
            response_analysis,
            uid_context,
            safe_job_details,
            next_question
        )
        
        # Ensure guardrails are applied after any scenario override
        try:
            previous_questions = get_previous_questions(anonymized_history)
            next_question = await enforce_question_guardrails(
                next_question,
                previous_questions,
                conversation_context,
                candidate_info,
                uid_context,
                session_id=session_id
            )
        except Exception:
            pass

        # Update conversation context
        await conversation_context.update_context(response_analysis, question_count)
        
        # Persist conversation context to ChromaDB (offloaded to thread pool)
        if session_id:
            await _rbi(_persist_conversation_context, session_id, conversation_context)
        
        # Rebuild contextual system message (contains ChromaDB read, offloaded)
        await _rbi(
            _rebuild_contextual_system_message,
            interview_req,
            uid_context,
            candidate_info,
            safe_job_details,
            names,
            conversation_context,
            question_count,
            response_analysis,
            session_id
        )

        # Determine if interview should end intelligently
        should_end = conversation_context.should_end_interview(interview_req.end_interview)
        log_event(
            "phase_update",
            session_id=session_id,
            phase=conversation_context.interview_phase,
            question_count=question_count,
            engagement=conversation_context.engagement_level,
            coverage=getattr(conversation_context, 'coverage', None)
        )
        if should_end:
            return await _conclude_interview(
                interview_req,
                interview_evaluator,
                question_count,
                uid_context,
                candidate_info,
                safe_job_details,
                names,
                conversation_context,
                response_analysis,
                session_id,
                start_time,
                anonymizer
            )
        
        # Generate intelligent, context-aware question using ChromaDB and UID context
        # previous_questions already prepared above
        
        # enhanced_candidate_info already prepared above
        
        # Check if this is the first question (no previous questions and no answer)
        if is_first_question:
            # First question - use personalized opener, with hard overrides for generic tracks
            topic = (interview_req.interview_topic or "").strip()
            topic_lower = topic.lower()
            psychometric_keywords = ["psychometric", "psychological", "behavioral", "aptitude", "personality"]
            communication_keywords = ["communication", "communication test", "communication skills", "speaking", "presentation", "listening"]
            
            if topic and any(kw in topic_lower for kw in psychometric_keywords):
                next_question = (
                    "Tell me about a time you had to make a tough decision with limited information. "
                    "What was your thought process and how did it turn out?"
                )
                log.info(f"First question override (psychometric/personality) for topic: {topic}")
            elif topic and any(kw in topic_lower for kw in communication_keywords):
                next_question = (
                    "Describe a time you had to explain a complex idea to someone unfamiliar with it. "
                    "How did you adapt your message and what was the outcome?"
                )
                log.info(f"First question override (communication) for topic: {topic}")
            elif interview_req.interview_topic and interview_req.structured_resume:
                # Topic-focused personalized opener
                next_question = get_topic_focused_opener(uid_context, interview_req.interview_topic, candidate_info, interview_req.structured_resume)
                log.info(f"First question (topic-focused): {interview_req.interview_topic}")
            else:
                # Generic personalized opener for non-topic-focused interviews
                job_title = get_job_title(uid_context, safe_job_details, candidate_info)
                next_question = get_personalized_opener(uid_context, job_title)
                log.info(f"First question (personalized opener) for role: {job_title}")

        # Safety net override: ensure first-turn topic-focused questions honor non-meta requirement
        if is_first_question and next_question:
            topic = (interview_req.interview_topic or "").strip().lower()
            if any(kw in topic for kw in ["psychometric", "psychological", "behavioral", "aptitude", "personality"]):
                next_question = (
                    "Tell me about a time you had to make a tough decision with limited information. "
                    "What was your thought process and how did it turn out?"
                )
                log.info("Safety override applied for psychometric/personality first question.")
            elif any(kw in topic for kw in ["communication", "communication test", "communication skills", "speaking", "presentation", "listening"]):
                next_question = (
                    "Describe a time you had to explain a complex idea to someone unfamiliar with it. "
                    "How did you adapt your message and what was the outcome?"
                )
                log.info("Safety override applied for communication first question.")
        else:
            # For topic-focused interviews, use the question already generated by _generate_next_question
            # (which uses generate_adaptive_question with interview_topic)
            if interview_req.interview_topic and interview_req.structured_resume:
                # Topic-focused question already generated - use it (don't override with generic questions)
                log.debug(f"Topic-focused question #{question_count + 1}: {interview_req.interview_topic}")
            else:
                # Circuit-breaker: if tripped for this session, serve role fallback; otherwise generate
                cb_info = _CB.get(session_id)
                if cb_info and cb_info.get("until_ts", 0) > time.time():
                    job_title = get_job_title(uid_context, safe_job_details, candidate_info)
                    next_question = await get_natural_fallback_question(
                        conversation_context,
                        candidate_info,
                        session_id,
                        uid_context
                    )
                    log.warning(f"Using natural fallback question for session {session_id}")
                    
                    log.warning(f"Circuit breaker active for session {session_id}, using fallback question")
                else:
                    # Generate contextual question based on the conversation
                    try:
                        start_t = time.time()
                        # Determine coverage hint: enforce strict rotation (max 2 consecutive same topic)
                        cov = getattr(conversation_context, 'coverage', {}) or {}
                        hint = None
                        if cov:
                            last_bins = getattr(conversation_context, 'last_bins', [])
                            
                            # Hard cap: if last 2 bins are the same, force rotation
                            if len(last_bins) >= 2 and last_bins[-1] == last_bins[-2]:
                                # Force rotation to any other bin
                                bins_sorted = sorted(cov.items(), key=lambda kv: kv[1])
                                for b, _ in bins_sorted:
                                    if b != last_bins[-1]:  # Different from last bin
                                        hint = b
                                        break
                            else:
                                # Normal rotation: prefer least-covered bins
                                bins_sorted = sorted(cov.items(), key=lambda kv: kv[1])
                                for b, _ in bins_sorted:
                                    if not last_bins or b != last_bins[-1]:
                                        hint = b
                                        break
                                hint = hint or bins_sorted[0][0]
                        
                        # Add natural follow-up logic based on response analysis
                        follow_up_hint = ""
                        if response_analysis.get("engagement_score", 0.5) < 0.4:
                            follow_up_hint = "Ask an engaging follow-up to increase participation"
                        elif response_analysis.get("word_count", 0) < 30:
                            follow_up_hint = "Ask for more specific details and examples"
                        elif response_analysis.get("type") == "technical" and response_analysis.get("confidence", 0) > 0.7:
                            follow_up_hint = "Dive deeper into technical implementation details"
                        elif response_analysis.get("type") == "behavioral":
                            follow_up_hint = "Explore decision-making process and outcomes"
                        
                        # Use follow-up hint if no coverage hint is available
                        if not hint and follow_up_hint:
                            hint = follow_up_hint
                        
                        # Use sophisticated follow-up logic for better questions
                        if interview_req.answer.strip():  # Not first question
                            # Build candidate context for sophisticated follow-up
                            candidate_context = {
                                "skills": enhanced_candidate_info.get("skills", "").split(", ") if enhanced_candidate_info.get("skills") else [],
                                "job_requirements": safe_job_details.get("requirements", []),
                                "experience_years": enhanced_candidate_info.get("experience_years", "0")
                            }
                            
                            # Build session context for conversation flow analysis
                            session_context = {
                                "topics_discussed": conversation_context.topics_discussed,
                                "question_count": question_count,
                                "coverage_bins": conversation_context.coverage
                            }
                            
                            # Generate sophisticated follow-up question with circuit breaker
                            start_sophisticated = time.time()
                            
                            if llm_circuit_breaker.can_execute():
                                try:
                                    sophisticated_question = generate_sophisticated_follow_up(
                                        response_analysis, candidate_context, session_context
                                    )
                                    llm_circuit_breaker.record_success()
                                    sophisticated_time = time.time() - start_sophisticated
                                    
                                    # Use sophisticated question if it's substantial
                                    if len(sophisticated_question) > 50:  # Substantial question
                                        next_question = sophisticated_question
                                        log.debug(f"Using sophisticated follow-up (took {sophisticated_time:.3f}s): {sophisticated_question[:100]}...")
                                    else:
                                        # Fallback to ChromaDB
                                        log.debug(f"Sophisticated question too short ({len(sophisticated_question)} chars), using ChromaDB fallback")
                                except Exception as e:
                                    llm_circuit_breaker.record_failure()
                                    log.warning(f"Sophisticated follow-up failed, using fast fallback: {e}")
                                    next_question = generate_fast_continuation_question(
                                        response_analysis.get("engagement_score", 0.5),
                                        response_analysis
                                    )
                            else:
                                log.warning("Circuit breaker open, using fast fallback")
                                next_question = generate_fast_continuation_question(
                                    response_analysis.get("engagement_score", 0.5),
                                    response_analysis
                                )
                            
                        # SINGLE CALL ARCHITECTURE - Question already generated above
                        # Apply guardrails to the generated question
                        next_question = await enforce_question_guardrails(
                            next_question,
                            previous_questions,
                            conversation_context,
                            candidate_info,
                            uid_context,
                            session_id=session_id
                        )
                        
                        log.debug(f"Single call question generated - Type: {response_analysis.get('type', 'general')}, Engagement: {response_analysis.get('engagement_score', 0.5):.2f}")
                        
                        # Reset circuit-breaker on success path
                        if session_id in _CB:
                            _CB.pop(session_id, None)
                    except asyncio.TimeoutError:
                        log.warning("Question generation timed out, using contextual fallback question")
                        log_event("llm_timeout", session_id=session_id)
                        
                        # Use context-aware fallback instead of hardcoded questions
                        next_question = await get_natural_fallback_question(conversation_context, candidate_info, session_id, uid_context)
                        
                        log.warning(f"LLM question generation timeout (20s) for session {session_id}, using fallback question")
                        
                        # Update circuit-breaker (timeout counts as failure)
                        info = _CB.get(session_id, {"failures": 0, "until_ts": 0})
                        info["failures"] = info.get("failures", 0) + 1
                        if info["failures"] >= 2:
                            info["until_ts"] = time.time() + 90  # 90s cooldown
                        _CB[session_id] = info
                        
                        # Persist circuit breaker state to ChromaDB (offloaded)
                        await _rbi(_persist_circuit_breaker, session_id, info)
        
        # Add the question to conversation history
        anonymized_history.append({"role": "assistant", "content": next_question})
        
        # De-anonymize the question for user display
        deanonymized_question = anonymizer.deanonymize_text(next_question)
        
        # Add to original (non-anonymized) history
        interview_req.conversation_history.append({
            "role": "assistant", 
            "content": deanonymized_question
        })
        
        # Persist server-side question snapshot and session data
        # Store question and update count
        stored_qid = f"q_{session_id}_{question_count + 1}"
        
        # Ensure question count is properly incremented
        new_q_count = question_count + 1
        log.debug(f"Question count: {question_count} -> {new_q_count}")
        session_data = {
            "candidate_info": candidate_info,
            "job_details": safe_job_details,
            "conversation_history": anonymized_history,
            "question_count": new_q_count,
            "status": "ongoing",
            "response_analysis": response_analysis,
            "conversation_context": {
                "topics_discussed": conversation_context.topics_discussed,
                "engagement_level": conversation_context.engagement_level,
                "interview_phase": conversation_context.interview_phase
            },
            "uid": interview_req.uid,
            "callback_url": interview_req.callback_url,
            "last_question": {"question_id": stored_qid, "question_text": next_question},
            "rolling_memory": build_rolling_memory(interview_req.conversation_history),
            "facts_memory": locals().get("facts_memory", {}),
        }
        # Build and store snapshot without external schema dependency
        try:
            # Convert conversation history to a safe, minimal list of dicts
            ch_msgs = []
            for m in interview_req.conversation_history:
                try:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    ch_msgs.append({"role": role, "content": f"{content}"})
                except Exception:
                    continue

            snap = {
                "session_id": session_id,
                "uid": interview_req.uid,
                "status": "ongoing",
                "question_count": new_q_count,
                "last_question": session_data.get("last_question"),
                "interview_phase": conversation_context.interview_phase,
                "conversation_history": ch_msgs,
                "conversation_context": session_data.get("conversation_context"),
                "rolling_memory": session_data.get("rolling_memory"),
                "facts_memory": session_data.get("facts_memory"),
                "response_analysis": response_analysis,
                "job_details": safe_job_details,
                "candidate_info": candidate_info,
                "callback_url": interview_req.callback_url,
            }
            await _rbi(interview_chroma.store_interview_session, session_id, snap)
        except Exception as e:
            log.error(f"Failed to store interview session snapshot: {e}")

        # Log interviewer response
        log.info(f"Question #{new_q_count} - Phase: {conversation_context.interview_phase}, Engagement: {conversation_context.engagement_level:.2f}")
        
        # Determine current state - use topic-focused states if interview_topic is provided
        if interview_req.interview_topic:
            # Use topic-focused state based on question count
            if question_count == 0:
                current_state = InterviewState.TOPIC_INTRODUCTION
            elif question_count < 3:
                current_state = InterviewState.TOPIC_FUNDAMENTALS
            elif question_count < 8:
                current_state = InterviewState.TOPIC_PROBLEM_SOLVING
            elif question_count < 12:
                current_state = InterviewState.TOPIC_DEEP_DIVE
            else:
                current_state = InterviewState.TOPIC_FEEDBACK
        else:
            # Fallback to old state mapping for non-topic-focused interviews
            current_state = _map_phase_to_state(conversation_context.interview_phase)
        
        # Build response
        response = InterviewResponse(
            conversation_history=interview_req.conversation_history,
            agent_message=deanonymized_question,
            status="ongoing",
            current_state=current_state,
            question_count=new_q_count,
            response_analysis=response_analysis,
            session_metadata={
                "session_id": session_id,
                "duration_minutes": (time.time() - start_time) / 60,
                "anonymization_summary": anonymizer.get_anonymization_summary(),
                "conversation_context": {
                    "topics_discussed": conversation_context.topics_discussed,
                    "engagement_level": conversation_context.engagement_level,
                    "interview_phase": conversation_context.interview_phase
                },
                "chroma_analytics": {
                    "question_analytics": {},
                    "session_stored": True
                },
                "last_question_id": stored_qid
            }
        )
        
        # Log output response
        log.info(f"Interview response - Status: {response.status}, State: {response.current_state}, Question #{response.question_count}, Duration: {response.session_metadata.get('duration_minutes', 0):.2f} min")
        
        return response
        
    except ValueError as e:
        log.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        # Log error details for debugging
        log.error("🚨 INTERVIEW AGENT ERROR 🚨")
        log.error(f"Error Type: {type(e).__name__}")
        log.error(f"Error Message: {str(e)}")
        log.error(f"Full Traceback:\n{error_details}")
        
        # Provide more specific error information
        error_message = f"Error: {str(e)}"
        if "timeout" in str(e).lower():
            error_message = "Request timed out. Please try again."
        elif "connection" in str(e).lower():
            error_message = "Connection error. Please check your network and try again."
        elif "validation" in str(e).lower():
            error_message = "Invalid request format. Please check your input and try again."
        elif "permission" in str(e).lower() or "unauthorized" in str(e).lower():
            error_message = "Authentication error. Please check your credentials."
        elif "unexpected keyword argument" in str(e).lower():
            error_message = f"Function signature error: {str(e)}"
        
        return InterviewResponse(
            conversation_history=interview_req.conversation_history if 'interview_req' in locals() else [],
            agent_message="Hello! Let's get started. Could you tell me about your background and what interests you about this role?",
            status="error",
            error=error_message
        )

# -------------------------
# Generate Interview Summary with De-anonymization
# -------------------------
async def generate_interview_summary(
    candidate_info: Dict[str, str],
    anonymized_history: List[Dict],
    anonymizer: PIIAnonymizer,
    original_name: str,
    question_count: int
) -> str:
    """Generate interview summary with anonymized data, then de-anonymize for final output"""
    # Use compressed history for summary to prevent token bloat
    compressed_history = _compress_conversation_history(anonymized_history[1:], max_recent_turns=8, max_message_length=500)
    
    summary_prompt = f"""
Based on this interview for a {candidate_info['roles']} position, provide a brief evaluation:

CONVERSATION HISTORY (compressed):
{json.dumps(compressed_history, indent=2)}

Provide a 4-6 sentence summary covering:
1. Overall impression of the candidate
2. Key strengths observed
3. Areas that need improvement or clarification
4. Recommendation (Strong Hire/Hire/Maybe/No Hire)

Keep the evaluation professional and constructive.
    """
    
    eval_conversation = [
        {"role": "system", "content": summary_prompt},
        {"role": "user", "content": "Generate the interview evaluation summary."}
    ]
    
    try:
        eval_response = await invoke_llm(
            prompt=eval_conversation,
            task_type="interview",
            agent_name="interview_agent"
        )
        summary = getattr(eval_response, "content", str(eval_response))
        # De-anonymize the summary for final output
        return anonymizer.deanonymize_text(summary)
    except Exception as e:
        log.error(f"Summary generation failed: {str(e)}")
        return f"Interview completed. {question_count} questions were asked. Please review the conversation history for detailed evaluation."

# ==================== PORTFOLIO / PROJECT LINK DIGESTER ====================
async def _fetch_text(session, url: str, timeout: float = 8.0) -> str:
    try:
        async with session.get(url, timeout=timeout) as resp:
            if resp.status != 200:
                return ""
            txt = await resp.text(errors="ignore")
            return txt or ""
    except Exception:
        return ""

def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    raw = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<style[\s\S]*?</style>", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw[:20000]

def _extract_tech_and_roles(text: str) -> Dict[str, List[str]]:
    text_l = text.lower()
    tech_keywords = [
        "python","javascript","typescript","react","vue","angular","node",
        "java","kotlin","swift","go","rust","django","flask","fastapi",
        "mysql","postgres","mongodb","redis","kafka","spark","hadoop",
        "aws","gcp","azure","docker","kubernetes","raspberry pi","esp32",
        "figma","adobe xd","accessibility","a11y","responsive","verilog","fpga"
    ]
    role_keywords = ["web designer","frontend","backend","full stack","embedded","firmware","data engineer","ml engineer","ui/ux","product designer"]
    found_tech = sorted({t for t in tech_keywords if t in text_l})[:20]
    found_roles = sorted({r for r in role_keywords if r in text_l})[:10]
    words = [w for w in re.split(r"[^a-z0-9+]+", text_l) if len(w) > 3]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    common = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:30]
    topics = [w for w, c in common if w not in tech_keywords][:10]
    return {"tech": found_tech, "roles": found_roles, "topics": topics}

async def digest_portfolio_links(urls: List[str]) -> Dict[str, Any]:
    if not urls:
        return {}
    try:
        import aiohttp
    except Exception:
        return {}
    results = []
    try:
        async with aiohttp.ClientSession() as sess:
            for url in urls[:3]:
                txt = await _fetch_text(sess, url)
                clean = _strip_html(txt)
                if clean:
                    meta = _extract_tech_and_roles(clean)
                    snippet = clean[:500]
                    results.append({"url": url, "snippet": snippet, **meta})
    except Exception:
        pass
    agg = {"urls": [r["url"] for r in results]}
    tech = set(); roles = set(); topics = set()
    for r in results:
        tech.update(r.get("tech", []))
        roles.update(r.get("roles", []))
        topics.update(r.get("topics", []))
    agg["tech"] = sorted(list(tech))[:20]
    agg["roles"] = sorted(list(roles))[:10]
    agg["topics"] = sorted(list(topics))[:15]
    parts = []
    if agg["roles"]:
        parts.append("Roles: " + ", ".join(agg["roles"]))
    if agg["tech"]:
        parts.append("Tech: " + ", ".join(agg["tech"][:8]))
    if agg["topics"]:
        parts.append("Topics: " + ", ".join(agg["topics"][:6]))
    agg["summary"] = " | ".join(parts)
    return agg

"""
state_manager.py

Clean, production-ready interview state machine for the LLM-first architecture.
- Topic-focused InterviewState enum only
- StateConfig dataclass for validated configs
- Simplified state transitions matching hybrid progression
- No dependency on core.config
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass

from .config import get_interview_state_config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ENUMS
# ---------------------------------------------------------------------------

class InterviewState(str, Enum):
    # Topic-focused states only (system is topic-focused only)
    TOPIC_INTRODUCTION = "topic_introduction"
    TOPIC_FUNDAMENTALS = "topic_fundamentals"
    TOPIC_DEEP_DIVE = "topic_deep_dive"
    TOPIC_FEEDBACK = "topic_feedback"
    COMPLETED = "completed"


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


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------------------------------

class InterviewRequest(BaseModel):
    uid: str
    callback_url: Optional[str] = None
    interview_topic: Optional[str] = Field(
        default=None,
        description="Interview topic (optional). If provided, generates topic-focused questions. If not provided, generates general interview questions."
    )
    structured_resume: Optional[Dict[str, Any]] = None
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
    
    # Multi-modal input support (optional, backward compatible)
    input_modality: Literal["text", "audio", "avatar"] = Field(
        default="text",
        description="Input modality: text (typed), audio (speech-to-text), or avatar (avatar-mediated)"
    )
    input_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata about input (duration, hesitation markers, confidence, etc.)"
    )
    interview_mode: Optional[Literal["practice", "hiring"]] = Field(
        default=None,
        description="Interview mode: practice (coaching) or hiring (evaluation)"
    )
    interviewer_type: Optional[Literal["ai", "human"]] = Field(
        default=None,
        description="Interviewer type: ai (AI interviewer) or human (human interviewer with AI copilot)"
    )
    generate_avatar: bool = Field(
        default=False,
        description="If true, generate avatar video for the question and return as base64 MP4 via SSE"
    )

    @validator("answer")
    def validate_answer(cls, v):
        if len(v) > 5000:
            raise ValueError("Answer too long")
        return v.strip()


class InterviewResponse(BaseModel):
    conversation_history: List[Dict[str, str]]
    question: str
    session_id: Optional[str] = None
    current_state: Optional[InterviewState] = None
    evaluation_summary: Optional[Dict[str, Any]] = None
    detailed_summary: Optional[str] = None
    session_metadata: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    question_count: Optional[int] = None
    final_summary: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None


class SingleCallQuestionResult(BaseModel):
    question: str
    response_analysis: Dict[str, Any]
    engagement_score: float = Field(ge=0.0, le=1.0)
    suggested_follow_ups: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    topics_mentioned: List[str]
    skills_demonstrated: List[str]
    strengths: List[str]
    areas_to_explore: List[str]


# ---------------------------------------------------------------------------
# STATE CONFIG
# ---------------------------------------------------------------------------

@dataclass
class StateConfig:
    max_questions: int = 1
    focus: str = ""
    next_states: List[InterviewState] = None
    timeout_minutes: int = 5

    def __post_init__(self):
        if self.next_states is None:
            self.next_states = []


def _normalize_key(k: str) -> str:
    return (k or "").strip().lower()


def build_state_configurations() -> Dict[InterviewState, StateConfig]:
    """
    Build validated state configurations from config file.
    """
    raw = get_interview_state_config() or {}

    # dynamic lookup: both "opening_rapport" and "OPENING_RAPPORT" should resolve
    enum_lookup = {e.value.lower(): e for e in InterviewState}
    enum_lookup.update({e.name.lower(): e for e in InterviewState})

    mapped: Dict[InterviewState, StateConfig] = {}

    for key, cfg in raw.items():
        normalized = _normalize_key(key)
        state = enum_lookup.get(normalized)
        if not state:
            log.debug(f"Unknown state config key: {key}")
            continue

        # Build next state list
        next_list = []
        for nxt in cfg.get("next_states", []):
            ns = enum_lookup.get(_normalize_key(nxt))
            if ns:
                next_list.append(ns)

        mapped[state] = StateConfig(
            max_questions=cfg.get("max_questions", 1),
            focus=cfg.get("focus", ""),
            next_states=next_list,
            timeout_minutes=cfg.get("timeout_minutes", 5),
        )

    return mapped


# cache
_STATE_CONFIG_CACHE: Optional[Dict[InterviewState, StateConfig]] = None


def get_state_config(state: InterviewState) -> StateConfig:
    global _STATE_CONFIG_CACHE
    if _STATE_CONFIG_CACHE is None:
        _STATE_CONFIG_CACHE = build_state_configurations()
    return _STATE_CONFIG_CACHE.get(state, StateConfig())


# ---------------------------------------------------------------------------
# BASIC TRANSITIONS
# ---------------------------------------------------------------------------

def map_phase_to_state(phase: str) -> InterviewState:
    """
    Map phase string to InterviewState (topic-focused only).
    Handles both topic-focused phase names and legacy phase names for backward compatibility.
    """
    p = (phase or "").lower()
    
    # Topic-focused phases (primary)
    if p in ["topic_introduction", "introduction", "opening"]:
        return InterviewState.TOPIC_INTRODUCTION
    if p in ["topic_fundamentals", "fundamentals", "exploration"]:
        return InterviewState.TOPIC_FUNDAMENTALS
    if p in ["topic_deep_dive", "deep_dive"]:
        return InterviewState.TOPIC_DEEP_DIVE
    if p in ["topic_feedback", "feedback"]:
        return InterviewState.TOPIC_FEEDBACK
    if p in ["completed", "closing"]:
        return InterviewState.COMPLETED
    
    # Default to introduction
    return InterviewState.TOPIC_INTRODUCTION


def get_next_state(current_state: InterviewState, question_count: int = 0) -> InterviewState:
    """
    Suggest next state based on question count and state configuration.
    
    NOTE: This is a helper function. Orchestrator controls actual state transitions.
    Do not call this directly to advance state - use orchestrator logic instead.
    """
    cfg = get_state_config(current_state)
    if question_count >= cfg.max_questions and cfg.next_states:
        return cfg.next_states[0]
    return current_state


def can_transition(from_state: InterviewState, to_state: InterviewState) -> bool:
    cfg = get_state_config(from_state)
    return to_state in cfg.next_states or to_state == from_state


# ---------------------------------------------------------------------------
# ADVANCED DECISION LOGIC
# ---------------------------------------------------------------------------

"""
IMPORTANT ARCHITECTURAL NOTE:
This module defines interview states and validation rules.
Actual state progression decisions are centralized in orchestrator.py.
This file MUST NOT introduce progression logic based on modality or metadata.

The state machine is authoritative for:
- State definitions and valid transitions
- State configuration and validation

The orchestrator is authoritative for:
- When to transition between states
- How question_count and analysis influence progression
- Stage determination based on conversation flow
"""

def determine_next_state(
    current_state: InterviewState,
    analysis: Optional[Dict[str, Any]] = None,
    session: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> InterviewState:
    """
    Simplified state logic for topic-focused interviews.
    Note: Orchestrator handles hybrid stage progression, so this is mainly for compatibility.
    
    CRITICAL: Metadata MUST NOT influence state transitions (bias + modality leakage risk).
    This function accepts metadata parameter for API compatibility but does not use it.
    """
    analysis = analysis or {}
    session = session or {}
    metadata = metadata or {}
    
    # Metadata MUST NOT influence state transitions (bias + modality leakage risk)
    # This parameter exists for API compatibility only - do not use it for decisions

    # Circuit breaker check - fast exit to completed
    if session.get("circuit_open"):
        return InterviewState.COMPLETED

    # Negative behavior fast-exit
    sentiment = analysis.get("sentiment")
    engage = analysis.get("engagement_score")
    if sentiment == "negative" and (engage is not None and engage < 0.2):
        return InterviewState.COMPLETED

    # Simplified: stay in current state (orchestrator handles progression via hybrid logic)
    # This function is kept for backward compatibility but orchestrator is the source of truth
    return current_state
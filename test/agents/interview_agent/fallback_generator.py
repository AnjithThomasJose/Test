"""
Fallback question generation for interview agent.

Provides fallback questions when primary question generation fails
or when natural fallback questions are needed.
"""

import random
import logging
import yaml
import os
import asyncio
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from difflib import SequenceMatcher

from .state_manager import InterviewState
from .conversation_context import ConversationContext

log = logging.getLogger(__name__)

_FALLBACK_CACHE = None
_FALLBACK_LOCK = asyncio.Lock()

# Injectable selector function for testing (defaults to random.choice)
_question_selector: Callable[[List[str]], str] = random.choice


def set_question_selector(selector: Callable[[List[str]], str]) -> None:
    """
    Set a custom question selector function for testing.
    
    Args:
        selector: Function that takes a list and returns one item
    """
    global _question_selector
    _question_selector = selector


def _get_question_selector() -> Callable[[List[str]], str]:
    """Get the current question selector function."""
    return _question_selector


def _get_fallback_questions_path() -> Path:
    """
    Get path to fallback_questions.yaml, handling bundled/zipped environments.
    
    Returns:
        Path to fallback_questions.yaml file
    """
    try:
        # Try standard path first
        base_dir = Path(__file__).resolve().parent / "constants"
        fallback_path = base_dir / "fallback_questions.yaml"
        if fallback_path.exists():
            return fallback_path
    except (AttributeError, OSError):
        pass
    
    # Fallback: try relative to current working directory
    try:
        fallback_path = Path("agents/interview_agent/constants/fallback_questions.yaml")
        if fallback_path.exists():
            return fallback_path
    except Exception:
        pass
    
    # Last resort: try to find it in common locations
    for base in [Path.cwd(), Path(__file__).parent.parent.parent]:
        fallback_path = base / "agents" / "interview_agent" / "constants" / "fallback_questions.yaml"
        if fallback_path.exists():
            return fallback_path
    
    # Return default path (will fail gracefully if not found)
    return Path(__file__).parent / "constants" / "fallback_questions.yaml"


async def _load_fallback_questions() -> Dict[str, Any]:
    """
    Async, cached fallback loader. Reads YAML only once.
    Validates state mappings on load.
    """
    global _FALLBACK_CACHE
    async with _FALLBACK_LOCK:
        if _FALLBACK_CACHE is not None:
            return _FALLBACK_CACHE
        try:
            fallback_path = _get_fallback_questions_path()
            
            def _read_yaml():
                with open(fallback_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            
            _FALLBACK_CACHE = await asyncio.to_thread(_read_yaml)
            config = _FALLBACK_CACHE or {}
            
            # Validate state mappings on load
            _validate_state_mappings(config)
            
            return config
        except FileNotFoundError:
            log.error(f"fallback_questions.yaml not found at {_get_fallback_questions_path()}")
            _FALLBACK_CACHE = {}
            return {}
        except Exception as e:
            log.error(f"Failed to load fallback questions: {e}", exc_info=True)
            _FALLBACK_CACHE = {}
            return {}


def _validate_state_mappings(config: Dict[str, Any]) -> None:
    """
    Validate that state mappings in YAML match expected keys.
    Logs warnings for missing or unexpected keys.
    """
    # Expected YAML sections
    expected_sections = ["topic_aware", "generic", "topic_defaults", "role_specific", "engagement_based"]
    
    # Expected state keys in topic_aware and generic (legacy keys for backward compatibility)
    expected_legacy_keys = {
        "opening_rapport", "stage_setting", "main_questions", 
        "candidate_questions", "feedback", "closing"
    }
    
    # Expected topic-focused state keys (matching InterviewState enum)
    expected_topic_states = {
        "topic_introduction", "topic_fundamentals", "topic_problem_solving",
        "topic_deep_dive", "topic_feedback", "completed"
    }
    
    # Validate topic_aware section (can have legacy keys OR topic-focused keys)
    topic_aware = config.get("topic_aware", {})
    for key in topic_aware.keys():
        key_lower = key.lower()
        if (key_lower not in {k.lower() for k in expected_legacy_keys} and 
            key_lower not in {k.lower() for k in expected_topic_states}):
            log.warning(f"Unexpected state key in topic_aware: {key}")
    
    # Validate generic section (can have legacy keys OR topic-focused keys)
    generic = config.get("generic", {})
    for key in generic.keys():
        key_lower = key.lower()
        if (key_lower not in {k.lower() for k in expected_legacy_keys} and 
            key_lower not in {k.lower() for k in expected_topic_states}):
            log.warning(f"Unexpected state key in generic: {key}")
    
    # Validate topic_defaults section (allows both topic-focused keys and legacy keys for backward compatibility)
    topic_defaults = config.get("topic_defaults", {})
    for key in topic_defaults.keys():
        key_lower = key.lower()
        if (key_lower not in {k.lower() for k in expected_topic_states} and 
            key_lower not in {k.lower() for k in expected_legacy_keys}):
            log.warning(f"Unexpected state key in topic_defaults: {key}")


def _topic_similarity(topic1: str, topic2: str) -> float:
    """
    Calculate similarity ratio between two topics using SequenceMatcher.
    
    Args:
        topic1: First topic string
        topic2: Second topic string
        
    Returns:
        Similarity ratio between 0.0 and 1.0
    """
    return SequenceMatcher(None, topic1.lower(), topic2.lower()).ratio()


def _has_topic_reference(question: str, interview_topic: str, threshold: float = 0.85) -> bool:
    """
    Check if question references interview_topic using fuzzy matching.
    
    For short topics (<= 4 chars), uses exact matching only.
    For longer topics, uses fuzzy matching with higher threshold.
    
    Args:
        question: Question string to check
        interview_topic: Topic to look for
        threshold: Similarity threshold for fuzzy matching (default 0.85)
        
    Returns:
        True if topic is referenced (exact match or fuzzy match above threshold)
    """
    question_lower = question.lower()
    topic_lower = interview_topic.lower()
    
    # Exact substring match (always check first)
    if topic_lower in question_lower:
        return True
    
    # For short topics (AI, ML, NLP, API), use exact matching only
    if len(topic_lower) <= 4:
        return topic_lower in question_lower
    
    # For longer topics, use fuzzy matching with tighter threshold
    # Check for fuzzy matches in question words
    question_words = re.findall(r'\b\w+\b', question_lower)
    for word in question_words:
        if len(word) >= 3:  # Only check words of 3+ chars
            similarity = _topic_similarity(word, topic_lower)
            if similarity >= threshold:
                return True
    
    # Check if topic words appear in question (exact match for individual words)
    topic_words = re.findall(r'\b\w+\b', topic_lower)
    for topic_word in topic_words:
        if len(topic_word) >= 3 and topic_word in question_lower:
            return True
    
    return False


def _sanitize_question(question: str, interview_topic: Optional[str] = None, template: Optional[str] = None) -> str:
    """
    Sanitize and validate a fallback question.
    
    - Strip greetings completely (hi, hello, etc.)
    - Enforce topic reference using fuzzy matching
    - Add topic reference only if missing
    - Always end with "?"
    - Enforce 2 sentences max (truncate to two sentences)
    - Removes unformatted placeholders
    - Logs warnings for invalid templates
    
    Args:
        question: Raw question string
        interview_topic: Optional interview topic to enforce
        template: Original template string (for logging)
        
    Returns:
        Sanitized question string
    """
    if not question:
        if interview_topic:
            return f"Can you tell me about your experience with {interview_topic}?"
        return "Can you tell me more about your background?"
    
    original_question = question
    
    # Strip greetings completely (violates system_message_template rules)
    # Match common greeting patterns at start of string
    question = re.sub(r"^(hello|hi|hey|greetings|welcome|good\s+(morning|afternoon|evening))[!,\.]?\s*", "", question, flags=re.IGNORECASE)
    question = question.strip()
    
    # Check for unformatted placeholders (e.g., {unknown_key})
    placeholder_pattern = r"\{[^}]*\}"
    placeholders = re.findall(placeholder_pattern, question)
    if placeholders:
        log.warning(f"Unformatted placeholders found in question: {placeholders}. Template: {template or 'N/A'}")
        question = re.sub(placeholder_pattern, "", question)
        question = question.strip()
    
    # Normalize whitespace (fix double spaces after placeholder removal)
    question = re.sub(r"\s+", " ", question).strip()
    
    # Global spacing normalization: ensure proper spacing between words
    # Fixes cases where words are concatenated without spaces (e.g., "whereyou" -> "where you")
    question = re.sub(r'([a-z])([A-Z])', r'\1 \2', question)  # "whereYou" -> "where You"
    question = re.sub(r'(\w)([A-Z][a-z])', r'\1 \2', question)  # More general case
    question = re.sub(r'\s+', ' ', question).strip()  # Normalize whitespace again
    
    # Enforce topic reference using fuzzy matching - add only if missing
    if interview_topic:
        if not _has_topic_reference(question, interview_topic, threshold=0.85):
            # Append topic reference if missing
            question = question.rstrip("?.,! ")
            question = f"{question} in the context of {interview_topic}?"
    else:
        # Ensure ends with "?" if no topic
        if not question.endswith("?"):
            question = question.rstrip(".,! ") + "?"
    
    # Enforce 2 sentences max (matches question_generator and prompts)
    from .question_generator import _truncate_to_two_sentences, _normalize_question_grammar
    question = _truncate_to_two_sentences(question)
    if not question.endswith("?"):
        question = question.rstrip(".,! ") + "?"
    
    # Global grammar fix: missing preposition after "experience", missing verb after "how did/do you"
    question = _normalize_question_grammar(question)
    
    # Final check: always end with "?"
    if not question.endswith("?"):
        question = question.rstrip(".,! ") + "?"
    
    # Log if question was significantly modified
    if original_question != question and template:
        log.debug(f"Question sanitized. Original: {original_question[:100]}... Modified: {question[:100]}...")
    
    return question.strip()


async def get_default_question(state: InterviewState, interview_topic: Optional[str] = None) -> str:
    """
    Get topic-focused default question for a given state.
    
    RULES:
    - Never includes greetings (violates system_message_template)
    - Always references interview_topic when provided
    - Maximum 2 sentences
    - Must end with "?"
    
    Args:
        state: Current InterviewState
        interview_topic: Optional interview topic (MANDATORY for topic-focused interviews)
        
    Returns:
        Default question string (sanitized)
    """
    fallback_config = await _load_fallback_questions()
    topic_defaults = fallback_config.get("topic_defaults", {})
    
    # Try to get from YAML first (case-insensitive lookup)
    state_key = state.value.lower()
    template = None
    for key, value in topic_defaults.items():
        if key.lower() == state_key:
            template = value
            break
    
    # If found in YAML, format it
    if template and interview_topic:
        try:
            question = template.format(interview_topic=interview_topic)
            return _sanitize_question(question, interview_topic, template)
        except (KeyError, ValueError) as e:
            # Template has invalid placeholders, log and use fallback
            log.warning(f"Invalid template or placeholder in: {template}. Error: {e}")
            pass
    
    # Topic-focused defaults (always reference topic)
    # NOTE: COMPLETED state is excluded as it's not a question (violates rules)
    if interview_topic:
        default_questions = {
            InterviewState.TOPIC_INTRODUCTION: f"Can you tell me about your experience with {interview_topic}?",
            InterviewState.TOPIC_FUNDAMENTALS: f"Can you explain a core concept related to {interview_topic}?",
            InterviewState.TOPIC_DEEP_DIVE: f"Can you walk me through a real project where you applied {interview_topic}?",
            InterviewState.TOPIC_FEEDBACK: f"What aspects of {interview_topic} do you feel most confident discussing?",
            # COMPLETED state removed - it's not a question and violates rules
        }
    else:
        # Generic defaults (no topic available - should be rare in topic-focused mode)
        default_questions = {
            InterviewState.TOPIC_INTRODUCTION: "Can you tell me about your background?",
            InterviewState.TOPIC_FUNDAMENTALS: "Can you describe a challenging project you've worked on?",
            InterviewState.TOPIC_DEEP_DIVE: "Can you go deeper into your technical approach?",
            InterviewState.TOPIC_FEEDBACK: "What questions do you have about the role?",
            # COMPLETED state removed - it's not a question and violates rules
        }
    
    # Handle COMPLETED state separately (not a question, return empty or skip)
    if state == InterviewState.COMPLETED:
        log.debug("COMPLETED state requested - returning empty (not a question)")
        return ""
    
    question = default_questions.get(state, "Can you tell me more about your background?")
    return _sanitize_question(question, interview_topic, template)


def _normalize_state_key(state_key: str) -> str:
    """
    Normalize state key for case-insensitive YAML lookup.
    
    Uses exact state keys from InterviewState (no mapping).
    YAML sections should match these keys directly.
    """
    return state_key.lower().strip()


def _build_questions_from_config(
    config: Dict[str, Any],
    section: str,
    state_key: str,
    question_count: int = 0,
    interview_topic: Optional[str] = None,
    candidate_info: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Extract and format questions from YAML config.
    
    Centralized logic for building questions from config to reduce duplication.
    
    Args:
        config: Fallback config dictionary
        section: Section name ("topic_aware" or "generic")
        state_key: Normalized state key for lookup
        question_count: Current question count (for first_question vs follow_up)
        interview_topic: Optional interview topic for formatting
        candidate_info: Optional candidate info for formatting
        
    Returns:
        List of formatted question strings
    """
    section_data = config.get(section, {})
    if not section_data:
        return []
    
    # Case-insensitive lookup
    questions = None
    for key, value in section_data.items():
        if key.lower() == state_key.lower():
            questions = value
            break
    
    if not questions:
        return []
    
    # Handle nested structure (e.g., topic_introduction with first_question/follow_up)
    if isinstance(questions, dict):
        if question_count == 0:
            question_list = questions.get("first_question", [])
        else:
            question_list = questions.get("follow_up", [])
    elif isinstance(questions, list):
        question_list = questions
    else:
        return []
    
    # Format questions with placeholders
    formatted_questions = []
    name = "Candidate"
    role = "role"
    if candidate_info:
        name = candidate_info.get('name', 'Candidate')
        if isinstance(name, list):
            name = name[0] if name else 'Candidate'
        name = str(name) if name else 'Candidate'
        role = candidate_info.get('roles', 'role')
    
    for q in question_list:
        if not isinstance(q, str):
            continue
        
        try:
            # Build format kwargs
            format_kwargs = {}
            if interview_topic:
                format_kwargs["interview_topic"] = interview_topic
            if "{name}" in q or "{role}" in q:
                format_kwargs["name"] = name
                format_kwargs["role"] = role
            
            formatted_q = q.format(**format_kwargs)
            formatted_questions.append(formatted_q)
        except (KeyError, ValueError) as e:
            # Log the offending template for debugging
            log.warning(f"Invalid template or placeholder in: {q}. Error: {e}")
            # Try to inject topic if missing (using fuzzy matching with tighter threshold)
            if interview_topic and not _has_topic_reference(q, interview_topic, threshold=0.85):
                formatted_q = f"{q.rstrip('?.,! ')} in the context of {interview_topic}?"
            else:
                formatted_q = q
            formatted_questions.append(formatted_q)
    
    return formatted_questions


async def get_fallback_question_for_state(
    state: InterviewState,
    candidate_info: Dict[str, str],
    question_count: int = 0,
    interview_topic: Optional[str] = None
) -> str:
    """
    State-aware fallback questions that reference interview_topic when available.
    
    RULES:
    - Always references interview_topic when provided
    - No greetings
    - Maximum 2 sentences
    - Ends with "?"
    
    Args:
        state: Current InterviewState
        candidate_info: Candidate information dictionary
        question_count: Current question count
        interview_topic: Optional interview topic (MANDATORY for topic-focused interviews)
        
    Returns:
        Fallback question string (sanitized)
    """
    fallback_config = await _load_fallback_questions()
    
    # Extract candidate name
    name = candidate_info.get('name', 'Candidate')
    if isinstance(name, list):
        name = name[0] if name else 'Candidate'
    name = str(name) if name else 'Candidate'
    
    state_key = state.value
    yaml_key = _normalize_state_key(state_key)
    
    # Try topic_defaults first (uses exact topic-focused state keys)
    formatted_questions = []
    if interview_topic:
        topic_defaults = fallback_config.get("topic_defaults", {})
        template = None
        for key, value in topic_defaults.items():
            if key.lower() == yaml_key:
                template = value
                break
        
        if template:
            try:
                formatted_q = template.format(interview_topic=interview_topic)
                formatted_questions.append(formatted_q)
            except (KeyError, ValueError) as e:
                log.warning(f"Invalid template or placeholder in topic_defaults: {template}. Error: {e}")
    
    # Fall back to topic_aware/generic if topic_defaults didn't work
    if not formatted_questions:
        # Map topic-focused states to legacy YAML keys for topic_aware/generic sections
        legacy_mapping = {
            "topic_introduction": "opening_rapport",
            "topic_fundamentals": "main_questions",
            "topic_deep_dive": "main_questions",
            "topic_feedback": "feedback",
        }
        legacy_key = legacy_mapping.get(yaml_key, yaml_key)
        
        if interview_topic:
            formatted_questions = _build_questions_from_config(
                fallback_config,
                "topic_aware",
                legacy_key,
                question_count,
                interview_topic,
                candidate_info
            )
        else:
            formatted_questions = _build_questions_from_config(
                fallback_config,
                "generic",
                legacy_key,
                question_count,
                None,
                candidate_info
            )
    
    if formatted_questions:
        selector = _get_question_selector()
        selected = selector(formatted_questions)
        return _sanitize_question(selected, interview_topic)
    
    # Ultimate fallback (always topic-focused and specific if topic provided)
    if interview_topic:
        return _sanitize_question(f"Walk me through a specific project where you applied {interview_topic} to solve a real problem.", interview_topic)
    return _sanitize_question("Describe a technical project you're proud of and the key decisions you made.", None)


async def get_natural_fallback_question(
    conversation_context: ConversationContext,
    candidate_info: Dict[str, str],
    session_id: Optional[str] = None,
    uid_context: Optional[Dict[str, Any]] = None,
    interview_topic: Optional[str] = None
) -> str:
    """
    Generate intelligent fallback questions with context awareness.
    
    RULES:
    - Always references interview_topic when provided (MANDATORY for topic-focused interviews)
    - No greetings
    - Maximum 2 sentences
    - Ends with "?"
    
    Args:
        conversation_context: ConversationContext object
        candidate_info: Candidate information dictionary
        session_id: Optional session ID
        uid_context: Optional UID context
        interview_topic: Optional interview topic (MANDATORY for topic-focused interviews)
        
    Returns:
        Fallback question string (sanitized)
    """
    phase = conversation_context.interview_phase
    topics = conversation_context.topics_discussed
    engagement_level = conversation_context.engagement_level
    question_count = conversation_context.conversation_depth
    
    fallback_config = await _load_fallback_questions()
    phase_questions = []
    
    # PRIMARY: Use interview_topic if provided (topic-focused mode)
    if interview_topic:
        # Build topic-focused questions that are specific, not generic
        topic_questions = [
            f"Walk me through a specific project where you applied {interview_topic}.",
            f"What technical trade-offs have you encountered when working with {interview_topic}?",
            f"Describe a complex problem you solved using {interview_topic} and your approach.",
            f"What would you do differently if you could redo one of your {interview_topic} projects?",
        ]
        phase_questions.extend(topic_questions)
        
        # Add engagement-based questions (with topic reference)
        engagement_based = fallback_config.get("engagement_based", {})
        if engagement_level < 0.3:
            low_engagement = engagement_based.get("low_engagement", [])
            # Inject topic into low engagement questions (using fuzzy matching)
            for q in low_engagement:
                if not _has_topic_reference(q, interview_topic):
                    q = f"{q.rstrip('?.,! ')} related to {interview_topic}?"
                phase_questions.append(q)
        elif engagement_level > 0.7:
            high_engagement = engagement_based.get("high_engagement", [])
            # Inject topic into high engagement questions (using fuzzy matching)
            for q in high_engagement:
                if not _has_topic_reference(q, interview_topic):
                    q = f"{q.rstrip('?.,! ')} with {interview_topic}?"
                phase_questions.append(q)
        
        # Add topic-based questions from topics_discussed (but ensure they reference interview_topic)
        if topics and len(topics) > 0:
            recent_topic = topics[-1]
            # Only use if it's related to interview_topic (avoid drift) - use fuzzy matching with tighter threshold
            if _has_topic_reference(recent_topic, interview_topic, threshold=0.85):
                topic_questions = [
                    f"That's great insight about {recent_topic}. Can you elaborate on how this relates to {interview_topic}?",
                    f"I'm curious about your approach to {recent_topic}. How does this connect to {interview_topic}?",
                ]
                phase_questions.extend(topic_questions)
    else:
        # FALLBACK: Generic questions (should be rare in topic-focused mode)
        # Try to get from role_specific as fallback (but don't rely on it)
        role_specific = fallback_config.get("role_specific", {})
        # Try TECHNICAL_CODING first, then any available
        role_questions = role_specific.get("TECHNICAL_CODING", {})
        if not role_questions and role_specific:
            # Use first available role
            role_questions = list(role_specific.values())[0]
        
        if isinstance(role_questions, dict):
            phase_questions = role_questions.get(phase, ["Can you tell me more about your background?"])
        else:
            phase_questions = ["Can you tell me more about your background?"]
        
        # Add engagement-based questions (generic)
        engagement_based = fallback_config.get("engagement_based", {})
        if engagement_level < 0.3:
            phase_questions.extend(engagement_based.get("low_engagement", []))
        elif engagement_level > 0.7:
            phase_questions.extend(engagement_based.get("high_engagement", []))
        
        # Add topic-based questions from topics_discussed (generic)
        if topics and len(topics) > 0:
            recent_topic = topics[-1]
            topic_questions = [
                f"That's great insight about {recent_topic}. Can you elaborate on your experience with that?",
                f"I'm curious about your approach to {recent_topic}. What methodology did you use?",
            ]
            phase_questions.extend(topic_questions)
    
    # Deduplicate questions (engagement-based can add duplicates)
    phase_questions = list(dict.fromkeys(phase_questions))
    
    # Select a question using injectable selector
    selector = _get_question_selector()
    if len(phase_questions) > 0:
        selected_question = selector(phase_questions)
    else:
        if interview_topic:
            selected_question = f"Can you tell me more about your experience with {interview_topic}?"
        else:
            selected_question = "Can you tell me more about your background and experience?"
    
    # Sanitize and ensure topic is referenced
    return _sanitize_question(selected_question, interview_topic)


"""
Anti-repetition module for preventing duplicate questions.

This module provides functions to detect and prevent similar questions
from being asked multiple times during an interview session.
"""

import re
import logging
import asyncio
from typing import List, Optional

from .config import get_interview_config
from . import llm_utils
from .session_manager import load_question_fingerprints, save_question_fingerprints

log = logging.getLogger(__name__)

# Async-safe in-memory cache (session_id -> list[list[str]])
_QUESTION_FINGERPRINTS: dict = {}
_QFP_LOCK = asyncio.Lock()
_MAX_CACHE_PER_SESSION = 50


def generate_ngram_fingerprint(question: str, n: int = 3) -> List[str]:
    """
    Generate n-gram fingerprints from a question for similarity detection.
    Normalizes text and creates overlapping n-grams of words.
    Excludes common interview words that would cause false positives.
    
    Args:
        question: The question text to fingerprint
        n: Size of n-grams (default: 3)
    
    Returns:
        List of n-gram strings
    """
    if not question or not isinstance(question, str):
        return []
    
    # Common words to exclude from fingerprinting (cause false positives)
    EXCLUDE_WORDS = {
        # Generic interview words
        "can", "you", "tell", "me", "about", "your", "experience", "with",
        "how", "do", "did", "what", "when", "where", "why", "which",
        "please", "describe", "explain", "walk", "through", "give",
        "example", "specific", "particular", "more", "the", "a", "an",
        "have", "has", "had", "would", "could", "should", "will",
        "project", "projects", "work", "worked", "working",
        # Common tech terms that appear in many questions
        "using", "used", "use", "built", "build", "building",
        "applied", "apply", "applying", "solve", "solved", "solving",
    }
    
    # Normalize: lowercase, remove punctuation, split into words
    normalized = re.sub(r'[^\w\s]', '', question.lower())
    words = [w for w in normalized.split() if w not in EXCLUDE_WORDS]
    
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
    Enhanced to catch paraphrased questions more effectively.
    
    Args:
        fingerprint1: First fingerprint (list of n-grams)
        fingerprint2: Second fingerprint (list of n-grams)
        threshold: Similarity threshold (0.0 to 1.0) - lowered to catch more paraphrases
    
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
    
    # Jaccard similarity
    jaccard_similarity = intersection / union
    
    # Also check if one set is mostly contained in the other (catches paraphrases)
    # This helps catch cases like "what is X" vs "what does X involve" or "can you explain X"
    min_set_size = min(len(set1), len(set2))
    if min_set_size > 0:
        containment_ratio = intersection / min_set_size
        # Raised to 0.7 (70%) to reduce false positives - only block truly similar questions
        if containment_ratio >= 0.7:
            return True
    
    return jaccard_similarity >= threshold


async def check_question_similarity(
    question: str,
    session_id: str,
    window_size: int = None,
    similarity_threshold: float = None,
    ngram_size: int = None
) -> bool:
    """
    Async: Check if a question is too similar to recent questions in the session.
    Returns True if the question should be blocked (too similar), False otherwise.
    """
    # Load config defaults if not provided
    try:
        cfg = get_interview_config().get("anti_repetition", {})
        window_size = window_size or cfg.get("window_size", 5)
        # Raised threshold to 0.75 to reduce false positives (was 0.6)
        similarity_threshold = similarity_threshold or cfg.get("similarity_threshold", 0.75)
        ngram_size = ngram_size or cfg.get("ngram_size", 3)
    except Exception:
        window_size = window_size or 5
        # Raised threshold to reduce false positives
        similarity_threshold = similarity_threshold or 0.75
        ngram_size = ngram_size or 3

    # Generate fingerprint for current question
    current_fingerprint = generate_ngram_fingerprint(question, ngram_size)
    if not current_fingerprint:
        return False

    # Ensure cache loaded (async)
    async with _QFP_LOCK:
        if session_id not in _QUESTION_FINGERPRINTS:
            try:
                # load_question_fingerprints is async in session_manager; await it
                recovered = await load_question_fingerprints(session_id)
                _QUESTION_FINGERPRINTS[session_id] = recovered or []
            except Exception:
                _QUESTION_FINGERPRINTS[session_id] = []
        recent_fingerprints = list(_QUESTION_FINGERPRINTS.get(session_id, []))

    # Check local n-gram Jaccard similarity first (fast)
    for past_fingerprint in recent_fingerprints[-window_size:]:
        if is_semantically_similar(current_fingerprint, past_fingerprint, similarity_threshold):
            log.debug("Anti-Repetition: blocked by ngram similarity")
            return True

    # NOTE: LLM semantic similarity check disabled - it was not properly implemented
    # (no actual question history was being passed to compare against).
    # The n-gram fingerprint check above is sufficient for detecting similar questions.
    # Future improvement: Store actual question texts alongside fingerprints and implement
    # proper LLM-based semantic comparison if needed.

    return False


async def track_question_fingerprint(question: str, session_id: str, ngram_size: int = None) -> None:
    """
    Async: track a question fingerprint (persisted to Chroma via session_manager).
    """
    try:
        cfg = get_interview_config().get("anti_repetition", {})
        ngram_size = ngram_size or cfg.get("ngram_size", 3)
    except Exception:
        ngram_size = ngram_size or 3

    fingerprint = generate_ngram_fingerprint(question, ngram_size)
    if not fingerprint:
        return

    async with _QFP_LOCK:
        lst = _QUESTION_FINGERPRINTS.setdefault(session_id, [])
        lst.append(fingerprint)
        # keep bounded
        if len(lst) > _MAX_CACHE_PER_SESSION:
            _QUESTION_FINGERPRINTS[session_id] = lst[-_MAX_CACHE_PER_SESSION:]

        # persist to Chroma (session_manager.save_question_fingerprints is async)
        try:
            await save_question_fingerprints(session_id, list(_QUESTION_FINGERPRINTS[session_id]))
        except Exception:
            # best-effort: log and keep in-memory
            log.exception("Failed to persist question fingerprints for session %s", session_id)


__all__ = [
    "generate_ngram_fingerprint",
    "is_semantically_similar",
    "check_question_similarity",
    "track_question_fingerprint",
]

# NOTE: check_question_similarity and track_question_fingerprint are async.
# Callers must `await` them: `await check_question_similarity(...)`.


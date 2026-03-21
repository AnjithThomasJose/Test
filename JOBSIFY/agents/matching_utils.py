"""
Shared anti-hallucination utilities for job_matcher and ranker.
Avoids circular imports between job_matcher and ranker.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Set

log = logging.getLogger(__name__)


def _normalize_for_match(s: str) -> str:
    """Normalize skill for matching (lowercase, strip, collapse spaces)."""
    if not s or not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.lower().strip())


def _job_skill_matches_candidate(job_skill: str, candidate_skills: Set[str]) -> bool:
    """Check if job_skill has a match in candidate_skills (flexible: React.js=React, C#=.NET)."""
    if not candidate_skills or not job_skill:
        return False
    job_norm = _normalize_for_match(job_skill)
    if not job_norm:
        return False
    # Extract key tokens from job skill (e.g. "react.js" -> "react", "reactjs")
    job_tokens = set(t for t in re.findall(r"\b\w+\b", job_norm) if len(t) >= 3)
    job_tokens.add(job_norm.replace(".", "").replace(" ", ""))  # "react.js" -> "reactjs"
    for cs in candidate_skills:
        cs_norm = _normalize_for_match(str(cs))
        if not cs_norm:
            continue
        if job_norm == cs_norm or job_norm in cs_norm or cs_norm in job_norm:
            return True
        # Shared significant token (min 3 chars to avoid "in", "or", etc.)
        if any(t in cs_norm for t in job_tokens):
            return True
        cs_tokens = set(t for t in re.findall(r"\b\w+\b", cs_norm) if len(t) >= 3)
        if any(t in job_norm for t in cs_tokens):
            return True
    return False


def append_rationale_skill_drop_note(rationale: str, dropped_skills: List[str]) -> str:
    """
    When skills were dropped during validation, append a note so the rationale
    stays consistent with the corrected skills_matched.
    """
    if not rationale or not dropped_skills:
        return rationale
    skills_str = ", ".join(dropped_skills[:5])  # Limit to 5 for brevity
    if len(dropped_skills) > 5:
        skills_str += f" (+{len(dropped_skills) - 5} more)"
    note = f" Note: The following skills were not found in the resume during validation: {skills_str}. The match score has been adjusted accordingly."
    return (rationale.rstrip() + note).strip()


def validate_skills_against_candidate(
    llm_skills_matched: List[str],
    candidate_skills: Set[str],
    job_required_skills: List[str],
) -> List[str]:
    """
    Deterministic validation: only keep skills_matched that have a corresponding candidate skill.
    Prevents LLM from attributing skills the candidate does not have.
    """
    cand_set = set(str(s) for s in (candidate_skills or []))
    if not cand_set:
        return []
    validated = []
    for s in llm_skills_matched:
        if s not in job_required_skills:
            continue
        if _job_skill_matches_candidate(s, cand_set):
            validated.append(s)
        else:
            log.debug(f"Validated filter: '{s}' has no match in candidate skills")
    return validated

import json
import re
import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Literal
from models.llm_invoker import invoke_llm, invoke_structured_llm
from core.model_registry import TaskType
from agents.prompt_generator import generate_scoring_and_roles_prompt
from langsmith.run_helpers import traceable
from utils.session_manager import session_manager
from chroma import get_chat_session, update_chat_session, upsert_resume_score_doc
from core.error_handler import with_error_handling
from core.utils import (
    _mask,
    _sanitize_text_for_llm,
    _to_text,
    _clean_json_text,
    _scan_balanced_json,
    _safe_json_loads,
    _extract_json_from_response,
    _coerce_score,
    _merge_dedupe,
    _normalize_chat_history,
    _create_error_response,
    _validate_state_inputs,
    _generate_request_id,
    _calculate_processing_time,
    create_agent_state,
    filter_resume_data_for_agent,
    run_blocking_io,
    map_assessment_score_to_skill_score_floor,
)
from core.config import get_agent_config
from core.security import (
    validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm,
    PII_PATTERNS, INJECTION_FILTERS
)
from core.memory import BaseAgentMemory, get_agent_memory
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion

log = logging.getLogger(__name__)

# Get centralized configuration
config = get_agent_config("resume_scorer")
TIMEOUT_SECONDS = config.timeout_seconds
MAX_PROMPT_CHARS = config.max_prompt_chars
MAX_RESPONSE_LENGTH = config.max_response_length
MAX_CHAT_HISTORY_ITEMS = config.max_chat_history_items
LLM_MODEL = config.llm_model

# Type-safe analysis methods (LLM-only approach)
AnalysisMethod = Literal["llm", "error", "llm_timeout"]

# Score categories constant to avoid drift (Skills + Format only; Experience and Education removed)
SCORE_CATEGORIES = ('SkillsScore', 'FormatScore')

# --- Agentic AI Constants ---
CONFIDENCE_THRESHOLD = 0.75   # Minimum confidence for scoring results
ADAPTATION_WINDOW = 50        # Number of recent analyses to consider
MIN_SCORE_CATEGORIES = 2      # Minimum scoring categories required (Skills, Format)
MIN_RESUME_SCORE = 0.0        # Minimum valid resume score
MAX_RESUME_SCORE = 100.0      # Maximum valid resume score
SCORE_WEIGHTS = {
    "SkillsScore": 0.75,
    "FormatScore": 0.25,
}


def _weighted_ats_total_from_breakdown(breakdown: Dict[str, Any]) -> Optional[float]:
    """
    Derive overall ATS score from category breakdown.
    skills → SkillsScore weight; presentation → FormatScore weight.
    When both exist: 0.75 * skills + 0.25 * presentation (matches SCORE_WEIGHTS).
    """
    if not isinstance(breakdown, dict):
        return None
    sk = _coerce_score(breakdown.get("skills"))
    pr = _coerce_score(breakdown.get("presentation"))
    w_sk = SCORE_WEIGHTS["SkillsScore"]
    w_pr = SCORE_WEIGHTS["FormatScore"]
    if sk is not None and pr is not None:
        total = w_sk * float(sk) + w_pr * float(pr)
        return max(MIN_RESUME_SCORE, min(round(total, 1), MAX_RESUME_SCORE))
    if sk is not None:
        return max(MIN_RESUME_SCORE, min(round(float(sk), 1), MAX_RESUME_SCORE))
    if pr is not None:
        return max(MIN_RESUME_SCORE, min(round(float(pr), 1), MAX_RESUME_SCORE))
    return None


def _reconcile_resume_score_totals_with_breakdown(rs: Dict[str, Any]) -> None:
    """Mutate resumeScore dict so total/ats_score match weighted breakdown (fixes inconsistent LLM output)."""
    bd = rs.get("breakdown")
    computed = _weighted_ats_total_from_breakdown(bd if isinstance(bd, dict) else {})
    if computed is None:
        return
    raw = _coerce_score(rs.get("total"))
    if raw is not None and abs(float(raw) - float(computed)) >= 3.0:
        log.info(
            "RESUME_SCORER: Reconciling resumeScore.total with weighted breakdown "
            "(llm_total=%s → %s, breakdown=%s)",
            raw,
            computed,
            bd,
        )
    rs["total"] = computed
    rs["ats_score"] = computed
# Max points SkillsScore can increase per assessment, by difficulty (configurable via config)
# Values loaded from config to allow environment variable overrides
MAX_SKILLS_BOOST_BY_DIFFICULTY = {
    "easy": config.skills_boost_easy,
    "medium": config.skills_boost_medium,
    "hard": config.skills_boost_hard,
    "expert": config.skills_boost_expert,
}
DEFAULT_SKILLS_BOOST_PER_ASSESSMENT = config.skills_boost_default

# PII patterns and injection filters are now imported from core.security

# Custom memory class for resume scorer (extends base memory)
class ResumeScorerMemory(BaseAgentMemory):
    def __init__(self, tenant_id: str = "default_tenant"):
        super().__init__(tenant_id, config.adaptation_window)
        self.scoring_trends = {}
        self.score_effectiveness = {}

async def get_scorer_memory(tenant_id: str = "default_tenant") -> ResumeScorerMemory:
    """Get or create tenant-scoped resume scorer memory."""
    return await get_agent_memory("resume_scorer", tenant_id, ResumeScorerMemory)

def _calculate_confidence_score(scoring_result: Dict[str, Any], method: str) -> float:
    """Calculate confidence score for resume scoring."""
    if not scoring_result:
        return 0.0
    
    score = 0.3  # Base score
    
    # Score validity check
    resume_score = scoring_result.get('ResumeScore', {})
    if isinstance(resume_score, dict):
        overall_score = resume_score.get('OverallScore')
        if isinstance(overall_score, (int, float)) and MIN_RESUME_SCORE <= overall_score <= MAX_RESUME_SCORE:
            score += 0.3
    
    # Category completeness
    present_categories = sum(1 for cat in SCORE_CATEGORIES if resume_score.get(cat) is not None)
    if present_categories >= MIN_SCORE_CATEGORIES:
        score += 0.2
    
    # Quality indicators
    if scoring_result.get('CandidateType'):
        score += 0.1
    if scoring_result.get('ImprovementSuggestions'):
        score += 0.05
    if scoring_result.get('StrengthAreas'):
        score += 0.05
    
    # Method-specific adjustments
    if method == 'llm':
        score += 0.05  # Slight bonus for LLM comprehensiveness
    elif method == 'hybrid':
        score += 0.1   # Bonus for hybrid approach
    
    return max(0.0, min(score, 1.0))

def _assess_scoring_quality(scoring_result: Dict[str, Any]) -> float:
    """Assess quality of resume scoring."""
    if not scoring_result:
        return 0.0
    
    quality_score = 0.0
    
    # Structure validation
    resume_score = scoring_result.get('ResumeScore', {})
    if isinstance(resume_score, dict):
        # Check for valid overall score
        overall = resume_score.get('OverallScore')
        if isinstance(overall, (int, float)) and 0 <= overall <= 100:
            quality_score += 0.3
        
        # Check individual category scores (Skills + Format only)
        score_categories = ['SkillsScore', 'FormatScore']
        valid_categories = 0
        for cat in score_categories:
            cat_score = resume_score.get(cat)
            if isinstance(cat_score, (int, float)) and 0 <= cat_score <= 100:
                valid_categories += 1
        quality_score += (valid_categories / len(score_categories)) * 0.3
    
    # Content quality assessment
    if scoring_result.get('CandidateType') and isinstance(scoring_result['CandidateType'], str):
        quality_score += 0.2
    
    if scoring_result.get('ImprovementSuggestions') and isinstance(scoring_result['ImprovementSuggestions'], list):
        quality_score += 0.1
    
    if scoring_result.get('StrengthAreas') and isinstance(scoring_result['StrengthAreas'], list):
        quality_score += 0.1
    
    return min(quality_score, 1.0)

# Utility functions are now imported from core.utils

# Maximum length for improvement suggestions to prevent overly long LLM responses
MAX_IMPROVEMENT_LEN = 600


def truncate_text(text: str, max_len: int):
    """Safely truncate text without cutting words in the middle."""
    if not isinstance(text, str):
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def _transform_to_expected_format(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform scoring data to expected callback format."""
    log.info(f"🔍 TRANSFORM: Input data type: {type(data)}")
    log.info(f"🔍 TRANSFORM: Input data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    
    if not isinstance(data, dict):
        log.warning(f"🔍 TRANSFORM: Data is not a dict, returning empty")
        return {}
    
    transformed_data = {}
    
    # Check if data is already in new format
    if 'resumeScore' in data and isinstance(data['resumeScore'], dict) and 'breakdown' in data['resumeScore']:
        log.info(f"🔍 TRANSFORM: Data already in new format")
        rs = data['resumeScore']
        # Ensure ats_score/total match weighted breakdown; preserve breakdown_rationale
        rs_copy = dict(rs)
        _reconcile_resume_score_totals_with_breakdown(rs_copy)
        if 'ats_score' not in rs_copy and 'total' in rs_copy:
            rs_copy['ats_score'] = rs_copy['total']
        if isinstance(rs_copy.get('rationale'), str) and len(rs_copy['rationale']) > 2500:
            rs_copy['rationale'] = rs_copy['rationale'][:2497] + "..."
        transformed_data['resumeScore'] = rs_copy
        return transformed_data
    
    # Transform old format (ResumeScore) to new format (resumeScore with breakdown)
    resume_score = data.get('ResumeScore', {})
    log.info(f"🔍 TRANSFORM: ResumeScore found: {type(resume_score)}, keys: {list(resume_score.keys()) if isinstance(resume_score, dict) else 'Not a dict'}")
    
    if isinstance(resume_score, dict):
        log.info(f"🔍 TRANSFORM: Processing ResumeScore dict")
        # Create breakdown object with lowercase category names (Skills + Format only)
        breakdown = {}
        category_mapping = {
            'SkillsScore': 'skills',
            'FormatScore': 'presentation'
        }
        
        for old_cat, new_cat in category_mapping.items():
            score = _coerce_score(resume_score.get(old_cat))
            if score is not None:
                breakdown[new_cat] = max(0.0, min(score, 100.0))
        
        log.info(f"🔍 TRANSFORM: Breakdown created: {breakdown}")
        
        # Get total score
        total = _coerce_score(resume_score.get('OverallScore'))
        if total is None and breakdown:
            # Calculate average if no total provided
            total = sum(breakdown.values()) / len(breakdown)
        
        log.info(f"🔍 TRANSFORM: Total score: {total}")
        
        # Build rationale: overall only (Explanation + Reasoning). Per-category lives in breakdown_rationale only.
        rationale = ""
        if resume_score.get('Explanation'):
            rationale = resume_score['Explanation']
        if resume_score.get('Reasoning') and rationale:
            rationale += " " + resume_score['Reasoning']
        elif resume_score.get('Reasoning'):
            rationale = resume_score['Reasoning']
        sk_r = (resume_score.get('SkillsRationale') or "").strip()
        fmt_r = (resume_score.get('FormatRationale') or "").strip()
        breakdown_rationale = {
            "skills": sk_r,
            "presentation": fmt_r,
        }
        sk_how = truncate_text((resume_score.get('SkillsHowToImprove') or "").strip(), MAX_IMPROVEMENT_LEN)
        fmt_how = truncate_text((resume_score.get('FormatHowToImprove') or "").strip(), MAX_IMPROVEMENT_LEN)
        if not sk_how:
            sk_how = "Take the recommended assessments for your skills and explore recommended courses in the platform to improve this score."
        if not fmt_how:
            fmt_how = "Use the resume download option in the platform for a more ATS-friendly resume."
        breakdown_how_to_improve = {
            "skills": sk_how,
            "presentation": fmt_how,
        }
        log.info(f"🔍 TRANSFORM: Rationale length: {len(rationale)}")
        total_val = max(MIN_RESUME_SCORE, min(total or 0, MAX_RESUME_SCORE))
        transformed_data['resumeScore'] = {
            'breakdown': breakdown,
            'total': total_val,
            'ats_score': total_val,  # OverallScore = ATS compatibility score (alias for clarity)
            'rationale': rationale[:2500] if rationale else "Resume analysis completed successfully.",
            'breakdown_rationale': breakdown_rationale,
            'breakdown_how_to_improve': breakdown_how_to_improve,
        }
    else:
        log.warning(f"🔍 TRANSFORM: ResumeScore is not a dict: {type(resume_score)}")
    
    return transformed_data

def _validate_scoring_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize scoring data."""
    if not isinstance(data, dict):
        return {}
    
    validated_data = {}
    
    # Check for new format first (resumeScore with breakdown)
    resume_score = data.get('resumeScore', {})
    if isinstance(resume_score, dict) and 'breakdown' in resume_score:
        # New format validation
        validated_score = {}
        
        # Validate breakdown (Skills + Format only)
        breakdown = resume_score.get('breakdown', {})
        validated_breakdown: Dict[str, Any] = {}
        if isinstance(breakdown, dict):
            for category in ['skills', 'presentation']:
                score = _coerce_score(breakdown.get(category))
                if score is not None:
                    validated_breakdown[category] = max(0.0, min(score, 100.0))
            validated_score['breakdown'] = validated_breakdown

        # Weighted total must match breakdown (LLMs often return inconsistent overall scores)
        computed_total = _weighted_ats_total_from_breakdown(validated_breakdown)
        if computed_total is not None:
            raw_total = _coerce_score(resume_score.get('total'))
            if raw_total is not None and abs(float(raw_total) - float(computed_total)) >= 3.0:
                log.info(
                    "RESUME_SCORER: Reconciling validated resumeScore.total (llm_total=%s → %s, breakdown=%s)",
                    raw_total,
                    computed_total,
                    validated_breakdown,
                )
            validated_score['total'] = computed_total
            validated_score['ats_score'] = computed_total
        else:
            total = _coerce_score(resume_score.get('total'))
            if total is not None:
                clamped = max(MIN_RESUME_SCORE, min(total, MAX_RESUME_SCORE))
                validated_score['total'] = clamped
                validated_score['ats_score'] = clamped
        
        # Validate rationale (allow longer for detailed + per-category rationales)
        rationale = resume_score.get('rationale')
        if isinstance(rationale, str):
            validated_score['rationale'] = rationale[:2500]  # Cap length
        # Validate breakdown_rationale (optional per-category rationales; Skills + Format only)
        br = resume_score.get('breakdown_rationale')
        if isinstance(br, dict):
            validated_score['breakdown_rationale'] = {
                k: (v[:350] if isinstance(v, str) else "") for k, v in br.items()
                if k in ("skills", "presentation")
            }
        # Validate breakdown_how_to_improve (per-category how-to-improve; Skills + Format only)
        bhi = resume_score.get('breakdown_how_to_improve')
        if isinstance(bhi, dict):
            validated_score['breakdown_how_to_improve'] = {
                k: (v[:300] if isinstance(v, str) else "") for k, v in bhi.items()
                if k in ("skills", "presentation")
            }
        
        validated_data['resumeScore'] = validated_score
    
    # Check for old format (ResumeScore with category scores)
    elif data.get('ResumeScore'):
        resume_score = data.get('ResumeScore', {})
        if isinstance(resume_score, dict):
            validated_score = {}
            
            # Validate overall score
            overall = _coerce_score(resume_score.get('OverallScore'))
            if overall is not None:
                validated_score['OverallScore'] = max(MIN_RESUME_SCORE, min(overall, MAX_RESUME_SCORE))
            
            # Validate category scores
            for category in SCORE_CATEGORIES:
                cat_score = _coerce_score(resume_score.get(category))
                if cat_score is not None:
                    validated_score[category] = max(0.0, min(cat_score, 100.0))
            
            # Validate explanations and per-category rationales
            for field in ['Explanation', 'Reasoning']:
                if field in resume_score and isinstance(resume_score[field], str):
                    validated_score[field] = resume_score[field][:2500]  # Cap length
            for field in ['SkillsRationale', 'FormatRationale']:
                if field in resume_score and isinstance(resume_score[field], str):
                    validated_score[field] = resume_score[field][:350]
            for field in ['SkillsHowToImprove', 'FormatHowToImprove']:
                if field in resume_score and isinstance(resume_score[field], str):
                    validated_score[field] = resume_score[field][:300]
            
            validated_data['ResumeScore'] = validated_score
    
    # Validate CandidateType
    candidate_type = data.get('CandidateType')
    if isinstance(candidate_type, str):
        validated_data['CandidateType'] = candidate_type[:100]
    
    # Validate improvement suggestions
    improvements = data.get('ImprovementSuggestions', [])
    if isinstance(improvements, list):
        validated_improvements = []
        for suggestion in improvements[:10]:  # Cap suggestions
            if isinstance(suggestion, str):
                validated_improvements.append(suggestion[:300])  # Cap length
        validated_data['ImprovementSuggestions'] = validated_improvements
    
    # Validate strength areas
    strengths = data.get('StrengthAreas', [])
    if isinstance(strengths, list):
        validated_strengths = []
        for strength in strengths[:10]:  # Cap strengths
            if isinstance(strength, str):
                validated_strengths.append(strength[:200])  # Cap length
        validated_data['StrengthAreas'] = validated_strengths
    
    return validated_data

def _normalize_resume_score(
    score: Dict[str, Any],
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Ensure all keys exist, clamp values, and recompute OverallScore if needed."""
    if not isinstance(score, dict):
        score = {}
    out = {}
    # Fill each category from score or fallback (then clamp)
    for cat in SCORE_CATEGORIES:
        v = _coerce_score(score.get(cat))
        if v is None and isinstance(fallback, dict):
            v = _coerce_score(fallback.get(cat))
        v = 0.0 if v is None else max(0.0, min(v, 100.0))
        out[cat] = v
    # Explanation / Reasoning passthrough (capped)
    for field, cap in (("Explanation", 2500), ("Reasoning", 2500)):
        if isinstance(score.get(field), str):
            out[field] = score[field][:cap]
    # Per-category rationale passthrough (capped; Skills + Format only)
    for field in ("SkillsRationale", "FormatRationale"):
        if isinstance(score.get(field), str):
            out[field] = score[field][:350]
    # Per-category how-to-improve passthrough (capped)
    for field in ("SkillsHowToImprove", "FormatHowToImprove"):
        if isinstance(score.get(field), str):
            out[field] = score[field][:300]
    # Overall: use provided if valid, else recompute from weights
    overall = _coerce_score(score.get("OverallScore"))
    if overall is None:
        overall = sum(out[cat] * SCORE_WEIGHTS[cat] for cat in SCORE_CATEGORIES)
    out["OverallScore"] = max(MIN_RESUME_SCORE, min(round(float(overall), 1), MAX_RESUME_SCORE))
    return out


def _apply_assessment_boost_to_scoring_result(
    scoring_result: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Use assessment_results (if present) to ensure SkillsScore is at least as strong
    as a good assessment warrants, without hardcoding absolute scores.

    Logic:
    - Look at assessment_results.total_score (0–100). If <60 → no change.
    - Use map_assessment_score_to_skill_score_floor to get a floor for SkillsScore.
    - Read difficulty from the matching assessment_plan entry (topic match).
    - Set SkillsScore = max(SkillsScore, floor) and re-normalize ResumeScore.
    """
    try:
        assessment_results = state.get("assessment_results") or {}
        if not isinstance(assessment_results, dict):
            return scoring_result

        total_score = assessment_results.get("total_score")
        if total_score is None:
            return scoring_result

        # Locate difficulty from assessment_plan/prior_assessment_plan using assessment_topic
        assessment_topic = (
            state.get("assessment_topic")
            or assessment_results.get("assessment_topic")
            or ""
        )
        topic_normalized = str(assessment_topic).strip().lower()

        assessment_plan = (
            state.get("assessment_plan")
            or state.get("prior_assessment_plan")
            or []
        )
        plan_items = []
        if isinstance(assessment_plan, dict):
            plan_items = assessment_plan.get("assessment_plan") or assessment_plan.get("plan") or []
        elif isinstance(assessment_plan, list):
            plan_items = assessment_plan

        difficulty = None
        if topic_normalized and plan_items:
            for item in plan_items:
                if not isinstance(item, dict):
                    continue
                item_topic = str(item.get("topic") or "").strip().lower()
                if item_topic and item_topic == topic_normalized:
                    difficulty = item.get("difficulty")
                    break

        floor = map_assessment_score_to_skill_score_floor(total_score, difficulty)
        resume_score = scoring_result.get("ResumeScore") or {}
        if floor is None or not isinstance(resume_score, dict):
            return scoring_result

        current_skills = _coerce_score(resume_score.get("SkillsScore"))
        if current_skills is None:
            current_skills = 0.0

        # Cap increase by difficulty (configurable via RESUME_SCORER_SKILLS_BOOST_* env vars)
        # Defaults: easy=2, medium=5, hard/expert=8
        # Rationale: One assessment validates one topic; boost should be limited to prevent
        # single-topic dominance (e.g., 75→100 from one perfect assessment).
        diff_key = str(difficulty).strip().lower() if difficulty else ""
        max_boost = MAX_SKILLS_BOOST_BY_DIFFICULTY.get(diff_key, DEFAULT_SKILLS_BOOST_PER_ASSESSMENT)
        uncapped = max(float(current_skills), float(floor))
        new_skills = min(float(current_skills) + max_boost, uncapped)

        resume_score["SkillsScore"] = new_skills
        # Re-normalize (recomputes OverallScore from weighted categories when needed)
        scoring_result["ResumeScore"] = _normalize_resume_score(resume_score)

        # Build human-readable scoring rationale for Calculation Insights
        points_gain = float(new_skills) - float(current_skills)
        diff_label = "Unknown" if (difficulty is None or (isinstance(difficulty, str) and not str(difficulty).strip())) else str(difficulty).strip().capitalize()
        if points_gain >= 0.1:
            if new_skills <= current_skills + max_boost - 0.1:
                rationale = (
                    f"Resume Skills score: {int(round(current_skills))}. "
                    f"Assessment score: {total_score}% ({diff_label}). "
                    f"Max boost for this difficulty: +{int(max_boost)}. "
                    f"Final Skills score: {int(round(new_skills))} (+{int(round(points_gain))} from assessment)."
                )
            else:
                rationale = (
                    f"Resume Skills score: {int(round(current_skills))}. "
                    f"Assessment score: {total_score}% ({diff_label}). "
                    f"Boost applied: +{int(round(points_gain))} (capped by difficulty; max +{int(max_boost)}). "
                    f"Final Skills score: {int(round(new_skills))} (floor-limited)."
                )
        else:
            rationale = (
                f"Resume Skills score: {int(round(current_skills))}. "
                f"Assessment score: {total_score}% ({diff_label}). "
                f"No change applied (score already at or above assessment floor)."
            )

        # Attach lightweight telemetry and rationale so downstream / frontend can show Calculation Insights
        scoring_result["_assessment_boost_applied"] = True
        scoring_result["_assessment_boost_details"] = {
            "assessment_topic": assessment_topic,
            "total_score": total_score,
            "difficulty": difficulty,
            "max_boost_used": max_boost,
            "skills_score_before": float(current_skills),
            "skills_score_after": float(new_skills),
            "floor": float(floor),
            "scoring_rationale": rationale,
        }

        log.info(
            f"✅ Applied assessment-based boost to SkillsScore: "
            f"topic='{assessment_topic}', total_score={total_score}, difficulty={difficulty}, "
            f"old_skills={current_skills}, new_skills={new_skills}"
        )
        return scoring_result
    except Exception as e:
        log.warning(f"⚠️ Failed to apply assessment-based SkillsScore boost: {e}", exc_info=True)
        return scoring_result

def _validate_state_inputs(state: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate input state for security and completeness."""
    # Use centralized validation with resume-specific requirements
    required_fields = ["structured_resume"]
    is_valid, message = _validate_state_inputs(state, required_fields)
    
    if not is_valid:
        return False, message
    
    # Additional resume-specific validation
    structured_resume = state.get("structured_resume")
    if isinstance(structured_resume, dict) and "error" in structured_resume:
        return False, "Resume parsing error detected"
    
    # Validate chat history if present
    chat_history = state.get("chat_history", [])
    if not isinstance(chat_history, list):
        return False, "Invalid chat_history format"
    
    if len(chat_history) > MAX_CHAT_HISTORY_ITEMS:
        return False, f"Too many chat history items (max {MAX_CHAT_HISTORY_ITEMS})"
    
    return True, "Valid"

# ---------- Enhanced skill analysis functions ----------

def _analyze_skills_detailed(skills_data: List[Dict]) -> Dict[str, Any]:
    """Analyze skills for detailed scoring and role matching."""
    if not skills_data:
        return {"categories": {}, "summary": "No skills data", "dominant_domain": "unknown"}
    
    # Enhanced skill categorization
    skill_categories = {
        "frontend": {
            "keywords": ["React", "Vue", "Angular", "JavaScript", "TypeScript", "HTML", "CSS", "SASS", "Next.js", "Nuxt.js", "Tailwind", "Bootstrap"],
            "weight": 0.9,
            "roles": ["Frontend Developer", "UI Developer", "Frontend Engineer"]
        },
        "backend": {
            "keywords": ["Python", "Java", "Node.js", "C#", "Go", "Rust", "PHP", "Ruby", "Django", "Flask", "Express", "Spring", "ASP.NET"],
            "weight": 0.95,
            "roles": ["Backend Developer", "API Developer", "Backend Engineer"]
        },
        "database": {
            "keywords": ["SQL", "PostgreSQL", "MongoDB", "Redis", "MySQL", "Oracle", "Cassandra", "Elasticsearch", "DynamoDB"],
            "weight": 0.8,
            "roles": ["Database Developer", "Data Engineer"]
        },
        "cloud": {
            "keywords": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Jenkins", "CI/CD", "Lambda", "S3"],
            "weight": 1.0,
            "roles": ["Cloud Engineer", "DevOps Engineer", "Cloud Architect"]
        },
        "data_science": {
            "keywords": ["Python", "R", "TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn", "Jupyter", "Machine Learning", "AI"],
            "weight": 0.85,
            "roles": ["Data Scientist", "ML Engineer", "AI Engineer"]
        },
        "mobile": {
            "keywords": ["React Native", "Flutter", "iOS", "Android", "Swift", "Kotlin", "Xamarin", "Mobile"],
            "weight": 0.9,
            "roles": ["Mobile Developer", "iOS Developer", "Android Developer"]
        },
        "devops": {
            "keywords": ["Docker", "Kubernetes", "Jenkins", "GitLab", "GitHub Actions", "Terraform", "Ansible", "CI/CD"],
            "weight": 1.0,
            "roles": ["DevOps Engineer", "SRE", "Platform Engineer"]
        }
    }
    
    categorized_skills = {}
    skill_scores = {}
    
    for skill in skills_data:
        skill_name = skill.get("SkillName", "")
        proficiency = skill.get("Proficiency", "5/10")
        
        if not skill_name:
            continue
        
        # Extract proficiency score
        try:
            proficiency_score = int(proficiency.split("/")[0]) if "/" in proficiency else 5
        except:
            proficiency_score = 5
        
        # Categorize skill
        skill_category = "other"
        for category, config in skill_categories.items():
            if any(keyword.lower() in skill_name.lower() for keyword in config["keywords"]):
                skill_category = category
                break
        
        if skill_category not in categorized_skills:
            categorized_skills[skill_category] = []
            skill_scores[skill_category] = 0
        
        categorized_skills[skill_category].append({
            "name": skill_name,
            "proficiency": proficiency_score,
            "proficiency_text": proficiency
        })
        
        # Calculate weighted score for this category
        category_weight = skill_categories.get(skill_category, {}).get("weight", 0.5)
        skill_scores[skill_category] += proficiency_score * category_weight
    
    # Find dominant domain
    dominant_domain = max(skill_scores.items(), key=lambda x: x[1])[0] if skill_scores else "unknown"
    
    return {
        "categories": categorized_skills,
        "scores": skill_scores,
        "dominant_domain": dominant_domain,
        "summary": f"Dominant: {dominant_domain}, Categories: {len(categorized_skills)}"
    }

# ---------- Deterministic scoring functions ----------

def _generate_role_fit(structured_resume: Dict[str, Any], overall_score: float) -> List[str]:
    """Generate specific job roles based on detailed skills analysis."""
    roles = []
    
    # Get detailed skills analysis
    skills_analysis = _analyze_skills_detailed(structured_resume.get('skills', []))
    categories = skills_analysis.get("categories", {})
    dominant_domain = skills_analysis.get("dominant_domain", "unknown")
    
    # Extract experience for seniority assessment
    experience = structured_resume.get('experience', [])
    exp_years = 0
    senior_titles = []
    
    if isinstance(experience, list):
        for exp in experience:
            if isinstance(exp, dict):
                # Extract years from experience
                dates = exp.get('dates', '')
                if dates and '-' in dates:
                    try:
                        start_year = int(dates.split('-')[0].strip())
                        end_year = int(dates.split('-')[1].strip()) if dates.split('-')[1].strip().isdigit() else 2024
                        exp_years += (end_year - start_year)
                    except:
                        pass
                
                # Check for senior titles
                title = exp.get('job_title', '').lower()
                if any(word in title for word in ['senior', 'lead', 'principal', 'architect', 'manager']):
                    senior_titles.append(title)
    
    # Determine seniority level
    is_senior = overall_score >= 80 or exp_years >= 5 or len(senior_titles) > 0
    
    # Generate roles based on skill categories
    role_mappings = {
        "frontend": {
            "senior": ["Senior Frontend Developer", "Frontend Architect", "Lead UI Developer", "Principal Frontend Engineer"],
            "junior": ["Frontend Developer", "React Developer", "UI Developer", "Frontend Engineer"]
        },
        "backend": {
            "senior": ["Senior Backend Developer", "Backend Architect", "Lead Software Engineer", "Principal Backend Engineer"],
            "junior": ["Backend Developer", "Python Developer", "Java Developer", "Backend Engineer"]
        },
        "cloud": {
            "senior": ["Senior Cloud Engineer", "Cloud Architect", "Principal DevOps Engineer", "Lead Platform Engineer"],
            "junior": ["Cloud Engineer", "DevOps Engineer", "Platform Engineer", "Infrastructure Engineer"]
        },
        "data_science": {
            "senior": ["Senior Data Scientist", "Principal ML Engineer", "Lead AI Engineer", "Data Science Manager"],
            "junior": ["Data Scientist", "ML Engineer", "AI Engineer", "Data Analyst"]
        },
        "mobile": {
            "senior": ["Senior Mobile Developer", "Mobile Architect", "Lead Mobile Engineer", "Principal Mobile Developer"],
            "junior": ["Mobile Developer", "iOS Developer", "Android Developer", "React Native Developer"]
        },
        "devops": {
            "senior": ["Senior DevOps Engineer", "DevOps Architect", "Lead SRE", "Principal Platform Engineer"],
            "junior": ["DevOps Engineer", "SRE", "Platform Engineer", "Infrastructure Engineer"]
        }
    }
    
    # Generate roles based on dominant domain
    if dominant_domain in role_mappings:
        seniority = "senior" if is_senior else "junior"
        domain_roles = role_mappings[dominant_domain][seniority]
        roles.extend(domain_roles[:2])  # Take top 2 roles
    
    # Add cross-domain roles if multiple strong categories
    strong_categories = [cat for cat, skills in categories.items() if len(skills) >= 2]
    
    if len(strong_categories) >= 2:
        if "frontend" in strong_categories and "backend" in strong_categories:
            roles.append("Full-Stack Developer")
        if "backend" in strong_categories and "cloud" in strong_categories:
            roles.append("Cloud-Native Developer")
        if "data_science" in strong_categories and "backend" in strong_categories:
            roles.append("ML Platform Engineer")
    
    # Add specific technology-based roles
    all_skills = []
    for category_skills in categories.values():
        all_skills.extend([s["name"].lower() for s in category_skills])
    
    if "react" in all_skills and "node" in all_skills:
        roles.append("React + Node.js Developer")
    if "python" in all_skills and "django" in all_skills:
        roles.append("Django Developer")
    if "aws" in all_skills and "python" in all_skills:
        roles.append("AWS Python Developer")
    
    # Remove duplicates and limit to 5 roles
    unique_roles = []
    for role in roles:
        if role not in unique_roles:
            unique_roles.append(role)
    
    return unique_roles[:5]

# Deterministic functions removed - using LLM-only approach


# Deterministic fallback removed - using LLM-only approach


@traceable(name="resume_scorer_agent")
async def resume_scorer_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced resume scoring agent with enterprise security and agentic AI capabilities.
    
    Features:
    - Military-grade tenant isolation
    - Comprehensive PII redaction
    - Injection attack prevention
    - LLM-only analysis approach
    - Self-monitoring and learning
    - Multi-agent coordination
    - Performance optimization
    """
    
    # Extract tenant_id from state
    tenant_id = state.get("tenant_id", "default_tenant")
    
    # Use centralized logging
    log_context = create_log_context("resume_scorer", tenant_id)
    start_time = log_context["start_time"]
    request_id = log_context["request_id"]

    # Granular timing breakdown for performance monitoring
    timing_breakdown = {
        "input_optimization": 0.0,
        "chat_history_normalization": 0.0,
        "memory_setup": 0.0,
        "llm_scoring": 0.0,
        "result_processing": 0.0,
        "total": 0.0
    }

    llm_event: Optional[str] = None  # "timeout" | "error" | None
    chosen_method: str = "llm"  # LLM-only approach
    analysis_method: AnalysisMethod = "llm"
    confidence_score = 0.0

    # Use optimized inputs - only send required data
    opt_start = time.time()
    from core.utils import create_optimized_resume_scorer_inputs
    
    optimized_inputs = create_optimized_resume_scorer_inputs(state)
    structured_resume = optimized_inputs.get("structured_resume", {})
    chat_history = optimized_inputs.get("chat_history", [])
    timing_breakdown["input_optimization"] = time.time() - opt_start
    
    # Log data optimization
    original_size = len(str(structured_resume))
    log.info(f"📊 RESUME_SCORER: Processing resume data (optimized size: {original_size} chars)")
    
    # The structured_resume now contains only the fields needed for scoring:
    # - experience, skills, education, projects, certifications, Summary
    # - Much smaller payload than full resume!
    
    # Normalize and clamp chat history for prompt size & safety
    norm_start = time.time()
    chat_history = _normalize_chat_history(chat_history, MAX_CHAT_HISTORY_ITEMS, 2000)
    timing_breakdown["chat_history_normalization"] = time.time() - norm_start

    # Section 8 Issue 1: agent-level cache for same resume + chat
    from core.utils import get_cached_response, cache_response
    cache_input = {"structured_resume": structured_resume, "chat_history": chat_history}
    cached_result = get_cached_response("resume_scorer", cache_input)
    if cached_result:
        log.info("Cache hit for resume_scorer - returning cached result")
        return cached_result
    
    # Get tenant-scoped memory for adaptive behavior
    mem_start = time.time()
    scorer_memory = await get_scorer_memory(tenant_id)
    
    # Share context with other agents
    await scorer_memory.share_context("scoring_start_time", start_time)
    await scorer_memory.share_context("tenant_id", tenant_id)
    timing_breakdown["memory_setup"] = time.time() - mem_start
    
    # LLM-ONLY APPROACH: Always use LLM for scoring
    AgentLogger.log_method_selection(log_context, chosen_method, 1.0, 0.0)
    
    # OPTIMIZATION: Check for cached skills analysis from resume_assembler
    # Check both structured_resume and state (fallback) for cached analysis
    cached_skills_analysis = structured_resume.get("_cached_skills_analysis") or state.get("_cached_skills_analysis")
    if cached_skills_analysis:
        log.info(f"✅ RESUME_SCORER: Using cached skills analysis (dominant_domain: {cached_skills_analysis.get('dominant_domain', 'unknown')})")
    else:
        log.info("ℹ️ RESUME_SCORER: No cached skills analysis found, will compute during prompt generation")

    # Execute LLM analysis (pass uid for resume summary access and cached skills analysis)
    uid = state.get("uid")
    llm_start = time.time()
    scoring_result = await _execute_llm_scoring(
        structured_resume, chat_history, tenant_id, uid=uid, cached_skills_analysis=cached_skills_analysis
    )
    # NEW: If assessment_results are present (re-run flow), gently boost SkillsScore
    # so that strong assessments are reflected in the skills component.
    scoring_result = _apply_assessment_boost_to_scoring_result(scoring_result, state)
    # Expose telemetry fields from assessment boost (if any) back into state
    if scoring_result.get("_assessment_boost_applied"):
        state["assessment_boost_applied"] = True
        details = scoring_result.get("_assessment_boost_details", {})
        state["assessment_boost_details"] = details
        # Expose for Premium Insights → Calculation Insights (scoring rationale only)
        state["calculation_insights"] = {
            "skill_score": {
                "scoring_rationale": details.get("scoring_rationale", ""),
                "skills_score_before": details.get("skills_score_before"),
                "skills_score_after": details.get("skills_score_after"),
                "points_gain": (details.get("skills_score_after") or 0) - (details.get("skills_score_before") or 0),
                "assessment_score": details.get("total_score"),
                "difficulty": details.get("difficulty"),
                "max_boost_used": details.get("max_boost_used"),
            }
        }
    else:
        # Re-run with assessment_results but no boost (e.g. score < 60): still expose calculation_insights
        # so the UI always has something to show in Calculation Insights.
        assessment_results = state.get("assessment_results") or {}
        if isinstance(assessment_results, dict) and assessment_results.get("total_score") is not None:
            total_score = assessment_results.get("total_score")
            try:
                total_score_num = float(total_score)
            except (TypeError, ValueError):
                total_score_num = None
            resume_score = scoring_result.get("ResumeScore") or {}
            current_skills = _coerce_score(resume_score.get("SkillsScore"))
            if current_skills is None:
                current_skills = 0.0
            if total_score_num is not None and total_score_num < 60:
                no_boost_rationale = (
                    f"Assessment score: {int(round(total_score_num))}% (below 60). "
                    "No skill score boost applied. Improve assessment performance or complete more assessments to see a boost."
                )
            else:
                no_boost_rationale = (
                    "Assessment was considered but no skill score boost was applied."
                )
            state["calculation_insights"] = {
                "skill_score": {
                    "scoring_rationale": no_boost_rationale,
                    "skills_score_before": float(current_skills),
                    "skills_score_after": float(current_skills),
                    "points_gain": 0.0,
                    "assessment_score": total_score,
                    "difficulty": None,
                    "max_boost_used": None,
                }
            }
    timing_breakdown["llm_scoring"] = time.time() - llm_start
    analysis_method = "llm"
    
    # Keep analysis_method honest if LLM path failed
    if scoring_result.get("analysis_method") == "error":
        analysis_method = "error"
    
    # Calculate confidence and assess quality
    proc_start = time.time()
    confidence_score = _calculate_confidence_score(scoring_result, analysis_method)
    quality_score = _assess_scoring_quality(scoring_result)
    timing_breakdown["result_processing"] = time.time() - proc_start
    processing_time = time.time() - start_time
    timing_breakdown["total"] = processing_time
    
    # LLM-only approach - no deterministic fallback

    # Record analysis attempt for learning
    success = bool(
        scoring_result
        and confidence_score >= CONFIDENCE_THRESHOLD
        and quality_score >= 0.65
        and isinstance(scoring_result.get("ResumeScore", {}), dict)
        and all(
            k in scoring_result["ResumeScore"]
            for k in ("OverallScore",) + SCORE_CATEGORIES
        )
    )
    await scorer_memory.record_attempt(
        input_type="resume_scoring",
        method=analysis_method,
        success=success,
        confidence=confidence_score,
        processing_time=processing_time
    )
    
    # Update structured resume with scoring results
    structured_resume.update(scoring_result)
    structured_resume["_scoring_metadata"] = {
        "analysis_method": analysis_method,
        "chosen_method": chosen_method,
        "llm_event": llm_event,  # None | "timeout" | "error"
        "confidence_score": confidence_score,
        "quality_score": quality_score,
        "processing_time": processing_time,
        "timing_breakdown": timing_breakdown,  # NEW: Granular timing
        "tenant_id": _mask(tenant_id),
        "timestamp": time.time(),
        "request_id": request_id
    }
    
    # Share results with other agents
    await scorer_memory.share_context("scoring_confidence", confidence_score)
    await scorer_memory.share_context("scoring_quality", quality_score)
    await scorer_memory.share_context("analysis_complete", True)
    
    # Log completion using centralized logging with timing breakdown
    result_data = {
        "success": success,
        "confidence_score": confidence_score,
        "analysis_method": analysis_method,
        "chosen_method": chosen_method,
        "llm_event": llm_event,
        "quality_score": quality_score,
        "timing_breakdown": timing_breakdown  # NEW: Include timing breakdown
    }
    log_agent_completion(log_context, result_data, analysis_method, processing_time)
    
    # Log timing breakdown for performance monitoring
    log.info(
        f"⏱️ RESUME_SCORER timing breakdown: "
        f"input_opt={timing_breakdown['input_optimization']*1000:.1f}ms, "
        f"chat_norm={timing_breakdown['chat_history_normalization']*1000:.1f}ms, "
        f"memory={timing_breakdown['memory_setup']*1000:.1f}ms, "
        f"llm={timing_breakdown['llm_scoring']*1000:.1f}ms, "
        f"result_proc={timing_breakdown['result_processing']*1000:.1f}ms, "
        f"total={timing_breakdown['total']*1000:.1f}ms"
    )
    
    # Session Management Integration - MOVED TO BACKGROUND (non-blocking)
    uid = state.get("uid")
    if uid:
        # Fire-and-forget: Don't block response on session updates
        async def _update_session_background():
            try:
                session_start = time.time()
                log.info(f"🔄 RESUME_SCORER: Starting background session storage for UID={uid}")
                
                # Get or reuse existing session (ensures UID always uses same session ID)
                existing_session = await run_blocking_io(
                    session_manager.get_or_reuse_session,
                    owner_id=uid,
                    kind="candidate_pipeline",
                    owner_type="candidate",
                    initial_step="resume_score",
                    initial_data={"structured_resume": structured_resume}
                )
                log.info(f"✅ RESUME_SCORER: Using session: {existing_session.session_id} for UID={uid}")
                
                # Update session step
                session_update_result = await run_blocking_io(
                    session_manager.update_step,
                    session_id=existing_session.session_id,
                    step="resume_score",
                    data={"analysis_method": analysis_method, "confidence_score": confidence_score},
                    progress=0.5  # 50% complete after resume scoring
                )
                
                # Build resume_score payload (same structure for both storages)
                resume_score_payload = {
                    "ResumeScore": structured_resume.get("ResumeScore", {}),
                    "CandidateType": structured_resume.get("CandidateType"),
                    "ImprovementSuggestions": structured_resume.get("ImprovementSuggestions", []),
                    "StrengthAreas": structured_resume.get("StrengthAreas", []),
                    "analysis_method": analysis_method,
                    "confidence_score": confidence_score,
                    "quality_score": quality_score,
                    "processing_time": processing_time
                }
                metadata_rs = {
                    "agent": "resume_score",
                    "uid": uid,
                    "status": "scoring_complete",
                    "method": analysis_method
                }
                # Primary: store in dedicated per-UID document (uid_resume_score)
                await run_blocking_io(
                    upsert_resume_score_doc,
                    uid,
                    resume_score_payload,
                    metadata_rs
                )
                log.info(f"✅ RESUME_SCORER: Stored resume_score in uid_resume_score for UID={uid}")
                # Backward compat: also update chat session so existing session-based readers still work
                session_data = await run_blocking_io(get_chat_session, existing_session.session_id)
                if session_data:
                    session_data["resume_score"] = resume_score_payload
                    await run_blocking_io(
                        update_chat_session,
                        session_id=existing_session.session_id,
                        session_data=session_data,
                        metadata=metadata_rs
                    )
                    session_time = time.time() - session_start
                    log.info(f"✅ RESUME_SCORER: Background session update completed in {session_time*1000:.1f}ms for UID={uid}")
                else:
                    log.warning(f"⚠️ RESUME_SCORER: No existing chat session data found for session_id={existing_session.session_id}")
            except Exception as e:
                log.error(f"❌ RESUME_SCORER: Background session update failed for UID={uid}: {e}", exc_info=True)
        
        # Fire-and-forget: Start background task without awaiting
        asyncio.create_task(_update_session_background())
        log.info(f"🚀 RESUME_SCORER: Started background session update task for UID={uid} (non-blocking)")
    
    # Transform the structured_resume to expected callback format
    # OPTIMIZATION: Use debug-level logging to avoid performance impact
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"RESUME_SCORER: structured_resume keys: {list(structured_resume.keys()) if isinstance(structured_resume, dict) else 'Not a dict'}")
    
    # ✅ FIX: Add error handling around transformation to ensure we always return valid data
    try:
        transformed_resume = _transform_to_expected_format(structured_resume)
    except Exception as e:
        log.error(f"❌ RESUME_SCORER: Transformation failed with exception: {e}", exc_info=True)
        transformed_resume = {
            "resumeScore": {
                "breakdown": {"skills": 0, "presentation": 0},
                "total": 0,
                "rationale": f"Resume scoring completed but transformation failed: {str(e)}",
                "breakdown_rationale": {"skills": "Transformation failed.", "presentation": "Transformation failed."},
                "breakdown_how_to_improve": {"skills": "Take recommended assessments and explore recommended courses in the platform.", "presentation": "Use the resume download option in the platform for a more ATS-friendly resume."}
            }
        }
    
    # Ensure we always return a valid structure
    if not transformed_resume or not isinstance(transformed_resume, dict):
        log.warning(f"⚠️ RESUME_SCORER: Transformation failed, returning fallback structure")
        transformed_resume = {
            "resumeScore": {
                "breakdown": {"skills": 0, "presentation": 0},
                "total": 0,
                "rationale": "Resume scoring failed - unable to process data",
                "breakdown_rationale": {"skills": "Scoring unavailable.", "presentation": "Scoring unavailable."},
                "breakdown_how_to_improve": {"skills": "Take recommended assessments and explore recommended courses in the platform.", "presentation": "Use the resume download option in the platform for a more ATS-friendly resume."}
            }
        }
    
    # OPTIMIZATION: Only log return details at debug level
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"RESUME_SCORER: Returning transformed resume (type: {type(transformed_resume)})")
    
    # Ensure we always return valid data, never None
    if transformed_resume is None or not isinstance(transformed_resume, dict):
        log.warning("🔄 RESUME_SCORER: transformed_resume is None or invalid, using fallback")
        fallback_data = {
            "resumeScore": {
                "breakdown": {"skills": 0, "presentation": 0},
                "total": 0,
                "rationale": "Resume scoring failed - no valid data returned",
                "breakdown_rationale": {"skills": "Scoring unavailable.", "presentation": "Scoring unavailable."},
                "breakdown_how_to_improve": {"skills": "Take recommended assessments and explore recommended courses in the platform.", "presentation": "Use the resume download option in the platform for a more ATS-friendly resume."}
            }
        }
        log.info(f"🔄 RESUME_SCORER: Returning fallback data: {fallback_data}")
        return fallback_data
    
    # Fire-and-forget Novu notification: resume_parsed (Python side)
    try:
        from novu_notification_service import trigger_resume_parsed
        asyncio.create_task(trigger_resume_parsed(state))
    except Exception:
        # Notification failures must never break scoring flow
        pass

    # Include calculation_insights and boost fields in return so they appear in agent output / merged state
    if state.get("calculation_insights"):
        transformed_resume["calculation_insights"] = state["calculation_insights"]
    if state.get("assessment_boost_applied") is not None:
        transformed_resume["assessment_boost_applied"] = state["assessment_boost_applied"]
    if state.get("assessment_boost_details"):
        transformed_resume["assessment_boost_details"] = state["assessment_boost_details"]

    cache_response("resume_scorer", cache_input, transformed_resume)
    return transformed_resume

# Deterministic execution function removed - using LLM-only approach

async def _execute_llm_scoring(
    structured_resume: Dict[str, Any], 
    chat_history: List[Dict], 
    tenant_id: str,
    uid: Optional[str] = None,  # NEW: For accessing cached Gemini summary
    cached_skills_analysis: Optional[Dict[str, Any]] = None  # NEW: Pre-computed skills analysis from resume_assembler
) -> Dict[str, Any]:
    """Execute LLM-based resume scoring with security safeguards."""
    # Create log context for this function
    log_context = create_log_context("resume_scorer_llm", tenant_id)
    
    # Generate secure prompt (uses resume_summary from structured_resume - no blocking async calls)
    # OPTIMIZATION: Pass cached skills analysis to avoid redundant computation
    # OPTIMIZATION: Resume summary should already be in structured_resume from resume_summary_agent
    prompt_start = time.time()
    prompt = await generate_scoring_and_roles_prompt(
        structured_resume, chat_history, uid=uid, cached_skills_analysis=cached_skills_analysis
    )
    prompt_time = time.time() - prompt_start
    log.info(f"⏱️ RESUME_SCORER: Prompt generation took {prompt_time*1000:.1f}ms (no blocking async calls)")
    
    # Use configured max prompt size (default 30000) for better accuracy
    # Only truncate if absolutely necessary, and do it intelligently
    max_prompt_size = MAX_PROMPT_CHARS
    if len(prompt) > max_prompt_size:
        AgentLogger.log_warning(log_context, f"Prompt too large, truncating from {len(prompt)} to {max_prompt_size}")
        # Intelligent truncation: preserve instructions and key resume sections
        # Find where the resume JSON data starts
        json_start = prompt.find("**Full Resume Data:**")
        if json_start > 0:
            # Keep all instructions (before JSON) + truncate JSON intelligently
            instructions = prompt[:json_start + len("**Full Resume Data:**\n")]
            json_data = prompt[json_start + len("**Full Resume Data:**\n"):]
            # Calculate available space for JSON
            available_json_space = max_prompt_size - len(instructions)
            if available_json_space > 1000:  # Only truncate if we have reasonable space
                # Try to truncate JSON at a reasonable boundary (end of a section)
                # Find last complete JSON object/array boundary
                truncated_json = json_data[:available_json_space]
                # Find last complete JSON structure
                last_brace = truncated_json.rfind('}')
                last_bracket = truncated_json.rfind(']')
                last_comma = truncated_json.rfind(',')
                # Use the latest valid boundary
                cutoff = max(last_brace, last_bracket, last_comma)
                if cutoff > available_json_space * 0.8:  # If we found a good boundary
                    truncated_json = truncated_json[:cutoff + 1] + "\n}"
                else:
                    # Fallback: just truncate and close JSON
                    truncated_json = truncated_json[:available_json_space - 10] + "\n}"
                prompt = instructions + truncated_json
            else:
                # Not enough space, use simple truncation as fallback
                prompt = prompt[:max_prompt_size]
        else:
            # Fallback: simple truncation if we can't find JSON section
            prompt = prompt[:max_prompt_size]
    else:
        AgentLogger.log_debug(log_context, f"Prompt size: {len(prompt)} chars (within limit of {max_prompt_size})")
    
    # Sanitize prompt content using centralized security
    sanitized_prompt = sanitize_text_for_llm(prompt, True, True)
    
    # Optimized structured output for resume scoring
    from pydantic import BaseModel, Field
    from typing import List as _List, Optional as _Optional, Dict as _Dict
    from settings import settings as _settings

    # Simplified scoring model for better LLM performance (Skills + Format only; Experience and Education removed)
    # OverallScore = ATS (Applicant Tracking System) compatibility score
    class OptimizedResumeScore(BaseModel):
        OverallScore: int = Field(ge=0, le=100, description="ATS compatibility score 0-100 - how well the resume would parse and rank in Applicant Tracking Systems (standard sections, keyword visibility, clean format, parseability)")
        SkillsScore: int = Field(ge=0, le=100, description="Skills score 0-100")
        FormatScore: int = Field(ge=0, le=100, description="Format/presentation score 0-100")
        Explanation: str = Field(max_length=1200, description="Detailed overall scoring explanation: how OverallScore was determined and why points were deducted (e.g. if 80, where the missing 20 went—parseability, keyword visibility, format, content).")
        SkillsRationale: _Optional[str] = Field(default=None, max_length=500, description="Basis for Skills score and why points were deducted only. Do NOT include how-to-improve or platform nudge; that goes in SkillsHowToImprove. Keep under 350 chars.")
        FormatRationale: _Optional[str] = Field(default=None, max_length=500, description="Basis for Format score and why points were deducted only. Do NOT include how-to-improve or platform nudge; that goes in FormatHowToImprove. Keep under 350 chars.")
        SkillsHowToImprove: _Optional[str] = Field(default=None, max_length=450, description="Specific actionable steps to improve skills score. Nudge to platform: recommended assessments (validate level) and recommended courses (strengthen skills). Suggest according to situation. Do not repeat rationale. Keep under 300 chars.")
        FormatHowToImprove: _Optional[str] = Field(default=None, max_length=450, description="Specific actionable steps to improve format score. Nudge to platform: resume download option for ATS-friendly resume. Suggest according to situation. Do not repeat rationale. Keep under 300 chars.")

    class OptimizedResumeScoringResult(BaseModel):
        ResumeScore: OptimizedResumeScore
        ImprovementSuggestions: _List[str] = Field(
            min_length=0,
            max_length=4,
            description="2-4 concrete, actionable suggestions to improve the resume (e.g., 'Add quantifiable metrics to experience bullets', 'Expand technical skills section with proficiency levels', 'Include project impact and outcomes'). Provide at least 2 if possible."
        )
        StrengthAreas: _List[str] = Field(
            min_length=0,
            max_length=3,
            description="2-3 areas where the resume is already strong (e.g., 'Clear career progression', 'Strong technical depth', 'Quantified achievements in experience section'). Provide at least 2 if possible."
        )

    try:
        llm_call_start = time.time()
        log.info(f"⏱️ RESUME_SCORER: Calling LLM (timeout={TIMEOUT_SECONDS}s, model={getattr(_settings, 'GEMINI_MODEL', 'N/A')})")
        structured_result: OptimizedResumeScoringResult = await invoke_structured_llm(
            sanitized_prompt,
            OptimizedResumeScoringResult,
            task_type=TaskType.RESUME_ANALYSIS,
            preferred_model=_settings.GEMINI_MODEL,
            agent_name="resume_scorer",
            temperature=0.0,
            max_output_tokens=3500,
            timeout=float(TIMEOUT_SECONDS),
            raise_on_fallback=False,
        )
        llm_call_time = time.time() - llm_call_start
        log.info(f"⏱️ RESUME_SCORER: LLM call completed in {llm_call_time*1000:.1f}ms")
    
        if structured_result is None or getattr(structured_result, "ResumeScore", None) is None:
            log.warning("RESUME_SCORER: Structured output returned None or missing ResumeScore — using fallback.")
            raise ValueError("Structured output missing ResumeScore")
    
        # Truncate explanation if it's still too long (safety check)
        explanation = structured_result.ResumeScore.Explanation
        if len(explanation) > 1200:
            explanation = explanation[:1197] + "..."
        rs = structured_result.ResumeScore
        skills_rationale = (getattr(rs, "SkillsRationale", "") or "")[:350]
        format_rationale = (getattr(rs, "FormatRationale", "") or "")[:350]
        skills_how_to = (getattr(rs, "SkillsHowToImprove", "") or "").strip()[:300]
        format_how_to = (getattr(rs, "FormatHowToImprove", "") or "").strip()[:300]
        # Ensure how-to-improve is never empty: nudge to system features
        if not skills_how_to:
            skills_how_to = "Take the recommended assessments for your skills and explore recommended courses in the platform to improve this score."
        if not format_how_to:
            format_how_to = "Use the resume download option in the platform for a more ATS-friendly resume."

        # Extract improvement suggestions and strength areas from LLM
        improvement_suggestions = []
        strength_areas = []
        if hasattr(structured_result, "ImprovementSuggestions") and structured_result.ImprovementSuggestions:
            improvement_suggestions = [str(s).strip()[:300] for s in structured_result.ImprovementSuggestions if s]
        if hasattr(structured_result, "StrengthAreas") and structured_result.StrengthAreas:
            strength_areas = [str(s).strip()[:200] for s in structured_result.StrengthAreas if s]

        # Convert to expected format (include per-category rationales and how-to-improve; Skills + Format only)
        scoring_result = {
            "ResumeScore": {
                "OverallScore": structured_result.ResumeScore.OverallScore,
                "SkillsScore": structured_result.ResumeScore.SkillsScore,
                "FormatScore": structured_result.ResumeScore.FormatScore,
                "Explanation": explanation,
                "SkillsRationale": skills_rationale,
                "FormatRationale": format_rationale,
                "SkillsHowToImprove": skills_how_to,
                "FormatHowToImprove": format_how_to,
            },
            "CandidateType": "Professional",
            "ImprovementSuggestions": improvement_suggestions or [],
            "StrengthAreas": strength_areas or []
        }
        
    except ValueError as e:
        if "Structured output missing ResumeScore" in str(e):
            log.warning("RESUME_SCORER: Structured output was None or invalid — using fallback (no traceback).")
        else:
            raise
        skills_data = structured_resume.get('skills', [])
        skills_score = _calculate_skills_score_weighted(skills_data)
        scoring_result = {
            "ResumeScore": {
                "OverallScore": int(skills_score),
                "SkillsScore": int(skills_score),
                "FormatScore": 80,
                "Explanation": "Resume analysis completed with automated scoring based on skills assessment.",
                "SkillsRationale": "Score based on skills section, proficiency levels, and relevance. Taking recommended assessments and exploring recommended courses and other platform features can help improve this score. Fallback scoring used.",
                "FormatRationale": "Score based on resume structure, section clarity, and ATS parseability. Use the resume download option in the platform for a more ATS-friendly resume. Fallback scoring used.",
                "SkillsHowToImprove": "Take the recommended assessments for your skills and explore recommended courses in the platform to improve this score. Fallback.",
                "FormatHowToImprove": "Use the resume download option in the platform for a more ATS-friendly resume. Fallback.",
            },
            "CandidateType": "Professional",
            "ImprovementSuggestions": [
                "Add quantifiable metrics to experience bullets",
                "Expand skills section with proficiency levels where applicable",
            ],
            "StrengthAreas": [
                "Skills and technical competencies identified",
                "Structured resume format",
            ]
        }
    except asyncio.TimeoutError as e:
        log.warning(f"RESUME_SCORER: LLM timeout after {TIMEOUT_SECONDS}s — using fallback. Increase timeout or reduce prompt size.")
        log.exception("Resume scorer timeout (full traceback).")
        skills_data = structured_resume.get('skills', [])
        skills_score = _calculate_skills_score_weighted(skills_data)
        scoring_result = {
            "ResumeScore": {
                "OverallScore": int(skills_score),
                "SkillsScore": int(skills_score),
                "FormatScore": 80,
                "Explanation": "Resume analysis completed with automated scoring based on skills assessment.",
                "SkillsRationale": "Score based on skills section, proficiency levels, and relevance. Taking recommended assessments and exploring recommended courses and other platform features can help improve this score. Fallback scoring used.",
                "FormatRationale": "Score based on resume structure, section clarity, and ATS parseability. Use the resume download option in the platform for a more ATS-friendly resume. Fallback scoring used.",
                "SkillsHowToImprove": "Take the recommended assessments for your skills and explore recommended courses in the platform to improve this score. Fallback.",
                "FormatHowToImprove": "Use the resume download option in the platform for a more ATS-friendly resume. Fallback.",
            },
            "CandidateType": "Professional",
            "ImprovementSuggestions": [
                "Add quantifiable metrics to experience bullets",
                "Expand skills section with proficiency levels where applicable",
            ],
            "StrengthAreas": [
                "Skills and technical competencies identified",
                "Structured resume format",
            ]
        }
    except Exception as e:
        # Fallback if structured output fails (validation, API error, etc.)
        exc_type = type(e).__name__
        log.warning(f"RESUME_SCORER: Structured output failed ({exc_type}): {e} — using fallback.")
        try:
            from pydantic import ValidationError
            if isinstance(e, ValidationError):
                errs = e.errors() if callable(getattr(e, "errors", None)) else getattr(e, "errors", []) or []
                for err in errs:
                    loc = err.get("loc", ())
                    msg = err.get("msg", "")
                    log.warning(f"RESUME_SCORER: Validation error at {loc}: {msg}")
        except Exception:
            pass
        log.exception("Resume scorer structured output failed (full traceback); per-category rationales will be empty.")
        
        # Generate fallback response with basic scoring
        skills_data = structured_resume.get('skills', [])
        skills_score = _calculate_skills_score_weighted(skills_data)
        
        scoring_result = {
            "ResumeScore": {
                "OverallScore": int(skills_score),
                "SkillsScore": int(skills_score),
                "FormatScore": 80,
                "Explanation": "Resume analysis completed with automated scoring based on skills assessment.",
                "SkillsRationale": "Score based on skills section, proficiency levels, and relevance. Taking recommended assessments and exploring recommended courses and other platform features can help improve this score. Fallback scoring used.",
                "FormatRationale": "Score based on resume structure, section clarity, and ATS parseability. Use the resume download option in the platform for a more ATS-friendly resume. Fallback scoring used.",
                "SkillsHowToImprove": "Take the recommended assessments for your skills and explore recommended courses in the platform to improve this score. Fallback.",
                "FormatHowToImprove": "Use the resume download option in the platform for a more ATS-friendly resume. Fallback.",
            },
            "CandidateType": "Professional",
            "ImprovementSuggestions": [
                "Add quantifiable metrics to experience bullets",
                "Expand skills section with proficiency levels where applicable",
            ],
            "StrengthAreas": [
                "Skills and technical competencies identified",
                "Structured resume format",
            ]
        }
    
    # Validate and sanitize data
    validated_result = _validate_scoring_data(scoring_result)
    # Normalize ResumeScore (fills missing categories & ensures OverallScore)
    if "ResumeScore" in validated_result:
        validated_result["ResumeScore"] = _normalize_resume_score(validated_result["ResumeScore"])
    if not validated_result:
        raise ValueError("LLM response validation failed")
    
    return validated_result

def _calculate_skills_score_weighted(skills_data: List[Dict]) -> float:
    """Calculate weighted skills score based on market demand and proficiency."""
    if not skills_data:
        return 0.0
    
    # Market demand weights for different skill categories
    market_demand_weights = {
        "frontend": 0.9,
        "backend": 0.95,
        "cloud": 1.0,
        "data_science": 0.85,
        "mobile": 0.9,
        "devops": 1.0,
        "database": 0.8,
        "other": 0.5
    }
    
    # Get skills analysis
    skills_analysis = _analyze_skills_detailed(skills_data)
    categories = skills_analysis.get("categories", {})
    
    total_score = 0
    total_weight = 0
    
    for category, skills in categories.items():
        if skills:
            # Calculate average proficiency for this category
            avg_proficiency = sum(s["proficiency"] for s in skills) / len(skills)
            
            # Apply market demand weight
            weight = market_demand_weights.get(category, 0.5)
            
            # Calculate weighted score (0-100 scale)
            category_score = avg_proficiency * weight * 10
            total_score += category_score
            total_weight += weight
    
    # Return weighted average score
    return total_score / total_weight if total_weight > 0 else 0.0

# Hybrid scoring function removed - using LLM-only approach

# _merge_dedupe is now imported from core.utils

def _create_error_response(error_message: str) -> Dict[str, Any]:
    """Create standardized error response for resume scorer."""
    base_response = _create_error_response(error_message, "resume_scorer")
    return {
        **base_response,
        'ResumeScore': {
            'Error': error_message,
            'OverallScore': 0,
            'SkillsScore': 0,
            'FormatScore': 0,
            'Explanation': 'Analysis failed - using error response'
        },
        'CandidateType': 'Unknown',
        'ImprovementSuggestions': ['Please retry resume analysis'],
        'StrengthAreas': ['Analysis incomplete']
    }

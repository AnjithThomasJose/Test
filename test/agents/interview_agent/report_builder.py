"""
report_builder.py

Builds structured interview reports by combining evaluation results and LLM-generated summaries.

Responsibilities:
- Combine numerical scores from evaluator with narrative text from summary_generator
- Build a complete structured JSON object for UI consumption
- Extract and organize metrics, breakdowns, strengths, and concerns
- Provide safe defaults when data is missing
"""

import logging
from typing import Dict, Any, List, Optional

log = logging.getLogger(__name__)

# Allowed rating values for key_skill_breakdowns (UI expects only these three)
_ALLOWED_RATINGS = frozenset({"Strong", "Clear", "High"})

# Allowed overall signal levels (UI-facing)
_ALLOWED_SIGNAL_LEVELS = frozenset(
    {
        "Strong senior-level readiness",
        "Good mid-level performance",
        "Moderate performance",
        "Needs improvement",
        "Limited evaluation signal",
    }
)

# Topic -> 3 skill names for fallback when LLM report is not used
_TOPIC_SKILL_NAMES: Dict[str, List[str]] = {
    "menu development & engineering": [
        "Menu Design & Engineering",
        "Operational Knowledge",
        "Customer Focus",
    ],
    "communication": ["Clarity", "Structure", "Listening"],
    "communication test": ["Clarity", "Structure", "Listening"],
    "backend / platform engineering": [
        "System Design",
        "Technical Depth",
        "Scalability",
    ],
    "backend": ["System Design", "Technical Depth", "Code Quality"],
    "leadership": ["Decision Making", "Team Influence", "Strategic Thinking"],
    "technical": ["Technical Depth", "Problem Solving", "Application"],
}


def _get_topic_based_skill_names(interview_topic: str) -> List[str]:
    """Return exactly 3 skill names relevant to the interview topic."""
    if not interview_topic or not isinstance(interview_topic, str):
        return ["Problem Solving", "Communication", "Decision Judgment"]
    key = interview_topic.strip().lower()
    if key in _TOPIC_SKILL_NAMES:
        return _TOPIC_SKILL_NAMES[key]
    # Partial match: e.g. "Menu Development" -> menu development & engineering
    for topic_key, skills in _TOPIC_SKILL_NAMES.items():
        if topic_key in key or key in topic_key:
            return skills
    # Derive from topic: use topic as first skill (truncated), then two generic
    first = interview_topic.strip() if len(interview_topic.strip()) <= 35 else interview_topic.strip()[:32] + "..."
    return [first, "Knowledge & Application", "Communication & Clarity"]


def _normalize_ratings_to_strong_clear_high(
    key_skill_breakdowns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Normalize skill ratings to only Strong, Clear, or High."""
    result = []
    for item in key_skill_breakdowns:
        rating = (item.get("rating") or "").strip()
        if rating not in _ALLOWED_RATINGS:
            # Map common LLM outputs to allowed values
            r_lower = rating.lower()
            if r_lower in ("needs improvement", "moderate", "low", "weak"):
                rating = "Clear"
            elif r_lower in ("strong", "excellent"):
                rating = "Strong"
            elif r_lower in ("high", "good"):
                rating = "High"
            else:
                rating = "Clear"
        result.append({**item, "rating": rating})
    return result


def _normalize_signal_level(level: str, fallback: str) -> str:
    """
    Normalize LLM-provided overall_interview_signal.level into a small, safe set.

    We never surface raw technical phrasing like "Limited Signal Due to Technical Issues"
    to the user. Instead, we clamp to one of the UI-facing enums in _ALLOWED_SIGNAL_LEVELS.
    """
    if not level:
        return fallback

    candidate = level.strip()
    low = candidate.lower()

    # Direct matches to known values
    if candidate in _ALLOWED_SIGNAL_LEVELS:
        return candidate

    # Map common patterns
    if "limited" in low and ("signal" in low or "data" in low or "technical" in low):
        return "Limited evaluation signal"
    if "strong" in low:
        return "Strong senior-level readiness"
    if "good" in low or "mid-level" in low or "mid level" in low:
        return "Good mid-level performance"
    if "moderate" in low:
        return "Moderate performance"
    if "need" in low and "improve" in low:
        return "Needs improvement"

    # Fallback to score-based level decided by caller
    return fallback


def build_final_report(
    evaluation_result: Dict[str, Any],
    structured_report: Optional[Dict[str, Any]] = None,
    conversation_context: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Build a complete structured candidate-facing interview report matching UI format.
    
    Args:
        evaluation_result: Output from evaluator.run_full_evaluation()
            Expected keys: overall_score, category_scores, strengths, weaknesses, summary
        structured_report: Optional structured report from LLM (with overall_interview_signal, etc.)
        conversation_context: ConversationContext dict with topics_discussed, engagement_level, etc.
        metadata: Additional metadata (candidate_info, job_details, etc.)
    
    Returns:
        Structured report dict matching UI format:
        {
            "topic_evaluated": "string",
            "overall_interview_signal": {"level": "...", "description": "..."},
            "key_skill_breakdowns": [...],
            "observed_strengths": [...],
            "improvement_opportunities": [...],
            "personalized_recommendations": [...]
        }
    """
    if conversation_context is None:
        conversation_context = {}
    if metadata is None:
        metadata = {}
    # Extract interview topic from metadata
    interview_topic = metadata.get("interview_topic", "") or ""

    # Compute base metrics from evaluation_result (used by both LLM and fallback paths)
    overall_score = float(evaluation_result.get("overall_score", 0.0) or 0.0)
    category_scores = evaluation_result.get("category_scores", {})
    strengths = evaluation_result.get("strengths", []) or []
    weaknesses = evaluation_result.get("weaknesses", []) or []
    summary = evaluation_result.get("summary", "")

    # Score-based default overall signal level (used as fallback / normalization target)
    if overall_score >= 80:
        signal_level = "Strong senior-level readiness"
    elif overall_score >= 65:
        signal_level = "Good mid-level performance"
    elif overall_score >= 50:
        signal_level = "Moderate performance"
    else:
        signal_level = "Needs improvement"

    # Use structured_report if provided, otherwise build from evaluation_result
    if structured_report and isinstance(structured_report, dict):
        # Normalize ratings to only Strong / Clear / High (UI displays these three)
        key_skill_breakdowns_raw = structured_report.get("key_skill_breakdowns", [])
        key_skill_breakdowns = _normalize_ratings_to_strong_clear_high(
            key_skill_breakdowns_raw
        )

        overall_signal_raw = structured_report.get(
            "overall_interview_signal",
            {
                "level": signal_level,
                "description": "Report generation incomplete.",
            },
        )
        raw_level = (overall_signal_raw or {}).get("level", signal_level)
        normalized_level = _normalize_signal_level(raw_level, fallback=signal_level)
        overall_signal = {
            "level": normalized_level,
            "description": (overall_signal_raw or {}).get(
                "description",
                "Report generation incomplete.",
            ),
        }

        report = {
            "topic_evaluated": interview_topic,
            "overall_interview_signal": overall_signal,
            "key_skill_breakdowns": key_skill_breakdowns,
            "observed_strengths": structured_report.get("observed_strengths", []),
            "improvement_opportunities": structured_report.get(
                "improvement_opportunities", []
            ),
            "personalized_recommendations": structured_report.get(
                "personalized_recommendations", []
            ),
        }
        log.info("[REPORT_BUILDER] Using LLM-generated structured report")
        return report

    # Fallback: build minimal structure from evaluation_result
    log.warning(
        "[REPORT_BUILDER] No structured_report provided, building fallback from evaluation_result"
    )

    # Build skill breakdowns from category scores; skill names are topic-based
    problem_solving_score = category_scores.get("problem_solving", 0.0) or category_scores.get("technical", 0.0)
    communication_score = category_scores.get("communication", 0.0)
    decision_score = (category_scores.get("experience_relevance", 0.0) + category_scores.get("cultural_fit", 0.0)) / 2.0

    # Ratings must be one of: Strong, Clear, High only
    def _rating(score: float) -> str:
        if score >= 0.7:
            return "Strong"
        elif score >= 0.4:
            return "Clear"
        return "Clear"

    def _rating_high(score: float) -> str:
        if score >= 0.7:
            return "High"
        elif score >= 0.4:
            return "Strong"
        return "Clear"

    skill_names = _get_topic_based_skill_names(interview_topic)
    key_skill_breakdowns = [
        {
            "skill": skill_names[0],
            "rating": _rating(problem_solving_score),
            "description": "Based on problem-solving and technical evaluation."
        },
        {
            "skill": skill_names[1],
            "rating": _rating(communication_score),
            "description": "Based on communication clarity and structure."
        },
        {
            "skill": skill_names[2],
            "rating": _rating_high(decision_score),
            "description": "Based on experience relevance and cultural fit."
        }
    ]

    report = {
        "topic_evaluated": interview_topic,
        "overall_interview_signal": {
            "level": signal_level,
            "description": summary or f"Overall performance score: {overall_score:.1f}/100."
        },
        "key_skill_breakdowns": key_skill_breakdowns,
        "observed_strengths": strengths[:5] if isinstance(strengths, list) and len(strengths) > 0 else ["Demonstrated engagement in the interview process"],
        "improvement_opportunities": weaknesses[:5] if isinstance(weaknesses, list) and len(weaknesses) > 0 else ["Continue building domain-specific knowledge"],
        "personalized_recommendations": ["Practice articulating technical decisions with clear trade-offs", "Engage in mock interviews to refine communication"] if not strengths else []
    }

    log.info("[REPORT_BUILDER] Built fallback report structure")
    return report
    engagement = float(conversation_context.get("engagement_level", 0.5) or 0.5)
    interview_phase = conversation_context.get("interview_phase", "") or conversation_context.get("interview_stage", "")
    
    # Check for evaluation failure flag from guardrail
    failed_evaluation = evaluation_result.get("failed_evaluation", False)
    evaluation_error = evaluation_result.get("error", "")
    
    # Detect early phase - should limit report depth
    # MUST be defined before any use of is_early_phase or num_answers
    is_early_phase = interview_phase in ("topic_introduction", "opening_rapport", "stage_setting") or not interview_phase
    num_answers = evaluation_result.get("num_answers", 0) or 0
    
    # Extract overall score from evaluation
    # In early phases, prevent inflated or hallucinated scores
    overall_score = float(evaluation_result.get("overall_score", 0.0) or 0.0)
    
    # CRITICAL FIX: Don't cap scores for behavioral interviews (psychometric/personality/communication)
    # with sufficient answers, as they can have valid scores even in early phases
    interview_topic = metadata.get("interview_topic") or conversation_context.get("interview_topic") or ""
    topic_lower = str(interview_topic).lower()
    is_behavioral = any(kw in topic_lower for kw in [
        "psychometric", "psychological", "behavioral", "aptitude", "personality",
        "communication", "communication test", "communication skills"
    ])
    
    # Only cap scores for early-phase technical interviews with few answers
    # Behavioral interviews with 3+ answers should keep their scores
    should_cap_score = (is_early_phase or num_answers <= 1) and not (is_behavioral and num_answers >= 3)
    if should_cap_score:
        # Prevent inflated or hallucinated early-phase scores
        if overall_score > 30:
            overall_score = 30.0
    
    # If evaluation failed, log warning and prepare failure notice
    if failed_evaluation:
        log.warning(
            "[REPORT_BUILDER] Evaluation marked as failed: error=%s, num_answers=%s, overall_score=%.2f",
            evaluation_error,
            num_answers,
            overall_score
        )
    
    # Calculate AI confidence from evaluation quality
    # Higher confidence if we have multiple answers evaluated
    # Lower confidence in early phases or if evaluation failed
    # NOTE: ai_confidence is computed but NOT included in final report JSON
    if failed_evaluation:
        ai_confidence = 0.2  # Very low confidence when evaluation failed
    elif is_early_phase or num_answers == 0:
        ai_confidence = 0.3  # Low confidence in early stages
    else:
        ai_confidence = min(0.95, 0.5 + (num_answers * 0.1))
    
    # Sentiment must not be inferred from the score
    sentiment = "neutral"
    
    # Extract detailed breakdown from evaluation_result
    # Aggregate category scores from per_answer evaluations
    category_scores_map = {}
    all_strengths = []
    all_weaknesses = []
    
    per_answer = evaluation_result.get("per_answer", []) or []
    num_answers = evaluation_result.get("num_answers", 0) or 0
    
    # Try to extract category scores from per_answer evaluations
    for answer_eval in per_answer:
        eval_data = answer_eval.get("evaluation", {}) if isinstance(answer_eval, dict) else {}
        if isinstance(eval_data, dict):
            # Extract category scores
            cat_scores = eval_data.get("category_scores", []) or []
            for cat_score in cat_scores:
                if isinstance(cat_score, dict):
                    cat_name = cat_score.get("name", "")
                    cat_score_val = cat_score.get("score", 0.0)
                else:
                    # Pydantic model
                    cat_name = getattr(cat_score, "name", "")
                    cat_score_val = getattr(cat_score, "score", 0.0)
                
                if cat_name:
                    # Aggregate scores (average if multiple)
                    if cat_name not in category_scores_map:
                        category_scores_map[cat_name] = []
                    category_scores_map[cat_name].append(float(cat_score_val) if cat_score_val else 0.0)
            
            # Collect strengths and weaknesses
            strengths = eval_data.get("strengths", []) or []
            weaknesses = eval_data.get("weaknesses", []) or []
            if isinstance(strengths, list):
                all_strengths.extend([s for s in strengths if s and isinstance(s, str)])
            if isinstance(weaknesses, list):
                all_weaknesses.extend([w for w in weaknesses if w and isinstance(w, str)])
    
    # Also try to extract category scores from top-level evaluation_result
    if not category_scores_map:
        top_level_cat_scores = evaluation_result.get("category_scores") or []

        # New compact evaluator shape: dict of fractional scores in [0, 1]
        if isinstance(top_level_cat_scores, dict):
            for cat_name, frac in top_level_cat_scores.items():
                try:
                    frac_f = float(frac)
                except (TypeError, ValueError):
                    frac_f = 0.0
                # Clamp to [0, 1] then scale to 0–10 internal scale
                if frac_f < 0.0:
                    frac_f = 0.0
                if frac_f > 1.0:
                    frac_f = 1.0
                ten_point = frac_f * 10.0
                if cat_name not in category_scores_map:
                    category_scores_map[cat_name] = []
                category_scores_map[cat_name].append(ten_point)

        # Legacy shape: list of {"name": ..., "score": ...}
        else:
            for cat_score in top_level_cat_scores:
                if isinstance(cat_score, dict):
                    cat_name = cat_score.get("name", "")
                    cat_score_val = cat_score.get("score", 0.0)
                else:
                    # Pydantic model
                    cat_name = getattr(cat_score, "name", "")
                    cat_score_val = getattr(cat_score, "score", 0.0)
                
                if cat_name:
                    if cat_name not in category_scores_map:
                        category_scores_map[cat_name] = []
                    category_scores_map[cat_name].append(float(cat_score_val) if cat_score_val else 0.0)
    
    # Average category scores
    # Preserve exact category names from rubric (with spaces) for proper mapping
    detailed_breakdown = {}
    for cat_name, scores in category_scores_map.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            # Normalize to 0-100 scale (assuming max_score of 10)
            normalized_score = min(100.0, (avg_score / 10.0) * 100.0)
            # Keep original category name (with spaces) for exact matching
            detailed_breakdown[cat_name] = round(normalized_score, 2)
            # Also add lowercase version with underscores for backward compatibility
            detailed_breakdown[cat_name.lower().replace(" ", "_")] = round(normalized_score, 2)
    
    # Log category scores extraction for debugging
    if category_scores_map:
        log.debug(f"[REPORT] Found category_scores_map: {list(category_scores_map.keys())}")
    else:
        log.debug(f"[REPORT] No category_scores_map found. per_answer count: {len(per_answer)}")
        if per_answer:
            first_eval = per_answer[0].get("evaluation", {})
            log.debug(f"[REPORT] First evaluation keys: {list(first_eval.keys()) if isinstance(first_eval, dict) else 'not a dict'}")
            if isinstance(first_eval, dict):
                cat_scores = first_eval.get("category_scores", [])
                log.debug(f"[REPORT] First evaluation category_scores type: {type(cat_scores)}, value: {cat_scores}")
    
    # If no category scores found, derive from overall_score proportionally
    # BUT: In early phases, don't hallucinate scores - set to 0
    log.info(f"[REPORT] Checking detailed_breakdown: empty={not detailed_breakdown}, keys={list(detailed_breakdown.keys()) if detailed_breakdown else []}")
    log.info(f"[REPORT] Condition check: is_early_phase={is_early_phase}, num_answers={num_answers}, overall_score={overall_score}, interview_phase={interview_phase}")
    
    if not detailed_breakdown:
        if is_early_phase or num_answers == 0:
            # Early phase: ensure all category scores are zero (using fixed 5 categories)
            log.info(f"[REPORT] Early phase detected (phase={interview_phase}, num_answers={num_answers}), setting all scores to 0")
            detailed_breakdown = {
                "technical": 0.0,
                "problem solving": 0.0,  # Exact rubric name with space
                "communication": 0.0,
                "experience relevance": 0.0,  # Exact rubric name with space
                "cultural fit": 0.0  # Exact rubric name with space
            }
        elif overall_score > 0:
            log.info(f"[REPORT] No category scores found, deriving from overall_score ({overall_score}) proportionally")
            log.info(f"[REPORT] is_early_phase={is_early_phase}, num_answers={num_answers}, interview_phase={interview_phase}")
            # Distribute overall_score proportionally across fixed 5 categories
            # Use exact rubric category names (with spaces)
            standard_categories = ["technical", "problem solving", "communication", "experience relevance", "cultural fit"]
            score_per_category = overall_score / len(standard_categories)
            for cat in standard_categories:
                detailed_breakdown[cat] = round(score_per_category, 2)
            log.info(f"[REPORT] ✓ Distributed scores: {detailed_breakdown}")
        else:
            # No scores at all
            log.info(f"[REPORT] No overall_score, setting all category scores to 0")
            standard_categories = ["technical", "problem solving", "communication", "experience relevance", "cultural fit"]
            for cat in standard_categories:
                detailed_breakdown[cat] = 0.0
    else:
        log.info(f"[REPORT] detailed_breakdown already populated with {len(detailed_breakdown)} categories: {list(detailed_breakdown.keys())}")
    
    # Map fixed rubric categories to standard breakdown fields
    # Use exact category names from fixed rubric: "Technical", "Problem Solving", "Communication", "Experience Relevance", "Cultural Fit"
    
    # Technical (exact match from rubric)
    technical_skills = (
        detailed_breakdown.get("technical") or
        detailed_breakdown.get("technical_skills") or
        detailed_breakdown.get("technical_knowledge") or
        0.0
    )
    
    # Problem Solving (exact match from rubric - note the space, not underscore)
    problem_solving = (
        detailed_breakdown.get("problem solving") or  # Exact rubric name with space
        detailed_breakdown.get("problem_solving") or
        detailed_breakdown.get("problem_solving_ability") or
        detailed_breakdown.get("analytical_thinking") or
        0.0
    )
    
    # Communication (exact match from rubric)
    communication = (
        detailed_breakdown.get("communication") or
        detailed_breakdown.get("communication_skills") or
        detailed_breakdown.get("clarity") or
        0.0
    )
    
    # Experience Relevance (exact match from rubric - note the space, not underscore)
    experience_relevance = (
        detailed_breakdown.get("experience relevance") or  # Exact rubric name with space
        detailed_breakdown.get("experience_relevance") or
        detailed_breakdown.get("experience") or
        detailed_breakdown.get("relevant_experience") or
        0.0
    )
    
    # Cultural Fit (exact match from rubric - note the space, not underscore)
    cultural_fit = (
        detailed_breakdown.get("cultural fit") or  # Exact rubric name with space
        detailed_breakdown.get("cultural_fit") or
        detailed_breakdown.get("culture_fit") or
        detailed_breakdown.get("behavioral") or
        0.0
    )
    
    # Build standard breakdown with canonical mapping
    # Canonical mapping: evaluator rubric format -> output format
    # This ensures UI devs can reliably map between evaluator output and report output
    canonical_breakdown = {
        "Technical": technical_skills,
        "Problem Solving": problem_solving,
        "Communication": communication,
        "Experience Relevance": experience_relevance,
        "Cultural Fit": cultural_fit
    }
    
    # Also provide snake_case format for backward compatibility
    standard_breakdown = {
        "technical_skills": technical_skills,
        "experience_relevance": experience_relevance,
        "problem_solving": problem_solving,
        "communication": communication,
        "cultural_fit": cultural_fit
    }
    
    # Include canonical mapping in standard_breakdown for future-proofing
    standard_breakdown["_canonical"] = canonical_breakdown
    
    log.info(f"[REPORT] Final standard_breakdown before report: {standard_breakdown}")
    log.debug(f"[REPORT] Canonical mapping: {canonical_breakdown}")
    
    # ===== EXTRACT CANDIDATE_FEEDBACK FROM EVALUATOR (PRIMARY SOURCE) =====
    # The evaluator's candidate_feedback is the primary source of truth for all actionable content
    candidate_feedback = evaluation_result.get("candidate_feedback", {}) or {}
    
    # Extract strengths and weaknesses from candidate_feedback (preferred) or fallback to per_answer aggregation
    if isinstance(candidate_feedback, dict):
        # Use evaluator's strengths/weaknesses (from aggregate analysis)
        eval_strengths = candidate_feedback.get("strengths", []) or []
        eval_weaknesses = candidate_feedback.get("weaknesses", []) or []
        
        if isinstance(eval_strengths, list) and eval_strengths:
            strengths_list = [s for s in eval_strengths if s and isinstance(s, str)][:10]
            log.info("[REPORT_BUILDER] Using evaluator candidate_feedback for strengths")
        else:
            # Fallback: use evaluation_result strengths first, then per-answer aggregation
            eval_strengths_fallback = evaluation_result.get("strengths", []) or []
            if isinstance(eval_strengths_fallback, list) and eval_strengths_fallback:
                strengths_list = [s for s in eval_strengths_fallback if s and isinstance(s, str)][:10]
                log.info("[REPORT_BUILDER] Using evaluation_result strengths fallback (candidate_feedback.strengths missing)")
            else:
                strengths_list = list(dict.fromkeys(all_strengths))[:10]
                log.debug("[REPORT_BUILDER] Using per_answer aggregation for strengths (no evaluator strengths)")
        
        if isinstance(eval_weaknesses, list) and eval_weaknesses:
            concerns_list = [w for w in eval_weaknesses if w and isinstance(w, str)][:10]
            log.info("[REPORT_BUILDER] Using evaluator candidate_feedback for weaknesses")
        else:
            # Fallback: use evaluation_result weaknesses first, then per-answer aggregation
            eval_weaknesses_fallback = evaluation_result.get("weaknesses", []) or []
            if isinstance(eval_weaknesses_fallback, list) and eval_weaknesses_fallback:
                concerns_list = [w for w in eval_weaknesses_fallback if w and isinstance(w, str)][:10]
                log.info("[REPORT_BUILDER] Using evaluation_result weaknesses fallback (candidate_feedback.weaknesses missing)")
            else:
                concerns_list = list(dict.fromkeys(all_weaknesses))[:10]
                log.debug("[REPORT_BUILDER] Using per_answer aggregation for weaknesses (no evaluator weaknesses)")
    else:
        # Fallback: use per_answer aggregation
        eval_strengths = evaluation_result.get("strengths", []) or []
        eval_weaknesses = evaluation_result.get("weaknesses", []) or []
        if isinstance(eval_strengths, list) and eval_strengths:
            strengths_list = [s for s in eval_strengths if s and isinstance(s, str)][:10]
            log.info("[REPORT_BUILDER] Using evaluation_result strengths fallback")
        else:
            strengths_list = list(dict.fromkeys(all_strengths))[:10]
            log.debug("[REPORT_BUILDER] Using per_answer aggregation for strengths (no evaluator strengths)")
        
        if isinstance(eval_weaknesses, list) and eval_weaknesses:
            concerns_list = [w for w in eval_weaknesses if w and isinstance(w, str)][:10]
            log.info("[REPORT_BUILDER] Using evaluation_result weaknesses fallback")
        else:
            concerns_list = list(dict.fromkeys(all_weaknesses))[:10]
            log.debug("[REPORT_BUILDER] Using per_answer aggregation for weaknesses (no evaluator weaknesses)")
        log.debug("[REPORT_BUILDER] candidate_feedback not available, using per_answer aggregation")
    
    # In early phases, suppress only if nothing confident; otherwise keep evaluator-provided strengths/concerns
    if (is_early_phase or num_answers <= 1) and not strengths_list and not concerns_list:
        strengths_list = []
        concerns_list = []
    
    # ===== EXTRACT NARRATIVE TEXT FROM LLM SUMMARY (ONLY FOR REPORT TEXT) =====
    # LLM summary is ONLY used for narrative report text, not for actionable content
    report_text = ""
    if isinstance(llm_summary, dict):
        # New format: DetailedReport dict - extract only narrative text
        report_text = llm_summary.get("report", "")
    elif isinstance(llm_summary, str):
        # Old format: just text
        report_text = llm_summary
    
    # If no report text from LLM, use personalized_feedback from evaluator as fallback
    if not report_text and isinstance(candidate_feedback, dict):
        personalized_feedback = candidate_feedback.get("personalized_feedback", {}) or {}
        if isinstance(personalized_feedback, dict):
            report_text = personalized_feedback.get("long_summary", "") or personalized_feedback.get("short_summary", "")
            if report_text:
                log.info("[REPORT_BUILDER] Using evaluator personalized_feedback for report text (LLM summary not available)")
    
    # Enrich LLM narrative text if it's too short or seems incomplete
    # Use evaluator's personalized_feedback to enhance the narrative
    if report_text and len(report_text) < 200 and isinstance(candidate_feedback, dict):
        personalized_feedback = candidate_feedback.get("personalized_feedback", {}) or {}
        if isinstance(personalized_feedback, dict):
            long_summary = personalized_feedback.get("long_summary", "")
            short_summary = personalized_feedback.get("short_summary", "")
            
            # Append evaluator's feedback to enrich the narrative
            enrichment_text = long_summary or short_summary
            if enrichment_text and enrichment_text not in report_text:
                report_text = f"{report_text}\n\n{enrichment_text}"
                log.info("[REPORT_BUILDER] Enriched LLM narrative with evaluator personalized_feedback (LLM text was < 200 chars)")
    
    # If evaluation failed, prepend a warning notice to the report text
    if failed_evaluation:
        failure_notice = (
            "⚠️ EVALUATION SYSTEM NOTICE: The automated evaluation system encountered an issue "
            "and could not reliably assess this interview. The scores shown below may not accurately "
            "reflect candidate performance. Please review the interview transcript manually. "
        )
        if report_text:
            report_text = failure_notice + "\n\n" + report_text
        else:
            report_text = failure_notice
    
    # ===== EXTRACT ACTIONABLE CONTENT FROM EVALUATOR'S CANDIDATE_FEEDBACK =====
    # All actionable fields come from evaluator, NOT from LLM summary
    improvement_plan_7_days = []
    study_resources = []
    practice_questions = []
    projects = []
    final_recommendation = "Intermediate"
    confidence_score = 0.5
    skill_breakdown = {}
    
    if isinstance(candidate_feedback, dict):
        # Learning plan (7 days)
        learning_plan = candidate_feedback.get("learning_plan_7_days", []) or []
        if isinstance(learning_plan, list) and learning_plan:
            improvement_plan_7_days = learning_plan[:7]  # Ensure max 7 days
            log.info("[REPORT_BUILDER] Using evaluator candidate_feedback for learning_plan_7_days")
        else:
            log.debug("[REPORT_BUILDER] learning_plan_7_days not available in candidate_feedback")
        
        # Recommended resources
        resources = candidate_feedback.get("recommended_resources", []) or []
        if isinstance(resources, list) and resources:
            study_resources = resources[:10]  # Limit to 10 resources
            log.info("[REPORT_BUILDER] Using evaluator candidate_feedback for recommended_resources")
        else:
            log.debug("[REPORT_BUILDER] recommended_resources not available in candidate_feedback")
        
        # Practice questions
        questions = candidate_feedback.get("practice_questions", []) or []
        if isinstance(questions, list) and questions:
            practice_questions = questions[:10]  # Limit to 10 questions
            log.info("[REPORT_BUILDER] Using evaluator candidate_feedback for practice_questions")
        else:
            log.debug("[REPORT_BUILDER] practice_questions not available in candidate_feedback")
        
        # Projects
        proj_list = candidate_feedback.get("projects", []) or []
        if isinstance(proj_list, list) and proj_list:
            projects = proj_list[:5]  # Limit to 5 projects
            log.info("[REPORT_BUILDER] Using evaluator candidate_feedback for projects")
        else:
            log.debug("[REPORT_BUILDER] projects not available in candidate_feedback")
        
        # Overall competency level -> final_recommendation
        competency_level = candidate_feedback.get("overall_competency_level", "")
        if competency_level and isinstance(competency_level, str):
            # Map competency levels to final_recommendation format
            competency_map = {
                "novice": "Beginner",
                "intermediate": "Intermediate",
                "proficient": "Advanced",
                "advanced": "Advanced",
                "expert": "Ready for next round"
            }
            final_recommendation = competency_map.get(competency_level.lower(), "Intermediate")
            log.info(f"[REPORT_BUILDER] Using evaluator candidate_feedback for final_recommendation: {final_recommendation} (from {competency_level})")
        else:
            log.debug("[REPORT_BUILDER] overall_competency_level not available in candidate_feedback")
        
        # Confidence score
        conf_score = candidate_feedback.get("confidence_score")
        if conf_score is not None:
            # Clamp confidence score to valid range (0.0-1.0) to prevent unexpected values
            confidence_score = max(0.0, min(1.0, float(conf_score)))
            log.info(f"[REPORT_BUILDER] Using evaluator candidate_feedback for confidence_score: {confidence_score} (clamped to 0.0-1.0)")
        else:
            # Fallback to computed confidence (already clamped by ai_confidence calculation)
            confidence_score = ai_confidence
            log.debug("[REPORT_BUILDER] confidence_score not available in candidate_feedback, using computed value")
    else:
        log.warning("[REPORT_BUILDER] candidate_feedback not available in evaluation_result, using fallbacks")
        # Use computed ai_confidence as fallback
        confidence_score = ai_confidence
    
    # If evaluation failed, cap confidence score to reflect uncertainty
    if failed_evaluation:
        confidence_score = min(confidence_score, 0.3)
        log.debug("[REPORT_BUILDER] Capped confidence_score to 0.3 due to failed evaluation")
    
    # ===== BUILD PER-ANSWER FEEDBACK FROM PER_ANSWER_EXTENDED =====
    # Build per_answer_feedback from per_answer_extended (contains deep diagnostics)
    per_answer_feedback = []
    per_answer_extended = evaluation_result.get("per_answer_extended", []) or []
    
    if per_answer_extended:
        # Use per_answer_extended which contains deep diagnostics
        for answer_ext in per_answer_extended:
            if isinstance(answer_ext, dict):
                question = answer_ext.get("question", "Question not available")
                answer = answer_ext.get("answer", answer_ext.get("full_answer", ""))
                deep_diag = answer_ext.get("deep_diagnostics", {}) or {}
                eval_data = answer_ext.get("evaluation", {}) or {}
                
                # Extract ideal answer from deep diagnostics
                ideal_answer_text = deep_diag.get("ideal_answer", "")
                ideal_answer_list = []
                if ideal_answer_text:
                    # Split ideal answer into bullet points if it's a string
                    if isinstance(ideal_answer_text, str):
                        # Try to split by common separators
                        lines = ideal_answer_text.split("\n")
                        ideal_answer_list = [line.strip() for line in lines if line.strip()][:6]
                        if not ideal_answer_list:
                            ideal_answer_list = [ideal_answer_text[:200]]  # Fallback to first 200 chars
                    elif isinstance(ideal_answer_text, list):
                        ideal_answer_list = ideal_answer_text[:6]
                
                # Extract competency level
                skill_level = deep_diag.get("predicted_competency_level", "intermediate")
                # Map to readable format
                skill_level_map = {
                    "novice": "Beginner",
                    "intermediate": "Intermediate",
                    "proficient": "Advanced",
                    "advanced": "Advanced",
                    "expert": "Expert"
                }
                skill_level = skill_level_map.get(skill_level.lower(), "Intermediate")
                
                # Extract strengths and weaknesses for this answer
                answer_strengths = eval_data.get("strengths", []) or []
                answer_weaknesses = eval_data.get("weaknesses", []) or []
                
                # Build feedback entry
                feedback_entry = {
                    "question": question,
                    "candidate_answer": answer[:200] + "..." if len(answer) > 200 else answer,
                    "correct_what_was_good": ", ".join(answer_strengths) if answer_strengths else "Review your answer to identify strengths",
                    "missing_what_was_incomplete": ", ".join(answer_weaknesses) if answer_weaknesses else "Review your answer to identify gaps",
                    "ideal_answer": ideal_answer_list if ideal_answer_list else ["Key concept 1", "Best practice 2", "Additional detail 3"],
                    "skill_level": skill_level
                }
                per_answer_feedback.append(feedback_entry)
        
        log.info(f"[REPORT_BUILDER] Using evaluator per_answer_extended for per_answer_feedback ({len(per_answer_feedback)} entries)")
    elif per_answer:
        # Fallback: build from basic per_answer data
        for answer_data in per_answer:
            if isinstance(answer_data, dict):
                question = answer_data.get("question", "Question not available")
                answer = answer_data.get("answer", answer_data.get("full_answer", ""))
                eval_data = answer_data.get("evaluation", {}) or {}
                
                answer_strengths = eval_data.get("strengths", []) or []
                answer_weaknesses = eval_data.get("weaknesses", []) or []
                
                feedback_entry = {
                    "question": question,
                    "candidate_answer": answer[:200] + "..." if len(answer) > 200 else answer,
                    "correct_what_was_good": ", ".join(answer_strengths) if answer_strengths else "Review your answer to identify strengths",
                    "missing_what_was_incomplete": ", ".join(answer_weaknesses) if answer_weaknesses else "Review your answer to identify gaps",
                    "ideal_answer": ["Key concept 1", "Best practice 2", "Additional detail 3"],
                    "skill_level": "Intermediate"
                }
                per_answer_feedback.append(feedback_entry)
        
        log.debug("[REPORT_BUILDER] Using per_answer for per_answer_feedback (per_answer_extended not available)")
    
    # ===== BUILD SKILL BREAKDOWN FROM TOPIC_MASTERY =====
    # Build skill_breakdown from topic_mastery in evaluation_result
    topic_mastery = evaluation_result.get("topic_mastery", {}) or {}
    if isinstance(topic_mastery, dict) and topic_mastery:
        # Convert topic_mastery (0-1 scale) to skill breakdown format (0-100 scale)
        skill_breakdown = {}
        for skill, mastery_score in topic_mastery.items():
            if skill and isinstance(mastery_score, (int, float)):
                skill_breakdown[str(skill)] = {
                    "score": round(float(mastery_score) * 100, 2),
                    "explanation": f"Mastery level: {round(float(mastery_score) * 100, 0)}%",
                    "improvement_suggestions": []
                }
        log.info(f"[REPORT_BUILDER] Using evaluator topic_mastery for skill_breakdown ({len(skill_breakdown)} skills)")
    else:
        # Fallback: build from category scores
        if detailed_breakdown:
            skill_breakdown = {}
            for cat_name, score in detailed_breakdown.items():
                if cat_name and score:
                    skill_breakdown[str(cat_name)] = {
                        "score": round(float(score), 2),
                        "explanation": f"Performance in {cat_name}",
                        "improvement_suggestions": []
                    }
            log.debug("[REPORT_BUILDER] Using detailed_breakdown for skill_breakdown (topic_mastery not available)")
    
    # Build final report structure with all candidate-facing fields
    # All actionable content comes from evaluator's candidate_feedback
    report = {
        "candidate_name": candidate_name,
        "job_title": job_title,
        "position": position,
        "session_metrics": {
            "engagement": round(engagement, 2),
            "sentiment": sentiment,
            "confidence_score": round(confidence_score, 2)  # From evaluator
        },
        "overall_score": round(overall_score, 2),
        "detailed_breakdown": standard_breakdown,
        "strengths": strengths_list,  # From evaluator candidate_feedback
        "concerns": concerns_list,  # From evaluator candidate_feedback
        "interview_phase": interview_phase,
        "num_answers": num_answers,
        # Candidate-facing fields (ALL from evaluator, NOT from LLM)
        "report": report_text,  # Narrative text (from LLM summary or evaluator personalized_feedback)
        "per_answer_feedback": per_answer_feedback,  # From evaluator per_answer_extended
        "skill_breakdown": skill_breakdown,  # From evaluator topic_mastery
        "improvement_plan_7_days": improvement_plan_7_days,  # From evaluator candidate_feedback
        "study_resources": study_resources,  # From evaluator candidate_feedback
        "practice_questions": practice_questions,  # From evaluator candidate_feedback
        "projects": projects,  # From evaluator candidate_feedback
        "final_recommendation": final_recommendation,  # From evaluator candidate_feedback.overall_competency_level
        # Backward compatibility
        "llm_report_text": report_text
    }
    
    # Add evaluation status metadata if evaluation failed
    if failed_evaluation:
        report["evaluation_status"] = "failed"
        report["evaluation_error"] = evaluation_error
        report["evaluation_warning"] = (
            "The evaluation system could not reliably assess this interview. "
            "Scores may not accurately reflect candidate performance."
        )
    
    # Debug mode logging (if enabled)
    DEBUG_REPORT_MODE = metadata.get("debug_report_mode", False)
    if DEBUG_REPORT_MODE:
        log.info("[DEBUG_REPORT_MODE] Extracted answers count: %d", len(per_answer))
        log.info("[DEBUG_REPORT_MODE] Per-answer feedback count: %d", len(per_answer_feedback))
        log.info("[DEBUG_REPORT_MODE] Skill breakdown keys: %s", list(skill_breakdown.keys()))
        log.info("[DEBUG_REPORT_MODE] Improvement plan days: %d", len(improvement_plan_7_days))
        log.info("[DEBUG_REPORT_MODE] Study resources count: %d", len(study_resources))
        log.info("[DEBUG_REPORT_MODE] Practice questions count: %d", len(practice_questions))
        log.info("[DEBUG_REPORT_MODE] Projects count: %d", len(projects))
        log.info("[DEBUG_REPORT_MODE] Final recommendation: %s", final_recommendation)
        log.info("[DEBUG_REPORT_MODE] Confidence score: %s", confidence_score)
        log.info("[DEBUG_REPORT_MODE] Candidate feedback available: %s", bool(candidate_feedback))
    
    # Summary logging
    log.info("[REPORT_BUILDER] Report built successfully:")
    log.info(f"  - Narrative report: {'Yes' if report_text else 'No'} ({len(report_text)} chars)")
    log.info(f"  - Per-answer feedback: {len(per_answer_feedback)} entries")
    log.info(f"  - Learning plan: {len(improvement_plan_7_days)} days")
    log.info(f"  - Resources: {len(study_resources)} items")
    log.info(f"  - Practice questions: {len(practice_questions)} items")
    log.info(f"  - Projects: {len(projects)} items")
    log.info(f"  - Final recommendation: {final_recommendation}")
    
    return report


"""
Context Aggregator - Utility to load all candidate data.

Single responsibility: Aggregate data from all sources.
Reusable across all career coach components.

Loads resume, gap_doc, assessments_doc (and optionally session) in parallel for speed.
Optional in-memory TTL cache to avoid repeated loads within the same session window.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Tuple

from chroma import (
    get_resume_doc,
    get_gap_doc,
    get_resume_score_doc,
    get_assessments_doc,
    fetch_structured_resume,
    get_chat_session,
    normalize_assessment_status,
)
from core.utils import run_blocking_io, get_missing_skills_flat

log = logging.getLogger(__name__)

# In-memory cache: (uid, session_id or "") -> (context_dict, expiry_time)
_context_cache: Dict[Tuple[str, str], Tuple[Dict[str, Any], float]] = {}
_CONTEXT_CACHE_TTL_SECONDS = 60
_CONTEXT_CACHE_MAX_ENTRIES = 200


def _extract_location_from_resume(resume_data: Optional[Dict[str, Any]]) -> str:
    """Extract location from resume (ContactDetails or work experience)."""
    if not resume_data or not isinstance(resume_data, dict):
        return ""
    contact = resume_data.get("ContactDetails") or resume_data.get("contact_details", {})
    if isinstance(contact, dict):
        loc = contact.get("Location") or contact.get("location", "")
        if loc:
            return str(loc).strip()
    work = resume_data.get("work_experience") or resume_data.get("experience", [])
    if work and isinstance(work, list) and len(work) > 0:
        first = work[0]
        if isinstance(first, dict):
            loc = first.get("location") or first.get("Location", "")
            if loc:
                return str(loc).strip()
    return ""


def _derive_currency_from_location(location: str) -> tuple:
    """
    Map location string to (currency_symbol, region_name).
    Default: Indian Rupees if location missing or unrecognized.
    """
    if not location or not location.strip():
        return ("₹", "Indian")
    loc_lower = location.lower().strip()
    # India / Indian cities
    india_keywords = ["india", "bangalore", "mumbai", "delhi", "hyderabad", "chennai", "pune", "kolkata", "gurgaon", "noida", "gurugram", "kochi"]
    if any(kw in loc_lower for kw in india_keywords):
        return ("₹", "Indian")
    # US
    us_keywords = ["usa", "united states", "u.s.", "new york", "san francisco", "seattle", "austin", "boston", "chicago", "los angeles", "california", "texas"]
    if any(kw in loc_lower for kw in us_keywords):
        return ("$", "US")
    # UK
    uk_keywords = ["uk", "united kingdom", "london", "manchester", "birmingham", "edinburgh"]
    if any(kw in loc_lower for kw in uk_keywords):
        return ("£", "UK")
    # UAE
    uae_keywords = ["uae", "dubai", "abu dhabi", "emirates"]
    if any(kw in loc_lower for kw in uae_keywords):
        return ("AED", "UAE")
    # Singapore
    if "singapore" in loc_lower:
        return ("SGD", "Singapore")
    # Australia
    au_keywords = ["australia", "sydney", "melbourne"]
    if any(kw in loc_lower for kw in au_keywords):
        return ("AUD", "Australian")
    # Canada
    if "canada" in loc_lower or "toronto" in loc_lower or "vancouver" in loc_lower:
        return ("CAD", "Canadian")
    # Default: Indian market (platform default)
    return ("₹", "Indian")


def derive_assessment_results(assessments_doc: Optional[Dict[str, Any]]) -> list:
    """
    Derive assessment_results from assessments_doc (top-level, assessment_history, or completed assessment_plan).
    Single source of truth so mentor and get_career_advice see completion as soon as it's stored.
    """
    if not assessments_doc:
        return []
    results = list(assessments_doc.get("assessment_results") or [])
    if results:
        return results
    history = assessments_doc.get("assessment_history") or []
    if isinstance(history, list):
        for e in history:
            if isinstance(e, dict) and (e.get("total_score") is not None or (e.get("result") or {}).get("score") is not None):
                results.append({
                    "topic": e.get("topic"),
                    "assessment_topic": e.get("topic"),
                    "total_score": e.get("total_score") or (e.get("result") or {}).get("score"),
                    "score": e.get("total_score") or (e.get("result") or {}).get("score"),
                    "result": e.get("result"),
                })
    if results:
        return results
    ar = assessments_doc.get("assessment_recommender") or {}
    plan = ar.get("assessment_plan") or []
    for item in plan:
        if isinstance(item, dict) and normalize_assessment_status(item.get("status")) == "completed":
            results.append({
                "topic": item.get("topic"),
                "assessment_topic": item.get("topic"),
                "total_score": item.get("score"),
                "score": item.get("score"),
                "completed_at": item.get("completed_at"),
            })
    return results


def _empty_context() -> Dict[str, Any]:
    """Return a fresh empty context skeleton (same keys as load_comprehensive_context)."""
    return {
        "resume_data": {},
        "resume_summary": "",
        "resume_score": {},
        "skills": [],
        "skill_gaps": [],
        "career_paths": [],
        "career_advice": [],
        "improvement_recommendations": [],
        "assessment_results": [],
        "assessment_reports": {},
        "assessment_performance": {},
        "assessment_topics": [],
        "assessment_plan": [],
        "pending_assessments": [],
        "course_recommendations": [],
        "market_insights": {},
        "salary_trends": {},
        "trending_skills": [],
        "user_interests": [],
        "career_goals": [],
        "aspirations": {},
        "conversation_insights": {},
        "career_chat_history": [],
        "candidate_location": "",
        "candidate_currency": "₹",
        "salary_region": "Indian",
        "relevant_jobs": [],
        "role_fit_suggestions": [],
        "overall_performance": {},
        "strengths": [],
        "weaknesses": [],
        "last_updated": None,
    }


async def load_comprehensive_context(uid: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Load ALL available candidate information from every source.
    This is the single source of truth for candidate context.
    Runs resume_doc, gap_doc, assessments_doc (and session data if session_id) in parallel.
    Results are cached in memory for CONTEXT_CACHE_TTL_SECONDS to avoid repeated loads.
    
    Args:
        uid: User ID
        session_id: Optional session ID for additional context
        
    Returns:
        Dictionary containing all candidate data from all sources
    """
    cache_key = (uid, session_id or "")
    now = time.time()
    if cache_key in _context_cache:
        cached_ctx, expiry = _context_cache[cache_key]
        if now < expiry:
            log.debug(f"✅ Context cache hit for uid={uid}")
            return dict(cached_ctx)
        else:
            del _context_cache[cache_key]

    context = _empty_context()
    
    try:
        # Run all data fetches in parallel (no dependency between them)
        if session_id:
            resume_doc, gap_doc, resume_score_doc, assessments_doc, session_data, structured_resume = await asyncio.gather(
                run_blocking_io(get_resume_doc, uid),
                run_blocking_io(get_gap_doc, uid),
                run_blocking_io(get_resume_score_doc, uid),
                run_blocking_io(get_assessments_doc, uid),
                run_blocking_io(get_chat_session, session_id, uid),
                run_blocking_io(fetch_structured_resume, uid, None, session_id),
                return_exceptions=True,
            )
            if isinstance(session_data, Exception):
                log.debug(f"Could not load session: {session_data}")
                session_data = None
            if isinstance(structured_resume, Exception):
                log.debug(f"Could not fetch structured resume from session: {structured_resume}")
                structured_resume = None
        else:
            resume_doc, gap_doc, resume_score_doc, assessments_doc = await asyncio.gather(
                run_blocking_io(get_resume_doc, uid),
                run_blocking_io(get_gap_doc, uid),
                run_blocking_io(get_resume_score_doc, uid),
                run_blocking_io(get_assessments_doc, uid),
            )
            session_data, structured_resume = None, None

        # 1. Resume data (prefer structured_resume from session if available)
        if resume_doc and not isinstance(resume_doc, Exception):
            context["resume_data"] = resume_doc.get("structured_resume") or resume_doc
            context["resume_summary"] = context["resume_data"].get("resume_summary", "")
            context["skills"] = context["resume_data"].get("skills", [])
        if structured_resume and isinstance(structured_resume, dict):
            context["resume_data"] = structured_resume
            context["resume_summary"] = structured_resume.get("resume_summary", "")
            context["skills"] = structured_resume.get("skills", [])

        location = _extract_location_from_resume(context.get("resume_data"))
        currency, region = _derive_currency_from_location(location)
        context["candidate_location"] = location
        context["candidate_currency"] = currency
        context["salary_region"] = region

        # 2. Gap Analysis & Career Advisor Data
        if gap_doc and not isinstance(gap_doc, Exception):
            # Career Advisor Output
            career_advisor = gap_doc.get("skill_and_career_advisor", {})
            gap_analysis = career_advisor.get("raw_skill_gap_analysis_output", {})
            
            context["skill_gaps"] = get_missing_skills_flat(gap_analysis)
            context["career_paths"] = gap_analysis.get("career_paths", [])
            context["career_advice"] = gap_analysis.get("career_advice", [])
            context["improvement_recommendations"] = gap_analysis.get("improvement_recommendations", [])
            
            # Market & Course Recommender
            market_data = gap_doc.get("market_and_course_recommender", {})
            context["course_recommendations"] = market_data.get("course_recommendations", [])
            context["market_insights"] = market_data.get("market_insights", {})
            context["salary_trends"] = market_data.get("salary_trends", {})
            context["trending_skills"] = market_data.get("trending_skills", [])
            
            # Resume Score: prefer dedicated uid_resume_score doc, then gap_doc
            resume_score_data = {}
            if resume_score_doc and not isinstance(resume_score_doc, Exception) and isinstance(resume_score_doc, dict):
                resume_score_data = resume_score_doc
            if not resume_score_data and gap_doc:
                resume_score_data = gap_doc.get("resume_score", {}) or {}
            # Support both formats: ResumeScore (SkillsScore, FormatScore) or resumeScore (breakdown.skills, breakdown.presentation)
            rs = resume_score_data.get("ResumeScore") or resume_score_data.get("resumeScore") or resume_score_data
            context["resume_score"] = rs if isinstance(rs, dict) else {}
        elif resume_score_doc and not isinstance(resume_score_doc, Exception) and isinstance(resume_score_doc, dict):
            # No gap_doc but we have dedicated uid_resume_score doc
            rs = resume_score_doc.get("ResumeScore") or resume_score_doc.get("resumeScore") or resume_score_doc
            context["resume_score"] = rs if isinstance(rs, dict) else {}
        
        # 3. Assessment Data (already loaded in parallel)
        if assessments_doc and not isinstance(assessments_doc, Exception):
            context["assessment_results"] = derive_assessment_results(assessments_doc)
            
            # Assessment Reports
            if "report_generator" in assessments_doc:
                report_data = assessments_doc["report_generator"]
                context["assessment_reports"] = report_data.get("report", {})
                context["assessment_performance"] = report_data.get("assessment_performance", {})
            
            # Extract assessment topics
            for assessment in context["assessment_results"]:
                topic = assessment.get("assessment_topic") or assessment.get("topic")
                if topic:
                    context["assessment_topics"].append(topic)
            
            # Load Assessment Plan with Status
            assessment_recommender_data = assessments_doc.get("assessment_recommender", {})
            if assessment_recommender_data:
                assessment_plan = assessment_recommender_data.get("assessment_plan", [])
                context["assessment_plan"] = assessment_plan
                # Pending = plan items not completed. Track by assessment_id (from history) first; fallback to status.
                history_list = assessments_doc.get("assessment_history") or []
                completed_ids = {
                    h.get("assessment_id") for h in history_list
                    if isinstance(h, dict) and h.get("assessment_id")
                }
                def _is_completed(item):
                    if not isinstance(item, dict):
                        return False
                    plan_id = item.get("assessment_id") or item.get("assessment_id_ref")
                    if plan_id and plan_id in completed_ids:
                        return True
                    return normalize_assessment_status(item.get("status")) == "completed"
                pending_assessments = [
                    item for item in assessment_plan
                    if isinstance(item, dict) and not _is_completed(item)
                ]
                context["pending_assessments"] = pending_assessments
        
        # 4. User Interests / goals / aspirations / chat history (from session already loaded in parallel)
        if session_data and isinstance(session_data, dict):
            interest_data = session_data.get("interest_filler", {})
            context["user_interests"] = interest_data.get("user_interests", [])
            context["career_goals"] = session_data.get("career_goals", [])
            context["aspirations"] = session_data.get("career_aspirations", {})
            context["conversation_insights"] = session_data.get("conversation_insights", {})
            context["career_chat_history"] = session_data.get("career_chat_history", []) or []
        
        # 5. Load Role Fit Suggestions from gap_doc
        if gap_doc:
            career_advisor = gap_doc.get("skill_and_career_advisor", {})
            gap_analysis = career_advisor.get("raw_skill_gap_analysis_output", {})
            context["role_fit_suggestions"] = gap_analysis.get("role_fit_suggestions", [])
            
            # Also check resume score for role suggestions (prefer uid_resume_score doc)
            resume_score_data = (resume_score_doc if resume_score_doc and not isinstance(resume_score_doc, Exception) else None) or gap_doc.get("resume_score", {})
            if resume_score_data and isinstance(resume_score_data, dict):
                role_suggestions = resume_score_data.get("RoleSuggestions", [])
                if role_suggestions:
                    context["relevant_jobs"] = [
                        {
                            "title": role.get("Role", ""),
                            "match_score": role.get("Score", 0.7),
                            "rationale": role.get("Rationale", "")
                        }
                        for role in role_suggestions[:5]
                    ]
        
        # 6. Calculate Overall Performance Metrics
        if context["assessment_results"]:
            scores = []
            for assessment in context["assessment_results"]:
                score = assessment.get("total_score") or assessment.get("score")
                if isinstance(score, (int, float)):
                    scores.append(score)
            
            if scores:
                avg_score = sum(scores) / len(scores)
                context["overall_performance"] = {
                    "average_score": avg_score,
                    "assessment_count": len(scores),
                    "performance_level": (
                        "Outstanding" if avg_score >= 90 else
                        "Excellent" if avg_score >= 80 else
                        "Good" if avg_score >= 70 else
                        "Fair" if avg_score >= 50 else
                        "Needs Improvement"
                    )
                }
        
        # 7. Extract Strengths & Weaknesses from Reports
        if context["assessment_reports"]:
            report = context["assessment_reports"]
            strengths_text = report.get("strengths_analysis", "")
            weaknesses_text = report.get("development_areas", "")
            
            if isinstance(strengths_text, str) and strengths_text:
                context["strengths"] = [s.strip() for s in strengths_text.split("\n") if s.strip()][:5]
            if isinstance(weaknesses_text, str) and weaknesses_text:
                context["weaknesses"] = [w.strip() for w in weaknesses_text.split("\n") if w.strip()][:5]
        
        # 8. Set last updated timestamp
        from datetime import datetime
        context["last_updated"] = datetime.utcnow().isoformat()
        
        log.info(
            f"✅ Loaded comprehensive context for uid={uid}: "
            f"resume={bool(context['resume_data'])}, "
            f"assessments={len(context['assessment_results'])}, "
            f"skill_gaps={len(context['skill_gaps'])}"
        )
        
        # Cache for TTL to avoid repeated loads in the same session window
        _context_cache[cache_key] = (context, time.time() + _CONTEXT_CACHE_TTL_SECONDS)
        if len(_context_cache) > _CONTEXT_CACHE_MAX_ENTRIES:
            # Evict expired entries; if still over limit, drop oldest by expiry
            now = time.time()
            expired = [k for k, (_, ex) in _context_cache.items() if ex <= now]
            for k in expired:
                del _context_cache[k]
            if len(_context_cache) > _CONTEXT_CACHE_MAX_ENTRIES:
                by_expiry = sorted(_context_cache.items(), key=lambda x: x[1][1])
                for k, _ in by_expiry[: len(_context_cache) - _CONTEXT_CACHE_MAX_ENTRIES]:
                    del _context_cache[k]
        
    except Exception as e:
        log.error(f"Error loading comprehensive context: {e}", exc_info=True)
    
    return context


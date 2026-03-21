"""
Function Tools for Career Chatbot - Smart data retrieval with agent fallback.

Strategy:
1. Check existing data from ChromaDB first (fast, no LLM cost)
2. Only call agents if data is missing or explicitly requested
3. Cache results for future use
"""

import json
import logging
import asyncio
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from agents.market_and_course_recommender import market_and_course_recommender_agent
from agents.prompt_generator import generate_salary_only_prompt
from agents.skill_and_career_advisor import skill_and_career_advisor_agent
from agents.resume_score import resume_scorer_agent
from agents.enhanced_role_fit_agent import enhanced_role_fit_agent
from agents.assessment_recommender import (
    UnifiedAssessmentRecommender,
    assessment_recommender_agent,
    generate_unique_assessment_id,
    assign_unique_ids_to_assessments,
)
from agents.course_knowledge_base import CourseKnowledgeBase
from chroma import (
    fetch_structured_resume,
    get_gap_doc,
    get_resume_score_doc,
    get_assessments_doc,
    upsert_assessments_doc,
    get_chat_session,
    update_chat_session,
    query_job_descriptions,
    normalize_assessment_status,
)
from core.utils import create_agent_state, run_blocking_io, _generate_request_id, _calculate_processing_time, get_missing_skills_flat, _to_text
from models.llm_invoker import invoke_llm
from .context_aggregator import (
    load_comprehensive_context,
    derive_assessment_results,
    _extract_location_from_resume,
    _derive_currency_from_location,
)
from .conversation_analyzer import extract_aspirations_from_conversation, extract_goals_from_message
import time

log = logging.getLogger(__name__)

# Cache freshness threshold (hours)
DATA_FRESHNESS_HOURS = 24  # Consider data fresh if less than 24 hours old


def _normalize_date(date_str: Optional[str]) -> Optional[str]:
    """
    Convert natural language dates to ISO format (YYYY-MM-DD).
    Handles common phrases like "day after tomorrow", "next week", etc.
    
    Args:
        date_str: Natural language date string or ISO date string
        
    Returns:
        ISO format date string (YYYY-MM-DD) or None if invalid/unparseable
    """
    if not date_str:
        return None
    
    date_str = date_str.strip().lower()
    
    # If already in ISO format, return as-is
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str
    
    # Get current date
    today = datetime.utcnow().date()
    
    # Common natural language patterns
    patterns = {
        r'today': today,
        r'tomorrow': today + timedelta(days=1),
        r'day after tomorrow': today + timedelta(days=2),
        r'next week': today + timedelta(days=7),
        r'in (\d+) days?': lambda m: today + timedelta(days=int(m.group(1))),
        r'in (\d+) weeks?': lambda m: today + timedelta(weeks=int(m.group(1))),
        r'in (\d+) months?': lambda m: today + timedelta(days=int(m.group(1)) * 30),
    }
    
    for pattern, value in patterns.items():
        match = re.search(pattern, date_str)
        if match:
            if callable(value):
                try:
                    result_date = value(match)
                except:
                    continue
            else:
                result_date = value
            return result_date.isoformat()
    
    # Try to parse as date string (e.g., "January 23, 2026", "May 17th", "May 17, 2024")
    try:
        # Remove ordinal suffixes (st, nd, rd, th) for easier parsing
        date_str_clean = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
        
        # Common date formats (try with and without year)
        formats = [
            '%Y-%m-%d',           # 2024-05-17
            '%m/%d/%Y',           # 05/17/2024
            '%d/%m/%Y',           # 17/05/2024
            '%B %d, %Y',          # May 17, 2024
            '%b %d, %Y',          # May 17, 2024
            '%B %d %Y',           # May 17 2024
            '%b %d %Y',           # May 17 2024
            '%B %d',              # May 17 (assume current year)
            '%b %d',              # May 17 (assume current year)
        ]
        
        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str_clean, fmt)
                # If no year in format, assume current year (or next year if date has passed)
                if '%Y' not in fmt:
                    parsed_date = parsed.replace(year=datetime.utcnow().year).date()
                    # If date has passed this year, use next year
                    if parsed_date < today:
                        parsed_date = parsed.replace(year=datetime.utcnow().year + 1).date()
                    return parsed_date.isoformat()
                else:
                    return parsed.date().isoformat()
            except ValueError:
                continue
    except Exception as e:
        log.debug(f"Date parsing error: {e}")
    
    # If we can't parse it, return None (will be stored as-is but logged)
    log.warning(f"⚠️ Could not normalize date: {date_str}, storing as-is")
    return date_str


def _estimate_task_duration(task: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> float:
    """
    Estimate task duration in hours based on task type, difficulty, and context.
    
    Args:
        task: Task dictionary with title, type, difficulty, etc.
        context: Optional context with course/assessment data
    
    Returns:
        Estimated hours to complete the task
    """
    title_lower = task.get("title", "").lower()
    task_type = task.get("type", "").lower()
    difficulty = task.get("difficulty", "medium").lower()
    
    # Assessment tasks
    if "assessment" in title_lower or task_type == "assessment" or task.get("assessment_topic"):
        assessment_time = task.get("assessment_time_minutes", 30)
        # Add prep time based on difficulty
        prep_multiplier = {"easy": 0.5, "medium": 1.0, "hard": 2.0, "expert": 3.0}.get(difficulty, 1.0)
        return (assessment_time / 60.0) + (prep_multiplier * 0.5)  # 0.5-1.5 hours prep
    
    # Course tasks
    if "course" in title_lower or task_type == "course":
        # Check if we have course duration from context
        if context and "course_recommendations" in context:
            # Try to match course by title
            for course in context.get("course_recommendations", []):
                course_title = course.get("title", "").lower()
                if course_title and any(word in title_lower for word in course_title.split()[:3]):
                    duration_str = course.get("duration", "")
                    # Parse duration (e.g., "30 days", "Self-paced", "10 hours")
                    if "hour" in duration_str.lower():
                        match = re.search(r'(\d+)', duration_str)
                        if match:
                            hours = float(match.group(1))
                            return hours
                    elif "day" in duration_str.lower():
                        match = re.search(r'(\d+)', duration_str)
                        if match:
                            days = float(match.group(1))
                            return days * 2  # Assume 2 hours per day
                    elif "week" in duration_str.lower():
                        match = re.search(r'(\d+)', duration_str)
                        if match:
                            weeks = float(match.group(1))
                            return weeks * 10  # Assume 10 hours per week
        
        # Default course estimates based on difficulty
        base_hours = {"easy": 5, "medium": 10, "hard": 20, "expert": 40}.get(difficulty, 10)
        return base_hours
    
    # Project/practice tasks
    if any(kw in title_lower for kw in ["project", "build", "practice", "implement", "create", "develop"]):
        return {"easy": 4, "medium": 8, "hard": 16, "expert": 32}.get(difficulty, 8)
    
    # Reading/study tasks
    if any(kw in title_lower for kw in ["read", "study", "learn", "review", "watch", "tutorial"]):
        return {"easy": 2, "medium": 4, "hard": 8, "expert": 16}.get(difficulty, 4)
    
    # Default estimate based on difficulty
    return {"easy": 2, "medium": 4, "hard": 8, "expert": 16}.get(difficulty, 4)


def _calculate_due_date_from_duration(effort_hours: float, start_date: Optional[datetime] = None) -> str:
    """
    Calculate due date from effort hours, assuming reasonable daily commitment.
    
    Args:
        effort_hours: Estimated hours to complete
        start_date: Start date (defaults to today)
    
    Returns:
        ISO format date string
    """
    if start_date is None:
        start_date = datetime.utcnow()
    
    # Assume 2-3 hours per day commitment (realistic for working professionals)
    hours_per_day = 2.5
    days_needed = max(1, int(effort_hours / hours_per_day))
    
    # Add buffer: 20% extra time for unexpected delays
    days_with_buffer = int(days_needed * 1.2)
    
    # Cap at reasonable maximum (90 days)
    days_with_buffer = min(days_with_buffer, 90)
    
    due_date = start_date + timedelta(days=days_with_buffer)
    return due_date.date().isoformat()


# ============================================================================
# HELPER: Suggest assessment topics based on user profile
# ============================================================================

async def suggest_assessment_topics(uid: str, session_id: Optional[str] = None) -> List[str]:
    """
    Intelligently suggest assessment topics based on user's profile, skill gaps, and career goals.
    This helps the chatbot proactively recommend relevant assessments.
    
    Args:
        uid: User ID
        session_id: Optional session ID
    
    Returns:
        List of suggested topic names
    """
    try:
        context = await load_comprehensive_context(uid, session_id)
        suggested_topics = []
        
        # 1. Suggest topics based on skill gaps
        skill_gaps = context.get("skill_gaps", [])
        if skill_gaps:
            # Take top 3-5 skill gaps as assessment topics
            suggested_topics.extend([gap for gap in skill_gaps[:5] if gap])
        
        # 2. Suggest topics based on career paths
        career_paths = context.get("career_paths", [])
        if career_paths:
            for path in career_paths[:3]:  # Top 3 career paths
                if isinstance(path, dict):
                    title = path.get("title", "")
                    required_skills = path.get("required_skills", [])
                    if title:
                        suggested_topics.append(title)
                    # Add key required skills
                    suggested_topics.extend([skill for skill in required_skills[:2] if skill])
        
        # 3. Suggest topics based on missing skills from career advice
        improvement_recs = context.get("improvement_recommendations", [])
        if improvement_recs:
            # Extract skill names from improvement recommendations
            for rec in improvement_recs[:3]:
                if isinstance(rec, str):
                    # Try to extract skill names (simple heuristic)
                    words = rec.split()
                    for word in words:
                        if len(word) > 4 and word[0].isupper():  # Likely a skill name
                            suggested_topics.append(word)
        
        # 4. Remove duplicates and limit to 5-7 topics
        unique_topics = list(dict.fromkeys(suggested_topics))[:7]
        
        log.info(f"💡 Suggested {len(unique_topics)} assessment topics based on profile: {unique_topics}")
        return unique_topics
        
    except Exception as e:
        log.warning(f"Error suggesting assessment topics: {e}")
        return []


# ============================================================================
# HELPER: Lightweight course lookup for topic-specific asks
# ============================================================================

def _dedupe_courses(courses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate courses by course_id, then URL, then title while preserving order."""
    seen = set()
    deduped = []
    for course in courses:
        key = course.get("course_id") or course.get("url") or course.get("title") or ""
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(course)
    return deduped


def _synthetic_course_id() -> str:
    """Generate a stable id for courses that lack course_id (e.g. from KB or merged context)."""
    return f"course_{_generate_request_id()}"


def _norm_course_item(c: Dict[str, Any], default_course_id: Optional[str] = None) -> Dict[str, Any]:
    """Normalize a course item to canonical shape: course_id, title, url, provider, relevance_score, description."""
    course_id = c.get("course_id") or c.get("id") or default_course_id or _synthetic_course_id()
    title = c.get("title") or c.get("course") or c.get("name") or "Unknown"
    url = c.get("url") or c.get("link") or ""
    provider = c.get("provider") or c.get("platform") or ""
    score = c.get("relevance_score") or c.get("similarity_score") or c.get("score")
    if score is None:
        score = 0.8
    if isinstance(score, (int, float)) and score > 1:
        score = score / 100.0
    relevance_score = float(score) if score is not None else 0.8
    description = c.get("description") or c.get("summary") or ""
    return {
        "course_id": course_id,
        "title": title,
        "url": url,
        "provider": provider,
        "relevance_score": relevance_score,
        "description": description,
    }


def _build_topic_queries(topics: List[str], structured_resume: Optional[Dict[str, Any]]) -> List[str]:
    """
    Build simple semantic search queries for the course KB based on topics and profile.
    Keeps it lightweight to avoid running the full market_and_course_recommender agent.
    """
    if not topics:
        return []
    
    skills: List[str] = []
    if structured_resume and isinstance(structured_resume, dict):
        raw_skills = structured_resume.get("skills", []) or []
        # Normalize skills to strings; some resumes may store skills as dicts
        for s in raw_skills:
            if isinstance(s, str):
                skills.append(s)
            elif isinstance(s, dict):
                # try common keys
                name = s.get("name") or s.get("skill") or s.get("title")
                if isinstance(name, str):
                    skills.append(name)
            else:
                # fallback string conversion
                skills.append(str(s))
    
    queries = []
    skills_snippet = ", ".join(skills[:6]) if skills else ""
    for topic in topics:
        topic_str = topic.strip()
        if not topic_str:
            continue
        if skills_snippet:
            queries.append(f"{topic_str} courses for {skills_snippet}")
        queries.append(f"{topic_str} courses")
    return queries


def _quick_course_lookup_by_topics(
    topics: List[str],
    structured_resume: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Fast path: pull course recommendations directly from the course knowledge base
    using lightweight semantic search, without invoking the full course agent.
    """
    if not topics:
        return []
    
    kb = CourseKnowledgeBase()
    queries = _build_topic_queries(topics, structured_resume)
    all_courses: List[Dict[str, Any]] = []
    
    for query in queries:
        try:
            all_courses.extend(kb.search_courses(query, top_k=5, filters={"type": "course"}))
        except Exception as e:
            log.warning(f"KB course search failed for query '{query}': {e}")
    
    deduped = _dedupe_courses(all_courses)
    return [_norm_course_item(c) for c in deduped]


# ============================================================================
# SMART DATA RETRIEVAL - Check existing data first
# ============================================================================

async def get_course_recommendations(
    uid: str,
    session_id: Optional[str] = None,
    force_refresh: bool = False,
    topics: Optional[List[str]] = None,
    courses_only: bool = False
) -> Dict[str, Any]:
    """
    Get course recommendations - checks existing data first, calls agent if needed.
    
    Args:
        uid: User ID
        session_id: Optional session ID
        force_refresh: If True, always call agent (ignore existing data)
        topics: Optional list of specific topics to target courses for (fast path)
        courses_only: If True, skip market/salary analysis and return courses only
    
    Returns:
        Dict with course_recommendations, market_insights, source (existing|fresh)
    """
    try:
        start_time = time.time()
        request_id = _generate_request_id()
        topics_list = [
            t.strip() for t in (topics or []) if isinstance(t, str) and t.strip()
        ]
        courses_only_mode = courses_only or bool(topics_list)
        
        log.info(f"🎓 Getting course recommendations for uid={uid} (force_refresh={force_refresh})")
        
        # Step 1: Check existing data first (unless force_refresh)
        if not force_refresh and not courses_only_mode:
            context = await load_comprehensive_context(uid, session_id)
            existing_courses = context.get("course_recommendations", [])
            existing_insights = context.get("market_insights", [])
            existing_salary_trends = context.get("salary_trends", {})
            existing_career_paths = context.get("career_paths", [])
            existing_skill_demand = context.get("skill_demand_analysis", {})
            existing_gap_analysis = context.get("skill_gaps", {})
            
            if existing_courses:
                log.info(f"✅ Using existing course data ({len(existing_courses)} courses)")
                normalized_courses = [_norm_course_item(c) for c in existing_courses[:10]]
                processing_time = _calculate_processing_time(start_time)
                return {
                    # Standard agent format (same as market_and_course_recommender_agent - ALL fields)
                    "status": "completed",
                    "node": "get_course_recommendations",
                    "output": {
                        "raw_skill_gap_analysis_output": existing_gap_analysis,
                        "market_insights": existing_insights,
                        "salary_trends": existing_salary_trends,
                        "career_paths": existing_career_paths,
                        "skill_demand_analysis": existing_skill_demand,
                        "course_recommendations": normalized_courses
                    },
                    
                    # Metadata fields (same as agents)
                    "recommendations": {},
                    "raw_skill_gap_analysis_output": existing_gap_analysis,
                    "analysis_status": "success",
                    "analysis_method": "existing_data",
                    "confidence_score": 0.8,
                    "processing_time": processing_time,
                    "request_id": request_id,
                    "enhanced_course_flow": False,
                    "validated_courses_count": len(existing_courses),
                    
                    # Original fields for backward compatibility
                    "success": True,
                    "course_recommendations": normalized_courses,
                    "market_insights": existing_insights,
                    "salary_trends": existing_salary_trends,
                    "career_paths": existing_career_paths,
                    "skill_demand_analysis": existing_skill_demand,
                    "total_courses": len(existing_courses),
                    "source": "existing",
                    "message": "Here are your personalized course recommendations based on your profile."
                }
        
        # Step 1b: Fast path for topic-specific or courses-only requests (no full context / market analysis)
        if courses_only_mode:
            log.info(f"🎯 Fast path: topic-specific courses-only lookup (topics={topics_list}) — skipping context/gap load")
            
            # Load resume only (needed for KB lookup); do NOT load comprehensive context or gap_doc
            structured_resume = await run_blocking_io(fetch_structured_resume, uid, None, session_id)
            if not structured_resume:
                processing_time = _calculate_processing_time(start_time)
                return {
                    "status": "error",
                    "node": "get_course_recommendations",
                    "error": "No resume data found. Please upload your resume first.",
                    "processing_time": processing_time,
                    "request_id": request_id
                }
            
            # KB topic search only; no merge with existing context (avoids load_comprehensive_context + get_gap_doc)
            new_courses = _quick_course_lookup_by_topics(topics_list or ["courses"], structured_resume)
            merged = _dedupe_courses(new_courses or [])[:15]
            normalized_merged = [_norm_course_item(c) for c in merged[:10]]
            if merged:
                processing_time = _calculate_processing_time(start_time)
                return {
                    "status": "completed",
                    "node": "get_course_recommendations",
                    "output": {
                        "raw_skill_gap_analysis_output": {},
                        "market_insights": [],
                        "salary_trends": {},
                        "career_paths": [],
                        "skill_demand_analysis": {},
                        "course_recommendations": normalized_merged
                    },
                    "recommendations": {},
                    "raw_skill_gap_analysis_output": {},
                    "analysis_status": "success",
                    "analysis_method": "kb_topic_search",
                    "confidence_score": 0.7,
                    "processing_time": processing_time,
                    "request_id": request_id,
                    "enhanced_course_flow": False,
                    "validated_courses_count": len(merged),
                    "success": True,
                    "course_recommendations": normalized_merged,
                    "market_insights": [],
                    "salary_trends": {},
                    "career_paths": [],
                    "skill_demand_analysis": {},
                    "total_courses": len(merged),
                    "source": "courses_only",
                    "message": "Here are course recommendations based on the topic(s) you asked for and your profile."
                }
            
            log.info("⚠️ Courses-only KB lookup returned no results; falling back to full course agent.")
            # If no results, fall through to full agent below
        
        # Step 2: Data missing or force_refresh - call agent
        log.info(f"🔄 Calling course recommender agent (existing data not available or refresh requested)")
        
        # Load resume data
        structured_resume = await run_blocking_io(fetch_structured_resume, uid, None, session_id)
        if not structured_resume:
            processing_time = _calculate_processing_time(start_time)
            return {
                "status": "error",
                "node": "get_course_recommendations",
                "error": "No resume data found. Please upload your resume first.",
                "processing_time": processing_time,
                "request_id": request_id
            }
        
        # Get gap analysis
        gap_doc = await run_blocking_io(get_gap_doc, uid)
        skill_gap_analysis = {}
        if gap_doc:
            career_advisor = gap_doc.get("skill_and_career_advisor", {})
            skill_gap_analysis = career_advisor.get("raw_skill_gap_analysis_output", {})
        
        # Prepare state for market_and_course_recommender_agent
        base_state = {
            "uid": uid,
            "tenant_id": uid,
            "session_id": session_id,
            "structured_resume": structured_resume,
            "raw_skill_gap_analysis_output": skill_gap_analysis,
            "user_interests": []
        }
        state = create_agent_state(base_state, "market_and_course_recommender")
        
        # Call existing agent
        result = await market_and_course_recommender_agent(state, tenant_id=uid)
        
        # Extract course recommendations (use agent's output format - preserve ALL fields)
        output = result.get("output", {})
        courses = output.get("course_recommendations", [])
        normalized_courses_fresh = [_norm_course_item(c) for c in courses[:10]]
        market_insights = output.get("market_insights", [])
        salary_trends = output.get("salary_trends", {})
        career_paths = output.get("career_paths", [])
        skill_demand_analysis = output.get("skill_demand_analysis", {})
        raw_skill_gap = output.get("raw_skill_gap_analysis_output", {}) or skill_gap_analysis
        
        processing_time = _calculate_processing_time(start_time)
        
        log.info(f"✅ Course recommender returned {len(courses)} courses (fresh data)")
        
        return {
            # Standard agent format (same as market_and_course_recommender_agent - ALL fields)
            "status": "completed",
            "node": "get_course_recommendations",
            "output": {
                "raw_skill_gap_analysis_output": raw_skill_gap,
                "market_insights": market_insights,
                "salary_trends": salary_trends,
                "career_paths": career_paths,
                "skill_demand_analysis": skill_demand_analysis,
                "course_recommendations": normalized_courses_fresh
            },
            
            # Metadata fields (same as agents - preserve all from result)
            "recommendations": result.get("recommendations", {}),
            "raw_skill_gap_analysis_output": raw_skill_gap,
            "analysis_status": result.get("analysis_status", "success"),
            "analysis_method": result.get("analysis_method", "enhanced_llm"),
            "confidence_score": result.get("confidence_score", 0.8),
            "processing_time": processing_time,
            "request_id": result.get("request_id", request_id),
            "enhanced_course_flow": result.get("enhanced_course_flow", True),
            "validated_courses_count": result.get("validated_courses_count", len(courses)),
            
            # Original fields for backward compatibility
            "success": True,
            "course_recommendations": normalized_courses_fresh,
            "market_insights": market_insights,
            "salary_trends": salary_trends,
            "career_paths": career_paths,
            "skill_demand_analysis": skill_demand_analysis,
            "total_courses": len(courses),
            "source": "fresh",
            "message": "I've generated fresh course recommendations based on your latest profile."
        }
    
    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        request_id = _generate_request_id() if 'request_id' not in locals() else request_id
        log.error(f"Error getting course recommendations: {e}", exc_info=True)
        return {
            "status": "error",
            "node": "get_course_recommendations",
            "error": f"Failed to get course recommendations: {str(e)}",
            "processing_time": processing_time,
            "request_id": request_id
        }


async def get_career_goals(
    uid: str,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve all career goals (with milestones/tasks) for the user.
    """
    try:
        start_time = time.time()
        goals = await _load_goals(uid, session_id)
        processing_time = _calculate_processing_time(start_time)
        return {
            "status": "completed",
            "node": "get_career_goals",
            "output": {
                "goals": goals,
                "total_goals": len(goals)
            },
            "goals": goals,
            "total_goals": len(goals),
            "analysis_method": "chatbot_goal_manager",
            "confidence_score": 0.8,
            "processing_time": processing_time,
            "success": True,
            "message": f"Found {len(goals)} goals."
        }
    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        log.error(f"Error getting goals: {e}", exc_info=True)
        return {
            "status": "error",
            "node": "get_career_goals",
            "error": str(e),
            "processing_time": processing_time,
            "success": False
        }


async def update_career_goal(
    uid: str,
    goal_id: str,
    session_id: Optional[str] = None,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    priority: Optional[str] = None,
    target_date: Optional[str] = None,
    add_milestones: Optional[List[Dict[str, Any]]] = None,
    add_tasks: Optional[List[Dict[str, Any]]] = None,
    complete_tasks: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Update a career goal (status/progress/priority/target_date) and optionally add milestones/tasks or mark tasks complete.
    """
    try:
        start_time = time.time()
        goals = await _load_goals(uid, session_id)
        goal = next((g for g in goals if g.get("goal_id") == goal_id), None)
        if not goal:
            raise ValueError(f"Goal not found: {goal_id}")

        # Update simple fields
        if status:
            goal["status"] = status
        if progress is not None:
            try:
                goal["progress"] = max(0.0, min(1.0, float(progress)))
            except Exception:
                goal["progress"] = 0.0
        if priority:
            goal["priority"] = priority
        if target_date:
            goal["target_date"] = target_date

        # Add milestones
        if add_milestones:
            # Load context for better task duration estimation in milestones
            context = await load_comprehensive_context(uid, session_id)
            
            ms_list = goal.get("milestones", [])
            for ms in add_milestones[:5]:
                if isinstance(ms, dict):
                    ms_list.append(_normalize_milestone(ms, context))
            goal["milestones"] = ms_list

        # Add tasks (goal-level or under first milestone if milestone_id provided)
        if add_tasks:
            # Load context for better task duration estimation
            context = await load_comprehensive_context(uid, session_id)
            
            goal_tasks = goal.get("tasks", [])
            milestones = goal.get("milestones", [])
            for t in add_tasks[:20]:
                if not isinstance(t, dict):
                    continue
                milestone_id = t.get("milestone_id")
                norm_task = _normalize_task(t, context)
                if milestone_id:
                    target_ms = next((m for m in milestones if m.get("milestone_id") == milestone_id), None)
                    if target_ms:
                        target_ms.setdefault("tasks", []).append(norm_task)
                    else:
                        # If milestone not found, add to goal-level tasks
                        goal_tasks.append(norm_task)
                else:
                    goal_tasks.append(norm_task)
            goal["tasks"] = goal_tasks
            goal["milestones"] = milestones

        # Complete tasks by id across milestones and goal-level
        if complete_tasks:
            ids = set([tid for tid in complete_tasks if isinstance(tid, str)])
            for t in goal.get("tasks", []):
                if t.get("task_id") in ids:
                    t["status"] = "completed"
            for ms in goal.get("milestones", []):
                for t in ms.get("tasks", []):
                    if t.get("task_id") in ids:
                        t["status"] = "completed"

        goal["updated_at"] = datetime.utcnow().isoformat()

        await _save_goals(uid, session_id, goals)
        processing_time = _calculate_processing_time(start_time)

        return {
            "status": "completed",
            "node": "update_career_goal",
            "output": {
                "goals": goals,
                "updated_goal": goal,
                "total_goals": len(goals)
            },
            "goals": goals,
            "updated_goal": goal,
            "total_goals": len(goals),
            "analysis_method": "chatbot_goal_manager",
            "confidence_score": 0.8,
            "processing_time": processing_time,
            "success": True,
            "message": "Goal updated successfully."
        }
    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        log.error(f"Error updating career goal: {e}", exc_info=True)
        return {
            "status": "error",
            "node": "update_career_goal",
            "error": str(e),
            "processing_time": processing_time,
            "success": False
        }


async def get_career_advice(
    uid: str,
    session_id: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Get career advice - checks existing data first, calls agent if needed.
    
    Returns:
        Dict with career_paths, skill_gaps, improvement_recommendations, source
    """
    try:
        start_time = time.time()
        request_id = _generate_request_id()
        
        log.info(f"💼 Getting career advice for uid={uid} (force_refresh={force_refresh})")
        
        # Step 1: Check existing data first
        if not force_refresh:
            context = await load_comprehensive_context(uid, session_id)
            
            career_paths = context.get("career_paths", [])
            skill_gaps = context.get("skill_gaps", [])
            improvement_recs = context.get("improvement_recommendations", [])
            career_advice = context.get("career_advice", [])
            
            if career_paths or skill_gaps or improvement_recs:
                log.info(f"✅ Using existing career advice data")
                processing_time = _calculate_processing_time(start_time)
                # Build output format from existing context
                raw_skill_gap_analysis = {
                    "career_paths": career_paths,
                    "missing_skills": skill_gaps,
                    "improvement_recommendations": improvement_recs,
                    "career_advice": career_advice
                }
                return {
                    # Standard agent format (same as skill_and_career_advisor_agent)
                    "status": "completed",
                    "node": "get_career_advice",
                    "output": {
                        "raw_skill_gap_analysis_output": raw_skill_gap_analysis
                    },
                    
                    # Metadata fields (same as agents)
                    "raw_skill_gap_analysis_output": raw_skill_gap_analysis,
                    "analysis_status": "success",
                    "analysis_method": "existing_data",
                    "confidence_score": 0.8,
                    "processing_time": processing_time,
                    "request_id": request_id,
                    
                    # Original fields for backward compatibility
                    "success": True,
                    "career_paths": career_paths,
                    "skill_gaps": skill_gaps,
                    "improvement_recommendations": improvement_recs,
                    "career_advice": career_advice,
                    "source": "existing",
                    "message": "Here's your personalized career guidance based on your profile."
                }
        
        # Step 2: Call career_advisor agent for fresh data (single agent for speed)
        log.info(f"🔄 Calling career advisor agent (existing data not available or refresh requested)")
        
        # Load resume data
        structured_resume = await run_blocking_io(fetch_structured_resume, uid, None, session_id)
        if not structured_resume:
            processing_time = _calculate_processing_time(start_time)
            return {
                "status": "error",
                "node": "get_career_advice",
                "error": "No resume data found. Please upload your resume first.",
                "processing_time": processing_time,
                "request_id": request_id
            }
        
        # Get assessment results from dedicated Chroma user_assessments collection
        assessments_doc = await run_blocking_io(get_assessments_doc, uid)
        assessment_results = derive_assessment_results(assessments_doc or {})
        
        # Prepare state and call career_advisor only (single agent for speed)
        base_state = {
            "uid": uid,
            "tenant_id": uid,
            "session_id": session_id,
            "structured_resume": structured_resume,
            "user_interests": [],
            "assessment_results": assessment_results
        }
        state = create_agent_state(base_state, "skill_and_career_advisor")
        result = await skill_and_career_advisor_agent(state, tenant_id=uid)
        
        # Extract career advice (use agent's format directly - preserve ALL fields)
        output = result.get("output", {}).get("raw_skill_gap_analysis_output", {}) or result.get("raw_skill_gap_analysis_output", {})
        processing_time = _calculate_processing_time(start_time)
        
        log.info(f"✅ Career advisor returned fresh analysis")
        
        return {
            # Standard agent format (same as skill_and_career_advisor_agent - preserve all fields)
            "status": result.get("status", "completed"),
            "node": "get_career_advice",
            "output": {
                "raw_skill_gap_analysis_output": output
            },
            
            # Metadata fields (same as agents - preserve all from result)
            "raw_skill_gap_analysis_output": output,
            "analysis_status": result.get("analysis_status", "success"),
            "analysis_method": result.get("analysis_method", "llm"),
            "confidence_score": result.get("confidence_score", 0.8),
            "processing_time": processing_time,
            "request_id": result.get("request_id", request_id),
            
            # Original fields for backward compatibility
            "success": True,
            "career_paths": output.get("career_paths", []),
            "skill_gaps": get_missing_skills_flat(output),
            "improvement_recommendations": output.get("improvement_recommendations", []),
            "career_advice": output.get("career_advice", []),
            "source": "fresh",
            "message": "I've generated fresh career advice based on your latest profile."
        }
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        request_id = _generate_request_id() if 'request_id' not in locals() else request_id
        log.error(f"Error getting career advice: {e}", exc_info=True)
        return {
            "status": "error",
            "node": "get_career_advice",
            "error": f"Failed to get career advice: {str(e)}",
            "processing_time": processing_time,
            "request_id": request_id
        }


async def get_enhanced_role_fit(
    uid: str,
    session_id: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Get enhanced role fit (roles with why suggested, fit %, skill gaps, how to overcome).
    Uses enhanced_role_fit_agent. Call when user wants detailed role fit analysis.
    """
    try:
        start_time = time.time()
        request_id = _generate_request_id()
        log.info(f"🎯 Getting enhanced role fit for uid={uid} (force_refresh={force_refresh})")
        
        # Lightweight path: no load_comprehensive_context — use gap_doc only for existing role fits
        if not force_refresh:
            gap_doc = await run_blocking_io(get_gap_doc, uid)
            role_fits = []
            if gap_doc:
                career_advisor = gap_doc.get("skill_and_career_advisor", {})
                gap_analysis = career_advisor.get("raw_skill_gap_analysis_output", {})
                role_fits = gap_analysis.get("role_fit_suggestions", []) or []
            if role_fits:
                # Normalize so mentor teaser has role + fit_percentage
                def _norm(r):
                    if not isinstance(r, dict):
                        return r
                    role = r.get("role") or r.get("role_name") or r.get("title") or ""
                    pct = r.get("fit_percentage")
                    if pct is None and "match_percentage" in r:
                        pct = r.get("match_percentage")
                    if pct is None and "match_score" in r:
                        try:
                            pct = int(float(r["match_score"]) * 100) if r["match_score"] is not None else None
                        except (TypeError, ValueError):
                            pct = None
                    out = dict(r)
                    if role:
                        out["role"] = role
                    if pct is not None:
                        out["fit_percentage"] = pct
                    return out
                role_fits = [_norm(r) for r in role_fits if isinstance(r, dict)]
                if role_fits:
                    log.info(f"✅ Using existing role fit data from gap_doc ({len(role_fits)} roles)")
                    processing_time = _calculate_processing_time(start_time)
                    return {
                        "status": "completed",
                        "node": "get_enhanced_role_fit",
                        "success": True,
                        "enhanced_role_fit": role_fits,
                        "source": "existing",
                        "processing_time": processing_time,
                        "request_id": request_id,
                        "message": "Here's your role fit analysis from your profile."
                    }
        
        log.info(f"🔄 Calling enhanced_role_fit_agent")
        structured_resume = await run_blocking_io(fetch_structured_resume, uid, None, session_id)
        user_interests = []
        if session_id:
            try:
                session_data = await run_blocking_io(get_chat_session, session_id, uid)
                if session_data:
                    interest_data = session_data.get("interest_filler", {})
                    user_interests = interest_data.get("user_interests", [])
            except Exception as e:
                log.debug(f"Could not load user interests: {e}")
        
        if not structured_resume and not user_interests:
            processing_time = _calculate_processing_time(start_time)
            return {
                "status": "error",
                "node": "get_enhanced_role_fit",
                "success": False,
                "error": "No resume or interests found. Upload a resume or share your interests first.",
                "processing_time": processing_time,
                "request_id": request_id
            }
        
        base_state = {
            "uid": uid,
            "tenant_id": uid,
            "session_id": session_id,
            "structured_resume": structured_resume,
            "user_interests": user_interests,
        }
        state = create_agent_state(base_state, "enhanced_role_fit")
        result = await enhanced_role_fit_agent(state, tenant_id=uid)
        
        processing_time = _calculate_processing_time(start_time)
        enhanced = result.get("enhanced_role_fit", [])
        log.info(f"✅ Enhanced role fit returned {len(enhanced)} roles")
        return {
            "status": "completed",
            "node": "get_enhanced_role_fit",
            "success": True,
            "enhanced_role_fit": enhanced,
            "source": "fresh",
            "processing_time": processing_time,
            "request_id": request_id,
            "message": "I've generated a detailed role fit analysis with skill gaps and how to improve."
        }
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        request_id = _generate_request_id() if 'request_id' not in locals() else _generate_request_id()
        log.error(f"Error getting enhanced role fit: {e}", exc_info=True)
        return {
            "status": "error",
            "node": "get_enhanced_role_fit",
            "success": False,
            "error": str(e),
            "processing_time": processing_time,
            "request_id": request_id
        }


async def get_resume_score(
    uid: str,
    session_id: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Get resume score and role suggestions - checks existing data first, calls resume_scorer_agent if needed.
    """
    try:
        start_time = time.time()
        request_id = _generate_request_id()
        log.info(f"📊 Getting resume score for uid={uid} (force_refresh={force_refresh})")
        
        if not force_refresh:
            # Prefer dedicated uid_resume_score document, then gap_doc
            resume_score_data = await run_blocking_io(get_resume_score_doc, uid)
            if not resume_score_data or not isinstance(resume_score_data, dict):
                gap_doc = await run_blocking_io(get_gap_doc, uid)
                resume_score_data = gap_doc.get("resume_score", {}) if gap_doc else {}
            if resume_score_data and isinstance(resume_score_data, dict):
                rs = resume_score_data.get("ResumeScore", resume_score_data.get("resumeScore", {}))
                if rs:
                    log.info(f"✅ Using existing resume score data")
                    processing_time = _calculate_processing_time(start_time)
                    return {
                        "status": "completed",
                        "node": "get_resume_score",
                        "success": True,
                        "resume_score": resume_score_data,
                        "ResumeScore": rs,
                        "RoleSuggestions": resume_score_data.get("RoleSuggestions", []),
                        "source": "existing",
                        "processing_time": processing_time,
                        "request_id": request_id,
                        "message": "Here's your resume analysis based on your profile."
                    }
        
        log.info(f"🔄 Calling resume_scorer_agent (existing data not available or refresh requested)")
        structured_resume = await run_blocking_io(fetch_structured_resume, uid, None, session_id)
        if not structured_resume:
            processing_time = _calculate_processing_time(start_time)
            return {
                "status": "error",
                "node": "get_resume_score",
                "success": False,
                "error": "No resume data found. Please upload your resume first.",
                "processing_time": processing_time,
                "request_id": request_id
            }
        
        base_state = {
            "uid": uid,
            "tenant_id": uid,
            "session_id": session_id,
            "structured_resume": structured_resume,
            "chat_history": []
        }
        state = create_agent_state(base_state, "resume_scorer")
        result = await resume_scorer_agent(state)
        
        processing_time = _calculate_processing_time(start_time)
        resume_score = result.get("resumeScore", result)
        role_suggestions = (state.get("structured_resume") or {}).get("RoleSuggestions", [])
        
        log.info(f"✅ Resume scorer returned fresh analysis")
        return {
            "status": "completed",
            "node": "get_resume_score",
            "success": True,
            "resume_score": result,
            "ResumeScore": resume_score,
            "RoleSuggestions": role_suggestions,
            "source": "fresh",
            "processing_time": processing_time,
            "request_id": request_id,
            "message": "I've generated a fresh resume analysis based on your latest profile."
        }
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        request_id = _generate_request_id() if 'request_id' not in locals() else _generate_request_id()
        log.error(f"Error getting resume score: {e}", exc_info=True)
        return {
            "status": "error",
            "node": "get_resume_score",
            "success": False,
            "error": str(e),
            "processing_time": processing_time,
            "request_id": request_id
        }


def _pending_only(plan: list) -> list:
    """Return only pending (not completed) items from an assessment plan."""
    return [i for i in (plan or []) if isinstance(i, dict) and i.get("status", "pending") != "completed"]


def _normalize_plan_item_to_recommender_format(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single assessment plan item to match assessment_recommender output structure.
    Returns items matching exact structure: type, topic, difficulty, assessment_time_minutes,
    num_questions (only for "multi" type), rationale (optional), source (optional), status,
    assessment_id, completed_at, score.
    """
    if not isinstance(item, dict):
        return {}
    # Unwrap if nested under "assessment" (recommender can return either)
    assessment = item.get("assessment", item)
    if not isinstance(assessment, dict):
        return {}
    topic = (assessment.get("topic") or assessment.get("assessment_topic") or "").strip()
    if not topic:
        return {}
    
    assessment_type = (assessment.get("type") or "multi").strip() or "multi"
    if assessment_type not in ["multi", "mcq"]:
        assessment_type = "multi"
    
    difficulty = assessment.get("difficulty", "medium")
    if isinstance(difficulty, str):
        difficulty = difficulty.strip().lower() or "medium"
    if difficulty not in ["easy", "medium", "hard", "expert"]:
        difficulty = "medium"
    
    aid = assessment.get("assessment_id") or assessment.get("id")
    if not aid:
        aid = generate_unique_assessment_id()
    
    # Build normalized item in exact order: type, topic, difficulty, assessment_time_minutes, num_questions (if multi), rationale, source, status, assessment_id, completed_at, score
    normalized = {
        "type": assessment_type,
        "topic": topic,
        "difficulty": difficulty,
        "assessment_time_minutes": int(assessment.get("assessment_time_minutes", 30)),
    }
    
    # num_questions only for "multi" type (not "mcq")
    if assessment_type == "multi":
        num_q = assessment.get("num_questions") or {}
        if not isinstance(num_q, dict):
            num_q = {}
        normalized["num_questions"] = {
            "mcq": int(num_q.get("mcq", 5)),
            "short": int(num_q.get("short", 2)),
            "long": int(num_q.get("long", 0)),
            "coding": int(num_q.get("coding", 0)),
        }
    
    # Optional fields: rationale and source
    rationale = assessment.get("rationale")
    if rationale and isinstance(rationale, str) and rationale.strip():
        normalized["rationale"] = rationale.strip()
    
    source = assessment.get("source")
    if source and isinstance(source, str) and source.strip():
        normalized["source"] = source.strip()
    
    # Required fields: status, assessment_id, completed_at, score
    normalized["status"] = (assessment.get("status") or "pending").strip() or "pending"
    normalized["assessment_id"] = aid
    normalized["completed_at"] = assessment.get("completed_at")
    normalized["score"] = assessment.get("score")
    
    return normalized


def _normalize_assessment_plan_for_output(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize full plan to assessment_recommender output structure; ensure unique assessment_ids."""
    if not plan or not isinstance(plan, list):
        return []
    normalized = []
    seen_ids = set()
    for item in plan:
        n = _normalize_plan_item_to_recommender_format(item)
        if not n:
            continue
        aid = n.get("assessment_id")
        if aid and aid in seen_ids:
            aid = generate_unique_assessment_id()
            n["assessment_id"] = aid
        if aid:
            seen_ids.add(aid)
        normalized.append(n)
    return normalized


async def get_assessment_recommendations(
    uid: str,
    session_id: Optional[str] = None,
    force_refresh: bool = False,
    topics: Optional[List[str]] = None,
    difficulty: Optional[str] = None,
    suggest_new_topics: bool = False
) -> Dict[str, Any]:
    """
    Get assessment recommendations - checks existing data first, calls agent if needed.
    Can also create recommendations for specific topics (from mentor or user).
    Returns only pending (not completed) assessments plus any newly created ones.

    Args:
        uid: User ID
        session_id: Optional session ID
        force_refresh: If True, always call agent (ignore existing data)
        topics: Optional list of topics to create assessments for (e.g., ["Python", "React", "System Design"])
                 If provided, creates custom assessments for these topics (even if they exist)
        difficulty: Optional difficulty level ("easy", "medium", "hard", "expert"). Defaults to "medium"
        suggest_new_topics: If True and no topics provided, suggest topics based on profile and create
                           assessments only for topics that don't already exist

    Returns:
        Dict with assessment_plan (pending + new only), source (existing|fresh|custom)
    """
    try:
        start_time = time.time()
        request_id = _generate_request_id()
        log.info(f"📝 Getting assessment recommendations for uid={uid} (force_refresh={force_refresh}, topics={topics}, suggest_new_topics={suggest_new_topics})")
        
        # Step 1: If custom topics provided (user requested or mentor suggested), create assessment plan for those topics
        if topics and len(topics) > 0:
            log.info(f"🎯 Creating custom assessment recommendations for topics: {topics}")
            
            # Default difficulty to medium when topics provided — avoid load_comprehensive_context (slow, can cause timeout)
            if not difficulty:
                difficulty = "medium"
            
            # Normalize difficulty
            difficulty = difficulty.lower().strip()
            if difficulty not in ["easy", "medium", "hard", "expert"]:
                difficulty = "medium"
            
            # Create assessment plan items for each topic (exact structure matching assessment_recommender)
            assessment_plan = []
            for topic in topics:
                new_id = generate_unique_assessment_id()
                topic_str = str(topic).strip()
                # Base item structure: type, topic, difficulty, assessment_time_minutes, num_questions (for multi), rationale, source, status, assessment_id, completed_at, score
                assessment_item = {
                    "type": "multi",
                    "topic": topic_str,
                    "difficulty": difficulty,
                    "assessment_time_minutes": 30,
                    "num_questions": {
                        "mcq": 5,
                        "short": 2,
                        "long": 0,
                        "coding": 0
                    },
                    "rationale": f"Recommended to validate and showcase your {topic_str} skills.",
                    "source": "custom_from_chatbot",
                    "status": "pending",
                    "assessment_id": new_id,
                    "completed_at": None,
                    "score": None
                }
                # Adjust based on difficulty
                if difficulty == "easy":
                    assessment_item["num_questions"] = {"mcq": 5, "short": 1, "long": 0, "coding": 0}
                    assessment_item["assessment_time_minutes"] = 20
                elif difficulty == "hard":
                    assessment_item["num_questions"] = {"mcq": 5, "short": 2, "long": 1, "coding": 0}
                    assessment_item["assessment_time_minutes"] = 45
                elif difficulty == "expert":
                    assessment_item["num_questions"] = {"mcq": 5, "short": 2, "long": 1, "coding": 1}
                    assessment_item["assessment_time_minutes"] = 60
                assessment_plan.append(assessment_item)
            
            log.info(f"✅ Created {len(assessment_plan)} custom assessment recommendations")
            
            # Store custom assessment plan in ChromaDB (same format as assessment_recommender)
            # Merge with existing assessments if they exist (add new topics to existing plan)
            try:
                assessments_doc = await run_blocking_io(get_assessments_doc, uid) or {}
                existing_assessment_data = assessments_doc.get("assessment_recommender", {})
                existing_plan = existing_assessment_data.get("assessment_plan", [])
                
                # Get existing topics to avoid duplicates
                existing_topics = {item.get("topic", "").lower().strip() for item in existing_plan if isinstance(item, dict) and item.get("topic")}
                
                # Merge: Add new topics to existing plan (don't duplicate)
                merged_plan = list(existing_plan)  # Start with existing
                for new_item in assessment_plan:
                    topic = new_item.get("topic", "").lower().strip()
                    if topic and topic not in existing_topics:
                        merged_plan.append(new_item)
                        existing_topics.add(topic)
                        log.info(f"➕ Added new assessment topic: {new_item.get('topic')}")
                    else:
                        log.info(f"⏭️ Skipped duplicate topic: {new_item.get('topic')}")
                # Ensure every plan item has id/assessment_id (same as recommender) for UI and callback
                merged_plan = assign_unique_ids_to_assessments(merged_plan)

                # Update assessment_needs
                existing_needs = existing_assessment_data.get("assessment_needs", {})
                merged_needs = {
                    **existing_needs,
                    "custom_topics": topics,
                    "difficulty": difficulty,
                    "source": "custom_from_chatbot_merged" if existing_plan else "custom_from_chatbot"
                }
                
                assessments_doc["assessment_recommender"] = {
                    "assessment_needs": merged_needs,
                    "assessment_plan": merged_plan,  # Merged plan with new topics added
                    "personalization_score": 0.8,  # Default score for custom recommendations
                    "updated_at": time.time(),
                    "version": (existing_assessment_data.get("version", 0) + 1)
                }
                await run_blocking_io(
                    upsert_assessments_doc,
                    uid,
                    assessments_doc,
                    metadata={
                        "agent": "career_chatbot",
                        "uid": uid,
                        "status": "custom_assessment_recommendations_created"
                    }
                )
                log.info(f"💾 Stored merged assessment plan ({len(merged_plan)} total items, {len(assessment_plan)} new) to ChromaDB for uid={uid}")
                
                # Auto-create tasks in goal system (fire-and-forget; don't block response)
                asyncio.create_task(_create_assessment_tasks(uid, assessment_plan, session_id))
                
                # Use merged plan and needs for return
                final_plan = merged_plan
                final_needs = merged_needs
                existing_count = len(existing_plan)
            except Exception as e_store:
                log.warning(f"⚠️ Failed to store custom assessment plan: {e_store}")
                final_plan = assessment_plan
                final_needs = {
                    "topics": topics,
                    "difficulty": difficulty,
                    "source": "custom_from_chatbot"
                }
                existing_count = 0
            
            return_plan = _normalize_assessment_plan_for_output(_pending_only(final_plan))
            processing_time = _calculate_processing_time(start_time)
            
            log.info(f"✅ Returning custom assessment recommendations for {len(topics)} topic(s), {len(return_plan)} pending")
            return {
                # Standard agent format (same as assessment_recommender_agent)
                "status": "completed",
                "node": "get_assessment_recommendations",
                "output": {
                    "assessment_needs": final_needs,
                    "assessment_plan": return_plan
                },
                
                # Metadata fields (same as agents)
                "assessment_needs": final_needs,
                "assessment_plan": return_plan,
                "personalization_score": 0.8,
                "analysis_method": "custom_from_chatbot",
                "confidence_score": 0.8,
                "processing_time": processing_time,
                "request_id": request_id,
                
                # Original fields for backward compatibility
                "success": True,
                "total_assessments": len(return_plan),
                "source": "custom",
                "topics": topics,
                "difficulty": difficulty,
                "message": f"I've created assessment recommendations for {len(topics)} topic(s): {', '.join(topics)}. You have {len(return_plan)} pending assessment(s)."
            }
        
        # Step 2: If suggest_new_topics=True and no topics provided, suggest topics and create assessments for new ones
        if suggest_new_topics and (not topics or len(topics) == 0):
            log.info(f"💡 Suggesting new assessment topics based on profile")
            
            # Get existing assessment plan to check what topics already exist
            assessments_doc = await run_blocking_io(get_assessments_doc, uid)
            existing_topics = set()
            if assessments_doc:
                assessment_data = assessments_doc.get("assessment_recommender", {})
                if isinstance(assessment_data, dict):
                    existing_plan = assessment_data.get("assessment_plan", [])
                    for item in existing_plan:
                        if isinstance(item, dict):
                            topic = item.get("topic", "")
                            if topic:
                                existing_topics.add(topic.lower().strip())
            
            # Suggest topics based on profile
            suggested_topics = await suggest_assessment_topics(uid, session_id)
            
            # Filter out topics that already have assessments
            new_topics = [t for t in suggested_topics if t.lower().strip() not in existing_topics]
            
            if new_topics:
                log.info(f"🎯 Found {len(new_topics)} new topics without existing assessments: {new_topics}")
                # Create assessments for new topics
                return await get_assessment_recommendations(
                    uid=uid,
                    session_id=session_id,
                    force_refresh=False,
                    topics=new_topics,
                    difficulty=difficulty,
                    suggest_new_topics=False  # Prevent recursion
                )
            else:
                log.info(f"ℹ️ All suggested topics already have assessments, returning existing plan")
                # Fall through to return existing plan
        
        # Step 3: Check existing data first (if no custom topics and not suggesting new)
        if not force_refresh:
            # Try to get assessment plan from assessments_doc (where assessment_recommender stores it)
            assessments_doc = await run_blocking_io(get_assessments_doc, uid)
            if assessments_doc:
                assessment_data = assessments_doc.get("assessment_recommender", {})
                if isinstance(assessment_data, dict):
                    existing_plan = assessment_data.get("assessment_plan", [])
                    existing_needs = assessment_data.get("assessment_needs", {})
                    # Ensure all items have status and id/assessment_id (same as recommender for UI)
                    needs_update = False
                    for item in existing_plan:
                        if isinstance(item, dict) and "status" not in item:
                            item["status"] = "pending"
                            needs_update = True
                    has_missing_ids = any(
                        isinstance(i, dict) and not (i.get("id") or i.get("assessment_id"))
                        for i in existing_plan
                    )
                    existing_plan = assign_unique_ids_to_assessments(existing_plan)
                    if has_missing_ids:
                        needs_update = True
                    # Update stored plan if we added status fields or ids
                    if needs_update:
                        try:
                            assessments_doc["assessment_recommender"] = {
                                **assessment_data,
                                "assessment_plan": existing_plan,
                                "updated_at": time.time()
                            }
                            await run_blocking_io(
                                upsert_assessments_doc,
                                uid,
                                assessments_doc,
                                metadata={"agent": "career_chatbot", "uid": uid, "status": "assessment_plan_normalized"}
                            )
                            log.info(f"✅ Normalized assessment plan with status fields")
                        except Exception as e_norm:
                            log.warning(f"⚠️ Failed to normalize assessment plan: {e_norm}")
                    
                    if existing_plan and len(existing_plan) > 0:
                        return_plan = _normalize_assessment_plan_for_output(_pending_only(existing_plan))
                        log.info(f"✅ Using existing assessment plan ({len(return_plan)} pending)")
                        processing_time = _calculate_processing_time(start_time)
                        return {
                            # Standard agent format (same as assessment_recommender_agent)
                            "status": "completed",
                            "node": "get_assessment_recommendations",
                            "output": {
                                "assessment_needs": existing_needs,
                                "assessment_plan": return_plan
                            },
                            
                            # Metadata fields (same as agents)
                            "assessment_needs": existing_needs,
                            "assessment_plan": return_plan,
                            "personalization_score": 0.8,
                            "analysis_method": "existing_data",
                            "confidence_score": 0.8,
                            "processing_time": processing_time,
                            
                            # Original fields for backward compatibility
                            "success": True,
                            "total_assessments": len(return_plan),
                            "source": "existing",
                            "message": "Here are your pending assessment recommendations based on your profile."
                        }
            
            # Fallback: check assessment_results from context
            context = await load_comprehensive_context(uid, session_id)
            existing_assessments = context.get("assessment_results", [])
            if existing_assessments:
                return_plan = _normalize_assessment_plan_for_output(_pending_only(existing_assessments))
                log.info(f"✅ Using existing assessment results ({len(return_plan)} pending)")
                # Extract topics from existing assessments
                topics_list = [a.get("assessment_topic") or a.get("topic", "") for a in existing_assessments if a.get("assessment_topic") or a.get("topic")]
                
                # Convert existing_assessments to assessment_plan format (normalized above)
                assessment_needs_from_context = {"source": "context_fallback"}
                
                processing_time = _calculate_processing_time(start_time)
                
                return {
                    # Standard agent format (same as assessment_recommender_agent)
                    "status": "completed",
                    "node": "get_assessment_recommendations",
                    "output": {
                        "assessment_needs": assessment_needs_from_context,
                        "assessment_plan": return_plan
                    },
                    
                    # Metadata fields (same as agents)
                    "assessment_needs": assessment_needs_from_context,
                    "assessment_plan": return_plan,
                    "personalization_score": 0.0,
                    "analysis_method": "existing_data_from_context",
                    "confidence_score": 0.8,
                    "processing_time": processing_time,
                    
                    # Original fields for backward compatibility
                    "success": True,
                    "recommended_topics": topics_list,
                    "total_assessments": len(return_plan),
                    "source": "existing",
                    "message": "Here are your pending assessment recommendations based on your profile."
                }
        
        # Step 4: Call agent for fresh recommendations (no topics, no existing data, or force_refresh)
        log.info(f"🔄 Calling assessment recommender agent (existing data not available or refresh requested)")
        
        # Load resume data
        structured_resume = await run_blocking_io(fetch_structured_resume, uid, None, session_id)
        if not structured_resume:
            processing_time = _calculate_processing_time(start_time)
            return {
                "status": "error",
                "node": "get_assessment_recommendations",
                "error": "No resume data found. Please upload your resume first.",
                "processing_time": processing_time
            }
        
        # Prepare state
        base_state = {
            "uid": uid,
            "tenant_id": uid,
            "session_id": session_id,
            "structured_resume": structured_resume,
            "user_interests": []
        }
        state = create_agent_state(base_state, "assessment_recommender")
        
        # Call assessment_recommender_agent (which internally uses UnifiedAssessmentRecommender)
        # This ensures proper storage and format matching
        result = await assessment_recommender_agent(state)
        
        # Extract data in same format as assessment_recommender returns (preserve ALL fields)
        # The agent stores the data automatically, but we extract it for return
        output = result.get("output", {})
        assessment_needs = output.get("assessment_needs", {}) or result.get("assessment_needs", {})
        assessment_plan = output.get("assessment_plan", []) or result.get("assessment_plan", [])
        
        # Also get from stored data to ensure we have the latest
        assessments_doc = await run_blocking_io(get_assessments_doc, uid)
        if assessments_doc:
            stored_data = assessments_doc.get("assessment_recommender", {})
            if stored_data.get("assessment_plan"):
                assessment_plan = stored_data.get("assessment_plan", assessment_plan)
                # Ensure all items have status field (backward compatibility)
                for item in assessment_plan:
                    if isinstance(item, dict) and "status" not in item:
                        item["status"] = "pending"
            if stored_data.get("assessment_needs"):
                assessment_needs = stored_data.get("assessment_needs", assessment_needs)
        
        # Auto-create tasks for new assessments (fire-and-forget; don't block response)
        asyncio.create_task(_create_assessment_tasks(uid, assessment_plan, session_id))
        
        return_plan = _normalize_assessment_plan_for_output(_pending_only(assessment_plan))
        log.info(f"✅ Assessment recommender agent returned {len(return_plan)} pending assessments (fresh data, stored in ChromaDB)")
        
        processing_time = _calculate_processing_time(start_time)
        
        return {
            # Standard agent format (same as assessment_recommender_agent - preserve all fields)
            "status": result.get("status", "completed"),
            "node": "get_assessment_recommendations",
            "output": {
                "assessment_needs": assessment_needs,
                "assessment_plan": return_plan
            },
            
            # Metadata fields (same as agents - preserve all from result)
            "assessment_needs": assessment_needs,
            "assessment_plan": return_plan,
            "personalization_score": result.get("personalization_score", 0.8),
            "analysis_method": result.get("analysis_method", "llm"),
            "confidence_score": result.get("confidence_score", 0.8),
            "processing_time": result.get("processing_time", processing_time),
            
            # Original fields for backward compatibility
            "success": True,
            "total_assessments": len(return_plan),
            "source": "fresh",
            "message": f"I've generated fresh assessment recommendations based on your latest profile. You have {len(return_plan)} pending assessment(s)."
        }
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        req_id = request_id if 'request_id' in locals() else _generate_request_id()
        log.error(f"Error getting assessment recommendations: {e}", exc_info=True)
        return {
            "status": "error",
            "node": "get_assessment_recommendations",
            "error": f"Failed to get assessment recommendations: {str(e)}",
            "processing_time": processing_time,
            "request_id": req_id
        }


async def get_career_goals(
    uid: str,
    session_id: Optional[str] = None,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get user's career goals from ChromaDB or session.
    
    Args:
        uid: User ID
        session_id: Optional session ID
        status: Optional filter by status (active, completed, paused)
    
    Returns:
        Dict with goals list and metadata
    """
    try:
        log.info(f"🎯 Getting career goals for uid={uid} (status={status})")
        
        goals = []
        
        # Try to load from session first (temporary storage until goal_manager is implemented)
        if session_id:
            try:
                session_data = await run_blocking_io(get_chat_session, session_id, uid)
                if session_data:
                    stored_goals = session_data.get("career_goals", [])
                    if isinstance(stored_goals, list):
                        goals.extend(stored_goals)
            except Exception as e:
                log.debug(f"Could not load goals from session: {e}")
        
        # TODO: When goal_manager is implemented, load from ChromaDB career_goals collection
        # goals = await run_blocking_io(get_career_goals, uid, status)
        
        # Filter by status if provided
        if status:
            goals = [g for g in goals if g.get("status", "active") == status]
        
        log.info(f"✅ Found {len(goals)} career goals for uid={uid}")
        
        return {
            "success": True,
            "goals": goals,
            "count": len(goals),
            "message": f"Found {len(goals)} career goal(s)."
        }
        
    except Exception as e:
        log.error(f"Error getting career goals: {e}", exc_info=True)
        return {"error": f"Failed to get career goals: {str(e)}"}


def _normalize_task(task: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Normalize a task payload into a consistent structure.
    Auto-calculates due_date and effort_hours if not provided.
    
    Args:
        task: Task dictionary
        context: Optional context with course/assessment data for better estimation
    """
    # Get due_date from either due_date or target_date field, and normalize it
    due_date_raw = task.get("due_date") or task.get("target_date")
    due_date = _normalize_date(due_date_raw) if due_date_raw else None
    
    # Get or estimate effort_hours
    effort_hours = task.get("effort_hours")
    if not effort_hours or effort_hours <= 0:
        # Estimate effort based on task type and difficulty
        effort_hours = _estimate_task_duration(task, context)
        log.debug(f"⏱️ Auto-estimated effort for task '{task.get('title')}': {effort_hours}h")
    
    # If no due_date provided, calculate from effort_hours
    if not due_date:
        due_date = _calculate_due_date_from_duration(effort_hours)
        log.debug(f"📅 Auto-calculated due_date for task '{task.get('title')}': {due_date} (from {effort_hours}h)")
    
    return {
        "task_id": task.get("task_id") or _generate_request_id(),
        "title": str(task.get("title", "")).strip()[:200],
        "status": task.get("status", "not_started"),
        "priority": task.get("priority", "medium"),
        "due_date": due_date,  # Normalized or auto-calculated date
        "effort_hours": effort_hours,
        "notes": task.get("notes", ""),
        "created_at": task.get("created_at") or datetime.utcnow().isoformat(),
        "assessment_topic": task.get("assessment_topic"),  # Link to assessment
        "assessment_id": task.get("assessment_id")  # Link to assessment ID
    }


def _normalize_milestone(ms: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalize a milestone payload with nested tasks."""
    tasks = ms.get("tasks", [])
    norm_tasks = [_normalize_task(t, context) for t in tasks if isinstance(t, dict)]
    return {
        "milestone_id": ms.get("milestone_id") or _generate_request_id(),
        "title": str(ms.get("title", "")).strip()[:200],
        "status": ms.get("status", "not_started"),
        "progress": float(ms.get("progress", 0) or 0),
        "due_date": ms.get("due_date"),
        "tasks": norm_tasks,
        "created_at": ms.get("created_at") or datetime.utcnow().isoformat()
    }


async def _load_goals(uid: str, session_id: Optional[str]) -> List[Dict[str, Any]]:
    """Load goals from session storage (ChromaDB-backed)."""
    sid = session_id or f"{uid}_goals"
    session_data = await run_blocking_io(get_chat_session, sid, uid) or {}
    goals = session_data.get("career_goals", [])
    return goals if isinstance(goals, list) else []


async def _save_goals(uid: str, session_id: Optional[str], goals: List[Dict[str, Any]]) -> None:
    """Persist goals back to session storage."""
    sid = session_id or f"{uid}_goals"
    session_data = await run_blocking_io(get_chat_session, sid, uid) or {}
    session_data["career_goals"] = goals
    session_data["uid"] = uid
    session_data["last_updated"] = time.time()
    # Chroma requires uid, timestamp, status (truthy). Ensure set when creating from empty session.
    if not session_data.get("timestamp"):
        session_data["timestamp"] = time.time()
    if not session_data.get("status"):
        session_data["status"] = "active"
    await run_blocking_io(update_chat_session, sid, session_data)


async def _create_assessment_tasks(uid: str, assessment_plan: List[Dict[str, Any]], session_id: Optional[str] = None) -> None:
    """
    Auto-create tasks in goal system for each recommended assessment.
    This allows tracking and nudging users about pending assessments.
    """
    try:
        if not assessment_plan:
            return
        
        # Load existing goals
        goals = await _load_goals(uid, session_id)
        
        # Find or create "Skill Validation" goal
        skill_validation_goal = None
        for goal in goals:
            if goal.get("title", "").lower() in ["skill validation", "assessments", "skill assessment"]:
                skill_validation_goal = goal
                break
        
        if not skill_validation_goal:
            # Create new goal for assessments
            skill_validation_goal = {
                "goal_id": f"goal_{_generate_request_id()}",
                "uid": uid,
                "title": "Skill Validation",
                "description": "Complete recommended assessments to validate and improve your skills",
                "target_date": None,
                "priority": "medium",
                "status": "in_progress",
                "progress": 0.0,
                "milestones": [],
                "tasks": [],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            goals.append(skill_validation_goal)
        
        # Get existing task topics to avoid duplicates
        existing_tasks = skill_validation_goal.get("tasks", [])
        existing_topics = {t.get("assessment_topic", "").lower().strip() for t in existing_tasks if t.get("assessment_topic")}
        
        # Create tasks for new assessments
        new_tasks = []
        for assessment in assessment_plan:
            topic = assessment.get("topic", "").strip()
            if not topic:
                continue
            
            topic_lower = topic.lower().strip()
            if topic_lower in existing_topics:
                continue  # Skip if task already exists
            
            # Create task for this assessment
            task = {
                "task_id": f"task_{_generate_request_id()}",
                "title": f"Take {topic} Assessment",
                "status": "not_started",
                "priority": assessment.get("priority", "medium"),
                "due_date": None,
                "effort_hours": assessment.get("assessment_time_minutes", 30) / 60.0,  # Convert minutes to hours
                "notes": f"Assessment on {topic} ({assessment.get('difficulty', 'medium')} difficulty)",
                "created_at": datetime.utcnow().isoformat(),
                "assessment_topic": topic,
                "assessment_id": assessment.get("assessment_id")
            }
            new_tasks.append(task)
            existing_topics.add(topic_lower)
        
        if new_tasks:
            skill_validation_goal["tasks"].extend(new_tasks)
            skill_validation_goal["updated_at"] = datetime.utcnow().isoformat()
            await _save_goals(uid, session_id, goals)
            log.info(f"✅ Created {len(new_tasks)} assessment tasks in goal system")
        
    except Exception as e:
        log.warning(f"⚠️ Error creating assessment tasks: {e}", exc_info=True)


async def update_assessment_status(
    uid: str,
    assessment_topic: str,
    status: str,
    assessment_id: Optional[str] = None,
    score: Optional[float] = None,
    completed_at: Optional[str] = None
) -> bool:
    """
    Update the status of an assessment in the assessment plan.
    Called when an assessment is completed or started (e.g. during evaluation).
    Status is normalized and stored in the dedicated Chroma user_assessments collection.
    """
    try:
        assessments_doc = await run_blocking_io(get_assessments_doc, uid) or {}
        assessment_data = assessments_doc.get("assessment_recommender", {})
        assessment_plan = assessment_data.get("assessment_plan", [])
        topic_lower = (assessment_topic or "").lower().strip()

        # Normalize status for consistent storage (completed|pending|in_progress)
        status_normalized = normalize_assessment_status(status)

        # Find and update the matching assessment in the plan: prefer assessment_id, then topic
        updated = False
        matched_item = None
        for item in assessment_plan:
            if not isinstance(item, dict):
                continue
            # 1) Match by assessment_id (stable; avoids "Python" vs "Python 3" mismatch)
            if assessment_id and (
                item.get("assessment_id") == assessment_id or item.get("assessment_id_ref") == assessment_id
            ):
                item["status"] = status_normalized
                item["assessment_id"] = assessment_id
                if score is not None:
                    item["score"] = score
                if completed_at:
                    item["completed_at"] = completed_at
                updated = True
                matched_item = item
                log.info(f"✅ Updated assessment status by ID: {assessment_id} -> {status_normalized} (topic: {item.get('topic', assessment_topic)})")
                break
        if not updated and topic_lower:
            # 2) Fallback: match by topic (backward compat when assessment_id not provided)
            for item in assessment_plan:
                if not isinstance(item, dict):
                    continue
                item_topic = (item.get("topic") or "").lower().strip()
                if item_topic == topic_lower:
                    item["status"] = status_normalized
                    if assessment_id:
                        item["assessment_id"] = assessment_id
                    if score is not None:
                        item["score"] = score
                    if completed_at:
                        item["completed_at"] = completed_at
                    updated = True
                    matched_item = item
                    log.info(f"✅ Updated assessment status by topic: {assessment_topic} -> {status_normalized}")
                    break

        # When completed: append to assessment_history immediately so mentor sees result as soon as user submits
        history_updated = False
        if status_normalized == "completed" and score is not None and assessment_topic:
            history = list(assessments_doc.get("assessment_history") or [])
            # Avoid duplicate by assessment_id if present
            hist_id = assessment_id or f"submit-{topic_lower}-{int(time.time())}"
            if not any(isinstance(h, dict) and h.get("assessment_id") == hist_id for h in history):
                history.append({
                    "topic": assessment_topic,
                    "assessment_id": hist_id,
                    "total_score": score,
                    "result": {"score": score},
                    "timestamp": time.time(),
                    "completed_at": completed_at or datetime.utcnow().isoformat(),
                    "source": "evaluation",
                })
                assessments_doc["assessment_history"] = history
                history_updated = True
                log.info(f"✅ Appended assessment_history for {assessment_topic} (score {score}) — mentor will see completion immediately")
        if updated:
            # Ensure all items have status field (backward compatibility)
            for item in assessment_plan:
                if isinstance(item, dict) and "status" not in item:
                    item["status"] = "pending"  # normalized value
            assessments_doc["assessment_recommender"] = {
                **assessment_data,
                "assessment_plan": assessment_plan,
                "updated_at": time.time(),
                "version": (assessment_data.get("version", 0) + 1)
            }
        if updated or history_updated:
            # Persist to dedicated Chroma user_assessments collection (status normalized on write)
            await run_blocking_io(
                upsert_assessments_doc,
                uid,
                assessments_doc,
                metadata={
                    "agent": "career_chatbot",
                    "uid": uid,
                    "status": "assessment_status_updated"
                }
            )
        if updated:
            # Also update corresponding task in goal system (use matched item topic when we matched by ID)
            topic_for_goal = (matched_item.get("topic") if matched_item else None) or assessment_topic or ""
            topic_for_goal_lower = topic_for_goal.lower().strip()
            try:
                goals = await _load_goals(uid, None)
                for goal in goals:
                    tasks = goal.get("tasks", [])
                    for task in tasks:
                        task_topic = (task.get("assessment_topic") or "").lower().strip()
                        if task_topic == topic_for_goal_lower:
                            if status_normalized == "completed":
                                task["status"] = "completed"
                                if score is not None:
                                    task["notes"] = f"{task.get('notes', '')} | Score: {score:.1f}%"
                            elif status_normalized == "in_progress":
                                task["status"] = "in_progress"
                            goal["updated_at"] = datetime.utcnow().isoformat()
                            await _save_goals(uid, None, goals)
                            log.info(f"✅ Updated corresponding task status for {topic_for_goal}")
                            break
            except Exception as e_task:
                log.warning(f"⚠️ Failed to update task status: {e_task}")
        return updated or history_updated
        
    except Exception as e:
        log.error(f"Error updating assessment status: {e}", exc_info=True)
        return False


async def create_career_goal(
    uid: str,
    goal_title: str,
    goal_description: str,
    target_date: Optional[str] = None,
    priority: str = "medium",
    session_id: Optional[str] = None,
    milestones: Optional[List[Dict[str, Any]]] = None,
    tasks: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Create a career goal with optional milestones/tasks.
    Stores in session-backed storage (ChromaDB) for persistence across requests.
    """
    try:
        start_time = time.time()
        log.info(f"🎯 Creating career goal for uid={uid}: {goal_title}")

        # Load context for better task duration estimation
        context = await load_comprehensive_context(uid, session_id)
        
        normalized_milestones = []
        if milestones:
            for ms in milestones[:5]:
                if isinstance(ms, dict):
                    normalized_milestones.append(_normalize_milestone(ms, context))

        normalized_tasks = []
        if tasks:
            for t in tasks[:20]:
                if isinstance(t, dict):
                    normalized_tasks.append(_normalize_task(t, context))
        
        # Normalize goal's target_date
        normalized_target_date = _normalize_date(target_date) if target_date else None
        
        # If goal doesn't have target_date but tasks do, use the earliest task due_date
        if not normalized_target_date and normalized_tasks:
            task_dates = [t.get("due_date") for t in normalized_tasks if t.get("due_date")]
            if task_dates:
                # Use earliest task date as goal target_date
                try:
                    task_dates_parsed = [datetime.fromisoformat(d).date() if isinstance(d, str) else d for d in task_dates if d]
                    if task_dates_parsed:
                        normalized_target_date = min(task_dates_parsed).isoformat()
                        log.info(f"📅 Set goal target_date from earliest task: {normalized_target_date}")
                except Exception as e:
                    log.debug(f"Could not parse task dates for goal target_date: {e}")

        goal = {
            "goal_id": f"goal_{_generate_request_id()}",
            "uid": uid,
            "title": goal_title.strip()[:200],
            "description": goal_description.strip()[:2000],
            "target_date": normalized_target_date,  # Normalized date
            "priority": priority,
            "status": "in_progress",
            "progress": 0.0,
            "milestones": normalized_milestones,
            "tasks": normalized_tasks,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        goals = await _load_goals(uid, session_id)
        goals.append(goal)
        await _save_goals(uid, session_id, goals)
        log.info(f"💾 Stored goal: {goal['goal_id']} (total goals: {len(goals)})")

        processing_time = _calculate_processing_time(start_time)

        log.info(f"✅ Created career goal: {goal['goal_id']}")
        
        return {
            "status": "completed",
            "node": "create_career_goal",
            "output": {
                "goals": goals,
                "created_goal": goal,
                "total_goals": len(goals)
            },
            "goals": goals,
            "created_goal": goal,
            "total_goals": len(goals),
            "analysis_method": "chatbot_goal_manager",
            "confidence_score": 0.8,
            "processing_time": processing_time,
            "success": True,
            "message": f"Goal '{goal_title}' created successfully! I'll help you track your progress."
        }
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        log.error(f"Error creating goal: {e}", exc_info=True)
        return {
            "status": "error",
            "node": "create_career_goal",
            "error": f"Failed to create goal: {str(e)}",
            "processing_time": processing_time
        }


async def identify_relevant_jobs(
    uid: str,
    session_id: Optional[str] = None,
    search_query: Optional[str] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Find the best job matches using existing job matching agents.
    Uses enhanced_llm_job_matcher to find top 3 matches aligned with user's skills and qualifications.
    
    Args:
        uid: User ID
        session_id: Optional session ID
        search_query: Optional custom search query (not used, kept for compatibility)
        top_k: Number of top jobs to return (default: 3, max: 3)
    
    Returns:
        Dict with top job matches from job matching agents, match scores, and detailed matching info
    """
    try:
        start_time = time.time()
        log.info(f"💼 Finding best job matches using job matching agents for uid={uid}")
        
        # Limit to max 3 matches as requested (coerce to int; LLM may pass float e.g. 3.0)
        top_k = min(int(top_k), 3)
        
        # Load context to get user profile
        context = await load_comprehensive_context(uid, session_id)
        
        # Get structured resume (required for job matching)
        # Try fetch_structured_resume first, then check groq_resume_parser output
        structured_resume = await run_blocking_io(fetch_structured_resume, uid)
        
        # If not found, try getting from groq_resume_parser or resume_parser output in session
        if not structured_resume:
            try:
                from chroma import get_chat_session
                # Try to get from any recent session
                session_data = await run_blocking_io(get_chat_session, session_id or f"{uid}_latest", uid)
                if session_data:
                    # Check groq_resume_parser output
                    if "groq_resume_parser" in session_data and "structured_resume" in session_data["groq_resume_parser"]:
                        structured_resume = session_data["groq_resume_parser"]["structured_resume"]
                        log.info(f"✅ Found resume from groq_resume_parser output for uid={uid}")
                    # Check resume_parser output
                    elif "resume_parser" in session_data and "structured_resume" in session_data["resume_parser"]:
                        structured_resume = session_data["resume_parser"]["structured_resume"]
                        log.info(f"✅ Found resume from resume_parser output for uid={uid}")
            except Exception as e:
                log.debug(f"Could not get resume from session: {e}")
        
        if not structured_resume:
            log.warning(f"⚠️ No structured resume found for uid={uid}, falling back to role fit suggestions")
            return await _fallback_to_role_fit_suggestions(uid, session_id, context, top_k)
        
        log.info(f"✅ Using structured resume for job matching (type: {type(structured_resume).__name__})")
        
        # Use the full job_matcher_agent (same as normal flow)
        from agents.job_matcher import job_matcher_agent
        
        log.info("🔍 Running job_matcher_agent for comprehensive job matching...")
        
        # Prepare state for job_matcher_agent (same format as normal usage)
        # Add chatbot_top_k to limit results when called from chatbot (3-4 matches)
        agent_state = {
            "uid": uid,
            "tenant_id": uid,
            "session_id": session_id,
            "structured_resume": structured_resume,
            "raw_skill_gap_analysis_output": context.get("skill_gaps", {}),
            "user_interests": [],
            "chatbot_top_k": top_k  # Limit results when called from chatbot
        }
        
        # Run the job matcher agent
        matcher_result = await job_matcher_agent(agent_state)
        
        # Extract matched jobs from result
        matched_jobs = []
        if matcher_result and isinstance(matcher_result, dict):
            # The agent returns results in 'top_matches' key
            matched_jobs = matcher_result.get("top_matches", [])
            
            # If no top_matches, check other possible keys
            if not matched_jobs:
                matched_jobs = matcher_result.get("matches", []) or matcher_result.get("job_matches", [])
        
        if not matched_jobs:
            log.warning("⚠️ Job matcher agent returned no matches, falling back to role fit suggestions")
            fallback_result = await _fallback_to_role_fit_suggestions(uid, session_id, context, top_k)
            # Wrap fallback result in standard format
            if fallback_result.get("success"):
                processing_time = _calculate_processing_time(start_time)
                return {
                    # Standard agent format
                    "status": "completed",
                    "node": "identify_relevant_jobs",
                    "output": {
                        "top_matches": fallback_result.get("relevant_jobs", []),
                        "total_matches_found": fallback_result.get("count", 0),
                        "matching_method": "role_fit_fallback",
                        "job_matcher_status": "fallback"
                    },
                    
                    # Metadata fields
                    "matched_jobs": fallback_result.get("relevant_jobs", []),
                    "top_matches": fallback_result.get("relevant_jobs", []),
                    "total_matches_found": fallback_result.get("count", 0),
                    "job_matcher_status": "fallback",
                    "confidence_score": 0.7,  # Lower confidence for fallback
                    "processing_time_seconds": round(processing_time, 3),
                    "analysis_method": "role_fit_fallback",
                    
                    # Original fields
                    "success": True,
                    "relevant_jobs": fallback_result.get("relevant_jobs", []),
                    "count": fallback_result.get("count", 0),
                    "source": fallback_result.get("source", "role_fit_data"),
                    "message": fallback_result.get("message", "")
                }
            else:
                # Fallback failed, continue to error handling
                raise Exception(f"Fallback failed: {fallback_result.get('error', 'Unknown error')}")
        
        log.info(f"✅ Found {len(matched_jobs)} job matches from job_matcher_agent")
        
        # Sort by match score (descending) and take top_k
        # Handle different score formats (0-1 or 0-100)
        def get_score(match):
            score = match.get("match_score", 0) or match.get("score", 0)
            # If score is > 1, it's likely a percentage, normalize to 0-1
            if score > 1:
                score = score / 100.0
            return score
        
        matched_jobs.sort(key=get_score, reverse=True)
        top_matches = matched_jobs[:top_k]
        
        # Merge new (job_matcher) with existing (relevant_jobs, role_fit_suggestions)
        existing_list = (context.get("relevant_jobs") or []) + (context.get("role_fit_suggestions") or [])
        merged = _merge_jobs_new_and_existing(top_matches, existing_list, cap=max(top_k * 2, 10))
        
        log.info(f"✅ Returning {len(merged)} matches (new + existing)")
        
        # Return jobs in the SAME format as job_matcher_agent (no reformatting)
        # job_matcher_agent returns: rank, job_id, job_title, company, match_score,
        # skill_match_percentage, skill_match_count, skills_matched, skills_unmatched,
        # rationale, vector_similarity
        
        # Calculate metrics (same as job_matcher_agent)
        processing_time = _calculate_processing_time(start_time)
        confidence_score = (
            sum(j.get("match_score", 0) for j in merged) / len(merged)
            if merged else 0.0
        )
        
        log.info(f"✅ Found {len(merged)} job matches for uid={uid} (new + existing)")
        for i, job in enumerate(merged[:5], 1):
            job_title = job.get("job_title", job.get("title", "Unknown Position"))
            company = job.get("company", "Unknown Company")
            match_score = job.get("match_score", 0)
            match_pct = int(match_score * 100) if match_score <= 1 else int(match_score)
            log.info(f"   {i}. {job_title} at {company} (Match: {match_pct}%)")
        
        return {
            # Standard agent format (same as job_matcher_agent)
            "status": "completed",
            "node": "identify_relevant_jobs",
            "output": {
                "top_matches": merged,
                "total_matches_found": len(merged),
                "matching_method": "chatbot_triggered_job_matching",
                "job_matcher_status": "success"
            },
            
            # Metadata fields (same as job_matcher_agent)
            "matched_jobs": merged,
            "top_matches": merged,
            "total_matches_found": len(merged),
            "job_matcher_status": "success",
            "confidence_score": round(confidence_score, 4),
            "processing_time_seconds": round(processing_time, 3),
            "analysis_method": "job_matcher_agent",
            
            # Additional fields from job_matcher_agent state (preserve all metadata)
            "total_jobs_matched": len(merged),
            "semantic_filter_threshold": matcher_result.get("semantic_filter_threshold", 0.3),
            "processing_method": matcher_result.get("processing_method", "semantic_filter_batch_llm_v2"),
            "batches_processed": matcher_result.get("batches_processed", 0),
            
            # Original fields for backward compatibility
            "success": True,
            "relevant_jobs": merged,
            "count": len(merged),
            "source": "job_matching_agent",
            "message": f"Found {len(merged)} job matches that align with your skills and profile."
        }
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        log.error(f"Error finding job matches: {e}", exc_info=True)
        # Fallback to role fit suggestions on error
        try:
            context = await load_comprehensive_context(uid, session_id)
            fallback_result = await _fallback_to_role_fit_suggestions(uid, session_id, context, top_k)
            # Wrap fallback result in standard format
            if fallback_result.get("success"):
                return {
                    # Standard agent format
                    "status": "completed",
                    "node": "identify_relevant_jobs",
                    "output": {
                        "top_matches": fallback_result.get("relevant_jobs", []),
                        "total_matches_found": fallback_result.get("count", 0),
                        "matching_method": "role_fit_fallback",
                        "job_matcher_status": "fallback"
                    },
                    
                    # Metadata fields
                    "matched_jobs": fallback_result.get("relevant_jobs", []),
                    "top_matches": fallback_result.get("relevant_jobs", []),
                    "total_matches_found": fallback_result.get("count", 0),
                    "job_matcher_status": "fallback",
                    "confidence_score": 0.7,  # Lower confidence for fallback
                    "processing_time_seconds": round(processing_time, 3),
                    "analysis_method": "role_fit_fallback",
                    
                    # Original fields
                    "success": True,
                    "relevant_jobs": fallback_result.get("relevant_jobs", []),
                    "count": fallback_result.get("count", 0),
                    "source": fallback_result.get("source", "role_fit_data"),
                    "message": fallback_result.get("message", "")
                }
            else:
                raise e  # If fallback also failed, raise original error
        except Exception as fallback_error:
            log.error(f"Fallback also failed: {fallback_error}")
            return {
                "status": "error",
                "node": "identify_relevant_jobs",
                "error": f"Failed to find job matches: {str(e)}",
                "processing_time_seconds": round(processing_time, 3),
                # Original fields
                "success": False,
                "relevant_jobs": [],
                "count": 0
            }


def _synthetic_job_id() -> str:
    """Generate a stable id for fallback/merged job items so UI can key on job_id (same as pipeline)."""
    return f"job_fit_{_generate_request_id()}"


def _norm_job_item(e: Dict[str, Any], default_job_id: Optional[str] = None) -> Dict[str, Any]:
    """Normalize a job item to job_matcher shape: job_id, job_title, company, match_score, rationale."""
    job_id = e.get("job_id") or e.get("jd_id") or e.get("id") or default_job_id or _synthetic_job_id()
    job_title = (
        e.get("job_title")
        or e.get("jobTitle")
        or e.get("title")
        or e.get("position")
        or e.get("role")
        or e.get("designation")
        or e.get("path")
        or e.get("Role")
        or "Unknown"
    )
    company = e.get("company") or "Profile-based"
    match_score = e.get("match_score") or e.get("score") or e.get("match_percentage", 70) / 100.0 if isinstance(e.get("match_percentage"), (int, float)) else 0.7
    if match_score > 1:
        match_score = match_score / 100.0
    rationale = e.get("rationale") or e.get("match_reason") or "From your profile"
    return {
        "job_id": job_id,
        "job_title": job_title,
        "company": company,
        "match_score": float(match_score),
        "rationale": rationale,
    }


def _merge_jobs_new_and_existing(
    new_matches: List[Dict[str, Any]],
    existing_list: List[Dict[str, Any]],
    cap: int = 10
) -> List[Dict[str, Any]]:
    """Merge new job matches with existing (relevant_jobs, role_fit). New first, dedupe by (job_title, company), cap. All items get job_id for UI."""
    def _key(j: Dict[str, Any]) -> tuple:
        return (str(j.get("job_title") or j.get("title") or "").strip().lower(), str(j.get("company") or "").strip().lower())
    seen = {_key(j) for j in new_matches}
    out = list(new_matches)
    for e in existing_list or []:
        n = _norm_job_item(e)
        if _key(n) not in seen:
            seen.add(_key(n))
            out.append(n)
    return out[:cap]


async def _fallback_to_role_fit_suggestions(
    uid: str,
    session_id: Optional[str],
    context: Dict[str, Any],
    top_k: int
) -> Dict[str, Any]:
    """Fallback to role fit suggestions when job matching agents can't be used."""
    try:
        gap_doc = await run_blocking_io(get_gap_doc, uid)
        role_fit_suggestions = []
        
        if gap_doc:
            career_advisor = gap_doc.get("skill_and_career_advisor", {})
            gap_analysis = career_advisor.get("raw_skill_gap_analysis_output", {})
            role_fit_suggestions = gap_analysis.get("role_fit_suggestions", [])
        
        # If no role fit data, use career paths as job suggestions
        if not role_fit_suggestions:
            career_paths = context.get("career_paths", [])
            role_fit_suggestions = [
                {
                    "title": path.get("title") if isinstance(path, dict) else str(path),
                    "match_score": 0.8,
                    "match_percentage": 80,
                    "match_reason": "Based on your career path recommendations",
                    "source": "career_paths"
                }
                for path in career_paths[:top_k]
            ]
        
        # Normalize to job_matcher shape (job_id, job_title, company, match_score, rationale) for UI consistency
        normalized = [_norm_job_item(j) for j in role_fit_suggestions[:top_k]]
        return {
            "success": True,
            "relevant_jobs": normalized,
            "count": len(normalized),
            "source": "role_fit_data",
            "message": f"Found {len(normalized)} job suggestions based on your profile."
        }
    except Exception as e:
        log.error(f"Error in fallback: {e}")
        return {
            "success": False,
            "error": f"Failed to get job suggestions: {str(e)}",
            "relevant_jobs": [],
            "count": 0
        }


# ============================================================================
# FUNCTION DEFINITIONS FOR LLM (LangChain/Gemini function calling)
# ============================================================================

# ============================================================================
# QUICK WIN FEATURES: Resume Optimization, Salary Insights, Learning Path
# ============================================================================

async def get_resume_optimization_suggestions(
    uid: str,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze resume and provide optimization suggestions for ATS compatibility,
    keyword optimization, achievement quantification, and format improvements.
    """
    start_time = time.time()
    try:
        log.info(f"🔧 Handling function call: get_resume_optimization_suggestions | uid={uid}")
        
        # Load comprehensive context
        context = await load_comprehensive_context(uid, session_id)
        resume_data = context.get("resume_data", {})
        
        if not resume_data:
            processing_time = _calculate_processing_time(start_time)
            request_id = _generate_request_id()
            return {
                "status": "error",
                "node": "get_resume_optimization_suggestions",
                "error": "No resume data found. Please upload your resume first.",
                "suggestions": [],
                "processing_time": processing_time,
                "request_id": request_id,
                "success": False
            }
        
        suggestions = []
        
        # 1. Skills Analysis
        skills = resume_data.get("skills", []) or context.get("skills", [])
        if not skills or len(skills) < 5:
            suggestions.append({
                "category": "skills",
                "priority": "high",
                "issue": "Limited skills listed",
                "suggestion": "Add more relevant technical and soft skills. Aim for 10-15 key skills relevant to your target roles.",
                "action": "Review job descriptions for your target roles and add missing skills"
            })
        
        # 2. Experience Quantification
        experience = resume_data.get("work_experience") or resume_data.get("experience", [])
        quantified_count = 0
        for exp in experience:
            if isinstance(exp, dict):
                responsibilities = exp.get("responsibilities", []) or []
                desc = exp.get("description", "")
                text = " ".join(responsibilities) if isinstance(responsibilities, list) else str(desc)
                # Check for numbers (metrics, percentages, etc.)
                if re.search(r'\d+', text):
                    quantified_count += 1
        
        if quantified_count < len(experience) * 0.5:  # Less than 50% quantified
            suggestions.append({
                "category": "achievements",
                "priority": "high",
                "issue": "Missing quantified achievements",
                "suggestion": "Add numbers, metrics, and percentages to your work experience. For example: 'Increased sales by 30%' or 'Managed team of 5 developers'.",
                "action": "Review each role and add specific metrics (revenue, users, team size, time saved, etc.)"
            })
        
        # 3. ATS Keywords
        target_roles = context.get("role_fit_suggestions", []) or context.get("career_paths", [])
        if target_roles:
            # Extract common keywords from target roles
            role_text = " ".join([
                str(r.get("title", r.get("path", "")) if isinstance(r, dict) else r)
                for r in target_roles[:5]
            ])
            # Simple keyword extraction (in production, use more sophisticated analysis)
            common_keywords = ["experience", "skills", "development", "management", "analysis", "design", "implementation"]
            missing_keywords = []
            resume_text = str(resume_data).lower()
            for keyword in common_keywords:
                if keyword not in resume_text:
                    missing_keywords.append(keyword)
            
            if missing_keywords:
                suggestions.append({
                    "category": "ats_optimization",
                    "priority": "medium",
                    "issue": "Potential missing ATS keywords",
                    "suggestion": f"Consider including industry-standard keywords: {', '.join(missing_keywords[:5])}",
                    "action": "Review job descriptions for your target roles and naturally incorporate relevant keywords"
                })
        
        # 4. Education Section
        education = resume_data.get("education", [])
        if not education:
            suggestions.append({
                "category": "education",
                "priority": "medium",
                "issue": "Education section missing",
                "suggestion": "Add your educational background, including degrees, institutions, and graduation years if relevant.",
                "action": "Include at least your highest degree"
            })
        
        # 5. Projects Section
        projects = resume_data.get("projects", [])
        if not projects or len(projects) < 2:
            suggestions.append({
                "category": "projects",
                "priority": "medium",
                "issue": "Limited project showcase",
                "suggestion": "Add 2-3 key projects that demonstrate your skills. Include technologies used, your role, and impact/results.",
                "action": "Highlight projects that align with your target roles"
            })
        
        # 6. Summary/Objective
        resume_summary = context.get("resume_summary", "")
        if not resume_summary or len(resume_summary) < 50:
            suggestions.append({
                "category": "summary",
                "priority": "low",
                "issue": "Missing or brief professional summary",
                "suggestion": "Add a 2-3 sentence professional summary highlighting your key strengths, experience level, and career focus.",
                "action": "Write a concise summary that captures your value proposition"
            })
        
        # 7. Contact Information
        contact = resume_data.get("ContactDetails", {}) or resume_data.get("contact_details", {})
        if not contact or not contact.get("email"):
            suggestions.append({
                "category": "contact",
                "priority": "high",
                "issue": "Missing contact information",
                "suggestion": "Ensure your resume includes email, phone, and LinkedIn profile (if applicable).",
                "action": "Add complete contact details"
            })
        
        processing_time = _calculate_processing_time(start_time)
        request_id = _generate_request_id()
        
        return {
            "status": "completed",
            "node": "get_resume_optimization_suggestions",
            "output": {
                "suggestions": suggestions,
                "total_suggestions": len(suggestions),
                "high_priority_count": len([s for s in suggestions if s.get("priority") == "high"]),
                "resume_analysis": {
                    "has_skills": len(skills) > 0,
                    "skills_count": len(skills),
                    "experience_count": len(experience),
                    "quantified_experience_count": quantified_count,
                    "has_education": len(education) > 0,
                    "has_projects": len(projects) > 0,
                    "has_summary": bool(resume_summary)
                }
            },
            "suggestions": suggestions,
            "total_suggestions": len(suggestions),
            "analysis_method": "resume_analyzer",
            "confidence_score": 0.8,
            "processing_time": processing_time,
            "request_id": request_id,
            "success": True,
            "message": f"Found {len(suggestions)} optimization suggestions for your resume."
        }
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        request_id = _generate_request_id()
        log.error(f"Error getting resume optimization suggestions: {e}", exc_info=True)
        return {
            "status": "error",
            "node": "get_resume_optimization_suggestions",
            "error": str(e),
            "suggestions": [],
            "processing_time": processing_time,
            "request_id": request_id,
            "success": False
        }


def _normalize_rationale(val: Any) -> str:
    """Convert rationale (list or str) to string."""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val if x)
    return str(val)


def _extract_salary_trends_from_llm_response(text: str) -> Dict[str, Any]:
    """Extract salary_trends from LLM response (handles markdown-wrapped JSON and multiple shapes)."""
    if not text or not text.strip():
        return {}

    def _normalize_trends(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Return dict with current_level (and optional rationale) for downstream."""
        if not isinstance(parsed, dict):
            return {}
        # Already in salary_trends shape (current_level, current_level_rationale, etc.)
        if "current_level" in parsed:
            return parsed
        inner = parsed.get("salary_trends")
        if isinstance(inner, dict) and "current_level" in inner:
            return inner
        return {}

    # 1) Markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1).strip())
            out = _normalize_trends(parsed)
            if out:
                return out
        except json.JSONDecodeError:
            pass

    # 2) Raw JSON with "salary_trends" key
    obj_match = re.search(r"\{\s*[\"']salary_trends[\"']\s*:[\s\S]*\}", text)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group(0))
            out = _normalize_trends(parsed)
            if out:
                return out
        except json.JSONDecodeError:
            pass

    # 3) Whole text as JSON (no markdown)
    if '"current_level"' in text or '"salary_trends"' in text:
        try:
            parsed = json.loads(text.strip())
            out = _normalize_trends(parsed)
            if out:
                return out
        except json.JSONDecodeError:
            pass
    return {}


async def get_salary_insights(
    uid: str,
    session_id: Optional[str] = None,
    force_refresh: bool = False,
    lightweight: bool = True
) -> Dict[str, Any]:
    """
    Get salary insights. Lightweight path (default): resume + get_gap_doc only; uses
    cached salary_trends from gap_doc or runs salary-only LLM. Avoids load_comprehensive_context.
    When lightweight=False, uses full context first (for callers that need existing market data).
    """
    start_time = time.time()
    request_id = _generate_request_id()
    try:
        resume_data: Dict[str, Any] = {}
        salary_trends: Dict[str, Any] = {}
        market_insights_raw: Any = []
        data_source = "gap_doc_cache"

        if lightweight:
            # Lightweight path: no load_comprehensive_context — resume + gap_doc only (like courses_only)
            structured_resume = await run_blocking_io(fetch_structured_resume, uid, None, session_id)
            if not structured_resume:
                processing_time = _calculate_processing_time(start_time)
                return {
                    "status": "error",
                    "node": "get_salary_insights",
                    "error": "No resume data found. Please upload your resume first.",
                    "salary_insights": {},
                    "processing_time": processing_time,
                    "request_id": request_id,
                    "success": False,
                }
            resume_data = structured_resume or {}
            gap_doc = await run_blocking_io(get_gap_doc, uid)
            if gap_doc and not force_refresh:
                market_data = gap_doc.get("market_and_course_recommender", {})
                salary_trends = market_data.get("salary_trends", {}) or {}
                market_insights_raw = market_data.get("market_insights", []) or []
            if not salary_trends:
                log.info(f"🔧 get_salary_insights: Lightweight path, no cache — running salary-only LLM | uid={uid}")
                career_advisor = (gap_doc or {}).get("skill_and_career_advisor", {})
                skill_gap_analysis = career_advisor.get("raw_skill_gap_analysis_output", {}) or {}
                location = _extract_location_from_resume(resume_data)
                currency, region = _derive_currency_from_location(location)
                salary_prompt = generate_salary_only_prompt(resume_data, skill_gap_analysis, currency=currency, region=region)
                try:
                    llm_response = await asyncio.wait_for(
                        invoke_llm(
                            prompt=salary_prompt,
                            task_type="salary_insights",
                            agent_name="career_coach",
                            max_output_tokens=800,
                        ),
                        timeout=15,
                    )
                    llm_text = _to_text(llm_response) if llm_response else ""
                    salary_trends = _extract_salary_trends_from_llm_response(llm_text)
                    data_source = "salary_only_llm"
                except (asyncio.TimeoutError, Exception) as e:
                    log.warning(f"Salary-only LLM call failed: {e}")
        else:
            # Full context path (e.g. dashboard reusing existing analysis)
            context = await load_comprehensive_context(uid, session_id)
            salary_trends = context.get("salary_trends", {}) if not force_refresh else {}
            market_insights_raw = context.get("market_insights", []) or []
            resume_data = context.get("resume_data", {}) or {}
            data_source = "context" if salary_trends else "salary_only_llm"
            if not salary_trends:
                log.info(f"🔧 get_salary_insights: No context data, running salary-only LLM | uid={uid}")
                gap_doc = await run_blocking_io(get_gap_doc, uid)
                skill_gap_analysis = {}
                if gap_doc:
                    career_advisor = gap_doc.get("skill_and_career_advisor", {})
                    skill_gap_analysis = career_advisor.get("raw_skill_gap_analysis_output", {}) or {}
                currency = context.get("candidate_currency", "₹")
                region = context.get("salary_region", "Indian")
                salary_prompt = generate_salary_only_prompt(resume_data, skill_gap_analysis, currency=currency, region=region)
                try:
                    llm_response = await asyncio.wait_for(
                        invoke_llm(
                            prompt=salary_prompt,
                            task_type="salary_insights",
                            agent_name="career_coach",
                            max_output_tokens=800,
                        ),
                        timeout=15,
                    )
                    llm_text = _to_text(llm_response) if llm_response else ""
                    salary_trends = _extract_salary_trends_from_llm_response(llm_text)
                except (asyncio.TimeoutError, Exception) as e:
                    log.warning(f"Salary-only LLM call failed: {e}")

        location = _extract_location_from_resume(resume_data)
        currency, region = _derive_currency_from_location(location)
        loc = location or "Not specified"

        if not salary_trends:
            processing_time = _calculate_processing_time(start_time)
            return {
                "status": "completed",
                "node": "get_salary_insights",
                "output": {
                    "salary_insights": {},
                    "data_source": data_source,
                    "message": f"Salary data not available. Provide general compensation guidance using {currency} and {region} market (candidate location from resume: {loc})."
                },
                "salary_insights": {},
                "analysis_method": "context_only",
                "processing_time": processing_time,
                "request_id": request_id,
                "success": True,
                "message": f"No personalized salary data. Share general compensation advice using {currency} and {region} market (candidate location: {loc})."
            }

        current_level = salary_trends.get("current_level", "Not available")
        current_rationale = _normalize_rationale(salary_trends.get("current_level_rationale", ""))
        next_level = salary_trends.get("next_level", "Not available")
        next_rationale = _normalize_rationale(salary_trends.get("next_level_rationale", ""))
        future_expectations = salary_trends.get("future_expectations", "Not available")
        future_rationale = _normalize_rationale(salary_trends.get("future_expectations_rationale", ""))

        insights = {
            "current_level": {"salary_range": current_level, "rationale": current_rationale},
            "next_level": {"salary_range": next_level, "rationale": next_rationale, "growth_potential": "20-40% increase"},
            "future_expectations": {"salary_range": future_expectations, "rationale": future_rationale, "timeline": "5+ years", "growth_potential": "50-100% increase"},
            "market_insights": market_insights_raw[:3] if isinstance(market_insights_raw, list) else []
        }

        processing_time = _calculate_processing_time(start_time)
        return {
            "status": "completed",
            "node": "get_salary_insights",
            "output": {"salary_insights": insights, "data_source": data_source},
            "salary_insights": insights,
            "analysis_method": "gap_doc_cache" if data_source == "gap_doc_cache" else ("context" if data_source == "context" else "salary_only_llm"),
            "confidence_score": 0.8,
            "processing_time": processing_time,
            "request_id": request_id,
            "success": True,
            "message": "Salary insights retrieved based on your profile and market data."
        }

    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        log.error(f"Error getting salary insights: {e}", exc_info=True)
        return {
            "status": "error",
            "node": "get_salary_insights",
            "error": str(e),
            "salary_insights": {},
            "processing_time": processing_time,
            "request_id": request_id,
            "success": False
        }


async def get_market_insights_only(
    uid: str,
    session_id: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Get market insights, salary trends, and career paths only — no course recommender.
    Lightweight: tries gap_doc cache first; if missing, composes from get_career_advice +
    get_salary_insights (no market_and_course_recommender_agent). Use when user asks for
    market analysis / salary outlook / career paths but NOT for courses.
    """
    start_time = time.time()
    request_id = _generate_request_id()
    try:
        gap_doc = await run_blocking_io(get_gap_doc, uid)
        market_data = (gap_doc or {}).get("market_and_course_recommender", {})
        market_insights = market_data.get("market_insights", []) or []
        salary_trends = market_data.get("salary_trends", {}) or {}
        career_paths = market_data.get("career_paths", []) or []

        if (market_insights or salary_trends or career_paths) and not force_refresh:
            log.info(f"✅ get_market_insights_only: Using cached market data from gap_doc | uid={uid}")
            processing_time = _calculate_processing_time(start_time)
            return {
                "status": "completed",
                "node": "get_market_insights_only",
                "output": {
                    "market_insights": market_insights,
                    "salary_trends": salary_trends,
                    "career_paths": career_paths,
                },
                "market_insights": market_insights,
                "salary_trends": salary_trends,
                "career_paths": career_paths,
                "source": "cached",
                "processing_time": processing_time,
                "request_id": request_id,
                "success": True,
                "message": "Market insights retrieved from your existing analysis."
            }

        log.info(f"🔄 get_market_insights_only: No cache — composing from career_advice + salary_insights | uid={uid}")
        career_result = await get_career_advice(uid=uid, session_id=session_id, force_refresh=force_refresh)
        salary_result = await get_salary_insights(uid=uid, session_id=session_id, force_refresh=force_refresh, lightweight=True)

        career_paths = career_result.get("career_paths", []) or []
        salary_insights = salary_result.get("salary_insights", {}) or {}
        salary_trends = salary_insights
        market_insights = ["Your market outlook and career paths are ready. Unlock your full report for detailed insights."]

        processing_time = _calculate_processing_time(start_time)
        return {
            "status": "completed",
            "node": "get_market_insights_only",
            "output": {
                "market_insights": market_insights,
                "salary_trends": salary_trends,
                "career_paths": career_paths,
            },
            "market_insights": market_insights,
            "salary_trends": salary_trends,
            "career_paths": career_paths,
            "source": "composed",
            "processing_time": processing_time,
            "request_id": request_id,
            "success": True,
            "message": "Market insights prepared from your profile (career paths and salary outlook)."
        }
    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        log.error(f"Error in get_market_insights_only: {e}", exc_info=True)
        return {
            "status": "error",
            "node": "get_market_insights_only",
            "error": str(e),
            "market_insights": [],
            "salary_trends": {},
            "career_paths": [],
            "processing_time": processing_time,
            "request_id": request_id,
            "success": False,
        }


async def get_learning_path(
    uid: str,
    session_id: Optional[str] = None,
    topic: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a structured learning path from courses and assessments, organized by difficulty
    and progression (beginner → intermediate → advanced).
    """
    start_time = time.time()
    try:
        log.info(f"🔧 Handling function call: get_learning_path | uid={uid}, topic={topic}")
        
        # Load comprehensive context
        context = await load_comprehensive_context(uid, session_id)
        
        # Get courses and assessments
        courses = context.get("course_recommendations", []) or []
        assessments = context.get("assessment_plan", []) or []
        
        # When topic is specified: merge existing + new from KB topic search, then filter by topic
        if topic:
            topic_lower = topic.lower()
            # Add new courses from KB for this topic (existing + new)
            structured_resume = context.get("resume_data")
            new_courses = _quick_course_lookup_by_topics([topic], structured_resume)
            courses = _dedupe_courses((courses or []) + (new_courses or []))
            courses = [
                c for c in courses
                if topic_lower in str(c.get("course") or c.get("title", "")).lower() or
                   topic_lower in str(c.get("description", "")).lower() or
                   topic_lower in str(c.get("target_skill", "")).lower()
            ]
            assessments = [
                a for a in assessments
                if topic_lower in str(a.get("topic", "")).lower()
            ]
        
        # Normalize courses to canonical shape (course_id, title, url, provider, relevance_score, description)
        courses = [_norm_course_item(c) for c in courses]
        
        # Organize courses by difficulty/level
        beginner_courses = []
        intermediate_courses = []
        advanced_courses = []
        
        difficulty_keywords = {
            "beginner": ["beginner", "intro", "introduction", "basics", "fundamentals", "101", "getting started", "easy"],
            "intermediate": ["intermediate", "advanced", "deep dive", "expert", "mastery", "professional"]
        }
        
        for course in courses:
            # Handle both "course" and "title" field names
            course_name = str(course.get("course") or course.get("title", "")).lower()
            desc = str(course.get("description", "")).lower()
            text = f"{course_name} {desc}"
            
            if any(kw in text for kw in difficulty_keywords["beginner"]):
                beginner_courses.append(course)
            elif any(kw in text for kw in difficulty_keywords["intermediate"]):
                if "expert" in text or "mastery" in text:
                    advanced_courses.append(course)
                else:
                    intermediate_courses.append(course)
            else:
                # Default to intermediate if unclear
                intermediate_courses.append(course)
        
        # Organize assessments by difficulty
        beginner_assessments = []
        intermediate_assessments = []
        advanced_assessments = []
        
        for assessment in assessments:
            difficulty = str(assessment.get("difficulty", "medium")).lower()
            if difficulty in ["easy", "beginner"]:
                beginner_assessments.append(assessment)
            elif difficulty in ["hard", "expert", "advanced"]:
                advanced_assessments.append(assessment)
            else:
                intermediate_assessments.append(assessment)
        
        # Build learning path stages
        learning_path = {
            "topic": topic or "General Skills Development",
            "stages": []
        }
        
        # Stage 1: Foundation (Beginner)
        if beginner_courses or beginner_assessments:
            learning_path["stages"].append({
                "stage": 1,
                "level": "beginner",
                "name": "Foundation",
                "description": "Build fundamental knowledge and skills",
                "courses": beginner_courses[:3],  # Top 3 beginner courses
                "assessments": beginner_assessments[:2],  # Top 2 beginner assessments
                "estimated_duration": "2-4 weeks",
                "next_stage": "intermediate"
            })
        
        # Stage 2: Intermediate
        if intermediate_courses or intermediate_assessments:
            learning_path["stages"].append({
                "stage": 2,
                "level": "intermediate",
                "name": "Intermediate Skills",
                "description": "Develop practical skills and apply knowledge",
                "courses": intermediate_courses[:3],
                "assessments": intermediate_assessments[:2],
                "estimated_duration": "4-8 weeks",
                "prerequisites": "Complete foundation stage",
                "next_stage": "advanced"
            })
        
        # Stage 3: Advanced
        if advanced_courses or advanced_assessments:
            learning_path["stages"].append({
                "stage": 3,
                "level": "advanced",
                "name": "Advanced Mastery",
                "description": "Master advanced concepts and specialize",
                "courses": advanced_courses[:3],
                "assessments": advanced_assessments[:2],
                "estimated_duration": "8-12 weeks",
                "prerequisites": "Complete intermediate stage"
            })
        
        # Calculate total duration
        total_weeks = sum([
            int(re.search(r'(\d+)', stage.get("estimated_duration", "0")).group(1))
            if re.search(r'(\d+)', stage.get("estimated_duration", "0"))
            else 0
            for stage in learning_path["stages"]
        ])
        
        learning_path["total_estimated_duration"] = f"{total_weeks} weeks"
        learning_path["total_courses"] = len(courses)
        learning_path["total_assessments"] = len(assessments)
        
        processing_time = _calculate_processing_time(start_time)
        request_id = _generate_request_id()
        
        return {
            "status": "completed",
            "node": "get_learning_path",
            "output": {
                "learning_path": learning_path,
                "summary": {
                    "total_stages": len(learning_path["stages"]),
                    "total_courses": learning_path["total_courses"],
                    "total_assessments": learning_path["total_assessments"],
                    "estimated_duration": learning_path["total_estimated_duration"]
                }
            },
            "learning_path": learning_path,
            "analysis_method": "learning_path_builder",
            "confidence_score": 0.8,
            "processing_time": processing_time,
            "request_id": request_id,
            "success": True,
            "message": f"Created a {len(learning_path['stages'])}-stage learning path for {learning_path['topic']}."
        }
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time) if 'start_time' in locals() else 0.0
        request_id = _generate_request_id()
        log.error(f"Error getting learning path: {e}", exc_info=True)
        return {
            "status": "error",
            "node": "get_learning_path",
            "error": str(e),
            "learning_path": {},
            "processing_time": processing_time,
            "request_id": request_id,
            "success": False
        }


def get_function_definitions() -> List[Dict[str, Any]]:
    """
    Return function definitions for LangChain/Gemini function calling.
    These tell the LLM what functions are available and when to use them.
    """
    return [
        {
            "name": "get_course_recommendations",
            "description": (
                "Get personalized course recommendations based on user's skills, gaps, and career goals. "
                "This function checks existing data first (fast), and only generates fresh recommendations if needed. "
                "Use this when user asks about courses, learning resources, skill development, or training. "
                "If the user asks for courses on specific topics, pass those topics and set courses_only=true to trigger a fast courses-only path (skips market/salary). "
                "Always extract explicit topics (e.g., 'Python', 'React') from the user's request and include them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "string",
                        "description": "User ID"
                    },
                    "force_refresh": {
                        "type": "boolean",
                        "description": "If true, generate fresh recommendations even if existing data is available. Default: false (use existing data if available).",
                        "default": False
                    },
                    "topics": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "Optional list of specific topics to target courses for (e.g., ['Python', 'React']). "
                            "If provided, the function will run a lightweight courses-only lookup without market/salary analysis."
                        )
                    },
                    "courses_only": {
                        "type": "boolean",
                        "description": (
                            "If true, return courses only (skip market insights/salary paths). "
                            "Automatically true when topics are provided."
                        ),
                        "default": False
                    }
                },
                "required": ["uid"]
            }
        },
        {
            "name": "get_assessment_recommendations",
            "description": (
                "Get personalized assessment recommendations to test user's skills. "
                "This function checks existing data first (fast), and only generates fresh recommendations if needed. "
                "Can create recommendations for specific topics in two ways:\n"
                "1. PROACTIVE SUGGESTIONS: You can suggest topics based on user's skill gaps, career paths, or improvement areas. "
                "   Analyze their profile and identify 3-5 relevant topics they should assess (e.g., missing skills, career path requirements).\n"
                "2. USER REQUESTS: If the user explicitly asks for assessments on specific topics (e.g., 'I want to test my Python skills'), "
                "   use those exact topics.\n"
                "Use this when:\n"
                "- User asks about assessments, skill tests, or wants to evaluate their abilities\n"
                "- You identify skill gaps or areas they should assess (proactively suggest topics)\n"
                "- User requests assessments for specific topics (use their requested topics)\n"
                "If no topics are provided, the function will return existing recommendations or generate fresh ones from the assessment recommender agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "string",
                        "description": "User ID"
                    },
                    "force_refresh": {
                        "type": "boolean",
                        "description": "If true, generate fresh recommendations even if existing data is available. Default: false (use existing data if available).",
                        "default": False
                    },
                    "topics": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": (
                            "Optional list of specific topics to create assessments for (e.g., ['Python', 'React', 'System Design']). "
                            "If provided, creates custom assessment recommendations for these topics (even if assessments already exist for other topics). "
                            "Can be: (1) Topics the user explicitly requests (e.g., 'I want to test my Python skills' → use ['Python']), "
                            "or (2) Topics you proactively suggest based on skill gaps/career goals. "
                            "IMPORTANT: If user asks for a specific topic, extract that topic and pass it here. "
                            "If no topics provided and user just asks for 'assessments', set suggest_new_topics=true instead."
                        )
                    },
                    "suggest_new_topics": {
                        "type": "boolean",
                        "description": (
                            "If true and no topics provided, the function will suggest topics based on the user's profile "
                            "and create assessments only for topics that don't already have assessments. "
                            "Use this when user asks for 'assessments' or 'what assessments should I take' without specifying topics."
                        ),
                        "default": False
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard", "expert"],
                        "description": "Optional difficulty level for custom topic assessments. Defaults to 'medium' if not specified. Only used when topics are provided."
                    }
                },
                "required": ["uid"]
            }
        },
        {
            "name": "get_resume_score",
            "description": (
                "Get resume score and role suggestions. Use when user asks about resume quality, "
                "how their resume scores, role fit, or what roles suit their profile. "
                "Checks existing data first; calls resume scorer agent if needed or if user requests refresh."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string", "description": "User ID"},
                    "force_refresh": {
                        "type": "boolean",
                        "description": "If true, run fresh resume scoring even if data exists.",
                        "default": False
                    }
                },
                "required": ["uid"]
            }
        },
        {
            "name": "get_career_advice",
            "description": (
                "Get career advice: career paths, role fit suggestions, skill gaps, improvement recommendations. "
                "Uses career advisor agent (fast). Use when user asks about career paths, "
                "what roles they're qualified for, skill gaps, or career guidance. "
                "Checks existing data first; calls agent if needed or if user requests refresh."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string", "description": "User ID"},
                    "force_refresh": {
                        "type": "boolean",
                        "description": "If true, run fresh career analysis even if data exists.",
                        "default": False
                    }
                },
                "required": ["uid"]
            }
        },
        {
            "name": "get_salary_insights",
            "description": (
                "Get personalized salary insights (current level, rationale, location comparison). "
                "Uses existing data first (instant); falls back to lightweight salary-only LLM call if no prior analysis. "
                "Use when user asks about salary, compensation, expected pay, or salary range for their profile."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string", "description": "User ID"},
                    "force_refresh": {
                        "type": "boolean",
                        "description": "If true, run fresh salary analysis even if data exists.",
                        "default": False
                    }
                },
                "required": ["uid"]
            }
        },
        {
            "name": "get_market_insights_only",
            "description": (
                "Get market insights, salary trends, and career paths only — no courses. "
                "Lightweight: uses cache or composes from career + salary (no full course recommender). "
                "Call when user asks for market analysis, salary outlook, or career/market overview but NOT for course recommendations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string", "description": "User ID"},
                    "force_refresh": {
                        "type": "boolean",
                        "description": "If true, recompute market/career/salary even if cached.",
                        "default": False
                    }
                },
                "required": ["uid"]
            }
        },
        {
            "name": "create_career_goal",
            "description": (
                "Create a career goal for the user with optional milestones and tasks. "
                "Use this when the user wants to set a goal, target, or objective for their career development. "
                "Always extract the goal title and description from the user's message. "
                "Include milestone/task details if provided by the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "string",
                        "description": "User ID"
                    },
                    "goal_title": {
                        "type": "string",
                        "description": "Title of the career goal (e.g., 'Become Senior Product Manager')"
                    },
                    "goal_description": {
                        "type": "string",
                        "description": "Detailed description of the goal and why it's important"
                    },
                    "target_date": {
                        "type": "string",
                        "description": "Target completion date (YYYY-MM-DD format, optional)"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Priority level of the goal",
                        "default": "medium"
                    },
                    "milestones": {
                        "type": "array",
                        "description": "Optional milestones for this goal (max 5).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "due_date": {"type": "string"},
                                "status": {"type": "string"},
                                "progress": {"type": "number"},
                                "tasks": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "title": {"type": "string"},
                                            "status": {"type": "string"},
                                            "priority": {"type": "string"},
                                            "due_date": {"type": "string", "description": "Due date in ISO format or natural language (e.g., 'in 2 days', 'next week'). If not provided, system will auto-calculate from effort_hours and difficulty."},
                                            "effort_hours": {"type": "number", "description": "Estimated hours to complete. Always include this field. If not provided, system will estimate from task type and difficulty."},
                                            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard", "expert"], "description": "Task difficulty level (helps with effort estimation if effort_hours not provided)"},
                                            "type": {"type": "string", "enum": ["course", "assessment", "project", "study"], "description": "Task type (helps with effort estimation if effort_hours not provided)"},
                                            "notes": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "tasks": {
                        "type": "array",
                        "description": "Optional goal-level tasks (max 20). Each task should have: title, status, priority, due_date (ISO date or natural language), effort_hours (estimated hours), difficulty (easy/medium/hard/expert), type (course/assessment/project/study), notes. IMPORTANT: Always include effort_hours and due_date - if user specifies time, use that; otherwise estimate based on task type and difficulty.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "status": {"type": "string"},
                                "priority": {"type": "string"},
                                "due_date": {"type": "string", "description": "Due date in ISO format or natural language (e.g., 'in 2 days', 'next week'). If not provided, system will auto-calculate from effort_hours and difficulty."},
                                "effort_hours": {"type": "number", "description": "Estimated hours to complete. Always include this field. If not provided, system will estimate from task type and difficulty."},
                                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard", "expert"], "description": "Task difficulty level (helps with effort estimation if effort_hours not provided)"},
                                "type": {"type": "string", "enum": ["course", "assessment", "project", "study"], "description": "Task type (helps with effort estimation if effort_hours not provided)"},
                                "notes": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["uid", "goal_title", "goal_description"]
            }
        },
        {
            "name": "get_career_goals",
            "description": (
                "Retrieve all career goals (with milestones and tasks) for the user. "
                "Use when user asks to see their goals, milestones, or tasks, or to check progress."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "string",
                        "description": "User ID"
                    }
                },
                "required": ["uid"]
            }
        },
        {
            "name": "update_career_goal",
            "description": (
                "Update a career goal's status, progress, priority, target_date, or add milestones/tasks, "
                "or mark tasks complete. Use when the user wants to update progress or add steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string", "description": "User ID"},
                    "goal_id": {"type": "string", "description": "Goal ID to update"},
                    "status": {"type": "string", "description": "New status"},
                    "progress": {"type": "number", "description": "Progress between 0 and 1"},
                    "priority": {"type": "string", "description": "Priority (low|medium|high)"},
                    "target_date": {"type": "string", "description": "New target date (YYYY-MM-DD)"},
                    "add_milestones": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Milestones to add (max 5 per call)"
                    },
                    "add_tasks": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Tasks to add (max 20 per call). Optionally include milestone_id to attach."
                    },
                    "complete_tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task IDs to mark completed"
                    }
                },
                "required": ["uid", "goal_id"]
            }
        },
        {
            "name": "identify_relevant_jobs",
            "description": (
                "Search the job database and return actual job listings with companies and locations. "
                "ONLY call this when the user EXPLICITLY asks to SEE, FIND, or SEARCH for job listings from the database. "
                "\n"
                "Examples of when to CALL this function (wants database search):\n"
                "- 'Show me available jobs'\n"
                "- 'Find jobs that match my profile'\n"
                "- 'What job opportunities are in the database?'\n"
                "- 'Search for suitable jobs for me'\n"
                "- 'Are there any open positions?'\n"
                "\n"
                "Examples of when NOT to call (answer from context instead):\n"
                "- 'What jobs would I be suitable for?' → Use career_paths from context\n"
                "- 'What careers should I pursue?' → Use career_paths from context\n"
                "- 'What roles am I qualified for?' → Use role_fit_suggestions from context\n"
                "- 'What career options do I have?' → Use career_paths from context\n"
                "\n"
                "Key distinction: If they want DATABASE SEARCH/LISTINGS with companies and locations, call this. "
                "If they want GENERAL CAREER ADVICE about roles/paths, use the career_paths and role_fit_suggestions already in your context. "
                "This function is expensive (runs job_matcher_agent) - only use when user clearly wants actual job listings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "string",
                        "description": "User ID"
                    },
                    "search_query": {
                        "type": "string",
                        "description": "Optional custom search query. If not provided, automatically builds from user's skills and career goals."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top jobs to return (default: 3, max: 3). Returns the best matches aligned with user's skills.",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 3
                    }
                },
                "required": ["uid"]
            }
        },
        {
            "name": "get_resume_optimization_suggestions",
            "description": (
                "Analyze the user's resume and provide optimization suggestions for ATS compatibility, "
                "keyword optimization, achievement quantification, format improvements, and missing sections. "
                "Use this when the user asks about improving their resume, resume tips, ATS optimization, "
                "or resume feedback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "string",
                        "description": "User ID"
                    }
                },
                "required": ["uid"]
            }
        },
        {
            "name": "get_learning_path",
            "description": (
                "Create a structured learning path from courses and assessments, organized by difficulty "
                "and progression (beginner → intermediate → advanced). "
                "Use this when the user asks for a learning roadmap, structured learning plan, "
                "or how to progress in a skill/topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "string",
                        "description": "User ID"
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional specific topic to create a learning path for (e.g., 'Python', 'Data Science'). If not provided, creates a general learning path."
                    }
                },
                "required": ["uid"]
            }
        }
    ]


# ============================================================================
# FUNCTION CALL HANDLER
# ============================================================================

async def handle_function_call(
    function_name: str,
    arguments: Dict[str, Any],
    uid: str,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle function calls from LLM.
    Routes to appropriate function based on function_name.
    
    Args:
        function_name: Name of the function to call
        arguments: Function arguments (from LLM)
        uid: User ID (will override if in arguments)
        session_id: Optional session ID
    
    Returns:
        Function result dictionary
    """
    log.info(f"🔧 Handling function call: {function_name} | uid={uid}")
    
    try:
        # Handle nested kwargs structure (LangChain sometimes wraps args in 'kwargs')
        if isinstance(arguments, dict) and "kwargs" in arguments and len(arguments) == 1:
            arguments = arguments["kwargs"]
            log.debug(f"📦 Unwrapped kwargs in handle_function_call: {arguments}")
        
        # Ensure uid is set (from arguments or parameter)
        arguments["uid"] = arguments.get("uid", uid)
        force_refresh = arguments.get("force_refresh", False)
        
        if function_name == "get_course_recommendations":
            return await get_course_recommendations(
                uid=arguments["uid"],
                session_id=session_id,
                force_refresh=force_refresh,
                topics=arguments.get("topics"),
                courses_only=arguments.get("courses_only", False)
            )
        
        elif function_name == "get_assessment_recommendations":
            return await get_assessment_recommendations(
                uid=arguments["uid"],
                session_id=session_id,
                force_refresh=force_refresh,
                topics=arguments.get("topics"),
                difficulty=arguments.get("difficulty"),
                suggest_new_topics=arguments.get("suggest_new_topics", False)
            )
        
        elif function_name == "get_resume_score":
            return await get_resume_score(
                uid=arguments["uid"],
                session_id=session_id,
                force_refresh=arguments.get("force_refresh", False)
            )
        
        elif function_name == "get_career_advice":
            return await get_career_advice(
                uid=arguments["uid"],
                session_id=session_id,
                force_refresh=arguments.get("force_refresh", False)
            )
        
        elif function_name == "get_enhanced_role_fit":
            return await get_enhanced_role_fit(
                uid=arguments["uid"],
                session_id=session_id,
                force_refresh=arguments.get("force_refresh", False)
            )
        
        elif function_name == "get_salary_insights":
            return await get_salary_insights(
                uid=arguments["uid"],
                session_id=session_id,
                force_refresh=arguments.get("force_refresh", False),
                lightweight=True
            )
        
        elif function_name == "get_market_insights_only":
            return await get_market_insights_only(
                uid=arguments["uid"],
                session_id=session_id,
                force_refresh=arguments.get("force_refresh", False)
            )
        
        elif function_name == "create_career_goal":
            # Support both "title"/"description" and "goal_title"/"goal_description" for flexibility
            goal_title = arguments.get("goal_title") or arguments.get("title", "")
            goal_description = arguments.get("goal_description") or arguments.get("description", "")
            target_date_raw = arguments.get("target_date")
            
            # Normalize date from natural language to ISO format
            target_date = _normalize_date(target_date_raw) if target_date_raw else None
            
            return await create_career_goal(
                uid=arguments["uid"],
                goal_title=goal_title,
                goal_description=goal_description,
                target_date=target_date,
                priority=arguments.get("priority", "medium"),
                session_id=session_id,
                milestones=arguments.get("milestones"),
                tasks=arguments.get("tasks")
            )
        
        elif function_name == "get_career_goals":
            return await get_career_goals(
                uid=arguments["uid"],
                session_id=session_id
            )
        
        elif function_name == "update_career_goal":
            return await update_career_goal(
                uid=arguments["uid"],
                goal_id=arguments.get("goal_id", ""),
                session_id=session_id,
                status=arguments.get("status"),
                progress=arguments.get("progress"),
                priority=arguments.get("priority"),
                target_date=arguments.get("target_date"),
                add_milestones=arguments.get("add_milestones"),
                add_tasks=arguments.get("add_tasks"),
                complete_tasks=arguments.get("complete_tasks")
            )
        
        elif function_name == "identify_relevant_jobs":
            return await identify_relevant_jobs(
                uid=arguments["uid"],
                session_id=session_id,
                search_query=arguments.get("search_query"),
                top_k=arguments.get("top_k", 10)
            )
        
        elif function_name == "get_resume_optimization_suggestions":
            return await get_resume_optimization_suggestions(
                uid=arguments["uid"],
                session_id=session_id
            )
        
        elif function_name == "get_learning_path":
            return await get_learning_path(
                uid=arguments["uid"],
                session_id=session_id,
                topic=arguments.get("topic")
            )
        
        else:
            return {"error": f"Unknown function: {function_name}"}
            
    except Exception as e:
        log.error(f"Error handling function call {function_name}: {e}", exc_info=True)
        return {"error": f"Function call failed: {str(e)}"}


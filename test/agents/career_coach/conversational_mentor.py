"""
Career Chatbot - Conversational career guidance agent with streaming support.

Reuses patterns from interview agent:
- Session management for conversation history
- Request parsing and validation
- Context loading (using comprehensive context aggregator)
- Simple conversational flow

NEW: Streaming LLM responses for faster perceived performance

CONCURRENCY SAFETY:
- All operations are async and non-blocking
- Session isolation: Each user's sessions are isolated by uid
- Thread-safe: Uses run_blocking_io for ChromaDB operations
- Unique session IDs: Format {uid}_{timestamp}_{random} ensures uniqueness
- No shared mutable state: Each request has isolated context and history
- Supports multiple concurrent users without conflicts
"""

import ast
import json
import logging
import re
import time
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, AsyncIterator
from pydantic import BaseModel, Field
from fastapi import Request
from fastapi.responses import StreamingResponse
from langsmith.run_helpers import traceable
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from models.llm_invoker import _create_model_instance
from .function_tools import get_function_definitions, handle_function_call
from core.model_registry import get_model_for_task, TaskType
from chroma import get_chat_session, update_chat_session, list_sessions_by_uid, normalize_assessment_status
from core.utils import run_blocking_io, _to_text, _extract_json_from_response
from core.config import get_agent_config
from core.logging_helpers import create_log_context, log_agent_completion
from core.security import sanitize_text_for_llm
from settings import settings
from .context_aggregator import load_comprehensive_context
from .conversation_analyzer import (
    extract_aspirations_from_conversation,
    extract_goals_from_message,
    analyze_conversation_patterns
)

log = logging.getLogger(__name__)


def _extract_text_from_chunk(content: Any) -> str:
    """
    Extract plain displayable text from LLM chunk content.
    Handles Gemini-style list of dicts (e.g. [{'type': 'text', 'text': '...'}]),
    or content that was stringified to that form. Returns only text; no extras/signature.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        s = content.strip()
        # Already plain text
        if not (s.startswith("[") and "'text'" in s or '"text"' in s):
            return content
        # Try to parse as Python repr of list of dicts
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                parts = []
                for item in parsed:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(str(item["text"]))
                if parts:
                    return "".join(parts)
        except (ValueError, SyntaxError):
            pass
        # Try JSON (double-quoted)
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                parts = []
                for item in parsed:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(str(item["text"]))
                if parts:
                    return "".join(parts)
        except (json.JSONDecodeError, TypeError):
            pass
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        if parts:
            return "".join(parts)
        return ""
    return str(content)


# Premium tools: chat must only show a teaser; full data is in Premium Insights section.
# We pass a minimal "report ready" payload to the LLM so it cannot leak specific insights.
PREMIUM_INSIGHTS_TOOLS = frozenset({
    "get_salary_insights",
    "get_market_insights_only",
    "get_career_advice",
    "get_enhanced_role_fit",
})

# Job matches: in chat show only title, company, match score; rationale is paid (Jobs section in JobsifyAI).
JOBS_SECTION_TEASER_TOOLS = frozenset({"identify_relevant_jobs"})


def _tool_result_for_llm(func_name: str, func_result: Dict[str, Any]) -> str:
    """
    For premium tools, return a minimal teaser payload so the LLM cannot quote or paraphrase
    specific insights. Full result is still stored in tool_results for callbacks/dashboard.
    For job matches (identify_relevant_jobs): show only title, company, match_score; nudge to Jobs section in JobsifyAI.
    """
    if func_name in JOBS_SECTION_TEASER_TOOLS:
        jobs = func_result.get("relevant_jobs") or func_result.get("top_matches") or func_result.get("output", {}).get("top_matches") or []
        minimal = []
        for j in (jobs if isinstance(jobs, list) else [])[:10]:
            if isinstance(j, dict):
                title = (j.get("job_title") or j.get("title") or j.get("jobTitle") or "Role").strip()
                company = (j.get("company") or "Company").strip()
                score = j.get("match_score") or j.get("score") or 0
                if isinstance(score, (int, float)) and score > 1:
                    score = score / 100.0
                minimal.append({"job_title": title or "Role", "company": company or "Company", "match_score": float(score)})
        teaser = {
            "jobs_teaser": True,
            "jobs": minimal,
            "message": (
                "Show only job title, company, and match score (e.g. 85% Match) for each job. "
                "Do NOT show rationale, strong alignment, minor gaps, or any detailed matching info. "
                "Then add one short line: 'See the **Jobs** section in JobsifyAI for more matches and detailed match rationale.'"
            ),
        }
        return json.dumps(teaser, default=str)
    if func_name not in PREMIUM_INSIGHTS_TOOLS:
        return json.dumps(func_result, default=str)
    labels = {
        "get_salary_insights": "Salary Insights",
        "get_market_insights_only": "Market Insights",
        "get_career_advice": "Career Path Analysis",
        "get_enhanced_role_fit": "Detailed Role Fit",
    }
    label = labels.get(func_name, "Premium Insights")
    teaser = {
        "premium_teaser": True,
        "report_ready": True,
        "report_type": label,
        "destination": "Premium Insights section",
        "success": func_result.get("success", True),
    }
    if func_name == "get_market_insights_only":
        # Extract first market insight for a one-line teaser (rest in Premium Insights)
        market_insights = func_result.get("market_insights") or []
        one_insight = None
        if isinstance(market_insights, list) and len(market_insights) > 0:
            first = market_insights[0]
            if isinstance(first, dict):
                one_insight = (first.get("insight") or first.get("text") or "").strip()
            elif isinstance(first, str):
                one_insight = first.strip()
        if one_insight:
            teaser["one_insight"] = one_insight
            teaser["message"] = (
                "Show the user this ONE market insight only (as one bullet or sentence): use the text in 'one_insight'. "
                "Then add one short line: 'The rest can be seen in the Premium Insights section.' "
                "Do NOT show more than one insight. Do NOT mention salary or career paths."
            )
        else:
            teaser["message"] = (
                f"Reply with exactly ONE short line: 'Your {label} report is ready — unlock it in the Premium Insights section.' "
                f"Only mention this report ({label}). Do NOT mention other premium features. Keep it to this one line only."
            )
    elif func_name == "get_salary_insights":
        # Extract one salary amount (current level) for teaser; details in Premium Insights
        salary_insights = func_result.get("salary_insights") or {}
        current = salary_insights.get("current_level") or {}
        one_salary = (current.get("salary_range") or "").strip()
        if one_salary and one_salary.lower() not in ("not available", "n/a", ""):
            teaser["one_salary"] = one_salary
            teaser["message"] = (
                "Show the user this ONE salary amount only (the value in 'one_salary' — e.g. current-level range). "
                "Then add one short line: 'Details can be viewed in the Premium Insights section.' "
                "Do NOT show rationale, next level, or other figures."
            )
        else:
            # No specific salary: use guidance message if present (e.g. "general compensation in ₹ and Indian market")
            output = func_result.get("output") or {}
            msg = (func_result.get("message") or output.get("message") or "").strip()
            if msg and ("general compensation" in msg.lower() or "salary data not available" in msg.lower()):
                teaser["general_guidance"] = True
                teaser["message"] = (
                    "Reply with ONE short line: we can give the user general compensation guidance in their currency and region based on their profile — "
                    "then add: 'See the Premium Insights section for the full report.' Do NOT mention other premium features. Keep it to one line."
                )
            else:
                teaser["message"] = (
                    f"Reply with exactly ONE short line: 'Your {label} report is ready — unlock it in the Premium Insights section.' "
                    f"Only mention this report ({label}). Do NOT mention other premium features. Keep it to this one line only."
                )
    elif func_name == "get_career_advice":
        # Extract 1 or 2 career paths with minimal info (title + optional brief); rest in Premium Insights
        career_paths = func_result.get("career_paths") or []
        if isinstance(career_paths, list) and len(career_paths) > 0:
            minimal_paths = []
            for p in career_paths[:2]:
                if isinstance(p, dict):
                    title = (p.get("title") or p.get("name") or "").strip()
                    if title:
                        desc = (p.get("description") or "").strip()
                        minimal_paths.append({
                            "title": title,
                            "brief": (desc[:60] + "..." if len(desc) > 60 else desc) if desc else ""
                        })
                elif isinstance(p, str) and p.strip():
                    minimal_paths.append({"title": p.strip(), "brief": ""})
            if minimal_paths:
                teaser["one_or_two_career_paths"] = minimal_paths
                teaser["message"] = (
                    "Show the user only 1 or 2 career path titles from 'one_or_two_career_paths' (you may add one very short line per path from 'brief' if present). "
                    "Do NOT list required_skills or full descriptions. Then add one short line: 'Details can be viewed in the Premium Insights section.'"
                )
            else:
                teaser["message"] = (
                    f"Reply with exactly ONE short line: 'Your {label} report is ready — unlock it in the Premium Insights section.' "
                    f"Only mention this report ({label}). Do NOT mention other premium features. Keep it to this one line only."
                )
        else:
            teaser["message"] = (
                f"Reply with exactly ONE short line: 'Your {label} report is ready — unlock it in the Premium Insights section.' "
                f"Only mention this report ({label}). Do NOT mention other premium features. Keep it to this one line only."
            )
    elif func_name == "get_enhanced_role_fit":
        # Enhanced role fit: show 1–2 role names + fit % only; rationale/details on Resume Analysis page
        enhanced = func_result.get("enhanced_role_fit") or []
        minimal_roles = []
        for r in (enhanced if isinstance(enhanced, list) else [])[:2]:
            if isinstance(r, dict):
                role_name = (r.get("role") or r.get("role_name") or r.get("title") or "").strip()
                if role_name:
                    fit_pct = r.get("fit_percentage")
                    if fit_pct is None:
                        fit_pct = 0
                    try:
                        fit_pct = max(0, min(100, int(fit_pct)))
                    except (TypeError, ValueError):
                        fit_pct = 0
                    minimal_roles.append({"role": role_name, "fit_percentage": fit_pct})
        if minimal_roles:
            teaser["one_or_two_roles"] = minimal_roles
            teaser["destination"] = "Resume Analysis page"
            teaser["message"] = (
                "Show the user only 1–2 role names and fit percentage from 'one_or_two_roles' (e.g. 'Software Engineer – 85% fit'). "
                "Do NOT show why_suggested, skill_gaps_to_reach_role, how_to_overcome_gaps, or any rationale. "
                "Then add one short line: 'See the Resume Analysis page for more details and match rationale.'"
            )
        else:
            teaser["destination"] = "Resume Analysis page"
            teaser["message"] = (
                "Reply with exactly ONE short line: 'Your Detailed Role Fit report is ready — see the Resume Analysis page for more details and match rationale.' "
                "Do NOT mention other premium features. Keep it to this one line only."
            )
    else:
        # Fallback for any other premium tool
        teaser["message"] = (
            f"Reply with exactly ONE short line: 'Your {label} report is ready — unlock it in the Premium Insights section.' "
            f"Only mention this report ({label}). Do NOT mention other premium features (salary, career paths, market insights). "
            f"Keep it to this one line only; no extra sentences."
        )
    return json.dumps(teaser, default=str)


# Configuration
config = get_agent_config("career_mentor")
TIMEOUT_SECONDS = getattr(config, 'timeout_seconds', 60)
MAX_RESPONSE_LENGTH = getattr(config, 'max_response_length', 3000)
LLM_MODEL = getattr(config, 'llm_model', "gemini-2.0-flash-exp")


# ============================================================================
# DATA MODELS
# ============================================================================

class CareerChatRequest(BaseModel):
    """Request model for career chatbot"""
    uid: str
    message: str
    session_id: Optional[str] = None
    callback_url: Optional[str] = None
    stream: bool = Field(default=True, description="Enable streaming response")


class CareerChatResponse(BaseModel):
    """Response model for career chatbot (non-streaming)"""
    response: str
    session_id: str
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    suggested_topics: List[str] = Field(default_factory=list)  # Deprecated - use suggested_questions
    suggested_questions: List[str] = Field(default_factory=list)  # NEW: Contextually relevant questions
    action_items: List[Dict[str, str]] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.8)


# ============================================================================
# CALLBACK SUPPORT
# ============================================================================

async def _send_streaming_callback(
    callback_url: str,
    uid: str,
    session_id: str,
    full_response: str,
    suggested_topics: List[str],  # Deprecated
    suggested_questions: List[str],  # NEW
    action_items: List[Dict[str, str]],
):
    """
    Send first callback (response only) after streaming completes.
    Tool outputs are sent separately via _send_tool_outputs_callback.
    """
    try:
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from app import send_to_callback

        output_payload = {
            "response": full_response,
            "content_type": "markdown",
            "session_id": session_id,
            "suggested_questions": suggested_questions,
            "suggested_topics": suggested_topics,
            "action_items": action_items,
            "confidence_score": 0.8
        }

        callback_payload = {
            "type": "response_complete",
            "node": "career_chatbot",
            "status": "completed",
            "output": output_payload
        }

        log.info(f"📤 Sending callback (response_complete) ({len(full_response)} chars) | uid={uid} | session_id={session_id}")
        try:
            payload_for_log = json.loads(json.dumps(callback_payload, default=str))
            if isinstance(payload_for_log.get("output"), dict) and isinstance(payload_for_log["output"].get("response"), str):
                r = payload_for_log["output"]["response"]
                payload_for_log["output"]["response"] = r[:500] + ("..." if len(r) > 500 else "")
            log.info(f"📦 Callback payload (response truncated): %s", json.dumps(payload_for_log, default=str, ensure_ascii=False)[:4000])
        except Exception as _e:
            log.debug(f"Could not log callback payload: {_e}")
        await send_to_callback(callback_url, uid, callback_payload)
        log.info(f"✅ Callback (response_complete) sent | uid={uid}")
    except Exception as e:
        log.error(f"❌ Failed to send streaming callback: {e}", exc_info=True)


async def _send_tool_outputs_callback(
    callback_url: str,
    uid: str,
    session_id: str,
    tool_results: Dict[str, Any],
    assessment_plan: Optional[List[Dict[str, Any]]] = None,
    assessment_needs: Optional[Dict[str, Any]] = None,
):
    """
    Send second callback (tool outputs only) to the same callback_url.
    Client can distinguish by payload.type == "tool_outputs".
    """
    if not tool_results:
        return
    try:
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        from app import send_to_callback

        output_payload: Dict[str, Any] = {
            "session_id": session_id,
            "uid": uid,
            "tool_results": tool_results,
        }
        if assessment_plan is not None:
            output_payload["assessment_plan"] = assessment_plan
        if assessment_needs is not None:
            output_payload["assessment_needs"] = assessment_needs

        # When mentor returns assessment recommendations, expose them in assessment_recommender output structure
        # so the client can handle mentor assessment responses identically to assessment_recommender agent output.
        if "get_assessment_recommendations" in tool_results:
            ar_result = tool_results["get_assessment_recommendations"][0].get("result", {})
            if isinstance(ar_result, dict) and ar_result.get("success"):
                output_payload["assessment_recommender_output"] = {
                    "status": ar_result.get("status", "completed"),
                    "node": "assessment_recommender",
                    "output": {
                        "assessment_needs": ar_result.get("assessment_needs") or ar_result.get("output", {}).get("assessment_needs") or {},
                        "assessment_plan": ar_result.get("assessment_plan") or ar_result.get("output", {}).get("assessment_plan") or [],
                    },
                    "assessment_needs": ar_result.get("assessment_needs") or ar_result.get("output", {}).get("assessment_needs") or {},
                    "assessment_plan": ar_result.get("assessment_plan") or ar_result.get("output", {}).get("assessment_plan") or [],
                    "personalization_score": ar_result.get("personalization_score", 0.8),
                    "analysis_method": ar_result.get("analysis_method", "custom_from_chatbot"),
                    "confidence_score": ar_result.get("confidence_score", 0.8),
                    "processing_time": ar_result.get("processing_time", 0.0),
                }

        if "identify_relevant_jobs" in tool_results:
            job_result = tool_results["identify_relevant_jobs"][0]["result"]
            if isinstance(job_result, dict) and job_result.get("success"):
                output_payload["top_matches"] = job_result.get("relevant_jobs", [])
                output_payload["total_matches_found"] = job_result.get("count", 0)
                output_payload["matching_method"] = "chatbot_triggered_job_matching"
                output_payload["job_matcher_status"] = "success"

        if "get_course_recommendations" in tool_results:
            course_result = tool_results["get_course_recommendations"][0]["result"]
            if isinstance(course_result, dict) and course_result.get("success"):
                output_payload["course_recommendations"] = course_result.get("course_recommendations", [])
                output_payload["market_insights"] = course_result.get("market_insights", {})

        if "get_market_insights_only" in tool_results:
            market_result = tool_results["get_market_insights_only"][0]["result"]
            if isinstance(market_result, dict) and market_result.get("success"):
                output_payload["market_insights"] = market_result.get("market_insights", [])
                output_payload["salary_trends"] = market_result.get("salary_trends", {})
                output_payload["career_paths"] = market_result.get("career_paths", [])

        callback_payload = {
            "type": "tool_outputs",
            "node": "career_chatbot",
            "status": "completed",
            "output": output_payload
        }

        log.info(f"📤 Sending callback (tool_outputs) | uid={uid} | tools={list(tool_results.keys())}")
        await send_to_callback(callback_url, uid, callback_payload)
        log.info(f"✅ Callback (tool_outputs) sent | uid={uid}")
    except Exception as e:
        log.error(f"❌ Failed to send tool_outputs callback: {e}", exc_info=True)


# ============================================================================
# SESSION MANAGEMENT (Reusing interview agent pattern)
# ============================================================================

async def _load_conversation_history(session_id: str, uid: str) -> List[Dict[str, str]]:
    """
    Load conversation history from session.
    Thread-safe: Uses uid for proper session isolation.
    
    Args:
        session_id: Session identifier
        uid: User ID for proper session isolation
        
    Returns:
        List of conversation messages
    """
    if not session_id:
        return []
    
    try:
        # CRITICAL: Always include uid for proper session isolation
        # This ensures each user's sessions are properly isolated
        session_data = await run_blocking_io(get_chat_session, session_id, uid)
        if session_data:
            # Verify uid matches (security check)
            if session_data.get("uid") != uid:
                log.warning(f"Session uid mismatch: expected {uid}, got {session_data.get('uid')}")
                return []
            
            history = session_data.get("career_chat_history", [])
            if isinstance(history, list):
                return history
    except Exception as e:
        log.debug(f"Could not load conversation history: {e}")
    
    return []


async def _load_prior_conversation_context(
    uid: str,
    current_session_id: str,
    max_messages: int = 20,
) -> List[Dict[str, str]]:
    """
    Load last N messages from the most recent *other* session (Option B: always add
    recent prior). Excludes current session and goals pseudo-session.
    """
    if not uid or not current_session_id:
        return []
    try:
        other_sids = await run_blocking_io(
            list_sessions_by_uid,
            uid,
            5,
            current_session_id,
        )
        if not other_sids:
            return []
        sid = other_sids[0]
        session_data = await run_blocking_io(get_chat_session, sid, uid)
        if not session_data or session_data.get("uid") != uid:
            return []
        history = session_data.get("career_chat_history", [])
        if not isinstance(history, list):
            return []
        return history[-max_messages:]
    except Exception as e:
        log.debug(f"Could not load prior conversation context: {e}")
        return []


async def _save_conversation_history(session_id: str, history: List[Dict[str, str]], uid: str):
    """
    Save conversation history to session.
    Thread-safe: Uses uid for proper session isolation and handles concurrent writes.
    
    Args:
        session_id: Session identifier
        history: Conversation history to save
        uid: User ID for proper session isolation
    """
    if not session_id or not uid:
        return
    
    try:
        # CRITICAL: Always include uid for proper session isolation
        # Load existing session with uid to ensure we get the right one
        session_data = await run_blocking_io(get_chat_session, session_id, uid) or {}
        
        # Verify uid matches (security check)
        if session_data.get("uid") and session_data.get("uid") != uid:
            log.warning(f"Session uid mismatch on save: expected {uid}, got {session_data.get('uid')}")
            return
        
        # Update conversation history atomically
        # Use timestamp for optimistic locking (prevent race conditions)
        session_data["career_chat_history"] = history
        session_data["uid"] = uid  # Ensure uid is set
        session_data["timestamp"] = time.time()
        session_data["status"] = "active"
        session_data["last_updated"] = time.time()  # For conflict detection
        
        # Save to ChromaDB (reusing existing function)
        # update_chat_session handles uid-based isolation internally
        await run_blocking_io(update_chat_session, session_id, session_data)
        
        log.debug(f"✅ Saved conversation history ({len(history)} messages) for session={session_id}, uid={uid}")
    except Exception as e:
        log.warning(f"Failed to save conversation history: {e}", exc_info=True)


# ============================================================================
# CONTEXTUAL QUESTION GENERATION
# ============================================================================

async def _generate_contextual_questions(
    user_message: str,
    conversation_history: List[Dict[str, str]],
    context: Dict[str, Any],
    assistant_response: str
) -> List[str]:
    """
    Generate contextually relevant follow-up questions based on conversation context.
    
    Args:
        user_message: The user's current message
        conversation_history: Full conversation history
        context: User's profile context (resume, goals, etc.)
        assistant_response: The assistant's response to the user
        
    Returns:
        List of 2-4 contextually relevant questions
    """
    try:
        # Build context summary for question generation
        context_summary = []
        
        # Add user's goals/aspirations
        if context.get("career_goals"):
            goals = context["career_goals"][:3]
            goal_titles = [g.get("title", "") if isinstance(g, dict) else str(g) for g in goals]
            if goal_titles:
                context_summary.append(f"User's goals: {', '.join(goal_titles)}")
        
        # Add recent conversation topics
        recent_topics = []
        for msg in conversation_history[-5:]:
            content = msg.get("content", "").lower()
            if "course" in content or "learn" in content:
                recent_topics.append("learning/courses")
            if "assessment" in content or "test" in content:
                recent_topics.append("assessments")
            if "job" in content or "career" in content or "role" in content:
                recent_topics.append("jobs/career")
            if "skill" in content:
                recent_topics.append("skills")
        
        if recent_topics:
            context_summary.append(f"Recent topics: {', '.join(set(recent_topics))}")
        
        # Add skill gaps teaser if available (limit to 2 for premium protection)
        if context.get("skill_gaps"):
            gaps = context["skill_gaps"][:2]
            if gaps:
                context_summary.append(f"Skill gaps (teaser): {', '.join(gaps)}")
        
        context_text = "\n".join(context_summary) if context_summary else "No specific context available"
        
        # Build question generation prompt
        question_prompt = f"""Based on this conversation, generate 2-4 natural, contextually relevant follow-up questions that the USER can click to ask.

User's message: "{user_message}"

Your response: "{assistant_response[:500]}"

User context:
{context_text}

**CRITICAL: These questions will be CLICKABLE - when clicked, they become the USER'S next message to the chatbot.**

Generate questions that:
1. Are phrased as if the USER is asking them (not the chatbot asking)
2. Are natural and conversational (not formal)
3. Relate to what was just discussed
4. Are specific and actionable - they should trigger specific actions or requests
5. Consider the user's goals and context
6. When clicked, should make sense as a user's message to continue the conversation

**GOOD Examples (clickable - user can send these as their message):**
- "Can you show me courses to help me bridge my AWS skill gap?"
- "I'd like to take an assessment to see my current proficiency in Python or other AI/ML concepts"
- "Which of these areas (Agentic AI, Gen AI, e-commerce recommendations, Oracle HCM, RPA) should I focus on first?"
- "Can you find job opportunities that align with these career paths?"
- "What courses would help me improve in Python?"
- "Show me assessments I can take for data analysis"
- "Find me jobs that match my profile"

**BAD Examples (chatbot asking, not user asking):**
- "Would you like to explore courses to bridge your AWS skill gap?" (chatbot asking)
- "Are you interested in taking an assessment..." (chatbot asking)
- "Do you want to learn more?" (chatbot asking)
- "What else can I help with?" (chatbot asking)
- "Python" (not a question, just a topic)

**Format:**
Questions should be phrased as:
- Direct requests: "Can you show me...", "Find me...", "I'd like to...", "Show me..."
- Natural queries: "What courses...", "Which career paths...", "How can I..."
- Action-oriented: Questions that will trigger specific chatbot actions (show courses, find jobs, create assessments)

Return ONLY a JSON array of questions (no markdown, no code blocks):
["question 1", "question 2", "question 3"]
"""
        
        # Call LLM to generate questions
        from models.llm_invoker import invoke_llm
        response = await invoke_llm(
            prompt=question_prompt,
            task_type=TaskType.TEXT_GENERATION,
            agent_name="career_chatbot",
            max_output_tokens=200
        )
        
        # Parse response
        response_text = _to_text(response)
        questions = _extract_json_from_response(response_text)
        
        if isinstance(questions, list) and len(questions) > 0:
            # Filter out invalid questions
            valid_questions = [
                q for q in questions 
                if isinstance(q, str) and len(q.strip()) > 10 and "?" in q
            ]
            if valid_questions:
                log.info(f"✅ Generated {len(valid_questions)} contextual questions")
                return valid_questions[:4]  # Max 4 questions
        
        # Fallback: Generate simple questions based on context
        return _generate_fallback_questions(user_message, context)
        
    except Exception as e:
        log.warning(f"Error generating contextual questions: {e}")
        return _generate_fallback_questions(user_message, context)


def _generate_fallback_questions(
    user_message: str,
    context: Dict[str, Any]
) -> List[str]:
    """
    Fallback: Generate simple questions based on keywords and context.
    Questions are phrased as user queries (clickable).
    """
    questions = []
    message_lower = user_message.lower()
    
    # Check what user is asking about
    if any(kw in message_lower for kw in ["course", "learn", "training", "education"]):
        questions.append("Can you show me specific courses that match my learning goals?")
        if context.get("skill_gaps"):
            questions.append(f"Can you find courses to help me improve in {context['skill_gaps'][0]}?")
    
    if any(kw in message_lower for kw in ["assessment", "test", "evaluate", "quiz"]):
        questions.append("Can you show me assessments I can take to test my skills?")
    
    if any(kw in message_lower for kw in ["job", "career", "role", "position", "opportunity"]):
        questions.append("Can you find job opportunities that match my profile?")
        questions.append("What specific role or industry should I focus on?")
    
    if any(kw in message_lower for kw in ["skill", "ability", "competency"]):
        questions.append("Can you help me identify which skills to focus on developing?")
    
    if context.get("career_goals"):
        questions.append("Can you help me break down my career goals into actionable steps?")
    
    # Default questions if nothing specific
    if not questions:
        questions.append("Can you show me courses, assessments, or job opportunities?")
        questions.append("What should I focus on next in my career development?")
    
    return questions[:4]  # Max 4 questions


# ============================================================================
# PROMPT BUILDING
# ============================================================================

def _skill_to_str(s) -> str:
    """Extract skill name from dict (SkillName, Name, name) or return str(s)."""
    if isinstance(s, dict):
        return (
            s.get("SkillName") or s.get("Name") or s.get("name") or s.get("skill", "")
        ).strip() or str(s)
    return str(s).strip()


def _build_resume_detail_for_prompt(resume: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Build a rich resume-detail block (skills, experience, education, projects, certs)
    so the LLM can answer resume-related questions accurately.
    Truncates sensibly to fit token budget.
    """
    out = []
    skills_raw = resume.get("skills") or resume.get("Skills") or context.get("skills", [])
    skills_strs = [_skill_to_str(s) for s in skills_raw if _skill_to_str(s)][:25]
    if skills_strs:
        out.append("**Skills:** " + ", ".join(skills_strs))

    exp = resume.get("work_experience") or resume.get("experience") or resume.get("Experience") or resume.get("WorkExperience") or []
    if exp and isinstance(exp, list):
        out.append("**Work Experience:**")
        for e in exp[:8]:
            if not isinstance(e, dict):
                continue
            title = e.get("job_title") or e.get("title") or e.get("role", "")
            company = e.get("company") or e.get("organization", "")
            dates = e.get("dates") or e.get("duration", "")
            loc = e.get("location", "")
            parts = [p for p in [title, company, dates, loc] if p]
            line = " • ".join(str(p) for p in parts)
            if line:
                out.append(f"  - {line}")
            resp = (e.get("responsibilities") or e.get("description") or [])
            if isinstance(resp, list) and resp:
                brief = resp[0] if isinstance(resp[0], str) else str(resp[0])
                out.append(f"    {brief[:120]}{'...' if len(brief) > 120 else ''}")

    edu = resume.get("education") or resume.get("Education") or []
    if edu and isinstance(edu, list):
        out.append("**Education:**")
        for e in edu[:5]:
            if not isinstance(e, dict):
                out.append(f"  - {e}")
                continue
            degree = e.get("degree") or e.get("Degree") or ""
            major = e.get("major") or e.get("Major") or e.get("field", "")
            uni = e.get("university") or e.get("institution") or e.get("school") or e.get("University") or ""
            years = e.get("years") or e.get("duration", "")
            parts = [p for p in [f"{degree} {major}".strip(), uni, years] if p]
            edu_line = ", ".join(str(p) for p in parts) if parts else "(see resume)"
            out.append("  - " + edu_line)

    proj = resume.get("projects") or resume.get("Projects") or []
    if proj and isinstance(proj, list):
        out.append("**Projects:**")
        for p in proj[:5]:
            if not isinstance(p, dict):
                out.append(f"  - {p}")
                continue
            name = p.get("project_name") or p.get("name") or p.get("title", "")
            desc = p.get("description") or ""
            if name:
                out.append(f"  - {name}")
                if desc:
                    out.append(f"    {desc[:100]}{'...' if len(desc) > 100 else ''}")

    certs = resume.get("certifications") or resume.get("Certifications") or []
    if certs and isinstance(certs, list):
        cert_names = []
        for c in certs[:5]:
            if isinstance(c, dict):
                cert_names.append(c.get("name") or c.get("Name") or str(c))
            else:
                cert_names.append(str(c))
        if cert_names:
            out.append("**Certifications:** " + ", ".join(cert_names))

    if not out:
        return ""
    return "\n".join(out)


def _build_career_chat_prompt(
    user_message: str,
    context: Dict[str, Any],
    conversation_history: List[Dict[str, str]]
) -> str:
    """
    Build comprehensive prompt using all candidate context.
    """
    
    # Build resume summary + rich detail (for resume-related questions)
    resume_section = ""
    if context.get("resume_data"):
        resume = context["resume_data"]
        name = resume.get("Name") or resume.get("name", "Candidate")
        skills = context.get("skills", [])
        experience = resume.get("work_experience") or resume.get("experience", [])
        
        resume_section = f"""
**PROFESSIONAL PROFILE:**
- Name: {name}
- Total Experience: {len(experience)} position(s)
- Skills: {len(skills)} skill(s) identified
- Resume Summary: {context.get('resume_summary', 'Not available')[:500]}
"""
        detail = _build_resume_detail_for_prompt(resume, context)
        if detail:
            resume_section += f"""
**RESUME DETAIL (use this to answer resume-related questions accurately—e.g. "What's on my resume?", "What skills do I have?", "Summarize my experience"):**
{detail}
"""
    
    # Build candidate location & currency (for salary/market - derived from resume)
    location = context.get("candidate_location", "") or ""
    currency = context.get("candidate_currency", "₹") or "₹"
    region = context.get("salary_region", "Indian") or "Indian"
    location_currency_section = "\n**CANDIDATE LOCATION & CURRENCY (use for salary/compensation):**\n"
    location_currency_section += f"- Location (from resume): {location or 'Not specified'}\n"
    location_currency_section += f"- Currency: {currency} | Region: {region}\n"
    location_currency_section += "- For salary/compensation figures, use the above currency and region. If location is Not specified, default to ₹ and Indian market.\n"
    
    # Build skills & gaps (PREMIUM - only include for internal reference, NOT for display)
    skills_section = ""
    if context.get("skill_gaps"):
        all_gaps = context['skill_gaps']
        # Show only first 2 for teaser purposes in chat
        teaser_gaps = all_gaps[:2]
        skills_section = f"""
**SKILL GAPS (PREMIUM CONTENT - DO NOT DISPLAY FULL LIST):**
- Teaser (show only these 1-2 in chat): {', '.join(teaser_gaps)}
- Total skill gaps identified: {len(all_gaps)}
- Full analysis available in: Premium Insights section
- IMPORTANT: When user asks about skill gaps, show ONLY the 1-2 teaser gaps above, then nudge to Premium Insights for the complete analysis.
"""
    
    # Build resume score section (skill score + presentation score from resume scorer)
    resume_score_section = ""
    if context.get("resume_score"):
        rs = context["resume_score"]
        if isinstance(rs, dict):
            breakdown = rs.get("breakdown", {})
            skills_val = breakdown.get("skills") or rs.get("SkillsScore") or rs.get("skills")
            presentation_val = breakdown.get("presentation") or rs.get("FormatScore") or rs.get("presentation")
            total_val = rs.get("total") or rs.get("OverallScore") or rs.get("ats_score")
            if skills_val is not None or presentation_val is not None or total_val is not None:
                resume_score_section = "\n**RESUME SCORE (from resume analysis—use this when user asks for skill score, presentation score, or resume quality):**\n"
                if total_val is not None:
                    resume_score_section += f"- Overall/ATS Score: {int(round(float(total_val)))}/100\n"
                if skills_val is not None:
                    resume_score_section += f"- Skill Score: {int(round(float(skills_val)))}/100\n"
                if presentation_val is not None:
                    resume_score_section += f"- Presentation Score (format/ATS compatibility): {int(round(float(presentation_val)))}/100\n"
                rationale = rs.get("rationale") or rs.get("Explanation")
                if rationale:
                    resume_score_section += f"- Rationale: {str(rationale)[:300]}...\n" if len(str(rationale)) > 300 else f"- Rationale: {rationale}\n"
                resume_score_section += "If user asks for skill score or presentation score, use the values above. When stating these scores to the user, use whole numbers only (e.g. 85/100), never decimals (e.g. 85.0/100). Do NOT confuse these with assessment average (which is from completed assessments).\n"
    
    # Build career paths
    career_paths_section = ""
    if context.get("career_paths"):
        career_paths_section = "\n**RECOMMENDED CAREER PATHS:**\n"
        for path in context["career_paths"][:5]:
            title = path.get("title", "N/A") if isinstance(path, dict) else str(path)
            career_paths_section += f"- {title}\n"
    
    # Build assessment performance
    assessment_section = ""
    if context.get("assessment_results"):
        assessment_section = f"""
**ASSESSMENT PERFORMANCE:**
- Total Assessments Completed: {len(context['assessment_results'])}
"""
        if context.get("overall_performance"):
            perf = context["overall_performance"]
            assessment_section += f"- Average Score: {perf.get('average_score', 'N/A')}/100\n"
            assessment_section += f"- Performance Level: {perf.get('performance_level', 'N/A')}\n"
        # Per-topic results so the model can answer "how is my X performance"
        _lines = []
        for r in context["assessment_results"][:10]:
            topic = r.get("assessment_topic") or r.get("topic") or "Unknown"
            score = r.get("total_score") or r.get("score")
            perf_level = (r.get("result") or {}).get("performance_level") if isinstance(r.get("result"), dict) else None
            if score is not None:
                perf_str = f" ({perf_level})" if perf_level else ""
                _lines.append(f"- {topic}: {score}/100{perf_str}\n")
        if _lines:
            assessment_section += "\n**Completed assessments (per-topic):**\n" + "".join(_lines)
    
    # Detect weak areas from assessment results (for proactive suggestions)
    weak_areas = []
    if context.get("assessment_results"):
        for r in context["assessment_results"]:
            topic = r.get("assessment_topic") or r.get("topic") or "Unknown"
            score = r.get("total_score") or r.get("score")
            result = r.get("result", {})
            
            # Flag overall low scores
            if score is not None and score < 50:
                weak_areas.append(f"- {topic}: {score}/100 (needs improvement)")
            
            # Check question-type breakdown if available
            if isinstance(result, dict):
                coding_score = result.get("coding_score") or result.get("coding_questions_score")
                if coding_score is not None and coding_score < 30:
                    weak_areas.append(f"- Coding questions in {topic}: {coding_score}% (critical gap)")
                mcq_score = result.get("mcq_score") or result.get("mcq_questions_score")
                if mcq_score is not None and mcq_score < 40:
                    weak_areas.append(f"- MCQ questions in {topic}: {mcq_score}% (needs practice)")
    
    if weak_areas:
        assessment_section += "\n**AREAS NEEDING ATTENTION (for proactive suggestions):**\n"
        assessment_section += "\n".join(weak_areas)
        assessment_section += "\n"
    
    # Add pending assessments section for nudging
    pending_assessments = context.get("pending_assessments", [])
    if pending_assessments:
        assessment_section += f"""
**PENDING ASSESSMENTS (Not Yet Completed):**
You have {len(pending_assessments)} recommended assessment(s) that haven't been completed yet:
"""
        for item in pending_assessments[:5]:  # Show top 5
            topic = item.get("topic", "Unknown")
            difficulty = item.get("difficulty", "medium")
            status = normalize_assessment_status(item.get("status"))  # consistent: completed|pending|in_progress
            assessment_section += f"- {topic} ({difficulty} difficulty) - Status: {status}\n"
    
    # Build course recommendations (include link when url is available)
    courses_section = ""
    if context.get("course_recommendations"):
        lines = []
        for c in context["course_recommendations"][:5]:
            title = c.get("title") or c.get("name") or c.get("course", "Course")
            url = c.get("url") or ""
            if isinstance(c.get("url_validation"), dict):
                url = url or (c.get("url_validation", {}).get("url") or "")
            url = str(url or "").strip()
            provider = c.get("provider") or c.get("platform", "")
            if url and url.startswith(("http://", "https://")):
                line = f"- [{title}]({url})" + (f" ({provider})" if provider else "")
            else:
                line = f"- {title}" + (f" ({provider})" if provider else "")
            lines.append(line)
        courses_section = f"""
**RECOMMENDED COURSES:**
{chr(10).join(lines)}
"""
    
    # Build conversation history
    history_section = ""
    if conversation_history:
        recent = conversation_history[-5:]  # Last 5 exchanges
        history_section = "\n**RECENT CONVERSATION:**\n"
        for exchange in recent:
            role = exchange.get("role", "user")
            content = exchange.get("content", "")
            history_section += f"{role.capitalize()}: {content}\n"
    
    # Build aspirations section (from conversation or context)
    aspirations_section = ""
    if context.get("aspirations"):
        aspirations = context["aspirations"]
        if aspirations.get("career_goals") or aspirations.get("aspirations"):
            aspirations_section = "\n**USER'S STATED ASPIRATIONS (from conversation):**\n"
            if aspirations.get("career_goals"):
                aspirations_section += f"Career Goals: {', '.join(aspirations['career_goals'][:3])}\n"
            if aspirations.get("aspirations"):
                aspirations_section += f"Aspirations: {', '.join(aspirations['aspirations'][:3])}\n"
            if aspirations.get("skills_of_interest"):
                aspirations_section += f"Skills of Interest: {', '.join(aspirations['skills_of_interest'][:3])}\n"
    
    # Build goals section
    goals_section = ""
    if context.get("career_goals"):
        goals = context["career_goals"]
        if goals:
            goals_section = "\n**ACTIVE CAREER GOALS:**\n"
            for goal in goals[:3]:
                if isinstance(goal, dict):
                    title = goal.get("title", "N/A")
                    status = goal.get("status", "active")
                    goals_section += f"- {title} (Status: {status})\n"
    
    # Build job/role identification section
    jobs_section = ""
    if context.get("relevant_jobs") or context.get("role_fit_suggestions"):
        jobs_section = "\n**RELEVANT JOBS/ROLES FOR YOU:**\n"
        relevant_jobs = context.get("relevant_jobs", [])
        role_fits = context.get("role_fit_suggestions", [])
        
        # Combine both sources
        all_jobs = relevant_jobs[:3] + role_fits[:3]
        for job in all_jobs[:5]:
            if isinstance(job, dict):
                title = job.get("role") or job.get("role_name") or job.get("title") or job.get("Role", "N/A")
                jobs_section += f"- {title}\n"
            elif isinstance(job, str):
                jobs_section += f"- {job}\n"
    
    # Build conversation insights section (from pattern analysis)
    insights_section = ""
    if context.get("conversation_insights"):
        insights = context["conversation_insights"]
        insights_section = "\n**CONVERSATION INSIGHTS (use to personalize your response):**\n"
        
        # Preferred topics (what user asks about most)
        preferred_topics = insights.get("preferred_topics", [])
        if preferred_topics:
            insights_section += f"- User frequently asks about: {', '.join(preferred_topics[:3])}\n"
        
        # Frequently discussed specific topics (skills, areas of interest)
        freq_topics = insights.get("frequently_discussed_topics", [])
        if freq_topics:
            insights_section += f"- Areas of interest: {', '.join(freq_topics[:4])}\n"
        
        # Location preference (from prior conversations)
        location_pref = insights.get("location_preference")
        if location_pref:
            location_display = {
                "india": "India",
                "usa": "USA/United States",
                "uk": "UK",
                "uae": "UAE/Dubai",
                "global": "Global/International",
                "remote": "Remote work"
            }.get(location_pref, location_pref.title())
            insights_section += f"- **Location preference:** User has shown interest in {location_display} opportunities\n"
        
        # Work type preference
        work_type_pref = insights.get("work_type_preference")
        if work_type_pref:
            work_display = {"remote": "remote", "hybrid": "hybrid", "onsite": "on-site/office"}.get(work_type_pref, work_type_pref)
            insights_section += f"- **Work style preference:** User prefers {work_display} work\n"
        
        # Conversation themes
        themes = insights.get("conversation_themes", [])
        if themes:
            insights_section += f"- Recurring themes: {', '.join(themes[:3])}\n"
        
        # Engagement level
        engagement = insights.get("engagement_level", 0.0)
        if engagement > 0.7:
            insights_section += "- User is highly engaged - provide detailed, comprehensive responses\n"
        elif engagement > 0.4:
            insights_section += "- User is moderately engaged - balanced responses work well\n"
        else:
            insights_section += "- User prefers concise responses - keep it brief and focused\n"
        
        # Learning preferences
        learning_prefs = insights.get("learning_preferences", {})
        if learning_prefs.get("preferred_learning_method") == "structured_learning":
            insights_section += "- User prefers structured learning (courses, certifications)\n"
        elif learning_prefs.get("preferred_learning_method") == "assessment_based":
            insights_section += "- User prefers assessment-based learning\n"
        
        if learning_prefs.get("response_preference") == "detailed":
            insights_section += "- User appreciates detailed explanations\n"
        elif learning_prefs.get("response_preference") == "concise":
            insights_section += "- User prefers concise, to-the-point responses\n"
        
        # Add continuity note
        if location_pref or freq_topics:
            insights_section += "\n**CONTINUITY:** When relevant, reference prior conversations (e.g., 'Since you've been exploring [topic]...' or 'Continuing with your interest in [location] roles...')\n"
    
    prompt = f"""You are an expert AI Career Mentor for **JobsifyAI** having a CONVERSATION with a user. You have COMPLETE visibility into their entire professional profile.

**PRODUCT CONTEXT:**
You are part of JobsifyAI - an AI-powered career guidance platform. When referring to sections/pages, always mention "JobsifyAI" so users know where to go:
- "Premium Insights section in JobsifyAI" (or just "Premium Insights" if context is clear)
- "Resume Analysis page in JobsifyAI"
- "Jobs section in JobsifyAI"
- "Assessments section in JobsifyAI"
Never say "the website" or "our platform" - always say "JobsifyAI" by name.

You have access to:
- Their full resume and professional background
- All assessment results and performance data
- Skill gaps and career recommendations
- Course recommendations and market insights
- Their strengths, weaknesses, and growth areas

**CONVERSATION STYLE:**
- This is a CONTINUOUS conversation (like ChatGPT) - not a one-time Q&A session
- Keep responses CONCISE and conversational - answer their question directly without overwhelming them
- Be friendly, helpful, and natural - like talking to a knowledgeable friend
- Don't write long formal advice paragraphs - keep it conversational and engaging
- Always expect follow-up questions - end responses in a way that invites further conversation
- Only end the conversation if the user explicitly says goodbye, thanks you and indicates they're done, or asks to end

**FORMATTING FOR READABILITY (CRITICAL - follow strictly):**
Your responses MUST be easy to scan. Users read on small chat windows - dense paragraphs are hard to read.

RULES:
1. **NEVER write paragraphs longer than 2 sentences.** Break up information with line breaks.
2. **Use bullet points for ANY list of 2+ items.** Start each bullet with "* " (asterisk space).
3. **One idea per line.** Don't cram multiple points into one sentence.
4. **Use bold for key terms** like role names, percentages, section names.

BAD (dense paragraph - DON'T DO THIS):
"Arvind, I can certainly provide you with insights into your market outlook, career paths, and salary. Your market outlook and career paths are ready. Unlock your full report for detailed insights. The rest can be seen in the Premium Insights section. Based on your extensive experience as a Technical Program Manager and Senior Project Manager, you're likely in a senior-level compensation band. For detailed salary ranges and growth outlook, check the Premium Insights section."

GOOD (scannable with bullets):
"Arvind, here's a quick overview:

* **Market outlook & career paths** are ready - unlock your full report in the **Premium Insights** section for detailed insights

* **Salary:** Based on your experience as a Technical Program Manager, you're in a **senior-level compensation band**

For detailed salary ranges and growth outlook, check the **Premium Insights** section."

ALWAYS format your responses like the GOOD example - short lines, bullets, bold key terms.

**MARKET & CURRENCY (salary/compensation):**
- Use the candidate's location from context to determine currency and market. The **CANDIDATE LOCATION & CURRENCY** section below shows: location (from resume), currency symbol, and region. Use that for any salary/compensation figures.
- If location is missing, default to Indian Rupees (₹) and Indian market.

**PREMIUM FEATURES – TEASER ONLY IN CHAT (salary, market, career paths, role fit, skill gaps):**
**Where to nudge:** Role fit details (why suggested, fit %, rationale) → **Resume Analysis page**. Career paths / alternate career paths / full career analysis / skill gaps → **Premium Insights section**. Salary and market → **Premium Insights section**.
When you receive a tool result with `premium_teaser: true` or `report_ready: true`:
1. **Market Insights:** If the payload includes `one_insight`, show that single insight (one bullet or sentence), then: "The rest can be seen in the Premium Insights section." Do NOT show more than one insight.
2. **Salary Insights:** If the payload includes `one_salary`, show that one salary amount (e.g. current-level range), then: "Details can be viewed in the Premium Insights section." Do NOT show rationale, next level, or other figures. If the payload has `general_guidance: true` (no specific salary), reply with one short line that we can give general compensation guidance in their currency and region based on their profile, then: "See the Premium Insights section for the full report."
3. **Career Path Analysis:** If the payload includes `one_or_two_career_paths`, show only 1–2 career path titles (and at most one short line per path from `brief` if present). Do NOT list skills or full descriptions. Then: "Details can be viewed in the Premium Insights section."
3a. **Career paths nudge (use only one):** When you **called get_career_advice**, the tool result already tells you to say "Details can be viewed in the Premium Insights section." Use that line once only — do NOT add a second line like "For the full career path analysis...". When you **did not** call get_career_advice (you answered from context only), then add once: "For the full career path analysis and skill gaps, check the Premium Insights section."
4. **Detailed Role Fit:** If the payload includes `one_or_two_roles`, show only 1–2 role names and fit percentage (e.g. "Software Engineer – 85% fit"). Do NOT show why_suggested, skill_gaps_to_reach_role, how_to_overcome_gaps, or any rationale. Then: "See the Resume Analysis page for more details and match rationale." If no one_or_two_roles, reply with one short line: "Your Detailed Role Fit report is ready — see the Resume Analysis page for more details and match rationale."
5. **Skill Gaps (PREMIUM):** When user asks "what are my skill gaps?", "show me my skill gaps", "can you show my skill gaps?", or similar:
   - Show ONLY 1-2 skill gap names from the teaser in context (e.g., "Based on your profile, I've identified gaps in **Cloud architecture patterns** and **Cloud security best practices**.")
   - Then ALWAYS add: "For the complete skill gap analysis and recommendations on how to close them, check the **Premium Insights** section."
   - Do NOT list more than 2 skill gaps. Do NOT show the full list. Do NOT explain how to close them in chat.
6. **Any other report without a specific teaser:** Reply with exactly one short line: "Your [report_type] report is ready — unlock it in the Premium Insights section." Do not show any specific content.
7. **Do NOT** mention other premium features (e.g. for market analysis, do not say "salary trends and career path breakdowns"). Only refer to the report that was just generated.
8. **Do NOT** use content from the tool except `one_insight`, `one_salary`, `one_or_two_career_paths`, or `one_or_two_roles` when provided. No extra sentences beyond the teaser (if any) and the nudge line.

**CLARITY & UNDERSTANDABILITY:**
- Make all advice and suggestions CLEAR and EASY TO UNDERSTAND
- Avoid jargon or technical terms without explanation - if you must use them, explain them simply
- Break down complex concepts into simple, digestible points
- Use concrete examples when explaining abstract ideas
- Make recommendations actionable - tell them what to do, not just what to know
- If suggesting courses, skills, or paths, explain WHY they're relevant and HOW they help
- When recommending courses, include the course link (url) as a clickable markdown link [Title](url) when available, so the user can open it directly
- Use plain language - write as if explaining to someone who might not be familiar with the field

**MULTI-TURN CLARIFICATION:**
When a user's request is ambiguous or could have multiple interpretations, ask a brief clarifying question BEFORE providing a full answer. This leads to more targeted, relevant responses.

Examples of when to clarify:
- **Skill level ambiguity:** "Are you looking for beginner-level courses or more advanced content?"
- **Assessment scope:** "Would you prefer a quick 10-minute test or a comprehensive evaluation?"
- **Career direction:** "Are you thinking about advancing in your current field or exploring a different domain?"
- **Skill focus:** "Should I focus on technical skills or soft skills like communication and leadership?"
- **Learning format:** "Do you prefer video courses, hands-on projects, or reading-based learning?"

When NOT to clarify (just answer directly):
- The user's intent is clear from their message or conversation history
- You already have enough context from their profile (resume, goals, assessments) to give a targeted response
- The question is simple with an obvious answer
- The user explicitly asks for a general/broad overview

Keep clarifying questions SHORT (one question at a time) and conversational. Don't over-clarify simple requests.

**PROACTIVE INSIGHTS & SUGGESTIONS:**
Based on the user's profile, OCCASIONALLY offer helpful suggestions when they're relevant to the conversation. Don't force them into every response—use judgment.

**When to offer proactive suggestions:**
1. **Assessment performance gaps:** If you see a low score (< 50%) on a completed assessment or question type in the AREAS NEEDING ATTENTION section:
   - "I noticed you scored X% on [topic]. Would you like some resources to strengthen that area?"
   - "Your coding questions could use some practice. Want me to find some coding exercises or courses?"

2. **Pending assessments:** When the conversation touches on a topic with a pending assessment:
   - "By the way, you have a pending [topic] assessment. Would you like some preparation tips before taking it?"
   - "Since we're talking about [topic], you might want to take the assessment I recommended earlier. Ready to start?"

3. **Skill gaps related to goals:** When discussing a user's career goal and there's a relevant skill gap:
   - "To reach your goal of [goal], you'll want to strengthen [skill]. Want me to suggest some courses?"

4. **Stale knowledge:** If an assessment topic is discussed and the user hasn't tested it recently:
   - "Want to take a quick assessment to see where you stand on [topic]?"

**How to offer (tone & style):**
- Keep it brief and conversational (one sentence offer)
- Don't interrupt the flow—add proactive suggestions at the END of your response after answering their question

**PERSONALIZATION FROM USER PATTERNS:**
Use the CONVERSATION INSIGHTS section above to personalize your responses. Adapt based on what you know about this user:

1. **Location preferences:** If user has shown interest in a specific region (India, USA, etc.), default to that region when discussing jobs, salaries, or opportunities. Example: "Since you've been exploring roles in India..."

2. **Work style:** If user prefers remote/hybrid/onsite, factor that into job and career discussions.

3. **Frequently discussed topics:** If user repeatedly asks about certain skills (e.g., Python, AWS), proactively connect those to your advice. Example: "Given your interest in AWS..."

4. **Response style:** Match the user's preferred response length (concise vs. detailed) based on their interaction patterns.

5. **Continuity:** Reference prior conversations when relevant:
   - "Last time we discussed [topic], and..."
   - "Building on your interest in [area]..."
   - "Since you've been focusing on [skill]..."

6. **Learning style:** If user prefers courses over assessments (or vice versa), lean toward their preferred method when making recommendations.
- Make it easy to decline: "Would you like...", "Want me to...", "Interested in..."
- Maximum ONE proactive suggestion per response
- Don't repeat the same suggestion if user doesn't engage with it

**YOUR TOOLS:**
You have access to tools that can fetch data or call agents when needed:
- get_course_recommendations: Get personalized course recommendations (checks existing data first, fast)
- get_assessment_recommendations: Get skill assessment recommendations (checks existing data first, fast)
- get_resume_score: Get resume score and role suggestions (uses resume scorer agent when context is empty or user requests refresh)
- get_career_advice: Get career paths, role fit suggestions, skill gaps (uses career advisor agent when context is empty or user requests refresh)
- get_market_insights_only: Get market insights, salary trends, career paths only (no courses)—lightweight; use when user asks for market analysis / career outlook but NOT courses. In chat: PREMIUM – teaser + nudge to unlock.
- **Salary/compensation:** You do NOT have a salary tool in chat. Use the candidate's profile (resume, experience, skills) in your context to infer their current level (e.g. role, seniority). Give one short line, then nudge to Premium Insights for full salary details.
- create_career_goal: Create career goals for the user
- identify_relevant_jobs: Find the best job matches using advanced job matching agents (returns top 3-4 matches aligned with user's skills and qualifications)
- get_resume_optimization_suggestions: Analyze resume and provide optimization suggestions (ATS compatibility, keywords, achievements, format)
- get_learning_path: Create a structured learning path from courses and assessments (beginner → intermediate → advanced)

**WHEN TO USE TOOLS:**
**IMPORTANT: Only call tools when the user EXPLICITLY requests data or actions. For general questions, use your existing knowledge from the context sections above.**

**PARAMETERIZATION RULE:** When you do call a tool, ALWAYS extract and pass relevant arguments from the user's request (e.g., topics, search_query, top_k, courses_only, force_refresh) so the tool can run the minimal required flow.

1. **get_course_recommendations**: 
   - **For general course requests (no specific topic):** Use `course_recommendations` from your context directly. Do NOT call the tool. Just show the courses from context.
   - **ONLY call the tool when:** (a) user mentions a **specific topic** (e.g., "courses for Python", "React courses", "find courses for machine learning"), OR (b) user explicitly asks to refresh/update courses
   - If the user mentions a specific topic (e.g., "Python", "React", "machine learning"), extract it and pass in tool args: topics=["Python"] and set courses_only=true to avoid market/salary paths (courses-only fast path)
   - If user asks to refresh, set force_refresh=true
   - DO NOT call for general questions about learning (e.g., "how can I learn Python?" - answer from your knowledge)
   - **INCLUDE LINKS:** When presenting course recommendations to the user, ALWAYS include the course URL as a clickable link when the course has a `url` field. Use markdown: [Course Title](url). If url is missing, show title and provider only. Format each course as a new line starting with "* " (asterisk space) then the link or title, so the UI can render bullets and clickable links consistently.

2. **get_career_advice** (career paths, alternate career paths, skill gaps) – PREMIUM / TEASER:
   - Use context first: career_paths, role_fit_suggestions from context (but NOT full skill_gaps list - that's premium)
   - CALL get_career_advice when: (a) context has no career_paths/role_fit_suggestions and user asks about careers/roles, OR (b) user explicitly asks to refresh career analysis or for alternate career paths
   - Clear indicators: "What career paths suit me?", "What roles am I qualified for?", "Refresh my career analysis", "alternate career paths"
   - If user asks for fresh/updated career advice, set force_refresh=true
   - **In chat:** Do NOT present full career_paths, skill_gaps, or improvement_recommendations. You will receive only a report-ready payload. Nudge to **Premium Insights section** for full career path analysis and skill gaps (NOT Resume Analysis page).
   - **Nudge once only:** If you used context only (no get_career_advice call), end with: "For the full career path analysis and skill gaps, check the Premium Insights section." If you called get_career_advice, use only the Premium line from the tool result ("Details can be viewed in the Premium Insights section") and do not add a second nudge.

2a. **Skill gaps (PREMIUM - use context teaser ONLY):** When user asks "what are my skill gaps?", "show me my skill gaps", "can you show my skill gaps?":
   - Check your context for the **SKILL GAPS (PREMIUM CONTENT)** section
   - Show ONLY the 1-2 teaser gaps listed there (e.g., "I've identified some skill gaps including **Cloud architecture patterns** and **Cloud security best practices**.")
   - ALWAYS end with: "For the complete skill gap analysis and actionable recommendations, check the **Premium Insights** section."
   - Do NOT list more than 2 gaps. Do NOT show the full list. Do NOT explain remediation steps in chat.

2b. **Role fit (NO tool call):** You do NOT have get_enhanced_role_fit in chat. When the user asks for "role fits", "show role fits", "what roles suit me", "what roles am I qualified for", "which roles fit my profile", etc.:
   - Use **role_fit_suggestions** and **career_paths** from your context (see RELEVANT JOBS/ROLES FOR YOU and career_paths in context). Show 1–2 **role names only** (e.g. from role_fit_suggestions or career_paths titles). Do NOT call any tool.
   - Then add one short line: "See the Resume Analysis page for more details and match rationale."
   - If context has no role_fit_suggestions and no career_paths, say briefly that role fit is based on their resume and goals, then: "See the Resume Analysis page for a detailed role fit analysis."

3. **get_resume_score** (resume score, role suggestions):
   - Use context first: resume_score, RoleSuggestions from context if available
   - CALL get_resume_score when: (a) context has no resume_score and user asks about resume quality, scoring, or role fit from resume, OR (b) user explicitly asks to refresh resume score
   - Clear indicators: "How does my resume score?", "Rate my resume", "What roles suit my resume?", "Refresh my resume analysis"
   - If user asks for fresh resume analysis, set force_refresh=true

3b. **Salary/compensation (NO tool call):** When user asks about salary, compensation, expected pay, or salary range, do NOT call any tool. Use the candidate's profile (resume, experience, skills, location) already in your context to infer their **current level** (e.g. role, seniority band). Reply with **one short line** stating that level (e.g. "Based on your experience as [role], you're likely in a mid-level band"). Then add: "For detailed salary ranges and growth outlook, check the Premium Insights section." Do not reveal exact figures; full salary report is only in Premium Insights.

3c. **get_market_insights_only** (market analysis, no courses) – PREMIUM / TEASER:
   - CALL when user asks for market analysis, market outlook, or combined salary + career paths overview but does NOT want course recommendations.
   - Clear indicators: "What's my market outlook?", "Market analysis", "Salary and career outlook", "Full market report" (without asking for courses).
   - Do NOT call for course requests—use get_course_recommendations (with courses_only=true for topic-specific) instead.
   - **In chat:** Do NOT present full market_insights, salary_trends, or career_paths. You will receive only a report-ready payload. Follow **PREMIUM FEATURES**: teaser + nudge to the **Premium Insights section**.

4. **get_assessment_recommendations**: 
   - ONLY call when user EXPLICITLY requests assessments or skill tests
   - Clear indicators: "I want assessments", "test my skills", "evaluate my abilities", "create assessment plan"
   - If the user names topics (e.g., "Python", "React"), pass topics=["Python", ...]; set force_refresh if they ask for new/updated recs
   - If user says "suggest topics", set suggest_new_topics=true
   - DO NOT call for general questions about skills

5. **create_career_goal / get_career_goals / update_career_goal (GOAL BRAIN):**
   - You are the career mentor brain: ALWAYS consider the user's goals/milestones/tasks and progress.
   - On every turn, if goals are known in context, briefly factor them into your advice.
   
   **PROACTIVE GOAL OFFERS:**
   - When user expresses a clear career aspiration (e.g., "I want to expand into cybersecurity", "I'm interested in data science", "I want to become a product manager"), and you've just provided courses/assessments for that topic, OFFER to create a goal
   - After providing course or assessment recommendations, if the user expressed an aspiration, naturally offer: "Would you like me to create a goal to track your [topic] journey? I can set up milestones for the courses and assessments we just discussed."
   - Keep it conversational - don't be pushy, just offer it as a helpful option
   - If user says yes or agrees, THEN call create_career_goal with:
     * Title: Clear, actionable goal based on their aspiration
     * Description: Why this goal matters based on their profile
     * Milestones: 2-4 key milestones (e.g., "Complete foundational courses", "Pass assessment", "Build project")
     * Tasks: Link the courses/assessments you just recommended as tasks under this goal
   
   **EXPLICIT GOAL REQUESTS:**
   - When user explicitly says "my goal is", "set a goal", "track my goal": call create_career_goal immediately (include milestones/tasks if given).
   - When user asks about progress, status, next steps, or to see their plan: call get_career_goals.
   - When user wants to update progress, add tasks, or mark done: call update_career_goal (set status/progress, add tasks/milestones, complete tasks).
   - Indicators to use goal tools: "my goal is", "set a goal", "track my goal", "update my progress", "add a task", "mark task done", "what's next".
   - Nudge gently: propose 1–3 next tasks with due dates/effort; ask for confirmation before adding.
   
   **LINKING RECOMMENDATIONS TO GOALS:**
   - When creating a goal after providing courses/assessments, automatically add those recommendations as tasks:
     * Each course becomes a task with type="course", effort_hours estimated from course duration
     * Each assessment becomes a task with type="assessment", effort_hours = assessment_time_minutes / 60
     * Set realistic due dates based on effort_hours and 2-3 hours/day commitment
   
   **INTELLIGENT DATE SETTING FOR GOALS/TASKS:**
   - When creating goals or tasks, ALWAYS set realistic due dates and effort_hours based on:
     1. **User's explicit time request (HIGHEST PRIORITY)**: If user says "by next week", "in 2 days", "by May 17th", use that exactly as target_date/due_date
     2. **Task difficulty and type** (if no user time specified):
        * Assessments: 30-60 min + prep time (easy: 0.5h prep, medium: 1h, hard: 2h, expert: 3h prep)
        * Courses: Check course duration from context if available, or estimate (easy: 5h, medium: 10h, hard: 20h, expert: 40h)
        * Projects/Build tasks: easy: 4h, medium: 8h, hard: 16h, expert: 32h
        * Reading/Study tasks: easy: 2h, medium: 4h, hard: 8h, expert: 16h
     3. **Default assumption**: 2-3 hours per day commitment (realistic for working professionals)
     4. **Buffer**: System adds 20% buffer for unexpected delays automatically
   - **ALWAYS include effort_hours** in task dict when creating tasks (helps with date calculation)
   - **ALWAYS include due_date or target_date** in task dict (system will auto-calculate if missing, but explicit is better)
   - If user says "as soon as possible" or "urgent", set priority="high" and use shorter timeframes
   - If user doesn't specify time, calculate from task difficulty/type and include effort_hours

6. **identify_relevant_jobs**: 
   - ONLY call when user EXPLICITLY asks to SEE, FIND, or SEARCH for actual job listings from the database
   - If they give a query/role/location, pass search_query and top_k (default 10) accordingly
   - Clear indicators to CALL the function (wants database search):
     * "What jobs are available for me?" 
     * "Show me suitable jobs"
     * "Find jobs that match my profile"
     * "Search for jobs for me"
     * "What job opportunities can you find?"
     * "Are there any open positions for me?"
   - DO NOT call for general questions (answer from context instead):
     * "What careers should I pursue?" → Use career_paths from context
     * "What roles am I qualified for?" → Use role_fit_suggestions from context  
     * "What jobs would I be suitable for?" → Use career_paths and role_fit_suggestions from context
     * "What career paths can I take?" → Use career_paths from context
   - The key difference: "find/show/search/available" = call function to search database, "what/which/suitable" = use existing context
   - **When presenting job matches:** You will receive a payload with `jobs_teaser: true` and a list `jobs` with only `job_title`, `company`, and `match_score`. For each job show **only**: job title, company name, and match score (e.g. "**85% Match**"). Do **not** show rationale, strong alignment, minor gaps, or any detailed matching info (rationale is a paid feature). After the list, add one short line: "See the **Jobs** section in JobsifyAI for more matches and detailed match rationale."

7. **get_resume_optimization_suggestions**: 
   - Call when user asks about improving their resume, resume tips, ATS optimization, resume feedback, or resume optimization
   - Clear indicators: "How can I improve my resume?", "Resume tips", "ATS optimization", "Resume feedback", "Optimize my resume"
   - Returns actionable suggestions with priorities (high/medium/low) and specific actions

8. **get_learning_path**: 
   - Call when user asks for a learning roadmap, structured learning plan, or how to progress in a skill/topic
   - Clear indicators: "Learning path", "Learning roadmap", "How should I learn [topic]?", "Structured learning plan"
   - If user mentions a specific topic (e.g., "Python", "Data Science"), pass it in the topic parameter
   - Returns stages (beginner → intermediate → advanced) with courses and assessments for each stage

**SMART TOOL USAGE:**
- Use your existing context (resume, career_paths, skill_gaps, course_recommendations, etc.) to answer MOST questions
- Only call tools when the user needs fresh data, database search, or explicit actions
- Goal-first: if goals exist, tailor answers to move the user forward on their active goals/milestones/tasks
- If no goals exist and user is open, propose creating one succinct goal and optional milestones
- Think: "Does the user want me to FETCH/SEARCH/FIND new data from systems, or are they asking a QUESTION I can answer from what I already know?"
- Default to using context unless the user clearly wants a database search or new recommendations

**ASSESSMENT RECOMMENDATIONS - THREE WAYS TO WORK:**

1. **USER REQUESTS SPECIFIC TOPIC (Highest Priority):**
   - If user explicitly requests assessments for a specific topic (e.g., "I want to test my Python skills", "assessments for modern hairstyles")
   - Extract the topic from their message and use it exactly
   - Call get_assessment_recommendations with topics=["Python"] or topics=["modern hairstyles"]
   - This creates a NEW assessment plan for that topic (even if other assessments exist)

2. **PROACTIVE SUGGESTIONS (You suggest topics):**
   - If user asks "What assessments should I take?" or "recommend assessments" without specifying topics
   - Analyze their profile, skill gaps, career paths, and improvement areas
   - Identify 3-5 relevant topics they should assess
   - Call get_assessment_recommendations with suggest_new_topics=true
   - The function will suggest topics and create assessments ONLY for topics that don't already have assessments
   - This ensures you don't duplicate existing assessments

3. **GENERAL RECOMMENDATIONS (No topics specified):**
   - If user just asks for "assessments" without specifics
   - Call get_assessment_recommendations with suggest_new_topics=true
   - This will suggest new topics based on profile and create assessments for topics without existing plans

**IMPORTANT RULES:**
- If user mentions a specific topic → ALWAYS use that topic (topics=["user_topic"])
- If user asks generally → Use suggest_new_topics=true to avoid duplicates
- Always create assessments in the same format as the assessment recommender agent

**HOW TO USE TOOLS:**
1. When user requests something that requires a tool, call the appropriate function automatically
2. The tools are smart - they check existing data first (instant), and only call agents if data is missing
3. Present the results naturally in your response - don't mention "I called a function"
4. If user explicitly asks for "fresh" or "new" recommendations, set force_refresh=true
5. For assessments, you can proactively suggest topics based on skill gaps or career goals

**YOUR ROLE:**
Provide holistic, personalized career guidance that considers EVERYTHING we know about them.
Connect insights across all data sources to give comprehensive advice.

**MAKING ADVICE UNDERSTANDABLE:**
- Always explain recommendations in simple, clear language
- When mentioning technical skills or concepts, explain what they are and why they matter
- Use real-world examples to illustrate points (e.g., "Python is a programming language used for data analysis - think of it like Excel but much more powerful")
- Break down complex career paths into understandable steps
- When suggesting courses or skills, explain:
  * What the skill/course is about
  * Why it's relevant to their goals
  * How it will help them advance
  * What they'll be able to do after learning it
- Avoid industry jargon without explanation
- Make comparisons to familiar concepts when explaining new ideas
- Prefer asterisk bullets (* at the start of each point) and short sentences over long paragraphs for clarity and UI-friendly formatting

**UNDERSTANDING USER'S ASPIRATIONS & GOALS:**
- Pay attention to their stated aspirations and career goals (shown above)
- Reference their goals when giving advice
- Align recommendations with what they want to achieve
- If they mention new aspirations in conversation, acknowledge and incorporate them
- Help them progress toward their stated goals

{resume_section}
{location_currency_section}
{resume_score_section}
{skills_section}
{career_paths_section}
{assessment_section}
{courses_section}
{history_section}
{aspirations_section}
{goals_section}
{jobs_section}
{insights_section}

**USER'S QUESTION/MESSAGE:**
{user_message}

**INSTRUCTIONS:**
1. Answer their question directly and concisely - don't write long formal advice paragraphs
2. **Format for readability:** Use short sentences. For lists (advice, recommendations, steps, options), start each item with "* " (asterisk space) so the UI can format consistently. Keep paragraphs to 2-3 sentences when you use them.
3. Synthesize insights from ALL available data sources when relevant
4. Reference specific assessments, skills, or recommendations naturally when they help answer the question
5. Be conversational and natural - like you're chatting with a friend
6. Keep responses focused - answer what they asked, don't dump everything you know
7. **Resume-related questions:** Candidates often ask about their resume (e.g. "What skills do I have?", "Summarize my experience", "What's on my resume?"). Use the **RESUME DETAIL** section above to answer accurately with specifics—list actual skills, roles, education, projects—don't guess or paraphrase only the summary.
8. **Skill score & presentation score:** These come from **RESUME SCORE** (resume analysis), NOT from assessments. When user asks for skill score or presentation score, use the **RESUME SCORE** section if available. If not available, call get_resume_score to fetch it. Do NOT confuse with assessment average (which is from completed skill tests).
9. **Resume optimization:** When users ask about improving their resume, use get_resume_optimization_suggestions to provide actionable, prioritized suggestions.
10. **Salary questions:** When users ask about salary, compensation, expected pay, or salary range, do NOT call any tool. Use the candidate's profile (resume, experience, skills, location) in your context to infer their **current level** (e.g. role, seniority). Give **one short line** (e.g. "Based on your experience as [role], you're likely in a mid-level band"), then: "For detailed salary ranges and growth outlook, check the Premium Insights section." Do not reveal exact figures; full salary report is only in Premium Insights.
11. **Learning paths:** When users want a structured learning plan, use get_learning_path to create a beginner → intermediate → advanced progression.
12. Make all advice CLEAR and UNDERSTANDABLE:
   - Explain technical terms in simple language
   - Use concrete examples to illustrate points
   - Break down complex recommendations into simple steps
   - Explain WHY something is recommended, not just WHAT
   - Make suggestions actionable and specific
13. If appropriate, you can naturally mention related topics (e.g., "If you're curious about X, I can help with that too"), but keep it brief
14. End your response in a way that invites further conversation (unless they're clearly ending)

**DYNAMIC QUESTION GENERATION:**
After your response, generate 2-4 contextually relevant follow-up questions that:
- Are phrased as if the USER is asking them (clickable - when clicked, they become the user's next message)
- Relate to what was just discussed
- Are natural, conversational questions (not formal)
- Help guide the conversation forward
- Are actionable and specific to their situation
- Consider their profile, goals, and aspirations
- When clicked, should trigger specific actions (show courses, find jobs, create assessments, etc.)

Examples of good clickable questions (user can send these as their message):
- "Can you show me courses to help me bridge my AWS skill gap?"
- "I'd like to take an assessment to see my current proficiency in Python or other AI/ML concepts"
- "Which of these areas (Agentic AI, Gen AI, e-commerce recommendations, Oracle HCM, RPA) should I focus on first?"
- "Can you find job opportunities that align with these career paths?"
- "What courses would help me improve in Python?"
- "Show me assessments I can take for data analysis"

Examples of bad questions (chatbot asking, not user asking):
- "Would you like to explore courses..." (chatbot offering)
- "Are you interested in taking..." (chatbot asking)
- "Do you want to learn more?" (chatbot asking)
- "What else can I help with?" (chatbot asking)
- "Python" (this is a topic, not a question)

**Response Format (UI-friendly):**
Your message is rendered by the client using markdown. Use rich formatting for better readability and professional appearance:

**1. Lists & Bullets:**
* For bullet points, start each item with "* " (asterisk space)
* For numbered steps (sequential actions), use "1. ", "2. ", etc.

**2. Code Blocks:**
When showing technical content (commands, code snippets, config examples, technical syntax), use fenced code blocks with the language specified:
```bash
pip install tensorflow
```
```python
model.fit(X_train, y_train)
```

**3. Comparison Tables:**
When comparing options (roles, career paths, skills, courses, technologies), use markdown tables for easy scanning:
| Aspect | Tech Lead | Engineering Manager |
|--------|-----------|---------------------|
| Focus | Technical architecture | People & delivery |
| Skills needed | System design, coding | Leadership, hiring |
| Your fit | 85% | 70% |

**4. Progress/Roadmap Visualizations:**
For learning paths, career progressions, or skill development journeys, use visual indicators:
* 📍 **Current:** Junior Developer
* ➡️ **Next Step:** Mid-level Developer (6-12 months)
* 🎯 **Goal:** Senior Developer (2-3 years)

Or use horizontal flow for simple progressions:
Beginner → Intermediate → Advanced → Expert

**5. Inline Links:**
Always use [link text](url) for clickable links. Never show raw URLs. Example: [Python Fundamentals](https://example.com/course)

**6. Emphasis:**
* Use **bold** for key terms, role names, percentages, or important numbers
* Use `backticks` for technical terms, tool names, programming languages, or specific skills

**7. Section Headers:**
For longer responses with multiple topics, use headers to organize:
### Your Skill Gaps
### Recommended Courses
### Next Steps

Keep responses conversational but well-structured. Use formatting to make content scannable, not to make it longer.

**FINAL CHECK before responding:**
- Did I use bullets for lists? (If not, add them)
- Is any paragraph longer than 2 sentences? (If yes, break it up)
- Did I bold key terms? (Role names, percentages, section names)
- Is it easy to scan in a small chat window? (If not, restructure)

Provide your response naturally in a conversational style. At the end, include a JSON section with:
```json
{{
    "suggested_questions": [
        "Can you show me courses to help me bridge my AWS skill gap?",
        "I'd like to take an assessment to see my current proficiency in Python or other AI/ML concepts",
        "Which of these areas should I focus on first?"
    ],
    "action_items": [
        {{"action": "action description", "priority": "high|medium|low"}},
        {{"action": "action description", "priority": "high|medium|low"}}
    ]
}}
```

**IMPORTANT:** The suggested_questions and action_items are for the system to track - they should NOT be explicitly mentioned in your conversational response unless naturally relevant. Your response should feel like a natural conversation, not a formal report.
"""
    
    return prompt


# ============================================================================
# STREAMING LLM INVOCATION
# ============================================================================

async def _stream_llm_response_with_tools(
    prompt: str,
    uid: str,
    session_id: Optional[str] = None,
    agent_name: str = "career_chatbot",
    tool_results: Optional[Dict[str, Any]] = None
) -> AsyncIterator[str]:
    """
    Stream LLM response with function calling support.
    Checks existing data first, calls agents only when needed.
    
    Yields:
        str: Individual tokens/chunks from the LLM
    """
    try:
        # Get model config
        model_config = get_model_for_task(TaskType.TEXT_GENERATION, preferred_model=LLM_MODEL)
        
        # Create model instance
        model_instance = await _create_model_instance(
            model_config,
            response_mime_type=None,  # No structured output for streaming
            max_output_tokens=MAX_RESPONSE_LENGTH,
        )
        
        # Create LangChain tools from function definitions
        # Exclude get_salary_insights: LLM infers level from context and nudges to Premium Insights
        tools = []
        excluded_tools = {"get_salary_insights"}
        function_defs = [f for f in get_function_definitions() if f.get("name") not in excluded_tools]
        
        for func_def in function_defs:
            func_name = func_def["name"]
            
            # Create async tool wrapper that properly captures func_name, uid, and session_id
            # Use a factory function to create proper closure for each function
            def make_tool_wrapper(fn_name: str):
                async def tool_wrapper(**kwargs):
                    return await handle_function_call(fn_name, kwargs, uid, session_id)
                return tool_wrapper
            
            # Call factory with current func_name to create closure
            tool_func = make_tool_wrapper(func_name)
            
            tool = StructuredTool.from_function(
                func=tool_func,
                name=func_def["name"],
                description=func_def["description"]
            )
            tools.append(tool)
        
        # Bind tools to model
        model_with_tools = model_instance.bind_tools(tools)
        
        # Convert prompt to messages
        messages = [
            SystemMessage(content=(
                "You are an expert AI Career Mentor with access to tools that can fetch data or call agents. "
                "When a user asks for courses, career advice, assessments, or wants to set goals, use the appropriate tool. "
                "The tools are smart - they check existing data first (fast) and only call agents if needed. "
                "Present results naturally without mentioning function calls."
            )),
            HumanMessage(content=prompt)
        ]
        
        # Stream response (with potential function calls)
        function_calls_made = []
        tool_calls_detected = False
        # Note: Tool results are captured in tool_results dict, which is used in callback
        # All tools automatically save their outputs to ChromaDB
        
        async for chunk in model_with_tools.astream(messages):
            # Check if chunk contains function calls
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                tool_calls_detected = True
                # Handle function calls
                # First, add the AIMessage with tool calls to messages
                messages.append(chunk)
                
                for tool_call in chunk.tool_calls:
                    # Handle different tool_call formats (dict or object)
                    if isinstance(tool_call, dict):
                        func_name = tool_call.get("name", "")
                        func_args = tool_call.get("args", {})
                        tool_call_id = tool_call.get("id", "")
                    else:
                        # ToolCall object - use getattr with fallbacks
                        func_name = getattr(tool_call, "name", None)
                        if not func_name:
                            # Try accessing via function attribute
                            func_obj = getattr(tool_call, "function", None)
                            if func_obj:
                                func_name = getattr(func_obj, "name", None) if hasattr(func_obj, "name") else (func_obj.get("name") if isinstance(func_obj, dict) else None)
                        
                        func_args = getattr(tool_call, "args", {})
                        if not func_args:
                            func_obj = getattr(tool_call, "function", None)
                            if func_obj:
                                if hasattr(func_obj, "arguments"):
                                    func_args = func_obj.arguments
                                elif isinstance(func_obj, dict):
                                    func_args = func_obj.get("arguments", {})
                        
                        tool_call_id = getattr(tool_call, "id", None) or getattr(tool_call, "tool_call_id", "")
                    
                    # Parse args if it's a string
                    if isinstance(func_args, str):
                        try:
                            func_args = json.loads(func_args)
                        except:
                            func_args = {}
                    
                    # Handle nested kwargs structure (LangChain sometimes wraps args in 'kwargs')
                    if isinstance(func_args, dict) and "kwargs" in func_args and len(func_args) == 1:
                        func_args = func_args["kwargs"]
                        log.debug(f"📦 Unwrapped kwargs: {func_args}")
                    
                    if not func_name:
                        log.warning(f"⚠️ Could not extract function name from tool_call: {type(tool_call)}")
                        continue
                    
                    if not tool_call_id:
                        # Generate a tool_call_id if missing
                        tool_call_id = f"call_{func_name}_{uuid.uuid4().hex[:8]}"
                        log.warning(f"⚠️ Missing tool_call_id, generated: {tool_call_id}")
                    
                    log.info(f"🔧 LLM requested function: {func_name} with args: {func_args}")
                    function_calls_made.append(func_name)
                    
                    # ✅ NEW: Send status event to client BEFORE executing tool (for UI feedback)
                    tool_status_messages = {
                        "identify_relevant_jobs": "🔍 Searching for jobs that match your profile...",
                        "get_assessment_recommendations": "📝 Creating personalized assessment recommendations...",
                        "get_course_recommendations": "🎓 Finding personalized courses for you...",
                        "get_resume_score": "📊 Analyzing your resume and scoring...",
                        "get_career_advice": "💼 Generating career paths and role fit analysis...",
                        "get_enhanced_role_fit": "🎯 Generating detailed role fit analysis...",
                        "get_salary_insights": "💰 Analyzing your salary expectations...",
                        "get_market_insights_only": "📈 Preparing your market and career outlook...",
                        "create_career_goal": "🎯 Setting up your career goal..."
                    }
                    status_message = tool_status_messages.get(func_name, f"⚙️ Processing {func_name.replace('_', ' ')}...")
                    
                    # Yield status event (special format that will be handled separately)
                    status_event = {
                        "type": "tool_start",
                        "tool": func_name,
                        "message": status_message
                    }
                    yield f"__STATUS_EVENT__:{json.dumps(status_event)}"
                    log.info(f"📡 Sent tool_start status: {func_name}")
                    
                    # Execute function (this may take time - user will see status message)
                    func_result = await handle_function_call(func_name, func_args, uid, session_id)
                    
                    # ✅ NEW: Send completion event
                    status_complete = {
                        "type": "tool_complete",
                        "tool": func_name
                    }
                    yield f"__STATUS_EVENT__:{json.dumps(status_complete)}"
                    log.info(f"📡 Sent tool_complete status: {func_name}")
                    
                    # Capture tool/agent results for callback
                    # IMPORTANT: All tools automatically save their outputs to ChromaDB
                    # We capture here to include in callback without reloading from DB
                    if tool_results is not None:
                        try:
                            tool_results.setdefault(func_name, []).append({
                                "arguments": func_args,
                                "result": func_result,
                            })
                            source = func_result.get("source", "unknown") if isinstance(func_result, dict) else "unknown"
                            if isinstance(func_result, dict) and func_result.get("success"):
                                log.info(f"📋 Captured {func_name} result for callback (source: {source}, data saved to ChromaDB)")
                            else:
                                log.debug(f"📋 Captured {func_name} result for callback (source: {source})")
                        except Exception as capture_err:
                            log.debug(f"Failed to record tool result for {func_name}: {capture_err}")
                    
                    # Create ToolMessage: for premium tools, pass minimal teaser so LLM cannot leak insights
                    content_for_llm = _tool_result_for_llm(func_name, func_result)
                    tool_message = ToolMessage(
                        content=content_for_llm,
                        tool_call_id=tool_call_id
                    )
                    messages.append(tool_message)
                    
                    log.info(f"✅ Function {func_name} completed, source: {func_result.get('source', 'unknown')}")
                
                # After function calls, continue streaming the final response
                # IMPORTANT: Use astream to continue streaming, not ainvoke
                log.info("🔄 Streaming final response after function calls...")
                async for final_chunk in model_with_tools.astream(messages):
                    # Skip tool calls in final response (we already handled them)
                    if hasattr(final_chunk, 'tool_calls') and final_chunk.tool_calls:
                        continue
                    
                    # Stream content chunks (plain text only; strip Gemini list-of-dicts)
                    if hasattr(final_chunk, 'content'):
                        content = final_chunk.content
                        if content:
                            yield _extract_text_from_chunk(content)
                    elif isinstance(final_chunk, str):
                        yield _extract_text_from_chunk(final_chunk)
                    else:
                        yield _extract_text_from_chunk(str(final_chunk) if final_chunk else "")
                break
            else:
                # Regular content chunk - yield plain text only (strip Gemini list-of-dicts)
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    if content:
                        yield _extract_text_from_chunk(content)
                elif isinstance(chunk, str):
                    yield _extract_text_from_chunk(chunk)
                else:
                    yield _extract_text_from_chunk(str(chunk) if chunk else "")
        
        if function_calls_made:
            log.info(f"📊 Function calls made: {function_calls_made}")
        
    except Exception as e:
        log.error(f"Streaming LLM error: {e}", exc_info=True)
        yield f"\n\n[Error: {str(e)}]"


async def _stream_llm_response(
    prompt: str,
    agent_name: str = "career_chatbot"
) -> AsyncIterator[str]:
    """
    Stream LLM response token by token (fallback without tools).
    
    Yields:
        str: Individual tokens/chunks from the LLM
    """
    try:
        # Get model config
        model_config = get_model_for_task(TaskType.TEXT_GENERATION, preferred_model=LLM_MODEL)
        
        # Create model instance
        model_instance = await _create_model_instance(
            model_config,
            response_mime_type=None,  # No structured output for streaming
            max_output_tokens=MAX_RESPONSE_LENGTH,
        )
        
        # Convert prompt to messages
        messages = [
            SystemMessage(content="You are an expert AI Career Mentor."),
            HumanMessage(content=prompt)
        ]
        
        # Stream response (plain text only; strip Gemini list-of-dicts)
        async for chunk in model_instance.astream(messages):
            if hasattr(chunk, 'content'):
                content = chunk.content
                if content:
                    yield _extract_text_from_chunk(content)
            elif isinstance(chunk, str):
                yield _extract_text_from_chunk(chunk)
        
    except Exception as e:
        log.error(f"Streaming LLM error: {e}", exc_info=True)
        yield f"\n\n[Error: {str(e)}]"


# ============================================================================
# STREAMING RESPONSE GENERATOR
# ============================================================================

async def _generate_streaming_response(
    request: CareerChatRequest,
    context: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    history_to_save: Optional[List[Dict[str, str]]] = None,
) -> AsyncIterator[str]:
    """
    Generate streaming response chunks.
    Thread-safe: Each request has isolated session and context.
    conversation_history: used for prompt, extraction, analysis (may include prior-session context).
    history_to_save: appended to and persisted; if None, use conversation_history.
    Format: Server-Sent Events (SSE) format
    """
    save_list = history_to_save if history_to_save is not None else conversation_history
    full_response = ""
    # CRITICAL: Generate unique session_id per user to ensure isolation
    # Format: {uid}_{timestamp}_{random} ensures uniqueness even for concurrent requests
    session_id = request.session_id or f"{request.uid}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    try:
        # Extract aspirations and goals from conversation (async, non-blocking)
        # Do this before building prompt so we can include in context
        try:
            # Extract aspirations from full conversation history
            aspirations = await extract_aspirations_from_conversation(conversation_history, request.uid)
            
            # Extract goals from current message
            goals_from_message = await extract_goals_from_message(request.message, request.uid)
            
            # Store aspirations and goals in session (if we have new data)
            if aspirations.get("career_goals") or aspirations.get("aspirations") or goals_from_message:
                try:
                    from chroma import get_chat_session, update_chat_session
                    session_data = await run_blocking_io(get_chat_session, session_id, request.uid) or {}
                    
                    # Update aspirations
                    if aspirations.get("career_goals") or aspirations.get("aspirations"):
                        existing_aspirations = session_data.get("career_aspirations", {})
                        # Merge new aspirations with existing
                        merged_aspirations = {
                            "career_goals": list(set(existing_aspirations.get("career_goals", []) + aspirations.get("career_goals", [])))[:10],
                            "aspirations": list(set(existing_aspirations.get("aspirations", []) + aspirations.get("aspirations", [])))[:10],
                            "skills_of_interest": list(set(existing_aspirations.get("skills_of_interest", []) + aspirations.get("skills_of_interest", [])))[:10],
                            "career_paths_of_interest": list(set(existing_aspirations.get("career_paths_of_interest", []) + aspirations.get("career_paths_of_interest", [])))[:10],
                            "timeline": aspirations.get("timeline") or existing_aspirations.get("timeline"),
                            "confidence": max(aspirations.get("confidence", 0.0), existing_aspirations.get("confidence", 0.0))
                        }
                        session_data["career_aspirations"] = merged_aspirations
                        context["aspirations"] = merged_aspirations
                    
                    # Add new goals from message
                    if goals_from_message:
                        existing_goals = session_data.get("career_goals", [])
                        if not isinstance(existing_goals, list):
                            existing_goals = []
                        # Add new goals (avoid duplicates)
                        existing_goal_titles = {g.get("title", "").lower() for g in existing_goals if isinstance(g, dict)}
                        for new_goal in goals_from_message:
                            if isinstance(new_goal, dict) and new_goal.get("title", "").lower() not in existing_goal_titles:
                                new_goal["goal_id"] = f"goal_{request.uid}_{int(time.time() * 1000)}"
                                new_goal["uid"] = request.uid
                                new_goal["status"] = "active"
                                new_goal["created_at"] = datetime.utcnow().isoformat()
                                existing_goals.append(new_goal)
                                existing_goal_titles.add(new_goal.get("title", "").lower())
                        session_data["career_goals"] = existing_goals
                        context["career_goals"] = existing_goals
                    
                    # Update conversation insights (build profile over time)
                    existing_profile = session_data.get("conversation_insights", {})
                    conversation_insights = await analyze_conversation_patterns(
                        conversation_history, 
                        request.uid,
                        existing_profile=existing_profile  # Merge with existing profile
                    )
                    # Add timestamp
                    # from datetime import datetime
                    conversation_insights["last_analyzed"] = datetime.utcnow().isoformat()
                    session_data["conversation_insights"] = conversation_insights
                    context["conversation_insights"] = conversation_insights
                    
                    # Save updated session
                    await run_blocking_io(update_chat_session, session_id, session_data)
                    log.info(f"💾 Updated session with aspirations and goals for uid={request.uid}")
                except Exception as e:
                    log.warning(f"Could not store aspirations/goals in session: {e}")
        except Exception as e:
            log.warning(f"Error extracting aspirations/goals: {e}")
        
        # Build prompt (context is already loaded per-request, so it's isolated)
        prompt = _build_career_chat_prompt(request.message, context, conversation_history)
        
        # Send initial event
        yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"
        
        # Stream LLM response WITH function calling support (each stream is independent)
        # IMPORTANT: We accumulate chunks in full_response while streaming to client
        # After stream completes, we send the FULL response to callback
        chunk_count = 0
        log.info("🔄 Starting LLM stream with function calling support...")
        log.info(f"📡 Streaming chunks to client while accumulating full response for callback...")
        
        # Container to capture any tool/agent outputs during this streamed response
        tool_results: Dict[str, Any] = {}
        
        async for chunk in _stream_llm_response_with_tools(
            prompt,
            request.uid,
            session_id,
            tool_results=tool_results
        ):
            # Ensure chunk is a string and extract plain text (strip Gemini list-of-dicts)
            raw = str(chunk) if not isinstance(chunk, str) else chunk
            chunk_str = _extract_text_from_chunk(raw)
            
            # ✅ NEW: Check if this is a status event (not regular content)
            if chunk_str.startswith("__STATUS_EVENT__:"):
                # Extract status event JSON
                try:
                    status_json_str = chunk_str.replace("__STATUS_EVENT__:", "")
                    status_data = json.loads(status_json_str)
                    # Send as SSE status event to client (for UI updates/loading indicators)
                    # status_data already has "type": "tool_start" or "tool_complete"
                    # We wrap it with "event": "status" to distinguish from content chunks
                    status_event_payload = {
                        "type": "status",
                        "status_type": status_data.get("type"),  # tool_start or tool_complete
                        "tool": status_data.get("tool"),
                        "message": status_data.get("message", "")
                    }
                    yield f"data: {json.dumps(status_event_payload)}\n\n"
                    log.debug(f"📡 Sent status event to client: {status_data.get('type')} for {status_data.get('tool', 'unknown')}")
                except Exception as e:
                    log.warning(f"Failed to parse status event: {e}")
                # Don't accumulate status events in full_response - they're just UI updates
                continue
            
            # Regular content chunk - accumulate for callback
            full_response += chunk_str
            chunk_count += 1
            
            # Log chunks in terminal (first 3, then every 10th)
            if chunk_count <= 3:
                log.info(f"📦 Chunk #{chunk_count} (to client): {chunk_str[:100]}{'...' if len(chunk_str) > 100 else ''}")
            elif chunk_count % 10 == 0:
                log.info(f"📦 Chunk #{chunk_count} (progress) | Accumulated: {len(full_response)} chars")
            
            # Send chunk as SSE event to client (real-time streaming)
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk_str})}\n\n"
        
        # Log streaming completion
        log.info(f"✅ Stream completed: {chunk_count} chunks received")
        log.info(f"📦 Full response accumulated: {len(full_response)} characters (ready for callback)")
        log.info(f"🤖 Assistant Response (streamed): {full_response[:200]}{'...' if len(full_response) > 200 else ''}")
        
        # Parse final response for structured data
        suggested_topics = []  # Deprecated - kept for backward compatibility
        suggested_questions = []  # NEW: Contextually relevant questions
        action_items = []
        
        # Try to extract JSON from end of response
        # Look for JSON block (could be in ```json code block or plain JSON)
        try:
            # First, try to find JSON in code block
            json_code_block_pattern = r'```json\s*(\{.*?\})\s*```'
            json_match = re.search(json_code_block_pattern, full_response, re.DOTALL)
            
            if json_match:
                json_block = json_match.group(1)
            else:
                # Look for JSON block at the end (plain JSON)
                json_start = full_response.rfind('{')
                if json_start > 0:
                    json_text = full_response[json_start:]
                    # Try to find closing brace (handle nested objects)
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(json_text):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    
                    if json_end > 0:
                        json_block = json_text[:json_end]
                    else:
                        json_block = None
                else:
                    json_block = None
            
            if json_block:
                parsed = json.loads(json_block)
                # Support both old (suggested_topics) and new (suggested_questions) format
                extracted_questions = parsed.get("suggested_questions", [])
                extracted_topics = parsed.get("suggested_topics", [])  # Fallback for backward compatibility
                extracted_action_items = parsed.get("action_items", [])
                
                # Use extracted data (ensure all items are included, no filtering)
                if extracted_questions:
                    suggested_questions = extracted_questions  # Use all questions from JSON
                if extracted_topics:
                    suggested_topics = extracted_topics
                if extracted_action_items:
                    action_items = extracted_action_items  # Use all action items from JSON
                
                # Remove JSON from response text (both code block and plain JSON)
                if json_match:
                    # Remove the entire code block
                    full_response = re.sub(json_code_block_pattern, '', full_response, flags=re.DOTALL).strip()
                else:
                    # Remove plain JSON block
                    full_response = full_response[:json_start].strip()
                    
                log.debug(f"✅ Extracted {len(suggested_questions)} questions and {len(action_items)} action items from JSON block")
        except Exception as e:
            log.debug(f"Could not extract JSON from response: {e}")
            pass  # If JSON parsing fails, continue without structured data
        
        # Generate questions if LLM didn't provide them (fallback)
        if not suggested_questions:
            suggested_questions = await _generate_contextual_questions(
                user_message=request.message,
                conversation_history=conversation_history,
                context=context,
                assistant_response=full_response
            )
        
        # Add assistant response to history we persist (current session only)
        save_list.append({
            "role": "assistant",
            "content": full_response,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Log conversation summary
        log.info(f"💬 Conversation Summary | session_id={session_id} | history_length={len(save_list)} messages")
        # Log conversation insights if available
        if context.get("conversation_insights"):
            insights = context["conversation_insights"]
            log.info(f"📊 Conversation Insights: engagement={insights.get('engagement_level', 0):.2f}, preferred_topics={insights.get('preferred_topics', [])[:3]}")
        log.info(f"❓ Suggested Questions: {len(suggested_questions)} questions")
        if suggested_questions:
            for i, q in enumerate(suggested_questions[:3], 1):
                log.info(f"   {i}. {q[:80]}{'...' if len(q) > 80 else ''}")
        log.info(f"✅ Action Items: {len(action_items)} items")
        
        # Save conversation history (non-blocking, but with proper uid isolation)
        # Use create_task to avoid blocking the stream, but ensure uid is passed
        asyncio.create_task(_save_conversation_history(session_id, save_list, request.uid))
        
        # Log final response summary
        log.info(f"📋 Final Response Summary | session_id={session_id}")
        log.info(f"   - Full response: {len(full_response)} chars")
        log.info(f"   - Suggested questions: {len(suggested_questions)}")
        log.info(f"   - Action items: {len(action_items)}")
        
        # Send final event with metadata
        final_data = {
            'type': 'done',
            'session_id': session_id,
            'content_type': 'markdown',
            'suggested_questions': suggested_questions,  # NEW
            'suggested_topics': suggested_topics,  # Keep for backward compatibility
            'action_items': action_items,
            'full_response': full_response
        }
        # Log final SSE payload for dev (truncate full_response so VM logs stay readable)
        try:
            log_data = {**final_data, 'full_response': full_response[:400] + ("..." if len(full_response) > 400 else "")}
            log.info(f"📦 Final SSE payload (done): %s", json.dumps(log_data, default=str, ensure_ascii=False)[:3500])
        except Exception as _e:
            log.debug(f"Could not log final payload: {_e}")
        yield f"data: {json.dumps(final_data)}\n\n"
        
        # Send callback if provided (non-blocking, after stream completes)
        # NOTE: Callback receives the FULL accumulated response, not individual chunks
        # This happens AFTER all chunks have been streamed to the client
        if request.callback_url:
            log.info(f"📤 Sending callbacks to {request.callback_url} | uid={request.uid}")
            log.info(f"   Response preview: {full_response[:150]}...")

            # 1) First callback: response only (type=response_complete)
            asyncio.create_task(_send_streaming_callback(
                request.callback_url,
                request.uid,
                session_id,
                full_response,
                suggested_topics,
                suggested_questions,
                action_items,
            ))

            # 2) Second callback: tool outputs only (type=tool_outputs), same URL
            if tool_results:
                assessment_plan = None
                assessment_needs = None
                if "get_assessment_recommendations" in tool_results:
                    assessment_result = tool_results["get_assessment_recommendations"][0]["result"]
                    if isinstance(assessment_result, dict) and assessment_result.get("success"):
                        assessment_plan = assessment_result.get("assessment_plan")
                        assessment_needs = assessment_result.get("assessment_needs")
                log.info(f"📦 Sending tool_outputs callback for {len(tool_results)} tool(s): {list(tool_results.keys())}")
                asyncio.create_task(_send_tool_outputs_callback(
                    request.callback_url,
                    request.uid,
                    session_id,
                    tool_results,
                    assessment_plan,
                    assessment_needs,
                ))
        else:
            log.info(f"Callback skipped: no callback_url in request (streaming response sent to client) | uid={request.uid}")
        
    except Exception as e:
        log.error(f"Streaming error: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


# ============================================================================
# MAIN AGENT FUNCTION (Non-streaming fallback)
# ============================================================================

@traceable(name="career_chatbot_agent")
async def career_chatbot_agent(
    request: CareerChatRequest,
    tenant_id: str = "default_tenant"
) -> CareerChatResponse:
    """
    Career chatbot agent - provides conversational career guidance.
    Non-streaming version (fallback).
    """
    log_context = create_log_context("career_chatbot", tenant_id)
    start_time = log_context["start_time"]
    
    uid = request.uid
    user_message = sanitize_text_for_llm(request.message)
    # CRITICAL: Generate unique session_id per user to ensure isolation
    # Format: {uid}_{timestamp}_{random} ensures uniqueness even for concurrent requests
    session_id = request.session_id or f"{uid}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    log.info(f"Career chatbot request (non-streaming) | uid={uid} | session_id={session_id}")
    log.info(f"📝 User Message: {user_message[:200]}{'...' if len(user_message) > 200 else ''}")
    
    try:
        # Load context and prior-session history in parallel (current-session history from context)
        context, prior = await asyncio.gather(
            load_comprehensive_context(uid, session_id),
            _load_prior_conversation_context(uid, session_id, max_messages=20),
        )
        current = context.get("career_chat_history") or []
        if not isinstance(current, list):
            current = []
        log.info(f"📚 Loaded conversation history: {len(current)} current, {len(prior)} prior-session messages")
        if current:
            recent_messages = current[-3:]
            for msg in recent_messages:
                role = msg.get("role", "unknown")
                content_preview = msg.get("content", "")[:100]
                log.debug(f"  - {role}: {content_preview}{'...' if len(msg.get('content', '')) > 100 else ''}")
        
        # Add user message to current (what we persist)
        current.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.utcnow().isoformat()
        })
        context_for_prompt = prior + list(current)
        
        # Extract aspirations and goals from conversation (async, non-blocking)
        try:
            # Issue 8.3: Parallelize independent LLM calls for better latency
            aspirations_task = extract_aspirations_from_conversation(context_for_prompt, uid)
            goals_task = extract_goals_from_message(user_message, uid)
            
            aspirations, goals_from_message = await asyncio.gather(
                aspirations_task,
                goals_task,
                return_exceptions=True
            )
            
            # Handle exceptions from gather
            if isinstance(aspirations, Exception):
                log.warning(f"Failed to extract aspirations: {aspirations}")
                aspirations = {}
            if isinstance(goals_from_message, Exception):
                log.warning(f"Failed to extract goals: {goals_from_message}")
                goals_from_message = []
            
            # Store aspirations and goals in session (if we have new data)
            if aspirations.get("career_goals") or aspirations.get("aspirations") or goals_from_message:
                try:
                    from chroma import get_chat_session, update_chat_session
                    session_data = await run_blocking_io(get_chat_session, session_id, uid) or {}
                    
                    # Update aspirations
                    if aspirations.get("career_goals") or aspirations.get("aspirations"):
                        existing_aspirations = session_data.get("career_aspirations", {})
                        # Merge new aspirations with existing
                        merged_aspirations = {
                            "career_goals": list(set(existing_aspirations.get("career_goals", []) + aspirations.get("career_goals", [])))[:10],
                            "aspirations": list(set(existing_aspirations.get("aspirations", []) + aspirations.get("aspirations", [])))[:10],
                            "skills_of_interest": list(set(existing_aspirations.get("skills_of_interest", []) + aspirations.get("skills_of_interest", [])))[:10],
                            "career_paths_of_interest": list(set(existing_aspirations.get("career_paths_of_interest", []) + aspirations.get("career_paths_of_interest", [])))[:10],
                            "timeline": aspirations.get("timeline") or existing_aspirations.get("timeline"),
                            "confidence": max(aspirations.get("confidence", 0.0), existing_aspirations.get("confidence", 0.0))
                        }
                        session_data["career_aspirations"] = merged_aspirations
                        context["aspirations"] = merged_aspirations
                    
                    # Add new goals from message
                    if goals_from_message:
                        existing_goals = session_data.get("career_goals", [])
                        if not isinstance(existing_goals, list):
                            existing_goals = []
                        # Add new goals (avoid duplicates)
                        existing_goal_titles = {g.get("title", "").lower() for g in existing_goals if isinstance(g, dict)}
                        for new_goal in goals_from_message:
                            if isinstance(new_goal, dict) and new_goal.get("title", "").lower() not in existing_goal_titles:
                                new_goal["goal_id"] = f"goal_{uid}_{int(time.time() * 1000)}"
                                new_goal["uid"] = uid
                                new_goal["status"] = "active"
                                new_goal["created_at"] = datetime.utcnow().isoformat()
                                existing_goals.append(new_goal)
                                existing_goal_titles.add(new_goal.get("title", "").lower())
                        session_data["career_goals"] = existing_goals
                        context["career_goals"] = existing_goals
                    
                    # Update conversation insights
                    # Update conversation insights (build profile over time)
                    existing_profile = session_data.get("conversation_insights", {})
                    conversation_insights = await analyze_conversation_patterns(
                        context_for_prompt, 
                        uid,
                        existing_profile=existing_profile  # Merge with existing profile
                    )
                    # Add timestamp
                    # from datetime import datetime
                    conversation_insights["last_analyzed"] = datetime.utcnow().isoformat()
                    session_data["conversation_insights"] = conversation_insights
                    context["conversation_insights"] = conversation_insights
                    
                    # Save updated session
                    await run_blocking_io(update_chat_session, session_id, session_data)
                    log.info(f"💾 Updated session with aspirations and goals for uid={uid}")
                except Exception as e:
                    log.warning(f"Could not store aspirations/goals in session: {e}")
        except Exception as e:
            log.warning(f"Error extracting aspirations/goals: {e}")
        
        # Build prompt with all context
        prompt = _build_career_chat_prompt(user_message, context, context_for_prompt)
        
        # Call LLM (non-streaming)
        from models.llm_invoker import invoke_llm
        llm_response = await invoke_llm(
            prompt=prompt,
            task_type=TaskType.TEXT_GENERATION,
            agent_name="career_chatbot",
            response_mime_type=None,
            preferred_model=LLM_MODEL,
            max_output_tokens=MAX_RESPONSE_LENGTH
        )
        
        # Parse response
        response_text = _to_text(llm_response)
        
        # Log assistant response
        log.info(f"🤖 Assistant Response: {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
        log.info(f"📊 Response length: {len(response_text)} characters")
        
        # Extract structured data
        suggested_topics = []  # Deprecated
        suggested_questions = []  # NEW
        action_items = []
        try:
            json_start = response_text.rfind('{')
            if json_start > 0:
                json_text = response_text[json_start:]
                json_end = json_text.rfind('}') + 1
                if json_end > 0:
                    parsed = json.loads(json_text[:json_end])
                    # Support both old and new format
                    suggested_questions = parsed.get("suggested_questions", [])
                    suggested_topics = parsed.get("suggested_topics", [])  # Fallback
                    action_items = parsed.get("action_items", [])
                    response_text = response_text[:json_start].strip()
        except Exception:
            pass
        
        # Generate questions if LLM didn't provide them (fallback)
        if not suggested_questions:
            suggested_questions = await _generate_contextual_questions(
                user_message=user_message,
                conversation_history=context_for_prompt,
                context=context,
                assistant_response=response_text
            )
        
        # Add mentor response to current-session history (prior never persisted)
        current.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Save conversation history (reusing interview agent pattern)
        await _save_conversation_history(session_id, current, uid)
        
        processing_time = time.time() - start_time
        
        # Log conversation summary
        log.info(f"💬 Conversation Summary | session_id={session_id} | history_length={len(current)} messages")
        log.info(f"❓ Suggested Questions: {len(suggested_questions)} questions")
        if suggested_questions:
            for i, q in enumerate(suggested_questions[:3], 1):
                log.info(f"   {i}. {q[:80]}{'...' if len(q) > 80 else ''}")
        log.info(f"✅ Action Items: {len(action_items)} items")
        
        log_agent_completion(log_context, {
            "success": True,
            "session_id": session_id,
            "response_length": len(response_text),
            "suggested_questions_count": len(suggested_questions),
            "suggested_topics_count": len(suggested_topics),  # Deprecated
            "action_items_count": len(action_items)
        }, "llm", processing_time)
        
        return CareerChatResponse(
            response=response_text,
            session_id=session_id,
            conversation_history=current,
            suggested_questions=suggested_questions,  # NEW
            suggested_topics=suggested_topics,  # Keep for backward compatibility
            action_items=action_items,
            confidence_score=0.8
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        log.error(f"Career chatbot failed: {e}", exc_info=True)
        
        error_response = "I apologize, but I encountered an error. Please try again."
        
        return CareerChatResponse(
            response=error_response,
            session_id=session_id,
            conversation_history=current if 'current' in locals() else [],
            confidence_score=0.0
        )


# ============================================================================
# STREAMING AGENT FUNCTION
# ============================================================================

@traceable(name="career_chatbot_agent_stream")
async def career_chatbot_agent_stream(
    request: CareerChatRequest,
    tenant_id: str = "default_tenant"
) -> StreamingResponse:
    """
    Career chatbot agent with streaming support.
    
    Returns:
        StreamingResponse: Server-Sent Events (SSE) stream
    """
    log_context = create_log_context("career_chatbot_stream", tenant_id)
    start_time = log_context["start_time"]
    
    uid = request.uid
    user_message = sanitize_text_for_llm(request.message)
    # CRITICAL: Generate unique session_id per user to ensure isolation
    # Format: {uid}_{timestamp}_{random} ensures uniqueness even for concurrent requests
    session_id = request.session_id or f"{uid}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    log.info(f"Career chatbot request (streaming) | uid={uid} | session_id={session_id}")
    log.info(f"📝 User Message: {user_message[:200]}{'...' if len(user_message) > 200 else ''}")
    
    try:
        # Load context and prior-session history in parallel (current-session history from context)
        context, prior = await asyncio.gather(
            load_comprehensive_context(uid, session_id),
            _load_prior_conversation_context(uid, session_id, max_messages=20),
        )
        current = context.get("career_chat_history") or []
        if not isinstance(current, list):
            current = []
        log.info(f"📚 Loaded conversation history: {len(current)} current, {len(prior)} prior-session messages")
        if current:
            recent_messages = current[-3:]
            for msg in recent_messages:
                role = msg.get("role", "unknown")
                content_preview = msg.get("content", "")[:100]
                log.debug(f"  - {role}: {content_preview}{'...' if len(msg.get('content', '')) > 100 else ''}")
        
        # Add user message to current (what we persist)
        current.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.utcnow().isoformat()
        })
        # Context for prompt includes prior + current (prior never persisted to this session)
        context_for_prompt = prior + list(current)
        
        # Generate streaming response
        stream_generator = _generate_streaming_response(
            request, context, context_for_prompt, history_to_save=current
        )
        
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        log.error(f"Streaming setup failed: {e}", exc_info=True)
        
        # Return error as SSE
        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream"
        )


# Alias for backward compatibility
conversational_mentor_agent = career_chatbot_agent


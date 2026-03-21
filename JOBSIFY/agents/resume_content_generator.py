"""
Resume Content Generator Agent

This agent generates a complete, ATS-friendly, well-formatted resume content in Markdown.
It considers all context received throughout other agent calls including:
- Structured resume data (REQUIRED - minimum)
- User interests (OPTIONAL)
- Career advisor recommendations (OPTIONAL)
- Market and course recommendations (OPTIONAL)
- Assessment recommendations (OPTIONAL)
- Interview insights (OPTIONAL)
- Skill analysis (OPTIONAL)

The agent will work with just the structured resume if that's all that's available.
"""

from typing import Dict, Any, Optional, List
from langsmith.run_helpers import traceable
import logging
import json
from chroma import get_resume_doc, get_chat_session, fetch_structured_resume, get_resume
from utils.session_manager import session_manager
from core.utils import run_blocking_io, _calculate_processing_time, _create_error_response
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion
from core.config import get_agent_config
from core.memory import BaseAgentMemory, get_agent_memory
from models.llm_invoker import invoke_llm

# Get centralized configuration
config = get_agent_config("resume_content_generator")

# Custom memory class for resume content generator
class ResumeContentGeneratorMemory(BaseAgentMemory):
    def __init__(self, tenant_id: str = "default"):
        super().__init__(tenant_id, config.adaptation_window)

# Use centralized memory management
async def get_resume_content_generator_memory(tenant_id: str = "default") -> ResumeContentGeneratorMemory:
    """Get or create tenant-scoped resume content generator memory."""
    return await get_agent_memory("resume_content_generator", tenant_id, ResumeContentGeneratorMemory)

log = logging.getLogger(__name__)

# Thresholds for improvement priorities (scores 0-100)
WEAK_AREA_THRESHOLD = 70
STRONG_AREA_THRESHOLD = 80


def _normalize_resume_score_for_context(score_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize resume score data from various formats (ResumeScore, resumeScore, session format)
    into a unified structure for the content generator.
    Returns: {total, breakdown, rationale, breakdown_rationale, breakdown_how_to_improve,
              improvement_suggestions, strength_areas, weak_areas, strong_areas} or None
    """
    if not score_data or not isinstance(score_data, dict):
        return None

    result = {
        "total": None,
        "breakdown": {},
        "rationale": "",
        "breakdown_rationale": {},
        "breakdown_how_to_improve": {},
        "improvement_suggestions": [],
        "strength_areas": [],
        "weak_areas": [],
        "strong_areas": [],
    }

    # Handle resumeScore format (breakdown, total, rationale, breakdown_rationale, breakdown_how_to_improve)
    resume_score = score_data.get("resumeScore") or score_data
    if isinstance(resume_score, dict):
        result["total"] = resume_score.get("total") or resume_score.get("total_score") or resume_score.get("score")
        result["rationale"] = (resume_score.get("rationale") or resume_score.get("explanation") or "")[:2500]
        bd = resume_score.get("breakdown", {})
        if bd:
            result["breakdown"] = dict(bd)
        br = resume_score.get("breakdown_rationale")
        if isinstance(br, dict):
            result["breakdown_rationale"] = {k: (v[:350] if isinstance(v, str) else "") for k, v in br.items()}
        bhi = resume_score.get("breakdown_how_to_improve")
        if isinstance(bhi, dict):
            result["breakdown_how_to_improve"] = {k: (v[:300] if isinstance(v, str) else "") for k, v in bhi.items() if k in ("skills", "presentation")}

    # Handle ResumeScore format (SkillsScore, FormatScore only; Experience and Education removed)
    old_score = score_data.get("ResumeScore", {})
    if isinstance(old_score, dict) and not result["breakdown"]:
        cat_map = {
            "SkillsScore": "skills",
            "FormatScore": "presentation",
        }
        for old_cat, new_cat in cat_map.items():
            v = old_score.get(old_cat)
            if v is not None:
                try:
                    result["breakdown"][new_cat] = float(v)
                except (TypeError, ValueError):
                    pass
        if result["total"] is None:
            result["total"] = old_score.get("OverallScore")
        if not result["rationale"]:
            result["rationale"] = (old_score.get("Explanation") or old_score.get("Reasoning") or "")[:2500]
        # Copy per-category rationales from ResumeScore into breakdown_rationale (Skills + Format only)
        if not result["breakdown_rationale"]:
            sk_r = (old_score.get("SkillsRationale") or "").strip()[:350]
            fmt_r = (old_score.get("FormatRationale") or "").strip()[:350]
            if sk_r or fmt_r:
                result["breakdown_rationale"] = {"skills": sk_r, "presentation": fmt_r}
        # Copy per-category how-to-improve from ResumeScore into breakdown_how_to_improve
        if not result["breakdown_how_to_improve"]:
            sk_how = (old_score.get("SkillsHowToImprove") or "").strip()[:300]
            fmt_how = (old_score.get("FormatHowToImprove") or "").strip()[:300]
            if sk_how or fmt_how:
                result["breakdown_how_to_improve"] = {"skills": sk_how, "presentation": fmt_how}

    # Improvement suggestions (ImprovementSuggestions or improvement_suggestions)
    sugg = score_data.get("ImprovementSuggestions") or score_data.get("improvement_suggestions") or []
    if isinstance(sugg, list):
        result["improvement_suggestions"] = [str(s).strip()[:300] for s in sugg[:6] if s]

    # Strength areas (StrengthAreas, strength_areas, or strengths)
    str_areas = score_data.get("StrengthAreas") or score_data.get("strength_areas") or score_data.get("strengths") or []
    if isinstance(str_areas, list):
        result["strength_areas"] = [str(s).strip()[:200] for s in str_areas[:5] if s]

    # Derive weak/strong from breakdown
    if result["breakdown"]:
        for area, score in result["breakdown"].items():
            try:
                s = float(score)
                if s < WEAK_AREA_THRESHOLD:
                    result["weak_areas"].append(f"{area}: {s}/100")
                elif s >= STRONG_AREA_THRESHOLD:
                    result["strong_areas"].append(f"{area}: {s}/100")
            except (TypeError, ValueError):
                pass

    # Only return if we have at least total or breakdown
    if result["total"] is not None or result["breakdown"]:
        return result
    return None


def _build_context_summary(context_data: Dict[str, Any]) -> str:
    """Build a comprehensive context summary from all agent outputs. All fields are optional."""
    summary_parts = []
    
    # Structured Resume (this should always be present, but handle gracefully)
    structured_resume = context_data.get("structured_resume", {})
    if structured_resume and structured_resume.get("Name"):
        summary_parts.append("=== RESUME DATA ===")
        summary_parts.append(f"Name: {structured_resume.get('Name', 'N/A')}")
        if structured_resume.get('total_experience_years'):
            summary_parts.append(f"Total Experience: {structured_resume.get('total_experience_years', 0)} years")
        if structured_resume.get('education'):
            summary_parts.append(f"Education: {len(structured_resume.get('education', []))} entries")
        if structured_resume.get('experience'):
            summary_parts.append(f"Work Experience: {len(structured_resume.get('experience', []))} entries")
        if structured_resume.get('skills'):
            summary_parts.append(f"Skills: {len(structured_resume.get('skills', []))} skills")
        summary_parts.append("")
    
    # User Interests (OPTIONAL)
    user_interests = context_data.get("user_interests", [])
    if user_interests and isinstance(user_interests, list) and len(user_interests) > 0:
        summary_parts.append("=== USER INTERESTS ===")
        for interest in user_interests[:5]:  # Limit to top 5
            if isinstance(interest, dict):
                interest_text = interest.get('interest') or interest.get('topic') or interest.get('name')
                if interest_text:
                    summary_parts.append(f"- {interest_text}")
        if summary_parts[-1] == "=== USER INTERESTS ===":
            summary_parts.pop()  # Remove header if no interests added
        else:
            summary_parts.append("")
    
    # Career Advisor Insights (OPTIONAL)
    career_advisor = context_data.get("career_advisor", {})
    if career_advisor and isinstance(career_advisor, dict):
        recommendations = career_advisor.get("recommendations", [])
        if recommendations and isinstance(recommendations, list) and len(recommendations) > 0:
            summary_parts.append("=== CAREER INSIGHTS ===")
            summary_parts.append("Career Recommendations:")
            for rec in recommendations[:3]:
                if isinstance(rec, dict):
                    rec_text = rec.get('recommendation') or rec.get('suggestion') or rec.get('advice')
                    if rec_text:
                        summary_parts.append(f"- {rec_text}")
            summary_parts.append("")
    
    # Market and Course Recommendations (OPTIONAL)
    market_course = context_data.get("market_and_course_recommender", {})
    if market_course and isinstance(market_course, dict):
        # Check for courses (stored as "course_recommendations" in gap_doc, or "recommended_courses" in session)
        courses = market_course.get("course_recommendations") or market_course.get("recommended_courses", [])
        market_insights = market_course.get("market_insights", [])
        
        if (courses and isinstance(courses, list) and len(courses) > 0) or (market_insights and isinstance(market_insights, list) and len(market_insights) > 0):
            summary_parts.append("=== MARKET & COURSE INSIGHTS ===")
            
            if market_insights and isinstance(market_insights, list) and len(market_insights) > 0:
                summary_parts.append("Market Insights:")
                for insight in market_insights[:3]:
                    if isinstance(insight, dict):
                        insight_text = insight.get('insight') or insight.get('text') or insight.get('summary')
                        if insight_text:
                            summary_parts.append(f"- {insight_text}")
                    elif isinstance(insight, str):
                        summary_parts.append(f"- {insight}")
                summary_parts.append("")
            
            if courses and isinstance(courses, list) and len(courses) > 0:
                summary_parts.append("Recommended Courses:")
                for course in courses[:3]:
                    if isinstance(course, dict):
                        course_text = course.get('course_name') or course.get('name') or course.get('title') or course.get('course_title')
                        if course_text:
                            summary_parts.append(f"- {course_text}")
                summary_parts.append("")
    
    # Assessment Evaluator Results (OPTIONAL) - actual assessment scores
    assessment_evaluator = context_data.get("assessment_evaluator", {})
    if assessment_evaluator and isinstance(assessment_evaluator, dict):
        assessment_results = assessment_evaluator.get("assessment_results", {})
        if assessment_results and isinstance(assessment_results, dict):
            total_score = assessment_results.get("total_score")
            max_score = assessment_results.get("max_score")
            section_scores = assessment_results.get("section_scores", {})
            if total_score is not None or section_scores:
                summary_parts.append("=== ASSESSMENT RESULTS ===")
                if total_score is not None and max_score is not None:
                    summary_parts.append(f"Overall Score: {total_score}/{max_score}")
                if section_scores:
                    summary_parts.append("Section Scores:")
                    for section, score in list(section_scores.items())[:3]:
                        summary_parts.append(f"- {section}: {score}")
                summary_parts.append("")
    
    # Report Generator (OPTIONAL) - detailed assessment report
    report = context_data.get("report", {})
    if report and isinstance(report, dict):
        summary = report.get("summary")
        feedback = report.get("feedback")
        performance_level = report.get("performance_level")
        suggested_next_steps = report.get("suggested_next_steps", [])
        topics = report.get("topics", [])
        
        if summary or feedback or performance_level:
            summary_parts.append("=== ASSESSMENT REPORT ===")
            if summary:
                summary_parts.append(f"Summary: {summary}")
            if performance_level:
                summary_parts.append(f"Performance Level: {performance_level}")
            if feedback:
                summary_parts.append(f"Feedback: {feedback}")
            if topics and isinstance(topics, list) and len(topics) > 0:
                summary_parts.append("Key Topics:")
                for topic in topics[:3]:
                    if isinstance(topic, dict):
                        topic_name = topic.get("topic_name") or topic.get("name") or topic.get("topic")
                        if topic_name:
                            summary_parts.append(f"- {topic_name}")
            if suggested_next_steps and isinstance(suggested_next_steps, list) and len(suggested_next_steps) > 0:
                summary_parts.append("Suggested Next Steps:")
                for step in suggested_next_steps[:3]:
                    if isinstance(step, str) and step.strip():
                        summary_parts.append(f"- {step.strip()}")
            summary_parts.append("")
    
    # Interview Evaluation (OPTIONAL)
    interview_evaluation = context_data.get("interview_evaluation", {})
    if interview_evaluation and isinstance(interview_evaluation, dict):
        overall_score = interview_evaluation.get("overall_score")
        strengths = interview_evaluation.get("strengths", [])
        areas_for_improvement = interview_evaluation.get("areas_for_improvement", [])
        if overall_score is not None or strengths or areas_for_improvement:
            summary_parts.append("=== INTERVIEW INSIGHTS ===")
            if overall_score is not None:
                summary_parts.append(f"Overall Interview Score: {overall_score}/100")
            if strengths:
                summary_parts.append("Key Strengths:")
                for strength in strengths[:3]:
                    if strength:
                        summary_parts.append(f"- {strength}")
            if areas_for_improvement:
                summary_parts.append("Areas for Improvement:")
                for area in areas_for_improvement[:3]:
                    if area:
                        summary_parts.append(f"- {area}")
            summary_parts.append("")
    
    # Resume Score/Analysis (OPTIONAL) - normalized for improvement-driven content generation
    resume_score_raw = context_data.get("resume_score", {})
    resume_score = _normalize_resume_score_for_context(resume_score_raw)
    if resume_score:
        summary_parts.append("=== RESUME SCORE & IMPROVEMENT PRIORITIES ===")
        if resume_score.get("total") is not None:
            summary_parts.append(f"Overall Score: {resume_score['total']}/100")
        if resume_score.get("breakdown"):
            summary_parts.append("Category Scores:")
            for area, score in resume_score["breakdown"].items():
                summary_parts.append(f"- {area}: {score}/100")
        if resume_score.get("rationale"):
            summary_parts.append(f"Scorer Feedback: {resume_score['rationale']}")
        if resume_score.get("weak_areas"):
            summary_parts.append("PRIORITY: Weak Areas to Strengthen (focus improvement here):")
            for w in resume_score["weak_areas"]:
                summary_parts.append(f"- {w}")
        if resume_score.get("improvement_suggestions"):
            summary_parts.append("Improvement Suggestions (implement these):")
            for s in resume_score["improvement_suggestions"]:
                summary_parts.append(f"- {s}")
        if resume_score.get("strong_areas"):
            summary_parts.append("Strength Areas (preserve and emphasize):")
            for s in resume_score["strong_areas"]:
                summary_parts.append(f"- {s}")
        if resume_score.get("strength_areas"):
            summary_parts.append("Key Strengths:")
            for s in resume_score["strength_areas"]:
                summary_parts.append(f"- {s}")
        summary_parts.append("")
    
    # If no additional context was found, return empty string
    if len(summary_parts) == 0 or (len(summary_parts) == 1 and summary_parts[0].startswith("===")):
        return ""
    
    return "\n".join(summary_parts)


@traceable(name="resume_content_generator_agent")
async def resume_content_generator_agent(
    uid: str,
    tenant_id: str = "default_tenant",
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a complete plaintext resume considering all context from previous agent calls.
    Works with minimal data - only requires structured resume. All other context is optional.
    
    Args:
        uid: User ID (REQUIRED)
        tenant_id: Tenant ID (OPTIONAL, defaults to "default_tenant")
        session_id: Optional session ID to retrieve session-specific context (OPTIONAL)
        
    Returns:
        Dict with 'resume_content' (plaintext) and metadata
    """
    log_context = create_log_context("resume_content_generator", tenant_id)
    start_time = log_context["start_time"]
    request_id = log_context["request_id"]
    
    log.info(f"--- Entering Resume Content Generator Agent --- uid={uid}")
    
    # Get tenant-scoped memory
    generator_memory = await get_resume_content_generator_memory(tenant_id)
    
    try:
        # Step 1: Collect all context from previous agent calls (all optional except structured resume)
        context_data = {}
        
        # Get structured resume (REQUIRED - but handle gracefully if missing)
        # PRIORITY: Try to get the FULL resume from the resume collection first
        # This contains all sections (experience, education, skills, certifications, projects, etc.)
        # without truncation, unlike the lite version stored in chat_sessions
        structured_resume = {}
        full_resume = None  # Track if we got the full resume
        try:
            # Step 1: Get full resume from chat_sessions (uid_resume; chunked if >16KB)
            # get_resume() resolves from chat_sessions first; full structured_resume is stored there by resume_assembler
            full_resume = await run_blocking_io(get_resume, uid)
            if full_resume and isinstance(full_resume, dict):
                # get_resume returns the structured_resume directly (not wrapped in a doc)
                # Check if it has the expected structure (Name or name field)
                if full_resume.get("Name") or full_resume.get("name"):
                    structured_resume = full_resume
                    log.info(f"✅ Retrieved FULL structured resume from resume collection for uid={uid}")
                    log.info(f"📊 Full resume sections: Name={bool(structured_resume.get('Name') or structured_resume.get('name'))}, "
                             f"Education={len(structured_resume.get('education', []))} entries, "
                             f"Experience={len(structured_resume.get('experience', []))} entries, "
                             f"Skills={len(structured_resume.get('skills', []))} entries, "
                             f"Certifications={len(structured_resume.get('certifications', []))} entries, "
                             f"Projects={len(structured_resume.get('projects', []))} entries")
                else:
                    full_resume = None  # Reset if it doesn't have valid structure
            
            # Step 2: Fall back to lite resume from chat_sessions if full resume not available
            if not structured_resume or not (structured_resume.get("Name") or structured_resume.get("name")):
                log.debug(f"Full resume not found, trying lite resume from chat_sessions for uid={uid}")
                resume_doc = await run_blocking_io(get_resume_doc, uid) or {}
                structured_resume = resume_doc.get("structured_resume") or {}
                
                if structured_resume and (structured_resume.get("Name") or structured_resume.get("name")):
                    log.info(f"✅ Retrieved lite structured resume from chat_sessions for uid={uid}")
                    log.info(f"📊 Lite resume sections: Name={bool(structured_resume.get('Name'))}, "
                             f"Education={len(structured_resume.get('education', []))} entries, "
                             f"Experience={len(structured_resume.get('experience', []))} entries, "
                             f"Skills={len(structured_resume.get('skills', []))} entries")
            
            # Step 3: Try fetching from session if still not found
            if not structured_resume or not (structured_resume.get("Name") or structured_resume.get("name")):
                # Try fetching from session
                if session_id:
                    try:
                        structured_resume = await run_blocking_io(fetch_structured_resume, uid, None, session_id) or {}
                        if structured_resume and (structured_resume.get("Name") or structured_resume.get("name")):
                            log.info(f"✅ Retrieved structured resume from session {session_id}")
                    except Exception as e:
                        log.debug(f"Could not fetch from session: {e}")
                
                # If still no structured resume, try to get from any available session
                if not structured_resume or not (structured_resume.get("Name") or structured_resume.get("name")):
                    try:
                        from chroma import find_session_by_uid
                        found_session_id = await run_blocking_io(find_session_by_uid, uid)
                        if found_session_id:
                            structured_resume = await run_blocking_io(fetch_structured_resume, uid, None, found_session_id) or {}
                            if structured_resume and (structured_resume.get("Name") or structured_resume.get("name")):
                                log.info(f"✅ Retrieved structured resume from found session {found_session_id}")
                    except Exception as e:
                        log.debug(f"Could not find session: {e}")
            
            context_data["structured_resume"] = structured_resume
            if structured_resume and (structured_resume.get("Name") or structured_resume.get("name")):
                # Log which source was used
                if full_resume and isinstance(full_resume, dict) and (full_resume.get("Name") or full_resume.get("name")):
                    log.info(f"✅ Using FULL structured resume (all sections included) for uid={uid}")
                else:
                    log.info(f"✅ Using lite/fallback structured resume for uid={uid}")
                
                # Check if user_interests_summary is in structured_resume (for existing users)
                user_interests_summary = structured_resume.get("user_interests_summary")
                if user_interests_summary and isinstance(user_interests_summary, str) and user_interests_summary.strip():
                    # Convert summary string back to list format for consistency
                    # user_interests_summary is stored as "answer1; answer2; answer3"
                    if not context_data.get("user_interests"):
                        interests_list = [{"answer": item.strip()} for item in user_interests_summary.split(";") if item.strip()]
                        if interests_list:
                            context_data["user_interests"] = interests_list
                            log.info(f"✅ Retrieved {len(interests_list)} user_interests from structured_resume.user_interests_summary for uid={uid}")
            else:
                log.warning(f"⚠️ No structured resume found for uid={uid}")
        except Exception as e:
            log.warning(f"⚠️ Could not retrieve structured resume: {e}")
            context_data["structured_resume"] = {}
        
        # Validate that we have at least basic resume data
        if not structured_resume or not (structured_resume.get("Name") or structured_resume.get("name")):
            processing_time = _calculate_processing_time(start_time)
            AgentLogger.log_warning(log_context, "No structured resume data found")
            return _create_error_response(
                "No structured resume data found. Please ensure resume has been processed first.", 
                processing_time
            )
        
        # Load optional uploaded_certificates from resume_doc and merge into certifications
        uploaded_certificates: List[Dict[str, Any]] = []
        try:
            resume_doc = await run_blocking_io(get_resume_doc, uid) or {}
            uploaded_certificates = (resume_doc.get("uploaded_certificates") or []) if isinstance(resume_doc, dict) else []
            if uploaded_certificates and isinstance(uploaded_certificates, list):
                def _normalize_cert(c: Any) -> Dict[str, Any]:
                    if not isinstance(c, dict):
                        return {"certification_name": str(c)[:500], "issuing_organization": "", "year": None}
                    return {
                        "certification_name": c.get("certification_name") or c.get("name") or c.get("title") or "",
                        "issuing_organization": c.get("issuing_organization") or c.get("issuer") or "",
                        "year": c.get("year") or c.get("date"),
                    }
                normalized = [_normalize_cert(item) for item in uploaded_certificates if _normalize_cert(item).get("certification_name")]
                if normalized:
                    existing = structured_resume.get("certifications") or []
                    if not isinstance(existing, list):
                        existing = []
                    structured_resume["certifications"] = existing + normalized
                    log.info(f"✅ Merged {len(normalized)} uploaded certificate(s) into resume for uid={uid}")
        except Exception as e:
            log.debug(f"Could not load uploaded_certificates from resume_doc: {e}")
        
        # Get session data (OPTIONAL - contains all agent outputs)
        session_data = None
        if session_id:
            try:
                session_data = await run_blocking_io(get_chat_session, session_id, uid)
                if session_data:
                    log.info(f"✅ Retrieved session data for session_id={session_id}")
            except Exception as e:
                log.debug(f"Could not retrieve session data: {e}")
        
        # If no session_id provided, try to find the most recent session (OPTIONAL)
        if not session_data:
            try:
                from chroma import find_session_by_uid
                found_session_id = await run_blocking_io(find_session_by_uid, uid)
                if found_session_id:
                    session_data = await run_blocking_io(get_chat_session, found_session_id, uid)
                    if session_data:
                        log.info(f"✅ Retrieved most recent session data for uid={uid}")
            except Exception as e:
                log.debug(f"Could not find session by uid: {e}")
        
        # Extract agent outputs from session data (all optional)
        if session_data:
            try:
                # User interests (OPTIONAL)
                interest_filler = session_data.get("interest_filler", {})
                if interest_filler:
                    context_data["user_interests"] = interest_filler.get("user_interests", [])
                
                # Career advisor (OPTIONAL)
                if "career_advisor" in session_data:
                    context_data["career_advisor"] = session_data.get("career_advisor", {})
                
                # Market and course recommender (OPTIONAL)
                if "market_and_course_recommender" in session_data:
                    context_data["market_and_course_recommender"] = session_data.get("market_and_course_recommender", {})
                
                # Resume score (OPTIONAL)
                if "resume_score" in session_data:
                    context_data["resume_score"] = session_data.get("resume_score", {})
                
                # Resume analysis (OPTIONAL)
                if "resume_analysis" in session_data:
                    context_data["resume_analysis"] = session_data.get("resume_analysis", {})
                
                # Report generator (OPTIONAL) - detailed assessment report
                if "report_generator" in session_data:
                    report_generator_data = session_data.get("report_generator", {})
                    if report_generator_data and isinstance(report_generator_data, dict) and len(report_generator_data) > 0:
                        report = report_generator_data.get("report")
                        if report and isinstance(report, dict) and len(report) > 0:
                            context_data["report"] = report
                            log.info(f"✅ Retrieved report from report_generator for uid={uid}")
                
                # Assessment evaluator results (OPTIONAL) - actual assessment scores/results
                if "assessment_evaluator" in session_data:
                    assessment_evaluator_data = session_data.get("assessment_evaluator", {})
                    if assessment_evaluator_data:
                        context_data["assessment_evaluator"] = assessment_evaluator_data
                        log.info(f"✅ Retrieved assessment evaluator results for uid={uid}")
            except Exception as e:
                log.debug(f"Error extracting session data: {e}")
        
        # Also try getting from assessments_doc directly (OPTIONAL) - assessment data stored here
        try:
            from chroma import get_assessments_doc
            assessments_doc = await run_blocking_io(get_assessments_doc, uid) or {}
            log.info(f"🔍 Checking assessments_doc for uid={uid}, keys present: {list(assessments_doc.keys()) if assessments_doc else 'empty'}")
            if assessments_doc:
                # Check for assessment_history (contains past assessment results)
                assessment_history = assessments_doc.get("assessment_history", [])
                if assessment_history and isinstance(assessment_history, list) and len(assessment_history) > 0:
                    # Use the most recent assessment result if available
                    latest_assessment = assessment_history[-1] if assessment_history else {}
                    if latest_assessment and isinstance(latest_assessment, dict):
                        # Extract assessment results from history
                        if not context_data.get("assessment_evaluator"):
                            # Try to construct assessment_evaluator format from history
                            result = latest_assessment.get("result", {})
                            if result and isinstance(result, dict):
                                context_data["assessment_evaluator"] = {
                                    "assessment_results": result
                                }
                                log.info(f"✅ Retrieved assessment results from assessment_history for uid={uid}")
        except Exception as e:
            log.warning(f"⚠️ Could not get assessments_doc: {e}")
            import traceback
            log.debug(traceback.format_exc())
        
        # Prefer dedicated uid_resume_score document when resume_score not in session
        if not context_data.get("resume_score") and uid:
            try:
                from chroma import get_resume_score_doc
                resume_score_doc = await run_blocking_io(get_resume_score_doc, uid)
                if resume_score_doc and isinstance(resume_score_doc, dict) and len(resume_score_doc) > 0:
                    context_data["resume_score"] = resume_score_doc
                    log.info(f"✅ Retrieved resume_score from uid_resume_score for uid={uid}")
            except Exception as e:
                log.debug(f"Could not get resume_score_doc: {e}")
        
        # Also try getting from gap_doc directly (OPTIONAL) - many agents store data here
        try:
            from chroma import get_gap_doc
            gap_doc = await run_blocking_io(get_gap_doc, uid) or {}
            log.info(f"🔍 Checking gap_doc for uid={uid}, keys present: {list(gap_doc.keys()) if gap_doc else 'empty'}")
            if gap_doc:
                # Check for career_advisor (stored as "career_advisor" or "skill_and_career_advisor")
                if not context_data.get("career_advisor"):
                    career_advisor = gap_doc.get("career_advisor") or gap_doc.get("skill_and_career_advisor")
                    if career_advisor and isinstance(career_advisor, dict) and len(career_advisor) > 0:
                        context_data["career_advisor"] = career_advisor
                        log.info(f"✅ Retrieved career_advisor from gap_doc for uid={uid}")
                
                # Check for market_and_course_recommender
                if not context_data.get("market_and_course_recommender"):
                    market_course = gap_doc.get("market_and_course_recommender")
                    if market_course and isinstance(market_course, dict) and len(market_course) > 0:
                        context_data["market_and_course_recommender"] = market_course
                        log.info(f"✅ Retrieved market_and_course_recommender from gap_doc for uid={uid} (keys: {list(market_course.keys())})")
                    elif market_course:
                        log.warning(f"⚠️ market_and_course_recommender in gap_doc but empty or invalid: {type(market_course)}")
                
                # Check for resume_score
                if not context_data.get("resume_score"):
                    resume_score = gap_doc.get("resume_score")
                    if resume_score and isinstance(resume_score, dict) and len(resume_score) > 0:
                        context_data["resume_score"] = resume_score
                        log.info(f"✅ Retrieved resume_score from gap_doc for uid={uid}")
        except Exception as e:
            log.warning(f"⚠️ Could not get gap_doc: {e}")
            import traceback
            log.debug(traceback.format_exc())
        
        # Also try getting from user_profile helper (OPTIONAL) - fallback
        try:
            from chroma import get_user_profile
            user_profile = await run_blocking_io(get_user_profile, uid)
            log.info(f"🔍 Checking user_profile for uid={uid}, keys present: {list(user_profile.keys()) if user_profile else 'empty'}")
            if user_profile:
                # Merge with existing context (don't overwrite if already present)
                if not context_data.get("career_advisor"):
                    career_advisor = user_profile.get("career_advisor")
                    if career_advisor and isinstance(career_advisor, dict) and len(career_advisor) > 0:
                        context_data["career_advisor"] = career_advisor
                        log.info(f"✅ Retrieved career_advisor from user_profile for uid={uid}")
                
                if not context_data.get("market_and_course_recommender"):
                    market_course = user_profile.get("market_and_course_recommender")
                    if market_course and isinstance(market_course, dict) and len(market_course) > 0:
                        context_data["market_and_course_recommender"] = market_course
                        log.info(f"✅ Retrieved market_and_course_recommender from user_profile for uid={uid} (keys: {list(market_course.keys())})")
                    elif market_course:
                        log.warning(f"⚠️ market_and_course_recommender in user_profile but empty or invalid: {type(market_course)}")
                
                if not context_data.get("resume_score"):
                    resume_score = user_profile.get("resume_score")
                    if resume_score and isinstance(resume_score, dict) and len(resume_score) > 0:
                        context_data["resume_score"] = resume_score
                        log.info(f"✅ Retrieved resume_score from user_profile for uid={uid}")
        except Exception as e:
            log.warning(f"⚠️ Could not get user profile: {e}")
            import traceback
            log.debug(traceback.format_exc())
        
        # Get interview insights/evaluations (OPTIONAL) - from separate interview collection
        try:
            from agents.interview_chroma import InterviewChromaManager
            interview_chroma = InterviewChromaManager()
            
            # Try to find interview session for this uid
            # Interview sessions might have different session_id format, so we search by uid
            # Note: This is a best-effort retrieval - interview sessions may not always be linked to main session
            if session_id:
                try:
                    interview_session = await run_blocking_io(interview_chroma.get_interview_session, session_id)
                    if interview_session:
                        evaluation = interview_session.get("evaluation")
                        if evaluation:
                            context_data["interview_evaluation"] = evaluation
                            log.info(f"✅ Retrieved interview evaluation for session_id={session_id}")
                except Exception as e:
                    log.debug(f"Could not retrieve interview session by session_id: {e}")
            
            # Also try to get any recent interview evaluations for this uid
            # (This is a fallback if session_id doesn't match)
            try:
                # Note: InterviewChromaManager doesn't have a direct get_by_uid method
                # We'll rely on session_id matching for now
                pass
            except Exception as e:
                log.debug(f"Could not retrieve interview data by uid: {e}")
        except Exception as e:
            log.debug(f"Could not retrieve interview insights: {e}")
        
        # Step 2: Build comprehensive prompt for LLM
        # Log what context data we actually have before building summary
        log.info(f"📊 Context data keys before building summary: {list(context_data.keys())}")
        for key, value in context_data.items():
            if key != "structured_resume":  # Skip structured_resume as it's large
                if isinstance(value, dict):
                    log.info(f"📊 {key}: dict with {len(value)} keys: {list(value.keys())[:10]}")
                    # For market_and_course_recommender, log the structure
                    if key == "market_and_course_recommender":
                        courses = value.get("course_recommendations") or value.get("recommended_courses", [])
                        market_insights = value.get("market_insights", [])
                        log.info(f"📊 market_and_course_recommender: {len(courses) if isinstance(courses, list) else 0} courses, {len(market_insights) if isinstance(market_insights, list) else 0} insights")
                elif isinstance(value, list):
                    log.info(f"📊 {key}: list with {len(value)} items")
                else:
                    log.info(f"📊 {key}: {type(value).__name__}")
        
        # Build context summary (handles missing data gracefully)
        context_summary = _build_context_summary(context_data)
        log.info(f"📊 Context summary length: {len(context_summary)} characters")
        if context_summary:
            log.info(f"📊 Context summary preview (first 300 chars): {context_summary[:300]}...")
        else:
            log.warning(f"⚠️ Context summary is empty - no additional context will be included")
        
        # Build the prompt - works with or without additional context
        _common_output_structure = """
ATS-FRIENDLY REQUIREMENTS (industry standards—MUST follow):
- **Standard section headers only**: Use EXACTLY "Work Experience", "Education", "Skills", "Professional Summary", "Certifications", "Projects". No creative names. Omit Certifications and Projects if the candidate has none—do NOT include empty sections.
- **Keyword-rich**: Use industry-standard terms for roles, skills, and technologies (e.g., "Software Engineer" not "Code Ninja"; "Machine Learning" not "ML" in first mention).
- **Parseable structure**: Clear hierarchy; no tables, multi-column layout, or graphics in text; no special Unicode characters that break ATS parsing.
- **Action verbs**: Start every bullet with a strong verb (Developed, Led, Implemented, Designed, Achieved, Increased, Reduced).
- **Quantifiable achievements**: Include numbers, percentages, scale where possible (e.g., "Increased revenue by 25%", "Managed team of 5", "Served 10K+ users").
- **Reverse chronological**: Most recent experience and education first.
- **Spell out acronyms**: On first use, spell out then abbreviate (e.g., "Artificial Intelligence (AI)").
- **Contact clarity**: Name, location, email, phone in a parseable format; full URLs for LinkedIn/GitHub.
- **Skills**: Use common industry terms; group by category; include technologies, tools, methodologies.
- **Length**: Concise; typical 1-page for under 5 years experience, up to 2 pages for senior roles.

WELL-FORMATTED MARKDOWN RULES:
1. **Blank lines**: One blank line AFTER each ## header; one blank line BETWEEN sections
2. **URLs**: Full https:// URLs in links: [LinkedIn](https://linkedin.com/in/...), [GitHub](https://github.com/...)
3. **Section headers**: ## Work Experience, ## Education, ## Skills, ## Professional Summary; add ## Certifications and ## Projects only when the candidate has that data
4. **Bullets**: Use `-` consistently; one space after dash; max 2 lines per bullet
5. **Experience**: **Job Title** | Company Name | Date Range (e.g., "Jan 2020 - Dec 2022" or "January 2025 - Present")
6. **Education**: **Degree** | Institution | Year; GPA if 3.0+
7. **Projects**: ### Project Name, then - bullets with impact/outcomes
8. **Skills**: **Category:** item1, item2, item3 (comma-separated); use standard terms
9. **Summary**: 2-3 sentences; keyword-rich; tailored to role type; no first-person ("I")
10. **Readability**: Reasonable line length; proper breaks; no walls of text; professional tone throughout

STRUCTURE (strict order—omit sections the candidate has no data for):
- ## [Full Name]
- Contact line: Location | Email | Phone | [LinkedIn](https://...), [GitHub](https://...)
- ## Professional Summary
- ## Work Experience
- ## Education
- ## Skills
- ## Certifications — include ONLY if the candidate has certifications; omit entirely if none
- ## Projects — include ONLY if the candidate has projects; omit entirely if none

Do NOT wrap output in triple backticks. Output raw markdown only."""

        if context_summary.strip():
            prompt = f"""You are an expert resume writer specializing in ATS-friendly, industry-standard resumes. Your PRIMARY GOAL is to produce content that would score higher on ATS (Applicant Tracking System) criteria and human review (experience, skills, education, presentation).

SECONDARY GOALS:
- Preserve ALL factual data; do not fabricate information
- Follow ATS best practices and industry norms throughout
- Use clear, well-formatted structure that parses reliably
- Respect user interests and context when provided

RESUME DATA:
{json.dumps(structured_resume, indent=2, ensure_ascii=False)}

ADDITIONAL CONTEXT (scoring feedback, insights, recommendations):
{context_summary}

CRITICAL INSTRUCTIONS:
1. If "RESUME SCORE & IMPROVEMENT PRIORITIES" is provided above: address weak areas (low scores), implement improvement suggestions, preserve/emphasize strengths
2. Incorporate relevant insights from career advisor, market/course recommendations, assessment results, interview feedback where they add value
3. For low skills score: expand skills section with industry-standard terms; add relevant certifications or projects
4. For low experience score: strengthen bullet points with quantifiable achievements (metrics, percentages, scale); use action verbs
5. For low education score: clarify degrees, institutions, dates; add GPA if 3.0+
6. For low presentation score: use standard ATS section headers; improve structure, clarity, consistency; ensure parseable format
7. Do NOT fabricate—only refine, rephrase, restructure what is provided
8. Use standard section headers only: Work Experience, Education, Skills, Professional Summary. Include Certifications and Projects ONLY if the candidate has them—omit these sections if none exist.
9. Format dates consistently (e.g., "Jan 2020 - Dec 2022", "January 2025 - Present")
10. Start each bullet with a strong action verb; keep summary to 2-3 keyword-rich sentences
{_common_output_structure}

RESUME (Markdown):"""
        else:
            # Simplified prompt when no additional context is available
            prompt = f"""You are an expert resume writer specializing in ATS-friendly, industry-standard resumes. Generate a complete, well-formatted resume that would score well on ATS (Applicant Tracking System) and human review.

SECONDARY GOALS:
- Preserve ALL factual data; do not fabricate
- Follow ATS best practices and industry norms
- Use clear, professional, well-formatted Markdown structure

RESUME DATA:
{json.dumps(structured_resume, indent=2, ensure_ascii=False)}

INSTRUCTIONS:
1. Generate a complete, ATS-friendly resume in Markdown format
2. Use standard section headers: Work Experience, Education, Skills, Professional Summary. Include Certifications and Projects ONLY if the candidate has them—do NOT add empty Certifications or Projects sections.
3. Start bullets with action verbs; include quantifiable achievements where possible
4. Do NOT fabricate—only use data from the resume above
5. Format dates consistently (e.g., "Jan 2020 - Dec 2022", "January 2025 - Present")
6. Keep summary to 2-3 keyword-rich sentences; professional tone throughout
{_common_output_structure}

RESUME (Markdown):"""

        # Step 3: Generate resume content using LLM
        log.info("Generating resume content with LLM...")
        resume_content = await invoke_llm(
            prompt=prompt,
            task_type="text_generation",
            agent_name="resume_content_generator",
            max_output_tokens=8000  # Allow for comprehensive resume
        )
        
        if not resume_content or not resume_content.strip():
            processing_time = _calculate_processing_time(start_time)
            AgentLogger.log_error(log_context, "Empty response from LLM", processing_time)
            return _create_error_response("Failed to generate resume content", processing_time)
        
        # Clean up the response (remove any markdown code blocks if present)
        resume_content = resume_content.strip()
        if resume_content.startswith("```"):
            # Remove markdown code blocks
            lines = resume_content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            resume_content = "\n".join(lines).strip()
        
        # Normalize escaped newlines/tabs so Markdown-to-HTML/PDF converters render correctly.
        # If the content contains literal backslash-n or backslash-t (e.g. from LLM or JSON),
        # replace them with actual newline/tab so sections and bullets display properly.
        resume_content = resume_content.replace("\\n", "\n").replace("\\t", "\t")
        
        # Step 4: Record success
        processing_time = _calculate_processing_time(start_time)
        await generator_memory.record_attempt(
            'resume_generation', 'llm', True, 0.9, processing_time
        )
        
        # Determine which context was available
        # Helper to check if data is actually present and non-empty
        def _has_data(key: str) -> bool:
            data = context_data.get(key)
            if not data:
                return False
            if isinstance(data, dict):
                return len(data) > 0
            if isinstance(data, list):
                return len(data) > 0
            return bool(data)
        
        context_used = {
            "has_structured_resume": bool(structured_resume and structured_resume.get("Name")),
            "has_user_interests": _has_data("user_interests"),
            "has_career_advisor": _has_data("career_advisor"),
            "has_market_course": _has_data("market_and_course_recommender"),
            "has_assessment_results": _has_data("assessment_evaluator"),
            "has_report": _has_data("report"),
            "has_interview_evaluation": _has_data("interview_evaluation"),
            "has_resume_score": _has_data("resume_score"),
            "has_resume_analysis": _has_data("resume_analysis"),
            "has_uploaded_certificates": bool(uploaded_certificates),
        }
        
        # Log what context was actually found
        log.info(f"📊 Context summary for uid={uid}: {json.dumps(context_used, indent=2)}")
        log.info(f"📊 Available context keys: {list(context_data.keys())}")
        
        # Log success
        log_agent_completion(log_context, {
            "success": True,
            "resume_length": len(resume_content),
            "has_additional_context": bool(context_summary.strip()),
            **context_used
        }, "llm", processing_time)
        
        return {
            "resume_content": resume_content,
            "success": True,
            "processing_time_seconds": processing_time,
            "context_used": context_used
        }
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        AgentLogger.log_error(log_context, f"Resume content generation failed: {str(e)}", processing_time)
        
        # Record failure
        await generator_memory.record_attempt(
            'resume_generation', 'llm', False, 0.0, processing_time
        )
        
        return _create_error_response(f"Resume content generation failed: {str(e)}", processing_time)


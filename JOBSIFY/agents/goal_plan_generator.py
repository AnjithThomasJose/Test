"""
Goal Plan Generator Agent

Generates personalized career plans (milestones + tasks) from a goal description.
Uses user profile (resume, skills, gaps, assessments) to create relevant, actionable plans.
Also provides AI-recommended goals based on user data.
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from models.llm_invoker import invoke_llm
from core.model_registry import TaskType
from core.utils import run_blocking_io, _generate_request_id, _calculate_processing_time
from agents.career_coach.context_aggregator import load_comprehensive_context

log = logging.getLogger(__name__)


PLAN_GENERATOR_PROMPT = """You are a career planning expert for JobsifyAI. Create a detailed, personalized career plan.

## USER PROFILE
{user_profile}

## CURRENT SKILLS
{current_skills}

## SKILL GAPS
{skill_gaps}

## ASSESSMENT RESULTS
{assessment_results}

## CAREER PATHS (from analysis)
{career_paths}

## USER'S GOAL
{goal_text}

## TARGET TIMELINE
{target_timeline}

## INSTRUCTIONS

Create a comprehensive, actionable plan with sequential milestones. Each milestone achieved brings the user closer to their goal.

RULES:
1. Create 3-5 milestones (sequential phases)
2. Each milestone has 3-6 specific, actionable tasks
3. Tasks should be concrete (not vague like "learn more")
4. Consider the user's CURRENT skills — skip what they already know
5. Prioritize closing identified skill gaps
6. Assign realistic time estimates
7. Each task must have a task_type from: course, assessment, project, certification, job_application, resume_update, interview_prep, networking, learning, custom
8. Order milestones logically (foundation → intermediate → advanced → application)

Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{{
  "goal": "concise goal title",
  "description": "1-2 sentence description of the goal and approach",
  "estimated_duration_months": <number>,
  "milestones": [
    {{
      "title": "milestone name",
      "description": "what this milestone achieves",
      "order": 1,
      "estimated_duration_weeks": <number>,
      "tasks": [
        {{
          "title": "specific task description",
          "description": "what to do and why",
          "task_type": "course|assessment|project|certification|job_application|resume_update|interview_prep|networking|learning|custom",
          "priority": "high|medium|low",
          "estimated_hours": <number>,
          "completion_criteria": "how to know this task is done"
        }}
      ]
    }}
  ],
  "skill_gaps_addressed": ["skill1", "skill2"],
  "profile_analysis": {{
    "current_strengths": ["strength1", "strength2"],
    "gaps_to_close": ["gap1", "gap2"],
    "estimated_weekly_effort": "X-Y hours per week"
  }}
}}"""


GOAL_RECOMMENDER_PROMPT = """You are a career advisor for JobsifyAI. Based on the user's profile, suggest 3-5 career goals they should consider.

## USER PROFILE
{user_profile}

## CURRENT SKILLS
{current_skills}

## SKILL GAPS
{skill_gaps}

## CAREER PATHS (from analysis)
{career_paths}

## ROLE FIT SUGGESTIONS
{role_fit}

## ASSESSMENT RESULTS
{assessment_results}

## INSTRUCTIONS

Recommend 3-5 goals ordered by relevance. Mix different goal types:
- Career transition goals (new roles based on career paths)
- Skill improvement goals (based on gaps or low assessment scores)
- Certification/credential goals (if relevant to career paths)

For each goal, provide 3 key milestones as a preview.

Return ONLY valid JSON (no markdown) with this structure:
[
  {{
    "goal": "goal title",
    "description": "why this goal is relevant to the user",
    "category": "career_transition|skill_improvement|certification|leadership",
    "role_fit_percentage": <number or null>,
    "estimated_duration_months": <number>,
    "key_milestones": ["milestone 1", "milestone 2", "milestone 3"],
    "skill_gaps_to_close": ["skill1", "skill2"],
    "source": "career_paths|role_fit|skill_gaps|assessment_gaps|market_trends",
    "priority": "high|medium|low"
  }}
]"""


def _format_profile(context: Dict[str, Any]) -> str:
    parts = []
    resume = context.get("structured_resume") or context.get("resume_data") or {}
    if isinstance(resume, dict):
        name = resume.get("Name") or resume.get("name") or ""
        if name:
            parts.append(f"Name: {name}")
        title = resume.get("Title") or resume.get("current_title") or ""
        if title:
            parts.append(f"Current Role: {title}")
        exp = resume.get("experience") or resume.get("work_experience") or []
        if isinstance(exp, list) and exp:
            years_count = len(exp)
            parts.append(f"Work Experience: {years_count} positions")
            if isinstance(exp[0], dict):
                latest = exp[0]
                parts.append(f"Latest: {latest.get('title', '')} at {latest.get('company', '')}")
        education = resume.get("education") or resume.get("Education") or []
        if isinstance(education, list) and education:
            for edu in education[:2]:
                if isinstance(edu, dict):
                    degree = edu.get("degree") or edu.get("Degree") or ""
                    inst = edu.get("institution") or edu.get("Institution") or ""
                    if degree or inst:
                        parts.append(f"Education: {degree} from {inst}")
    return "\n".join(parts) if parts else "Profile information not available"


def _format_skills(context: Dict[str, Any]) -> str:
    skills = []
    resume = context.get("structured_resume") or context.get("resume_data") or {}
    if isinstance(resume, dict):
        tech = resume.get("technical_skills") or resume.get("TechnicalSkills") or []
        if isinstance(tech, list):
            skills.extend(tech[:20])
        soft = resume.get("soft_skills") or resume.get("SoftSkills") or []
        if isinstance(soft, list):
            skills.extend(soft[:10])
    if not skills:
        raw = context.get("raw_resume_text", "")
        if raw and isinstance(raw, str):
            return f"Skills extracted from resume (summary): {raw[:300]}"
    return ", ".join(skills) if skills else "No skills data available"


def _format_gaps(context: Dict[str, Any]) -> str:
    gaps = context.get("skill_gaps") or []
    if isinstance(gaps, list) and gaps:
        return "\n".join([f"- {g}" for g in gaps[:15]])
    return "No skill gaps identified yet"


def _format_assessments(context: Dict[str, Any]) -> str:
    results = context.get("assessment_results") or []
    if not results or not isinstance(results, list):
        return "No assessments completed yet"
    parts = []
    for r in results[:10]:
        if isinstance(r, dict):
            topic = r.get("assessment_topic") or r.get("topic") or "Unknown"
            score = r.get("total_score") or r.get("score")
            if score is not None:
                parts.append(f"- {topic}: {score}/100")
    return "\n".join(parts) if parts else "No assessments completed yet"


def _format_career_paths(context: Dict[str, Any]) -> str:
    paths = context.get("career_paths") or []
    if not paths or not isinstance(paths, list):
        return "No career paths analyzed yet"
    parts = []
    for p in paths[:5]:
        if isinstance(p, dict):
            title = p.get("title") or p.get("path_title") or "Unknown"
            match = p.get("match_percentage") or p.get("fit_percentage") or ""
            desc = p.get("description") or p.get("brief") or ""
            line = f"- {title}"
            if match:
                line += f" ({match}% match)"
            if desc:
                line += f": {str(desc)[:100]}"
            parts.append(line)
    return "\n".join(parts) if parts else "No career paths analyzed yet"


def _format_role_fit(context: Dict[str, Any]) -> str:
    roles = context.get("role_fit_suggestions") or []
    if not roles or not isinstance(roles, list):
        return "No role fit analysis available"
    parts = []
    for r in roles[:5]:
        if isinstance(r, dict):
            name = r.get("role_name") or r.get("role") or r.get("title") or "Unknown"
            fit = r.get("fit_percentage") or r.get("match_score") or ""
            line = f"- {name}"
            if fit:
                line += f" ({fit}% fit)"
            parts.append(line)
    return "\n".join(parts) if parts else "No role fit analysis available"


def _extract_json(text: str) -> Any:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()
    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        array_match = re.search(r'(\[[\s\S]*\])', text)
        if array_match:
            try:
                return json.loads(array_match.group(1))
            except json.JSONDecodeError:
                pass
        obj_match = re.search(r'(\{[\s\S]*\})', text)
        if obj_match:
            try:
                return json.loads(obj_match.group(1))
            except json.JSONDecodeError:
                pass
    return None


def _normalize_plan_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a task from LLM output into the standard goal task format."""
    valid_types = {
        "course", "assessment", "project", "certification",
        "job_application", "resume_update", "interview_prep",
        "networking", "learning", "custom"
    }
    task_type = str(task.get("task_type", "custom")).lower().strip()
    if task_type not in valid_types:
        task_type = "custom"

    return {
        "task_id": f"task_{_generate_request_id()}",
        "title": str(task.get("title", "")).strip()[:200],
        "description": str(task.get("description", "")).strip()[:500],
        "task_type": task_type,
        "status": "not_started",
        "priority": task.get("priority", "medium"),
        "estimated_hours": max(1, int(task.get("estimated_hours", 4) or 4)),
        "completion_criteria": str(task.get("completion_criteria", "")).strip()[:300],
        "linked_resource": task.get("linked_resource"),
        "created_at": datetime.utcnow().isoformat(),
    }


def _normalize_plan_milestone(ms: Dict[str, Any], milestone_order: int) -> Dict[str, Any]:
    """Normalize a milestone from LLM output into the standard format."""
    tasks = ms.get("tasks", [])
    normalized_tasks = []
    for i, t in enumerate(tasks[:8]):
        if isinstance(t, dict):
            nt = _normalize_plan_task(t)
            nt["order"] = i + 1
            normalized_tasks.append(nt)

    return {
        "milestone_id": f"ms_{_generate_request_id()}",
        "title": str(ms.get("title", "")).strip()[:200],
        "description": str(ms.get("description", "")).strip()[:500],
        "order": ms.get("order", milestone_order),
        "status": "not_started",
        "progress": 0.0,
        "estimated_duration_weeks": max(1, int(ms.get("estimated_duration_weeks", 4) or 4)),
        "tasks": normalized_tasks,
        "created_at": datetime.utcnow().isoformat(),
    }


def _calculate_target_date(duration_months: int) -> str:
    """Calculate a target date from estimated duration in months."""
    target = datetime.utcnow() + timedelta(days=duration_months * 30)
    return target.strftime("%Y-%m-%d")


async def generate_goal_plan(
    uid: str,
    goal_text: str,
    target_date: Optional[str] = None,
    priority: str = "high",
    session_id: Optional[str] = None,
    save: bool = False,
) -> Dict[str, Any]:
    """
    Generate a complete career plan (milestones + tasks) from a goal description.

    Args:
        uid: User ID
        goal_text: Free-text career goal
        target_date: Optional target completion date
        priority: Goal priority (high/medium/low)
        session_id: Optional session ID
        save: If True, auto-save the goal to storage

    Returns:
        Dict with generated_plan, profile_analysis, and metadata
    """
    start_time = time.time()
    log.info(f"🎯 Generating goal plan for uid={uid}: {goal_text[:80]}")

    try:
        context = await load_comprehensive_context(uid, session_id)

        target_timeline = target_date or "Flexible (AI will estimate)"
        prompt = PLAN_GENERATOR_PROMPT.format(
            user_profile=_format_profile(context),
            current_skills=_format_skills(context),
            skill_gaps=_format_gaps(context),
            assessment_results=_format_assessments(context),
            career_paths=_format_career_paths(context),
            goal_text=goal_text,
            target_timeline=target_timeline,
        )

        response = await invoke_llm(
            prompt,
            task_type=TaskType.TEXT_GENERATION,
            agent_name="goal_plan_generator",
            response_mime_type="application/json",
            max_output_tokens=4096,
        )

        plan_data = _extract_json(response)
        if not plan_data or not isinstance(plan_data, dict):
            raise ValueError("LLM returned invalid plan structure")

        milestones_raw = plan_data.get("milestones", [])
        if not milestones_raw:
            raise ValueError("LLM plan has no milestones")

        normalized_milestones = []
        for i, ms in enumerate(milestones_raw[:5]):
            if isinstance(ms, dict):
                normalized_milestones.append(_normalize_plan_milestone(ms, i + 1))

        duration = max(1, int(plan_data.get("estimated_duration_months", 6) or 6))
        final_target = target_date or _calculate_target_date(duration)

        generated_plan = {
            "goal": str(plan_data.get("goal", goal_text)).strip()[:200],
            "description": str(plan_data.get("description", "")).strip()[:1000],
            "estimated_duration_months": duration,
            "target_date": final_target,
            "priority": priority,
            "milestones": normalized_milestones,
            "skill_gaps_addressed": plan_data.get("skill_gaps_addressed", [])[:10],
            "profile_analysis": plan_data.get("profile_analysis", {}),
        }

        total_tasks = sum(len(m.get("tasks", [])) for m in normalized_milestones)
        processing_time = _calculate_processing_time(start_time)
        log.info(
            f"✅ Plan generated: {len(normalized_milestones)} milestones, "
            f"{total_tasks} tasks in {processing_time:.1f}s"
        )

        result = {
            "success": True,
            "generated_plan": generated_plan,
            "metadata": {
                "milestones_count": len(normalized_milestones),
                "total_tasks": total_tasks,
                "processing_time": processing_time,
            },
        }

        if save:
            from agents.career_coach.function_tools import _load_goals, _save_goals
            goal_obj = {
                "goal_id": f"goal_{_generate_request_id()}",
                "uid": uid,
                "title": generated_plan["goal"],
                "description": generated_plan["description"],
                "target_date": generated_plan["target_date"],
                "priority": generated_plan["priority"],
                "status": "in_progress",
                "progress": 0.0,
                "estimated_duration_months": generated_plan["estimated_duration_months"],
                "milestones": generated_plan["milestones"],
                "tasks": [],
                "skill_gaps_addressed": generated_plan.get("skill_gaps_addressed", []),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "created_by": "growth_plan_page",
            }
            goals = await _load_goals(uid, session_id)
            goals.append(goal_obj)
            await _save_goals(uid, session_id, goals)
            result["goal_id"] = goal_obj["goal_id"]
            result["saved"] = True
            log.info(f"💾 Goal auto-saved: {goal_obj['goal_id']}")

        return result

    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        log.error(f"❌ Plan generation failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "processing_time": processing_time,
        }


async def get_recommended_goals(
    uid: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate AI-recommended goals based on user profile.

    Pulls from career paths, role fit, skill gaps, and assessment results
    to suggest relevant goals the user should consider.

    Returns:
        Dict with recommended_goals list and metadata
    """
    start_time = time.time()
    log.info(f"📌 Generating goal recommendations for uid={uid}")

    try:
        context = await load_comprehensive_context(uid, session_id)

        prompt = GOAL_RECOMMENDER_PROMPT.format(
            user_profile=_format_profile(context),
            current_skills=_format_skills(context),
            skill_gaps=_format_gaps(context),
            career_paths=_format_career_paths(context),
            role_fit=_format_role_fit(context),
            assessment_results=_format_assessments(context),
        )

        response = await invoke_llm(
            prompt,
            task_type=TaskType.TEXT_GENERATION,
            agent_name="goal_recommender",
            response_mime_type="application/json",
            max_output_tokens=3000,
        )

        recommendations = _extract_json(response)
        if not recommendations or not isinstance(recommendations, list):
            raise ValueError("LLM returned invalid recommendations")

        valid_categories = {"career_transition", "skill_improvement", "certification", "leadership"}
        cleaned = []
        for rec in recommendations[:5]:
            if not isinstance(rec, dict) or not rec.get("goal"):
                continue
            cat = str(rec.get("category", "skill_improvement")).lower()
            if cat not in valid_categories:
                cat = "skill_improvement"
            cleaned.append({
                "goal": str(rec["goal"]).strip()[:200],
                "description": str(rec.get("description", "")).strip()[:500],
                "category": cat,
                "role_fit_percentage": rec.get("role_fit_percentage"),
                "estimated_duration_months": max(1, int(rec.get("estimated_duration_months", 3) or 3)),
                "key_milestones": [str(m)[:100] for m in rec.get("key_milestones", [])[:5]],
                "skill_gaps_to_close": [str(s)[:50] for s in rec.get("skill_gaps_to_close", [])[:5]],
                "source": rec.get("source", "career_paths"),
                "priority": rec.get("priority", "medium"),
            })

        processing_time = _calculate_processing_time(start_time)
        log.info(f"✅ Generated {len(cleaned)} goal recommendations in {processing_time:.1f}s")

        return {
            "success": True,
            "recommended_goals": cleaned,
            "total": len(cleaned),
            "sources_used": list({r.get("source", "") for r in cleaned}),
            "processing_time": processing_time,
        }

    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        log.error(f"❌ Goal recommendation failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "recommended_goals": [],
            "processing_time": processing_time,
        }

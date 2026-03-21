"""
Standalone agent to generate mock interview questions for a candidate and job.
Inputs: job_id, uid. Fetches JD and candidate from ChromaDB, generates Q&A based on the job.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from chroma import get_job_description, fetch_structured_resume
from core.utils import run_blocking_io
from models.llm_invoker import invoke_structured_llm
from core.model_registry import TaskType

log = logging.getLogger(__name__)


# Issue 5.4: Define Pydantic models for structured LLM output
class MockInterviewQA(BaseModel):
    """A single mock interview question-answer pair."""
    question: str = Field(description="The interview question")
    expected_answer: str = Field(description="The expected/sample answer (2-4 sentences)")


class MockInterviewQAList(BaseModel):
    """List of mock interview Q&A pairs."""
    mock_interview_qa: List[MockInterviewQA] = Field(default_factory=list, description="List of Q&A pairs")


def _build_candidate_summary(structured_resume: Optional[Dict[str, Any]]) -> str:
    """Build a short text summary of candidate for prompt context."""
    if not structured_resume:
        return "Candidate details not available."
    parts = []
    skills = structured_resume.get("skills", [])
    if isinstance(skills, list):
        names = []
        for s in skills[:15]:
            if isinstance(s, str):
                names.append(s)
            elif isinstance(s, dict):
                n = s.get("SkillName") or s.get("skillName") or s.get("name", "")
                if n:
                    names.append(n)
        if names:
            parts.append(f"Skills: {', '.join(names)}")
    exp = structured_resume.get("experience", [])
    if isinstance(exp, list) and exp:
        e = exp[0]
        if isinstance(e, dict):
            title = e.get("job_title") or e.get("title", "")
            company = e.get("company", "")
            if title:
                parts.append(f"Recent role: {title} at {company}")
    return " | ".join(parts) if parts else "Candidate details not available."


async def _generate_mock_qa_llm(
    job_description: Dict[str, Any],
    candidate_summary: str,
    num_questions: int,
) -> List[Dict[str, str]]:
    """Call LLM to generate mock interview Q&A based on JD (and optional candidate context)."""
    try:
        from settings import settings as _settings

        job_title = job_description.get("jobTitle") or job_description.get("title", "N/A")
        required_skills = job_description.get("requiredSkills") or job_description.get("required_skills", [])
        if isinstance(required_skills, str):
            required_skills = [s.strip() for s in required_skills.split(",")]
        skills_str = ", ".join(required_skills[:20]) if required_skills else "Not specified"
        jd_desc = (
            job_description.get("fullJobDescription")
            or job_description.get("description")
            or job_description.get("full_job_description")
            or ""
        )
        jd_desc_trunc = (jd_desc[:800] + "...") if len(jd_desc) > 800 else jd_desc
        responsibilities = job_description.get("responsibilities") or job_description.get("keyResponsibilities") or []
        resp_text = "\n".join([f"- {r}" for r in responsibilities]) if isinstance(responsibilities, list) else str(responsibilities)

        prompt = f"""You are an expert interviewer. Generate {num_questions} mock interview questions with sample/expected answers for a candidate preparing for this role.

JOB TITLE: {job_title}
REQUIRED SKILLS: {skills_str}
JOB DESCRIPTION (excerpt): {jd_desc_trunc}
KEY RESPONSIBILITIES:
{resp_text or "Not specified"}

CANDIDATE CONTEXT (use to tailor 1-2 questions if relevant; focus on job-based questions):
{candidate_summary}

Requirements:
- Mix of behavioral (e.g., past experience, teamwork) and technical/role-specific questions.
- Each item must have "question" (one clear question) and "expected_answer" (2-4 sentences: key points a strong answer would cover).
- Questions should be specific to this role and its requirements.
- Return ONLY valid JSON, no markdown or extra text.

Format:
{{"mock_interview_qa": [
  {{"question": "...", "expected_answer": "..."}},
  ...
]}}
"""

        response = await invoke_structured_llm(
            prompt,
            MockInterviewQAList,
            task_type=TaskType.INTERVIEW,
            preferred_model=_settings.GEMINI_MODEL,
            agent_name="mock_interview_prep",
            temperature=0.2,
            timeout=60.0,
            raise_on_fallback=False,
        )
        
        if response is None or not hasattr(response, 'mock_interview_qa'):
            log.warning("Mock interview Q&A: structured output returned None or invalid")
            return []
        
        out = []
        for item in response.mock_interview_qa[:num_questions]:
            if item.question and item.expected_answer:
                out.append({
                    "question": item.question.strip(),
                    "expected_answer": item.expected_answer.strip()
                })
        log.info(f"Generated {len(out)} mock interview Q&A for JD")
        return out
    except Exception as e:
        log.warning(f"Mock interview Q&A generation failed: {e}")
        return []


async def generate_mock_interview_qa(
    job_id: str,
    uid: str,
    num_questions: int = 6,
) -> Dict[str, Any]:
    """
    Retrieve JD and candidate, then generate mock interview Q&A based on the job.
    Independent call (not part of graph).
    """
    log.info(f"Mock interview prep: job_id={job_id}, uid={uid}, num_questions={num_questions}")
    job_description = await run_blocking_io(get_job_description, job_id)
    if not job_description:
        return {
            "job_id": job_id,
            "uid": uid,
            "error": "Job description not found",
            "mock_interview_qa": [],
        }
    structured_resume = await run_blocking_io(fetch_structured_resume, uid)
    candidate_summary = _build_candidate_summary(structured_resume)
    mock_qa = await _generate_mock_qa_llm(job_description, candidate_summary, num_questions)
    job_title = job_description.get("jobTitle") or job_description.get("title") or job_id
    return {
        "job_id": job_id,
        "uid": uid,
        "job_title": job_title,
        "mock_interview_qa": mock_qa,
    }

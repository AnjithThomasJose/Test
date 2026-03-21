"""
Prescreening Questions Agent

Generates JD-specific prescreening questions for recruiters to quickly identify
qualified candidates before a full interview. Runs in the corporate flow after
job_description_parser.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional

from models.llm_invoker import invoke_llm
from core.utils import _safe_json_loads, _calculate_processing_time, run_blocking_io
from core.logging_helpers import create_log_context

log = logging.getLogger(__name__)

# Default question count
DEFAULT_NUM_QUESTIONS = 9
MIN_QUESTIONS = 5
MAX_QUESTIONS = 15

# Role level mapping from experience string
EXPERIENCE_TO_LEVEL = {
    "0-1": "junior", "1-3": "junior", "3-5": "mid",
    "5-8": "senior", "8-12": "lead", "12+": "executive",
}


def _infer_role_level(job_description: Dict[str, Any]) -> str:
    """Infer role level from job description experience or experienceLevel."""
    level = (job_description.get("experienceLevel") or "").strip().lower()
    if level:
        level_map = {"junior": "junior", "mid": "mid", "middle": "mid", "senior": "senior",
                     "lead": "lead", "principal": "lead", "executive": "executive"}
        return level_map.get(level, "mid")

    exp = (job_description.get("experience") or "").strip().lower()
    for pattern, role in EXPERIENCE_TO_LEVEL.items():
        if pattern in exp.lower():
            return role
    return "mid"


def _build_jd_context(job_description: Dict[str, Any]) -> str:
    """Build context string from parsed JD for the prompt."""
    parts = []
    if job_description.get("jobTitle"):
        parts.append(f"Job Title: {job_description['jobTitle']}")
    if job_description.get("company"):
        parts.append(f"Company: {job_description['company']}")
    if job_description.get("requiredSkills"):
        skills = job_description["requiredSkills"]
        if isinstance(skills, list):
            parts.append(f"Required Skills: {', '.join(skills)}")
        else:
            parts.append(f"Required Skills: {skills}")
    if job_description.get("preferredSkills"):
        skills = job_description["preferredSkills"]
        if isinstance(skills, list):
            parts.append(f"Preferred Skills: {', '.join(skills)}")
    if job_description.get("experience"):
        parts.append(f"Experience: {job_description['experience']}")
    if job_description.get("educationRequired"):
        parts.append(f"Education: {job_description['educationRequired']}")
    if job_description.get("fullJobDescription"):
        desc = job_description["fullJobDescription"]
        if len(desc) > 8000:
            desc = desc[:8000] + "..."
        parts.append(f"\nFull Job Description:\n{desc}")
    return "\n".join(parts) if parts else "No job description context available."


def _generate_prescreening_prompt(
    jd_context: str,
    role_level: str,
    num_questions: int,
    question_types: str,
) -> str:
    """Build the prescreening prompt from the user's template."""
    return f"""You are an expert technical recruiter and hiring consultant. Your task is to generate prescreening questions for a job opening based on the provided Job Description (JD).

## INPUT
Job Description:
{jd_context}

Role Level: {role_level}
Question Count: {num_questions}
Question Types: {question_types}

## OBJECTIVE
Generate {num_questions} prescreening questions that help recruiters quickly identify whether a candidate is a strong fit BEFORE a full interview.

## QUESTION DESIGN RULES
1. **Must-have signals**: At least 40% of questions should test non-negotiable requirements from the JD (mandatory skills, experience thresholds, certifications).
2. **Killer questions first**: Start with 2-3 disqualifying questions — if answered wrong, the candidate is clearly unfit. These should be binary (yes/no or specific threshold).
3. **Depth questions**: Include 2-3 questions that reveal depth of experience, not just presence of a skill.
4. **Culture/context fit**: 1-2 questions on work style, availability, or logistics (notice period, location, remote preference) if relevant in the JD.
5. **Avoid vanity questions**: Do not ask generic questions like "Tell me about yourself" or "Why do you want this job".
6. **Each question must map to a JD requirement**: In your output, tag each question with the JD section it tests.

## OUTPUT FORMAT
Return a JSON array with this structure:
[
  {{
    "question_id": 1,
    "question_text": "...",
    "question_type": "yes_no | mcq | short_answer",
    "jd_mapping": "skill/requirement this tests",
    "is_disqualifier": true/false,
    "ideal_answer_hint": "what a good answer looks like",
    "options": ["..."]
  }}
]

## TONE
Professional, direct, and neutral. Avoid leading questions. Avoid bias based on age, gender, nationality, or education pedigree.

Return ONLY valid JSON. No markdown, no code blocks, no explanatory text. Start with [ and end with ]."""


def _parse_questions_response(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM response into list of question dicts."""
    if not raw or not isinstance(raw, str):
        return []

    raw = raw.strip()
    # Remove markdown code blocks if present
    if "```json" in raw:
        m = re.search(r"```json\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
    elif "```" in raw:
        m = re.search(r"```\s*([\s\S]*?)\s*```", raw)
        if m:
            raw = m.group(1).strip()

    parsed = _safe_json_loads(raw)
    if isinstance(parsed, list) and len(parsed) > 0:
        return parsed

    # Try extracting from object
    if isinstance(parsed, dict):
        for key in ("questions", "prescreening_questions", "prescreeningQuestions"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]

    # Try balanced array scan
    first = raw.find("[")
    if first >= 0:
        depth = 0
        for i in range(first, len(raw)):
            if raw[i] == "[":
                depth += 1
            elif raw[i] == "]":
                depth -= 1
                if depth == 0:
                    chunk = raw[first : i + 1]
                    arr = _safe_json_loads(chunk)
                    if isinstance(arr, list):
                        return arr
                    break

    return []


# System instruction for JSON-only output (cached by Gemini)
PRESCREENING_SYSTEM_INSTRUCTION = """You are an expert technical recruiter. Generate prescreening questions as a JSON array only. Return no markdown, no code blocks, no explanatory text. Output must be valid JSON starting with [ and ending with ]."""


async def prescreening_questions_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate prescreening questions for a job description.

    Inputs from state:
        - job_description: Parsed JD dict
        - job_id: Job ID (optional, for ChromaDB update)
        - body: Optional prescreening_num_questions, prescreening_question_types

    Outputs:
        - prescreening_questions: List of question dicts
        - job_description: Updated with prescreening_questions if job_id present
    """
    log_context = create_log_context("prescreening_questions", state.get("tenant_id", "default"))
    start_time = log_context["start_time"]

    job_description = state.get("job_description")
    job_id = state.get("job_id")
    body = state.get("body") or {}

    if not job_description:
        processing_time = _calculate_processing_time(start_time)
        return {
            "prescreening_questions": [],
            "error": "No job_description in state",
            "processing_time_seconds": processing_time,
        }

    # Params from state (passed by job_description_parser) or body
    num_questions = (
        state.get("prescreening_num_questions")
        or body.get("prescreening_num_questions")
        or body.get("prescreeningNumQuestions")
        or DEFAULT_NUM_QUESTIONS
    )
    num_questions = max(MIN_QUESTIONS, min(MAX_QUESTIONS, int(num_questions)))

    question_types = (
        state.get("prescreening_question_types")
        or body.get("prescreening_question_types")
        or body.get("prescreeningQuestionTypes")
        or "mixed"
    )

    role_level = _infer_role_level(job_description)
    jd_context = _build_jd_context(job_description)

    prompt = _generate_prescreening_prompt(
        jd_context=jd_context,
        role_level=role_level,
        num_questions=num_questions,
        question_types=question_types,
    )

    log.info(f"[PRESCREENING] Generating {num_questions} questions for role_level={role_level}, types={question_types}")

    try:
        raw_response = await invoke_llm(
            prompt,
            task_type="assessment_generation",
            agent_name="prescreening_questions",
            response_mime_type="application/json",
            preferred_model="gemini-2.5-flash",
            max_output_tokens=4096,
            system_instruction=PRESCREENING_SYSTEM_INSTRUCTION,
        )

        questions = _parse_questions_response(raw_response)

        if not questions:
            log.warning("[PRESCREENING] LLM returned no valid questions, using empty list")
            questions = []

        processing_time = _calculate_processing_time(start_time)
        log.info(f"[PRESCREENING] Generated {len(questions)} questions in {processing_time:.2f}s")

        result = {
            "prescreening_questions": questions,
            "processing_time_seconds": processing_time,
        }

        # Optionally merge into job_description and persist to ChromaDB
        if job_id and questions:
            try:
                from chroma import update_job_description
                updated_jd = {**job_description, "prescreening_questions": questions}
                await run_blocking_io(update_job_description, job_id, updated_jd)
                result["job_description"] = updated_jd
                log.info(f"[PRESCREENING] Updated ChromaDB with prescreening_questions for job_id={job_id}")
            except Exception as e:
                log.warning(f"[PRESCREENING] Failed to update ChromaDB: {e}")

        return result

    except Exception as e:
        log.error(f"[PRESCREENING] Error: {e}")
        import traceback
        log.debug(traceback.format_exc())
        processing_time = _calculate_processing_time(start_time)
        return {
            "prescreening_questions": [],
            "error": str(e),
            "processing_time_seconds": processing_time,
        }

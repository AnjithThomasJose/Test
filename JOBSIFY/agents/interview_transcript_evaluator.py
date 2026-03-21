"""
Interview Transcript Evaluator Agent

Evaluates interview transcripts and provides feedback to the interviewer by comparing
candidate responses against the job description and interviewer questions.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from models.llm_invoker import invoke_llm
from core.config import get_agent_config
from core.utils import _extract_json_from_response, _safe_json_loads

log = logging.getLogger(__name__)


def _jd_dict_to_text(jd_dict: Dict[str, Any]) -> str:
    """Build job description text from Chroma JD dict."""
    if not jd_dict or not isinstance(jd_dict, dict):
        return ""
    parts = []
    title = jd_dict.get("jobTitle") or jd_dict.get("title") or ""
    if title:
        parts.append(f"Title: {title}")
    desc = jd_dict.get("fullJobDescription") or jd_dict.get("description") or ""
    if desc:
        parts.append(f"Description: {desc}")
    skills = jd_dict.get("requiredSkills") or []
    if skills:
        parts.append(f"Required Skills: {', '.join(skills) if isinstance(skills, list) else skills}")
    exp = jd_dict.get("experience") or ""
    if exp:
        parts.append(f"Experience: {exp}")
    company = jd_dict.get("company") or ""
    if company:
        parts.append(f"Company: {company}")
    return "\n\n".join(parts) if parts else json.dumps(jd_dict, default=str)[:8000]


def _conversation_history_to_qa_pairs(history: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Convert interview conversation_history (role/content) to Q&A pairs.
    Interview format: assistant = interviewer (AI), user = candidate."""
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i < len(history):
        msg = history[i]
        role = (msg.get("role") or msg.get("speaker") or "").lower()
        content = (msg.get("content") or msg.get("text") or "").strip()
        if not content:
            i += 1
            continue
        # assistant = interviewer question, user = candidate answer
        if role in ("assistant", "interviewer", "interview"):
            question = content
            answer_parts: List[str] = []
            i += 1
            while i < len(history):
                next_msg = history[i]
                next_role = (next_msg.get("role") or next_msg.get("speaker") or "").lower()
                next_content = (next_msg.get("content") or next_msg.get("text") or "").strip()
                if next_role in ("assistant", "interviewer"):
                    break
                if next_content and next_role in ("user", "candidate"):
                    answer_parts.append(next_content)
                i += 1
            answer = " ".join(answer_parts).strip()
            if question or answer:
                pairs.append((question, answer))
        else:
            i += 1
    return pairs

config = get_agent_config("interview_transcript_evaluator")


# =============================================================================
# Pydantic Models
# =============================================================================


class TranscriptTurn(BaseModel):
    """A single turn in the transcript."""
    speaker: str = Field(description="'interviewer' or 'candidate'")
    text: str = Field(description="What was said")
    timestamp: Optional[str] = Field(None, description="Optional timestamp")

    @field_validator("speaker")
    @classmethod
    def normalize_speaker(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v in ("interviewer", "int", "i"):
            return "interviewer"
        if v in ("candidate", "cand", "c"):
            return "candidate"
        return v or "unknown"


class JdAlignmentItem(BaseModel):
    """How a candidate answer aligns with a specific JD requirement."""
    jd_item: str = Field(description="Job requirement or skill from JD")
    evidence_from_answer: str = Field(description="Evidence from candidate's answer")
    alignment_score: int = Field(ge=0, le=5, description="1-5 alignment strength")
    notes: Optional[str] = None


class PerAnswerEvaluation(BaseModel):
    """Evaluation of a single candidate answer."""
    question_id: str = Field(description="Identifier for the question")
    question_text: str = Field(description="The interviewer's question")
    answer_summary: str = Field(description="Brief summary of the candidate's answer")
    relevance_score: int = Field(ge=0, le=5, description="1-5: Did they answer the question?")
    depth_score: int = Field(ge=0, le=5, description="1-5: Depth and detail of response")
    evidence_score: int = Field(ge=0, le=5, description="1-5: Use of concrete examples/STAR")
    communication_score: int = Field(ge=0, le=5, description="1-5: Clarity and structure")
    jd_alignment: List[JdAlignmentItem] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    follow_up_suggestions: List[str] = Field(default_factory=list)


class OverallSummary(BaseModel):
    """Overall candidate assessment vs job description."""
    strengths_vs_jd: List[str] = Field(default_factory=list)
    gaps_vs_jd: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    recommendation: str = Field(description="e.g. 'Strong yes', 'Leaning yes', 'Needs more signal', 'Leaning no'")
    recommendation_rationale: str = Field(default="")


class InterviewerFeedback(BaseModel):
    """Feedback for the interviewer."""
    jd_coverage_gaps: List[str] = Field(
        default_factory=list,
        description="JD skills/competencies not probed by questions"
    )
    question_quality_notes: List[str] = Field(default_factory=list)
    suggested_follow_up_questions: List[str] = Field(default_factory=list)


# =============================================================================
# Transcript Parsing
# =============================================================================


def _parse_transcript_into_turns(transcript: Any) -> List[Dict[str, str]]:
    """
    Parse transcript into list of turns.
    Supports:
    - List of {speaker, text} or {speaker, text, timestamp}
    - Raw text with "Interviewer:" / "Candidate:" prefixes
    - Raw text with "Q:" / "A:" prefixes
    """
    turns: List[Dict[str, str]] = []

    if isinstance(transcript, list):
        for item in transcript:
            if isinstance(item, dict):
                speaker = (item.get("speaker") or item.get("role") or "").strip().lower()
                text = (item.get("text") or item.get("content") or item.get("message") or "").strip()
                if not text:
                    continue
                if speaker in ("interviewer", "int", "i", "questioner"):
                    speaker = "interviewer"
                elif speaker in ("candidate", "cand", "c", "applicant"):
                    speaker = "candidate"
                else:
                    speaker = speaker or "unknown"
                turns.append({"speaker": speaker, "text": text})
        return turns

    if isinstance(transcript, str):
        text = transcript.strip()
        # Match "Interviewer: ..." or "Candidate: ..." or "Q: ..." or "A: ..." blocks
        pattern = re.compile(
            r"(?:^(?:Interviewer|Candidate|Q|A|Int|Cand)\s*:\s*)(.*?)(?=^(?:Interviewer|Candidate|Q|A|Int|Cand)\s*:|$)",
            re.DOTALL | re.IGNORECASE | re.MULTILINE
        )
        # Alternative: split on lines that start with speaker prefix
        parts = re.split(r"\n(?=(?:Interviewer|Candidate|Q|A|Int|Cand)\s*:)", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                match = re.match(r"^(Interviewer|Candidate|Q|A|Int|Cand)\s*:\s*(.*)", part, re.DOTALL | re.I)
                if match:
                    role = "interviewer" if match.group(1).lower() in ("interviewer", "q", "int") else "candidate"
                    content = (match.group(2) or "").strip()
                    if content:
                        turns.append({"speaker": role, "text": content})
                else:
                    # No prefix - alternate with previous speaker
                    prev = turns[-1]["speaker"] if turns else "interviewer"
                    next_speaker = "candidate" if prev == "interviewer" else "interviewer"
                    turns.append({"speaker": next_speaker, "text": part})
        else:
            # No clear structure - treat as single candidate response
            if text:
                turns.append({"speaker": "candidate", "text": text})

    return turns


def _group_turns_into_qa_pairs(turns: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """
    Group transcript turns into (question, answer) pairs.
    Each interviewer turn is followed by concatenated candidate turns until next interviewer.
    """
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i < len(turns):
        turn = turns[i]
        if turn["speaker"] != "interviewer":
            i += 1
            continue
        question = turn["text"].strip()
        if not question:
            i += 1
            continue
        # Collect all consecutive candidate turns
        answer_parts: List[str] = []
        i += 1
        while i < len(turns) and turns[i]["speaker"] == "candidate":
            answer_parts.append(turns[i]["text"].strip())
            i += 1
        answer = " ".join(answer_parts).strip() if answer_parts else ""
        if question or answer:
            pairs.append((question, answer))
    return pairs


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an expert hiring partner that evaluates interview transcripts.
You compare candidate answers to the job description and interviewer questions.
Focus on evidence, not vibes. Be concise and structured.
Return valid JSON only. Do not make up facts not present in the transcript."""


def _build_per_answer_prompt(
    job_description: str,
    question: str,
    answer: str,
    question_id: str,
) -> str:
    return f"""Evaluate this interview Q&A against the job description.

## Job Description
{job_description[:6000]}

## Interviewer Question (ID: {question_id})
{question}

## Candidate Answer
{answer or "[No response or inaudible]"}

Return a JSON object with this exact structure (no other text):
{{
  "question_id": "{question_id}",
  "question_text": "{question[:500]}",
  "answer_summary": "1-2 sentence summary of what the candidate said",
  "relevance_score": 1-5,
  "depth_score": 1-5,
  "evidence_score": 1-5,
  "communication_score": 1-5,
  "jd_alignment": [
    {{"jd_item": "requirement from JD", "evidence_from_answer": "quote or paraphrase", "alignment_score": 1-5, "notes": "optional"}}
  ],
  "strengths": ["strength1", "strength2"],
  "concerns": ["concern1"],
  "follow_up_suggestions": ["suggested follow-up question"]
}}

Scoring: 1=very weak, 3=adequate, 5=excellent. Be evidence-based."""


def _build_overall_prompt(
    job_description: str,
    per_answer_json: List[Dict[str, Any]],
) -> str:
    answers_str = json.dumps(per_answer_json, indent=2, default=str)[:12000]
    return f"""Given the job description and per-answer evaluations below, produce an overall candidate assessment.

## Job Description
{job_description[:4000]}

## Per-Answer Evaluations
{answers_str}

Return a JSON object:
{{
  "strengths_vs_jd": ["strength1", "strength2"],
  "gaps_vs_jd": ["gap1", "gap2"],
  "risk_factors": ["risk1"],
  "recommendation": "Strong yes | Leaning yes | Needs more signal | Leaning no | No",
  "recommendation_rationale": "2-3 sentence rationale"
}}"""


def _build_interviewer_feedback_prompt(
    job_description: str,
    questions_asked: List[str],
    jd_items_covered: List[str],
) -> str:
    questions_str = "\n".join(f"- {q[:200]}" for q in questions_asked[:20])
    covered_str = "\n".join(f"- {c}" for c in jd_items_covered[:30])
    return f"""As a hiring expert, suggest feedback for the interviewer.

## Job Description
{job_description[:4000]}

## Questions the interviewer asked
{questions_str}

## JD items that were covered (from candidate answers)
{covered_str}

Return a JSON object:
{{
  "jd_coverage_gaps": ["JD skill/competency not probed"],
  "question_quality_notes": ["note on question clarity or effectiveness"],
  "suggested_follow_up_questions": ["question to ask in next round"]
}}"""


# =============================================================================
# Agent
# =============================================================================


async def evaluate_interview_transcript(
    job_id: Optional[str] = None,
    uid: Optional[str] = None,
    job_description: Optional[str] = None,
    transcript: Optional[Any] = None,
    interviewer_questions: Optional[List[Dict[str, str]]] = None,
    session_id: Optional[str] = None,
    include_interviewer_feedback: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate an interview transcript against a job description.

    Args:
        job_id: Job ID to fetch JD from Chroma (required if job_description not provided)
        uid: Candidate UID for identification
        job_description: Full JD text override (if provided, skips Chroma fetch)
        transcript: Transcript as list of {speaker, text} or raw string with Q/A prefixes
        interviewer_questions: List of {question, answer} to use instead of parsing
        session_id: If provided, fetch transcript from interview_chroma
        include_interviewer_feedback: Whether to include feedback for the interviewer

    Returns:
        Dict with job_id, uid, per_answer_evaluations, overall_summary, interviewer_feedback
    """
    start = time.time()

    # Resolve job description
    jd_text = (job_description or "").strip()
    if not jd_text and job_id:
        try:
            from chroma import get_job_description
            from core.utils import run_blocking_io
            jd_dict = await run_blocking_io(get_job_description, job_id)
            if jd_dict:
                jd_text = _jd_dict_to_text(jd_dict)
            else:
                return {
                    "error": f"Job description not found for job_id={job_id}",
                    "job_id": job_id,
                    "uid": uid,
                    "per_answer_evaluations": [],
                    "overall_summary": {},
                    "interviewer_feedback": {},
                    "processing_time_seconds": round(time.time() - start, 3),
                }
        except Exception as e:
            log.warning(f"Failed to fetch JD for job_id={job_id}: {e}")
            return {
                "error": f"Failed to fetch job description: {str(e)}",
                "job_id": job_id,
                "uid": uid,
                "per_answer_evaluations": [],
                "overall_summary": {},
                "interviewer_feedback": {},
                "processing_time_seconds": round(time.time() - start, 3),
            }
    if not jd_text:
        return {
            "error": "job_id or job_description is required",
            "per_answer_evaluations": [],
            "overall_summary": {},
            "interviewer_feedback": {},
            "processing_time_seconds": 0,
        }

    # Resolve transcript / Q&A pairs
    qa_pairs: List[Tuple[str, str]] = []
    if interviewer_questions and isinstance(interviewer_questions, list) and len(interviewer_questions) > 0:
        qa_pairs = [
            (q.get("question") or q.get("q") or "", q.get("answer") or q.get("a") or "")
            for q in interviewer_questions
        ]
        qa_pairs = [(q, a) for q, a in qa_pairs if q or a]
    elif session_id:
        try:
            from agents.interview_chroma import InterviewChromaManager
            chroma_mgr = InterviewChromaManager()
            session_data = chroma_mgr.get_interview_session(session_id)
            if session_data and session_data.get("conversation_history"):
                qa_pairs = _conversation_history_to_qa_pairs(session_data["conversation_history"])
            else:
                return {
                    "error": f"No conversation history found for session_id={session_id}",
                    "job_id": job_id,
                    "uid": uid,
                    "per_answer_evaluations": [],
                    "overall_summary": {},
                    "interviewer_feedback": {},
                    "processing_time_seconds": round(time.time() - start, 3),
                }
        except Exception as e:
            log.warning(f"Failed to fetch session {session_id}: {e}")
            return {
                "error": f"Failed to fetch interview session: {str(e)}",
                "job_id": job_id,
                "uid": uid,
                "per_answer_evaluations": [],
                "overall_summary": {},
                "interviewer_feedback": {},
                "processing_time_seconds": round(time.time() - start, 3),
            }
    elif transcript is not None:
        turns = _parse_transcript_into_turns(transcript)
        qa_pairs = _group_turns_into_qa_pairs(turns)
    else:
        return {
            "error": "One of interviewer_questions, transcript, or session_id must be provided",
            "job_id": job_id,
            "uid": uid,
            "per_answer_evaluations": [],
            "overall_summary": {},
            "interviewer_feedback": {},
            "processing_time_seconds": round(time.time() - start, 3),
        }

    if not qa_pairs:
        return {
            "error": "No question-answer pairs found in transcript",
            "per_answer_evaluations": [],
            "overall_summary": {},
            "interviewer_feedback": {},
            "processing_time_seconds": round(time.time() - start, 3),
        }

    # Per-answer evaluation
    per_answer_results: List[Dict[str, Any]] = []
    jd_items_covered: List[str] = []

    for idx, (question, answer) in enumerate(qa_pairs):
        qid = f"Q{idx + 1}"
        prompt = _build_per_answer_prompt(jd_text, question, answer, qid)
        try:
            raw = await invoke_llm(
                prompt,
                agent_name="interview_transcript_evaluator",
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=2048,
            )
            parsed = _extract_json_from_response(raw) or _safe_json_loads(raw)
            if isinstance(parsed, dict) and parsed.get("question_id"):
                per_answer_results.append(parsed)
                for item in parsed.get("jd_alignment") or []:
                    jd_item = item.get("jd_item") if isinstance(item, dict) else None
                    if jd_item:
                        jd_items_covered.append(jd_item)
        except Exception as e:
            log.warning(f"Per-answer eval failed for {qid}: {e}")
            per_answer_results.append({
                "question_id": qid,
                "question_text": question[:200],
                "answer_summary": "Evaluation failed",
                "relevance_score": 0,
                "depth_score": 0,
                "evidence_score": 0,
                "communication_score": 0,
                "jd_alignment": [],
                "strengths": [],
                "concerns": [str(e)[:200]],
                "follow_up_suggestions": [],
            })

    # Overall summary
    overall = {}
    try:
        overall_prompt = _build_overall_prompt(jd_text, per_answer_results)
        raw = await invoke_llm(
            overall_prompt,
            agent_name="interview_transcript_evaluator",
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            max_output_tokens=1024,
        )
        overall = _extract_json_from_response(raw) or _safe_json_loads(raw) or {}
    except Exception as e:
        log.warning(f"Overall summary failed: {e}")
        overall = {
            "strengths_vs_jd": [],
            "gaps_vs_jd": [],
            "risk_factors": [],
            "recommendation": "Unable to assess",
            "recommendation_rationale": str(e)[:300],
        }

    # Interviewer feedback (optional)
    interviewer_fb = {}
    if include_interviewer_feedback:
        try:
            questions_asked = [q for q, _ in qa_pairs]
            fb_prompt = _build_interviewer_feedback_prompt(jd_text, questions_asked, jd_items_covered)
            raw = await invoke_llm(
                fb_prompt,
                agent_name="interview_transcript_evaluator",
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=1024,
            )
            interviewer_fb = _extract_json_from_response(raw) or _safe_json_loads(raw) or {}
        except Exception as e:
            log.warning(f"Interviewer feedback failed: {e}")
            interviewer_fb = {
                "jd_coverage_gaps": [],
                "question_quality_notes": [],
                "suggested_follow_up_questions": [],
            }

    elapsed = round(time.time() - start, 3)
    log.info(f"Interview transcript evaluation complete: job_id={job_id} uid={uid} {len(per_answer_results)} answers in {elapsed}s")

    return {
        "job_id": job_id,
        "uid": uid,
        "per_answer_evaluations": per_answer_results,
        "overall_summary": overall,
        "interviewer_feedback": interviewer_fb,
        "processing_time_seconds": elapsed,
        "total_qa_pairs": len(qa_pairs),
    }

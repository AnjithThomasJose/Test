"""
Question generation for interview agent.

- Default: topic-based, one question at a time (aiinterview-style).
- Generic tests (psychometric, personality, communication): dedicated behavioral-only prompt
  so the model never asks definition/meta questions like "how do you define X?" or "what is a X?".
"""

import json
import logging
from typing import Any, Dict, List

from .streaming_config import StreamingConfig

log = logging.getLogger(__name__)

# Topic keywords that use the generic-test (behavioral-only) prompt
GENERIC_TEST_KEYWORDS = [
    "psychometric", "psychological", "behavioral", "aptitude", "personality",
    "personality test", "personality assessment", "trait",
    "communication", "communication test", "communication skills",
    "speaking", "presentation", "listening",
]


def _is_generic_test_topic(topic: str) -> bool:
    """True if topic is a generic test type (psychometric, personality, communication)."""
    if not topic or not topic.strip():
        return False
    t = topic.strip().lower()
    return any(kw in t for kw in GENERIC_TEST_KEYWORDS)


def _resume_summary(resume: Dict[str, Any]) -> str:
    """Build a short text summary of the resume for the prompt."""
    if not resume:
        return "No resume provided."
    parts = []
    name = resume.get("name") or resume.get("Name")
    if name:
        parts.append(f"Candidate: {name}")
    email = resume.get("Email") or resume.get("email")
    if email:
        parts.append(f"Email: {email}")
    exp_years = resume.get("total_experience_years")
    if exp_years is not None:
        parts.append(f"Total experience: {exp_years} years")
    skills = resume.get("skills") or resume.get("Skills") or []
    if skills:
        names = []
        for s in skills:
            if isinstance(s, dict):
                names.append(s.get("SkillName") or s.get("skill_name") or s.get("name"))
            elif isinstance(s, str):
                names.append(s)
        parts.append("Skills: " + ", ".join(filter(None, names)))
    exp = resume.get("experience") or resume.get("projects") or []
    if exp:
        jobs = []
        for e in exp:
            if isinstance(e, dict):
                jobs.append(f"{e.get('job_title', e.get('title', ''))} at {e.get('company', e.get('company_name', ''))}")
        if jobs:
            parts.append("Experience: " + "; ".join(filter(None, jobs)))
    return "\n".join(parts) if parts else json.dumps(resume)[:2000]


def _system_instruction_default(resume: Dict[str, Any], topic: str) -> str:
    """Default topic-based system instruction."""
    summary = _resume_summary(resume)
    return (
        "You are a professional interviewer conducting a topic-based interview. "
        f"The interview topic is: {topic}. "
        "Ask one clear question at a time. Be concise and relevant to the topic. "
        "When this is the first question (no prior messages), you MUST start your reply with 'Hello [candidate name],' then ask the question. "
        "Candidate context from their resume:\n" + summary
    )


def _system_instruction_generic_test(resume: Dict[str, Any], topic: str) -> str:
    """
    Dedicated prompt for generic tests (psychometric, personality, communication).
    Behavioral/scenario only; never ask definition or meta questions.
    """
    summary = _resume_summary(resume)
    return (
        "You are a professional interviewer conducting a behavioral interview. "
        f"The interview focus is: {topic}. "
        "CRITICAL: Do NOT ask the candidate to define the topic or the test (e.g. no 'how do you define X?', 'what is a X?', 'what comes to mind when you hear X?'). "
        "Do NOT mention 'test', 'assessment', or 'section' in your question. "
        "Ask only scenario-based or past-behavior questions (e.g. 'Tell me about a time when...', 'How did you handle...', 'Describe a situation where...'). "
        "Ask exactly ONE complete question that ends with a question mark. "
        "When this is the first question (no prior messages), you MUST start your reply with 'Hello [candidate name],' then the question. For follow-ups, start directly with the question. "
        "No filler (no 'Thanks for sharing', 'Let\'s move on'). "
        "Candidate context from their resume:\n" + summary
    )


def _system_instruction(resume: Dict[str, Any], topic: str) -> str:
    """Pick system instruction: generic-test prompt for psychometric/personality/communication, else default."""
    if _is_generic_test_topic(topic):
        return _system_instruction_generic_test(resume, topic)
    return _system_instruction_default(resume, topic)


def build_messages(
    conversation_history: List[Dict[str, str]],
    resume: Dict[str, Any],
    topic: str,
    candidate_name: str,
) -> List[Dict[str, str]]:
    """
    Build messages for question generation.
    Uses generic-test prompt when topic is psychometric/personality/communication, else default.
    """
    topic = (topic or "general").strip()
    name = (candidate_name or "there").strip() or "there"
    system_content = _system_instruction(resume or {}, topic)

    if not conversation_history:
        user_content = (
            f"This is the first question. Start the interview on the topic: {topic}. "
            f"Your reply MUST start with exactly: Hello {name}, (then a comma and space, then your one interview question). "
            f"Do not add anything before 'Hello {name},' and do not add anything after the single question. Output only that one line."
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    messages = [{"role": "system", "content": system_content}]
    for turn in conversation_history:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "model":
            role = "assistant"
        if role not in ("user", "assistant"):
            role = "user"
        messages.append({"role": role, "content": content})
    if messages and messages[-1].get("role") == "assistant":
        messages.append({
            "role": "user",
            "content": "Based on the conversation above, ask your next single interview question.",
        })
    return messages


async def generate_question_nostream(
    conversation_history: List[Dict[str, str]],
    resume: Dict[str, Any],
    topic: str,
    candidate_name: str,
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Generate one interview question (non-streaming). Uses default or generic-test prompt by topic."""
    from . import llm_utils
    messages = build_messages(conversation_history, resume or {}, topic, candidate_name)
    result = await llm_utils.invoke_llm(
        messages,
        model=model,
        enforce_json=False,
        max_tokens=StreamingConfig.QUESTION_MAX_TOKENS,
        temperature=0.7,
        max_retries=1,
    )
    if not result.get("ok"):
        return {"question": "", "error": result.get("error", "LLM call failed")}
    raw = (result.get("raw") or "").strip()
    question = raw.split("\n")[0].strip() if raw else ""
    return {"question": question}

"""
JD Enhancer Agent

Enhances raw job description text for candidate-facing display (before validation/parsing).
Used when enhance_jd_only is true: returns enhanced text so the user can send it again for validation.
"""

import logging
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from settings import settings
from core.config import get_agent_config

log = logging.getLogger(__name__)

config = get_agent_config("jd_enhancer")
ENHANCER_TIMEOUT_SECONDS = getattr(config, "timeout_seconds", 90)
ENHANCER_MAX_RETRIES = getattr(config, "llm_retry_attempts", 2)
ENHANCER_MAX_CHARS = getattr(config, "max_prompt_chars", 30000)

SYSTEM_PROMPT = """You are an expert at improving job descriptions for candidates.

Your task: Given a job description (which may be from a file or manually entered), produce a single enhanced version that:

1. **Preserves all factual content** – Do not add, remove, or change facts (role, requirements, skills, salary, company, etc.).
2. **Improves structure** – Use clear sections such as: About the role / Position summary, Responsibilities, Requirements / Qualifications, Preferred skills, Benefits, Application instructions (if present).
3. **Improves clarity** – Fix grammar, normalize bullets and headings, use consistent formatting.
4. **Candidate-friendly** – Professional, neutral tone; easy to scan.

Output ONLY the enhanced job description text. No preamble, no "Here is the enhanced version", no meta-commentary. Start directly with the first section heading or the first line of the JD."""


def _truncate_for_llm(raw_text: str, max_chars: int = None) -> str:
    """Truncate JD text to fit within LLM limits."""
    if not raw_text:
        return ""
    limit = max_chars or ENHANCER_MAX_CHARS
    if len(raw_text) <= limit:
        return raw_text.strip()
    s = raw_text[:limit]
    last_newline = s.rfind("\n")
    if last_newline > limit * 0.8:
        s = s[: last_newline + 1]
    log.debug("JD enhancer: truncated input from %d to %d chars", len(raw_text), len(s))
    return s.strip()


def _create_groq_model() -> ChatGroq:
    return ChatGroq(
        model=settings.GROQ_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0.2,
        max_retries=ENHANCER_MAX_RETRIES,
        timeout=ENHANCER_TIMEOUT_SECONDS,
        model_kwargs={"top_p": 0.2, "max_completion_tokens": 8000},
    )


async def enhance_jd_text(jd_text: str) -> Optional[str]:
    """
    Enhance raw JD text for candidate display. Preserves facts; improves structure and clarity.

    Args:
        jd_text: Raw job description (from file or manual input).

    Returns:
        Enhanced JD string, or None on failure.
    """
    if not jd_text or not str(jd_text).strip():
        log.warning("JD enhancer: empty input")
        return None

    text = _truncate_for_llm(str(jd_text).strip())
    if not text:
        return None

    model = _create_groq_model()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=text),
    ]

    # Issue 5.1: Use safe_llm_call for retry, timeout, and rate limit handling
    try:
        from utils.llm_error_handler import safe_llm_call, LLMError
        
        response = await safe_llm_call(
            lambda: model.ainvoke(messages),
            timeout=ENHANCER_TIMEOUT_SECONDS,
            max_retries=ENHANCER_MAX_RETRIES,
            agent_name="jd_enhancer"
        )
        out = getattr(response, "content", None) or ""
        out = (out or "").strip()
        if not out:
            log.warning("JD enhancer: empty LLM response")
            return None
        log.info("JD enhancer: produced %d chars", len(out))
        return out
    except LLMError as e:
        log.error("JD enhancer LLM error: %s", e, exc_info=True)
        return None
    except Exception as e:
        log.error("JD enhancer failed: %s", e, exc_info=True)
        return None

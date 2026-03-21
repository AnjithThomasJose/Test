"""
Enhanced Role Fit Agent

Runs before skill_and_career_advisor on the second call (with user interest answers).
Produces enhanced role fit: for each suggested role, why it was suggested, skill gaps
to reach that role, and how to overcome those gaps. Output is stored in state["enhanced_role_fit"]
and consumed by the skill_and_career_advisor agent.
"""

import json
import logging
import asyncio
import time
from typing import Dict, Any, List

from models.llm_invoker import invoke_llm, invoke_structured_llm
from core.model_registry import TaskType
from langsmith.run_helpers import traceable

from agents.prompt_generator import generate_enhanced_role_fit_prompt
from core.utils import (
    _to_text,
    _extract_json_from_response,
    _calculate_processing_time,
    create_optimized_career_advisor_inputs,
)
from core.config import get_agent_config
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion

log = logging.getLogger(__name__)

config = get_agent_config("enhanced_role_fit")
TIMEOUT_SECONDS = getattr(config, "timeout_seconds", 90)
FALLBACK_TIMEOUT_SECONDS = getattr(config, "timeout_seconds", 90) * 2  # Fallback gets 2x (last resort)
MAX_PROMPT_CHARS = getattr(config, "max_prompt_chars", 8000)
MAX_RESPONSE_LENGTH = getattr(config, "max_response_length", 6000)


@traceable(name="enhanced_role_fit_agent")
async def enhanced_role_fit_agent(
    state: Dict[str, Any], tenant_id: str = "default_tenant"
) -> Dict[str, Any]:
    """
    Generate enhanced role fit (career paths with why suggested, skill gaps, how to overcome).
    Runs on 2nd call before skill_and_career_advisor. Writes state["enhanced_role_fit"].
    """
    log_context = create_log_context("enhanced_role_fit", tenant_id)
    start_time = log_context["start_time"]

    optimized = create_optimized_career_advisor_inputs(state)
    structured_resume = optimized.get("structured_resume", {})
    user_interests = optimized.get("user_interests", [])

    if not structured_resume and not user_interests:
        log.warning("Enhanced role fit: no resume or interests; returning empty enhanced_role_fit")
        return {
            "enhanced_role_fit": [],
            "status": "completed",
            "node": "enhanced_role_fit",
        }

    log.info("Enhanced role fit: has resume and interests, calling LLM (structured output preferred)")

    def _normalize_role_item(item: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return None
        fit_pct = item.get("fit_percentage")
        if fit_pct is not None:
            try:
                fit_pct = max(0, min(100, int(fit_pct)))
            except (TypeError, ValueError):
                fit_pct = 0
        else:
            fit_pct = 0
        lacking = (item.get("where_candidate_is_lacking") or item.get("lacking_rationale") or "").strip()
        return {
            "role": item.get("role", "") or item.get("title", ""),
            "why_suggested": item.get("why_suggested", "") or item.get("rationale", ""),
            "fit_percentage": fit_pct,
            "where_candidate_is_lacking": lacking[:500] if lacking else "",
            "skill_gaps_to_reach_role": item.get("skill_gaps_to_reach_role") or item.get("skill_gaps", []) or [],
            "how_to_overcome_gaps": item.get("how_to_overcome_gaps") or item.get("how_to_overcome", []) or [],
            "platform_actions": item.get("platform_actions") or [],
        }

    def _parse_enhanced_list(parsed: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        raw = None
        if isinstance(parsed, dict):
            raw = parsed.get("enhanced_role_fit") or parsed.get("role_fit") or parsed.get("roles")
        if isinstance(parsed, list):
            raw = parsed
        if isinstance(raw, list):
            for item in raw:
                n = _normalize_role_item(item) if isinstance(item, dict) else None
                if n and (n.get("role") or n.get("why_suggested")):
                    out.append(n)
        return out

    enhanced_list: List[Dict[str, Any]] = []
    try:
        prompt = generate_enhanced_role_fit_prompt(structured_resume, user_interests)
        if len(prompt) > MAX_PROMPT_CHARS:
            prompt = prompt[:MAX_PROMPT_CHARS]

        # Prefer structured output (Gemini) for reliable JSON shape when API key is set
        try:
            from pydantic import BaseModel, Field
            from typing import List as _List
            from settings import settings as _settings

            api_key = getattr(_settings, "GOOGLE_API_KEY", None) if _settings else None
            if not api_key or not str(api_key).strip():
                raise ValueError("GOOGLE_API_KEY not set; using invoke_llm fallback")

            class RoleFitItem(BaseModel):
                role: str = ""
                why_suggested: str = ""
                fit_percentage: int = Field(default=0, ge=0, le=100, description="0-100 how close candidate is to this role")
                where_candidate_is_lacking: str = Field(default="", max_length=500, description="1-3 sentences on where candidate falls short")
                skill_gaps_to_reach_role: _List[str] = Field(default_factory=list)
                how_to_overcome_gaps: _List[str] = Field(default_factory=list)
                platform_actions: _List[str] = Field(default_factory=list, description="Platform-specific actions matched to gaps (e.g., recommended assessments/courses, resume download for ATS/format)")

            class EnhancedRoleFitResponse(BaseModel):
                enhanced_role_fit: _List[RoleFitItem] = Field(default_factory=list)

            structured_result = await invoke_structured_llm(
                prompt,
                EnhancedRoleFitResponse,
                task_type=TaskType.COMPLEX_REASONING,
                preferred_model="gemini-2.5-flash",
                agent_name="enhanced_role_fit",
                max_retries=2,
                max_output_tokens=MAX_RESPONSE_LENGTH,
                temperature=0.2,
                timeout=float(TIMEOUT_SECONDS),
                raise_on_fallback=False,
            )
            raw_list = getattr(structured_result, "enhanced_role_fit", None) or []
            for r in raw_list:
                fit_pct = getattr(r, "fit_percentage", 0)
                if not isinstance(fit_pct, int):
                    try:
                        fit_pct = max(0, min(100, int(fit_pct)))
                    except (TypeError, ValueError):
                        fit_pct = 0
                else:
                    fit_pct = max(0, min(100, fit_pct))
                lacking = (getattr(r, "where_candidate_is_lacking", "") or "").strip()[:500]
                platform_actions = list(getattr(r, "platform_actions", []) or [])
                enhanced_list.append({
                    "role": getattr(r, "role", "") or "",
                    "why_suggested": getattr(r, "why_suggested", "") or "",
                    "fit_percentage": fit_pct,
                    "where_candidate_is_lacking": lacking,
                    "skill_gaps_to_reach_role": list(getattr(r, "skill_gaps_to_reach_role", []) or []),
                    "how_to_overcome_gaps": list(getattr(r, "how_to_overcome_gaps", []) or []),
                    "platform_actions": platform_actions,
                })
            if not enhanced_list:
                log.warning("Enhanced role fit: structured output returned 0 roles (model may have returned empty list)")
        except (Exception, asyncio.CancelledError) as struct_err:
            # CancelledError (BaseException) can occur when parent times out; treat as transient
            log.warning("Enhanced role fit structured output failed, using invoke_llm fallback: %s", struct_err, exc_info=True)
            # Same pattern as interest_filler: fallback prompt must demand JSON-only so extraction works
            fallback_json_instruction = "\n\nYou must respond with ONLY a valid JSON object (no markdown, no code fence, no explanation). Use the exact key \"enhanced_role_fit\" with an array of role objects as in the structure above."
            fallback_prompt = prompt + fallback_json_instruction
            llm_response = await asyncio.wait_for(
                invoke_llm(
                    prompt=fallback_prompt,
                    task_type="enhanced_role_fit",
                    agent_name="enhanced_role_fit",
                    response_mime_type="application/json",
                    preferred_model="gemini-2.5-flash",
                    max_output_tokens=2000,
                ),
                timeout=FALLBACK_TIMEOUT_SECONDS,
            )
            text = _to_text(llm_response)
            parsed = _extract_json_from_response(text, MAX_RESPONSE_LENGTH)
            enhanced_list = _parse_enhanced_list(parsed)
            log.info("Enhanced role fit: fallback parse got %s roles", len(enhanced_list))
            if not enhanced_list and text.strip():
                log.warning("Enhanced role fit: LLM returned text but no valid JSON/list. parsed=%s, text_len=%s", type(parsed).__name__, len(text))

        if not enhanced_list:
            log.warning("Enhanced role fit: no roles produced; returning empty list")

        processing_time = _calculate_processing_time(start_time)
        log_agent_completion(
            log_context,
            {"success": True, "roles_count": len(enhanced_list)},
            "llm",
            processing_time,
        )

        return {
            "enhanced_role_fit": enhanced_list,
            "status": "completed",
            "node": "enhanced_role_fit",
        }
    except asyncio.TimeoutError:
        processing_time = _calculate_processing_time(start_time)
        AgentLogger.log_error(log_context, "Enhanced role fit LLM timeout", processing_time)
        return {
            "enhanced_role_fit": [],
            "status": "completed",
            "node": "enhanced_role_fit",
        }
    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        AgentLogger.log_error(log_context, f"Enhanced role fit failed: {e}", processing_time)
        return {
            "enhanced_role_fit": [],
            "status": "completed",
            "node": "enhanced_role_fit",
        }

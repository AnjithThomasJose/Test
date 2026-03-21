"""
summary_generator.py

Simple narrative report generator for the interview agent.

Responsibilities:
- Compress conversation history
- Call LLM to generate narrative text only
- Return simple JSON: { "report": "<string>" }

Return Shape:
    {
        "ok": bool,  # True if LLM succeeded and validation passed
        "raw": str,  # Raw LLM response text
        "report": { "report": "<string>" }  # Narrative text only
    }
    
    On failure (ok=False), report contains fallback message.
"""

from typing import Any, Dict, List, Optional
import logging
import json
import yaml
from pydantic import BaseModel, Field, ValidationError

from . import llm_utils
from .compression_utils import compress_conversation_history
from .config import load_prompts
from utils.llm_json_sanitizer import extract_json
from utils.llm_telemetry import log_invoke_llm

log = logging.getLogger(__name__)


# ----------------------------
# Minimal Pydantic schema
# ----------------------------
class NarrativeReport(BaseModel):
    """Minimal schema for narrative report output."""
    report: str = Field(..., min_length=1)


class StructuredInterviewReport(BaseModel):
    """Schema for structured UI-formatted interview report."""
    overall_interview_signal: Dict[str, str] = Field(..., description="Overall signal with level and description")
    key_skill_breakdowns: List[Dict[str, str]] = Field(..., description="3 skills: Problem Solving, Communication, Decision Judgment")
    observed_strengths: List[str] = Field(..., description="List of specific strengths")
    improvement_opportunities: List[str] = Field(..., description="List of specific improvement areas")
    personalized_recommendations: List[str] = Field(..., description="List of actionable recommendations")


# ----------------------------
# Prompt loader
# ----------------------------
def _load_prompts(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load prompts from YAML file or config.
    
    Args:
        path: Optional path to prompts.yaml. If None, uses config.load_prompts()
        
    Returns:
        Dict with prompt templates
    """
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data
        except FileNotFoundError:
            log.warning("prompts.yaml not found at %s; using config.load_prompts()", path)
        except Exception as e:
            log.exception("Failed to load prompts.yaml from %s: %s", path, e)
    
    # Fallback to config.load_prompts()
    try:
        return load_prompts()
    except Exception as e:
        log.exception("Failed to load prompts via config: %s", e)
        return {}


def _safe_format_template(template: str, placeholders: Dict[str, Any]) -> str:
    """
    Safely format a template string, handling edge cases.
    """
    try:
        return template.format(**placeholders)
    except KeyError as e:
        missing_key = str(e).strip("'\"")
        log.warning(f"Missing placeholder in template: '{missing_key}'")
        # Use safe_format utility to handle missing placeholders gracefully
        from utils.safe_format import safe_format
        return safe_format(template, placeholders)
    except ValueError as e:
        log.warning(f"Template format error: {e}")
        # Attempt to escape single braces that aren't placeholders
        try:
            escaped = template.replace("{", "{{").replace("}", "}}")
            for key, value in placeholders.items():
                escaped = escaped.replace(f"{{{{{key}}}}}", f"{{{key}}}")
            return escaped.format(**placeholders)
        except Exception:
            # Last resort: use safe_format
            from utils.safe_format import safe_format
            try:
                return safe_format(template, placeholders)
            except Exception:
                # Absolute last resort: return template as-is
                return template


# ----------------------------
# Public API
# ----------------------------
async def generate_interview_summary(
    *,
    session: Dict[str, Any],
    evaluation_summary: Dict[str, Any],
    conversation_context: Optional[Dict[str, Any]] = None,
    mode: str = "interview_summary",  # Ignored - always uses narrative_report_prompt
    prompts_path: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate a simple narrative report.
    
    Args:
        session: session dict containing at least 'conversation_history' (list)
        evaluation_summary: output from evaluator (ignored - evaluator produces all actionable content)
        conversation_context: optional additional context (ignored)
        mode: ignored - always uses narrative_report_prompt
        prompts_path: path to prompts.yaml
        model: model name passed to llm_utils
        client: optional custom LLM client to pass to llm_utils
    
    Returns:
        {
            "ok": bool,
            "raw": str,
            "report": { "report": "<string>" }
        }
    """
    prompts = _load_prompts(prompts_path)

    # Prepare compressed history
    history = session.get("conversation_history", []) if session else []
    compressed = ""
    try:
        compressed_result = compress_conversation_history(history, max_recent_turns=10, max_message_length=300)
        
        # Handle both List[Dict] and string returns
        if isinstance(compressed_result, str):
            compressed_list = [{"role": "assistant", "content": compressed_result}]
        elif isinstance(compressed_result, list):
            compressed_list = compressed_result
        else:
            log.warning(f"Unexpected compression result type: {type(compressed_result)}")
            compressed_list = history[-20:] if history else []
        
        # Format as "role: content" to preserve role information
        compressed = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" 
            for m in compressed_list 
            if isinstance(m, dict)
        ])
    except Exception as e:
        log.warning(f"History compression failed: {e}")
        # Fallback: join last N messages with roles preserved
        try:
            compressed = "\n".join([
                f"{m.get('role', 'unknown')}: {m.get('content', '')}" 
                for m in history[-20:] 
                if isinstance(m, dict)
            ])
        except Exception:
            compressed = ""

    # Build placeholders - only compressed_history is needed
    placeholders = {
        "compressed_history": compressed,
    }

    # Always use narrative_report_prompt
    prompt_key = "narrative_report_prompt"
    
    # Validate prompt key exists
    if prompt_key not in prompts:
        log.warning(f"Prompt key '{prompt_key}' not found in prompts; available keys: {list(prompts.keys())[:5]}")
        # Fallback: try to find any report/summary prompt
        fallback_key = next((k for k in prompts.keys() if "narrative" in k.lower() or "report" in k.lower() or "summary" in k.lower()), None)
        if fallback_key:
            log.info(f"Using fallback prompt key: {fallback_key}")
            prompt_key = fallback_key
        else:
            # Last resort: return error
            log.error(f"No suitable prompt found. Available keys: {list(prompts.keys())[:10]}")
            return {
                "ok": False,
                "raw": "",
                "report": {"report": "Interview summary unavailable. Prompt configuration error."}
            }

    # Build prompt - simple user message only (no system message)
    template = prompts.get(prompt_key)
    if not template:
        log.error(f"Prompt template '{prompt_key}' is empty")
        return {
            "ok": False,
            "raw": "",
            "report": {"report": "Interview summary unavailable. Prompt template error."}
        }
    
    user_content = _safe_format_template(template, placeholders)
    messages = [{"role": "user", "content": user_content}]

    # Call LLM via llm_utils with telemetry
    import time
    llm_start_ts = time.time()
    session_id = session.get("session_id") or session.get("metadata", {}).get("session_id")
    
    try:
        llm_ret = await llm_utils.invoke_llm(messages, model=model, enforce_json=True, client=client)
    except Exception as e:
        log.exception("LLM invocation failed: %s", e)
        llm_ret = {"ok": False, "raw": "", "json": None, "error": str(e)}
    
    # Log telemetry
    json_valid = bool(llm_ret.get("json"))
    try:
        log_invoke_llm(
            session_id=session_id,
            stage="narrative_summary",
            model=model,
            start_ts=llm_start_ts,
            ok=llm_ret.get("ok", False),
            json_valid=json_valid,
            fallback_used=False
        )
    except Exception:
        log.debug("Telemetry logging failed; continuing")
    
    raw = llm_ret.get("raw", "")
    parsed = llm_ret.get("json")
    
    # If LLM call failed or returned invalid JSON, try JSON sanitizer
    if not llm_ret.get("ok") or not parsed or not isinstance(parsed, dict):
        error_msg = llm_ret.get("error", "unknown_error")
        if not llm_ret.get("ok"):
            log.warning("LLM failed to generate narrative report: %s", error_msg)
        
        # Try JSON sanitizer if we have raw content
        if raw:
            try:
                sanitized = extract_json(raw)
                if sanitized and isinstance(sanitized, dict):
                    parsed = sanitized
                    log.info("JSON sanitizer recovered valid JSON from raw output")
                    llm_ret["json"] = parsed
                    llm_ret["ok"] = True
                else:
                    log.warning("JSON sanitizer could not extract valid JSON from raw output")
            except Exception as e:
                log.debug(f"JSON sanitizer failed: {e}")
        else:
            log.warning("LLM returned empty raw content and no JSON")
        
        # Fallback if we still don't have valid JSON
        if not parsed or not isinstance(parsed, dict):
            return {
                "ok": False,
                "raw": raw,
                "report": {"report": "Interview summary unavailable. Please review the conversation history."}
            }
    
    # Validate against NarrativeReport schema
    try:
        # Extract "report" field from parsed JSON
        report_text = parsed.get("report", "")
        if not report_text:
            # Try "summary" as fallback
            report_text = parsed.get("summary", "")
        
        if not report_text:
            # Last resort: use raw text (truncated)
            report_text = str(raw)[:2000] if raw else "Interview summary unavailable."
        
        # Validate with Pydantic
        report_obj = NarrativeReport(report=report_text)
        return {
            "ok": True,
            "raw": raw,
            "report": report_obj.model_dump()
        }
    except ValidationError as ve:
        log.warning("Narrative report JSON did not validate: %s", ve)
        # Fallback: use raw text or parsed report field
        report_text = parsed.get("report", "") if isinstance(parsed, dict) else ""
        if not report_text:
            report_text = str(raw)[:2000] if raw else "Interview summary unavailable. Please review the conversation history."
        
        return {
            "ok": False,
            "raw": raw,
            "report": {"report": report_text}
        }


async def generate_structured_interview_report(
    *,
    session: Dict[str, Any],
    evaluation_summary: Dict[str, Any],
    job_title: str = "",
    interview_topic: str = "",
    prompts_path: Optional[str] = None,
    model: Optional[str] = None,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Generate a structured interview report matching the UI format.
    
    Args:
        session: session dict containing at least 'conversation_history' (list)
        evaluation_summary: output from evaluator with scores, strengths, weaknesses
        job_title: job title for the interview
        interview_topic: interview topic/domain
        prompts_path: path to prompts.yaml
        model: model name passed to llm_utils
        client: optional custom LLM client to pass to llm_utils
    
    Returns:
        {
            "ok": bool,
            "raw": str,
            "report": {
                "overall_interview_signal": {"level": "...", "description": "..."},
                "key_skill_breakdowns": [...],
                "observed_strengths": [...],
                "improvement_opportunities": [...],
                "personalized_recommendations": [...]
            }
        }
    """
    prompts = _load_prompts(prompts_path)

    # Prepare compressed history
    history = session.get("conversation_history", []) if session else []
    compressed = ""
    try:
        compressed_result = compress_conversation_history(history, max_recent_turns=10, max_message_length=300)
        
        if isinstance(compressed_result, str):
            compressed_list = [{"role": "assistant", "content": compressed_result}]
        elif isinstance(compressed_result, list):
            compressed_list = compressed_result
        else:
            compressed_list = history[-20:] if history else []
        
        compressed = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" 
            for m in compressed_list 
            if isinstance(m, dict)
        ])
    except Exception as e:
        log.warning(f"History compression failed: {e}")
        try:
            compressed = "\n".join([
                f"{m.get('role', 'unknown')}: {m.get('content', '')}" 
                for m in history[-20:] 
                if isinstance(m, dict)
            ])
        except Exception:
            compressed = ""

    # Extract evaluation data
    overall_score = evaluation_summary.get("overall_score", 0.0)
    category_scores = evaluation_summary.get("category_scores", {})
    strengths = evaluation_summary.get("strengths", [])
    weaknesses = evaluation_summary.get("weaknesses", [])
    summary = evaluation_summary.get("summary", "")

    # Build placeholders
    placeholders = {
        "overall_score": overall_score,
        "category_scores": str(category_scores),
        "strengths": str(strengths),
        "weaknesses": str(weaknesses),
        "summary": summary,
        "job_title": job_title,
        "interview_topic": interview_topic,
        "compressed_history": compressed,
    }

    prompt_key = "structured_interview_report_prompt"
    
    if prompt_key not in prompts:
        log.error(f"Prompt key '{prompt_key}' not found in prompts")
        return {
            "ok": False,
            "raw": "",
            "report": {
                "overall_interview_signal": {"level": "Unknown", "description": "Report generation failed."},
                "key_skill_breakdowns": [],
                "observed_strengths": ["Demonstrated engagement in the interview process"],
                "improvement_opportunities": ["Continue building domain-specific knowledge"],
                "personalized_recommendations": ["Practice articulating technical decisions with clear trade-offs"]
            }
        }

    template = prompts.get(prompt_key)
    if not template:
        log.error(f"Prompt template '{prompt_key}' is empty")
        return {
            "ok": False,
            "raw": "",
            "report": {
                "overall_interview_signal": {"level": "Unknown", "description": "Report generation failed."},
                "key_skill_breakdowns": [],
                "observed_strengths": ["Demonstrated engagement in the interview process"],
                "improvement_opportunities": ["Continue building domain-specific knowledge"],
                "personalized_recommendations": ["Practice articulating technical decisions with clear trade-offs"]
            }
        }
    
    user_content = _safe_format_template(template, placeholders)
    messages = [{"role": "user", "content": user_content}]

    # Call LLM
    import time
    llm_start_ts = time.time()
    session_id = session.get("session_id") or session.get("metadata", {}).get("session_id")
    
    try:
        llm_ret = await llm_utils.invoke_llm(messages, model=model, enforce_json=True, client=client)
    except Exception as e:
        log.exception("LLM invocation failed: %s", e)
        llm_ret = {"ok": False, "raw": "", "json": None, "error": str(e)}
    
    raw = llm_ret.get("raw", "")
    parsed = llm_ret.get("json")
    
    # Try JSON sanitizer if needed
    if not llm_ret.get("ok") or not parsed or not isinstance(parsed, dict):
        if raw:
            try:
                sanitized = extract_json(raw)
                if sanitized and isinstance(sanitized, dict):
                    parsed = sanitized
                    log.info("JSON sanitizer recovered valid JSON from raw output")
                    llm_ret["json"] = parsed
                    llm_ret["ok"] = True
            except Exception:
                pass
        
        if not parsed or not isinstance(parsed, dict):
            # Fallback: build minimal structure from evaluation_summary
            log.warning("Failed to generate structured report, using fallback")
            return {
                "ok": False,
                "raw": raw,
                "report": {
                    "overall_interview_signal": {
                        "level": "Evaluation completed",
                        "description": summary or "Interview evaluation completed."
                    },
                    "key_skill_breakdowns": [
                        {
                            "skill": "Problem Solving",
                            "rating": "Moderate",
                            "description": "Based on evaluation scores."
                        },
                        {
                            "skill": "Communication",
                            "rating": "Moderate",
                            "description": "Based on evaluation scores."
                        },
                        {
                            "skill": "Decision Judgment",
                            "rating": "Moderate",
                            "description": "Based on evaluation scores."
                        }
                    ],
                    "observed_strengths": strengths if isinstance(strengths, list) else [],
                    "improvement_opportunities": weaknesses if isinstance(weaknesses, list) else [],
                    "personalized_recommendations": []
                }
            }
    
    # Validate and return
    try:
        # Ensure all required fields exist
        observed_strengths_raw = parsed.get("observed_strengths", [])
        # Ensure observed_strengths is never empty - add fallback if needed
        if not observed_strengths_raw or len(observed_strengths_raw) == 0:
            observed_strengths_raw = ["Demonstrated engagement in the interview process"]
        
        report_data = {
            "overall_interview_signal": parsed.get("overall_interview_signal", {
                "level": "Unknown",
                "description": "Report generation incomplete."
            }),
            "key_skill_breakdowns": parsed.get("key_skill_breakdowns", []),
            "observed_strengths": observed_strengths_raw,
            "improvement_opportunities": parsed.get("improvement_opportunities", []),
            "personalized_recommendations": parsed.get("personalized_recommendations", [])
        }
        
        # Validate with Pydantic
        report_obj = StructuredInterviewReport(**report_data)
        return {
            "ok": True,
            "raw": raw,
            "report": report_obj.model_dump()
        }
    except ValidationError as ve:
        log.warning("Structured report JSON did not validate: %s", ve)
        # Return parsed data even if validation fails
        return {
            "ok": False,
            "raw": raw,
            "report": parsed if isinstance(parsed, dict) else {
                "overall_interview_signal": {"level": "Unknown", "description": "Validation failed."},
                "key_skill_breakdowns": [],
                "observed_strengths": ["Demonstrated engagement in the interview process"],
                "improvement_opportunities": ["Continue building domain-specific knowledge"],
                "personalized_recommendations": ["Practice articulating technical decisions with clear trade-offs"]
            }
        }


# convenience sync wrapper
def generate_interview_summary_sync(*args, **kwargs) -> Dict[str, Any]:
    """
    Synchronous wrapper for generate_interview_summary.

    WARNING: Do not call from the async event loop thread (e.g. FastAPI request
    handlers). This uses fut.result() and will block the event loop. In async
    code use generate_interview_summary() and await it instead. For CLI/scripts
    only.

    If event loop is running, uses run_coroutine_threadsafe with 30s timeout.
    Otherwise, creates new event loop with asyncio.run().
    """
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # Running in async context - use thread-safe execution with timeout
        fut = asyncio.run_coroutine_threadsafe(generate_interview_summary(*args, **kwargs), loop)
        try:
            return fut.result(timeout=30.0)  # 30 second timeout
        except asyncio.TimeoutError:
            log.error("Summary generation timed out after 30s")
            return {
                "ok": False,
                "raw": "",
                "report": {"report": "Summary generation timed out."}
            }
    else:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(generate_interview_summary(*args, **kwargs))

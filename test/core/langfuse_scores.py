"""Langfuse model-based evaluation (LLM-as-judge) for pipeline outputs."""
import asyncio
import logging
from typing import Any, Dict, Optional

from core.utils import schedule_background_task

log = logging.getLogger(__name__)

MODEL_EVAL_TIMEOUT_SECONDS = 30


async def run_model_based_eval(
    trace_id: Optional[str],
    result: Dict[str, Any],
    final_node: Optional[str],
    settings: Any,
) -> None:
    """
    Run an LLM-based evaluation on pipeline output and submit a score to Langfuse.

    Runs in the background; does not block. Only runs if trace_id is set,
    LANGFUSE_MODEL_EVAL_ENABLED is True, and Langfuse is configured.

    Args:
        trace_id: Langfuse trace ID to score
        result: Pipeline result dict (output from the final node)
        final_node: Name of the final node (e.g. resume_scorer, ranker, assessment_evaluator)
        settings: App settings (for GOOGLE_API_KEY, etc.)
    """
    if not trace_id:
        return
    if not getattr(settings, "LANGFUSE_MODEL_EVAL_ENABLED", False):
        return
    if not getattr(settings, "LANGFUSE_PUBLIC_KEY", None):
        return

    try:
        from langfuse import get_client
        from models.llm_invoker import invoke_llm
        from core.model_registry import TaskType
    except ImportError:
        log.debug("Langfuse or LangChain not available for model eval")
        return

    async def _eval():
        try:
            client = get_client()
            if not client or not getattr(client, "_tracing_enabled", True):
                return

            # Build a minimal context for the judge based on final_node
            context_parts = []
            if final_node == "resume_scorer":
                score = result.get("resume_score") or result.get("overall_score")
                summary = (result.get("resume_summary") or result.get("summary"))[:500] if result.get("resume_summary") or result.get("summary") else ""
                if score is not None:
                    context_parts.append(f"Score: {score}")
                if summary:
                    context_parts.append(f"Summary: {summary[:300]}...")
            elif final_node == "ranker":
                top = result.get("top_matches") or result.get("matches") or []
                count = len(top) if isinstance(top, list) else 0
                context_parts.append(f"Matches count: {count}")
            elif final_node == "assessment_evaluator":
                report = result.get("report") or {}
                total = report.get("total_score") if isinstance(report, dict) else None
                if total is not None:
                    context_parts.append(f"Assessment score: {total}")
            else:
                context_parts.append(f"Node: {final_node}")

            if not context_parts:
                context_parts.append(str(result)[:400])

            context_str = "\n".join(context_parts)

            system_prompt = """You are an evaluator. Rate the quality of this LLM pipeline output from 0.0 (poor) to 1.0 (excellent).
Consider: completeness, relevance, coherence, and usefulness. Reply with ONLY a number between 0.0 and 1.0."""

            # Issue 5.3: Add timeout to prevent unbounded LLM evaluation
            try:
                raw = await asyncio.wait_for(
                    invoke_llm(
                        prompt=context_str,
                        task_type=TaskType.TEXT_GENERATION,
                        preferred_model=getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash"),
                        agent_name="langfuse_model_eval",
                        system_instruction=system_prompt,
                    ),
                    timeout=MODEL_EVAL_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                log.warning("Langfuse model eval timed out after %ds for trace %s", MODEL_EVAL_TIMEOUT_SECONDS, trace_id[:8])
                return
            
            content = (raw or "").strip()
            try:
                score_val = float(content.replace(",", ".").split()[0])
                score_val = max(0.0, min(1.0, score_val))
            except (ValueError, IndexError):
                score_val = 0.5

            client.create_score(
                trace_id=trace_id,
                name="model-eval-quality",
                value=score_val,
                data_type="NUMERIC",
                comment=f"LLM judge on {final_node or 'pipeline'} output",
            )
            log.debug("Langfuse model eval score created: %.2f for trace %s", score_val, trace_id[:8])
        except Exception as e:
            log.warning("Langfuse model eval failed: %s", e)

    # Issue 5.3: Use schedule_background_task for proper exception logging
    schedule_background_task(_eval(), "langfuse_model_eval")

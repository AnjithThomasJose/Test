from typing import Dict, Any
from langsmith.run_helpers import traceable
import logging
from chroma import get_resume_doc, upsert_resume_doc
from core.utils import _calculate_processing_time
from core.config import get_agent_config
from core.memory import BaseAgentMemory, get_agent_memory
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion

# Get centralized configuration (uses default if not found)
try:
    config = get_agent_config("resume_summary")
except:
    # Fallback to default config if resume_summary config doesn't exist
    config = get_agent_config("resume_assembler")  # Use similar config

# Custom memory class for resume summary agent (extends base memory)
class ResumeSummaryMemory(BaseAgentMemory):
    def __init__(self, tenant_id: str = "default"):
        super().__init__(tenant_id, config.adaptation_window)

# Use centralized memory management
async def get_resume_summary_memory(tenant_id: str = "default") -> ResumeSummaryMemory:
    """Get or create tenant-scoped resume summary memory."""
    return await get_agent_memory("resume_summary", tenant_id, ResumeSummaryMemory)

log = logging.getLogger(__name__)


@traceable(name="resume_summary_agent")
async def resume_summary_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dedicated agent for generating and caching resume summaries using Gemini LLM.
    
    This agent:
    1. Takes structured_resume from state
    2. Generates/caches resume summary using Gemini (via prompt_generator)
    3. Stores summary in resume_doc for future use
    4. Adds summary to structured_resume in state for downstream agents
    
    Args:
        state: Agent state containing structured_resume and uid
        
    Returns:
        Updated state with resume_summary added to structured_resume
    """
    log_context = create_log_context("resume_summary_agent", state)
    start_time = log_context["start_time"]
    
    tenant_id = state.get("tenant_id", "default_tenant")
    uid = state.get("uid", "")
    
    # ✅ CRITICAL: Get structured_resume from groq_resume_parser output (preferred source)
    # This ensures we use the fresh, complete data from groq_resume_parser
    structured_resume = state.get("structured_resume", {})
    
    # Validate that we have structured_resume from groq_resume_parser
    if structured_resume and isinstance(structured_resume, dict):
        # Check if it has the expected fields from groq_resume_parser
        has_groq_fields = (
            structured_resume.get("name") or 
            structured_resume.get("work_experience") or 
            structured_resume.get("skills") or
            structured_resume.get("education")
        )
        if has_groq_fields:
            log.info(f"✅ RESUME_SUMMARY_AGENT: Using structured_resume from groq_resume_parser output")
        else:
            log.warning("⚠️ RESUME_SUMMARY_AGENT: structured_resume exists but missing expected groq_resume_parser fields")
    
    # Initialize memory
    summary_memory = await get_resume_summary_memory(tenant_id)
    
    # Validate required inputs - check for None, empty dict, or invalid types
    if not structured_resume or structured_resume is None:
        log.warning("⚠️ RESUME_SUMMARY_AGENT: No structured_resume found in state (expected from groq_resume_parser)")
        # Try to get from cache as fallback
        if uid:
            try:
                from core.utils import run_blocking_io
                rdoc = await run_blocking_io(get_resume_doc, uid) or {}
                cached_resume = rdoc.get("structured_resume")
                if isinstance(cached_resume, dict) and (cached_resume.get("name") or cached_resume.get("work_experience")):
                    log.info("✅ RESUME_SUMMARY_AGENT: Using cached structured_resume as fallback")
                    structured_resume = cached_resume
                else:
                    return {
                        "resume_summary_status": "skipped",
                        "resume_summary_error": "No structured_resume in state or cache"
                    }
            except Exception as e:
                log.warning(f"⚠️ RESUME_SUMMARY_AGENT: Failed to get cached resume: {e}")
                return {
                    "resume_summary_status": "skipped",
                    "resume_summary_error": "No structured_resume in state"
                }
        else:
            return {
                "resume_summary_status": "skipped",
                "resume_summary_error": "No structured_resume in state"
            }
    
    # FIXED: Check if structured_resume is actually a dict (not a slice or other type)
    if not isinstance(structured_resume, dict):
        error_type = type(structured_resume).__name__
        error_repr = repr(structured_resume) if not isinstance(structured_resume, slice) else f"slice({structured_resume.start}, {structured_resume.stop}, {structured_resume.step})"
        log.error(f"❌ RESUME_SUMMARY_AGENT: structured_resume is not a dict (type: {error_type}, value: {error_repr}), cannot process")
        # If it's a slice, this is a serious data corruption issue - try to get structured_resume from state differently
        if isinstance(structured_resume, slice):
            log.error(f"❌ RESUME_SUMMARY_AGENT: CRITICAL - structured_resume is a slice object! This indicates data corruption in state.")
            # Try to recover by getting structured_resume from a different source
            try:
                from chroma import get_resume_doc
                from core.utils import run_blocking_io
                if uid:
                    rdoc = await run_blocking_io(get_resume_doc, uid) or {}
                    recovered_resume = rdoc.get("structured_resume")
                    if isinstance(recovered_resume, dict):
                        log.warning(f"⚠️ RESUME_SUMMARY_AGENT: Recovered structured_resume from resume_doc for UID {uid}")
                        structured_resume = recovered_resume
                    else:
                        raise ValueError("Recovered resume is also invalid")
            except Exception as recover_error:
                log.error(f"❌ RESUME_SUMMARY_AGENT: Failed to recover structured_resume: {recover_error}")
                return {
                    "resume_summary_status": "skipped",
                    "resume_summary_error": f"structured_resume is a slice object (data corruption) and recovery failed: {recover_error}"
                }
        else:
            return {
                "resume_summary_status": "skipped",
                "resume_summary_error": f"structured_resume is not a dict (type: {error_type})"
            }
    
    if not uid:
        log.warning("⚠️ RESUME_SUMMARY_AGENT: No UID found in state, cannot cache summary")
        return {
            "resume_summary_status": "skipped",
            "resume_summary_error": "No UID in state"
        }
    
    try:
        # Generate and cache Gemini resume summary
        from agents.prompt_generator import get_or_generate_resume_summary
        
        # ✅ CRITICAL: Use structured_resume from groq_resume_parser (fresh output)
        # Log what we're using for transparency
        resume_source = "groq_resume_parser_output"
        if structured_resume.get("name") or structured_resume.get("work_experience"):
            log.info(f"✅ RESUME_SUMMARY_AGENT: Using structured_resume from {resume_source} with {len(structured_resume.get('work_experience', []))} work experiences, {len(structured_resume.get('skills', []))} skills")
        else:
            log.warning(f"⚠️ RESUME_SUMMARY_AGENT: structured_resume from {resume_source} may be incomplete")
        
        log.info(f"🔄 RESUME_SUMMARY_AGENT: Generating/caching Gemini resume summary for UID {uid} using {resume_source}")
        resume_summary = await get_or_generate_resume_summary(
            structured_resume,  # ✅ This is from groq_resume_parser output
            uid, 
            force_regenerate=False  # Use cache if hash matches, otherwise regenerate
        )
        
        # Ensure summary is stored in resume_doc (get_or_generate_resume_summary does this, but verify)
        try:
            from core.utils import run_blocking_io
            rdoc = await run_blocking_io(get_resume_doc, uid) or {}
            if "resume_summary_cache" not in rdoc:
                rdoc["resume_summary_cache"] = {}
            rdoc["resume_summary_cache"]["summary"] = resume_summary
            await run_blocking_io(upsert_resume_doc, uid, rdoc)
            log.info(f"✅ RESUME_SUMMARY_AGENT: Verified summary stored in resume_doc for UID {uid}")
        except Exception as e_store:
            log.warning(f"⚠️ RESUME_SUMMARY_AGENT: Failed to verify summary storage: {e_store}")
        
        # Add summary to structured_resume in state so downstream agents can use it
        # Create a deep copy to avoid mutating the original and handle nested structures safely
        # This avoids TypeError("unhashable type: 'slice'") when structured_resume 
        # contains problematic types like slice objects
        try:
            import copy
            # Use deep copy to safely handle nested structures and avoid slice object issues
            updated_structured_resume = copy.deepcopy(structured_resume)
        except (TypeError, ValueError) as copy_error:
            # If deep copy fails (e.g., due to unhashable types like slice), build a clean dict with hashable keys only
            log.warning(f"⚠️ RESUME_SUMMARY_AGENT: Deep copy failed ({copy_error}), building dict with hashable keys only")
            try:
                updated_structured_resume = {
                    k: v for k, v in structured_resume.items()
                    if isinstance(k, (str, int, float, bool, type(None)))
                }
            except Exception as shallow_error:
                # If even shallow copy fails, create new dict and preserve what we can
                log.error(f"❌ RESUME_SUMMARY_AGENT: Both deep and shallow copy failed, creating new dict")
                updated_structured_resume = {}
                # Only use hashable keys (e.g. str) to avoid "unhashable type: 'slice'" when keys are corrupted
                for key, value in structured_resume.items():
                    if not isinstance(key, (str, int, float, bool)) and not (isinstance(key, type(None))):
                        log.warning(f"⚠️ RESUME_SUMMARY_AGENT: Skipping unhashable key type {type(key).__name__}")
                        continue
                    if isinstance(value, (str, int, float, bool, type(None))):
                        updated_structured_resume[key] = value
                    elif isinstance(value, (list, dict)):
                        try:
                            updated_structured_resume[key] = copy.deepcopy(value)
                        except (TypeError, ValueError):
                            log.warning(f"⚠️ RESUME_SUMMARY_AGENT: Skipping key '{key}' due to unhashable types in value")
        
        updated_structured_resume["resume_summary"] = resume_summary
        
        log.info(f"✅ RESUME_SUMMARY_AGENT: Gemini resume summary generated and cached for UID {uid} ({len(resume_summary)} chars)")
        
        # Record successful completion
        processing_time = _calculate_processing_time(start_time)
        await summary_memory.record_attempt(
            'resume_summary_generation', 'llm', True, 0.9, processing_time
        )
        
        log_agent_completion(log_context, {
            "success": True,
            "summary_length": len(resume_summary),
            "uid": uid
        }, "llm", processing_time)
        
        out = {
            "resume_summary_status": "completed",
            "structured_resume": updated_structured_resume
        }
        # Preserve job_id and body for compare-candidate-job flow
        body = state.get("body") or {}
        if body.get("request_type") == "candidate_job_match" or body.get("job_id"):
            out["body"] = body
            job_id = state.get("job_id") or body.get("job_id")
            if job_id:
                out["job_id"] = job_id
        return out
        
    except Exception as e:
        log.error(f"❌ RESUME_SUMMARY_AGENT: Failed to generate resume summary: {e}")
        import traceback
        log.error(traceback.format_exc())
        
        # Record failure
        processing_time = _calculate_processing_time(start_time)
        await summary_memory.record_attempt(
            'resume_summary_generation', 'llm', False, 0.0, processing_time
        )
        
        log_agent_completion(log_context, {
            "success": False,
            "error": str(e),
            "uid": uid
        }, "llm", processing_time)
        
        # Return state without summary - downstream agents will use fallback
        return {
            "resume_summary_status": "failed",
            "resume_summary_error": str(e)
        }


from typing import Dict, Any
from langsmith.run_helpers import traceable
import logging
from chroma import get_chat_session, update_chat_session
from chroma import upsert_resume_doc  # NEW: per-UID resume doc in chat_sessions
from utils.session_manager import session_manager
from core.utils import (
    _mask, _sanitize_text_for_llm, _validate_state_inputs, _to_text, _clean_json_text, 
    _scan_balanced_json, _safe_json_loads, _extract_json_from_response, _create_error_response,
    _generate_request_id, _calculate_processing_time, run_blocking_io
)
from core.config import get_agent_config
from core.security import validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm, PII_PATTERNS, INJECTION_FILTERS
from core.memory import BaseAgentMemory, get_agent_memory
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion

# Get centralized configuration
config = get_agent_config("resume_assembler")

# Custom memory class for resume assembler (extends base memory)
class ResumeAssemblerMemory(BaseAgentMemory):
    def __init__(self, tenant_id: str = "default"):
        super().__init__(tenant_id, config.adaptation_window)
        # Add any resume assembler specific fields here if needed

# Use centralized memory management
async def get_resume_assembler_memory(tenant_id: str = "default") -> ResumeAssemblerMemory:
    """Get or create tenant-scoped resume assembler memory."""
    return await get_agent_memory("resume_assembler", tenant_id, ResumeAssemblerMemory)

log = logging.getLogger(__name__)


def _deduplicate_skills(skills_list):
    """Remove duplicate skills based on SkillName (case-insensitive).
    If duplicates exist, prefer the one with X/10 format proficiency (from skill_proficiency_analyzer).
    Since analyzer output comes last due to state merging, later occurrences with X/10 format will replace earlier ones."""
    if not skills_list:
        return []
    
    seen = {}
    # Process in normal order - analyzer's skills come last, so they'll replace earlier duplicates
    for skill in skills_list:
        if not isinstance(skill, dict):
            continue
        
        skill_name = skill.get("SkillName", "").strip()
        if not skill_name:
            continue
        
        # Normalize skill name for comparison (case-insensitive)
        key = skill_name.lower()
        
        # Check if proficiency is in X/10 format (from analyzer)
        prof = skill.get("Proficiency", "")
        is_x10_format = isinstance(prof, str) and prof.strip() and "/10" in prof
        
        # If we haven't seen this skill, add it
        if key not in seen:
            seen[key] = skill
        else:
            # If duplicate, prefer the one with X/10 format proficiency (from analyzer)
            existing_prof = seen[key].get("Proficiency", "")
            existing_is_x10 = isinstance(existing_prof, str) and existing_prof.strip() and "/10" in existing_prof
            
            # Always prefer X/10 format (from analyzer) over other formats like "Fluent"
            if is_x10_format and not existing_is_x10:
                seen[key] = skill
            # If both have X/10 format, prefer the later one (from analyzer)
            elif is_x10_format and existing_is_x10:
                seen[key] = skill
            # If current one has non-empty proficiency and existing doesn't, prefer current
            elif prof and prof.strip() and not (existing_prof and existing_prof.strip()):
                seen[key] = skill
            # Otherwise keep the existing one
    
    return list(seen.values())


def _filter_skills_with_proficiency(skills_list):
    """Return only skills that have a non-empty Proficiency value."""
    filtered = []
    for skill in (skills_list or []):
        if not isinstance(skill, dict):
            continue
        prof = skill.get("Proficiency")
        # Accept non-empty strings or truthy non-strings
        if isinstance(prof, str):
            if prof.strip():
                filtered.append(skill)
        elif prof:
            filtered.append(skill)
    return filtered

@traceable(name="resume_assembler_agent")
async def resume_assembler_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced resume assembler with centralized utilities and LLM-only approach.
    """
    # Use centralized logging
    log_context = create_log_context("resume_assembler", state.get("tenant_id", "default"))
    start_time = log_context["start_time"]
    request_id = log_context["request_id"]
    
    log.info("--- Entering Resume Assembler Agent ---")
    
    # Log synchronization timing - all parsers have completed
    log.info(
        "🚀 PARALLEL_TIMING: resume_assembler started - all parsers completed"
    )
    
    # Get tenant-scoped memory
    assembler_memory = await get_resume_assembler_memory(state.get("tenant_id", "default"))
    
    log.info("Assembling structured resume from parsed components.")

    uid = state.get("uid")
    if not uid:
        processing_time = _calculate_processing_time(start_time)
        AgentLogger.log_warning(log_context, "Missing UID in state. Cannot save to database.")
        return _create_error_response("Missing UID in state", processing_time)

    try:
        # Check if we have structured_resume from Groq parser (new flow)
        groq_structured_resume = state.get("structured_resume")
        
        if groq_structured_resume and isinstance(groq_structured_resume, dict):
            # Groq parser already provided complete structured_resume
            log.info("✅ Using structured_resume from Groq parser")
            # Handle both "Name" and "name" (case-insensitive name extraction)
            name = (
                groq_structured_resume.get("Name") or 
                groq_structured_resume.get("name") or
                groq_structured_resume.get("fullName") or
                groq_structured_resume.get("full_name") or
                "N/A"
            )
            # Handle nested Personal_Information structure
            if (name == "N/A" or not name) and "Personal_Information" in groq_structured_resume:
                personal_info = groq_structured_resume.get("Personal_Information", {})
                if isinstance(personal_info, dict):
                    name = (
                        personal_info.get("Name") or
                        personal_info.get("name") or
                        personal_info.get("FullName") or
                        personal_info.get("fullName") or
                        personal_info.get("full_name") or
                        name
                    )
            # Handle both "ContactDetails" and "contact_details" (case-insensitive)
            contact_details = (
                groq_structured_resume.get("ContactDetails") or
                groq_structured_resume.get("contact_details") or
                groq_structured_resume.get("Contact_Details") or
                {}
            )
            # Handle nested contact details in Personal_Information
            if not contact_details and "Personal_Information" in groq_structured_resume:
                personal_info = groq_structured_resume.get("Personal_Information", {})
                if isinstance(personal_info, dict):
                    contact_details = (
                        personal_info.get("ContactDetails") or
                        personal_info.get("contact_details") or
                        personal_info.get("Contact_Details") or
                        contact_details
                    )
            education = (
                groq_structured_resume.get("education") or
                groq_structured_resume.get("Education") or
                groq_structured_resume.get("Education_Information") or
                []
            )
            work_experience = (
                groq_structured_resume.get("work_experience") or
                groq_structured_resume.get("experience") or
                groq_structured_resume.get("WorkExperience") or
                groq_structured_resume.get("Work_Experience") or
                []
            )
            # Note: Skills are ONLY taken from skill_proficiency_analyzer output, not from structured_resume
            certifications = (
                groq_structured_resume.get("certifications") or
                groq_structured_resume.get("Certifications") or
                []
            )
            projects = (
                groq_structured_resume.get("projects") or
                groq_structured_resume.get("Projects") or
                []
            )
            extras = (
                groq_structured_resume.get("extras") or
                groq_structured_resume.get("Extras") or
                []
            )
            total_experience_years = groq_structured_resume.get("total_experience_years", 0) or groq_structured_resume.get("totalExperienceYears", 0)
            _prof = (
                groq_structured_resume.get("professional_summary")
                or groq_structured_resume.get("Professional_Summary")
                or groq_structured_resume.get("professionalSummary")
                or ""
            )
            professional_summary = _prof if isinstance(_prof, str) else str(_prof or "")
            
            # When structured_resume is partial (e.g. only "skills" from skill_proficiency_analyzer),
            # supplement from top-level state so groq parser data is not lost.
            sr_keys = set(groq_structured_resume.keys())
            if not education:
                education = (
                    state.get("education") or
                    state.get("Education") or
                    []
                )
                if not isinstance(education, list):
                    education = [education] if education else []
            if not work_experience:
                work_experience = (
                    state.get("work_experience") or
                    state.get("experience") or
                    state.get("WorkExperience") or
                    state.get("Work_Experience") or
                    []
                )
                if not isinstance(work_experience, list):
                    work_experience = [work_experience] if work_experience else []
            if (not name or name == "N/A") and state.get("name"):
                _sn = state.get("name")
                name = _sn[0] if isinstance(_sn, list) and _sn else (_sn if isinstance(_sn, str) else "N/A")
            if not contact_details and state.get("contact_details"):
                contact_details = state.get("contact_details", {})
            if not certifications and state.get("certifications"):
                certifications = state.get("certifications", [])
            if not projects and state.get("projects"):
                projects = state.get("projects", [])
            if not extras and state.get("extras"):
                extras = state.get("extras", [])
            if not total_experience_years and state.get("total_experience_years"):
                total_experience_years = state.get("total_experience_years", 0)
            if not professional_summary and (state.get("professional_summary") or state.get("Summary") or state.get("Objective")):
                _ps = state.get("professional_summary") or state.get("Summary") or state.get("Objective") or ""
                professional_summary = _ps if isinstance(_ps, str) else str(_ps or "")
            if len(sr_keys) == 1 and "skills" in sr_keys:
                log.info("🔍 RESUME_ASSEMBLER: structured_resume had only 'skills'; supplemented from top-level state")
            
            log.info(f"🔍 RESUME_ASSEMBLER: Extracted name='{name}' from structured_resume (keys: {list(groq_structured_resume.keys())})")
        else:
            # Fallback: Extract data from top-level fields (backward compatibility)
            name = state.get("name", "N/A")
            contact_details = state.get("contact_details", {})
            education = state.get("education", [])
            work_experience = state.get("work_experience", [])
            # Note: Skills are ONLY taken from skill_proficiency_analyzer output, not from other sources
            certifications = state.get("certifications", [])
            projects = state.get("projects", [])
            extras = state.get("extras", [])
            total_experience_years = state.get("total_experience_years", 0)
            _ps = state.get("professional_summary") or state.get("Summary") or state.get("Objective") or ""
            professional_summary = _ps if isinstance(_ps, str) else str(_ps or "")
    
        # If data not found at top level, try to extract from parser results
        if not name or name == "N/A":
            # Try multiple fallback sources for name
            personal_info = state.get("personal_info_parser", {})
            if isinstance(personal_info, dict):
                name = personal_info.get("name") or personal_info.get("output", {}).get("name", "N/A")
                contact_details = personal_info.get("contact_details") or personal_info.get("output", {}).get("contact_details", {})
            
            # If still no name, try to extract from resume text as last resort
            if not name or name == "N/A":
                resume_text = state.get("resume_text", "")
                if resume_text:
                    # Simple name extraction from first line
                    first_line = resume_text.split('\n')[0].strip()
                    if first_line and len(first_line.split()) >= 2:
                        name = first_line
                        log.warning(f"Extracted name from resume text as fallback: {name}")
        
        if not education:
            education_data = state.get("education_parser", {})
            if isinstance(education_data, dict):
                education = education_data.get("education", [])
        
        if not work_experience:
            experience_data = state.get("experience_parser", {})
            if isinstance(experience_data, dict):
                work_experience = experience_data.get("work_experience", [])
                # Only overwrite certifications, projects, extras if they are empty
                if not certifications:
                    certifications = experience_data.get("certifications", [])
                if not projects:
                    projects = experience_data.get("projects", [])
                if not extras:
                    extras = experience_data.get("extras", [])
                total_experience_years = experience_data.get("total_experience_years", 0)
        
        # Fallback: try to get professional_summary from parsers if still empty
        if not (professional_summary or "").strip():
            exp = state.get("experience_parser") or {}
            personal = state.get("personal_info_parser") or {}
            if isinstance(exp, dict):
                _e = exp.get("professional_summary") or exp.get("summary") or exp.get("Summary") or ""
                if _e and isinstance(_e, str):
                    professional_summary = _e
            if isinstance(personal, dict) and not (professional_summary or "").strip():
                _p = personal.get("professional_summary") or personal.get("summary") or personal.get("Summary") or ""
                if _p and isinstance(_p, str):
                    professional_summary = _p
        
        # CRITICAL: COMPLETELY REPLACE skills field with output from skill_proficiency_analyzer
        # The analyzer output contains ALL skills with proficiency data in X/10 format
        # We use the exact output without any filtering or modification
        skills = state.get("skills", [])
        
        if not skills:
            log.warning("⚠️ RESUME_ASSEMBLER: No skills found from skill_proficiency_analyzer output. Using empty skills list.")
            skills = []
        else:
            log.info(f"✅ RESUME_ASSEMBLER: Replacing skills field with {len(skills)} skills from skill_proficiency_analyzer output")
            # Deduplicate only if needed (in case state merging created duplicates)
            # This preserves the analyzer's output while removing any accidental duplicates
            original_count = len(skills)
            skills = _deduplicate_skills(skills)
            if len(skills) != original_count:
                log.info(f"🔍 RESUME_ASSEMBLER: Removed {original_count - len(skills)} duplicate skills (after deduplication: {len(skills)} skills)")
            else:
                log.info(f"🔍 RESUME_ASSEMBLER: Using all {len(skills)} skills from analyzer (no duplicates found)")

        # Extract candidate_domains from Groq parser output (optional, for domain-based retrieval)
        candidate_domains = []
        if groq_structured_resume and isinstance(groq_structured_resume, dict):
            candidate_domains = (
                groq_structured_resume.get("candidate_domains")
                or groq_structured_resume.get("candidateDomains")
                or []
            )
        if not candidate_domains:
            candidate_domains = state.get("candidate_domains") or state.get("candidateDomains") or []

        structured_resume = {
            "Name": name,
            "ContactDetails": contact_details,
            "professional_summary": str(professional_summary or ""),
            "education": education,
            "experience": work_experience,
            "skills": skills,
            "certifications": certifications,
            "projects": projects,
            "total_experience_years": total_experience_years,
            "extras": extras,
            "candidate_domains": candidate_domains[:3] if candidate_domains else []
        }
        
        # ✅ PERFORMANCE: Calculate and cache resume hash for reuse by downstream agents
        # This avoids recalculating the hash multiple times in prompt_generator and other agents
        try:
            from agents.prompt_generator import _calculate_resume_hash
            resume_hash = _calculate_resume_hash(structured_resume)
            structured_resume["_resume_hash"] = resume_hash
            log.debug(f"✅ RESUME_ASSEMBLER: Cached resume hash for reuse: {resume_hash[:8]}...")
        except Exception as hash_error:
            log.debug(f"⚠️ RESUME_ASSEMBLER: Could not calculate resume hash: {hash_error}")

        
        log.info(f"🔍 RESUME_ASSEMBLER DEBUG: Name='{name}', Education count={len(education)}, Work experience count={len(work_experience)}, Skills count={len(skills)}, Certifications count={len(certifications)}, Projects count={len(projects)}, Extras count={len(extras)}, Total experience years={total_experience_years}")
        
        # Debug: Check what data sources were used
        log.info(f"🔍 RESUME_ASSEMBLER DEBUG: Data sources - personal_info_parser: {'personal_info_parser' in state}, education_parser: {'education_parser' in state}, experience_parser: {'experience_parser' in state}, skills_parser: {'skills_parser' in state}")
        log.info(f"🔍 RESUME_ASSEMBLER DEBUG: Experience years source - from state: {state.get('total_experience_years', 'NOT_FOUND')}, from experience_parser: {state.get('experience_parser', {}).get('total_experience_years', 'NOT_FOUND') if isinstance(state.get('experience_parser'), dict) else 'NOT_DICT'}")

        # --- Save the assembled resume to the database ---
        if uid:
            try:
                log.debug(f"🔍 RESUME_ASSEMBLER: Starting database save for UID {uid}")
                # insert_resume is done in groq_resume_parser (pure Groq output).
                # Do NOT call insert_resume here — by this point structured_resume in state
                # has been merged with skill_proficiency_analyzer output, which would overwrite
                # the pure Groq data in ChromaDB with proficiency-enriched skills.
                from chroma import upsert_resume_doc, get_resume_doc
                from core.utils import run_blocking_io

                existing_doc = await run_blocking_io(get_resume_doc, uid)
                existing_doc = existing_doc or {}

                log.info(f"🔍 RESUME_ASSEMBLER: get_resume_doc completed for UID {uid}")
                log.debug(
                    "🔍 RESUME_ASSEMBLER: About to enter inner try block for upsert_resume_doc"
                )
                # NEW: Persist full structured_resume to chat_sessions (uid_resume); chunking handles >16KB
                try:
                    log.debug(
                        f"🔍 RESUME_ASSEMBLER: About to upsert_resume_doc for UID {uid}"
                    )
                    existing_sr = existing_doc.get("structured_resume") or {}
                    preserved_summary = existing_sr.get("user_interests_summary")

                    # Full resume for content generator and others; chunking in chroma handles size
                    full_sr = dict(structured_resume)
                    if preserved_summary:
                        full_sr["user_interests_summary"] = preserved_summary
                        log.debug(
                            "🔍 RESUME_ASSEMBLER: Preserving user_interests_summary from existing doc"
                        )
                    full_payload = {
                        "structured_resume": full_sr,
                        "timestamp": log_context["start_time"],
                    }

                    log.debug(
                        "🔍 RESUME_ASSEMBLER: Calling upsert_resume_doc with full structured_resume "
                        f"(keys: {list(full_sr.keys())})"
                    )
                    await run_blocking_io(upsert_resume_doc, uid, full_payload, metadata={"agent": "resume_assembler"})
                    log.info(f"✅ Stored full structured_resume in chat_sessions as uid_resume for UID {uid}")
                    # ✅ OPTIMIZATION: Removed redundant verification call - upsert_resume_doc already confirms success
                    # Verification was adding ~1-2s latency without providing critical value
                except Exception as e_store:
                    log.error(f"❌ Could not upsert uid_resume doc for UID {uid}: {e_store}")
                    import traceback
                    error_tb = traceback.format_exc()
                    log.error(f"❌ Full traceback: {error_tb}")
                    log.error(f"❌ ERROR storing resume doc: {e_store}")
                    log.error(f"❌ ERROR traceback: {error_tb}")
            
                # Session Management Integration
                try:
                    log.info(f"🔄 RESUME_ASSEMBLER: Starting session storage for UID={uid}")
                
                    # Use the session_id from state if available, otherwise get/reuse session
                    current_session_id = state.get("session_id")
                    if current_session_id:
                        log.info(f"✅ RESUME_ASSEMBLER: Using existing session_id from state: {current_session_id}")
                        try:
                            existing_session = await run_blocking_io(session_manager.get_session, current_session_id)
                            if not existing_session:
                                log.warning(f"⚠️ RESUME_ASSEMBLER: Session {current_session_id} not found, creating new one")
                                existing_session = await run_blocking_io(
                                    session_manager.get_or_reuse_session,
                                    owner_id=uid,
                                    kind="candidate_pipeline",
                                    owner_type="candidate",
                                    initial_step="resume_assembler",
                                    initial_data={"structured_resume": structured_resume}
                                )
                        except Exception as e:
                            log.error(f"❌ RESUME_ASSEMBLER: Error getting session {current_session_id}: {e}")
                            existing_session = await run_blocking_io(
                                session_manager.get_or_reuse_session,
                                owner_id=uid,
                                kind="candidate_pipeline",
                                owner_type="candidate",
                                initial_step="resume_assembler",
                                initial_data={"structured_resume": structured_resume}
                            )
                    else:
                        log.info(f"🔄 RESUME_ASSEMBLER: No session_id in state, getting/reusing session for UID={uid}")
                        existing_session = await run_blocking_io(
                            session_manager.get_or_reuse_session,
                            owner_id=uid,
                            kind="candidate_pipeline",
                            owner_type="candidate",
                            initial_step="resume_assembler",
                            initial_data={"structured_resume": structured_resume}
                        )
                    log.info(f"✅ RESUME_ASSEMBLER: Using session: {existing_session.session_id} for UID={uid}")
                    
                    # Update session step
                    log.info(f"🔄 RESUME_ASSEMBLER: Updating session step for session_id={existing_session.session_id}")
                    session_update_result = await run_blocking_io(
                        session_manager.update_step,
                        session_id=existing_session.session_id,
                        step="resume_assembler",
                        data={"has_structured_resume": True},
                        progress=0.4  # 40% complete after resume assembly
                    )
                    log.info(f"📊 RESUME_ASSEMBLER: Session update result: {session_update_result}")
                    
                    # Note: Resume assembler data is NOT stored in chat_sessions
                    # Only stored in resume collection and Firebase
                    log.info(f"ℹ️ RESUME_ASSEMBLER: Skipping chat_sessions storage (not required)")
                    log.info(
                        "ℹ️ Resume assembler data stored in resume collection and "
                        "Firebase (not in chat_sessions)"
                    )
                    
                except Exception as session_error:
                    log.error(f"❌ RESUME_ASSEMBLER: Error in session management: {session_error}")
                    import traceback
                    log.error(f"Resume assembler session management traceback: {traceback.format_exc()}")
                    log.error(
                        f"❌ Error in resume assembler session management: {session_error}"
                    )
                    
            except Exception as e:
                log.error(f"Failed to save structured_resume to database for UID {uid}: {e}")
                # Decide if you want to stop the flow or just log the error
                # For now, we will log and continue
    
        if not structured_resume.get("Name") or structured_resume.get("Name") == "N/A":
            log.warning("Assembled resume is missing a name. The scorer might fail.")

        if not structured_resume.get("skills"):
            log.warning("Assembled resume is missing skills. The scorer might fail.")

        # OPTIMIZATION: Pre-compute and cache skills analysis for resume scorer
        # This avoids redundant analysis in resume_scorer_agent
        if structured_resume.get("skills"):
            try:
                from agents.resume_score import _analyze_skills_detailed
                skills_analysis = _analyze_skills_detailed(structured_resume.get("skills", []))
                structured_resume["_cached_skills_analysis"] = skills_analysis
                log.info(f"✅ RESUME_ASSEMBLER: Cached skills analysis for resume scorer (dominant_domain: {skills_analysis.get('dominant_domain', 'unknown')})")
            except Exception as e:
                log.warning(f"⚠️ RESUME_ASSEMBLER: Failed to cache skills analysis: {e}")
                # Continue without cache - resume scorer will compute it

        # Record successful assembly
        processing_time = _calculate_processing_time(start_time)
        await assembler_memory.record_attempt(
            'resume_assembly', 'llm', True, 0.8, processing_time
        )
        
        # Log success
        log_agent_completion(log_context, {
            "success": True,
            "name": structured_resume.get("Name", "N/A"),
            "education_count": len(structured_resume.get("education", [])),
            "work_experience_count": len(structured_resume.get("experience", [])),
            "skills_count": len(structured_resume.get("skills", [])),
            "certifications_count": len(structured_resume.get("certifications", [])),
            "projects_count": len(structured_resume.get("projects", [])),
            "extras_count": len(structured_resume.get("extras", []))
        }, "llm", processing_time)

        # FINAL GUARANTEE: upsert full structured_resume into uid_resume (chunked if >16KB)
        try:
            log.debug(
                f"🔒 RESUME_ASSEMBLER: Final guarantee upsert to uid_resume for UID {uid}"
            )
            from chroma import upsert_resume_doc
            from core.utils import run_blocking_io
            if 'existing_doc' not in locals() or existing_doc is None:
                from chroma import get_resume_doc
                existing_doc = await run_blocking_io(get_resume_doc, uid) or {}
            existing_sr = existing_doc.get("structured_resume") or {}
            preserved_summary = existing_sr.get("user_interests_summary")

            full_sr = dict(structured_resume)
            if preserved_summary:
                full_sr["user_interests_summary"] = preserved_summary
                log.debug("🔒 RESUME_ASSEMBLER: Preserving user_interests_summary from existing doc")
            final_payload = {
                "structured_resume": full_sr,
                "timestamp": start_time,
            }
            await run_blocking_io(upsert_resume_doc, uid, final_payload, metadata={"agent": "resume_assembler", "stage": "final"})
            log.info(f"✅ RESUME_ASSEMBLER: Final upsert (full structured_resume) done for {uid}_resume")
        except Exception as _final_e:
            import traceback
            log.error(
                f"❌ RESUME_ASSEMBLER: Final upsert failed for {uid}: {_final_e}"
            )
            log.error(traceback.format_exc())

        # Preserve job_id and other routing fields in the return (critical for compare-candidate-job flow)
        result = {"structured_resume": structured_resume}
        body = state.get("body") or {}
        job_id = state.get("job_id") or body.get("job_id")
        is_compare_flow = (
            body.get("request_type") == "candidate_job_match"
            or state.get("endpoint_name") == "compare_candidate_job"
            or body.get("endpoint_name") == "compare_candidate_job"
        )
        if job_id:
            result["job_id"] = job_id
        # Always preserve body for compare flow so preprocessor can read job_id; ensure job_id is in body
        if is_compare_flow or (body and (body.get("request_type") == "candidate_job_match" or body.get("job_id"))):
            body_to_return = dict(body) if body else {}
            if job_id:
                body_to_return["job_id"] = job_id
            body_to_return.setdefault("request_type", "candidate_job_match")
            body_to_return.setdefault("endpoint_name", "compare_candidate_job")
            result["body"] = body_to_return
            result["endpoint_name"] = state.get("endpoint_name") or "compare_candidate_job"
            result["request_type"] = state.get("request_type") or "candidate_job_match"

        return result
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        AgentLogger.log_error(log_context, f"Resume assembly failed: {str(e)}", processing_time)
        
        # Record failure
        await assembler_memory.record_attempt(
            'resume_assembly', 'llm', False, 0.0, processing_time
        )
        
        return _create_error_response(f"Resume assembly failed: {str(e)}", processing_time)
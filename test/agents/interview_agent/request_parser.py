"""
Request parsing and validation for interview agent.

Handles parsing, validation, and normalization of interview requests.

NOTE:
This module performs request normalization only.
It MUST NOT contain interview flow, scoring, or state transition logic.
"""

import json
import logging
import asyncio
from typing import Dict, List, Tuple, Any
from fastapi import Request, HTTPException

from utils.interview_utils import mask_sensitive_data, build_context_from_payload, process_uid_context
from .anonymizer_module import PIIAnonymizer
from .state_manager import InterviewRequest
from .session_manager import load_conversation_history
from .input_adapter import NormalizedAnswer

log = logging.getLogger(__name__)


async def parse_and_validate_request(request: Request) -> InterviewRequest:
    """
    Handles parsing, validation, and auth token extraction.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        Validated InterviewRequest object
        
    Raises:
        HTTPException: If validation fails
    """
    data = await request.json()

    # Extract auth token from header (preferred) or body (fallback)
    auth_token = request.headers.get("authorization", "").replace("Bearer ", "")
    auth_method = "header" if auth_token else "none"

    if not auth_token and "auth_token" in data:
        auth_token = data.get("auth_token")
        auth_method = "body"

    # Add auth_token to data if found in header
    if auth_token and "auth_token" not in data:
        data["auth_token"] = auth_token

    # Log auth method without exposing token
    log.info(f"Auth token method: {auth_method}, token present: {bool(auth_token)}")

    # Handle normalized_answer input (multi-modal support)
    # If normalized_answer is provided, extract answer_text and metadata
    if "normalized_answer" in data:
        try:
            normalized = NormalizedAnswer(**data["normalized_answer"])
            # Map answer_text to answer field (existing interface)
            data["answer"] = normalized.answer_text
            # Merge metadata instead of overwriting (preserve any existing input_metadata)
            existing_metadata = data.get("input_metadata") or {}
            data["input_metadata"] = {**existing_metadata, **(normalized.metadata or {})}
            # Extract source from metadata and set input_modality
            source = normalized.metadata.get("source", "text") if normalized.metadata else "text"
            if source in ["text", "audio", "avatar"]:
                data["input_modality"] = source
            else:
                data["input_modality"] = "text"  # Default fallback
            log.debug(f"Processed normalized_answer: modality={data.get('input_modality')}, has_metadata={bool(data.get('input_metadata'))}")
            # Clean up normalized_answer from data after processing
            data.pop("normalized_answer", None)
        except Exception as e:
            log.warning(f"Failed to parse normalized_answer: {e}. Falling back to answer field.")
            # Clean up failed normalized_answer to prevent pydantic validation confusion
            data.pop("normalized_answer", None)
            # Fall through to use existing answer field if normalized_answer parsing fails

    # Normalize payload: handle structured_resume sent at top level
    if "structured_resume" in data:
        structured_resume_value = data["structured_resume"]
        # If structured_resume is a JSON string, parse it; if empty string, use None
        if isinstance(structured_resume_value, str):
            if structured_resume_value.strip():
                try:
                    # FIX: For large JSON strings, use asyncio.to_thread() to prevent blocking
                    if len(structured_resume_value) > 10000:
                        structured_resume_value = await asyncio.to_thread(
                            json.loads, 
                            structured_resume_value
                        )
                    else:
                        structured_resume_value = json.loads(structured_resume_value)
                except (json.JSONDecodeError, ValueError):
                    # If not valid JSON, treat as empty
                    structured_resume_value = None
            else:
                structured_resume_value = None

        # Keep structured_resume at top level for topic-focused interviews
        if structured_resume_value:
            data["structured_resume"] = structured_resume_value
            # Also wrap in resume object for backward compatibility
            if "resume" not in data:
                data["resume"] = {"structured_resume": structured_resume_value}
        else:
            data["structured_resume"] = None
            if "resume" not in data:
                data["resume"] = None

    # Check if this is the first call (question_count is 0 AND conversation_history is empty)
    is_first_call = (
        data.get("question_count", 0) == 0 and
        len(data.get("conversation_history", []) or []) == 0
    )
    
    # Only log full input payload on the first call
    if is_first_call:
        try:
            # Mask sensitive data before logging
            masked_data = mask_sensitive_data(data)
            log.info("=" * 80)
            log.info("INTERVIEW AGENT INPUT PAYLOAD (FIRST CALL):")
            log.info("=" * 80)
            log.info(json.dumps(masked_data, indent=2, default=str))
            log.info("=" * 80)
            log.info(
                f"INTERVIEW REQ: uid={data.get('uid')} | "
                f"session_id={data.get('session_id')} | "
                f"end={data.get('end_interview')}"
            )
            log.info(f"Payload keys: {sorted(list(data.keys()))}")
        except Exception as e:
            log.error(f"Failed to log input payload: {e}")
    else:
        # For subsequent calls, just log minimal info
        log.debug(
            f"Interview turn | uid={data.get('uid')} | "
            f"session_id={data.get('session_id')} | "
            f"question_count={data.get('question_count', 0)} | "
            f"has_answer={bool(data.get('answer'))}"
        )

    # Skip loading conversation_history from Chroma before question generation for latency optimization
    # History will be loaded only if needed for evaluation/report generation (not for question generation)
    # Question generation now uses minimal context (last 3 turns) and doesn't need full history
    session_id_for_load = data.get("session_id")
    existing_history = data.get("conversation_history", [])
    history_length = len(existing_history) if existing_history else 0
    
    log.debug(f"[LOAD] Skipping Chroma load for question generation (latency optimization). session_id={session_id_for_load}, existing_history_length={history_length}")

    interview_req = InterviewRequest(**data)

    # Extract structured_resume from resume if not at top level
    try:
        if not interview_req.structured_resume and interview_req.resume:
            if isinstance(interview_req.resume, dict):
                # If resume contains a "structured_resume" key, use that
                if "structured_resume" in interview_req.resume:
                    interview_req.structured_resume = interview_req.resume["structured_resume"]
                    log.debug("Extracted structured_resume from resume['structured_resume']")
                # Otherwise, if resume is a dict with actual data (has keys like skills, experience, etc.),
                # treat the entire resume dict as the structured_resume (universal for any industry/role)
                elif interview_req.resume and len(interview_req.resume) > 0:
                    # Check if it looks like a structured resume (has typical resume fields - universal across all industries)
                    resume_keys = set(interview_req.resume.keys())
                    # Universal resume fields that appear across all industries and roles
                    typical_resume_keys = {
                        # Core fields (universal)
                        "skills", "experience", "work_experience", "education", "name",
                        # Contact fields (various formats)
                        "Email", "Phone", "Location", "contact_details", "LinkedIn", "GitHub",
                        # Additional fields (universal)
                        "projects", "certifications", "professional_summary", "summary",
                        # Extended fields (may be present)
                        "RoleFit", "resumeScore", "total_experience_years", "extras",
                        "professional_affiliations", "awards", "publications"
                    }
                    if resume_keys.intersection(typical_resume_keys):
                        interview_req.structured_resume = interview_req.resume
                        log.debug(
                            f"Using entire resume dict as structured_resume "
                            f"(has {len(resume_keys)} keys, detected as structured resume)"
                        )
    except Exception as e:
        log.warning(f"Error extracting structured_resume from resume: {e}. Continuing without structured_resume.")
        # Don't fail the request, just log the warning
    
    # Log interview topic and structured_resume status for debugging
    if interview_req.interview_topic:
        log.info(f"Interview topic provided: {interview_req.interview_topic}")
        log.info(f"Structured resume available: {bool(interview_req.structured_resume)}")
        if interview_req.structured_resume:
            # Cap resume size before logging to avoid log bloat
            resume_keys = list(interview_req.structured_resume.keys())[:10]
            log.info(f"Structured resume keys: {resume_keys}")
            # Log if skills are available for topic matching
            if "skills" in interview_req.structured_resume:
                skills = interview_req.structured_resume.get("skills", [])
                skills_count = len(skills) if isinstance(skills, list) else 0
                log.info(f"Structured resume has {skills_count} skills for topic matching")
        else:
            log.warning(
                f"Interview topic '{interview_req.interview_topic}' provided but structured_resume is None - "
                f"topic-focused features may not work optimally"
            )
    else:
        log.info("Interview topic not provided - will generate general interview questions")

    # Validate critical fields and cap history
    if not interview_req.uid:
        raise HTTPException(status_code=400, detail="uid is required")
    if interview_req.conversation_history and len(interview_req.conversation_history) > 50:
        interview_req.conversation_history = interview_req.conversation_history[-50:]
        log.info("Capped conversation_history to last 50 messages")

    return interview_req


def build_interview_context_data(
    interview_req: InterviewRequest
) -> Tuple[Dict, Dict, Dict, List[str], List[str], str, float, PIIAnonymizer]:
    """
    Builds all context data needed for the interview.
    
    Args:
        interview_req: Validated InterviewRequest object
        
    Returns:
        Tuple of (uid_context, candidate_info, safe_job_details, names, companies, session_id, start_time, anonymizer)
    """
    import time
    from utils.interview_utils import (
        extract_and_anonymize_candidate_info,
        sanitize_job_details
    )
    
    # Build context from the request payload
    uid_context = build_context_from_payload(interview_req)
    log.info(f"Built context for UID {interview_req.uid}: {list(uid_context.keys())}")

    # Quick context stats for logging
    try:
        sr = (uid_context.get("resume_data") or {}) if isinstance(uid_context, dict) else {}
        exp = sr.get("experience") or []
        skills = sr.get("skills") or []
        jt = (interview_req.job_details or {}).get("title") or (
            exp[0].get("job_title") if exp and isinstance(exp[0], dict) else "Professional"
        )
        name = sr.get("Name") or "Candidate"
        log.info(f"CTX name={name} exp_count={len(exp)} skills_count={len(skills)} job_title={jt}")
    except Exception:
        pass

    # Process UID context and merge with interview request data
    resume_data, job_details, names, companies = process_uid_context(uid_context, interview_req)

    log.info(
        f"Processed resume_data: Name='{resume_data.get('Name', 'N/A')}', "
        f"RoleFit='{resume_data.get('RoleFit', [])}'"
    )

    # Initialize session metadata with UUID for uniqueness
    # Using UUID prevents session collision in concurrent requests
    from .session_manager import generate_session_id
    session_id = interview_req.session_id or generate_session_id(interview_req.uid)
    start_time = interview_req.start_time or time.time()

    # Initialize enhanced anonymizer with session ID
    anonymizer = PIIAnonymizer()
    anonymizer.set_session_id(session_id)

    # Extract and anonymize candidate info
    candidate_info, names, companies = extract_and_anonymize_candidate_info(
        resume_data,
        anonymizer
    )

    # Enhance candidate info with ChromaDB data
    if uid_context and uid_context.get("personal_info") and "name" in uid_context["personal_info"]:
        candidate_info["name"] = uid_context["personal_info"]["name"]

    # Add skills from ChromaDB
    if uid_context and uid_context.get("skills_analysis") and "skills" in uid_context["skills_analysis"]:
        skills_list = []
        for skill in uid_context["skills_analysis"]["skills"]:
            if "SkillName" in skill:
                skills_list.append(skill["SkillName"])
        if skills_list:
            candidate_info["skills"] = ", ".join(skills_list)

    # Sanitize job details
    safe_job_details = sanitize_job_details(job_details)

    return uid_context, candidate_info, safe_job_details, names, companies, session_id, start_time, anonymizer


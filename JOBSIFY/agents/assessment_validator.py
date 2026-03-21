"""
Assessment validator agent for validating the assessment flow.

This agent validates the assessment evaluation flow, including checking that all
required data is available and that the flow executed correctly.
"""

import logging
import time
from typing import Dict, Any
from utils.assessment_flow_validator import AssessmentFlowValidator
from utils.session_manager import session_manager
from chroma import get_chat_session, update_chat_session
from core.logging_helpers import AgentLogger, create_log_context
from core.utils import run_blocking_io

log = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 12000  # Character limit for prompts
MAX_RESUME_TOKENS = 6000  # Token limit for resume text

async def assessment_validator_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the assessment evaluation flow.
    
    Args:
        state: The current state dictionary
        
    Returns:
        Updated state with validation results
    """
    start_time = time.time()
    tenant_id = state.get("tenant_id", "default-tenant")
    log_context = create_log_context("assessment_validator", tenant_id)
    AgentLogger.log_info(log_context, "Assessment validator started")
    
    # Get session ID
    session_id = state.get("session_id")
    if not session_id:
        processing_time = time.time() - start_time
        return {
            "status": "error",
            "message": "No session ID provided",
            "processing_time": processing_time
        }
    
    # Check if we're in the recommendation phase (no submission) or evaluation phase (with submission)
    has_submission = state.get("submission") is not None
    has_assessment_plan = state.get("assessment_plan") is not None
    
    # Only validate flow trigger if we have a submission (evaluation phase)
    # If we only have assessment_plan (recommendation phase), skip submission validation
    if has_submission:
        is_valid, message = AssessmentFlowValidator.validate_flow_trigger(state)
        if not is_valid:
            processing_time = time.time() - start_time
            AgentLogger.log_error(log_context, f"Flow trigger validation failed: {message}", processing_time)
            return {
                "status": "error",
                "message": f"Flow trigger validation failed: {message}",
                "processing_time": processing_time
            }
    elif has_assessment_plan:
        # We're in recommendation phase - no submission validation needed
        AgentLogger.log_info(log_context, "Assessment plan found - skipping submission validation")
    else:
        processing_time = time.time() - start_time
        AgentLogger.log_warning(log_context, "No submission or assessment plan found")
        return {
            "status": "skipped",
            "message": "No submission or assessment plan to validate",
            "processing_time": processing_time
        }
        
    try:
        # Canonical Chroma key for one-doc-per-UID sessions (must match update_step / chat session).
        effective_session_id = session_id
        session = await run_blocking_io(session_manager.get_session, session_id)
        if not session:
            # One-document-per-UID: row may not exist yet if this node runs before any
            # code path called get_or_reuse_session (common race with parallel branches).
            ensure_key = (state.get("uid") or "").strip() or session_id
            AgentLogger.log_info(
                log_context,
                "No session document yet; ensuring UID session before validation",
            )
            await run_blocking_io(
                session_manager.get_or_reuse_session,
                ensure_key,
                "candidate_pipeline",
                "candidate",
            )
            session = await run_blocking_io(session_manager.get_session, ensure_key)
            if session and ensure_key != session_id:
                AgentLogger.log_warning(
                    log_context,
                    "session_id differs from uid document key; using uid for validator I/O",
                )
                effective_session_id = ensure_key
        if not session:
            processing_time = time.time() - start_time
            AgentLogger.log_error(log_context, "Session not found", processing_time)
            return {
                "status": "error",
                "message": "Session not found",
                "processing_time": processing_time
            }
            
        # Update session step - run blocking I/O in thread pool
        await run_blocking_io(
            session_manager.update_step, effective_session_id, "assessment_validator"
        )
        
        # Validate each step in the flow
        validation_results = {}
        flow_steps = [
            "assessment_evaluator",
            "report_generator",
            "career_advisor",
            "market_and_course_recommender",
            "assessment_recommender"
        ]
        
        overall_success = True
        
        for step in flow_steps:
            is_valid, message, step_data = await AssessmentFlowValidator.validate_step_completion(
                effective_session_id, step
            )
            validation_results[step] = {
                "valid": is_valid,
                "message": message
            }
            
            # Log validation result
            await AssessmentFlowValidator.log_validation_result(
                effective_session_id, step, is_valid, message
            )
            
            if not is_valid:
                overall_success = False
                AgentLogger.log_warning(log_context, f"Step validation failed: {step} - {message}")
            else:
                AgentLogger.log_info(log_context, f"Step validation passed: {step}")
                
        # Get chat session data - run blocking I/O in thread pool
        session_data = await run_blocking_io(get_chat_session, effective_session_id)
        if session_data:
            # Add validation results to session data
            from datetime import datetime
            session_data["assessment_validation"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "steps": validation_results,
                "overall_success": overall_success
            }
            
            # Update session data - run blocking I/O in thread pool
            await run_blocking_io(
                update_chat_session, effective_session_id, session_data
            )
        
        processing_time = time.time() - start_time
        
        return {
            "status": "success" if overall_success else "partial_success",
            "validation_results": validation_results,
            "overall_success": overall_success,
            "processing_time": processing_time
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        AgentLogger.log_error(log_context, f"Error in assessment validator: {str(e)}", processing_time)
        return {
            "status": "error",
            "message": f"Error in assessment validator: {str(e)}",
            "processing_time": processing_time
        }

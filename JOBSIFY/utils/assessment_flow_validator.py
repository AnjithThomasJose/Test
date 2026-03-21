"""
Assessment flow validator for the assessment evaluation flow.

This module provides utilities for validating the assessment flow execution,
including validating agent outputs at each step and ensuring data dependencies
are available.
"""

import logging
from typing import Any, Dict, Tuple

from chroma import get_chat_session, update_chat_session
from core.utils import run_blocking_io
from core.validation import (
    validate_assessment_evaluator_output,
    validate_report_generator_output,
    validate_career_advisor_output,
    validate_market_and_course_recommender_output,
    validate_assessment_recommender_output
)

log = logging.getLogger(__name__)


def _sync_validate_step_completion(session_id: str, step_name: str) -> Tuple[bool, str, Any]:
    """Sync Chroma/session reads — run via run_blocking_io from async contexts."""
    if step_name == "assessment_evaluator":
        return validate_assessment_evaluator_output(session_id)
    if step_name == "report_generator":
        return validate_report_generator_output(session_id)
    if step_name == "career_advisor":
        return validate_career_advisor_output(session_id)
    if step_name == "market_and_course_recommender":
        return validate_market_and_course_recommender_output(session_id)
    if step_name == "assessment_recommender":
        return validate_assessment_recommender_output(session_id)
    return False, f"Unknown step name: {step_name}", None


def _sync_log_validation_result(
    session_id: str, step_name: str, is_valid: bool, message: str
) -> None:
    """Sync get/update chat session — run via run_blocking_io."""
    if not session_id:
        log.warning(f"Cannot log validation for {step_name}: No session ID provided")
        return
    try:
        session_data = get_chat_session(session_id)
        if not session_data:
            log.warning(f"Cannot log validation for {step_name}: No session data found")
            return

        if "validation" not in session_data:
            session_data["validation"] = {}

        from datetime import datetime

        session_data["validation"][step_name] = {
            "valid": is_valid,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }

        update_chat_session(session_id, session_data)

    except Exception as e:
        log.error(f"Error logging validation result: {str(e)}")


def _sync_merge_flow_validation(session_id: str, results: Dict[str, Any]) -> None:
    """Persist flow_validation blob — sync Chroma I/O for thread offload."""
    try:
        session_data = get_chat_session(session_id)
        if session_data:
            session_data["flow_validation"] = results
            update_chat_session(session_id, session_data)
    except Exception as e:
        log.error(f"Error updating flow validation: {str(e)}")


class AssessmentFlowValidator:
    """Validator for the assessment evaluation flow."""
    
    @staticmethod
    def validate_flow_trigger(state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that the flow trigger is present and valid.
        
        Args:
            state: The current state dictionary
            
        Returns:
            Tuple of (is_valid, message)
        """
        submission = state.get("submission")
        if not submission:
            return False, "No submission found in state"
        
        # Check for required fields in submission
        required_fields = ["topic", "questions"]
        missing_fields = [field for field in required_fields if field not in submission]
        
        if missing_fields:
            return False, f"Missing required fields in submission: {', '.join(missing_fields)}"
        
        questions = submission.get("questions", [])
        if not questions or not isinstance(questions, list) or len(questions) == 0:
            return False, "No questions found in submission"
            
        # Check if each question has answers
        for i, question in enumerate(questions):
            if not question.get("user_answer"):
                return False, f"Missing user_answer in question {i+1}"
        
        return True, "Flow trigger validation passed"
    
    @staticmethod
    async def validate_step_completion(session_id: str, step_name: str) -> Tuple[bool, str, Any]:
        """
        Validate that a specific step has been completed.
        
        Args:
            session_id: The ID of the session
            step_name: The name of the step to validate
            
        Returns:
            Tuple of (is_valid, message, step_data)
        """
        return await run_blocking_io(_sync_validate_step_completion, session_id, step_name)
    
    @staticmethod
    async def log_validation_result(session_id: str, step_name: str, is_valid: bool, message: str) -> None:
        """
        Log validation result and update session data.
        
        Args:
            session_id: The ID of the session
            step_name: The name of the step that was validated
            is_valid: Whether the validation passed
            message: Validation message
        """
        await run_blocking_io(_sync_log_validation_result, session_id, step_name, is_valid, message)
    
    @staticmethod
    async def validate_flow_completion(session_id: str) -> Dict[str, Any]:
        """
        Validate the entire assessment flow completion.
        
        Args:
            session_id: The ID of the session
            
        Returns:
            Validation results for each step
        """
        steps = [
            "assessment_evaluator",
            "report_generator",
            "career_advisor",
            "market_and_course_recommender",
            "assessment_recommender"
        ]
        
        from datetime import datetime
        results = {
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "steps": {},
            "overall_success": True
        }
        
        for step in steps:
            is_valid, message, _ = await AssessmentFlowValidator.validate_step_completion(session_id, step)
            results["steps"][step] = {
                "valid": is_valid,
                "message": message
            }
            
            if not is_valid:
                results["overall_success"] = False

        await run_blocking_io(_sync_merge_flow_validation, session_id, results)

        return results

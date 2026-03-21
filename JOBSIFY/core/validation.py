"""
Validation framework for Assessment Evaluation Flow.

This module provides utilities for validating each step in the assessment
evaluation flow, including validating agent outputs and ensuring data
dependencies are available before proceeding to the next step.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple, Set, Union
from chroma import get_chat_session
from datetime import datetime

log = logging.getLogger(__name__)

def validate_session_output(session_id: str, agent_name: str, required_keys: List[str] = None) -> Tuple[bool, str, Dict]:
    """
    Validate that a specific agent's output exists in a session and has the required keys.
    
    Args:
        session_id: The ID of the session to check
        agent_name: The name of the agent whose output to validate
        required_keys: Optional list of keys that must be present in the output
        
    Returns:
        Tuple of (success, message, data)
        - success: Boolean indicating if validation passed
        - message: Message describing validation result
        - data: The agent's output data if available, or empty dict
    """
    if not session_id:
        return False, "No session ID provided", {}
    
    try:
        session_data = get_chat_session(session_id)
        if not session_data:
            return False, f"No session data found for session ID: {session_id}", {}
        
        # Check if agent data exists in session
        if agent_name not in session_data:
            return False, f"No {agent_name} data found in session", {}
        
        agent_data = session_data[agent_name]
        
        # Check for required keys if specified
        if required_keys:
            missing_keys = [key for key in required_keys if key not in agent_data]
            if missing_keys:
                return False, f"Missing required keys in {agent_name} data: {', '.join(missing_keys)}", agent_data
        
        return True, f"{agent_name} data validation successful", agent_data
    
    except Exception as e:
        return False, f"Error validating {agent_name} data: {str(e)}", {}

def validate_assessment_evaluator_output(session_id: str) -> Tuple[bool, str, Dict]:
    """
    Validate the assessment evaluator's output in the session.
    
    Args:
        session_id: The ID of the session to check
        
    Returns:
        Tuple of (success, message, data)
    """
    # Required keys for assessment_evaluator output
    required_keys = ["assessment_results"]
    
    success, message, data = validate_session_output(session_id, "assessment_evaluator", required_keys)
    
    # Additional validation specific to assessment_evaluator
    if success and "assessment_results" in data:
        assessment_results = data["assessment_results"]
        
        # Check for required structure in assessment_results
        if not isinstance(assessment_results, dict):
            return False, "assessment_results is not a dictionary", data
        
        # Check for required fields in assessment_results
        ar_required_keys = ["total_score", "max_score"]
        missing_ar_keys = [key for key in ar_required_keys if key not in assessment_results]
        
        if missing_ar_keys:
            return False, f"Missing required keys in assessment_results: {', '.join(missing_ar_keys)}", data
        
        # Validate score values
        try:
            total_score = float(assessment_results["total_score"])
            max_score = float(assessment_results["max_score"])
            
            if total_score < 0 or total_score > max_score:
                return False, f"Invalid score values: total_score={total_score}, max_score={max_score}", data
                
        except (ValueError, TypeError):
            return False, "Score values must be numbers", data
    
    return success, message, data

def validate_report_generator_output(session_id: str) -> Tuple[bool, str, Dict]:
    """
    Validate the report generator's output in the session.
    
    Args:
        session_id: The ID of the session to check
        
    Returns:
        Tuple of (success, message, data)
    """
    # Required keys for report_generator output
    required_keys = ["report"]
    
    success, message, data = validate_session_output(session_id, "report_generator", required_keys)
    
    # Additional validation specific to report_generator
    if success and "report" in data:
        report = data["report"]
        
        # Check for required structure in report
        if not isinstance(report, dict):
            return False, "report is not a dictionary", data
        
        # Check for required fields in report
        report_required_keys = ["total_score", "max_score", "summary"]
        missing_report_keys = [key for key in report_required_keys if key not in report]
        
        if missing_report_keys:
            return False, f"Missing required keys in report: {', '.join(missing_report_keys)}", data
    
    return success, message, data

def validate_career_advisor_output(session_id: str) -> Tuple[bool, str, Dict]:
    """
    Validate the skill and career advisor's output in the session.
    
    Args:
        session_id: The ID of the session to check
        
    Returns:
        Tuple of (success, message, data)
    """
    # Required keys for career_advisor output
    required_keys = ["raw_skill_gap_analysis_output"]
    
    success, message, data = validate_session_output(session_id, "career_advisor", required_keys)
    
    # Additional validation specific to career_advisor
    if success and "raw_skill_gap_analysis_output" in data:
        raw_skill_gap_analysis_output = data["raw_skill_gap_analysis_output"]
        
        # Check for required structure in raw_skill_gap_analysis_output
        if not isinstance(raw_skill_gap_analysis_output, dict):
            return False, "raw_skill_gap_analysis_output is not a dictionary", data
        
        # Check for required fields in raw_skill_gap_analysis_output
        sg_required_keys = ["career_paths", "missing_skills"]
        missing_sg_keys = [key for key in sg_required_keys if key not in raw_skill_gap_analysis_output]
        
        if missing_sg_keys:
            return False, f"Missing required keys in raw_skill_gap_analysis_output: {', '.join(missing_sg_keys)}", data
    
    return success, message, data

def validate_market_and_course_recommender_output(session_id: str) -> Tuple[bool, str, Dict]:
    """
    Validate the market and course recommender's output in the session.
    
    Args:
        session_id: The ID of the session to check
        
    Returns:
        Tuple of (success, message, data)
    """
    # Required keys for market_and_course_recommender output
    required_keys = ["raw_skill_gap_analysis_output"]
    
    success, message, data = validate_session_output(session_id, "market_and_course_recommender", required_keys)
    
    # Additional validation specific to market_and_course_recommender
    if success and "raw_skill_gap_analysis_output" in data:
        raw_skill_gap_analysis_output = data["raw_skill_gap_analysis_output"]
        
        # Check for required structure in raw_skill_gap_analysis_output
        if not isinstance(raw_skill_gap_analysis_output, dict):
            return False, "raw_skill_gap_analysis_output is not a dictionary", data
        
        # Check for required fields in raw_skill_gap_analysis_output
        # Accept both legacy 'recommendation' and canonical 'course_recommendations'
        has_market = "market_insights" in raw_skill_gap_analysis_output
        has_courses = (
            "course_recommendations" in raw_skill_gap_analysis_output or
            "recommendation" in raw_skill_gap_analysis_output
        )
        if not (has_market and has_courses):
            missing = []
            if not has_market:
                missing.append("market_insights")
            if not has_courses:
                missing.append("course_recommendations|recommendation")
            return False, f"Missing required keys in raw_skill_gap_analysis_output: {', '.join(missing)}", data
    
    return success, message, data

def validate_assessment_recommender_output(session_id: str) -> Tuple[bool, str, Dict]:
    """
    Validate the assessment recommender's output in the session.
    
    Args:
        session_id: The ID of the session to check
        
    Returns:
        Tuple of (success, message, data)
    """
    # Required keys for assessment_recommender output
    required_keys = ["assessment_plan", "assessment_needs"]
    
    success, message, data = validate_session_output(session_id, "assessment_recommender", required_keys)
    
    # Additional validation specific to assessment_recommender
    if success:
        assessment_plan = data.get("assessment_plan")
        assessment_needs = data.get("assessment_needs")
        
        # Check for required structure in assessment_plan
        if not isinstance(assessment_plan, list):
            return False, "assessment_plan is not a list", data
        
        # Check for required structure in assessment_needs
        if not isinstance(assessment_needs, dict):
            return False, "assessment_needs is not a dictionary", data
    
    return success, message, data

def validate_assessment_evaluation_flow(session_id: str) -> Dict[str, Any]:
    """
    Validate the entire assessment evaluation flow for a given session.
    
    Args:
        session_id: The ID of the session to check
        
    Returns:
        Dictionary with validation results for each step
    """
    results = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "steps": {},
        "overall_success": True
    }
    
    # Validate each step in the flow
    evaluation_steps = [
        ("assessment_evaluator", validate_assessment_evaluator_output),
        ("report_generator", validate_report_generator_output),
        ("career_advisor", validate_career_advisor_output),
        ("market_and_course_recommender", validate_market_and_course_recommender_output),
        ("assessment_recommender", validate_assessment_recommender_output)
    ]
    
    for step_name, validation_func in evaluation_steps:
        success, message, _ = validation_func(session_id)
        results["steps"][step_name] = {
            "success": success,
            "message": message
        }
        
        if not success:
            results["overall_success"] = False
    
    return results

def validate_dependency_availability(state: Dict[str, Any], dependencies: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that all required dependencies are available in the state.
    
    Args:
        state: The current state dictionary
        dependencies: List of keys that must be present in the state
        
    Returns:
        Tuple of (success, missing_dependencies)
    """
    if not dependencies:
        return True, []
    
    missing = [dep for dep in dependencies if dep not in state]
    return len(missing) == 0, missing

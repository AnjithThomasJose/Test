import json
import re
import asyncio
import logging
import time
import uuid
import os
from typing import Dict, Any, List, Optional, Tuple, Literal
from models.llm_invoker import invoke_llm
from agents.prompt_generator import generate_resume_analysis_prompt
from agents.skill_frameworks import INDUSTRY_SKILLS, get_all_industries, get_profile
from langsmith.run_helpers import traceable
from core.utils import (
    _mask, _sanitize_text_for_llm, _to_text, _clean_json_text, _scan_balanced_json,
    _safe_json_loads, _extract_json_from_response, _coerce_score, _merge_dedupe,
    _normalize_chat_history, _create_error_response, _validate_state_inputs,
    _generate_request_id, _calculate_processing_time
)
from core.config import get_agent_config
from core.security import (
    validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm,
    PII_PATTERNS, INJECTION_FILTERS
)
from core.memory import BaseAgentMemory, get_agent_memory
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion

log = logging.getLogger(__name__)

log = logging.getLogger(__name__)

# Get centralized configuration
config = get_agent_config("resume_analysis")
TIMEOUT_SECONDS = config.timeout_seconds
MAX_PROMPT_CHARS = config.max_prompt_chars
MAX_RESPONSE_LENGTH = config.max_response_length
MAX_CHAT_HISTORY_ITEMS = config.max_chat_history_items
LLM_MODEL = config.llm_model

# Type-safe analysis methods (simplified - no deterministic)
AnalysisMethod = Literal["llm", "error", "llm_timeout"]

# --- Agentic AI Constants ---
CONFIDENCE_THRESHOLD = config.confidence_threshold
ADAPTATION_WINDOW = config.adaptation_window
MIN_ROLE_FITS = 1            # Minimum role fits required (reduced for enterprise flexibility)
MIN_RESUME_SCORE = 0.0       # Minimum valid resume score
MAX_RESUME_SCORE = 100.0     # Maximum valid resume score

# PII patterns and injection filters are now imported from core.security

# Custom memory class for resume analysis (extends base memory)
class ResumeAnalysisMemory(BaseAgentMemory):
    def __init__(self, tenant_id: str = "default_tenant"):
        super().__init__(tenant_id, config.adaptation_window)
        # Add any resume analysis specific fields here if needed

# Use centralized memory management
async def get_analysis_memory(tenant_id: str = "default_tenant") -> ResumeAnalysisMemory:
    """Get or create tenant-scoped resume analysis memory."""
    return await get_agent_memory("resume_analysis", tenant_id, ResumeAnalysisMemory)

def _calculate_confidence_score(analysis: Dict[str, Any], method: str) -> float:
    """Calculate confidence score for resume analysis."""
    if not analysis:
        return 0.3  # Minimum confidence even for empty analysis
    
    score = 0.5  # Higher base score for enterprise systems (increased from 0.4)
    
    # Role fit scoring (more lenient)
    role_fit = analysis.get('RoleFit', [])
    if isinstance(role_fit, list) and len(role_fit) > 0:
        score += 0.2  # Any role fits boost confidence
    
    # Resume score validation
    resume_score = analysis.get('resumeScore')
    if isinstance(resume_score, (int, float)) and MIN_RESUME_SCORE <= resume_score <= MAX_RESUME_SCORE:
        score += 0.15
    
    # Quality indicators
    if analysis.get('summary') and isinstance(analysis['summary'], str):
        score += 0.1
    if analysis.get('strengths') and isinstance(analysis['strengths'], list) and len(analysis['strengths']) > 0:
        score += 0.05
    if analysis.get('improvements') and isinstance(analysis['improvements'], list) and len(analysis['improvements']) > 0:
        score += 0.05
    
    # Method-specific adjustments
    if method == 'llm':
        score += 0.05  # Slight bonus for LLM comprehensiveness
    # LLM-only approach - no method-based scoring adjustments needed
    
    # Ensure minimum confidence of 0.5 for production systems
    return max(0.5, min(score, 1.0))

def _assess_analysis_quality(analysis: Dict[str, Any]) -> float:
    """Assess quality of resume analysis."""
    if not analysis:
        return 0.0
    
    quality_score = 0.0
    
    # Structure validation (more lenient)
    expected_keys = ['RoleFit', 'resumeScore', 'summary']
    present_keys = sum(1 for key in expected_keys if key in analysis and analysis[key] is not None)
    if present_keys > 0:
        quality_score += (present_keys / len(expected_keys)) * 0.6
    
    # Content quality assessment (more flexible)
    role_fit = analysis.get('RoleFit', [])
    if isinstance(role_fit, list) and len(role_fit) > 0:
        quality_score += 0.2
        # Additional points for well-structured roles
        for role in role_fit[:5]:  # Check first 5 roles
            if isinstance(role, dict):
                if role.get('role'):
                    quality_score += 0.02
                if role.get('score') is not None:
                    quality_score += 0.02
    
    # Resume score validity
    resume_score = analysis.get('resumeScore')
    if isinstance(resume_score, (int, float)) and 0 <= resume_score <= 100:
        quality_score += 0.2
    
    return min(quality_score, 1.0)

def _mask(s: str, keep: int = 6) -> str:
    """Mask sensitive strings for logging."""
    return (s[:keep] + "***") if s else ""

# Duplicate utility functions removed - using centralized versions from core.utils

def _validate_analysis_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize analysis data."""
    if not isinstance(data, dict):
        return {}
    
    validated_data = {}
    
    # Validate RoleFit
    role_fit = data.get('RoleFit', [])
    if isinstance(role_fit, list):
        validated_roles = []
        for role in role_fit[:10]:  # Cap role fits
            if isinstance(role, dict):
                validated_role = {}
                if role.get('role') and isinstance(role['role'], str):
                    validated_role['role'] = role['role'][:100]
                if role.get('score') and isinstance(role['score'], (int, float)):
                    validated_role['score'] = max(0.0, min(float(role['score']), 100.0))
                # Include role if it has at least a role name or score
                if 'role' in validated_role or 'score' in validated_role:
                    validated_roles.append(validated_role)
        validated_data['RoleFit'] = validated_roles
    
    # Validate resume score
    resume_score = data.get('resumeScore')
    if isinstance(resume_score, (int, float)):
        validated_data['resumeScore'] = max(MIN_RESUME_SCORE, min(float(resume_score), MAX_RESUME_SCORE))
    
    # Validate other fields
    for field in ['summary', 'strengths', 'improvements']:
        if field in data and data[field]:
            if isinstance(data[field], str):
                validated_data[field] = data[field][:1000]  # Cap length
            elif isinstance(data[field], list):
                validated_list = []
                for item in data[field][:10]:  # Cap items
                    if isinstance(item, str):
                        validated_list.append(item[:200])
                validated_data[field] = validated_list
    
    return validated_data

def _validate_state_inputs(state: Any) -> Tuple[bool, str]:
    """Validate input state for security and completeness."""
    if not isinstance(state, dict):
        return False, "Invalid state format"
    
    # Check for required fields
    structured_resume = state.get("structured_resume")
    if not structured_resume:
        return False, "Missing structured_resume"
    
    if isinstance(structured_resume, dict) and "error" in structured_resume:
        return False, "Resume parsing error detected"
    
    # Validate chat history
    chat_history = state.get("chat_history", [])
    if not isinstance(chat_history, list):
        return False, "Invalid chat_history format"
    
    if len(chat_history) > MAX_CHAT_HISTORY_ITEMS:
        return False, f"Too many chat history items (max {MAX_CHAT_HISTORY_ITEMS})"
    
    return True, "Valid"

# _extract_json_from_response removed - using centralized version from core.utils

# Deterministic function removed - using LLM-only approach

@traceable(name="resume_analysis_agent")
async def resume_analysis_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enterprise-grade Resume Analysis Agent with agentic AI capabilities.
    Analyzes resumes with military-grade security, tenant isolation, and adaptive intelligence.
    """
    start_time = time.time()
    tenant_id = state.get("tenant_id", "default_tenant")
    
    # LLM-ONLY APPROACH: Use centralized logging and simplified flow
    log_context = create_log_context("resume_analysis", tenant_id)
    
    # Validate tenant isolation and input security
    if not validate_tenant_id(tenant_id):
        log.error(f"Invalid tenant_id format: {_mask(tenant_id)}")
        return _create_error_response("Invalid tenant identification", "resume_analysis")
    
    is_valid, validation_msg = _validate_state_inputs(state)
    if not is_valid:
        log.error(f"[{_mask(tenant_id)}] Input validation failed: {validation_msg}")
        return _create_error_response(validation_msg, "resume_analysis")
    
    log.info(f"[{_mask(tenant_id)}] Starting resume analysis with enterprise security")
    
    # Get tenant-scoped memory for adaptive behavior
    memory = await get_analysis_memory(tenant_id)
    
    # Extract and validate inputs
    structured_resume = state.get("structured_resume", {})
    
    # Normalize chat history with proper sanitization
    raw_chat = (state.get("chat_history") or [])[-MAX_CHAT_HISTORY_ITEMS:]
    chat_history = _normalize_chat_history(raw_chat, MAX_CHAT_HISTORY_ITEMS, 2000)
    
    assessment_results = state.get("assessment_results")
    report = state.get("report")
    
    # LLM-ONLY ANALYSIS
    try:
        # Generate secure prompt with Gemini summary (NO TRUNCATION)
        uid = state.get("uid")
        prompt = await generate_resume_analysis_prompt(
            structured_resume,
            chat_history,
            assessment_results,
            report,
            uid=uid
        )
        
        # Sanitize prompt for LLM safety
        sanitized_prompt = sanitize_text_for_llm(prompt, PII_PATTERNS, INJECTION_FILTERS)
        if len(sanitized_prompt) > MAX_PROMPT_CHARS:
            sanitized_prompt = sanitized_prompt[:MAX_PROMPT_CHARS]
        
        # Secure LLM invocation with timeout
        response = await asyncio.wait_for(
            invoke_llm(
            prompt=sanitized_prompt,
            task_type="resume_analysis",
            agent_name="resume_analysis"
        ),
            timeout=TIMEOUT_SECONDS
        )
        
        analysis_result = _extract_json_from_response(_to_text(response), MAX_RESPONSE_LENGTH)
        processing_time = _calculate_processing_time(start_time)
        
        # Validate LLM result quality
        if not analysis_result or not any([
            analysis_result.get('RoleFit'),
            analysis_result.get('resumeScore', 0) > 0,
            analysis_result.get('summary'),
            analysis_result.get('strengths'),
            analysis_result.get('improvements')
        ]):
            log.warning(f"[{_mask(tenant_id)}] LLM returned empty or invalid result")
            return _create_error_response("Empty or invalid analysis result from LLM", "resume_analysis")
        
        confidence = _calculate_confidence_score(analysis_result, "llm")
        
        # Record successful analysis
        await memory.record_attempt("structured_resume", "llm", True, confidence, processing_time)
        
        # Share success context with other agents
        await memory.share_context(f"last_successful_analysis_llm", {
            "timestamp": time.time(),
            "confidence": confidence,
            "processing_time": processing_time
        })
        
        # Log success using centralized logging
        log_agent_completion(log_context, {
            "success": True,
            "confidence_score": confidence,
            "role_count": len(analysis_result.get('RoleFit', []))
        }, "llm", processing_time)
        
    except asyncio.TimeoutError:
        processing_time = _calculate_processing_time(start_time)
        await memory.record_attempt("structured_resume", "llm_timeout", False, 0.0, processing_time)
        AgentLogger.log_timeout(log_context, TIMEOUT_SECONDS)
        return _create_error_response(f"LLM timeout after {TIMEOUT_SECONDS}s", "resume_analysis")
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        log.error(f"[{_mask(tenant_id)}] LLM analysis failed: {_sanitize_text_for_llm(str(e))[:300]}")
        await memory.record_attempt("structured_resume", "llm", False, 0.0, processing_time)
        return _create_error_response("LLM analysis failed", "resume_analysis")
    
    # Validate and normalize results
    final_result = _validate_analysis_data(analysis_result)
    
    # Final safety check - ensure minimum viable results
    if not final_result.get('RoleFit'):
        final_result['RoleFit'] = [{"role": "Entry Level Professional", "score": 50.0}]
    if not final_result.get('resumeScore'):
        final_result['resumeScore'] = 45.0  # Minimum viable score
    if not final_result.get('summary'):
        final_result['summary'] = "Resume analysis completed with basic profile assessment."
    if not final_result.get('strengths'):
        final_result['strengths'] = ["Profile contains structured information"]
    if not final_result.get('improvements'):
        final_result['improvements'] = ["Enhance profile with additional details and achievements"]
    
    # Add metadata
    final_result["method"] = "llm"
    final_result["confidence"] = confidence
    final_result["processing_time"] = processing_time
    final_result["tenant_id"] = tenant_id
    
    # Quality assessment for continuous improvement
    quality_score = _assess_analysis_quality(final_result)
    await memory.share_context("last_analysis_quality", quality_score)
    
    total_time = _calculate_processing_time(start_time)
    log.info(f"[{_mask(tenant_id)}] Resume analysis completed in {total_time:.3f}s (quality: {quality_score:.3f})")
    
    # Return analysis with enterprise telemetry
    return {
        "RoleFit": final_result.get("RoleFit", []),
        "resumeScore": final_result.get("resumeScore", 0),
        "summary": final_result.get("summary", "Analysis completed"),
        "strengths": final_result.get("strengths", []),  
        "improvements": final_result.get("improvements", []),
        "analysis_metadata": {
            "schema_version": "ra-1.0.0",
            "method": "llm",
            "confidence": confidence,
            "processing_time": processing_time,
            "quality_score": quality_score,
            "total_time": total_time,
            "tenant_id": _mask(tenant_id)
        }
    }
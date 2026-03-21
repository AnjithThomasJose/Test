"""
Centralized logging helpers for all KAFIN agents.

This module provides standardized logging patterns and utilities
to ensure consistent logging across all agents.
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional
from core.utils import _mask


log = logging.getLogger(__name__)


class AgentLogger:
    """Centralized logging for all agents."""
    
    @staticmethod
    def start_agent_log(agent_name: str, tenant_id: str) -> Dict[str, Any]:
        """Start agent execution logging."""
        start_time = time.time()
        request_id = str(uuid.uuid4())[:8]
        
        log.info(f"[TENANT:{_mask(tenant_id)}] {agent_name}: Starting analysis (request_id={request_id})")
        
        return {
            "start_time": start_time,
            "request_id": request_id,
            "agent_name": agent_name,
            "tenant_id": tenant_id
        }
    
    @staticmethod
    def log_success(log_context: Dict[str, Any], method: str, confidence: float, processing_time: float):
        """Log successful agent execution."""
        log.info(
            f"[TENANT:{_mask(log_context['tenant_id'])}] {log_context['agent_name']}: "
            f"Analysis complete (method={method}, confidence={confidence:.2f}, "
            f"time={processing_time:.2f}s)"
        )
    
    @staticmethod
    def log_error(log_context: Dict[str, Any], error: str, processing_time: float):
        """Log agent error."""
        log.error(
            f"[TENANT:{_mask(log_context['tenant_id'])}] {log_context['agent_name']}: "
            f"Error: {error} (time={processing_time:.2f}s)"
        )
    
    @staticmethod
    def log_info(log_context: Dict[str, Any], message: str):
        """Log agent info message."""
        log.info(
            f"[TENANT:{_mask(log_context['tenant_id'])}] {log_context['agent_name']}: "
            f"{message}"
        )
    
    @staticmethod
    def log_warning(log_context: Dict[str, Any], message: str):
        """Log agent warning."""
        log.warning(
            f"[TENANT:{_mask(log_context['tenant_id'])}] {log_context['agent_name']}: "
            f"Warning: {message}"
        )
    
    @staticmethod
    def log_debug(log_context: Dict[str, Any], message: str):
        """Log agent debug message."""
        log.debug(
            f"[TENANT:{_mask(log_context['tenant_id'])}] {log_context['agent_name']}: "
            f"Debug: {message}"
        )
    
    @staticmethod
    def log_method_selection(log_context: Dict[str, Any], chosen_method: str, 
                           llm_success_rate: float, deterministic_success_rate: float):
        """Log method selection decision."""
        log.info(
            f"[TENANT:{_mask(log_context['tenant_id'])}] {log_context['agent_name']}: "
            f"Using analysis method: {chosen_method} "
            f"(LLM success: {llm_success_rate:.2f}, Deterministic: {deterministic_success_rate:.2f})"
        )
    
    @staticmethod
    def log_llm_call(log_context: Dict[str, Any], prompt_size: int, model: str):
        """Log LLM call details."""
        log.debug(
            f"[TENANT:{_mask(log_context['tenant_id'])}] {log_context['agent_name']}: "
            f"LLM call (prompt_size={prompt_size}, model={model})"
        )
    
    @staticmethod
    def log_timeout(log_context: Dict[str, Any], timeout_seconds: int):
        """Log timeout event."""
        log.warning(
            f"[TENANT:{_mask(log_context['tenant_id'])}] {log_context['agent_name']}: "
            f"Timeout after {timeout_seconds}s"
        )
    
    @staticmethod
    def log_fallback(log_context: Dict[str, Any], fallback_method: str, reason: str):
        """Log fallback method usage."""
        log.info(
            f"[TENANT:{_mask(log_context['tenant_id'])}] {log_context['agent_name']}: "
            f"Using fallback method: {fallback_method} (reason: {reason})"
        )
    
    @staticmethod
    def log_validation_failure(log_context: Dict[str, Any], validation_error: str):
        """Log validation failure."""
        log.warning(
            f"[TENANT:{_mask(log_context['tenant_id'])}] {log_context['agent_name']}: "
            f"Validation failed: {validation_error}"
        )
    
    @staticmethod
    def log_security_event(log_context: Dict[str, Any], event_type: str, details: str):
        """Log security-related events."""
        log.warning(
            f"[TENANT:{_mask(log_context['tenant_id'])}] {log_context['agent_name']}: "
            f"Security event [{event_type}]: {details}"
        )
    
    @staticmethod
    def log_performance(log_context: Dict[str, Any], metric: str, value: float, unit: str = ""):
        """Log performance metrics."""
        log.info(
            f"[TENANT:{_mask(log_context['tenant_id'])}] {log_context['agent_name']}: "
            f"Performance [{metric}]: {value:.2f}{unit}"
        )


def create_log_context(agent_name: str, tenant_id: str) -> Dict[str, Any]:
    """Create a standardized log context for an agent."""
    return AgentLogger.start_agent_log(agent_name, tenant_id)


def log_agent_completion(log_context: Dict[str, Any], result: Dict[str, Any], 
                        method: str, processing_time: float):
    """Log agent completion with result details."""
    confidence = result.get('confidence_score', 0.0)
    success = result.get('success', True)
    
    if success:
        AgentLogger.log_success(log_context, method, confidence, processing_time)
    else:
        error_msg = result.get('error', 'Unknown error')
        AgentLogger.log_error(log_context, error_msg, processing_time)


def log_agent_start(agent_name: str, tenant_id: str, method: str = None) -> Dict[str, Any]:
    """Log agent start and return log context."""
    log_context = create_log_context(agent_name, tenant_id)
    if method:
        AgentLogger.log_debug(log_context, f"Starting with method: {method}")
    return log_context


def log_agent_error(log_context: Dict[str, Any], error: Exception, processing_time: float):
    """Log agent error with exception details."""
    error_msg = str(error) if error else "Unknown error"
    AgentLogger.log_error(log_context, error_msg, processing_time)

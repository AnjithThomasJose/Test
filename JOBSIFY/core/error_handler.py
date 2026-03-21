"""
Centralized error handling middleware for all agents.
This module provides standardized error handling, logging, and recovery mechanisms.
"""

import asyncio
import logging
import time
import traceback
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Callable, Awaitable, Union

log = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"           # Minor issues, can continue
    MEDIUM = "medium"      # Significant issues, may need fallback
    HIGH = "high"         # Critical issues, requires fallback
    CRITICAL = "critical"  # System failure, stop processing


class ErrorCategory(Enum):
    """Error categories for better classification and handling."""
    VALIDATION = "validation"           # Input validation errors
    TIMEOUT = "timeout"                # Timeout errors
    LLM_ERROR = "llm_error"            # LLM service errors
    NETWORK = "network"                # Network connectivity issues
    PERMISSION = "permission"           # Authorization/authentication errors
    RESOURCE = "resource"              # Resource exhaustion (memory, CPU)
    DATA = "data"                      # Data processing errors
    SYSTEM = "system"                 # System-level errors
    UNKNOWN = "unknown"                # Unclassified errors


@dataclass
class ErrorContext:
    """Context information for error handling."""
    agent_name: str
    tenant_id: str
    request_id: str
    error_type: type
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: float
    processing_time: float
    fallback_available: bool = False
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class ErrorResponse:
    """Standardized error response structure."""
    success: bool = False
    error_code: str = ""
    error_message: str = ""
    error_category: str = ""
    error_severity: str = ""
    fallback_used: bool = False
    retry_count: int = 0
    request_id: str = ""
    timestamp: float = 0.0
    processing_time: float = 0.0
    agent_status: str = "error"
    confidence_score: float = 0.0


class AgentErrorHandler:
    """Centralized error handler for all agents."""
    
    def __init__(self):
        self.error_stats = {}
        self.circuit_breakers = {}
        self.retry_configs = {
            ErrorCategory.TIMEOUT: {"max_retries": 2, "backoff_factor": 1.5},
            ErrorCategory.LLM_ERROR: {"max_retries": 3, "backoff_factor": 2.0},
            ErrorCategory.NETWORK: {"max_retries": 3, "backoff_factor": 1.5},
            ErrorCategory.SYSTEM: {"max_retries": 1, "backoff_factor": 1.0},
        }
    
    def _classify_error(self, error: Exception, agent_name: str) -> tuple[ErrorSeverity, ErrorCategory]:
        """Classify error severity and category."""
        error_type = type(error)
        error_msg = str(error).lower()
        
        # Timeout errors
        if isinstance(error, asyncio.TimeoutError):
            return ErrorSeverity.MEDIUM, ErrorCategory.TIMEOUT
        
        # Cancelled errors (async operations cancelled)
        if isinstance(error, asyncio.CancelledError):
            return ErrorSeverity.MEDIUM, ErrorCategory.TIMEOUT
        
        # LLM-related errors
        if "llm" in error_msg or "model" in error_msg or "api" in error_msg:
            return ErrorSeverity.HIGH, ErrorCategory.LLM_ERROR
        
        # Network errors
        if any(term in error_msg for term in ["connection", "network", "timeout", "unreachable"]):
            return ErrorSeverity.HIGH, ErrorCategory.NETWORK
        
        # Permission errors
        if any(term in error_msg for term in ["permission", "unauthorized", "forbidden", "access denied"]):
            return ErrorSeverity.HIGH, ErrorCategory.PERMISSION
        
        # Resource errors
        if any(term in error_msg for term in ["memory", "resource", "quota", "limit"]):
            return ErrorSeverity.HIGH, ErrorCategory.RESOURCE
        
        # Validation errors
        if isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorSeverity.MEDIUM, ErrorCategory.VALIDATION
        
        # System errors
        if isinstance(error, (OSError, RuntimeError)):
            return ErrorSeverity.CRITICAL, ErrorCategory.SYSTEM
        
        # Default classification
        return ErrorSeverity.MEDIUM, ErrorCategory.UNKNOWN
    
    def _should_retry(self, context: ErrorContext) -> bool:
        """Determine if error should be retried."""
        if context.retry_count >= context.max_retries:
            return False
        
        # Don't retry validation errors
        if context.category == ErrorCategory.VALIDATION:
            return False
        
        # Don't retry permission errors
        if context.category == ErrorCategory.PERMISSION:
            return False
        
        # Don't retry critical system errors
        if context.severity == ErrorSeverity.CRITICAL:
            return False
        
        return True
    
    def _calculate_backoff_delay(self, context: ErrorContext) -> float:
        """Calculate backoff delay for retries."""
        config = self.retry_configs.get(context.category, {"backoff_factor": 1.5})
        base_delay = 1.0
        return base_delay * (config["backoff_factor"] ** context.retry_count)
    
    def _create_error_response(self, context: ErrorContext, fallback_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Create standardized error response."""
        response = ErrorResponse(
            error_code=f"{context.category.value}_{context.severity.value}",
            error_message=context.error_message,
            error_category=context.category.value,
            error_severity=context.severity.value,
            fallback_used=context.fallback_available,
            retry_count=context.retry_count,
            request_id=context.request_id,
            timestamp=context.timestamp,
            processing_time=context.processing_time,
            agent_status=f"{context.agent_name}_error",
            confidence_score=0.0
        )
        
        # Convert to dict and add fallback data if available
        result = {
            **response.__dict__,
            "success": False
        }
        
        if fallback_data:
            result.update(fallback_data)
            result["fallback_used"] = True
        
        return result
    
    def _log_error(self, context: ErrorContext, traceback_str: str):
        """Log error with appropriate level based on severity."""
        log_data = {
            "agent": context.agent_name,
            "tenant_id": context.tenant_id,
            "request_id": context.request_id,
            "error_type": context.error_type.__name__,
            "error_message": context.error_message,
            "severity": context.severity.value,
            "category": context.category.value,
            "processing_time": context.processing_time,
            "retry_count": context.retry_count
        }
        
        if context.severity == ErrorSeverity.CRITICAL:
            log.critical(f"CRITICAL ERROR in {context.agent_name}: {context.error_message}", extra=log_data)
        elif context.severity == ErrorSeverity.HIGH:
            log.error(f"HIGH SEVERITY ERROR in {context.agent_name}: {context.error_message}", extra=log_data)
        elif context.severity == ErrorSeverity.MEDIUM:
            log.warning(f"MEDIUM SEVERITY ERROR in {context.agent_name}: {context.error_message}", extra=log_data)
        else:
            log.info(f"LOW SEVERITY ERROR in {context.agent_name}: {context.error_message}", extra=log_data)
        
        # Log traceback for high severity errors
        if context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            log.debug(f"Traceback for {context.agent_name}: {traceback_str}")
    
    async def handle_error(
        self,
        error: Exception,
        agent_name: str,
        tenant_id: str,
        request_id: str,
        processing_time: float,
        fallback_fn: Optional[Callable] = None,
        retry_fn: Optional[Callable] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """Main error handling method."""
        
        # Classify error
        severity, category = self._classify_error(error, agent_name)
        
        # Safely convert error message to string, handling edge cases like slice objects
        try:
            if isinstance(error, slice):
                error_message = f"slice({error.start}, {error.stop}, {error.step})"
            else:
                error_message = str(error)
        except Exception:
            # Fallback if string conversion fails
            error_message = f"{type(error).__name__}: Unable to convert error to string"
        
        # Create error context
        context = ErrorContext(
            agent_name=agent_name,
            tenant_id=tenant_id,
            request_id=request_id,
            error_type=type(error),
            error_message=error_message,
            severity=severity,
            category=category,
            timestamp=time.time(),
            processing_time=processing_time,
            retry_count=retry_count,
            max_retries=self.retry_configs.get(category, {}).get("max_retries", 3)
        )
        
        # Log error
        traceback_str = traceback.format_exc()
        self._log_error(context, traceback_str)
        
        # Try fallback if available
        if fallback_fn and context.fallback_available:
            try:
                log.info(f"Attempting fallback for {agent_name}")
                fallback_result = await fallback_fn()
                context.fallback_available = True
                return self._create_error_response(context, fallback_result)
            except Exception as fallback_error:
                log.error(f"Fallback failed for {agent_name}: {fallback_error}")
        
        # Try retry if appropriate
        if retry_fn and self._should_retry(context):
            delay = self._calculate_backoff_delay(context)
            log.info(f"Retrying {agent_name} after {delay}s (attempt {retry_count + 1})")
            await asyncio.sleep(delay)
            
            try:
                return await retry_fn()
            except Exception as retry_error:
                # Recursively handle retry error
                return await self.handle_error(
                    retry_error, agent_name, tenant_id, request_id,
                    processing_time, fallback_fn, retry_fn, retry_count + 1
                )
        
        # Return error response
        return self._create_error_response(context)


# Global error handler instance
error_handler = AgentErrorHandler()


def with_error_handling(
    agent_name: str,
    fallback_fn: Optional[Callable] = None,
    enable_retry: bool = True
):
    """Decorator to add centralized error handling to agent functions."""
    
    def decorator(agent_fn: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        async def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
            tenant_id = state.get("tenant_id", "default")
            request_id = state.get("request_id", str(uuid.uuid4())[:8])
            start_time = time.time()
            
            async def retry_fn():
                return await agent_fn(state)
            
            try:
                result = await agent_fn(state)
                
                # Log success
                processing_time = time.time() - start_time
                log.info(f"Agent {agent_name} completed successfully", extra={
                    "agent": agent_name,
                    "tenant_id": tenant_id,
                    "request_id": request_id,
                    "processing_time": processing_time
                })
                
                return result
                
            except Exception as error:
                processing_time = time.time() - start_time
                
                # Use centralized error handling
                return await error_handler.handle_error(
                    error=error,
                    agent_name=agent_name,
                    tenant_id=tenant_id,
                    request_id=request_id,
                    processing_time=processing_time,
                    fallback_fn=fallback_fn,
                    retry_fn=retry_fn if enable_retry else None
                )
        
        return wrapped
    return decorator


# Convenience functions for common error scenarios
def create_timeout_error_response(agent_name: str, timeout_seconds: float) -> Dict[str, Any]:
    """Create standardized timeout error response."""
    return {
        "success": False,
        "error_code": "timeout_high",
        "error_message": f"Agent {agent_name} timed out after {timeout_seconds}s",
        "error_category": "timeout",
        "error_severity": "high",
        "agent_status": f"{agent_name}_timeout",
        "confidence_score": 0.0,
        "processing_time": timeout_seconds
    }


def create_validation_error_response(agent_name: str, validation_error: str) -> Dict[str, Any]:
    """Create standardized validation error response."""
    return {
        "success": False,
        "error_code": "validation_medium",
        "error_message": f"Validation error in {agent_name}: {validation_error}",
        "error_category": "validation",
        "error_severity": "medium",
        "agent_status": f"{agent_name}_validation_error",
        "confidence_score": 0.0
    }


def create_llm_error_response(agent_name: str, llm_error: str) -> Dict[str, Any]:
    """Create standardized LLM error response."""
    return {
        "success": False,
        "error_code": "llm_error_high",
        "error_message": f"LLM error in {agent_name}: {llm_error}",
        "error_category": "llm_error",
        "error_severity": "high",
        "agent_status": f"{agent_name}_llm_error",
        "confidence_score": 0.0
    }


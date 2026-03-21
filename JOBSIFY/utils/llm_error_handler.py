"""
Enhanced LLM Error Handler for CancelledError and Timeout Management

This module provides robust error handling for LLM operations, specifically
addressing CancelledError exceptions that can occur during async operations.
"""

import asyncio
import logging
import os
import random
from typing import Any, Callable, Optional, Dict, Union
from functools import wraps

log = logging.getLogger(__name__)


def _retry_sleep_seconds(attempt: int, backoff_factor: float) -> float:
    """Exponential backoff with additive jitter (reduces thundering herd after outages).

    Base delay is ``backoff_factor ** attempt``; adds uniform random in ``[0, jitter_max]``.
    Configure max jitter via env ``LLM_RETRY_JITTER_MAX_SECONDS`` (default ``1.0``, ``0`` disables jitter).
    """
    base = float(backoff_factor ** attempt)
    jitter_max = float(os.getenv("LLM_RETRY_JITTER_MAX_SECONDS", "1.0"))
    jitter = random.uniform(0.0, max(0.0, jitter_max))
    total = base + jitter
    log.debug(
        "LLM retry sleep: attempt=%s base=%.3fs jitter_max=%.3fs actual=%.3fs",
        attempt + 1,
        base,
        jitter_max,
        total,
    )
    return total

def _extract_retry_after(exception: Exception) -> Optional[float]:
    """Extract Retry-After value from an exception if available.
    
    Tries multiple approaches to find the retry-after value:
    1. Check for 'retry_after' attribute on the exception
    2. Check for 'response' attribute with 'headers' containing 'Retry-After'
    3. Parse the error message for retry-after hints (e.g., "retry after X seconds")
    
    Args:
        exception: The caught exception
        
    Returns:
        Retry-after value in seconds, or None if not found
    """
    # Try direct attribute (some SDKs set this)
    if hasattr(exception, 'retry_after'):
        try:
            return float(exception.retry_after)
        except (ValueError, TypeError):
            pass
    
    # Try response headers (HTTP-based errors)
    if hasattr(exception, 'response'):
        response = exception.response
        if hasattr(response, 'headers'):
            retry_header = response.headers.get('Retry-After') or response.headers.get('retry-after')
            if retry_header:
                try:
                    return float(retry_header)
                except ValueError:
                    # Could be HTTP-date format, but we'll skip that complexity
                    pass
    
    # Try to parse from error message (fallback)
    error_str = str(exception).lower()
    import re
    # Match patterns like "retry after 30 seconds" or "wait 60s"
    patterns = [
        r'retry.?after\s*:?\s*(\d+(?:\.\d+)?)\s*(?:seconds?|s\b)',
        r'wait\s+(\d+(?:\.\d+)?)\s*(?:seconds?|s\b)',
        r'(\d+(?:\.\d+)?)\s*(?:seconds?|s)\s*(?:backoff|wait|delay)',
    ]
    for pattern in patterns:
        match = re.search(pattern, error_str)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    
    return None


class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass

class LLMTimeoutError(LLMError):
    """LLM operation timed out"""
    pass

class LLMCancelledError(LLMError):
    """LLM operation was cancelled"""
    pass

class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded.
    
    Attributes:
        retry_after: Suggested wait time in seconds from Retry-After header (if available)
    """
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after

async def safe_llm_call(
    llm_callable: Callable,
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    agent_name: str = "unknown"
) -> Any:
    """
    Safely execute an LLM call with comprehensive error handling.
    
    Args:
        llm_callable: Async callable that performs the LLM operation
        timeout: Timeout in seconds for each attempt
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier (per attempt: ``backoff_factor ** attempt``)
        agent_name: Name of the calling agent for logging

    Retry delays add **additive jitter** (uniform ``[0, LLM_RETRY_JITTER_MAX_SECONDS]``, default 1.0s)
    on top of exponential backoff to reduce synchronized retries after API recovery.

    Returns:
        Result of the LLM call
        
    Raises:
        LLMTimeoutError: If all attempts timeout
        LLMCancelledError: If operation is cancelled
        LLMRateLimitError: If rate limited
        LLMError: For other LLM-related errors
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            log.info(f"🔄 {agent_name}: LLM attempt {attempt + 1}/{max_retries}")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                llm_callable(),
                timeout=timeout
            )
            
            log.info(f"✅ {agent_name}: LLM call successful on attempt {attempt + 1}")
            return result
            
        except asyncio.CancelledError as e:
            error_msg = f"{agent_name}: LLM operation cancelled on attempt {attempt + 1}"
            log.warning(f"💥 {error_msg}")
            last_error = LLMCancelledError(error_msg)
            
            if attempt == max_retries - 1:
                break
                
            wait_time = _retry_sleep_seconds(attempt, backoff_factor)
            log.info(f"⏳ {agent_name}: Waiting {wait_time:.2f}s before retry (cancelled)...")
            await asyncio.sleep(wait_time)
            continue
            
        except asyncio.TimeoutError as e:
            error_msg = f"{agent_name}: LLM call timed out after {timeout}s on attempt {attempt + 1}"
            log.warning(f"💥 {error_msg}")
            last_error = LLMTimeoutError(error_msg)
            
            if attempt == max_retries - 1:
                break
                
            wait_time = _retry_sleep_seconds(attempt, backoff_factor)
            log.info(f"⏳ {agent_name}: Waiting {wait_time:.2f}s before retry (timeout)...")
            await asyncio.sleep(wait_time)
            continue
            
        except Exception as e:
            error_msg = f"{agent_name}: LLM call failed on attempt {attempt + 1}: {str(e)}"
            log.error(f"💥 {error_msg}")
            
            # Classify the error and extract Retry-After if available
            if "rate" in str(e).lower() or "quota" in str(e).lower():
                retry_after = _extract_retry_after(e)
                last_error = LLMRateLimitError(error_msg, retry_after=retry_after)
                if retry_after:
                    log.info(f"📋 {agent_name}: Retry-After header found: {retry_after}s")
            elif "timeout" in str(e).lower():
                last_error = LLMTimeoutError(error_msg)
            elif "cancelled" in str(e).lower():
                last_error = LLMCancelledError(error_msg)
            else:
                last_error = LLMError(error_msg)
            
            if attempt == max_retries - 1:
                break
                
            wait_time = _retry_sleep_seconds(attempt, backoff_factor)
            log.info(f"⏳ {agent_name}: Waiting {wait_time:.2f}s before retry...")
            await asyncio.sleep(wait_time)
            continue
    
    # All attempts failed
    log.error(f"❌ {agent_name}: All {max_retries} LLM attempts failed")
    raise last_error

def with_llm_error_handling(
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_factor: float = 2.0
):
    """
    Decorator to add comprehensive LLM error handling to async functions.
    
    Args:
        timeout: Timeout in seconds for each attempt
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            agent_name = getattr(func, '__name__', 'unknown')
            
            async def llm_call():
                return await func(*args, **kwargs)
            
            return await safe_llm_call(
                llm_call,
                timeout=timeout,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                agent_name=agent_name
            )
        return wrapper
    return decorator

async def handle_llm_cancellation(
    operation_name: str,
    fallback_value: Any = None,
    log_context: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Handle CancelledError gracefully with proper logging and fallback.
    
    Args:
        operation_name: Name of the operation being performed
        fallback_value: Value to return if operation is cancelled
        log_context: Additional context for logging
        
    Returns:
        Result of operation or fallback value
    """
    try:
        # This would be called within a try-except block
        pass
    except asyncio.CancelledError:
        context_str = f" | {log_context}" if log_context else ""
        log.warning(f"⚠️ {operation_name} was cancelled{context_str}")
        
        if fallback_value is not None:
            log.info(f"🔄 {operation_name}: Using fallback value")
            return fallback_value
        else:
            log.error(f"❌ {operation_name}: No fallback available for cancelled operation")
            raise LLMCancelledError(f"{operation_name} was cancelled and no fallback available")

def create_robust_llm_call(
    model_instance,
    prompt: str,
    timeout: float = 30.0,
    max_retries: int = 3
) -> Callable:
    """
    Create a robust LLM call function with error handling.
    
    Args:
        model_instance: The LLM model instance
        prompt: The prompt to send
        timeout: Timeout for each attempt
        max_retries: Maximum retry attempts
        
    Returns:
        Async callable that handles errors gracefully
    """
    async def robust_call():
        return await safe_llm_call(
            lambda: model_instance.ainvoke(prompt),
            timeout=timeout,
            max_retries=max_retries,
            agent_name=f"{type(model_instance).__name__}"
        )
    
    return robust_call

# Utility function for common LLM error patterns
def is_cancellation_error(error: Exception) -> bool:
    """Check if an error is related to cancellation."""
    return (
        isinstance(error, asyncio.CancelledError) or
        isinstance(error, LLMCancelledError) or
        "cancelled" in str(error).lower()
    )

def is_timeout_error(error: Exception) -> bool:
    """Check if an error is related to timeout."""
    return (
        isinstance(error, asyncio.TimeoutError) or
        isinstance(error, LLMTimeoutError) or
        "timeout" in str(error).lower()
    )

def is_rate_limit_error(error: Exception) -> bool:
    """Check if an error is related to rate limiting."""
    return (
        isinstance(error, LLMRateLimitError) or
        "rate" in str(error).lower() or
        "quota" in str(error).lower() or
        "limit" in str(error).lower()
    )


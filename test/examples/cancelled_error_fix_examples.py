"""
Example: How to Update Existing Agents to Handle CancelledError

This file shows how to update existing agent code to properly handle
CancelledError exceptions using the new error handling utilities.
"""

import asyncio
import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI

# Import the new error handling utilities
from utils.llm_error_handler import (
    safe_llm_call, with_llm_error_handling, 
    LLMCancelledError, LLMTimeoutError, LLMRateLimitError
)
from settings import settings

log = logging.getLogger(__name__)

# Example 1: Using the decorator approach
@with_llm_error_handling(timeout=30.0, max_retries=3)
async def example_agent_with_decorator(state: Dict[str, Any]) -> Dict[str, Any]:
    """Example agent using the decorator for error handling."""
    
    # Your existing agent logic here
    prompt = "Your prompt here"
    
    # Create model instance
    model = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=0.3,
        google_api_key=settings.GOOGLE_API_KEY
    )
    
    # The decorator will handle all CancelledError, TimeoutError, etc.
    response = await model.ainvoke(prompt)
    
    return {
        **state,
        "result": response.content,
        "success": True
    }

# Example 2: Using safe_llm_call directly
async def example_agent_with_safe_call(state: Dict[str, Any]) -> Dict[str, Any]:
    """Example agent using safe_llm_call for error handling."""
    
    try:
        prompt = "Your prompt here"
        
        # Create model instance
        model = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0.3,
            google_api_key=settings.GOOGLE_API_KEY
        )
        
        # Use safe_llm_call with comprehensive error handling
        response = await safe_llm_call(
            lambda: model.ainvoke(prompt),
            timeout=30.0,
            max_retries=3,
            agent_name="example_agent"
        )
        
        return {
            **state,
            "result": response.content,
            "success": True
        }
        
    except LLMCancelledError as e:
        log.warning(f"Agent operation was cancelled: {e}")
        return {
            **state,
            "result": "Operation was cancelled",
            "success": False,
            "error": "cancelled"
        }
        
    except LLMTimeoutError as e:
        log.warning(f"Agent operation timed out: {e}")
        return {
            **state,
            "result": "Operation timed out",
            "success": False,
            "error": "timeout"
        }
        
    except LLMRateLimitError as e:
        log.warning(f"Agent hit rate limit: {e}")
        return {
            **state,
            "result": "Rate limit exceeded",
            "success": False,
            "error": "rate_limit"
        }

# Example 3: Manual error handling (for complex scenarios)
async def example_agent_manual_handling(state: Dict[str, Any]) -> Dict[str, Any]:
    """Example agent with manual error handling for complex scenarios."""
    
    prompt = "Your prompt here"
    
    # Create model instance
    model = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=0.3,
        google_api_key=settings.GOOGLE_API_KEY
    )
    
    try:
        # Manual timeout and error handling
        response = await asyncio.wait_for(
            model.ainvoke(prompt),
            timeout=30.0
        )
        
        return {
            **state,
            "result": response.content,
            "success": True
        }
        
    except asyncio.CancelledError:
        log.warning("LLM operation was cancelled")
        # Handle cancellation gracefully
        return {
            **state,
            "result": "Operation was cancelled",
            "success": False,
            "error": "cancelled"
        }
        
    except asyncio.TimeoutError:
        log.warning("LLM operation timed out")
        # Handle timeout gracefully
        return {
            **state,
            "result": "Operation timed out",
            "success": False,
            "error": "timeout"
        }
        
    except Exception as e:
        log.error(f"LLM operation failed: {e}")
        # Handle other errors
        return {
            **state,
            "result": f"Operation failed: {str(e)}",
            "success": False,
            "error": "unknown"
        }

# Example 4: Updating existing agent code
async def update_existing_agent_pattern(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Example showing how to update existing agent code that might be causing CancelledError.
    
    BEFORE (problematic):
    ```python
    try:
        llm_result = await asyncio.wait_for(
            structured_model.ainvoke(prompt),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        # Handle timeout
    except Exception as e:
        # Handle other errors
    ```
    
    AFTER (robust):
    ```python
    try:
        llm_result = await safe_llm_call(
            lambda: structured_model.ainvoke(prompt),
            timeout=30.0,
            max_retries=3,
            agent_name="your_agent_name"
        )
    except LLMCancelledError as e:
        # Handle cancellation
    except LLMTimeoutError as e:
        # Handle timeout
    except LLMRateLimitError as e:
        # Handle rate limiting
    except LLMError as e:
        # Handle other LLM errors
    ```
    """
    
    prompt = "Your prompt here"
    
    # Create model instance
    model = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=0.3,
        google_api_key=settings.GOOGLE_API_KEY
    )
    
    try:
        # Use the robust error handling
        llm_result = await safe_llm_call(
            lambda: model.ainvoke(prompt),
            timeout=30.0,
            max_retries=3,
            agent_name="update_existing_agent_pattern"
        )
        
        return {
            **state,
            "result": llm_result.content,
            "success": True
        }
        
    except LLMCancelledError as e:
        log.warning(f"Operation was cancelled: {e}")
        return {
            **state,
            "result": "Operation was cancelled",
            "success": False,
            "error": "cancelled"
        }
        
    except LLMTimeoutError as e:
        log.warning(f"Operation timed out: {e}")
        return {
            **state,
            "result": "Operation timed out",
            "success": False,
            "error": "timeout"
        }
        
    except LLMRateLimitError as e:
        log.warning(f"Rate limit exceeded: {e}")
        return {
            **state,
            "result": "Rate limit exceeded",
            "success": False,
            "error": "rate_limit"
        }
        
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        return {
            **state,
            "result": f"Unexpected error: {str(e)}",
            "success": False,
            "error": "unknown"
        }

# Example 5: For structured output models
async def structured_output_example(state: Dict[str, Any]) -> Dict[str, Any]:
    """Example for agents using structured output."""
    
    from pydantic import BaseModel, Field
    from typing import List
    
    class StructuredResponse(BaseModel):
        result: str
        confidence: float
        metadata: List[str] = Field(default_factory=list)
    
    prompt = "Your prompt here"
    
    # Create structured model
    model = ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        temperature=0.3,
        google_api_key=settings.GOOGLE_API_KEY
    ).with_structured_output(StructuredResponse)
    
    try:
        # Use safe_llm_call for structured output
        structured_result = await safe_llm_call(
            lambda: model.ainvoke(prompt),
            timeout=30.0,
            max_retries=3,
            agent_name="structured_output_example"
        )
        
        return {
            **state,
            "result": structured_result.result,
            "confidence": structured_result.confidence,
            "metadata": structured_result.metadata,
            "success": True
        }
        
    except LLMCancelledError as e:
        log.warning(f"Structured output operation was cancelled: {e}")
        return {
            **state,
            "result": "Operation was cancelled",
            "confidence": 0.0,
            "metadata": [],
            "success": False,
            "error": "cancelled"
        }
        
    except Exception as e:
        log.error(f"Structured output operation failed: {e}")
        return {
            **state,
            "result": f"Operation failed: {str(e)}",
            "confidence": 0.0,
            "metadata": [],
            "success": False,
            "error": "unknown"
        }


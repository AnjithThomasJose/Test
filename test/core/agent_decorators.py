"""
Decorators for agent functions to provide common functionality.

These decorators handle validation of inputs and outputs, error handling,
and automatic persistence of agent outputs to Chroma DB.
"""

import functools
import logging
import traceback
import time
from typing import Dict, Any, List, Callable, Union, Optional
from datetime import datetime
from core.validation import validate_dependency_availability

log = logging.getLogger(__name__)

def validate_agent_flow(
    required_inputs: List[str] = None, 
    validate_output_func: Callable = None
):
    """
    Decorator for agent functions that validates inputs and outputs.
    
    Args:
        required_inputs: List of keys that must be present in the input state
        validate_output_func: Optional function to validate the output of the agent
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(state: Dict[str, Any], *args, **kwargs):
            start_time = time.time()
            agent_name = func.__name__.replace("_agent", "")
            log_prefix = f"[{agent_name}]"
            
            # Validate inputs
            if required_inputs:
                success, missing = validate_dependency_availability(state, required_inputs)
                if not success:
                    log.error(f"{log_prefix} Missing required inputs: {', '.join(missing)}")
                    return {
                        "status": "error",
                        "message": f"Missing required inputs: {', '.join(missing)}",
                        "processing_time": time.time() - start_time
                    }
            
            # Call the original function
            try:
                result = await func(state, *args, **kwargs)
                
                # Validate output if a validation function is provided
                if validate_output_func and callable(validate_output_func):
                    is_valid, validation_message = validate_output_func(result)
                    if not is_valid:
                        log.error(f"{log_prefix} Output validation failed: {validation_message}")
                        if isinstance(result, dict) and "status" in result:
                            result["status"] = "error"
                            result["validation_error"] = validation_message
                        else:
                            result = {
                                "status": "error",
                                "message": f"Output validation failed: {validation_message}",
                                "processing_time": time.time() - start_time
                            }
                
                return result
                
            except Exception as e:
                log.error(f"{log_prefix} Error in agent execution: {str(e)}")
                log.error(traceback.format_exc())
                return {
                    "status": "error",
                    "message": f"Error in agent execution: {str(e)}",
                    "processing_time": time.time() - start_time
                }
                
        return wrapper
    return decorator

def log_execution_time(func):
    """
    Decorator for logging the execution time of an agent function.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    async def wrapper(state: Dict[str, Any], *args, **kwargs):
        start_time = time.time()
        agent_name = func.__name__.replace("_agent", "")
        
        result = await func(state, *args, **kwargs)
        
        execution_time = time.time() - start_time
        log.info(f"[{agent_name}] Execution time: {execution_time:.2f}s")
        
        if isinstance(result, dict) and "processing_time" not in result:
            result["processing_time"] = execution_time
            
        return result
        
    return wrapper

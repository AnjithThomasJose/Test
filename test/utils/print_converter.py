"""
Print-to-Log Conversion Utility

This module provides utilities to automatically categorize print statements
and convert them to appropriate log levels based on content analysis.
"""

import logging
import re
from typing import Optional


def get_log_level_from_content(message: str) -> int:
    """
    Auto-categorize print statement content to determine appropriate log level.
    
    Rules:
    - ERROR: Contains "error", "exception", "failed", "❌", "failed to", "unable to"
    - WARNING: Contains "warning", "warn", "⚠️", "deprecated", "legacy"
    - CRITICAL: Contains "critical", "fatal", "💥", "system down"
    - DEBUG: Contains "debug", "🔍", "verbose", detailed variable dumps, step-by-step traces
    - INFO: Default for all other cases
    
    Args:
        message: The print statement message content
        
    Returns:
        logging level constant (logging.DEBUG, logging.INFO, etc.)
    """
    if not message:
        return logging.INFO
    
    message_lower = message.lower()
    
    # CRITICAL - highest priority, check first
    critical_patterns = [
        r'\bcritical\b',
        r'\bfatal\b',
        r'💥',
        r'system down',
        r'system failure',
        r'catastrophic',
        r'emergency'
    ]
    for pattern in critical_patterns:
        if re.search(pattern, message_lower):
            return logging.CRITICAL
    
    # ERROR
    error_patterns = [
        r'\berror\b',
        r'\bexception\b',
        r'\bfailed\b',
        r'❌',
        r'failed to',
        r'unable to',
        r'cannot',
        r'could not',
        r'error:',
        r'exception:',
        r'traceback',
        r'stack trace'
    ]
    for pattern in error_patterns:
        if re.search(pattern, message_lower):
            return logging.ERROR
    
    # WARNING
    warning_patterns = [
        r'\bwarning\b',
        r'\bwarn\b',
        r'⚠️',
        r'⚠',
        r'\bdeprecated\b',
        r'\blegacy\b',
        r'\bobsolete\b',
        r'caution',
        r'attention'
    ]
    for pattern in warning_patterns:
        if re.search(pattern, message_lower):
            return logging.WARNING
    
    # DEBUG - verbose, detailed information
    debug_patterns = [
        r'\bdebug\b',
        r'🔍',
        r'\bverbose\b',
        r'step \d+',
        r'processing',
        r'checking',
        r'validating',
        r'verifying',
        r'detailed',
        r'trace:',
        r'variable:',
        r'value:',
        r'type:',
        r'length:',
        r'size:',
        r'count:',
        r'iteration',
        r'loop',
        r'🔧',  # wrench emoji often used for debug
        r'📋',  # clipboard often used for detailed info
    ]
    for pattern in debug_patterns:
        if re.search(pattern, message_lower):
            return logging.DEBUG
    
    # INFO - default for status updates, progress, general information
    return logging.INFO


def get_logger_for_module(module_name: str, file_path: str = "") -> logging.Logger:
    """
    Get appropriate logger for a module based on its location and purpose.
    
    Args:
        module_name: Name of the module (e.g., __name__)
        file_path: Optional file path to help determine logger
        
    Returns:
        Appropriate logger instance
    """
    # Determine logger based on module path
    if 'agents' in module_name.lower() or '/agents/' in file_path or '\\agents\\' in file_path:
        return logging.getLogger('agents')
    elif 'api' in module_name.lower() or 'app.py' in file_path or 'main.py' in file_path:
        return logging.getLogger('api')
    elif 'core' in module_name.lower() or '/core/' in file_path or '\\core\\' in file_path:
        return logging.getLogger('main')
    elif 'utils' in module_name.lower() or '/utils/' in file_path or '\\utils\\' in file_path:
        return logging.getLogger('main')
    elif 'chroma' in module_name.lower() or 'chroma.py' in file_path:
        return logging.getLogger('main')  # ChromaDB operations use main logger
    else:
        return logging.getLogger('main')


def should_keep_as_print(message: str, context: str = "") -> bool:
    """
    Determine if a print statement should remain as print (for critical system failures).
    
    Args:
        message: The print statement message
        context: Additional context (e.g., file name, function name)
        
    Returns:
        True if print should be kept, False if it should be converted to logging
    """
    # Keep prints for logging system failures
    if 'log' in context.lower() and 'handler' in context.lower():
        if any(keyword in message.lower() for keyword in ['error', 'failed', 'exception', 'warning']):
            return True
    
    # Keep prints in settings.py before logging is initialized
    if 'settings.py' in context:
        return True
    
    # Keep prints for bootstrap/startup critical errors
    critical_bootstrap_keywords = [
        'failed to load',
        'missing required',
        'environment variable',
        'settings',
        'bootstrap',
        'initialization failed'
    ]
    if any(keyword in message.lower() for keyword in critical_bootstrap_keywords):
        if 'settings' in context.lower() or 'bootstrap' in context.lower():
            return True
    
    return False


def convert_print_to_log(print_statement: str, module_name: str = "", file_path: str = "") -> tuple:
    """
    Convert a print statement to appropriate logging call.
    
    Args:
        print_statement: The print statement code (e.g., 'print("Hello")')
        module_name: Module name for logger selection
        file_path: File path for logger selection
        
    Returns:
        Tuple of (logger_name, log_level, converted_code, should_keep_as_print)
    """
    # Extract message from print statement
    # Simple regex to extract string content from print()
    message_match = re.search(r'print\s*\(\s*["\']([^"\']*)["\']', print_statement)
    if not message_match:
        # Try f-string or complex expressions
        message_match = re.search(r'print\s*\([^)]+\)', print_statement)
        if message_match:
            # For complex expressions, default to INFO
            logger = get_logger_for_module(module_name, file_path)
            return (logger.name, logging.INFO, None, False)
    
    if message_match:
        message = message_match.group(1) if message_match.groups() else print_statement
    else:
        message = print_statement
    
    # Check if should keep as print
    if should_keep_as_print(message, file_path):
        return (None, None, None, True)
    
    # Get appropriate logger and log level
    logger = get_logger_for_module(module_name, file_path)
    log_level = get_log_level_from_content(message)
    
    # Convert print statement to logging call
    # This is a helper - actual conversion will be done manually with context
    log_method = logging.getLevelName(log_level).lower()
    
    return (logger.name, log_level, log_method, False)

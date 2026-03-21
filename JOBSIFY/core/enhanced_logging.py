"""
Enhanced Logging Utilities for PII Redaction and Log Management

This module provides centralized logging utilities to:
- Redact PII from log messages
- Manage log levels appropriately
- Remove duplicate logging patterns
- Optimize log volume
"""

import re
import logging
from typing import Any, Dict, Optional
from functools import wraps

# PII patterns for redaction
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
    'phone_india': r'\b\+91[-.\s]?[0-9]{5}[-.\s]?[0-9]{5}\b',
    'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Simple name pattern
    'linkedin': r'linkedin\.com/in/[a-zA-Z0-9\-]+',
    'github': r'github\.com/[a-zA-Z0-9\-]+',
    'url': r'https?://[^\s]+',
}

# Log level configuration
# Note: This is kept consistent with the main logging policy in `log_handler.py`:
# - development/dev: DEBUG (DEBUG + INFO visible)
# - qa/production: WARNING (INFO suppressed)
LOG_LEVELS = {
    'production': logging.WARNING,
    'development': logging.DEBUG,
    'testing': logging.WARNING,
    'debug': logging.DEBUG,
}

class PIIRedactor:
    """PII redaction utility for log messages."""
    
    def __init__(self, redact_pii: bool = True):
        self.redact_pii = redact_pii
        self.compiled_patterns = {}
        
        if redact_pii:
            for name, pattern in PII_PATTERNS.items():
                self.compiled_patterns[name] = re.compile(pattern, re.IGNORECASE)
    
    def redact_text(self, text: str) -> str:
        """Redact PII from text."""
        if not self.redact_pii or not text:
            return text
        
        redacted_text = text
        
        # Redact different types of PII
        redacted_text = self.compiled_patterns['email'].sub('[EMAIL]', redacted_text)
        redacted_text = self.compiled_patterns['phone'].sub('[PHONE]', redacted_text)
        redacted_text = self.compiled_patterns['phone_india'].sub('[PHONE]', redacted_text)
        redacted_text = self.compiled_patterns['ssn'].sub('[SSN]', redacted_text)
        redacted_text = self.compiled_patterns['credit_card'].sub('[CARD]', redacted_text)
        redacted_text = self.compiled_patterns['linkedin'].sub('[LINKEDIN]', redacted_text)
        redacted_text = self.compiled_patterns['github'].sub('[GITHUB]', redacted_text)
        redacted_text = self.compiled_patterns['url'].sub('[URL]', redacted_text)
        
        # Redact names (more conservative approach)
        redacted_text = self.compiled_patterns['name'].sub('[NAME]', redacted_text)
        
        return redacted_text
    
    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact PII from dictionary data."""
        if not self.redact_pii:
            return data
        
        redacted_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                redacted_data[key] = self.redact_text(value)
            elif isinstance(value, dict):
                redacted_data[key] = self.redact_dict(value)
            elif isinstance(value, list):
                redacted_data[key] = [self.redact_text(str(item)) if isinstance(item, str) else item for item in value]
            else:
                redacted_data[key] = value
        
        return redacted_data

class LogManager:
    """Centralized log management utility."""
    
    def __init__(self, environment: str = 'development', redact_pii: bool = True):
        self.environment = environment
        self.redactor = PIIRedactor(redact_pii)
        self.log_level = LOG_LEVELS.get(environment, logging.INFO)
        self.max_log_length = 1000  # Maximum log message length
        
        # Track logged content to prevent duplicates
        self.logged_content = set()
    
    def should_log(self, message: str, level: int) -> bool:
        """Determine if message should be logged based on level and content."""
        # Check log level
        if level < self.log_level:
            return False
        
        # Check for duplicate content (simple hash-based deduplication)
        message_hash = hash(message[:100])  # Use first 100 chars for deduplication
        if message_hash in self.logged_content:
            return False
        
        self.logged_content.add(message_hash)
        return True
    
    def truncate_message(self, message: str) -> str:
        """Truncate message if too long."""
        if len(message) <= self.max_log_length:
            return message
        
        return message[:self.max_log_length] + "... [TRUNCATED]"
    
    def prepare_log_message(self, message: str, level: int = logging.INFO) -> Optional[str]:
        """Prepare log message with redaction and truncation."""
        if not self.should_log(message, level):
            return None
        
        # Redact PII
        redacted_message = self.redactor.redact_text(message)
        
        # Truncate if necessary
        final_message = self.truncate_message(redacted_message)
        
        return final_message

# Global log manager instance
log_manager = LogManager()

def log_with_redaction(logger: logging.Logger, level: int, message: str, *args, **kwargs):
    """Log message with PII redaction and level checking."""
    prepared_message = log_manager.prepare_log_message(message, level)
    if prepared_message:
        logger.log(level, prepared_message, *args, **kwargs)

def log_info_safe(logger: logging.Logger, message: str, *args, **kwargs):
    """Safe info logging with PII redaction."""
    log_with_redaction(logger, logging.INFO, message, *args, **kwargs)

def log_debug_safe(logger: logging.Logger, message: str, *args, **kwargs):
    """Safe debug logging with PII redaction."""
    log_with_redaction(logger, logging.DEBUG, message, *args, **kwargs)

def log_warning_safe(logger: logging.Logger, message: str, *args, **kwargs):
    """Safe warning logging with PII redaction."""
    log_with_redaction(logger, logging.WARNING, message, *args, **kwargs)

def log_error_safe(logger: logging.Logger, message: str, *args, **kwargs):
    """Safe error logging with PII redaction."""
    log_with_redaction(logger, logging.ERROR, message, *args, **kwargs)

def redact_data(data: Any) -> Any:
    """Redact PII from any data structure."""
    if isinstance(data, str):
        return log_manager.redactor.redact_text(data)
    elif isinstance(data, dict):
        return log_manager.redactor.redact_dict(data)
    elif isinstance(data, list):
        return [redact_data(item) for item in data]
    else:
        return data

# Decorator for automatic PII redaction in function logs
def redact_logs(func):
    """Decorator to automatically redact PII from function logs."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get logger for the function
        logger = logging.getLogger(func.__module__)
        
        # Log function entry (debug level)
        log_debug_safe(logger, f"Entering {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            log_debug_safe(logger, f"Exiting {func.__name__} successfully")
            return result
        except Exception as e:
            log_error_safe(logger, f"Error in {func.__name__}: {str(e)}")
            raise
    
    return wrapper

# Configuration functions
def configure_logging(environment: str = 'development', redact_pii: bool = True):
    """Configure global logging settings."""
    global log_manager
    log_manager = LogManager(environment, redact_pii)
    
    # Set root logger level
    logging.getLogger().setLevel(log_manager.log_level)
    
    # Configure formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_manager.log_level)
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    
    return log_manager

# Utility functions for common logging patterns
def log_agent_start(logger: logging.Logger, agent_name: str, tenant_id: str = "default"):
    """Log agent start with safe information."""
    log_info_safe(logger, f"🤖 Starting {agent_name} agent for tenant: {tenant_id}")

def log_agent_completion(logger: logging.Logger, agent_name: str, success: bool = True, 
                        processing_time: float = 0.0, details: str = ""):
    """Log agent completion with safe information."""
    status = "✅ completed" if success else "❌ failed"
    message = f"{status} {agent_name} agent"
    if processing_time > 0:
        message += f" in {processing_time:.2f}s"
    if details:
        message += f" - {details}"
    log_info_safe(logger, message)

def log_data_processing(logger: logging.Logger, data_type: str, data_size: int, 
                       redacted: bool = True):
    """Log data processing with size information."""
    if redacted:
        log_info_safe(logger, f"📊 Processing {data_type} (size: {data_size} chars)")
    else:
        log_info_safe(logger, f"📊 Processing {data_type} (size: {data_size} chars)")

def log_performance_metrics(logger: logging.Logger, metrics: Dict[str, Any]):
    """Log performance metrics with PII redaction."""
    redacted_metrics = redact_data(metrics)
    log_info_safe(logger, f"📈 Performance metrics: {redacted_metrics}")

# Initialize logging configuration
configure_logging()

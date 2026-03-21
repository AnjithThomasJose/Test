import logging
import os
import json
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from typing import Dict, Any, Optional, Union
from contextvars import ContextVar
import traceback
import threading
from functools import wraps
import queue

# Module-level logger for this logging setup module
log = logging.getLogger(__name__)

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar('correlation_id', default='')

# Global queue listener for async-safe logging
_log_queue_listener: Optional[QueueListener] = None
_log_queue: Optional[queue.Queue] = None

# Logging metrics
_logging_metrics = {
    "queue_size": 0,
    "total_logs": 0,
    "handler_errors": 0,
    "dropped_logs": 0
}


class SafeQueueHandler(QueueHandler):
    """
    QueueHandler with error handling and queue-full protection.
    Drops oldest logs if queue is full (configurable behavior).
    """
    def __init__(self, queue_instance, drop_on_full=True):
        super().__init__(queue_instance)
        self.drop_on_full = drop_on_full
    
    def enqueue(self, record):
        """Enqueue record with queue-full handling."""
        try:
            # Try to put record (non-blocking)
            self.queue.put_nowait(record)
        except queue.Full:
            # Queue is full - handle based on policy
            _logging_metrics["dropped_logs"] += 1
            
            if self.drop_on_full:
                # Drop oldest log and add new one
                try:
                    # Remove oldest (non-blocking)
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                    # Try again
                    try:
                        self.queue.put_nowait(record)
                    except queue.Full:
                        # Still full - drop this log
                        pass
                except Exception:
                    # Last resort - drop this log
                    pass
            # If drop_on_full=False, log is silently dropped (queue full)
            
            # Warn if queue is consistently full
            if _logging_metrics["dropped_logs"] % 100 == 0:
                import sys
                print(f"⚠️ Log queue full - dropped {_logging_metrics['dropped_logs']} logs", file=sys.stderr)

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread_id": threading.get_ident(),
            "process_id": os.getpid()
        }
        
        # Add correlation ID if available
        corr_id = correlation_id.get()
        if corr_id:
            log_entry["correlation_id"] = corr_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)

class PerformanceFormatter(logging.Formatter):
    """Formatter for performance metrics logging."""
    
    def format(self, record):
        perf_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "performance",
            "operation": getattr(record, 'operation', 'unknown'),
            "duration_ms": getattr(record, 'duration_ms', 0),
            "status": getattr(record, 'status', 'unknown'),
            "metadata": getattr(record, 'metadata', {}),
            "correlation_id": correlation_id.get() or ''
        }
        
        return json.dumps(perf_entry, ensure_ascii=False, default=str)

class SecurityFormatter(logging.Formatter):
    """Formatter for security events logging."""
    
    def format(self, record):
        security_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "security",
            "event": getattr(record, 'event', 'unknown'),
            "severity": record.levelname,
            "source_ip": getattr(record, 'source_ip', 'unknown'),
            "user_id": getattr(record, 'user_id', 'unknown'),
            "action": getattr(record, 'action', 'unknown'),
            "result": getattr(record, 'result', 'unknown'),
            "metadata": getattr(record, 'metadata', {}),
            "correlation_id": correlation_id.get() or ''
        }
        
        return json.dumps(security_entry, ensure_ascii=False, default=str)

class BusinessFormatter(logging.Formatter):
    """Formatter for business logic events logging."""
    
    def format(self, record):
        business_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "business",
            "event": getattr(record, 'event', 'unknown'),
            "level": record.levelname,
            "agent": getattr(record, 'agent', 'unknown'),
            "node": getattr(record, 'node', 'unknown'),
            "uid": getattr(record, 'uid', 'unknown'),
            "tenant_id": getattr(record, 'tenant_id', 'unknown'),
            "session_id": getattr(record, 'session_id', 'unknown'),
            "progress": getattr(record, 'progress', 0.0),
            "status": getattr(record, 'status', 'unknown'),
            "message": record.getMessage(),
            "metadata": getattr(record, 'metadata', {}),
            "correlation_id": correlation_id.get() or ''
        }
        
        return json.dumps(business_entry, ensure_ascii=False, default=str)

def set_correlation_id(corr_id: str = None):
    """Set correlation ID for request tracing."""
    if corr_id is None:
        corr_id = str(uuid.uuid4())
    correlation_id.set(corr_id)
    return corr_id

def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id.get()

def log_performance(operation: str, duration_ms: float, status: str = "success", **metadata):
    """Log performance metrics."""
    perf_logger = logging.getLogger('performance')
    perf_logger.info(
        f"Performance: {operation}",
        extra={
            'operation': operation,
            'duration_ms': duration_ms,
            'status': status,
            'metadata': metadata
        }
    )

def log_security_event(event: str, severity: str = "INFO", source_ip: str = "unknown", 
                      user_id: str = "unknown", action: str = "unknown", 
                      result: str = "unknown", **metadata):
    """Log security events."""
    security_logger = logging.getLogger('security')
    level = getattr(logging, severity.upper(), logging.INFO)
    security_logger.log(
        level,
        f"Security: {event}",
        extra={
            'event': event,
            'source_ip': source_ip,
            'user_id': user_id,
            'action': action,
            'result': result,
            'metadata': metadata
        }
    )

def log_business_event(event: str, agent: str = "unknown", node: str = "unknown",
                      uid: str = "unknown", tenant_id: str = "unknown", 
                      session_id: str = "unknown", progress: float = 0.0,
                      status: str = "unknown", level: str = "INFO", **metadata):
    """Log business logic events."""
    business_logger = logging.getLogger('business')
    log_level = getattr(logging, level.upper(), logging.INFO)
    business_logger.log(
        log_level,
        f"Business: {event}",
        extra={
            'event': event,
            'agent': agent,
            'node': node,
            'uid': uid,
            'tenant_id': tenant_id,
            'session_id': session_id,
            'progress': progress,
            'status': status,
            'metadata': metadata
        }
    )

def performance_timer(operation_name: str = None):
    """Decorator to automatically log performance metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                log_performance(operation, duration_ms, "success")
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_performance(operation, duration_ms, "error", error=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                log_performance(operation, duration_ms, "success")
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_performance(operation, duration_ms, "error", error=str(e))
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

def _get_environment_log_level():
    """
    Determine log level based on APP_ENV environment variable.
    - development/dev: DEBUG (shows DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - qa: WARNING (shows WARNING, ERROR, CRITICAL)
    - production: WARNING (same as QA)
    - default: WARNING
    """
    app_env = os.getenv("APP_ENV", "development").lower()
    if app_env == "dev":
        app_env = "development"
    
    if app_env in ["development", "dev"]:
        return logging.DEBUG
    elif app_env in ["qa", "production", "prod"]:
        return logging.WARNING
    else:
        return logging.WARNING

def setup_loggers():
    """
    Sets up comprehensive structured logging with multiple loggers for different purposes.
    Uses QueueHandler + QueueListener pattern for async-safe, non-blocking logging.
    
    Architecture:
    - Main threads/FastAPI workers → push log records into in-memory queue (non-blocking)
    - Dedicated background thread → pulls from queue and writes to file/console (blocking I/O isolated)
    
    This isolates all slow disk I/O operations away from the async event loop.
    
    Returns main logger and error logger for backward compatibility.
    """
    global _log_queue_listener
    
    try:
        # Check if loggers are already configured to prevent duplicate setup
        main_logger = logging.getLogger('main')
        if main_logger.handlers:
            import sys
            print("✅ Loggers already configured, returning existing loggers", file=sys.stderr)
            return main_logger, logging.getLogger('error')
        
        log_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        # Get environment-based log level
        env_log_level = _get_environment_log_level()
        app_env = os.getenv("APP_ENV", "development").lower()
        if app_env == "dev":
            app_env = "development"
        # When forcing console (e.g. VM dev), use at least INFO so callback/request logs appear in VM logs
        force_console = os.getenv('FORCE_CONSOLE_LOGGING', '').lower() == 'true'
        effective_log_level = max(env_log_level, logging.INFO) if force_console else env_log_level

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(effective_log_level)
        
        # Clear existing handlers and ensure all handlers respect the environment level
        for handler in root_logger.handlers[:]:
            handler.setLevel(effective_log_level)  # Update handler level before removing
            root_logger.removeHandler(handler)

        # ========================================================================
        # ASYNC-SAFE LOGGING PIPELINE: QueueHandler + QueueListener
        # ========================================================================
        # Create a single queue for all loggers (thread-safe, lock-free)
        # Configurable max size (default: unlimited) to prevent memory issues
        max_queue_size = int(os.getenv('LOG_QUEUE_MAX_SIZE', '-1'))
        log_queue = queue.Queue(max_queue_size)
        global _log_queue
        _log_queue = log_queue
        
        # Create actual file handlers (these will run in background thread)
        # These handlers do the actual blocking I/O work
        
        # Helper function to create error-handled handler
        def create_safe_handler(handler, logger_name):
            """Wrap handler with error handling and logger name filter."""
            class ErrorHandlingHandler(logging.Handler):
                def __init__(self, wrapped_handler, logger_name):
                    super().__init__()
                    self.wrapped_handler = wrapped_handler
                    self.logger_name = logger_name
                
                def emit(self, record):
                    # Filter by logger name
                    if record.name != self.logger_name:
                        return
                    
                    try:
                        self.wrapped_handler.emit(record)
                        _logging_metrics["total_logs"] += 1
                    except Exception as e:
                        # Log to stderr (bypass queue to avoid recursion)
                        _logging_metrics["handler_errors"] += 1
                        try:
                            import sys
                            print(f"⚠️ Logging error in {self.logger_name} handler: {e}", file=sys.stderr)
                        except:
                            pass  # Last resort - can't log logging errors
            
            return ErrorHandlingHandler(handler, logger_name)
        
        # 1. Main Application Logger
        main_file_handler_raw = RotatingFileHandler(
            os.path.join(log_directory, 'app.log'), 
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        main_file_handler_raw.setFormatter(StructuredFormatter())
        main_file_handler = create_safe_handler(main_file_handler_raw, 'main')
        
        # 2. Error Logger
        error_file_handler_raw = RotatingFileHandler(
            os.path.join(log_directory, 'error.log'), 
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        error_file_handler_raw.setFormatter(StructuredFormatter())
        error_file_handler = create_safe_handler(error_file_handler_raw, 'error')
        
        # 3. Performance Logger
        perf_file_handler_raw = RotatingFileHandler(
            os.path.join(log_directory, 'performance.log'), 
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        perf_file_handler_raw.setFormatter(PerformanceFormatter())
        perf_file_handler = create_safe_handler(perf_file_handler_raw, 'performance')
        
        # 4. Security Logger
        security_file_handler_raw = RotatingFileHandler(
            os.path.join(log_directory, 'security.log'), 
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        security_file_handler_raw.setFormatter(SecurityFormatter())
        security_file_handler = create_safe_handler(security_file_handler_raw, 'security')
        
        # 5. Business Logic Logger
        business_file_handler_raw = RotatingFileHandler(
            os.path.join(log_directory, 'business.log'), 
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        business_file_handler_raw.setFormatter(BusinessFormatter())
        business_file_handler = create_safe_handler(business_file_handler_raw, 'business')
        
        # 6. Agent Logger
        agent_file_handler_raw = RotatingFileHandler(
            os.path.join(log_directory, 'agents.log'), 
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        agent_file_handler_raw.setFormatter(StructuredFormatter())
        agent_file_handler = create_safe_handler(agent_file_handler_raw, 'agents')
        
        # 7. API Logger
        api_file_handler_raw = RotatingFileHandler(
            os.path.join(log_directory, 'api.log'), 
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        api_file_handler_raw.setFormatter(StructuredFormatter())
        api_file_handler = create_safe_handler(api_file_handler_raw, 'api')
        
        # Console Handler (if enabled)
        # In dev: always show logs to stdout so VM/container logs show callback, request, etc.
        # FORCE_CONSOLE_LOGGING=true: always enable console + INFO (e.g. on VM where APP_ENV might be qa)
        # ENABLE_CONSOLE_LOGGING: normal toggle (default true)
        console_handlers = []
        enable_console_env = os.getenv('ENABLE_CONSOLE_LOGGING', 'true').lower() == 'true'
        enable_console = force_console or (app_env == "development") or enable_console_env
        if enable_console:
            console_handler_raw = logging.StreamHandler()
            console_handler_raw.setLevel(effective_log_level)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler_raw.setFormatter(console_formatter)
            # Console handler doesn't need error handling (stderr is always available)
            console_handlers.append(console_handler_raw)
        
        # Collect all file handlers for the queue listener
        # The QueueListener will route records to appropriate handlers based on logger name
        all_file_handlers = [
            main_file_handler,
            error_file_handler,
            perf_file_handler,
            security_file_handler,
            business_file_handler,
            agent_file_handler,
            api_file_handler,
        ] + console_handlers
        
        # Create QueueListener - this runs in a dedicated background thread
        # It pulls records from the queue and dispatches to appropriate handlers
        # Custom error handling wrapper for the listener itself
        class SafeQueueListener(QueueListener):
            def _monitor(self):
                """Override to add queue size monitoring while preserving parent behavior."""
                # Use getattr to safely access _stop_event (it's created in start() method)
                stop_event = getattr(self, '_stop_event', None)
                if stop_event is None:
                    # If _stop_event doesn't exist yet, wait a bit and retry
                    import time
                    time.sleep(0.1)
                    stop_event = getattr(self, '_stop_event', None)
                    if stop_event is None:
                        # Still doesn't exist - fall back to parent implementation
                        return super()._monitor()
                
                # Replicate parent's _monitor() logic with added monitoring
                while not stop_event.is_set():
                    try:
                        # Update queue size metric periodically
                        if hasattr(self.queue, 'qsize'):
                            _logging_metrics["queue_size"] = self.queue.qsize()
                            
                            # Check for queue backup warning
                            queue_size = _logging_metrics["queue_size"]
                            if queue_size > 1000:
                                import sys
                                print(f"⚠️ Log queue backing up: {queue_size} records pending", file=sys.stderr)
                        
                        # Parent's logic: get record from queue and handle it
                        # Use getattr for dequeue timeout (defaults to 1.0 seconds like parent)
                        dequeue_timeout = getattr(self, 'dequeue', 1.0)
                        try:
                            record = self.queue.get(True, dequeue_timeout)
                            self.handle(record)
                        except queue.Empty:
                            # Queue is empty, continue monitoring
                            pass
                    except Exception as e:
                        # Last resort error handling
                        try:
                            import sys
                            print(f"⚠️ QueueListener error: {e}", file=sys.stderr)
                        except:
                            pass
                        # Break on exception to avoid infinite loop
                        break
        
        _log_queue_listener = SafeQueueListener(log_queue, *all_file_handlers, respect_handler_level=True)
        _log_queue_listener.start()  # Start the background thread
        
        # Create QueueHandlers (non-blocking) for each logger
        # These push records into the queue without blocking
        # Use SafeQueueHandler for queue-full protection
        
        drop_on_full = os.getenv('LOG_DROP_ON_FULL', 'true').lower() == 'true'
        
        # 1. Main Application Logger (Structured JSON)
        main_logger = logging.getLogger('main')
        main_logger.setLevel(effective_log_level)  # Use effective level (INFO when FORCE_CONSOLE_LOGGING)
        main_logger.propagate = False
        main_queue_handler = SafeQueueHandler(log_queue, drop_on_full=drop_on_full)
        main_logger.addHandler(main_queue_handler)
        
        # 2. Error Logger (Structured JSON) - Always ERROR level minimum
        error_logger = logging.getLogger('error')
        error_logger.setLevel(logging.ERROR)  # Error logger always at ERROR level
        error_logger.propagate = False
        error_queue_handler = SafeQueueHandler(log_queue, drop_on_full=drop_on_full)
        error_logger.addHandler(error_queue_handler)
        
        # 3. Performance Logger (Structured JSON)
        perf_logger = logging.getLogger('performance')
        perf_logger.setLevel(env_log_level)  # Use environment-based level
        perf_logger.propagate = False
        perf_queue_handler = SafeQueueHandler(log_queue, drop_on_full=drop_on_full)
        perf_logger.addHandler(perf_queue_handler)
        
        # 4. Security Logger (Structured JSON)
        security_logger = logging.getLogger('security')
        security_logger.setLevel(env_log_level)  # Use environment-based level
        security_logger.propagate = False
        security_queue_handler = SafeQueueHandler(log_queue, drop_on_full=drop_on_full)
        security_logger.addHandler(security_queue_handler)
        
        # 5. Business Logic Logger (Structured JSON)
        business_logger = logging.getLogger('business')
        business_logger.setLevel(env_log_level)  # Use environment-based level
        business_logger.propagate = False
        business_queue_handler = SafeQueueHandler(log_queue, drop_on_full=drop_on_full)
        business_logger.addHandler(business_queue_handler)
        
        # 6. Agent Logger (Structured JSON)
        agent_logger = logging.getLogger('agents')
        agent_logger.setLevel(env_log_level)  # Use environment-based level
        agent_logger.propagate = False
        agent_queue_handler = SafeQueueHandler(log_queue, drop_on_full=drop_on_full)
        agent_logger.addHandler(agent_queue_handler)
        
        # 7. API Logger (Structured JSON)
        api_logger = logging.getLogger('api')
        api_logger.setLevel(env_log_level)  # Use environment-based level
        api_logger.propagate = False
        api_queue_handler = SafeQueueHandler(log_queue, drop_on_full=drop_on_full)
        api_logger.addHandler(api_queue_handler)
        
        # Configure third-party loggers to WARNING level (or env_log_level if in dev)
        # This ensures they don't spam INFO/DEBUG logs in QA/production
        third_party_level = logging.WARNING if app_env in ["qa", "production", "prod"] else env_log_level
        
        # Always suppress httpcore DEBUG logs in all environments
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('google').setLevel(third_party_level)
        logging.getLogger('google.genai').setLevel(third_party_level)
        logging.getLogger('google_genai').setLevel(third_party_level)
        logging.getLogger('google.genai.models').setLevel(third_party_level)
        logging.getLogger('chromadb').setLevel(logging.WARNING)
        logging.getLogger('chroma').setLevel(logging.WARNING)
        logging.getLogger('langchain').setLevel(third_party_level)
        logging.getLogger('langgraph').setLevel(third_party_level)
        
        # Configure application loggers to respect environment level
        # These loggers use __name__ and should inherit from root, but ensure they respect env
        app_loggers = ['core.middleware', 'utils.session_manager', 'core.logging_helpers', 
                      'core.error_handler', 'core.utils', 'utils.memory_manager']
        for logger_name in app_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(env_log_level)
            # Ensure handlers also respect the level
            for handler in logger.handlers:
                handler.setLevel(env_log_level)
        
        import sys
        print(f"✅ Async-safe structured logging initialized (QueueHandler + QueueListener)", file=sys.stderr)
        print(f"📁 Log files in: {log_directory}", file=sys.stderr)
        print(f"📊 Available loggers: main, error, performance, security, business, agents, api", file=sys.stderr)
        print(f"🚀 Logging is now non-blocking - all disk I/O runs in background thread", file=sys.stderr)
        print(f"⚙️  Queue max size: {max_queue_size if max_queue_size > 0 else 'unlimited'}", file=sys.stderr)
        print(f"🛡️  Error handling: Enabled (handlers wrapped with error recovery)", file=sys.stderr)
        print(f"🌍 Environment: {app_env} | Log Level: {logging.getLevelName(env_log_level)}", file=sys.stderr)
        
        return main_logger, error_logger
    except Exception as e:
        import sys
        print(f"❌ WARNING: setup_loggers() failed with error: {e}. Using basic fallback loggers.", file=sys.stderr)
        env_log_level = _get_environment_log_level()
        logging.basicConfig(level=env_log_level)
        fallback_log = logging.getLogger('fallback_main')
        fallback_error_log = logging.getLogger('fallback_error')
        return fallback_log, fallback_error_log


def shutdown_loggers():
    """
    Gracefully shutdown the logging system.
    Stops the QueueListener and ensures all queued logs are written.
    Should be called during application shutdown.
    """
    global _log_queue_listener, _log_queue
    
    if _log_queue_listener is not None:
        try:
            # Wait for queue to drain (with timeout)
            import time
            start_time = time.time()
            timeout = 10.0  # 10 second timeout
            
            while _log_queue and _log_queue.qsize() > 0:
                if time.time() - start_time > timeout:
                    log.warning(f"⚠️ Log queue not empty after {timeout}s, forcing shutdown")
                    break
                time.sleep(0.1)
            
            _log_queue_listener.stop()  # Stops the background thread gracefully
            log.info("✅ Logging system shutdown complete")
            log.info(f"📊 Final metrics: {_logging_metrics}")
        except Exception as e:
            log.error(f"⚠️ Error shutting down logging system: {e}")
        finally:
            _log_queue_listener = None
            _log_queue = None


def get_logging_metrics() -> Dict[str, Any]:
    """
    Get current logging system metrics.
    Useful for monitoring and debugging.
    
    Returns:
        Dictionary with metrics:
        - queue_size: Current number of records in queue
        - total_logs: Total logs processed
        - handler_errors: Number of handler errors
        - dropped_logs: Number of dropped logs (if queue full)
    """
    global _log_queue
    metrics = _logging_metrics.copy()
    
    # Update queue size from actual queue
    if _log_queue is not None:
        try:
            metrics["queue_size"] = _log_queue.qsize()
        except:
            pass
    
    return metrics


def get_log_queue_size() -> int:
    """
    Get current log queue size (for monitoring).
    
    Returns:
        Current queue size, or 0 if queue not initialized
    """
    global _log_queue
    if _log_queue is not None:
        try:
            return _log_queue.qsize()
        except:
            return 0
    return 0

# Import asyncio for the decorator
import asyncio

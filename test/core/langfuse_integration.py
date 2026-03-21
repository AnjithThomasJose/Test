"""
Langfuse integration for LLM observability and tracing.
Provides LangGraph callback handlers for Langfuse Cloud.

This module is designed to be completely non-blocking:
- All errors are caught and logged, never raised
- API requests will never fail due to Langfuse issues
- Graceful degradation if Langfuse is unavailable
"""
import logging
from typing import Optional, List, Any
from settings import settings

log = logging.getLogger(__name__)

_langfuse_enabled = False
_langfuse_initialization_error = None
_langfuse_available = False

def _safe_get_langfuse_handler(session_id: Optional[str] = None, user_id: Optional[str] = None):
    """
    Safely create Langfuse callback handler with comprehensive error handling.
    Never raises exceptions - always returns None on error.
    """
    global _langfuse_initialization_error, _langfuse_available
    
    # Check if Langfuse is enabled
    try:
        if not getattr(settings, 'LANGFUSE_ENABLED', False):
            return None
    except Exception as e:
        log.debug(f"Langfuse check failed (non-critical): {e}")
        return None
    
    # Check if required keys are set
    try:
        secret_key = getattr(settings, 'LANGFUSE_SECRET_KEY', None)
        public_key = getattr(settings, 'LANGFUSE_PUBLIC_KEY', None)
        host = getattr(settings, 'LANGFUSE_HOST', 'https://cloud.langfuse.com')
        
        if not secret_key or not public_key:
            return None
    except Exception as e:
        log.debug(f"Langfuse key retrieval failed (non-critical): {e}")
        return None
    
    # Try to import and create handler
    try:
        # Langfuse v3+ uses langchain integration module
        from langfuse.langchain import CallbackHandler
        from langfuse import get_client
        import os
        
        # Set environment variables for Langfuse client initialization
        # The CallbackHandler uses these to initialize the Langfuse client
        # These must be set before creating the handler
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_HOST"] = host
        # Also set BASE_URL (some versions use this instead of HOST)
        os.environ["LANGFUSE_BASE_URL"] = host
        tracing_env = getattr(settings, "LANGFUSE_TRACING_ENVIRONMENT", None) or "default"
        os.environ["LANGFUSE_TRACING_ENVIRONMENT"] = tracing_env

        # Log initialization for debugging
        log.debug(f"Langfuse env vars set: PUBLIC_KEY={public_key[:10]}..., HOST={host}, ENV={tracing_env}")
        
        # Initialize Langfuse client before creating handler
        # This ensures the client is properly set up to receive traces
        try:
            langfuse_client = get_client()
            log.debug("✅ Langfuse client initialized")
        except Exception as client_error:
            log.debug(f"Langfuse client initialization warning (non-critical): {client_error}")
        
        # Create handler - Langfuse v3 uses environment variables
        # The CallbackHandler will automatically use the environment variables we just set
        # For multi-tenant support, we'll use trace_context
        handler = CallbackHandler(
            public_key=public_key,  # Explicitly pass public_key (optional, env var also works)
            secret_key=secret_key,  # Also pass secret_key explicitly for better initialization
            update_trace=True,  # Update trace with chain input/output/metadata
            trace_context={
                "user_id": user_id,  # Multi-tenant support
                "session_id": session_id or "default"
            } if user_id or session_id else None
        )
        
        _langfuse_available = True
        _langfuse_initialization_error = None
        return handler
        
    except ImportError:
        if _langfuse_initialization_error != "import_error":
            log.debug("Langfuse not installed (non-critical). Install with: pip install langfuse")
            _langfuse_initialization_error = "import_error"
        return None
    except Exception as e:
        # Log only once to avoid spam
        if _langfuse_initialization_error != str(e):
            log.debug(f"Langfuse initialization failed (non-critical, will not block requests): {e}")
            _langfuse_initialization_error = str(e)
        _langfuse_available = False
        return None

def get_langfuse_callback_handler(session_id: Optional[str] = None, user_id: Optional[str] = None):
    """
    Get Langfuse callback handler for LangGraph.
    
    This function is completely safe and will never raise exceptions.
    Returns None if Langfuse is disabled, unavailable, or encounters any error.
    
    Args:
        session_id: Optional session identifier (for trace grouping)
        user_id: Optional user/tenant identifier (for multi-tenant support)
    
    Returns:
        Langfuse callback handler or None if disabled/unavailable
    """
    global _langfuse_enabled
    try:
        handler = _safe_get_langfuse_handler(session_id, user_id)
        if handler and not _langfuse_enabled:
            _langfuse_enabled = True
            log.info("✅ Langfuse callback handler available (non-blocking mode)")
        return handler
    except Exception as e:
        # Ultimate safety net - catch absolutely everything
        log.debug(f"Unexpected error in get_langfuse_callback_handler (non-critical): {e}")
        return None

def is_langfuse_enabled() -> bool:
    """
    Check if Langfuse is enabled and configured.
    Safe function that never raises exceptions.
    """
    try:
        return (
            getattr(settings, 'LANGFUSE_ENABLED', False) and
            bool(getattr(settings, 'LANGFUSE_SECRET_KEY', None)) and
            bool(getattr(settings, 'LANGFUSE_PUBLIC_KEY', None))
        )
    except Exception:
        # If settings access fails, assume disabled
        return False

def safe_add_langfuse_callbacks(config: dict, session_id: Optional[str] = None, user_id: Optional[str] = None) -> dict:
    """
    Safely add Langfuse callbacks to graph config.
    Never modifies the original config if Langfuse fails.
    
    Args:
        config: Graph configuration dict
        session_id: Optional session identifier
        user_id: Optional user/tenant identifier
    
    Returns:
        Updated config with Langfuse callbacks, or original config if Langfuse unavailable
    """
    try:
        if not is_langfuse_enabled():
            return config
        
        langfuse_handler = get_langfuse_callback_handler(session_id, user_id)
        if langfuse_handler:
            # Create a copy to avoid modifying original
            updated_config = config.copy()
            existing_callbacks = updated_config.get("callbacks", [])
            if not isinstance(existing_callbacks, list):
                existing_callbacks = []
            updated_config["callbacks"] = existing_callbacks + [langfuse_handler]
            return updated_config
        
        return config
    except Exception as e:
        # If anything fails, return original config unchanged
        log.debug(f"Failed to add Langfuse callbacks (non-critical): {e}")
        return config

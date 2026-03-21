"""Langfuse integration for LLM observability and tracing."""
import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Cache the handler to avoid recreating; Langfuse client is registered once per public_key
_cached_handler: Optional[Any] = None
_cached_public_key: Optional[str] = None
_last_failure_reason: Optional[str] = None


def get_langfuse_config(settings: Any) -> Dict[str, Any]:
    """
    Build config dict with Langfuse callback handler if enabled and configured.

    Langfuse's get_client(public_key) requires a client to be pre-registered.
    We explicitly instantiate Langfuse() first to register it, then create
    the CallbackHandler which uses get_client(public_key).

    Returns:
        {"callbacks": [CallbackHandler]} if Langfuse is enabled, else {}
    """
    global _cached_handler, _cached_public_key, _last_failure_reason
    _last_failure_reason = None

    if not getattr(settings, "LANGFUSE_ENABLED", True):
        _last_failure_reason = "LANGFUSE_ENABLED=False"
        log.warning("Langfuse disabled via LANGFUSE_ENABLED=False; no traces will be sent")
        return {}

    public_key = getattr(settings, "LANGFUSE_PUBLIC_KEY", None)
    secret_key = getattr(settings, "LANGFUSE_SECRET_KEY", None)

    if not public_key or not secret_key:
        _last_failure_reason = "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY missing"
        log.warning(
            "Langfuse keys missing in settings "
            "(LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY); no traces will be sent"
        )
        return {}

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        base_url = (
            getattr(settings, "LANGFUSE_HOST", None)
            or getattr(settings, "LANGFUSE_BASE_URL", None)
            or "https://cloud.langfuse.com"
        )
        environment = getattr(settings, "LANGFUSE_TRACING_ENVIRONMENT", None) or "default"

        # Must instantiate Langfuse first to register the client with LangfuseResourceManager.
        # CallbackHandler's get_client(public_key) looks up this registered instance.
        if _cached_public_key != public_key:
            log.info(
                "Initializing Langfuse client (host=%s, env=%s, public_key_prefix=%s...)",
                base_url,
                environment,
                str(public_key)[:10],
            )
            Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=base_url,
                environment=environment,
            )
            _cached_handler = CallbackHandler(public_key=public_key)
            _cached_public_key = public_key

        if not _cached_handler:
            _last_failure_reason = "handler cache not initialized"
            log.warning("Langfuse handler cache not initialized; no callbacks will be attached")
            return {}

        return {"callbacks": [_cached_handler]}
    except ImportError as e:
        _last_failure_reason = f"langfuse not installed: {e}"
        log.warning("Langfuse not installed in runtime environment; no traces will be sent: %s", e)
        return {}
    except Exception as e:
        _last_failure_reason = f"init failed: {e}"
        log.warning("Langfuse callback init failed; no traces will be sent: %s", e)
        return {}


def merge_langfuse_into_config(
    config: Optional[Dict[str, Any]],
    settings: Any,
    *,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    handler_out: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Merge Langfuse callbacks and session/user metadata into existing graph/config.

    Pass session_id and user_id to group traces in Langfuse Sessions UI.
    Langfuse reads langfuse_session_id and langfuse_user_id from config metadata.

    Pass handler_out=[my_list] to capture the CallbackHandler for trace_id retrieval.
    After graph execution: trace_id = my_list[0].last_trace_id if my_list else None

    Safe to call with None config; returns a new dict.
    """
    config = config.copy() if config else {}
    need_trace_capture = handler_out is not None and isinstance(handler_out, list)

    if need_trace_capture:
        # Create a fresh handler per execution so we can read last_trace_id
        try:
            from langfuse.langchain import CallbackHandler

            public_key = getattr(settings, "LANGFUSE_PUBLIC_KEY", None)
            if public_key and get_langfuse_config(settings):  # Ensures client is registered
                handler = CallbackHandler(public_key=public_key)
                handler_out.append(handler)
                existing = config.get("callbacks", [])
                if not isinstance(existing, list):
                    existing = [existing] if existing else []
                config["callbacks"] = existing + [handler]
            else:
                log.debug("Langfuse disabled or misconfigured; trace-capture handler not created")
        except Exception as e:
            log.debug("Could not create trace-capture handler: %s", e)
    else:
        langfuse_callbacks = get_langfuse_config(settings)
        if langfuse_callbacks:
            existing = config.get("callbacks", [])
            if not isinstance(existing, list):
                existing = [existing] if existing else []
            config["callbacks"] = existing + langfuse_callbacks.get("callbacks", [])

    # Add session/user metadata for Langfuse Sessions (groups related traces)
    if session_id or user_id:
        config.setdefault("metadata", {})
        if isinstance(config["metadata"], dict):
            if session_id and str(session_id).strip():
                config["metadata"]["langfuse_session_id"] = str(session_id)
            if user_id and str(user_id).strip():
                config["metadata"]["langfuse_user_id"] = str(user_id)

    return config


def get_langfuse_failure_reason(settings: Any) -> Optional[str]:
    """
    Diagnose why Langfuse callbacks are not being attached.
    Calls get_langfuse_config to trigger the full init path and capture the actual failure.
    Returns a short reason string, or None if Langfuse should work.
    """
    cfg = get_langfuse_config(settings)
    if cfg and cfg.get("callbacks"):
        return None
    return _last_failure_reason or "get_langfuse_config returned empty"


def get_trace_id_from_handler(handler_out: Optional[list]) -> Optional[str]:
    """Extract trace_id from handler captured via handler_out in merge_langfuse_into_config."""
    if not handler_out or not isinstance(handler_out, list) or len(handler_out) == 0:
        return None
    handler = handler_out[0]
    return getattr(handler, "last_trace_id", None) or None

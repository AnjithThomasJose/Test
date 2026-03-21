"""
llm_utils.py

Production-ready centralized LLM utilities for the interview agent.

Features:
- Markdown/code-fence stripping
- Robust JSON extraction from LLM responses
- Async invoke_llm with configurable retries and exponential backoff
- Safe fallback shapes and consistent return contract
- Lightweight pluggable client interface (pass your LLM client function)
- Structured logging with context
- Simple error classes for callers to react to specific failure modes

User-facing return format (always):
{
    "ok": bool,
    "raw": str,
    "json": dict | list | None,
    "error": str | None,
    "meta": { "model": str, "duration": float, ... }
}

Expected integration:
- Other modules call invoke_llm(...) and inspect .get("ok") and .get("json")
- For testing, pass mock_client to invoke_llm that returns a dict or string

Note: Replace `default_llm_client` with your real LLM invocation wrapper that
talks to OpenAI/Gemini/Anthropic/etc. For unit tests, pass a mock client.
"""

from typing import Any, Callable, Dict, Optional, Union
import asyncio
import json
import logging
import re
import time

log = logging.getLogger(__name__)

# --- Errors -----------------------------------------------------------------
class LLMError(Exception):
    """Base LLM error."""

class TimeoutError(LLMError):
    pass

class RateLimitError(LLMError):
    pass

class InvalidJSONError(LLMError):
    pass

class EmptyResponseError(LLMError):
    pass

# --- Helpers ---------------------------------------------------------------
_RE_JSON_BLOCK = re.compile(r"(\{[\s\S]*\}|\[[\s\S]*\])", re.MULTILINE)
_FENCE_RE = re.compile(r"^(```+)(?:\w+)?\n|```+$", re.MULTILINE)

def strip_markdown_fences(text: str) -> str:
    """
    Remove triple-backtick fences and common leading/trailing whitespace.
    Keeps inline code intact.
    """
    if not isinstance(text, str):
        return text
    # Remove leading/trailing backtick fences and any language hints
    # Replace windows newlines
    s = text.replace("\r\n", "\n")
    # Remove starting fence
    s = re.sub(r"^\s*```(?:[\w+-]*)\s*\n", "", s)
    # Remove ending fence
    s = re.sub(r"\n?\s*```\s*$", "", s)
    return s.strip()

def extract_json_from_text(text: str) -> Optional[Union[Dict[str, Any], list]]:
    """
    Try to extract a JSON object or array from a noisy LLM text response.
    Returns parsed JSON or None.
    """
    if not text or not isinstance(text, str):
        return None
    cleaned = strip_markdown_fences(text).strip()
    # First, try to load the whole cleaned string
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # If that fails, search for the first JSON-like block
    m = _RE_JSON_BLOCK.search(cleaned)
    if not m:
        return None
    json_text = m.group(1)
    try:
        return json.loads(json_text)
    except Exception:
        # Last-ditch: try to fix common issues (trailing commas)
        fixed = re.sub(r",\s*([\}\]])", r"\1", json_text)
        try:
            return json.loads(fixed)
        except Exception:
            return None

# --- Default mock client ---------------------------------------------------
async def default_llm_client(prompt: Union[str, list], model: Optional[str] = None, timeout: Optional[float] = None, max_tokens: Optional[int] = None, temperature: Optional[float] = None, response_mime_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Gemini-based LLM client for default use in agents.

    This implementation proxies to the core Gemini client logic in models/llm_invoker.py,
    ensuring all Gemini usage routes through our main invocation, observability, and quota system.

    Args:
        prompt: Prompt string or list to send to Gemini.
        model: Gemini model name (optional).
        timeout: Timeout in seconds (optional).
        max_tokens: Maximum tokens to generate (optional).
        temperature: Temperature for generation (optional).
        response_mime_type: MIME type for response (e.g. "application/json").

    Returns:
        dict with at least: {"content": ..., "model": ...}
    """
    # Import invoke_llm directly from the main invocation layer
    from models.llm_invoker import invoke_llm, TaskType

    # Compose prompt as string for Gemini input
    # Handle both string prompts and messages list format
    if isinstance(prompt, list):
        # If it's a list of message dicts, convert to string format
        if prompt and isinstance(prompt[0], dict):
            # Messages format: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            parts = []
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    if role == "system":
                        parts.append(f"System: {content}")
                    elif role == "assistant":
                        parts.append(f"Assistant: {content}")
                    else:
                        parts.append(content)
            prompt_str = "\n\n".join(parts)
        else:
            # List of strings
            prompt_str = "\n\n".join(str(p) for p in prompt)
    else:
        prompt_str = str(prompt)
    
    task_type = TaskType.TEXT_GENERATION if hasattr(TaskType, "TEXT_GENERATION") else "text_generation"

    # NOTE: model argument is advisory; actual model selection is routed via invoke_llm
    # Store max_tokens and temperature for potential use in model instance creation
    # For now, we pass them but the underlying invoke_llm may not support them directly
    # We'll need to handle this in the model instance creation if needed
    content = await invoke_llm(
        prompt_str, 
        task_type=task_type, 
        preferred_model=model,
        response_mime_type=response_mime_type
    )
    
    # Store max_tokens and temperature in return dict for potential downstream use
    result = {
        "content": content,
        "model": model or "gemini"
    }
    if max_tokens is not None:
        result["max_tokens"] = max_tokens
    if temperature is not None:
        result["temperature"] = temperature
    return result

# --- Core invoke logic -----------------------------------------------------
async def _invoke_with_retry(
    client: Callable[..., Any],
    prompt: Union[str, list],
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: int = 2,
    initial_backoff: float = 0.5,
    backoff_factor: float = 2.0,
    retry_on_exceptions: tuple = (TimeoutError,),
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    response_mime_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Internal helper: call client with retries and exponential backoff.
    The client should be an async callable that accepts (prompt, model, timeout, max_tokens, temperature, response_mime_type).
    Returns a dict from the client or raises the last exception.
    """
    attempt = 0
    backoff = initial_backoff
    last_exc = None
    while attempt <= max_retries:
        try:
            start = time.time()
            # Pass max_tokens, temperature, and response_mime_type to client if provided
            client_kwargs = {"prompt": prompt, "model": model, "timeout": timeout}
            if max_tokens is not None:
                client_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                client_kwargs["temperature"] = temperature
            if response_mime_type is not None:
                client_kwargs["response_mime_type"] = response_mime_type
                
            raw_resp = await client(**client_kwargs)
            duration = time.time() - start
            if not isinstance(raw_resp, dict) and not isinstance(raw_resp, str):
                # Coerce to a dict with 'content'
                raw_resp = {"content": str(raw_resp), "model": model}
            # If client returned a string, wrap it
            if isinstance(raw_resp, str):
                raw_resp = {"content": raw_resp, "model": model}
            raw_resp.setdefault("model", model)
            raw_resp.setdefault("duration", duration)
            return raw_resp
        except Exception as exc:
            last_exc = exc
            # Map common error names to our error classes if possible
            # For external clients you may need to inspect exc.__class__.__name__
            name = exc.__class__.__name__.lower()
            if "timeout" in name:
                mapped = TimeoutError(str(exc))
            elif "rate" in name or "ratelimit" in name:
                mapped = RateLimitError(str(exc))
            else:
                mapped = LLMError(str(exc))
            log.warning("LLM client error on attempt %d/%d: %s", attempt+1, max_retries+1, mapped)
            attempt += 1
            if attempt > max_retries:
                raise mapped from exc
            await asyncio.sleep(backoff)
            backoff *= backoff_factor
    # If we exited loop unexpectedly
    raise last_exc or LLMError("Unknown LLM client error")

async def invoke_llm(
    prompt: Union[str, list],
    *,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: int = 2,
    client: Optional[Callable[..., Any]] = None,
    enforce_json: bool = True,
    session_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    response_mime_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Public LLM invocation helper.

    Args:
        prompt: str or list (messages) to send to LLM
        model: model name (optional)
        timeout: seconds
        max_retries: number of retries for transient errors
        client: async callable(client_prompt, model=model, timeout=timeout, max_tokens=max_tokens, temperature=temperature, response_mime_type=response_mime_type) -> dict
        enforce_json: if True, attempt to parse JSON from the LLM response and set 'json' key. Also sets response_mime_type="application/json" if not overridden.
        session_id: optional session id for structured logging
        max_tokens: maximum tokens to generate (optional)
        temperature: temperature for generation (optional)
        response_mime_type: MIME type for response (e.g. "application/json")

    Returns:
        {
            "ok": bool,
            "raw": "<full raw text>",
            "json": dict | list | None,
            "error": None | "<error message>",
            "meta": {"model": str, "duration": float}
        }
    """
    client = client or default_llm_client
    start = time.time()
    
    # Auto-enable Native JSON Mode if enforce_json is True and no mime type specified
    if enforce_json and response_mime_type is None:
        response_mime_type = "application/json"
        
    try:
        raw_resp = await _invoke_with_retry(
            client, prompt, model=model, timeout=timeout, max_retries=max_retries,
            max_tokens=max_tokens, temperature=temperature, response_mime_type=response_mime_type
        )
    except Exception as e:
        duration = time.time() - start
        log.error("LLM invocation failed (session=%s): %s", session_id, str(e), exc_info=False)
        return {"ok": False, "raw": "", "json": None, "error": str(e), "meta": {"model": model, "duration": duration}}

    # Normalize content
    content = ""
    if isinstance(raw_resp, dict):
        # Typical client shape: {"content": "...", "model": "gpt-4o", "duration": 0.2}
        if "content" in raw_resp:
            content = raw_resp["content"]
        elif "message" in raw_resp and isinstance(raw_resp["message"], dict):
            # Some clients return {"message": {"content": "..."}}
            msg = raw_resp["message"]
            content = msg.get("content") or (msg.get("choices") and msg["choices"][0].get("message", {}).get("content", "")) or ""
        elif "choices" in raw_resp and isinstance(raw_resp["choices"], list):
            # OpenAI style
            try:
                # extract the first non-empty text
                for ch in raw_resp["choices"]:
                    if isinstance(ch, dict):
                        if "message" in ch and isinstance(ch["message"], dict) and ch["message"].get("content"):
                            content = ch["message"]["content"]
                            break
                        if "text" in ch and ch.get("text"):
                            content = ch["text"]
                            break
            except Exception:
                content = json.dumps(raw_resp)
        else:
            # Fallback: stringify
            content = str(raw_resp)
    elif isinstance(raw_resp, str):
        content = raw_resp
    content = (content or "").strip()
    duration = raw_resp.get("duration") if isinstance(raw_resp, dict) and raw_resp.get("duration") is not None else (time.time() - start)

    # If empty response
    if not content:
        log.warning("LLM returned empty content (session=%s)", session_id)
        return {"ok": False, "raw": "", "json": None, "error": "empty_response", "meta": {"model": model, "duration": duration}}

    # Optionally enforce JSON parsing
    parsed = None
    if enforce_json:
        parsed = extract_json_from_text(content)
        if parsed is None:
            # Try looser parse: if content looks like single-line key:value pairs, return raw
            log.warning("LLM returned non-JSON content while enforce_json=True (session=%s); attempting best-effort", session_id)
            return {"ok": False, "raw": content, "json": None, "error": "invalid_json", "meta": {"model": model, "duration": duration}}
    else:
        # Best-effort parse but not required
        parsed = extract_json_from_text(content)

    # Success
    return {"ok": True, "raw": content, "json": parsed, "error": None, "meta": {"model": model, "duration": duration}}

# --- Convenience synchronous wrapper for callers that are not async -----------
def invoke_llm_sync(*args, **kwargs) -> Dict[str, Any]:
    """
    Synchronous wrapper around invoke_llm for synchronous code paths.
    This will run the async call on the running loop if available or create a new one.

    WARNING: Do not call from the async event loop thread (e.g. FastAPI request
    handlers). This uses fut.result() and will block the event loop. In async
    code use invoke_llm() and await it instead.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # We are inside an event loop; schedule and wait
        fut = asyncio.run_coroutine_threadsafe(invoke_llm(*args, **kwargs), loop)
        return fut.result()
    else:
        return asyncio.run(invoke_llm(*args, **kwargs))


# --- LLM Connection Warmup --------------------------------------------------
_llm_warmed_up = False

async def warmup_llm_connection() -> None:
    """
    Pre-warm LLM connection on server startup to reduce first-request latency.
    
    This sends a minimal prompt to initialize the LLM client connection,
    avoiding cold-start delays on the first actual user request.
    """
    global _llm_warmed_up
    if _llm_warmed_up:
        return
    
    try:
        log.info("[WARMUP] Pre-warming LLM connection...")
        start = time.time()
        
        # Send a minimal warmup prompt
        warmup_prompt = "Say 'ready' in one word."
        result = await default_llm_client(
            prompt=warmup_prompt,
            max_tokens=10,
            temperature=0.0,
            timeout=30.0
        )
        
        duration = time.time() - start
        # default_llm_client returns {"content": "...", "model": "..."}, not {"ok", "raw", "error"}
        if result.get("content"):
            _llm_warmed_up = True
            log.info(f"[WARMUP] LLM connection warmed up successfully in {duration:.2f}s")
        else:
            log.warning(f"[WARMUP] LLM warmup returned unexpected result: {result}")
    except Exception as e:
        log.warning(f"[WARMUP] LLM warmup failed (non-critical): {e}")


def is_llm_warmed_up() -> bool:
    """Check if LLM connection has been pre-warmed."""
    return _llm_warmed_up


# --- End of file ------------------------------------------------------------

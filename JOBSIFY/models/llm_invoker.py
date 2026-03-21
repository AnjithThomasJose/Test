import json
import os
import asyncio
import time
import logging
import hashlib
import sqlite3
from contextlib import closing
from dataclasses import replace
from typing import Dict, Any, Optional, Union, List, Tuple, Type, TypeVar, cast

from pydantic import BaseModel
from core.prompt_store import get_prompt, PromptType
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Only using Gemini models - no other providers needed
from settings import settings

# Import new infrastructure components
from core.model_registry import model_registry, TaskType, get_model_for_task, record_model_usage
from core.quota_manager import quota_manager, check_quota, record_usage, set_backoff, is_in_backoff
from core.observability import observability_manager, log_prompt_response

# Import enhanced error handling
from utils.llm_error_handler import (
    safe_llm_call, LLMError, LLMTimeoutError, LLMCancelledError, 
    LLMRateLimitError, is_cancellation_error, is_timeout_error, is_rate_limit_error
)


# =============================================================================
# Fallback Response Handling (Issue 5.4)
# =============================================================================
# When all LLM attempts fail, callers need to know they received a fallback.

class LLMFallbackError(LLMError):
    """Raised when LLM fails and caller requested raise_on_fallback=True.
    
    This allows callers to handle failures explicitly rather than receiving
    a generic fallback response that may not match their expected format.
    
    Attributes:
        fallback_response: The fallback response that would have been returned
        original_error: The original error that caused the fallback
    """
    def __init__(self, message: str, fallback_response: str = "", original_error: Optional[str] = None):
        super().__init__(message)
        self.fallback_response = fallback_response
        self.original_error = original_error


# Marker to identify fallback responses (Issue 5.4)
# Callers can check for this prefix to detect fallbacks
FALLBACK_RESPONSE_MARKER = "[LLM_FALLBACK]"


def is_fallback_response(response: str) -> bool:
    """Check if a response is a fallback response.
    
    Issue 5.4: Helper function for callers to easily detect fallback responses
    without needing to know the marker format.
    
    Args:
        response: The LLM response to check
        
    Returns:
        True if the response is a fallback, False otherwise
        
    Example:
        response = await invoke_llm(prompt)
        if is_fallback_response(response):
            # Handle fallback case - maybe retry, use cached data, or show error
            log.warning("Got fallback response, using cached data instead")
    """
    return response.startswith(FALLBACK_RESPONSE_MARKER)


def strip_fallback_marker(response: str) -> str:
    """Remove the fallback marker from a response if present.
    
    Issue 5.4: Helper to clean up fallback responses for display.
    
    Args:
        response: The LLM response (may or may not be a fallback)
        
    Returns:
        The response without the marker prefix
    """
    if response.startswith(FALLBACK_RESPONSE_MARKER):
        return response[len(FALLBACK_RESPONSE_MARKER):].lstrip()
    return response


def _estimate_tokens(text: str) -> int:
    """
    Issue 8.4: Character-based token estimation (more accurate than word-based).
    
    Uses character count with different ratios for text vs code/JSON.
    This is more accurate than word-based estimation for:
    - Code snippets (high token/word ratio)
    - JSON/structured data (high token/word ratio)
    - Non-English text (variable token/word ratio)
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    import re
    
    if not text:
        return 0
    
    # Remove extra whitespace for cleaner estimate
    cleaned = re.sub(r'\s+', ' ', text.strip())
    char_count = len(cleaned)
    
    # Detect if text looks like code or JSON (uses more tokens per character)
    code_indicators = ['{', '[', 'def ', 'function', 'class ', 'import ', 'return ', '};', '});']
    is_code_like = any(indicator in text for indicator in code_indicators)
    
    if is_code_like:
        # Code/JSON: ~3.5 chars per token
        return max(1, int(char_count / 3.5))
    else:
        # Natural language: ~4 chars per token
        return max(1, int(char_count / 4))


log = logging.getLogger(__name__)

try:
    import redis  # type: ignore
except Exception:
    redis = None


class _LLMCache:
    def __init__(self):
        self.backend = settings.LLM_CACHE_BACKEND.lower() if getattr(settings, "LLM_CACHE_BACKEND", None) else "sqlite"
        self.ttl = int(getattr(settings, "LLM_CACHE_TTL_SECONDS", 1800))
        self._memory: Dict[str, Any] = {}
        self._redis = None
        self._sqlite_path = getattr(settings, "SQLITE_CACHE_PATH", "/tmp/kafin_llm_cache.sqlite")
        self._last_cleanup = 0
        self._cleanup_interval = 300  # Cleanup every 5 minutes

        if self.backend == "redis" and getattr(settings, "REDIS_URL", None) and redis:
            try:
                self._redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
                self._redis.ping()
                log.info("✅ Redis cache backend initialized successfully")
            except Exception as e:
                log.warning(f"⚠️ Redis connection failed, falling back to SQLite: {e}")
                self._redis = None
                self.backend = "sqlite"

        if self.backend == "sqlite":
            try:
                with closing(sqlite3.connect(self._sqlite_path)) as conn:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS llm_cache (
                            k TEXT PRIMARY KEY,
                            v TEXT NOT NULL,
                            expires_at INTEGER NOT NULL
                        )
                        """
                    )
                    conn.commit()
            except Exception:
                # Fallback to in-memory if sqlite path is not writable
                self.backend = "memory"

    @staticmethod
    def _now() -> int:
        return int(time.time())

    def _is_expired(self, expires_at: int) -> bool:
        return self._now() >= expires_at

    def _make_key(self, provider: str, model: str, temperature: float, prompt: str, system_instruction: Optional[str] = None) -> str:
        payload = {
            "provider": provider,
            "model": model,
            "temperature": round(float(temperature), 3),
            "prompt_hash": hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest(),
        }
        if system_instruction:
            payload["system_instruction_hash"] = hashlib.sha256(system_instruction.strip().encode("utf-8")).hexdigest()
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _cleanup_expired_entries(self):
        """Proactively clean expired entries from SQLite cache."""
        if self.backend != "sqlite":
            return
        
        current_time = self._now()
        # Only cleanup if enough time has passed since last cleanup
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        try:
            with closing(sqlite3.connect(self._sqlite_path)) as conn:
                deleted = conn.execute(
                    "DELETE FROM llm_cache WHERE expires_at < ?",
                    (current_time,)
                ).rowcount
                conn.commit()
                if deleted > 0:
                    log.debug(f"🧹 Cleaned up {deleted} expired cache entries")
                self._last_cleanup = current_time
        except Exception as e:
            log.debug(f"Cache cleanup failed: {e}")

    def get(self, key: str) -> Optional[str]:
        if not getattr(settings, "USE_LLM_CACHE", True):
            return None

        if self.backend == "redis" and self._redis:
            try:
                return self._redis.get(key)
            except Exception:
                return None

        if self.backend == "sqlite":
            try:
                self._cleanup_expired_entries()
                
                with closing(sqlite3.connect(self._sqlite_path)) as conn:
                    row = conn.execute("SELECT v, expires_at FROM llm_cache WHERE k = ?", (key,)).fetchone()
                    if not row:
                        return None
                    v, expires_at = row
                    if self._is_expired(expires_at):
                        conn.execute("DELETE FROM llm_cache WHERE k = ?", (key,))
                        conn.commit()
                        return None
                    return v
            except Exception:
                return None

        val = self._memory.get(key)
        if not val:
            return None
        v, expires_at = val
        if self._is_expired(expires_at):
            self._memory.pop(key, None)
            return None
        return v

    async def aget(self, key: str) -> Optional[str]:
        """Async wrapper — offloads sync SQLite/Redis I/O to thread pool."""
        if not getattr(settings, "USE_LLM_CACHE", True):
            return None
        if self.backend == "memory":
            return self.get(key)
        return await asyncio.to_thread(self.get, key)

    def set(self, key: str, value: str):
        if not getattr(settings, "USE_LLM_CACHE", True):
            return

        expires_at = self._now() + self.ttl

        if self.backend == "redis" and self._redis:
            try:
                self._redis.setex(key, self.ttl, value)
                return
            except Exception as e:
                log.debug(f"Redis cache set failed: {e}")
                pass

        if self.backend == "sqlite":
            try:
                self._cleanup_expired_entries()
                
                with closing(sqlite3.connect(self._sqlite_path)) as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO llm_cache (k, v, expires_at) VALUES (?, ?, ?)",
                        (key, value, expires_at),
                    )
                    conn.commit()
                return
            except Exception as e:
                log.debug(f"SQLite cache set failed: {e}")
                pass

        self._memory[key] = (value, expires_at)

    async def aset(self, key: str, value: str):
        """Async wrapper — offloads sync SQLite/Redis I/O to thread pool."""
        if not getattr(settings, "USE_LLM_CACHE", True):
            return
        if self.backend == "memory":
            return self.set(key, value)
        await asyncio.to_thread(self.set, key, value)
    
    def clear(self):
        """Clear all cache entries."""
        if self.backend == "redis" and self._redis:
            try:
                self._redis.flushdb()
                log.info("🧹 Redis cache cleared")
            except Exception as e:
                log.warning(f"Failed to clear Redis cache: {e}")
        
        if self.backend == "sqlite":
            try:
                with closing(sqlite3.connect(self._sqlite_path)) as conn:
                    conn.execute("DELETE FROM llm_cache")
                    conn.commit()
                    log.info("🧹 SQLite cache cleared")
            except Exception as e:
                log.warning(f"Failed to clear SQLite cache: {e}")
        
        if self.backend == "memory":
            self._memory.clear()
            log.info("🧹 Memory cache cleared")
    
    def invalidate(self, key: str):
        """Invalidate a specific cache entry."""
        if self.backend == "redis" and self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass
        
        if self.backend == "sqlite":
            try:
                with closing(sqlite3.connect(self._sqlite_path)) as conn:
                    conn.execute("DELETE FROM llm_cache WHERE k = ?", (key,))
                    conn.commit()
            except Exception:
                pass
        
        if self.backend == "memory":
            self._memory.pop(key, None)
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching a pattern (SQL LIKE for SQLite, keys() scan for others)."""
        if self.backend == "redis" and self._redis:
            try:
                # Use SCAN to find matching keys
                cursor = 0
                while True:
                    cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
                    if keys:
                        self._redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception:
                pass
        
        if self.backend == "sqlite":
            try:
                with closing(sqlite3.connect(self._sqlite_path)) as conn:
                    conn.execute("DELETE FROM llm_cache WHERE k LIKE ?", (pattern,))
                    conn.commit()
            except Exception:
                pass
        
        if self.backend == "memory":
            keys_to_remove = [k for k in self._memory.keys() if pattern in k]
            for key in keys_to_remove:
                self._memory.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "backend": self.backend,
            "ttl_seconds": self.ttl,
        }
        
        if self.backend == "redis" and self._redis:
            try:
                stats["size"] = self._redis.dbsize()
            except Exception:
                stats["size"] = 0
        
        if self.backend == "sqlite":
            try:
                with closing(sqlite3.connect(self._sqlite_path)) as conn:
                    stats["size"] = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
                    stats["expired"] = conn.execute(
                        "SELECT COUNT(*) FROM llm_cache WHERE expires_at < ?",
                        (self._now(),)
                    ).fetchone()[0]
            except Exception:
                stats["size"] = 0
                stats["expired"] = 0
        
        if self.backend == "memory":
            stats["size"] = len(self._memory)
            stats["expired"] = sum(
                1 for _, expires_at in self._memory.values()
                if self._is_expired(expires_at)
            )
        
        return stats


_llm_cache = _LLMCache()

# Global semaphore to limit concurrent LLM calls across all agents (prevents rate limits, fair sharing)
_llm_semaphore: Optional[asyncio.Semaphore] = None


def _get_llm_semaphore() -> asyncio.Semaphore:
    """Get or create global LLM semaphore (lazy init). Limits concurrent API calls."""
    global _llm_semaphore
    if _llm_semaphore is None:
        limit = int(os.getenv("LLM_CONCURRENCY_LIMIT", "12"))
        _llm_semaphore = asyncio.Semaphore(max(1, limit))
    return _llm_semaphore


TStruct = TypeVar("TStruct", bound=BaseModel)


def _structured_schema_id(schema: Type[BaseModel]) -> str:
    return f"{schema.__module__}.{schema.__qualname__}"


def _normalize_structured_llm_input(
    llm_input: Union[str, List[BaseMessage]],
) -> Tuple[Any, str, Optional[str]]:
    """
    Build (invoke_input, prompt_for_cache_key, system_instruction_for_cache_key).
    """
    if isinstance(llm_input, str):
        text = llm_input.strip()
        return llm_input, text, None
    system_chunks: List[str] = []
    human_chunks: List[str] = []
    for msg in llm_input:
        if isinstance(msg, SystemMessage):
            system_chunks.append(str(msg.content))
        elif isinstance(msg, HumanMessage):
            human_chunks.append(str(msg.content))
        else:
            human_chunks.append(str(getattr(msg, "content", msg)))
    system_instruction = "\n\n".join(system_chunks) if system_chunks else None
    prompt = "\n\n".join(human_chunks) if human_chunks else ""
    return llm_input, prompt, system_instruction


def _make_structured_llm_cache_key(
    provider: str,
    model: str,
    temperature: float,
    prompt: str,
    system_instruction: Optional[str],
    schema_id: str,
) -> str:
    base = _llm_cache._make_key(provider, model, temperature, prompt, system_instruction)
    return hashlib.sha256(f"{base}|structured|{schema_id}".encode("utf-8")).hexdigest()


def _parse_structured_invoke_result(
    schema: Type[TStruct],
    result: Any,
    include_raw: bool,
) -> Union[TStruct, Tuple[TStruct, Any]]:
    """Normalize LangChain structured output (model instance or include_raw dict)."""
    want_raw = include_raw
    if want_raw and isinstance(result, dict) and "parsed" in result:
        parsed = result.get("parsed")
        raw = result.get("raw")
        if parsed is None:
            raise ValueError("Structured LLM returned include_raw dict with parsed=None")
        if isinstance(parsed, schema):
            return cast(TStruct, parsed), raw
        if isinstance(parsed, dict):
            validated = schema.model_validate(parsed)
            return validated, raw
        raise ValueError(f"Unexpected parsed type: {type(parsed)}")
    if want_raw:
        if isinstance(result, schema):
            return cast(TStruct, result), None
        if isinstance(result, BaseModel):
            validated = schema.model_validate(result.model_dump())
            return validated, None
        if isinstance(result, dict):
            return schema.model_validate(result), None
        raise ValueError(f"Unexpected structured LLM result type: {type(result)}")
    if isinstance(result, schema):
        return cast(TStruct, result)
    if isinstance(result, dict):
        return schema.model_validate(result)
    if isinstance(result, BaseModel):
        return schema.model_validate(result.model_dump())
    raise ValueError(f"Unexpected structured LLM result type: {type(result)}")


async def invoke_structured_llm(
    llm_input: Union[str, List[BaseMessage]],
    schema: Type[TStruct],
    task_type: Union[TaskType, str] = TaskType.TEXT_GENERATION,
    preferred_model: Optional[str] = None,
    max_retries: int = 3,
    agent_name: str = "unknown",
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: Optional[float] = None,
    raise_on_fallback: bool = True,
    skip_cache: bool = False,
    structured_output_kwargs: Optional[Dict[str, Any]] = None,
    include_raw: bool = False,
) -> Union[TStruct, Tuple[TStruct, Any]]:
    """
    Invoke Gemini with LangChain structured output under the same semaphore, quota,
    cache, and safe_llm_call path as invoke_llm.

    Args:
        llm_input: User prompt string or LangChain messages (SystemMessage/HumanMessage).
        schema: Pydantic model class for with_structured_output.
        include_raw: If True, returns (parsed_model, raw_message) for telemetry (e.g. record_direct_llm_usage).

    Returns:
        Validated Pydantic instance, or (instance, raw) when include_raw=True.
    """
    start_time = time.time()
    invoke_payload, cache_prompt, cache_system = _normalize_structured_llm_input(llm_input)
    schema_id = _structured_schema_id(schema)
    final_model: Optional[str] = None

    so_kw = dict(structured_output_kwargs or {})
    if include_raw:
        so_kw["include_raw"] = True

    try:
        model_config = get_model_for_task(task_type, preferred_model)
        if temperature is not None:
            model_config = replace(model_config, temperature=temperature)
        if timeout is not None:
            model_config = replace(model_config, timeout=max(model_config.timeout, int(timeout) + 5))
        final_model = model_config.model_name

        prompt_text = cache_prompt
        if cache_system:
            prompt_text = cache_system + "\n\n" + cache_prompt
        estimated_tokens = _estimate_tokens(prompt_text)
        estimated_cost = (
            model_config.cost_per_input_token * estimated_tokens
            + model_config.cost_per_output_token * estimated_tokens
        )

        if not getattr(settings, "DISABLE_QUOTA_CHECK", False):
            quota_status = check_quota(
                model_config.provider.value, estimated_cost, int(estimated_tokens)
            )
            if quota_status.value == "exceeded":
                log.warning(f"Quota exceeded for {model_config.provider.value}, trying fallback")
                fallback_model = model_registry.get_fallback_model(final_model)
                if fallback_model:
                    model_config = fallback_model
                    final_model = model_config.model_name
                    if temperature is not None:
                        model_config = replace(model_config, temperature=temperature)
                    if timeout is not None:
                        model_config = replace(
                            model_config,
                            timeout=max(model_config.timeout, int(timeout) + 5),
                        )
                else:
                    raise RuntimeError(
                        f"Quota exceeded for {model_config.provider.value} and no fallback available"
                    )
            if is_in_backoff(model_config.provider.value):
                log.warning(f"Provider {model_config.provider.value} is in backoff, trying fallback")
                fallback_model = model_registry.get_fallback_model(final_model)
                if fallback_model:
                    model_config = fallback_model
                    final_model = model_config.model_name
                    if temperature is not None:
                        model_config = replace(model_config, temperature=temperature)
                    if timeout is not None:
                        model_config = replace(
                            model_config,
                            timeout=max(model_config.timeout, int(timeout) + 5),
                        )
                else:
                    raise RuntimeError(
                        f"Provider {model_config.provider.value} is in backoff and no fallback available"
                    )

        cache_key = (
            _make_structured_llm_cache_key(
                model_config.provider.value,
                model_config.model_name,
                getattr(model_config, "temperature", 0.0),
                cache_prompt,
                cache_system,
                schema_id,
            )
            if not skip_cache
            else None
        )
        if not skip_cache and cache_key:
            try:
                cached = _llm_cache.get(cache_key)
                if cached:
                    latency_ms = (time.time() - start_time) * 1000
                    record_model_usage(model_config.model_name, 0, latency_ms, success=True)
                    log_prompt_response(
                        prompt=cache_prompt[:2000],
                        response=cached[:2000],
                        token_count=0,
                        cost=0.0,
                        latency_ms=latency_ms,
                        agent_name=agent_name,
                        success=True,
                        metadata={
                            "internal_cache_hit": True,
                            "cache_source": "internal_llm_cache_structured",
                            "schema_id": schema_id,
                        },
                    )
                    validated = schema.model_validate_json(cached)
                    if include_raw:
                        return validated, None
                    return validated
            except Exception:
                log.warning("Structured LLM cache read failed; proceeding without cache")

        async with _get_llm_semaphore():
            try:
                model_instance = await _create_model_instance(
                    model_config, response_mime_type=None, max_output_tokens=max_output_tokens
                )
                structured_runnable = model_instance.with_structured_output(schema, **so_kw)
                effective_timeout = (
                    float(timeout) if timeout is not None else float(model_config.timeout)
                )
                response = await safe_llm_call(
                    lambda: structured_runnable.ainvoke(invoke_payload),
                    timeout=effective_timeout,
                    max_retries=max_retries,
                    agent_name=f"{agent_name}_{model_config.display_name}_structured",
                )
                parsed_result = _parse_structured_invoke_result(
                    schema, response, include_raw=bool(include_raw or so_kw.get("include_raw"))
                )
            except LLMCancelledError as e:
                log.error(f"Structured LLM cancelled: {e}")
                raise
            except LLMTimeoutError as e:
                log.error(f"Structured LLM timeout: {e}")
                raise
            except LLMRateLimitError as e:
                backoff_seconds = e.retry_after if e.retry_after else 60
                backoff_seconds = max(5, min(backoff_seconds, 300))
                set_backoff(model_config.provider.value, backoff_seconds)
                raise

            if include_raw or so_kw.get("include_raw"):
                pair = cast(Tuple[TStruct, Any], parsed_result)
                validated, raw_msg = pair
                usage_meta = _extract_usage_metadata(raw_msg) if raw_msg is not None else {}
            else:
                validated = cast(TStruct, parsed_result)
                raw_msg = None
                usage_meta = {}

            content_repr = validated.model_dump_json()[:8000]
            if usage_meta.get("total_token_count"):
                actual_tokens = usage_meta["total_token_count"]
                prompt_tokens = usage_meta.get("prompt_token_count", 0)
                output_tokens = usage_meta.get("candidates_token_count", 0)
            else:
                actual_tokens = _estimate_tokens(prompt_text) + _estimate_tokens(content_repr)
                prompt_tokens = _estimate_tokens(prompt_text)
                output_tokens = _estimate_tokens(content_repr)

            latency_ms = (time.time() - start_time) * 1000
            record_usage(model_config.provider.value, estimated_cost, actual_tokens, success=True)
            record_model_usage(final_model or model_config.model_name, actual_tokens, latency_ms, success=True)
            log_prompt_response(
                prompt=cache_prompt[:2000],
                response=content_repr[:2000],
                token_count=actual_tokens,
                cost=estimated_cost,
                latency_ms=latency_ms,
                agent_name=agent_name,
                success=True,
                metadata={
                    "structured_output": True,
                    "schema_id": schema_id,
                    "prompt_token_count": prompt_tokens,
                    "output_token_count": output_tokens,
                    **usage_meta,
                },
            )

            if not skip_cache and cache_key:
                try:
                    _llm_cache.set(cache_key, validated.model_dump_json())
                except Exception:
                    log.warning("Structured LLM cache set failed; continuing")

            if include_raw or so_kw.get("include_raw"):
                return validated, raw_msg
            return validated

    except (LLMCancelledError, LLMTimeoutError, LLMRateLimitError, LLMFallbackError):
        raise
    except Exception as e:
        log.error(f"Structured LLM invocation failed: {e}")
        err_name = str(type(e).__name__)
        if final_model:
            latency_ms = (time.time() - start_time) * 1000
            record_model_usage(final_model, 0, latency_ms, success=False, error_type=err_name)
            log_prompt_response(
                prompt=cache_prompt[:2000],
                response="",
                token_count=0,
                cost=0.0,
                latency_ms=latency_ms,
                agent_name=agent_name,
                success=False,
                error_type=err_name,
            )
        fb_name = final_model or getattr(settings, "GEMINI_MODEL", None)
        fallback_model = model_registry.get_fallback_model(fb_name) if fb_name else None
        if fallback_model and final_model and fallback_model.model_name != final_model:
            return await invoke_structured_llm(
                llm_input,
                schema,
                task_type=task_type,
                preferred_model=fallback_model.model_name,
                max_retries=1,
                agent_name=agent_name,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                timeout=timeout,
                raise_on_fallback=raise_on_fallback,
                skip_cache=skip_cache,
                structured_output_kwargs=structured_output_kwargs,
                include_raw=include_raw,
            )
        if raise_on_fallback:
            raise LLMFallbackError(
                message=f"Structured LLM failed: {e}",
                fallback_response="",
                original_error=str(e),
            ) from e
        raise


async def invoke_llm(prompt: str, task_type: Union[TaskType, str] = TaskType.TEXT_GENERATION, 
                    preferred_model: Optional[str] = None, max_retries: int = 3, 
                    agent_name: str = "unknown", response_mime_type: Optional[str] = None,
                    max_output_tokens: Optional[int] = None, system_instruction: Optional[str] = None,
                    raise_on_fallback: bool = False, skip_cache: bool = False) -> str:
    """
    Enhanced LLM invoker with model registry, quota management, and observability.
    
    Args:
        prompt: The input prompt (dynamic content)
        task_type: Type of task (for model routing)
        preferred_model: Preferred model name (optional)
        max_retries: Maximum number of retry attempts
        agent_name: Name of the calling agent (for observability)
        response_mime_type: MIME type for response (e.g. "application/json")
        max_output_tokens: Maximum number of tokens to generate (optional)
        system_instruction: Static system instructions for prompt caching (optional)
                            This content will be cached by Gemini to reduce token costs
        raise_on_fallback: If True, raises LLMFallbackError instead of returning a 
                           fallback response when all attempts fail. Use this when
                           you need structured output or want explicit failure handling.
                           (Issue 5.4)
        skip_cache: If True, bypasses internal LLM cache (read and write). Use for
                    tasks like market_analysis where truncated cached responses cause
                    parsing failures.
        
    Returns:
        str: The LLM response content. If a fallback is returned and raise_on_fallback
             is False, the response will be prefixed with FALLBACK_RESPONSE_MARKER.
             
    Raises:
        LLMFallbackError: If raise_on_fallback=True and all LLM attempts fail
    """
    start_time = time.time()
    success = False
    error_type = None
    final_model = None
    
    try:
        # Get best model for task
        model_config = get_model_for_task(task_type, preferred_model)
        final_model = model_config.model_name
        
        # Check quota before making request
        # Include system_instruction in token estimation if provided
        prompt_text = prompt
        if system_instruction:
            prompt_text = system_instruction + "\n\n" + prompt
        
        # Issue 8.4: Character-based token estimation (more accurate than word-based)
        estimated_tokens = _estimate_tokens(prompt_text)
        estimated_cost = model_config.cost_per_input_token * estimated_tokens + model_config.cost_per_output_token * estimated_tokens
        
        # Skip internal quota/backoff checks when DISABLE_QUOTA_CHECK - trust Google's actual API limits.
        # Internal limits (60 RPM, backoff after 429) can cause false "Quota exceeded" when you have quota.
        if not getattr(settings, "DISABLE_QUOTA_CHECK", False):
            quota_status = check_quota(model_config.provider.value, estimated_cost, int(estimated_tokens))
            if quota_status.value == "exceeded":
                log.warning(f"Quota exceeded for {model_config.provider.value}, trying fallback")
                fallback_model = model_registry.get_fallback_model(final_model)
                if fallback_model:
                    model_config = fallback_model
                    final_model = model_config.model_name
                else:
                    raise RuntimeError(f"Quota exceeded for {model_config.provider.value} and no fallback available")
            if is_in_backoff(model_config.provider.value):
                log.warning(f"Provider {model_config.provider.value} is in backoff, trying fallback")
                fallback_model = model_registry.get_fallback_model(final_model)
                if fallback_model:
                    model_config = fallback_model
                    final_model = model_config.model_name
                else:
                    raise RuntimeError(f"Provider {model_config.provider.value} is in backoff and no fallback available")
        
        # Centralized cache: check for a cached response before invoking the LLM
        # Skip cache when skip_cache=True (e.g. market_analysis - truncated cached responses cause parse failures)
        cache_key = _llm_cache._make_key(
            provider=model_config.provider.value,
            model=model_config.model_name,
            temperature=getattr(model_config, "temperature", 0.0),
            prompt=prompt,
            system_instruction=system_instruction,
        ) if not skip_cache else None
        if not skip_cache:
            try:
                cached = await _llm_cache.aget(cache_key)
                if cached:
                    log.debug(f"🟩 Internal LLM cache HIT for {model_config.display_name}")
                    # Record usage and observability as a cached hit (zero cost/tokens assumed)
                    latency_ms = (time.time() - start_time) * 1000
                    record_model_usage(model_config.model_name, 0, latency_ms, success=True)
                    log_prompt_response(
                        prompt=prompt,
                        response=cached,
                        token_count=0,
                        cost=0.0,
                        latency_ms=latency_ms,
                        agent_name=agent_name,
                        success=True,
                        metadata={
                            "internal_cache_hit": True,
                            "cache_source": "internal_llm_cache"
                        }
                    )
                    return cached
                else:
                    log.debug(f"🟥 LLM cache MISS for {model_config.display_name}")
            except Exception as _:
                # Non-fatal cache error; proceed without cache
                log.warning("LLM cache check failed; proceeding without cache")
        else:
            log.debug("🟨 LLM cache SKIPPED (skip_cache=True)")

        # Acquire global semaphore to limit concurrent LLM calls (fair sharing across job_matcher, ranker, etc.)
        async with _get_llm_semaphore():
            # Execute with enhanced error handling
            try:
                # Create model instance based on provider
                model_instance = await _create_model_instance(model_config, response_mime_type, max_output_tokens)

                # Build input: use messages format for system_instruction (ChatGoogleGenerativeAI
                # does not accept system_instruction in constructor; it expects it via messages)
                if system_instruction:
                    llm_input = [SystemMessage(content=system_instruction), HumanMessage(content=prompt)]
                    log.debug(f"📝 Using system instruction via messages ({len(system_instruction)} chars)")
                else:
                    llm_input = prompt

                # Use safe LLM call with comprehensive error handling
                response = await safe_llm_call(
                    lambda: model_instance.ainvoke(llm_input),
                    timeout=model_config.timeout,
                    max_retries=max_retries,
                    agent_name=f"{agent_name}_{model_config.display_name}"
                )
                # Extract content from response
                content = _extract_response_content(response)
            
                # Extract usage metadata to check for Gemini API cache hits
                usage_metadata = _extract_usage_metadata(response)
                cached_tokens = usage_metadata.get("cached_content_token_count", 0)
                api_cache_hit = cached_tokens > 0

                if content and content.strip():
                    success = True

                    # Log cache status from API
                    if api_cache_hit:
                        log.debug(f"🟢 Gemini API cache HIT: {cached_tokens} cached tokens used for {model_config.display_name}")
                    else:
                        log.debug(f"🔴 Gemini API cache MISS: No cached tokens for {model_config.display_name}")

                    log.info(f"✅ Successfully used {model_config.display_name}")

                    # Use actual token counts from API if available, otherwise estimate
                    if usage_metadata.get("total_token_count"):
                        actual_tokens = usage_metadata["total_token_count"]
                        prompt_tokens = usage_metadata.get("prompt_token_count", 0)
                        output_tokens = usage_metadata.get("candidates_token_count", 0)
                    else:
                        # Fallback to estimation
                        actual_tokens = len(content.split()) + len(prompt.split())
                        prompt_tokens = len(prompt.split())
                        output_tokens = len(content.split())

                    latency_ms = (time.time() - start_time) * 1000

                    record_usage(model_config.provider.value, estimated_cost, actual_tokens, success=True)
                    record_model_usage(final_model, actual_tokens, latency_ms, success=True)

                    # Log for observability with cache information
                    log_prompt_response(
                        prompt=prompt,
                        response=content,
                        token_count=actual_tokens,
                        cost=estimated_cost,
                        latency_ms=latency_ms,
                        agent_name=agent_name,
                        success=True,
                        metadata={
                            "api_cache_hit": api_cache_hit,
                            "cached_content_token_count": cached_tokens,
                            "prompt_token_count": prompt_tokens,
                            "output_token_count": output_tokens,
                            **usage_metadata
                        }
                    )

                    # Store in cache for future calls (skip when skip_cache=True)
                    if not skip_cache and cache_key:
                        try:
                            await _llm_cache.aset(cache_key, content)
                        except Exception:
                            log.warning("LLM cache set failed; continuing")

                    return content
                else:
                    raise ValueError("Empty response content from LLM")

            except LLMCancelledError as e:
                error_msg = f"{model_config.display_name} operation was cancelled: {str(e)}"
                log.error(f"💥 {error_msg}")
                error_type = "cancelled"

            except LLMTimeoutError as e:
                error_msg = f"{model_config.display_name} timed out: {str(e)}"
                log.error(f"💥 {error_msg}")
                error_type = "timeout"

            except LLMRateLimitError as e:
                error_msg = f"{model_config.display_name} rate limited: {str(e)}"
                log.error(f"💥 {error_msg}")
                error_type = "rate_limit"
                # Use adaptive backoff: prefer Retry-After from API, fallback to 60s
                backoff_seconds = e.retry_after if e.retry_after else 60
                # Clamp to reasonable bounds: min 5s, max 300s (5 min)
                backoff_seconds = max(5, min(backoff_seconds, 300))
                log.info(f"⏳ Setting backoff for {model_config.provider.value}: {backoff_seconds}s")
                set_backoff(model_config.provider.value, backoff_seconds)

            except LLMError as e:
                error_msg = f"{model_config.display_name} failed: {str(e)}"
                log.error(f"💥 {error_msg}")
                error_type = "llm_error"

            # All attempts failed, try fallback model
            log.warning(f"All attempts failed for {final_model}, trying fallback")
            fallback_model = model_registry.get_fallback_model(final_model)

            if fallback_model:
                return await invoke_llm(prompt, task_type, fallback_model.model_name,
                                       max_retries=1, agent_name=agent_name, response_mime_type=response_mime_type,
                                       system_instruction=system_instruction, raise_on_fallback=raise_on_fallback)

            # No fallback available, use system fallback (Issue 5.4)
            return await _fallback_response(prompt, raise_on_fallback=raise_on_fallback,
                                            original_error=error_type, agent_name=agent_name)
        
    except Exception as e:
        log.error(f"LLM invocation failed: {e}")
        error_type = str(type(e).__name__)
        
        # Record failed usage
        if final_model:
            latency_ms = (time.time() - start_time) * 1000
            record_model_usage(final_model, 0, latency_ms, success=False, error_type=error_type)
            
            log_prompt_response(
                prompt=prompt,
                response="",
                token_count=0,
                cost=0.0,
                latency_ms=latency_ms,
                agent_name=agent_name,
                success=False,
                error_type=error_type
            )
        
        # Issue 5.4: Pass through raise_on_fallback and context
        return await _fallback_response(prompt, raise_on_fallback=raise_on_fallback, 
                                        original_error=str(e), agent_name=agent_name)

async def _create_model_instance(model_config, response_mime_type: Optional[str] = None, max_output_tokens: Optional[int] = None):
    """Create model instance - only supports Gemini models
    
    Args:
        model_config: Model configuration
        response_mime_type: MIME type for response (optional)
        max_output_tokens: Maximum output tokens (optional)
    
    Note: system_instruction is passed via messages in invoke_llm, not here.
    ChatGoogleGenerativeAI does not accept system_instruction in its constructor.
    """
    if model_config.provider.value == "gemini":
        # Build ChatGoogleGenerativeAI with direct parameters (not model_kwargs)
        # response_mime_type and max_output_tokens should be passed directly, not in model_kwargs
        kwargs = {
            "model": model_config.model_name,
            "temperature": model_config.temperature,
            "timeout": model_config.timeout - 5,
            "google_api_key": settings.GOOGLE_API_KEY,
        }
        
        # Add response_mime_type as direct parameter if provided
        if response_mime_type:
            kwargs["response_mime_type"] = response_mime_type
            # Disable thinking for JSON output - Gemini 2.5 Flash thinking tokens
            # can corrupt JSON (missing commas, prepended content). See:
            # https://discuss.ai.google.dev/t/2-5-flash-stopped-delivering-true-json-structures/100175
            if response_mime_type == "application/json":
                kwargs["thinking_budget"] = 0
        
        # Add max_output_tokens as direct parameter if provided
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
            
        return ChatGoogleGenerativeAI(**kwargs)
    else:
        raise ValueError(f"Only Gemini models are supported. Provider {model_config.provider.value} is not allowed.")

def _extract_response_content(response) -> str:
    """Extract content from various response types"""
    if hasattr(response, "content"):
        return response.content
    elif isinstance(response, str):
        return response
    elif isinstance(response, dict):
        return response.get("content", str(response))
    else:
        return str(response)

def estimate_tokens(text: str) -> int:
    """Public token estimation for pre-call validation and logging.
    
    Uses same logic as internal _estimate_tokens.
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    return _estimate_tokens(text)


def _extract_usage_metadata(response) -> Dict[str, Any]:
    """Extract usage metadata from Gemini API response, including cache information.
    
    Handles: Gemini API response, LangChain AIMessage (response_metadata), dict.
    
    Returns:
        Dict with usage metadata including:
        - prompt_token_count: Total input tokens
        - cached_content_token_count: Tokens served from cache (if > 0, cache was used)
        - candidates_token_count: Output tokens
        - total_token_count: Total tokens
    """
    usage_metadata = {}
    
    # Check for usage_metadata attribute (Gemini API response)
    if hasattr(response, "usage_metadata"):
        metadata = response.usage_metadata
        if hasattr(metadata, "prompt_token_count"):
            usage_metadata["prompt_token_count"] = metadata.prompt_token_count
        if hasattr(metadata, "cached_content_token_count"):
            usage_metadata["cached_content_token_count"] = metadata.cached_content_token_count
        if hasattr(metadata, "candidates_token_count"):
            usage_metadata["candidates_token_count"] = metadata.candidates_token_count
        if hasattr(metadata, "total_token_count"):
            usage_metadata["total_token_count"] = metadata.total_token_count
    
    # LangChain AIMessage: check response_metadata for usage
    elif hasattr(response, "response_metadata") and response.response_metadata:
        meta = response.response_metadata
        # Gemini-style (usage_metadata dict with prompt_token_count, etc.)
        if "usage_metadata" in meta:
            um = meta["usage_metadata"]
            if isinstance(um, dict):
                usage_metadata["prompt_token_count"] = um.get("prompt_token_count", 0)
                usage_metadata["cached_content_token_count"] = um.get("cached_content_token_count", 0)
                usage_metadata["candidates_token_count"] = um.get("candidates_token_count", 0)
                usage_metadata["total_token_count"] = um.get("total_token_count", 0)
        # LangChain-style: input_tokens/output_tokens at top level
        elif "input_tokens" in meta or "output_tokens" in meta:
            pt = meta.get("input_tokens", 0)
            ot = meta.get("output_tokens", 0)
            usage_metadata["prompt_token_count"] = pt
            usage_metadata["candidates_token_count"] = ot
            usage_metadata["total_token_count"] = pt + ot
    
    # Also check for dict-style access (some SDK versions)
    elif isinstance(response, dict):
        if "usage_metadata" in response:
            metadata = response["usage_metadata"]
            usage_metadata["prompt_token_count"] = metadata.get("prompt_token_count", 0)
            usage_metadata["cached_content_token_count"] = metadata.get("cached_content_token_count", 0)
            usage_metadata["candidates_token_count"] = metadata.get("candidates_token_count", 0)
            usage_metadata["total_token_count"] = metadata.get("total_token_count", 0)
        # Also check for camelCase (cachedContentTokenCount)
        elif "usageMetadata" in response:
            metadata = response["usageMetadata"]
            usage_metadata["prompt_token_count"] = metadata.get("promptTokenCount", 0)
            usage_metadata["cached_content_token_count"] = metadata.get("cachedContentTokenCount", 0)
            usage_metadata["candidates_token_count"] = metadata.get("candidatesTokenCount", 0)
            usage_metadata["total_token_count"] = metadata.get("totalTokenCount", 0)
    
    return usage_metadata


def record_direct_llm_usage(
    response: Any,
    prompt: str,
    content: str,
    agent_name: str,
    start_time: float,
    model_name: Optional[str] = None,
) -> None:
    """Record token usage for direct LangChain/API calls that bypass invoke_llm.
    
    Use this when calling ChatGoogleGenerativeAI.ainvoke() or similar directly,
    so usage is tracked in quota, model registry, and observability.
    
    Args:
        response: Raw LLM response (AIMessage or dict with usage_metadata)
        prompt: Input prompt text
        content: Extracted response content (string)
        agent_name: Name for observability (e.g. "job_matcher")
        start_time: time.time() from before the call
        model_name: Model used (optional; defaults to task model)
    """
    usage_metadata = _extract_usage_metadata(response)
    if usage_metadata.get("total_token_count"):
        actual_tokens = usage_metadata["total_token_count"]
        prompt_tokens = usage_metadata.get("prompt_token_count", 0)
        output_tokens = usage_metadata.get("candidates_token_count", 0)
    else:
        actual_tokens = _estimate_tokens(prompt) + _estimate_tokens(content)
        prompt_tokens = _estimate_tokens(prompt)
        output_tokens = _estimate_tokens(content)
    
    model_config = get_model_for_task(TaskType.TEXT_GENERATION, model_name or settings.GEMINI_MODEL)
    estimated_cost = (
        model_config.cost_per_input_token * prompt_tokens
        + model_config.cost_per_output_token * output_tokens
    )
    latency_ms = (time.time() - start_time) * 1000
    
    record_usage(model_config.provider.value, estimated_cost, actual_tokens, success=True)
    record_model_usage(model_config.model_name, actual_tokens, latency_ms, success=True)
    log_prompt_response(
        prompt=prompt,
        response=content,
        token_count=actual_tokens,
        cost=estimated_cost,
        latency_ms=latency_ms,
        agent_name=agent_name,
        success=True,
        metadata={
            "prompt_token_count": prompt_tokens,
            "output_token_count": output_tokens,
            **usage_metadata,
        },
    )


async def _fallback_response(prompt: str, raise_on_fallback: bool = False,
                            original_error: Optional[str] = None,
                            agent_name: str = "unknown") -> str:
    """Fallback response when LLM fails completely.
    
    Issue 5.4: Enhanced to support explicit failure handling.
    
    Args:
        prompt: The original prompt that failed
        raise_on_fallback: If True, raises LLMFallbackError instead of returning fallback
        original_error: The error that caused the fallback (for diagnostics)
        agent_name: Name of the calling agent (for logging)
        
    Returns:
        str: A fallback response prefixed with FALLBACK_RESPONSE_MARKER
        
    Raises:
        LLMFallbackError: If raise_on_fallback=True
    """
    log.warning(f"🆘 Using fallback response for agent '{agent_name}' due to LLM failures. "
                f"Original error: {original_error or 'unknown'}")
    
    # Generate context-aware fallback based on prompt content
    prompt_lower = prompt.lower()
    if "question" in prompt_lower or "interview" in prompt_lower:
        fallback = "I'd like to learn more about your background and experience. Can you tell me about a recent project you worked on?"
    elif "analysis" in prompt_lower or "evaluate" in prompt_lower:
        fallback = "Based on our conversation, I can see you have relevant experience. Let's continue exploring your skills and background."
    elif "json" in prompt_lower or "structured" in prompt_lower:
        # For structured output requests, provide minimal valid JSON
        fallback = '{"status": "fallback", "message": "LLM temporarily unavailable"}'
    else:
        fallback = "Thank you for sharing that information. I'd like to ask you a few more questions about your experience."
    
    # Issue 5.4: Allow callers to handle failures explicitly
    if raise_on_fallback:
        raise LLMFallbackError(
            message=f"LLM invocation failed after all retries. Original error: {original_error or 'unknown'}",
            fallback_response=fallback,
            original_error=original_error
        )
    
    # Return marked fallback so callers can detect it programmatically
    # Callers can check: if response.startswith(FALLBACK_RESPONSE_MARKER)
    return f"{FALLBACK_RESPONSE_MARKER} {fallback}"


async def rewrite_steps_with_context(templates: List[str], context: Dict[str, Any]) -> List[str]:
    """
    Rewrite/enrich scaffolded steps with user/context specifics while preserving count and safety.

    Constraints:
    - Do not invent URLs. If resources are missing, suggest generic guidance.
    - Keep 4-6 concise steps. Avoid duplicates.
    - Keep steps actionable with milestones/examples.
    """
    try:
        topic = str(context.get("topic", "the topic")).strip()
        band = str(context.get("band", "general")).strip()
        audience = str(context.get("seniority", "practitioner")).strip()
        timeline = str(context.get("timeline", "2-4 weeks")).strip()
        resources = context.get("resources", {})

        # Centralized prompt content with fallback
        try:
            prompt_template = get_prompt("report_steps_rewrite")
        except Exception:
            prompt_template = (
                "You are enhancing a short plan of learning/improvement steps.\n"
                "Rules:\n"
                "- Keep 4-6 steps, concise and actionable.\n"
                "- No external links unless explicitly provided in resources.\n"
                "- Avoid duplicates. Include concrete examples or milestones.\n"
                "Context: {context_json}.\n"
                "Templates to refine (keep structure, improve clarity):\n{templates_block}\n"
                "Return only a numbered list of steps."
            )

        context_json = json.dumps({
            "topic": topic,
            "band": band,
            "audience": audience,
            "timeline": timeline,
            "resources": resources,
        }, ensure_ascii=False)

        templates_block = "\n".join([f"{i+1}. {t}" for i, t in enumerate(templates)])
        prompt = prompt_template.replace("{context_json}", context_json).replace("{templates_block}", templates_block)
        raw = await invoke_llm(prompt, agent_name="report_generator")
        # Parse numbered list back into steps
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        steps: List[str] = []
        for ln in lines:
            # Strip leading numbering like '1. ' or '1) '
            cleaned = ln
            if cleaned[:2].isdigit() and cleaned[1:2] in [".", ")"]:
                cleaned = cleaned[2:].strip()
            elif cleaned[:3].isdigit() and cleaned[2:3] in [".", ")"]:
                cleaned = cleaned[3:].strip()
            steps.append(cleaned)
        steps = [s for s in steps if s]
        if 3 <= len(steps) <= 8:
            return steps[:6]
        return templates[:6]
    except Exception:
        return templates[:6]

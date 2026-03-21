"""
Centralized background task scheduling with error tracking and optional timeout.
Ensures fire-and-forget tasks log failures and can be bounded by a task-level timeout.
"""
import asyncio
import logging
import os
from typing import Optional, Coroutine, Any

log = logging.getLogger(__name__)

# Optional task-level timeout in seconds (env BACKGROUND_TASK_TIMEOUT_SECONDS).
# Default None = no outer timeout (preserves existing behavior).
BACKGROUND_TASK_TIMEOUT_SECONDS: Optional[float] = None


def _parse_timeout() -> Optional[float]:
    raw = os.getenv("BACKGROUND_TASK_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return None
    try:
        val = float(raw)
        return val if val > 0 else None
    except ValueError:
        return None


def _log_task_done(task: asyncio.Task, task_name: str) -> None:
    """Done callback: log exception if task failed."""
    try:
        exc = task.exception()
        if exc is not None:
            log.exception(
                "Background task %s failed: %s",
                task_name,
                exc,
            )
    except asyncio.CancelledError:
        log.warning("Background task %s was cancelled", task_name)
    except Exception as e:
        log.exception("Unexpected error in background task callback for %s: %s", task_name, e)


def schedule_background_task(
    coro: Coroutine[Any, Any, Any],
    task_name: str,
    timeout_seconds: Optional[float] = None,
) -> asyncio.Task:
    """
    Schedule a coroutine as a background task with optional error logging and timeout.

    - Adds a done callback so exceptions are logged (not silently dropped).
    - If timeout_seconds is provided (or BACKGROUND_TASK_TIMEOUT_SECONDS is set and
      timeout_seconds is None), the coroutine is run under asyncio.wait_for so the
      task completes or times out; on timeout, TimeoutError is logged.

    Does not change return values or response bodies; only adds observability and
    optional bounded execution.
    """
    global BACKGROUND_TASK_TIMEOUT_SECONDS
    if BACKGROUND_TASK_TIMEOUT_SECONDS is None:
        BACKGROUND_TASK_TIMEOUT_SECONDS = _parse_timeout()

    effective_timeout = timeout_seconds if timeout_seconds is not None else BACKGROUND_TASK_TIMEOUT_SECONDS

    async def _run() -> None:
        if effective_timeout is not None and effective_timeout > 0:
            try:
                await asyncio.wait_for(asyncio.shield(coro), timeout=effective_timeout)
            except asyncio.TimeoutError:
                log.error(
                    "Background task %s timed out after %s seconds",
                    task_name,
                    effective_timeout,
                )
        else:
            await coro

    task = asyncio.create_task(_run())
    task.add_done_callback(lambda t: _log_task_done(t, task_name))
    return task

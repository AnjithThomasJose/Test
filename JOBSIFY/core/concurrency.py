"""
Global concurrency management for the agent system.
Provides semaphore-based concurrency control at the graph boundary.
"""
import asyncio
import os
from typing import Optional

# Global concurrency limit for LangGraph parallel branches (configurable via env).
# Default 12 improves multi-branch flows (compare/career) under concurrent HTTP load;
# tune per LLM/Chroma capacity (lower if you hit 429s).
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "12"))

# Global semaphore for graph-level concurrency control
_global_semaphore: Optional[asyncio.Semaphore] = None
_semaphore_lock = asyncio.Lock()

async def get_global_semaphore() -> asyncio.Semaphore:
    """Get or create the global concurrency semaphore."""
    global _global_semaphore
    if _global_semaphore is None:
        async with _semaphore_lock:
            if _global_semaphore is None:
                _global_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    return _global_semaphore

async def acquire_concurrency_slot():
    """Acquire a slot from the global semaphore."""
    semaphore = await get_global_semaphore()
    return await semaphore.acquire()

def release_concurrency_slot():
    """Release a slot back to the global semaphore."""
    if _global_semaphore is not None:
        _global_semaphore.release()

async def with_concurrency_limit(coro):
    """Execute a coroutine with concurrency limit."""
    await acquire_concurrency_slot()
    try:
        return await coro
    finally:
        release_concurrency_slot()

def get_max_concurrency() -> int:
    """Get the current max concurrency setting."""
    return MAX_CONCURRENCY







"""
Timeout utilities with jitter and exponential backoff.
Provides strict timeout configuration and retry logic.
"""
import asyncio
import random
import time
from typing import Optional, Callable, Any
import httpx

# Default timeout values
CONNECT_TIMEOUT = 3.0  # Connection timeout in seconds
READ_TIMEOUT = 90.0    # Read timeout in seconds
WRITE_TIMEOUT = 10.0   # Write timeout in seconds
POOL_TIMEOUT = 5.0     # Pool timeout in seconds

# Default retry configuration
MAX_RETRIES = 4
BASE_BACKOFF = 0.6     # Base backoff in seconds
BACKOFF_CAP = 8.0      # Maximum backoff in seconds

def get_timeout_config(
    connect: Optional[float] = None,
    read: Optional[float] = None,
    write: Optional[float] = None,
    pool: Optional[float] = None
) -> httpx.Timeout:
    """Get a timeout configuration with defaults."""
    return httpx.Timeout(
        connect=connect or CONNECT_TIMEOUT,
        read=read or READ_TIMEOUT,
        write=write or WRITE_TIMEOUT,
        pool=pool or POOL_TIMEOUT
    )

def calculate_backoff(attempt: int, base: float = BASE_BACKOFF, cap: float = BACKOFF_CAP) -> float:
    """
    Calculate exponential backoff with full jitter.
    
    Full jitter: sleep_time = random(0, min(cap, base * 2^attempt))
    This provides better distribution of retry attempts.
    """
    exponential = min(cap, base * (2 ** attempt))
    return random.uniform(0, exponential)

async def retry_with_backoff(
    func: Callable,
    max_retries: int = MAX_RETRIES,
    base_backoff: float = BASE_BACKOFF,
    backoff_cap: float = BACKOFF_CAP,
    *args,
    **kwargs
) -> Any:
    """
    Execute a function with exponential backoff and full jitter.
    
    Args:
        func: Async function to execute
        max_retries: Maximum number of retry attempts
        base_backoff: Base backoff time in seconds
        backoff_cap: Maximum backoff time in seconds
        *args, **kwargs: Arguments to pass to func
    
    Returns:
        Result from func
    
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            last_exception = e
            if attempt < max_retries:
                backoff = calculate_backoff(attempt, base_backoff, backoff_cap)
                await asyncio.sleep(backoff)
            else:
                raise
        except Exception as e:
            # Don't retry on non-network errors
            raise
    
    if last_exception:
        raise last_exception

async def with_timeout(
    coro: Callable,
    timeout_seconds: float,
    *args,
    **kwargs
) -> Any:
    """
    Execute a coroutine with a timeout.
    
    Args:
        coro: Coroutine to execute
        timeout_seconds: Timeout in seconds
        *args, **kwargs: Arguments to pass to coro
    
    Returns:
        Result from coro
    
    Raises:
        asyncio.TimeoutError if timeout is exceeded
    """
    return await asyncio.wait_for(coro(*args, **kwargs), timeout=timeout_seconds)







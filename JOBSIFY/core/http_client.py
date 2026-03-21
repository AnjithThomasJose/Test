"""
Shared HTTP client with keep-alive and connection pooling.
Provides a singleton httpx.AsyncClient for efficient HTTP requests.
"""
import httpx
import asyncio
import os
from typing import Optional

# Check if HTTP/2 is available
try:
    import h2  # noqa: F401
    HTTP2_AVAILABLE = True
except ImportError:
    HTTP2_AVAILABLE = False

# Global HTTP client instance
_http_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()

# Default timeout configuration (configurable via environment variables)
DEFAULT_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "10.0"))
DEFAULT_READ_TIMEOUT = float(os.getenv("HTTP_READ_TIMEOUT", "90.0"))
DEFAULT_WRITE_TIMEOUT = float(os.getenv("HTTP_WRITE_TIMEOUT", "10.0"))
DEFAULT_POOL_TIMEOUT = float(os.getenv("HTTP_POOL_TIMEOUT", "5.0"))

# Default connection limits (configurable via environment variables)
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "50"))
DEFAULT_MAX_CONNECTIONS = int(os.getenv("HTTP_MAX_CONNECTIONS", "200"))

async def get_http_client() -> httpx.AsyncClient:
    """
    Get or create the global shared HTTP client.
    Uses connection pooling and keep-alive for efficiency.
    """
    global _http_client
    if _http_client is None:
        async with _client_lock:
            if _http_client is None:
                timeout = httpx.Timeout(
                    connect=DEFAULT_CONNECT_TIMEOUT,
                    read=DEFAULT_READ_TIMEOUT,
                    write=DEFAULT_WRITE_TIMEOUT,
                    pool=DEFAULT_POOL_TIMEOUT
                )
                limits = httpx.Limits(
                    max_keepalive_connections=DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
                    max_connections=DEFAULT_MAX_CONNECTIONS
                )
                # Enable HTTP/2 only if h2 package is installed
                _http_client = httpx.AsyncClient(
                    timeout=timeout,
                    limits=limits,
                    http2=HTTP2_AVAILABLE  # Enable HTTP/2 if available
                )
    return _http_client

async def close_http_client():
    """Close the global HTTP client (call on shutdown)."""
    global _http_client
    if _http_client is not None:
        async with _client_lock:
            if _http_client is not None:
                await _http_client.aclose()
                _http_client = None


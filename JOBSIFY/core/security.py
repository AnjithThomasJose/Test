"""
Centralized security utilities for all KAFIN agents.

This module provides common security functions including PII redaction,
tenant validation, injection attack prevention, token verification,
and callback URL whitelisting.
"""

import os
import re
import asyncio
import logging
import socket
import ipaddress
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

log = logging.getLogger(__name__)


# Centralized security patterns
TENANT_ID_RX = re.compile(r"^[a-zA-Z0-9_\-]{8,64}$")
USER_ID_RX = re.compile(r"^[a-zA-Z0-9_\-]{8,64}$")

# PII patterns for redaction (precompiled for performance)
PII_PATTERNS = [
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b', re.IGNORECASE),                      # SSN
    re.compile(r'\b(?:\d[ -]*?){13,16}\b', re.IGNORECASE),                    # Credit card
    re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', re.IGNORECASE),  # Email
    re.compile(r'\b(?:\+?\d[\d(). -]{8,}\d)\b', re.IGNORECASE),               # Phone numbers
    re.compile(r'\bhttps?://[^\s)]+|\bwww\.[^\s)]+\b', re.IGNORECASE),        # URLs
    re.compile(r'(?<![A-Za-z0-9_])@[a-z][A-Za-z0-9_]*(?![A-Za-z0-9_])'),     # Social handles
]

# Injection attack filters
INJECTION_FILTERS = [re.compile(p) for p in [
    r'(?i)ignore\s+all?\s+previous\s+instructions?',
    r'(?i)disregard\s+previous\s+instructions?',
    r'(?i)\[system\]',
    r'(?i)\[assistant\]',
    r'(?i)new\s+instructions?',
    r'(?i)you\s+are\s+now\s+',
    r'(?i)forget\s+everything',
    r'(?i)override\s+your\s+instructions'
]]


def validate_tenant_id(tenant_id: str) -> bool:
    """Validate tenant ID format."""
    return bool(tenant_id and TENANT_ID_RX.fullmatch(tenant_id))


def validate_user_id(user_id: str) -> bool:
    """Validate user ID format."""
    return bool(user_id and USER_ID_RX.fullmatch(user_id))


def redact_pii(text: str, custom_patterns: List[re.Pattern] = None) -> str:
    """Redact PII from text."""
    patterns = custom_patterns or PII_PATTERNS
    for pattern in patterns:
        text = pattern.sub("[REDACTED]", text)
    return text


def filter_injection_attempts(text: str, custom_filters: List[re.Pattern] = None) -> str:
    """Filter injection attack attempts from text."""
    filters = custom_filters or INJECTION_FILTERS
    for pattern in filters:
        text = pattern.sub("[FILTERED]", text)
    return text


def sanitize_text_for_llm(text: str, redact_pii_flag: bool = True, filter_injection_flag: bool = True) -> str:
    """Comprehensive text sanitization for LLM processing."""
    if not isinstance(text, str):
        return ""
    
    # Keep printable chars
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
    
    # Redact PII
    if redact_pii_flag:
        text = redact_pii(text)
    
    # Filter injection attempts
    if filter_injection_flag:
        text = filter_injection_attempts(text)
    
    return text.strip()


def validate_payload_size(payload: Dict[str, Any], max_size: int = 100000) -> bool:
    """Validate payload size to prevent memory exhaustion."""
    try:
        import json
        payload_str = json.dumps(payload)
        return len(payload_str) <= max_size
    except (TypeError, ValueError):
        return False


def validate_resume_length(resume_text: str, min_length: int = 100, max_length: int = 500000) -> bool:
    """Validate resume text length."""
    if not isinstance(resume_text, str):
        return False
    return min_length <= len(resume_text.strip()) <= max_length


def validate_skills_count(skills: List[Any], max_count: int = 15) -> bool:
    """Validate skills list length."""
    if not isinstance(skills, list):
        return False
    return len(skills) <= max_count


def validate_experience_entries(experience: List[Any], max_entries: int = 25) -> bool:
    """Validate experience entries count."""
    if not isinstance(experience, list):
        return False
    return len(experience) <= max_entries


def validate_education_entries(education: List[Any], max_entries: int = 20) -> bool:
    """Validate education entries count."""
    if not isinstance(education, list):
        return False
    return len(education) <= max_entries


# --- User-supplied resume/JD/callback URL policy (HTTPS + DNS + optional host allowlist) ---
RESUME_URL_ALLOWED_SCHEMES = frozenset({"https"})


def get_resume_url_allowlist_hosts() -> set:
    """Lowercased hostnames from RESUME_ALLOWLIST env (comma-separated). Empty set = no extra restriction."""
    return {h.strip().lower() for h in os.getenv("RESUME_ALLOWLIST", "").split(",") if h.strip()}


def _ip_blocked_for_untrusted_url(ip: Any) -> bool:
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        return True
    if ip.version == 4:
        parts = str(ip).split(".")
        if len(parts) == 4 and parts[0] == "169" and parts[1] == "254":
            return True
    if ip.version == 6:
        s = str(ip).lower()
        if s.startswith("fd") or s.startswith("fe80:"):
            return True
    return False


def host_is_public_for_resume_url(host: str) -> bool:
    """
    Sync DNS resolution for URL policy. Blocks private/link-local/metadata-style ranges.
    Call from threads or sync code; use host_is_public_for_resume_url_async from async handlers.
    """
    if not host:
        return False
    try:
        addrs = {ai[4][0] for ai in socket.getaddrinfo(host, None)}
        for a in addrs:
            ip = ipaddress.ip_address(a)
            if _ip_blocked_for_untrusted_url(ip):
                return False
        return len(addrs) > 0
    except Exception:
        return False


async def host_is_public_for_resume_url_async(host: str) -> bool:
    """Non-blocking DNS check for the event loop."""
    return await asyncio.to_thread(host_is_public_for_resume_url, host)


def sanitize_url(url: str) -> Optional[str]:
    """Validate HTTPS URL, optional RESUME_ALLOWLIST, and public DNS targets (sync)."""
    if not url:
        return None
    p = urlparse(url)
    if p.scheme not in RESUME_URL_ALLOWED_SCHEMES:
        return None
    host = (p.hostname or "").lower()
    allow = get_resume_url_allowlist_hosts()
    if allow and host not in allow:
        return None
    if not host_is_public_for_resume_url(host):
        return None
    return f"{p.scheme}://{p.netloc}{p.path}{('?' + p.query) if p.query else ''}"


async def sanitize_url_async(url: str) -> Optional[str]:
    """Same as sanitize_url but runs DNS resolution in a worker thread."""
    if not url:
        return None
    p = urlparse(url)
    if p.scheme not in RESUME_URL_ALLOWED_SCHEMES:
        return None
    host = (p.hostname or "").lower()
    allow = get_resume_url_allowlist_hosts()
    if allow and host not in allow:
        return None
    if not await host_is_public_for_resume_url_async(host):
        return None
    return f"{p.scheme}://{p.netloc}{p.path}{('?' + p.query) if p.query else ''}"


def mask_sensitive_data(data: Dict[str, Any], sensitive_keys: List[str] = None) -> Dict[str, Any]:
    """Mask sensitive data in dictionaries for logging."""
    if sensitive_keys is None:
        sensitive_keys = ['password', 'token', 'key', 'secret', 'ssn', 'email', 'phone']
    
    masked_data = data.copy()
    
    def _mask_value(value):
        if isinstance(value, str) and len(value) > 6:
            return value[:6] + "***"
        return "***"
    
    def _recursive_mask(obj):
        if isinstance(obj, dict):
            return {k: _recursive_mask(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_recursive_mask(item) for item in obj]
        elif isinstance(obj, str) and any(key in str(obj).lower() for key in sensitive_keys):
            return _mask_value(obj)
        else:
            return obj
    
    return _recursive_mask(masked_data)


def create_security_error_response(error_type: str, message: str) -> Dict[str, Any]:
    """Create standardized security error response."""
    return {
        "success": False,
        "error_type": "security_error",
        "error_category": error_type,
        "error_message": message,
        "timestamp": __import__("time").time(),
        "confidence_score": 0.0,
        "analysis_method": "security_error"
    }


def sanitize_error_message(error: Exception, include_details: bool = False) -> str:
    """
    Sanitize error messages to prevent information leakage.
    
    Args:
        error: The exception to sanitize
        include_details: If True, include safe error details (default: False for production)
        
    Returns:
        Sanitized error message safe to return to clients
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    
    # Map common errors to user-friendly messages
    if "timeout" in error_str or isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        return "Request timed out. Please try again."
    
    if "connection" in error_str or "network" in error_str or "unreachable" in error_str:
        return "Connection error. Please check your network and try again."
    
    if "validation" in error_str or isinstance(error, (ValueError, TypeError)):
        return "Invalid request format. Please check your input and try again."
    
    if "permission" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
        return "Authentication error. Please check your credentials."
    
    if "not found" in error_str or isinstance(error, KeyError):
        return "Requested resource not found."
    
    if "rate limit" in error_str or "too many" in error_str:
        return "Rate limit exceeded. Please try again later."
    
    # For development, include more details if requested
    if include_details and os.getenv("APP_ENV", "").lower() in ["development", "dev"]:
        # Only include error type, not full message
        return f"Error occurred: {error_type}"
    
    # Default: generic error message
    return "An error occurred while processing your request. Please try again."


# ============================================================================
# Token Verification Functions
# ============================================================================

def is_oauth_token(token: str) -> bool:
    """
    Check if token is a Google OAuth access token.
    
    OAuth tokens typically start with 'ya29.' for Google services.
    
    Args:
        token: Token string to check
        
    Returns:
        True if token appears to be an OAuth token
    """
    if not token:
        return False
    # Google OAuth access tokens start with 'ya29.'
    # Service account tokens may start with different patterns
    return token.startswith("ya29.") or len(token) > 100  # OAuth tokens are typically longer


async def verify_oauth_token(token: str) -> bool:
    """
    Verify a Google OAuth access token.
    
    For OAuth tokens, we accept them if they have the correct format.
    Optionally validates via Google's tokeninfo endpoint if network is available.
    
    Args:
        token: OAuth access token to verify
        
    Returns:
        True if token appears valid (format check or tokeninfo validation)
    """
    # Basic format validation - OAuth tokens from Google typically start with 'ya29.'
    # and are reasonably long
    if not token.startswith("ya29.") and len(token) < 50:
        log.warning("OAuth token format validation failed: invalid format")
        return False
    
    # Try to validate via Google's tokeninfo endpoint (best effort)
    try:
        import httpx
        from core.http_client import get_http_client
        
        # Call Google's tokeninfo endpoint to validate the token
        client = await get_http_client()
        response = await client.get(
            f"https://oauth2.googleapis.com/tokeninfo?access_token={token}",
            timeout=3.0  # Short timeout to avoid blocking
        )
        
        if response.status_code == 200:
            data = response.json()
            # Check if token is valid (not expired and has required scopes)
            if "error" not in data:
                log.debug("OAuth token validated successfully via Google tokeninfo")
                return True
            else:
                log.warning(f"OAuth token validation failed: {data.get('error', 'unknown error')}")
                # Still accept if format is correct (token might be expired but format is valid)
                return True
        else:
            # If tokeninfo fails, accept based on format
            log.debug(f"OAuth token tokeninfo unavailable (HTTP {response.status_code}), accepting based on format")
            return True
            
    except Exception as e:
        # If validation fails (network issues, etc.), accept based on format
        log.debug(f"OAuth token tokeninfo validation unavailable: {e}, accepting based on format")
        return True


async def verify_genai_token(token: str, token_source: str = "unknown") -> bool:
    """
    Verify that the OAuth token sent from Node API is valid.
    
    Only accepts Google OAuth access tokens (starts with 'ya29.' or long tokens).
    
    Args:
        token: The OAuth token to verify
        token_source: Where the token was found (e.g., "header", "body") for logging
        
    Returns:
        True if token is valid, False otherwise
    """
    # Mask token for logging (show first 6 chars only)
    masked_token = (token[:6] + "***") if token and len(token) > 6 else "***"
    
    if not token:
        log.warning(f"Empty OAuth token provided for verification | source={token_source}")
        # Log security event for missing token
        try:
            from log_handler import log_security_event
            log_security_event(
                event="token_verification",
                severity="WARNING",
                action="verify_genai_token",
                result="failed",
                metadata={"reason": "empty_token", "source": token_source}
            )
        except Exception:
            pass
        return False
    
    try:
        # Verify OAuth token
        log.debug(f"Verifying OAuth token | source={token_source} | token={masked_token}")
        is_valid = await verify_oauth_token(token)
        
        if is_valid:
            log.info(f"OAuth token verification successful | source={token_source} | token={masked_token}")
            # Log security event for successful verification
            try:
                from log_handler import log_security_event
                log_security_event(
                    event="token_verification",
                    severity="INFO",
                    action="verify_genai_token",
                    result="success",
                    metadata={"token_type": "oauth", "source": token_source, "token_preview": masked_token}
                )
            except Exception:
                pass
            return True
        else:
            log.warning(f"OAuth token verification failed | source={token_source} | token={masked_token}")
            # Log security event for failed verification
            try:
                from log_handler import log_security_event
                log_security_event(
                    event="token_verification",
                    severity="WARNING",
                    action="verify_genai_token",
                    result="failed",
                    metadata={"reason": "oauth_validation_failed", "source": token_source, "token_preview": masked_token}
                )
            except Exception:
                pass
            return False
            
    except Exception as e:
        log.error(f"Error verifying OAuth token: {e} | source={token_source} | token={masked_token}")
        # Log security event for error
        try:
            from log_handler import log_security_event
            log_security_event(
                event="token_verification",
                severity="ERROR",
                action="verify_genai_token",
                result="error",
                metadata={"error": str(e), "source": token_source}
            )
        except Exception:
            pass
        return False


async def verify_request_token(request, body: Optional[Dict[str, Any]] = None) -> bool:
    """
    Extract and verify OAuth token from FastAPI Request object.
    
    The Node API should send a Google OAuth access token to prove authorization.
    Checks for token in:
    1. Authorization header (Bearer token) - preferred
    2. Request body (auth_token, ai_token, or google_api_key field) - if body is provided
    
    In development/local environments, token verification is optional (temporarily).
    In production and QA, token verification is required.
    
    Args:
        request: FastAPI Request object
        body: Optional pre-parsed request body (to avoid consuming body twice)
        
    Returns:
        True if OAuth token is valid or if in development mode, False otherwise
    """
    try:
        # Check environment - skip token verification in development/local (temporarily)
        app_env = os.getenv("APP_ENV", "development").lower()
        if app_env == "dev":
            app_env = "development"
        
        # In development/local, allow requests without token for testing
        # Token verification is completely optional - even invalid tokens are allowed
        if app_env in ["development", "dev", "local"]:
            log.info(f"{app_env.upper()} mode: Token verification optional - allowing all requests")
            # Still try to verify if token is provided for logging purposes, but don't block on failure
            auth_header = request.headers.get("authorization", "")
            token_source = None
            if auth_header.startswith("Bearer "):
                auth_token = auth_header.replace("Bearer ", "").strip()
                token_source = "header"
            else:
                auth_token = auth_header.strip() if auth_header else None
            
            if not auth_token and body:
                if body.get("auth_token"):
                    auth_token = body.get("auth_token")
                    token_source = "body(auth_token)"
                elif body.get("ai_token"):
                    auth_token = body.get("ai_token")
                    token_source = "body(ai_token)"
                elif body.get("google_api_key"):
                    auth_token = body.get("google_api_key")
                    token_source = "body(google_api_key)"
            
            # If no token provided, allow in development/local
            if not auth_token:
                log.info(f"{app_env.upper()} mode: Allowing request without OAuth token")
                return True
            
            # If token is provided, verify it for logging but always allow in dev/local mode
            is_valid = await verify_genai_token(auth_token, token_source or "unknown")
            if is_valid:
                log.info(f"{app_env.upper()} mode: OAuth token verified successfully")
            else:
                log.info(f"{app_env.upper()} mode: OAuth token invalid but allowing request ({app_env} mode)")
            # Always return True in development/local mode regardless of token validity
            return True
        
        # Production and QA: Token verification required
        # Extract token from header (preferred)
        auth_header = request.headers.get("authorization", "")
        token_source = None
        if auth_header.startswith("Bearer "):
            auth_token = auth_header.replace("Bearer ", "").strip()
            token_source = "header"
        else:
            auth_token = auth_header.strip() if auth_header else None
        
        # Fallback to body if not in header and body is provided
        if not auth_token and body:
            if body.get("auth_token"):
                auth_token = body.get("auth_token")
                token_source = "body(auth_token)"
            elif body.get("ai_token"):
                auth_token = body.get("ai_token")
                token_source = "body(ai_token)"
            elif body.get("google_api_key"):
                auth_token = body.get("google_api_key")
                token_source = "body(google_api_key)"
        
        if not auth_token:
            log.warning("No OAuth token found in request headers or body")
            # Log security event for missing token in production
            try:
                from log_handler import log_security_event
                client_ip = getattr(request, "client", None)
                client_ip = client_ip.host if client_ip else "unknown"
                log_security_event(
                    event="token_verification",
                    severity="WARNING",
                    source_ip=client_ip,
                    action="verify_request_token",
                    result="failed",
                    metadata={"reason": "token_not_found"}
                )
            except Exception:
                pass
            return False
        
        return await verify_genai_token(auth_token, token_source or "unknown")
        
    except Exception as e:
        log.error(f"Error extracting/verifying request token: {e}")
        # In development/local, allow on error; in production/qa, deny
        app_env = os.getenv("APP_ENV", "development").lower()
        if app_env in ["development", "dev", "local"]:
            log.debug(f"{app_env.upper()} mode: Allowing request despite verification error")
            return True
        return False


# ============================================================================
# Callback URL Whitelisting Functions
# ============================================================================

def get_whitelisted_callback_domains() -> List[str]:
    """
    Get whitelisted callback domains from environment variable.
    
    Format: comma-separated list of domains (e.g., "api.example.com,webhook.example.com")
    Wildcards are supported (e.g., "*.example.com" matches all subdomains)
    
    Returns:
        List of whitelisted domain patterns
    """
    domains_str = os.getenv("CALLBACK_WHITELIST_DOMAINS", "")
    if not domains_str:
        log.warning("CALLBACK_WHITELIST_DOMAINS not configured. Callback URL whitelisting disabled.")
        return []
    
    domains = [domain.strip().lower() for domain in domains_str.split(",") if domain.strip()]
    log.debug(f"Loaded {len(domains)} whitelisted callback domains")
    return domains


def is_callback_url_whitelisted(callback_url: str) -> bool:
    """
    Check if callback URL is from a whitelisted domain.
    
    Supports:
    - Exact domain matches (e.g., "api.example.com")
    - Subdomain matches (e.g., "webhook.api.example.com" matches "api.example.com")
    - Wildcard patterns (e.g., "*.example.com" matches all subdomains)
    - Development mode exceptions (specific domains allowed in dev only)
    
    Args:
        callback_url: The callback URL to validate
        
    Returns:
        True if URL is whitelisted, False otherwise
    """
    if not callback_url:
        return False
    
    try:
        # Parse URL to extract domain
        parsed = urlparse(callback_url)
        domain = parsed.netloc.lower()
        
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        # Permanent whitelist: Always allowed domains (regardless of environment)
        permanent_whitelist = [
            "apis-buh3qzwapq-uc.a.run.app",  # Cloud Run API endpoint
        ]
        if domain in permanent_whitelist:
            log.info(f"Callback URL whitelisted (permanent): {domain}")
            return True
        
        # Development mode: Allow specific dev-only domains
        app_env = os.getenv("APP_ENV", "development").lower()
        if app_env == "dev":
            app_env = "development"
        
        if app_env in ["development", "dev", "local"]:
            # Dev-only whitelisted domains
            dev_whitelist = [
                "apis-buh3qzwapq-uc.a.run.app",  # Cloud Run API endpoint for dev
            ]
            if domain in dev_whitelist:
                log.info(f"Development mode: Allowing callback URL from dev whitelist: {domain}")
                return True
        
        # Get whitelisted domains
        whitelisted = get_whitelisted_callback_domains()
        
        # If no whitelist configured, allow all (for backward compatibility during migration)
        if not whitelisted:
            log.warning(f"No callback whitelist configured. Allowing callback URL: {domain}")
            return True
        
        # Check exact match
        if domain in whitelisted:
            log.debug(f"Callback URL whitelisted (exact match): {domain}")
            return True
        
        # Check subdomain matches and wildcard patterns
        for pattern in whitelisted:
            # Handle wildcard patterns (e.g., *.example.com)
            if pattern.startswith("*."):
                base_domain = pattern[2:]  # Remove "*."
                if domain == base_domain or domain.endswith(f".{base_domain}"):
                    log.debug(f"Callback URL whitelisted (wildcard match): {domain} matches {pattern}")
                    return True
            # Handle exact domain matches (check if domain is a subdomain of pattern)
            elif domain.endswith(f".{pattern}"):
                log.debug(f"Callback URL whitelisted (subdomain match): {domain} matches {pattern}")
                return True
        
        log.warning(f"Callback URL not whitelisted: {domain} (from {callback_url})")
        return False
        
    except Exception as e:
        log.error(f"Error validating callback URL {callback_url}: {e}")
        return False


def validate_callback_url(callback_url: str) -> Tuple[bool, str]:
    """
    Comprehensive callback URL validation including format and whitelist.
    
    Args:
        callback_url: The callback URL to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if URL is valid and whitelisted
        - error_message: Empty string if valid, error description if invalid
    """
    # Format validation
    if not callback_url:
        return False, "Callback URL is required"
    
    if not isinstance(callback_url, str):
        return False, "Callback URL must be a string"
    
    if not callback_url.startswith(('http://', 'https://')):
        return False, "Invalid callback URL format. Must start with http:// or https://"
    
    # Whitelist validation
    if not is_callback_url_whitelisted(callback_url):
        parsed = urlparse(callback_url)
        domain = parsed.netloc.split(':')[0] if ':' in parsed.netloc else parsed.netloc
        return False, f"Callback URL domain '{domain}' is not whitelisted"
    
    return True, ""

from __future__ import annotations

import asyncio
import json
import re
import time
import hashlib
from typing import Callable, Awaitable, Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from .rate_limit import TokenBucketRateLimiter
from .validators import is_valid_content_type, is_valid_json_size
from .sanitizer import add_security_headers, sanitize_response_json


# Content Safety Classes
class SafetyLevel(Enum):
    """Safety validation levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ContentType(Enum):
    """Types of content to validate"""
    TEXT = "text"
    JSON = "json"
    CODE = "code"
    EMAIL = "email"
    URL = "url"
    PHONE = "phone"

@dataclass
class SafetyViolation:
    """Represents a safety violation"""
    violation_type: str
    severity: SafetyLevel
    message: str
    detected_content: str
    position: Optional[int] = None
    suggested_action: str = "block"

@dataclass
class SafetyResult:
    """Result of safety validation"""
    is_safe: bool
    violations: List[SafetyViolation] = field(default_factory=list)
    sanitized_content: Optional[str] = None
    confidence_score: float = 1.0
    processing_time_ms: float = 0.0

class PIIDetector:
    """Detects and redacts personally identifiable information"""
    
    def __init__(self):
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',
            "api_key": r'\b(sk-|pk-|ak-|AIza|ya29|1//)[A-Za-z0-9_-]+\b',  # Common API key patterns
            "password": r'\b(password|pass|pwd)\s*[:=]\s*["\']?[A-Za-z0-9!@#$%^&*()_+-=]+\b',
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "mac_address": r'\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b'
        }
        
        self.redaction_map = {
            "email": "[EMAIL_REDACTED]",
            "phone": "[PHONE_REDACTED]",
            "ssn": "[SSN_REDACTED]",
            "credit_card": "[CARD_REDACTED]",
            "api_key": "[API_KEY_REDACTED]",
            "password": "[PASSWORD_REDACTED]",
            "ip_address": "[IP_REDACTED]",
            "mac_address": "[MAC_REDACTED]"
        }
    
    def detect_and_redact(self, text: str, redact: bool = True) -> Tuple[str, List[SafetyViolation]]:
        """Detect PII and optionally redact it"""
        violations = []
        sanitized_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                violation = SafetyViolation(
                    violation_type=f"pii_{pii_type}",
                    severity=SafetyLevel.HIGH if pii_type == "api_key" else SafetyLevel.MEDIUM,
                    message=f"PII detected: {pii_type}",
                    detected_content=match.group(),
                    position=match.start(),
                    suggested_action="redact"
                )
                violations.append(violation)
                
                if redact:
                    sanitized_text = sanitized_text.replace(match.group(), self.redaction_map[pii_type])
        
        return sanitized_text, violations

class ContentSafetyChecker:
    """Checks content for safety violations"""
    
    def __init__(self):
        self.toxic_patterns = [
            (r'\b(hate|kill|murder|violence|abuse)\b', SafetyLevel.HIGH),
            (r'\b(discrimination|racist|sexist|homophobic)\b', SafetyLevel.CRITICAL),
            (r'\b(suicide|self.harm|depression)\b', SafetyLevel.HIGH),
            (r'\b(drugs|illegal|crime)\b', SafetyLevel.MEDIUM),
            (r'\b(terrorism|bomb|weapon)\b', SafetyLevel.CRITICAL)
        ]
        
        self.security_patterns = [
            (r'\b(password|secret|key|token)\s*[:=]\s*\w+', SafetyLevel.HIGH),
            (r'\b(admin|root|sudo)\b', SafetyLevel.MEDIUM),
            (r'\b(exploit|hack|crack|bypass)\b', SafetyLevel.HIGH),
            (r'\b(sql\s*injection|xss|csrf)\b', SafetyLevel.CRITICAL)
        ]
    
    def check_safety(self, text: str) -> List[SafetyViolation]:
        """Check text for safety violations"""
        violations = []
        text_lower = text.lower()
        
        # Check toxic content
        for pattern, severity in self.toxic_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                violation = SafetyViolation(
                    violation_type="toxic_content",
                    severity=severity,
                    message=f"Toxic content detected: {match.group()}",
                    detected_content=match.group(),
                    position=match.start(),
                    suggested_action="block"
                )
                violations.append(violation)
        
        # Check security issues
        for pattern, severity in self.security_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                violation = SafetyViolation(
                    violation_type="security_risk",
                    severity=severity,
                    message=f"Security risk detected: {match.group()}",
                    detected_content=match.group(),
                    position=match.start(),
                    suggested_action="block"
                )
                violations.append(violation)
        
        return violations

class ContentSafetyGuardrails:
    """Content safety and guardrails system for LLM inputs/outputs"""
    
    def __init__(self, safety_level: SafetyLevel = SafetyLevel.MEDIUM):
        self.safety_level = safety_level
        self.pii_detector = PIIDetector()
        self.content_checker = ContentSafetyChecker()
        self.violation_history = deque(maxlen=1000)
        
        # Safety level configurations
        self.level_configs = {
            SafetyLevel.LOW: {
                "check_pii": False,
                "check_toxicity": False,
                "check_security": False
            },
            SafetyLevel.MEDIUM: {
                "check_pii": True,
                "check_toxicity": True,
                "check_security": False
            },
            SafetyLevel.HIGH: {
                "check_pii": True,
                "check_toxicity": True,
                "check_security": True
            },
            SafetyLevel.CRITICAL: {
                "check_pii": True,
                "check_toxicity": True,
                "check_security": True,
                "block_on_violation": True
            }
        }
    
    def validate_input(self, content: str, content_type: ContentType = ContentType.TEXT) -> SafetyResult:
        """Validate input content for safety"""
        start_time = time.time()
        violations = []
        sanitized_content = content
        config = self.level_configs[self.safety_level]
        
        # PII detection and redaction
        if config["check_pii"]:
            sanitized_content, pii_violations = self.pii_detector.detect_and_redact(content, redact=True)
            violations.extend(pii_violations)
        
        # Content safety check
        if config["check_toxicity"] or config["check_security"]:
            safety_violations = self.content_checker.check_safety(content)
            violations.extend(safety_violations)
        
        # Determine if content is safe
        is_safe = len(violations) == 0
        if config.get("block_on_violation", False) and violations:
            is_safe = False
        
        # Calculate confidence score
        confidence_score = 1.0
        for violation in violations:
            if violation.severity == SafetyLevel.CRITICAL:
                confidence_score -= 0.3
            elif violation.severity == SafetyLevel.HIGH:
                confidence_score -= 0.2
            elif violation.severity == SafetyLevel.MEDIUM:
                confidence_score -= 0.1
        
        confidence_score = max(0.0, confidence_score)
        
        # Record violations
        for violation in violations:
            self.violation_history.append(violation)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SafetyResult(
            is_safe=is_safe,
            violations=violations,
            sanitized_content=sanitized_content if violations else None,
            confidence_score=confidence_score,
            processing_time_ms=processing_time
        )
    
    def validate_output(self, content: str, input_content: str = "") -> SafetyResult:
        """Validate output content for safety"""
        # Similar to validate_input but with additional checks
        result = self.validate_input(content)
        
        # Additional output-specific checks
        if input_content:
            # Check for data leakage
            leakage_violations = self._check_data_leakage(content, input_content)
            result.violations.extend(leakage_violations)
            result.is_safe = result.is_safe and len(leakage_violations) == 0
        
        return result
    
    def _check_data_leakage(self, output: str, input_content: str) -> List[SafetyViolation]:
        """Check for data leakage between input and output"""
        violations = []
        
        # Look for potential PII leakage
        for pii_type, pattern in self.pii_detector.pii_patterns.items():
            input_matches = re.findall(pattern, input_content, re.IGNORECASE)
            output_matches = re.findall(pattern, output, re.IGNORECASE)
            
            for match in output_matches:
                if match in input_matches:
                    violation = SafetyViolation(
                        violation_type="data_leakage",
                        severity=SafetyLevel.HIGH,
                        message=f"Potential {pii_type} leakage detected",
                        detected_content=match,
                        suggested_action="redact"
                    )
                    violations.append(violation)
        
        return violations
    
    def get_violation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of violations in the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_violations = [
            v for v in self.violation_history 
            if hasattr(v, 'timestamp') and v.timestamp >= cutoff_time
        ]
        
        violation_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for violation in recent_violations:
            violation_counts[violation.violation_type] += 1
            severity_counts[violation.severity.value] += 1
        
        return {
            "total_violations": len(recent_violations),
            "violation_types": dict(violation_counts),
            "severity_breakdown": dict(severity_counts),
            "time_period_hours": hours
        }


class GuardrailMiddleware(BaseHTTPMiddleware):
    """Edge security middleware: IP rate limit, JSON/body validation, security headers, response sanitization."""

    def __init__(
        self,
        app,
        max_requests_per_minute: int = 120,
        max_json_bytes: int = 100 * 1024,  # 100 KB
    ):
        super().__init__(app)
        self.rate_limiter = TokenBucketRateLimiter(
            max_tokens=max_requests_per_minute,
            refill_rate=max_requests_per_minute / 60.0,
        )
        self.max_json_bytes = max_json_bytes
        
        # Security scanning detection patterns
        self.security_scan_patterns = [
            r'\.env',
            r'\.env\.local',
            r'\.env\.production',
            r'\.env\.development',
            r'config\.js',
            r'config\.json',
            r'\.git/config',
            r'\.git/HEAD',
            r'\.gitignore',
            r'\.htaccess',
            r'\.htpasswd',
            r'wp-config\.php',
            r'web\.config',
            r'\.aws/credentials',
            r'\.ssh/id_rsa',
            r'\.ssh/id_dsa',
            r'\.ssh/known_hosts',
            r'\.dockerignore',
            r'Dockerfile',
            r'docker-compose\.yml',
            r'package\.json',
            r'package-lock\.json',
            r'requirements\.txt',
            r'pom\.xml',
            r'\.idea/',
            r'\.vscode/',
            r'\.DS_Store',
            r'\.env\.backup',
            r'\.env\.old',
            r'\.env\.bak',
            r'\.env\.save',
            r'\.env\.swp',
            r'\.env\.tmp',
        ]
        
        # Compile patterns for faster matching
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.security_scan_patterns]
        
        # Track scanning attempts per IP
        self.scan_attempts = defaultdict(int)
        self.scan_attempt_times = defaultdict(list)

    def _is_security_scan_attempt(self, path: str) -> Tuple[bool, Optional[str]]:
        """Check if the request path matches security scanning patterns."""
        path_lower = path.lower()
        for pattern in self.compiled_patterns:
            if pattern.search(path_lower):
                return True, pattern.pattern
        return False, None

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # Check for security scanning attempts
        is_scan, matched_pattern = self._is_security_scan_attempt(path)
        
        if is_scan:
            # Log security event
            try:
                from log_handler import log_security_event, set_correlation_id
                corr_id = set_correlation_id()
                
                # Track scan attempts
                self.scan_attempts[client_ip] += 1
                self.scan_attempt_times[client_ip].append(time.time())
                
                # Keep only last hour of attempts
                cutoff = time.time() - 3600
                self.scan_attempt_times[client_ip] = [
                    t for t in self.scan_attempt_times[client_ip] if t > cutoff
                ]
                
                # Determine severity based on frequency
                recent_attempts = len(self.scan_attempt_times[client_ip])
                severity = "WARNING" if recent_attempts < 5 else "CRITICAL"
                
                log_security_event(
                    event="security_scan_attempt",
                    severity=severity,
                    source_ip=client_ip,
                    action="path_scan",
                    result="blocked",
                    correlation_id=corr_id,
                    metadata={
                        "path": path,
                        "method": request.method,
                        "matched_pattern": matched_pattern,
                        "user_agent": request.headers.get("user-agent", "unknown"),
                        "total_attempts": self.scan_attempts[client_ip],
                        "recent_attempts_1h": recent_attempts
                    }
                )
                
                # Block if too many attempts from same IP
                if recent_attempts >= 10:
                    return JSONResponse(
                        {"ok": False, "error": "forbidden"},
                        status_code=403
                    )
            except Exception as e:
                # Don't fail the request if logging fails
                pass

        # Rate limit early
        if not self.rate_limiter.allow(client_ip):
            return JSONResponse({"ok": False, "error": "rate_limited"}, status_code=429)

        # Validate content-type for JSON endpoints
        if request.method in {"POST", "PUT", "PATCH"}:
            if not is_valid_content_type(request.headers.get("content-type", "")):
                return JSONResponse({"ok": False, "error": "invalid_content_type"}, status_code=415)

            # Peek at body size without consuming it entirely (Starlette stores body for reuse)
            body_bytes = await request.body()
            if not is_valid_json_size(body_bytes, self.max_json_bytes):
                return JSONResponse({"ok": False, "error": "payload_too_large"}, status_code=413)

            # Ensure valid JSON
            try:
                json.loads(body_bytes or b"{}")
            except Exception:
                return JSONResponse({"ok": False, "error": "invalid_json"}, status_code=400)

        # Proceed to downstream app
        response = await call_next(request)

        # Add security headers
        add_security_headers(response)

        # Sanitize JSON responses
        try:
            if isinstance(response, JSONResponse) and isinstance(response.body, (bytes, bytearray)):
                data = json.loads(response.body.decode("utf-8"))
                sanitized = sanitize_response_json(data)
                return JSONResponse(sanitized, status_code=response.status_code, headers=dict(response.headers))
        except Exception:
            # Best-effort sanitization; never fail the response
            pass

        return response


# Global instances for easy access
content_safety_guardrails = ContentSafetyGuardrails()

# Convenience functions for content safety
def validate_input_content(content: str, content_type: ContentType = ContentType.TEXT) -> SafetyResult:
    """Validate input content for safety"""
    return content_safety_guardrails.validate_input(content, content_type)

def validate_output_content(content: str, input_content: str = "") -> SafetyResult:
    """Validate output content for safety"""
    return content_safety_guardrails.validate_output(content, input_content)

def set_content_safety_level(level: SafetyLevel):
    """Set the content safety level"""
    content_safety_guardrails.safety_level = level

def get_content_violation_summary(hours: int = 24) -> Dict[str, Any]:
    """Get content violation summary"""
    return content_safety_guardrails.get_violation_summary(hours)

def detect_and_redact_pii(text: str, redact: bool = True) -> Tuple[str, List[SafetyViolation]]:
    """Detect and redact PII from text"""
    return content_safety_guardrails.pii_detector.detect_and_redact(text, redact)

def check_content_safety(text: str) -> List[SafetyViolation]:
    """Check text for safety violations"""
    return content_safety_guardrails.content_checker.check_safety(text)



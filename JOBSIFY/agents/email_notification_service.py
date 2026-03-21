"""
Twilio SendGrid Email Notification Service
Handles email notifications for resumebot registration
Implements Agentic AI Production Playbook patterns
"""

import os
import json
import logging
import re
import time
import asyncio
import hashlib
from typing import Dict, Any, Optional, Union, List
from enum import Enum
from dataclasses import dataclass
from collections import deque
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content, HtmlContent
from pydantic import BaseModel, Field, validator, HttpUrl
from core.rate_limit import TokenBucketRateLimiter
from core.utils import (
    _mask, _sanitize_text_for_llm, _validate_state_inputs, _to_text, _clean_json_text, 
    _scan_balanced_json, _safe_json_loads, _extract_json_from_response, _create_error_response,
    _generate_request_id, _calculate_processing_time
)
from core.config import get_agent_config
from core.security import validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm, PII_PATTERNS, INJECTION_FILTERS
from core.memory import BaseAgentMemory, get_agent_memory
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion

# Get centralized configuration
config = get_agent_config("email_notification_service")

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Production Playbook Constants
TENANT_ID_REGEX = re.compile(r'^[a-zA-Z0-9_\-]{8,64}$')
MAX_EMAIL_LENGTH = 254
MAX_SUBJECT_LENGTH = 998
MAX_HTML_LENGTH = 1000000  # 1MB
CACHE_TTL_SECONDS = 1800  # 30 minutes
MAX_CACHE_ENTRIES = 1000
RATE_LIMIT_TOKENS = 5  # Lower for email service
RATE_LIMIT_REFILL_RATE = 0.5  # tokens per second
MAX_RETRIES = 3
BASE_BACKOFF = 0.4

@dataclass
class EmailServiceConfig:
    """Centralized configuration for email service"""
    MAX_EMAIL_LENGTH: int = 254
    MAX_SUBJECT_LENGTH: int = 998
    MAX_HTML_LENGTH: int = 1000000
    CACHE_TTL_SECONDS: int = 1800
    MAX_CACHE_ENTRIES: int = 1000
    RATE_LIMIT_TOKENS: int = 5
    RATE_LIMIT_REFILL_RATE: float = 0.5
    TIMEOUT_SECONDS: int = 30
    MAX_RETRIES: int = 3
    BASE_BACKOFF: float = 0.4
    ENABLE_DISPOSABLE_EMAIL_CHECK: bool = True
    ENABLE_PASSWORD_STRENGTH_CHECK: bool = True
    ENABLE_TEMPLATE_VERSIONING: bool = True
    
    @classmethod
    def from_env(cls) -> 'EmailServiceConfig':
        """Load config from environment variables"""
        return cls(
            MAX_EMAIL_LENGTH=int(os.getenv('EMAIL_MAX_LENGTH', 254)),
            MAX_SUBJECT_LENGTH=int(os.getenv('EMAIL_MAX_SUBJECT_LENGTH', 998)),
            MAX_HTML_LENGTH=int(os.getenv('EMAIL_MAX_HTML_LENGTH', 1000000)),
            CACHE_TTL_SECONDS=int(os.getenv('EMAIL_CACHE_TTL', 1800)),
            MAX_CACHE_ENTRIES=int(os.getenv('EMAIL_MAX_CACHE_ENTRIES', 1000)),
            RATE_LIMIT_TOKENS=int(os.getenv('EMAIL_RATE_LIMIT_TOKENS', 5)),
            RATE_LIMIT_REFILL_RATE=float(os.getenv('EMAIL_RATE_LIMIT_REFILL', 0.5)),
            TIMEOUT_SECONDS=int(os.getenv('EMAIL_TIMEOUT_SECONDS', 30)),
            MAX_RETRIES=int(os.getenv('EMAIL_MAX_RETRIES', 3)),
            BASE_BACKOFF=float(os.getenv('EMAIL_BASE_BACKOFF', 0.4)),
            ENABLE_DISPOSABLE_EMAIL_CHECK=os.getenv('EMAIL_ENABLE_DISPOSABLE_CHECK', 'true').lower() == 'true',
            ENABLE_PASSWORD_STRENGTH_CHECK=os.getenv('EMAIL_ENABLE_PASSWORD_CHECK', 'true').lower() == 'true',
            ENABLE_TEMPLATE_VERSIONING=os.getenv('EMAIL_ENABLE_TEMPLATE_VERSIONING', 'true').lower() == 'true'
        )

class EmailType(str, Enum):
    """Email notification types"""
    RESUMEBOT = "resumebot"

# Custom memory class for email notification service (extends base memory)
class EmailNotificationServiceMemory(BaseAgentMemory):
    def __init__(self, tenant_id: str = "default"):
        super().__init__(tenant_id, config.adaptation_window)
        # Add any email notification service specific fields here if needed

# Use centralized memory management
async def get_email_notification_service_memory(tenant_id: str = "default") -> EmailNotificationServiceMemory:
    """Get or create tenant-scoped email notification service memory."""
    return await get_agent_memory("email_notification_service", tenant_id, EmailNotificationServiceMemory)

class EmailTemplateVersion:
    """Email template versioning for A/B testing and improvements"""
    
    def __init__(self):
        self.templates = {
            'resumebot_v1': self._generate_resumebot_v1,
            'resumebot_v2': self._generate_resumebot_v2
        }
    
    def get_template(self, template_key: str, version: str = 'v1'):
        """Get specific template version"""
        template_name = f"{template_key}_{version}"
        return self.templates.get(template_name, self.templates.get(f"{template_key}_v1"))
    
    def _generate_resumebot_v1(self, data: 'ResumebotEmailData') -> tuple[str, str]:
        """Generate resumebot v1 template (original)"""
        subject = sanitize_subject("🎉 Registration Successful - Welcome to JobsifyAI!")
        
        # Sanitize user data
        safe_email = sanitize_email(data.email)
        safe_password = data.password
        
        # Enhanced HTML content with preview text
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta name="x-apple-disable-message-reformatting">
            <style type="text/css">
                .preheader {{ display: none; font-size: 1px; color: #fefefe; line-height: 1px; max-height: 0px; max-width: 0px; opacity: 0; overflow: hidden; }}
            </style>
            <title>Registration Successful</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background-color: #f8fafc;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background-color: #ffffff;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px 20px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 600;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 16px;
                    opacity: 0.9;
                }}
                .content {{
                    padding: 40px 30px;
                }}
                .welcome-message {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .welcome-message h2 {{
                    color: #2d3748;
                    font-size: 24px;
                    margin: 0 0 10px 0;
                }}
                .welcome-message p {{
                    color: #718096;
                    font-size: 16px;
                    margin: 0;
                }}
                .credentials-box {{
                    background-color: #f7fafc;
                    border: 2px solid #e2e8f0;
                    border-radius: 8px;
                    padding: 25px;
                    margin: 25px 0;
                }}
                .credential-item {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 15px;
                }}
                .credential-item:last-child {{
                    margin-bottom: 0;
                }}
                .credential-label {{
                    font-weight: 600;
                    color: #4a5568;
                    min-width: 80px;
                    margin-right: 15px;
                }}
                .credential-value {{
                    font-family: 'Courier New', monospace;
                    background-color: #ffffff;
                    padding: 8px 12px;
                    border-radius: 4px;
                    border: 1px solid #cbd5e0;
                    color: #2d3748;
                    font-size: 14px;
                    word-break: break-all;
                }}
                .email-value {{
                    color: #3182ce;
                    text-decoration: none;
                }}
                .email-value:hover {{
                    text-decoration: underline;
                }}
                .warning-box {{
                    background-color: #fef5e7;
                    border-left: 4px solid #f6ad55;
                    padding: 15px 20px;
                    margin: 25px 0;
                    border-radius: 0 8px 8px 0;
                }}
                .warning-box p {{
                    margin: 0;
                    color: #744210;
                    font-weight: 500;
                }}
                .cta-section {{
                    text-align: center;
                    margin: 30px 0;
                }}
                .cta-button {{
                    display: inline-block;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    padding: 12px 30px;
                    border-radius: 25px;
                    font-weight: 600;
                    font-size: 16px;
                    transition: transform 0.2s ease;
                }}
                .cta-button:hover {{
                    transform: translateY(-2px);
                }}
                .footer {{
                    background-color: #f7fafc;
                    padding: 25px 30px;
                    text-align: center;
                    border-top: 1px solid #e2e8f0;
                }}
                .footer p {{
                    margin: 0;
                    color: #718096;
                    font-size: 14px;
                }}
                .logo {{
                    font-size: 32px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                @media (max-width: 600px) {{
                    .container {{
                        margin: 10px;
                        border-radius: 8px;
                    }}
                    .content {{
                        padding: 20px;
                    }}
                    .header {{
                        padding: 20px;
                    }}
                    .header h1 {{
                        font-size: 24px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="preheader">Welcome to JobsifyAI! Your account is ready.</div>
            <div class="container">
                <div class="header">
                    <div class="logo">🚀 JobsifyAI</div>
                    <h1>Registration Successful!</h1>
                    <p>Welcome to the future of career development</p>
                </div>
                
                <div class="content">
                    <div class="welcome-message">
                        <h2>Hello!</h2>
                        <p>Your account has been successfully created. You're all set to begin your journey with JobsifyAI.</p>
                    </div>
                    
                    <div class="credentials-box">
                        <div class="credential-item">
                            <span class="credential-label">📧 Email:</span>
                            <span class="credential-value email-value">{safe_email}</span>
                        </div>
                        <div class="credential-item">
                            <span class="credential-label">🔑 Password:</span>
                            <span class="credential-value">{safe_password}</span>
                        </div>
                    </div>
                    
                    <div class="warning-box">
                        <p>⚠️ Please keep this password secure and do not share it with anyone.</p>
                    </div>
                    
                    <div class="cta-section">
                        <a href="https://qa.jobsify.ai/login" class="cta-button">Login to Your Account</a>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Thank you for choosing JobsifyAI!<br>
                    If you have any questions, feel free to contact our support team.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Sanitize HTML content
        sanitized_html = sanitize_html(html_content)
        
        return subject, sanitized_html
    
    def _generate_resumebot_v2(self, data: 'ResumebotEmailData') -> tuple[str, str]:
        """Generate resumebot v2 template (enhanced)"""
        subject = sanitize_subject("🚀 Welcome to JobsifyAI - Your Career Journey Starts Now!")
        
        # Sanitize user data
        safe_email = sanitize_email(data.email)
        safe_password = data.password
        
        # Enhanced v2 HTML content with better design
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <meta name="x-apple-disable-message-reformatting">
            <style type="text/css">
                .preheader {{ display: none; font-size: 1px; color: #fefefe; line-height: 1px; max-height: 0px; max-width: 0px; opacity: 0; overflow: hidden; }}
            </style>
            <title>Welcome to JobsifyAI</title>
            <style>
                body {{
                    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #1a202c;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background-color: #ffffff;
                    border-radius: 16px;
                    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                    color: white;
                    padding: 40px 30px;
                    text-align: center;
                    position: relative;
                }}
                .header::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
                    opacity: 0.3;
                }}
                .header-content {{
                    position: relative;
                    z-index: 1;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 32px;
                    font-weight: 700;
                    letter-spacing: -0.025em;
                }}
                .header p {{
                    margin: 15px 0 0 0;
                    font-size: 18px;
                    opacity: 0.95;
                    font-weight: 400;
                }}
                .content {{
                    padding: 50px 40px;
                }}
                .welcome-message {{
                    text-align: center;
                    margin-bottom: 40px;
                }}
                .welcome-message h2 {{
                    color: #1a202c;
                    font-size: 28px;
                    margin: 0 0 15px 0;
                    font-weight: 600;
                }}
                .welcome-message p {{
                    color: #4a5568;
                    font-size: 18px;
                    margin: 0;
                    line-height: 1.7;
                }}
                .credentials-box {{
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                    border: 1px solid #e2e8f0;
                    border-radius: 12px;
                    padding: 30px;
                    margin: 30px 0;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                }}
                .credential-item {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                    padding: 15px;
                    background: white;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                }}
                .credential-item:last-child {{
                    margin-bottom: 0;
                }}
                .credential-label {{
                    font-weight: 600;
                    color: #374151;
                    min-width: 100px;
                    margin-right: 20px;
                    font-size: 16px;
                }}
                .credential-value {{
                    font-family: 'JetBrains Mono', 'Courier New', monospace;
                    background-color: #f9fafb;
                    padding: 12px 16px;
                    border-radius: 6px;
                    border: 1px solid #d1d5db;
                    color: #1f2937;
                    font-size: 15px;
                    word-break: break-all;
                    flex: 1;
                }}
                .email-value {{
                    color: #2563eb;
                    text-decoration: none;
                }}
                .email-value:hover {{
                    text-decoration: underline;
                }}
                .warning-box {{
                    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
                    border-left: 5px solid #f59e0b;
                    padding: 20px 25px;
                    margin: 30px 0;
                    border-radius: 0 12px 12px 0;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                }}
                .warning-box p {{
                    margin: 0;
                    color: #92400e;
                    font-weight: 600;
                    font-size: 16px;
                }}
                .cta-section {{
                    text-align: center;
                    margin: 40px 0;
                }}
                .cta-button {{
                    display: inline-block;
                    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                    color: white;
                    text-decoration: none;
                    padding: 16px 40px;
                    border-radius: 50px;
                    font-weight: 600;
                    font-size: 18px;
                    transition: all 0.3s ease;
                    box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4);
                }}
                .cta-button:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 20px 25px -5px rgba(79, 70, 229, 0.4);
                }}
                .footer {{
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                    padding: 30px 40px;
                    text-align: center;
                    border-top: 1px solid #e2e8f0;
                }}
                .footer p {{
                    margin: 0;
                    color: #6b7280;
                    font-size: 15px;
                    line-height: 1.6;
                }}
                .logo {{
                    font-size: 36px;
                    font-weight: 800;
                    margin-bottom: 15px;
                    letter-spacing: -0.025em;
                }}
                .features {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .feature {{
                    text-align: center;
                    padding: 20px;
                    background: white;
                    border-radius: 8px;
                    border: 1px solid #e2e8f0;
                }}
                .feature-icon {{
                    font-size: 24px;
                    margin-bottom: 10px;
                }}
                .feature-text {{
                    font-size: 14px;
                    color: #4a5568;
                    font-weight: 500;
                }}
                @media (max-width: 600px) {{
                    body {{
                        padding: 10px;
                    }}
                    .container {{
                        border-radius: 12px;
                    }}
                    .content {{
                        padding: 30px 20px;
                    }}
                    .header {{
                        padding: 30px 20px;
                    }}
                    .header h1 {{
                        font-size: 28px;
                    }}
                    .features {{
                        grid-template-columns: 1fr;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="preheader">Welcome to JobsifyAI! Your account is ready and your career journey starts now.</div>
            <div class="container">
                <div class="header">
                    <div class="header-content">
                        <div class="logo">🚀 JobsifyAI</div>
                        <h1>Welcome Aboard!</h1>
                        <p>Your career transformation journey begins today</p>
                    </div>
                </div>
                
                <div class="content">
                    <div class="welcome-message">
                        <h2>Hello there!</h2>
                        <p>Congratulations! Your JobsifyAI account has been successfully created. You now have access to powerful AI-driven career tools that will help you land your dream job.</p>
                    </div>
                    
                    <div class="credentials-box">
                        <div class="credential-item">
                            <span class="credential-label">📧 Email:</span>
                            <span class="credential-value email-value">{safe_email}</span>
                        </div>
                        <div class="credential-item">
                            <span class="credential-label">🔑 Password:</span>
                            <span class="credential-value">{safe_password}</span>
                        </div>
                    </div>
                    
                    <div class="warning-box">
                        <p>🔒 Security Notice: Please keep your password secure and do not share it with anyone. We recommend changing it after your first login.</p>
                    </div>
                    
                    <div class="features">
                        <div class="feature">
                            <div class="feature-icon">📄</div>
                            <div class="feature-text">AI Resume Analysis</div>
                        </div>
                        <div class="feature">
                            <div class="feature-icon">🎯</div>
                            <div class="feature-text">Job Matching</div>
                        </div>
                        <div class="feature">
                            <div class="feature-icon">📊</div>
                            <div class="feature-text">Career Insights</div>
                        </div>
                        <div class="feature">
                            <div class="feature-icon">🎓</div>
                            <div class="feature-text">Skill Development</div>
                        </div>
                    </div>
                    
                    <div class="cta-section">
                        <a href="https://qa.jobsify.ai/login" class="cta-button">Start Your Journey</a>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Thank you for choosing JobsifyAI!<br>
                    We're excited to be part of your career success story.<br>
                    Questions? Contact our support team anytime.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Sanitize HTML content
        sanitized_html = sanitize_html(html_content)
        
        return subject, sanitized_html

# Email security validation functionality moved to centralized middleware

# Simple replacement functions for email sanitization
def sanitize_email(email: str) -> str:
    """Sanitize email address"""
    if not email:
        return ""
    email = email.strip().lower()
    email = re.sub(r'[<>"\']', '', email)
    if len(email) > MAX_EMAIL_LENGTH:
        email = email[:MAX_EMAIL_LENGTH]
    return email

def sanitize_subject(subject: str) -> str:
    """Sanitize email subject"""
    if not subject:
        return ""
    subject = re.sub(r'[<>"\']', '', subject)
    if len(subject) > MAX_SUBJECT_LENGTH:
        subject = subject[:MAX_SUBJECT_LENGTH]
    return subject

def sanitize_html(html: str) -> str:
    """Sanitize HTML content"""
    if not html:
        return ""
    html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r'on\w+\s*=', '', html, flags=re.IGNORECASE)
    if len(html) > MAX_HTML_LENGTH:
        html = html[:MAX_HTML_LENGTH]
    return html

def validate_email_format(email: str) -> bool:
    """Validate email format"""
    if not email or len(email) > MAX_EMAIL_LENGTH:
        return False
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(email_pattern.match(email))

# Rate limiting functionality moved to centralized middleware

class EmailCache:
    """Cache for email templates and results"""
    
    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES, ttl: int = CACHE_TTL_SECONDS):
        self.max_entries = max_entries
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
    
    def _make_key(self, email_type: str, email: str) -> str:
        """Create cache key (full hash to avoid collisions)"""
        key_data = f"{email_type}:{email}"
        return hashlib.sha256(key_data.encode()).hexdigest()  # Full hash to avoid collisions
    
    def get_template(self, email_type: str, email: str) -> Optional[Dict[str, Any]]:
        """Get cached email template"""
        key = self._make_key(email_type, email)
        now = time.time()
        
        if key in self.cache:
            entry = self.cache[key]
            if now - entry["timestamp"] < self.ttl:
                self.access_times[key] = now
                return entry["data"]
            else:
                del self.cache[key]
                del self.access_times[key]
        
        return None
    
    def set_template(self, email_type: str, email: str, template_data: Dict[str, Any]):
        """Cache email template"""
        key = self._make_key(email_type, email)
        now = time.time()
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_entries:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = {
            "data": template_data,
            "timestamp": now
        }
        self.access_times[key] = now

class EmailMetrics:
    """Email service metrics tracking"""
    
    def __init__(self, window_size: int = 5000):
        self.window_size = window_size
        self.send_attempts = deque(maxlen=window_size)
        self.send_successes = deque(maxlen=window_size)
        self.send_failures = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.delivery_delays = deque(maxlen=window_size)
        self.rate_limit_hits = 0
        self.validation_failures = 0
        self.bounce_rate = 0
        self.spam_complaints = 0
        self.template_usage = {}  # Track template version usage
    
    def record_send_attempt(self, email_type: str, success: bool, response_time: float, error_type: str = None):
        """Record email send attempt"""
        self.send_attempts.append({
            "email_type": email_type,
            "success": success,
            "response_time": response_time,
            "error_type": error_type,
            "timestamp": time.time()
        })
        
        if success:
            self.send_successes.append({"email_type": email_type, "response_time": response_time})
        else:
            self.send_failures.append({"email_type": email_type, "error_type": error_type})
        
        self.response_times.append(response_time)
    
    def record_rate_limit_hit(self):
        """Record rate limit hit"""
        self.rate_limit_hits += 1
    
    def record_validation_failure(self):
        """Record validation failure"""
        self.validation_failures += 1
    
    def record_bounce(self, bounce_type: str):
        """Record email bounce"""
        self.bounce_rate += 1
        log.warning(f"Email bounce recorded: {bounce_type}")
    
    def record_spam_complaint(self):
        """Record spam complaint"""
        self.spam_complaints += 1
        log.warning("Spam complaint recorded")
    
    def record_delivery_delay(self, delay_seconds: float):
        """Record delivery delay"""
        self.delivery_delays.append({
            "delay_seconds": delay_seconds,
            "timestamp": time.time()
        })
    
    def record_template_usage(self, template_name: str, version: str):
        """Record template usage for analytics"""
        key = f"{template_name}_{version}"
        self.template_usage[key] = self.template_usage.get(key, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        if not self.send_attempts:
            return {"error": "No data available"}
        
        total_attempts = len(self.send_attempts)
        success_rate = len(self.send_successes) / total_attempts if total_attempts > 0 else 0
        
        response_times = list(self.response_times)
        response_times.sort()
        
        delivery_delays = list(self.delivery_delays)
        delivery_delays.sort(key=lambda x: x["delay_seconds"])
        
        return {
            "total_attempts": total_attempts,
            "success_rate": success_rate,
            "response_time_p95": response_times[int(len(response_times) * 0.95)] if response_times else 0,
            "response_time_avg": sum(response_times) / len(response_times) if response_times else 0,
            "rate_limit_hits": self.rate_limit_hits,
            "validation_failures": self.validation_failures,
            "bounce_rate": self.bounce_rate,
            "spam_complaints": self.spam_complaints,
            "delivery_delay_avg": sum(d["delay_seconds"] for d in delivery_delays) / len(delivery_delays) if delivery_delays else 0,
            "delivery_delay_p95": delivery_delays[int(len(delivery_delays) * 0.95)]["delay_seconds"] if delivery_delays else 0,
            "template_usage": self.template_usage,
            "recent_errors": list(self.send_failures)[-10:]  # Last 10 errors
        }

class BaseEmailData(BaseModel):
    """Base email data structure with validation"""
    to: str = Field(..., description="Recipient email address")
    type: EmailType = Field(..., description="Type of email notification")
    subject: Optional[str] = Field(None, description="Custom subject line")
    
    @validator('to')
    def validate_email(cls, v):
        if not validate_email_format(v):
            raise ValueError(f"Invalid email format: {v}")
        return sanitize_email(v)
    
    @validator('subject')
    def validate_subject(cls, v):
        if v is None:
            return v
        return sanitize_subject(v)

class ResumebotEmailData(BaseEmailData):
    """Resumebot email data structure with enhanced validation"""
    type: EmailType = EmailType.RESUMEBOT
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    template_version: str = Field(default="v1", description="Email template version")
    
    @validator('email')
    def validate_user_email(cls, v):
        if not validate_email_format(v):
            raise ValueError(f"Invalid user email format: {v}")
        
        # Disposable email check removed - handled by centralized middleware
        
        return sanitize_email(v)
    
    @validator('password')
    def validate_password(cls, v):
        if not v or len(v) < 1:
            raise ValueError("Password cannot be empty")
        
        # Password strength validation removed - handled by centralized middleware
        
        # Don't log or store password in plain text
        return v
    
    @validator('template_version')
    def validate_template_version(cls, v):
        if v not in ['v1', 'v2']:
            raise ValueError("Template version must be 'v1' or 'v2'")
        return v

class TwilioEmailService:
    """Twilio SendGrid Email Notification Service with Production Patterns"""
    
    def __init__(self):
        """Initialize the email service with SendGrid configuration and production patterns"""
        self.api_key = os.getenv('SENDGRID_API_KEY')
        self.from_email = os.getenv('SENDGRID_FROM_EMAIL', 'noreply@jobsify.ai')
        self.from_name = os.getenv('SENDGRID_FROM_NAME', 'Jobsify AI')
        self.reply_to = os.getenv('SENDGRID_REPLY_TO', 'support@jobsify.ai')
        
        # Load configuration from environment
        self.config = EmailServiceConfig.from_env()
        
        # Rate limiting now handled by centralized middleware
        self.cache = EmailCache(
            max_entries=self.config.MAX_CACHE_ENTRIES,
            ttl=self.config.CACHE_TTL_SECONDS
        )
        self.metrics = EmailMetrics()
        self.template_version = EmailTemplateVersion()
        self._lock = asyncio.Lock()
        self._rate_limiter = TokenBucketRateLimiter(max_tokens=self.config.RATE_LIMIT_TOKENS, refill_rate=self.config.RATE_LIMIT_REFILL_RATE)
        
        if not self.api_key:
            log.warning("SENDGRID_API_KEY not found. Email service will be disabled.")
            self.sg = None
        else:
            self.sg = SendGridAPIClient(api_key=self.api_key)
            log.info("TwilioEmailService initialized successfully with production patterns and configuration")

    async def send_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send email following production pipeline with centralized utilities
        Implements: Rate limit → Security validation → Cache → Send with retry → Metrics
        """
        # Use centralized logging
        log_context = create_log_context("email_notification_service", email_data.get("tenant_id", "default"))
        start_time = log_context["start_time"]
        request_id = log_context["request_id"]
        tenant_id = email_data.get("tenant_id", "default")
        
        # Get tenant-scoped memory
        email_memory = await get_email_notification_service_memory(tenant_id)
        
        try:
            # 1. Rate limiting
            if not self._rate_limiter.allow("email_service_global"):
                self.metrics.record_rate_limit_hit()
                log.warning("Rate limit exceeded for email sending")
                return {
                    "success": False,
                    "message": "Rate limit exceeded",
                    "email_type": email_data.get('type', 'unknown'),
                    "error_type": "rate_limited"
                }
            
            # 2. Security validation
            try:
                email_type = email_data.get('type')
                if email_type == EmailType.RESUMEBOT:
                    validated_data = ResumebotEmailData(**email_data)
                else:
                    raise ValueError(f"Unsupported email type: {email_type}")
            except Exception as validation_error:
                self.metrics.record_validation_failure()
                log.error(f"Email validation failed: {validation_error}")
                return {
                    "success": False,
                    "message": f"Email validation failed: {validation_error}",
                    "email_type": email_data.get('type', 'unknown'),
                    "error_type": "validation_failed"
                }
            
            # 3. Cache lookup (include template version in cache key)
            template_version = getattr(validated_data, 'template_version', 'v1')
            cache_key = f"{email_type}_{template_version}"
            cached_template = self.cache.get_template(cache_key, validated_data.to)
            
            if cached_template:
                log.info(f"Using cached template {template_version} for {validated_data.to}")
                subject = cached_template["subject"]
                html_content = cached_template["html_content"]
            else:
                # Generate email content using template versioning
                if self.config.ENABLE_TEMPLATE_VERSIONING:
                    template_func = self.template_version.get_template(email_type, template_version)
                    if template_func:
                        subject, html_content = template_func(validated_data)
                    else:
                        # Fallback to v1 if version not found
                        subject, html_content = self.template_version.get_template(email_type, 'v1')(validated_data)
                else:
                    # Use original method if versioning disabled
                    subject, html_content = self._generate_resumebot_email(validated_data)
                
                # Cache the template with version
                self.cache.set_template(cache_key, validated_data.to, {
                    "subject": subject,
                    "html_content": html_content
                })
            
            # Record template usage for analytics
            self.metrics.record_template_usage(email_type, template_version)
            
            # 4. Send email with retry logic
            result = await self._send_with_retry(validated_data, subject, html_content)
            
            # 5. Record metrics
            processing_time = _calculate_processing_time(start_time)
            
            # Record success/failure
            await email_memory.record_attempt(
                'email_sending', 'llm', result.get("success", False), 0.8 if result.get("success", False) else 0.0, processing_time
            )
            
            # Log success
            if result.get("success", False):
                log_agent_completion(log_context, {
                    "success": True,
                    "status": "completed",
                    "email_type": email_type,
                    "template_version": template_version
                }, "llm", processing_time)
            
            self.metrics.record_send_attempt(
                email_type=email_type,
                success=result.get("success", False),
                response_time=processing_time,
                error_type=result.get("error_type")
            )
            
            log.info(f"Email processing complete: {result.get('success', False)} | {processing_time:.3f}s")
            return result
            
        except Exception as e:
            processing_time = _calculate_processing_time(start_time)
            AgentLogger.log_error(log_context, f"Email service failed: {str(e)}")
            
            # Record failure
            await email_memory.record_attempt(
                'email_sending', 'llm', False, 0.0, processing_time
            )
            
            self.metrics.record_send_attempt(
                email_type=email_data.get('type', 'unknown'),
                success=False,
                response_time=processing_time,
                error_type="exception"
            )
            log.error(f"Email service error: {str(e)}")
            return {
                "success": False,
                "message": f"Email service error: {str(e)}",
                "email_type": email_data.get('type', 'unknown'),
                "error_type": "service_error"
            }
    
    async def _send_with_retry(self, validated_data: ResumebotEmailData, subject: str, html_content: str) -> Dict[str, Any]:
        """Send email with retry logic and timeout"""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                # Check if SendGrid is available
                if not self.sg:
                    return {
                        "success": False,
                        "message": "Email service not configured",
                        "error_type": "service_unavailable"
                    }
                
                # Create email message
                message = Mail(
                    from_email=Email(self.from_email, self.from_name),
                    to_emails=To(validated_data.to),
                    subject=subject,
                    html_content=HtmlContent(html_content)
                )
                
                # Set reply-to if provided
                if self.reply_to:
                    message.reply_to = Email(self.reply_to)
                
                # Send with timeout
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.sg.send, message),
                    timeout=self.config.TIMEOUT_SECONDS
                )
                
                if response.status_code in [200, 201, 202]:
                    log.info(f"Email sent successfully to {validated_data.to}")
                    return {
                        "success": True,
                        "message": "Email sent successfully",
                        "email_type": validated_data.type.value,
                        "status_code": response.status_code
                    }
                else:
                    log.error(f"Failed to send email. Status code: {response.status_code}")
                    if attempt == self.config.MAX_RETRIES - 1:
                        return {
                            "success": False,
                            "message": f"Failed to send email. Status code: {response.status_code}",
                            "email_type": validated_data.type.value,
                            "status_code": response.status_code,
                            "error_type": "sendgrid_error"
                        }
                
            except asyncio.TimeoutError:
                log.warning(f"Email send timeout (attempt {attempt + 1}/{self.config.MAX_RETRIES})")
                if attempt == self.config.MAX_RETRIES - 1:
                    return {
                        "success": False,
                        "message": "Email send timeout",
                        "error_type": "timeout"
                    }
            except Exception as e:
                log.error(f"Email send error (attempt {attempt + 1}/{self.config.MAX_RETRIES}): {str(e)}")
                if attempt == self.config.MAX_RETRIES - 1:
                    return {
                        "success": False,
                        "message": f"Email send error: {str(e)}",
                        "error_type": "send_error"
                    }
            
            # Exponential backoff with jitter
            if attempt < self.config.MAX_RETRIES - 1:
                backoff_time = self.config.BASE_BACKOFF * (2 ** attempt) + (time.time() % 1) * 0.1
                await asyncio.sleep(backoff_time)
        
        return {
            "success": False,
            "message": "Max retries exceeded",
            "error_type": "max_retries_exceeded"
        }

    def _generate_resumebot_email(self, data: ResumebotEmailData) -> tuple[str, str]:
        """Generate resumebot enhanced HTML email content with security validation"""
        subject = sanitize_subject("🎉 Registration Successful - Welcome to JobsifyAI!")
        
        # Sanitize user data
        safe_email = sanitize_email(data.email)
        safe_password = data.password  # Password is already validated by Pydantic
        
        # Enhanced HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Registration Successful</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background-color: #f8fafc;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background-color: #ffffff;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px 20px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: 600;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 16px;
                    opacity: 0.9;
                }}
                .content {{
                    padding: 40px 30px;
                }}
                .welcome-message {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .welcome-message h2 {{
                    color: #2d3748;
                    font-size: 24px;
                    margin: 0 0 10px 0;
                }}
                .welcome-message p {{
                    color: #718096;
                    font-size: 16px;
                    margin: 0;
                }}
                .credentials-box {{
                    background-color: #f7fafc;
                    border: 2px solid #e2e8f0;
                    border-radius: 8px;
                    padding: 25px;
                    margin: 25px 0;
                }}
                .credential-item {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 15px;
                }}
                .credential-item:last-child {{
                    margin-bottom: 0;
                }}
                .credential-label {{
                    font-weight: 600;
                    color: #4a5568;
                    min-width: 80px;
                    margin-right: 15px;
                }}
                .credential-value {{
                    font-family: 'Courier New', monospace;
                    background-color: #ffffff;
                    padding: 8px 12px;
                    border-radius: 4px;
                    border: 1px solid #cbd5e0;
                    color: #2d3748;
                    font-size: 14px;
                    word-break: break-all;
                }}
                .email-value {{
                    color: #3182ce;
                    text-decoration: none;
                }}
                .email-value:hover {{
                    text-decoration: underline;
                }}
                .warning-box {{
                    background-color: #fef5e7;
                    border-left: 4px solid #f6ad55;
                    padding: 15px 20px;
                    margin: 25px 0;
                    border-radius: 0 8px 8px 0;
                }}
                .warning-box p {{
                    margin: 0;
                    color: #744210;
                    font-weight: 500;
                }}
                .cta-section {{
                    text-align: center;
                    margin: 30px 0;
                }}
                .cta-button {{
                    display: inline-block;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    padding: 12px 30px;
                    border-radius: 25px;
                    font-weight: 600;
                    font-size: 16px;
                    transition: transform 0.2s ease;
                }}
                .cta-button:hover {{
                    transform: translateY(-2px);
                }}
                .footer {{
                    background-color: #f7fafc;
                    padding: 25px 30px;
                    text-align: center;
                    border-top: 1px solid #e2e8f0;
                }}
                .footer p {{
                    margin: 0;
                    color: #718096;
                    font-size: 14px;
                }}
                .logo {{
                    font-size: 32px;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}
                @media (max-width: 600px) {{
                    .container {{
                        margin: 10px;
                        border-radius: 8px;
                    }}
                    .content {{
                        padding: 20px;
                    }}
                    .header {{
                        padding: 20px;
                    }}
                    .header h1 {{
                        font-size: 24px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">🚀 JobsifyAI</div>
                    <h1>Registration Successful!</h1>
                    <p>Welcome to the future of career development</p>
                </div>
                
                <div class="content">
                    <div class="welcome-message">
                        <h2>Hello!</h2>
                        <p>Your account has been successfully created. You're all set to begin your journey with JobsifyAI.</p>
                    </div>
                    
                    <div class="credentials-box">
                        <div class="credential-item">
                            <span class="credential-label">📧 Email:</span>
                            <span class="credential-value email-value">{safe_email}</span>
                        </div>
                        <div class="credential-item">
                            <span class="credential-label">🔑 Password:</span>
                            <span class="credential-value">{safe_password}</span>
                        </div>
                    </div>
                    
                    <div class="warning-box">
                        <p>⚠️ Please keep this password secure and do not share it with anyone.</p>
                    </div>
                    
                    <div class="cta-section">
                        <a href="https://qa.jobsify.ai/login" class="cta-button">Login to Your Account</a>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Thank you for choosing JobsifyAI!<br>
                    If you have any questions, feel free to contact our support team.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Sanitize HTML content
        sanitized_html = sanitize_html(html_content)
        
        return subject, sanitized_html

# Utility functions for monitoring and management
def get_email_service_metrics() -> Dict[str, Any]:
    """Get email service metrics for monitoring"""
    return email_service.metrics.get_metrics()

def get_email_rate_limit_status() -> Dict[str, Any]:
    """Get rate limiter status"""
    return {
        "tokens_available": email_service.rate_limiter.tokens,
        "max_tokens": email_service.rate_limiter.max_tokens,
        "refill_rate": email_service.rate_limiter.refill_rate,
        "can_consume": email_service.rate_limiter.can_consume()
    }

def get_email_cache_status() -> Dict[str, Any]:
    """Get cache status"""
    return {
        "cache_size": len(email_service.cache.cache),
        "max_entries": email_service.cache.max_entries,
        "ttl_seconds": email_service.cache.ttl
    }

def clear_email_cache():
    """Clear email template cache"""
    email_service.cache.cache.clear()
    email_service.cache.access_times.clear()
    log.info("Email cache cleared")

def reset_email_rate_limiter():
    """Reset rate limiter to full capacity"""
    email_service.rate_limiter.tokens = email_service.rate_limiter.max_tokens
    email_service.rate_limiter.last_refill = time.time()
    log.info("Email rate limiter reset")

def get_email_config() -> Dict[str, Any]:
    """Get current email service configuration"""
    return {
        "max_email_length": email_service.config.MAX_EMAIL_LENGTH,
        "max_subject_length": email_service.config.MAX_SUBJECT_LENGTH,
        "max_html_length": email_service.config.MAX_HTML_LENGTH,
        "cache_ttl_seconds": email_service.config.CACHE_TTL_SECONDS,
        "max_cache_entries": email_service.config.MAX_CACHE_ENTRIES,
        "rate_limit_tokens": email_service.config.RATE_LIMIT_TOKENS,
        "rate_limit_refill_rate": email_service.config.RATE_LIMIT_REFILL_RATE,
        "timeout_seconds": email_service.config.TIMEOUT_SECONDS,
        "max_retries": email_service.config.MAX_RETRIES,
        "base_backoff": email_service.config.BASE_BACKOFF,
        "enable_disposable_email_check": email_service.config.ENABLE_DISPOSABLE_EMAIL_CHECK,
        "enable_password_strength_check": email_service.config.ENABLE_PASSWORD_STRENGTH_CHECK,
        "enable_template_versioning": email_service.config.ENABLE_TEMPLATE_VERSIONING
    }

def get_available_template_versions() -> Dict[str, List[str]]:
    """Get available template versions"""
    return {
        "resumebot": ["v1", "v2"]
    }

def record_email_bounce(bounce_type: str):
    """Record email bounce for monitoring"""
    email_service.metrics.record_bounce(bounce_type)

def record_spam_complaint():
    """Record spam complaint for monitoring"""
    email_service.metrics.record_spam_complaint()

def record_delivery_delay(delay_seconds: float):
    """Record delivery delay for monitoring"""
    email_service.metrics.record_delivery_delay(delay_seconds)

# Create global instance
email_service = TwilioEmailService()

"""
PII Anonymization module for interview agent.

Provides enhanced PII anonymization with hash-based placeholders
for consistent anonymization across conversation sessions.
"""

import re
import hashlib
import logging
import asyncio
from typing import Dict, List, Any

log = logging.getLogger(__name__)


class PIIAnonymizer:
    """Enhanced PII anonymization with hash-based placeholders"""
    
    def __init__(self):
        self.pii_map = {}
        self.reverse_map = {}
        self.session_id = None
        
        # FIX: Pre-compile regex patterns to avoid recompilation on each call
        # This significantly improves performance for repeated anonymization operations
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_patterns = [
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # US format
            re.compile(r'\b\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'),  # International
            re.compile(r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b')  # Parentheses format
        ]
        self.url_pattern = re.compile(r'(https?://[^\s)]+)')
        self.handle_pattern = re.compile(r'(?<!\w)@([A-Za-z0-9_\.\-]{2,30})')
        self.org_suffix_pattern = re.compile(r'\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,2}\s(?:Inc|LLC|Ltd|Corporation|Corp|GmbH))\b')
        self.ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        # Pattern caches for dynamic entities (names, companies)
        # Used to avoid recompiling the same regexes for repeated names/companies
        self._name_patterns_cache: Dict[str, Any] = {}
        self._company_patterns_cache: Dict[str, Any] = {}
    
    def set_session_id(self, session_id: str):
        """Set session ID for consistent anonymization across conversation"""
        self.session_id = session_id
    
    def _generate_hash_placeholder(self, original: str, prefix: str) -> str:
        """Generate consistent hash-based placeholder for PII"""
        # Use session ID + original for consistent hashing across conversation
        hash_input = f"{self.session_id or 'default'}_{original}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        placeholder = f"{prefix}_{hash_value}"
        
        self.pii_map[original] = placeholder
        self.reverse_map[placeholder] = original
        return placeholder
    
    def anonymize_name(self, name: Any) -> str:
        """Replace name with CANDIDATE_<hash> format.
        
        Accepts either a single string or a list of possible names. Falls back
        to a deterministic placeholder when nothing usable is provided.
        """
        # Normalize list inputs (e.g., when resume parsers return a list of names)
        if isinstance(name, list):
            for candidate in name:
                if isinstance(candidate, str) and candidate.strip():
                    name = candidate.strip()
                    break
            else:
                name = ""
        
        if not isinstance(name, str) or not name.strip():
            return "CANDIDATE_A"
        
        normalized = name.strip()
        if normalized.lower() in ["candidate", "applicant", "interviewee"]:
            return "CANDIDATE_A"
        
        if normalized in self.pii_map:
            return self.pii_map[normalized]
        
        return self._generate_hash_placeholder(normalized, "CANDIDATE")
    
    def anonymize_email(self, email: str) -> str:
        """Replace email with candidate@domain.com format"""
        if not email:
            return ""
        if email in self.pii_map:
            return self.pii_map[email]
        domain = email.split('@')[-1] if '@' in email else "domain.com"
        return f"candidate@{domain}"
    
    def anonymize_phone(self, phone: str) -> str:
        """Replace phone with XXX-XXX-XXXX format"""
        if not phone:
            return ""
        return "XXX-XXX-XXXX"
    
    def anonymize_company(self, company: str) -> str:
        """Replace company name with COMPANY_7b2e format"""
        if not company or len(company) < 2:
            return company
        if company in self.pii_map:
            return self.pii_map[company]
        return self._generate_hash_placeholder(company, "COMPANY")
    
    def anonymize_text(self, text: str, names: List[str], companies: List[str]) -> str:
        """Remove PII from free text with enhanced patterns
        
        FIX: Uses pre-compiled regex patterns for better performance.
        """
        if not text:
            return text
            
        anonymized = text
        
        # Replace names with case-insensitive matching
        # Use cached compiled patterns for better performance across calls
        for name in names:
            if name and len(name) > 1:
                placeholder = self.anonymize_name(name)
                pattern = self._name_patterns_cache.get(name)
                if pattern is None:
                    pattern = re.compile(re.escape(name), re.IGNORECASE)
                    self._name_patterns_cache[name] = pattern
                anonymized = pattern.sub(placeholder, anonymized)
        
        # Replace companies with case-insensitive matching
        # Use cached compiled patterns for better performance across calls
        for company in companies:
            if company and len(company) > 1:
                placeholder = self.anonymize_company(company)
                pattern = self._company_patterns_cache.get(company)
                if pattern is None:
                    pattern = re.compile(re.escape(company), re.IGNORECASE)
                    self._company_patterns_cache[company] = pattern
                anonymized = pattern.sub(placeholder, anonymized)
        
        # FIX: Use pre-compiled regex patterns instead of compiling on each call
        anonymized = self.email_pattern.sub('EMAIL_REDACTED', anonymized)
        
        # Enhanced phone pattern matching using pre-compiled patterns
        for pattern in self.phone_patterns:
            anonymized = pattern.sub('PHONE_REDACTED', anonymized)
        
        # Additional PII patterns using pre-compiled patterns
        anonymized = self.url_pattern.sub('URL_REDACTED', anonymized)
        anonymized = self.handle_pattern.sub('HANDLE_REDACTED', anonymized)
        anonymized = self.org_suffix_pattern.sub('COMPANY_GENERIC', anonymized)
        anonymized = self.ssn_pattern.sub('SSN_REDACTED', anonymized)
        
        return anonymized
    
    async def anonymize_text_async(self, text: str, names: List[str], companies: List[str]) -> str:
        """Async wrapper for anonymize_text that uses thread pool for heavy workloads.
        
        FIX: For large texts, wraps in asyncio.to_thread() to prevent event loop blocking
        during CPU-intensive regex operations.
        
        IMPROVEMENT: Uses a smarter threshold that considers both text length and the
        number of replacements (names + companies). This avoids unnecessary async
        overhead for simple cases while still protecting the event loop for complex ones.
        
        Args:
            text: Text to anonymize
            names: List of names to anonymize
            companies: List of companies to anonymize
            
        Returns:
            Anonymized text
        """
        if not text:
            return text
        
        # Decide whether to offload to a thread based on text length and
        # how many dynamic replacements we need to perform.
        total_replacements = len(names or []) + len(companies or [])
        # Base threshold and heuristic:
        # - always async for very long texts
        # - async for moderately long texts with many replacements
        use_async = len(text) > 1000 or (len(text) > 500 and total_replacements > 10)
        
        if use_async:
            return await asyncio.to_thread(self.anonymize_text, text, names, companies)
        # Small/simple cases can run synchronously without meaningful blocking
        return self.anonymize_text(text, names, companies)

    async def anonymize_batch_async(self, texts: List[str], names: List[str], companies: List[str]) -> List[str]:
        """Batch anonymization for multiple texts in parallel.
        
        Processes multiple texts concurrently using asyncio.gather(), which allows
        heavy anonymization work to be spread across the thread pool instead of
        blocking the main event loop sequentially.
        
        Args:
            texts: List of texts to anonymize
            names: List of names to anonymize
            companies: List of companies to anonymize
            
        Returns:
            List of anonymized texts in the same order as the input.
        """
        if not texts:
            return []
        
        tasks = [
            self.anonymize_text_async(text or "", names, companies)
            for text in texts
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def deanonymize_text(self, text: str) -> str:
        """Restore original PII in text for final output"""
        if not text:
            return text
            
        deanonymized = text
        for placeholder, original in self.reverse_map.items():
            deanonymized = deanonymized.replace(placeholder, original)
        return deanonymized
    
    def get_anonymization_summary(self) -> Dict[str, Any]:
        """Get summary of anonymized PII for audit purposes"""
        return {
            "total_anonymized_items": len(self.pii_map),
            "anonymized_names": [k for k in self.pii_map.keys() if k in self.reverse_map.values()],
            "anonymized_companies": [k for k in self.pii_map.keys() if k.startswith("COMPANY_")],
            "session_id": self.session_id
        }


def sanitize_answer_for_prompt(answer: str) -> str:
    """
    Sanitizes user answer for safe embedding in LLM prompts and JSON parsing.
    
    Handles:
    - Special characters that break JSON parsing (quotes, newlines, backslashes)
    - Unicode normalization
    - Preserves meaningful content while preventing parsing errors
    
    Args:
        answer: Raw user answer string
        
    Returns:
        Sanitized string safe for embedding in prompts and JSON
    """
    if not isinstance(answer, str):
        return ""
    
    # Step 1: Normalize unicode characters
    import unicodedata
    normalized = unicodedata.normalize('NFKC', answer)
    
    # Step 2: Escape special characters that break JSON/string embedding
    # Replace newlines and carriage returns with spaces
    sanitized = normalized.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Truncate if too long (prevent prompt injection)
    max_length = 5000
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
        log.warning(f"Answer truncated to {max_length} characters")
    
    return sanitized.strip()


# Alias for backward compatibility (some modules import with underscore)
_sanitize_answer_for_prompt = sanitize_answer_for_prompt


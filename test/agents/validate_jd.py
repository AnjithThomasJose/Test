import json
import re
import asyncio
import sys
import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Ensure we import from the root settings.py, not agents/settings.py
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
from settings import settings

from core.utils import (
    _mask, _sanitize_text_for_llm, _to_text, _clean_json_text, _scan_balanced_json,
    _safe_json_loads, _extract_json_from_response, _create_error_response, _validate_state_inputs,
    _generate_request_id, _calculate_processing_time
)
from core.config import get_agent_config
from core.security import (
    validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm,
    PII_PATTERNS, INJECTION_FILTERS
)



# ✅ Define structured output schema
class StructuredJDValidationOutput(BaseModel):
    is_valid_jd: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = []
    detected_sections: list[str] = []


# ✅ Create Groq model instance
def _create_groq_model():
    return ChatGroq(
        model=settings.GROQ_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0.0,
        model_kwargs={"top_p": 0.0, "max_completion_tokens": 2048}
    )


# ✅ ENHANCED SYSTEM PROMPT - LLM does all validation
SYSTEM_PROMPT = """You are an expert job description validator with balanced validation rules.

Your task is to determine whether the provided text is a VALID Job Description (JD) for a hiring position.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALID JD REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A VALID Job Description MUST contain AT LEAST 3 of these 5 core elements:

1. **Job Title/Position** → Clear role name (e.g., "Software Engineer", "Marketing Manager", "Movie Director")
2. **Experience Requirements** → Years of experience, seniority level (e.g., "5+ years", "Senior level", "Minimum 3-5 years")
3. **Skills/Technologies** → Specific technical or soft skills required (e.g., "Python", "Leadership", "Tamil fluency")
4. **Responsibilities** → What the person will do in the role (e.g., "Design systems", "Manage team", "Direct films")
5. **Company/Location Info** → Company name, industry, location, or employment type (e.g., "Chennai, Tamil Nadu", "Full-Time")

**Additional Validation Rules:**
- Must be written from HIRING perspective (employer seeking candidate)
- Must describe a FUTURE role (not past employment history)
- Must have coherent, complete sentences (not just random keyword lists)
- Must be substantial content (minimum 100 characters)
- May include compensation information, application instructions, or company details

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REJECT THESE IMMEDIATELY (return is_valid_jd: false)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**1. Resumes/CVs** 
   - Written in first-person PAST tense: "I worked at...", "I achieved...", "My experience includes..."
   - Contains personal accomplishments with dates: "Worked at Company X (2018-2022)"
   - Lists personal GPA or academic grades: "GPA: 3.8/4.0"
   - Multiple personal contact details as the applicant

**2. Bills & Invoices**
   - Invoice numbers: "Invoice #12345"
   - Account numbers: "Account Number: 987654321"
   - Payment due dates: "Amount Due: $500.00 by Jan 15"
   - Billing statements with transaction IDs

**3. Financial Documents**
   - Bank statements with transaction history
   - Tax forms (W-2, 1099, etc.)
   - Receipts with purchase details
   - Credit/debit card transactions

**4. Legal Documents**
   - Contracts with "Whereas", "Hereby", "Party of the first part"
   - Terms of Service or User Agreements
   - Legal disclaimers or liability clauses

**5. Academic Documents**
   - Grade transcripts or mark sheets
   - Degree certificates awarded to individuals
   - Course syllabi or curriculum documents

**6. Random/Broken Content**
   - Keyword spam without context
   - Corrupted or garbled text
   - Extremely short fragments (< 100 chars)
   - Excessive special characters or gibberish

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: DO NOT REJECT VALID JDs FOR THESE REASONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Valid JDs often include:**
- Compensation/Salary information: "₹15-40 Lakhs per film" or "$80K-$120K annually"
- Application instructions: "Send resume to careers@company.com"
- Company contact email: "apply at jobs@company.com"
- Employment benefits: "Medical insurance", "Travel allowance"
- Application deadlines: "Apply by November 30, 2025"
- Detailed qualifications: Specific degrees, certifications, language requirements
- Industry-specific terminology: Technical jargon, domain-specific skills

**These are NORMAL parts of job descriptions and should NOT trigger rejection.**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION LOGIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**STEP 1**: Check perspective and tense
   - Is it written from employer's perspective? ("We are seeking..." vs ❌ "I worked...")
   - Does it describe a FUTURE role? ("You will manage..." vs ❌ "I managed...")

**STEP 2**: Check for immediate rejection criteria
   - Is it a resume, bill, financial doc, legal doc, or broken content?
   - If YES → return is_valid_jd: false

**STEP 3**: Count core JD elements present (0-5)
   - Job Title: Clear position name?
   - Experience: Years or level specified?
   - Skills: Specific technical/soft skills listed?
   - Responsibilities: Role duties described?
   - Company Info: Location, company name, or job type?

**STEP 4**: Validate content quality
   - Coherent and complete sentences?
   - Substantial length (100+ characters)?
   - Logical structure and flow?

**Final Decision:**
- is_valid_jd: true → If has 3+ core elements AND written from hiring perspective
- is_valid_jd: false → If has < 3 core elements OR is non-JD document type

**When in doubt**: If it looks like a legitimate job posting with clear hiring intent, lean towards TRUE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (JSON only - no markdown, no code blocks)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "is_valid_jd": true or false,
  "confidence": 0.0 to 1.0,
  "reasons": ["specific reason 1", "specific reason 2", "reason 3"],
  "detected_sections": ["Job Title", "Requirements", "Skills", etc.]
}

**Examples of reasons for TRUE:**
- "Contains job title, experience requirements, responsibilities, skills, and location"
- "Written from hiring perspective with clear role expectations"
- "All 5 core JD elements present"

**Examples of reasons for FALSE:**
- "Written in first-person past tense - appears to be a resume"
- "Contains invoice number and billing information - this is a bill"
- "Only contains random keywords without context or structure"
"""


# ✅ Main validation agent (NO PRE-FILTER)
async def validate_jd_agent(state_or_jd_text: str | dict) -> dict:
    """
    Validate whether the provided text is a Job Description (JD) using Groq LLM only.
    Returns a dictionary containing validation results.
    
    Accepts either:
    - A string: direct JD text
    - A dict: state dict (from middleware) with jd_text and optionally job_details
    
    No pre-filtering - relies entirely on LLM judgment.
    """
    
    # 🔥 EXTRACT jd_text AND job_details FROM INPUT
    jd_text = None
    job_details = None
    
    if isinstance(state_or_jd_text, dict):
        # Extract from state dict
        jd_text = (
            state_or_jd_text.get('jd_text') or 
            state_or_jd_text.get('text') or 
            state_or_jd_text.get('content') or 
            state_or_jd_text.get('description') or 
            ""
        )
        job_details = state_or_jd_text.get('job_details') or {}
    else:
        # Direct text input
        jd_text = str(state_or_jd_text) if state_or_jd_text else ""
    
    if not jd_text:
        jd_text = ""
    
    if not isinstance(jd_text, str):
        jd_text = str(jd_text)
    
    # 🔥 ENRICH jd_text WITH job_details IF AVAILABLE (only if fields are present and non-empty)
    if job_details and isinstance(job_details, dict):
        enrichment_parts = []
        
        # Helper function to check if a value is non-empty
        def is_not_empty(value):
            if value is None:
                return False
            if isinstance(value, str):
                return value.strip() != ""
            if isinstance(value, list):
                # Filter out empty strings and return True if any non-empty items remain
                return len([s for s in value if s and str(s).strip()]) > 0
            return bool(value)
        
        # Extract structured fields from job_details (only if present)
        job_title = job_details.get("job_title") or job_details.get("jobTitle")
        if is_not_empty(job_title):
            enrichment_parts.append(f"Job Title: {job_title}")
        
        must_have_skills = job_details.get("must_have_skills") or job_details.get("mustHaveSkills") or []
        if is_not_empty(must_have_skills):
            skills_str = ", ".join([str(s).strip() for s in must_have_skills if s and str(s).strip()]) if isinstance(must_have_skills, list) else str(must_have_skills).strip()
            if skills_str:
                enrichment_parts.append(f"Must-Have Skills: {skills_str}")
        
        required_skills = job_details.get("required_skills") or job_details.get("requiredSkills") or []
        if is_not_empty(required_skills):
            skills_str = ", ".join([str(s).strip() for s in required_skills if s and str(s).strip()]) if isinstance(required_skills, list) else str(required_skills).strip()
            if skills_str:
                enrichment_parts.append(f"Required Skills: {skills_str}")
        
        good_to_have_skills = job_details.get("good_to_have_skills") or job_details.get("goodToHaveSkills") or []
        if is_not_empty(good_to_have_skills):
            skills_str = ", ".join([str(s).strip() for s in good_to_have_skills if s and str(s).strip()]) if isinstance(good_to_have_skills, list) else str(good_to_have_skills).strip()
            if skills_str:
                enrichment_parts.append(f"Preferred Skills: {skills_str}")
        
        experience = job_details.get("experience")
        if is_not_empty(experience):
            enrichment_parts.append(f"Experience Required: {experience}")
        
        education_qualification = job_details.get("education_qualification") or job_details.get("educationQualification")
        if is_not_empty(education_qualification):
            enrichment_parts.append(f"Education Qualification: {education_qualification}")
        
        location = job_details.get("location")
        if is_not_empty(location):
            enrichment_parts.append(f"Location: {location}")
        
        work_location = job_details.get("work_location") or job_details.get("workLocation")
        if is_not_empty(work_location):
            enrichment_parts.append(f"Work Location: {work_location}")
        
        company_name = job_details.get("companyName") or job_details.get("company_name")
        if is_not_empty(company_name):
            enrichment_parts.append(f"Company: {company_name}")
        
        employment_type = job_details.get("employment_type") or job_details.get("employmentType")
        if is_not_empty(employment_type):
            enrichment_parts.append(f"Employment Type: {employment_type}")
        
        salary_range = job_details.get("salary_range") or job_details.get("salaryRange")
        if is_not_empty(salary_range):
            enrichment_parts.append(f"Salary Range: {salary_range}")
        
        # Combine jd_text with enriched information (only if we have enrichment parts)
        if enrichment_parts:
            enriched_text = "\n".join(enrichment_parts)
            jd_text = f"{enriched_text}\n\n---\n\n{jd_text}" if jd_text.strip() else enriched_text
    
    # Only check for completely empty input
    if not jd_text or len(jd_text.strip()) < 10:
        return {
            "is_valid_jd": False,
            "confidence": 1.0,
            "reasons": ["Text is empty or too short (minimum 10 characters)"],
            "detected_sections": []
        }
    
    # 🔥 LLM VALIDATION (NO PRE-FILTER)
    prompt = f"""TEXT TO VALIDATE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{jd_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analyze the text above carefully and return ONLY the JSON object (no code blocks, no markdown, no explanations).

Remember: Compensation info, application emails, and benefits are NORMAL in JDs. Focus on perspective (hiring vs personal history) and core elements."""

    model = _create_groq_model()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    def _parse_validation_response(text: str) -> dict:
        """Extract JSON from LLM response; strip <function=...> prefix if present (Groq sometimes adds it)."""
        if not text or not isinstance(text, str):
            return {}
        s = text.strip()
        # Strip <function=StructuredJDValidationOutput> or similar prefix that causes tool_use_failed
        if s.startswith("<function="):
            idx = s.find(">")
            if idx >= 0:
                s = s[idx + 1 :].strip()
        parsed = _safe_json_loads(s)
        if parsed and "is_valid_jd" in parsed:
            return parsed
        # Try extracting from markdown/code block
        extracted = _extract_json_from_response(s)
        if extracted and "is_valid_jd" in extracted:
            return extracted
        return parsed or extracted

    try:
        from utils.llm_error_handler import safe_llm_call, LLMError

        config = get_agent_config("validate_jd")
        timeout = getattr(config, "timeout_seconds", 60)
        max_retries = getattr(config, "llm_retry_attempts", 3)

        # Use plain invoke (no with_structured_output) to avoid Groq tool_use_failed on <function=...> prefix
        response = await safe_llm_call(
            lambda: model.ainvoke(messages),
            timeout=timeout,
            max_retries=max_retries,
            agent_name="validate_jd",
        )

        if response is None:
            raise ValueError("LLM returned None")

        content = getattr(response, "content", None) or str(response)
        parsed = _parse_validation_response(content)
        if not parsed or "is_valid_jd" not in parsed:
            raise ValueError("LLM response did not contain valid JSON with is_valid_jd")

        # Normalize to schema (confidence 0.0–1.0)
        conf = float(parsed.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        out = StructuredJDValidationOutput(
            is_valid_jd=bool(parsed.get("is_valid_jd", False)),
            confidence=conf,
            reasons=list(parsed.get("reasons", [])) if isinstance(parsed.get("reasons"), list) else [],
            detected_sections=list(parsed.get("detected_sections", [])) if isinstance(parsed.get("detected_sections"), list) else [],
        )
        return out.model_dump()
    except LLMError as e:
        return {
            "is_valid_jd": False,
            "confidence": 0.0,
            "reasons": [f"LLM validation error: {str(e)}"],
            "detected_sections": [],
        }
    except Exception as e:
        return {
            "is_valid_jd": False,
            "confidence": 0.0,
            "reasons": [f"Validation failed: {str(e)}"],
            "detected_sections": [],
        }


# ✅ Utility wrapper for quick boolean check
async def is_valid_jd(jd_text: str | dict) -> bool:
    """Returns True if the JD is valid based on Groq validation."""
    result = await validate_jd_agent(jd_text)
    return result.get("is_valid_jd", False)


# ✅ Synchronous wrapper for non-async code
def validate_jd_sync(jd_text: str | dict) -> dict:
    """
    Synchronous version of validate_jd_agent.
    If called from async context, this will raise RuntimeError - use validate_jd_agent directly instead.
    """
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        # We're in async context - this shouldn't be called from async code
        # Caller should use validate_jd_agent directly
        raise RuntimeError("validate_jd_sync() should not be called from async context. Use validate_jd_agent() directly.")
    except RuntimeError as e:
        if "get_running_loop" in str(e):
            # No event loop, safe to use asyncio.run()
            return asyncio.run(validate_jd_agent(jd_text))
        else:
            # Re-raise if it's our custom error
            raise


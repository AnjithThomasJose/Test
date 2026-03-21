import re
import time
import logging
import asyncio
import json
import sys
import os
from typing import Dict, Any, List
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from core.utils import (
    _mask, _sanitize_text_for_llm, _to_text, _clean_json_text, _scan_balanced_json,
    _safe_json_loads, _extract_json_from_response, _create_error_response, _validate_state_inputs,
    _generate_request_id, _calculate_processing_time, run_blocking_io
)
from core.config import get_agent_config
from core.security import (
    validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm,
    PII_PATTERNS, INJECTION_FILTERS
)

# Ensure we import from the root settings.py, not agents/settings.py
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
from settings import settings
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ================= CONFIGURATION =================
MIN_RESUME_LENGTH = 80
MIN_UNIQUE_WORDS = 30
JD_SIMILARITY_THRESHOLD = 0.55
MIN_CONFIDENCE_THRESHOLD = 0.65

# Groq validation LLM: cap wall-clock per call (async layer; ChatGroq also has client timeout)
_GROQ_VALIDATE_LLM_TIMEOUT = float(os.getenv("GROQ_VALIDATE_LLM_TIMEOUT_SECONDS", "60"))


# ================= GROQ-BASED VALIDATOR =================
# Define structured output schema
class StructuredResumeValidationOutput(BaseModel):
    is_valid_resume: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = []
    detected_sections: list[str] = []


# Create Groq model instance
def _create_groq_model():
    return ChatGroq(
        model=settings.GROQ_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0.0,
        model_kwargs={"top_p": 0.0, "max_completion_tokens": 2048}
    )


# Enhanced system prompt for resume validation
RESUME_VALIDATION_SYSTEM_PROMPT = """You are an expert resume validator with balanced validation rules.

Your task is to determine whether the provided text is a VALID Resume/CV for a job candidate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALID RESUME REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A VALID Resume MUST contain AT LEAST 3 of these 5 core elements:

1. **Personal Information** → Name, contact details (email, phone, location), LinkedIn/profile links
2. **Work Experience/Employment History** → Past or current jobs with company names, job titles, dates, responsibilities
3. **Education** → Degrees, universities, graduation years, GPA (if mentioned)
4. **Skills** → Technical skills, soft skills, tools, technologies, languages, competencies
5. **Additional Sections** → Certifications, projects, achievements, awards, volunteer work, or professional summary

**Additional Validation Rules:**
- Must be written from CANDIDATE's perspective (first-person past tense or third-person)
- Must describe PAST or CURRENT employment/education (not future roles)
- Must have coherent, structured content (not just random keyword lists)
- Must be substantial content (minimum 80 characters)
- May include personal details, achievements, career objectives, or professional summaries

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REJECT THESE IMMEDIATELY (return is_valid_resume: false)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**1. Job Descriptions (JDs)**
   - Written from HIRING perspective: "We are seeking...", "Join our team...", "You will be responsible..."
   - Contains application instructions: "Apply now", "Send your resume to...", "Application deadline"
   - Describes FUTURE roles: "You will manage...", "The candidate will..."
   - Company benefits or compensation details as primary content

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
   - Grade transcripts or mark sheets (without personal info context)
   - Course syllabi or curriculum documents
   - Academic papers or research abstracts

**6. Random/Broken Content**
   - Keyword spam without context
   - Corrupted or garbled text
   - Extremely short fragments (< 80 chars)
   - Excessive special characters or gibberish

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: DO NOT REJECT VALID RESUMES FOR THESE REASONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Valid Resumes often include:**
- Professional summaries or career objectives
- Industry-specific terminology: Technical jargon, domain-specific skills
- Achievement metrics: "Increased sales by 30%", "Managed team of 10"
- Detailed work responsibilities and accomplishments
- Various resume formats: Chronological, functional, combination
- Skills sections in various formats (lists, bullet points, or narrative)
- Contact information in various formats
- Educational achievements, GPA, honors
- Certifications, licenses, professional development

**These are NORMAL parts of resumes and should NOT trigger rejection.**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIAL CASES: RECRUITER/HIRING MANAGER RESUMES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Recruiter resumes often contain phrases like:**
- "Hiring for...", "Recruitment", "Talent acquisition", "Sourcing candidates"
- "Fulfilling hiring needs", "Building teams", "Leadership hiring"
- "Candidate sourcing", "Interview coordination"

**These are VALID resume content when:**
- Written in first-person past tense ("I hired...", "I recruited...")
- Contains personal employment history with companies and dates
- Includes personal contact information
- Describes the candidate's OWN work experience (not job requirements)

**DO NOT reject recruiter resumes just because they mention hiring/recruitment activities.**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION LOGIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**STEP 1**: Check perspective and tense
   - Is it written from candidate's perspective? ("I worked...", "My experience...", "Worked at...")
   - Does it describe PAST/CURRENT roles? ("May 2024 - June 2025", "Currently working...")

**STEP 2**: Check for immediate rejection criteria
   - Is it a JD, bill, financial doc, legal doc, or broken content?
   - If YES → return is_valid_resume: false

**STEP 3**: Count core resume elements present (0-5)
   - Personal Info: Name, contact details, location?
   - Work Experience: Past/current jobs with details?
   - Education: Degrees, universities, years?
   - Skills: Technical/soft skills, tools, technologies?
   - Additional: Certifications, projects, achievements?

**STEP 4**: Validate content quality
   - Coherent and structured content?
   - Substantial length (80+ characters)?
   - Logical resume structure and flow?

**Final Decision:**
- is_valid_resume: true → If has 3+ core elements AND written from candidate perspective
- is_valid_resume: false → If has < 3 core elements OR is non-resume document type

**When in doubt**: If it looks like a legitimate resume with clear candidate information and work history, lean towards TRUE.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (JSON only - no markdown, no code blocks)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
  "is_valid_resume": true or false,
  "confidence": 0.0 to 1.0,
  "reasons": ["specific reason 1", "specific reason 2", "reason 3"],
  "detected_sections": ["Contact", "Experience", "Education", "Skills", etc.]
}

**Examples of reasons for TRUE:**
- "Contains personal info, work experience, education, skills, and certifications"
- "Written from candidate perspective with clear employment history"
- "All 5 core resume elements present with structured format"

**Examples of reasons for FALSE:**
- "Written from hiring perspective with application instructions - appears to be a job description"
- "Contains invoice number and billing information - this is a bill"
- "Only contains random keywords without context or structure"
"""


async def validate_resume_with_groq(resume_text: str | dict) -> dict:
    """
    Validate whether the provided text is a valid Resume using Groq LLM.
    Returns a dictionary containing validation results.
    
    No pre-filtering - relies entirely on LLM judgment.
    """
    
    # Input validation
    if isinstance(resume_text, dict):
        resume_text = (
            resume_text.get('text') or 
            resume_text.get('content') or 
            resume_text.get('resume_text') or 
            resume_text.get('resume_data') or 
            str(resume_text)
        )
    
    if not isinstance(resume_text, str):
        resume_text = str(resume_text)
    
    # Only check for completely empty input
    if not resume_text or len(resume_text.strip()) < 10:
        return {
            "is_valid": False,
            "confidence": 1.0,
            "reasons": ["Text is empty or too short (minimum 10 characters)"],
            "detected_sections": [],
            "method": "groq_llm",
            "processing_time_ms": 0.0
        }
    
    start_time = time.time()
    
    # LLM Validation
    prompt = f"""TEXT TO VALIDATE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{resume_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analyze the text above carefully and return ONLY the JSON object (no code blocks, no markdown, no explanations).

Remember: Recruiter resumes mentioning "hiring" or "recruitment" are VALID if they describe the candidate's own work experience. Focus on perspective (candidate's history vs employer's requirements) and core elements."""

    model = _create_groq_model()

    try:
        resp = await asyncio.wait_for(
            model.ainvoke([
                SystemMessage(content=RESUME_VALIDATION_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]),
            timeout=_GROQ_VALIDATE_LLM_TIMEOUT,
        )

        if not resp or not getattr(resp, "content", None):
            raise ValueError("Groq response is empty")

        raw = resp.content.strip()

        # Extract JSON from response
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No valid JSON found in response: {raw}")

        json_str = raw[start:end + 1]
        data = json.loads(json_str)

        # Validate with Pydantic
        result = StructuredResumeValidationOutput(**data)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Convert to expected format
        return {
            "is_valid": result.is_valid_resume,
            "confidence": result.confidence,
            "reasons": result.reasons,
            "detected_sections": result.detected_sections,
            "method": "groq_llm",
            "processing_time_ms": processing_time_ms
        }

    except Exception as e:
        log.error(f"Groq resume validation error: {e}")
        processing_time_ms = (time.time() - start_time) * 1000
        return {
            "is_valid": False,
            "confidence": 0.0,
            "reasons": [f"Validation failed: {str(e)}"],
            "detected_sections": [],
            "method": "groq_llm_error",
            "processing_time_ms": processing_time_ms
        }


# Load model once (local)
try:
    # Disable progress bars to reduce log noise
    # Note: show_progress_bar parameter removed in newer sentence-transformers versions
    MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5")  # Better quality, same 384 dimensions
    # Disable progress bars after initialization (if method exists)
    if hasattr(MODEL, 'set_show_progress_bar'):
        MODEL.set_show_progress_bar(False)
    log.info("✅ MiniLM model loaded successfully (progress bars disabled).")
except Exception as e:
    MODEL = None
    log.warning(f"⚠️ Could not load MiniLM model: {e}. Using regex-only validation.")

class RegexValidator:
    """Regex-based fast resume structure validation for all professional domains."""

    def __init__(self):
        self.section_patterns = {
            # --- Contact Information ---
            'contact': re.compile(
                r'(email|phone|mobile|contact|@|linkedin|github|portfolio|\+?\d{7,}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
                re.IGNORECASE
            ),

            # --- Education Section ---
            'education': re.compile(
                r'(education|degree|bachelor|master|university|college|institute|diploma|school|cgpa|gpa|ph\.?d)',
                re.IGNORECASE
            ),

            'experience': re.compile(
                r'(experience|employment|career|intern(ship)?|job title|designation|position|company|organization|'
                r'manager|supervisor|lead|developer|engineer|technician|doctor|nurse|recruiter|consultant|specialist|'
                r'assistant|officer|analyst|teacher|professor|trainer|operator|driver|mechanic|clerk|executive|staff)',
                re.IGNORECASE
            ),
            # --- Skills Section (Tech + Non-Tech + Domain-Specific) ---
            'skills': re.compile(
                r'(skills?|technical|soft skills|expertise|languages|tools|frameworks|'
                r'competencies|key competencies|core competencies|technical expertise|'
                r'areas of expertise|professional skills|core skills|technical proficiencies|'
                r'skill areas|expertise areas|capabilities|strengths|tech stack|'
                r'proficiencies|specializations|interpersonal skills|communication skills|'
                r'python|java|c\+\+|sql|javascript|react|node|aws|excel|erp|crm|communication|leadership|'
                r'negotiation|training|recruiting|sourcing|employee|onboarding|benefits|payroll|'
                r'pharmacy|surgery|diagnostic|treatment|patient|medical|clinical|'
                r'machine|repair|maintenance|technician|electrician|hardware|support|testing|automation)',
                re.IGNORECASE
            ),

            # --- Projects / Research / Portfolio ---
            'projects': re.compile(
                r'(projects?|research|develop(ed|ing)?|design(ed|ing)?|created|built|implemented|'
                r'deployment|case study|publication|thesis|github|portfolio|prototype|system)',
                re.IGNORECASE
            ),

            # --- Certifications / Achievements ---
            'certifications': re.compile(
                r'(certifications?|courses?|training|awards?|achievements?|honors?|recognition|workshop)',
                re.IGNORECASE
            ),

            # --- JD Reject Pattern (Job Descriptions) - Strong indicators only ---
            # Strong JD patterns that are unlikely to appear in resumes
            'jd_strong': re.compile(
                r'(we are (seeking|hiring|looking for)|job description|about the role|position overview|'
                r'join our team|our company|why join us|benefits we offer|what you will do|you will be responsible|'
                r'candidate must|ideal candidate|we offer|competitive salary|send your resume|'
                r'submit your application|application deadline|job opening|vacancy|position available|'
                r'looking to hire|seeking candidates|apply to|send applications|job posting)',
                re.IGNORECASE
            ),
            # Weak JD patterns that might appear in resumes too
            'jd_weak': re.compile(
                r'\b(responsibilities include|requirements|qualifications)\b',
                re.IGNORECASE
            ),
            # Resume-specific patterns (first person, past tense, personal accomplishments)
            # Note: "certified" may match in JDs (e.g., "board-certified"), but rule 2b handles this
            'resume_indicators': re.compile(
                r'\b(i (worked|developed|managed|designed|implemented|achieved|led|created)|'
                r'my (experience|skills|education|projects|responsibilities)|'
                r'worked at|developed|managed|designed|implemented|achieved|led|created|'
                r'gpa|cgpa|graduated|completed|certified|awarded)\b',
                re.IGNORECASE
            ),
            
            # --- Personal Contact Pattern (Resume-specific, not JD application instructions) ---
            # Matches phone numbers and emails, but we'll filter out application emails separately
            'personal_contact': re.compile(
                r'(\+?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\+?\d{10,12}|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b)',
                re.IGNORECASE
            )
        }

    def validate(self, text: str) -> tuple[bool, float, list[str]]:
        """Return tuple: (is_valid, confidence, reasons)."""
        if not text or len(text) < MIN_RESUME_LENGTH:
            return False, 0.0, ["❌ Too short to be a resume."]

        # Clean text
        text_lower = re.sub(r'\s+', ' ', text.lower())
        text_original = text  # Keep original for better pattern matching

        # BALANCED JD REJECTION: Use scoring system to avoid false positives
        # Check for resume indicators first (give resumes benefit of the doubt)
        resume_indicators_count = len(self.section_patterns['resume_indicators'].findall(text_lower))
        has_resume_indicators = resume_indicators_count >= 2  # At least 2 first-person/past-tense phrases
        
        # Check for strong JD patterns
        jd_strong_matches = len(self.section_patterns['jd_strong'].findall(text_lower))
        jd_weak_matches = len(self.section_patterns['jd_weak'].findall(text_lower))
        
        # Check for application-related phrases (very strong JD indicator)
        application_phrases = re.compile(r'\b(apply now|send your resume|submit your application|application deadline|careers@|jobs@|hiring@)\b', re.IGNORECASE)
        has_application_instructions = application_phrases.search(text_lower) is not None
        
        # Check for personal contact info (filter out application emails)
        contact_matches = self.section_patterns['personal_contact'].findall(text_original)
        application_email_patterns = re.compile(r'(careers|jobs|hiring|apply|recruitment|hr|human\.resources)@', re.IGNORECASE)
        personal_contacts = []
        for contact in contact_matches:
            contact_str = contact if isinstance(contact, str) else (contact[0] if contact else '')
            if contact_str and '@' in contact_str:
                if not application_email_patterns.search(contact_str):
                    personal_contacts.append(contact_str)
            elif contact_str:
                personal_contacts.append(contact_str)
        has_personal_contact = len(personal_contacts) > 0
        
        has_education = self.section_patterns['education'].search(text_lower)
        has_experience = self.section_patterns['experience'].search(text_lower)
        has_resume_structure = has_education or has_experience
        
        # Check for first-person pronouns (strong resume indicator)
        first_person_pattern = re.compile(r'\b(i |my |me |myself )\b', re.IGNORECASE)
        first_person_count = len(first_person_pattern.findall(text_lower))
        has_strong_resume_indicators = first_person_count >= 2  # Multiple first-person references
        
        # JD REJECTION LOGIC (balanced approach):
        # 1. If has application instructions + JD patterns → definitely reject
        if has_application_instructions and (jd_strong_matches > 0 or jd_weak_matches >= 2):
            return False, 0.0, ["❌ Contains job description patterns and application instructions."]
        
        # 2. If has strong JD patterns (3+) AND no resume indicators → reject
        if jd_strong_matches >= 3 and not has_resume_indicators:
            return False, 0.0, ["❌ Contains multiple job description patterns without resume indicators."]
        
        # 2b. If JD patterns significantly outweigh resume indicators (4+ JD vs 2 weak indicators) → reject
        # This catches cases where "certified" or other weak matches create false positives
        if jd_strong_matches >= 4 and resume_indicators_count <= 2 and not has_strong_resume_indicators:
            return False, 0.0, ["❌ Contains multiple strong job description patterns that outweigh weak resume indicators."]
        
        # 3. If has strong JD patterns (2+) AND application instructions → reject
        if jd_strong_matches >= 2 and has_application_instructions:
            return False, 0.0, ["❌ Contains job description patterns and application instructions."]
        
        # 4. If has JD patterns but also has strong resume indicators (first-person) → allow (likely a resume)
        if has_strong_resume_indicators and has_resume_structure:
            # Strong first-person indicators override JD patterns
            pass  # Continue validation
        
        # 4b. If has JD patterns but also has resume indicators → allow only if JD patterns are weak
        elif has_resume_indicators and has_resume_structure and jd_strong_matches < 3:
            # Resume indicators override weak JD patterns
            pass  # Continue validation
        
        # 5. If has JD patterns but has personal contact + resume structure → allow (likely a resume)
        elif has_personal_contact and has_resume_structure and jd_strong_matches < 3:
            # Personal contact + structure suggests resume, but only if JD patterns are weak
            pass  # Continue validation
        
        # 6. If has strong JD patterns (2+) without resume indicators or structure → reject
        elif jd_strong_matches >= 2 and not has_resume_indicators and not (has_personal_contact and has_resume_structure):
            return False, 0.0, ["❌ Looks like a job description (missing resume indicators)."]
        
        # 7. If has very strong JD patterns (3+) without strong resume indicators → reject
        elif jd_strong_matches >= 3 and not has_strong_resume_indicators:
            return False, 0.0, ["❌ Contains multiple job description patterns without strong resume indicators (first-person)."]

        unique_words = len(set(text_lower.split()))
        if unique_words < MIN_UNIQUE_WORDS:
            return False, 0.2, ["❌ Not enough unique words."]

        # Count section hits (use text_lower for consistency, but text_original for contact patterns)
        scores = {}
        for n, p in self.section_patterns.items():
            if n in ['jd_strong', 'jd_weak', 'resume_indicators']:
                continue  # Skip JD/resume indicator patterns from section scoring
            if n == 'personal_contact':
                # Use original text for contact patterns to preserve case
                scores[n] = len(p.findall(text_original))
            else:
                scores[n] = len(p.findall(text_lower))

        # Check for contact info (use personal_contact if available, otherwise fall back to contact)
        has_contact = scores.get('personal_contact', 0) >= 1 or scores.get('contact', 0) >= 1
        has_edu_or_exp = scores['education'] >= 1 or scores['experience'] >= 1
        sections_found = sum(1 for s in scores.values() if s >= 1)

        # Require core resume structure: some education/experience and at least 3 sections overall.
        if not has_edu_or_exp or sections_found < 3:
            return False, 0.45, [f"❌ Missing or weak sections: {scores}"]

        # Confidence based on richness of information. Penalize slightly if contact info is missing,
        # but still consider the resume valid to support pasted text without headers.
        confidence = min(0.9, sum(min(s / 5, 0.25) for s in scores.values()))
        reasons = [f"✅ Regex passed ({sections_found} sections found)."]
        if not has_contact:
            confidence = min(confidence, 0.7)
            reasons.append("ℹ️ No explicit contact info found; treated as valid with lower confidence.")
        return True, confidence, reasons

# ================= SEMANTIC JD DETECTOR =================
class SemanticJDDetector:
    """Detect job descriptions using semantic similarity."""

    def __init__(self):
        self.model = MODEL
        self.jd_reference = (
            "We are hiring, job description, position overview, qualifications, "
            "join our team, apply now, candidate must have, responsibilities include."
        )

    def is_job_description(self, text: str) -> tuple[bool, float]:
        """Return (is_jd, similarity_score)"""
        if not self.model:
            return False, 0.0
        embeddings = self.model.encode([text[:512], self.jd_reference], convert_to_tensor=True)
        sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        return sim > JD_SIMILARITY_THRESHOLD, sim


# ================= MAIN VALIDATOR =================
class ProductionResumeValidator:
    """Production-ready resume validator (regex + semantic)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.regex_validator = RegexValidator()
        self.semantic_detector = SemanticJDDetector()

    async def validate(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        reasons = []

        passed_regex, regex_conf, regex_reasons = self.regex_validator.validate(text)
        reasons.extend(regex_reasons)

        if not passed_regex:
            return {
                "is_valid": False,
                "confidence": regex_conf,
                "method": "regex_reject",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "reasons": reasons
            }

        is_jd, jd_score = await run_blocking_io(self.semantic_detector.is_job_description, text)
        if is_jd:
            reasons.append(f"❌ Semantic JD detector triggered (similarity={jd_score:.3f})")
            return {
                "is_valid": False,
                "confidence": jd_score,
                "method": "semantic_jd_reject",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "reasons": reasons
            }

        final_conf = regex_conf * 0.7 + (1 - jd_score) * 0.3
        is_valid = final_conf >= MIN_CONFIDENCE_THRESHOLD
        reasons.append(f"✅ Final confidence: {final_conf:.2f}")

        return {
            "is_valid": is_valid,
            "confidence": final_conf,
            "method": "regex_semantic_combined",
            "processing_time_ms": (time.time() - start_time) * 1000,
            "reasons": reasons
        }


# ================= AGENT INTERFACES =================
_validator = ProductionResumeValidator()


async def validate_resume_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Asynchronous agent-style validator using Groq LLM (Code 1 compatible)."""
    resume_text = state.get("resume_text") or state.get("resume_data")
    if not resume_text:
        return {"status": "error", "output": False, "validation_error": "Missing resume text"}

    # Use Groq-based validator
    result = await validate_resume_with_groq(resume_text)
    return {
        "status": "completed",
        "node": "is_valid_resume",
        "output": result["is_valid"],
        "is_resume": result["is_valid"],
        "resume_confidence": result["confidence"],
        "is_valid_resume": result["is_valid"],
        "validation_metadata": {
            "method": result["method"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time_ms"] / 1000,
            "reasons": result["reasons"],
            "detected_sections": result.get("detected_sections", [])
        }
    }


def is_valid_resume(state: Dict[str, Any]) -> bool:
    """Boolean compatibility function."""
    if not isinstance(state, dict):
        return False

    if "is_valid_resume" in state and isinstance(state["is_valid_resume"], bool):
        return state["is_valid_resume"]

    if "is_resume" in state and "resume_confidence" in state:
        return bool(state["is_resume"]) and float(state["resume_confidence"]) >= MIN_CONFIDENCE_THRESHOLD

    return False

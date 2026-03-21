"""
Interview Agent Utility Functions

This module contains utility functions extracted from interview_agent.py
to improve code organization and maintainability.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

# PII Redaction Patterns
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")

_PROHIBITED_TOPICS = [
    "age", "religion", "pregnant", "pregnancy", "married", "marital status",
    "sexual orientation", "gender identity", "disability", "medical condition",
]


log = logging.getLogger("main")


# ==================== Data Masking & Logging ====================

def mask_sensitive_data(data: Any, sensitive_keys: List[str] = None) -> Any:
    """
    Recursively mask sensitive data in dictionaries, lists, and strings.
    Default sensitive keys: auth_token, authorization, password, secret, api_key
    """
    if sensitive_keys is None:
        sensitive_keys = ["auth_token", "authorization", "password", "secret", "api_key", "token"]
    
    if isinstance(data, dict):
        masked_data = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
                # Mask the sensitive value
                if isinstance(value, str) and len(value) > 8:
                    masked_data[key] = f"{value[:4]}...{value[-4:]}"
                elif isinstance(value, str) and len(value) > 0:
                    masked_data[key] = "***"
                else:
                    masked_data[key] = "***"
            else:
                # Recursively mask nested structures
                masked_data[key] = mask_sensitive_data(value, sensitive_keys)
        return masked_data
    elif isinstance(data, list):
        return [mask_sensitive_data(item, sensitive_keys) for item in data]
    elif isinstance(data, str):
        # Check if string looks like a bearer token
        if data.startswith("Bearer ") and len(data) > 15:
            return f"Bearer {data[7:11]}...{data[-4:]}"
        return data
    else:
        return data


def log_event(event: str, **fields: Any) -> None:
    """Structured event logger (JSON line)"""
    try:
        # Mask sensitive data before logging
        masked_fields = mask_sensitive_data(fields)
        payload = {"event": event, **masked_fields}
        log.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        # Fallback to simple log
        log.info(f"{event} | {fields}")


# ==================== PII Anonymization ====================

def _redact_pii(text: str) -> str:
    """Redact PII from text using regex patterns"""
    if not isinstance(text, str):
        return text
    text = _EMAIL_RE.sub("[redacted_email]", text)
    text = _PHONE_RE.sub("[redacted_phone]", text)
    return text


def _policy_compliant(text: str) -> bool:
    """Check if text contains prohibited topics"""
    if not isinstance(text, str):
        return True
    t = text.lower()
    return not any(tok in t for tok in _PROHIBITED_TOPICS)


def _rewrite_to_safe(text: str) -> str:
    """Minimal rewrite: neutralize prohibited phrasing"""
    safe = text
    for tok in _PROHIBITED_TOPICS:
        safe = re.sub(re.escape(tok), "background", safe, flags=re.IGNORECASE)
    return safe


def extract_and_anonymize_candidate_info(
    resume: Dict[str, Any], 
    anonymizer: Any  # PIIAnonymizer type from interview_agent
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """
    Extract and anonymize candidate information.
    
    OPTIMIZATION: This function only uses specific resume fields:
    - Name: For candidate identification
    - RoleFit: For role suggestions
    - skills: For skill extraction
    - experience: For company and experience analysis
    
    The resume data passed to this function should be filtered to contain
    only these fields for optimal performance.
    """
    # Extract raw info - check multiple name field variations (matching resume_assembler pattern)
    name = (
        resume.get("Name") or 
        resume.get("name") or
        resume.get("fullName") or
        resume.get("full_name") or
        "Candidate"
    )
    
    # Handle nested Personal_Information structure
    if (name == "Candidate" or not name) and "Personal_Information" in resume:
        personal_info = resume.get("Personal_Information", {})
        if isinstance(personal_info, dict):
            name = (
                personal_info.get("Name") or
                personal_info.get("name") or
                personal_info.get("FullName") or
                personal_info.get("fullName") or
                personal_info.get("full_name") or
                name
            )
    # Handle RoleFit being either a list of strings or a list of dicts
    role_fit_raw = resume.get("RoleFit", [])
    roles_list: List[str] = []
    if isinstance(role_fit_raw, list):
        if role_fit_raw and isinstance(role_fit_raw[0], dict):
            for item in role_fit_raw:
                if isinstance(item, dict):
                    # Prefer common keys; fallback to any stringable value
                    candidate_role = (
                        item.get("role")
                        or item.get("title")
                        or item.get("name")
                        or None
                    )
                    if isinstance(candidate_role, str) and candidate_role.strip():
                        roles_list.append(candidate_role.strip())
        else:
            roles_list = [str(r).strip() for r in role_fit_raw if isinstance(r, (str, int, float)) and str(r).strip()]
    roles = ", ".join(roles_list) or "Software Developer"
    
    skills = []
    skills_raw = resume.get("skills") or resume.get("skill_details") or []
    if skills_raw:
        if isinstance(skills_raw, list):
            for category in skills_raw:
                if isinstance(category, dict):
                    items = category.get("Items")
                    if items:
                        skills.extend([
                            item.get("Name", item.get("name", "")) if isinstance(item, dict) else str(item)
                            for item in items
                        ])
                    # Single skill name in dict (e.g. {"Name": "Python"})
                    name = category.get("Name") or category.get("name")
                    if name and isinstance(name, str) and name.strip():
                        skills.append(name.strip())
                elif isinstance(category, str) and category.strip():
                    skills.append(category.strip())
        elif isinstance(skills_raw, dict):
            # skill_details as dict: {"CategoryName": [skills...], ...}
            for category_values in skills_raw.values():
                if isinstance(category_values, list):
                    for item in category_values:
                        if isinstance(item, dict):
                            name = item.get("Name") or item.get("name")
                            if name and isinstance(name, str) and name.strip():
                                skills.append(name.strip())
                        elif isinstance(item, str) and item.strip():
                            skills.append(item.strip())
    
    companies = []
    experience_list = resume.get("experience") or []
    if experience_list and isinstance(experience_list, list):
        for exp in experience_list:
            if isinstance(exp, dict):
                company = exp.get("company", exp.get("Company", ""))
                if company and isinstance(company, str):
                    companies.append(company)
    
    # Anonymize
    anonymized_name = anonymizer.anonymize_name(name)
    anonymized_companies = [anonymizer.anonymize_company(c) for c in companies if c]
    
    return {
        "name": anonymized_name,
        "roles": roles,
        "skills": ", ".join(skills[:10]) or "N/A",
        "companies": ", ".join(anonymized_companies[:5]) or "N/A",
        "years_experience": len(resume.get("experience", []))
    }, [name], companies


def anonymize_conversation_history(
    history: List[Dict[str, str]], 
    anonymizer: Any,  # Typically PIIAnonymizer from interview_agent
    names: List[str],
    companies: List[str]
) -> List[Dict[str, str]]:
    """Anonymize PII in conversation history.
    
    Notes:
        - This helper is intentionally synchronous and is best suited for
          small histories (e.g., for logging, debugging, or non-hot paths).
        - The main interview flow uses async-aware anonymization via the
          orchestrator and `PIIAnonymizer.anonymize_text_async` /
          `anonymize_batch_async` to avoid blocking the event loop for
          large histories.
        - Callers that need to anonymize many messages in latency-sensitive
          paths should prefer the async batch APIs instead of this helper.
    """
    anonymized: List[Dict[str, str]] = []
    for msg in history:
        content = msg.get("content") or msg.get("answer") or msg.get("question") or ""
        role = msg.get("role") or (
            "user" if msg.get("answer") else "assistant" if msg.get("question") else "user"
        )
        anonymized_content = anonymizer.anonymize_text(content, names, companies)
        anonymized.append(
            {
                "role": role,
                "content": anonymized_content,
            }
        )
    return anonymized


# ==================== History & Data Sanitization ====================

def count_questions_in_history(conversation_history: List[Dict]) -> int:
    """Count actual interview questions in conversation history"""
    count = 0
    for msg in conversation_history:
        if msg.get("role") == "assistant" and msg.get("content", "").strip().endswith("?"):
            count += 1
    return count


def get_previous_questions(conversation_history: List[Dict]) -> List[str]:
    """Extract all previous questions from conversation history"""
    questions = []
    for msg in conversation_history:
        if msg.get("role") == "assistant":
            content = msg.get("content", "").strip()
            if content.endswith("?"):
                questions.append(content)
    return questions


def sanitize_history(history: List[Dict[str, str]], names: List[str]) -> List[Dict[str, str]]:
    """Remove leading greetings/filler and name repetition from assistant messages"""
    cleaned = []
    
    # Patterns to strip at start of assistant messages
    greeting_lead_pattern = re.compile(
        r'^\s*(hi|hello|hey|welcome|greetings|good morning|good afternoon|good evening)[^.!?]*[.!?\,]?\s*',
        flags=re.IGNORECASE
    )
    filler_lead_pattern = re.compile(
        r'^\s*(great to (meet|connect|have you)|nice to (meet|connect)|it(\'s| is) (great|nice) to (meet|connect|have you)|thanks for (sharing|that)|thank you for (sharing|that))[^\n]*[.!?]?\s*',
        flags=re.IGNORECASE
    )

    # Remove candidate name tokens
    name_patterns = []
    for n in names:
        if n and len(n) > 1:
            name_patterns.append(re.compile(re.escape(n), flags=re.IGNORECASE))

    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        
        if role == "assistant" and content:
            text = content
            # Strip leading greeting and filler
            text = greeting_lead_pattern.sub("", text)
            text = filler_lead_pattern.sub("", text)
            
            # Remove occurrences of candidate names
            for pat in name_patterns:
                text = pat.sub("", text)
            
            # Collapse whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            cleaned.append({"role": role, "content": text if text else content})
        else:
            cleaned.append({"role": role, "content": content})
    
    return cleaned


def sanitize_job_details(job_details: Dict[str, Any]) -> Dict[str, Any]:
    """Remove PII from job details"""
    sanitized = {}
    safe_fields = ["title", "department", "level", "type", "requirements",
                   "responsibilities", "technologies", "skills_required"]
    
    for field in safe_fields:
        if field in job_details:
            sanitized[field] = job_details[field]
    
    return sanitized


# ==================== Candidate Info Helpers ====================

def get_candidate_name(uid_context: Dict[str, Any], candidate_info: Dict[str, str], names: List[str]) -> str:
    """Get candidate name prioritizing UID context"""
    if uid_context and uid_context.get("personal_info") and uid_context["personal_info"].get("name"):
        name = uid_context["personal_info"]["name"]
    elif names:
        name = names[0]
    else:
        name = candidate_info.get('name', 'Candidate')
    
    # Handle case where name is a list (e.g., ['Harold Das'])
    if isinstance(name, list):
        name = name[0] if name else 'Candidate'
    
    # Ensure name is a string
    return str(name) if name else 'Candidate'


def get_job_title(uid_context: Dict[str, Any], safe_job_details: Dict[str, Any], candidate_info: Dict[str, str]) -> str:
    """Get job title prioritizing UID context"""
    if safe_job_details.get('title'):
        return safe_job_details['title']
    elif uid_context and uid_context.get("experience_data") and uid_context["experience_data"].get("work_experience"):
        work_experience = uid_context["experience_data"]["work_experience"]
        if work_experience and len(work_experience) > 0:
            latest_job = work_experience[0]
            return latest_job.get("job_title", "Professional")
    elif candidate_info.get('roles'):
        return candidate_info['roles']
    else:
        return "Professional"


def get_candidate_skills(uid_context: Dict[str, Any], candidate_info: Dict[str, str]) -> str:
    """Get candidate skills prioritizing UID context"""
    ctx = uid_context or {}
    skills_block = ctx.get("skills_analysis") or {}
    skills_list = skills_block.get("skills") or []

    if isinstance(skills_list, list) and skills_list:
        skills = [
            skill.get("SkillName")
            for skill in skills_list
            if isinstance(skill, dict) and "SkillName" in skill
        ]
        skills = [s for s in skills if s]
        if skills:
            return ", ".join(skills)

    if candidate_info.get('skills'):
        return candidate_info['skills']
    return "Programming"


def get_experience_years(uid_context: Dict[str, Any], candidate_info: Dict[str, str]) -> str:
    """Get experience years prioritizing UID context"""
    if uid_context and uid_context.get("experience_data") and uid_context["experience_data"].get("work_experience"):
        work_exp = uid_context["experience_data"]["work_experience"]
        if work_exp:
            # Calculate years from work experience dates
            total_months = 0
            for exp in work_exp:
                if "dates" in exp:
                    dates = exp["dates"]
                    # Simple parsing for date ranges (e.g., "Sept 2024–Feb 2025")
                    if "–" in dates or "-" in dates:
                        try:
                            # Extract years and calculate months
                            parts = dates.replace("–", "-").split("-")
                            if len(parts) >= 2:
                                # Rough calculation: assume 6 months per job if we can't parse dates
                                total_months += 6
                        except Exception:
                            total_months += 6
            years = max(1, total_months // 12)  # At least 1 year
            return str(years)
    elif candidate_info.get('years_experience'):
        return candidate_info['years_experience']
    else:
        return "2"


# ==================== Opener & Suggestion Functions ====================

def get_personalized_opener(uid_context: Dict[str, Any], job_title: str) -> str:
    """Get personalized opener using UID context"""
    name = get_candidate_name(uid_context, {}, [])
    
    if name != "Candidate":
        if job_title and job_title != "Professional":
            return f"Hello {name}! I'm excited to learn about your background in {job_title}. Can you tell me about your experience and what drew you to this field?"
        else:
            return f"Hello {name}! I'm excited to learn about your background. Can you tell me about your experience and what drew you to this field?"
    else:
        if job_title and job_title != "Professional":
            return f"Hello! I'm excited to learn about your background in {job_title}. Can you tell me about your experience and what drew you to this field?"
        else:
            return "Hello! I'm excited to learn about your background. Can you tell me about your experience and what drew you to this field?"


def get_topic_focused_opener(uid_context: Dict[str, Any], interview_topic: str, candidate_info: Dict[str, str], structured_resume: Optional[Dict[str, Any]] = None) -> str:
    """Get topic-focused personalized opener using UID context and interview topic"""
    name = get_candidate_name(uid_context, candidate_info, [])
    
    # Extract topic-relevant skills from structured_resume (preferred) or candidate_info
    topic_skills = []
    
    # First, try to extract from structured_resume (more reliable)
    if structured_resume and isinstance(structured_resume, dict) and "skills" in structured_resume:
        all_skills = structured_resume.get("skills", [])
        if isinstance(all_skills, list):
            for skill in all_skills:
                if isinstance(skill, dict) and "SkillName" in skill:
                    skill_name = skill.get("SkillName", "").strip()
                    if skill_name:
                        # Check if skill is related to the interview topic
                        skill_lower = skill_name.lower()
                        topic_lower = interview_topic.lower()
                        
                        # Skip if skill is essentially the same as the topic (avoid redundancy)
                        if skill_lower == topic_lower:
                            continue
                        
                        # Include if skill contains topic or vice versa, but they're not identical
                        if topic_lower in skill_lower or skill_lower in topic_lower or any(
                            word in skill_lower for word in topic_lower.split() if len(word) > 3
                        ):
                            topic_skills.append(skill_name)
    
    # Fallback to candidate_info if no skills found in structured_resume
    if not topic_skills:
        skills_str = candidate_info.get('skills', '')
        if skills_str and isinstance(skills_str, str):
            # Check if any skills mention the topic
            for skill in skills_str.split(','):
                skill = skill.strip()
                skill_lower = skill.lower()
                topic_lower = interview_topic.lower()
                
                # Skip if skill is essentially the same as the topic
                if skill_lower == topic_lower:
                    continue
                
                if skill and (topic_lower in skill_lower or skill_lower in topic_lower):
                    topic_skills.append(skill)
    
    # Build the opener with topic-relevant skills if found (only if they add value)
    if name != "Candidate":
        if topic_skills:
            # Only mention specific skills if we have complementary/related skills (not redundant)
            skills_mention = f" and {', '.join(topic_skills[:2])}" if len(topic_skills) <= 2 else f" and related areas like {', '.join(topic_skills[:2])}"
            return f"Hello {name}! I'd like to learn about your experience with {interview_topic}{skills_mention}. Can you tell me about a project where you've applied {interview_topic}?"
        else:
            return f"Hello {name}! I'd like to learn about your experience with {interview_topic}. Can you tell me about a project where you've applied {interview_topic}?"
    else:
        if topic_skills:
            skills_mention = f" and {', '.join(topic_skills[:2])}" if len(topic_skills) <= 2 else f" and related areas like {', '.join(topic_skills[:2])}"
            return f"Hello! I'd like to learn about your experience with {interview_topic}{skills_mention}. Can you tell me about a project where you've applied {interview_topic}?"
        else:
            return f"Hello! I'd like to learn about your experience with {interview_topic}. Can you tell me about a project where you've applied {interview_topic}?"


def get_contextual_suggestions(uid_context: Dict[str, Any]) -> List[str]:
    """Get contextual follow-up suggestions using UID context"""
    suggestions = []
    
    interests_block = (uid_context.get("user_interests") or {}) if uid_context else {}
    if interests_block.get("user_interests"):
        interests = interests_block["user_interests"]
        if any("VLSI" in interest for interest in interests):
            suggestions.append("Ask about VLSI design experience")
        if any("embedded" in interest.lower() for interest in interests):
            suggestions.append("Explore embedded systems projects")
        if any("UI/UX" in interest for interest in interests):
            suggestions.append("Discuss UI/UX design approach")
        if any("Python" in interest for interest in interests):
            suggestions.append("Ask about Python projects")
    
    skills_block = (uid_context.get("skills_analysis") or {}) if uid_context else {}
    if skills_block.get("skills"):
        skills = [skill["SkillName"] for skill in skills_block["skills"]]
        if "Python" in skills and "Ask about Python projects" not in suggestions:
            suggestions.append("Ask about Python projects")
        if "Embedded Systems" in skills and "Explore embedded systems projects" not in suggestions:
            suggestions.append("Explore embedded systems projects")
        if "UI/UX Design" in skills and "Discuss UI/UX design approach" not in suggestions:
            suggestions.append("Discuss UI/UX design approach")
    
    return suggestions if suggestions else ["Ask about background and experience"]


def get_problem_solving_keywords(uid_context: Dict[str, Any]) -> List[str]:
    """Get problem-solving keywords using UID context"""
    base_keywords = ["problem", "solve", "challenge", "approach", "solution"]
    
    skills_block = (uid_context.get("skills_analysis") or {}) if uid_context else {}
    if skills_block.get("skills"):
        skills = [skill["SkillName"] for skill in skills_block["skills"]]
        if "Python" in skills:
            base_keywords.extend(["debug", "optimize", "algorithm", "code"])
        if "Embedded Systems" in skills:
            base_keywords.extend(["debug", "fix", "hardware", "firmware", "sensor"])
        if "UI/UX Design" in skills:
            base_keywords.extend(["design", "user experience", "interface", "usability"])
    
    return base_keywords


# ==================== Context Building ====================

def build_context_from_payload(req: Any) -> Dict[str, Any]:  # InterviewRequest type from interview_agent
    """Build context dictionary from interview request payload"""
    ctx: Dict[str, Any] = {
        "session_id": req.session_id or req.uid,
        "resume_data": None,
        "job_details": None,
        "skills_analysis": None,
        "personal_info": None,
        "experience_data": None,
        "user_interests": None,
    }
    r = req.resume or {}
    if isinstance(r, dict):
        ctx["resume_data"] = r.get("structured_resume", r)
        sr = ctx["resume_data"] or {}
        if sr:
            # Check multiple name field variations (matching extract_and_anonymize_candidate_info pattern)
            name = (
                sr.get("Name") or 
                sr.get("name") or
                sr.get("fullName") or
                sr.get("full_name")
            )
            
            # Handle nested Personal_Information structure
            if not name and "Personal_Information" in sr:
                personal_info = sr.get("Personal_Information", {})
                if isinstance(personal_info, dict):
                    name = (
                        personal_info.get("Name") or
                        personal_info.get("name") or
                        personal_info.get("FullName") or
                        personal_info.get("fullName") or
                        personal_info.get("full_name")
                    )
            contact = sr.get("ContactDetails") or sr.get("contact_details", {})
            if name or contact:
                ctx["personal_info"] = {"name": name or "Candidate", "contact_details": contact}
            exp = sr.get("experience") or sr.get("work_experience") or []
            if isinstance(exp, list):
                ctx["experience_data"] = {"work_experience": exp}
                if exp and isinstance(exp[0], dict):
                    latest = exp[0]
                    ctx["job_details"] = ctx["job_details"] or {
                        "title": latest.get("job_title", "Professional"),
                        "company": latest.get("company", ""),
                        "location": latest.get("location", ""),
                        "requirements": latest.get("responsibilities", []) or []
                    }
            skills = sr.get("skills") or []
            if isinstance(skills, list):
                ctx["skills_analysis"] = {"skills": skills}
            if sr.get("user_interests_summary"):
                ctx["user_interests_summary"] = sr["user_interests_summary"]
                ctx["user_interests"] = {"user_interests": [sr["user_interests_summary"]]}
    if req.job_details and isinstance(req.job_details, dict):
        ctx["job_details"] = req.job_details
    return ctx


def process_uid_context(context: Dict[str, Any], interview_req) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], List[str]]:
    """Process UID context and merge with interview request data"""
    
    # Use resume from context if not provided in request
    resume_data = interview_req.resume
    if not resume_data and context.get("resume_data"):
        resume_data = context["resume_data"]
    elif not resume_data:
        # Fallback to basic resume structure
        resume_data = {
            "Name": "Candidate",
            "RoleFit": ["Software Developer"],
            "skills": [{"Category": "General", "Items": [{"Name": "Programming"}]}],
            "experience": []
        }
    
    # Use job details from context if not provided in request
    job_details = interview_req.job_details
    if not job_details and context.get("job_details"):
        job_details = context["job_details"]
    elif not job_details:
        # Try to extract from experience data
        if context.get("experience_data") and "work_experience" in context["experience_data"]:
            work_experience = context["experience_data"]["work_experience"]
            if work_experience and len(work_experience) > 0:
                latest_job = work_experience[0]
                job_details = {
                    "title": latest_job.get("job_title", "Professional"),
                    "company": latest_job.get("company", ""),
                    "location": latest_job.get("location", ""),
                    "requirements": latest_job.get("responsibilities", []) or []
                }
            else:
                job_details = {"title": "Professional", "requirements": []}
        else:
            # Final fallback - use generic title
            job_details = {"title": "Professional", "requirements": []}
    
    # Extract names and companies for anonymization
    names = []
    companies = []
    
    if resume_data:
        if "Name" in resume_data:
            names.append(resume_data["Name"])
        
        if "experience" in resume_data:
            for exp in resume_data["experience"]:
                if "company" in exp:
                    companies.append(exp["company"])
    
    return resume_data, job_details, names, companies


"""
Gemini-powered Skill Proficiency Analyzer
Analyzes extracted skills and determines proficiency levels based on work experience and education.
"""

import logging
import asyncio
import os
import re
import time
import html
from typing import Dict, Any, List, Optional, Tuple
from langsmith.run_helpers import traceable
from pydantic import BaseModel, Field

# Local imports
from core.model_registry import TaskType
from models.llm_invoker import invoke_llm
from settings import settings
from agents.resume_schema import SkillItem
from core.utils import (
    _calculate_processing_time,
    _create_error_response,
    map_assessment_score_to_proficiency,
)
from core.security import validate_tenant_id
from core.logging_helpers import create_log_context, AgentLogger, log_agent_completion

log = logging.getLogger(__name__)


class SkillProficiencyAssessment(BaseModel):
    """Schema for individual skill proficiency assessment."""
    SkillName: str = Field(description="Name of the skill")
    Proficiency: str = Field(description="Proficiency level (e.g., '7/10', 'Intermediate', 'Advanced')")
    PositiveRationale: str = Field(default="", description="Why the candidate got this score (e.g. why 8/10)—evidence from experience, projects, education")
    NegativeRationale: str = Field(default="", description="Why points were deducted (e.g. why they lost the remaining 2 points)—gaps, missing evidence")
    HowToImprove: str = Field(default="", description="1-2 sentences on how the candidate can improve this skill's score (e.g. add certification, mention in project, add quantifiable impact)")


class SkillsProficiencyAnalysis(BaseModel):
    """Schema for complete skills proficiency analysis."""
    skills: List[SkillProficiencyAssessment] = Field(description="List of skills with proficiency assessments")


def _normalize_skill_name(skill_name: str) -> str:
    """
    Normalize skill name by decoding HTML entities and cleaning up.
    This ensures consistent matching between input skills and LLM responses.
    
    Args:
        skill_name: Raw skill name (may contain HTML entities like &amp;)
        
    Returns:
        Normalized skill name with HTML entities decoded
    """
    if not skill_name or not isinstance(skill_name, str):
        return ""
    
    # Decode HTML entities (e.g., &amp; -> &, &lt; -> <, etc.)
    normalized = html.unescape(skill_name)
    
    # Strip whitespace
    normalized = normalized.strip()
    
    return normalized


# Known tech/skill tokens that often appear in certification names (lowercase for matching)
_CERT_SKILL_ALLOWLIST = frozenset({
    "sql", "aws", "azure", "gcp", "java", "python", "pmp", "mongodb", "react", "node", "javascript",
    "typescript", "kubernetes", "docker", "terraform", "scrum", "agile", "salesforce", "google",
    "microsoft", "oracle", "red hat", "linux", "comptia", "cisco", "security", "data", "cloud",
    "devops", "machine learning", "ml", "ai", "analytics", "tableau", "power bi", "excel",
})
_CERT_SKILL_STOPLIST = frozenset({
    "certification", "certified", "cert", "professional", "associate", "expert", "level", "foundations",
    "administrator", "developer", "engineer", "specialist", "practitioner", "accreditation",
})


def extract_skills_from_certificate_names(certifications: List[Dict[str, Any]]) -> List[str]:
    """
    Derive skill names from certification names (e.g. 'SQL' from 'SQL Certification', 'AWS' from 'AWS Certified').
    Used when certificates are added via API so skills can be merged into the resume.
    
    Args:
        certifications: List of dicts with at least 'certification_name' (or 'name'/'title').
        
    Returns:
        List of distinct skill name strings (display form), deduped by normalized name.
    """
    import re
    seen_normalized: set = set()
    result: List[str] = []
    
    for cert in certifications or []:
        if not isinstance(cert, dict):
            continue
        name = (cert.get("certification_name") or cert.get("name") or cert.get("title") or "").strip()
        if not name:
            continue
        # Split on common delimiters and "Certified"/"Certification"
        parts = re.split(r"\s*[:\-–—()]\s*|\s+Certified\s+|\s+Certification\s*|\s+Cert\s*", name, flags=re.IGNORECASE)
        for part in parts:
            token = part.strip()
            if len(token) < 2:
                continue
            # Skip pure digits or year-like
            if token.isdigit() or (len(token) == 4 and token.isdigit()):
                continue
            normalized = _normalize_skill_name(token).lower()
            normalized_ws = " ".join(normalized.split())
            if not normalized_ws or normalized_ws in seen_normalized or normalized_ws in _CERT_SKILL_STOPLIST:
                continue
            # Keep if in allowlist or looks like a skill (title-case word or known acronym)
            if normalized_ws in _CERT_SKILL_ALLOWLIST or token[0].isupper() or (len(token) <= 5 and token.upper() == token):
                seen_normalized.add(normalized_ws)
                result.append(token)
    return result


def _normalize_skill_name_for_match(skill_name: str) -> str:
    """
    Canonical form for matching: &/and, punctuation, whitespace, case.
    Use when comparing input skill names to LLM-returned names.
    """
    if not skill_name or not isinstance(skill_name, str):
        return ""
    s = _normalize_skill_name(skill_name)
    # Treat "&" and "and" the same (LLM may return either)
    s = s.replace("&", " and ")
    s = " ".join(s.split()).strip().lower()
    return s


# Boilerplate words to strip from certification names when deriving skills
_CERT_BOILERPLATE = frozenset({
    "certificate", "certification", "certified", "professional", "program",
    "course", "diploma", "license", "licensure", "accredited", "accreditation",
    "foundation", "fundamentals", "essentials", "specialist", "associate",
    "practitioner", "expert", "master", "advanced", "beginner", "intermediate",
})


def extract_skills_from_certificate_names(certificates_list: List[Dict[str, Any]]) -> List[str]:
    """
    Derive skill names from certification names. Used when merging new certificates
    into a resume to add inferred skills (e.g. "AWS Certified Solutions Architect"
    -> "Solutions Architecture", "AWS").
    """
    seen: set = set()
    result: List[str] = []
    for cert in certificates_list or []:
        if not isinstance(cert, dict):
            continue
        name = (
            cert.get("certification_name") or cert.get("name") or cert.get("title") or ""
        ).strip()
        if not name:
            continue
        name = _normalize_skill_name(name)
        # Split by common delimiters
        parts = re.split(r"\s*[&/,\-–—|]\s*|\s+and\s+", name, flags=re.IGNORECASE)
        for part in parts:
            part = part.strip()
            if len(part) < 2:
                continue
            # Skip boilerplate words
            part_lower = part.lower()
            if part_lower in _CERT_BOILERPLATE:
                continue
            # Skip pure numbers or year-like
            if re.match(r"^[\d\s\-]+$", part):
                continue
            # Normalize for dedup
            key = part_lower
            if key in seen:
                continue
            seen.add(key)
            result.append(part)
    return result


def _find_matching_skill(skill_name: str, proficiency_map: Dict[str, Dict[str, str]]) -> Optional[str]:
    """
    Find matching skill in proficiency_map using multiple strategies.
    Handles exact matches, case-insensitive, whitespace, and &/and variations.
    """
    if not skill_name:
        return None

    skill_canonical = _normalize_skill_name_for_match(skill_name)

    # Strategy 1: Exact match (normalized)
    if skill_name in proficiency_map:
        return skill_name

    # Strategy 2: Case-insensitive match
    skill_lower = skill_name.lower()
    for key in proficiency_map.keys():
        if key.lower() == skill_lower:
            log.debug(f"✅ Found case-insensitive match: '{skill_name}' -> '{key}'")
            return key

    # Strategy 3: Whitespace-normalized match
    skill_normalized_ws = " ".join(skill_name.split())
    for key in proficiency_map.keys():
        key_normalized_ws = " ".join(key.split())
        if key_normalized_ws == skill_normalized_ws:
            log.debug(f"✅ Found whitespace-normalized match: '{skill_name}' -> '{key}'")
            return key

    # Strategy 4: Canonical match (& vs "and", case, whitespace)
    for key in proficiency_map.keys():
        if _normalize_skill_name_for_match(key) == skill_canonical:
            log.debug(f"✅ Found canonical match: '{skill_name}' -> '{key}'")
            return key

    return None


# Batch size to avoid response truncation; each batch gets one LLM call
PROFICIENCY_BATCH_SIZE = 15

# Cap parallel batch LLM calls (in addition to global LLM semaphore)
_SKILL_PROFICIENCY_MAX_PARALLEL_BATCHES = max(
    1, int(os.getenv("SKILL_PROFICIENCY_MAX_PARALLEL_BATCHES", "3"))
)

# Static system instruction for Gemini context caching (~2,500 tokens cached per resume)
# Same instructions for all proficiency calls; prompt = candidate-specific data only
_PROFICIENCY_SYSTEM_INSTRUCTION = """You are an expert career analyst. Analyze the following skills and determine proficiency levels.

CRITICAL: ALL skills listed below are EXPLICITLY MENTIONED in the candidate's resume skills section. They are NOT inferred - they are directly listed by the candidate.

IMPORTANT: Every skill above is from the candidate's skills section. Do NOT say "not explicitly mentioned" for any of these skills - they ARE explicitly listed.

SUPPORTING EVIDENCE: A skill has supporting evidence if it is mentioned or used in WORK EXPERIENCE, EDUCATION, or PROJECTS. A skill has NO supporting evidence if it appears ONLY in the skills section above and is not mentioned anywhere in work experience, education, or projects.

PLATFORM: This platform recommends assessments and courses per skill. In HowToImprove, always suggest the user take the recommended assessment for the skill and explore recommended courses in the platform (to create a link between features).

CRITICAL REQUIREMENTS - PRIORITY ORDER:
1. **BASE PROFICIENCY: ALL skills above are EXPLICITLY LISTED in the skills section**
   - Since ALL skills are from the skills section, assign 60-70/10 as BASE proficiency
   - This is the minimum for any skill in the list above

2. **REFINEMENT: Check if skill is mentioned in WORK EXPERIENCE, EDUCATION, PROJECTS, OR CERTIFICATIONS**
   - If mentioned in experience/education/projects/certifications: Increase to 70-80/10 (advanced)
   - If NOT mentioned in experience/education/projects/certifications: Keep at 60-70/10 (intermediate, explicitly listed but no supporting evidence)
   - You MUST explicitly identify which skills have NO supporting evidence (only in skills section) and list them in skills_without_supporting_evidence.

3. **PROFICIENCY ASSIGNMENT RULES:**
   - **Explicitly listed + mentioned in experience/education/projects/certifications**: 70-80/10 (advanced)
   - **Explicitly listed in skills section only (no mention in jobs, projects, or certifications)**: 60-70/10 (intermediate)
   - **Never assign below 60/10 for skills in the list above** (they are all explicitly listed)

4. **RATIONALE REQUIREMENTS (CRITICAL):**
   - MUST state that the skill is "Listed in skills section" or "Explicitly mentioned in skills section" or "Listed under [category]"
   - If mentioned in work experience: Add "Also mentioned in work experience at [company/role]"
   - If mentioned in projects: Add "Also used in project [name]" or "Mentioned in projects"
   - If ONLY in skills section (no experience/projects): Add "No supporting evidence in work experience or projects" or "Listed in skills section only"
   - NEVER say "not explicitly mentioned" - ALL skills above ARE explicitly mentioned
   - NEVER say "inferred" or "assumed" - these skills are directly listed by the candidate

5. **SPECIFIC EXAMPLES (ALL skills are from skills section):**
   - "Redux" → Proficiency: 60-70/10, Rationale: "Listed under Frontend Skills in the skills section. Indicates intermediate experience."
   - "MongoDB" → Proficiency: 60-70/10, Rationale: "Listed under Databases in the skills section. Explicitly mentioned."
   - "Next.js" → Proficiency: 60/10, Rationale: "Listed under Frontend Skills. Beginner to intermediate level."
   - "MySQL" → Proficiency: 50-60/10, Rationale: "Listed under Databases in the skills section."

6. Assign proficiency levels ONLY as X/10 format (e.g., "7/10", "9/10", "5/10")
   - Use 1-4/10 for beginner level (no evidence or minimal exposure)
   - Use 5-6/10 for intermediate level (some evidence or related experience)
   - Use 7-8/10 for advanced level (strong evidence of use)
   - Use 9-10/10 for expert level (extensive evidence and impact)

7. For each skill provide TWO rationales and how to improve:
   - **PositiveRationale**: Why they got this score (e.g. why 8/10). Cite evidence: listed in skills section, mentioned in work experience/education/projects, certifications, quantifiable impact. Keep to 1-2 sentences.
   - **NegativeRationale**: Why they lost the remaining points (e.g. why they lost 2 points). Cite gaps: no certification, not mentioned in experience, limited quantifiable impact, no proficiency level, etc. If score is 10/10, use "No significant deductions" or similar. Keep to 1-2 sentences.
   - **HowToImprove**: The platform recommends assessments and courses per skill. For HowToImprove, ALWAYS suggest the user (a) take the recommended assessment for this skill in the platform (to validate their level) and (b) explore the recommended courses (to strengthen it). You may add one short resume-level tip (e.g. add impact in experience bullets). Keep to 1-2 sentences. If score is 10/10, use "No specific improvement needed" or similar. Be actionable and link to in-system features.
   - NEVER leave Proficiency empty. Provide PositiveRationale, NegativeRationale, and HowToImprove for each skill.
   - NEVER say "not explicitly mentioned" if the skill IS listed in the skills section above

OUTPUT FORMAT:
Return a JSON object with TWO keys:
1. "skills": array of objects for EVERY skill listed above (MUST use X/10 format). Each object: {"SkillName": "...", "Proficiency": "X/10", "PositiveRationale": "...", "NegativeRationale": "...", "HowToImprove": "Suggest taking the recommended assessment for this skill and exploring recommended courses in the platform; optional one-line resume tip"}
2. "skills_without_supporting_evidence": array of EXACT skill names (strings) that appear ONLY in the skills section and are NOT mentioned or used in WORK EXPERIENCE, EDUCATION, PROJECTS, or CERTIFICATIONS. Use the exact skill names from the list above. If all skills have evidence, return an empty array [].

CRITICAL: Use the EXACT skill names as provided in the list above. Do NOT modify, abbreviate, or change the skill names in any way.

EXAMPLE 1 - Skill explicitly listed + mentioned in experience (8/10):
{"SkillName": "React.js", "Proficiency": "8/10", "PositiveRationale": "Listed under Frontend Skills. Mentioned in work experience at TechVista Labs where candidate built UI components using React.js; used in 2 projects. Demonstrates practical application.", "NegativeRationale": "No React-specific certification mentioned; limited quantifiable impact (e.g. performance metrics) in bullets.", "HowToImprove": "Take the recommended assessment for this skill in the platform to validate your level, and explore recommended courses to deepen it. Consider also quantifying impact in experience bullets (e.g. performance gains, user metrics)."}

EXAMPLE 2 - Skill explicitly listed but NOT mentioned in experience (6/10):
{"SkillName": "Redux", "Proficiency": "6/10", "PositiveRationale": "Listed under Frontend Skills in the skills section. Candidate has explicitly claimed this skill.", "NegativeRationale": "No mention in work experience or projects; no supporting evidence. Deducted points for lack of demonstrated use.", "HowToImprove": "Take the recommended assessment for Redux in the platform and explore recommended courses to strengthen it. Add Redux to a project or work experience bullet with state management or outcomes to demonstrate use."}

EXAMPLE 3 - Skill with strong evidence (7/10):
{"SkillName": "MongoDB", "Proficiency": "7/10", "PositiveRationale": "Listed under Databases. Used in projects (TaskFlow, ShopEase) where candidate built full-stack applications with MongoDB. Shows practical database experience.", "NegativeRationale": "No certification (e.g. MongoDB Certified); no quantifiable scale or performance metrics in project descriptions.", "HowToImprove": "Take the recommended assessment for this skill in the platform and explore recommended courses (e.g. MongoDB Certified Developer path). Add scale or performance metrics to project descriptions (e.g. data volume, query performance)."}
"""


def _build_proficiency_prompt(
    skills: List[Dict[str, Any]],
    work_experience: List[Dict[str, Any]],
    education: List[Dict[str, Any]],
    projects: Optional[List[Dict[str, Any]]] = None,
    certifications: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    """Build (system_instruction, prompt) for proficiency analysis.
    Supporting evidence = work experience, education, projects, and/or certifications.
    Returns (system_instruction, prompt) for Gemini context caching."""

    if projects is None:
        projects = []
    if certifications is None:
        certifications = []

    # Extract skill names
    skill_names = [skill.get("SkillName", "") for skill in skills if skill.get("SkillName")]

    # Format skills section (explicitly listed skills)
    skills_section_text = "\n".join([f"- {name}" for name in skill_names])

    # Format work experience
    work_exp_text = ""
    for exp in work_experience[:5]:  # Limit to last 5 jobs
        job_title = exp.get("job_title", "")
        company = exp.get("company", "")
        dates = exp.get("dates", "")
        resp = exp.get("responsibilities", [])
        if isinstance(resp, list):
            resp_text = "\n".join([f"- {r}" for r in resp[:3]])
        else:
            resp_text = ""

        work_exp_text += f"• {job_title} at {company} ({dates})\n{resp_text}\n\n"

    # Format education
    edu_text = ""
    for edu in education[:3]:  # Limit to last 3 education entries
        degree = edu.get("degree", "")
        major = edu.get("major", "")
        university = edu.get("university", "")
        edu_text += f"• {degree} in {major} from {university}\n"

    # Format projects (supporting evidence: skills used in projects)
    projects_text = ""
    for proj in (projects or [])[:5]:
        name = proj.get("project_name", "") or proj.get("name", "")
        desc = proj.get("description", "")
        tech = proj.get("technologies", [])
        if isinstance(tech, list):
            tech_str = ", ".join(str(t) for t in tech[:10]) if tech else ""
        else:
            tech_str = str(tech)[:200]
        projects_text += f"• {name}"
        if desc:
            projects_text += f": {desc[:300]}{'...' if len(desc) > 300 else ''}"
        if tech_str:
            projects_text += f"\n  Technologies: {tech_str}"
        projects_text += "\n\n"

    # Format certifications (supporting evidence)
    certs_text = ""
    for cert in certifications[:10]:
        if not isinstance(cert, dict):
            continue
        cname = (
            cert.get("certification_name")
            or cert.get("name")
            or cert.get("title")
            or ""
        )
        if not cname:
            continue
        org = cert.get("issuing_organization") or cert.get("organization") or ""
        year = cert.get("year") or ""
        line = f"• {cname}"
        if org:
            line += f" — {org}"
        if year:
            line += f" ({year})"
        certs_text += line + "\n"

    # Dynamic prompt = candidate-specific data only (enables Gemini to cache system_instruction)
    prompt = f"""SKILLS TO ANALYZE (ALL of these are EXPLICITLY LISTED in the resume's skills section):
{chr(10).join([f"{i+1}. {name}" for i, name in enumerate(skill_names)])}

EXPLICITLY LISTED SKILLS SECTION (for reference):
{skills_section_text}

WORK EXPERIENCE:
{work_exp_text if work_exp_text else "No work experience provided"}

EDUCATION:
{edu_text if edu_text else "No education provided"}

PROJECTS (use to find supporting evidence - skills used or mentioned here count as evidenced):
{projects_text if projects_text else "No projects provided"}

CERTIFICATIONS (skills related to these certs count as having supporting evidence):
{certs_text if certs_text else "No certifications provided"}

CRITICAL REMINDERS:
- ALL {len(skill_names)} skills above are EXPLICITLY LISTED in the skills section
- NEVER say "not explicitly mentioned" - they ARE all explicitly listed
- Minimum proficiency for any skill above: 60/10 (since they're all explicitly listed)
- If mentioned in experience/education/projects/certifications: 70-80/10
- If NOT mentioned in experience/education/projects/certifications: 60-70/10; and add that skill name to skills_without_supporting_evidence
- PositiveRationale should cite "Listed in skills section" or evidence from experience/education/projects/certifications
- Use EXACT skill names from the list above - do NOT modify them (e.g., if skill is "GST Filing & Compliance", return exactly "GST Filing & Compliance", not "GST Filing and Compliance" or any variation)
- You MUST include skills_without_supporting_evidence: list exact skill names that have no mention in work experience, education, projects, or certifications.

Begin your analysis now. Return ONLY valid JSON, no additional text. Use ONLY X/10 format for proficiency."""

    return _PROFICIENCY_SYSTEM_INSTRUCTION, prompt


def _parse_proficiency_response(response_text: str):
    """
    Parse LLM response into proficiency_map and skills_without_supporting_evidence.
    Returns (proficiency_map, skills_without_supporting_evidence).
    Raises ValueError if response is not valid JSON or missing required structure.
    """
    from core.utils import _extract_json_from_response
    json_data = _extract_json_from_response(response_text)
    if isinstance(json_data, list):
        json_data = {"skills": json_data, "skills_without_supporting_evidence": []}
    elif isinstance(json_data, dict) and "skills" not in json_data and json_data.get("SkillName") is not None:
        json_data = {"skills": [json_data], "skills_without_supporting_evidence": []}
    if not json_data or "skills" not in json_data:
        raise ValueError("Failed to extract skills from JSON response - LLM must return valid JSON")
    raw_no_evidence = json_data.get("skills_without_supporting_evidence")
    no_evidence: List[str] = []
    if isinstance(raw_no_evidence, list):
        for item in raw_no_evidence:
            if isinstance(item, str) and item.strip():
                no_evidence.append(_normalize_skill_name(item.strip()))
    skills_list = json_data.get("skills", []) or []
    proficiency_map = {}
    for skill_data in skills_list:
        if not isinstance(skill_data, dict):
            continue
        normalized_name = _normalize_skill_name(skill_data.get("SkillName", ""))
        if not normalized_name:
            continue
        proficiency_map[normalized_name] = {
            "Proficiency": skill_data.get("Proficiency", ""),
            "PositiveRationale": (skill_data.get("PositiveRationale") or "").strip()[:500],
            "NegativeRationale": (skill_data.get("NegativeRationale") or "").strip()[:500],
            "HowToImprove": (skill_data.get("HowToImprove") or "").strip()[:400],
        }
    return proficiency_map, no_evidence


# Default proficiency when LLM omits a skill (avoid hard failure)
DEFAULT_MISSING_PROFICIENCY = "5/10"
DEFAULT_MISSING_RATIONALE = "Not assessed (LLM did not return this skill; may be due to response length or naming mismatch)."


@traceable(name="skill_proficiency_analyzer_agent")
async def skill_proficiency_analyzer_agent(
    state: Dict[str, Any], 
    tenant_id: str = "default_tenant"
) -> Dict[str, Any]:
    """
    Analyzes skill proficiency using Gemini based on work experience and education.
    
    Args:
        state: Dictionary containing parsed resume data from groq_resume_parser
        tenant_id: Tenant identifier
        
    Returns:
        Dictionary with skills updated with proficiency and rationale
    """
    # Extract tenant_id from state if not provided as parameter
    actual_tenant_id = state.get("tenant_id", tenant_id)
    if actual_tenant_id == "default":
        actual_tenant_id = "default_tenant"
    
    start_time = time.time()
    log_context = create_log_context("skill_proficiency_analyzer", actual_tenant_id)
    
    log.info(f"🚀 Starting skill proficiency analyzer for tenant: {actual_tenant_id}")
    
    try:
        # Validate tenant
        if not validate_tenant_id(actual_tenant_id):
            log.error(f"Invalid tenant_id: {actual_tenant_id}")
            return _create_error_response("Invalid tenant identification", "skill_proficiency_analyzer")
        
        # Extract skills from state - check multiple locations
        skills = state.get("skills", [])
        
        # If no skills at top level, check parser outputs (fallback)
        if not skills:
            # Check if skills are in structured_resume
            structured_resume = state.get("structured_resume", {})
            if isinstance(structured_resume, dict):
                skills = structured_resume.get("skills", [])
            
            # If still no skills, check skills_parser output
            if not skills:
                skills_parser = state.get("skills_parser", {})
                if isinstance(skills_parser, dict):
                    skills = skills_parser.get("skills", [])
        
        # Normalize and validate skills format
        normalized_skills = []
        if skills:
            for skill in skills:
                if isinstance(skill, dict):
                    skill_name = skill.get("SkillName") or skill.get("skill") or skill.get("name") or ""
                    if skill_name and skill_name.strip():
                        # Normalize skill name (decode HTML entities)
                        normalized_name = _normalize_skill_name(skill_name)
                        if normalized_name:
                            normalized_skills.append({
                                "SkillName": normalized_name,
                                "Proficiency": skill.get("Proficiency", ""),
                                "PositiveRationale": skill.get("PositiveRationale", ""),
                                "NegativeRationale": skill.get("NegativeRationale", ""),
                                "HowToImprove": skill.get("HowToImprove", ""),
                            })
                elif isinstance(skill, str) and skill.strip():
                    # Handle string skills
                    normalized_name = _normalize_skill_name(skill)
                    if normalized_name:
                        normalized_skills.append({
                            "SkillName": normalized_name,
                            "Proficiency": "",
                            "PositiveRationale": "",
                            "NegativeRationale": "",
                            "HowToImprove": "",
                        })
        
        skills = normalized_skills
        
        # Debug logging
        log.info(f"🔍 SKILL_PROFICIENCY_ANALYZER: Found {len(skills) if skills else 0} skills in state (after normalization)")
        if skills:
            log.info(f"🔍 SKILL_PROFICIENCY_ANALYZER: First 3 skills: {[s.get('SkillName', 'N/A') for s in skills[:3]]}")
        
        # ✅ Use structured_resume from groq_resume_parser (already in state)
        structured_resume = state.get("structured_resume", {})
        if not isinstance(structured_resume, dict):
            log.warning("⚠️ structured_resume is not a dict, using work_experience, education, projects, certifications from state directly")
            work_experience = state.get("work_experience", [])
            education = state.get("education", [])
            projects = state.get("projects", [])
            certifications = state.get("certifications", [])
        else:
            # Extract from structured_resume (preferred - from groq_resume_parser)
            work_experience = structured_resume.get("work_experience") or structured_resume.get("experience") or state.get("work_experience", [])
            education = structured_resume.get("education") or state.get("education", [])
            projects = structured_resume.get("projects") or state.get("projects", [])
            certifications = structured_resume.get("certifications") or state.get("certifications", [])
            log.info(f"✅ Using structured_resume from groq_resume_parser: {len(work_experience)} work experiences, {len(education)} education entries, {len(projects)} projects, {len(certifications)} certifications")
        
        # Check if we need to analyze proficiency
        if not skills:
            log.error("No skills found in state, skipping proficiency analysis")
            log.error(f"🔍 DEBUG: State keys: {list(state.keys())}")
            log.error(f"🔍 DEBUG: structured_resume exists: {'structured_resume' in state}")
            log.error(f"🔍 DEBUG: skills_parser exists: {'skills_parser' in state}")
            return {
                "skill_proficiency_status": "skipped",
                "reason": "no_skills"
            }
        
        # Analyze ALL skills regardless of existing proficiency values
        # This ensures we get consistent X/10 format proficiency assessments
        log.info(f"📊 Analyzing proficiency for ALL {len(skills)} skills (ignoring existing proficiency values)...")
        log.info(f"📝 Using work experience: {len(work_experience)} jobs")
        log.info(f"🎓 Using education: {len(education)} entries")
        log.info(f"📁 Using projects: {len(projects)} entries")
        log.info(f"📜 Using certifications: {len(certifications)} entries")
        
        # Build prompt with system_instruction for Gemini context caching
        system_instruction, prompt = _build_proficiency_prompt(
            skills, work_experience, education, projects, certifications
        )

        # Log prompt length for debugging
        prompt_length = len(prompt)
        log.info(f"📏 Proficiency analysis prompt length: {prompt_length} characters")
        if prompt_length > 100000:
            log.warning(f"⚠️ Prompt is very long ({prompt_length} chars), may cause issues with Gemini")

        json_suffix = (
            "\n\nIMPORTANT: Return ONLY valid JSON in this exact format:\n{\n  \"skills\": [\n"
            '    {"SkillName": "...", "Proficiency": "X/10", "PositiveRationale": "...", '
            '"NegativeRationale": "...", "HowToImprove": "..."},\n    ...\n  ],\n'
            '  "skills_without_supporting_evidence": ["SkillA", "SkillB"]\n}\n\n'
            "Each skill MUST include PositiveRationale, NegativeRationale, and HowToImprove. "
            "skills_without_supporting_evidence must list EXACT skill names that appear ONLY in the skills section "
            "(not in work experience, education, projects, or certifications). Return ONLY the JSON, no additional text."
        )
        proficiency_map: Dict[str, Dict[str, str]] = {}
        skills_without_supporting_evidence: List[str] = []
        _batch_sem = asyncio.Semaphore(_SKILL_PROFICIENCY_MAX_PARALLEL_BATCHES)

        async def _run_proficiency_llm(sys_inst: str, user_prompt: str) -> str:
            return await invoke_llm(
                user_prompt + json_suffix,
                task_type=TaskType.SKILL_ANALYSIS,
                agent_name="skill_proficiency_analyzer",
                max_retries=3,
                system_instruction=sys_inst,
                max_output_tokens=12000,
                response_mime_type="application/json",
            )

        try:
            if len(skills) <= PROFICIENCY_BATCH_SIZE:
                log.info(f"📞 Calling Gemini for proficiency analysis of {len(skills)} skills (single batch)...")
                response_text = await _run_proficiency_llm(system_instruction, prompt)
                log.debug(f"📄 Raw Gemini response length: {len(response_text)} characters")
                proficiency_map, skills_without_supporting_evidence = _parse_proficiency_response(response_text)
            else:
                total_batches = (len(skills) + PROFICIENCY_BATCH_SIZE - 1) // PROFICIENCY_BATCH_SIZE
                log.info(f"📞 Batched proficiency analysis: {len(skills)} skills in {total_batches} batches (batch size {PROFICIENCY_BATCH_SIZE}) — parallel execution")

                async def _analyze_batch(start: int) -> tuple:
                    async with _batch_sem:
                        batch = skills[start : start + PROFICIENCY_BATCH_SIZE]
                        batch_sys_inst, batch_prompt = _build_proficiency_prompt(
                            batch, work_experience, education, projects, certifications
                        )
                        response_text = await _run_proficiency_llm(batch_sys_inst, batch_prompt)
                        return _parse_proficiency_response(response_text)

                results = await asyncio.gather(*[
                    _analyze_batch(start)
                    for start in range(0, len(skills), PROFICIENCY_BATCH_SIZE)
                ])

                for batch_num, (batch_map, batch_no_evidence) in enumerate(results, 1):
                    proficiency_map.update(batch_map)
                    skills_without_supporting_evidence.extend(batch_no_evidence)
                    log.info(f"✅ Batch {batch_num}/{total_batches}: got {len(batch_map)} skills")
            if skills_without_supporting_evidence:
                log.info(f"📋 Skills with no supporting evidence: {len(skills_without_supporting_evidence)} — {skills_without_supporting_evidence[:10]}{'...' if len(skills_without_supporting_evidence) > 10 else ''}")
        except Exception as llm_error:
            log.error(f"❌ LLM proficiency analysis failed: {llm_error}")
            import traceback
            log.error(f"Full error traceback: {traceback.format_exc()}")
            raise ValueError(f"LLM proficiency analysis failed - LLM must complete analysis. Error: {str(llm_error)}")

        # Update ALL skills with proficiency data; fill missing with default instead of failing
        updated_skills = []
        missing_skills = []
        for skill in skills:
            skill_name = skill.get("SkillName", "")
            matching_key = _find_matching_skill(skill_name, proficiency_map)
            if matching_key:
                prof_data = proficiency_map[matching_key]
                if not prof_data.get("Proficiency"):
                    prof_data = {**prof_data, "Proficiency": DEFAULT_MISSING_PROFICIENCY}
                updated_skill = {
                    **skill,
                    "Proficiency": prof_data["Proficiency"],
                    "PositiveRationale": prof_data.get("PositiveRationale") or "",
                    "NegativeRationale": prof_data.get("NegativeRationale") or "",
                    "HowToImprove": prof_data.get("HowToImprove") or "",
                }
                updated_skills.append(updated_skill)
            else:
                missing_skills.append(skill_name)
                log.warning(f"⚠️ Skill '{skill_name}' not found in Gemini response; assigning default proficiency")
                updated_skill = {
                    **skill,
                    "Proficiency": DEFAULT_MISSING_PROFICIENCY,
                    "PositiveRationale": "",
                    "NegativeRationale": DEFAULT_MISSING_RATIONALE,
                    "HowToImprove": "",
                }
                updated_skills.append(updated_skill)
        if missing_skills:
            log.warning(
                f"⚠️ {len(missing_skills)} skills were not in LLM response (assigned {DEFAULT_MISSING_PROFICIENCY}): {missing_skills[:5]}{'...' if len(missing_skills) > 5 else ''}"
            )
        log.info(f"✅ Successfully analyzed proficiency for {len(updated_skills)} skills")
        
        # ------------------------------------------------------------------
        # ✅ NEW: Integrate assessment results into per-skill proficiency
        # ------------------------------------------------------------------
        try:
            assessment_results = state.get("assessment_results") or {}
            if isinstance(assessment_results, dict) and assessment_results.get("total_score") is not None:
                total_score = assessment_results.get("total_score")
                # Prefer explicit assessment_topic from state, then from results
                assessment_topic = (
                    state.get("assessment_topic")
                    or assessment_results.get("assessment_topic")
                    or ""
                )
                topic_norm = _normalize_skill_name(str(assessment_topic))

                # Locate matching assessment config in plan (for difficulty + recommendation_source)
                assessment_plan = (
                    state.get("assessment_plan")
                    or state.get("prior_assessment_plan")
                    or []
                )
                plan_items = []
                if isinstance(assessment_plan, dict):
                    # Common patterns used across tools
                    plan_items = assessment_plan.get("assessment_plan") or assessment_plan.get("plan") or []
                elif isinstance(assessment_plan, list):
                    plan_items = assessment_plan

                recommendation_source = ""
                difficulty = None
                if topic_norm and plan_items:
                    for item in plan_items:
                        if not isinstance(item, dict):
                            continue
                        item_topic = _normalize_skill_name(str(item.get("topic") or ""))
                        if item_topic and item_topic.lower() == topic_norm.lower():
                            recommendation_source = str(item.get("recommendation_source") or item.get("rationale") or "")
                            difficulty = item.get("difficulty")
                            break

                source_lower = recommendation_source.lower()
                is_gap_assessment = "skill gap" in source_lower
                is_existing_skill_assessment = "existing skills" in source_lower

                # Track skills that were enriched / created from this assessment
                assessment_enriched_skills: List[str] = []

                # Map assessment score + difficulty → proficiency band
                # Rules: <60 = no change; otherwise bucketed + difficulty-aware, never downgrade
                def _apply_assessment_to_skill_list(skills_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                    if not topic_norm:
                        return skills_list
                    new_list: List[Dict[str, Any]] = []
                    found_index = -1
                    for idx, skill_item in enumerate(skills_list):
                        name_norm = _normalize_skill_name(skill_item.get("SkillName", ""))
                        if name_norm and name_norm.lower() == topic_norm.lower():
                            found_index = idx
                        new_list.append(skill_item)

                    # Derive new proficiency from assessment (may return None to indicate "no change")
                    existing_prof = None
                    if 0 <= found_index < len(new_list):
                        existing_prof = new_list[found_index].get("Proficiency")

                    new_prof = map_assessment_score_to_proficiency(
                        score=total_score,
                        difficulty=difficulty,
                        existing_proficiency=existing_prof,
                    )

                    # If score <60 or mapping says "no change", we keep original list
                    if not new_prof:
                        return new_list

                    # Case 1: skill already exists → update its proficiency
                    if 0 <= found_index < len(new_list):
                        if new_prof != existing_prof:
                            enriched_name = new_list[found_index].get("SkillName") or assessment_topic
                            if enriched_name and enriched_name not in assessment_enriched_skills:
                                assessment_enriched_skills.append(enriched_name)
                        new_list[found_index]["Proficiency"] = new_prof
                        return new_list

                    # Case 2: assessment came from a skill gap → add new skill with proficiency
                    if is_gap_assessment and topic_norm:
                        log.info(
                            f"✅ Adding new skill from gap assessment: '{assessment_topic}' "
                            f"with proficiency {new_prof} (score={total_score}, difficulty={difficulty})"
                        )
                        if assessment_topic and assessment_topic not in assessment_enriched_skills:
                            assessment_enriched_skills.append(assessment_topic)
                        new_list.append(
                            {
                                "SkillName": assessment_topic,
                                "Proficiency": new_prof,
                                "PositiveRationale": (
                                    f"Proficiency validated via assessment on '{assessment_topic}' "
                                    f"with score {total_score}."
                                ),
                                "NegativeRationale": (
                                    "Some concepts in this topic can still be strengthened; "
                                    "use additional practice and projects to deepen expertise."
                                ),
                                "HowToImprove": (
                                    "Take follow-up assessments for this topic when available and explore "
                                    "recommended courses/projects in the platform to further strengthen it."
                                ),
                            }
                        )
                        return new_list

                    # For existing-skill assessments where the skill is not yet on the list,
                    # we are conservative: do not silently add new skills unless explicitly marked as gap.
                    return new_list

                if assessment_topic:
                    before_count = len(updated_skills)
                    updated_skills = _apply_assessment_to_skill_list(updated_skills)
                    after_count = len(updated_skills)
                    log.info(
                        f"✅ Applied assessment-based proficiency update for topic='{assessment_topic}' "
                        f"(total_score={total_score}, difficulty={difficulty}, "
                        f"skills_before={before_count}, skills_after={after_count}, "
                        f"assessment_enriched_skills={assessment_enriched_skills})"
                    )
                    if assessment_enriched_skills:
                        # Expose telemetry in state so callbacks / downstream agents can surface it
                        state["assessment_enriched_skills"] = assessment_enriched_skills
        except Exception as e_assess:
            log.error(f"⚠️ Failed to apply assessment-based proficiency updates: {e_assess}", exc_info=True)
        
        # Calculate processing time
        processing_time = _calculate_processing_time(start_time)
        
        # Log success
        log_agent_completion(log_context, {
            "success": True,
            "skills_analyzed": len(updated_skills),
            "method": "gemini_proficiency_analysis"
        }, "llm", processing_time)
        
        # Ensure structured_resume keeps skills in sync for downstream agents
        if isinstance(structured_resume, dict):
            structured_resume["skills"] = updated_skills
        
        # Return both formats (same pattern as interest_filler):
        # 1. Strict JSON format for callbacks (handled by main callback flow in app.py)
        # 2. Original fields for internal pipeline use
        out = {
            # Strict JSON format for callbacks
            "status": "completed",
            "node": "skill_proficiency_analyzer",
            "output": {
                "skills": updated_skills,
                "skills_analyzed": len(updated_skills),
                "skills_without_supporting_evidence": skills_without_supporting_evidence,
                "assessment_enriched_skills": state.get("assessment_enriched_skills", []),
                "method": "gemini_proficiency_analysis"
            },
            # Original fields for internal pipeline use
            "skills": updated_skills,
            "structured_resume": structured_resume,
            "skills_without_supporting_evidence": skills_without_supporting_evidence,
            "assessment_enriched_skills": state.get("assessment_enriched_skills", []),
            "skill_proficiency_status": "success",
            "processing_time": processing_time,
            "method": "gemini_proficiency_analysis",
            "skills_analyzed": len(updated_skills)
        }
        # Preserve job_id and body for compare-candidate-job flow
        body = state.get("body") or {}
        if body.get("request_type") == "candidate_job_match" or body.get("job_id"):
            out["body"] = body
            job_id = state.get("job_id") or body.get("job_id")
            if job_id:
                out["job_id"] = job_id
        return out
            
    except asyncio.TimeoutError:
            log.error("❌ Gemini proficiency analysis timed out - LLM must complete analysis")
            log.error("❌ Cannot proceed without LLM analysis - no deterministic fallbacks")
            # Re-raise to fail the agent - LLM must do the work
            raise ValueError("Gemini proficiency analysis timed out - LLM analysis required, cannot use fallbacks")
            
    except Exception as e:
            log.error(f"❌ Gemini proficiency analysis failed: {e}")
            import traceback
            log.error(f"Full error traceback: {traceback.format_exc()}")
            log.error("❌ Cannot proceed without LLM analysis - no deterministic fallbacks")
            # Re-raise to fail the agent - LLM must do the work
            raise ValueError(f"Gemini proficiency analysis failed - LLM analysis required, cannot use fallbacks. Error: {str(e)}")
        
    except Exception as e:
        processing_time = _calculate_processing_time(start_time)
        log.error(f"❌ Skill proficiency analyzer error: {str(e)}")
        
        AgentLogger.log_error(log_context, f"Skill proficiency analysis failed: {str(e)}", processing_time)
        
        return _create_error_response(f"Skill proficiency analyzer error: {str(e)}", "skill_proficiency_analyzer")


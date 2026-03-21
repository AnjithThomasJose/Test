"""
Pydantic schemas for structured resume parsing output.
This schema matches the existing output format to ensure backward compatibility.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class ContactDetails(BaseModel):
    """Contact details schema matching existing format."""
    Email: Optional[str] = Field(default=None, description="Email address")
    Phone: Optional[str] = Field(default=None, description="Phone number")
    Location: Optional[str] = Field(default=None, description="Location/address")
    LinkedIn: Optional[str] = Field(default=None, description="LinkedIn profile URL")
    GitHub: Optional[str] = Field(default=None, description="GitHub profile URL")
    Website: Optional[str] = Field(default=None, description="Personal website URL")


class EducationItem(BaseModel):
    """Education entry schema matching existing format."""
    degree: str = Field(description="Degree name (e.g., Bachelor of Science, Master of Engineering)")
    major: str = Field(description="Major or field of study")
    university: str = Field(description="University or institution name")
    location: Optional[str] = Field(default="", description="Location (city, state/country)")
    years: Optional[str] = Field(default="", description="Education years (e.g., '2018-2022' or 'Jan 2020 - May 2024')")
    gpa: Optional[str] = Field(default="", description="GPA as written in the resume")
    capstone: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Capstone or thesis details if present (e.g., {title, outcome})")


class WorkExperienceItem(BaseModel):
    """Work experience entry schema matching existing format."""
    job_title: str = Field(description="Job title or position name")
    company: str = Field(description="Company or organization name")
    dates: Optional[str] = Field(default="", description="Employment dates (e.g., '2020-2023')")
    location: Optional[str] = Field(default="", description="Work location")
    responsibilities: Optional[List[str]] = Field(default=[], description="List of job responsibilities and achievements")


class SkillItem(BaseModel):
    """Skill entry schema matching existing format.
    
    CRITICAL: Each SkillItem represents ONE and ONLY ONE skill. 
    If multiple skills are listed together (e.g., "Python, Java, C++" or "MS Office, Excel, Word"),
    they MUST be split into separate SkillItem entries.
    """
    SkillName: str = Field(description="Name of a SINGLE skill. Each skill entry must contain exactly one skill name. Do not combine multiple skills into one entry.")
    Proficiency: Optional[str] = Field(default="5/10", description="Proficiency level (e.g., '7/10', 'Advanced')")
    PositiveRationale: Optional[str] = Field(default="", description="Why the candidate got this score (e.g. why 8/10)")
    NegativeRationale: Optional[str] = Field(default="", description="Why points were deducted (e.g. why they lost the remaining 2 points)")
    HowToImprove: Optional[str] = Field(default="", description="How the candidate can improve this skill's score (actionable 1-2 sentences)")


class CertificationItem(BaseModel):
    """Certification entry schema matching existing format."""
    certification_name: str = Field(description="Name of certification")
    issuing_organization: str = Field(description="Organization that issued the certification")
    year: Optional[str] = Field(default=None, description="Year obtained")


class ProjectItem(BaseModel):
    """Project entry schema matching existing format."""
    project_name: str = Field(description="Project name")
    description: Optional[str] = Field(default="", description="Project description")
    technologies: Optional[List[str]] = Field(default=[], description="Technologies used in the project")
    duration: Optional[str] = Field(default="", description="Project duration (e.g., '6 months', '2022-2023')")
    achievements: Optional[List[str]] = Field(default=[], description="Key achievements or outcomes")


class StructuredResumeOutput(BaseModel):
    """
    Main structured resume schema matching existing output format.
    This ensures backward compatibility with downstream agents.
    """
    name: str = Field(description="Full name of the person")
    professional_summary: Optional[str] = Field(default="", description="Professional summary/objective/profile text as written")
    contact_details: Optional[ContactDetails] = Field(default=None, description="Contact information")
    
    # Education entries
    education: Optional[List[EducationItem]] = Field(default=[], description="List of education entries")
    
    # Work experience entries
    work_experience: Optional[List[WorkExperienceItem]] = Field(default=[], description="List of work experience entries")
    
    # Skills
    skills: Optional[List[SkillItem]] = Field(default=[], description="List of skills")
    
    # Certifications
    certifications: Optional[List[CertificationItem]] = Field(default=[], description="List of certifications")
    
    # Projects
    projects: Optional[List[ProjectItem]] = Field(default=[], description="List of projects")
    
    # Extras (awards, volunteer work, etc.)
    extras: Optional[List[Dict[str, Any]]] = Field(default=[], description="Additional items like awards or volunteer work")
    
    # Professional affiliations / memberships
    professional_affiliations: Optional[List[Dict[str, Any]]] = Field(default=[], description="Professional affiliations or memberships with organization, membership, years")
    
    # Total experience in years
    # total_experience_years: float = Field(default=0.0, description="Total years of professional experience")
    total_experience_years: str = Field(
        default="0 months",
        description="Total professional experience (e.g. '11 months', '2 years 6 months')"
    )
    
    # Candidate domains (optional, for domain-based retrieval)
    candidate_domains: Optional[List[str]] = Field(default=[], description="Top 3 professional domains from DOMAIN_TAXONOMY")


# --- Relaxed schema (maximally tolerant) + coercion helper ---

class RelaxedContactDetails(BaseModel):
    Email: Optional[str] = None
    Phone: Optional[str] = None
    Location: Optional[str] = None
    LinkedIn: Optional[str] = None
    GitHub: Optional[str] = None
    Website: Optional[str] = None

    class Config:
        extra = "allow"


class RelaxedEducationItem(BaseModel):
    degree: Optional[str] = None
    major: Optional[str] = None
    university: Optional[str] = None
    location: Optional[str] = ""
    years: Optional[str] = ""
    gpa: Optional[str] = ""
    capstone: Optional[Dict[str, Any]] = None

    @field_validator('capstone', mode='before')
    @classmethod
    def normalize_capstone(cls, v):
        """Convert empty strings to empty dict for capstone field."""
        if v == "" or v is None:
            return {}
        if isinstance(v, dict):
            return v
        # Fallback: if it's not a dict and not empty string/None, return empty dict
        return {}

    class Config:
        extra = "allow"


class RelaxedWorkExperienceItem(BaseModel):
    job_title: Optional[str] = None
    company: Optional[str] = None
    dates: Optional[str] = ""
    location: Optional[str] = ""
    responsibilities: Optional[List[str]] = None

    class Config:
        extra = "allow"


class RelaxedSkillItem(BaseModel):
    """Relaxed skill item - each entry must still represent ONE skill only."""
    SkillName: Optional[str] = None
    Proficiency: Optional[str] = "5/10"
    PositiveRationale: Optional[str] = ""
    NegativeRationale: Optional[str] = ""
    HowToImprove: Optional[str] = ""

    class Config:
        extra = "allow"


class RelaxedCertificationItem(BaseModel):
    certification_name: Optional[str] = None
    issuing_organization: Optional[str] = None
    year: Optional[str] = None

    class Config:
        extra = "allow"


class RelaxedProjectItem(BaseModel):
    project_name: Optional[str] = None
    description: Optional[str] = ""
    technologies: Optional[List[str]] = None
    duration: Optional[str] = ""
    achievements: Optional[List[str]] = None

    class Config:
        extra = "allow"


class RelaxedResumeOutput(BaseModel):
    name: str
    professional_summary: Optional[str] = ""
    contact_details: Optional[RelaxedContactDetails] = None
    education: Optional[List[RelaxedEducationItem]] = None
    work_experience: Optional[List[RelaxedWorkExperienceItem]] = None
    skills: Optional[List[RelaxedSkillItem]] = None
    certifications: Optional[List[RelaxedCertificationItem]] = None
    projects: Optional[List[RelaxedProjectItem]] = None
    extras: Optional[List[Dict[str, Any]]] = None
    professional_affiliations: Optional[List[Dict[str, Any]]] = None
    # total_experience_years: Optional[float] = 0.0
    total_experience_years: Optional[str] = Field(
        default="0 months",
        description="Total professional experience (e.g. '11 months', '2 years 6 months')"
    )
    candidate_domains: Optional[List[str]] = None

    class Config:
        extra = "allow"
        # added
        populate_by_name = True 


def coerce_relaxed_to_structured(relaxed: RelaxedResumeOutput) -> StructuredResumeOutput:
    def _ensure_list(value):
        return value or []

    def _ensure_str(value):
        return value if isinstance(value, str) else ""

    def _ensure_opt_str(value):
        return value if (value is None or isinstance(value, str)) else ""

    # Contact details
    cd = relaxed.contact_details
    contact = (
        ContactDetails(
            Email=_ensure_opt_str(cd.Email),
            Phone=_ensure_opt_str(cd.Phone),
            Location=_ensure_opt_str(cd.Location),
            LinkedIn=_ensure_opt_str(cd.LinkedIn),
            GitHub=_ensure_opt_str(cd.GitHub),
            Website=_ensure_opt_str(cd.Website),
        )
        if cd
        else None
    )

    # Education
    education_items: List[EducationItem] = []
    for e in _ensure_list(relaxed.education):
        education_items.append(
            EducationItem(
                degree=_ensure_str(e.degree),
                major=_ensure_str(e.major),
                university=_ensure_str(e.university),
                location=_ensure_str(e.location),
                years=_ensure_str(e.years),
                gpa=_ensure_str(e.gpa),
                capstone=e.capstone or {},
            )
        )

    # Work experience
    work_items: List[WorkExperienceItem] = []
    for w in _ensure_list(relaxed.work_experience):
        work_items.append(
            WorkExperienceItem(
                job_title=_ensure_str(w.job_title),
                company=_ensure_str(w.company),
                dates=_ensure_str(w.dates),
                location=_ensure_str(w.location),
                responsibilities=_ensure_list(w.responsibilities),
            )
        )

    # Skills
    skill_items: List[SkillItem] = []
    for s in _ensure_list(relaxed.skills):
        skill_items.append(
            SkillItem(
                SkillName=_ensure_str(s.SkillName),
                Proficiency=_ensure_str(s.Proficiency or "5/10"),
                PositiveRationale=_ensure_str(getattr(s, "PositiveRationale", None) or ""),
                NegativeRationale=_ensure_str(getattr(s, "NegativeRationale", None) or ""),
                HowToImprove=_ensure_str(getattr(s, "HowToImprove", None) or ""),
            )
        )

    # Certifications
    cert_items: List[CertificationItem] = []
    for c in _ensure_list(relaxed.certifications):
        cert_items.append(
            CertificationItem(
                certification_name=_ensure_str(c.certification_name),
                issuing_organization=_ensure_str(c.issuing_organization),
                year=c.year if (c.year is None or isinstance(c.year, str)) else None,
            )
        )

    # Projects
    project_items: List[ProjectItem] = []
    for p in _ensure_list(relaxed.projects):
        project_items.append(
            ProjectItem(
                project_name=_ensure_str(p.project_name),
                description=_ensure_str(p.description),
                technologies=_ensure_list(p.technologies),
                duration=_ensure_str(p.duration),
                achievements=_ensure_list(p.achievements),
            )
        )

    return StructuredResumeOutput(
        name=relaxed.name,
        professional_summary=_ensure_str(relaxed.professional_summary),
        contact_details=contact,
        education=education_items,
        work_experience=work_items,
        skills=skill_items,
        certifications=cert_items,
        projects=project_items,
        extras=_ensure_list(relaxed.extras),
        professional_affiliations=_ensure_list(relaxed.professional_affiliations),
        # total_experience_years=float(relaxed.total_experience_years or 0.0),
        total_experience_years=relaxed.total_experience_years or "0 months",
        candidate_domains=_ensure_list(relaxed.candidate_domains),
    )


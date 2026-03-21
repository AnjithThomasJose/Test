from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ===============================
# ✅ Strict Schema
# ===============================

class CompanyDetails(BaseModel):
    name: str = Field(description="Company name")
    industry: Optional[str] = ""
    size: Optional[str] = ""
    website: Optional[str] = ""
    description: Optional[str] = ""


class LocationDetails(BaseModel):
    city: Optional[str] = ""
    state: Optional[str] = ""
    country: Optional[str] = ""
    is_remote: Optional[bool] = False
    work_mode: Optional[str] = ""


class CompensationDetails(BaseModel):
    min_salary: Optional[str] = ""
    max_salary: Optional[str] = ""
    currency: Optional[str] = ""
    period: Optional[str] = ""
    additional_benefits: Optional[str] = ""


class ResponsibilityItem(BaseModel):
    description: str
    category: Optional[str] = ""


class RequirementItem(BaseModel):
    description: str
    type: Optional[str] = "required"


class BenefitItem(BaseModel):
    benefit_name: str
    description: Optional[str] = ""
    category: Optional[str] = ""


class StructuredJDOutput(BaseModel):
    """Main strict schema."""
    jobTitle: str
    company: str
    location: str
    jobType: str = "Full-Time"
    workMode: Optional[str] = ""
    experience: str = ""
    experienceLevel: Optional[str] = ""
    educationRequired: Optional[str] = ""
    salary: str = ""
    requiredSkills: List[str] = []
    preferredSkills: Optional[List[str]] = []
    jobDomains: Optional[List[str]] = []
    softSkills: Optional[List[str]] = []
    responsibilities: Optional[List[str]] = []
    fullJobDescription: str
    fileUrl: Optional[str] = ""
    companyDetails: Optional[CompanyDetails] = None
    locationDetails: Optional[LocationDetails] = None
    compensationDetails: Optional[CompensationDetails] = None
    benefits: Optional[List[str]] = []
    certifications: Optional[List[str]] = []
    postedDate: Optional[str] = ""
    extras: Optional[Dict[str, Any]] = {}

    class Config:
        extra = "allow"


# ===============================
# ✅ Relaxed Schema (for tolerant parsing)
# ===============================

class RelaxedCompanyDetails(BaseModel):
    name: Optional[str] = None
    industry: Optional[str] = ""
    size: Optional[str] = ""
    website: Optional[str] = ""
    description: Optional[str] = ""

    class Config:
        extra = "allow"


class RelaxedLocationDetails(BaseModel):
    city: Optional[str] = ""
    state: Optional[str] = ""
    country: Optional[str] = ""
    is_remote: Optional[bool] = False
    work_mode: Optional[str] = ""

    class Config:
        extra = "allow"


class RelaxedCompensationDetails(BaseModel):
    min_salary: Optional[str] = ""
    max_salary: Optional[str] = ""
    currency: Optional[str] = ""
    period: Optional[str] = ""
    additional_benefits: Optional[str] = ""

    class Config:
        extra = "allow"


class RelaxedResponsibilityItem(BaseModel):
    description: Optional[str] = None
    category: Optional[str] = ""

    class Config:
        extra = "allow"


class RelaxedRequirementItem(BaseModel):
    description: Optional[str] = None
    type: Optional[str] = "required"

    class Config:
        extra = "allow"


class RelaxedBenefitItem(BaseModel):
    benefit_name: Optional[str] = None
    description: Optional[str] = ""
    category: Optional[str] = ""

    class Config:
        extra = "allow"


class RelaxedJDOutput(BaseModel):
    """Relaxed (maximally tolerant) schema for parsing."""
    jobTitle: str
    company: str
    location: str
    fullJobDescription: str
    jobType: Optional[str] = "Full-Time"
    workMode: Optional[str] = ""
    experience: Optional[str] = ""
    experienceLevel: Optional[str] = ""
    educationRequired: Optional[str] = ""
    salary: Optional[str] = ""
    requiredSkills: Optional[List[str]] = None
    preferredSkills: Optional[List[str]] = None
    jobDomains: Optional[List[str]] = None
    softSkills: Optional[List[str]] = None
    responsibilities: Optional[List[str]] = None
    companyDetails: Optional[RelaxedCompanyDetails] = None
    locationDetails: Optional[RelaxedLocationDetails] = None
    compensationDetails: Optional[RelaxedCompensationDetails] = None
    benefits: Optional[List[str]] = None
    certifications: Optional[List[str]] = None
    fileUrl: Optional[str] = ""
    postedDate: Optional[str] = ""
    extras: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


# ===============================
# ✅ Coercion Function
# ===============================

def coerce_relaxed_to_structured(relaxed: RelaxedJDOutput) -> StructuredJDOutput:
    """Convert relaxed JD into structured JD."""
    def _ensure_list(value):
        return value or []

    def _ensure_str(value):
        return value if isinstance(value, str) else ""

    company_details = (
        CompanyDetails(
            name=_ensure_str(relaxed.companyDetails.name)
            if relaxed.companyDetails and relaxed.companyDetails.name
            else relaxed.company,
            industry=_ensure_str(getattr(relaxed.companyDetails, "industry", "")),
            size=_ensure_str(getattr(relaxed.companyDetails, "size", "")),
            website=_ensure_str(getattr(relaxed.companyDetails, "website", "")),
            description=_ensure_str(getattr(relaxed.companyDetails, "description", "")),
        )
        if relaxed.companyDetails
        else None
    )

    location_details = (
        LocationDetails(
            city=_ensure_str(getattr(relaxed.locationDetails, "city", "")),
            state=_ensure_str(getattr(relaxed.locationDetails, "state", "")),
            country=_ensure_str(getattr(relaxed.locationDetails, "country", "")),
            is_remote=getattr(relaxed.locationDetails, "is_remote", False),
            work_mode=_ensure_str(getattr(relaxed.locationDetails, "work_mode", "")),
        )
        if relaxed.locationDetails
        else None
    )

    compensation_details = (
        CompensationDetails(
            min_salary=_ensure_str(getattr(relaxed.compensationDetails, "min_salary", "")),
            max_salary=_ensure_str(getattr(relaxed.compensationDetails, "max_salary", "")),
            currency=_ensure_str(getattr(relaxed.compensationDetails, "currency", "")),
            period=_ensure_str(getattr(relaxed.compensationDetails, "period", "")),
            additional_benefits=_ensure_str(getattr(relaxed.compensationDetails, "additional_benefits", "")),
        )
        if relaxed.compensationDetails
        else None
    )

    return StructuredJDOutput(
        jobTitle=_ensure_str(relaxed.jobTitle),
        company=_ensure_str(relaxed.company),
        location=_ensure_str(relaxed.location),
        jobType=_ensure_str(relaxed.jobType),
        workMode=_ensure_str(relaxed.workMode),
        experience=_ensure_str(relaxed.experience),
        experienceLevel=_ensure_str(relaxed.experienceLevel),
        educationRequired=_ensure_str(relaxed.educationRequired),
        salary=_ensure_str(relaxed.salary),
        requiredSkills=_ensure_list(relaxed.requiredSkills),
        preferredSkills=_ensure_list(relaxed.preferredSkills),
        jobDomains=_ensure_list(relaxed.jobDomains),
        softSkills=_ensure_list(relaxed.softSkills),
        responsibilities=_ensure_list(relaxed.responsibilities),
        fullJobDescription=_ensure_str(relaxed.fullJobDescription),
        fileUrl=_ensure_str(relaxed.fileUrl),
        companyDetails=company_details,
        locationDetails=location_details,
        compensationDetails=compensation_details,
        benefits=_ensure_list(relaxed.benefits),
        certifications=_ensure_list(relaxed.certifications),
        postedDate=_ensure_str(relaxed.postedDate),
        extras=relaxed.extras or {},
    )


# ===============================
# Minimal JD Extractor (Remove nested structure)
# ===============================

def extract_minimal_jd(structured_output: StructuredJDOutput) -> Dict[str, Any]:
    """
    Return flattened job_description (no nested "job_description")
    Includes jobTitle when parsed from the JD.
    """
    return {
        "company": structured_output.company,
        "experience": structured_output.experience,
        "educationRequired": structured_output.educationRequired or "",
        "fileUrl": structured_output.fileUrl,
        "fullJobDescription": structured_output.fullJobDescription,
        "jobTitle": (structured_output.jobTitle or "").strip(),
        "jobType": structured_output.jobType,
        "location": structured_output.location,
        "requiredSkills": structured_output.requiredSkills,
        "preferredSkills": structured_output.preferredSkills or [],
        "jobDomains": structured_output.jobDomains or [],
        "workMode": structured_output.workMode or "",
        "salary": structured_output.salary,
    }
"""
Recruitment Feature Data Models

Domain-specific models for candidate search and matching.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class RecruitmentQuery:
    """
    Structured recruitment query extracted from recruiter's natural language.
    
    Example:
        raw_query: "Senior Python Dev in Dubai"
        parsed_role: "Python Developer"
        parsed_location: "Dubai"
        parsed_experience: "Senior"
    """
    raw_query: str
    parsed_role: Optional[str] = None
    parsed_location: Optional[str] = None
    parsed_experience: Optional[str] = None
    parsed_skills: List[str] = field(default_factory=list)
    parsed_education: Optional[str] = None
    parsed_company_type: Optional[str] = None
    parsed_name: Optional[str] = None  # Candidate name if searching by name
    parsed_university: Optional[str] = None  # University/institution if searching by school
    
    # Pre-normalized values from combined parse+normalize LLM call
    normalized_location: Optional[str] = None
    normalized_seniority: Optional[str] = None
    normalized_education: Optional[str] = None

    # Optional limit extracted from query (e.g. "top 3", "give me 10") — overrides request top_k when set
    parsed_limit: Optional[int] = None

    # Metadata
    query_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "raw_query": self.raw_query,
            "parsed_role": self.parsed_role,
            "parsed_location": self.parsed_location,
            "parsed_experience": self.parsed_experience,
            "parsed_skills": self.parsed_skills,
            "parsed_education": self.parsed_education,
            "parsed_company_type": self.parsed_company_type,
            "parsed_name": self.parsed_name,
            "parsed_university": self.parsed_university,
            "normalized_location": self.normalized_location,
            "normalized_seniority": self.normalized_seniority,
            "normalized_education": self.normalized_education,
            "parsed_limit": self.parsed_limit,
            "query_id": self.query_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class CandidateMatch:
    """
    A single candidate match result with recruitment-specific enrichment.
    """
    candidate_id: str
    match_score: float
    candidate_name: str = ""  # Candidate's name
    candidate_skills: List[str] = field(default_factory=list)  # All candidate skills
    total_experience_years: float = 0.0
    current_location: str = ""
    seniority_level: str = ""
    education_level: str = ""
    
    # Metadata
    distance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""  # LLM-generated fit explanation (when using LLM re-rank)
    skills_matched: List[str] = field(default_factory=list)  # Skills from search context that candidate has (LLM re-rank)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "match_score": self.match_score,
            "candidate_skills": self.candidate_skills,
            "total_experience_years": round(self.total_experience_years, 1),
            "current_location": self.current_location,
            "seniority_level": self.seniority_level,
            "education_level": self.education_level,
            "distance_score": self.distance_score,
            "metadata": self.metadata,
            "rationale": self.rationale,
            "skills_matched": self.skills_matched
        }


@dataclass
class RecruitmentResult:
    """
    Complete recruitment search result with multiple candidate matches.
    """
    query: RecruitmentQuery
    matches: List[CandidateMatch] = field(default_factory=list)
    total_found: int = 0
    processing_time_ms: int = 0
    
    # Search metadata
    search_strategy: str = ""
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    avg_match_score: float = 0.0
    top_match_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "query": self.query.to_dict(),
            "matches": [m.to_dict() for m in self.matches],
            "total_found": self.total_found,
            "processing_time_ms": self.processing_time_ms,
            "search_strategy": self.search_strategy,
            "filters_applied": self.filters_applied,
            "avg_match_score": self.avg_match_score,
            "top_match_score": self.top_match_score
        }
    
    def calculate_metrics(self):
        """Calculate quality metrics from matches"""
        if self.matches:
            scores = [m.match_score for m in self.matches]
            self.avg_match_score = sum(scores) / len(scores)
            self.top_match_score = max(scores)
        else:
            self.avg_match_score = 0.0
            self.top_match_score = 0.0


# Seniority level mapping
SENIORITY_LEVELS = {
    "junior": ["junior", "jr", "entry", "entry-level", "graduate", "intern"],
    "mid": ["mid", "mid-level", "intermediate", "associate"],
    "senior": ["senior", "sr", "lead", "principal", "staff"],
    "lead": ["lead", "principal", "staff", "architect", "head"]
}

# Education level mapping
EDUCATION_LEVELS = {
    "high_school": ["high school", "secondary", "diploma"],
    "bachelors": ["bachelor", "bachelors", "bs", "ba", "bsc", "undergraduate"],
    "masters": ["master", "masters", "ms", "ma", "msc", "graduate"],
    "phd": ["phd", "doctorate", "doctoral", "ph.d"]
}

"""
Recruitment Feature

Domain-specific logic for candidate search and matching.
Converts recruiter queries into structured search requests.
"""

from .models import RecruitmentQuery, RecruitmentResult, CandidateMatch
from .query_parser import QueryParser
from .normalizer import Normalizer
from .adapter import RecruitmentAdapter

__all__ = [
    "RecruitmentQuery",
    "RecruitmentResult",
    "CandidateMatch",
    "QueryParser",
    "Normalizer",
    "RecruitmentAdapter",
]

__version__ = "1.0.0"

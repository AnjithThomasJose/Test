"""
Core Search Service - Generic Infrastructure Layer

This module provides a domain-agnostic search infrastructure that can be used
by any feature (recruitment, courses, etc.) without knowing about domain-specific
concepts like "resumes" or "jobs".

Key Components:
- search_gateway: Public interface for search operations
- models: Generic request/response data models

Design Principles:
- Domain Agnostic: Knows nothing about resumes, jobs, or courses
- Reusable: Can be used by multiple features without modification
- Simple: Thin wrapper around existing ChromaDB infrastructure
"""

from .models import (
    StandardSearchRequest,
    StandardSearchResponse,
    SearchResult,
    SearchFilter,
    SearchMetadata
)

from .search_gateway import SearchGateway

__all__ = [
    "StandardSearchRequest",
    "StandardSearchResponse",
    "SearchResult",
    "SearchFilter",
    "SearchMetadata",
    "SearchGateway"
]

__version__ = "1.0.0"

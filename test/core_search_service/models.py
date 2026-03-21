"""
Core Search Service - Data Models

Generic request/response models for the search infrastructure.
These models are domain-agnostic and can be used by any feature.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class SearchFilterOperator(str, Enum):
    """Filter operators for metadata filtering"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "nin"
    CONTAINS = "contains"


@dataclass
class SearchFilter:
    """
    Metadata filter for search operations.
    
    Examples:
        - SearchFilter(field="location", operator="eq", value="Dubai")
        - SearchFilter(field="experience_years", operator="gte", value=5)
        - SearchFilter(field="skills", operator="contains", value="Python")
    """
    field: str
    operator: SearchFilterOperator
    value: Any
    
    def to_chroma_filter(self) -> Dict[str, Any]:
        """
        Convert to ChromaDB filter format.
        
        ChromaDB filter format:
        - {"field": "value"} for equality
        - {"field": {"$gte": value}} for comparisons
        - {"field": {"$in": [values]}} for IN operations
        """
        if self.operator == SearchFilterOperator.EQUALS:
            return {self.field: self.value}
        elif self.operator == SearchFilterOperator.NOT_EQUALS:
            return {self.field: {"$ne": self.value}}
        elif self.operator == SearchFilterOperator.GREATER_THAN:
            return {self.field: {"$gt": self.value}}
        elif self.operator == SearchFilterOperator.GREATER_THAN_OR_EQUAL:
            return {self.field: {"$gte": self.value}}
        elif self.operator == SearchFilterOperator.LESS_THAN:
            return {self.field: {"$lt": self.value}}
        elif self.operator == SearchFilterOperator.LESS_THAN_OR_EQUAL:
            return {self.field: {"$lte": self.value}}
        elif self.operator == SearchFilterOperator.IN:
            return {self.field: {"$in": self.value}}
        elif self.operator == SearchFilterOperator.NOT_IN:
            return {self.field: {"$nin": self.value}}
        elif self.operator == SearchFilterOperator.CONTAINS:
            # For string contains, we'll use a simple equality check
            # (ChromaDB doesn't support substring matching in metadata)
            return {self.field: self.value}
        else:
            return {self.field: self.value}


@dataclass
class SearchMetadata:
    """Additional metadata about the search operation"""
    collection_name: str
    embedding_model: str
    total_documents: int
    filters_applied: List[str] = field(default_factory=list)
    cache_hit: bool = False


@dataclass
class SearchResult:
    """
    Single search result with ID, score, and metadata.
    
    Attributes:
        id: Unique identifier for the result
        score: Similarity score (0.0-1.0, higher is better)
        distance: Distance metric from ChromaDB (lower is better)
        metadata: Additional metadata about the result
        document: Optional document content
    """
    id: str
    score: float
    distance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    document: Optional[str] = None
    
    def __post_init__(self):
        """Ensure score is between 0 and 1"""
        if self.score < 0:
            self.score = 0.0
        elif self.score > 1:
            self.score = 1.0


@dataclass
class StandardSearchRequest:
    """
    Generic search request that can be used by any feature.
    
    This is the core interface for all search operations. Features convert
    their domain-specific queries into this standard format.
    
    Attributes:
        target_collection: ChromaDB collection to search in
        search_text: Text query for semantic search
        filters: List of metadata filters to apply
        top_k: Number of results to return
        include_documents: Whether to include document content in results
        min_score: Minimum similarity score threshold (0.0-1.0)
    
    Examples:
        # Simple text search
        request = StandardSearchRequest(
            target_collection="candidates_v1",
            search_text="Senior Python Developer",
            top_k=10
        )
        
        # Search with filters
        request = StandardSearchRequest(
            target_collection="candidates_v1",
            search_text="Python Developer",
            filters=[
                SearchFilter("location", SearchFilterOperator.EQUALS, "Dubai"),
                SearchFilter("experience_years", SearchFilterOperator.GREATER_THAN_OR_EQUAL, 5)
            ],
            top_k=20
        )
    """
    target_collection: str
    search_text: str
    filters: List[SearchFilter] = field(default_factory=list)
    top_k: int = 10
    include_documents: bool = False
    min_score: float = 0.0
    query_embeddings: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate request parameters"""
        if not self.target_collection:
            raise ValueError("target_collection is required")
        if not self.search_text:
            raise ValueError("search_text is required")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.top_k > 1000:
            raise ValueError("top_k cannot exceed 1000")
        if self.min_score < 0 or self.min_score > 1:
            raise ValueError("min_score must be between 0 and 1")
    
    def to_chroma_filters(self) -> Optional[Dict[str, Any]]:
        """
        Convert filters to ChromaDB format.
        
        Returns:
            ChromaDB-compatible filter dictionary or None if no filters
        """
        if not self.filters:
            return None
        
        if len(self.filters) == 1:
            return self.filters[0].to_chroma_filter()
        
        # Multiple filters: combine with $and
        chroma_filters = [f.to_chroma_filter() for f in self.filters]
        return {"$and": chroma_filters}


@dataclass
class StandardSearchResponse:
    """
    Generic search response returned by the search gateway.
    
    Attributes:
        results: List of search results
        total_found: Total number of results found
        processing_time_ms: Time taken to process the request (milliseconds)
        metadata: Additional metadata about the search operation
        request: Original request (for reference)
    """
    results: List[SearchResult]
    total_found: int
    processing_time_ms: int
    metadata: SearchMetadata
    request: StandardSearchRequest
    
    def get_top_ids(self, n: Optional[int] = None) -> List[str]:
        """Get top N result IDs"""
        n = n or len(self.results)
        return [r.id for r in self.results[:n]]
    
    def get_top_scores(self, n: Optional[int] = None) -> List[float]:
        """Get top N result scores"""
        n = n or len(self.results)
        return [r.score for r in self.results[:n]]
    
    def filter_by_score(self, min_score: float) -> List[SearchResult]:
        """Filter results by minimum score"""
        return [r for r in self.results if r.score >= min_score]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "results": [
                {
                    "id": r.id,
                    "score": r.score,
                    "distance": r.distance,
                    "metadata": r.metadata,
                    "document": r.document
                }
                for r in self.results
            ],
            "total_found": self.total_found,
            "processing_time_ms": self.processing_time_ms,
            "metadata": {
                "collection_name": self.metadata.collection_name,
                "embedding_model": self.metadata.embedding_model,
                "total_documents": self.metadata.total_documents,
                "filters_applied": self.metadata.filters_applied,
                "cache_hit": self.metadata.cache_hit
            },
            "request": {
                "target_collection": self.request.target_collection,
                "search_text": self.request.search_text,
                "top_k": self.request.top_k,
                "filters": [
                    {
                        "field": f.field,
                        "operator": f.operator.value,
                        "value": f.value
                    }
                    for f in self.request.filters
                ]
            }
        }

"""
Jobsify AI Candidate Matching Architecture - Data Models
Implements dynamic schema evolution with confidence-based processing
"""

from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import uuid


class ConfidenceLevel(str, Enum):
    """Confidence levels for facts, tags, and embeddings"""
    LOW = "low"          # 0.0 - 0.4
    MEDIUM = "medium"    # 0.4 - 0.7
    HIGH = "high"        # 0.7 - 1.0


class DataSource(str, Enum):
    """Sources of candidate data"""
    RESUME = "resume"
    CHAT_SESSION = "chat_session"
    ASSESSMENT = "assessment"
    INTERVIEW = "interview"
    LINKEDIN = "linkedin"
    PORTFOLIO = "portfolio"


class ProcessingStage(str, Enum):
    """Processing stages in the pipeline"""
    INGESTION = "ingestion"
    FACT_EXTRACTION = "fact_extraction"
    TAG_GENERATION = "tag_generation"
    EMBEDDING = "embedding"
    MERGING = "merging"
    QUALITY_GATE = "quality_gate"
    RETRIEVAL = "retrieval"


class FactType(str, Enum):
    """Types of structured facts"""
    SKILL = "skill"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    CERTIFICATION = "certification"
    PROJECT = "project"
    ACHIEVEMENT = "achievement"
    INTEREST = "interest"
    LOCATION = "location"
    LANGUAGE = "language"
    SOFT_SKILL = "soft_skill"


class TagCategory(str, Enum):
    """Categories for dynamic tags"""
    TECHNICAL_SKILL = "technical_skill"
    INDUSTRY = "industry"
    SENIORITY = "seniority"
    ROLE_TYPE = "role_type"
    LOCATION_PREFERENCE = "location_preference"
    WORK_STYLE = "work_style"
    CAREER_STAGE = "career_stage"
    SPECIALIZATION = "specialization"


class EmbeddingView(str, Enum):
    """Multi-view embedding types"""
    RESUME_CONTENT = "resume_content"
    SKILL_FOCUSED = "skill_focused"
    EXPERIENCE_FOCUSED = "experience_focused"
    ASSESSMENT_RESPONSES = "assessment_responses"
    CHAT_CONTEXT = "chat_context"
    COMPREHENSIVE = "comprehensive"


# ==================== CORE DATA MODELS ====================

class ConfidenceScore(BaseModel):
    """Confidence scoring for all data points"""
    value: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    level: ConfidenceLevel = Field(description="Confidence level category")
    reasoning: Optional[str] = Field(None, description="Explanation for confidence score")
    
    @validator('level', always=True)
    def set_confidence_level(cls, v, values):
        """Auto-set confidence level based on value"""
        if 'value' in values:
            score = values['value']
            if score >= 0.7:
                return ConfidenceLevel.HIGH
            elif score >= 0.4:
                return ConfidenceLevel.MEDIUM
            else:
                return ConfidenceLevel.LOW
        return v


class Provenance(BaseModel):
    """Data provenance tracking"""
    source: DataSource = Field(description="Original data source")
    source_id: str = Field(description="Unique identifier in source system")
    extraction_method: str = Field(description="Method used to extract this data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0", description="Data version")


class StructuredFact(BaseModel):
    """Structured fact with confidence and provenance"""
    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fact_type: FactType = Field(description="Type of fact")
    content: Dict[str, Any] = Field(description="Fact content (dynamic schema)")
    confidence: ConfidenceScore = Field(description="Confidence in this fact")
    provenance: Provenance = Field(description="Data provenance")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DynamicTag(BaseModel):
    """Dynamic tag with confidence scoring"""
    tag_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: TagCategory = Field(description="Tag category")
    value: str = Field(description="Tag value")
    confidence: ConfidenceScore = Field(description="Confidence in this tag")
    provenance: Provenance = Field(description="Data provenance")
    related_facts: List[str] = Field(default_factory=list, description="Related fact IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MultiViewEmbedding(BaseModel):
    """Multi-view embedding with confidence"""
    embedding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    view_type: EmbeddingView = Field(description="Type of embedding view")
    vector: List[float] = Field(description="Embedding vector")
    confidence: ConfidenceScore = Field(description="Confidence in embedding quality")
    source_data: Dict[str, Any] = Field(description="Source data used for embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CandidateProfile(BaseModel):
    """Core candidate profile with dynamic schema"""
    candidate_id: str = Field(description="Unique candidate identifier")
    uid: str = Field(description="User ID")
    tenant_id: str = Field(description="Tenant identifier")
    
    # Dynamic profile data
    profile_data: Dict[str, Any] = Field(default_factory=dict, description="Dynamic profile schema")
    
    # Structured components
    facts: List[StructuredFact] = Field(default_factory=list, description="Structured facts")
    tags: List[DynamicTag] = Field(default_factory=list, description="Dynamic tags")
    embeddings: List[MultiViewEmbedding] = Field(default_factory=list, description="Multi-view embeddings")
    
    # Processing metadata
    processing_stage: ProcessingStage = Field(default=ProcessingStage.INGESTION)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0")
    
    # Quality metrics
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    class Config:
        use_enum_values = True


class JobMatchResult(BaseModel):
    """Job match result with detailed scoring"""
    job_id: str = Field(description="Job identifier")
    candidate_id: str = Field(description="Candidate identifier")
    
    # Match scores
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall match score")
    skill_match_score: float = Field(ge=0.0, le=1.0, description="Skill matching score")
    experience_match_score: float = Field(ge=0.0, le=1.0, description="Experience matching score")
    education_match_score: float = Field(ge=0.0, le=1.0, description="Education matching score")
    cultural_fit_score: float = Field(ge=0.0, le=1.0, description="Cultural fit score")
    
    # Detailed matching
    matched_skills: List[str] = Field(default_factory=list)
    unmatched_skills: List[str] = Field(default_factory=list)
    matched_experience: List[Dict[str, Any]] = Field(default_factory=list)
    matched_education: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Confidence and explanation
    confidence: ConfidenceScore = Field(description="Confidence in this match")
    rationale: str = Field(description="AI-generated match explanation")
    
    # Retrieval stage information
    retrieval_stage: Literal["pre_filter", "vector_recall", "reranker"] = Field(description="Stage where match was found")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingEvent(BaseModel):
    """Event for event-driven processing"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(description="Type of processing event")
    candidate_id: str = Field(description="Candidate identifier")
    data: Dict[str, Any] = Field(description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: int = Field(default=0, description="Processing priority")
    retry_count: int = Field(default=0, description="Number of retries")
    
    class Config:
        use_enum_values = True


class QualityMetrics(BaseModel):
    """Quality metrics for data validation"""
    completeness_score: float = Field(ge=0.0, le=1.0, description="Data completeness score")
    accuracy_score: float = Field(ge=0.0, le=1.0, description="Data accuracy score")
    consistency_score: float = Field(ge=0.0, le=1.0, description="Data consistency score")
    freshness_score: float = Field(ge=0.0, le=1.0, description="Data freshness score")
    overall_quality: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    
    # Validation details
    missing_fields: List[str] = Field(default_factory=list)
    inconsistent_fields: List[str] = Field(default_factory=list)
    low_confidence_facts: List[str] = Field(default_factory=list)
    
    # Recommendations
    improvement_suggestions: List[str] = Field(default_factory=list)


class RetrievalConfig(BaseModel):
    """Configuration for 3-stage retrieval"""
    # Pre-filter stage
    pre_filter_enabled: bool = Field(default=True)
    pre_filter_tags: List[str] = Field(default_factory=list)
    pre_filter_facts: List[str] = Field(default_factory=list)
    
    # Vector recall stage
    vector_recall_enabled: bool = Field(default=True)
    vector_recall_top_k: int = Field(default=50, ge=1, le=1000)
    embedding_views: List[EmbeddingView] = Field(default_factory=lambda: [EmbeddingView.COMPREHENSIVE])
    
    # Reranker stage
    reranker_enabled: bool = Field(default=True)
    reranker_top_k: int = Field(default=10, ge=1, le=100)
    reranker_model: str = Field(default="cross_encoder", description="Reranker model type")
    
    # Quality thresholds
    min_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    min_quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


# ==================== RESPONSE MODELS ====================

class CandidateMatchingResponse(BaseModel):
    """Response model for candidate matching"""
    candidate_id: str = Field(description="Candidate identifier")
    total_matches_found: int = Field(description="Total number of matches found")
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    
    # Stage results
    pre_filter_results: int = Field(description="Number of candidates after pre-filter")
    vector_recall_results: int = Field(description="Number of candidates after vector recall")
    reranker_results: int = Field(description="Number of candidates after reranking")
    
    # Final matches
    top_matches: List[JobMatchResult] = Field(description="Top job matches")
    
    # Quality metrics
    overall_confidence: ConfidenceScore = Field(description="Overall confidence in results")
    quality_metrics: QualityMetrics = Field(description="Quality assessment")
    
    # Metadata
    retrieval_config: RetrievalConfig = Field(description="Configuration used")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ProcessingStatus(BaseModel):
    """Processing status for monitoring"""
    candidate_id: str = Field(description="Candidate identifier")
    current_stage: ProcessingStage = Field(description="Current processing stage")
    progress_percentage: float = Field(ge=0.0, le=100.0, description="Progress percentage")
    status: Literal["pending", "processing", "completed", "failed", "retrying"] = Field(description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


# ==================== UTILITY FUNCTIONS ====================

def create_confidence_score(value: float, reasoning: Optional[str] = None) -> ConfidenceScore:
    """Create a confidence score with auto-determined level"""
    return ConfidenceScore(value=value, reasoning=reasoning)


def create_provenance(source: DataSource, source_id: str, extraction_method: str) -> Provenance:
    """Create provenance information"""
    return Provenance(
        source=source,
        source_id=source_id,
        extraction_method=extraction_method
    )


def calculate_overall_confidence(scores: List[ConfidenceScore]) -> ConfidenceScore:
    """Calculate overall confidence from multiple scores"""
    if not scores:
        return create_confidence_score(0.0, "No confidence scores available")
    
    avg_score = sum(score.value for score in scores) / len(scores)
    return create_confidence_score(avg_score, f"Average of {len(scores)} confidence scores")


def validate_candidate_profile(profile: CandidateProfile) -> QualityMetrics:
    """Validate candidate profile and return quality metrics"""
    # Calculate completeness
    required_fields = ["candidate_id", "uid", "tenant_id"]
    missing_fields = [field for field in required_fields if not getattr(profile, field, None)]
    
    # Calculate confidence-based quality
    fact_confidences = [fact.confidence.value for fact in profile.facts]
    tag_confidences = [tag.confidence.value for tag in profile.tags]
    embedding_confidences = [emb.confidence.value for emb in profile.embeddings]
    
    avg_confidence = 0.0
    if fact_confidences or tag_confidences or embedding_confidences:
        all_confidences = fact_confidences + tag_confidences + embedding_confidences
        avg_confidence = sum(all_confidences) / len(all_confidences)
    
    # Calculate completeness score
    completeness = 1.0 - (len(missing_fields) / len(required_fields))
    
    # Calculate overall quality
    overall_quality = (completeness + avg_confidence) / 2
    
    return QualityMetrics(
        completeness_score=completeness,
        accuracy_score=avg_confidence,
        consistency_score=avg_confidence,  # Simplified for now
        freshness_score=1.0,  # Assume fresh data
        overall_quality=overall_quality,
        missing_fields=missing_fields,
        low_confidence_facts=[f.fact_id for f in profile.facts if f.confidence.value < 0.5]
    )

"""
Jobsify AI Candidate Matching Architecture - Main Integration Module
Integrates all components into a cohesive candidate matching system
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .candidate_matching_models import (
    CandidateProfile, JobMatchResult, CandidateMatchingResponse,
    ProcessingEvent, EventType, DataSource, ProcessingStage,
    StructuredFact, DynamicTag, MultiViewEmbedding, QualityMetrics
)
from .profile_event_processor import profile_event_processor, EventType as PE_EventType
from .fact_extractor import fact_extractor, ExtractionContext
from .tag_generator import tag_generator, TagContext
from .multi_view_embedder import multi_view_embedder, EmbeddingContext
from .intelligent_merger import intelligent_merger, MergeContext
from .quality_gate import quality_gate, ValidationLevel
from .retrieval_gateway import retrieval_gateway, JobDescription

log = logging.getLogger(__name__)


@dataclass
class CandidateMatchingRequest:
    """Request for candidate matching"""
    candidate_id: str
    uid: str
    tenant_id: str
    job_description: JobDescription
    config_name: str = "default"
    validation_level: str = "standard"


@dataclass
class ProcessingResult:
    """Result of processing pipeline"""
    candidate_profile: CandidateProfile
    processing_stage: ProcessingStage
    quality_metrics: QualityMetrics
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None


class JobsifyCandidateMatchingSystem:
    """
    Main integration class for the Jobsify AI Candidate Matching Architecture
    Orchestrates all components in the event-driven pipeline
    """
    
    def __init__(self):
        self.components = {
            "profile_event_processor": profile_event_processor,
            "fact_extractor": fact_extractor,
            "tag_generator": tag_generator,
            "multi_view_embedder": multi_view_embedder,
            "intelligent_merger": intelligent_merger,
            "quality_gate": quality_gate,
            "retrieval_gateway": retrieval_gateway
        }
        
        # Initialize processing pipeline
        self._initialize_pipeline()
        
    def _initialize_pipeline(self):
        """Initialize the processing pipeline"""
        log.info("Initializing Jobsify Candidate Matching Pipeline")
        
        # Start the event processor
        asyncio.create_task(self.components["profile_event_processor"].start_processing())
        
        log.info("Pipeline initialized successfully")
    
    async def process_candidate_data(self, candidate_id: str, uid: str, tenant_id: str,
                                   source_data: Dict[str, Any], source_type: DataSource) -> ProcessingResult:
        """
        Process candidate data through the complete pipeline
        """
        log.info(f"Processing candidate data for {candidate_id} from {source_type}")
        
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Publish event for data ingestion
            await self._publish_data_ingestion_event(candidate_id, source_data, source_type)
            
            # Step 2: Extract facts
            facts = await self._extract_facts(candidate_id, source_data, source_type)
            
            # Step 3: Generate tags
            tags = await self._generate_tags(candidate_id, facts, source_data)
            
            # Step 4: Create embeddings
            embeddings = await self._create_embeddings(candidate_id, facts, tags, source_data)
            
            # Step 5: Merge data from multiple sources
            merged_profile = await self._merge_candidate_data(candidate_id, uid, tenant_id, facts, tags, embeddings, source_data)
            
            # Step 6: Validate quality
            quality_metrics = await self._validate_quality(merged_profile)
            
            # Step 7: Update processing stage
            merged_profile.processing_stage = ProcessingStage.RETRIEVAL
            merged_profile.last_updated = datetime.utcnow()
            
            processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            log.info(f"Successfully processed candidate {candidate_id} in {processing_time_ms}ms")
            
            return ProcessingResult(
                candidate_profile=merged_profile,
                processing_stage=ProcessingStage.RETRIEVAL,
                quality_metrics=quality_metrics,
                processing_time_ms=processing_time_ms,
                success=True
            )
            
        except Exception as e:
            log.error(f"Error processing candidate {candidate_id}: {e}")
            processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return ProcessingResult(
                candidate_profile=None,
                processing_stage=ProcessingStage.INGESTION,
                quality_metrics=None,
                processing_time_ms=processing_time_ms,
                success=False,
                error_message=str(e)
            )
    
    async def find_job_matches(self, request: CandidateMatchingRequest) -> CandidateMatchingResponse:
        """
        Find job matches for a candidate using the 3-stage retrieval funnel
        """
        log.info(f"Finding job matches for candidate {request.candidate_id}")
        
        try:
            # Get candidate profile
            candidate_profile = await self._get_candidate_profile(request.candidate_id)
            
            if not candidate_profile:
                raise ValueError(f"Candidate profile not found for {request.candidate_id}")
            
            # Find matches using retrieval gateway
            matches = await self.components["retrieval_gateway"].find_job_matches(
                candidate_profile, request.job_description, request.config_name
            )
            
            log.info(f"Found {matches.total_matches_found} matches for candidate {request.candidate_id}")
            return matches
            
        except Exception as e:
            log.error(f"Error finding job matches for candidate {request.candidate_id}: {e}")
            raise
    
    async def _publish_data_ingestion_event(self, candidate_id: str, source_data: Dict[str, Any], source_type: DataSource):
        """Publish data ingestion event"""
        event_type_map = {
            DataSource.RESUME: PE_EventType.RESUME_UPLOADED,
            DataSource.CHAT_SESSION: PE_EventType.CHAT_SESSION_CREATED,
            DataSource.ASSESSMENT: PE_EventType.ASSESSMENT_COMPLETED,
            DataSource.INTERVIEW: PE_EventType.INTERVIEW_COMPLETED,
            DataSource.LINKEDIN: PE_EventType.PROFILE_UPDATED,
            DataSource.PORTFOLIO: PE_EventType.PROFILE_UPDATED
        }
        
        event_type = event_type_map.get(source_type, PE_EventType.PROFILE_UPDATED)
        
        await self.components["profile_event_processor"].publish_event(
            event_type, candidate_id, source_data
        )
    
    async def _extract_facts(self, candidate_id: str, source_data: Dict[str, Any], source_type: DataSource) -> List[StructuredFact]:
        """Extract facts from source data"""
        context = ExtractionContext(
            source_data=source_data,
            source_type=source_type,
            candidate_id=candidate_id,
            extraction_timestamp=datetime.utcnow()
        )
        
        return await self.components["fact_extractor"].extract_facts(context)
    
    async def _generate_tags(self, candidate_id: str, facts: List[StructuredFact], source_data: Dict[str, Any]) -> List[DynamicTag]:
        """Generate tags from facts and source data"""
        context = TagContext(
            candidate_id=candidate_id,
            facts=facts,
            profile_data=source_data,
            source_data=source_data,
            generation_timestamp=datetime.utcnow()
        )
        
        return await self.components["tag_generator"].generate_tags(context)
    
    async def _create_embeddings(self, candidate_id: str, facts: List[StructuredFact], 
                               tags: List[DynamicTag], source_data: Dict[str, Any]) -> List[MultiViewEmbedding]:
        """Create embeddings from facts, tags, and source data"""
        context = EmbeddingContext(
            candidate_id=candidate_id,
            facts=facts,
            tags=tags,
            profile_data=source_data,
            source_data=source_data,
            generation_timestamp=datetime.utcnow()
        )
        
        return await self.components["multi_view_embedder"].generate_embeddings(context)
    
    async def _merge_candidate_data(self, candidate_id: str, uid: str, tenant_id: str,
                                  facts: List[StructuredFact], tags: List[DynamicTag],
                                  embeddings: List[MultiViewEmbedding], source_data: Dict[str, Any]) -> CandidateProfile:
        """Merge candidate data from multiple sources"""
        # Get existing data from other sources
        existing_facts = await self._get_existing_facts(candidate_id)
        existing_tags = await self._get_existing_tags(candidate_id)
        existing_embeddings = await self._get_existing_embeddings(candidate_id)
        
        # Merge with new data
        all_facts = existing_facts + facts
        all_tags = existing_tags + tags
        all_embeddings = existing_embeddings + embeddings
        
        # Create merge context
        context = MergeContext(
            candidate_id=candidate_id,
            source_data_sets={DataSource.RESUME: source_data},  # Simplified for now
            facts_by_source={DataSource.RESUME: all_facts},
            tags_by_source={DataSource.RESUME: all_tags},
            embeddings_by_source={DataSource.RESUME: all_embeddings},
            merge_timestamp=datetime.utcnow()
        )
        
        return await self.components["intelligent_merger"].merge_candidate_data(context)
    
    async def _validate_quality(self, candidate_profile: CandidateProfile) -> QualityMetrics:
        """Validate candidate profile quality"""
        validation_level = ValidationLevel.STANDARD
        
        return await self.components["quality_gate"].validate_candidate_profile(
            candidate_profile, validation_level
        )
    
    async def _get_candidate_profile(self, candidate_id: str) -> Optional[CandidateProfile]:
        """Get candidate profile (mock implementation)"""
        # In production, this would query the database
        return CandidateProfile(
            candidate_id=candidate_id,
            uid=candidate_id,
            tenant_id="default",
            profile_data={},
            facts=[],
            tags=[],
            embeddings=[]
        )
    
    async def _get_existing_facts(self, candidate_id: str) -> List[StructuredFact]:
        """Get existing facts for candidate (mock implementation)"""
        return []
    
    async def _get_existing_tags(self, candidate_id: str) -> List[DynamicTag]:
        """Get existing tags for candidate (mock implementation)"""
        return []
    
    async def _get_existing_embeddings(self, candidate_id: str) -> List[MultiViewEmbedding]:
        """Get existing embeddings for candidate (mock implementation)"""
        return []
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        status = {
            "system_name": "Jobsify AI Candidate Matching Architecture",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Get status from each component
        for name, component in self.components.items():
            try:
                if hasattr(component, 'get_processing_stats'):
                    status["components"][name] = component.get_processing_stats()
                elif hasattr(component, 'get_extraction_stats'):
                    status["components"][name] = component.get_extraction_stats()
                elif hasattr(component, 'get_tag_generation_stats'):
                    status["components"][name] = component.get_tag_generation_stats()
                elif hasattr(component, 'get_embedding_stats'):
                    status["components"][name] = component.get_embedding_stats()
                elif hasattr(component, 'get_merge_stats'):
                    status["components"][name] = component.get_merge_stats()
                elif hasattr(component, 'get_quality_gate_stats'):
                    status["components"][name] = component.get_quality_gate_stats()
                elif hasattr(component, 'get_retrieval_gateway_stats'):
                    status["components"][name] = component.get_retrieval_gateway_stats()
                else:
                    status["components"][name] = {"status": "active"}
            except Exception as e:
                status["components"][name] = {"status": "error", "error": str(e)}
        
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        for name, component in self.components.items():
            try:
                # Simple health check - can be enhanced
                if hasattr(component, 'is_running'):
                    health_status["components"][name] = {
                        "status": "healthy" if component.is_running else "unhealthy",
                        "details": "Component is running" if component.is_running else "Component is not running"
                    }
                else:
                    health_status["components"][name] = {
                        "status": "healthy",
                        "details": "Component is available"
                    }
            except Exception as e:
                health_status["components"][name] = {
                    "status": "unhealthy",
                    "details": f"Error: {str(e)}"
                }
                health_status["overall_status"] = "unhealthy"
        
        return health_status
    
    async def shutdown(self):
        """Shutdown the system gracefully"""
        log.info("Shutting down Jobsify Candidate Matching System")
        
        # Stop event processor
        await self.components["profile_event_processor"].stop_processing()
        
        log.info("System shutdown completed")


# ==================== SINGLETON INSTANCE ====================

# Global instance for the application
jobsify_system = JobsifyCandidateMatchingSystem()


# ==================== CONVENIENCE FUNCTIONS ====================

async def process_resume_data(candidate_id: str, uid: str, tenant_id: str, resume_data: Dict[str, Any]) -> ProcessingResult:
    """Process resume data for a candidate"""
    return await jobsify_system.process_candidate_data(
        candidate_id, uid, tenant_id, resume_data, DataSource.RESUME
    )


async def process_chat_data(candidate_id: str, uid: str, tenant_id: str, chat_data: Dict[str, Any]) -> ProcessingResult:
    """Process chat session data for a candidate"""
    return await jobsify_system.process_candidate_data(
        candidate_id, uid, tenant_id, chat_data, DataSource.CHAT_SESSION
    )


async def process_assessment_data(candidate_id: str, uid: str, tenant_id: str, assessment_data: Dict[str, Any]) -> ProcessingResult:
    """Process assessment data for a candidate"""
    return await jobsify_system.process_candidate_data(
        candidate_id, uid, tenant_id, assessment_data, DataSource.ASSESSMENT
    )


async def process_interview_data(candidate_id: str, uid: str, tenant_id: str, interview_data: Dict[str, Any]) -> ProcessingResult:
    """Process interview data for a candidate"""
    return await jobsify_system.process_candidate_data(
        candidate_id, uid, tenant_id, interview_data, DataSource.INTERVIEW
    )


async def find_job_matches_for_candidate(candidate_id: str, job_description: JobDescription, 
                                       config_name: str = "default") -> CandidateMatchingResponse:
    """Find job matches for a candidate"""
    request = CandidateMatchingRequest(
        candidate_id=candidate_id,
        uid=candidate_id,  # Assuming candidate_id is the same as uid
        tenant_id="default",
        job_description=job_description,
        config_name=config_name
    )
    
    return await jobsify_system.find_job_matches(request)


async def get_system_status() -> Dict[str, Any]:
    """Get system status"""
    return await jobsify_system.get_system_status()


async def perform_health_check() -> Dict[str, Any]:
    """Perform health check"""
    return await jobsify_system.health_check()

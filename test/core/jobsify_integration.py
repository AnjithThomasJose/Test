"""
Jobsify AI Candidate Matching - LangGraph Integration
Integrates the enhanced architecture with existing LangGraph pipeline
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .candidate_matching_models import (
    CandidateProfile, JobMatchResult, ConfidenceScore, RetrievalConfig,
    CandidateMatchingResponse, QualityMetrics, ProcessingEvent, ProcessingStage
)
from .enhanced_retrieval_gateway import EnhancedRetrievalGateway, JobDescription, PerformanceMetrics
from .profile_event_processor import ProfileEventProcessor
from .fact_extractor import FactExtractor
from .tag_generator import TagGenerator
from .multi_view_embedder import MultiViewEmbedder
from .intelligent_merger import IntelligentMerger
from .quality_gate import QualityGate
from .firestore_schema import FirestoreSchema
from .chromadb_manager import ChromaDBManager

try:
    from core.langfuse_tracing import merge_langfuse_into_config
    from settings import settings
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False

log = logging.getLogger(__name__)


class JobsifyAgentState:
    """Enhanced agent state for Jobsify AI Candidate Matching"""
    
    def __init__(self):
        # Core candidate data
        self.candidate_id: str = ""
        self.uid: str = ""
        self.tenant_id: str = ""
        
        # Multi-source data
        self.resume_data: Dict[str, Any] = {}
        self.chat_data: Dict[str, Any] = {}
        self.assessment_data: Dict[str, Any] = {}
        self.interview_data: Dict[str, Any] = {}
        
        # Processing pipeline data
        self.profile_event: Optional[ProcessingEvent] = None
        self.extracted_facts: List[Dict[str, Any]] = []
        self.generated_tags: List[Dict[str, Any]] = []
        self.embeddings: List[Dict[str, Any]] = []
        self.merged_profile: Optional[CandidateProfile] = None
        self.quality_metrics: Optional[QualityMetrics] = None
        
        # Job matching results
        self.job_description: Optional[JobDescription] = None
        self.top_matches: List[JobMatchResult] = []
        self.job_matcher_status: str = "pending"
        self.total_matches_found: int = 0
        
        # Performance metrics
        self.performance_metrics: List[PerformanceMetrics] = []
        self.processing_time_ms: int = 0
        
        # Error handling
        self.errors: List[str] = []
        self.retry_count: int = 0
        
        # Configuration
        self.retrieval_config: str = "enterprise"
        self.confidence_threshold: float = 0.3
        self.quality_threshold: float = 0.5


class JobsifyIntegration:
    """
    Main integration class for Jobsify AI Candidate Matching
    Orchestrates the entire pipeline with event-driven processing
    """
    
    def __init__(self):
        # Initialize core components
        self.profile_event_processor = ProfileEventProcessor()
        self.fact_extractor = FactExtractor()
        self.tag_generator = TagGenerator()
        self.embedder = MultiViewEmbedder()
        self.merger = IntelligentMerger()
        self.quality_gate = QualityGate()
        self.retrieval_gateway = EnhancedRetrievalGateway()
        self.firestore_schema = FirestoreSchema()
        self.chromadb_manager = ChromaDBManager()
        
        # Initialize LangGraph
        self.graph = self._create_jobsify_graph()
        
        log.info("Jobsify AI Candidate Matching Integration initialized")
    
    def _create_jobsify_graph(self) -> StateGraph:
        """Create the LangGraph workflow for Jobsify AI Candidate Matching"""
        
        # Create the state graph
        workflow = StateGraph(JobsifyAgentState)
        
        # Add nodes for each processing stage
        workflow.add_node("profile_event_processor", self._process_profile_event)
        workflow.add_node("fact_extractor", self._extract_facts)
        workflow.add_node("tag_generator", self._generate_tags)
        workflow.add_node("embedder", self._create_embeddings)
        workflow.add_node("merger", self._merge_data)
        workflow.add_node("quality_gate", self._validate_quality)
        workflow.add_node("retrieval_gateway", self._find_job_matches)
        workflow.add_node("error_handler", self._handle_errors)
        
        # Define the workflow edges
        workflow.set_entry_point("profile_event_processor")
        
        workflow.add_edge("profile_event_processor", "fact_extractor")
        workflow.add_edge("fact_extractor", "tag_generator")
        workflow.add_edge("tag_generator", "embedder")
        workflow.add_edge("embedder", "merger")
        workflow.add_edge("merger", "quality_gate")
        
        # Conditional edge from quality gate
        workflow.add_conditional_edges(
            "quality_gate",
            self._should_proceed_to_matching,
            {
                "proceed": "retrieval_gateway",
                "retry": "fact_extractor",
                "fail": "error_handler"
            }
        )
        
        workflow.add_edge("retrieval_gateway", END)
        workflow.add_edge("error_handler", END)
        
        return workflow.compile()
    
    async def _process_profile_event(self, state: JobsifyAgentState) -> JobsifyAgentState:
        """Process profile event and extract multi-source data"""
        try:
            log.info(f"Processing profile event for candidate {state.candidate_id}")
            
            # Process the profile event
            event_result = await self.profile_event_processor.process_event(
                candidate_id=state.candidate_id,
                uid=state.uid,
                tenant_id=state.tenant_id,
                resume_data=state.resume_data,
                chat_data=state.chat_data,
                assessment_data=state.assessment_data,
                interview_data=state.interview_data
            )
            
            state.profile_event = event_result.profile_event
            state.resume_data = event_result.resume_data
            state.chat_data = event_result.chat_data
            state.assessment_data = event_result.assessment_data
            state.interview_data = event_result.interview_data
            
            log.info(f"Profile event processed successfully for candidate {state.candidate_id}")
            
        except Exception as e:
            log.error(f"Error processing profile event: {e}")
            state.errors.append(f"Profile event processing failed: {str(e)}")
        
        return state
    
    async def _extract_facts(self, state: JobsifyAgentState) -> JobsifyAgentState:
        """Extract structured facts from multi-source data"""
        try:
            log.info(f"Extracting facts for candidate {state.candidate_id}")
            
            # Extract facts from all sources
            facts_result = await self.fact_extractor.extract_facts(
                resume_data=state.resume_data,
                chat_data=state.chat_data,
                assessment_data=state.assessment_data,
                interview_data=state.interview_data,
                candidate_id=state.candidate_id
            )
            
            state.extracted_facts = facts_result.extracted_facts
            state.confidence_threshold = facts_result.confidence_threshold
            
            log.info(f"Extracted {len(state.extracted_facts)} facts for candidate {state.candidate_id}")
            
        except Exception as e:
            log.error(f"Error extracting facts: {e}")
            state.errors.append(f"Fact extraction failed: {str(e)}")
        
        return state
    
    async def _generate_tags(self, state: JobsifyAgentState) -> JobsifyAgentState:
        """Generate dynamic tags from extracted facts"""
        try:
            log.info(f"Generating tags for candidate {state.candidate_id}")
            
            # Generate tags from facts
            tags_result = await self.tag_generator.generate_tags(
                facts=state.extracted_facts,
                candidate_id=state.candidate_id
            )
            
            state.generated_tags = tags_result.generated_tags
            
            log.info(f"Generated {len(state.generated_tags)} tags for candidate {state.candidate_id}")
            
        except Exception as e:
            log.error(f"Error generating tags: {e}")
            state.errors.append(f"Tag generation failed: {str(e)}")
        
        return state
    
    async def _create_embeddings(self, state: JobsifyAgentState) -> JobsifyAgentState:
        """Create multi-view embeddings"""
        try:
            log.info(f"Creating embeddings for candidate {state.candidate_id}")
            
            # Create embeddings from all sources
            embeddings_result = await self.embedder.create_embeddings(
                resume_data=state.resume_data,
                chat_data=state.chat_data,
                assessment_data=state.assessment_data,
                interview_data=state.interview_data,
                facts=state.extracted_facts,
                tags=state.generated_tags,
                candidate_id=state.candidate_id
            )
            
            state.embeddings = embeddings_result.embeddings
            
            log.info(f"Created {len(state.embeddings)} embeddings for candidate {state.candidate_id}")
            
        except Exception as e:
            log.error(f"Error creating embeddings: {e}")
            state.errors.append(f"Embedding creation failed: {str(e)}")
        
        return state
    
    async def _merge_data(self, state: JobsifyAgentState) -> JobsifyAgentState:
        """Merge data from multiple sources"""
        try:
            log.info(f"Merging data for candidate {state.candidate_id}")
            
            # Merge all data sources
            merge_result = await self.merger.merge_data(
                resume_data=state.resume_data,
                chat_data=state.chat_data,
                assessment_data=state.assessment_data,
                interview_data=state.interview_data,
                facts=state.extracted_facts,
                tags=state.generated_tags,
                embeddings=state.embeddings,
                candidate_id=state.candidate_id,
                uid=state.uid,
                tenant_id=state.tenant_id
            )
            
            state.merged_profile = merge_result.merged_profile
            
            log.info(f"Data merged successfully for candidate {state.candidate_id}")
            
        except Exception as e:
            log.error(f"Error merging data: {e}")
            state.errors.append(f"Data merging failed: {str(e)}")
        
        return state
    
    async def _validate_quality(self, state: JobsifyAgentState) -> JobsifyAgentState:
        """Validate data quality and completeness"""
        try:
            log.info(f"Validating quality for candidate {state.candidate_id}")
            
            # Validate quality
            quality_result = await self.quality_gate.validate_quality(
                candidate_profile=state.merged_profile,
                facts=state.extracted_facts,
                tags=state.generated_tags,
                embeddings=state.embeddings
            )
            
            state.quality_metrics = quality_result.quality_metrics
            
            log.info(f"Quality validation completed for candidate {state.candidate_id}")
            
        except Exception as e:
            log.error(f"Error validating quality: {e}")
            state.errors.append(f"Quality validation failed: {str(e)}")
        
        return state
    
    def _should_proceed_to_matching(self, state: JobsifyAgentState) -> str:
        """Determine if we should proceed to job matching"""
        if state.errors:
            return "fail"
        
        if state.quality_metrics and state.quality_metrics.overall_quality < state.quality_threshold:
            if state.retry_count < 3:
                state.retry_count += 1
                return "retry"
            else:
                return "fail"
        
        return "proceed"
    
    async def _find_job_matches(self, state: JobsifyAgentState) -> JobsifyAgentState:
        """Find job matches using enhanced retrieval gateway"""
        try:
            log.info(f"Finding job matches for candidate {state.candidate_id}")
            
            if not state.merged_profile or not state.job_description:
                state.errors.append("Missing candidate profile or job description")
                return state
            
            # Find job matches
            matching_result = await self.retrieval_gateway.find_job_matches(
                candidate_profile=state.merged_profile,
                job_description=state.job_description,
                config_name=state.retrieval_config
            )
            
            state.top_matches = matching_result.top_matches
            state.total_matches_found = matching_result.total_matches_found
            state.job_matcher_status = "completed"
            state.performance_metrics = self.retrieval_gateway.get_performance_metrics()
            state.processing_time_ms = matching_result.processing_time_ms
            
            log.info(f"Found {state.total_matches_found} job matches for candidate {state.candidate_id}")
            
        except Exception as e:
            log.error(f"Error finding job matches: {e}")
            state.errors.append(f"Job matching failed: {str(e)}")
            state.job_matcher_status = "failed"
        
        return state
    
    async def _handle_errors(self, state: JobsifyAgentState) -> JobsifyAgentState:
        """Handle errors and provide fallback options"""
        log.error(f"Handling errors for candidate {state.candidate_id}: {state.errors}")
        
        # Set status to failed
        state.job_matcher_status = "failed"
        
        # Could implement retry logic or fallback matching here
        
        return state
    
    async def process_candidate(self, candidate_id: str, uid: str, tenant_id: str,
                             resume_data: Dict[str, Any],
                             chat_data: Optional[Dict[str, Any]] = None,
                             assessment_data: Optional[Dict[str, Any]] = None,
                             interview_data: Optional[Dict[str, Any]] = None,
                             job_description: Optional[JobDescription] = None,
                             retrieval_config: str = "enterprise") -> Dict[str, Any]:
        """
        Process a candidate through the entire Jobsify AI pipeline
        """
        log.info(f"Processing candidate {candidate_id} through Jobsify AI pipeline")
        
        # Initialize state
        state = JobsifyAgentState()
        state.candidate_id = candidate_id
        state.uid = uid
        state.tenant_id = tenant_id
        state.resume_data = resume_data
        state.chat_data = chat_data or {}
        state.assessment_data = assessment_data or {}
        state.interview_data = interview_data or {}
        state.job_description = job_description
        state.retrieval_config = retrieval_config
        
        # Run the workflow
        try:
            config = (
                merge_langfuse_into_config(
                    {}, settings,
                    session_id=candidate_id,
                    user_id=uid,
                )
                if _LANGFUSE_AVAILABLE
                else {}
            )
            final_state = await self.graph.ainvoke(state, config=config)
            
            # Return results
            return {
                "candidate_id": final_state.candidate_id,
                "status": final_state.job_matcher_status,
                "total_matches_found": final_state.total_matches_found,
                "top_matches": [match.dict() for match in final_state.top_matches],
                "processing_time_ms": final_state.processing_time_ms,
                "performance_metrics": [metric.__dict__ for metric in final_state.performance_metrics],
                "quality_metrics": final_state.quality_metrics.dict() if final_state.quality_metrics else None,
                "errors": final_state.errors,
                "retry_count": final_state.retry_count
            }
            
        except Exception as e:
            log.error(f"Error processing candidate {candidate_id}: {e}")
            return {
                "candidate_id": candidate_id,
                "status": "failed",
                "total_matches_found": 0,
                "top_matches": [],
                "processing_time_ms": 0,
                "performance_metrics": [],
                "quality_metrics": None,
                "errors": [str(e)],
                "retry_count": 0
            }
    
    async def batch_process_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple candidates in batch"""
        log.info(f"Batch processing {len(candidates)} candidates")
        
        results = []
        for candidate_data in candidates:
            result = await self.process_candidate(**candidate_data)
            results.append(result)
        
        log.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return {
            "components": {
                "profile_event_processor": "active",
                "fact_extractor": "active",
                "tag_generator": "active",
                "embedder": "active",
                "merger": "active",
                "quality_gate": "active",
                "retrieval_gateway": "active"
            },
            "retrieval_gateway_stats": self.retrieval_gateway.get_retrieval_gateway_stats(),
            "firestore_collections": self.firestore_schema.get_collections(),
            "chromadb_collections": self.chromadb_manager.get_collections(),
            "performance_targets": {
                "pre_filter": "O(log n)",
                "vector_recall": "O(log k)",
                "reranker": "O(10)",
                "total_time": "< 1 second"
            }
        }


# ==================== SINGLETON INSTANCE ====================

# Global instance for the application
jobsify_integration = JobsifyIntegration()

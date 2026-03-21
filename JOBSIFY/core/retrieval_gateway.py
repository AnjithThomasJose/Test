"""
RetrievalGateway - 3-stage retrieval funnel implementation
Implements Pre-Filter, Vector Recall, and Reranker stages for intelligent job matching
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import uuid

from sentence_transformers import CrossEncoder

from .candidate_matching_models import (
    CandidateProfile, JobMatchResult, ConfidenceScore, RetrievalConfig,
    CandidateMatchingResponse, QualityMetrics, EmbeddingView
)

log = logging.getLogger(__name__)


class RetrievalStage(str, Enum):
    """Retrieval stages"""
    PRE_FILTER = "pre_filter"
    VECTOR_RECALL = "vector_recall"
    RERANKER = "reranker"


class MatchStrategy(str, Enum):
    """Match strategies"""
    SKILL_BASED = "skill_based"
    EXPERIENCE_BASED = "experience_based"
    EDUCATION_BASED = "education_based"
    COMPREHENSIVE = "comprehensive"
    CULTURAL_FIT = "cultural_fit"


@dataclass
class JobDescription:
    """Job description data structure"""
    job_id: str
    job_title: str
    company_name: str
    location: str
    required_skills: List[str]
    preferred_skills: List[str]
    required_experience: str
    education_requirements: List[str]
    job_description: str
    company_culture: List[str]
    benefits: List[str]
    salary_range: Optional[str] = None
    job_type: Optional[str] = None
    remote_option: Optional[bool] = None


@dataclass
class PreFilterResult:
    """Pre-filter stage result"""
    candidate_ids: List[str]
    filter_criteria: Dict[str, Any]
    candidates_filtered: int
    processing_time_ms: int


@dataclass
class VectorRecallResult:
    """Vector recall stage result"""
    candidate_ids: List[str]
    similarity_scores: List[float]
    embedding_view: EmbeddingView
    candidates_recalled: int
    processing_time_ms: int


@dataclass
class RerankerResult:
    """Reranker stage result"""
    job_matches: List[JobMatchResult]
    reranker_model: str
    candidates_reranked: int
    processing_time_ms: int


class RetrievalGateway:
    """
    3-stage retrieval funnel for intelligent job-candidate matching
    Implements Pre-Filter, Vector Recall, and Reranker stages
    """
    
    def __init__(self):
        self.retrieval_configs = self._initialize_retrieval_configs()
        self.reranker_models = self._initialize_reranker_models()
        self.match_strategies = self._initialize_match_strategies()
        self.stage_processors = self._initialize_stage_processors()
        
    def _initialize_retrieval_configs(self) -> Dict[str, RetrievalConfig]:
        """Initialize retrieval configurations"""
        return {
            "default": RetrievalConfig(
                pre_filter_enabled=True,
                pre_filter_tags=["technical_skill", "seniority", "location_preference"],
                pre_filter_facts=["skill", "experience", "education"],
                vector_recall_enabled=True,
                vector_recall_top_k=50,
                embedding_views=[EmbeddingView.COMPREHENSIVE, EmbeddingView.SKILL_FOCUSED],
                reranker_enabled=True,
                reranker_top_k=10,
                reranker_model="cross_encoder",
                min_confidence_threshold=0.3,
                min_quality_threshold=0.5
            ),
            "fast": RetrievalConfig(
                pre_filter_enabled=True,
                pre_filter_tags=["technical_skill"],
                pre_filter_facts=["skill"],
                vector_recall_enabled=True,
                vector_recall_top_k=20,
                embedding_views=[EmbeddingView.SKILL_FOCUSED],
                reranker_enabled=False,
                reranker_top_k=5,
                reranker_model="cross_encoder",
                min_confidence_threshold=0.4,
                min_quality_threshold=0.6
            ),
            "comprehensive": RetrievalConfig(
                pre_filter_enabled=True,
                pre_filter_tags=["technical_skill", "seniority", "location_preference", "industry"],
                pre_filter_facts=["skill", "experience", "education", "certification"],
                vector_recall_enabled=True,
                vector_recall_top_k=100,
                embedding_views=[EmbeddingView.COMPREHENSIVE, EmbeddingView.SKILL_FOCUSED, EmbeddingView.EXPERIENCE_FOCUSED],
                reranker_enabled=True,
                reranker_top_k=20,
                reranker_model="cross_encoder",
                min_confidence_threshold=0.2,
                min_quality_threshold=0.4
            )
        }
    
    def _initialize_reranker_models(self) -> Dict[str, Any]:
        """Initialize reranker models container (lazy loading for Issue 3.4).
        
        Models are loaded on first use to avoid:
        - Slow cold start times (~500MB model download)
        - Import-time blocking
        - Memory waste when reranking is disabled
        """
        return {}  # Models loaded lazily via _get_reranker()
    
    def _get_reranker(self, model_name: str) -> Any:
        """Get or lazily initialize a reranker model (Issue 3.4).
        
        Args:
            model_name: Name of the reranker model (e.g., "cross_encoder")
            
        Returns:
            The reranker model instance, or None if loading failed
        """
        if model_name not in self.reranker_models:
            if model_name == "cross_encoder":
                try:
                    log.info("Lazy-loading CrossEncoder reranker model...")
                    self.reranker_models[model_name] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                    log.info("✅ Loaded CrossEncoder reranker model")
                except Exception as e:
                    log.error(f"Failed to load CrossEncoder model: {e}")
                    self.reranker_models[model_name] = None
            else:
                log.warning(f"Unknown reranker model: {model_name}")
                self.reranker_models[model_name] = None
        
        return self.reranker_models.get(model_name)
    
    def _initialize_match_strategies(self) -> Dict[MatchStrategy, callable]:
        """Initialize match strategies"""
        return {
            MatchStrategy.SKILL_BASED: self._match_by_skills,
            MatchStrategy.EXPERIENCE_BASED: self._match_by_experience,
            MatchStrategy.EDUCATION_BASED: self._match_by_education,
            MatchStrategy.COMPREHENSIVE: self._match_comprehensive,
            MatchStrategy.CULTURAL_FIT: self._match_by_cultural_fit
        }
    
    def _initialize_stage_processors(self) -> Dict[RetrievalStage, callable]:
        """Initialize stage processors"""
        return {
            RetrievalStage.PRE_FILTER: self._process_pre_filter,
            RetrievalStage.VECTOR_RECALL: self._process_vector_recall,
            RetrievalStage.RERANKER: self._process_reranker
        }
    
    async def find_job_matches(self, candidate_profile: CandidateProfile, 
                             job_description: JobDescription,
                             config_name: str = "default") -> CandidateMatchingResponse:
        """
        Find job matches using 3-stage retrieval funnel
        """
        log.info(f"Finding job matches for candidate {candidate_profile.candidate_id}")
        
        start_time = datetime.utcnow()
        
        # Get retrieval configuration
        config = self.retrieval_configs.get(config_name, self.retrieval_configs["default"])
        
        # Stage 1: Pre-Filter
        pre_filter_result = None
        if config.pre_filter_enabled:
            pre_filter_result = await self._process_pre_filter(candidate_profile, job_description, config)
            log.info(f"Pre-filter stage: {pre_filter_result.candidates_filtered} candidates")
        else:
            # If pre-filter disabled, get all candidates
            pre_filter_result = PreFilterResult(
                candidate_ids=[],  # Will be populated by vector recall
                filter_criteria={},
                candidates_filtered=0,
                processing_time_ms=0
            )
        
        # Stage 2: Vector Recall
        vector_recall_result = None
        if config.vector_recall_enabled:
            vector_recall_result = await self._process_vector_recall(
                candidate_profile, job_description, config, pre_filter_result
            )
            log.info(f"Vector recall stage: {vector_recall_result.candidates_recalled} candidates")
        else:
            # If vector recall disabled, use pre-filter results
            vector_recall_result = VectorRecallResult(
                candidate_ids=pre_filter_result.candidate_ids,
                similarity_scores=[1.0] * len(pre_filter_result.candidate_ids),
                embedding_view=EmbeddingView.COMPREHENSIVE,
                candidates_recalled=len(pre_filter_result.candidate_ids),
                processing_time_ms=0
            )
        
        # Stage 3: Reranker
        reranker_result = None
        if config.reranker_enabled:
            reranker_result = await self._process_reranker(
                candidate_profile, job_description, config, vector_recall_result
            )
            log.info(f"Reranker stage: {reranker_result.candidates_reranked} candidates")
        else:
            # If reranker disabled, create basic matches from vector recall
            reranker_result = await self._create_basic_matches(
                candidate_profile, job_description, vector_recall_result
            )
        
        # Calculate processing time
        processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Calculate overall confidence
        overall_confidence = await self._calculate_overall_confidence(reranker_result.job_matches)
        
        # Calculate quality metrics
        quality_metrics = await self._calculate_quality_metrics(candidate_profile, reranker_result.job_matches)
        
        # Create response
        response = CandidateMatchingResponse(
            candidate_id=candidate_profile.candidate_id,
            total_matches_found=len(reranker_result.job_matches),
            processing_time_ms=processing_time_ms,
            pre_filter_results=pre_filter_result.candidates_filtered,
            vector_recall_results=vector_recall_result.candidates_recalled,
            reranker_results=reranker_result.candidates_reranked,
            top_matches=reranker_result.job_matches,
            overall_confidence=overall_confidence,
            quality_metrics=quality_metrics,
            retrieval_config=config
        )
        
        log.info(f"Found {len(reranker_result.job_matches)} job matches for candidate {candidate_profile.candidate_id}")
        return response
    
    async def _process_pre_filter(self, candidate_profile: CandidateProfile, 
                                job_description: JobDescription, 
                                config: RetrievalConfig) -> PreFilterResult:
        """Process pre-filter stage"""
        start_time = datetime.utcnow()
        
        # Get all candidate IDs (in real implementation, this would query Firestore)
        all_candidate_ids = await self._get_all_candidate_ids()
        
        # Apply tag-based filtering
        tag_filtered_ids = await self._apply_tag_filtering(
            all_candidate_ids, job_description, config.pre_filter_tags
        )
        
        # Apply fact-based filtering
        fact_filtered_ids = await self._apply_fact_filtering(
            tag_filtered_ids, job_description, config.pre_filter_facts
        )
        
        # Apply quality filtering
        quality_filtered_ids = await self._apply_quality_filtering(
            fact_filtered_ids, config.min_quality_threshold
        )
        
        processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return PreFilterResult(
            candidate_ids=quality_filtered_ids,
            filter_criteria={
                "tag_filters": config.pre_filter_tags,
                "fact_filters": config.pre_filter_facts,
                "quality_threshold": config.min_quality_threshold
            },
            candidates_filtered=len(quality_filtered_ids),
            processing_time_ms=processing_time_ms
        )
    
    async def _process_vector_recall(self, candidate_profile: CandidateProfile,
                                   job_description: JobDescription,
                                   config: RetrievalConfig,
                                   pre_filter_result: PreFilterResult) -> VectorRecallResult:
        """Process vector recall stage"""
        start_time = datetime.utcnow()
        
        # Get candidate IDs from pre-filter or all candidates
        candidate_ids = pre_filter_result.candidate_ids if pre_filter_result.candidate_ids else await self._get_all_candidate_ids()
        
        # Create job description embedding
        job_embedding = await self._create_job_embedding(job_description)
        
        # Perform vector search for each embedding view
        all_similarities = {}
        for embedding_view in config.embedding_views:
            similarities = await self._perform_vector_search(
                candidate_ids, job_embedding, embedding_view, config.vector_recall_top_k
            )
            all_similarities[embedding_view] = similarities
        
        # Combine similarities from different views
        combined_similarities = await self._combine_similarities(all_similarities)
        
        # Sort by similarity and take top-k
        sorted_candidates = sorted(combined_similarities.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[:config.vector_recall_top_k]
        
        candidate_ids = [candidate_id for candidate_id, _ in top_candidates]
        similarity_scores = [score for _, score in top_candidates]
        
        processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return VectorRecallResult(
            candidate_ids=candidate_ids,
            similarity_scores=similarity_scores,
            embedding_view=EmbeddingView.COMPREHENSIVE,  # Combined view
            candidates_recalled=len(candidate_ids),
            processing_time_ms=processing_time_ms
        )
    
    async def _process_reranker(self, candidate_profile: CandidateProfile,
                              job_description: JobDescription,
                              config: RetrievalConfig,
                              vector_recall_result: VectorRecallResult) -> RerankerResult:
        """Process reranker stage"""
        start_time = datetime.utcnow()
        
        # Get candidate profiles for reranking
        candidate_profiles = await self._get_candidate_profiles(vector_recall_result.candidate_ids)
        
        # Perform reranking
        job_matches = []
        for candidate_id, similarity_score in zip(vector_recall_result.candidate_ids, vector_recall_result.similarity_scores):
            candidate_profile = candidate_profiles.get(candidate_id)
            if not candidate_profile:
                continue
            
            # Calculate detailed match scores
            match_result = await self._calculate_detailed_match(
                candidate_profile, job_description, similarity_score
            )
            
            if match_result:
                job_matches.append(match_result)
        
        # Sort by overall score
        job_matches.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Take top-k
        top_matches = job_matches[:config.reranker_top_k]
        
        processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        return RerankerResult(
            job_matches=top_matches,
            reranker_model=config.reranker_model,
            candidates_reranked=len(top_matches),
            processing_time_ms=processing_time_ms
        )
    
    async def _apply_tag_filtering(self, candidate_ids: List[str], 
                                 job_description: JobDescription,
                                 filter_tags: List[str]) -> List[str]:
        """Apply tag-based filtering"""
        filtered_ids = []
        
        for candidate_id in candidate_ids:
            # Get candidate tags (in real implementation, this would query Firestore)
            candidate_tags = await self._get_candidate_tags(candidate_id)
            
            # Check if candidate has required tags
            if await self._has_required_tags(candidate_tags, job_description, filter_tags):
                filtered_ids.append(candidate_id)
        
        return filtered_ids
    
    async def _apply_fact_filtering(self, candidate_ids: List[str],
                                  job_description: JobDescription,
                                  filter_facts: List[str]) -> List[str]:
        """Apply fact-based filtering"""
        filtered_ids = []
        
        for candidate_id in candidate_ids:
            # Get candidate facts (in real implementation, this would query Firestore)
            candidate_facts = await self._get_candidate_facts(candidate_id)
            
            # Check if candidate has required facts
            if await self._has_required_facts(candidate_facts, job_description, filter_facts):
                filtered_ids.append(candidate_id)
        
        return filtered_ids
    
    async def _apply_quality_filtering(self, candidate_ids: List[str],
                                    quality_threshold: float) -> List[str]:
        """Apply quality-based filtering"""
        filtered_ids = []
        
        for candidate_id in candidate_ids:
            # Get candidate quality score (in real implementation, this would query Firestore)
            quality_score = await self._get_candidate_quality_score(candidate_id)
            
            if quality_score >= quality_threshold:
                filtered_ids.append(candidate_id)
        
        return filtered_ids
    
    async def _perform_vector_search(self, candidate_ids: List[str],
                                   job_embedding: List[float],
                                   embedding_view: EmbeddingView,
                                   top_k: int) -> Dict[str, float]:
        """Perform vector search for a specific embedding view"""
        similarities = {}
        
        for candidate_id in candidate_ids:
            # Get candidate embedding (in real implementation, this would query ChromaDB)
            candidate_embedding = await self._get_candidate_embedding(candidate_id, embedding_view)
            
            if candidate_embedding:
                # Calculate cosine similarity
                similarity = self._calculate_cosine_similarity(job_embedding, candidate_embedding)
                similarities[candidate_id] = similarity
        
        return similarities
    
    async def _combine_similarities(self, all_similarities: Dict[EmbeddingView, Dict[str, float]]) -> Dict[str, float]:
        """Combine similarities from different embedding views"""
        combined = {}
        
        for candidate_id in set().union(*[sims.keys() for sims in all_similarities.values()]):
            similarities = []
            for view, sims in all_similarities.items():
                if candidate_id in sims:
                    similarities.append(sims[candidate_id])
            
            if similarities:
                # Use average similarity
                combined[candidate_id] = sum(similarities) / len(similarities)
        
        return combined
    
    async def _calculate_detailed_match(self, candidate_profile: CandidateProfile,
                                     job_description: JobDescription,
                                     similarity_score: float) -> Optional[JobMatchResult]:
        """Calculate detailed match scores"""
        
        # Calculate skill match score
        skill_match_score, matched_skills, unmatched_skills = await self._match_by_skills(
            candidate_profile, job_description
        )
        
        # Calculate experience match score
        experience_match_score, matched_experience = await self._match_by_experience(
            candidate_profile, job_description
        )
        
        # Calculate education match score
        education_match_score, matched_education = await self._match_by_education(
            candidate_profile, job_description
        )
        
        # Calculate cultural fit score
        cultural_fit_score = await self._match_by_cultural_fit(
            candidate_profile, job_description
        )
        
        # Calculate overall score
        overall_score = (
            skill_match_score * 0.4 +
            experience_match_score * 0.3 +
            education_match_score * 0.2 +
            cultural_fit_score * 0.1
        )
        
        # Calculate confidence
        confidence = ConfidenceScore(
            value=overall_score,
            reasoning=f"Skill: {skill_match_score:.2f}, Experience: {experience_match_score:.2f}, Education: {education_match_score:.2f}, Cultural: {cultural_fit_score:.2f}"
        )
        
        # Generate rationale
        rationale = await self._generate_match_rationale(
            candidate_profile, job_description, matched_skills, unmatched_skills,
            matched_experience, matched_education, overall_score
        )
        
        return JobMatchResult(
            job_id=job_description.job_id,
            candidate_id=candidate_profile.candidate_id,
            overall_score=overall_score,
            skill_match_score=skill_match_score,
            experience_match_score=experience_match_score,
            education_match_score=education_match_score,
            cultural_fit_score=cultural_fit_score,
            matched_skills=matched_skills,
            unmatched_skills=unmatched_skills,
            matched_experience=matched_experience,
            matched_education=matched_education,
            confidence=confidence,
            rationale=rationale,
            retrieval_stage="reranker"
        )
    
    async def _match_by_skills(self, candidate_profile: CandidateProfile, 
                             job_description: JobDescription) -> Tuple[float, List[str], List[str]]:
        """Match by skills"""
        candidate_skills = []
        for fact in candidate_profile.facts:
            if fact.fact_type.value == "skill":
                skill_name = fact.content.get("skill_name", "")
                if skill_name:
                    candidate_skills.append(skill_name.lower())
        
        required_skills = [skill.lower() for skill in job_description.required_skills]
        preferred_skills = [skill.lower() for skill in job_description.preferred_skills]
        
        matched_skills = []
        unmatched_skills = []
        
        # Check required skills
        for skill in required_skills:
            if any(skill in candidate_skill or candidate_skill in skill for candidate_skill in candidate_skills):
                matched_skills.append(skill)
            else:
                unmatched_skills.append(skill)
        
        # Check preferred skills
        for skill in preferred_skills:
            if any(skill in candidate_skill or candidate_skill in skill for candidate_skill in candidate_skills):
                matched_skills.append(skill)
        
        # Calculate score
        required_match_ratio = len(matched_skills) / len(required_skills) if required_skills else 1.0
        preferred_match_ratio = len([s for s in matched_skills if s in preferred_skills]) / len(preferred_skills) if preferred_skills else 0.0
        
        skill_match_score = required_match_ratio * 0.8 + preferred_match_ratio * 0.2
        
        return skill_match_score, matched_skills, unmatched_skills
    
    async def _match_by_experience(self, candidate_profile: CandidateProfile,
                                 job_description: JobDescription) -> Tuple[float, List[Dict[str, Any]]]:
        """Match by experience"""
        candidate_experience = []
        for fact in candidate_profile.facts:
            if fact.fact_type.value == "experience":
                candidate_experience.append(fact.content)
        
        matched_experience = []
        experience_match_score = 0.0
        
        if candidate_experience:
            # Simple experience matching - can be enhanced
            experience_match_score = min(1.0, len(candidate_experience) / 3.0)  # Normalize by experience count
            matched_experience = candidate_experience[:3]  # Take top 3 experiences
        
        return experience_match_score, matched_experience
    
    async def _match_by_education(self, candidate_profile: CandidateProfile,
                                job_description: JobDescription) -> Tuple[float, List[Dict[str, Any]]]:
        """Match by education"""
        candidate_education = []
        for fact in candidate_profile.facts:
            if fact.fact_type.value == "education":
                candidate_education.append(fact.content)
        
        matched_education = []
        education_match_score = 0.0
        
        if candidate_education:
            # Simple education matching - can be enhanced
            education_match_score = min(1.0, len(candidate_education) / 2.0)  # Normalize by education count
            matched_education = candidate_education
        
        return education_match_score, matched_education
    
    async def _match_by_cultural_fit(self, candidate_profile: CandidateProfile,
                                   job_description: JobDescription) -> float:
        """Match by cultural fit"""
        # Simple cultural fit matching - can be enhanced
        cultural_fit_score = 0.7  # Placeholder
        
        return cultural_fit_score
    
    async def _match_comprehensive(self, candidate_profile: CandidateProfile,
                                 job_description: JobDescription) -> float:
        """Comprehensive matching"""
        skill_score, _, _ = await self._match_by_skills(candidate_profile, job_description)
        experience_score, _ = await self._match_by_experience(candidate_profile, job_description)
        education_score, _ = await self._match_by_education(candidate_profile, job_description)
        cultural_score = await self._match_by_cultural_fit(candidate_profile, job_description)
        
        comprehensive_score = (
            skill_score * 0.4 +
            experience_score * 0.3 +
            education_score * 0.2 +
            cultural_score * 0.1
        )
        
        return comprehensive_score
    
    async def _generate_match_rationale(self, candidate_profile: CandidateProfile,
                                      job_description: JobDescription,
                                      matched_skills: List[str],
                                      unmatched_skills: List[str],
                                      matched_experience: List[Dict[str, Any]],
                                      matched_education: List[Dict[str, Any]],
                                      overall_score: float) -> str:
        """Generate match rationale"""
        rationale_parts = []
        
        # Skill rationale
        if matched_skills:
            rationale_parts.append(f"Strong match in skills: {', '.join(matched_skills[:3])}")
        
        if unmatched_skills:
            rationale_parts.append(f"Missing skills: {', '.join(unmatched_skills[:3])}")
        
        # Experience rationale
        if matched_experience:
            rationale_parts.append(f"Relevant experience: {len(matched_experience)} positions")
        
        # Education rationale
        if matched_education:
            rationale_parts.append(f"Educational background: {len(matched_education)} degrees")
        
        # Overall rationale
        if overall_score >= 0.8:
            rationale_parts.append("Excellent match for this position")
        elif overall_score >= 0.6:
            rationale_parts.append("Good match for this position")
        elif overall_score >= 0.4:
            rationale_parts.append("Moderate match for this position")
        else:
            rationale_parts.append("Limited match for this position")
        
        return ". ".join(rationale_parts) + "."
    
    async def _create_basic_matches(self, candidate_profile: CandidateProfile,
                                  job_description: JobDescription,
                                  vector_recall_result: VectorRecallResult) -> RerankerResult:
        """Create basic matches when reranker is disabled"""
        job_matches = []
        
        for candidate_id, similarity_score in zip(vector_recall_result.candidate_ids, vector_recall_result.similarity_scores):
            # Create basic match result
            match_result = JobMatchResult(
                job_id=job_description.job_id,
                candidate_id=candidate_id,
                overall_score=similarity_score,
                skill_match_score=similarity_score,
                experience_match_score=similarity_score,
                education_match_score=similarity_score,
                cultural_fit_score=similarity_score,
                matched_skills=[],
                unmatched_skills=[],
                matched_experience=[],
                matched_education=[],
                confidence=ConfidenceScore(value=similarity_score, reasoning="Vector similarity score"),
                rationale=f"Match based on vector similarity: {similarity_score:.2f}",
                retrieval_stage="vector_recall"
            )
            job_matches.append(match_result)
        
        return RerankerResult(
            job_matches=job_matches,
            reranker_model="vector_similarity",
            candidates_reranked=len(job_matches),
            processing_time_ms=0
        )
    
    async def _calculate_overall_confidence(self, job_matches: List[JobMatchResult]) -> ConfidenceScore:
        """Calculate overall confidence in results"""
        if not job_matches:
            return ConfidenceScore(value=0.0, reasoning="No matches found")
        
        avg_confidence = sum(match.confidence.value for match in job_matches) / len(job_matches)
        
        return ConfidenceScore(
            value=avg_confidence,
            reasoning=f"Average confidence across {len(job_matches)} matches"
        )
    
    async def _calculate_quality_metrics(self, candidate_profile: CandidateProfile,
                                       job_matches: List[JobMatchResult]) -> QualityMetrics:
        """Calculate quality metrics for the matching process"""
        # Simple quality metrics - can be enhanced
        completeness_score = 1.0 if job_matches else 0.0
        accuracy_score = sum(match.overall_score for match in job_matches) / len(job_matches) if job_matches else 0.0
        consistency_score = 0.8  # Placeholder
        freshness_score = 1.0  # Placeholder
        
        overall_quality = (completeness_score + accuracy_score + consistency_score + freshness_score) / 4
        
        return QualityMetrics(
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            freshness_score=freshness_score,
            overall_quality=overall_quality,
            missing_fields=[],
            inconsistent_fields=[],
            low_confidence_facts=[],
            improvement_suggestions=[]
        )
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    # ==================== MOCK DATA METHODS ====================
    # These would be replaced with actual database queries in production
    
    async def _get_all_candidate_ids(self) -> List[str]:
        """Get all candidate IDs (mock implementation)"""
        return [f"candidate_{i}" for i in range(1, 101)]  # Mock 100 candidates
    
    async def _get_candidate_tags(self, candidate_id: str) -> List[str]:
        """Get candidate tags (mock implementation)"""
        # Mock tags based on candidate ID
        return ["python_developer", "senior_level", "remote_preference"]
    
    async def _get_candidate_facts(self, candidate_id: str) -> List[Dict[str, Any]]:
        """Get candidate facts (mock implementation)"""
        # Mock facts based on candidate ID
        return [
            {"fact_type": "skill", "content": {"skill_name": "Python"}},
            {"fact_type": "experience", "content": {"title": "Software Engineer", "company": "Tech Corp"}}
        ]
    
    async def _get_candidate_quality_score(self, candidate_id: str) -> float:
        """Get candidate quality score (mock implementation)"""
        # Mock quality score
        return 0.8
    
    async def _get_candidate_embedding(self, candidate_id: str, embedding_view: EmbeddingView) -> Optional[List[float]]:
        """Get candidate embedding (mock implementation)"""
        # Mock embedding vector
        return [0.1] * 384
    
    async def _get_candidate_profiles(self, candidate_ids: List[str]) -> Dict[str, CandidateProfile]:
        """Get candidate profiles (mock implementation)"""
        profiles = {}
        for candidate_id in candidate_ids:
            # Create mock profile
            profiles[candidate_id] = CandidateProfile(
                candidate_id=candidate_id,
                uid=candidate_id,
                tenant_id="default",
                profile_data={},
                facts=[],
                tags=[],
                embeddings=[]
            )
        return profiles
    
    async def _create_job_embedding(self, job_description: JobDescription) -> List[float]:
        """Create job description embedding (mock implementation)"""
        # Mock embedding vector
        return [0.2] * 384
    
    async def _has_required_tags(self, candidate_tags: List[str], 
                               job_description: JobDescription,
                               filter_tags: List[str]) -> bool:
        """Check if candidate has required tags"""
        # Simple tag matching
        return len(candidate_tags) > 0
    
    async def _has_required_facts(self, candidate_facts: List[Dict[str, Any]],
                                job_description: JobDescription,
                                filter_facts: List[str]) -> bool:
        """Check if candidate has required facts"""
        # Simple fact matching
        return len(candidate_facts) > 0
    
    def get_retrieval_gateway_stats(self) -> Dict[str, Any]:
        """Get retrieval gateway statistics"""
        return {
            "retrieval_configs": list(self.retrieval_configs.keys()),
            "reranker_models": list(self.reranker_models.keys()),
            "match_strategies": [strategy.value for strategy in MatchStrategy],
            "stage_processors": [stage.value for stage in RetrievalStage]
        }


# ==================== SINGLETON INSTANCE ====================

# Global instance for the application
retrieval_gateway = RetrievalGateway()

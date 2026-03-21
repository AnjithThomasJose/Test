"""
Multi-View Embedder - Creates embeddings for different views of candidate data
Implements multi-view embedding generation with confidence scoring
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import hashlib
import os

# Disable progress bars globally before importing sentence_transformers
os.environ['TQDM_DISABLE'] = '1'

# Monkey-patch tqdm to disable all progress bars
try:
    import tqdm
    # Create a no-op tqdm class that does nothing
    class NoOpTqdm:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, *args, **kwargs):
            pass
        def close(self):
            pass
        def __iter__(self):
            return iter([])
        def __call__(self, *args, **kwargs):
            return self
    
    # Replace tqdm.tqdm with our no-op version
    tqdm.tqdm = NoOpTqdm
    # Disable tqdm auto module
    if hasattr(tqdm, 'auto'):
        tqdm.auto.tqdm = NoOpTqdm
except (ImportError, AttributeError):
    # tqdm not installed or doesn't have expected attributes, nothing to patch
    pass

from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .candidate_matching_models import (
    MultiViewEmbedding, ConfidenceScore, Provenance, EmbeddingView, DataSource,
    StructuredFact, DynamicTag, CandidateProfile
)
from .config import EMBEDDING_MODEL, EMBEDDING_DIMENSION

log = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str
    dimension: int
    confidence_threshold: float
    batch_size: int
    max_length: int


@dataclass
class EmbeddingContext:
    """Context for embedding generation"""
    candidate_id: str
    facts: List[StructuredFact]
    tags: List[DynamicTag]
    profile_data: Dict[str, Any]
    source_data: Dict[str, Any]
    generation_timestamp: datetime


class MultiViewEmbedder:
    """
    Creates multi-view embeddings for candidate data
    Implements different embedding views with confidence scoring
    """
    
    def __init__(self):
        self.embedding_configs = self._initialize_embedding_configs()
        self.embedding_models = self._initialize_embedding_models()
        self.view_processors = self._initialize_view_processors()
        self.confidence_weights = self._initialize_confidence_weights()
        
    def _initialize_embedding_configs(self) -> Dict[EmbeddingView, EmbeddingConfig]:
        """Initialize embedding configurations for different views.
        
        All views use the centralized EMBEDDING_MODEL and EMBEDDING_DIMENSION
        from core/config.py to ensure consistency across the codebase.
        """
        return {
            EmbeddingView.RESUME_CONTENT: EmbeddingConfig(
                model_name=EMBEDDING_MODEL,
                dimension=EMBEDDING_DIMENSION,
                confidence_threshold=0.8,
                batch_size=32,
                max_length=512
            ),
            EmbeddingView.SKILL_FOCUSED: EmbeddingConfig(
                model_name=EMBEDDING_MODEL,
                dimension=EMBEDDING_DIMENSION,
                confidence_threshold=0.9,
                batch_size=64,
                max_length=256
            ),
            EmbeddingView.EXPERIENCE_FOCUSED: EmbeddingConfig(
                model_name=EMBEDDING_MODEL,
                dimension=EMBEDDING_DIMENSION,
                confidence_threshold=0.8,
                batch_size=32,
                max_length=512
            ),
            EmbeddingView.ASSESSMENT_RESPONSES: EmbeddingConfig(
                model_name=EMBEDDING_MODEL,
                dimension=EMBEDDING_DIMENSION,
                confidence_threshold=0.7,
                batch_size=32,
                max_length=512
            ),
            EmbeddingView.CHAT_CONTEXT: EmbeddingConfig(
                model_name=EMBEDDING_MODEL,
                dimension=EMBEDDING_DIMENSION,
                confidence_threshold=0.6,
                batch_size=32,
                max_length=512
            ),
            EmbeddingView.COMPREHENSIVE: EmbeddingConfig(
                model_name=EMBEDDING_MODEL,
                dimension=EMBEDDING_DIMENSION,
                confidence_threshold=0.8,
                batch_size=16,
                max_length=1024
            )
        }
    
    def _initialize_embedding_models(self) -> Dict[str, SentenceTransformer]:
        """Initialize embedding models using centralized config."""
        models = {}
        model_names = set(config.model_name for config in self.embedding_configs.values())
        
        for model_name in model_names:
            try:
                # Disable progress bars to reduce log noise
                models[model_name] = SentenceTransformer(model_name)
                # Disable progress bars after initialization (if method exists)
                if hasattr(models[model_name], 'set_show_progress_bar'):
                    models[model_name].set_show_progress_bar(False)
                log.info(f"Loaded embedding model: {model_name} (progress bars disabled)")
            except Exception as e:
                log.error(f"Failed to load model {model_name}: {e}")
                # Fallback to centralized default model
                try:
                    models[model_name] = SentenceTransformer(EMBEDDING_MODEL)
                    if hasattr(models[model_name], 'set_show_progress_bar'):
                        models[model_name].set_show_progress_bar(False)
                except Exception as e2:
                    log.error(f"Failed to load fallback model: {e2}")
        
        return models
    
    def _initialize_view_processors(self) -> Dict[EmbeddingView, callable]:
        """Initialize view processors for different embedding views"""
        return {
            EmbeddingView.RESUME_CONTENT: self._process_resume_content,
            EmbeddingView.SKILL_FOCUSED: self._process_skill_focused,
            EmbeddingView.EXPERIENCE_FOCUSED: self._process_experience_focused,
            EmbeddingView.ASSESSMENT_RESPONSES: self._process_assessment_responses,
            EmbeddingView.CHAT_CONTEXT: self._process_chat_context,
            EmbeddingView.COMPREHENSIVE: self._process_comprehensive
        }
    
    def _initialize_confidence_weights(self) -> Dict[str, float]:
        """Initialize confidence weights for embedding generation"""
        return {
            "content_quality_weight": 0.3,
            "source_reliability_weight": 0.2,
            "completeness_weight": 0.2,
            "consistency_weight": 0.2,
            "model_confidence_weight": 0.1
        }
    
    async def generate_embeddings(self, context: EmbeddingContext) -> List[MultiViewEmbedding]:
        """
        Generate multi-view embeddings for candidate data
        """
        log.info(f"Generating embeddings for candidate {context.candidate_id}")
        
        embeddings = []
        
        # Generate embeddings for each view
        for view_type, processor in self.view_processors.items():
            try:
                embedding = await processor(context, view_type)
                if embedding:
                    embeddings.append(embedding)
                    log.info(f"Generated {view_type.value} embedding for candidate {context.candidate_id}")
            except Exception as e:
                log.error(f"Failed to generate {view_type.value} embedding: {e}")
        
        # Validate and score embeddings
        validated_embeddings = []
        for embedding in embeddings:
            validated_embedding = await self._validate_and_score_embedding(embedding, context)
            if validated_embedding:
                validated_embeddings.append(validated_embedding)
        
        log.info(f"Generated {len(validated_embeddings)} embeddings for candidate {context.candidate_id}")
        return validated_embeddings
    
    async def _process_resume_content(self, context: EmbeddingContext, view_type: EmbeddingView) -> Optional[MultiViewEmbedding]:
        """Process resume content for embedding"""
        # Extract resume text from profile data
        resume_text = self._extract_resume_text(context.profile_data)
        
        if not resume_text:
            return None
        
        # Generate embedding
        config = self.embedding_configs[view_type]
        model = self.embedding_models[config.model_name]
        
        # Truncate text if too long
        if len(resume_text) > config.max_length:
            resume_text = resume_text[:config.max_length]
        
        embedding_vector = model.encode(resume_text)
        
        # Calculate confidence based on content quality
        confidence = await self._calculate_content_confidence(resume_text, context)
        
        return MultiViewEmbedding(
            view_type=view_type,
            vector=embedding_vector.tolist(),
            confidence=confidence,
            source_data={"resume_text": resume_text},
            metadata={
                "candidate_id": context.candidate_id,
                "text_length": len(resume_text),
                "model_name": config.model_name,
                "generation_timestamp": context.generation_timestamp.isoformat()
            }
        )
    
    async def _process_skill_focused(self, context: EmbeddingContext, view_type: EmbeddingView) -> Optional[MultiViewEmbedding]:
        """Process skill-focused embedding"""
        # Extract skills from facts
        skill_facts = [f for f in context.facts if f.fact_type.value == "skill"]
        
        if not skill_facts:
            return None
        
        # Create skill-focused text
        skill_text = self._create_skill_focused_text(skill_facts)
        
        # Generate embedding
        config = self.embedding_configs[view_type]
        model = self.embedding_models[config.model_name]
        
        embedding_vector = model.encode(skill_text)
        
        # Calculate confidence based on skill data quality
        confidence = await self._calculate_skill_confidence(skill_facts, context)
        
        return MultiViewEmbedding(
            view_type=view_type,
            vector=embedding_vector.tolist(),
            confidence=confidence,
            source_data={"skill_text": skill_text, "skill_facts": [f.fact_id for f in skill_facts]},
            metadata={
                "candidate_id": context.candidate_id,
                "skill_count": len(skill_facts),
                "model_name": config.model_name,
                "generation_timestamp": context.generation_timestamp.isoformat()
            }
        )
    
    async def _process_experience_focused(self, context: EmbeddingContext, view_type: EmbeddingView) -> Optional[MultiViewEmbedding]:
        """Process experience-focused embedding"""
        # Extract experience from facts
        experience_facts = [f for f in context.facts if f.fact_type.value == "experience"]
        
        if not experience_facts:
            return None
        
        # Create experience-focused text
        experience_text = self._create_experience_focused_text(experience_facts)
        
        # Generate embedding
        config = self.embedding_configs[view_type]
        model = self.embedding_models[config.model_name]
        
        embedding_vector = model.encode(experience_text)
        
        # Calculate confidence based on experience data quality
        confidence = await self._calculate_experience_confidence(experience_facts, context)
        
        return MultiViewEmbedding(
            view_type=view_type,
            vector=embedding_vector.tolist(),
            confidence=confidence,
            source_data={"experience_text": experience_text, "experience_facts": [f.fact_id for f in experience_facts]},
            metadata={
                "candidate_id": context.candidate_id,
                "experience_count": len(experience_facts),
                "model_name": config.model_name,
                "generation_timestamp": context.generation_timestamp.isoformat()
            }
        )
    
    async def _process_assessment_responses(self, context: EmbeddingContext, view_type: EmbeddingView) -> Optional[MultiViewEmbedding]:
        """Process assessment responses embedding"""
        # Extract assessment data from source data
        assessment_data = context.source_data.get("assessment_data", {})
        
        if not assessment_data:
            return None
        
        # Create assessment-focused text
        assessment_text = self._create_assessment_focused_text(assessment_data)
        
        # Generate embedding
        config = self.embedding_configs[view_type]
        model = self.embedding_models[config.model_name]
        
        embedding_vector = model.encode(assessment_text)
        
        # Calculate confidence based on assessment data quality
        confidence = await self._calculate_assessment_confidence(assessment_data, context)
        
        return MultiViewEmbedding(
            view_type=view_type,
            vector=embedding_vector.tolist(),
            confidence=confidence,
            source_data={"assessment_text": assessment_text, "assessment_data": assessment_data},
            metadata={
                "candidate_id": context.candidate_id,
                "assessment_type": assessment_data.get("assessment_type", "unknown"),
                "model_name": config.model_name,
                "generation_timestamp": context.generation_timestamp.isoformat()
            }
        )
    
    async def _process_chat_context(self, context: EmbeddingContext, view_type: EmbeddingView) -> Optional[MultiViewEmbedding]:
        """Process chat context embedding"""
        # Extract chat data from source data
        chat_data = context.source_data.get("chat_data", {})
        
        if not chat_data:
            return None
        
        # Create chat-focused text
        chat_text = self._create_chat_focused_text(chat_data)
        
        # Generate embedding
        config = self.embedding_configs[view_type]
        model = self.embedding_models[config.model_name]
        
        embedding_vector = model.encode(chat_text)
        
        # Calculate confidence based on chat data quality
        confidence = await self._calculate_chat_confidence(chat_data, context)
        
        return MultiViewEmbedding(
            view_type=view_type,
            vector=embedding_vector.tolist(),
            confidence=confidence,
            source_data={"chat_text": chat_text, "chat_data": chat_data},
            metadata={
                "candidate_id": context.candidate_id,
                "chat_length": len(chat_text),
                "model_name": config.model_name,
                "generation_timestamp": context.generation_timestamp.isoformat()
            }
        )
    
    async def _process_comprehensive(self, context: EmbeddingContext, view_type: EmbeddingView) -> Optional[MultiViewEmbedding]:
        """Process comprehensive embedding combining all data"""
        # Create comprehensive text from all sources
        comprehensive_text = self._create_comprehensive_text(context)
        
        if not comprehensive_text:
            return None
        
        # Generate embedding
        config = self.embedding_configs[view_type]
        model = self.embedding_models[config.model_name]
        
        # Truncate text if too long
        if len(comprehensive_text) > config.max_length:
            comprehensive_text = comprehensive_text[:config.max_length]
        
        embedding_vector = model.encode(comprehensive_text)
        
        # Calculate confidence based on overall data quality
        confidence = await self._calculate_comprehensive_confidence(context)
        
        return MultiViewEmbedding(
            view_type=view_type,
            vector=embedding_vector.tolist(),
            confidence=confidence,
            source_data={"comprehensive_text": comprehensive_text},
            metadata={
                "candidate_id": context.candidate_id,
                "text_length": len(comprehensive_text),
                "fact_count": len(context.facts),
                "tag_count": len(context.tags),
                "model_name": config.model_name,
                "generation_timestamp": context.generation_timestamp.isoformat()
            }
        )
    
    def _extract_resume_text(self, profile_data: Dict[str, Any]) -> str:
        """Extract resume text from profile data"""
        text_parts = []
        
        # Add name
        if profile_data.get("Name"):
            text_parts.append(f"Candidate: {profile_data['Name']}")
        
        # Add skills
        skills = profile_data.get("Skills", [])
        if skills:
            skill_text = "Skills: "
            if isinstance(skills, list):
                skill_list = []
                for skill in skills:
                    if isinstance(skill, dict):
                        skill_list.append(skill.get("skill", skill.get("name", "")))
                    else:
                        skill_list.append(str(skill))
                skill_text += ", ".join(filter(None, skill_list))
            else:
                skill_text += str(skills)
            text_parts.append(skill_text)
        
        # Add work experience
        work_exp = profile_data.get("WorkExperience", [])
        if work_exp:
            exp_text = "Experience: "
            exp_parts = []
            for job in work_exp:
                if isinstance(job, dict):
                    title = job.get("title", job.get("position", ""))
                    company = job.get("company", "")
                    description = job.get("description", "")
                    if title:
                        exp_parts.append(f"{title} at {company}" if company else title)
                    if description:
                        exp_parts.append(description)
                else:
                    exp_parts.append(str(job))
            exp_text += " | ".join(exp_parts)
            text_parts.append(exp_text)
        
        # Add education
        education = profile_data.get("Education", [])
        if education:
            edu_text = "Education: "
            edu_parts = []
            for edu in education:
                if isinstance(edu, dict):
                    degree = edu.get("degree", "")
                    major = edu.get("major", "")
                    institution = edu.get("institution", "")
                    if degree or major:
                        edu_parts.append(f"{degree} {major}".strip())
                    if institution:
                        edu_parts.append(institution)
                else:
                    edu_parts.append(str(edu))
            edu_text += " | ".join(edu_parts)
            text_parts.append(edu_text)
        
        return " ".join(text_parts)
    
    def _create_skill_focused_text(self, skill_facts: List[StructuredFact]) -> str:
        """Create skill-focused text from skill facts"""
        skill_parts = []
        
        for fact in skill_facts:
            skill_name = fact.content.get("skill_name", "")
            skill_level = fact.content.get("skill_level", "")
            years_experience = fact.content.get("years_experience", 0)
            
            skill_desc = skill_name
            if skill_level:
                skill_desc += f" ({skill_level})"
            if years_experience:
                skill_desc += f" - {years_experience} years"
            
            skill_parts.append(skill_desc)
        
        return "Skills: " + ", ".join(skill_parts)
    
    def _create_experience_focused_text(self, experience_facts: List[StructuredFact]) -> str:
        """Create experience-focused text from experience facts"""
        exp_parts = []
        
        for fact in experience_facts:
            title = fact.content.get("title", "")
            company = fact.content.get("company", "")
            description = fact.content.get("description", "")
            duration = fact.content.get("duration", "")
            
            exp_desc = f"{title} at {company}" if title and company else title or company
            if duration:
                exp_desc += f" ({duration})"
            if description:
                exp_desc += f": {description}"
            
            exp_parts.append(exp_desc)
        
        return "Experience: " + " | ".join(exp_parts)
    
    def _create_assessment_focused_text(self, assessment_data: Dict[str, Any]) -> str:
        """Create assessment-focused text from assessment data"""
        text_parts = []
        
        # Add skill scores
        skill_scores = assessment_data.get("skill_scores", {})
        if skill_scores:
            skill_text = "Assessment Skills: "
            skill_parts = [f"{skill} ({score})" for skill, score in skill_scores.items()]
            skill_text += ", ".join(skill_parts)
            text_parts.append(skill_text)
        
        # Add personality traits
        personality_traits = assessment_data.get("personality_traits", {})
        if personality_traits:
            trait_text = "Personality: "
            trait_parts = [f"{trait} ({value})" for trait, value in personality_traits.items()]
            trait_text += ", ".join(trait_parts)
            text_parts.append(trait_text)
        
        # Add overall score
        overall_score = assessment_data.get("overall_score", 0)
        if overall_score:
            text_parts.append(f"Overall Assessment Score: {overall_score}")
        
        return " ".join(text_parts)
    
    def _create_chat_focused_text(self, chat_data: Dict[str, Any]) -> str:
        """Create chat-focused text from chat data"""
        text_parts = []
        
        # Add conversation text
        conversation_text = chat_data.get("conversation_text", "")
        if conversation_text:
            text_parts.append(f"Conversation: {conversation_text}")
        
        # Add interests
        interests = chat_data.get("interests", [])
        if interests:
            interest_text = "Interests: " + ", ".join(interests)
            text_parts.append(interest_text)
        
        # Add preferences
        preferences = chat_data.get("preferences", {})
        if preferences:
            pref_text = "Preferences: " + ", ".join([f"{k}: {v}" for k, v in preferences.items()])
            text_parts.append(pref_text)
        
        return " ".join(text_parts)
    
    def _create_comprehensive_text(self, context: EmbeddingContext) -> str:
        """Create comprehensive text from all data sources"""
        text_parts = []
        
        # Add resume content
        resume_text = self._extract_resume_text(context.profile_data)
        if resume_text:
            text_parts.append(resume_text)
        
        # Add skill-focused text
        skill_facts = [f for f in context.facts if f.fact_type.value == "skill"]
        if skill_facts:
            skill_text = self._create_skill_focused_text(skill_facts)
            text_parts.append(skill_text)
        
        # Add experience-focused text
        experience_facts = [f for f in context.facts if f.fact_type.value == "experience"]
        if experience_facts:
            experience_text = self._create_experience_focused_text(experience_facts)
            text_parts.append(experience_text)
        
        # Add assessment data
        assessment_data = context.source_data.get("assessment_data", {})
        if assessment_data:
            assessment_text = self._create_assessment_focused_text(assessment_data)
            text_parts.append(assessment_text)
        
        # Add chat data
        chat_data = context.source_data.get("chat_data", {})
        if chat_data:
            chat_text = self._create_chat_focused_text(chat_data)
            text_parts.append(chat_text)
        
        # Add tags
        if context.tags:
            tag_text = "Tags: " + ", ".join([f"{tag.category.value}: {tag.value}" for tag in context.tags])
            text_parts.append(tag_text)
        
        return " ".join(text_parts)
    
    async def _calculate_content_confidence(self, text: str, context: EmbeddingContext) -> ConfidenceScore:
        """Calculate confidence for content-based embedding"""
        # Base confidence on text quality
        text_quality = min(1.0, len(text) / 1000)  # Normalize by length
        
        # Adjust based on source reliability
        source_reliability = 0.9  # Resume is highly reliable
        
        # Calculate final confidence
        confidence = text_quality * source_reliability
        
        return ConfidenceScore(
            value=confidence,
            reasoning=f"Content quality: {text_quality:.2f}, Source reliability: {source_reliability:.2f}"
        )
    
    async def _calculate_skill_confidence(self, skill_facts: List[StructuredFact], context: EmbeddingContext) -> ConfidenceScore:
        """Calculate confidence for skill-focused embedding"""
        if not skill_facts:
            return ConfidenceScore(value=0.0, reasoning="No skill facts available")
        
        # Average confidence of skill facts
        avg_fact_confidence = sum(f.confidence.value for f in skill_facts) / len(skill_facts)
        
        # Bonus for skill count
        skill_count_bonus = min(0.2, len(skill_facts) * 0.02)
        
        confidence = min(1.0, avg_fact_confidence + skill_count_bonus)
        
        return ConfidenceScore(
            value=confidence,
            reasoning=f"Average skill fact confidence: {avg_fact_confidence:.2f}, Skill count bonus: {skill_count_bonus:.2f}"
        )
    
    async def _calculate_experience_confidence(self, experience_facts: List[StructuredFact], context: EmbeddingContext) -> ConfidenceScore:
        """Calculate confidence for experience-focused embedding"""
        if not experience_facts:
            return ConfidenceScore(value=0.0, reasoning="No experience facts available")
        
        # Average confidence of experience facts
        avg_fact_confidence = sum(f.confidence.value for f in experience_facts) / len(experience_facts)
        
        # Bonus for experience count
        exp_count_bonus = min(0.2, len(experience_facts) * 0.05)
        
        confidence = min(1.0, avg_fact_confidence + exp_count_bonus)
        
        return ConfidenceScore(
            value=confidence,
            reasoning=f"Average experience fact confidence: {avg_fact_confidence:.2f}, Experience count bonus: {exp_count_bonus:.2f}"
        )
    
    async def _calculate_assessment_confidence(self, assessment_data: Dict[str, Any], context: EmbeddingContext) -> ConfidenceScore:
        """Calculate confidence for assessment embedding"""
        # Base confidence on assessment completeness
        completeness_score = 0.0
        
        if assessment_data.get("skill_scores"):
            completeness_score += 0.4
        if assessment_data.get("personality_traits"):
            completeness_score += 0.3
        if assessment_data.get("overall_score"):
            completeness_score += 0.3
        
        # Adjust based on source reliability
        source_reliability = 0.8  # Assessment is reliable but not as much as resume
        
        confidence = completeness_score * source_reliability
        
        return ConfidenceScore(
            value=confidence,
            reasoning=f"Assessment completeness: {completeness_score:.2f}, Source reliability: {source_reliability:.2f}"
        )
    
    async def _calculate_chat_confidence(self, chat_data: Dict[str, Any], context: EmbeddingContext) -> ConfidenceScore:
        """Calculate confidence for chat embedding"""
        # Base confidence on chat content quality
        conversation_text = chat_data.get("conversation_text", "")
        text_quality = min(1.0, len(conversation_text) / 500)  # Normalize by length
        
        # Adjust based on source reliability
        source_reliability = 0.6  # Chat is less reliable than structured data
        
        confidence = text_quality * source_reliability
        
        return ConfidenceScore(
            value=confidence,
            reasoning=f"Chat text quality: {text_quality:.2f}, Source reliability: {source_reliability:.2f}"
        )
    
    async def _calculate_comprehensive_confidence(self, context: EmbeddingContext) -> ConfidenceScore:
        """Calculate confidence for comprehensive embedding"""
        # Calculate confidence based on all data sources
        confidences = []
        
        # Resume confidence
        resume_text = self._extract_resume_text(context.profile_data)
        if resume_text:
            resume_conf = await self._calculate_content_confidence(resume_text, context)
            confidences.append(resume_conf.value)
        
        # Skill confidence
        skill_facts = [f for f in context.facts if f.fact_type.value == "skill"]
        if skill_facts:
            skill_conf = await self._calculate_skill_confidence(skill_facts, context)
            confidences.append(skill_conf.value)
        
        # Experience confidence
        experience_facts = [f for f in context.facts if f.fact_type.value == "experience"]
        if experience_facts:
            exp_conf = await self._calculate_experience_confidence(experience_facts, context)
            confidences.append(exp_conf.value)
        
        # Assessment confidence
        assessment_data = context.source_data.get("assessment_data", {})
        if assessment_data:
            assessment_conf = await self._calculate_assessment_confidence(assessment_data, context)
            confidences.append(assessment_conf.value)
        
        # Chat confidence
        chat_data = context.source_data.get("chat_data", {})
        if chat_data:
            chat_conf = await self._calculate_chat_confidence(chat_data, context)
            confidences.append(chat_conf.value)
        
        if not confidences:
            return ConfidenceScore(value=0.0, reasoning="No data sources available")
        
        # Calculate weighted average
        avg_confidence = sum(confidences) / len(confidences)
        
        # Bonus for data diversity
        diversity_bonus = min(0.2, len(confidences) * 0.05)
        
        confidence = min(1.0, avg_confidence + diversity_bonus)
        
        return ConfidenceScore(
            value=confidence,
            reasoning=f"Average confidence: {avg_confidence:.2f}, Diversity bonus: {diversity_bonus:.2f}"
        )
    
    async def _validate_and_score_embedding(self, embedding: MultiViewEmbedding, context: EmbeddingContext) -> Optional[MultiViewEmbedding]:
        """Validate and score an embedding"""
        # Check if embedding vector is valid
        if not embedding.vector or len(embedding.vector) == 0:
            log.warning(f"Invalid embedding vector for {embedding.embedding_id}")
            return None
        
        # Check if confidence meets threshold
        config = self.embedding_configs[embedding.view_type]
        if embedding.confidence.value < config.confidence_threshold:
            log.warning(f"Low confidence embedding for {embedding.embedding_id}: {embedding.confidence.value}")
            return None
        
        # Check if vector dimension matches expected
        expected_dim = config.dimension
        if len(embedding.vector) != expected_dim:
            log.warning(f"Dimension mismatch for {embedding.embedding_id}: expected {expected_dim}, got {len(embedding.vector)}")
            return None
        
        return embedding
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        return {
            "embedding_configs": {
                view.value: {
                    "model_name": config.model_name,
                    "dimension": config.dimension,
                    "confidence_threshold": config.confidence_threshold
                }
                for view, config in self.embedding_configs.items()
            },
            "loaded_models": list(self.embedding_models.keys()),
            "confidence_weights": self.confidence_weights,
            "supported_views": [view.value for view in EmbeddingView]
        }


# ==================== SINGLETON INSTANCE ====================

# Global instance for the application
multi_view_embedder = MultiViewEmbedder()

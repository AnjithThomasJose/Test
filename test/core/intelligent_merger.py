"""
Intelligent Merger - Multi-source data fusion with conflict resolution
Merges data from multiple sources with intelligent conflict resolution
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

from .candidate_matching_models import (
    CandidateProfile, StructuredFact, DynamicTag, MultiViewEmbedding,
    ConfidenceScore, Provenance, DataSource, FactType, TagCategory,
    EmbeddingView, QualityMetrics
)

log = logging.getLogger(__name__)


@dataclass
class MergeContext:
    """Context for data merging"""
    candidate_id: str
    source_data_sets: Dict[DataSource, Dict[str, Any]]
    facts_by_source: Dict[DataSource, List[StructuredFact]]
    tags_by_source: Dict[DataSource, List[DynamicTag]]
    embeddings_by_source: Dict[DataSource, List[MultiViewEmbedding]]
    merge_timestamp: datetime


@dataclass
class ConflictResolution:
    """Conflict resolution strategy"""
    strategy: str  # "highest_confidence", "most_recent", "source_priority", "merge_content"
    source_priority: List[DataSource]
    confidence_threshold: float


class IntelligentMerger:
    """
    Intelligently merges data from multiple sources
    Implements conflict resolution and data fusion strategies
    """
    
    def __init__(self):
        self.conflict_resolution = self._initialize_conflict_resolution()
        self.source_priorities = self._initialize_source_priorities()
        self.merge_strategies = self._initialize_merge_strategies()
        self.quality_weights = self._initialize_quality_weights()
        
    def _initialize_conflict_resolution(self) -> Dict[str, ConflictResolution]:
        """Initialize conflict resolution strategies"""
        return {
            "facts": ConflictResolution(
                strategy="highest_confidence",
                source_priority=[DataSource.RESUME, DataSource.ASSESSMENT, DataSource.INTERVIEW, DataSource.CHAT_SESSION],
                confidence_threshold=0.5
            ),
            "tags": ConflictResolution(
                strategy="merge_content",
                source_priority=[DataSource.RESUME, DataSource.ASSESSMENT, DataSource.INTERVIEW, DataSource.CHAT_SESSION],
                confidence_threshold=0.3
            ),
            "embeddings": ConflictResolution(
                strategy="highest_confidence",
                source_priority=[DataSource.RESUME, DataSource.ASSESSMENT, DataSource.INTERVIEW, DataSource.CHAT_SESSION],
                confidence_threshold=0.6
            )
        }
    
    def _initialize_source_priorities(self) -> Dict[DataSource, int]:
        """Initialize source priorities (higher number = higher priority)"""
        return {
            DataSource.RESUME: 100,
            DataSource.ASSESSMENT: 80,
            DataSource.INTERVIEW: 70,
            DataSource.LINKEDIN: 60,
            DataSource.PORTFOLIO: 50,
            DataSource.CHAT_SESSION: 40
        }
    
    def _initialize_merge_strategies(self) -> Dict[str, callable]:
        """Initialize merge strategies for different data types"""
        return {
            "highest_confidence": self._merge_by_highest_confidence,
            "most_recent": self._merge_by_most_recent,
            "source_priority": self._merge_by_source_priority,
            "merge_content": self._merge_content,
            "weighted_average": self._merge_by_weighted_average
        }
    
    def _initialize_quality_weights(self) -> Dict[str, float]:
        """Initialize quality weights for merging"""
        return {
            "confidence_weight": 0.4,
            "completeness_weight": 0.3,
            "consistency_weight": 0.2,
            "freshness_weight": 0.1
        }
    
    async def merge_candidate_data(self, context: MergeContext) -> CandidateProfile:
        """
        Merge candidate data from multiple sources
        """
        log.info(f"Merging data for candidate {context.candidate_id}")
        
        # Merge facts
        merged_facts = await self._merge_facts(context)
        
        # Merge tags
        merged_tags = await self._merge_tags(context)
        
        # Merge embeddings
        merged_embeddings = await self._merge_embeddings(context)
        
        # Create merged profile
        merged_profile = await self._create_merged_profile(
            context, merged_facts, merged_tags, merged_embeddings
        )
        
        # Calculate quality metrics
        quality_metrics = await self._calculate_quality_metrics(merged_profile, context)
        merged_profile.completeness_score = quality_metrics.completeness_score
        merged_profile.quality_score = quality_metrics.overall_quality
        
        log.info(f"Successfully merged data for candidate {context.candidate_id}")
        return merged_profile
    
    async def _merge_facts(self, context: MergeContext) -> List[StructuredFact]:
        """Merge facts from multiple sources"""
        all_facts = []
        
        # Collect all facts
        for source, facts in context.facts_by_source.items():
            all_facts.extend(facts)
        
        if not all_facts:
            return []
        
        # Group facts by signature for deduplication
        fact_groups = self._group_facts_by_signature(all_facts)
        
        # Merge facts in each group
        merged_facts = []
        for signature, facts in fact_groups.items():
            if len(facts) == 1:
                merged_facts.append(facts[0])
            else:
                merged_fact = await self._merge_duplicate_facts(facts, "facts")
                if merged_fact:
                    merged_facts.append(merged_fact)
        
        log.info(f"Merged {len(all_facts)} facts into {len(merged_facts)} unique facts")
        return merged_facts
    
    async def _merge_tags(self, context: MergeContext) -> List[DynamicTag]:
        """Merge tags from multiple sources"""
        all_tags = []
        
        # Collect all tags
        for source, tags in context.tags_by_source.items():
            all_tags.extend(tags)
        
        if not all_tags:
            return []
        
        # Group tags by signature for deduplication
        tag_groups = self._group_tags_by_signature(all_tags)
        
        # Merge tags in each group
        merged_tags = []
        for signature, tags in tag_groups.items():
            if len(tags) == 1:
                merged_tags.append(tags[0])
            else:
                merged_tag = await self._merge_duplicate_tags(tags, "tags")
                if merged_tag:
                    merged_tags.append(merged_tag)
        
        log.info(f"Merged {len(all_tags)} tags into {len(merged_tags)} unique tags")
        return merged_tags
    
    async def _merge_embeddings(self, context: MergeContext) -> List[MultiViewEmbedding]:
        """Merge embeddings from multiple sources"""
        all_embeddings = []
        
        # Collect all embeddings
        for source, embeddings in context.embeddings_by_source.items():
            all_embeddings.extend(embeddings)
        
        if not all_embeddings:
            return []
        
        # Group embeddings by view type
        embedding_groups = self._group_embeddings_by_view(all_embeddings)
        
        # Merge embeddings in each group
        merged_embeddings = []
        for view_type, embeddings in embedding_groups.items():
            if len(embeddings) == 1:
                merged_embeddings.append(embeddings[0])
            else:
                merged_embedding = await self._merge_duplicate_embeddings(embeddings, "embeddings")
                if merged_embedding:
                    merged_embeddings.append(merged_embedding)
        
        log.info(f"Merged {len(all_embeddings)} embeddings into {len(merged_embeddings)} unique embeddings")
        return merged_embeddings
    
    def _group_facts_by_signature(self, facts: List[StructuredFact]) -> Dict[str, List[StructuredFact]]:
        """Group facts by signature for deduplication"""
        groups = defaultdict(list)
        
        for fact in facts:
            signature = self._create_fact_signature(fact)
            groups[signature].append(fact)
        
        return dict(groups)
    
    def _group_tags_by_signature(self, tags: List[DynamicTag]) -> Dict[str, List[DynamicTag]]:
        """Group tags by signature for deduplication"""
        groups = defaultdict(list)
        
        for tag in tags:
            signature = f"{tag.category.value}:{tag.value}"
            groups[signature].append(tag)
        
        return dict(groups)
    
    def _group_embeddings_by_view(self, embeddings: List[MultiViewEmbedding]) -> Dict[EmbeddingView, List[MultiViewEmbedding]]:
        """Group embeddings by view type"""
        groups = defaultdict(list)
        
        for embedding in embeddings:
            groups[embedding.view_type].append(embedding)
        
        return dict(groups)
    
    def _create_fact_signature(self, fact: StructuredFact) -> str:
        """Create a signature for fact deduplication"""
        if fact.fact_type == FactType.SKILL:
            return f"skill:{fact.content.get('skill_name', '').lower()}"
        elif fact.fact_type == FactType.EXPERIENCE:
            return f"exp:{fact.content.get('title', '').lower()}:{fact.content.get('company', '').lower()}"
        elif fact.fact_type == FactType.EDUCATION:
            return f"edu:{fact.content.get('degree', '').lower()}:{fact.content.get('institution', '').lower()}"
        elif fact.fact_type == FactType.CERTIFICATION:
            return f"cert:{fact.content.get('certification_name', '').lower()}"
        elif fact.fact_type == FactType.PROJECT:
            return f"proj:{fact.content.get('project_name', '').lower()}"
        elif fact.fact_type == FactType.ACHIEVEMENT:
            return f"ach:{fact.content.get('achievement_title', '').lower()}"
        elif fact.fact_type == FactType.INTEREST:
            return f"int:{fact.content.get('interest_name', '').lower()}"
        elif fact.fact_type == FactType.LOCATION:
            return f"loc:{fact.content.get('location_name', '').lower()}"
        elif fact.fact_type == FactType.LANGUAGE:
            return f"lang:{fact.content.get('language_name', '').lower()}"
        elif fact.fact_type == FactType.SOFT_SKILL:
            return f"soft:{fact.content.get('skill_name', '').lower()}"
        
        return f"other:{fact.fact_id}"
    
    async def _merge_duplicate_facts(self, facts: List[StructuredFact], data_type: str) -> Optional[StructuredFact]:
        """Merge duplicate facts using conflict resolution strategy"""
        resolution = self.conflict_resolution[data_type]
        strategy = self.merge_strategies[resolution.strategy]
        
        return await strategy(facts, resolution)
    
    async def _merge_duplicate_tags(self, tags: List[DynamicTag], data_type: str) -> Optional[DynamicTag]:
        """Merge duplicate tags using conflict resolution strategy"""
        resolution = self.conflict_resolution[data_type]
        strategy = self.merge_strategies[resolution.strategy]
        
        return await strategy(tags, resolution)
    
    async def _merge_duplicate_embeddings(self, embeddings: List[MultiViewEmbedding], data_type: str) -> Optional[MultiViewEmbedding]:
        """Merge duplicate embeddings using conflict resolution strategy"""
        resolution = self.conflict_resolution[data_type]
        strategy = self.merge_strategies[resolution.strategy]
        
        return await strategy(embeddings, resolution)
    
    async def _merge_by_highest_confidence(self, items: List[Any], resolution: ConflictResolution) -> Optional[Any]:
        """Merge by selecting item with highest confidence"""
        if not items:
            return None
        
        # Filter by confidence threshold
        valid_items = [item for item in items if item.confidence.value >= resolution.confidence_threshold]
        
        if not valid_items:
            return None
        
        # Select item with highest confidence
        best_item = max(valid_items, key=lambda x: x.confidence.value)
        
        # Update provenance to reflect merging
        if hasattr(best_item, 'provenance'):
            best_item.provenance = Provenance(
                source=best_item.provenance.source,
                source_id=best_item.provenance.source_id,
                extraction_method=f"merged_{resolution.strategy}",
                timestamp=datetime.utcnow()
            )
        
        return best_item
    
    async def _merge_by_most_recent(self, items: List[Any], resolution: ConflictResolution) -> Optional[Any]:
        """Merge by selecting most recent item"""
        if not items:
            return None
        
        # Select item with most recent timestamp
        most_recent = max(items, key=lambda x: x.provenance.timestamp)
        
        # Update provenance
        most_recent.provenance = Provenance(
            source=most_recent.provenance.source,
            source_id=most_recent.provenance.source_id,
            extraction_method=f"merged_{resolution.strategy}",
            timestamp=datetime.utcnow()
        )
        
        return most_recent
    
    async def _merge_by_source_priority(self, items: List[Any], resolution: ConflictResolution) -> Optional[Any]:
        """Merge by selecting item from highest priority source"""
        if not items:
            return None
        
        # Sort by source priority
        sorted_items = sorted(items, key=lambda x: self.source_priorities.get(x.provenance.source, 0), reverse=True)
        
        best_item = sorted_items[0]
        
        # Update provenance
        best_item.provenance = Provenance(
            source=best_item.provenance.source,
            source_id=best_item.provenance.source_id,
            extraction_method=f"merged_{resolution.strategy}",
            timestamp=datetime.utcnow()
        )
        
        return best_item
    
    async def _merge_content(self, items: List[Any], resolution: ConflictResolution) -> Optional[Any]:
        """Merge content from multiple items"""
        if not items:
            return None
        
        if len(items) == 1:
            return items[0]
        
        # Use the item with highest confidence as base
        base_item = max(items, key=lambda x: x.confidence.value)
        
        # Merge content from all items
        merged_content = base_item.content.copy()
        
        for item in items:
            if item != base_item:
                # Merge additional fields
                for key, value in item.content.items():
                    if key not in merged_content or not merged_content[key]:
                        merged_content[key] = value
                    elif isinstance(value, list) and isinstance(merged_content[key], list):
                        # Merge lists
                        merged_content[key] = list(set(merged_content[key] + value))
        
        # Calculate merged confidence
        avg_confidence = sum(item.confidence.value for item in items) / len(items)
        merged_confidence = ConfidenceScore(
            value=avg_confidence,
            reasoning=f"Merged from {len(items)} sources with average confidence {avg_confidence:.2f}"
        )
        
        # Create merged item
        merged_item = type(base_item)(
            **{**base_item.__dict__, 'content': merged_content, 'confidence': merged_confidence}
        )
        
        # Update provenance
        merged_item.provenance = Provenance(
            source=base_item.provenance.source,
            source_id=base_item.provenance.source_id,
            extraction_method=f"merged_{resolution.strategy}",
            timestamp=datetime.utcnow()
        )
        
        return merged_item
    
    async def _merge_by_weighted_average(self, items: List[Any], resolution: ConflictResolution) -> Optional[Any]:
        """Merge by weighted average (for embeddings)"""
        if not items:
            return None
        
        if len(items) == 1:
            return items[0]
        
        # Calculate weights based on confidence
        weights = [item.confidence.value for item in items]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return items[0]  # Fallback to first item
        
        # Normalize weights
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted average for embeddings
        if hasattr(items[0], 'vector'):
            vector_dim = len(items[0].vector)
            weighted_vector = [0.0] * vector_dim
            
            for item, weight in zip(items, normalized_weights):
                for i in range(vector_dim):
                    weighted_vector[i] += item.vector[i] * weight
            
            # Create merged embedding
            merged_item = MultiViewEmbedding(
                view_type=items[0].view_type,
                vector=weighted_vector,
                confidence=ConfidenceScore(
                    value=sum(item.confidence.value * weight for item, weight in zip(items, normalized_weights)),
                    reasoning=f"Weighted average from {len(items)} sources"
                ),
                source_data=items[0].source_data,
                metadata={
                    **items[0].metadata,
                    "merged_from": [item.embedding_id for item in items],
                    "merge_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            return merged_item
        
        return items[0]  # Fallback
    
    async def _create_merged_profile(self, context: MergeContext, 
                                   merged_facts: List[StructuredFact],
                                   merged_tags: List[DynamicTag],
                                   merged_embeddings: List[MultiViewEmbedding]) -> CandidateProfile:
        """Create merged candidate profile"""
        
        # Merge profile data from all sources
        merged_profile_data = {}
        for source, data in context.source_data_sets.items():
            merged_profile_data.update(data)
        
        # Create merged profile
        merged_profile = CandidateProfile(
            candidate_id=context.candidate_id,
            uid=context.candidate_id,  # Assuming candidate_id is the same as uid
            tenant_id="default",  # This should be passed in context
            profile_data=merged_profile_data,
            facts=merged_facts,
            tags=merged_tags,
            embeddings=merged_embeddings,
            processing_stage="merging",
            last_updated=context.merge_timestamp,
            version="1.0"
        )
        
        return merged_profile
    
    async def _calculate_quality_metrics(self, profile: CandidateProfile, context: MergeContext) -> QualityMetrics:
        """Calculate quality metrics for merged profile"""
        
        # Calculate completeness score
        completeness_score = await self._calculate_completeness_score(profile, context)
        
        # Calculate accuracy score
        accuracy_score = await self._calculate_accuracy_score(profile, context)
        
        # Calculate consistency score
        consistency_score = await self._calculate_consistency_score(profile, context)
        
        # Calculate freshness score
        freshness_score = await self._calculate_freshness_score(profile, context)
        
        # Calculate overall quality
        overall_quality = (
            completeness_score * self.quality_weights["completeness_weight"] +
            accuracy_score * self.quality_weights["confidence_weight"] +
            consistency_score * self.quality_weights["consistency_weight"] +
            freshness_score * self.quality_weights["freshness_weight"]
        )
        
        # Identify issues
        missing_fields = await self._identify_missing_fields(profile)
        inconsistent_fields = await self._identify_inconsistent_fields(profile)
        low_confidence_facts = [f.fact_id for f in profile.facts if f.confidence.value < 0.5]
        
        # Generate improvement suggestions
        improvement_suggestions = await self._generate_improvement_suggestions(profile, context)
        
        return QualityMetrics(
            completeness_score=completeness_score,
            accuracy_score=accuracy_score,
            consistency_score=consistency_score,
            freshness_score=freshness_score,
            overall_quality=overall_quality,
            missing_fields=missing_fields,
            inconsistent_fields=inconsistent_fields,
            low_confidence_facts=low_confidence_facts,
            improvement_suggestions=improvement_suggestions
        )
    
    async def _calculate_completeness_score(self, profile: CandidateProfile, context: MergeContext) -> float:
        """Calculate completeness score"""
        required_components = ["facts", "tags", "embeddings"]
        present_components = []
        
        if profile.facts:
            present_components.append("facts")
        if profile.tags:
            present_components.append("tags")
        if profile.embeddings:
            present_components.append("embeddings")
        
        return len(present_components) / len(required_components)
    
    async def _calculate_accuracy_score(self, profile: CandidateProfile, context: MergeContext) -> float:
        """Calculate accuracy score based on confidence"""
        if not profile.facts and not profile.tags and not profile.embeddings:
            return 0.0
        
        all_items = profile.facts + profile.tags + profile.embeddings
        if not all_items:
            return 0.0
        
        avg_confidence = sum(item.confidence.value for item in all_items) / len(all_items)
        return avg_confidence
    
    async def _calculate_consistency_score(self, profile: CandidateProfile, context: MergeContext) -> float:
        """Calculate consistency score"""
        # Check for consistency between different data sources
        consistency_checks = []
        
        # Check fact consistency
        if profile.facts:
            fact_consistency = await self._check_fact_consistency(profile.facts)
            consistency_checks.append(fact_consistency)
        
        # Check tag consistency
        if profile.tags:
            tag_consistency = await self._check_tag_consistency(profile.tags)
            consistency_checks.append(tag_consistency)
        
        if not consistency_checks:
            return 1.0
        
        return sum(consistency_checks) / len(consistency_checks)
    
    async def _calculate_freshness_score(self, profile: CandidateProfile, context: MergeContext) -> float:
        """Calculate freshness score"""
        # Check how recent the data is
        now = datetime.utcnow()
        freshness_scores = []
        
        for fact in profile.facts:
            age_hours = (now - fact.provenance.timestamp).total_seconds() / 3600
            freshness = max(0.0, 1.0 - (age_hours / (24 * 30)))  # Decay over 30 days
            freshness_scores.append(freshness)
        
        if not freshness_scores:
            return 1.0
        
        return sum(freshness_scores) / len(freshness_scores)
    
    async def _check_fact_consistency(self, facts: List[StructuredFact]) -> float:
        """Check consistency between facts"""
        # Simple consistency check - can be enhanced
        return 0.8  # Placeholder
    
    async def _check_tag_consistency(self, tags: List[DynamicTag]) -> float:
        """Check consistency between tags"""
        # Simple consistency check - can be enhanced
        return 0.8  # Placeholder
    
    async def _identify_missing_fields(self, profile: CandidateProfile) -> List[str]:
        """Identify missing fields"""
        missing_fields = []
        
        if not profile.facts:
            missing_fields.append("facts")
        if not profile.tags:
            missing_fields.append("tags")
        if not profile.embeddings:
            missing_fields.append("embeddings")
        
        return missing_fields
    
    async def _identify_inconsistent_fields(self, profile: CandidateProfile) -> List[str]:
        """Identify inconsistent fields"""
        # Placeholder - can be enhanced with actual consistency checks
        return []
    
    async def _generate_improvement_suggestions(self, profile: CandidateProfile, context: MergeContext) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if not profile.facts:
            suggestions.append("Add more structured facts from resume or assessments")
        
        if not profile.tags:
            suggestions.append("Generate more tags from available data")
        
        if not profile.embeddings:
            suggestions.append("Create embeddings from available text data")
        
        low_confidence_count = len([f for f in profile.facts if f.confidence.value < 0.5])
        if low_confidence_count > 0:
            suggestions.append(f"Improve confidence for {low_confidence_count} low-confidence facts")
        
        return suggestions
    
    def get_merge_stats(self) -> Dict[str, Any]:
        """Get merge statistics"""
        return {
            "conflict_resolution_strategies": {
                data_type: {
                    "strategy": resolution.strategy,
                    "source_priority": [s.value for s in resolution.source_priority],
                    "confidence_threshold": resolution.confidence_threshold
                }
                for data_type, resolution in self.conflict_resolution.items()
            },
            "source_priorities": {source.value: priority for source, priority in self.source_priorities.items()},
            "quality_weights": self.quality_weights,
            "merge_strategies": list(self.merge_strategies.keys())
        }


# ==================== SINGLETON INSTANCE ====================

# Global instance for the application
intelligent_merger = IntelligentMerger()

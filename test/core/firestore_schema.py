"""
Firestore Schema Implementation for Jobsify AI Candidate Matching
Implements dynamic collections with confidence-based processing
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from google.cloud import firestore
from google.oauth2 import service_account
import requests

from .candidate_matching_models import (
    CandidateProfile, StructuredFact, DynamicTag, MultiViewEmbedding,
    ConfidenceScore, Provenance, DataSource, FactType, TagCategory,
    EmbeddingView, QualityMetrics, JobMatchResult
)

log = logging.getLogger(__name__)


@dataclass
class FirestoreConfig:
    """Firestore configuration"""
    project_id: str
    database_id: str
    service_account_url: str
    collections: Dict[str, str]


class FirestoreSchemaManager:
    """
    Manages Firestore collections and schema for candidate matching
    Implements dynamic schema evolution with confidence-based processing
    """
    
    def __init__(self, config: FirestoreConfig):
        self.config = config
        self.db = self._initialize_firestore_client()
        self.collections = self._initialize_collections()
        
    def _initialize_firestore_client(self) -> firestore.Client:
        """Initialize Firestore client"""
        try:
            # Load service account credentials
            response = requests.get(self.config.service_account_url)
            service_account_info = response.json()
            
            # Create credentials
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            
            # Initialize Firestore client
            db = firestore.Client(
                project=service_account_info["project_id"],
                credentials=credentials,
                database=self.config.database_id
            )
            
            log.info(f"Firestore client initialized for project {service_account_info['project_id']}")
            return db
            
        except Exception as e:
            log.error(f"Failed to initialize Firestore client: {e}")
            raise
    
    def _initialize_collections(self) -> Dict[str, firestore.CollectionReference]:
        """Initialize Firestore collections"""
        collections = {}
        
        collection_names = [
            "candidates",
            "facts",
            "tags", 
            "embeddings",
            "assessments",
            "chat_sessions",
            "job_matches",
            "processing_events",
            "quality_metrics"
        ]
        
        for collection_name in collection_names:
            collections[collection_name] = self.db.collection(collection_name)
            log.info(f"Initialized collection: {collection_name}")
        
        return collections
    
    # ==================== CANDIDATE PROFILE OPERATIONS ====================
    
    async def save_candidate_profile(self, profile: CandidateProfile) -> bool:
        """Save candidate profile to Firestore"""
        try:
            doc_ref = self.collections["candidates"].document(profile.candidate_id)
            
            # Convert profile to Firestore-compatible format
            profile_data = {
                "candidate_id": profile.candidate_id,
                "uid": profile.uid,
                "tenant_id": profile.tenant_id,
                "profile_data": profile.profile_data,
                "processing_stage": profile.processing_stage.value,
                "last_updated": profile.last_updated,
                "version": profile.version,
                "completeness_score": profile.completeness_score,
                "quality_score": profile.quality_score,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            doc_ref.set(profile_data, merge=True)
            log.info(f"Saved candidate profile: {profile.candidate_id}")
            return True
            
        except Exception as e:
            log.error(f"Error saving candidate profile {profile.candidate_id}: {e}")
            return False
    
    async def get_candidate_profile(self, candidate_id: str) -> Optional[CandidateProfile]:
        """Get candidate profile from Firestore"""
        try:
            doc_ref = self.collections["candidates"].document(candidate_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return None
            
            data = doc.to_dict()
            
            # Get related facts, tags, and embeddings
            facts = await self.get_candidate_facts(candidate_id)
            tags = await self.get_candidate_tags(candidate_id)
            embeddings = await self.get_candidate_embeddings(candidate_id)
            
            # Reconstruct profile
            profile = CandidateProfile(
                candidate_id=data["candidate_id"],
                uid=data["uid"],
                tenant_id=data["tenant_id"],
                profile_data=data.get("profile_data", {}),
                facts=facts,
                tags=tags,
                embeddings=embeddings,
                processing_stage=ProcessingStage(data.get("processing_stage", "ingestion")),
                last_updated=data.get("last_updated", datetime.utcnow()),
                version=data.get("version", "1.0"),
                completeness_score=data.get("completeness_score", 0.0),
                quality_score=data.get("quality_score", 0.0)
            )
            
            return profile
            
        except Exception as e:
            log.error(f"Error getting candidate profile {candidate_id}: {e}")
            return None
    
    async def update_candidate_profile(self, candidate_id: str, updates: Dict[str, Any]) -> bool:
        """Update candidate profile"""
        try:
            doc_ref = self.collections["candidates"].document(candidate_id)
            updates["updated_at"] = datetime.utcnow()
            doc_ref.update(updates)
            
            log.info(f"Updated candidate profile: {candidate_id}")
            return True
            
        except Exception as e:
            log.error(f"Error updating candidate profile {candidate_id}: {e}")
            return False
    
    # ==================== FACTS OPERATIONS ====================
    
    async def save_facts(self, candidate_id: str, facts: List[StructuredFact]) -> bool:
        """Save facts for a candidate"""
        try:
            batch = self.db.batch()
            
            for fact in facts:
                doc_ref = self.collections["facts"].document(fact.fact_id)
                
                fact_data = {
                    "fact_id": fact.fact_id,
                    "candidate_id": candidate_id,
                    "fact_type": fact.fact_type.value,
                    "content": fact.content,
                    "confidence": {
                        "value": fact.confidence.value,
                        "level": fact.confidence.level.value,
                        "reasoning": fact.confidence.reasoning
                    },
                    "provenance": {
                        "source": fact.provenance.source.value,
                        "source_id": fact.provenance.source_id,
                        "extraction_method": fact.provenance.extraction_method,
                        "timestamp": fact.provenance.timestamp
                    },
                    "tags": fact.tags,
                    "metadata": fact.metadata,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                batch.set(doc_ref, fact_data, merge=True)
            
            batch.commit()
            log.info(f"Saved {len(facts)} facts for candidate {candidate_id}")
            return True
            
        except Exception as e:
            log.error(f"Error saving facts for candidate {candidate_id}: {e}")
            return False
    
    async def get_candidate_facts(self, candidate_id: str) -> List[StructuredFact]:
        """Get facts for a candidate"""
        try:
            query = self.collections["facts"].where("candidate_id", "==", candidate_id)
            docs = query.stream()
            
            facts = []
            for doc in docs:
                data = doc.to_dict()
                
                fact = StructuredFact(
                    fact_id=data["fact_id"],
                    fact_type=FactType(data["fact_type"]),
                    content=data["content"],
                    confidence=ConfidenceScore(
                        value=data["confidence"]["value"],
                        reasoning=data["confidence"]["reasoning"]
                    ),
                    provenance=Provenance(
                        source=DataSource(data["provenance"]["source"]),
                        source_id=data["provenance"]["source_id"],
                        extraction_method=data["provenance"]["extraction_method"]
                    ),
                    tags=data.get("tags", []),
                    metadata=data.get("metadata", {})
                )
                facts.append(fact)
            
            return facts
            
        except Exception as e:
            log.error(f"Error getting facts for candidate {candidate_id}: {e}")
            return []
    
    # ==================== TAGS OPERATIONS ====================
    
    async def save_tags(self, candidate_id: str, tags: List[DynamicTag]) -> bool:
        """Save tags for a candidate"""
        try:
            batch = self.db.batch()
            
            for tag in tags:
                doc_ref = self.collections["tags"].document(tag.tag_id)
                
                tag_data = {
                    "tag_id": tag.tag_id,
                    "candidate_id": candidate_id,
                    "category": tag.category.value,
                    "value": tag.value,
                    "confidence": {
                        "value": tag.confidence.value,
                        "level": tag.confidence.level.value,
                        "reasoning": tag.confidence.reasoning
                    },
                    "provenance": {
                        "source": tag.provenance.source.value,
                        "source_id": tag.provenance.source_id,
                        "extraction_method": tag.provenance.extraction_method,
                        "timestamp": tag.provenance.timestamp
                    },
                    "related_facts": tag.related_facts,
                    "metadata": tag.metadata,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                batch.set(doc_ref, tag_data, merge=True)
            
            batch.commit()
            log.info(f"Saved {len(tags)} tags for candidate {candidate_id}")
            return True
            
        except Exception as e:
            log.error(f"Error saving tags for candidate {candidate_id}: {e}")
            return False
    
    async def get_candidate_tags(self, candidate_id: str) -> List[DynamicTag]:
        """Get tags for a candidate"""
        try:
            query = self.collections["tags"].where("candidate_id", "==", candidate_id)
            docs = query.stream()
            
            tags = []
            for doc in docs:
                data = doc.to_dict()
                
                tag = DynamicTag(
                    tag_id=data["tag_id"],
                    category=TagCategory(data["category"]),
                    value=data["value"],
                    confidence=ConfidenceScore(
                        value=data["confidence"]["value"],
                        reasoning=data["confidence"]["reasoning"]
                    ),
                    provenance=Provenance(
                        source=DataSource(data["provenance"]["source"]),
                        source_id=data["provenance"]["source_id"],
                        extraction_method=data["provenance"]["extraction_method"]
                    ),
                    related_facts=data.get("related_facts", []),
                    metadata=data.get("metadata", {})
                )
                tags.append(tag)
            
            return tags
            
        except Exception as e:
            log.error(f"Error getting tags for candidate {candidate_id}: {e}")
            return []
    
    # ==================== EMBEDDINGS OPERATIONS ====================
    
    async def save_embeddings(self, candidate_id: str, embeddings: List[MultiViewEmbedding]) -> bool:
        """Save embeddings for a candidate"""
        try:
            batch = self.db.batch()
            
            for embedding in embeddings:
                doc_ref = self.collections["embeddings"].document(embedding.embedding_id)
                
                embedding_data = {
                    "embedding_id": embedding.embedding_id,
                    "candidate_id": candidate_id,
                    "view_type": embedding.view_type.value,
                    "vector": embedding.vector,
                    "confidence": {
                        "value": embedding.confidence.value,
                        "level": embedding.confidence.level.value,
                        "reasoning": embedding.confidence.reasoning
                    },
                    "source_data": embedding.source_data,
                    "metadata": embedding.metadata,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                batch.set(doc_ref, embedding_data, merge=True)
            
            batch.commit()
            log.info(f"Saved {len(embeddings)} embeddings for candidate {candidate_id}")
            return True
            
        except Exception as e:
            log.error(f"Error saving embeddings for candidate {candidate_id}: {e}")
            return False
    
    async def get_candidate_embeddings(self, candidate_id: str) -> List[MultiViewEmbedding]:
        """Get embeddings for a candidate"""
        try:
            query = self.collections["embeddings"].where("candidate_id", "==", candidate_id)
            docs = query.stream()
            
            embeddings = []
            for doc in docs:
                data = doc.to_dict()
                
                embedding = MultiViewEmbedding(
                    embedding_id=data["embedding_id"],
                    view_type=EmbeddingView(data["view_type"]),
                    vector=data["vector"],
                    confidence=ConfidenceScore(
                        value=data["confidence"]["value"],
                        reasoning=data["confidence"]["reasoning"]
                    ),
                    source_data=data["source_data"],
                    metadata=data.get("metadata", {})
                )
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            log.error(f"Error getting embeddings for candidate {candidate_id}: {e}")
            return []
    
    # ==================== JOB MATCHES OPERATIONS ====================
    
    async def save_job_matches(self, candidate_id: str, job_matches: List[JobMatchResult]) -> bool:
        """Save job matches for a candidate"""
        try:
            batch = self.db.batch()
            
            for match in job_matches:
                doc_ref = self.collections["job_matches"].document(f"{candidate_id}_{match.job_id}")
                
                match_data = {
                    "candidate_id": candidate_id,
                    "job_id": match.job_id,
                    "overall_score": match.overall_score,
                    "skill_match_score": match.skill_match_score,
                    "experience_match_score": match.experience_match_score,
                    "education_match_score": match.education_match_score,
                    "cultural_fit_score": match.cultural_fit_score,
                    "matched_skills": match.matched_skills,
                    "unmatched_skills": match.unmatched_skills,
                    "matched_experience": match.matched_experience,
                    "matched_education": match.matched_education,
                    "confidence": {
                        "value": match.confidence.value,
                        "level": match.confidence.level.value,
                        "reasoning": match.confidence.reasoning
                    },
                    "rationale": match.rationale,
                    "retrieval_stage": match.retrieval_stage,
                    "created_at": match.created_at,
                    "metadata": match.metadata,
                    "updated_at": datetime.utcnow()
                }
                
                batch.set(doc_ref, match_data, merge=True)
            
            batch.commit()
            log.info(f"Saved {len(job_matches)} job matches for candidate {candidate_id}")
            return True
            
        except Exception as e:
            log.error(f"Error saving job matches for candidate {candidate_id}: {e}")
            return False
    
    async def get_candidate_job_matches(self, candidate_id: str) -> List[JobMatchResult]:
        """Get job matches for a candidate"""
        try:
            query = self.collections["job_matches"].where("candidate_id", "==", candidate_id)
            docs = query.stream()
            
            matches = []
            for doc in docs:
                data = doc.to_dict()
                
                match = JobMatchResult(
                    job_id=data["job_id"],
                    candidate_id=data["candidate_id"],
                    overall_score=data["overall_score"],
                    skill_match_score=data["skill_match_score"],
                    experience_match_score=data["experience_match_score"],
                    education_match_score=data["education_match_score"],
                    cultural_fit_score=data["cultural_fit_score"],
                    matched_skills=data["matched_skills"],
                    unmatched_skills=data["unmatched_skills"],
                    matched_experience=data["matched_experience"],
                    matched_education=data["matched_education"],
                    confidence=ConfidenceScore(
                        value=data["confidence"]["value"],
                        reasoning=data["confidence"]["reasoning"]
                    ),
                    rationale=data["rationale"],
                    retrieval_stage=data["retrieval_stage"],
                    created_at=data["created_at"],
                    metadata=data.get("metadata", {})
                )
                matches.append(match)
            
            return matches
            
        except Exception as e:
            log.error(f"Error getting job matches for candidate {candidate_id}: {e}")
            return []
    
    # ==================== QUALITY METRICS OPERATIONS ====================
    
    async def save_quality_metrics(self, candidate_id: str, quality_metrics: QualityMetrics) -> bool:
        """Save quality metrics for a candidate"""
        try:
            doc_ref = self.collections["quality_metrics"].document(candidate_id)
            
            metrics_data = {
                "candidate_id": candidate_id,
                "completeness_score": quality_metrics.completeness_score,
                "accuracy_score": quality_metrics.accuracy_score,
                "consistency_score": quality_metrics.consistency_score,
                "freshness_score": quality_metrics.freshness_score,
                "overall_quality": quality_metrics.overall_quality,
                "missing_fields": quality_metrics.missing_fields,
                "inconsistent_fields": quality_metrics.inconsistent_fields,
                "low_confidence_facts": quality_metrics.low_confidence_facts,
                "improvement_suggestions": quality_metrics.improvement_suggestions,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            doc_ref.set(metrics_data, merge=True)
            log.info(f"Saved quality metrics for candidate {candidate_id}")
            return True
            
        except Exception as e:
            log.error(f"Error saving quality metrics for candidate {candidate_id}: {e}")
            return False
    
    async def get_quality_metrics(self, candidate_id: str) -> Optional[QualityMetrics]:
        """Get quality metrics for a candidate"""
        try:
            doc_ref = self.collections["quality_metrics"].document(candidate_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return None
            
            data = doc.to_dict()
            
            metrics = QualityMetrics(
                completeness_score=data["completeness_score"],
                accuracy_score=data["accuracy_score"],
                consistency_score=data["consistency_score"],
                freshness_score=data["freshness_score"],
                overall_quality=data["overall_quality"],
                missing_fields=data.get("missing_fields", []),
                inconsistent_fields=data.get("inconsistent_fields", []),
                low_confidence_facts=data.get("low_confidence_facts", []),
                improvement_suggestions=data.get("improvement_suggestions", [])
            )
            
            return metrics
            
        except Exception as e:
            log.error(f"Error getting quality metrics for candidate {candidate_id}: {e}")
            return None
    
    # ==================== QUERY OPERATIONS ====================
    
    async def query_candidates_by_tags(self, tags: List[str], limit: int = 100) -> List[str]:
        """Query candidates by tags"""
        try:
            candidate_ids = set()
            
            for tag in tags:
                query = self.collections["tags"].where("value", "==", tag).limit(limit)
                docs = query.stream()
                
                for doc in docs:
                    data = doc.to_dict()
                    candidate_ids.add(data["candidate_id"])
            
            return list(candidate_ids)
            
        except Exception as e:
            log.error(f"Error querying candidates by tags: {e}")
            return []
    
    async def query_candidates_by_facts(self, fact_type: str, limit: int = 100) -> List[str]:
        """Query candidates by fact type"""
        try:
            query = self.collections["facts"].where("fact_type", "==", fact_type).limit(limit)
            docs = query.stream()
            
            candidate_ids = []
            for doc in docs:
                data = doc.to_dict()
                candidate_ids.append(data["candidate_id"])
            
            return candidate_ids
            
        except Exception as e:
            log.error(f"Error querying candidates by facts: {e}")
            return []
    
    async def query_candidates_by_quality(self, min_quality: float, limit: int = 100) -> List[str]:
        """Query candidates by quality score"""
        try:
            query = self.collections["quality_metrics"].where("overall_quality", ">=", min_quality).limit(limit)
            docs = query.stream()
            
            candidate_ids = []
            for doc in docs:
                data = doc.to_dict()
                candidate_ids.append(data["candidate_id"])
            
            return candidate_ids
            
        except Exception as e:
            log.error(f"Error querying candidates by quality: {e}")
            return []
    
    # ==================== UTILITY OPERATIONS ====================
    
    async def get_all_candidate_ids(self) -> List[str]:
        """Get all candidate IDs"""
        try:
            docs = self.collections["candidates"].stream()
            candidate_ids = []
            
            for doc in docs:
                candidate_ids.append(doc.id)
            
            return candidate_ids
            
        except Exception as e:
            log.error(f"Error getting all candidate IDs: {e}")
            return []
    
    async def delete_candidate_data(self, candidate_id: str) -> bool:
        """Delete all data for a candidate"""
        try:
            # Delete from all collections
            collections_to_clean = ["candidates", "facts", "tags", "embeddings", "job_matches", "quality_metrics"]
            
            for collection_name in collections_to_clean:
                query = self.collections[collection_name].where("candidate_id", "==", candidate_id)
                docs = query.stream()
                
                batch = self.db.batch()
                for doc in docs:
                    batch.delete(doc.reference)
                batch.commit()
            
            log.info(f"Deleted all data for candidate {candidate_id}")
            return True
            
        except Exception as e:
            log.error(f"Error deleting candidate data {candidate_id}: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        stats = {}
        
        for collection_name, collection_ref in self.collections.items():
            try:
                docs = collection_ref.stream()
                count = sum(1 for _ in docs)
                stats[collection_name] = count
            except Exception as e:
                stats[collection_name] = f"Error: {str(e)}"
        
        return stats


# ==================== SINGLETON INSTANCE ====================

# Global instance for the application
# This would be initialized with actual configuration in production
firestore_schema_manager = None

def initialize_firestore_schema(config: FirestoreConfig) -> FirestoreSchemaManager:
    """Initialize Firestore schema manager"""
    global firestore_schema_manager
    firestore_schema_manager = FirestoreSchemaManager(config)
    return firestore_schema_manager

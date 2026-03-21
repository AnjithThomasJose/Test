"""
ChromaDB Collections Setup for Jobsify AI Candidate Matching
Implements multi-view embeddings storage and retrieval
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .candidate_matching_models import (
    MultiViewEmbedding, EmbeddingView, CandidateProfile, ConfidenceScore
)
from .config import EMBEDDING_MODEL

# Import metadata normalization function from chroma module
try:
    from chroma import normalize_metadata
except ImportError:
    # Fallback if chroma module is not available
    def normalize_metadata(meta: dict | None) -> dict:
        """Fallback normalization - converts lists/dicts to strings"""
        if not meta:
            return {}
        normalized = {}
        for k, v in meta.items():
            if isinstance(v, list):
                normalized[k] = ", ".join(str(item) for item in v) if v else ""
            elif isinstance(v, dict):
                normalized[k] = json.dumps(v, separators=(",", ":"), ensure_ascii=False)
            elif isinstance(v, (str, int, float, bool)) or v is None:
                normalized[k] = v
            else:
                normalized[k] = str(v)
        return normalized

log = logging.getLogger(__name__)


async def _to_thread(fn, *args, **kwargs):
    """Run a sync ChromaDB operation in the thread pool to avoid blocking the event loop."""
    if kwargs:
        return await asyncio.to_thread(lambda: fn(*args, **kwargs))
    return await asyncio.to_thread(fn, *args)


@dataclass
class ChromaDBConfig:
    """ChromaDB configuration"""
    api_key: str
    tenant: str
    database: str
    embedding_model: str = EMBEDDING_MODEL  # Centralized in core/config.py


class ChromaDBManager:
    """
    Manages ChromaDB collections for multi-view embeddings
    Implements embedding storage and retrieval for candidate matching
    """
    
    def __init__(self, config: ChromaDBConfig):
        self.config = config
        self.client = self._initialize_chromadb_client()
        self.embedding_function = self._initialize_embedding_function()
        self.collections = self._initialize_collections()
        
    def _initialize_chromadb_client(self) -> chromadb.CloudClient:
        """Initialize ChromaDB client"""
        try:
            client = chromadb.CloudClient(
                api_key=self.config.api_key,
                tenant=self.config.tenant,
                database=self.config.database
            )
            
            log.info(f"ChromaDB client initialized for tenant {self.config.tenant}")
            return client
            
        except Exception as e:
            log.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def _initialize_embedding_function(self) -> SentenceTransformerEmbeddingFunction:
        """Initialize embedding function"""
        try:
            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model
            )
            
            log.info(f"Embedding function initialized with model {self.config.embedding_model}")
            return embedding_function
            
        except Exception as e:
            log.error(f"Failed to initialize embedding function: {e}")
            raise
    
    def _initialize_collections(self) -> Dict[EmbeddingView, chromadb.Collection]:
        """Initialize ChromaDB collections for different embedding views"""
        collections = {}
        
        collection_mapping = {
            EmbeddingView.RESUME_CONTENT: "resume_embeddings",
            EmbeddingView.SKILL_FOCUSED: "skill_embeddings",
            EmbeddingView.EXPERIENCE_FOCUSED: "experience_embeddings",
            EmbeddingView.ASSESSMENT_RESPONSES: "assessment_embeddings",
            EmbeddingView.CHAT_CONTEXT: "chat_embeddings",
            EmbeddingView.COMPREHENSIVE: "comprehensive_embeddings"
        }
        
        for view_type, collection_name in collection_mapping.items():
            try:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                collections[view_type] = collection
                log.info(f"Initialized collection: {collection_name}")
                
            except Exception as e:
                log.error(f"Failed to initialize collection {collection_name}: {e}")
                raise
        
        return collections
    
    # ==================== TENANT ISOLATION HELPERS ====================
    
    def _validate_tenant_id(self, tenant_id: str, operation: str) -> bool:
        """Validate tenant_id is present and properly formatted.
        
        Args:
            tenant_id: The tenant identifier to validate
            operation: Description of the operation (for logging)
            
        Returns:
            True if valid, False otherwise
        """
        if not tenant_id or not isinstance(tenant_id, str) or not tenant_id.strip():
            log.error(f"Tenant isolation violation: missing or invalid tenant_id for {operation}")
            return False
        return True
    
    def _build_tenant_filter(self, tenant_id: str, additional_filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Build a ChromaDB where clause that includes tenant_id filter.
        
        Args:
            tenant_id: The tenant identifier (must be validated before calling)
            additional_filters: Optional additional filter conditions
            
        Returns:
            Combined where clause with tenant_id filter
        """
        if additional_filters:
            return {"$and": [{"tenant_id": tenant_id}, additional_filters]}
        return {"tenant_id": tenant_id}
    
    # ==================== EMBEDDING STORAGE OPERATIONS ====================
    
    async def store_candidate_embeddings(self, candidate_id: str, embeddings: List[MultiViewEmbedding], 
                                        tenant_id: str) -> bool:
        """Store embeddings for a candidate.
        
        Args:
            candidate_id: Unique candidate identifier
            embeddings: List of multi-view embeddings to store
            tenant_id: Tenant identifier for isolation (REQUIRED)
            
        Returns:
            True if successful, False otherwise
        """
        # Tenant isolation: validate tenant_id
        if not self._validate_tenant_id(tenant_id, f"store_candidate_embeddings({candidate_id})"):
            return False
            
        try:
            for embedding in embeddings:
                collection = self.collections.get(embedding.view_type)
                if not collection:
                    log.warning(f"No collection found for view type {embedding.view_type}")
                    continue
                
                # Prepare data for storage
                embedding_id = f"{tenant_id}_{candidate_id}_{embedding.view_type.value}"
                
                # Convert vector to list if it's numpy array
                vector = embedding.vector if isinstance(embedding.vector, list) else embedding.vector.tolist()
                
                # Prepare metadata (includes tenant_id for isolation, embedding_model for version tracking)
                metadata = {
                    "tenant_id": tenant_id,  # Tenant isolation: always set
                    "candidate_id": candidate_id,
                    "embedding_id": embedding.embedding_id,
                    "view_type": embedding.view_type.value,
                    "confidence_value": embedding.confidence.value,
                    "confidence_level": embedding.confidence.level.value,
                    "confidence_reasoning": embedding.confidence.reasoning,
                    "source_data_keys": list(embedding.source_data.keys()),
                    "created_at": datetime.utcnow().isoformat(),
                    "embedding_model": self.config.embedding_model,
                    **embedding.metadata
                }
                
                # Normalize metadata to ensure all values are ChromaDB-compatible primitives
                metadata = normalize_metadata(metadata)
                
                await _to_thread(
                    collection.upsert,
                    ids=[embedding_id],
                    embeddings=[vector],
                    metadatas=[metadata],
                    documents=[json.dumps(embedding.source_data)]
                )
                
                log.info(f"Stored {embedding.view_type.value} embedding for candidate {candidate_id} (tenant: {tenant_id})")
            
            return True
            
        except Exception as e:
            log.error(f"Error storing embeddings for candidate {candidate_id}: {e}")
            return False
    
    async def get_candidate_embeddings(self, candidate_id: str, tenant_id: str,
                                       view_type: Optional[EmbeddingView] = None) -> List[MultiViewEmbedding]:
        """Get embeddings for a candidate.
        
        Args:
            candidate_id: Unique candidate identifier
            tenant_id: Tenant identifier for isolation (REQUIRED)
            view_type: Optional specific view type to retrieve
            
        Returns:
            List of embeddings for the candidate
        """
        # Tenant isolation: validate tenant_id
        if not self._validate_tenant_id(tenant_id, f"get_candidate_embeddings({candidate_id})"):
            return []
            
        try:
            embeddings = []
            
            if view_type:
                collection = self.collections.get(view_type)
                if collection:
                    results = await _to_thread(
                        collection.get,
                        where=self._build_tenant_filter(tenant_id, {"candidate_id": candidate_id})
                    )
                    embeddings.extend(self._parse_embedding_results(results, view_type))
            else:
                for view, collection in self.collections.items():
                    results = await _to_thread(
                        collection.get,
                        where=self._build_tenant_filter(tenant_id, {"candidate_id": candidate_id})
                    )
                    embeddings.extend(self._parse_embedding_results(results, view))
            
            return embeddings
            
        except Exception as e:
            log.error(f"Error getting embeddings for candidate {candidate_id}: {e}")
            return []
    
    def _parse_embedding_results(self, results: Dict[str, Any], view_type: EmbeddingView) -> List[MultiViewEmbedding]:
        """Parse ChromaDB results into MultiViewEmbedding objects"""
        embeddings = []
        
        if not results or not results.get("ids"):
            return embeddings
        
        for i, embedding_id in enumerate(results["ids"]):
            try:
                embedding = MultiViewEmbedding(
                    embedding_id=results["metadatas"][i]["embedding_id"],
                    view_type=view_type,
                    vector=results["embeddings"][i],
                    confidence=ConfidenceScore(
                        value=results["metadatas"][i]["confidence_value"],
                        reasoning=results["metadatas"][i]["confidence_reasoning"]
                    ),
                    source_data=json.loads(results["documents"][i]) if results["documents"][i] else {},
                    metadata={
                        k: v for k, v in results["metadatas"][i].items()
                        if k not in ["candidate_id", "embedding_id", "view_type", "confidence_value", "confidence_level", "confidence_reasoning", "source_data_keys", "created_at"]
                    }
                )
                embeddings.append(embedding)
                
            except Exception as e:
                log.error(f"Error parsing embedding {embedding_id}: {e}")
                continue
        
        return embeddings
    
    # ==================== VECTOR SEARCH OPERATIONS ====================
    
    async def search_similar_candidates(self, query_embedding: List[float], 
                                      view_type: EmbeddingView, 
                                      tenant_id: str,
                                      top_k: int = 10,
                                      min_confidence: float = 0.0) -> List[Tuple[str, float]]:
        """Search for similar candidates using vector similarity.
        
        Args:
            query_embedding: Query vector for similarity search
            view_type: Which embedding view to search
            tenant_id: Tenant identifier for isolation (REQUIRED)
            top_k: Maximum number of results
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of (candidate_id, similarity_score) tuples
        """
        # Tenant isolation: validate tenant_id
        if not self._validate_tenant_id(tenant_id, "search_similar_candidates"):
            return []
            
        try:
            collection = self.collections.get(view_type)
            if not collection:
                log.error(f"No collection found for view type {view_type}")
                return []
            
            # Tenant isolation: filter by tenant_id AND confidence
            where_clause = self._build_tenant_filter(
                tenant_id, 
                {"confidence_value": {"$gte": min_confidence}}
            )
            
            results = await _to_thread(
                collection.query,
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause
            )
            
            # Parse results
            candidates = []
            if results["ids"] and results["ids"][0]:
                for i, candidate_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    candidates.append((candidate_id, similarity))
            
            log.info(f"Found {len(candidates)} similar candidates for view {view_type.value} (tenant: {tenant_id})")
            return candidates
            
        except Exception as e:
            log.error(f"Error searching similar candidates: {e}")
            return []
    
    async def search_candidates_by_text(self, query_text: str, 
                                      view_type: EmbeddingView, 
                                      tenant_id: str,
                                      top_k: int = 10,
                                      min_confidence: float = 0.0) -> List[Tuple[str, float]]:
        """Search for candidates using text query.
        
        Args:
            query_text: Text query to search for
            view_type: Which embedding view to search
            tenant_id: Tenant identifier for isolation (REQUIRED)
            top_k: Maximum number of results
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of (candidate_id, similarity_score) tuples
        """
        # Tenant isolation: validate tenant_id
        if not self._validate_tenant_id(tenant_id, "search_candidates_by_text"):
            return []
            
        try:
            collection = self.collections.get(view_type)
            if not collection:
                log.error(f"No collection found for view type {view_type}")
                return []
            
            # Tenant isolation: filter by tenant_id AND confidence
            where_clause = self._build_tenant_filter(
                tenant_id, 
                {"confidence_value": {"$gte": min_confidence}}
            )
            
            results = await _to_thread(
                collection.query,
                query_texts=[query_text],
                n_results=top_k,
                where=where_clause
            )
            
            # Parse results
            candidates = []
            if results["ids"] and results["ids"][0]:
                for i, candidate_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    candidates.append((candidate_id, similarity))
            
            log.info(f"Found {len(candidates)} candidates for text query in view {view_type.value} (tenant: {tenant_id})")
            return candidates
            
        except Exception as e:
            log.error(f"Error searching candidates by text: {e}")
            return []
    
    # ==================== JOB DESCRIPTION EMBEDDINGS ====================
    
    async def store_job_description_embedding(self, job_id: str, job_description: str, 
                                            tenant_id: str,
                                            view_type: EmbeddingView = EmbeddingView.COMPREHENSIVE) -> bool:
        """Store job description embedding.
        
        Args:
            job_id: Unique job identifier
            job_description: Job description text
            tenant_id: Tenant identifier for isolation (REQUIRED)
            view_type: Which embedding view to use
            
        Returns:
            True if successful, False otherwise
        """
        # Tenant isolation: validate tenant_id
        if not self._validate_tenant_id(tenant_id, f"store_job_description_embedding({job_id})"):
            return False
            
        try:
            collection = self.collections.get(view_type)
            if not collection:
                log.error(f"No collection found for view type {view_type}")
                return False
            
            # Prepare metadata (includes tenant_id for isolation, embedding_model for version tracking)
            metadata = {
                "tenant_id": tenant_id,  # Tenant isolation: always set
                "job_id": job_id,
                "view_type": view_type.value,
                "created_at": datetime.utcnow().isoformat(),
                "embedding_model": self.config.embedding_model
            }
            
            # Normalize metadata to ensure all values are ChromaDB-compatible primitives
            metadata = normalize_metadata(metadata)
            
            # Use tenant-scoped ID to prevent cross-tenant collisions
            scoped_job_id = f"{tenant_id}_{job_id}"
            
            await _to_thread(
                collection.upsert,
                ids=[scoped_job_id],
                documents=[job_description],
                metadatas=[metadata]
            )
            
            log.info(f"Stored job description embedding for job {job_id} (tenant: {tenant_id})")
            return True
            
        except Exception as e:
            log.error(f"Error storing job description embedding for job {job_id}: {e}")
            return False
    
    async def search_candidates_for_job(self, job_id: str, 
                                      tenant_id: str,
                                      view_type: EmbeddingView = EmbeddingView.COMPREHENSIVE,
                                      top_k: int = 50) -> List[Tuple[str, float]]:
        """Search candidates for a specific job.
        
        Args:
            job_id: Job identifier to match candidates against
            tenant_id: Tenant identifier for isolation (REQUIRED)
            view_type: Which embedding view to search
            top_k: Maximum number of candidates to return
            
        Returns:
            List of (candidate_id, similarity_score) tuples
        """
        # Tenant isolation: validate tenant_id
        if not self._validate_tenant_id(tenant_id, f"search_candidates_for_job({job_id})"):
            return []
            
        try:
            collection = self.collections.get(view_type)
            if not collection:
                log.error(f"No collection found for view type {view_type}")
                return []
            
            # Use tenant-scoped job ID
            scoped_job_id = f"{tenant_id}_{job_id}"
            
            job_results = await _to_thread(collection.get, ids=[scoped_job_id])
            if not job_results["ids"]:
                log.error(f"Job {job_id} not found in collection for tenant {tenant_id}")
                return []
            
            # Use job embedding to search for similar candidates
            job_embedding = job_results["embeddings"][0]
            
            # Tenant isolation: filter by tenant_id AND candidate_id exists
            where_clause = self._build_tenant_filter(
                tenant_id,
                {"candidate_id": {"$exists": True}}  # Only candidates, not jobs
            )
            
            results = await _to_thread(
                collection.query,
                query_embeddings=[job_embedding],
                n_results=top_k,
                where=where_clause
            )
            
            # Parse results
            candidates = []
            if results["ids"] and results["ids"][0]:
                for i, candidate_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    candidates.append((candidate_id, similarity))
            
            log.info(f"Found {len(candidates)} candidates for job {job_id} (tenant: {tenant_id})")
            return candidates
            
        except Exception as e:
            log.error(f"Error searching candidates for job {job_id}: {e}")
            return []
    
    # ==================== COLLECTION MANAGEMENT ====================
    
    async def delete_candidate_embeddings(self, candidate_id: str, tenant_id: str) -> bool:
        """Delete all embeddings for a candidate.
        
        Args:
            candidate_id: Unique candidate identifier
            tenant_id: Tenant identifier for isolation (REQUIRED)
            
        Returns:
            True if successful, False otherwise
        """
        # Tenant isolation: validate tenant_id
        if not self._validate_tenant_id(tenant_id, f"delete_candidate_embeddings({candidate_id})"):
            return False
            
        try:
            deleted_count = 0
            
            for view_type, collection in self.collections.items():
                results = await _to_thread(
                    collection.get,
                    where=self._build_tenant_filter(tenant_id, {"candidate_id": candidate_id})
                )
                
                if results["ids"]:
                    await _to_thread(collection.delete, ids=results["ids"])
                    deleted_count += len(results["ids"])
                    
                    log.info(f"Deleted {len(results['ids'])} {view_type.value} embeddings for candidate {candidate_id}")
            
            log.info(f"Deleted total {deleted_count} embeddings for candidate {candidate_id}")
            return True
            
        except Exception as e:
            log.error(f"Error deleting embeddings for candidate {candidate_id}: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections"""
        stats = {}
        
        for view_type, collection in self.collections.items():
            try:
                count = await _to_thread(collection.count)
                stats[view_type.value] = {
                    "count": count,
                    "name": collection.name
                }
            except Exception as e:
                stats[view_type.value] = {
                    "error": str(e),
                    "name": collection.name
                }
        
        return stats
    
    async def cleanup_old_embeddings(self, days_old: int = 30) -> int:
        """Cleanup old embeddings"""
        try:
            cutoff_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_old)
            
            deleted_count = 0
            
            for view_type, collection in self.collections.items():
                results = await _to_thread(
                    collection.get,
                    where={"created_at": {"$lt": cutoff_date.isoformat()}}
                )
                
                if results["ids"]:
                    await _to_thread(collection.delete, ids=results["ids"])
                    deleted_count += len(results["ids"])
                    
                    log.info(f"Deleted {len(results['ids'])} old {view_type.value} embeddings")
            
            log.info(f"Cleaned up total {deleted_count} old embeddings")
            return deleted_count
            
        except Exception as e:
            log.error(f"Error cleaning up old embeddings: {e}")
            return 0
    
    # ==================== UTILITY METHODS ====================
    
    async def get_candidate_embedding_by_view(self, candidate_id: str, tenant_id: str, 
                                              view_type: EmbeddingView) -> Optional[MultiViewEmbedding]:
        """Get specific embedding view for a candidate.
        
        Args:
            candidate_id: Unique candidate identifier
            tenant_id: Tenant identifier for isolation (REQUIRED)
            view_type: Which embedding view to retrieve
            
        Returns:
            The embedding if found, None otherwise
        """
        embeddings = await self.get_candidate_embeddings(candidate_id, tenant_id, view_type)
        return embeddings[0] if embeddings else None
    
    async def update_candidate_embedding(self, candidate_id: str, tenant_id: str, 
                                        embedding: MultiViewEmbedding) -> bool:
        """Update a specific embedding for a candidate.
        
        Args:
            candidate_id: Unique candidate identifier
            tenant_id: Tenant identifier for isolation (REQUIRED)
            embedding: The embedding to update
            
        Returns:
            True if successful, False otherwise
        """
        return await self.store_candidate_embeddings(candidate_id, [embedding], tenant_id)
    
    async def batch_store_embeddings(self, embeddings_by_candidate: Dict[str, List[MultiViewEmbedding]], 
                                    tenant_id: str) -> bool:
        """Batch store embeddings for multiple candidates.
        
        Args:
            embeddings_by_candidate: Dict mapping candidate_id to list of embeddings
            tenant_id: Tenant identifier for isolation (REQUIRED)
            
        Returns:
            True if all successful, False if any failed
        """
        # Tenant isolation: validate tenant_id
        if not self._validate_tenant_id(tenant_id, "batch_store_embeddings"):
            return False
            
        try:
            for candidate_id, embeddings in embeddings_by_candidate.items():
                success = await self.store_candidate_embeddings(candidate_id, embeddings, tenant_id)
                if not success:
                    log.error(f"Failed to store embeddings for candidate {candidate_id}")
                    return False
            
            log.info(f"Successfully batch stored embeddings for {len(embeddings_by_candidate)} candidates (tenant: {tenant_id})")
            return True
            
        except Exception as e:
            log.error(f"Error in batch store embeddings: {e}")
            return False


# ==================== SINGLETON INSTANCE ====================

# Global instance for the application
# This would be initialized with actual configuration in production
chromadb_manager = None

def initialize_chromadb_manager(config: ChromaDBConfig) -> ChromaDBManager:
    """Initialize ChromaDB manager"""
    global chromadb_manager
    chromadb_manager = ChromaDBManager(config)
    return chromadb_manager

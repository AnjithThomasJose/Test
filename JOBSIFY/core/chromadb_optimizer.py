"""
Optimized ChromaDB Architecture
Implements connection pooling, caching, batch operations, and performance monitoring
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import lru_cache
from collections import defaultdict
import hashlib

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from settings import settings
from chroma import normalize_metadata
from core.config import EMBEDDING_MODEL

log = logging.getLogger(__name__)


@dataclass
class QueryCacheEntry:
    """Cache entry for query results"""
    results: Dict[str, Any]
    timestamp: float
    hit_count: int = 0


@dataclass
class CollectionMetrics:
    """Metrics for a ChromaDB collection"""
    name: str
    query_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_query_time_ms: float = 0.0
    total_query_time_ms: float = 0.0
    last_accessed: Optional[float] = None
    document_count: int = 0


@dataclass
class ChromaDBOptimizerConfig:
    """Configuration for ChromaDB optimizer"""
    # Cache settings
    enable_query_cache: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes default
    max_cache_size: int = 1000  # Max cached queries per collection
    
    # Embedding model (used in cache key to invalidate when model changes)
    embedding_model: str = EMBEDDING_MODEL  # Centralized in core/config.py
    
    # Batch settings
    batch_size: int = 100  # Batch operations size
    batch_timeout_seconds: float = 1.0  # Max wait time for batch
    
    # Query optimization
    default_top_k: int = 10  # Default top_k for queries
    max_top_k: int = 100  # Maximum allowed top_k
    enable_query_optimization: bool = True
    
    # Performance monitoring
    enable_metrics: bool = True
    metrics_retention_hours: int = 24
    
    # Connection settings
    connection_timeout_seconds: int = 30
    max_retries: int = 3


class OptimizedChromaDBManager:
    """
    Optimized ChromaDB manager with:
    - Connection pooling and singleton pattern
    - Query result caching
    - Batch operations
    - Performance monitoring
    - Query optimization
    """
    
    _instance: Optional['OptimizedChromaDBManager'] = None
    _lock = asyncio.Lock()
    
    def __init__(self, config: Optional[ChromaDBOptimizerConfig] = None):
        if OptimizedChromaDBManager._instance is not None:
            raise RuntimeError("Use get_instance() to get the singleton instance")
        
        self.config = config or ChromaDBOptimizerConfig()
        self.client: Optional[chromadb.CloudClient] = None
        self.embedding_function: Optional[SentenceTransformerEmbeddingFunction] = None
        self.collections: Dict[str, chromadb.Collection] = {}
        self.collection_cache: Dict[str, QueryCacheEntry] = {}
        self.collection_metrics: Dict[str, CollectionMetrics] = {}
        self.batch_queue: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.batch_timers: Dict[str, float] = {}
        self._initialized = False
        
    @classmethod
    async def get_instance(cls, config: Optional[ChromaDBOptimizerConfig] = None) -> 'OptimizedChromaDBManager':
        """Get or create singleton instance"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
                    await cls._instance.initialize()
        return cls._instance
    
    async def initialize(self):
        """Initialize ChromaDB client and collections"""
        if self._initialized:
            return
        
        try:
            # Initialize client with retry logic
            self.client = await self._initialize_client_with_retry()
            self.embedding_function = self._initialize_embedding_function()
            
            # Lazy load collections (only when needed)
            log.info("ChromaDB optimizer initialized successfully")
            self._initialized = True
            
        except Exception as e:
            log.error(f"Failed to initialize ChromaDB optimizer: {e}")
            raise
    
    async def _initialize_client_with_retry(self) -> chromadb.CloudClient:
        """Initialize client with retry logic"""
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                client = chromadb.CloudClient(
                    api_key=settings.CHROMA_API_KEY,
                    tenant=settings.CHROMA_TENANT,
                    database=settings.CHROMA_DATABASE
                )
                # Test connection
                client.list_collections()
                log.info(f"ChromaDB client initialized (attempt {attempt + 1})")
                return client
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    log.warning(f"ChromaDB initialization failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    log.error(f"ChromaDB initialization failed after {self.config.max_retries} attempts")
        
        raise last_error
    
    def _initialize_embedding_function(self) -> SentenceTransformerEmbeddingFunction:
        """Initialize embedding function (reused across collections)"""
        return SentenceTransformerEmbeddingFunction(
            model_name=self.config.embedding_model  # Uses centralized config from core/config.py
        )
    
    def get_collection(self, name: str, create_if_not_exists: bool = True) -> chromadb.Collection:
        """Get or create collection with lazy loading"""
        if name in self.collections:
            return self.collections[name]
        
        if not self.client:
            raise RuntimeError("ChromaDB client not initialized")
        
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function
            )
            self.collections[name] = collection
            
            # Initialize metrics
            if self.config.enable_metrics:
                self.collection_metrics[name] = CollectionMetrics(
                    name=name,
                    document_count=collection.count()
                )
            
            log.debug(f"Collection '{name}' loaded")
            return collection
            
        except Exception as e:
            log.error(f"Failed to get/create collection '{name}': {e}")
            raise
    
    def _generate_cache_key(self, collection_name: str, query_text: Optional[str], 
                          query_embedding: Optional[List[float]], 
                          n_results: int, where_clause: Optional[Dict],
                          tenant_id: Optional[str] = None) -> str:
        """Generate cache key for query.
        
        Includes tenant_id to prevent cross-tenant cache hits (security),
        and embedding_model to auto-invalidate when model changes.
        """
        key_parts = [
            collection_name,
            tenant_id or "",  # Tenant isolation: prevents cross-tenant cache hits
            self.config.embedding_model,  # Invalidates cache if model changes
            str(query_text) if query_text else "",
            json.dumps(query_embedding, sort_keys=True) if query_embedding else "",
            str(n_results),
            json.dumps(where_clause, sort_keys=True) if where_clause else ""
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def query_with_cache(self, collection_name: str, 
                              query_texts: Optional[List[str]] = None,
                              query_embeddings: Optional[List[List[float]]] = None,
                              n_results: int = 10,
                              where: Optional[Dict] = None,
                              use_cache: bool = True,
                              tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query collection with caching and performance monitoring.
        
        Args:
            collection_name: Name of the collection to query
            query_texts: Text queries (will be embedded)
            query_embeddings: Pre-computed embeddings
            n_results: Number of results to return
            where: Filter clause
            use_cache: Whether to use cache (default True)
            tenant_id: Tenant identifier for cache isolation (prevents cross-tenant hits)
        """
        start_time = time.time()
        collection = self.get_collection(collection_name)
        
        # Validate n_results
        n_results = min(n_results, self.config.max_top_k)
        
        # Check cache (includes tenant_id for isolation)
        cache_key = None
        if use_cache and self.config.enable_query_cache and query_texts:
            cache_key = self._generate_cache_key(
                collection_name, query_texts[0] if query_texts else None,
                query_embeddings[0] if query_embeddings else None,
                n_results, where, tenant_id
            )
            
            if cache_key in self.collection_cache:
                entry = self.collection_cache[cache_key]
                age = time.time() - entry.timestamp
                
                if age < self.config.cache_ttl_seconds:
                    entry.hit_count += 1
                    if self.config.enable_metrics:
                        metrics = self.collection_metrics.get(collection_name)
                        if metrics:
                            metrics.cache_hits += 1
                            metrics.last_accessed = time.time()
                    
                    log.debug(f"Cache hit for query in '{collection_name}'")
                    return entry.results
        
        # Execute query
        try:
            query_params = {
                "n_results": n_results,
                "where": where,
                "include": ["metadatas", "documents", "distances"]  # Resume collection stores in metadata; document fallback
            }
            
            if query_texts:
                query_params["query_texts"] = query_texts
            elif query_embeddings:
                query_params["query_embeddings"] = query_embeddings
            else:
                raise ValueError("Either query_texts or query_embeddings must be provided")
            
            results = collection.query(**query_params)
            
            # Cache results
            if cache_key and use_cache and self.config.enable_query_cache:
                self._add_to_cache(collection_name, cache_key, results)
                if self.config.enable_metrics:
                    metrics = self.collection_metrics.get(collection_name)
                    if metrics:
                        metrics.cache_misses += 1
            
            # Update metrics
            query_time_ms = (time.time() - start_time) * 1000
            if self.config.enable_metrics:
                metrics = self.collection_metrics.get(collection_name)
                if metrics:
                    metrics.query_count += 1
                    metrics.total_query_time_ms += query_time_ms
                    metrics.avg_query_time_ms = metrics.total_query_time_ms / metrics.query_count
                    metrics.last_accessed = time.time()
            
            log.debug(f"Query completed in {query_time_ms:.2f}ms for '{collection_name}'")
            return results
            
        except Exception as e:
            log.error(f"Query failed for '{collection_name}': {e}")
            raise
    
    def _add_to_cache(self, collection_name: str, cache_key: str, results: Dict[str, Any]):
        """Add query results to cache with size management"""
        # Remove old entries if cache is full
        if len(self.collection_cache) >= self.config.max_cache_size:
            # Remove least recently used entries
            sorted_entries = sorted(
                self.collection_cache.items(),
                key=lambda x: (x[1].timestamp, -x[1].hit_count)
            )
            # Remove 10% of cache
            remove_count = max(1, self.config.max_cache_size // 10)
            for key, _ in sorted_entries[:remove_count]:
                del self.collection_cache[key]
        
        self.collection_cache[cache_key] = QueryCacheEntry(
            results=results,
            timestamp=time.time()
        )
    
    async def batch_upsert(self, collection_name: str, 
                          ids: List[str],
                          documents: Optional[List[str]] = None,
                          embeddings: Optional[List[List[float]]] = None,
                          metadatas: Optional[List[Dict]] = None,
                          flush_immediately: bool = False):
        """
        Batch upsert with automatic batching and flushing
        """
        collection = self.get_collection(collection_name)
        
        # Normalize metadatas
        if metadatas:
            metadatas = [normalize_metadata(meta) if meta else {} for meta in metadatas]
        
        # Add to batch queue
        batch_key = collection_name
        batch_item = {
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas
        }
        
        self.batch_queue[batch_key].append(batch_item)
        
        # Flush if batch is full or immediate flush requested
        if (flush_immediately or 
            len(self.batch_queue[batch_key]) >= self.config.batch_size):
            await self._flush_batch(collection_name)
        else:
            # Schedule flush after timeout
            await self._schedule_batch_flush(collection_name)
    
    async def _schedule_batch_flush(self, collection_name: str):
        """Schedule batch flush after timeout"""
        batch_key = collection_name
        current_time = time.time()
        
        if batch_key not in self.batch_timers:
            self.batch_timers[batch_key] = current_time
        
        # Check if timeout reached
        if current_time - self.batch_timers[batch_key] >= self.config.batch_timeout_seconds:
            await self._flush_batch(collection_name)
    
    async def _flush_batch(self, collection_name: str):
        """Flush batched operations"""
        batch_key = collection_name
        if not self.batch_queue[batch_key]:
            return
        
        collection = self.get_collection(collection_name)
        batch_items = self.batch_queue[batch_key]
        self.batch_queue[batch_key] = []
        
        if batch_key in self.batch_timers:
            del self.batch_timers[batch_key]
        
        try:
            # Combine all batch items
            all_ids = []
            all_documents = []
            all_embeddings = []
            all_metadatas = []
            
            for item in batch_items:
                all_ids.extend(item["ids"])
                if item["documents"]:
                    all_documents.extend(item["documents"])
                if item["embeddings"]:
                    all_embeddings.extend(item["embeddings"])
                if item["metadatas"]:
                    all_metadatas.extend(item["metadatas"])
            
            # Perform batch upsert
            upsert_params = {"ids": all_ids}
            if all_documents:
                upsert_params["documents"] = all_documents
            if all_embeddings:
                upsert_params["embeddings"] = all_embeddings
            if all_metadatas:
                upsert_params["metadatas"] = all_metadatas
            
            collection.upsert(**upsert_params)
            
            log.info(f"Flushed batch of {len(all_ids)} items to '{collection_name}'")
            
            # Update metrics
            if self.config.enable_metrics:
                metrics = self.collection_metrics.get(collection_name)
                if metrics:
                    metrics.document_count = collection.count()
            
        except Exception as e:
            log.error(f"Failed to flush batch for '{collection_name}': {e}")
            raise
    
    async def flush_all_batches(self):
        """Flush all pending batches"""
        for collection_name in list(self.batch_queue.keys()):
            if self.batch_queue[collection_name]:
                await self._flush_batch(collection_name)
    
    def get_metrics(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.config.enable_metrics:
            return {}
        
        if collection_name:
            metrics = self.collection_metrics.get(collection_name)
            if metrics:
                return {
                    "collection_name": metrics.name,
                    "query_count": metrics.query_count,
                    "cache_hits": metrics.cache_hits,
                    "cache_misses": metrics.cache_misses,
                    "cache_hit_rate": (metrics.cache_hits / max(metrics.query_count, 1)) * 100,
                    "avg_query_time_ms": metrics.avg_query_time_ms,
                    "document_count": metrics.document_count,
                    "last_accessed": datetime.fromtimestamp(metrics.last_accessed).isoformat() if metrics.last_accessed else None
                }
            return {}
        
        # Return all metrics
        return {
            name: {
                "query_count": m.query_count,
                "cache_hits": m.cache_hits,
                "cache_misses": m.cache_misses,
                "cache_hit_rate": (m.cache_hits / max(m.query_count, 1)) * 100,
                "avg_query_time_ms": m.avg_query_time_ms,
                "document_count": m.document_count,
                "last_accessed": datetime.fromtimestamp(m.last_accessed).isoformat() if m.last_accessed else None
            }
            for name, m in self.collection_metrics.items()
        }
    
    def clear_cache(self, collection_name: Optional[str] = None):
        """Clear query cache"""
        if collection_name:
            # Remove cache entries for specific collection
            keys_to_remove = [
                key for key in self.collection_cache.keys()
                if key.startswith(collection_name)
            ]
            for key in keys_to_remove:
                del self.collection_cache[key]
        else:
            self.collection_cache.clear()
        
        log.info(f"Cache cleared for {'collection' if collection_name else 'all collections'}")
    
    async def optimize_collection(self, collection_name: str) -> Dict[str, Any]:
        """Optimize collection by cleaning up and analyzing"""
        collection = self.get_collection(collection_name)
        
        optimization_results = {
            "collection_name": collection_name,
            "document_count": collection.count(),
            "cache_size": len([k for k in self.collection_cache.keys() if k.startswith(collection_name)]),
            "metrics": self.get_metrics(collection_name)
        }
        
        # Clear old cache entries
        current_time = time.time()
        keys_to_remove = []
        for key, entry in self.collection_cache.items():
            if key.startswith(collection_name):
                age = current_time - entry.timestamp
                if age > self.config.cache_ttl_seconds * 2:  # Remove entries older than 2x TTL
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.collection_cache[key]
        
        optimization_results["cache_cleaned"] = len(keys_to_remove)
        
        log.info(f"Optimization completed for '{collection_name}': {optimization_results}")
        return optimization_results


# Global instance getter
async def get_optimized_chromadb() -> OptimizedChromaDBManager:
    """Get optimized ChromaDB manager instance"""
    config = ChromaDBOptimizerConfig(
        enable_query_cache=settings.USE_LLM_CACHE,  # Reuse cache setting
        cache_ttl_seconds=settings.CACHE_TTL_SECONDS,
        batch_size=100,
        enable_metrics=True
    )
    return await OptimizedChromaDBManager.get_instance(config)


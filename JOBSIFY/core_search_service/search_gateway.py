"""
Core Search Service - Search Gateway

The main interface for all search operations. This is a thin wrapper around
the existing ChromaDB infrastructure in agents/chroma.py.

Design Principles:
- Domain Agnostic: Knows nothing about resumes, jobs, or courses
- Singleton Pattern: Single instance manages all search operations
- Reuses Existing Infrastructure: Wraps chroma.py functions
- Input Sanitization: Validates and sanitizes all inputs
"""

import logging
import time
from typing import Optional, Dict, Any, List, Tuple
import asyncio

from .models import (
    StandardSearchRequest,
    StandardSearchResponse,
    SearchResult,
    SearchMetadata
)

log = logging.getLogger(__name__)

# Import existing ChromaDB infrastructure
import sys
import os

# Add parent directory to path to import chroma module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import chroma
    from core.config import EMBEDDING_MODEL
    client = chroma.client
    _get_collection = chroma._get_collection
    embedding_fn = chroma.embedding_fn
    log.info("Successfully imported chroma module")
except ImportError as e:
    # Fallback for testing
    import traceback
    log.error(f"Failed to import chroma module: {e}")
    log.error(traceback.format_exc())
    client = None
    _get_collection = None
    embedding_fn = None
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Fallback default


class SearchGateway:
    """
    Main interface for search operations.
    
    This class provides a clean, domain-agnostic interface for vector search
    operations. It wraps the existing ChromaDB infrastructure and provides
    input validation, error handling, and response formatting.
    
    Usage:
        gateway = SearchGateway()
        
        request = StandardSearchRequest(
            target_collection="candidates_v1",
            search_text="Senior Python Developer",
            top_k=10
        )
        
        response = await gateway.search(request)
        
        for result in response.results:
            print(f"ID: {result.id}, Score: {result.score}")
    """
    
    _COUNT_CACHE_TTL = 60  # seconds

    def __init__(self):
        """Initialize the search gateway"""
        self._client = client
        self._embedding_fn = embedding_fn
        self._collection_cache: Dict[str, Any] = {}
        self._count_cache: Dict[str, Tuple[int, float]] = {}
        log.info("SearchGateway initialized")
    
    def _get_collection_safe(self, collection_name: str):
        """
        Get collection with caching and error handling.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            ChromaDB collection object
            
        Raises:
            ValueError: If collection doesn't exist
        """
        if collection_name in self._collection_cache:
            return self._collection_cache[collection_name]
        
        try:
            collection = _get_collection(collection_name)
            self._collection_cache[collection_name] = collection
            return collection
        except Exception as e:
            log.error(f"Failed to get collection '{collection_name}': {e}")
            raise ValueError(f"Collection '{collection_name}' not found or inaccessible")
    
    def _sanitize_search_text(self, text: str) -> str:
        """
        Sanitize search text to prevent injection attacks.
        
        Args:
            text: Raw search text
            
        Returns:
            Sanitized search text
        """
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Trim whitespace
        text = text.strip()
        
        # Limit length to prevent abuse
        max_length = 5000
        if len(text) > max_length:
            log.warning(f"Search text truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
        
        return text
    
    def _convert_distance_to_score(self, distance: float) -> float:
        """
        Convert ChromaDB distance to similarity score (0-1).
        
        ChromaDB returns L2 distance (lower is better).
        We convert to similarity score (higher is better).
        
        Args:
            distance: L2 distance from ChromaDB
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert L2 distance to similarity score
        # Using formula: similarity = 1 / (1 + distance)
        # This ensures: distance=0 → score=1, distance=∞ → score=0
        score = 1.0 / (1.0 + distance)
        return max(0.0, min(1.0, score))
    
    async def embed_text(self, text: str) -> Optional[List[float]]:
        """Pre-compute embedding vector for reuse across multiple search calls."""
        sanitized = self._sanitize_search_text(text)
        if not sanitized:
            return None
        try:
            fn = self._embedding_fn
            if fn is None or not callable(fn):
                try:
                    import chroma as _chroma
                    fn = _chroma._get_embedding_fn()
                except Exception:
                    return None
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, lambda: fn([sanitized]))
            if embeddings and len(embeddings) > 0:
                vec = embeddings[0]
                return vec.tolist() if hasattr(vec, 'tolist') else [float(x) for x in vec]
            return None
        except Exception as e:
            log.warning(f"Failed to pre-compute embedding: {e}")
            return None

    async def _get_cached_count(self, collection_name: str, collection) -> int:
        """Return collection document count, cached for _COUNT_CACHE_TTL seconds."""
        cached = self._count_cache.get(collection_name)
        if cached and (time.time() - cached[1]) < self._COUNT_CACHE_TTL:
            return cached[0]
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, collection.count)
        self._count_cache[collection_name] = (count, time.time())
        return count

    async def search(self, request: StandardSearchRequest) -> StandardSearchResponse:
        """
        Execute a search request.
        
        This is the main entry point for all search operations. It:
        1. Validates the request
        2. Sanitizes inputs
        3. Executes the vector search
        4. Formats the response
        
        Args:
            request: StandardSearchRequest with search parameters
            
        Returns:
            StandardSearchResponse with results
            
        Raises:
            ValueError: If request is invalid
            RuntimeError: If search fails
        """
        start_time = time.time()
        
        try:
            # Validate request
            if not request.target_collection:
                raise ValueError("target_collection is required")
            if not request.search_text:
                raise ValueError("search_text is required")
            
            # Sanitize search text
            sanitized_text = self._sanitize_search_text(request.search_text)
            if not sanitized_text:
                raise ValueError("search_text is empty after sanitization")
            
            log.info(
                f"Executing search: collection={request.target_collection}, "
                f"text='{sanitized_text[:50]}...', top_k={request.top_k}, "
                f"filters={len(request.filters)}"
            )
            
            # Get collection
            collection = self._get_collection_safe(request.target_collection)
            
            # Get total document count (cached to avoid repeated slow calls)
            total_documents = await self._get_cached_count(request.target_collection, collection)
            
            # Convert filters to ChromaDB format
            where_clause = request.to_chroma_filters()
            
            # Execute search — use pre-computed embedding when available to
            # avoid re-running SentenceTransformer on repeated fallback queries.
            loop = asyncio.get_event_loop()
            if request.query_embeddings:
                results = await loop.run_in_executor(
                    None,
                    lambda: collection.query(
                        query_embeddings=[request.query_embeddings],
                        n_results=request.top_k,
                        where=where_clause,
                        include=['documents', 'metadatas', 'distances']
                    )
                )
            else:
                results = await loop.run_in_executor(
                    None,
                    lambda: collection.query(
                        query_texts=[sanitized_text],
                        n_results=request.top_k,
                        where=where_clause,
                        include=['documents', 'metadatas', 'distances']
                    )
                )
            
            # Parse results
            search_results = []
            
            if results and results.get('ids') and results['ids'][0]:
                ids = results['ids'][0]
                distances = results.get('distances', [[]])[0]
                metadatas = results.get('metadatas', [[]])[0]
                documents = results.get('documents', [[]])[0] if request.include_documents else [None] * len(ids)
                
                for i, result_id in enumerate(ids):
                    distance = distances[i] if i < len(distances) else 0.0
                    score = self._convert_distance_to_score(distance)
                    
                    # Apply minimum score filter
                    if score < request.min_score:
                        continue
                    
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    document = documents[i] if i < len(documents) else None
                    
                    search_results.append(
                        SearchResult(
                            id=result_id,
                            score=score,
                            distance=distance,
                            metadata=metadata,
                            document=document
                        )
                    )
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create metadata
            filters_applied = [f"{f.field} {f.operator.value} {f.value}" for f in request.filters]
            
            search_metadata = SearchMetadata(
                collection_name=request.target_collection,
                embedding_model=EMBEDDING_MODEL,  # Centralized in core/config.py
                total_documents=total_documents,
                filters_applied=filters_applied,
                cache_hit=False
            )
            
            # Create response
            response = StandardSearchResponse(
                results=search_results,
                total_found=len(search_results),
                processing_time_ms=processing_time_ms,
                metadata=search_metadata,
                request=request
            )
            
            log.info(
                f"Search completed: found={len(search_results)}, "
                f"time={processing_time_ms}ms"
            )
            
            return response
            
        except ValueError as e:
            log.error(f"Invalid search request: {e}")
            raise
        except Exception as e:
            log.error(f"Search failed: {e}", exc_info=True)
            raise RuntimeError(f"Search operation failed: {str(e)}")
    
    async def search_by_id(self, collection_name: str, document_id: str) -> Optional[SearchResult]:
        """
        Retrieve a specific document by ID.
        
        Args:
            collection_name: Name of the collection
            document_id: Document ID to retrieve
            
        Returns:
            SearchResult if found, None otherwise
        """
        try:
            collection = self._get_collection_safe(collection_name)
            
            # Get document by ID
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: collection.get(
                    ids=[document_id],
                    include=['documents', 'metadatas']
                )
            )
            
            if results and results.get('ids') and results['ids']:
                metadata = results.get('metadatas', [{}])[0]
                document = results.get('documents', [None])[0]
                
                return SearchResult(
                    id=document_id,
                    score=1.0,  # Perfect match for direct ID lookup
                    distance=0.0,
                    metadata=metadata,
                    document=document
                )
            
            return None
            
        except Exception as e:
            log.error(f"Failed to retrieve document {document_id}: {e}")
            return None
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self._get_collection_safe(collection_name)
            
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(None, collection.count)
            
            return {
                "name": collection_name,
                "count": count,
                "embedding_model": EMBEDDING_MODEL  # Centralized in core/config.py
            }
            
        except Exception as e:
            log.error(f"Failed to get stats for collection {collection_name}: {e}")
            return {
                "name": collection_name,
                "error": str(e)
            }
    
    def clear_cache(self):
        """Clear the collection cache"""
        self._collection_cache.clear()
        log.info("Collection cache cleared")


# Singleton instance
_gateway_instance: Optional[SearchGateway] = None


def get_search_gateway() -> SearchGateway:
    """
    Get the singleton SearchGateway instance.
    
    Returns:
        SearchGateway instance
    """
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = SearchGateway()
    return _gateway_instance

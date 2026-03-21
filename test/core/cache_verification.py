"""
Utility functions for verifying cache hits from Gemini API responses.

This module provides functions to check if a request hit the cache by inspecting
the usage_metadata in the API response.
"""

from typing import Dict, Any, Optional
import logging
from core.observability import get_recent_metrics


log = logging.getLogger("main")


def verify_gemini_cache_hit(usage_metadata: Dict[str, Any]) -> bool:
    """
    Verify if a Gemini API request hit the cache.
    
    Args:
        usage_metadata: Usage metadata from Gemini API response
        
    Returns:
        True if cached_content_token_count > 0, False otherwise
        
    Example:
        >>> metadata = {"cached_content_token_count": 50, "prompt_token_count": 100}
        >>> verify_gemini_cache_hit(metadata)
        True
    """
    cached_tokens = usage_metadata.get("cached_content_token_count", 0)
    return cached_tokens > 0


def get_cache_statistics(limit: int = 100) -> Dict[str, Any]:
    """
    Get cache statistics from recent metrics.
    
    Args:
        limit: Number of recent metrics to analyze
        
    Returns:
        Dict with cache statistics including:
        - total_requests: Total number of requests
        - internal_cache_hits: Number of internal cache hits
        - api_cache_hits: Number of Gemini API cache hits
        - no_cache: Number of requests with no cache
        - internal_cache_hit_rate: Percentage of internal cache hits
        - api_cache_hit_rate: Percentage of API cache hits
        - total_cached_tokens: Total cached tokens used
    """
    metrics = get_recent_metrics(limit=limit)
    
    if not metrics:
        return {
            "total_requests": 0,
            "internal_cache_hits": 0,
            "api_cache_hits": 0,
            "no_cache": 0,
            "internal_cache_hit_rate": 0.0,
            "api_cache_hit_rate": 0.0,
            "total_cached_tokens": 0,
            "avg_cached_tokens_per_hit": 0.0
        }
    
    total = len(metrics)
    internal_hits = 0
    api_hits = 0
    total_cached_tokens = 0
    
    for metric in metrics:
        metadata = metric.get("metadata", {})
        
        # Check internal cache
        if metadata.get("internal_cache_hit") or metric.get("cache_hit"):
            # Distinguish between internal and API cache
            if metadata.get("cache_source") == "internal_llm_cache":
                internal_hits += 1
            elif metadata.get("api_cache_hit"):
                api_hits += 1
                total_cached_tokens += metadata.get("cached_content_token_count", 0)
        elif metadata.get("api_cache_hit"):
            api_hits += 1
            total_cached_tokens += metadata.get("cached_content_token_count", 0)
    
    no_cache = total - internal_hits - api_hits
    
    return {
        "total_requests": total,
        "internal_cache_hits": internal_hits,
        "api_cache_hits": api_hits,
        "no_cache": no_cache,
        "internal_cache_hit_rate": (internal_hits / total * 100) if total > 0 else 0.0,
        "api_cache_hit_rate": (api_hits / total * 100) if total > 0 else 0.0,
        "total_cached_tokens": total_cached_tokens,
        "avg_cached_tokens_per_hit": (total_cached_tokens / api_hits) if api_hits > 0 else 0.0
    }


def check_recent_cache_performance(limit: int = 50) -> None:
    """
    Print a summary of recent cache performance.
    
    Args:
        limit: Number of recent metrics to analyze
    """
    stats = get_cache_statistics(limit=limit)
    
    log.info("CACHE PERFORMANCE SUMMARY")
    log.info("Total Requests: %s", stats["total_requests"])
    log.info(
        "Internal Cache - Hits: %s | Hit Rate: %.1f%%",
        stats["internal_cache_hits"],
        stats["internal_cache_hit_rate"],
    )
    log.info(
        "Gemini API Cache - Hits: %s | Hit Rate: %.1f%%",
        stats["api_cache_hits"],
        stats["api_cache_hit_rate"],
    )
    log.info("Total Cached Tokens: %s", stats["total_cached_tokens"])
    log.info("Avg Cached Tokens/Hit: %.1f", stats["avg_cached_tokens_per_hit"])
    log.info("No Cache: %s", stats["no_cache"])


def find_cache_hits_by_agent(agent_name: str, limit: int = 100) -> list:
    """
    Find all cache hits for a specific agent.
    
    Args:
        agent_name: Name of the agent to filter by
        limit: Number of recent metrics to check
        
    Returns:
        List of metrics where cache was hit
    """
    metrics = get_recent_metrics(limit=limit)
    
    cache_hits = []
    for metric in metrics:
        if metric.get("agent_name") == agent_name:
            if metric.get("cache_hit") or metric.get("metadata", {}).get("api_cache_hit"):
                cache_hits.append(metric)
    
    return cache_hits


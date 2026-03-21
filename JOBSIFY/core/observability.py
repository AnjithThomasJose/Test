"""
Observability and metrics tracking for agent tasks.
Provides per-task metrics including tokens, latency, retries, cache hits, etc.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import deque
import threading

@dataclass
class TaskMetrics:
    """Metrics for a single task execution."""
    task_id: str
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    cache_hit: bool = False
    k: Optional[int] = None  # RAG k value
    context_tokens: Optional[int] = None  # RAG context tokens
    error_code: Optional[str] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms,
            "retries": self.retries,
            "cache_hit": self.cache_hit,
            "k": self.k,
            "context_tokens": self.context_tokens,
            "error_code": self.error_code,
            "success": self.success,
            "metadata": self.metadata
        }

class MetricsCollector:
    """Collects and stores task metrics."""
    
    def __init__(self, max_entries: int = 10000):
        self.metrics: deque = deque(maxlen=max_entries)
        self._lock = threading.Lock()
    
    def record(self, metrics: TaskMetrics):
        """Record task metrics."""
        with self._lock:
            self.metrics.append(metrics)
    
    def get_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics."""
        with self._lock:
            return [m.to_dict() for m in list(self.metrics)[-limit:]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        with self._lock:
            if not self.metrics:
                return {
                    "total_tasks": 0,
                    "success_rate": 0.0,
                    "avg_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "cache_hit_rate": 0.0,
                    "avg_retries": 0.0,
                    "total_tokens_in": 0,
                    "total_tokens_out": 0,
                    "total_estimated_cost": 0.0,
                }
            
            metrics_list = list(self.metrics)
            latencies = [m.latency_ms for m in metrics_list]
            latencies.sort()
            
            successful = sum(1 for m in metrics_list if m.success)
            cache_hits = sum(1 for m in metrics_list if m.cache_hit)
            total_retries = sum(m.retries for m in metrics_list)
            total_tokens_in = sum(m.tokens_in for m in metrics_list)
            total_tokens_out = sum(m.tokens_out for m in metrics_list)
            total_estimated_cost = sum(
                float(m.metadata.get("cost", 0) or 0) for m in metrics_list
            )
            
            n = len(metrics_list)
            p95_idx = int(n * 0.95)
            p99_idx = int(n * 0.99)
            
            return {
                "total_tasks": n,
                "success_rate": successful / n if n > 0 else 0.0,
                "avg_latency_ms": sum(latencies) / n if n > 0 else 0.0,
                "p95_latency_ms": latencies[p95_idx] if p95_idx < n else 0.0,
                "p99_latency_ms": latencies[p99_idx] if p99_idx < n else 0.0,
                "cache_hit_rate": cache_hits / n if n > 0 else 0.0,
                "avg_retries": total_retries / n if n > 0 else 0.0,
                "total_tokens_in": total_tokens_in,
                "total_tokens_out": total_tokens_out,
                "total_estimated_cost": total_estimated_cost,
            }

# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()

def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        with _collector_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()
    return _metrics_collector

def create_task_metrics(
    agent_name: str,
    task_id: Optional[str] = None
) -> TaskMetrics:
    """Create a new task metrics instance."""
    return TaskMetrics(
        task_id=task_id or str(uuid.uuid4()),
        agent_name=agent_name,
        start_time=time.time()
    )

def log_task_metrics(metrics: TaskMetrics):
    """Log task metrics to the collector."""
    collector = get_metrics_collector()
    metrics.end_time = time.time()
    metrics.latency_ms = (metrics.end_time - metrics.start_time) * 1000
    collector.record(metrics)

def get_task_stats() -> Dict[str, Any]:
    """Get aggregated task statistics."""
    collector = get_metrics_collector()
    return collector.get_stats()

def get_recent_metrics(limit: int = 100) -> List[Dict[str, Any]]:
    """Get recent task metrics."""
    collector = get_metrics_collector()
    return collector.get_recent(limit)

class ObservabilityManager:
    """Manager for observability operations."""
    
    def __init__(self):
        self.collector = get_metrics_collector()
    
    def log_prompt_response(
        self,
        prompt: str,
        response: str,
        token_count: int,
        cost: float,
        latency_ms: float,
        agent_name: str,
        success: bool = True,
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log a prompt/response pair for observability.
        
        Args:
            metadata: Optional dict containing additional metadata like:
                - api_cache_hit: bool indicating if Gemini API cache was used
                - cached_content_token_count: Number of cached tokens used
                - prompt_token_count: Actual prompt tokens from API
                - output_token_count: Actual output tokens from API
        """
        # Extract cache information from metadata if available
        api_cache_hit = False
        if metadata:
            api_cache_hit = metadata.get("api_cache_hit", False)
            cached_tokens = metadata.get("cached_content_token_count", 0)
            prompt_tokens = metadata.get("prompt_token_count")
            output_tokens = metadata.get("output_token_count")
        else:
            cached_tokens = 0
            prompt_tokens = None
            output_tokens = None
        
        # Use actual token counts from API if available
        tokens_in = prompt_tokens if prompt_tokens is not None else (len(prompt.split()) if prompt else 0)
        tokens_out = output_tokens if output_tokens is not None else (len(response.split()) if response else 0)
        
        # Create task metrics for this LLM call
        base_metadata = {
            "cost": cost,
            "token_count": token_count,
            "prompt_length": len(prompt) if prompt else 0,
            "response_length": len(response) if response else 0
        }
        
        # Add cache information if available
        if api_cache_hit:
            base_metadata["api_cache_hit"] = True
            base_metadata["cached_content_token_count"] = cached_tokens
        
        # Merge with any additional metadata
        if metadata:
            base_metadata.update(metadata)
        
        task_metrics = TaskMetrics(
            task_id=str(uuid.uuid4()),
            agent_name=agent_name,
            start_time=time.time() - (latency_ms / 1000.0),
            end_time=time.time(),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            retries=0,
            cache_hit=api_cache_hit,  # Set cache_hit based on API response
            success=success,
            error_code=error_type,
            metadata=base_metadata
        )
        self.collector.record(task_metrics)

# Global observability manager instance
observability_manager = ObservabilityManager()

def log_prompt_response(
    prompt: str,
    response: str,
    token_count: int,
    cost: float,
    latency_ms: float,
    agent_name: str,
    success: bool = True,
    error_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """Log a prompt/response pair for observability.
    
    Args:
        metadata: Optional dict containing additional metadata like cache information
    """
    observability_manager.log_prompt_response(
        prompt=prompt,
        response=response,
        token_count=token_count,
        cost=cost,
        latency_ms=latency_ms,
        agent_name=agent_name,
        success=success,
        error_type=error_type,
        metadata=metadata
    )

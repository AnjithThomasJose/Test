"""
Centralized memory management for all KAFIN agents.

This module provides a base memory class that eliminates duplication
across all agent-specific memory implementations.
"""

import asyncio
import time
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod


class BaseAgentMemory(ABC):
    """Base class for all agent memory management."""
    
    def __init__(self, tenant_id: str = "default_tenant", adaptation_window: int = 50):
        self.tenant_id = tenant_id
        self.history = []
        self.success_patterns = {}
        self.failure_patterns = {}
        self.shared_context = {}
        self.lock = asyncio.Lock()
        self.adaptation_window = adaptation_window
    
    async def record_attempt(self, input_type: str, method: str, success: bool, 
                           confidence: float, processing_time: float):
        """Record analysis attempt for learning and adaptation."""
        async with self.lock:
            self.history.append({
                'timestamp': time.time(),
                'input_type': input_type,
                'method': method,
                'success': success,
                'confidence': confidence,
                'processing_time': processing_time
            })
            
            # Keep only recent history
            if len(self.history) > self.adaptation_window:
                self.history = self.history[-self.adaptation_window:]
    
    async def get_success_rate(self, method: str) -> float:
        """Get success rate for a specific method."""
        async with self.lock:
            attempts = [h for h in self.history if h['method'] == method]
            if not attempts:
                return 0.0
            successful = [h for h in attempts if h['success']]
            return len(successful) / len(attempts)
    
    async def should_use_hybrid_approach(self) -> bool:
        """Adaptive decision: should we use hybrid approach?"""
        async with self.lock:
            llm_rate = await self.get_success_rate('llm')
            det_rate = await self.get_success_rate('deterministic')
            return llm_rate > 0.6 and det_rate > 0.5
    
    async def share_context(self, key: str, value: Any):
        """Share context with other agents."""
        async with self.lock:
            self.shared_context[key] = value
    
    async def get_context(self, key: str, default: Any = None) -> Any:
        """Get shared context."""
        async with self.lock:
            return self.shared_context.get(key, default)
    
    async def get_shared_context(self, key: str) -> Any:
        """Get shared context (alias for compatibility)."""
        return await self.get_context(key)
    
    async def get_recent_attempts(self, method: str = None, limit: int = 10) -> list:
        """Get recent attempts, optionally filtered by method."""
        async with self.lock:
            attempts = self.history
            if method:
                attempts = [h for h in attempts if h['method'] == method]
            return attempts[-limit:]
    
    async def get_average_confidence(self, method: str = None) -> float:
        """Get average confidence score."""
        async with self.lock:
            attempts = self.history
            if method:
                attempts = [h for h in attempts if h['method'] == method]
            if not attempts:
                return 0.0
            return sum(h['confidence'] for h in attempts) / len(attempts)
    
    async def get_average_processing_time(self, method: str = None) -> float:
        """Get average processing time."""
        async with self.lock:
            attempts = self.history
            if method:
                attempts = [h for h in attempts if h['method'] == method]
            if not attempts:
                return 0.0
            return sum(h['processing_time'] for h in attempts) / len(attempts)
    
    async def clear_history(self):
        """Clear history (useful for testing)."""
        async with self.lock:
            self.history = []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        async with self.lock:
            total_attempts = len(self.history)
            successful_attempts = len([h for h in self.history if h['success']])
            
            methods = set(h['method'] for h in self.history)
            method_stats = {}
            for method in methods:
                method_attempts = [h for h in self.history if h['method'] == method]
                method_successful = len([h for h in method_attempts if h['success']])
                method_stats[method] = {
                    'total': len(method_attempts),
                    'successful': method_successful,
                    'success_rate': method_successful / len(method_attempts) if method_attempts else 0.0,
                    'avg_confidence': sum(h['confidence'] for h in method_attempts) / len(method_attempts) if method_attempts else 0.0,
                    'avg_processing_time': sum(h['processing_time'] for h in method_attempts) / len(method_attempts) if method_attempts else 0.0
                }
            
            return {
                'tenant_id': self.tenant_id,
                'total_attempts': total_attempts,
                'successful_attempts': successful_attempts,
                'overall_success_rate': successful_attempts / total_attempts if total_attempts else 0.0,
                'methods': method_stats,
                'shared_context_keys': list(self.shared_context.keys())
            }


# Global memory instances storage
_memory_instances: Dict[str, Dict[str, BaseAgentMemory]] = {}
_memory_creation_lock = asyncio.Lock()


async def get_agent_memory(agent_name: str, tenant_id: str = "default_tenant", 
                          memory_class: type = BaseAgentMemory) -> BaseAgentMemory:
    """Get or create tenant-scoped agent memory."""
    if agent_name not in _memory_instances:
        _memory_instances[agent_name] = {}
    
    if tenant_id in _memory_instances[agent_name]:
        return _memory_instances[agent_name][tenant_id]
    
    async with _memory_creation_lock:
        # Double-check pattern to avoid race conditions
        if tenant_id not in _memory_instances[agent_name]:
            _memory_instances[agent_name][tenant_id] = memory_class(tenant_id)
        return _memory_instances[agent_name][tenant_id]


def clear_all_memory():
    """Clear all memory instances (useful for testing)."""
    global _memory_instances
    _memory_instances = {}


async def get_memory_stats() -> Dict[str, Any]:
    """Get statistics for all memory instances."""
    stats = {}
    for agent_name, tenant_memories in _memory_instances.items():
        stats[agent_name] = {}
        for tenant_id, memory in tenant_memories.items():
            stats[agent_name][tenant_id] = await memory.get_stats()
    return stats

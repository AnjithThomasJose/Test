"""
Granular Timing Utilities for ML Model Loading and Parsing Performance

This module provides utilities to track and separate:
- ML model loading time (cold start)
- Actual parsing logic time (warm start)
- Performance metrics and reporting
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class TimingMetrics:
    """Timing metrics for agent performance."""
    agent_name: str
    ml_load_time: float = 0.0
    parsing_time: float = 0.0
    total_time: float = 0.0
    is_cold_start: bool = True
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    @property
    def warm_parsing_time(self) -> float:
        """Get parsing time excluding ML load (warm start equivalent)."""
        return self.parsing_time
    
    @property
    def cold_start_overhead(self) -> float:
        """Get ML load overhead for cold starts."""
        return self.ml_load_time if self.is_cold_start else 0.0

class MLModelTimer:
    """Timer for tracking ML model loading and parsing separately."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.metrics = TimingMetrics(agent_name)
        self.start_time = None
        self.ml_load_start = None
        self.parsing_start = None
        
    def start_total(self):
        """Start total timing."""
        self.start_time = time.time()
        self.metrics.timestamp = self.start_time
        
    def start_ml_load(self):
        """Start ML model loading timing."""
        self.ml_load_start = time.time()
        
    def end_ml_load(self):
        """End ML model loading timing."""
        if self.ml_load_start:
            self.metrics.ml_load_time = time.time() - self.ml_load_start
            self.metrics.is_cold_start = True
            
    def start_parsing(self):
        """Start parsing logic timing."""
        self.parsing_start = time.time()
        
    def end_parsing(self):
        """End parsing logic timing."""
        if self.parsing_start:
            self.metrics.parsing_time = time.time() - self.parsing_start
            
    def end_total(self):
        """End total timing and calculate metrics."""
        if self.start_time:
            self.metrics.total_time = time.time() - self.start_time
            
    def get_metrics(self) -> TimingMetrics:
        """Get current timing metrics."""
        return self.metrics
    
    def log_metrics(self, logger: logging.Logger):
        """Log timing metrics with appropriate detail."""
        metrics = self.get_metrics()
        
        # Log ML load time separately
        if metrics.ml_load_time > 0:
            logger.info(f"🚀 ML_LOAD_TIMING: {self.agent_name} ML model load: {metrics.ml_load_time:.2f}s")
        
        # Log parsing time separately
        if metrics.parsing_time > 0:
            logger.info(f"⚡ PARSING_TIMING: {self.agent_name} parsing logic: {metrics.parsing_time:.2f}s")
        
        # Log total time
        logger.info(f"🚀 PARALLEL_TIMING: {self.agent_name} total: {metrics.total_time:.2f}s")
        
        # Log performance insights
        if metrics.is_cold_start and metrics.ml_load_time > 0:
            overhead_percentage = (metrics.ml_load_time / metrics.total_time) * 100
            logger.info(f"📊 PERFORMANCE: {self.agent_name} cold start overhead: {overhead_percentage:.1f}%")

class PerformanceTracker:
    """Global performance tracker for all agents."""
    
    def __init__(self):
        self.agent_metrics: Dict[str, TimingMetrics] = {}
        self.log = logging.getLogger(__name__)
        
    def record_metrics(self, metrics: TimingMetrics):
        """Record metrics for an agent."""
        self.agent_metrics[metrics.agent_name] = metrics
        
    def get_agent_metrics(self, agent_name: str) -> Optional[TimingMetrics]:
        """Get metrics for a specific agent."""
        return self.agent_metrics.get(agent_name)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all agents."""
        if not self.agent_metrics:
            return {}
        
        summary = {
            'total_agents': len(self.agent_metrics),
            'cold_start_agents': sum(1 for m in self.agent_metrics.values() if m.is_cold_start),
            'warm_start_agents': sum(1 for m in self.agent_metrics.values() if not m.is_cold_start),
            'total_ml_load_time': sum(m.ml_load_time for m in self.agent_metrics.values()),
            'total_parsing_time': sum(m.parsing_time for m in self.agent_metrics.values()),
            'total_execution_time': sum(m.total_time for m in self.agent_metrics.values()),
            'agent_breakdown': {}
        }
        
        for agent_name, metrics in self.agent_metrics.items():
            summary['agent_breakdown'][agent_name] = {
                'ml_load_time': metrics.ml_load_time,
                'parsing_time': metrics.parsing_time,
                'total_time': metrics.total_time,
                'is_cold_start': metrics.is_cold_start,
                'cold_start_overhead': metrics.cold_start_overhead
            }
        
        return summary
    
    def log_performance_summary(self):
        """Log comprehensive performance summary."""
        summary = self.get_performance_summary()
        if not summary:
            return
        
        self.log.info("📊 PERFORMANCE SUMMARY:")
        self.log.info(f"   Total Agents: {summary['total_agents']}")
        self.log.info(f"   Cold Start Agents: {summary['cold_start_agents']}")
        self.log.info(f"   Warm Start Agents: {summary['warm_start_agents']}")
        self.log.info(f"   Total ML Load Time: {summary['total_ml_load_time']:.2f}s")
        self.log.info(f"   Total Parsing Time: {summary['total_parsing_time']:.2f}s")
        self.log.info(f"   Total Execution Time: {summary['total_execution_time']:.2f}s")
        
        # Log individual agent performance
        for agent_name, breakdown in summary['agent_breakdown'].items():
            self.log.info(f"   {agent_name}: ML={breakdown['ml_load_time']:.2f}s, "
                           f"Parsing={breakdown['parsing_time']:.2f}s, "
                           f"Total={breakdown['total_time']:.2f}s, "
                           f"ColdStart={breakdown['is_cold_start']}")

# Global performance tracker
performance_tracker = PerformanceTracker()

@contextmanager
def track_ml_model_loading(agent_name: str, logger: logging.Logger):
    """Context manager for tracking ML model loading."""
    timer = MLModelTimer(agent_name)
    timer.start_total()
    timer.start_ml_load()
    
    try:
        yield timer
    finally:
        timer.end_ml_load()
        timer.end_total()
        timer.log_metrics(logger)
        performance_tracker.record_metrics(timer.get_metrics())

@contextmanager
def track_parsing_logic(agent_name: str, logger: logging.Logger):
    """Context manager for tracking parsing logic."""
    timer = MLModelTimer(agent_name)
    timer.start_total()
    timer.start_parsing()
    
    try:
        yield timer
    finally:
        timer.end_parsing()
        timer.end_total()
        timer.log_metrics(logger)
        performance_tracker.record_metrics(timer.get_metrics())

def timing_decorator(agent_name: str):
    """Decorator for automatic timing of agent functions."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            timer = MLModelTimer(agent_name)
            timer.start_total()
            
            try:
                result = await func(*args, **kwargs)
                timer.end_total()
                timer.log_metrics(logger)
                performance_tracker.record_metrics(timer.get_metrics())
                return result
            except Exception as e:
                timer.end_total()
                timer.log_metrics(logger)
                performance_tracker.record_metrics(timer.get_metrics())
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            timer = MLModelTimer(agent_name)
            timer.start_total()
            
            try:
                result = func(*args, **kwargs)
                timer.end_total()
                timer.log_metrics(logger)
                performance_tracker.record_metrics(timer.get_metrics())
                return result
            except Exception as e:
                timer.end_total()
                timer.log_metrics(logger)
                performance_tracker.record_metrics(timer.get_metrics())
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def get_performance_summary() -> Dict[str, Any]:
    """Get current performance summary."""
    return performance_tracker.get_performance_summary()

def log_performance_summary():
    """Log current performance summary."""
    performance_tracker.log_performance_summary()

# Utility functions for common timing patterns
def log_ml_model_load(logger: logging.Logger, agent_name: str, model_name: str, load_time: float):
    """Log ML model loading with details."""
    logger.info(f"🤖 ML_MODEL_LOAD: {agent_name} loaded {model_name} in {load_time:.2f}s")

def log_parsing_start(logger: logging.Logger, agent_name: str, data_size: int):
    """Log parsing start with data size."""
    logger.info(f"⚡ PARSING_START: {agent_name} processing {data_size} chars")

def log_parsing_complete(logger: logging.Logger, agent_name: str, parsing_time: float, 
                        items_found: int = 0):
    """Log parsing completion with metrics."""
    logger.info(f"⚡ PARSING_COMPLETE: {agent_name} completed in {parsing_time:.2f}s, "
               f"found {items_found} items")

def log_performance_insights(logger: logging.Logger, agent_name: str, metrics: TimingMetrics):
    """Log performance insights and recommendations."""
    if metrics.is_cold_start and metrics.ml_load_time > 0:
        overhead_percentage = (metrics.ml_load_time / metrics.total_time) * 100
        logger.info(f"💡 PERFORMANCE_INSIGHT: {agent_name} cold start overhead: {overhead_percentage:.1f}%")
        
        if overhead_percentage > 50:
            logger.warning(f"⚠️ PERFORMANCE_WARNING: {agent_name} has high cold start overhead")
        elif overhead_percentage < 20:
            logger.info(f"✅ PERFORMANCE_GOOD: {agent_name} has low cold start overhead")
    
    if metrics.parsing_time > 10:
        logger.warning(f"⚠️ PERFORMANCE_WARNING: {agent_name} parsing time is high: {metrics.parsing_time:.2f}s")
    elif metrics.parsing_time < 2:
        logger.info(f"✅ PERFORMANCE_GOOD: {agent_name} parsing time is fast: {metrics.parsing_time:.2f}s")

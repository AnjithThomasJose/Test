"""
Quota Management System for KAFIN Agents

This module provides comprehensive quota management with:
- Multi-provider quota tracking
- Request batching and optimization
- Exponential backoff for rate limits
- Cost monitoring and alerts
- Automatic fallback to alternative models
- Gemini tier 2 aware limits (2000 RPM, 8M TPM when GEMINI_TIER=2)
"""

import asyncio
import os
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque

log = logging.getLogger(__name__)

class QuotaStatus(Enum):
    """Quota status indicators"""
    AVAILABLE = "available"
    NEAR_LIMIT = "near_limit"
    EXCEEDED = "exceeded"
    UNKNOWN = "unknown"

class RateLimitType(Enum):
    """Types of rate limits"""
    REQUESTS_PER_MINUTE = "rpm"
    TOKENS_PER_MINUTE = "tpm"
    REQUESTS_PER_DAY = "rpd"
    TOKENS_PER_DAY = "tpd"
    COST_PER_DAY = "cpd"

@dataclass
class QuotaConfig:
    """Configuration for quota limits"""
    provider: str
    daily_request_limit: int = 10000
    hourly_request_limit: int = 1000
    per_minute_request_limit: int = 60
    daily_token_limit: int = 1000000
    hourly_token_limit: int = 100000
    per_minute_token_limit: int = 10000
    daily_cost_limit: float = 100.0
    hourly_cost_limit: float = 10.0
    warning_threshold: float = 0.8  # Alert when 80% of limit reached
    burst_limit_multiplier: float = 1.5  # Allow 50% burst over limits
    enabled: bool = True

@dataclass
class QuotaUsage:
    """Track quota usage for a provider"""
    provider: str
    daily_requests: int = 0
    hourly_requests: int = 0
    per_minute_requests: int = 0
    daily_tokens: int = 0
    hourly_tokens: int = 0
    per_minute_tokens: int = 0
    daily_cost: float = 0.0
    hourly_cost: float = 0.0
    last_reset_daily: datetime = field(default_factory=datetime.now)
    last_reset_hourly: datetime = field(default_factory=datetime.now)
    last_reset_minute: datetime = field(default_factory=datetime.now)
    
    def reset_if_needed(self):
        """Reset counters if time windows have passed"""
        now = datetime.now()
        
        # Reset daily counters
        if now.date() > self.last_reset_daily.date():
            self.daily_requests = 0
            self.daily_tokens = 0
            self.daily_cost = 0.0
            self.last_reset_daily = now
        
        # Reset hourly counters
        if now.hour != self.last_reset_hourly.hour or now.date() > self.last_reset_hourly.date():
            self.hourly_requests = 0
            self.hourly_tokens = 0
            self.hourly_cost = 0.0
            self.last_reset_hourly = now
        
        # Reset per-minute counters
        if now.minute != self.last_reset_minute.minute or now.hour != self.last_reset_minute.hour:
            self.per_minute_requests = 0
            self.per_minute_tokens = 0
            self.last_reset_minute = now

@dataclass
class RequestBatch:
    """Represents a batch of requests"""
    requests: List[Dict[str, Any]]
    provider: str
    created_at: datetime
    priority: int = 1  # 1-5, 5 being highest priority
    max_wait_time: float = 30.0  # Maximum time to wait for batch completion

class QuotaManager:
    """Centralized quota management system"""
    
    def __init__(self):
        self.quota_configs: Dict[str, QuotaConfig] = {}
        self.quota_usage: Dict[str, QuotaUsage] = {}
        self.batch_queues: Dict[str, deque] = defaultdict(deque)
        self.batch_processors: Dict[str, asyncio.Task] = {}
        self.rate_limit_windows: Dict[str, deque] = defaultdict(deque)
        self.backoff_timers: Dict[str, float] = {}
        self.cost_alerts_sent: Dict[str, datetime] = {}
        
        # Initialize default configurations
        self._initialize_default_configs()
        
        # Background tasks will be started lazily when event loop is available
        self._background_tasks_started = False
    
    def _get_gemini_quota_config(self) -> QuotaConfig:
        """Get Gemini quota config based on GEMINI_TIER env var.
        Tier 1 (default): 60 RPM, 20K TPM, 50K RPD
        Tier 2: 2000 RPM, 8M TPM, 100K RPD (matches Google Gemini 2.5 Flash tier 2)
        """
        tier = int(os.getenv("GEMINI_TIER", "1"))
        if tier >= 2:
            return QuotaConfig(
                provider="gemini",
                daily_request_limit=100000,
                hourly_request_limit=10000,
                per_minute_request_limit=2000,
                daily_token_limit=8000000,
                hourly_token_limit=800000,
                per_minute_token_limit=8000000,
                daily_cost_limit=100.0,
                hourly_cost_limit=10.0
            )
        return QuotaConfig(
            provider="gemini",
            daily_request_limit=50000,
            hourly_request_limit=5000,
            per_minute_request_limit=60,
            daily_token_limit=2000000,
            hourly_token_limit=200000,
            per_minute_token_limit=20000,
            daily_cost_limit=50.0,
            hourly_cost_limit=5.0
        )

    def _initialize_default_configs(self):
        """Initialize default quota configurations for providers"""
        gemini_config = self._get_gemini_quota_config()
        log.info(f"Gemini quota: tier={os.getenv('GEMINI_TIER', '1')}, "
                 f"RPM={gemini_config.per_minute_request_limit}, "
                 f"TPM={gemini_config.per_minute_token_limit}")
        default_configs = {
            "gemini": gemini_config,
            "openai": QuotaConfig(
                provider="openai",
                daily_request_limit=10000,
                hourly_request_limit=1000,
                per_minute_request_limit=40,
                daily_token_limit=1000000,
                hourly_token_limit=100000,
                per_minute_token_limit=80000,
                daily_cost_limit=100.0,
                hourly_cost_limit=10.0
            ),
            "anthropic": QuotaConfig(
                provider="anthropic",
                daily_request_limit=5000,
                hourly_request_limit=500,
                per_minute_request_limit=50,
                daily_token_limit=500000,
                hourly_token_limit=50000,
                per_minute_token_limit=90000,
                daily_cost_limit=75.0,
                hourly_cost_limit=7.5
            )
        }
        
        for provider, config in default_configs.items():
            self.set_quota_config(provider, config)
    
    def set_quota_config(self, provider: str, config: QuotaConfig):
        """Set quota configuration for a provider"""
        self.quota_configs[provider] = config
        if provider not in self.quota_usage:
            self.quota_usage[provider] = QuotaUsage(provider=provider)
        log.info(f"Set quota config for {provider}: {config.daily_request_limit} req/day, ${config.daily_cost_limit} cost/day")
    
    def check_quota(self, provider: str, request_cost: float = 0.0, 
                   token_count: int = 0, allow_burst: bool = False) -> QuotaStatus:
        """Check if request can be made within quota limits"""
        if provider not in self.quota_configs:
            return QuotaStatus.UNKNOWN
        
        config = self.quota_configs[provider]
        if not config.enabled:
            return QuotaStatus.AVAILABLE
        
        usage = self.quota_usage[provider]
        usage.reset_if_needed()
        
        # Check if we're in backoff period
        if provider in self.backoff_timers and time.time() < self.backoff_timers[provider]:
            return QuotaStatus.EXCEEDED
        
        # Apply burst multiplier if allowed
        multiplier = config.burst_limit_multiplier if allow_burst else 1.0
        
        # Check daily limits
        if usage.daily_requests >= config.daily_request_limit * multiplier:
            return QuotaStatus.EXCEEDED
        
        if usage.daily_tokens + token_count > config.daily_token_limit * multiplier:
            return QuotaStatus.EXCEEDED
        
        if usage.daily_cost + request_cost > config.daily_cost_limit * multiplier:
            return QuotaStatus.EXCEEDED
        
        # Check hourly limits
        if usage.hourly_requests >= config.hourly_request_limit * multiplier:
            return QuotaStatus.EXCEEDED
        
        if usage.hourly_tokens + token_count > config.hourly_token_limit * multiplier:
            return QuotaStatus.EXCEEDED
        
        if usage.hourly_cost + request_cost > config.hourly_cost_limit * multiplier:
            return QuotaStatus.EXCEEDED
        
        # Check per-minute limits
        if usage.per_minute_requests >= config.per_minute_request_limit * multiplier:
            return QuotaStatus.EXCEEDED
        
        if usage.per_minute_tokens + token_count > config.per_minute_token_limit * multiplier:
            return QuotaStatus.EXCEEDED
        
        # Check warning thresholds
        warning_threshold = config.warning_threshold
        if (usage.daily_requests >= config.daily_request_limit * warning_threshold or
            usage.daily_cost >= config.daily_cost_limit * warning_threshold):
            return QuotaStatus.NEAR_LIMIT
        
        return QuotaStatus.AVAILABLE
    
    def record_usage(self, provider: str, request_cost: float = 0.0, 
                    token_count: int = 0, success: bool = True):
        """Record usage for a provider"""
        if provider not in self.quota_usage:
            self.quota_usage[provider] = QuotaUsage(provider=provider)
        
        usage = self.quota_usage[provider]
        usage.reset_if_needed()
        
        # Update counters
        usage.daily_requests += 1
        usage.hourly_requests += 1
        usage.per_minute_requests += 1
        usage.daily_tokens += token_count
        usage.hourly_tokens += token_count
        usage.per_minute_tokens += token_count
        usage.daily_cost += request_cost
        usage.hourly_cost += request_cost
        
        # Check for alerts
        self._check_cost_alerts(provider)
        
        log.debug(f"Recorded usage for {provider}: {token_count} tokens, ${request_cost:.6f}")
    
    def _check_cost_alerts(self, provider: str):
        """Check if cost alerts should be sent"""
        if provider not in self.quota_configs:
            return
        
        config = self.quota_configs[provider]
        usage = self.quota_usage[provider]
        
        # Send alert if approaching daily cost limit
        if usage.daily_cost >= config.daily_cost_limit * config.warning_threshold:
            last_alert = self.cost_alerts_sent.get(provider)
            if not last_alert or datetime.now() - last_alert > timedelta(hours=1):
                log.warning(f"Cost alert for {provider}: ${usage.daily_cost:.2f} / ${config.daily_cost_limit:.2f}")
                self.cost_alerts_sent[provider] = datetime.now()
    
    async def batch_requests(self, provider: str, requests: List[Dict[str, Any]], 
                           batch_size: int = 10, delay_seconds: float = 1.0,
                           processor_func: Optional[Callable] = None) -> List[Any]:
        """Batch requests to optimize quota usage"""
        if not requests:
            return []
        
        # Check quota before batching
        total_cost = sum(req.get("estimated_cost", 0.0) for req in requests)
        total_tokens = sum(req.get("estimated_tokens", 0) for req in requests)
        
        quota_status = self.check_quota(provider, total_cost, total_tokens)
        if quota_status == QuotaStatus.EXCEEDED:
            raise RuntimeError(f"Quota exceeded for {provider}")
        
        # Create batches
        batches = [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]
        results = []
        
        for i, batch in enumerate(batches):
            try:
                # Process batch
                if processor_func:
                    batch_results = await processor_func(batch)
                else:
                    batch_results = await self._default_batch_processor(provider, batch)
                
                results.extend(batch_results)
                
                # Record usage for batch
                batch_cost = sum(req.get("estimated_cost", 0.0) for req in batch)
                batch_tokens = sum(req.get("estimated_tokens", 0) for req in batch)
                self.record_usage(provider, batch_cost, batch_tokens)
                
                # Rate limiting delay between batches
                if i < len(batches) - 1:  # Don't delay after last batch
                    await asyncio.sleep(delay_seconds)
                    
            except Exception as e:
                log.error(f"Batch processing failed for {provider}: {e}")
                # Record failed usage
                batch_cost = sum(req.get("estimated_cost", 0.0) for req in batch)
                batch_tokens = sum(req.get("estimated_tokens", 0) for req in batch)
                self.record_usage(provider, batch_cost, batch_tokens, success=False)
                raise
        
        return results
    
    async def _default_batch_processor(self, provider: str, batch: List[Dict[str, Any]]) -> List[Any]:
        """Default batch processor (placeholder)"""
        # This would be implemented based on the specific provider
        log.info(f"Processing batch of {len(batch)} requests for {provider}")
        return [{"status": "processed", "request": req} for req in batch]
    
    def set_backoff(self, provider: str, duration_seconds: float):
        """Set backoff period for a provider"""
        self.backoff_timers[provider] = time.time() + duration_seconds
        log.warning(f"Set backoff for {provider}: {duration_seconds}s")
    
    def clear_backoff(self, provider: str):
        """Clear backoff period for a provider"""
        if provider in self.backoff_timers:
            del self.backoff_timers[provider]
            log.info(f"Cleared backoff for {provider}")
    
    def is_in_backoff(self, provider: str) -> bool:
        """Check if provider is in backoff period"""
        if provider not in self.backoff_timers:
            return False
        
        if time.time() >= self.backoff_timers[provider]:
            del self.backoff_timers[provider]
            return False
        
        return True
    
    def get_quota_status(self, provider: str) -> Dict[str, Any]:
        """Get detailed quota status for a provider"""
        if provider not in self.quota_configs:
            return {"status": "unknown", "error": "Provider not configured"}
        
        config = self.quota_configs[provider]
        usage = self.quota_usage.get(provider, QuotaUsage(provider=provider))
        usage.reset_if_needed()
        
        # Calculate percentages
        daily_request_pct = usage.daily_requests / config.daily_request_limit
        daily_cost_pct = usage.daily_cost / config.daily_cost_limit
        daily_token_pct = usage.daily_tokens / config.daily_token_limit
        
        # Determine overall status
        if (daily_request_pct >= 1.0 or daily_cost_pct >= 1.0 or daily_token_pct >= 1.0):
            status = QuotaStatus.EXCEEDED
        elif (daily_request_pct >= config.warning_threshold or 
              daily_cost_pct >= config.warning_threshold or 
              daily_token_pct >= config.warning_threshold):
            status = QuotaStatus.NEAR_LIMIT
        else:
            status = QuotaStatus.AVAILABLE
        
        return {
            "provider": provider,
            "status": status.value,
            "daily_requests": {
                "used": usage.daily_requests,
                "limit": config.daily_request_limit,
                "percentage": daily_request_pct
            },
            "daily_cost": {
                "used": usage.daily_cost,
                "limit": config.daily_cost_limit,
                "percentage": daily_cost_pct
            },
            "daily_tokens": {
                "used": usage.daily_tokens,
                "limit": config.daily_token_limit,
                "percentage": daily_token_pct
            },
            "hourly_requests": {
                "used": usage.hourly_requests,
                "limit": config.hourly_request_limit
            },
            "per_minute_requests": {
                "used": usage.per_minute_requests,
                "limit": config.per_minute_request_limit
            },
            "in_backoff": self.is_in_backoff(provider),
            "backoff_remaining": max(0, self.backoff_timers.get(provider, 0) - time.time())
        }
    
    def get_all_quota_status(self) -> Dict[str, Dict[str, Any]]:
        """Get quota status for all providers"""
        return {provider: self.get_quota_status(provider) 
                for provider in self.quota_configs.keys()}
    
    def _start_background_tasks(self):
        """Start background tasks for quota management"""
        # Only start if event loop is running and not already started
        if self._background_tasks_started:
            return
        
        try:
            loop = asyncio.get_running_loop()
            # Event loop is running, safe to create task
            asyncio.create_task(self._quota_reset_task())
            self._background_tasks_started = True
            log.info("Started quota manager background tasks")
        except RuntimeError:
            # No event loop running yet, will be started later
            log.debug("No event loop available, background tasks will start later")
    
    async def _quota_reset_task(self):
        """Background task to reset quota counters"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                for usage in self.quota_usage.values():
                    usage.reset_if_needed()
            except Exception as e:
                log.error(f"Quota reset task error: {e}")
                await asyncio.sleep(60)
    
    def estimate_request_cost(self, provider: str, estimated_tokens: int) -> float:
        """Estimate cost for a request"""
        # This would be implemented based on provider pricing
        # For now, return a rough estimate
        if provider == "gemini":
            return estimated_tokens * 0.000001
        elif provider == "openai":
            return estimated_tokens * 0.00003
        elif provider == "anthropic":
            return estimated_tokens * 0.00001
        else:
            return estimated_tokens * 0.000001  # Default estimate
    
    def get_cost_summary(self, days: int = 1) -> Dict[str, Any]:
        """Get cost summary across all providers"""
        total_cost = 0.0
        provider_costs = {}
        
        for provider, usage in self.quota_usage.items():
            provider_costs[provider] = usage.daily_cost
            total_cost += usage.daily_cost
        
        return {
            "total_cost": total_cost,
            "provider_costs": provider_costs,
            "period_days": days,
            "daily_average": total_cost / max(days, 1)
        }
    
    def disable_provider(self, provider: str):
        """Disable quota management for a provider"""
        if provider in self.quota_configs:
            self.quota_configs[provider].enabled = False
            log.info(f"Disabled quota management for {provider}")
    
    def enable_provider(self, provider: str):
        """Enable quota management for a provider"""
        if provider in self.quota_configs:
            self.quota_configs[provider].enabled = True
            log.info(f"Enabled quota management for {provider}")

# Global instance
quota_manager = QuotaManager()

# Convenience functions
def check_quota(provider: str, request_cost: float = 0.0, token_count: int = 0) -> QuotaStatus:
    """Check quota status for a provider"""
    return quota_manager.check_quota(provider, request_cost, token_count)

def record_usage(provider: str, request_cost: float = 0.0, token_count: int = 0, success: bool = True):
    """Record usage for a provider"""
    quota_manager.record_usage(provider, request_cost, token_count, success)

def get_quota_status(provider: str) -> Dict[str, Any]:
    """Get quota status for a provider"""
    return quota_manager.get_quota_status(provider)

def set_backoff(provider: str, duration_seconds: float):
    """Set backoff period for a provider"""
    quota_manager.set_backoff(provider, duration_seconds)

def is_in_backoff(provider: str) -> bool:
    """Check if provider is in backoff period"""
    return quota_manager.is_in_backoff(provider)

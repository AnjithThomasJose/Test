"""
Centralized Model Registry for KAFIN Agents

This module provides a comprehensive model management system with:
- Multi-provider support (Gemini, OpenAI, Anthropic, Vertex AI)
- Task-specific model routing
- Automatic fallback mechanisms
- Cost tracking and optimization
- Performance monitoring
"""

import os
import json
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime, timedelta
import logging

log = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Supported model providers"""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VERTEX_AI = "vertex_ai"
    OLLAMA = "ollama"

class TaskType(Enum):
    """Supported task types for model routing"""
    ASSESSMENT_GENERATION = "assessment_generation"
    ASSESSMENT_EVALUATION = "assessment_evaluation"
    RESUME_ANALYSIS = "resume_analysis"
    RESUME_SUMMARIZATION = "resume_summarization"
    CODING = "coding"
    COMPLEX_REASONING = "complex_reasoning"
    TEXT_GENERATION = "text_generation"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    INTERVIEW = "interview"
    REPORT_GENERATION = "report_generation"
    SKILL_ANALYSIS = "skill_analysis"
    COURSE_VALIDATION = "course_validation"
    MARKET_ANALYSIS = "market_analysis"
    SALARY_INSIGHTS = "salary_insights"

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    provider: ModelProvider
    model_name: str
    display_name: str
    max_tokens: int
    temperature: float
    timeout: int
    cost_per_input_token: float
    cost_per_output_token: float
    fallback_models: List[str] = field(default_factory=list)
    supported_tasks: List[TaskType] = field(default_factory=list)
    priority: int = 1  # 1-5, 5 being highest priority
    is_active: bool = True
    rate_limit_rpm: int = 60  # requests per minute
    rate_limit_tpm: int = 100000  # tokens per minute
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class ModelUsage:
    """Track model usage statistics"""
    model_name: str
    requests_count: int = 0
    tokens_used: int = 0
    total_cost: float = 0.0
    avg_latency: float = 0.0
    error_count: int = 0
    last_used: Optional[datetime] = None

class ModelRegistry:
    """Centralized model registry with routing and fallback support"""
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.usage_stats: Dict[str, ModelUsage] = {}
        self.task_routing: Dict[TaskType, List[str]] = {}
        self._initialize_default_models()
        self._load_custom_config()
    
    def _initialize_default_models(self):
        """Initialize default model configurations.

        Same across all environments: gemini-2.5-flash primary and fallback.
        """
        primary_name = "gemini-2.5-flash"
        secondary_name = "gemini-2.5-flash"
        primary_fallbacks: List[str] = []
        secondary_fallbacks: List[str] = []

        default_models = {
            primary_name: ModelConfig(
                provider=ModelProvider.GEMINI,
                model_name=primary_name,
                display_name="Gemini 2.5 Flash Lite" if primary_name.endswith("lite") else "Gemini 2.5 Flash",
                max_tokens=8192,
                temperature=0.1,
                timeout=45 if primary_name.endswith("lite") else 60,  # match per-model tuning
                cost_per_input_token=0.0000005 if primary_name.endswith("lite") else 0.000001,
                cost_per_output_token=0.0000005 if primary_name.endswith("lite") else 0.000001,
                fallback_models=primary_fallbacks,
                supported_tasks=[
                    TaskType.ASSESSMENT_GENERATION,
                    TaskType.ASSESSMENT_EVALUATION,
                    TaskType.RESUME_ANALYSIS,
                    TaskType.RESUME_SUMMARIZATION,
                    TaskType.CODING,
                    TaskType.COMPLEX_REASONING,
                    TaskType.TEXT_GENERATION,
                    TaskType.CLASSIFICATION,
                    TaskType.SUMMARIZATION,
                    TaskType.TRANSLATION,
                    TaskType.INTERVIEW,
                    TaskType.REPORT_GENERATION,
                    TaskType.SKILL_ANALYSIS,
                    TaskType.COURSE_VALIDATION,
                    TaskType.MARKET_ANALYSIS,
                    TaskType.SALARY_INSIGHTS,
                ],
                priority=5,
                rate_limit_rpm=60 if primary_name.endswith("flash") and not primary_name.endswith("lite") else 120,
                rate_limit_tpm=100000 if primary_name.endswith("flash") and not primary_name.endswith("lite") else 200000,
            ),
            secondary_name: ModelConfig(
                provider=ModelProvider.GEMINI,
                model_name=secondary_name,
                display_name="Gemini 2.5 Flash Lite" if secondary_name.endswith("lite") else "Gemini 2.5 Flash",
                max_tokens=8192,
                temperature=0.1,
                timeout=45 if secondary_name.endswith("lite") else 60,
                cost_per_input_token=0.0000005 if secondary_name.endswith("lite") else 0.000001,
                cost_per_output_token=0.0000005 if secondary_name.endswith("lite") else 0.000001,
                fallback_models=secondary_fallbacks,
                supported_tasks=[
                    TaskType.ASSESSMENT_GENERATION,
                    TaskType.ASSESSMENT_EVALUATION,
                    TaskType.RESUME_ANALYSIS,
                    TaskType.RESUME_SUMMARIZATION,
                    TaskType.TEXT_GENERATION,
                    TaskType.CLASSIFICATION,
                    TaskType.SUMMARIZATION,
                    TaskType.INTERVIEW,
                    TaskType.REPORT_GENERATION,
                    TaskType.SKILL_ANALYSIS,
                    TaskType.COURSE_VALIDATION,
                    TaskType.MARKET_ANALYSIS,
                    TaskType.SALARY_INSIGHTS,
                ],
                priority=4,
                rate_limit_rpm=120 if secondary_name.endswith("lite") else 60,
                rate_limit_tpm=200000 if secondary_name.endswith("lite") else 100000,
            ),
        }
        
        for model_name, config in default_models.items():
            self.register_model(model_name, config)
    
    def _load_custom_config(self):
        """Load custom model configurations from environment or config file"""
        custom_config = os.getenv("MODEL_REGISTRY_CONFIG")
        if custom_config:
            try:
                config_data = json.loads(custom_config)
                for model_name, config_dict in config_data.items():
                    self._load_model_from_dict(model_name, config_dict)
            except Exception as e:
                log.warning(f"Failed to load custom model config: {e}")
    
    def register_model(self, model_name: str, config: ModelConfig):
        """Register a new model configuration"""
        self.models[model_name] = config
        self.usage_stats[model_name] = ModelUsage(model_name=model_name)
        
        # Update task routing
        for task in config.supported_tasks:
            if task not in self.task_routing:
                self.task_routing[task] = []
            
            # Insert based on priority (higher priority first)
            inserted = False
            for i, existing_model in enumerate(self.task_routing[task]):
                if self.models[existing_model].priority < config.priority:
                    self.task_routing[task].insert(i, model_name)
                    inserted = True
                    break
            
            if not inserted:
                self.task_routing[task].append(model_name)
        
        log.info(f"Registered model: {model_name} for tasks: {[t.value for t in config.supported_tasks]}")
    
    def get_model_for_task(self, task: Union[TaskType, str], 
                          preferred_model: Optional[str] = None,
                          budget_limit: Optional[float] = None) -> ModelConfig:
        """Get the best model for a specific task with fallback support"""
        if isinstance(task, str):
            try:
                task = TaskType(task)
            except ValueError:
                log.warning(f"Unknown task type: {task}, using default")
                task = TaskType.TEXT_GENERATION
        
        # Check if preferred model is available and suitable
        if preferred_model and preferred_model in self.models:
            model = self.models[preferred_model]
            if task in model.supported_tasks and model.is_active:
                if budget_limit is None or self._estimate_cost(model, task) <= budget_limit:
                    return model
        
        # Find best available model for task
        available_models = self.task_routing.get(task, [])
        for model_name in available_models:
            if model_name in self.models:
                model = self.models[model_name]
                if model.is_active:
                    if budget_limit is None or self._estimate_cost(model, task) <= budget_limit:
                        return model
        
        # Fallback to default model (same across all environments)
        default_name = "gemini-2.5-flash"
        default_model = self.models.get(default_name)
        if default_model:
            log.warning(f"No suitable model found for task {task.value}, using default")
            return default_model
        
        raise RuntimeError("No models available")
    
    def get_fallback_model(self, current_model: str) -> Optional[ModelConfig]:
        """Get fallback model for the current model"""
        if current_model not in self.models:
            return None
        
        current_config = self.models[current_model]
        for fallback_name in current_config.fallback_models:
            if fallback_name in self.models and self.models[fallback_name].is_active:
                return self.models[fallback_name]
        
        return None
    
    def record_usage(self, model_name: str, tokens_used: int, latency_ms: float, 
                    success: bool = True, error_type: Optional[str] = None):
        """Record model usage statistics"""
        if model_name not in self.usage_stats:
            self.usage_stats[model_name] = ModelUsage(model_name=model_name)
        
        usage = self.usage_stats[model_name]
        usage.requests_count += 1
        usage.tokens_used += tokens_used
        usage.last_used = datetime.now()
        
        # Calculate cost
        if model_name in self.models:
            config = self.models[model_name]
            # Estimate input/output token split (rough approximation)
            input_tokens = int(tokens_used * 0.7)
            output_tokens = int(tokens_used * 0.3)
            cost = (input_tokens * config.cost_per_input_token + 
                   output_tokens * config.cost_per_output_token)
            usage.total_cost += cost
        
        # Update average latency
        if usage.avg_latency == 0:
            usage.avg_latency = latency_ms
        else:
            usage.avg_latency = (usage.avg_latency * 0.9) + (latency_ms * 0.1)
        
        if not success:
            usage.error_count += 1
        
        log.debug(f"Recorded usage for {model_name}: {tokens_used} tokens, {latency_ms}ms, cost: ${usage.total_cost:.6f}")
    
    def get_usage_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics for models"""
        if model_name:
            if model_name in self.usage_stats:
                usage = self.usage_stats[model_name]
                return {
                    "model_name": model_name,
                    "requests_count": usage.requests_count,
                    "tokens_used": usage.tokens_used,
                    "total_cost": usage.total_cost,
                    "avg_latency": usage.avg_latency,
                    "error_count": usage.error_count,
                    "error_rate": usage.error_count / max(usage.requests_count, 1),
                    "last_used": usage.last_used.isoformat() if usage.last_used else None
                }
            return {}
        
        # Return stats for all models
        return {name: self.get_usage_stats(name) for name in self.usage_stats.keys()}
    
    def get_cost_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get cost summary for the specified period"""
        total_cost = sum(usage.total_cost for usage in self.usage_stats.values())
        total_tokens = sum(usage.tokens_used for usage in self.usage_stats.values())
        total_requests = sum(usage.requests_count for usage in self.usage_stats.values())
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "avg_cost_per_request": total_cost / max(total_requests, 1),
            "avg_tokens_per_request": total_tokens / max(total_requests, 1),
            "period_days": days
        }
    
    def _estimate_cost(self, model: ModelConfig, task: TaskType) -> float:
        """Estimate cost for a task (rough approximation)"""
        # Task-specific token estimates
        token_estimates = {
            TaskType.ASSESSMENT_GENERATION: 2000,
            TaskType.ASSESSMENT_EVALUATION: 1500,
            TaskType.RESUME_ANALYSIS: 3000,
            TaskType.RESUME_SUMMARIZATION: 1000,
            TaskType.CODING: 2500,
            TaskType.COMPLEX_REASONING: 2000,
            TaskType.TEXT_GENERATION: 1000,
            TaskType.CLASSIFICATION: 500,
            TaskType.SUMMARIZATION: 800,
            TaskType.INTERVIEW: 1200,
            TaskType.REPORT_GENERATION: 1800
        }
        
        estimated_tokens = token_estimates.get(task, 1000)
        input_tokens = int(estimated_tokens * 0.7)
        output_tokens = int(estimated_tokens * 0.3)
        
        return (input_tokens * model.cost_per_input_token + 
               output_tokens * model.cost_per_output_token)
    
    def _load_model_from_dict(self, model_name: str, config_dict: Dict[str, Any]):
        """Load model configuration from dictionary"""
        try:
            config = ModelConfig(
                provider=ModelProvider(config_dict["provider"]),
                model_name=model_name,
                display_name=config_dict.get("display_name", model_name),
                max_tokens=config_dict.get("max_tokens", 4096),
                temperature=config_dict.get("temperature", 0.1),
                timeout=config_dict.get("timeout", 45),
                cost_per_input_token=config_dict.get("cost_per_input_token", 0.000001),
                cost_per_output_token=config_dict.get("cost_per_output_token", 0.000001),
                fallback_models=config_dict.get("fallback_models", []),
                supported_tasks=[TaskType(t) for t in config_dict.get("supported_tasks", [])],
                priority=config_dict.get("priority", 1),
                is_active=config_dict.get("is_active", True),
                rate_limit_rpm=config_dict.get("rate_limit_rpm", 60),
                rate_limit_tpm=config_dict.get("rate_limit_tpm", 100000)
            )
            self.register_model(model_name, config)
        except Exception as e:
            log.error(f"Failed to load model {model_name}: {e}")
    
    def list_models(self, task: Optional[TaskType] = None) -> List[Dict[str, Any]]:
        """List available models, optionally filtered by task"""
        models = []
        for name, config in self.models.items():
            if not config.is_active:
                continue
            
            if task and task not in config.supported_tasks:
                continue
            
            models.append({
                "name": name,
                "display_name": config.display_name,
                "provider": config.provider.value,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "timeout": config.timeout,
                "priority": config.priority,
                "supported_tasks": [t.value for t in config.supported_tasks],
                "fallback_models": config.fallback_models,
                "rate_limit_rpm": config.rate_limit_rpm,
                "rate_limit_tpm": config.rate_limit_tpm
            })
        
        return sorted(models, key=lambda x: x["priority"], reverse=True)
    
    def disable_model(self, model_name: str):
        """Disable a model"""
        if model_name in self.models:
            self.models[model_name].is_active = False
            log.info(f"Disabled model: {model_name}")
    
    def enable_model(self, model_name: str):
        """Enable a model"""
        if model_name in self.models:
            self.models[model_name].is_active = True
            log.info(f"Enabled model: {model_name}")

# Global instance
model_registry = ModelRegistry()

# Convenience functions
def get_model_for_task(task: Union[TaskType, str], preferred_model: Optional[str] = None) -> ModelConfig:
    """Get the best model for a specific task"""
    return model_registry.get_model_for_task(task, preferred_model)

def record_model_usage(model_name: str, tokens_used: int, latency_ms: float, 
                     success: bool = True, error_type: Optional[str] = None):
    """Record model usage statistics"""
    model_registry.record_usage(model_name, tokens_used, latency_ms, success, error_type)

def get_model_usage_stats(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Get model usage statistics"""
    return model_registry.get_usage_stats(model_name)

def get_cost_summary(days: int = 7) -> Dict[str, Any]:
    """Get cost summary"""
    return model_registry.get_cost_summary(days)

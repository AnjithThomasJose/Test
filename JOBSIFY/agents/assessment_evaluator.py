import json
import asyncio
import re
import logging
import html
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import os
from pydantic import BaseModel, Field, field_validator
from models.llm_invoker import invoke_llm, invoke_structured_llm
from langsmith.run_helpers import traceable
from utils.session_manager import session_manager
from core.model_registry import TaskType
from chroma import get_chat_session, update_chat_session
from core.utils import (
    _create_error_response, _calculate_processing_time, run_blocking_io
)
from core.config import get_agent_config
from core.security import validate_tenant_id, redact_pii, filter_injection_attempts, sanitize_text_for_llm, PII_PATTERNS, INJECTION_FILTERS
from core.memory import BaseAgentMemory, get_agent_memory
from core.logging_helpers import AgentLogger, create_log_context, log_agent_completion
from enum import Enum

# Get centralized configuration
config = get_agent_config("assessment_evaluator")

# =============================================================================
# Structured Output Models for LLM Evaluation (Issue 5.1 + 5.3 Validation)
# =============================================================================
# These Pydantic models ensure type-safe, validated responses from LLM calls.
# Using with_structured_output() instead of manual json.loads() provides:
# - Automatic schema validation
# - Provider-native JSON mode (more reliable than regex extraction)
# - Type safety and IDE autocomplete
#
# Issue 5.3: Business-rule validation via field_validators

def _clamp_score(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """Clamp a score to valid range."""
    if isinstance(value, (int, float)):
        return max(min_val, min(max_val, float(value)))
    return 50.0  # Default mid-score

def _sanitize_str(value: str, max_length: int = 500, default: str = "") -> str:
    """Sanitize and truncate a string."""
    if not value or not isinstance(value, str):
        return default
    return value.strip()[:max_length]

def _sanitize_str_list(items: List[str], max_items: int = 10, max_item_len: int = 200) -> List[str]:
    """Sanitize a list of strings."""
    if not items or not isinstance(items, list):
        return []
    return [item.strip()[:max_item_len] for item in items[:max_items] if item and isinstance(item, str) and item.strip()]


class StrengthItem(BaseModel):
    """A strength identified in the assessment"""
    trait: str = Field(description="Name of the trait or competency")
    insight: str = Field(description="Detailed behavioral insight")
    description: str = Field(description="Professional description of the strength")
    
    @field_validator('trait', mode='before')
    @classmethod
    def validate_trait(cls, v):
        return _sanitize_str(v, max_length=100, default="Unknown Trait")
    
    @field_validator('insight', 'description', mode='before')
    @classmethod
    def validate_text(cls, v):
        return _sanitize_str(v, max_length=500, default="")


class DevelopmentAreaItem(BaseModel):
    """An area for development identified in the assessment"""
    trait: str = Field(description="Name of the trait or competency")
    insight: str = Field(description="Detailed behavioral insight explaining the development area")
    description: str = Field(description="Professional description of the growth opportunity")
    
    @field_validator('trait', mode='before')
    @classmethod
    def validate_trait(cls, v):
        return _sanitize_str(v, max_length=100, default="Unknown Area")
    
    @field_validator('insight', 'description', mode='before')
    @classmethod
    def validate_text(cls, v):
        return _sanitize_str(v, max_length=500, default="")


class ConstructiveFeedback(BaseModel):
    """Structured constructive feedback from the evaluation"""
    summary: str = Field(description="Overall summary of the behavioral analysis")
    strengths: List[StrengthItem] = Field(default_factory=list, description="List of identified strengths")
    areas_for_development: List[DevelopmentAreaItem] = Field(default_factory=list, description="List of development areas")
    behavioral_insights: List[str] = Field(default_factory=list, description="Key behavioral insights")
    development_focus: List[str] = Field(default_factory=list, description="Priority development areas")
    
    @field_validator('summary', mode='before')
    @classmethod
    def validate_summary(cls, v):
        return _sanitize_str(v, max_length=1000, default="Assessment completed.")
    
    @field_validator('strengths', mode='before')
    @classmethod
    def limit_strengths(cls, v):
        if isinstance(v, list):
            return v[:10]  # Max 10 strengths
        return []
    
    @field_validator('areas_for_development', mode='before')
    @classmethod
    def limit_development_areas(cls, v):
        if isinstance(v, list):
            return v[:10]  # Max 10 areas
        return []
    
    @field_validator('behavioral_insights', 'development_focus', mode='before')
    @classmethod
    def validate_str_lists(cls, v):
        return _sanitize_str_list(v, max_items=10, max_item_len=300)


class BehavioralEvaluationResult(BaseModel):
    """Complete structured output for behavioral assessment evaluation"""
    test_type: str = Field(description="Type of assessment (e.g., personality, psychometric)")
    constructive_feedback: ConstructiveFeedback = Field(description="Detailed constructive feedback")
    trait_scores: Dict[str, float] = Field(default_factory=dict, description="Scores for each trait (0-100)")
    interpretations: Dict[str, str] = Field(default_factory=dict, description="Interpretation for each trait")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    evaluation_method: str = Field(default="llm_behavioral_analysis", description="Method used for evaluation")
    
    @field_validator('test_type', mode='before')
    @classmethod
    def validate_test_type(cls, v):
        return _sanitize_str(v, max_length=50, default="behavioral_assessment")
    
    @field_validator('trait_scores', mode='before')
    @classmethod
    def validate_trait_scores(cls, v):
        """Clamp all trait scores to 0-100 range."""
        if not isinstance(v, dict):
            return {}
        result = {}
        for key, score in list(v.items())[:20]:  # Max 20 traits
            result[str(key)[:50]] = _clamp_score(score, 0.0, 100.0)
        return result
    
    @field_validator('interpretations', mode='before')
    @classmethod
    def validate_interpretations(cls, v):
        """Sanitize interpretation texts."""
        if not isinstance(v, dict):
            return {}
        result = {}
        for key, text in list(v.items())[:20]:  # Max 20 interpretations
            result[str(key)[:50]] = _sanitize_str(text, max_length=500, default="")
        return result
    
    @field_validator('recommendations', mode='before')
    @classmethod
    def validate_recommendations(cls, v):
        return _sanitize_str_list(v, max_items=10, max_item_len=300)
    
    @field_validator('evaluation_method', mode='before')
    @classmethod
    def validate_method(cls, v):
        return _sanitize_str(v, max_length=50, default="llm_behavioral_analysis")


# Token-Optimized Enhanced Evaluation System 
class TokenOptimizedPrompts:
    """Optimized prompt templates with minimal token usage"""
    
    @staticmethod
    def get_compact_evaluation_prompt(qtype: str, max_score: int) -> str:
        """Ultra-compact evaluation prompt"""
        if qtype in ["short_answer", "long_answer"]:
            return f"""Eval {qtype}: Q={{question}} A={{answer}} Max={max_score}
Rules: Answer must FULLY address ALL parts. No full marks for partial matches. Score based on completeness.
Score 0-{max_score}:"""
        else:
            return f"""Eval {qtype}: Q={{question}} A={{answer}} Max={max_score}
Score:"""
    
    @staticmethod
    def get_enhanced_evaluation_prompt(qtype: str, max_score: int, analysis_summary: str) -> str:
        """Enhanced but token-efficient prompt"""
        return f"""Eval {qtype} (Max:{max_score})
Q: {{question}}
A: {{answer}}
Analysis: {analysis_summary}

CRITICAL SCORING RULES:
- Answer must FULLY address ALL parts of the question to receive full marks
- Do NOT award full marks for partial matches (e.g., only first few words matching)
- Do NOT award points for trivial affirmations (e.g., "correct", "yes", "ok") - these must score 0
- Do NOT award points for answers with < 3 words - these must score 0
- Check completeness: if question asks for examples/explanations/comparisons, ALL must be provided
- Award partial credit only: score based on how much of the question is actually answered
- Incomplete answers (missing examples, explanations, or key points) should receive reduced scores
- Very short answers (≤6 words) should receive maximum 30% of max score even if partially correct
- Score 0-{max_score} based on completeness and accuracy"""
    
    @staticmethod
    def get_batch_evaluation_prompt(questions_data: List[Dict]) -> str:
        """Batch evaluation prompt for multiple questions"""
        prompt = "Batch Eval:\n"
        for i, q in enumerate(questions_data):
            prompt += f"{i+1}. {q['type']} (Max:{q['max_score']}) - {q['question'][:50]}...\n"
            prompt += f"   A: {q['answer'][:100]}...\n"
            prompt += f"   Score: \n"
        return prompt


class ContentOptimizer:
    """Optimizes content for token efficiency while preserving meaning"""
    
    def __init__(self):
        self.max_question_length = 200
        self.max_answer_length = 300
        self.max_analysis_length = 100
    
    def truncate_question(self, question: str) -> str:
        """Truncate question while preserving key information"""
        if len(question) <= self.max_question_length:
            return question
        
        # Try to truncate at sentence boundary
        truncated = self._truncate_at_sentence_boundary(question, self.max_question_length)
        if truncated:
            return truncated
        
        # Fallback to word boundary truncation
        words = question.split()
        truncated_words = words[:self.max_question_length // 6]  # ~6 chars per word
        return " ".join(truncated_words) + "..."
    
    def truncate_answer(self, answer: str, qtype: str) -> str:
        """Truncate answer based on question type"""
        if qtype in ["short_answer"]:
            max_len = 150
        elif qtype in ["long_answer"]:
            max_len = 300
        else:
            max_len = 200
        
        if len(answer) <= max_len:
            return answer
        
        # For short answers, preserve the beginning
        if qtype == "short_answer":
            return answer[:max_len-3] + "..."
        
        # For long answers, try to preserve key parts
        return self._smart_truncate_long_answer(answer, max_len)
    
    def _smart_truncate_long_answer(self, answer: str, max_len: int) -> str:
        """Smart truncation for long answers preserving key concepts"""
        if len(answer) <= max_len:
            return answer
        
        # Split into sentences
        sentences = answer.split('. ')
        truncated = []
        current_len = 0
        
        for sentence in sentences:
            if current_len + len(sentence) + 2 <= max_len - 10:  # Reserve space for "..."
                truncated.append(sentence)
                current_len += len(sentence) + 2
            else:
                break
        
        if truncated:
            return '. '.join(truncated) + "..."
        else:
            return answer[:max_len-3] + "..."
    
    def _truncate_at_sentence_boundary(self, text: str, max_length: int) -> str:
        """Truncate at sentence boundary if possible"""
        if len(text) <= max_length:
            return text
        
        # Find the last complete sentence within the limit
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclamation = truncated.rfind('!')
        
        last_sentence_end = max(last_period, last_question, last_exclamation)
        
        if last_sentence_end > max_length * 0.7:  # Only if we don't lose too much
            return text[:last_sentence_end + 1]
        
        return None


class TokenEfficientConceptAnalyzer:
    """Lightweight concept analysis with minimal token usage"""
    
    def __init__(self):
        # Pre-compiled regex patterns for efficiency
        self.concept_patterns = {
            "technical": re.compile(r'\b(?:algorithm|API|database|framework|system|design|optimization|architecture|performance|security)\b', re.IGNORECASE),
            "analytical": re.compile(r'\b(?:analyze|compare|evaluate|because|therefore|however|consequently|furthermore|moreover)\b', re.IGNORECASE),
            "practical": re.compile(r'\b(?:implement|develop|experience|project|solution|problem|challenge|build|create|execute)\b', re.IGNORECASE),
            "conceptual": re.compile(r'\b(?:concept|principle|theory|fundamental|core|essential|basic|understand|comprehend|grasp)\b', re.IGNORECASE)
        }
    
    def quick_concept_analysis(self, answer: str) -> str:
        """Fast concept analysis returning compact summary"""
        concept_counts = {}
        
        for category, pattern in self.concept_patterns.items():
            matches = len(pattern.findall(answer))
            if matches > 0:
                concept_counts[category] = matches
        
        if not concept_counts:
            return "basic"
        
        # Return most prominent concept category
        dominant = max(concept_counts.items(), key=lambda x: x[1])
        return f"{dominant[0]}({dominant[1]})"
    
    def get_quality_indicators(self, answer: str) -> str:
        """Get compact quality indicators"""
        word_count = len(answer.split())
        
        # Simple quality indicators
        has_structure = any(word in answer.lower() for word in ['first', 'second', 'also', 'however', 'therefore', 'moreover', 'furthermore'])
        has_examples = any(word in answer.lower() for word in ['example', 'such as', 'like', 'including', 'for instance', 'specifically'])
        has_explanation = any(word in answer.lower() for word in ['because', 'since', 'due to', 'explain', 'reason', 'why'])
        
        indicators = []
        if has_structure: indicators.append("struct")
        if has_examples: indicators.append("ex")
        if has_explanation: indicators.append("expl")
        
        return f"wc:{word_count} {'+'.join(indicators) if indicators else 'basic'}"


class TokenBudgetManager:
    """Manages token usage across evaluation tasks"""
    
    def __init__(self, max_tokens_per_evaluation: int = 2000):
        self.max_tokens_per_evaluation = max_tokens_per_evaluation
        self.current_budget = max_tokens_per_evaluation
    
    def estimate_prompt_tokens(self, prompt: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(prompt) // 4
    
    def can_afford_evaluation(self, prompt: str) -> bool:
        """Check if we can afford this evaluation"""
        estimated_tokens = self.estimate_prompt_tokens(prompt)
        return estimated_tokens <= self.current_budget
    
    def optimize_prompt_for_budget(self, prompt: str) -> str:
        """Optimize prompt to fit within token budget"""
        estimated_tokens = self.estimate_prompt_tokens(prompt)
        
        if estimated_tokens <= self.current_budget:
            return prompt
        
        # Calculate reduction factor
        reduction_factor = self.current_budget / estimated_tokens
        
        # Truncate proportionally
        target_length = int(len(prompt) * reduction_factor * 0.9)  # 10% safety margin
        return prompt[:target_length] + "..."
    
    def consume_budget(self, prompt: str):
        """Consume token budget for this evaluation"""
        estimated_tokens = self.estimate_prompt_tokens(prompt)
        self.current_budget -= estimated_tokens
    
    def reset_budget(self):
        """Reset budget for new evaluation session"""
        self.current_budget = self.max_tokens_per_evaluation


class TokenUsageMonitor:
    """Monitor and log token usage for optimization"""
    
    def __init__(self):
        self.total_tokens_used = 0
        self.evaluation_count = 0
        self.avg_tokens_per_evaluation = 0
    
    def log_evaluation_tokens(self, prompt_length: int, response_length: int):
        """Log token usage for an evaluation"""
        estimated_tokens = (prompt_length + response_length) // 4
        self.total_tokens_used += estimated_tokens
        self.evaluation_count += 1
        self.avg_tokens_per_evaluation = self.total_tokens_used / self.evaluation_count
        
        log.debug(f"Token usage: {estimated_tokens} tokens (avg: {self.avg_tokens_per_evaluation:.1f})")
    
    def get_efficiency_report(self) -> Dict[str, Any]:
        """Get token efficiency report"""
        return {
            "total_tokens": self.total_tokens_used,
            "evaluations": self.evaluation_count,
            "avg_tokens_per_evaluation": self.avg_tokens_per_evaluation,
            "efficiency_score": max(0, 100 - (self.avg_tokens_per_evaluation / 20))  # Efficiency score 0-100
        }


# Global instances for token optimization
content_optimizer = ContentOptimizer()
concept_analyzer = TokenEfficientConceptAnalyzer()
token_monitor = TokenUsageMonitor()

# Generic Test Evaluation Framework
class GenericTestType(Enum):
    """Enumeration of supported generic test types"""
    PERSONALITY = "personality"
    PSYCHOMETRIC = "psychometric"
    COMMUNICATION = "communication"
    COGNITIVE = "cognitive"
    APTITUDE = "aptitude"
    BEHAVIORAL = "behavioral"
    LEADERSHIP = "leadership"
    TEAMWORK = "teamwork"
    PROBLEM_SOLVING = "problem_solving"
    CRITICAL_THINKING = "critical_thinking"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    STRESS_MANAGEMENT = "stress_management"
    TIME_MANAGEMENT = "time_management"
    DECISION_MAKING = "decision_making"
    CREATIVITY = "creativity"
    ADAPTABILITY = "adaptability"
    WORK_STYLE = "work_style"
    MOTIVATION = "motivation"
    VALUES = "values"

class GenericTestEvaluator:
    """Evaluates generic tests using behavioral scoring matrices"""
    
    def __init__(self):
        self.scoring_matrices = self._initialize_scoring_matrices()
        self.trait_definitions = self._initialize_trait_definitions()
    
    def _initialize_scoring_matrices(self) -> Dict[GenericTestType, Dict[str, Dict[str, int]]]:
        """Initialize scoring matrices for each generic test type"""
        return {
            GenericTestType.PSYCHOMETRIC: {
                "problem_solving": {
                    "A": 3,  # Systematic approach
                    "B": 2,  # Collaborative approach
                    "C": 1,  # Trial and error
                    "D": 4   # Research-based approach
                },
                "stress_management": {
                    "A": 2,  # Avoidance
                    "B": 4,  # Control-focused
                    "C": 3,  # Social support
                    "D": 1   # Problem-solving
                },
                "decision_making": {
                    "A": 4,  # Information gathering
                    "B": 1,  # Intuitive
                    "C": 3,  # Collaborative
                    "D": 2   # Consequence-focused
                },
                "communication_style": {
                    "A": 3,  # Written
                    "B": 2,  # Verbal
                    "C": 4,  # Visual
                    "D": 1   # Hands-on
                },
                "work_environment": {
                    "A": 1,  # Quiet/private
                    "B": 3,  # Collaborative
                    "C": 2,  # Flexible
                    "D": 4   # Structured
                },
                "feedback_reception": {
                    "A": 1,  # Defensive
                    "B": 4,  # Open to feedback
                    "C": 2,  # Selective
                    "D": 3   # Justifying
                },
                "teamwork_style": {
                    "A": 4,  # Leadership
                    "B": 3,  # Collaborative
                    "C": 2,  # Task-focused
                    "D": 1   # Supportive
                },
                "time_pressure": {
                    "A": 2,  # Extra hours
                    "B": 4,  # Prioritization
                    "C": 3,  # Delegation
                    "D": 1   # Negotiation
                },
                "motivation": {
                    "A": 2,  # Recognition
                    "B": 4,  # Growth
                    "C": 1,  # Financial
                    "D": 3   # Impact
                },
                "learning_style": {
                    "A": 3,  # Structured
                    "B": 1,  # Hands-on
                    "C": 2,  # Theoretical
                    "D": 4   # Collaborative
                },
                "persistence": {
                    "A": 4,  # Keep trying
                    "B": 2,  # Take breaks
                    "C": 3,  # Ask for help
                    "D": 1   # Research solutions
                },
                "communication_preference": {
                    "A": 3,  # Face-to-face
                    "B": 2,  # Written
                    "C": 4,  # Team meetings
                    "D": 1   # Informal
                },
                "organization_style": {
                    "A": 4,  # Detailed lists
                    "B": 2,  # Flexible
                    "C": 3,  # Project-based
                    "D": 1   # Priority-based
                },
                "conflict_resolution": {
                    "A": 1,  # Avoidance
                    "B": 4,  # Direct resolution
                    "C": 3,  # Mediation
                    "D": 2   # Documentation
                },
                "project_preference": {
                    "A": 3,  # Long-term
                    "B": 1,  # Short-term
                    "C": 4,  # Collaborative
                    "D": 2   # Independent
                }
            },
            GenericTestType.PERSONALITY: {
                "social_behavior": {
                    "A": 4,  # Extraverted
                    "B": 3,  # Observant
                    "C": 2,  # Introverted
                    "D": 1   # Anxious
                },
                "work_preference": {
                    "A": 2,  # Independent
                    "B": 3,  # Collaborative
                    "C": 4,  # Leadership
                    "D": 1   # Systematic
                },
                "feedback_handling": {
                    "A": 1,  # Defensive
                    "B": 4,  # Open
                    "C": 2,  # Selective
                    "D": 3   # Justifying
                },
                "motivation": {
                    "A": 2,  # Recognition
                    "B": 4,  # Growth
                    "C": 1,  # Financial
                    "D": 3   # Impact
                },
                "leisure_preference": {
                    "A": 4,  # Social
                    "B": 3,  # Hobbies
                    "C": 1,  # Solitary
                    "D": 2   # Learning
                },
                "decision_style": {
                    "A": 3,  # Logical
                    "B": 1,  # Intuitive
                    "C": 2,  # Collaborative
                    "D": 4   # Analytical
                },
                "communication_style": {
                    "A": 4,  # Direct
                    "B": 3,  # Diplomatic
                    "C": 2,  # Detailed
                    "D": 1   # Concise
                },
                "environment_preference": {
                    "A": 4,  # Dynamic
                    "B": 1,  # Predictable
                    "C": 3,  # Team-oriented
                    "D": 2   # Focused
                },
                "change_adaptability": {
                    "A": 4,  # Embrace change
                    "B": 3,  # Gradual adaptation
                    "C": 1,  # Resist change
                    "D": 2   # Analyze change
                },
                "energy_source": {
                    "A": 4,  # Social interaction
                    "B": 3,  # Problem-solving
                    "C": 1,  # Quiet time
                    "D": 2   # Learning
                },
                "risk_tolerance": {
                    "A": 3,  # Calculated risks
                    "B": 1,  # Risk-averse
                    "C": 4,  # Bold risks
                    "D": 2   # Evaluated risks
                },
                "leadership_style": {
                    "A": 4,  # Directive
                    "B": 3,  # Collaborative
                    "C": 2,  # Supportive
                    "D": 1   # Strategic
                },
                "pressure_response": {
                    "A": 4,  # Thrive under pressure
                    "B": 3,  # Stay calm
                    "C": 1,  # Feel stressed
                    "D": 2   # Work ahead
                },
                "feedback_preference": {
                    "A": 2,  # Immediate
                    "B": 3,  # Detailed
                    "C": 4,  # Positive
                    "D": 1   # Direct
                },
                "learning_preference": {
                    "A": 1,  # Hands-on
                    "B": 2,  # Theoretical
                    "C": 3,  # Collaborative
                    "D": 4   # Observational
                },
                "career_motivation": {
                    "A": 3,  # Advancement
                    "B": 2,  # Work-life balance
                    "C": 4,  # Impact
                    "D": 1   # Financial
                }
            },
            GenericTestType.COMMUNICATION: {
                "conflict_resolution": {
                    "A": 1,  # Avoidance
                    "B": 4,  # Direct resolution
                    "C": 3,  # Mediation
                    "D": 2   # Documentation
                },
                "presentation_style": {
                    "A": 2,  # Detailed slides
                    "B": 4,  # Key points
                    "C": 3,  # Interactive
                    "D": 1   # Handouts
                },
                "listening_skills": {
                    "A": 4,  # Paraphrasing
                    "B": 3,  # Questioning
                    "C": 2,  # Note-taking
                    "D": 1   # Observation
                },
                "communication_preference": {
                    "A": 4,  # Face-to-face
                    "B": 2,  # Written
                    "C": 1,  # Phone
                    "D": 3   # Team meetings
                },
                "active_listening": {
                    "A": 4,  # Attentive
                    "B": 2,  # Note-taking
                    "C": 3,  # Eye contact
                    "D": 1   # Wait to respond
                },
                "difficult_conversations": {
                    "A": 3,  # Preparation
                    "B": 4,  # Direct honesty
                    "C": 2,  # Compromise
                    "D": 1   # Support seeking
                },
                "feedback_delivery": {
                    "A": 4,  # Private
                    "B": 2,  # Written
                    "C": 1,  # Public
                    "D": 3   # Informal
                },
                "conversation_management": {
                    "A": 3,  # Polite wait
                    "B": 1,  # Immediate stop
                    "C": 2,  # Acknowledge and return
                    "D": 4   # Incorporate input
                },
                "cross_cultural_communication": {
                    "A": 4,  # Adapt style
                    "B": 3,  # Simple language
                    "C": 2,  # Ask questions
                    "D": 1   # Patience
                },
                "instruction_reception": {
                    "A": 3,  # Written
                    "B": 2,  # Verbal
                    "C": 4,  # Visual
                    "D": 1   # Hands-on
                },
                "email_communication": {
                    "A": 4,  # Brief
                    "B": 2,  # Detailed
                    "C": 3,  # Formatted
                    "D": 1   # Comprehensive
                },
                "miscommunication_resolution": {
                    "A": 4,  # Immediate clarification
                    "B": 3,  # Take responsibility
                    "C": 2,  # Ask questions
                    "D": 1   # Provide context
                },
                "team_communication": {
                    "A": 3,  # Meetings
                    "B": 2,  # Documents
                    "C": 1,  # Informal
                    "D": 4   # Platforms
                },
                "virtual_communication": {
                    "A": 3,  # Equal participation
                    "B": 2,  # Visual aids
                    "C": 4,  # Focused meetings
                    "D": 1   # Written summaries
                },
                "disagreement_expression": {
                    "A": 3,  # Alternative solutions
                    "B": 2,  # Understanding questions
                    "C": 4,  # Evidence-based
                    "D": 1   # Compromise
                },
                "message_clarity": {
                    "A": 4,  # Confirmation
                    "B": 3,  # Simple language
                    "C": 2,  # Examples
                    "D": 1   # Repetition
                },
                "upward_communication": {
                    "A": 4,  # Concise summaries
                    "B": 2,  # Detailed reports
                    "C": 3,  # Business impact
                    "D": 1   # Regular updates
                },
                "crisis_communication": {
                    "A": 3,  # Frequent updates
                    "B": 4,  # Fact-focused
                    "C": 2,  # Stakeholder coordination
                    "D": 1   # Transparency
                }
            }
        }
    
    def _initialize_trait_definitions(self) -> Dict[GenericTestType, Dict[str, Dict[str, str]]]:
        """Initialize trait definitions and interpretations for each test type"""
        return {
            GenericTestType.PSYCHOMETRIC: {
                "problem_solving": {
                    "high": "Exceptional systematic problem-solver who approaches challenges with methodical analysis, breaking down complex issues into manageable components and developing innovative solutions through thorough investigation and creative thinking.",
                    "medium": "Collaborative problem-solver who seeks input from others and combines analytical thinking with team-based approaches to find effective solutions.",
                    "low": "Developing problem-solving skills through structured approaches and systematic analysis techniques to improve efficiency and solution quality."
                },
                "stress_management": {
                    "high": "Outstanding stress management capabilities with remarkable composure under pressure, utilizing effective coping strategies and maintaining peak performance in demanding situations.",
                    "medium": "Good stress management skills with room for growth in high-pressure scenarios and extreme stress situations.",
                    "low": "Developing stress management techniques and coping strategies to improve performance consistency and well-being."
                },
                "decision_making": {
                    "high": "Excellent decision-maker with strong analytical thinking, comprehensive consideration of multiple factors, and ability to make well-informed strategic choices.",
                    "medium": "Balanced decision-making approach considering both data and intuition with room for improvement in complex scenarios.",
                    "low": "Developing decision-making frameworks and analytical skills to improve confidence and effectiveness in making important choices."
                },
                "communication_style": {
                    "high": "Prefers visual and hands-on communication methods",
                    "medium": "Balanced between written and verbal communication",
                    "low": "Prefers written instructions and documentation"
                },
                "work_environment": {
                    "high": "Thrives in structured, predictable environments",
                    "medium": "Adapts well to different work environments",
                    "low": "Prefers quiet, private workspaces"
                }
            },
            GenericTestType.PERSONALITY: {
                "social_behavior": {
                    "high": "Highly extraverted, enjoys social interaction and meeting new people",
                    "medium": "Moderately social, comfortable in familiar social settings",
                    "low": "Introverted, prefers smaller groups and quiet environments"
                },
                "work_preference": {
                    "high": "Natural leader who enjoys coordinating teams and delegating tasks",
                    "medium": "Collaborative worker who enjoys team projects",
                    "low": "Independent worker who prefers systematic approaches"
                },
                "decision_style": {
                    "high": "Highly analytical, considers all possible outcomes before deciding",
                    "medium": "Balanced approach using both logic and intuition",
                    "low": "Intuitive decision-maker who trusts gut feelings"
                },
                "change_adaptability": {
                    "high": "Embraces change and sees it as an opportunity for growth",
                    "medium": "Adapts gradually to change with time to adjust",
                    "low": "Prefers stability and may resist unnecessary changes"
                },
                "risk_tolerance": {
                    "high": "Comfortable taking calculated risks for potential rewards",
                    "medium": "Evaluates risks carefully before proceeding",
                    "low": "Risk-averse, prefers to avoid risks when possible"
                }
            },
            GenericTestType.COMMUNICATION: {
                "conflict_resolution": {
                    "high": "Excellent conflict resolution skills, addresses issues directly",
                    "medium": "Uses mediation and documentation for conflict resolution",
                    "low": "May avoid confrontation and hope issues resolve themselves"
                },
                "listening_skills": {
                    "high": "Exceptional active listener who paraphrases and asks clarifying questions",
                    "medium": "Good listener who takes notes and observes non-verbal cues",
                    "low": "Basic listening skills, may need improvement in active listening"
                },
                "presentation_style": {
                    "high": "Engaging presenter who focuses on key points and encourages interaction",
                    "medium": "Balanced presenter using visual aids and handouts",
                    "low": "Detail-focused presenter who relies heavily on slides"
                },
                "feedback_delivery": {
                    "high": "Skilled at delivering feedback privately and constructively",
                    "medium": "Uses both written and informal methods for feedback",
                    "low": "May deliver feedback inappropriately in public settings"
                },
                "cross_cultural_communication": {
                    "high": "Excellent cross-cultural communication, adapts style to audience",
                    "medium": "Good cross-cultural skills, uses simple language and patience",
                    "low": "Basic cross-cultural communication skills"
                }
            }
        }
    
    def identify_generic_test_type(self, topic: Optional[str]) -> Optional[GenericTestType]:
        """Identify if a topic is a generic test type"""
        if not topic or not isinstance(topic, str):
            return None
        topic_lower = topic.lower().strip()
        if not topic_lower:
            return None
        
        # Priority order for checking (more specific first)
        test_type_priority = [
            GenericTestType.PSYCHOMETRIC,
            GenericTestType.PERSONALITY,
            GenericTestType.COMMUNICATION,
            GenericTestType.COGNITIVE,
            GenericTestType.APTITUDE,
            GenericTestType.BEHAVIORAL,
            GenericTestType.LEADERSHIP,
            GenericTestType.TEAMWORK,
            GenericTestType.PROBLEM_SOLVING,
            GenericTestType.CRITICAL_THINKING,
            GenericTestType.EMOTIONAL_INTELLIGENCE,
            GenericTestType.STRESS_MANAGEMENT,
            GenericTestType.TIME_MANAGEMENT,
            GenericTestType.DECISION_MAKING,
            GenericTestType.CREATIVITY,
            GenericTestType.ADAPTABILITY,
            GenericTestType.WORK_STYLE,
            GenericTestType.MOTIVATION,
            GenericTestType.VALUES
        ]
        
        # Keywords for each test type
        keywords = {
            GenericTestType.PSYCHOMETRIC: ["psychometric", "psychometric test", "psychometric assessment", "psychological test", "psychological assessment", "psych test", "psych assessment"],
            GenericTestType.PERSONALITY: ["personality", "personality test", "personality assessment", "personality questionnaire", "big five", "myers-briggs", "mbti", "disc", "enneagram", "personality traits"],
            GenericTestType.COMMUNICATION: ["communication", "communication test", "communication assessment", "communication skills", "communication questionnaire"]
        }
        
        for test_type in test_type_priority:
            if test_type in keywords:
                for keyword in keywords[test_type]:
                    keyword_lower = keyword.lower()
                    if (keyword_lower in topic_lower and 
                        (keyword_lower == topic_lower or 
                         topic_lower.startswith(keyword_lower + " ") or 
                         topic_lower.endswith(" " + keyword_lower) or 
                         " " + keyword_lower + " " in topic_lower)):
                        return test_type
        
        return None
    
    @traceable(
        name="generic_test_llm_evaluation",
        tags=["generic_test", "behavioral_analysis", "llm_evaluation"],
        metadata={
            "evaluation_method": "llm_behavioral_analysis"
        }
    )
    async def evaluate_generic_test(self, test_type: GenericTestType, questions: List[Dict], answers: List[Dict]) -> Dict[str, Any]:
        """Evaluate a generic test using LLM-based behavioral analysis"""
        if not questions or not answers:
            return {"error": "Missing questions or answers"}
        
        # Prepare data for LLM evaluation
        evaluation_data = {
            "test_type": test_type.value,
            "questions": questions,
            "answers": answers
        }
        
        # Update traceable metadata with dynamic information
        from langsmith import get_current_run_tree
        try:
            current_run = get_current_run_tree()
            if current_run:
                current_run.metadata.update({
                    "test_type": test_type.value,
                    "question_count": len(questions),
                    "answer_count": len(answers),
                    "evaluation_method": "llm_behavioral_analysis"
                })
        except Exception:
            pass  # Continue if metadata update fails
        
        # Generate LLM prompt for behavioral analysis
        prompt = self._create_llm_evaluation_prompt(test_type, questions, answers)
        
        try:
            structured_result: BehavioralEvaluationResult = await invoke_structured_llm(
                prompt,
                BehavioralEvaluationResult,
                task_type=TaskType.ASSESSMENT_EVALUATION,
                agent_name="assessment_evaluator_behavioral",
                temperature=0.2,
                max_output_tokens=3000,
                timeout=45.0,
                raise_on_fallback=False,
            )
            
            # Convert Pydantic model to dict for compatibility with existing code
            evaluation_result = structured_result.model_dump()
            
            # Ensure test_type matches (override if LLM returned different value)
            evaluation_result["test_type"] = test_type.value
            
            log.info(f"✅ Structured evaluation completed for {test_type.value}")
            return evaluation_result
            
        except asyncio.TimeoutError:
            log.error(f"LLM evaluation timed out for {test_type.value}")
            return self._fallback_deterministic_evaluation(test_type, questions, answers)
        except Exception as e:
            log.error(f"LLM evaluation failed: {e}")
            # Fallback to deterministic scoring
            return self._fallback_deterministic_evaluation(test_type, questions, answers)
    
    def _create_llm_evaluation_prompt(self, test_type: GenericTestType, questions: List[Dict], answers: List[Dict]) -> str:
        """Create LLM prompt for behavioral analysis"""
        test_type_display = test_type.value.replace('_', ' ').title()
        
        # Prepare questions and answers for analysis
        qa_pairs = []
        for i, question in enumerate(questions):
            if i < len(answers):
                answer = answers[i]
                qa_pairs.append({
                    "question": question.get("question_text", ""),
                    "answer": answer.get("selected_option", ""),
                    "category": question.get("category", "general")
                })
        
        prompt = f"""
You are an expert behavioral psychologist conducting a {test_type_display} assessment. Analyze the following responses and provide comprehensive behavioral insights.

ASSESSMENT TYPE: {test_type_display}

QUESTIONS AND RESPONSES:
"""
        
        for i, qa in enumerate(qa_pairs, 1):
            prompt += f"""
Question {i}: {qa['question']}
Response: {qa['answer']}
Category: {qa['category']}
"""
        
        prompt += f"""

ANALYSIS REQUIREMENTS:
1. Analyze behavioral patterns from the responses
2. Identify key strengths (at least 2-3) and areas for development (at least 2-3)
3. Provide detailed behavioral insights (at least 3)
4. Generate specific, actionable recommendations (at least 3)
5. Focus on constructive, growth-oriented feedback
6. Assign trait_scores as numbers between 0-100 for relevant traits
7. Provide interpretations for each scored trait

GUIDELINES:
- Provide detailed, professional behavioral analysis
- Focus on constructive feedback and growth opportunities
- Use specific examples from the responses where possible
- Ensure recommendations are actionable and practical
- Maintain a positive, development-focused tone
- Set test_type to "{test_type.value}"
"""
        
        return prompt
    
    def _parse_llm_evaluation_response(self, llm_response: str, test_type: GenericTestType) -> Dict[str, Any]:
        """Parse LLM response and extract evaluation results"""
        try:
            # Extract JSON from LLM response
            import json
            import re
            
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validate and enhance the result
                if "constructive_feedback" not in result:
                    result["constructive_feedback"] = {
                        "summary": f"LLM-based behavioral analysis for {test_type.value.replace('_', ' ').title()} assessment",
                        "strengths": [],
                        "areas_for_development": [],
                        "behavioral_insights": [],
                        "development_focus": []
                    }
                
                # Ensure all required fields exist
                result.setdefault("test_type", test_type.value)
                result.setdefault("trait_scores", {})
                result.setdefault("interpretations", {})
                result.setdefault("recommendations", [])
                result.setdefault("evaluation_method", "llm_behavioral_analysis")
                
                return result
            else:
                raise ValueError("No JSON found in LLM response")
                
        except Exception as e:
            log.error(f"Failed to parse LLM response: {e}")
            # Return a basic structure if parsing fails
            return {
                "test_type": test_type.value,
                "constructive_feedback": {
                    "summary": f"Behavioral analysis completed for {test_type.value.replace('_', ' ').title()} assessment",
                    "strengths": [],
                    "areas_for_development": [],
                    "behavioral_insights": ["LLM analysis completed successfully"],
                    "development_focus": ["Continue personal and professional development"]
                },
                "trait_scores": {},
                "interpretations": {},
                "recommendations": ["Focus on continued behavioral development"],
                "evaluation_method": "llm_behavioral_analysis"
            }
    
    def _fallback_deterministic_evaluation(self, test_type: GenericTestType, questions: List[Dict], answers: List[Dict]) -> Dict[str, Any]:
        """Fallback to deterministic scoring if LLM fails"""
        # Calculate trait scores using the old deterministic method
        trait_scores = {}
        category_scores = {}
        
        for i, question in enumerate(questions):
            if i >= len(answers):
                continue
                
            answer = answers[i]
            category = question.get("category", "general")
            question_text = question.get("question_text", "")
            
            # Get the selected option
            selected_option = answer.get("selected_option", "")
            if not selected_option:
                continue
            
            # If category is "general" or missing, try to infer from question text
            if category == "general" or not category:
                category = self._infer_category_from_question(question_text, test_type)
            
            # Get scoring matrix for this category
            if test_type in self.scoring_matrices and category in self.scoring_matrices[test_type]:
                score = self.scoring_matrices[test_type][category].get(selected_option, 0)
                
                # Accumulate scores by category
                if category not in category_scores:
                    category_scores[category] = {"total": 0, "count": 0}
                category_scores[category]["total"] += score
                category_scores[category]["count"] += 1
        
        # Calculate average scores for each category
        trait_scores = {}
        for category, data in category_scores.items():
            if data["count"] > 0:
                trait_scores[category] = data["total"] / data["count"]
        
        # Generate interpretations
        interpretations = self._generate_interpretations(test_type, trait_scores)
        
        # Generate constructive criticism instead of numerical scores
        constructive_feedback = self._generate_constructive_feedback(test_type, trait_scores, interpretations)
        
        return {
            "test_type": test_type.value,
            "constructive_feedback": constructive_feedback,
            "trait_scores": trait_scores,
            "interpretations": interpretations,
            "recommendations": self._generate_recommendations(test_type, trait_scores),
            "evaluation_method": "deterministic_fallback"
        }
    
    def _generate_interpretations(self, test_type: GenericTestType, trait_scores: Dict[str, float]) -> Dict[str, str]:
        """Generate behavioral interpretations based on trait scores"""
        interpretations = {}
        
        if test_type not in self.trait_definitions:
            return interpretations
        
        trait_defs = self.trait_definitions[test_type]
        
        for trait, score in trait_scores.items():
            if trait in trait_defs:
                if score >= 3.5:
                    level = "high"
                elif score >= 2.5:
                    level = "medium"
                else:
                    level = "low"
                
                interpretations[trait] = trait_defs[trait].get(level, "No interpretation available")
        
        return interpretations
    
    def _calculate_overall_score(self, trait_scores: Dict[str, float]) -> float:
        """Calculate overall score normalized to 100"""
        if not trait_scores:
            return 0.0
        
        # Average all trait scores and normalize to 100
        average_score = sum(trait_scores.values()) / len(trait_scores)
        # Convert from 1-4 scale to 0-100 scale
        normalized_score = ((average_score - 1) / 3) * 100
        return round(normalized_score, 2)
    
    def _generate_recommendations(self, test_type: GenericTestType, trait_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on trait scores"""
        recommendations = []
        
        if test_type == GenericTestType.PSYCHOMETRIC:
            # Check all psychometric traits and generate recommendations
            if trait_scores.get("stress_management", 0) < 2.5:
                recommendations.append("Consider stress management training to improve coping strategies")
            if trait_scores.get("decision_making", 0) < 2.5:
                recommendations.append("Develop decision-making frameworks to improve analytical thinking")
            if trait_scores.get("teamwork_style", 0) < 2.5:
                recommendations.append("Enhance collaboration skills through team-based projects")
            if trait_scores.get("problem_solving", 0) < 2.5:
                recommendations.append("Practice systematic problem-solving approaches and analytical thinking")
            if trait_scores.get("communication_style", 0) < 2.5:
                recommendations.append("Improve communication skills through practice and feedback")
            if trait_scores.get("work_environment", 0) < 2.5:
                recommendations.append("Explore different work environments to find your optimal setting")
            if trait_scores.get("feedback_reception", 0) < 2.5:
                recommendations.append("Work on receiving feedback constructively and using it for growth")
            if trait_scores.get("time_pressure", 0) < 2.5:
                recommendations.append("Develop time management and prioritization skills")
            if trait_scores.get("motivation", 0) < 2.5:
                recommendations.append("Identify what motivates you and align your work accordingly")
            if trait_scores.get("learning_style", 0) < 2.5:
                recommendations.append("Explore different learning methods to find what works best for you")
            if trait_scores.get("persistence", 0) < 2.5:
                recommendations.append("Develop persistence and resilience through challenging projects")
            if trait_scores.get("organization_style", 0) < 2.5:
                recommendations.append("Improve organizational skills and develop effective systems")
            if trait_scores.get("conflict_resolution", 0) < 2.5:
                recommendations.append("Learn conflict resolution techniques and practice in safe environments")
            if trait_scores.get("project_preference", 0) < 2.5:
                recommendations.append("Explore different types of projects to understand your preferences")
        
        elif test_type == GenericTestType.PERSONALITY:
            # Check all personality traits and generate recommendations
            if trait_scores.get("social_behavior", 0) < 2.5:
                recommendations.append("Consider opportunities to develop social confidence and networking skills")
            if trait_scores.get("change_adaptability", 0) < 2.5:
                recommendations.append("Work on flexibility and adaptability to better handle change")
            if trait_scores.get("risk_tolerance", 0) < 2.5:
                recommendations.append("Practice taking calculated risks to build confidence")
            if trait_scores.get("work_preference", 0) < 2.5:
                recommendations.append("Explore different work styles to find what suits you best")
            if trait_scores.get("feedback_handling", 0) < 2.5:
                recommendations.append("Develop skills for receiving and processing feedback constructively")
            if trait_scores.get("motivation", 0) < 2.5:
                recommendations.append("Identify your core values and what truly motivates you")
            if trait_scores.get("leisure_preference", 0) < 2.5:
                recommendations.append("Explore different leisure activities to find what energizes you")
            if trait_scores.get("decision_style", 0) < 2.5:
                recommendations.append("Practice different decision-making approaches to find your style")
            if trait_scores.get("communication_style", 0) < 2.5:
                recommendations.append("Develop your unique communication style and practice expressing yourself")
            if trait_scores.get("environment_preference", 0) < 2.5:
                recommendations.append("Experiment with different environments to find your ideal setting")
            if trait_scores.get("energy_source", 0) < 2.5:
                recommendations.append("Identify what energizes you and incorporate more of it into your life")
            if trait_scores.get("leadership_style", 0) < 2.5:
                recommendations.append("Explore different leadership approaches and find your authentic style")
            if trait_scores.get("pressure_response", 0) < 2.5:
                recommendations.append("Develop stress management techniques and pressure-handling strategies")
            if trait_scores.get("feedback_preference", 0) < 2.5:
                recommendations.append("Learn to ask for and receive feedback in ways that work for you")
            if trait_scores.get("learning_preference", 0) < 2.5:
                recommendations.append("Experiment with different learning methods to find your optimal approach")
            if trait_scores.get("career_motivation", 0) < 2.5:
                recommendations.append("Reflect on your career goals and what success means to you")
        
        elif test_type == GenericTestType.COMMUNICATION:
            # Check all communication traits and generate recommendations
            if trait_scores.get("listening_skills", 0) < 2.5:
                recommendations.append("Focus on improving active listening skills and empathy")
            if trait_scores.get("conflict_resolution", 0) < 2.5:
                recommendations.append("Develop conflict resolution skills through training and practice")
            if trait_scores.get("presentation_style", 0) < 2.5:
                recommendations.append("Enhance presentation skills and public speaking confidence")
            if trait_scores.get("communication_preference", 0) < 2.5:
                recommendations.append("Practice different communication methods to find your strengths")
            if trait_scores.get("active_listening", 0) < 2.5:
                recommendations.append("Develop active listening techniques and practice in conversations")
            if trait_scores.get("difficult_conversations", 0) < 2.5:
                recommendations.append("Learn techniques for having difficult but necessary conversations")
            if trait_scores.get("feedback_delivery", 0) < 2.5:
                recommendations.append("Practice delivering constructive feedback in a helpful manner")
            if trait_scores.get("conversation_management", 0) < 2.5:
                recommendations.append("Develop skills for managing conversations and group dynamics")
            if trait_scores.get("cross_cultural_communication", 0) < 2.5:
                recommendations.append("Learn about different cultural communication styles and adapt accordingly")
            if trait_scores.get("instruction_reception", 0) < 2.5:
                recommendations.append("Practice receiving and clarifying instructions effectively")
            if trait_scores.get("email_communication", 0) < 2.5:
                recommendations.append("Improve written communication skills and email etiquette")
            if trait_scores.get("miscommunication_resolution", 0) < 2.5:
                recommendations.append("Learn techniques for resolving misunderstandings quickly and effectively")
            if trait_scores.get("team_communication", 0) < 2.5:
                recommendations.append("Develop skills for effective team communication and collaboration")
            if trait_scores.get("virtual_communication", 0) < 2.5:
                recommendations.append("Improve virtual communication skills and online presence")
            if trait_scores.get("disagreement_expression", 0) < 2.5:
                recommendations.append("Learn to express disagreements constructively and professionally")
            if trait_scores.get("message_clarity", 0) < 2.5:
                recommendations.append("Practice making your messages clear, concise, and actionable")
            if trait_scores.get("upward_communication", 0) < 2.5:
                recommendations.append("Develop skills for communicating effectively with management")
            if trait_scores.get("crisis_communication", 0) < 2.5:
                recommendations.append("Learn crisis communication techniques and emergency protocols")
        
        # If no specific recommendations were generated, provide general ones
        if not recommendations:
            if test_type == GenericTestType.PSYCHOMETRIC:
                recommendations.extend([
                    "Focus on developing your problem-solving and decision-making skills",
                    "Work on stress management and time management techniques",
                    "Improve your communication and teamwork abilities"
                ])
            elif test_type == GenericTestType.PERSONALITY:
                recommendations.extend([
                    "Explore different social and work environments to understand your preferences",
                    "Work on adaptability and flexibility in various situations",
                    "Develop your unique strengths and work on areas for improvement"
                ])
            elif test_type == GenericTestType.COMMUNICATION:
                recommendations.extend([
                    "Practice active listening and clear communication",
                    "Work on conflict resolution and difficult conversation skills",
                    "Improve your presentation and public speaking abilities"
                ])
        
        return recommendations
    
    def _generate_constructive_feedback(self, test_type: GenericTestType, trait_scores: Dict[str, float], interpretations: Dict[str, str]) -> Dict[str, Any]:
        """Generate constructive criticism and behavioral insights"""
        feedback = {
            "summary": "",
            "strengths": [],
            "areas_for_development": [],
            "behavioral_insights": [],
            "development_focus": []
        }
        
        if not trait_scores:
            feedback["summary"] = f"Based on your {test_type.value.replace('_', ' ')} assessment, we've identified several areas for personal and professional growth."
            feedback["areas_for_development"] = [
                "Focus on developing core behavioral competencies",
                "Work on building self-awareness and emotional intelligence",
                "Practice skills in real-world situations"
            ]
            return feedback
        
        # Analyze trait scores to identify patterns
        high_scores = {trait: score for trait, score in trait_scores.items() if score >= 3.5}
        medium_scores = {trait: score for trait, score in trait_scores.items() if 2.5 <= score < 3.5}
        low_scores = {trait: score for trait, score in trait_scores.items() if score < 2.5}
        
        # Generate strengths
        for trait, score in high_scores.items():
            trait_display = trait.replace('_', ' ').title()
            interpretation = interpretations.get(trait, "")
            feedback["strengths"].append({
                "trait": trait_display,
                "insight": interpretation,
                "description": f"You demonstrate strong {trait_display.lower()} capabilities"
            })
        
        # Generate areas for development
        for trait, score in low_scores.items():
            trait_display = trait.replace('_', ' ').title()
            interpretation = interpretations.get(trait, "")
            feedback["areas_for_development"].append({
                "trait": trait_display,
                "insight": interpretation,
                "description": f"Focus on developing {trait_display.lower()} skills"
            })
        
        # Generate behavioral insights
        if test_type == GenericTestType.PSYCHOMETRIC:
            feedback["behavioral_insights"] = self._generate_psychometric_insights(trait_scores, interpretations)
        elif test_type == GenericTestType.PERSONALITY:
            feedback["behavioral_insights"] = self._generate_personality_insights(trait_scores, interpretations)
        elif test_type == GenericTestType.COMMUNICATION:
            feedback["behavioral_insights"] = self._generate_communication_insights(trait_scores, interpretations)
        
        # Generate development focus
        feedback["development_focus"] = self._generate_development_focus(test_type, trait_scores)
        
        # Generate summary
        feedback["summary"] = self._generate_feedback_summary(test_type, len(high_scores), len(medium_scores), len(low_scores))
        
        return feedback
    
    def _generate_psychometric_insights(self, trait_scores: Dict[str, float], interpretations: Dict[str, str]) -> List[str]:
        """Generate psychometric-specific behavioral insights"""
        insights = []
        
        # Problem-solving patterns
        if trait_scores.get("problem_solving", 0) >= 3.5:
            insights.append("You demonstrate exceptional problem-solving capabilities with a systematic, methodical approach. You break down complex challenges into manageable components and analyze each aspect thoroughly before developing solutions. This analytical mindset makes you highly effective at tackling intricate problems and finding innovative solutions.")
        elif trait_scores.get("problem_solving", 0) < 2.5:
            insights.append("Your problem-solving approach could benefit from more structured methodologies. Consider developing systematic frameworks for analyzing issues, such as root cause analysis or the 5-Why technique. This will help you approach challenges more methodically and improve your efficiency in finding solutions.")
        
        # Stress management patterns
        if trait_scores.get("stress_management", 0) >= 3.5:
            insights.append("You exhibit outstanding stress management skills and maintain remarkable composure under pressure. You have developed effective coping strategies that allow you to stay focused and productive even in high-pressure situations. This resilience is a significant asset in demanding professional environments.")
        elif trait_scores.get("stress_management", 0) < 2.5:
            insights.append("Developing stronger stress management techniques will significantly enhance your performance consistency. Consider exploring mindfulness practices, time management strategies, or physical exercise routines. Learning to recognize early stress signals and implementing proactive coping mechanisms will help you maintain peak performance.")
        
        # Decision-making patterns
        if trait_scores.get("decision_making", 0) >= 3.5:
            insights.append("Your decision-making process reflects strong analytical thinking and comprehensive consideration of multiple factors. You weigh pros and cons systematically, consider both short-term and long-term implications, and make well-informed choices. This balanced approach leads to sound, strategic decisions.")
        elif trait_scores.get("decision_making", 0) < 2.5:
            insights.append("Building robust decision-making frameworks will help you make more confident and effective choices. Consider developing structured approaches like decision trees, SWOT analysis, or cost-benefit analysis. Practice making decisions within time constraints to build confidence and improve your decision-making speed.")
        
        # Communication patterns
        if trait_scores.get("communication_style", 0) >= 3.5:
            insights.append("You communicate with exceptional clarity and effectiveness, adapting your style to different audiences and contexts. You excel at conveying complex information in accessible ways and actively listen to others' perspectives. This strong communication ability enhances collaboration and builds strong professional relationships.")
        elif trait_scores.get("communication_style", 0) < 2.5:
            insights.append("Enhancing your communication skills will significantly improve your ability to convey ideas clearly and build stronger professional relationships. Focus on active listening, clear articulation of thoughts, and adapting your communication style to different audiences. Practice presenting ideas concisely and asking clarifying questions.")
        
        # Work environment patterns
        if trait_scores.get("work_environment", 0) >= 3.5:
            insights.append("You have a clear understanding of your optimal work environment and thrive in settings that align with your preferences. You're able to create productive workspaces and adapt to different environmental conditions effectively. This self-awareness helps you maximize your performance potential.")
        elif trait_scores.get("work_environment", 0) < 2.5:
            insights.append("Exploring different work environments will help you identify settings that maximize your productivity and well-being. Consider experimenting with various workspace configurations, noise levels, and collaboration styles. Understanding your environmental preferences will help you create optimal conditions for peak performance.")
        
        # Feedback reception patterns
        if trait_scores.get("feedback_reception", 0) >= 3.5:
            insights.append("You receive feedback with remarkable openness and use it constructively for personal and professional growth. You view feedback as valuable information for improvement rather than criticism, which accelerates your development and strengthens relationships with colleagues and supervisors.")
        elif trait_scores.get("feedback_reception", 0) < 2.5:
            insights.append("Developing skills for receiving feedback constructively will accelerate your professional growth and strengthen workplace relationships. Practice separating feedback from personal criticism, ask clarifying questions, and focus on actionable improvements. Remember that feedback is valuable information for development.")
        
        return insights
    
    def _generate_personality_insights(self, trait_scores: Dict[str, float], interpretations: Dict[str, str]) -> List[str]:
        """Generate personality-specific behavioral insights"""
        insights = []
        
        # Social behavior patterns
        if trait_scores.get("social_behavior", 0) >= 3.5:
            insights.append("You are naturally social and exceptionally comfortable in group settings, which significantly enhances team collaboration and networking opportunities. You excel at building rapport quickly, facilitating group dynamics, and creating inclusive environments where others feel valued and heard. This social confidence is a powerful asset in leadership and collaborative roles.")
        elif trait_scores.get("social_behavior", 0) < 2.5:
            insights.append("Building social confidence will help you network more effectively and collaborate more comfortably in team environments. Consider practicing active listening, asking open-ended questions, and gradually expanding your comfort zone in social situations. Remember that authentic connections often develop through genuine interest in others' perspectives and experiences.")
        
        # Adaptability patterns
        if trait_scores.get("change_adaptability", 0) >= 3.5:
            insights.append("You embrace change with remarkable enthusiasm and adapt quickly to new situations, which is invaluable in dynamic professional environments. You view change as an opportunity for growth and learning rather than a threat, allowing you to thrive in evolving circumstances and lead others through transitions effectively.")
        elif trait_scores.get("change_adaptability", 0) < 2.5:
            insights.append("Developing flexibility will help you navigate change and uncertainty more effectively. Practice reframing change as an opportunity for growth, focus on what you can control, and build resilience through small, manageable challenges. Consider developing routines that provide stability while remaining open to new approaches.")
        
        # Risk tolerance patterns
        if trait_scores.get("risk_tolerance", 0) >= 3.5:
            insights.append("You're comfortable taking calculated risks and seizing opportunities that others might avoid, which can lead to significant innovation and career advancement. You balance risk-taking with careful analysis, making informed decisions that often result in breakthrough outcomes. This entrepreneurial mindset is valuable in leadership and innovation-focused roles.")
        elif trait_scores.get("risk_tolerance", 0) < 2.5:
            insights.append("Learning to take calculated risks will help you seize opportunities and grow professionally. Start with small, low-stakes risks to build confidence, practice analyzing potential outcomes, and gradually expand your comfort zone. Remember that calculated risks often lead to the most significant personal and professional growth.")
        
        # Work preference patterns
        if trait_scores.get("work_preference", 0) >= 3.5:
            insights.append("You have a clear understanding of your work preferences and excel in environments that align with your natural working style. You're able to communicate your needs effectively and create conditions that maximize your productivity and job satisfaction. This self-awareness helps you make informed career decisions and advocate for optimal working conditions.")
        elif trait_scores.get("work_preference", 0) < 2.5:
            insights.append("Exploring different work styles will help you identify approaches that maximize your productivity and satisfaction. Consider experimenting with various work methods, collaboration styles, and project structures. Understanding your preferences will help you communicate your needs effectively and create optimal working conditions.")
        
        # Leadership style patterns
        if trait_scores.get("leadership_style", 0) >= 3.5:
            insights.append("You demonstrate strong leadership capabilities with a clear understanding of your authentic leadership style. You inspire others through your actions and words, create environments where team members can thrive, and balance direction with empowerment. This leadership presence is valuable in both formal and informal leadership roles.")
        elif trait_scores.get("leadership_style", 0) < 2.5:
            insights.append("Developing your leadership skills will enhance your ability to influence others and drive positive change. Focus on building emotional intelligence, practicing clear communication, and learning to delegate effectively. Consider taking on small leadership opportunities to build confidence and develop your unique leadership style.")
        
        return insights
    
    def _generate_communication_insights(self, trait_scores: Dict[str, float], interpretations: Dict[str, str]) -> List[str]:
        """Generate communication-specific behavioral insights"""
        insights = []
        
        # Listening patterns
        if trait_scores.get("listening_skills", 0) >= 3.5:
            insights.append("You are an exceptional listener who demonstrates genuine empathy and understanding in conversations. You focus intently on others' words, ask thoughtful follow-up questions, and create safe spaces for open dialogue. This deep listening ability builds strong relationships and helps you understand complex situations from multiple perspectives.")
        elif trait_scores.get("listening_skills", 0) < 2.5:
            insights.append("Developing active listening skills will significantly improve your understanding and relationships. Practice focusing entirely on the speaker, avoiding interruptions, and asking clarifying questions. Learn to listen not just to words but to emotions and underlying messages. This skill will enhance your communication effectiveness and build stronger connections.")
        
        # Conflict resolution patterns
        if trait_scores.get("conflict_resolution", 0) >= 3.5:
            insights.append("You handle conflicts with remarkable skill and diplomacy, finding solutions that work for everyone involved. You approach disagreements with empathy, focus on underlying interests rather than positions, and create win-win outcomes. This conflict resolution ability is invaluable in team environments and leadership roles.")
        elif trait_scores.get("conflict_resolution", 0) < 2.5:
            insights.append("Learning conflict resolution techniques will help you navigate disagreements more effectively and maintain positive relationships. Practice active listening during conflicts, focus on finding common ground, and develop skills in mediation and negotiation. Remember that most conflicts arise from miscommunication or unmet needs.")
        
        # Presentation patterns
        if trait_scores.get("presentation_style", 0) >= 3.5:
            insights.append("You present information with exceptional clarity and engagement, making complex topics accessible and compelling to diverse audiences. You structure your presentations logically, use visual aids effectively, and adapt your delivery style to connect with different learning preferences. This presentation ability enhances your influence and professional impact.")
        elif trait_scores.get("presentation_style", 0) < 2.5:
            insights.append("Developing presentation skills will enhance your ability to influence and inform others effectively. Focus on structuring content clearly, using visual aids appropriately, and practicing delivery techniques. Learn to read your audience and adapt your style accordingly. Strong presentation skills are essential for career advancement and leadership roles.")
        
        # Cross-cultural communication patterns
        if trait_scores.get("cross_cultural_communication", 0) >= 3.5:
            insights.append("You excel at cross-cultural communication and adapt your approach to work effectively with diverse teams and global audiences. You demonstrate cultural sensitivity, respect for different communication styles, and the ability to bridge cultural gaps. This skill is increasingly valuable in our globalized workplace.")
        elif trait_scores.get("cross_cultural_communication", 0) < 2.5:
            insights.append("Developing cross-cultural communication skills will help you work more effectively in diverse environments. Learn about different cultural communication norms, practice adapting your style, and develop awareness of cultural differences in body language, directness, and decision-making processes.")
        
        # Virtual communication patterns
        if trait_scores.get("virtual_communication", 0) >= 3.5:
            insights.append("You communicate exceptionally well in virtual environments, maintaining engagement and clarity despite technological barriers. You use digital tools effectively, manage virtual meetings skillfully, and create inclusive online spaces. This ability is crucial in today's remote and hybrid work environments.")
        elif trait_scores.get("virtual_communication", 0) < 2.5:
            insights.append("Improving virtual communication skills will enhance your effectiveness in remote and hybrid work environments. Focus on clear audio/video setup, engaging virtual meeting techniques, and using digital collaboration tools effectively. Learn to compensate for the lack of physical presence through enhanced verbal and written communication.")
        
        return insights
    
    def _generate_development_focus(self, test_type: GenericTestType, trait_scores: Dict[str, float]) -> List[str]:
        """Generate focused development areas"""
        focus_areas = []
        
        # Identify the lowest scoring traits for focused development
        sorted_traits = sorted(trait_scores.items(), key=lambda x: x[1])
        
        for trait, score in sorted_traits[:3]:  # Focus on top 3 lowest scores
            if score < 2.5:
                trait_display = trait.replace('_', ' ').title()
                focus_areas.append(f"Prioritize developing {trait_display.lower()} skills")
        
        if not focus_areas:
            focus_areas.append("Continue building on your existing strengths")
            focus_areas.append("Explore advanced applications of your current skills")
        
        return focus_areas
    
    def _generate_feedback_summary(self, test_type: GenericTestType, high_count: int, medium_count: int, low_count: int) -> str:
        """Generate a constructive summary of the assessment"""
        test_name = test_type.value.replace('_', ' ').title()
        
        if high_count > medium_count + low_count:
            return f"Your {test_name} assessment reveals strong behavioral competencies with several areas of excellence. You demonstrate natural strengths that can be leveraged for continued growth and development."
        elif low_count > high_count + medium_count:
            return f"Your {test_name} assessment identifies several opportunities for personal and professional development. Focus on building core competencies that will enhance your effectiveness in various situations."
        else:
            return f"Your {test_name} assessment shows a balanced profile with both strengths and areas for development. This provides a solid foundation for targeted skill building and personal growth."
    
    def _infer_category_from_question(self, question_text: str, test_type: GenericTestType) -> str:
        """Infer category from question text when category is missing"""
        question_lower = question_text.lower()
        
        # Psychometric test category inference
        if test_type == GenericTestType.PSYCHOMETRIC:
            if any(word in question_lower for word in ["problem", "solve", "approach", "solution"]):
                return "problem_solving"
            elif any(word in question_lower for word in ["stress", "pressure", "handle", "cope"]):
                return "stress_management"
            elif any(word in question_lower for word in ["decision", "decide", "choose", "select"]):
                return "decision_making"
            elif any(word in question_lower for word in ["communicate", "communication", "talk", "discuss"]):
                return "communication_style"
            elif any(word in question_lower for word in ["environment", "workplace", "office", "space"]):
                return "work_environment"
            elif any(word in question_lower for word in ["feedback", "criticism", "review"]):
                return "feedback_reception"
            elif any(word in question_lower for word in ["team", "teamwork", "collaborate", "group"]):
                return "teamwork_style"
            elif any(word in question_lower for word in ["deadline", "time", "pressure", "urgent"]):
                return "time_pressure"
            elif any(word in question_lower for word in ["motivate", "motivation", "drive", "inspire"]):
                return "motivation"
            elif any(word in question_lower for word in ["learn", "learning", "study", "training"]):
                return "learning_style"
            elif any(word in question_lower for word in ["persist", "persistence", "continue", "keep"]):
                return "persistence"
            elif any(word in question_lower for word in ["organize", "organization", "plan", "structure"]):
                return "organization_style"
            elif any(word in question_lower for word in ["conflict", "disagree", "argument", "dispute"]):
                return "conflict_resolution"
            elif any(word in question_lower for word in ["project", "task", "assignment", "work"]):
                return "project_preference"
        
        # Personality test category inference
        elif test_type == GenericTestType.PERSONALITY:
            if any(word in question_lower for word in ["social", "people", "meet", "interact"]):
                return "social_behavior"
            elif any(word in question_lower for word in ["work", "job", "career", "professional"]):
                return "work_preference"
            elif any(word in question_lower for word in ["feedback", "criticism", "review"]):
                return "feedback_handling"
            elif any(word in question_lower for word in ["motivate", "motivation", "drive", "inspire"]):
                return "motivation"
            elif any(word in question_lower for word in ["leisure", "free time", "hobby", "relax"]):
                return "leisure_preference"
            elif any(word in question_lower for word in ["decision", "decide", "choose", "select"]):
                return "decision_style"
            elif any(word in question_lower for word in ["communicate", "communication", "talk", "discuss"]):
                return "communication_style"
            elif any(word in question_lower for word in ["environment", "workplace", "office", "space"]):
                return "environment_preference"
            elif any(word in question_lower for word in ["change", "adapt", "flexible", "new"]):
                return "change_adaptability"
            elif any(word in question_lower for word in ["energy", "energize", "recharge", "tired"]):
                return "energy_source"
            elif any(word in question_lower for word in ["risk", "safe", "danger", "adventure"]):
                return "risk_tolerance"
            elif any(word in question_lower for word in ["lead", "leadership", "manage", "direct"]):
                return "leadership_style"
            elif any(word in question_lower for word in ["pressure", "stress", "urgent", "deadline"]):
                return "pressure_response"
            elif any(word in question_lower for word in ["feedback", "criticism", "review"]):
                return "feedback_preference"
            elif any(word in question_lower for word in ["learn", "learning", "study", "training"]):
                return "learning_preference"
            elif any(word in question_lower for word in ["career", "advancement", "success", "goal"]):
                return "career_motivation"
        
        # Communication test category inference
        elif test_type == GenericTestType.COMMUNICATION:
            if any(word in question_lower for word in ["conflict", "disagree", "argument", "dispute"]):
                return "conflict_resolution"
            elif any(word in question_lower for word in ["present", "presentation", "speak", "talk"]):
                return "presentation_style"
            elif any(word in question_lower for word in ["listen", "listening", "hear", "attention"]):
                return "listening_skills"
            elif any(word in question_lower for word in ["communicate", "communication", "talk", "discuss"]):
                return "communication_preference"
            elif any(word in question_lower for word in ["active", "attentive", "focus", "concentrate"]):
                return "active_listening"
            elif any(word in question_lower for word in ["difficult", "hard", "challenging", "tough"]):
                return "difficult_conversations"
            elif any(word in question_lower for word in ["feedback", "criticism", "review", "suggest"]):
                return "feedback_delivery"
            elif any(word in question_lower for word in ["interrupt", "interruption", "stop", "wait"]):
                return "conversation_management"
            elif any(word in question_lower for word in ["cultural", "culture", "diverse", "different"]):
                return "cross_cultural_communication"
            elif any(word in question_lower for word in ["instruction", "direction", "guidance", "tell"]):
                return "instruction_reception"
            elif any(word in question_lower for word in ["email", "message", "write", "written"]):
                return "email_communication"
            elif any(word in question_lower for word in ["misunderstand", "confusion", "clear", "clarify"]):
                return "miscommunication_resolution"
            elif any(word in question_lower for word in ["team", "teamwork", "collaborate", "group"]):
                return "team_communication"
            elif any(word in question_lower for word in ["virtual", "online", "remote", "video"]):
                return "virtual_communication"
            elif any(word in question_lower for word in ["disagree", "disagreement", "oppose", "against"]):
                return "disagreement_expression"
            elif any(word in question_lower for word in ["clear", "clarity", "understand", "confuse"]):
                return "message_clarity"
            elif any(word in question_lower for word in ["manager", "boss", "supervisor", "upward"]):
                return "upward_communication"
            elif any(word in question_lower for word in ["crisis", "emergency", "urgent", "critical"]):
                return "crisis_communication"
        
        # Default fallback
        return "general"

# Global instance
generic_test_evaluator = GenericTestEvaluator()

@traceable(name="evaluate_generic_test")
async def _evaluate_generic_test(
    state: Dict[str, Any], 
    submission: Dict[str, Any], 
    test_type: GenericTestType, 
    config: 'EvaluatorConfig',
    log_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a generic test using behavioral scoring matrices.
    
    Args:
        state: Original state data
        submission: Validated submission data
        test_type: Identified generic test type
        config: Configuration instance
        log_context: Logging context
        
    Returns:
        Dictionary with generic test evaluation results
    """
    try:
        # Extract questions and answers from submission
        questions = []
        answers = []
        
        # Collect all questions and their corresponding answers
        for qtype, qlist in submission.items():
            if isinstance(qlist, list):
                for q in qlist:
                    if isinstance(q, dict):
                        questions.append(q)
                        # Extract answer from the question data
                        # Accept common MCQ fields to be robust across payload shapes
                        sel = (
                            q.get("selected_option")
                            or q.get("user_answer")
                            or q.get("selected")
                            or q.get("selected_choice")
                            or q.get("choice")
                            or q.get("answer")
                        )
                        # Normalize dict/number answers to a string code if needed (e.g., 0->A)
                        try:
                            if isinstance(sel, int):
                                sel = "ABCD"[sel] if 0 <= sel < 4 else str(sel)
                            elif isinstance(sel, dict):
                                sel = sel.get("option") or sel.get("key") or sel.get("label") or ""
                            elif sel is None:
                                sel = ""
                        except Exception:
                            sel = str(sel or "")
                        answer = {
                            "selected_option": sel,
                            "question_id": q.get("question_id", ""),
                            "question_text": q.get("question_text", "")
                        }
                        answers.append(answer)
        
        # Debug logging for question counting
        log.info(f"Question extraction debug: Found {len(questions)} questions, {len(answers)} answers")
        answered_count = len([a for a in answers if str(a.get("selected_option", "")).strip()])
        log.info(f"Answered questions count: {answered_count}")
        
        # Additional debug logging for submission structure
        log.info(f"Submission structure debug: {list(submission.keys())}")
        for qtype, qlist in submission.items():
            if isinstance(qlist, list):
                log.info(f"Question type '{qtype}': {len(qlist)} questions")
                for i, q in enumerate(qlist[:3]):  # Log first 3 questions for debugging
                    sel_dbg = q.get('selected_option') or q.get('user_answer', '')
                    log.info(f"  Question {i+1}: has_answer={bool(sel_dbg)}, selected='{sel_dbg}'")
        
        if not questions or not answers:
            processing_time = _calculate_processing_time(log_context["start_time"])
            AgentLogger.log_warning(log_context, "No questions or answers found for generic test evaluation")
            return _create_error_response("No questions or answers found", processing_time)
        
        log.info(f"Evaluating {len(questions)} questions for {test_type.value} test")
        
        # Use the generic test evaluator
        evaluation_result = await generic_test_evaluator.evaluate_generic_test(test_type, questions, answers)
        
        if "error" in evaluation_result:
            processing_time = _calculate_processing_time(log_context["start_time"])
            AgentLogger.log_error(log_context, f"Generic test evaluation failed: {evaluation_result['error']}", processing_time)
            return _create_error_response(f"Generic test evaluation failed: {evaluation_result['error']}", processing_time)
        
        # Format the result to match the expected structure
        processing_time = _calculate_processing_time(log_context["start_time"])
        
        result = {
            "uid": state.get("uid", ""),
            "tenant_id": state.get("tenant_id", "default"),
            "callback_url": state.get("callback_url", ""),
            "assessment_id": state.get("assessment_id", ""),
            "assessment_topic": _sanitize_input(str(state.get("assessment_topic", "")), config=config),
            "question_doc_id": _sanitize_input(str(state.get("question_doc_id", "")), config=config),
            "assessment_results": {
                "test_type": evaluation_result["test_type"],
                "constructive_feedback": evaluation_result["constructive_feedback"],
                "trait_scores": evaluation_result["trait_scores"],
                "interpretations": evaluation_result["interpretations"],
                "recommendations": evaluation_result["recommendations"],
                "evaluation_method": evaluation_result["evaluation_method"],
                "total_questions": len(questions),
                "answered_questions": answered_count
            },
            "confidence_score": 0.9,  # High confidence for behavioral scoring
            "confidence_level": "high",
            "processing_time_seconds": processing_time,
            "error": "",
            "validation_error": "",
            "security_error": "",
            "rate_limit_error": "",
            "session_id": state.get("session_id", "")
        }
        
        # Log successful evaluation
        AgentLogger.log_info(log_context, f"Generic test evaluation completed successfully for {test_type.value}")
        log.info(f"Generic test evaluation result: traits={len(evaluation_result['trait_scores'])}, insights={len(evaluation_result['constructive_feedback'].get('behavioral_insights', []))}")
        
        # Debug logging for trait profile
        constructive_feedback = evaluation_result.get("constructive_feedback", {})
        log.info(f"Trait profile debug: strengths={len(constructive_feedback.get('strengths', []))}, areas_for_development={len(constructive_feedback.get('areas_for_development', []))}")
        log.info(f"Behavioral insights: {len(constructive_feedback.get('behavioral_insights', []))}")
        log.info(f"Summary available: {bool(constructive_feedback.get('summary'))}")
        
        return result
        
    except Exception as e:
        processing_time = _calculate_processing_time(log_context["start_time"])
        error_msg = f"Generic test evaluation error: {str(e)}"
        AgentLogger.log_error(log_context, error_msg, processing_time)
        log.error(f"Generic test evaluation failed: {e}", exc_info=True)
        return _create_error_response(error_msg, processing_time)

# Constants
DEFAULT_MAX_SCORE = 5
NORMALIZED_MAX_SCORE = 100
ALLOWED_QUESTION_TYPES = {  # Security: Whitelist allowed question types
    "multiple_choice", "mcq", "essay", "short_answer", "long_answer", "coding", "true_false"
}


class Constants:
    """Named constants to replace magic numbers"""
    MAX_LLM_RESPONSE_LENGTH = config.max_response_length
    CONFIDENCE_SCORE_LOWER_BOUND = 0.2
    CONFIDENCE_SCORE_UPPER_BOUND = 0.8
    BASE_CONFIDENCE = 0.5
    NUMERIC_RESPONSE_CONFIDENCE_BONUS = 0.3
    EDGE_SCORE_CONFIDENCE_PENALTY = 0.2
    MID_RANGE_CONFIDENCE_BONUS = 0.2
    SUCCESS_RATE_THRESHOLD = 0.8
    LOW_SUCCESS_RATE_THRESHOLD = 0.7
    LARGE_QUESTION_COUNT_THRESHOLD = 20
    MAX_SANITIZED_ERROR_LENGTH = 200
    MAX_SANITIZED_FIELD_LENGTH = 50

    # Validation constants
    MIN_VALID_SCORE = 0
    MAX_VALID_SCORE = 100
    AVERAGE_WORD_SCORE_MULTIPLIER = 0.6
    POOR_WORD_SCORE_MULTIPLIER = 0.3

# Setup logging - respect environment-based log level from log_handler
log = logging.getLogger(__name__)
# Do not hardcode log level - let it inherit from root logger or environment configuration
# The log_handler.setup_loggers() sets the appropriate level based on APP_ENV
# - development/dev: DEBUG
# - qa/production: WARNING

# Use centralized injection filters
INJECTION_FILTERS = INJECTION_FILTERS

# Custom memory class for assessment evaluator (extends base memory)
class AssessmentEvaluatorMemory(BaseAgentMemory):
    def __init__(self, tenant_id: str = "default"):
        super().__init__(tenant_id, config.adaptation_window)
        # Add any assessment evaluator specific fields here if needed

# Use centralized memory management
async def get_assessment_evaluator_memory(tenant_id: str = "default") -> AssessmentEvaluatorMemory:
    """Get or create tenant-scoped assessment evaluator memory."""
    return await get_agent_memory("assessment_evaluator", tenant_id, AssessmentEvaluatorMemory)

SCORE_EXTRACTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in [
        r'(?:score|rating|points?):\s*(\d+)',  # "Score: 8"
        r'(\d+)\s*(?:out of|/)\s*\d+',        # "8 out of 10"
        r'(?:final|total)\s*(?:score|rating):\s*(\d+)',  # "Final score: 8"
        r'(?:grade|mark):\s*(\d+)',           # "Grade: 8"
        r'(\d+)\s*(?:points?|pts)',           # "8 points"
    ]
]

NUMBER_PATTERN = re.compile(r'\b(\d+)\b')


# Custom Exceptions
class EvaluationError(Exception):
    """Base exception for evaluation errors"""
    pass


class ValidationError(EvaluationError):
    """Input validation failed"""
    pass


class ProcessingError(EvaluationError):
    """Processing failed"""
    pass


@dataclass
class EvaluatorConfig:
    """Configuration for the evaluator with environment variable support"""
    max_concurrent_tasks: int = 50
    max_input_length: int = 10000
    evaluation_timeout: float = 300.0
    confidence_threshold_simple: float = 0.8
    confidence_threshold_complex: float = 0.9
    confidence_threshold_adaptive: float = 0.95

    @classmethod
    def from_env(cls):
        """Create configuration from environment variables"""
        return cls(
            max_concurrent_tasks=int(os.getenv('EVALUATOR_MAX_CONCURRENT_TASKS', 50)),
            max_input_length=int(os.getenv('EVALUATOR_MAX_INPUT_LENGTH', 10000)),
            evaluation_timeout=float(os.getenv('EVALUATOR_TIMEOUT', 300.0)),
            confidence_threshold_simple=float(os.getenv('EVALUATOR_CONFIDENCE_SIMPLE', 0.8)),
            confidence_threshold_complex=float(os.getenv('EVALUATOR_CONFIDENCE_COMPLEX', 0.9)),
            confidence_threshold_adaptive=float(os.getenv('EVALUATOR_CONFIDENCE_ADAPTIVE', 0.95))
        )


# Agentic AI: Dynamic evaluation strategies based on configuration
def get_evaluation_strategies(config: EvaluatorConfig) -> Dict[str, Dict[str, Any]]:
    """Get evaluation strategies with current configuration values"""
    return {
        "simple": {"max_retries": 1, "confidence_threshold": config.confidence_threshold_simple},
        "complex": {"max_retries": 2, "confidence_threshold": config.confidence_threshold_complex},
        "adaptive": {"max_retries": 3, "confidence_threshold": config.confidence_threshold_adaptive}
    }


def _filter_prompt_injection(text: str) -> str:
    """
    Filter potential prompt injection attempts from user input using compiled patterns.

    Args:
        text: Input text to filter

    Returns:
        Filtered text with dangerous patterns replaced
    """
    if not isinstance(text, str):
        return ""

    filtered_text = text
    for pattern in INJECTION_FILTERS:
        if pattern.search(text):
            log.warning("Potential prompt injection attempt detected and filtered")
            filtered_text = pattern.sub("[CONTENT FILTERED FOR SECURITY]", filtered_text)

    return filtered_text


class EvaluationContext:
    """Agentic AI: Thread-safe context and memory for evaluation decisions"""
    def __init__(self):
        self._lock = asyncio.Lock()
        self.evaluation_count = 0
        self.success_rate = 1.0
        self.avg_processing_time = 0.0
        self.failed_question_types = set()

    async def update_performance(self, success: bool, processing_time: float, qtype: str):
        """Async thread-safe performance update with atomic operations"""
        async with self._lock:
            try:
                # Atomic increment and update to prevent race conditions
                self.evaluation_count += 1
                count = self.evaluation_count

                # Use running average for better numerical stability
                if count == 1:
                    self.success_rate = 1.0 if success else 0.0
                    self.avg_processing_time = processing_time
                else:
                    # Incremental average update
                    self.success_rate = (self.success_rate * (count - 1) + (1 if success else 0)) / count
                    self.avg_processing_time = (self.avg_processing_time * (count - 1) + processing_time) / count

                if not success:
                    self.failed_question_types.add(qtype)
            except Exception as e:
                # If update fails, decrement count to maintain consistency
                self.evaluation_count = max(0, self.evaluation_count - 1)
                log.error(f"Failed to update performance metrics: {type(e).__name__}")
                raise

    async def should_use_complex_strategy(self, qtype: str) -> bool:
        """Async thread-safe strategy decision based on historical performance"""
        async with self._lock:
            return qtype in self.failed_question_types or self.success_rate < 0.8

    async def get_stats(self) -> Dict[str, Any]:
        """Get async thread-safe statistics snapshot"""
        async with self._lock:
            return {
                "evaluation_count": self.evaluation_count,
                "success_rate": round(self.success_rate, 3),
                "avg_processing_time": round(self.avg_processing_time, 3),
                "failed_question_types": list(self.failed_question_types)
            }


async def _plan_evaluation_strategy(questions: List[Dict], context: EvaluationContext) -> str:
    """
    Agentic AI: Plan evaluation strategy based on question complexity and history

    Args:
        questions: List of questions to evaluate
        context: Historical performance context

    Returns:
        Strategy name ('simple', 'complex', 'adaptive')
    """
    # Analyze question complexity
    has_essays = any(q.get("qtype") == "essay" for q in questions)
    has_coding = any(q.get("qtype") == "coding" for q in questions)
    question_count = len(questions)

    # Get current performance stats asynchronously
    stats = await context.get_stats()
    success_rate = stats["success_rate"]

    # Adaptive planning based on context and complexity
    if success_rate < Constants.LOW_SUCCESS_RATE_THRESHOLD or has_coding:
        return "adaptive"
    elif has_essays or question_count > Constants.LARGE_QUESTION_COUNT_THRESHOLD:
        return "complex"
    else:
        return "simple"


def _assess_confidence(score: int, max_score: int, response: str) -> float:
    """
    Agentic AI: Self-reflection - assess confidence in evaluation using named constants.

    Args:
        score: Assigned score
        max_score: Maximum possible score
        response: Raw LLM response

    Returns:
        Confidence score between 0 and 1
    """
    confidence = Constants.BASE_CONFIDENCE

    # Higher confidence for clear numeric responses
    if response.strip().isdigit():
        confidence += Constants.NUMERIC_RESPONSE_CONFIDENCE_BONUS

    # Lower confidence for edge scores (might be uncertain)
    if score == 0 or score == max_score:
        confidence -= Constants.EDGE_SCORE_CONFIDENCE_PENALTY

    # Higher confidence for mid-range scores
    if (Constants.CONFIDENCE_SCORE_LOWER_BOUND * max_score <= score <=
        Constants.CONFIDENCE_SCORE_UPPER_BOUND * max_score):
        confidence += Constants.MID_RANGE_CONFIDENCE_BONUS

    return max(0.0, min(1.0, confidence))

def _sanitize_input(text: str, max_length: int = None, config: EvaluatorConfig = None) -> str:
    """
    Comprehensive sanitization of user input with security filtering.

    Args:
        text: Raw user input text
        max_length: Maximum allowed length (uses config if not specified)
        config: Configuration instance (required for proper operation)

    Returns:
        Sanitized and truncated text

    Raises:
        ValueError: If config is None and max_length is not provided
    """
    if not isinstance(text, str):
        return ""

    if config is None and max_length is None:
        raise ValueError("Either config or max_length must be provided")

    if max_length is None:
        max_length = config.max_input_length

    # Step 1: Filter potential prompt injection
    filtered_text = _filter_prompt_injection(text)

    # Step 2: Remove markdown code blocks and formatting
    # Remove triple backticks and language identifiers (e.g., ```python, ```verilog)
    sanitized = re.sub(r'```[\w]*\n?', '', filtered_text)
    
    # Remove inline code backticks
    sanitized = re.sub(r'`([^`]+)`', r'\1', sanitized)
    
    # Remove markdown bold/italic markers but keep the text
    sanitized = re.sub(r'\*\*([^\*]+)\*\*', r'\1', sanitized)  
    sanitized = re.sub(r'\*([^\*]+)\*', r'\1', sanitized)      
    sanitized = re.sub(r'__([^_]+)__', r'\1', sanitized)       
    sanitized = re.sub(r'_([^_]+)_', r'\1', sanitized)         
    
    # Remove HTML tags while preserving text content
    sanitized = re.sub(r'<[^>]+>', '', sanitized)
    
    # Unescape any HTML entities that might be present in the input
    sanitized = html.unescape(sanitized)

    # Step 3: Remove excessive whitespace and clean up
    sanitized = re.sub(r'\s+', ' ', sanitized)
    sanitized = sanitized.strip()

    # Step 4: Length limiting
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "... [truncated]"
        log.warning(f"Input truncated to {max_length} characters for security")

    return sanitized


def _validate_question_type(qtype: str) -> str:
    """
    Validate and sanitize question type.

    Args:
        qtype: Question type from user input

    Returns:
        Validated question type or default
    """
    if not isinstance(qtype, str):
        return "unknown"

    # Security: Only allow whitelisted question types
    sanitized_type = qtype.lower().strip()
    if sanitized_type in ALLOWED_QUESTION_TYPES:
        return sanitized_type
    else:
        log.warning(f"Invalid question type '{qtype}' replaced with 'unknown'")
        return "unknown"


def _validate_max_score(score: Any) -> int:
    """
    Validate and sanitize max score value.

    Args:
        score: Score value from user input

    Returns:
        Valid integer score between 1 and 100
    """
    try:
        if isinstance(score, (int, float)):
            score_int = int(score)
            # Security: Prevent negative or excessively large scores
            return max(1, min(score_int, 100))
        return DEFAULT_MAX_SCORE
    except (ValueError, TypeError):
        return DEFAULT_MAX_SCORE

def _extract_score_from_response(response: str, max_score: int) -> int:
    """
    Robustly extract numeric score from LLM response.
    
    COMPLETE FIX: Better extraction with comprehensive logging
    """
    try:
        if not isinstance(response, str):
            log.warning(f"Invalid response type: {type(response)}, returning 0")
            return 0

        response = response[:Constants.MAX_LLM_RESPONSE_LENGTH].strip()

        # Strategy 1: Direct integer conversion
        try:
            score = int(response)
            clamped = max(0, min(score, max_score))
            log.debug(f"Score extracted (direct): '{response}' -> {clamped}")
            return clamped
        except ValueError:
            pass

        # Strategy 2: Pattern matching
        for pattern in SCORE_EXTRACTION_PATTERNS:
            match = pattern.search(response)
            if match:
                score = int(match.group(1))
                clamped = max(0, min(score, max_score))
                log.debug(f"Score extracted (pattern): '{response}' -> {clamped}")
                return clamped

        # Strategy 3: Find any valid numbers
        numbers = NUMBER_PATTERN.findall(response)
        if numbers:
            for num_str in numbers:
                num = int(num_str)
                if 0 <= num <= max_score:
                    log.debug(f"Score extracted (valid number): '{response}' -> {num}")
                    return num
            
            # Use first number if no valid one found
            score = int(numbers[0])
            clamped = max(0, min(score, max_score))
            log.debug(f"Score extracted (first number): '{response}' -> {clamped}")
            return clamped

        # Strategy 4: Word-based scores
        word_scores = {
            'zero': 0, 'none': 0, 'fail': 0,
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'excellent': max_score, 'perfect': max_score, 'full': max_score,
            'good': int(max_score * 0.8),
            'average': int(max_score * 0.6),
            'poor': int(max_score * 0.3)
        }

        response_lower = response.lower()
        for word, score in word_scores.items():
            if word in response_lower:
                clamped = max(0, min(score, max_score))
                log.debug(f"Score extracted (word): '{response}' -> '{word}' -> {clamped}")
                return clamped

        log.warning(f"⚠ Could not extract score from: '{response}', returning 0")
        return 0

    except Exception as e:
        log.error(f"Score extraction exception: {type(e).__name__} for '{response}', returning 0")
        return 0
        
@traceable(name="evaluate_question_with_llm")
async def _evaluate_question_with_llm(
    question_text: str,
    user_answer: str,
    qtype: str,
    max_score: int,
    strategy: str,
    config: EvaluatorConfig,
    expected_answer: str = None
) -> Tuple[int, float]:
    """
    Evaluate a single question using LLM with optional expected answer comparison.
    
    CRITICAL FIX: Always use expected_answer comparison if available
    """
    # FIX: Log the inputs for debugging
    log.info(f"=== Evaluating Question ===")
    log.info(f"Type: {qtype}, Max Score: {max_score}")
    log.info(f"Question: {question_text[:100]}...")
    log.info(f"User Answer: '{user_answer}'")
    log.info(f"Expected Answer: '{expected_answer}'")
    log.info(f"Has Expected Answer: {bool(expected_answer and str(expected_answer).strip())}")
    
    # CRITICAL FIX: Use expected answer comparison if available (highest priority)
    if expected_answer and str(expected_answer).strip():
        log.info(f"✓ Using expected_answer comparison for {qtype}")
        return await _evaluate_question_with_expected_answer(
            question_text, user_answer, expected_answer, qtype, max_score, strategy, config
        )
    
    # Fallback to standard LLM evaluation only if no expected answer
    log.info(f"⚠ No expected_answer available, using standard LLM evaluation for {qtype}")
    
    # Use enhanced evaluation for short and long answer questions
    if qtype in ["short_answer", "long_answer"]:
        try:
            score, confidence, analysis_details = await evaluate_short_long_answer_enhanced(
                question_text, user_answer, qtype, max_score, config
            )
            log.debug(f"Enhanced evaluation for {qtype}: score={score}, confidence={confidence:.2f}")
            return score, confidence
        except Exception as e:
            log.warning(f"Enhanced evaluation failed for {qtype}, falling back to standard: {e}")
    
    # Standard evaluation for other question types or fallback
    strategies = get_evaluation_strategies(config)
    strategy_config = strategies.get(strategy, strategies["simple"])
    max_retries = strategy_config["max_retries"]
    confidence_threshold = strategy_config["confidence_threshold"]

    prompt_template = """EVALUATION TASK
=================
Type: {question_type}
Question: {question_content}
Candidate Response: {candidate_response}
Maximum Points: {max_points}

INSTRUCTIONS:
- Grade based on accuracy, completeness, and clarity
- CRITICAL: Answer must FULLY address ALL parts of the question to receive full marks
- Do NOT award full marks for partial matches (e.g., only first few words matching)
- Do NOT award points for trivial affirmations (e.g., "correct", "yes", "ok") - these must score 0
- Do NOT award points for answers with < 3 words - these must score 0
- Check completeness: if question asks for examples/explanations/comparisons, ALL must be provided
- Award partial credit ONLY for partially correct answers - incomplete answers should receive reduced scores
- Very short answers (≤6 words) should receive maximum 30% of max score even if partially correct
- Score must be integer between 0 and {max_points}
- Respond with ONLY the numeric score
- No explanations or additional text

SCORE: """

    best_score, best_confidence = 0, 0.0

    for attempt in range(max_retries):
        try:
            prompt = prompt_template.format(
                question_type=qtype,
                question_content=question_text,
                candidate_response=user_answer,
                max_points=max_score
            )

            start_time = time.time()
            llm_response = await invoke_llm(
                prompt=prompt,
                task_type="assessment_evaluation",
                agent_name="assessment_evaluator"
            )

            if not llm_response:
                raise ProcessingError("Empty LLM response")

            content = getattr(llm_response, "content", str(llm_response)).strip()

            if not content:
                raise ProcessingError("Empty LLM content")

            score = _extract_score_from_response(content, max_score)
            confidence = _assess_confidence(score, max_score, content)

            if confidence > best_confidence:
                best_score, best_confidence = score, confidence

            if confidence >= confidence_threshold:
                log.debug(f"High confidence ({confidence:.2f}) achieved on attempt {attempt + 1}")
                break

        except Exception as e:
            log.error(f"LLM evaluation failed for question type {qtype}, attempt {attempt + 1}: {type(e).__name__}")
            if attempt == max_retries - 1:
                raise ProcessingError(f"All evaluation attempts failed for {qtype}")
            continue

    log.info(f"Standard evaluation result: score={best_score}/{max_score}, confidence={best_confidence:.2f}")
    return best_score, best_confidence

@traceable(name="evaluate_question_with_expected_answer")
async def _evaluate_question_with_expected_answer(
    question_text: str,
    user_answer: str,
    expected_answer: str,
    qtype: str,
    max_score: int,
    strategy: str,
    config: EvaluatorConfig
) -> Tuple[int, float]:
    """
    Evaluate a question by comparing user answer with expected answer
    
    COMPLETE FIX: Robust MCQ comparison with multiple normalization strategies
    """
    
    # Universal guard: empty/blank user answers receive zero
    try:
        if not str(user_answer or "").strip():
            return 0, 1.0
    except Exception:
        return 0, 1.0
    
    # Guard against trivial answers for free-form types
    try:
        if qtype in ("short_answer", "long_answer"):
            _ua = str(user_answer or "").strip()
            if _is_trivial_affirmation(_ua) or len(_ua.split()) < 3:
                return 0, 0.9
    except Exception:
        pass
    
    if qtype == "mcq":
        # CRITICAL FIX: Comprehensive MCQ answer comparison
        log.info(f"MCQ Comparison - User: '{user_answer}' vs Expected: '{expected_answer}'")
        
        # Normalize both answers
        user_normalized = str(user_answer).strip().upper()
        expected_normalized = str(expected_answer).strip().upper()
        
        log.debug(f"Normalized - User: '{user_normalized}' vs Expected: '{expected_normalized}'")
        
        # Strategy 1: Direct comparison after normalization
        if user_normalized == expected_normalized:
            log.info(f"✓ MCQ CORRECT (direct match): '{user_answer}' == '{expected_answer}'")
            return max_score, 1.0
        
        # Strategy 2: Extract single letter from both (handles "A", "a", "A)", "A.", etc.)
        user_letter = None
        expected_letter = None
        
        # Extract from user answer
        if len(user_normalized) == 1 and user_normalized.isalpha():
            user_letter = user_normalized
        else:
            # Try to extract letter from formatted answer (e.g., "A)", "A.", "A:")
            match = re.match(r'^([A-D])[:\)\.\s]', user_normalized)
            if match:
                user_letter = match.group(1)
        
        # Extract from expected answer
        if len(expected_normalized) == 1 and expected_normalized.isalpha():
            expected_letter = expected_normalized
        else:
            # Try to extract letter from formatted answer
            match = re.match(r'^([A-D])[:\)\.\s]', expected_normalized)
            if match:
                expected_letter = match.group(1)
        
        log.debug(f"Extracted letters - User: '{user_letter}' vs Expected: '{expected_letter}'")
        
        # Compare extracted letters
        if user_letter and expected_letter:
            if user_letter == expected_letter:
                log.info(f"✓ MCQ CORRECT (letter match): '{user_letter}' == '{expected_letter}'")
                return max_score, 1.0
            else:
                log.info(f"✗ MCQ INCORRECT: '{user_letter}' != '{expected_letter}'")
                return 0, 1.0
        
        # Strategy 3: Fallback - if one is single letter, compare with the other
        if user_letter and not expected_letter:
            if user_letter in expected_normalized:
                log.info(f"✓ MCQ CORRECT (letter in expected): '{user_letter}' in '{expected_normalized}'")
                return max_score, 1.0
        
        if expected_letter and not user_letter:
            if expected_letter in user_normalized:
                log.info(f"✓ MCQ CORRECT (letter in user): '{expected_letter}' in '{user_normalized}'")
                return max_score, 1.0
        
        # No match found
        log.info(f"✗ MCQ INCORRECT (no match): User '{user_answer}' != Expected '{expected_answer}'")
        return 0, 1.0
    
    else:
        # For short_answer, long_answer, coding - use LLM with partial credit
        log.info(f"Using LLM comparison for {qtype}")
        
        strategies = get_evaluation_strategies(config)
        strategy_config = strategies.get(strategy, strategies["simple"])
        max_retries = strategy_config["max_retries"]
        confidence_threshold = strategy_config["confidence_threshold"]

        prompt_template = """EVALUATION TASK
=================
Type: {question_type}
Question: {question_content}
Expected Answer: {expected_answer}
Candidate Response: {candidate_response}
Maximum Points: {max_points}

INSTRUCTIONS:
- Compare the candidate's response with the expected answer
- Look for key concepts, technical accuracy, and completeness
- CRITICAL: Answer must FULLY address ALL parts of the question to receive full marks
- CRITICAL: Do NOT award points for superficial or prefix-only matches (e.g., matching the first 3 words)
- CRITICAL: Full marks ({max_points}) ONLY if ≥85% of key concepts in Expected Answer are covered accurately AND comprehensively
- If response is a trivial affirmation (e.g., "correct", "yes") or < 3 words, score 0
- Check completeness: if question asks for examples/explanations/comparisons, ALL must be provided
- Award points based on:
  * Accuracy (40%): Correctness of information
  * Completeness (35%): Coverage of key points - this is CRITICAL for short_answer questions
  * Clarity (15%): Expression quality
  * Relevance (10%): Addresses the question
- Score must be integer between 0 and {max_points}
- Respond with ONLY the numeric score
- No explanations or additional text

PARTIAL CREDIT GUIDELINES (for short_answer and long_answer):
- FULL marks ({max_points}): Answer covers ≥85% of key concepts with accurate, comprehensive explanation
- HIGH partial (70-84% of max): Answer covers MOST key concepts (60-84%) but missing some important points
- MEDIUM partial (40-69% of max): Answer covers SOME key concepts (30-59%) with basic understanding
- LOW partial (20-39% of max): Answer shows MINIMAL understanding, only touches on 1-2 key points
- ZERO (0): Answer is completely wrong, irrelevant, trivial, or < 3 words

IMPORTANT: 
- Brief answers that only mention ONE aspect of a multi-part expected answer should receive LOW partial credit (20-40%)
- Answers must demonstrate understanding of the COMPLETE concept, not just a fragment
- For questions asking "why" or "explain", the answer must include the explanation, not just state a fact

Respond with ONLY the numeric score (integer between 0 and {max_points}).

SCORE: """

        best_score, best_confidence = 0, 0.0

        for attempt in range(max_retries):
            try:
                prompt = prompt_template.format(
                    question_type=qtype,
                    question_content=question_text,
                    expected_answer=expected_answer,
                    candidate_response=user_answer,
                    max_points=max_score
                )

                llm_response = await invoke_llm(
                    prompt=prompt,
                    task_type="assessment_evaluation",
                    agent_name="assessment_evaluator"
                )

                if not llm_response:
                    raise ProcessingError("Empty LLM response")

                content = getattr(llm_response, "content", str(llm_response)).strip()
                score = _extract_score_from_response(content, max_score)
                confidence = _assess_confidence(score, max_score, content)

                log.info(f"{qtype} LLM evaluation: score={score}/{max_score}, confidence={confidence:.2f}")

                if confidence > best_confidence:
                    best_score, best_confidence = score, confidence

                if confidence >= confidence_threshold:
                    break

            except Exception as e:
                log.error(f"Evaluation attempt {attempt + 1} failed: {e}")
                continue

        log.info(f"Final {qtype} score: {best_score}/{max_score}")
        return best_score, best_confidence


@traceable(name="evaluate_short_long_answer_enhanced")
async def evaluate_short_long_answer_enhanced(
    question_text: str,
    user_answer: str,
    qtype: str,
    max_score: int,
    config: EvaluatorConfig
) -> Tuple[int, float, Dict[str, Any]]:
    """Token-optimized enhanced evaluation for short and long answer questions"""
    
    # Early guard: trivial answers or very short answers get zero immediately
    try:
        _ua = str(user_answer or "").strip()
        if not _ua:
            return 0, 1.0, {"word_count": 0, "reason": "empty_answer"}
        if _is_trivial_affirmation(_ua) or len(_ua.split()) < 3:
            log.debug(f"Early guard: trivial or too short answer detected ({len(_ua.split())} words)")
            return 0, 0.9, {"word_count": len(_ua.split()), "reason": "trivial_or_too_short"}
    except Exception:
        pass
    
    # Initialize budget manager for this evaluation
    budget_manager = TokenBudgetManager(max_tokens_per_evaluation=1500)  # Reduced budget for efficiency
    
    # Optimize content for token efficiency
    optimized_question = content_optimizer.truncate_question(question_text)
    optimized_answer = content_optimizer.truncate_answer(user_answer, qtype)
    
    # Quick concept analysis
    concept_summary = concept_analyzer.quick_concept_analysis(user_answer)
    quality_summary = concept_analyzer.get_quality_indicators(user_answer)
    
    # Create compact analysis summary
    analysis_summary = f"{concept_summary}|{quality_summary}"
    
    # Create optimized prompt
    prompt_template = TokenOptimizedPrompts.get_enhanced_evaluation_prompt(
        qtype, max_score, analysis_summary
    )
    
    prompt = prompt_template.format(
        question=optimized_question,
        answer=optimized_answer
    )
    
    # Check and optimize for token budget
    if not budget_manager.can_afford_evaluation(prompt):
        prompt = budget_manager.optimize_prompt_for_budget(prompt)
        log.debug(f"Optimized prompt for token budget: {len(prompt)} chars")
    
    # Single LLM call with retry logic (reduced retries for token efficiency)
    best_score, best_confidence = 0, 0.0
    strategies = get_evaluation_strategies(config)
    strategy_config = strategies.get("simple", strategies["simple"])  # Use simple for token efficiency
    
    for attempt in range(min(2, strategy_config["max_retries"])):  # Limit retries for token efficiency
        try:
            start_time = time.time()
            llm_response = await invoke_llm(
                prompt=prompt,
                task_type="assessment_evaluation_enhanced",
                agent_name="assessment_evaluator"
            )
            
            if llm_response:
                content = getattr(llm_response, "content", str(llm_response)).strip()
                
                # Parse compact response
                score = _extract_score_from_response(content, max_score)
                confidence = _assess_confidence(score, max_score, content)
                
                # Post-score safeguards: prevent false positives from prefix-only matches
                try:
                    ua = str(user_answer or "")
                    # Heuristic 1: very short answers that got high scores should be capped
                    word_count = len(ua.split())
                    if word_count <= 6 and score > max_score * 0.5:
                        # If answer is very short but got high score, reduce it
                        score = min(score, int(max_score * 0.3))
                        confidence = min(confidence, 0.7)
                        log.debug(f"Post-score safeguard: short answer ({word_count} words) with high score, capped to {score}")
                except Exception:
                    # Best-effort safeguards; ignore failures
                    pass
                
                if confidence > best_confidence:
                    best_score, best_confidence = score, confidence
                
                # Log token usage
                token_monitor.log_evaluation_tokens(len(prompt), len(content))
                
                if confidence >= 0.7:  # Lower threshold for token efficiency
                    log.debug(f"Enhanced evaluation completed with confidence {confidence:.2f}")
                    break
                    
        except Exception as e:
            log.error(f"Enhanced evaluation attempt {attempt + 1} failed: {e}")
            continue
    
    # Apply lightweight quality adjustments
    word_count = len(user_answer.split())
    if qtype == "short_answer" and word_count < 10:
        best_score = int(best_score * 0.8)  # Penalty for very short answers
        log.debug(f"Applied short answer penalty: {word_count} words")
    elif qtype == "long_answer" and word_count < 30:
        best_score = int(best_score * 0.7)  # Penalty for insufficient length
        log.debug(f"Applied long answer penalty: {word_count} words")
    
    # Hard guard: trivial affirmations get zero for short/long answers (redundant but safe)
    if qtype in ("short_answer", "long_answer") and _is_trivial_affirmation(user_answer):
        log.debug("Final guard: detected trivial affirmation in free-form answer; forcing score to 0.")
        best_score = 0
        best_confidence = min(best_confidence, 0.6)
    
    # Additional safeguard: very short answers should not get high scores
    word_count = len(user_answer.split())
    if qtype == "short_answer" and word_count <= 6 and best_score > max_score * 0.5:
        best_score = min(best_score, int(max_score * 0.3))
        log.debug(f"Final safeguard: short answer ({word_count} words) capped to {best_score}")
    elif qtype == "long_answer" and word_count <= 10 and best_score > max_score * 0.5:
        best_score = min(best_score, int(max_score * 0.3))
        log.debug(f"Final safeguard: short long answer ({word_count} words) capped to {best_score}")
    
    # Consume budget
    budget_manager.consume_budget(prompt)
    
    return best_score, best_confidence, {
        "concept_summary": concept_summary,
        "quality_summary": quality_summary,
        "word_count": word_count,
        "optimized": True,
        "token_efficiency": budget_manager.current_budget / budget_manager.max_tokens_per_evaluation
    }


def _parse_enhanced_response(content: str, max_score: int) -> Tuple[int, float, str]:
    """Parse enhanced evaluation response"""
    try:
        lines = content.strip().split('\n')
        score = 0
        confidence = 0.5
        reasoning = ""
        
        for line in lines:
            if line.strip().startswith('Score'):
                score_text = line.split(':')[1].strip() if ':' in line else line.split()[-1]
                try:
                    score = int(score_text.split()[0])
                    score = max(0, min(score, max_score))
                except (ValueError, IndexError):
                    score = 0
            
            elif line.strip().startswith('Confidence'):
                conf_text = line.split(':')[1].strip() if ':' in line else line.split()[-1]
                try:
                    confidence = float(conf_text.split()[0])
                    confidence = max(0.0, min(confidence, 1.0))
                except (ValueError, IndexError):
                    confidence = 0.5
            
            elif line.strip().startswith('Reasoning'):
                reasoning = line.split(':', 1)[1].strip() if ':' in line else ""
        
        return score, confidence, reasoning
        
    except Exception as e:
        log.warning(f"Enhanced response parsing failed: {e}")
        return 0, 0.0, ""


def _apply_quality_adjustments(
    base_score: int,
    max_score: int,
    concept_summary: str,
    quality_summary: str,
    word_count: int,
    qtype: str
) -> int:
    """Apply quality-based score adjustments"""
    adjusted_score = base_score
    
    # Word count adjustments
    if qtype == "short_answer":
        if word_count < 10:
            adjusted_score = int(adjusted_score * 0.7)  # Severe penalty
        elif word_count < 15:
            adjusted_score = int(adjusted_score * 0.9)  # Light penalty
        elif 20 <= word_count <= 50:
            adjusted_score = min(max_score, int(adjusted_score * 1.05))  # Bonus for optimal length
    
    elif qtype == "long_answer":
        if word_count < 30:
            adjusted_score = int(adjusted_score * 0.6)  # Severe penalty
        elif word_count < 50:
            adjusted_score = int(adjusted_score * 0.8)  # Penalty
        elif 100 <= word_count <= 300:
            adjusted_score = min(max_score, int(adjusted_score * 1.03))  # Bonus for optimal length
    
    # Concept coverage adjustments
    if "technical" in concept_summary or "analytical" in concept_summary:
        adjusted_score = min(max_score, int(adjusted_score * 1.02))  # Small bonus for technical content
    
    # Quality indicator bonuses
    if "struct" in quality_summary or "expl" in quality_summary:
        adjusted_score = min(max_score, int(adjusted_score * 1.01))  # Small bonus for structure/explanation
    
    return max(0, min(max_score, adjusted_score))


def _calculate_enhanced_confidence(
    score: int,
    max_score: int,
    concept_summary: str,
    quality_summary: str,
    word_count: int,
    base_confidence: float
) -> float:
    """Calculate enhanced confidence based on multiple quality indicators"""
    confidence = base_confidence
    
    # Word count confidence factor
    if word_count >= 20:  # Adequate length
        confidence += 0.1
    elif word_count < 10:  # Too short
        confidence -= 0.1
    
    # Concept coverage confidence factor
    if concept_summary != "basic":
        confidence += 0.05
    
    # Quality indicators confidence factor
    if "struct" in quality_summary or "expl" in quality_summary:
        confidence += 0.05
    
    # Edge case penalty
    if score == 0 or score == max_score:
        confidence -= 0.05
    
    return max(0.0, min(1.0, confidence))


class BatchEvaluator:
    """Process multiple questions in batches to reduce token overhead while maintaining quality"""
    
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        self.content_optimizer = ContentOptimizer()
        self.concept_analyzer = TokenEfficientConceptAnalyzer()
    
    async def evaluate_batch_with_expected_answers(
        self, 
        questions_batch: List[Dict], 
        config: EvaluatorConfig
    ) -> List[Tuple[int, float]]:
        """Evaluate a batch of questions with expected answers, maintaining quality standards"""
        
        # Prepare quality-preserving batch prompt
        batch_prompt = self._create_quality_batch_prompt(questions_batch)
        
        # Single LLM call for the entire batch
        try:
            llm_response = await invoke_llm(
                prompt=batch_prompt,
                task_type="batch_assessment_evaluation",
                agent_name="assessment_evaluator"
            )
            
            # Parse batch response with quality checks
            return self._parse_quality_batch_response(llm_response, questions_batch)
        except Exception as e:
            log.error(f"Batch evaluation failed: {e}")
            # Return zero scores for all questions in batch
            return [(0, 0.0) for _ in questions_batch]
    
    def _create_quality_batch_prompt(self, questions_batch: List[Dict]) -> str:
        """Create batch evaluation prompt that maintains all quality criteria"""
        prompt = """BATCH EVALUATION TASK
====================
Evaluate multiple questions with the SAME quality standards as individual evaluation.

CRITICAL SCORING RULES (apply to ALL questions):
- Answer must FULLY address ALL parts of the question to receive full marks
- Do NOT award points for superficial or prefix-only matches (e.g., matching the first 3 words)
- Do NOT award points for trivial affirmations (e.g., "correct", "yes", "ok") - these must score 0
- Do NOT award points for answers with < 3 words - these must score 0
- Check completeness: if question asks for examples/explanations/comparisons, ALL must be provided
- Very short answers (≤6 words) should receive maximum 30% of max score even if partially correct

PARTIAL CREDIT GUIDELINES (for short_answer and long_answer):
- FULL marks: Answer covers ≥85% of key concepts with accurate, comprehensive explanation
- HIGH partial (70-84% of max): Answer covers MOST key concepts (60-84%) but missing some important points
- MEDIUM partial (40-69% of max): Answer covers SOME key concepts (30-59%) with basic understanding
- LOW partial (20-39% of max): Answer shows MINIMAL understanding, only touches on 1-2 key points
- ZERO (0): Answer is completely wrong, irrelevant, trivial, or < 3 words

For MCQ: Score max_points if answer matches expected answer exactly (case-insensitive, letter extraction), otherwise 0.

EVALUATE EACH QUESTION INDEPENDENTLY:
"""
        
        for i, q in enumerate(questions_batch):
            qtype = q.get('qtype', q.get('type', 'unknown'))
            question_text = q.get('question_text', q.get('question', ''))
            user_answer = q.get('user_answer', q.get('answer', ''))
            expected_answer = q.get('expected_answer', '')
            max_score = q.get('max_score', 10)
            
            # Don't truncate - maintain full quality by keeping complete context
            prompt += f"\n--- QUESTION {i+1} ---\n"
            prompt += f"Type: {qtype}\n"
            prompt += f"Question: {question_text}\n"
            if expected_answer:
                prompt += f"Expected Answer: {expected_answer}\n"
            prompt += f"Candidate Response: {user_answer}\n"
            prompt += f"Maximum Points: {max_score}\n"
            prompt += f"Score (integer 0-{max_score}):\n"
        
        prompt += "\n\nIMPORTANT: Respond with ONLY the scores, one per line, in order (Question 1 score, Question 2 score, etc.).\n"
        prompt += "Format: Q1: <score>\nQ2: <score>\n...\n"
        
        return prompt
    
    def _parse_quality_batch_response(
        self, 
        response: Any, 
        questions_batch: List[Dict]
    ) -> List[Tuple[int, float]]:
        """Parse batch response with quality validation"""
        if not response:
            return [(0, 0.0) for _ in questions_batch]
            
        content = getattr(response, "content", str(response)).strip()
        batch_size = len(questions_batch)
        scores = []
        
        # Try multiple parsing strategies
        # Strategy 1: Look for "Q1:", "Q2:" pattern
        q_pattern = re.compile(r'Q(\d+):\s*(\d+)', re.IGNORECASE)
        matches = q_pattern.findall(content)
        if matches:
            # Sort by question number
            sorted_matches = sorted(matches, key=lambda x: int(x[0]))
            for qnum, score_str in sorted_matches:
                try:
                    score = int(score_str)
                    max_score = questions_batch[int(qnum) - 1].get('max_score', 10)
                    score = max(0, min(score, max_score))  # Validate range
                    confidence = _assess_confidence(score, max_score, content)
                    scores.append((score, confidence))
                except (ValueError, IndexError):
                    scores.append((0, 0.5))
        
        # Strategy 2: Look for numbered lines with scores
        if len(scores) != batch_size:
            lines = content.split('\n')
            for line in lines:
                # Look for patterns like "1. 8" or "Question 1: 8" or just numbers
                num_match = re.search(r'(\d+)', line)
                if num_match:
                    try:
                        score = int(num_match.group(1))
                        if 0 <= score <= 100:  # Reasonable score range
                            max_score = questions_batch[len(scores)].get('max_score', 10) if len(scores) < batch_size else 10
                            score = min(score, max_score)
                            confidence = _assess_confidence(score, max_score, content)
                            scores.append((score, confidence))
                            if len(scores) >= batch_size:
                                break
                    except (ValueError, IndexError):
                        continue
        
        # Strategy 3: Extract all numbers and use first N
        if len(scores) != batch_size:
            all_numbers = re.findall(r'\b(\d+)\b', content)
            for num_str in all_numbers[:batch_size]:
                try:
                    score = int(num_str)
                    if len(scores) < batch_size:
                        max_score = questions_batch[len(scores)].get('max_score', 10)
                        score = max(0, min(score, max_score))
                        confidence = _assess_confidence(score, max_score, content)
                        scores.append((score, confidence))
                except (ValueError, IndexError):
                    continue
        
        # Ensure we have the right number of scores
        while len(scores) < batch_size:
            scores.append((0, 0.5))
        
        return scores[:batch_size]
    
    async def generate_batch_rationales(
        self,
        questions_batch: List[Dict],
        config: EvaluatorConfig
    ) -> List[str]:
        """Generate rationales for a batch of questions"""
        prompt = """BATCH RATIONALE GENERATION
===========================
Generate brief explanations for each question in SECOND-PERSON (use 'Your answer', 'You selected', etc.).

CRITICAL RULES:
- Write in SECOND-PERSON - use 'Your answer', 'You selected', NOT 'Candidate Response' or 'The candidate'
- If score > 0: Start with 'Your answer is correct' or 'You selected the correct answer'
- If score = 0: Start with 'Your answer is incorrect' or 'You selected the wrong answer'
- Do NOT say the answer is correct if the score indicates it is incorrect
- Keep each explanation under 2 sentences
- Avoid markdown formatting, code blocks, and special characters

QUESTIONS:
"""
        for i, q in enumerate(questions_batch):
            prompt += f"\n--- QUESTION {i+1} ---\n"
            prompt += f"Type: {q.get('qtype', 'unknown')}\n"
            prompt += f"Question: {q.get('question_text', '')}\n"
            prompt += f"Your Answer: {q.get('user_answer', '')}\n"
            prompt += f"Score: {q.get('score', 0)}/{q.get('max_score', 10)}\n"
            prompt += f"Rationale:\n"
        
        prompt += "\n\nProvide rationales in order, one per question. Format: Q1: <rationale>\nQ2: <rationale>\n..."
        
        try:
            llm_response = await invoke_llm(
                prompt=prompt,
                task_type="batch_rationale_generation",
                agent_name="assessment_evaluator"
            )
            content = getattr(llm_response, "content", str(llm_response)).strip()
            return self._parse_batch_rationales(content, len(questions_batch))
        except Exception as e:
            log.error(f"Batch rationale generation failed: {e}")
            return ["Brief explanation unavailable." for _ in questions_batch]
    
    def _parse_batch_rationales(self, content: str, batch_size: int) -> List[str]:
        """Parse batch rationale response"""
        rationales = []
        # Try to extract Q1:, Q2: patterns
        q_pattern = re.compile(r'Q(\d+):\s*(.+?)(?=Q\d+:|$)', re.IGNORECASE | re.DOTALL)
        matches = q_pattern.findall(content)
        if matches:
            sorted_matches = sorted(matches, key=lambda x: int(x[0]))
            for qnum, rationale in sorted_matches:
                rationale = rationale.strip()
                # Clean up markdown
                rationale = re.sub(r'```[\w]*\n?', '', rationale)
                rationale = re.sub(r'`', '', rationale)
                rationale = re.sub(r'\*\*|\*|__?', '', rationale)
                rationale = _convert_to_second_person(rationale)
                rationales.append(_truncate_at_sentence_boundary(rationale, 300))
        
        # Fallback: split by question markers or use simple line breaks
        if len(rationales) != batch_size:
            # Try splitting by "--- QUESTION" markers
            parts = re.split(r'--- QUESTION \d+ ---', content)
            for part in parts[1:batch_size+1] if len(parts) > 1 else []:
                rationale = part.strip()
                rationale = re.sub(r'```[\w]*\n?', '', rationale)
                rationale = re.sub(r'`', '', rationale)
                rationale = re.sub(r'\*\*|\*|__?', '', rationale)
                rationale = _convert_to_second_person(rationale)
                rationales.append(_truncate_at_sentence_boundary(rationale, 300))
        
        while len(rationales) < batch_size:
            rationales.append("Brief explanation unavailable.")
        
        return rationales[:batch_size]
    
    async def generate_batch_correct_answers(
        self,
        questions_batch: List[Dict],
        config: EvaluatorConfig
    ) -> List[str]:
        """Generate correct answers for a batch of questions"""
        prompt = """BATCH CORRECT ANSWER GENERATION
===============================
Generate concise correct answers for each question.

RULES:
- Provide the correct answer succinctly
- If coding, provide a minimal correct snippet only
- Avoid extra commentary and markdown formatting
- Do not use code blocks (```) or special characters
- Provide plain text answer only

QUESTIONS:
"""
        for i, q in enumerate(questions_batch):
            prompt += f"\n--- QUESTION {i+1} ---\n"
            prompt += f"Type: {q.get('qtype', 'unknown')}\n"
            prompt += f"Question: {q.get('question_text', '')}\n"
            prompt += f"Correct Answer:\n"
        
        prompt += "\n\nProvide answers in order. Format: Q1: <answer>\nQ2: <answer>\n..."
        
        try:
            llm_response = await invoke_llm(
                prompt=prompt,
                task_type="batch_correct_answer_generation",
                agent_name="assessment_evaluator"
            )
            content = getattr(llm_response, "content", str(llm_response)).strip()
            return self._parse_batch_correct_answers(content, len(questions_batch))
        except Exception as e:
            log.error(f"Batch correct answer generation failed: {e}")
            return ["Correct answer unavailable." for _ in questions_batch]
    
    def _parse_batch_correct_answers(self, content: str, batch_size: int) -> List[str]:
        """Parse batch correct answer response"""
        answers = []
        # Try to extract Q1:, Q2: patterns
        q_pattern = re.compile(r'Q(\d+):\s*(.+?)(?=Q\d+:|$)', re.IGNORECASE | re.DOTALL)
        matches = q_pattern.findall(content)
        if matches:
            sorted_matches = sorted(matches, key=lambda x: int(x[0]))
            for qnum, answer in sorted_matches:
                answer = answer.strip()
                # Clean up markdown
                answer = re.sub(r'```[\w]*\n?', '', answer)
                answer = re.sub(r'`', '', answer)
                answer = re.sub(r'\*\*|\*|__?', '', answer)
                answers.append(_truncate_at_sentence_boundary(answer, 600))
        
        # Fallback: split by question markers
        if len(answers) != batch_size:
            parts = re.split(r'--- QUESTION \d+ ---', content)
            for part in parts[1:batch_size+1] if len(parts) > 1 else []:
                answer = part.strip()
                answer = re.sub(r'```[\w]*\n?', '', answer)
                answer = re.sub(r'`', '', answer)
                answer = re.sub(r'\*\*|\*|__?', '', answer)
                answers.append(_truncate_at_sentence_boundary(answer, 600))
        
        while len(answers) < batch_size:
            answers.append("Correct answer unavailable.")
        
        return answers[:batch_size]


def _group_questions_for_batching(
    qpayloads: List[Tuple[str, str, str, str]],
    qmeta: List[Tuple[str, str, int]],
    batch_size: int = 5
) -> List[List[Dict[str, Any]]]:
    """Group questions intelligently for batching by type and max_score"""
    # Create question dicts with all needed info
    questions = []
    for (qid, question_text, user_answer, expected_answer), (qid2, qtype, max_score) in zip(qpayloads, qmeta):
        if qid == qid2:
            questions.append({
                'qid': qid,
                'qtype': qtype,
                'question_text': question_text,
                'user_answer': user_answer,
                'expected_answer': expected_answer,
                'max_score': max_score
            })
    
    # Group by (qtype, max_score) for better batching quality
    groups = {}
    for q in questions:
        key = (q['qtype'], q['max_score'])
        if key not in groups:
            groups[key] = []
        groups[key].append(q)
    
    # Create batches, trying to keep similar questions together
    batches = []
    for key, group_questions in groups.items():
        # Split large groups into batches
        for i in range(0, len(group_questions), batch_size):
            batches.append(group_questions[i:i + batch_size])
    
    # If we have small batches, try to combine them (but keep similar types together)
    if len(batches) > 1:
        optimized_batches = []
        current_batch = []
        for batch in batches:
            if len(current_batch) + len(batch) <= batch_size:
                current_batch.extend(batch)
            else:
                if current_batch:
                    optimized_batches.append(current_batch)
                current_batch = batch
        if current_batch:
            optimized_batches.append(current_batch)
        batches = optimized_batches
    
    return batches


def _validate_submission_data(state: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Validate and extract submission data from state.

    Args:
        state: Input state dictionary

    Returns:
        Validated submission data

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(state, dict):
        raise ValidationError("Invalid state format provided")

    submission: Optional[Dict[str, List[Dict[str, Any]]]] = (
        state.get("submission") or state.get("body", {}).get("submission")
    )

    if not submission or not isinstance(submission, dict):
        raise ValidationError("No valid submission found in state")

    return submission

def _prepare_evaluation_tasks(
    submission: Dict[str, List[Dict[str, Any]]],
    strategy: str,
    config: EvaluatorConfig
) -> Tuple[List, List, int, List[Tuple[str, str, str, str]]]:
    """
    Prepare evaluation tasks from submission data.
    
    COMPLETE FIX: Robust extraction of all required fields
    """
    tasks, qmeta = [], []
    task_count = 0
    qpayloads: List[Tuple[str, str, str, str]] = []

    log.info(f"📋 Preparing tasks from submission with {len(submission)} question types")

    for qtype, questions in submission.items():
        if not isinstance(questions, list):
            log.warning(f"Invalid questions format for type {qtype}")
            continue

        log.info(f"Processing {len(questions)} questions of type '{qtype}'")

        for idx, q in enumerate(questions):
            if not isinstance(q, dict):
                log.warning(f"Question {idx} in {qtype} is not a dict, skipping")
                continue

            if task_count >= config.max_concurrent_tasks:
                log.warning(f"Maximum task limit ({config.max_concurrent_tasks}) reached")
                break

            # Get question type
            individual_qtype = q.get("question_type", qtype)
            validated_qtype = _validate_question_type(individual_qtype)
            qid = f"{validated_qtype}_{idx}"

            # Validate and sanitize inputs
            max_score = _validate_max_score(q.get("max_score", DEFAULT_MAX_SCORE))
            question_text = _sanitize_input(str(q.get("question_text", "")), config=config)
            
            # CRITICAL FIX: Comprehensive user_answer extraction
            user_answer_raw = (
                q.get("user_answer") or 
                q.get("selected_option") or 
                q.get("answer") or 
                q.get("response") or 
                q.get("candidate_answer") or
                q.get("selected") or
                q.get("user_response") or
                ""
            )
            
            # Handle the case where user_answer might be a dict (for MCQ with options)
            if isinstance(user_answer_raw, dict):
                user_answer_raw = user_answer_raw.get("option") or user_answer_raw.get("key") or ""
            
            user_answer = _sanitize_input(str(user_answer_raw), config=config)
            
            # CRITICAL FIX: Extract expected_answer
            expected_answer_raw = q.get("expected_answer", "")
            
            # Handle dict expected_answer
            if isinstance(expected_answer_raw, dict):
                expected_answer_raw = expected_answer_raw.get("option") or expected_answer_raw.get("key") or ""
            
            expected_answer = str(expected_answer_raw).strip() if expected_answer_raw else ""

            # Debug logging
            log.debug(f"Question {qid}:")
            log.debug(f"  - question_text: {question_text[:50]}...")
            log.debug(f"  - user_answer_raw: '{user_answer_raw}'")
            log.debug(f"  - user_answer: '{user_answer}'")
            log.debug(f"  - expected_answer: '{expected_answer}'")
            log.debug(f"  - max_score: {max_score}")

            # Validation: Skip only if question_text is empty
            if not question_text:
                log.warning(f"Skipping question {qid}: empty question_text")
                continue
            
            # Allow empty user_answer (will be scored as 0)
            if not user_answer:
                log.warning(f"Question {qid} has empty user_answer - will score as 0")
                user_answer = ""  # Ensure it's empty string, not None

            qmeta.append((qid, validated_qtype, max_score))
            tasks.append(
                _evaluate_question_with_llm(
                    question_text,
                    user_answer,
                    validated_qtype,
                    max_score,
                    strategy,
                    config,
                    expected_answer  # Pass expected_answer
                )
            )
            task_count += 1
            qpayloads.append((qid, question_text, user_answer, expected_answer))

    log.info(f"✓ Prepared {task_count} evaluation tasks")
    return tasks, qmeta, task_count, qpayloads

def _truncate_at_sentence_boundary(text: str, max_length: int) -> str:
    """Truncate text at sentence boundary to avoid breaking mid-sentence.
    
    Args:
        text: Text to truncate
        max_length: Maximum allowed length
        
    Returns:
        Truncated text ending at a complete sentence
    """
    if len(text) <= max_length:
        return text
    
    # Find the last sentence-ending punctuation before max_length
    truncated = text[:max_length]
    
    # Look for sentence endings: period, question mark, exclamation mark
    sentence_endings = ['.', '!', '?']
    last_sentence_end = -1
    
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > last_sentence_end:
            last_sentence_end = pos
    
    # If we found a sentence boundary, truncate there
    if last_sentence_end > 0:
        return text[:last_sentence_end + 1].strip()
    
    # If no sentence boundary found, look for comma or space as fallback
    last_comma = truncated.rfind(',')
    last_space = truncated.rfind(' ')
    
    fallback_pos = max(last_comma, last_space)
    if fallback_pos > max_length * 0.7:  # Only use if we're keeping at least 70% of content
        return text[:fallback_pos].strip() + "..."
    
    # Last resort: hard truncate but indicate it's incomplete
    return truncated.strip() + "..."


def _convert_to_second_person(text: str) -> str:
    """Convert third-person phrases to second-person for better user experience.
    
    Converts phrases like "Candidate Response B is correct" to "Your answer is correct"
    or "You selected the correct answer".
    """
    if not text:
        return text
    
    # Common third-person patterns to convert
    replacements = [
        # "Candidate Response X is correct" -> "Your answer is correct" or "You selected the correct answer"
        (r'\bCandidate Response ([A-Z])\s+is\s+correct\b', r'Your answer is correct'),
        (r'\bThe candidate response\s+is\s+correct\b', r'Your answer is correct'),
        (r'\bCandidate Response ([A-Z])\s+is\s+incorrect\b', r'Your answer is incorrect'),
        (r'\bThe candidate response\s+is\s+incorrect\b', r'Your answer is incorrect'),
        # "The candidate selected..." -> "You selected..."
        (r'\bThe candidate\s+selected\b', r'You selected'),
        (r'\bThe candidate\s+answered\b', r'You answered'),
        # "Candidate Response" at start -> "Your answer"
        (r'^\s*Candidate Response\s+([A-Z])\s*[:\.,]?\s*', r'Your answer '),
        # "The candidate's response" -> "Your answer"
        (r'\bThe candidate\'?s\s+response\b', r'Your answer'),
        (r'\bThe candidate\'?s\s+answer\b', r'Your answer'),
    ]
    
    result = text
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result


def _is_trivial_affirmation(answer: str) -> bool:
    """Detect single-word or trivial affirmations that should not earn credit for short/long answers."""
    if not isinstance(answer, str):
        return False
    normalized = answer.strip().lower()
    if not normalized:
        return False
    # Single token or extremely short generic acknowledgements
    trivial_set = {
        "correct", "yes", "yeah", "yep", "yup", "ok", "okay", "k", "true", "false",
        "agree", "right", "sure", "indeed", "same", "as above", "good", "thanks",
        "noted", "confirmed", "affirmative", "negative"
    }
    # Consider also punctuation-only or emoji-like acknowledgements
    punctuation_only = set(".,!~")
    if all(ch in punctuation_only for ch in normalized):
        return True
    # One or two-word trivial responses
    if normalized in trivial_set:
        return True
    if len(normalized.split()) <= 2 and normalized.endswith(" correct"):
        return True
    return False


def _keyword_coverage(expected_text: str, answer_text: str) -> float:
    """Compute a lightweight keyword coverage ratio between expected answer and user answer.
    Returns a value in [0,1]. Uses simple tokenization and a tiny stopword list."""
    try:
        import re as _re
        stop = {
            "the","a","an","and","or","of","to","in","on","for","with","is","are","was","were",
            "that","this","these","those","at","by","be","as","it","its","from","into","about",
            "than","then","so","such","their","there","which","who","whom","what","when","where",
            "how","why","if","but","also","can","may","might","should","would","could","etc",
        }
        def tokens(s: str) -> set[str]:
            words = [w for w in _re.split(r"[^a-z0-9]+", (s or "").lower()) if w]
            return {w for w in words if w not in stop and len(w) > 2}
        exp = tokens(expected_text)
        ans = tokens(answer_text)
        if not exp:
            return 0.0
        overlap = len(exp & ans)
        return overlap / max(1, len(exp))
    except Exception:
        return 0.0


@traceable(name="generate_rationale_with_llm")
async def _generate_rationale_with_llm(
    question_text: str,
    user_answer: str,
    qtype: str,
    config: EvaluatorConfig,
    score: int = None,
    max_score: int = None
) -> str:
    """Generate a brief explanation of the candidate's answer.

    If the answer is correct, affirm and highlight key points succinctly.
    If incorrect, explain what is missing or wrong and what is expected.
    Returns a short response (max ~300 chars)."""
    try:
        # Guard: if no answer provided, return a deterministic rationale
        if not str(user_answer or "").strip():
            return "Your answer is incorrect. No response was provided; please attempt an answer."
        
        # Determine if answer is correct based on score
        is_correct = False
        if score is not None and max_score is not None:
            is_correct = score > 0
        elif score is not None:
            is_correct = score > 0
        
        correctness_context = ""
        if score is not None and max_score is not None:
            correctness_context = f"\nScore: {score}/{max_score} ({'CORRECT' if is_correct else 'INCORRECT'})"
        elif score is not None:
            correctness_context = f"\nScore: {score} ({'CORRECT' if is_correct else 'INCORRECT'})"
        
        # Build instructions based on correctness
        if is_correct:
            correctness_instruction = (
                "CRITICAL: The answer is CORRECT (score > 0). Start with 'Your answer is correct' or 'You selected the correct answer'.\n"
                "- Affirm correctness and highlight the key idea succinctly.\n"
            )
        else:
            correctness_instruction = (
                "CRITICAL: The answer is INCORRECT (score = 0). DO NOT start with 'Correct' or 'Correct.'\n"
                "- Start with 'Your answer is incorrect' or 'You selected the wrong answer'.\n"
                "- Explain what is wrong or missing and provide the key idea.\n"
                "- For MCQ questions: Explain why the selected option is wrong and what the correct answer is.\n"
            )
        
        prompt = (
            "Rationale Task\n"
            "================\n"
            f"Type: {qtype}\n"
            f"Question: {question_text}\n"
            f"Your Answer: {user_answer}{correctness_context}\n\n"
            "Instructions:\n"
            "- Write the explanation in SECOND-PERSON (use 'Your answer', 'You selected', etc.) - NOT third-person\n"
            f"{correctness_instruction}"
            "- Do NOT use phrases like 'Candidate Response' or 'The candidate' - use 'You' or 'Your answer' instead\n"
            "- Do NOT say the answer is correct if the score indicates it is incorrect\n"
            "- Do NOT use phrases like 'The response correctly' or 'The response emphasizes' when the answer is incorrect\n"
            "- Do NOT start with just 'Correct.' or 'Correct' - always use full phrases like 'Your answer is correct' or 'Your answer is incorrect'\n"
            "- Do NOT reveal sensitive data.\n"
            "- Keep it under 2 sentences.\n"
            "- Avoid markdown formatting, code blocks, and special characters.\n"
            "- Provide plain text explanation only.\n\n"
            "Rationale:"
        )
        llm_response = await invoke_llm(
            prompt=prompt,
            task_type="assessment_evaluation",
            agent_name="assessment_evaluator"
        )
        content = getattr(llm_response, "content", str(llm_response)).strip()
        if not content:
            return "Brief explanation unavailable."

        # Remove any markdown or special formatting
        content = re.sub(r'```[\w]*\n?', '', content)  # Remove code blocks
        content = re.sub(r'`', '', content)  # Remove backticks
        content = re.sub(r'\*\*|\*|__?', '', content)  # Remove bold/italic markers
        content = content.strip()
        
        # Convert third-person to second-person for better user experience
        content = _convert_to_second_person(content)
        
        # Sanitize and trim
        # return content[:300]
        # Sanitize and trim at sentence boundary
        return _truncate_at_sentence_boundary(content, 300)
    except Exception:
        return "Brief explanation unavailable."


@traceable(name="generate_correct_answer_with_llm")
async def _generate_correct_answer_with_llm(
    question_text: str,
    qtype: str,
    config: EvaluatorConfig
) -> str:
    """Generate a concise correct answer for the given question.

    For coding questions, return a minimal working snippet. For MCQ/short answers, return a brief statement.
    """
    try:
        prompt = (
            "Answer Synthesis Task\n"
            "=====================\n"
            f"Type: {qtype}\n"
            f"Question: {question_text}\n\n"
            "Instructions:\n"
            "- Provide the correct answer succinctly.\n"
            "- If coding, provide a minimal correct snippet only.\n"
            "- Avoid extra commentary and markdown formatting.\n"
            "- Do not use code blocks (```) or special characters.\n"
            "- Provide plain text answer only.\n\n"
            "Correct Answer:"
        )
        llm_response = await invoke_llm(
            prompt=prompt,
            task_type="assessment_evaluation",
            agent_name="assessment_evaluator"
        )
        content = getattr(llm_response, "content", str(llm_response)).strip()
        if not content:
            return "Correct answer unavailable."

        # Remove any remaining markdown or special formatting
        content = re.sub(r'```[\w]*\n?', '', content)  
        content = re.sub(r'`', '', content)  
        content = re.sub(r'\*\*|\*|__?', '', content)  
        content = content.strip()
        # return content[:600]
        # Sanitize and trim at sentence boundary
        return _truncate_at_sentence_boundary(content, 600)
    except Exception:
        return "Correct answer unavailable."


@traceable(name="assessment_evaluator_agent")
async def assessment_evaluator_agent(state: Dict[str, Any], config: EvaluatorConfig = None) -> Dict[str, Any]:
    """
    Enhanced assessment evaluator with centralized utilities and LLM-only approach.

    Args:
        state: Dictionary containing assessment data including submission
        config: Configuration instance (creates default if not provided)

    Returns:
        Dictionary with evaluation results normalized to 100-point scale
    """
    # Use centralized logging
    log_context = create_log_context("assessment_evaluator", state.get("tenant_id", "default"))
    start_time = log_context["start_time"]
    request_id = log_context["request_id"]
    
    # Test DEBUG logging
    log.debug("🔍 ASSESSMENT_EVALUATOR_DEBUG: Starting assessment evaluation")
    log.debug(f"🔍 ASSESSMENT_EVALUATOR_DEBUG: State keys: {list(state.keys())}")
    log.debug(f"🔍 ASSESSMENT_EVALUATOR_DEBUG: Request ID: {request_id}")
    
    log.info("--- Entering Assessment Evaluator Agent ---")
    
    # Get tenant-scoped memory
    assessment_memory = await get_assessment_evaluator_memory(state.get("tenant_id", "default"))
    
    # **FIX:** Ensure config is an EvaluatorConfig object
    if not isinstance(config, EvaluatorConfig):
        config = EvaluatorConfig.from_env()
    log.debug(f"EvaluatorConfig: timeout={config.evaluation_timeout}, max_concurrent={config.max_concurrent_tasks}")

    # Agentic AI: Initialize context for decision making
    context = EvaluationContext()

    try:
        # Step 1: Validate input data
        try:
            submission = _validate_submission_data(state)
        except ValidationError:
            processing_time = _calculate_processing_time(start_time)
            AgentLogger.log_warning(log_context, "No valid submission found in state")
            return _create_error_response("Submission missing or invalid", processing_time)
        # DEBUG: submission breakdown
        try:
            debug_counts = {k: len(v) for k, v in submission.items() if isinstance(v, list)}
            log.debug(f"Submission received | counts by type: {debug_counts}")
        except Exception:
            pass

        # Step 1.5: Check if this is a generic test (psychometric, personality, communication)
        # Try assessment_topic first, then fallback to submission.test_type
        assessment_topic = state.get("assessment_topic") or submission.get("test_type") or ""
        generic_test_type = generic_test_evaluator.identify_generic_test_type(assessment_topic)
        
        if generic_test_type:
            log.info(f"Detected generic test type: {generic_test_type.value}")
            # Use generic test evaluation instead of standard evaluation
            return await _evaluate_generic_test(state, submission, generic_test_type, config, log_context)

        # Granular timing breakdown for performance monitoring
        timing_breakdown = {
            "strategy_planning": 0.0,
            "task_preparation": 0.0,
            "evaluation_execution": 0.0,
            "result_processing": 0.0,
            "rationale_generation": 0.0,
            "total": 0.0
        }
        
        # Step 2: Collect all questions for strategic planning
        all_questions = []
        for qtype, questions in submission.items():
            if isinstance(questions, list):
                for q in questions:
                    if isinstance(q, dict):
                        all_questions.append({**q, "qtype": qtype})

        # Step 3: Plan evaluation strategy based on question complexity
        strategy_start = time.time()
        strategy = await _plan_evaluation_strategy(all_questions, context)
        timing_breakdown["strategy_planning"] = time.time() - strategy_start
        log.info(f"Selected evaluation strategy: {strategy} for {len(all_questions)} questions")

        # Step 4: Prepare evaluation tasks (offloaded — heavy regex sanitization)
        prep_start = time.time()
        tasks, qmeta, task_count, qpayloads = await asyncio.to_thread(
            _prepare_evaluation_tasks, submission, strategy, config
        )
        timing_breakdown["task_preparation"] = time.time() - prep_start
        log.debug(f"Prepared tasks: {task_count} | qmeta={len(qmeta)} | payloads={len(qpayloads)}")

        if not tasks:
            log.warning("No valid questions found for evaluation")
            return _create_empty_result(state, config)

        # Step 5: Execute evaluations with intelligent batching
        eval_start = time.time()
        # Initialize batch evaluator for potential use in evaluation and rationale generation
        batch_evaluator = BatchEvaluator(batch_size=5)
        try:
            # Use batching for better performance while maintaining quality
            use_batching = len(tasks) >= 5  # Only batch if we have 5+ questions
            
            if use_batching:
                log.info(f"Using batch evaluation for {len(tasks)} questions")
                # Group questions for batching
                question_batches = _group_questions_for_batching(qpayloads, qmeta, batch_size=5)
                
                # Evaluate batches concurrently
                batch_tasks = []
                batch_indices = []  # Track which questions belong to which batch
                question_idx = 0
                
                for batch in question_batches:
                    batch_data = []
                    batch_qids = []
                    for q in batch:
                        batch_data.append({
                            'qtype': q['qtype'],
                            'question_text': q['question_text'],
                            'user_answer': q['user_answer'],
                            'expected_answer': q['expected_answer'],
                            'max_score': q['max_score']
                        })
                        batch_qids.append(q['qid'])
                    
                    batch_indices.append((batch_qids, question_idx, question_idx + len(batch)))
                    question_idx += len(batch)
                    
                    # Create batch evaluation task
                    batch_tasks.append(
                        batch_evaluator.evaluate_batch_with_expected_answers(batch_data, config)
                    )
                
                # Execute all batches concurrently
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*batch_tasks, return_exceptions=True),
                    timeout=config.evaluation_timeout
                )
                
                # Flatten batch results back to individual question results
                results = []
                for (batch_qids, start_idx, end_idx), batch_result in zip(batch_indices, batch_results):
                    if isinstance(batch_result, Exception):
                        # If batch failed, fall back to individual evaluation for that batch
                        log.warning(f"Batch evaluation failed, falling back to individual calls: {batch_result}")
                        fallback_tasks = []
                        for qid in batch_qids:
                            # Find the original task for this question
                            qidx = next(i for i, (qid2, _, _) in enumerate(qmeta) if qid2 == qid)
                            fallback_tasks.append(tasks[qidx])
                        fallback_results = await asyncio.gather(*fallback_tasks, return_exceptions=True)
                        results.extend(fallback_results)
                    else:
                        results.extend(batch_result)
                
                # Ensure we have results for all questions
                while len(results) < len(tasks):
                    results.append((0, 0.0))
                results = results[:len(tasks)]
            else:
                # For small question sets, use individual concurrent calls
                log.info(f"Using individual concurrent evaluation for {len(tasks)} questions")
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=config.evaluation_timeout
                )
            
            timing_breakdown["evaluation_execution"] = time.time() - eval_start
        except asyncio.TimeoutError:
            timing_breakdown["evaluation_execution"] = time.time() - eval_start
            processing_time = _calculate_processing_time(start_time)
            AgentLogger.log_error(log_context, f"Evaluation timeout exceeded ({config.evaluation_timeout}s)", processing_time)
            return _create_error_response("Evaluation timeout", processing_time)

        # Step 6: Process results with performance tracking
        proc_start = time.time()
        question_scores, section_scores = {}, {}
        question_confidences = {}
        question_rationales: Dict[str, str] = {}
        section_max_scores: Dict[str, int] = {}
        total_score, total_max_score = 0, 0
        successful_evaluations = 0

        log.debug(f"Collected {len(results)} evaluation results for {len(qmeta)} questions")
        for (qid, qtype, max_score), result in zip(qmeta, results):
            if isinstance(result, Exception):
                log.error(f"Failed to evaluate question {qid}: {type(result).__name__}")
                score, confidence = 0, 0.0
                await context.update_performance(False, 0.0, qtype)
            else:
                score, confidence = result
                successful_evaluations += 1
                processing_time = (time.time() - start_time) / max(1, len(tasks))
                await context.update_performance(True, processing_time, qtype)
                log.debug(f"Scored {qid} ({qtype}) -> score={score}, conf={confidence:.2f}")

            question_scores[qid] = score
            question_confidences[qid] = confidence
            section_scores[qtype] = section_scores.get(qtype, 0) + score
            total_score += score
            total_max_score += max_score
            section_max_scores[qtype] = section_max_scores.get(qtype, 0) + max_score
        timing_breakdown["result_processing"] = time.time() - proc_start

        # Step 6b: Generate explanations for all responses, and correct answers for incorrect ones
        rationale_start = time.time()
        try:
            types_by_id = {qid: qtype for (qid, qtype, _maxs) in qmeta}
            
            # Use batching for rationale and correct answer generation if we have enough questions
            use_batching_rationale = len(qpayloads) >= 5
            
            if use_batching_rationale:
                log.info(f"Using batch rationale/correct answer generation for {len(qpayloads)} questions")
                # Prepare batch data
                rationale_batches = []
                correct_answer_batches = []
                batch_qids = []
                
                # Group questions for batching
                question_batches = _group_questions_for_batching(qpayloads, qmeta, batch_size=5)
                
                for batch in question_batches:
                    rationale_batch_data = []
                    correct_answer_batch_data = []
                    batch_qid_list = []
                    
                    for q in batch:
                        qid = q['qid']
                        qtype = types_by_id.get(qid, "unknown")
                        score = question_scores.get(qid, 0)
                        max_score = q['max_score']
                        
                        # Prepare rationale batch data
                        rationale_batch_data.append({
                            'qtype': qtype,
                            'question_text': q['question_text'],
                            'user_answer': q['user_answer'],
                            'score': score,
                            'max_score': max_score
                        })
                        
                        # Prepare correct answer batch data
                        correct_answer_batch_data.append({
                            'qtype': qtype,
                            'question_text': q['question_text']
                        })
                        
                        batch_qid_list.append(qid)
                    
                    rationale_batches.append((batch_qid_list, rationale_batch_data))
                    correct_answer_batches.append((batch_qid_list, correct_answer_batch_data))
                
                # Execute batch rationale generation
                rationale_batch_tasks = [
                    batch_evaluator.generate_batch_rationales(batch_data, config)
                    for _, batch_data in rationale_batches
                ]
                correct_answer_batch_tasks = [
                    batch_evaluator.generate_batch_correct_answers(batch_data, config)
                    for _, batch_data in correct_answer_batches
                ]
                
                rationale_batch_results, correct_answer_batch_results = await asyncio.gather(
                    asyncio.gather(*rationale_batch_tasks, return_exceptions=True),
                    asyncio.gather(*correct_answer_batch_tasks, return_exceptions=True)
                )
                
                # Flatten batch results
                for (batch_qids, _), rationale_batch_result in zip(rationale_batches, rationale_batch_results):
                    if isinstance(rationale_batch_result, Exception):
                        # Fallback to individual calls if batch failed
                        log.warning(f"Batch rationale generation failed, using fallback: {rationale_batch_result}")
                        for qid in batch_qids:
                            qtype = types_by_id.get(qid, "unknown")
                            score = question_scores.get(qid, 0)
                            max_score = next((ms for (qid2, _, ms) in qmeta if qid2 == qid), 10)
                            qdata = next((q for (qid2, _, _, _) in qpayloads if qid2 == qid), None)
                            if qdata:
                                _, question_text, user_answer, _ = qdata
                                try:
                                    rationale = await _generate_rationale_with_llm(
                                        question_text, user_answer, qtype, config, score=score, max_score=max_score
                                    )
                                    question_rationales[qid] = rationale
                                except Exception:
                                    question_rationales[qid] = "Rationale unavailable due to an internal error."
                    else:
                        for qid, rationale in zip(batch_qids, rationale_batch_result):
                            question_rationales[qid] = rationale
                
                # Flatten correct answer batch results
                for (batch_qids, _), correct_answer_batch_result in zip(correct_answer_batches, correct_answer_batch_results):
                    if isinstance(correct_answer_batch_result, Exception):
                        # Fallback to individual calls if batch failed
                        log.warning(f"Batch correct answer generation failed, using fallback: {correct_answer_batch_result}")
                        for qid in batch_qids:
                            qtype = types_by_id.get(qid, "unknown")
                            qdata = next((q for (qid2, _, _, _) in qpayloads if qid2 == qid), None)
                            if qdata:
                                _, question_text, _, _ = qdata
                                try:
                                    correct_answer = await _generate_correct_answer_with_llm(
                                        question_text, qtype, config
                                    )
                                    question_rationales[f"{qid}__correct"] = correct_answer
                                except Exception:
                                    question_rationales[f"{qid}__correct"] = "Correct answer unavailable."
                    else:
                        for qid, correct_answer in zip(batch_qids, correct_answer_batch_result):
                            question_rationales[f"{qid}__correct"] = correct_answer
            else:
                # Use individual calls for small question sets
                log.info(f"Using individual rationale/correct answer generation for {len(qpayloads)} questions")
                rationale_tasks = []
                rationale_ids = []
                correct_answer_tasks = []
                for (qid, question_text, user_answer, expected_answer) in qpayloads:
                    qtype = types_by_id.get(qid, "unknown")
                    score = question_scores.get(qid, 0)
                    max_score = next((ms for (qid2, _, ms) in qmeta if qid2 == qid), None)
                    rationale_tasks.append(_generate_rationale_with_llm(question_text, user_answer, qtype, config, score=score, max_score=max_score))
                    rationale_ids.append(qid)
                    correct_answer_tasks.append(_generate_correct_answer_with_llm(question_text, qtype, config))
                
                if rationale_tasks:
                    gather_correct = asyncio.gather(*correct_answer_tasks, return_exceptions=True) if correct_answer_tasks else asyncio.sleep(0)
                    rationale_results, correct_answer_results = await asyncio.gather(
                        asyncio.gather(*rationale_tasks, return_exceptions=True),
                        gather_correct
                    )
                    for idx, qid in enumerate(rationale_ids):
                        r = rationale_results[idx]
                        if isinstance(r, Exception):
                            question_rationales[qid] = "Rationale unavailable due to an internal error."
                        else:
                            question_rationales[qid] = str(r)
                    if correct_answer_tasks:
                        for idx, qid in enumerate(rationale_ids):
                            if idx < len(correct_answer_results):
                                ca = correct_answer_results[idx] if isinstance(correct_answer_results, list) else "Correct answer unavailable."
                                if isinstance(ca, Exception):
                                    ca_text = "Correct answer unavailable."
                                else:
                                    ca_text = str(ca)
                                question_rationales[f"{qid}__correct"] = ca_text
            
            timing_breakdown["rationale_generation"] = time.time() - rationale_start
            log.debug(f"Rationales generated: {len(question_rationales)}")
        except Exception as e:
            log.exception(f"Post-processing (rationales/correct answers) failed: {type(e).__name__}")

        # Step 7: Self-reflection on overall performance
        overall_confidence = sum(question_confidences.values()) / len(question_confidences) if question_confidences else 0.0
        success_rate = successful_evaluations / len(tasks) if tasks else 0.0

        log.info(f"Evaluation completed: {successful_evaluations}/{len(tasks)} successful, "
                   f"overall confidence: {overall_confidence:.2f}, strategy: {strategy}")

        # Step 8: Normalize scores and create final result
        log.debug(f"Assembling result: total_score={total_score}, total_max_score={total_max_score}, questions={len(question_scores)}")
        
        question_max_scores = {qid: max_score for (qid, _, max_score) in qmeta}
        result = await asyncio.to_thread(
            lambda: _normalize_and_format_results(
                state, question_scores, section_scores, total_score, total_max_score, config,
                question_rationales=question_rationales,
                per_question_payloads=qpayloads,
                section_max_scores=section_max_scores,
                question_max_scores=question_max_scores,
            )
        )
        log.debug(f"Result ready | total_score_100={result.get('assessment_results', {}).get('total_score')} | sections={len(result.get('assessment_results', {}).get('section_scores', {}))}")

        # Meta information removed from callback output
        timing_breakdown["total"] = time.time() - start_time
        
        # Log timing breakdown
        log.info(
            f"⏱️ ASSESSMENT_EVALUATOR timing breakdown: "
            f"strategy={timing_breakdown['strategy_planning']*1000:.1f}ms, "
            f"prep={timing_breakdown['task_preparation']*1000:.1f}ms, "
            f"evaluation={timing_breakdown['evaluation_execution']*1000:.1f}ms, "
            f"result_proc={timing_breakdown['result_processing']*1000:.1f}ms, "
            f"rationale={timing_breakdown['rationale_generation']*1000:.1f}ms, "
            f"total={timing_breakdown['total']*1000:.1f}ms"
        )

        # Session Management Integration - MOVED TO BACKGROUND (non-blocking)
        uid = state.get("uid")
        session_id = state.get("session_id")
        if uid and session_id:
            # Fire-and-forget: Don't block response on session updates
            async def _update_session_background():
                try:
                    session_start = time.time()
                    log.info(f"🔄 ASSESSMENT_EVALUATOR: Starting background session storage for UID={uid}")
                    
                    # Update session step
                    session_update_result = await run_blocking_io(
                        session_manager.update_step,
                        session_id=session_id,
                        step="assessment_evaluator",
                        data={
                            "processing_time": round(timing_breakdown["total"], 2),
                            "method": "agentic_evaluation",
                            "strategy": strategy,
                            "max_score": result["assessment_results"].get("max_score")
                        },
                        progress=0.9  # 90% complete after assessment evaluation
                    )
                    
                    # Store assessment evaluation data in existing session
                    session_data = await run_blocking_io(get_chat_session, session_id)
                    if session_data:
                        session_data["assessment_evaluator"] = {
                            "assessment_results": result["assessment_results"],
                            "processing_time": round(timing_breakdown["total"], 2),
                            "timing_breakdown": timing_breakdown,
                            "method": "agentic_evaluation",
                            "strategy": strategy,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        await run_blocking_io(
                            update_chat_session,
                            session_id=session_id,
                            session_data=session_data,
                            metadata={
                                "agent": "assessment_evaluator",
                                "uid": uid,
                                "status": "assessment_evaluator_complete",
                                "method": "agentic_evaluation"
                            }
                        )
                        session_time = time.time() - session_start
                        log.info(f"✅ ASSESSMENT_EVALUATOR: Background session update completed in {session_time*1000:.1f}ms for UID={uid}")
                    else:
                        log.warning(f"⚠️ ASSESSMENT_EVALUATOR: No existing chat session data found for session_id={session_id}")
                except Exception as e:
                    log.error(f"❌ ASSESSMENT_EVALUATOR: Background session update failed for UID={uid}: {e}", exc_info=True)
            
            # Fire-and-forget: Start background task without awaiting
            asyncio.create_task(_update_session_background())
            log.info(f"🚀 ASSESSMENT_EVALUATOR: Started background session update task for UID={uid} (non-blocking)")

        # Record successful evaluation
        processing_time = _calculate_processing_time(start_time)
        await assessment_memory.record_attempt(
            'assessment_evaluation', 'llm', True, 0.8, processing_time
        )
        
        # Log success (use computed result value instead of undefined local)
        log_agent_completion(log_context, {
            "success": True,
            "total_score": int(result.get("assessment_results", {}).get("total_score", 0)),
            "questions_evaluated": len(question_scores)
        }, "llm", processing_time)

        # Debug: Log assessment results being returned in state
        log.info(f"🔍 ASSESSMENT_EVALUATOR: Returning assessment results in state: {result.get('assessment_results', {})}")
        log.info(f"✅ Assessment evaluator returning assessment results: total_score={result.get('assessment_results', {}).get('total_score', 'N/A')}")
        
        # Update assessment status in dedicated user_assessments store (normalized status, completion + score)
        uid = state.get("uid")
        submission = state.get("submission") or {}
        assessment_topic = (
            state.get("assessment_topic")
            or submission.get("test_type")
            or (result.get("assessment_topic") or "").strip()
            or "General Assessment"
        )
        assessment_id = (state.get("assessment_id") or result.get("assessment_id") or "").strip() or None
        total_score = result.get("assessment_results", {}).get("total_score", 0)
        # Pass score even when 0 so completion is stored correctly (score is not None required for history)
        score_value = float(total_score) if total_score is not None else None

        if uid and assessment_topic:
            try:
                # Import here to avoid circular dependency
                from agents.career_coach.function_tools import update_assessment_status
                from datetime import datetime

                await update_assessment_status(
                    uid=uid,
                    assessment_topic=assessment_topic,
                    status="completed",
                    assessment_id=assessment_id,
                    score=score_value,
                    completed_at=datetime.utcnow().isoformat(),
                )
                log.info(f"✅ Updated assessment status: {assessment_topic} -> completed (score: {total_score})")
            except Exception as e_status:
                log.warning(f"⚠️ Failed to update assessment status: {e_status}")
        
        # Add token efficiency report to results
        efficiency_report = token_monitor.get_efficiency_report()
        result["token_efficiency"] = efficiency_report
        log.info(f"Token efficiency: {efficiency_report['efficiency_score']:.1f}% (avg {efficiency_report['avg_tokens_per_evaluation']:.1f} tokens/eval)")
        
        return result

    except ValidationError as e:
        log.warning(f"Validation error in assessment evaluation: {str(e)}")
        return _create_error_result(state, f"Validation error: {str(e)}", config)
    except ProcessingError as e:
        log.error(f"Processing error in assessment evaluation: {str(e)}")
        return _create_error_result(state, "Processing error occurred", config)
    except asyncio.TimeoutError:
        log.error("Assessment evaluation timeout")
        return _create_error_result(state, "Evaluation timeout", config)
    except Exception as e:
        # Security: log full exception but keep sanitized output
        log.exception(f"Unexpected error in assessment evaluation: {type(e).__name__}")
        return _create_error_result(state, f"Internal processing error ({type(e).__name__})", config)


def _create_empty_result(state: Dict[str, Any], config: EvaluatorConfig) -> Dict[str, Any]:
    """Create result structure for empty submissions with sanitized output."""
    return {
        "assessment_id": _sanitize_input(str(state.get("assessment_id", "")), config=config),
        "assessment_topic": _sanitize_input(str(state.get("assessment_topic", "")), config=config),
        "question_doc_id": _sanitize_input(str(state.get("question_doc_id", "")), config=config),
        "assessment_results": {"reason": "No questions found for evaluation."},
    }


def _create_error_result(state: Dict[str, Any], error_message: str, config: EvaluatorConfig) -> Dict[str, Any]:
    """Create result structure for error cases with sanitized output."""
    # Security: Sanitize error message and don't expose internal details
    safe_error = _sanitize_input(str(error_message), Constants.MAX_SANITIZED_ERROR_LENGTH, config)

    return {
        "assessment_id": _sanitize_input(str(state.get("assessment_id", "")), config=config),
        "assessment_topic": _sanitize_input(str(state.get("assessment_topic", "")), config=config),
        "question_doc_id": _sanitize_input(str(state.get("question_doc_id", "")), config=config),
        "assessment_results": {"reason": f"Evaluation failed: {safe_error}"},
    }


def _normalize_and_format_results(
    state: Dict[str, Any],
    question_scores: Dict[str, int],
    section_scores: Dict[str, int],
    total_score: int,
    total_max_score: int,
    config: EvaluatorConfig,
    question_rationales: Optional[Dict[str, str]] = None,
    per_question_payloads: Optional[List[Tuple[str, str, str, str]]] = None,
    section_max_scores: Optional[Dict[str, int]] = None,
    question_max_scores: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Normalize scores to 100-point scale and format results with sanitized output."""
    def _is_attempted_answer(user_answer: Any) -> bool:
        return bool(str(user_answer or "").strip())

    if total_max_score > 0:
        scale_factor = NORMALIZED_MAX_SCORE / total_max_score
        total_score_100 = total_score * scale_factor
        section_scores_100 = {k: v * scale_factor for k, v in section_scores.items()}
        question_scores_100 = {k: v * scale_factor for k, v in question_scores.items()}
    else:
        log.warning("Total max score is 0, cannot normalize scores")
        total_score_100 = 0
        section_scores_100, question_scores_100 = section_scores, question_scores

    # Section scores as percentage (0-100) per section: raw_section_score / section_max_score * 100
    normalized_section_scores: Dict[str, int] = {}
    if section_max_scores:
        for section, raw_score in section_scores.items():
            max_s = max(1, section_max_scores.get(section, 0))
            normalized_section_scores[section] = int(round(max(0, min((raw_score / max_s) * NORMALIZED_MAX_SCORE, NORMALIZED_MAX_SCORE))))
    else:
        normalized_section_scores = section_scores_100  # fallback

    qmax = question_max_scores or {}
    question_scores_with_max: Dict[str, str] = {
        _sanitize_input(str(qid), Constants.MAX_SANITIZED_FIELD_LENGTH, config): f"{question_scores.get(qid, 0)}/{max(1, qmax.get(qid, DEFAULT_MAX_SCORE))}"
        for qid in question_scores
    }
    attempted_question_ids = {
        qid
        for qid, _, user_answer, _ in (per_question_payloads or [])
        if _is_attempted_answer(user_answer)
    }
    counted_question_ids = attempted_question_ids if per_question_payloads is not None else set(question_scores.keys())
    attempted_questions = len(counted_question_ids)
    fully_correct_questions = sum(
        1
        for qid in counted_question_ids
        if question_scores.get(qid, 0) >= max(1, qmax.get(qid, DEFAULT_MAX_SCORE))
    )
    partially_correct_questions = sum(
        1
        for qid in counted_question_ids
        if 0 < question_scores.get(qid, 0) < max(1, qmax.get(qid, DEFAULT_MAX_SCORE))
    )
    incorrect_questions = sum(
        1
        for qid in counted_question_ids
        if question_scores.get(qid, 0) <= 0
    )
    legacy_correct_answers = fully_correct_questions + partially_correct_questions
    weighted_score_percentage = round((total_score / total_max_score) * 100, 2) if total_max_score > 0 else 0.0
    question_correctness_rate = round((fully_correct_questions / attempted_questions) * 100, 2) if attempted_questions > 0 else 0.0

    result = {
        "assessment_id": _sanitize_input(str(state.get("assessment_id", "")), config=config),
        "assessment_topic": _sanitize_input(str(state.get("assessment_topic", "")), config=config),
        "question_doc_id": _sanitize_input(str(state.get("question_doc_id", "")), config=config),
        "assessment_results": {
            "total_score": int(round(total_score_100)),
            "max_score": NORMALIZED_MAX_SCORE,
            "weighted_score": total_score,
            "weighted_max_score": total_max_score,
            "weighted_score_percentage": weighted_score_percentage,
            "total_questions": len(question_scores),
            "answered_questions": attempted_questions,
            "attempted_questions": attempted_questions,
            "fully_correct_questions": fully_correct_questions,
            "partially_correct_questions": partially_correct_questions,
            "incorrect_questions": incorrect_questions,
            # Legacy aliases group any credit-earning answer under "correct".
            "correct_answers": legacy_correct_answers,
            "incorrect_answers": incorrect_questions,
            "question_correctness_rate": question_correctness_rate,
            "score_explanation": (
                "Official score is based on weighted points earned across the assessment, "
                "normalized to a 0-100 scale."
            ),
            "section_scores": {
                _sanitize_input(str(k), Constants.MAX_SANITIZED_FIELD_LENGTH, config): int(round(max(0, min(v, NORMALIZED_MAX_SCORE))))
                for k, v in normalized_section_scores.items()
            },
            "question_scores": question_scores_with_max,
        },
    }

    # Build per_question structure grouped by section/type
    if per_question_payloads:
        per_question: Dict[str, List[Dict[str, Any]]] = {}
        qmax = question_max_scores or {}
        for qid, question_text, user_answer, expected_answer in per_question_payloads:
            section = qid.split('_')[0] if '_' in qid else 'unknown'
            score_raw = question_scores.get(qid, 0)
            max_for_q = qmax.get(qid, DEFAULT_MAX_SCORE)
            entry: Dict[str, Any] = {
                "question": question_text,
                "user_answer": user_answer,
                "expected_answer": expected_answer,
                "evaluation": {
                    "score": score_raw,
                    "max_score": int(max(1, max_for_q)),
                }
            }
            # Pull explanation from generated rationales map if available
            rationale_text = None
            if question_rationales:
                rationale_text = question_rationales.get(qid)
            
            # Always set explanation, but normalize misleading prefixes vs score
            if rationale_text:
                explanation_text = rationale_text
            else:
                explanation_text = "Your answer is correct" if score_raw > 0 else "Your answer is incorrect"

            try:
                import re as _re
                # Convert third-person to second-person for better user experience
                if isinstance(explanation_text, str):
                    explanation_text = _convert_to_second_person(explanation_text)
                
                # If score is zero but explanation indicates correctness, rewrite to indicate incorrectness
                if (isinstance(score_raw, (int, float)) and score_raw <= 0) and isinstance(explanation_text, str):
                    # Check for patterns that incorrectly indicate correctness
                    incorrect_patterns = [
                        r"^\s*(correct|that's correct|that is correct|yes|your answer is correct)\b",
                        r"\bthe response correctly\b",
                        r"\bthe response (is|was) (correct|right|accurate)\b",
                        r"\b(correctly|rightly|accurately) (emphasizes?|explains?|describes?|states?)\b",
                        r"\b(is|was) (correct|right|accurate)\b",
                    ]
                    
                    needs_correction = False
                    for pattern in incorrect_patterns:
                        if _re.search(pattern, explanation_text, _re.IGNORECASE):
                            needs_correction = True
                            break
                    
                    # Also check if explanation talks about "the response" in a positive way when score is 0
                    if not needs_correction and _re.search(r"\bthe response\b", explanation_text, _re.IGNORECASE):
                        # If it mentions "the response" but doesn't clearly indicate incorrectness, it might be misleading
                        if not _re.search(r"\b(incorrect|wrong|missing|incomplete|failed|lacks)\b", explanation_text, _re.IGNORECASE):
                            needs_correction = True
                    
                    if needs_correction:
                        # Replace incorrect affirmative phrases at the start
                        explanation_text = _re.sub(
                            r"^\s*(correct\.?|that's correct\.?|that is correct\.?|yes[,\.]?|your answer is correct\.?)\s*",
                            "Your answer is incorrect. ",
                            explanation_text,
                            flags=_re.IGNORECASE
                        )
                        # Replace phrases that incorrectly indicate correctness throughout the text
                        explanation_text = _re.sub(
                            r"\bthe response correctly\b",
                            "Your answer incorrectly",
                            explanation_text,
                            flags=_re.IGNORECASE
                        )
                        explanation_text = _re.sub(
                            r"\bthe response (is|was) (correct|right|accurate)\b",
                            "Your answer is incorrect",
                            explanation_text,
                            flags=_re.IGNORECASE
                        )
                        explanation_text = _re.sub(
                            r"\b(correctly|rightly|accurately) (emphasizes?|explains?|describes?|states?)\b",
                            "incorrectly states",
                            explanation_text,
                            flags=_re.IGNORECASE
                        )
                        # Replace "The response emphasizes..." with "Your answer is incorrect. The response..."
                        explanation_text = _re.sub(
                            r"\bthe response (emphasizes?|explains?|describes?|states?|highlights?)\b",
                            "Your answer is incorrect. The response",
                            explanation_text,
                            flags=_re.IGNORECASE
                        )
                        # Ensure it starts with an incorrect indicator if it doesn't already
                        if not _re.match(r"^\s*(incorrect|your answer is incorrect|wrong)", explanation_text, _re.IGNORECASE):
                            explanation_text = "Your answer is incorrect. " + explanation_text.lstrip()
                
                # If score is positive but explanation starts with "Incorrect", flip to a neutral affirmation
                if (isinstance(score_raw, (int, float)) and score_raw > 0) and isinstance(explanation_text, str):
                    if _re.match(r"^\s*incorrect\b", explanation_text, _re.IGNORECASE):
                        explanation_text = _re.sub(r"^\s*incorrect\.?\s*", "Your answer is correct. ", explanation_text, flags=_re.IGNORECASE)
            except Exception:
                # Best-effort normalization; ignore failures
                pass

            entry["evaluation"]["explanation"] = explanation_text
            
            # Always try to set correct answer if available
            correct_key = f"{qid}__correct"
            correct_text = question_rationales.get(correct_key) if question_rationales else ""
            if correct_text:
                entry["evaluation"]["correct_answer"] = correct_text
            else:
                entry["evaluation"]["correct_answer"] = ""

            per_question.setdefault(section, []).append(entry)

        result["assessment_results"]["per_question"] = per_question

        return result

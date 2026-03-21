"""
Intelligent Adaptive Field Filtering System for Agent Outputs

This module provides intelligent, adaptive filtering of agent outputs that:
1. Automatically discovers agent capabilities through introspection
2. Adapts to new agents without manual configuration
3. Learns field patterns from agent outputs
4. Provides intelligent field filtering based on workflow context
"""

from typing import Dict, Any, List, Set, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import inspect
import re
from collections import defaultdict, Counter
import json

log = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of agents in the system."""
    PARSER = "parser"
    ANALYZER = "analyzer" 
    RECOMMENDER = "recommender"
    EVALUATOR = "evaluator"
    NOTIFIER = "notifier"
    RANKER = "ranker"
    GENERATOR = "generator"
    UNKNOWN = "unknown"

@dataclass
class FieldPattern:
    """Pattern information for a field."""
    field_name: str
    frequency: int = 0
    confidence: float = 0.0
    data_types: Set[str] = field(default_factory=set)
    is_required: bool = False
    is_output: bool = False
    is_input: bool = False
    agent_sources: Set[str] = field(default_factory=set)

@dataclass
class AgentFieldSchema:
    """Schema defining what fields an agent produces and consumes."""
    agent_name: str
    agent_type: AgentType
    produces: Set[str] = field(default_factory=set)
    consumes: Set[str] = field(default_factory=set)
    required_for_output: Set[str] = field(default_factory=set)
    optional_for_output: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    last_updated: float = 0.0
    field_patterns: Dict[str, FieldPattern] = field(default_factory=dict)

@dataclass
class WorkflowPattern:
    """Pattern information for a workflow context."""
    context_name: str
    common_fields: Set[str] = field(default_factory=set)
    agent_sequences: List[List[str]] = field(default_factory=list)
    field_dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    confidence: float = 0.0
    frequency: int = 0

class AdaptiveFieldFilterManager:
    """Intelligent, adaptive field filtering that learns from agent behavior."""
    
    def __init__(self):
        self.agent_schemas: Dict[str, AgentFieldSchema] = {}
        self.workflow_patterns: Dict[str, WorkflowPattern] = {}
        self.field_patterns: Dict[str, FieldPattern] = {}
        self.agent_discovery_cache: Dict[str, Any] = {}
        self.learning_data: List[Dict[str, Any]] = []
        self.agent_type_patterns = self._initialize_agent_type_patterns()
        self.field_classification_rules = self._initialize_field_classification_rules()
        
        # Initialize with some basic patterns
        self._initialize_basic_patterns()
        
        # Initialize legacy workflow_contexts for backward compatibility
        self.workflow_contexts = self._initialize_legacy_workflow_contexts()
    
    def _initialize_agent_type_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting agent types based on names and behavior."""
        return {
            "parser": ["parser", "extract", "parse", "parse_", "info_parser", "description_parser"],
            "analyzer": ["analyzer", "analysis", "analyze", "scorer", "score", "evaluator"],
            "recommender": ["recommender", "recommend", "suggestion", "advisor", "advisor"],
            "evaluator": ["evaluator", "evaluate", "assessment", "test", "exam"],
            "generator": ["generator", "generate", "create", "build", "question_generator"],
            "notifier": ["notifier", "notification", "email", "mail", "send"],
            "ranker": ["ranker", "rank", "sort", "order", "priority"]
        }
    
    def _initialize_field_classification_rules(self) -> Dict[str, Callable[[str, Any], bool]]:
        """Initialize rules for classifying fields based on names and values."""
        return {
            "is_required": lambda name, value: (
                "status" in name.lower() or 
                "error" in name.lower() or
                "id" in name.lower() or
                (isinstance(value, (str, list, dict)) and value) or
                (isinstance(value, (int, float)) and value > 0)
            ),
            "is_output": lambda name, value: (
                not name.startswith("input_") and
                not name.startswith("raw_") and
                not name.endswith("_input") and
                not name.endswith("_text") and
                name not in ["resume_text", "jd_text", "body", "uid", "tenant_id"]
            ),
            "is_input": lambda name, value: (
                name.startswith("input_") or
                name.startswith("raw_") or
                name.endswith("_input") or
                name.endswith("_text") or
                name in ["resume_text", "jd_text", "body", "uid", "tenant_id", "callback_url"]
            ),
            "is_jd_related": lambda name, value: (
                "jd_" in name.lower() or
                "job_description" in name.lower() or
                "ranked_candidates" in name.lower() or
                "ranker" in name.lower() or
                name in ["jd_text", "jd_url", "is_valid_jd"]
            ),
            "is_resume_related": lambda name, value: (
                "resume" in name.lower() or
                "structured_resume" in name.lower() or
                "skill_gap" in name.lower() or
                "market_analysis" in name.lower() or
                "course_recommendations" in name.lower() or
                "assessment_plan" in name.lower() or
                name in ["resume_text", "resume_url", "is_valid_resume", "user_interests"]
            )
        }
    
    def _initialize_basic_patterns(self):
        """Initialize with some basic known patterns."""
        # Add some basic workflow patterns
        self.workflow_patterns = {
            "resume_processing": WorkflowPattern(
                context_name="resume_processing",
                common_fields={"structured_resume", "resume_score", "skill_gap_analysis"},
                agent_sequences=[["personal_info_parser", "experience_parser", "education_parser", "skills_parser"]],
                confidence=0.8
            ),
            "job_description_processing": WorkflowPattern(
                context_name="job_description_processing", 
                common_fields={"job_description", "ranked_candidates"},
                agent_sequences=[["job_description_parser", "ranker"]],
                confidence=0.8
            )
        }
    
    def _initialize_agent_schemas(self) -> Dict[str, AgentFieldSchema]:
        """Initialize field schemas for all agents."""
        return {
            # Resume Parsing Agents
            "personal_info_parser": AgentFieldSchema(
                agent_name="personal_info_parser",
                agent_type=AgentType.PARSER,
                produces={"name", "contact_details", "personal_info_parser_status", "confidence_score"},
                consumes={"resume_text"},
                required_for_output={"name", "contact_details"},
                optional_for_output={"personal_info_parser_status", "confidence_score", "processing_time"}
            ),
            
            "experience_parser": AgentFieldSchema(
                agent_name="experience_parser", 
                agent_type=AgentType.PARSER,
                produces={"work_experience", "experience_parser_status", "confidence_score"},
                consumes={"resume_text"},
                required_for_output={"work_experience"},
                optional_for_output={"experience_parser_status", "confidence_score", "processing_time"}
            ),
            
            "education_parser": AgentFieldSchema(
                agent_name="education_parser",
                agent_type=AgentType.PARSER, 
                produces={"education", "education_parser_status", "confidence_score"},
                consumes={"resume_text"},
                required_for_output={"education"},
                optional_for_output={"education_parser_status", "confidence_score", "processing_time"}
            ),
            
            "skills_parser": AgentFieldSchema(
                agent_name="skills_parser",
                agent_type=AgentType.PARSER,
                produces={"skills", "skills_parser_status", "confidence_score"},
                consumes={"resume_text"},
                required_for_output={"skills"},
                optional_for_output={"skills_parser_status", "confidence_score", "processing_time"}
            ),
            
            "resume_assembler": AgentFieldSchema(
                agent_name="resume_assembler",
                agent_type=AgentType.PARSER,
                produces={"structured_resume", "resume_assembler_status"},
                consumes={"name", "contact_details", "work_experience", "education", "skills", "certifications", "projects", "extras"},
                required_for_output={"structured_resume"},
                optional_for_output={"resume_assembler_status", "processing_time"}
            ),
            
            # Analysis Agents
            "resume_scorer": AgentFieldSchema(
                agent_name="resume_scorer",
                agent_type=AgentType.ANALYZER,
                produces={"resume_score", "resume_scorer_status", "confidence_score", "resume_analysis"},
                consumes={"structured_resume"},
                required_for_output={"resume_score"},
                optional_for_output={"resume_scorer_status", "confidence_score", "resume_analysis", "processing_time"}
            ),
            
            "career_advisor": AgentFieldSchema(
                agent_name="career_advisor",
                agent_type=AgentType.ANALYZER,
                produces={"raw_skill_gap_analysis_output", "skill_gap_analysis", "analysis_status", "confidence_score"},
                consumes={"structured_resume", "user_interests"},
                required_for_output={"raw_skill_gap_analysis_output"},
                optional_for_output={"skill_gap_analysis", "analysis_status", "confidence_score", "processing_time"}
            ),
            
            "market_and_course_recommender": AgentFieldSchema(
                agent_name="market_and_course_recommender",
                agent_type=AgentType.RECOMMENDER,
                produces={"market_analysis", "course_recommendations", "recommender_status", "confidence_score"},
                consumes={"structured_resume", "raw_skill_gap_analysis_output"},
                required_for_output={"market_analysis", "course_recommendations"},
                optional_for_output={"recommender_status", "confidence_score", "processing_time"}
            ),
            
            "assessment_recommender": AgentFieldSchema(
                agent_name="assessment_recommender",
                agent_type=AgentType.RECOMMENDER,
                produces={"assessment_plan", "assessment_recommender_status", "confidence_score"},
                consumes={"structured_resume", "raw_skill_gap_analysis_output"},
                required_for_output={"assessment_plan"},
                optional_for_output={"assessment_recommender_status", "confidence_score", "processing_time"}
            ),
            
            # Evaluation Agents
            "assessment_evaluator": AgentFieldSchema(
                agent_name="assessment_evaluator",
                agent_type=AgentType.EVALUATOR,
                produces={"assessment_results", "evaluation_status", "confidence_score"},
                consumes={"submission", "structured_resume"},
                required_for_output={"assessment_results"},
                optional_for_output={"evaluation_status", "confidence_score", "processing_time"}
            ),
            
            "assessment_question_generator": AgentFieldSchema(
                agent_name="assessment_question_generator",
                agent_type=AgentType.GENERATOR,
                produces={"generated_questions", "question_generator_status", "confidence_score"},
                consumes={"assessment_plan", "structured_resume"},
                required_for_output={"generated_questions"},
                optional_for_output={"question_generator_status", "confidence_score", "processing_time"}
            ),
            
            # Job Description Agents
            "job_description_parser": AgentFieldSchema(
                agent_name="job_description_parser",
                agent_type=AgentType.PARSER,
                produces={"job_description", "jd_parser_status", "confidence_score"},
                consumes={"jd_text"},
                required_for_output={"job_description"},
                optional_for_output={"jd_parser_status", "confidence_score", "processing_time"}
            ),
            
            "ranker": AgentFieldSchema(
                agent_name="ranker",
                agent_type=AgentType.RANKER,
                produces={"ranked_candidates", "ranker_status", "confidence_score"},
                consumes={"job_description", "structured_resume"},
                required_for_output={"ranked_candidates"},
                optional_for_output={"ranker_status", "confidence_score", "processing_time"}
            ),
            
            # Notification Agent
            "notification_agent": AgentFieldSchema(
                agent_name="notification_agent",
                agent_type=AgentType.NOTIFIER,
                produces={"notification_status", "email_sent", "notification_result"},
                consumes={"user_mail", "email", "structured_resume"},
                required_for_output={"notification_status"},
                optional_for_output={"email_sent", "notification_result", "processing_time"}
            ),
            
            # Report Generator
            "report_generator": AgentFieldSchema(
                agent_name="report_generator",
                agent_type=AgentType.GENERATOR,
                produces={"report", "report_generator_status", "confidence_score"},
                consumes={"assessment_results", "structured_resume"},
                required_for_output={"report"},
                optional_for_output={"report_generator_status", "confidence_score", "processing_time"}
            )
        }
    
    def _initialize_legacy_workflow_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize legacy workflow contexts for backward compatibility."""
        return {
            "resume_processing": {
                "required_agents": ["personal_info_parser", "experience_parser", "education_parser", "skills_parser", "resume_assembler"],
                "analysis_agents": ["resume_scorer", "career_advisor", "market_and_course_recommender", "assessment_recommender"],
                "core_fields": {"structured_resume", "resume_score", "raw_skill_gap_analysis_output", "market_analysis", "assessment_plan"},
                "optional_fields": {"name", "contact_details", "work_experience", "education", "skills", "certifications", "projects", "extras", "course_recommendations"}
            },
            
            "job_description_processing": {
                "required_agents": ["job_description_parser", "ranker"],
                "core_fields": {"job_description", "ranked_candidates"},
                "optional_fields": {"jd_parser_status", "ranker_status", "confidence_score"}
            },
            
            "assessment_evaluation": {
                "required_agents": ["assessment_evaluator", "report_generator"],
                "core_fields": {"assessment_results", "report"},
                "optional_fields": {"evaluation_status", "report_generator_status", "confidence_score"}
            },
            
            "assessment_generation": {
                "required_agents": ["assessment_question_generator"],
                "core_fields": {"generated_questions"},
                "optional_fields": {"question_generator_status", "confidence_score"}
            },
            
            "notification": {
                "required_agents": ["notification_agent"],
                "core_fields": {"notification_status"},
                "optional_fields": {"email_sent", "notification_result"}
            }
        }
    
    def discover_agent_type(self, agent_name: str) -> AgentType:
        """Automatically discover agent type based on name patterns."""
        agent_name_lower = agent_name.lower()
        
        for agent_type, patterns in self.agent_type_patterns.items():
            for pattern in patterns:
                if pattern in agent_name_lower:
                    return AgentType(agent_type)
        
        return AgentType.UNKNOWN
    
    def analyze_agent_behavior(self, agent_name: str, input_state: Dict[str, Any], output_state: Dict[str, Any]) -> AgentFieldSchema:
        """Analyze agent behavior to build its field schema dynamically."""
        import time
        
        # Get or create schema
        if agent_name not in self.agent_schemas:
            agent_type = self.discover_agent_type(agent_name)
            self.agent_schemas[agent_name] = AgentFieldSchema(
                agent_name=agent_name,
                agent_type=agent_type,
                confidence=0.0
            )
        
        schema = self.agent_schemas[agent_name]
        
        # Analyze input fields (what the agent consumes)
        input_fields = set(input_state.keys())
        
        # Analyze output fields (what the agent produces)
        output_fields = set(output_state.keys()) - input_fields
        
        # Classify fields using rules
        for field_name, field_value in output_state.items():
            if field_name not in schema.field_patterns:
                schema.field_patterns[field_name] = FieldPattern(field_name=field_name)
            
            pattern = schema.field_patterns[field_name]
            pattern.frequency += 1
            pattern.data_types.add(type(field_value).__name__)
            pattern.agent_sources.add(agent_name)
            
            # Apply classification rules
            for rule_name, rule_func in self.field_classification_rules.items():
                if rule_func(field_name, field_value):
                    if rule_name == "is_required":
                        pattern.is_required = True
                    elif rule_name == "is_output":
                        pattern.is_output = True
                    elif rule_name == "is_input":
                        pattern.is_input = True
        
        # Update schema based on analysis
        schema.produces.update(output_fields)
        schema.consumes.update(input_fields)
        
        # Update required/optional fields based on patterns
        for field_name, pattern in schema.field_patterns.items():
            if pattern.is_required and pattern.is_output:
                schema.required_for_output.add(field_name)
            elif pattern.is_output:
                schema.optional_for_output.add(field_name)
        
        # Update confidence based on frequency
        total_fields = len(schema.field_patterns)
        if total_fields > 0:
            high_confidence_fields = sum(1 for p in schema.field_patterns.values() if p.frequency > 2)
            schema.confidence = high_confidence_fields / total_fields
        
        schema.last_updated = time.time()
        
        # Store learning data
        self.learning_data.append({
            "timestamp": time.time(),
            "agent_name": agent_name,
            "input_fields": list(input_fields),
            "output_fields": list(output_fields),
            "field_patterns": {k: v.__dict__ for k, v in schema.field_patterns.items()}
        })
        
        # Keep only recent learning data (last 1000 entries)
        if len(self.learning_data) > 1000:
            self.learning_data = self.learning_data[-1000:]
        
        return schema
    
    def determine_workflow_context(self, state: Dict[str, Any]) -> str:
        """Determine which workflow context is active based on state."""
        body = state.get("body", {})
        
        if body.get("user_mail") or body.get("email"):
            return "notification"
        elif body.get("resume_url"):
            return "resume_processing"
        elif body.get("jd_url"):
            return "job_description_processing"
        elif body.get("submission"):
            return "assessment_evaluation"
        elif body.get("plan"):
            return "assessment_generation"
        else:
            return "resume_processing"  # Default fallback
    
    def learn_from_workflow(self, state_sequence: List[Dict[str, Any]], workflow_context: str):
        """Learn workflow patterns from a sequence of states."""
        if workflow_context not in self.workflow_patterns:
            self.workflow_patterns[workflow_context] = WorkflowPattern(context_name=workflow_context)
        
        pattern = self.workflow_patterns[workflow_context]
        pattern.frequency += 1
        
        # Analyze field evolution through the workflow
        all_fields = set()
        for state in state_sequence:
            all_fields.update(state.keys())
        
        pattern.common_fields.update(all_fields)
        
        # Learn agent sequences
        agent_sequence = []
        for state in state_sequence:
            # Try to identify which agent produced this state
            for agent_name, schema in self.agent_schemas.items():
                if any(field in state for field in schema.produces):
                    agent_sequence.append(agent_name)
                    break
        
        if agent_sequence:
            pattern.agent_sequences.append(agent_sequence)
        
        # Update confidence
        pattern.confidence = min(1.0, pattern.frequency / 10.0)
    
    def get_active_agents(self, state: Dict[str, Any]) -> List[str]:
        """Determine which agents are currently active based on learned patterns."""
        active_agents = []
        
        # Check all known agents to see if they have produced output
        for agent_name, schema in self.agent_schemas.items():
            if self._agent_has_output(state, agent_name):
                active_agents.append(agent_name)
        
        # If no agents are detected, try to infer from field patterns
        if not active_agents:
            workflow_context = self.determine_workflow_context(state)
            if workflow_context in self.workflow_patterns:
                pattern = self.workflow_patterns[workflow_context]
                # Look for fields that match common patterns
                for field_name in state.keys():
                    for agent_name, schema in self.agent_schemas.items():
                        if field_name in schema.produces:
                            active_agents.append(agent_name)
                            break
        
        return active_agents
    
    def _agent_has_output(self, state: Dict[str, Any], agent_name: str) -> bool:
        """Check if an agent has produced output in the state."""
        schema = self.agent_schemas.get(agent_name)
        if not schema:
            return False
        
        # Check if any of the agent's produced fields exist in state
        return any(field in state for field in schema.produces)
    
    def predict_agent_output(self, agent_name: str, input_state: Dict[str, Any]) -> Set[str]:
        """Predict what fields an agent will produce based on learned patterns."""
        if agent_name not in self.agent_schemas:
            return set()
        
        schema = self.agent_schemas[agent_name]
        
        # Base prediction on learned patterns
        predicted_fields = set()
        
        for field_name, pattern in schema.field_patterns.items():
            if pattern.is_output and pattern.confidence > 0.5:
                predicted_fields.add(field_name)
        
        # Add fields that are commonly produced by this agent type
        agent_type = schema.agent_type
        for other_agent, other_schema in self.agent_schemas.items():
            if other_schema.agent_type == agent_type and other_agent != agent_name:
                predicted_fields.update(other_schema.produces)
        
        return predicted_fields
    
    def filter_state_for_display(self, state: Dict[str, Any], target_agent: Optional[str] = None) -> Dict[str, Any]:
        """
        Intelligently filter state to only include relevant fields for display.
        
        Args:
            state: The current agent state
            target_agent: Specific agent to filter for, or None for general filtering
            
        Returns:
            Filtered state with only relevant fields
        """
        if target_agent:
            return self._filter_for_specific_agent(state, target_agent)
        else:
            return self._filter_for_workflow_context(state)
    
    def _filter_for_specific_agent(self, state: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Intelligently filter state for a specific agent's needs."""
        schema = self.agent_schemas.get(agent_name)
        if not schema:
            # If no schema exists, try to create one from the state
            log.info(f"No schema found for agent: {agent_name}, creating adaptive schema")
            schema = self.analyze_agent_behavior(agent_name, {}, state)
        
        # Include fields this agent produces and consumes
        relevant_fields = schema.produces | schema.consumes
        
        # Add predicted fields based on learned patterns
        predicted_fields = self.predict_agent_output(agent_name, state)
        relevant_fields.update(predicted_fields)
        
        # Always include core system fields
        core_fields = {
            "uid", "tenant_id", "callback_url", "analysis_id", "session_id",
            "processing_time_seconds", "analysis_method", "confidence_score",
            "ok", "error", "validation_error", "security_error", "rate_limit_error"
        }
        relevant_fields.update(core_fields)
        
        # Add fields that are commonly associated with this agent type
        agent_type = schema.agent_type
        for other_agent, other_schema in self.agent_schemas.items():
            if other_schema.agent_type == agent_type and other_agent != agent_name:
                # Add fields that are commonly produced by similar agents
                for field_name, pattern in other_schema.field_patterns.items():
                    if pattern.confidence > 0.7 and pattern.is_output:
                        relevant_fields.add(field_name)
        
        # Filter state to only include relevant fields
        filtered_state = {k: v for k, v in state.items() if k in relevant_fields}
        
        log.info(f"Adaptive filtered state for {agent_name}: {list(filtered_state.keys())}")
        return filtered_state
    
    def _filter_for_workflow_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently filter state based on the current workflow context."""
        workflow_context = self.determine_workflow_context(state)
        
        # Get learned workflow pattern
        workflow_pattern = self.workflow_patterns.get(workflow_context)
        
        # Start with common fields for this workflow
        if workflow_pattern:
            relevant_fields = set(workflow_pattern.common_fields)
        else:
            relevant_fields = set()
        
        # Add fields from active agents
        active_agents = self.get_active_agents(state)
        for agent_name in active_agents:
            schema = self.agent_schemas.get(agent_name)
            if schema:
                relevant_fields.update(schema.produces)
                # Add high-confidence optional fields
                for field_name, pattern in schema.field_patterns.items():
                    if pattern.confidence > 0.6 and pattern.is_output:
                        relevant_fields.add(field_name)
        
        # Add fields that are commonly associated with this workflow context
        for other_agent, other_schema in self.agent_schemas.items():
            # Check if this agent type is commonly used in this workflow
            if workflow_pattern and any(agent_name in seq for seq in workflow_pattern.agent_sequences for agent_name in [other_agent]):
                for field_name, pattern in other_schema.field_patterns.items():
                    if pattern.confidence > 0.5 and pattern.is_output:
                        relevant_fields.add(field_name)
        
        # Apply workflow-specific exclusions using intelligent field classification
        if workflow_context == "resume_processing":
            # Exclude JD-related fields for resume processing
            jd_excluded_fields = set()
            for field_name in state.keys():
                if self.field_classification_rules["is_jd_related"](field_name, state[field_name]):
                    jd_excluded_fields.add(field_name)
            relevant_fields = relevant_fields - jd_excluded_fields
        elif workflow_context == "job_description_processing":
            # Exclude resume-related fields for JD processing
            resume_excluded_fields = set()
            for field_name in state.keys():
                if self.field_classification_rules["is_resume_related"](field_name, state[field_name]):
                    resume_excluded_fields.add(field_name)
            relevant_fields = relevant_fields - resume_excluded_fields
        
        # Always include core system fields
        core_fields = {
            "uid", "tenant_id", "callback_url", "analysis_id", "session_id",
            "processing_time_seconds", "analysis_method", "confidence_score",
            "ok", "error", "validation_error", "security_error", "rate_limit_error"
        }
        relevant_fields.update(core_fields)
        
        # Filter state
        filtered_state = {k: v for k, v in state.items() if k in relevant_fields}
        
        log.info(f"Adaptive filtered state for {workflow_context} workflow: {list(filtered_state.keys())}")
        return filtered_state
    
    def get_agent_output_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get a summary of what each active agent has produced."""
        active_agents = self.get_active_agents(state)
        summary = {}
        
        for agent_name in active_agents:
            schema = self.agent_schemas.get(agent_name)
            if not schema:
                continue
            
            agent_output = {}
            for field in schema.produces:
                if field in state:
                    agent_output[field] = state[field]
            
            if agent_output:
                summary[agent_name] = {
                    "status": "completed",
                    "output_fields": list(agent_output.keys()),
                    "data": agent_output
                }
        
        return summary
    
    def validate_agent_output(self, agent_name: str, output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that an agent's output contains required fields."""
        schema = self.agent_schemas.get(agent_name)
        if not schema:
            return False, [f"No schema found for agent: {agent_name}"]
        
        missing_required = []
        for field in schema.required_for_output:
            if field not in output or output[field] is None:
                missing_required.append(field)
        
        if missing_required:
            return False, [f"Missing required fields for {agent_name}: {missing_required}"]
        
        return True, []
    
    def get_workflow_progress(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get progress information for the current workflow."""
        workflow_context = self.determine_workflow_context(state)
        context_config = self.workflow_contexts.get(workflow_context, {})
        active_agents = self.get_active_agents(state)
        
        required_agents = context_config.get("required_agents", [])
        analysis_agents = context_config.get("analysis_agents", [])
        
        completed_required = [agent for agent in required_agents if agent in active_agents]
        completed_analysis = [agent for agent in analysis_agents if agent in active_agents]
        
        total_required = len(required_agents)
        total_analysis = len(analysis_agents)
        
        return {
            "workflow_context": workflow_context,
            "required_agents_progress": {
                "completed": len(completed_required),
                "total": total_required,
                "percentage": (len(completed_required) / total_required * 100) if total_required > 0 else 0,
                "agents": completed_required
            },
            "analysis_agents_progress": {
                "completed": len(completed_analysis),
                "total": total_analysis,
                "percentage": (len(completed_analysis) / total_analysis * 100) if total_analysis > 0 else 0,
                "agents": completed_analysis
            },
            "overall_progress": {
                "total_agents": total_required + total_analysis,
                "completed_agents": len(completed_required) + len(completed_analysis),
                "percentage": ((len(completed_required) + len(completed_analysis)) / (total_required + total_analysis) * 100) if (total_required + total_analysis) > 0 else 0
            }
        }

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning data."""
        if not self.learning_data:
            return {"message": "No learning data available"}
        
        # Analyze field patterns
        field_frequencies = Counter()
        agent_frequencies = Counter()
        
        for entry in self.learning_data:
            agent_frequencies[entry["agent_name"]] += 1
            for field in entry["output_fields"]:
                field_frequencies[field] += 1
        
        # Get most common fields and agents
        most_common_fields = field_frequencies.most_common(10)
        most_common_agents = agent_frequencies.most_common(10)
        
        # Analyze agent type distribution
        agent_type_distribution = Counter()
        for agent_name, schema in self.agent_schemas.items():
            agent_type_distribution[schema.agent_type.value] += 1
        
        return {
            "total_learning_entries": len(self.learning_data),
            "most_common_fields": most_common_fields,
            "most_common_agents": most_common_agents,
            "agent_type_distribution": dict(agent_type_distribution),
            "learned_agents": list(self.agent_schemas.keys()),
            "learned_workflows": list(self.workflow_patterns.keys()),
            "average_confidence": sum(schema.confidence for schema in self.agent_schemas.values()) / len(self.agent_schemas) if self.agent_schemas else 0
        }
    
    def export_learned_patterns(self) -> Dict[str, Any]:
        """Export learned patterns for persistence or analysis."""
        return {
            "agent_schemas": {
                name: {
                    "agent_name": schema.agent_name,
                    "agent_type": schema.agent_type.value,
                    "produces": list(schema.produces),
                    "consumes": list(schema.consumes),
                    "required_for_output": list(schema.required_for_output),
                    "optional_for_output": list(schema.optional_for_output),
                    "confidence": schema.confidence,
                    "last_updated": schema.last_updated,
                    "field_patterns": {
                        field_name: {
                            "frequency": pattern.frequency,
                            "confidence": pattern.confidence,
                            "data_types": list(pattern.data_types),
                            "is_required": pattern.is_required,
                            "is_output": pattern.is_output,
                            "is_input": pattern.is_input,
                            "agent_sources": list(pattern.agent_sources)
                        }
                        for field_name, pattern in schema.field_patterns.items()
                    }
                }
                for name, schema in self.agent_schemas.items()
            },
            "workflow_patterns": {
                name: {
                    "context_name": pattern.context_name,
                    "common_fields": list(pattern.common_fields),
                    "agent_sequences": pattern.agent_sequences,
                    "field_dependencies": {k: list(v) for k, v in pattern.field_dependencies.items()},
                    "confidence": pattern.confidence,
                    "frequency": pattern.frequency
                }
                for name, pattern in self.workflow_patterns.items()
            },
            "learning_insights": self.get_learning_insights()
        }

# Global instance
field_filter_manager = AdaptiveFieldFilterManager()

def filter_agent_output(state: Dict[str, Any], target_agent: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to filter agent output using adaptive filtering.
    
    Args:
        state: The current agent state
        target_agent: Specific agent to filter for, or None for general filtering
        
    Returns:
        Filtered state with only relevant fields
    """
    return field_filter_manager.filter_state_for_display(state, target_agent)

def get_workflow_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a comprehensive summary of the current workflow state.
    
    Args:
        state: The current agent state
        
    Returns:
        Dictionary containing workflow summary information
    """
    return {
        "agent_outputs": field_filter_manager.get_agent_output_summary(state),
        "workflow_progress": field_filter_manager.get_workflow_progress(state),
        "active_agents": field_filter_manager.get_active_agents(state),
        "workflow_context": field_filter_manager.determine_workflow_context(state),
        "learning_insights": field_filter_manager.get_learning_insights()
    }

def learn_from_agent_execution(agent_name: str, input_state: Dict[str, Any], output_state: Dict[str, Any]):
    """
    Learn from an agent execution to improve future filtering.
    
    Args:
        agent_name: Name of the agent that executed
        input_state: State before agent execution
        output_state: State after agent execution
    """
    field_filter_manager.analyze_agent_behavior(agent_name, input_state, output_state)

def learn_from_workflow(workflow_states: List[Dict[str, Any]], workflow_context: str):
    """
    Learn from a complete workflow execution.
    
    Args:
        workflow_states: Sequence of states through the workflow
        workflow_context: Context of the workflow (e.g., "resume_processing")
    """
    field_filter_manager.learn_from_workflow(workflow_states, workflow_context)

def get_adaptive_insights() -> Dict[str, Any]:
    """Get insights from the adaptive learning system."""
    return field_filter_manager.get_learning_insights()

def export_learned_patterns() -> Dict[str, Any]:
    """Export all learned patterns for analysis or persistence."""
    return field_filter_manager.export_learned_patterns()

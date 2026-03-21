"""
Career Coach Module - Conversational Career Guidance

Phase 1: Conversational Mentor (Chatbot) with streaming support
"""

from .conversational_mentor import (
    career_chatbot_agent,
    career_chatbot_agent_stream,
    conversational_mentor_agent,  # Alias for backward compatibility
    CareerChatRequest,
    CareerChatResponse
)
from .context_aggregator import load_comprehensive_context
from .function_tools import (
    get_course_recommendations,
    get_career_advice,
    get_assessment_recommendations,
    get_market_insights_only,
    suggest_assessment_topics,
    create_career_goal,
    get_career_goals,
    identify_relevant_jobs,
    get_function_definitions,
    handle_function_call
)
from .conversation_analyzer import (
    extract_aspirations_from_conversation,
    extract_goals_from_message,
    analyze_conversation_patterns
)

__all__ = [
    "career_chatbot_agent",
    "career_chatbot_agent_stream",
    "conversational_mentor_agent",
    "CareerChatRequest",
    "CareerChatResponse",
    "load_comprehensive_context",
    "get_course_recommendations",
    "get_career_advice",
    "get_assessment_recommendations",
    "get_market_insights_only",
    "suggest_assessment_topics",
    "create_career_goal",
    "get_career_goals",
    "identify_relevant_jobs",
    "get_function_definitions",
    "handle_function_call",
    "extract_aspirations_from_conversation",
    "extract_goals_from_message",
    "analyze_conversation_patterns",
]


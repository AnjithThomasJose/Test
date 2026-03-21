"""
Workflow router for unified conversational agent.

Routes requests to appropriate workflow (interview or career guidance)
based on payload structure. Does not modify existing interview agent code.
"""

import logging
from enum import Enum
from typing import Dict, Any
from fastapi import Request

log = logging.getLogger(__name__)


class WorkflowType(str, Enum):
    """Workflow types detected from payload"""
    INTERVIEW = "interview"
    CAREER_GUIDANCE = "career_guidance"


def detect_workflow_from_payload(body: Dict[str, Any]) -> WorkflowType:
    """
    Detect workflow type from request payload.
    
    Detection logic:
    1. Explicit "workflow" field (highest priority)
    2. "message" field = career guidance
    3. "interview_topic" or interview-specific fields = interview
    4. Conversation history pattern analysis
    5. Default to interview for backward compatibility
    
    Args:
        body: Request body dictionary
        
    Returns:
        WorkflowType enum
    """
    # Priority 1: Explicit workflow field (if provided)
    if "workflow" in body:
        workflow_str = body.get("workflow", "").lower()
        if workflow_str == "career_guidance":
            return WorkflowType.CAREER_GUIDANCE
        elif workflow_str == "interview":
            return WorkflowType.INTERVIEW
    
    # Priority 2: Check for career guidance indicators
    has_message = "message" in body and body.get("message")
    if has_message:
        log.debug("Detected career guidance workflow (has 'message' field)")
        return WorkflowType.CAREER_GUIDANCE
    
    # Priority 3: Check for interview indicators
    has_interview_topic = "interview_topic" in body and body.get("interview_topic")
    has_answer = "answer" in body
    has_question_count = "question_count" in body
    
    if has_interview_topic or (has_answer and has_question_count):
        log.debug("Detected interview workflow (has interview-specific fields)")
        return WorkflowType.INTERVIEW
    
    # Priority 4: Check conversation history pattern
    history = body.get("conversation_history", [])
    if history:
        # Analyze conversation pattern
        last_message = history[-1].get("content", "").lower() if history else ""
        
        # If last message is a question (ends with ?), likely interview
        if last_message.endswith("?"):
            log.debug("Detected interview workflow (last message is a question)")
            return WorkflowType.INTERVIEW
        
        # If last message contains guidance keywords, likely career guidance
        guidance_keywords = ["recommend", "suggest", "advice", "help", "courses", "career", "skills"]
        if any(keyword in last_message for keyword in guidance_keywords):
            log.debug("Detected career guidance workflow (guidance keywords in history)")
            return WorkflowType.CAREER_GUIDANCE
    
    # Default: interview for backward compatibility
    log.debug("Could not determine workflow from payload, defaulting to interview")
    return WorkflowType.INTERVIEW


async def route_workflow(request: Request) -> Any:
    """
    Route request to appropriate workflow based on payload.
    
    Does not modify existing interview agent code - just routes to it.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        Response from the appropriate workflow handler
    """
    try:
        body = await request.json()
        workflow = detect_workflow_from_payload(body)
        
        log.info(f"Routing to workflow: {workflow.value}")
        
        if workflow == WorkflowType.INTERVIEW:
            # Route to existing interview agent - NO CHANGES to interview code
            from .interview_agent import ai_interview_agent_intelligent
            return await ai_interview_agent_intelligent(request)
        
        elif workflow == WorkflowType.CAREER_GUIDANCE:
            # Route to career guidance workflow
            from .workflows.career_guidance_workflow import handle_career_guidance
            return await handle_career_guidance(request, body)
        
        else:
            raise ValueError(f"Unknown workflow: {workflow}")
            
    except Exception as e:
        log.error(f"Error routing workflow: {e}", exc_info=True)
        raise





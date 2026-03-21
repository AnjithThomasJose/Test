"""
Interview workflow handler.

Wraps existing interview agent logic without modifying it.
"""

import logging
from fastapi import Request

log = logging.getLogger(__name__)


async def handle_interview(request: Request):
    """
    Handle interview workflow.
    
    Simply calls the existing interview agent function without any modifications.
    This ensures the interview agent code remains completely unchanged.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        InterviewResponse from the interview agent
    """
    log.debug("Handling interview workflow")
    
    # Call existing interview agent - NO CHANGES to interview code
    from ..interview_agent import ai_interview_agent_intelligent
    return await ai_interview_agent_intelligent(request)





"""
Career guidance workflow handler.

Routes career guidance requests to the career coach chatbot.
Uses shared utilities for request handling and session management.
"""

import logging
from typing import Dict, Any
from fastapi import Request

log = logging.getLogger(__name__)


async def handle_career_guidance(request: Request, body: Dict[str, Any]):
    """
    Handle career guidance workflow.
    
    Uses existing career coach logic but routes through the unified endpoint.
    Uses shared utilities for request handling and session management.
    
    Args:
        request: FastAPI Request object
        body: Parsed request body
        
    Returns:
        Response from career coach (StreamingResponse or CareerChatResponse)
    """
    log.info("Handling career guidance workflow")
    
    try:
        # Import career coach components
        from agents.career_coach.conversational_mentor import (
            CareerChatRequest,
            career_chatbot_agent_stream,
            career_chatbot_agent
        )
        
        # Convert body to CareerChatRequest format
        chat_request = CareerChatRequest(
            uid=body.get("uid"),
            message=body.get("message", ""),
            session_id=body.get("session_id"),
            callback_url=body.get("callback_url"),
            stream=body.get("stream", True)
        )
        
        # Validate required fields
        if not chat_request.message:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail="'message' field is required for career guidance workflow"
            )
        
        log.info(f"Career guidance request | uid={chat_request.uid} | stream={chat_request.stream}")
        
        # Route to streaming or non-streaming based on request
        if chat_request.stream:
            return await career_chatbot_agent_stream(chat_request)
        else:
            return await career_chatbot_agent(chat_request)
            
    except Exception as e:
        log.error(f"Error in career guidance workflow: {e}", exc_info=True)
        raise





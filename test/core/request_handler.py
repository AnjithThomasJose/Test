"""
Shared request handling utilities for all conversational agents.

Provides common request parsing and authentication functionality
that can be used by both interview and career guidance workflows.
"""

import logging
import time
import uuid
from typing import Dict, Any, Tuple
from fastapi import Request, HTTPException

log = logging.getLogger(__name__)


async def parse_and_authenticate_request(request: Request) -> Tuple[Dict[str, Any], str]:
    """
    Parse JSON body and verify authentication token.
    
    Shared utility for both interview and career guidance workflows.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        Tuple of (parsed_body, uid)
        
    Raises:
        HTTPException: If authentication fails or uid is missing
    """
    try:
        body = await request.json()
        
        # Verify GenAI token
        from app import verify_request_token
        is_valid_token = await verify_request_token(request, body)
        if not is_valid_token:
            log.warning("Invalid or missing GenAI token")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing authentication token"
            )
        
        # Extract and validate uid
        uid = body.get("uid")
        if not uid:
            raise HTTPException(
                status_code=400,
                detail="uid is required"
            )
        
        return body, uid
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error parsing request: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request format: {str(e)}"
        )


def generate_session_id(uid: str) -> str:
    """
    Generate unique session ID.
    
    Format: {uid}_{timestamp}_{random}
    Ensures uniqueness even for concurrent requests.
    
    Args:
        uid: User ID
        
    Returns:
        Unique session ID string
    """
    return f"{uid}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"





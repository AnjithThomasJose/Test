"""
HeyGen Streaming API Integration
Uses HeyGen's Streaming API for real-time avatar video streaming via WebRTC
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional, AsyncIterator
from settings import settings
from core.http_client import get_http_client
from .streaming_config import StreamingConfig

log = logging.getLogger(__name__)

HEYGEN_API_KEY = getattr(settings, "HEYGEN_API_KEY", None)
HEYGEN_STREAMING_BASE_URL = "https://api.heygen.com/v1/streaming"


async def create_streaming_session(
    avatar_id: Optional[str] = None,
    voice_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new streaming session with HeyGen Streaming API.
    
    Args:
        avatar_id: Avatar ID (default: from StreamingConfig)
        voice_id: Voice ID (default: from StreamingConfig)
    
    Returns:
        Dict with session_id and connection details
    """
    if not HEYGEN_API_KEY:
        raise ValueError("HeyGen API key not configured")
    
    url = f"{HEYGEN_STREAMING_BASE_URL}/new"
    headers = {
        "X-Api-Key": HEYGEN_API_KEY,
        "Content-Type": "application/json",
    }
    
    payload = {
        "avatar_id": avatar_id or StreamingConfig.DEFAULT_AVATAR_ID,
        "voice_id": voice_id or StreamingConfig.DEFAULT_VOICE_ID,
    }
    
    client = await get_http_client()
    response = await client.post(url, json=payload, headers=headers, timeout=30.0)
    response.raise_for_status()
    return response.json()


async def send_streaming_task(
    session_id: str,
    text: str
) -> Dict[str, Any]:
    """
    Send a task (text) to the streaming session for avatar to speak.
    
    Args:
        session_id: Streaming session ID
        text: Text for avatar to speak
    
    Returns:
        Task response
    """
    if not HEYGEN_API_KEY:
        raise ValueError("HeyGen API key not configured")
    
    url = f"{HEYGEN_STREAMING_BASE_URL}/task"
    headers = {
        "X-Api-Key": HEYGEN_API_KEY,
        "Content-Type": "application/json",
    }
    
    payload = {
        "session_id": session_id,
        "text": text,
    }
    
    client = await get_http_client()
    response = await client.post(url, json=payload, headers=headers, timeout=30.0)
    response.raise_for_status()
    return response.json()


async def generate_avatar_video_streaming_api(
    text: str,
    avatar_id: Optional[str] = None,
    voice_id: Optional[str] = None
) -> AsyncIterator[Dict[str, Any]]:
    """
    Generate avatar video using HeyGen Streaming API (WebRTC-based real-time streaming).
    
    This uses HeyGen's Streaming API which provides:
    - Real-time video streaming via WebRTC
    - Low latency (< 1 second)
    - No polling required
    - Direct streaming to client
    
    Args:
        text: Text to convert to video
        avatar_id: Avatar ID (default: from StreamingConfig)
        voice_id: Voice ID (default: from StreamingConfig)
    
    Yields:
        Dict with streaming updates including WebRTC connection details
    """
    if not HEYGEN_API_KEY:
        yield {
            "type": "error",
            "message": "HeyGen API key not configured"
        }
        return
    
    try:
        # Step 1: Create streaming session
        yield {
            "type": "status",
            "message": "Creating streaming session...",
            "stage": "initiating",
            "progress": 0.1
        }
        
        session_data = await create_streaming_session(avatar_id, voice_id)
        session_id = session_data.get("data", {}).get("session_id")
        
        if not session_id:
            yield {
                "type": "error",
                "message": "Failed to create streaming session"
            }
            return
        
        # Get WebRTC connection details
        webrtc_data = session_data.get("data", {}).get("sdp_answer") or {}
        ice_servers = session_data.get("data", {}).get("ice_servers", [])
        
        yield {
            "type": "status",
            "message": "Streaming session created",
            "stage": "connected",
            "session_id": session_id,
            "progress": 0.3
        }
        
        # Step 2: Send task (text) to streaming session
        yield {
            "type": "status",
            "message": "Sending text to avatar...",
            "stage": "sending",
            "progress": 0.5
        }
        
        task_data = await send_streaming_task(session_id, text)
        task_id = task_data.get("data", {}).get("task_id")
        
        if not task_id:
            yield {
                "type": "error",
                "message": "Failed to send streaming task"
            }
            return
        
        # Step 3: Return WebRTC connection details for client
        # Client will establish WebRTC connection and receive video stream directly
        yield {
            "type": "video",
            "delivery_method": "webrtc_streaming",
            "session_id": session_id,
            "task_id": task_id,
            "webrtc": {
                "sdp_answer": webrtc_data,
                "ice_servers": ice_servers
            },
            "format": "webrtc",
            "is_streaming": True,
            "is_complete": True
        }
        
        log.info(f"HeyGen Streaming API session created: {session_id}, task: {task_id}")
        
    except Exception as e:
        log.exception(f"Error in HeyGen Streaming API: {e}")
        yield {
            "type": "error",
            "message": f"Error in streaming API: {str(e)}"
        }

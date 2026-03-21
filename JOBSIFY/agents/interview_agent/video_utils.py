"""
Video Interview Utilities
Handles VideoSDK meeting creation and HeyGen avatar video generation
"""

import logging
import time
import jwt
import asyncio
import base64
import re
from typing import Dict, Any, Optional, AsyncIterator
from settings import settings
from core.http_client import get_http_client
from .streaming_config import StreamingConfig
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

log = logging.getLogger(__name__)

# VideoSDK Configuration
VIDEOSDK_API_KEY = getattr(settings, "VIDEOSDK_API_KEY", None)
VIDEOSDK_API_SECRET = getattr(settings, "VIDEOSDK_API_SECRET", None)

# HeyGen Configuration
HEYGEN_API_KEY = getattr(settings, "HEYGEN_API_KEY", None)

# Get configurable settings
AVATAR_ENABLED = getattr(settings, "AVATAR_ENABLED", True)
AVATAR_GENERATION_TIMEOUT = getattr(settings, "AVATAR_GENERATION_TIMEOUT", StreamingConfig.AVATAR_DEFAULT_TIMEOUT)
AVATAR_FALLBACK_TO_URL = getattr(settings, "AVATAR_FALLBACK_TO_URL", True)
HEYGEN_MOCK_MODE = getattr(settings, "HEYGEN_MOCK_MODE", False)


def sanitize_question_for_avatar(question: str, max_length: int = 500) -> str:
    """
    Sanitize question text before sending to HeyGen API.
    
    Args:
        question: Question text to sanitize
        max_length: Maximum length in characters
    
    Returns:
        Sanitized question text
    """
    if not question:
        return ""
    
    # Remove potential PII patterns
    question = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', question)  # SSN pattern
    question = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', question)  # Phone pattern
    question = re.sub(r'\b\d{10}\b', '[NUMBER]', question)  # 10-digit numbers
    
    # Limit length
    question = question[:max_length]
    
    # Validate question format (ensure it ends with ?)
    question = question.strip()
    if not question.endswith('?'):
        question = f"{question.rstrip('.!')}?"
    
    # Remove special characters that might break API (keep basic punctuation)
    question = re.sub(r'[^\w\s\?\.\,\!\-\'\"]', '', question)
    
    return question


def generate_videosdk_token() -> Optional[str]:
    """
    Generate JWT token for VideoSDK.
    
    Returns:
        JWT token string or None if configuration is missing
    """
    if not VIDEOSDK_API_KEY or not VIDEOSDK_API_SECRET:
        log.warning("VideoSDK API key or secret not configured")
        return None
    
    try:
        payload = {
            "apikey": VIDEOSDK_API_KEY,
            "permissions": ["allow_join"],
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,  # 1 hour expiry
        }
        return jwt.encode(payload, VIDEOSDK_API_SECRET, algorithm="HS256")
    except Exception as e:
        log.exception(f"Error generating VideoSDK token: {e}")
        return None


async def create_videosdk_meeting() -> Dict[str, Any]:
    """
    Create a VideoSDK meeting room.
    
    Returns:
        Dict with roomId and token, or error dict
    """
    if not VIDEOSDK_API_KEY:
        return {
            "error": "VideoSDK API key not configured",
            "configured": False
        }
    
    try:
        url = "https://api.videosdk.live/v2/rooms"
        headers = {
            "authorization": VIDEOSDK_API_KEY,
            "Content-Type": "application/json",
        }
        
        client = await get_http_client()
        response = await client.post(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        room_data = response.json()
        
        if "roomId" not in room_data:
            log.error(f"Failed to create VideoSDK room: {room_data}")
            return {
                "error": "Failed to create room",
                "videosdk_response": room_data
            }
        
        token = generate_videosdk_token()
        if not token:
            return {
                "error": "Failed to generate VideoSDK token",
                "roomId": room_data["roomId"]
            }
        
        log.info(f"VideoSDK meeting created: {room_data['roomId']}")
        return {
            "roomId": room_data["roomId"],
            "token": token,
        }
    except Exception as e:
        log.exception(f"Error creating VideoSDK meeting: {e}")
        return {
            "error": f"Error creating meeting: {str(e)}"
        }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutError))
)
async def _poll_video_status(video_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Poll HeyGen API for video status with retry logic.
    
    Args:
        video_id: HeyGen video ID
        headers: Request headers
    
    Returns:
        Status response data
    """
    status_url = f"https://api.heygen.com/v1/video_status.get?video_id={video_id}"
    client = await get_http_client()
    response = await client.get(status_url, headers=headers, timeout=10.0)
    response.raise_for_status()
    return response.json()


async def generate_avatar_video(
    text: str,
    avatar_id: Optional[str] = None,
    voice_id: Optional[str] = None,
    return_base64: bool = True
) -> Dict[str, Any]:
    """
    Generate avatar video using HeyGen API.
    
    Args:
        text: Text to convert to video
        avatar_id: Avatar ID (default: DEFAULT_AVATAR_ID)
        voice_id: Voice ID (default: DEFAULT_VOICE_ID)
        return_base64: If True, download video and return as base64. If False, return URL.
    
    Returns:
        Dict with video_url or video_base64, or error dict
    """
    if not HEYGEN_API_KEY:
        return {
            "error": "HeyGen API key not configured",
            "configured": False
        }
    
    if not text or not text.strip():
        return {
            "error": "Text is required for avatar video generation"
        }
    
    # Sanitize text before sending
    text = sanitize_question_for_avatar(text)
    
    try:
        # Step 1: Create video generation request
        url = "https://api.heygen.com/v2/video/generate"
        headers = {
            "X-Api-Key": HEYGEN_API_KEY,
            "Content-Type": "application/json",
        }
        
        payload = {
            "video_inputs": [{
                "character": {
                    "type": "avatar",
                    "avatar_id": avatar_id or StreamingConfig.DEFAULT_AVATAR_ID,
                    "avatar_style": "normal"
                },
                "voice": {
                    "type": "text",
                    "input_text": text,
                    "voice_id": voice_id or StreamingConfig.DEFAULT_VOICE_ID,
                    "speed": 1.0
                }
            }],
            "dimension": {
                "width": StreamingConfig.AVATAR_VIDEO_WIDTH,
                "height": StreamingConfig.AVATAR_VIDEO_HEIGHT
            },
            "aspect_ratio": StreamingConfig.AVATAR_VIDEO_ASPECT_RATIO,
            "test": False
        }
        
        log.info(f"Requesting HeyGen video generation for text: {text[:50]}...")
        client = await get_http_client()
        response = await client.post(url, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        
        if "data" not in data or "video_id" not in data["data"]:
            error_msg = data.get("message", "Unknown error")
            log.error(f"HeyGen video generation failed: {error_msg}")
            return {
                "error": f"Failed to create video: {error_msg}",
                "heygen_response": data
            }
        
        video_id = data["data"]["video_id"]
        log.info(f"HeyGen video ID: {video_id}")
        
        # Step 2: Poll for video status
        max_attempts = AVATAR_GENERATION_TIMEOUT
        
        for attempt in range(max_attempts):
            await asyncio.sleep(StreamingConfig.AVATAR_POLL_INTERVAL)
            
            try:
                status_data = await _poll_video_status(video_id, headers)
                
                if "data" in status_data:
                    status = status_data["data"].get("status")
                    
                    if status == "completed":
                        video_url = status_data["data"].get("video_url")
                        log.info(f"HeyGen video completed: {video_url}")
                        
                        if return_base64:
                            # Download video and convert to base64
                            try:
                                video_response = await client.get(video_url, timeout=30.0)
                                video_response.raise_for_status()
                                video_base64 = base64.b64encode(video_response.content).decode('utf-8')
                                log.info(f"Video converted to base64: {len(video_base64)} chars")
                                return {
                                    "video_base64": video_base64,
                                    "video_url": video_url,
                                    "format": "mp4",
                                    "video_id": video_id
                                }
                            except Exception as download_error:
                                log.exception(f"Error downloading video: {download_error}")
                                # Fallback to URL if base64 conversion fails
                                if AVATAR_FALLBACK_TO_URL:
                                    return {
                                        "video_url": video_url,
                                        "video_id": video_id,
                                        "error": f"Failed to convert to base64: {str(download_error)}"
                                    }
                                return {
                                    "error": f"Failed to download video: {str(download_error)}",
                                    "video_id": video_id
                                }
                        else:
                            return {
                                "video_url": video_url,
                                "video_id": video_id
                            }
                    elif status == "failed":
                        error_msg = status_data["data"].get("error", {}).get("message", "Video generation failed")
                        log.error(f"HeyGen video generation failed: {error_msg}")
                        return {
                            "error": error_msg,
                            "video_id": video_id
                        }
            except Exception as status_error:
                log.warning(f"Error checking video status (attempt {attempt + 1}): {status_error}")
                # Continue polling
        
        log.warning(f"HeyGen video generation timed out after {max_attempts} attempts")
        return {
            "error": "Video generation timed out",
            "video_id": video_id
        }
        
    except Exception as e:
        log.exception(f"Error generating avatar video: {e}")
        return {
            "error": f"Error generating avatar: {str(e)}"
        }


async def generate_avatar_video_streaming(
    text: str,
    avatar_id: Optional[str] = None,
    voice_id: Optional[str] = None,
    use_streaming_api: bool = True
) -> AsyncIterator[Dict[str, Any]]:
    """
    Generate avatar video with streaming status updates via SSE.
    
    Supports two modes:
    1. HeyGen Streaming API (WebRTC) - Real-time streaming, low latency
    2. Video Generation API (MP4) - Pre-generated videos with polling
    
    Args:
        text: Text to convert to video
        avatar_id: Avatar ID (default: DEFAULT_AVATAR_ID)
        voice_id: Voice ID (default: DEFAULT_VOICE_ID)
        use_streaming_api: If True, use HeyGen Streaming API (WebRTC). If False, use Video Generation API (MP4)
    
    Yields:
        Dict with status updates and video data
    """
    # Try HeyGen Streaming API first (real-time WebRTC streaming)
    if use_streaming_api and StreamingConfig.AVATAR_USE_STREAMING_API:
        try:
            from .heygen_streaming_api import generate_avatar_video_streaming_api
            log.info("Using HeyGen Streaming API (WebRTC) for real-time streaming")
            async for update in generate_avatar_video_streaming_api(text, avatar_id, voice_id):
                yield update
            return
        except ImportError:
            log.warning("HeyGen Streaming API module not available, falling back to Video Generation API")
        except Exception as e:
            log.warning(f"HeyGen Streaming API failed: {e}, falling back to Video Generation API")
    
    # Fallback to Video Generation API (MP4 with polling)
    log.info("Using HeyGen Video Generation API (MP4) with optimized polling")
    # Mock mode for testing
    if HEYGEN_MOCK_MODE:
        yield {
            "type": "status",
            "message": "Mock mode: Generating video...",
            "stage": "initiating"
        }
        await asyncio.sleep(0.1)
        yield {
            "type": "status",
            "message": "Mock mode: Video completed",
            "stage": "processing",
            "progress": 1.0
        }
        yield {
            "type": "video",
            "video_base64": "mock_base64_video_data",
            "video_url": "https://mock.heygen.com/video.mp4",
            "format": "mp4",
            "video_id": "mock_video_id",
            "size_bytes": 1000
        }
        return
    
    if not HEYGEN_API_KEY:
        yield {
            "type": "error",
            "message": "HeyGen API key not configured"
        }
        return
    
    if not AVATAR_ENABLED:
        yield {
            "type": "error",
            "message": "Avatar generation is disabled"
        }
        return
    
    try:
        # Sanitize text before sending
        text = sanitize_question_for_avatar(text)
        
        # Step 1: Create video generation request
        url = "https://api.heygen.com/v2/video/generate"
        headers = {
            "X-Api-Key": HEYGEN_API_KEY,
            "Content-Type": "application/json",
        }
        
        payload = {
            "video_inputs": [{
                "character": {
                    "type": "avatar",
                    "avatar_id": avatar_id or StreamingConfig.DEFAULT_AVATAR_ID,
                    "avatar_style": "normal"
                },
                "voice": {
                    "type": "text",
                    "input_text": text,
                    "voice_id": voice_id or StreamingConfig.DEFAULT_VOICE_ID,
                    "speed": 1.0
                }
            }],
            "dimension": {
                "width": StreamingConfig.AVATAR_VIDEO_WIDTH,
                "height": StreamingConfig.AVATAR_VIDEO_HEIGHT
            },
            "aspect_ratio": StreamingConfig.AVATAR_VIDEO_ASPECT_RATIO,
            "test": False
        }
        
        yield {
            "type": "status",
            "message": "Requesting video generation...",
            "stage": "initiating",
            "progress": 0.0
        }
        
        client = await get_http_client()
        response = await client.post(url, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        
        if "data" not in data or "video_id" not in data["data"]:
            error_msg = data.get("message", "Unknown error")
            yield {
                "type": "error",
                "message": f"Failed to create video: {error_msg}"
            }
            return
        
        video_id = data["data"]["video_id"]
        yield {
            "type": "status",
            "message": "Video generation started",
            "stage": "processing",
            "video_id": video_id,
            "progress": 0.05
        }
        
        # Step 2: Poll for video status with optimized exponential backoff polling
        max_attempts = AVATAR_GENERATION_TIMEOUT
        start_time = time.time()
        
        # Exponential backoff: start fast (0.5s), gradually increase to max (2.0s)
        # This reduces API calls while maintaining responsiveness
        for attempt in range(max_attempts):
            # Calculate exponential backoff interval
            if attempt == 0:
                # First poll: immediate (no sleep)
                poll_interval = 0.0
            elif attempt < 5:
                # First 5 polls: fast (0.5s)
                poll_interval = StreamingConfig.AVATAR_POLL_INTERVAL_INITIAL
            else:
                # After 5 polls: exponential backoff (0.5s * 1.2^attempt, capped at 2.0s)
                exponential = StreamingConfig.AVATAR_POLL_INTERVAL_INITIAL * (1.2 ** min(attempt - 5, 10))
                poll_interval = min(exponential, StreamingConfig.AVATAR_POLL_INTERVAL_MAX)
            
            if poll_interval > 0:
                await asyncio.sleep(poll_interval)
            
            try:
                status_data = await _poll_video_status(video_id, headers)
                
                if "data" in status_data:
                    status = status_data["data"].get("status")
                    elapsed = time.time() - start_time
                    progress = min(0.95, elapsed / (max_attempts * StreamingConfig.AVATAR_POLL_INTERVAL_MAX))
                    estimated_remaining = max(0, int((max_attempts - attempt - 1) * poll_interval))
                    
                    # Send progress update with enhanced information
                    yield {
                        "type": "status",
                        "message": f"Processing video... ({attempt + 1}/{max_attempts})",
                        "stage": "processing",
                        "progress": progress,
                        "progress_percent": round(progress * 100, 1),
                        "estimated_seconds_remaining": estimated_remaining,
                        "elapsed_seconds": round(elapsed, 1)
                    }
                    
                    if status == "completed":
                        video_url = status_data["data"].get("video_url")
                        
                        # TRUE STREAMING: Stream video download immediately when URL is available
                        # Don't wait for full download - stream chunks as they arrive
                        if StreamingConfig.AVATAR_USE_DIRECT_URL:
                            # OPTION 1: Direct URL (browser handles streaming) - fastest
                            try:
                                # Quick HEAD request to get size (non-blocking)
                                try:
                                    head_response = await client.head(video_url, timeout=5.0)
                                    content_length = head_response.headers.get("content-length")
                                    size_bytes = int(content_length) if content_length else 0
                                except Exception:
                                    size_bytes = 0
                                
                                # Send video URL immediately - browser will stream it
                                yield {
                                    "type": "video",
                                    "video_url": video_url,
                                    "format": "mp4",
                                    "video_id": video_id,
                                    "size_bytes": size_bytes,
                                    "delivery_method": "direct_url",  # Browser handles streaming
                                    "is_complete": True
                                }
                                log.info(f"Video URL sent immediately: {video_url} (size: {size_bytes} bytes, browser streaming)")
                                return
                            except Exception as url_error:
                                log.warning(f"Error sending direct URL, falling back to chunked download: {url_error}")
                                # Fall through to chunked download
                        
                        # OPTION 2: Progressive download streaming (true streaming via SSE)
                        # Stream video chunks as they download from HeyGen CDN
                        yield {
                            "type": "status",
                            "message": "Video ready, streaming download...",
                            "stage": "downloading",
                            "video_url": video_url,
                            "progress": 0.95
                        }
                        
                        try:
                            # Stream video download immediately - don't wait for full download
                            async with client.stream('GET', video_url, timeout=60.0) as video_stream:
                                video_stream.raise_for_status()
                                
                                # Get content length if available
                                content_length = video_stream.headers.get("content-length")
                                size_bytes = int(content_length) if content_length else 0
                                
                                # Send start event immediately
                                yield {
                                    "type": "video",
                                    "video_url": video_url,
                                    "format": "mp4",
                                    "video_id": video_id,
                                    "size_bytes": size_bytes,
                                    "is_streaming": True,
                                    "is_start": True,
                                    "delivery_method": "progressive_download"  # True streaming
                                }
                                
                                chunk_buffer = b''
                                total_size = 0
                                chunk_index = 0
                                
                                # Stream chunks as they arrive (true streaming)
                                async for chunk in video_stream.aiter_bytes():
                                    chunk_buffer += chunk
                                    total_size += len(chunk)
                                    
                                    # Send chunks in reasonable sizes for SSE (base64 encoded)
                                    if len(chunk_buffer) >= 7500:  # ~10KB base64
                                        base64_chunk = base64.b64encode(chunk_buffer).decode('utf-8')
                                        yield {
                                            "type": "video",
                                            "video_base64": base64_chunk,
                                            "video_url": video_url,
                                            "format": "mp4",
                                            "video_id": video_id,
                                            "chunk_index": chunk_index,
                                            "chunk_size": len(base64_chunk),
                                            "is_streaming": True,
                                            "delivery_method": "progressive_download"
                                        }
                                        chunk_buffer = b''
                                        chunk_index += 1
                                
                                # Send remaining buffer
                                if chunk_buffer:
                                    base64_chunk = base64.b64encode(chunk_buffer).decode('utf-8')
                                    yield {
                                        "type": "video",
                                        "video_base64": base64_chunk,
                                        "video_url": video_url,
                                        "format": "mp4",
                                        "video_id": video_id,
                                        "chunk_index": chunk_index,
                                        "size_bytes": total_size if not size_bytes else size_bytes,
                                        "chunk_size": len(base64_chunk),
                                        "is_streaming": True,
                                        "is_last": True,
                                        "delivery_method": "progressive_download"
                                    }
                                
                                # Send completion event
                                yield {
                                    "type": "video",
                                    "video_url": video_url,
                                    "format": "mp4",
                                    "video_id": video_id,
                                    "size_bytes": total_size if not size_bytes else size_bytes,
                                    "is_complete": True,
                                    "delivery_method": "progressive_download"
                                }
                                log.info(f"Video streaming completed: {video_url} ({total_size} bytes streamed)")
                                return
                        except Exception:
                            # Legacy mode: download and send base64 chunks (fallback)
                            yield {
                                "type": "status",
                                "message": "Video completed, downloading...",
                                "stage": "downloading",
                                "video_url": video_url,
                                "progress": 0.95
                            }
                            
                            # Stream video download and chunk base64 encoding
                            try:
                                async with client.stream('GET', video_url, timeout=30.0) as video_stream:
                                    video_stream.raise_for_status()
                                    
                                    # Get content length if available
                                    content_length = video_stream.headers.get("content-length")
                                    size_bytes = int(content_length) if content_length else 0
                                    
                                    # Send start event before streaming chunks
                                    yield {
                                        "type": "video",
                                        "video_url": video_url,
                                        "format": "mp4",
                                        "video_id": video_id,
                                        "size_bytes": size_bytes,
                                        "is_streaming": True,
                                        "is_start": True,
                                        "delivery_method": "base64_chunks"
                                    }
                                    
                                    chunk_buffer = b''
                                    total_size = 0
                                    
                                    async for chunk in video_stream.aiter_bytes():
                                        chunk_buffer += chunk
                                        total_size += len(chunk)
                                        
                                        # When buffer reaches chunk size, encode and yield
                                        if len(chunk_buffer) >= 7500:  # ~10KB base64
                                            base64_chunk = base64.b64encode(chunk_buffer).decode('utf-8')
                                            yield {
                                                "type": "video",
                                                "video_base64": base64_chunk,
                                                "video_url": video_url,
                                                "format": "mp4",
                                                "video_id": video_id,
                                                "chunk_size": len(base64_chunk),
                                                "is_streaming": True
                                            }
                                            chunk_buffer = b''
                                    
                                    # Encode remaining buffer
                                    if chunk_buffer:
                                        base64_chunk = base64.b64encode(chunk_buffer).decode('utf-8')
                                        yield {
                                            "type": "video",
                                            "video_base64": base64_chunk,
                                            "video_url": video_url,
                                            "format": "mp4",
                                            "video_id": video_id,
                                            "size_bytes": total_size if not size_bytes else size_bytes,
                                            "chunk_size": len(base64_chunk),
                                            "is_streaming": True,
                                            "is_last": True
                                        }
                                    
                                    # Send final complete event
                                    yield {
                                        "type": "video",
                                        "video_url": video_url,
                                        "format": "mp4",
                                        "video_id": video_id,
                                        "size_bytes": total_size if not size_bytes else size_bytes,
                                        "is_complete": True
                                    }
                                    return
                                    
                            except Exception as download_error:
                                log.exception(f"Error downloading video: {download_error}")
                                # Fallback: send URL if download fails
                                if AVATAR_FALLBACK_TO_URL:
                                    yield {
                                        "type": "video",
                                        "video_url": video_url,
                                        "format": "mp4",
                                        "video_id": video_id,
                                        "delivery_method": "direct_url_fallback",
                                        "is_complete": True,
                                        "warning": "Download failed, using direct URL"
                                    }
                                    return
                                else:
                                    yield {
                                        "type": "error",
                                        "message": f"Failed to download video: {str(download_error)}",
                                        "video_url": video_url
                                    }
                                    return
                    elif status == "failed":
                        error_msg = status_data["data"].get("error", {}).get("message", "Video generation failed")
                        yield {
                            "type": "error",
                            "message": error_msg,
                            "video_id": video_id
                        }
                        return
            except Exception as status_error:
                log.warning(f"Error checking video status: {status_error}")
                # Continue polling with backoff
        
        yield {
            "type": "error",
            "message": "Video generation timed out",
            "video_id": video_id
        }
        
    except Exception as e:
        log.exception(f"Error in streaming avatar generation: {e}")
        # Graceful degradation: yield error but don't fail completely
        yield {
            "type": "error",
            "message": f"Error generating avatar: {str(e)}",
            "fallback_message": "Avatar unavailable, using text-only mode"
        }

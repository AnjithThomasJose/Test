"""
Callback utilities for interview agent.

Handles sending callback notifications to external servers.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from utils.interview_utils import mask_sensitive_data
from core.security import validate_callback_url

log = logging.getLogger(__name__)


def _normalize_indexed_char_dict(value: Any) -> Any:
    """
    Normalize JS-style '{\"0\": \"t\", \"1\": \"h\", ...}' char maps into strings.

    Some clients accidentally spread a string into an object on the JS/Node side
    (e.g. `{ ...response }`), which results in payloads like:

        { "response": { "0": "t", "1": "h", "2": "a", ... } }

    This helper detects that pattern and converts it back to a string:

        { "response": "thank you" }

    The detection is conservative:
    - keys must be digit-only strings
    - values must all be strings
    """
    if not isinstance(value, dict) or not value:
        return value

    if all(isinstance(k, str) and k.isdigit() for k in value.keys()) and \
       all(isinstance(v, str) for v in value.values()):
        try:
            chars = [value[str(i)] for i in sorted((int(k) for k in value.keys()))]
            return "".join(chars)
        except Exception:
            return value

    return value


def _normalize_payload(data: Any) -> Any:
    """
    Recursively normalize callback payload data.

    - Detects and fixes char-map objects created by spreading strings in JS
    - Traverses dicts and lists to apply normalization deeply
    """
    normalized = _normalize_indexed_char_dict(data)
    if normalized is not data:
        return normalized

    if isinstance(data, dict):
        return {k: _normalize_payload(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_normalize_payload(v) for v in data]
    return data


async def send_callback_notification(
    callback_url: str,
    uid: str,
    session_id: str,
    status: str,
    evaluation_summary: Optional[Dict[str, Any]] = None,
    detailed_summary: Optional[str] = None,
    structured_report: Optional[Dict[str, Any]] = None,
    auth_token: Optional[str] = None,
    conversation_context: Optional[Dict[str, Any]] = None,
    response_analysis: Optional[Dict[str, Any]] = None,
    session_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Send callback notification to the provided URL.
    
    This function sends a PATCH request to the callback URL with interview
    results and metadata. It handles authentication, error logging, and
    response validation.
    
    Args:
        callback_url: URL to send callback to
        uid: User ID
        session_id: Interview session ID
        status: Interview status (e.g., "completed", "in_progress")
        evaluation_summary: Optional evaluation summary dictionary
        detailed_summary: Optional detailed text summary
        auth_token: Optional authentication token
        conversation_context: Optional conversation context dictionary
        response_analysis: Optional response analysis dictionary
        session_metadata: Optional session metadata dictionary
    """
    try:
        # Validate callback URL whitelist
        is_valid, error_msg = validate_callback_url(callback_url)
        if not is_valid:
            log.error(f"Callback URL validation failed: {error_msg} | url={callback_url} | uid={uid}")
            return
        
        import aiohttp
        
        # Format callback data to match callback_validator.py schema
        # Send structured_report if available, otherwise fallback to evaluation_summary
        callback_data = {
            "node": "interview_agent",
            "status": status,
            "output": {
                "uid": uid,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "evaluation_summary": evaluation_summary,  # Keep for backward compatibility
                "detailed_summary": detailed_summary,
                "structured_report": structured_report,  # New UI-formatted report
                "conversation_context": conversation_context,
                "response_analysis": response_analysis,
                "session_metadata": session_metadata
            }
        }
        
        # Normalize payload to handle any JS char-map objects (e.g. {\"0\": \"t\", \"1\": \"h\", ...})
        callback_data = _normalize_payload(callback_data)

        # Log callback data
        log.debug(f"Callback data - UID: {uid}, Session: {session_id}, Status: {status}")
        
        if evaluation_summary:
            overall_score = evaluation_summary.get('overall_score', 'N/A')
            eval_status = evaluation_summary.get('status', 'N/A')
            questions_asked = evaluation_summary.get('questions_asked', 'N/A')
            log.info(
                f"Evaluation summary - Overall Score: {overall_score}, "
                f"Status: {eval_status}, Questions: {questions_asked}"
            )
        
        # Prepare headers with auth token if provided
        headers = {
            "Content-Type": "application/json"
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
            # Mask token in logs
            masked_token = mask_sensitive_data({"auth_token": auth_token})["auth_token"]
            log.debug(f"Using auth token for callback: {masked_token}")
        else:
            log.warning("No auth token provided for callback")
        
        # Send callback request
        async with aiohttp.ClientSession() as session:
            async with session.patch(
                callback_url,
                json=callback_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response_text = await response.text()
                if response.status == 200:
                    log.info(f"Callback sent successfully to {callback_url}")
                else:
                    log.warning(
                        f"Callback failed with status {response.status}: {response_text}"
                    )
    
    except Exception as e:
        log.error(f"Error sending callback to {callback_url}: {e}", exc_info=True)


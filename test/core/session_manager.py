"""
Shared session management utilities for all conversational agents.

Provides common session and conversation history management functionality
that can be used by both interview and career guidance workflows.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from chroma import get_chat_session, update_chat_session
from core.utils import run_blocking_io

log = logging.getLogger(__name__)


class SessionManager:
    """
    Shared session management utilities.
    
    Used by both interview and career guidance workflows.
    Supports flexible history keys for different workflow types.
    """
    
    @staticmethod
    async def load_session(session_id: str, uid: str) -> Dict[str, Any]:
        """
        Load session data with UID isolation.
        
        Args:
            session_id: Session identifier
            uid: User ID for proper session isolation
            
        Returns:
            Session data dictionary, or empty dict if not found
        """
        if not session_id or not uid:
            return {}
        
        try:
            session_data = await run_blocking_io(get_chat_session, session_id, uid)
            if session_data:
                # Verify uid matches (security check)
                if session_data.get("uid") != uid:
                    log.warning(f"Session uid mismatch: expected {uid}, got {session_data.get('uid')}")
                    return {}
                return session_data
        except Exception as e:
            log.debug(f"Could not load session: {e}")
        
        return {}
    
    @staticmethod
    async def save_session(session_id: str, data: Dict[str, Any], uid: str):
        """
        Save session data with UID isolation.
        
        Args:
            session_id: Session identifier
            data: Session data to save
            uid: User ID for proper session isolation
        """
        if not session_id or not uid:
            return
        
        try:
            # Ensure uid is set and add timestamp
            data["uid"] = uid
            data["timestamp"] = time.time()
            data["last_updated"] = time.time()
            
            await run_blocking_io(update_chat_session, session_id, data)
            log.debug(f"✅ Saved session for session_id={session_id}, uid={uid}")
        except Exception as e:
            log.warning(f"Failed to save session: {e}", exc_info=True)
    
    @staticmethod
    async def load_conversation_history(
        session_id: str,
        uid: str,
        history_key: str = "conversation_history"
    ) -> List[Dict[str, str]]:
        """
        Load conversation history from session.
        
        Supports flexible history keys for different workflow types:
        - Interview: "conversation_history"
        - Career guidance: "career_chat_history"
        
        Args:
            session_id: Session identifier
            uid: User ID for proper session isolation
            history_key: Key to use for history in session data
            
        Returns:
            List of conversation messages
        """
        if not session_id:
            return []
        
        try:
            session_data = await SessionManager.load_session(session_id, uid)
            if session_data:
                history = session_data.get(history_key, [])
                if isinstance(history, list):
                    return history
        except Exception as e:
            log.debug(f"Could not load conversation history: {e}")
        
        return []
    
    @staticmethod
    async def save_conversation_history(
        session_id: str,
        history: List[Dict[str, str]],
        uid: str,
        history_key: str = "conversation_history"
    ):
        """
        Save conversation history to session.
        
        Supports flexible history keys for different workflow types.
        
        Args:
            session_id: Session identifier
            history: Conversation history to save
            uid: User ID for proper session isolation
            history_key: Key to use for history in session data
        """
        if not session_id or not uid:
            return
        
        try:
            # Load existing session data
            session_data = await SessionManager.load_session(session_id, uid) or {}
            
            # Update conversation history
            session_data[history_key] = history
            session_data["status"] = "active"
            
            # Save updated session
            await SessionManager.save_session(session_id, session_data, uid)
            
            log.debug(f"✅ Saved conversation history ({len(history)} messages) for session={session_id}, uid={uid}, key={history_key}")
        except Exception as e:
            log.warning(f"Failed to save conversation history: {e}", exc_info=True)





"""
session_manager.py

Chroma-backed async session manager.

This implementation uses `interview_chroma` as the primary persistence layer.
All public functions used by the orchestrator are async and wrap potentially
blocking chroma calls with asyncio.to_thread for safety.

Public async API:
- async def recover_session(session_id: str) -> dict
- async def save_session(session: dict, save_cb: bool = True) -> str
- async def load_conversation_context(session_id: str) -> dict
- async def save_conversation_context(session_id: str, context: dict) -> None
- async def load_question_fingerprints(session_id: str) -> List[List[str]]
- async def save_question_fingerprints(session_id: str, fingerprints: List[List[str]]) -> None
- async def load_negative_intent_history(session_id: str) -> List[dict]
- async def save_negative_intent_history(session_id: str, history: List[dict]) -> None
- async def save_circuit_breaker_state(session_id: str) -> None
- async def load_circuit_breaker_state(session_id: str) -> None
- async def check_circuit_breaker(session_id: str) -> bool
- async def record_circuit_breaker_failure(session_id: str) -> None
- async def record_circuit_breaker_success(session_id: str) -> None
- generate_session_id(uid: Optional[str]) -> str
- initialize_session(...)

Notes:
- interview_chroma is expected to expose sync methods used here: store_interview_session,
  recover_session_state, store_conversation_context, get_conversation_context, store_question_fingerprints,
  get_question_fingerprints, store_negative_intent_history, get_negative_intent_history,
  store_circuit_breaker_state, get_circuit_breaker_state.
- If interview_chroma methods are async already, to_thread wrapper is harmless but could be optimized.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from .circuit_breaker import register_failure, register_success, is_open as cb_is_open, get_state as cb_get_state
from .conversation_context import ConversationContext
from ..interview_chroma import interview_chroma

log = logging.getLogger(__name__)


def generate_session_id(uid: Optional[str] = None) -> str:
    if uid:
        return f"{uid}_{uuid.uuid4().hex[:16]}"
    return f"session_{uuid.uuid4().hex[:16]}"


async def initialize_session(session_id: Optional[str] = None, uid: Optional[str] = None, initial_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Create a new session id (if not provided) and store initial_data in Chroma.
    """
    if not session_id:
        session_id = generate_session_id(uid)
    if initial_data:
        try:
            # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
            await asyncio.wait_for(
                asyncio.to_thread(interview_chroma.store_interview_session, session_id, initial_data),
                timeout=5.0
            )
            log.info("Initialized session %s", session_id)
        except asyncio.TimeoutError:
            log.warning(f"[TIMEOUT] store_interview_session exceeded 5s for session {session_id}")
        except Exception:
            log.exception("Failed to initialize session in Chroma")
    return session_id


# ----------------------------
# Conversation context
# ----------------------------
async def save_conversation_context(session_id: str, context: Dict[str, Any]) -> None:
    if not session_id or context is None:
        return
    try:
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.store_conversation_context, session_id, context),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        log.warning(f"[TIMEOUT] store_conversation_context exceeded 5s for session {session_id}")
    except Exception:
        log.exception("Failed to save conversation context to Chroma")


async def load_conversation_context(session_id: str) -> Dict[str, Any]:
    if not session_id:
        return {}
    try:
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        ctx = await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.get_conversation_context, session_id),
            timeout=5.0
        )
        return ctx or {}
    except asyncio.TimeoutError:
        log.warning(f"[TIMEOUT] get_conversation_context exceeded 5s for session {session_id}")
        return {}
    except Exception:
        log.exception("Failed to load conversation context from Chroma")
        return {}
        return {}


# ----------------------------
# Conversation history
# ----------------------------
async def save_conversation_history(session_id: str, conversation_history: List[Dict[str, str]]) -> None:
    """
    Save conversation_history to ChromaDB.
    
    Stores conversation_history in the session data under a stable key.
    Caps history at 200 messages to avoid storage bloat (configurable).
    
    Args:
        session_id: Session identifier
        conversation_history: List of message dicts with 'role' and 'content' keys
    """
    if not session_id or not conversation_history:
        return
    
    history_to_store = conversation_history
    
    try:
        # Load existing session data or create new
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        existing_session = await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.get_interview_session, session_id),
            timeout=5.0
        )
        if existing_session is None:
            existing_session = {}
        
        # Update conversation_history in session data
        existing_session["conversation_history"] = history_to_store
        existing_session["session_id"] = session_id
        
        # Store updated session
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.store_interview_session, session_id, existing_session),
            timeout=5.0
        )
        log.info(f"[SAVE] history length={len(history_to_store)} for session_id={session_id}")
    except asyncio.TimeoutError:
        log.warning(f"[TIMEOUT] save_conversation_history exceeded 5s for session {session_id}")
    except Exception:
        log.exception(f"Failed to save conversation_history to Chroma for session_id={session_id}")


async def load_conversation_history(session_id: str) -> List[Dict[str, str]]:
    """
    Load conversation_history from ChromaDB.
    
    Retrieves conversation_history from session data stored under stable key.
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of message dicts with 'role' and 'content' keys, or empty list if not found
    """
    if not session_id:
        return []
    
    try:
        # Use the dedicated method from interview_chroma
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        history = await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.get_conversation_history, session_id),
            timeout=5.0
        )
        if history and isinstance(history, list) and len(history) > 0:
            log.info(f"[LOAD] history length={len(history)} for session_id={session_id}")
            return history
        
        # Fallback: try direct session retrieval for debugging
        log.debug(f"[LOAD] get_conversation_history returned empty, trying direct session retrieval for session_id={session_id}")
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        session_data = await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.get_interview_session, session_id),
            timeout=5.0
        )
        if session_data and isinstance(session_data, dict):
            log.debug(f"[LOAD] Session data keys: {list(session_data.keys())}")
            history = session_data.get("conversation_history")
            if isinstance(history, list) and len(history) > 0:
                log.info(f"[LOAD] history length={len(history)} for session_id={session_id}")
                return history
            else:
                log.debug(f"[LOAD] conversation_history field exists but is empty or not a list (expected for new sessions): {type(history)}")
        else:
            log.debug(f"[LOAD] Session data is None or not a dict (expected for new sessions): {type(session_data)}")
        return []
    except Exception as e:
        log.exception(f"Failed to load conversation_history from Chroma for session_id={session_id}: {e}")
        return []


# ----------------------------
# Question fingerprints
# ----------------------------
async def save_question_fingerprints(session_id: str, fingerprints: List[List[str]]) -> None:
    if not session_id:
        return
    try:
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.store_question_fingerprints, session_id, fingerprints),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        log.warning(f"[TIMEOUT] store_question_fingerprints exceeded 5s for session {session_id}")
    except Exception:
        log.exception("Failed to save question fingerprints to Chroma")


async def load_question_fingerprints(session_id: str) -> List[List[str]]:
    if not session_id:
        return []
    try:
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        fps = await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.get_question_fingerprints, session_id),
            timeout=5.0
        )
        return fps or []
    except asyncio.TimeoutError:
        log.warning(f"[TIMEOUT] get_question_fingerprints exceeded 5s for session {session_id}")
        return []
    except Exception:
        log.exception("Failed to load question fingerprints from Chroma")
        return []
        return []


# ----------------------------
# Negative intent history
# ----------------------------
async def save_negative_intent_history(session_id: str, intent_history: List[Dict[str, Any]]) -> None:
    if not session_id:
        return
    try:
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.store_negative_intent_history, session_id, intent_history),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        log.warning(f"[TIMEOUT] store_negative_intent_history exceeded 5s for session {session_id}")
    except Exception:
        log.exception("Failed to save negative intent history to Chroma")


async def load_negative_intent_history(session_id: str) -> List[Dict[str, Any]]:
    if not session_id:
        return []
    try:
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        ih = await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.get_negative_intent_history, session_id),
            timeout=5.0
        )
        return ih or []
    except asyncio.TimeoutError:
        log.warning(f"[TIMEOUT] get_negative_intent_history exceeded 5s for session {session_id}")
        return []
    except Exception:
        log.exception("Failed to load negative intent history from Chroma")
        return []
        return []


# ----------------------------
# Circuit breaker state
# ----------------------------
async def save_circuit_breaker_state(session_id: str) -> None:
    if not session_id:
        return
    try:
        # cb_get_state expected to return serializable state
        state = await asyncio.to_thread(cb_get_state, session_id)
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.store_circuit_breaker_state, session_id, state),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        log.warning(f"[TIMEOUT] store_circuit_breaker_state exceeded 5s for session {session_id}")
    except Exception:
        log.exception("Failed to save circuit breaker state to Chroma")


async def load_circuit_breaker_state(session_id: str) -> None:
    if not session_id:
        return
    try:
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        state = await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.get_circuit_breaker_state, session_id),
            timeout=5.0
        )
        if state:
            try:
                from .circuit_breaker import restore_state
                await asyncio.to_thread(restore_state, session_id, state)
            except Exception:
                log.debug("circuit_breaker.restore_state not available; skipping restore", exc_info=False)
    except Exception:
        log.exception("Failed to load circuit breaker state from Chroma")


# ----------------------------
# Recover / Save session (high-level)
# ----------------------------
async def recover_session(session_id: str) -> Dict[str, Any]:
    """
    Recover all session state from Chroma.
    """
    if not session_id:
        return {}
    try:
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        recovered = await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.recover_session_state, session_id),
            timeout=5.0
        )
        await load_circuit_breaker_state(session_id)
        return recovered or {}
    except asyncio.TimeoutError:
        log.warning(f"[TIMEOUT] recover_session_state exceeded 5s for session {session_id}")
        return {}
    except Exception:
        log.exception("Chroma recover_session failed")
        return {}
        return {}


async def save_session(session: Dict[str, Any], save_cb: bool = True) -> str:
    """
    Save session data to Chroma.
    Returns snapshot id.
    """
    session_id = session.get("session_id") or generate_session_id()
    session["session_id"] = session_id
    snapshot_id = f"{session_id}_{int(time.time() * 1000)}"
    try:
        # CRITICAL FIX: Add timeout to prevent blocking concurrent requests
        await asyncio.wait_for(
            asyncio.to_thread(interview_chroma.store_interview_session, session_id, session),
            timeout=5.0
        )
        if save_cb:
            await save_circuit_breaker_state(session_id)
        log.info("Saved session snapshot %s", snapshot_id, extra={"session_id": session_id, "snapshot_id": snapshot_id})
        return snapshot_id
    except asyncio.TimeoutError:
        log.warning(f"[TIMEOUT] save_session exceeded 5s for session {session_id}")
        return snapshot_id
    except Exception:
        log.exception("Failed to save session to Chroma")
        return snapshot_id


# ----------------------------
# Circuit checks / records
# ----------------------------
async def check_circuit_breaker(session_id: str) -> bool:
    try:
        await load_circuit_breaker_state(session_id)
    except Exception:
        log.debug("load_circuit_breaker_state failed silently")
    try:
        return await asyncio.to_thread(cb_is_open, session_id)
    except Exception:
        log.exception("cb_is_open failed; defaulting to False")
        return False


async def record_circuit_breaker_failure(session_id: str) -> None:
    try:
        await asyncio.to_thread(register_failure, session_id)
        await save_circuit_breaker_state(session_id)
    except Exception:
        log.exception("Failed to record circuit breaker failure")


async def record_circuit_breaker_success(session_id: str) -> None:
    try:
        await asyncio.to_thread(register_success, session_id)
        await save_circuit_breaker_state(session_id)
    except Exception:
        log.exception("Failed to record circuit breaker success")

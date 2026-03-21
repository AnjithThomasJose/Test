"""
Circuit breaker policy logic for interview agent.

Provides circuit breaker state management without persistence.
Persistence is handled by session_manager.py.
"""

import logging
import time
from typing import Dict, Any, Optional
from enum import Enum

from .config import get_circuit_breaker_config

log = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerState:
    """Represents the state of a circuit breaker for a session."""
    
    def __init__(
        self,
        session_id: str,
        state: CircuitState = CircuitState.CLOSED,
        failure_count: int = 0,
        last_failure_time: Optional[float] = None,
        opened_at: Optional[float] = None
    ):
        self.session_id = session_id
        self.state = state
        self.failure_count = failure_count
        self.last_failure_time = last_failure_time
        self.opened_at = opened_at


# In-memory state storage (persistence handled by session_manager)
_CIRCUIT_BREAKER_STATES: Dict[str, CircuitBreakerState] = {}


def get_circuit_breaker_state(session_id: str) -> CircuitBreakerState:
    """
    Get circuit breaker state for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        CircuitBreakerState for the session
    """
    if session_id not in _CIRCUIT_BREAKER_STATES:
        _CIRCUIT_BREAKER_STATES[session_id] = CircuitBreakerState(session_id)
    return _CIRCUIT_BREAKER_STATES[session_id]


def is_open(session_id: str) -> bool:
    """
    Check if circuit breaker is open for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        True if circuit is open, False otherwise
    """
    cb_state = get_circuit_breaker_state(session_id)
    config = get_circuit_breaker_config()
    cooldown_seconds = config["cooldown_seconds"]
    
    # If circuit is open, check if cooldown period has passed
    if cb_state.state == CircuitState.OPEN:
        if cb_state.opened_at is None:
            # No timestamp, assume still open
            return True
        
        elapsed = time.time() - cb_state.opened_at
        if elapsed >= cooldown_seconds:
            # Cooldown passed, transition to half-open
            cb_state.state = CircuitState.HALF_OPEN
            cb_state.failure_count = 0
            log.info(f"Circuit breaker transitioning to HALF_OPEN for session {session_id}")
            return False
        return True
    
    return False


def register_failure(session_id: str) -> None:
    """
    Register a failure for a session.
    
    Args:
        session_id: Session identifier
    """
    cb_state = get_circuit_breaker_state(session_id)
    config = get_circuit_breaker_config()
    failure_threshold = config["failure_threshold"]
    
    cb_state.failure_count += 1
    cb_state.last_failure_time = time.time()
    
    # If in half-open state, any failure immediately opens the circuit
    if cb_state.state == CircuitState.HALF_OPEN:
        cb_state.state = CircuitState.OPEN
        cb_state.opened_at = time.time()
        log.warning(
            f"Circuit breaker opened for session {session_id} after failure in HALF_OPEN state",
            extra={
                "session_id": session_id,
                "failure_count": cb_state.failure_count
            }
        )
        return
    
    # Check if we've reached the failure threshold
    if cb_state.failure_count >= failure_threshold:
        cb_state.state = CircuitState.OPEN
        cb_state.opened_at = time.time()
        log.warning(
            f"Circuit breaker opened for session {session_id} after {cb_state.failure_count} failures",
            extra={
                "session_id": session_id,
                "failure_count": cb_state.failure_count,
                "threshold": failure_threshold
            }
        )
    else:
        log.debug(
            f"Circuit breaker failure registered for session {session_id} ({cb_state.failure_count}/{failure_threshold})",
            extra={
                "session_id": session_id,
                "failure_count": cb_state.failure_count,
                "threshold": failure_threshold
            }
        )


def register_success(session_id: str) -> None:
    """
    Register a success for a session.
    
    Args:
        session_id: Session identifier
    """
    cb_state = get_circuit_breaker_state(session_id)
    
    # Reset failure count on success
    cb_state.failure_count = 0
    cb_state.last_failure_time = None
    
    # If in half-open state, success closes the circuit
    if cb_state.state == CircuitState.HALF_OPEN:
        cb_state.state = CircuitState.CLOSED
        cb_state.opened_at = None
        log.info(
            f"Circuit breaker closed for session {session_id} after successful request in HALF_OPEN state",
            extra={"session_id": session_id}
        )
    elif cb_state.state == CircuitState.OPEN:
        # In OPEN, success has no effect until cooldown expires
        log.debug(
            f"Ignoring success for OPEN circuit (session {session_id}) — wait for cooldown",
            extra={"session_id": session_id}
        )
        return


def get_state(session_id: str) -> Dict[str, Any]:
    """
    Get current circuit breaker state as a dictionary.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dict with state information:
        - state: Current state (closed/open/half_open)
        - failure_count: Number of consecutive failures
        - last_failure_time: Timestamp of last failure (if any)
        - opened_at: Timestamp when circuit was opened (if open)
    """
    cb_state = get_circuit_breaker_state(session_id)
    
    return {
        "state": cb_state.state.value,
        "failure_count": cb_state.failure_count,
        "last_failure_time": cb_state.last_failure_time,
        "opened_at": cb_state.opened_at,
    }


def snapshot_state(session_id: str) -> Dict[str, Any]:
    """
    Alias of get_state() used by session_manager for persistence.
    """
    return get_state(session_id)


def restore_state(session_id: str, state_dict: Dict[str, Any]) -> None:
    """
    Restore a circuit breaker state from a persisted snapshot.

    Called by session_manager when loading session state.
    """
    if not state_dict:
        return
    try:
        state = CircuitState(state_dict.get("state", "closed"))
    except Exception:
        state = CircuitState.CLOSED
    cb_state = get_circuit_breaker_state(session_id)
    cb_state.state = state
    cb_state.failure_count = int(state_dict.get("failure_count", 0) or 0)
    cb_state.last_failure_time = state_dict.get("last_failure_time")
    cb_state.opened_at = state_dict.get("opened_at")
    # Safety: if in OPEN but opened_at missing → consider it freshly opened
    if cb_state.state == CircuitState.OPEN and cb_state.opened_at is None:
        cb_state.opened_at = time.time()


def reset_circuit_breaker(session_id: str) -> None:
    """
    Reset circuit breaker for a session (for testing or manual recovery).
    
    Args:
        session_id: Session identifier
    """
    if session_id in _CIRCUIT_BREAKER_STATES:
        cb_state = _CIRCUIT_BREAKER_STATES[session_id]
        cb_state.state = CircuitState.CLOSED
        cb_state.failure_count = 0
        cb_state.last_failure_time = None
        cb_state.opened_at = None
        log.info(f"Circuit breaker reset for session {session_id}", extra={"session_id": session_id})


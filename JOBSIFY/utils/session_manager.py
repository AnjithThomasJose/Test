#!/usr/bin/env python3
"""
Enhanced Session Manager for TTL-based session management.
Implements proper session schema with owner tracking, heartbeats, and auto-expiry.
"""

import datetime
import uuid
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from chroma import chat_sessions_collection, insert_chat_session, update_chat_session, get_chat_session, find_session_by_uid
from .audit_trail import audit_trail, AuditEventType, AuditSeverity

log = logging.getLogger(__name__)

@dataclass
class SessionState:
    """Represents the current state of a session."""
    step: str
    data: Dict[str, Any]
    progress: float = 0.0  # 0.0 to 1.0
    last_updated: str = ""

@dataclass
class Session:
    """Enhanced session schema matching PDF requirements."""
    session_id: str
    owner_type: str  # "candidate" | "employee"
    owner_id: str
    kind: str  # "candidate_pipeline" | "job_pipeline" | "interview"
    state: SessionState
    started_at: str
    last_heartbeat_at: str
    expires_at: str
    status: str = "active"  # "active" | "completed" | "expired" | "failed"
    metadata: Dict[str, Any] = None

class SessionManager:
    """
    Manages TTL-based sessions with heartbeat mechanism and step progression.
    """
    
    def __init__(self):
        self.collection = chat_sessions_collection
        self.default_ttl_minutes = 120  # 2 hours
        self.heartbeat_extension_minutes = 30  # 30 minutes
        self.completion_ttl_minutes = 15  # 15 minutes
    
    def create_session(
        self, 
        owner_type: str, 
        owner_id: str, 
        kind: str, 
        initial_step: str = "start",
        initial_data: Dict[str, Any] = None,
        ttl_minutes: Optional[int] = None
    ) -> Session:
        """
        Create a new session with TTL.
        
        Args:
            owner_type: "candidate" | "employee"
            owner_id: Unique identifier for the owner
            kind: "candidate_pipeline" | "job_pipeline" | "interview"
            initial_step: Starting step name
            initial_data: Initial step data
            ttl_minutes: Custom TTL in minutes (default: 120)
        
        Returns:
            Created Session object
        """
        now = datetime.datetime.utcnow()
        ttl = ttl_minutes or self.default_ttl_minutes
        expires_at = now + datetime.timedelta(minutes=ttl)
        
        # One-document-per-UID model: use UID as the session ID
        session_id = owner_id
        
        session = Session(
            session_id=session_id,
            owner_type=owner_type,
            owner_id=owner_id,
            kind=kind,
            state=SessionState(
                step=initial_step,
                data=initial_data or {},
                last_updated=now.isoformat()
            ),
            started_at=now.isoformat(),
            last_heartbeat_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            status="active",
            metadata={}
        )
        
        # Store in ChromaDB
        session_data = {
            "session_id": session_id,
            "owner_type": owner_type,
            "owner_id": owner_id,
            "kind": kind,
            "state": asdict(session.state),
            "started_at": session.started_at,
            "last_heartbeat_at": session.last_heartbeat_at,
            "expires_at": session.expires_at,
            "status": session.status,
            "metadata": session.metadata or {},
            "chat_history": [],  # Initialize empty chat history
            "timestamp": now.isoformat(),
            "uid": owner_id  # Add uid field for compatibility
        }
        
        insert_chat_session(
            session_id=session_id,
            session_data=session_data,
            metadata={
                "owner_type": owner_type,
                "owner_id": owner_id,
                "kind": kind,
                "status": "active"
            }
        )
        
        # Log session creation
        audit_trail.log_session_event(
            event_type=AuditEventType.SESSION_CREATED,
            session_id=session_id,
            user_id=owner_id,
            action="session_created",
            description=f"Created new {kind} session",
            details={
                "owner_type": owner_type,
                "kind": kind,
                "initial_step": initial_step,
                "ttl_minutes": ttl
            }
        )
        
        log.info(f"✅ Created session {session_id} for {owner_type}:{owner_id} ({kind})")
        return session
    
    def update_heartbeat(self, session_id: str, extend_ttl: bool = True) -> bool:
        """
        Update session heartbeat and optionally extend TTL.
        
        Args:
            session_id: Session to update
            extend_ttl: Whether to extend TTL by heartbeat_extension_minutes
        
        Returns:
            True if successful, False if session not found
        """
        try:
            # Get current session
            session_data = get_chat_session(session_id)
            if not session_data:
                log.warning(f"❌ Session {session_id} not found for heartbeat update")
                return False
            
            now = datetime.datetime.utcnow()
            
            # Update heartbeat time
            session_data["last_heartbeat_at"] = now.isoformat()
            
            # Extend TTL if requested
            if extend_ttl:
                # Check if expires_at exists and is valid
                if "expires_at" in session_data and session_data["expires_at"] and session_data["expires_at"].strip():
                    try:
                        current_expires = datetime.datetime.fromisoformat(session_data["expires_at"])
                        new_expires = now + datetime.timedelta(minutes=self.heartbeat_extension_minutes)
                        # Only extend if new time is later than current
                        if new_expires > current_expires:
                            session_data["expires_at"] = new_expires.isoformat()
                    except ValueError as e:
                        log.warning(f"⚠️ Invalid expires_at format for session {session_id}: {e}")
                        # Set a default expires_at if parsing fails
                        session_data["expires_at"] = (now + datetime.timedelta(hours=24)).isoformat()
                else:
                    # If no expires_at or empty string, set a default
                    session_data["expires_at"] = (now + datetime.timedelta(hours=24)).isoformat()
            
            # Update in ChromaDB
            update_chat_session(
                session_id=session_id,
                session_data=session_data,
                metadata={
                    "last_heartbeat": now.isoformat(),
                    "extended_ttl": extend_ttl
                }
            )
            
            log.debug(f"💓 Heartbeat updated for session {session_id}")
            return True
            
        except Exception as e:
            log.error(f"❌ Error updating heartbeat for session {session_id}: {e}")
            return False
    
    def update_step(
        self, 
        session_id: str, 
        step: str, 
        data: Dict[str, Any] = None,
        progress: float = None
    ) -> bool:
        """
        Update session step and data.
        
        Args:
            session_id: Session to update
            step: New step name
            data: Step-specific data (keep minimal)
            progress: Progress percentage (0.0 to 1.0)
        
        Returns:
            True if successful, False if session not found
        """
        try:
            # Get current session
            session_data = get_chat_session(session_id)
            if not session_data:
                log.warning(f"❌ Session {session_id} not found for step update")
                return False
            
            now = datetime.datetime.utcnow()
            
            # Update state
            current_state = session_data.get("state", {})
            current_state["step"] = step
            current_state["data"] = data or {}
            current_state["last_updated"] = now.isoformat()
            
            if progress is not None:
                current_state["progress"] = max(0.0, min(1.0, progress))
            
            session_data["state"] = current_state
            session_data["last_heartbeat_at"] = now.isoformat()
            
            # Update in ChromaDB
            update_chat_session(
                session_id=session_id,
                session_data=session_data,
                metadata={
                    "step": step,
                    "progress": current_state.get("progress", 0.0),
                    "last_updated": now.isoformat()
                }
            )
            
            log.info(f"📝 Updated session {session_id} to step '{step}' (progress: {current_state.get('progress', 0.0):.1%})")
            return True
            
        except Exception as e:
            log.error(f"❌ Error updating step for session {session_id}: {e}")
            return False
    
    def complete_session(self, session_id: str, final_ttl_minutes: Optional[int] = None, force: bool = False) -> bool:
        """
        Mark session as completed and set short TTL.
        
        Args:
            session_id: Session to complete
            final_ttl_minutes: Final TTL in minutes (default: 15)
            force: If True, complete even if not at terminal step (for error cases)
        
        Returns:
            True if successful, False if session not found
        """
        try:
            # Get current session
            session_data = get_chat_session(session_id)
            if not session_data:
                log.warning(f"❌ Session {session_id} not found for completion")
                return False
            
            # Guard: only terminal nodes may complete the session (unless forced)
            state = session_data.get("state", {})
            current_step = (state.get("step") or "").strip()
            terminal_steps = {"end", "end_invalid_resume", "end_invalid_jd"}
            if not force and current_step not in terminal_steps:
                log.warning(f"⚠️ Prevented premature completion for session {session_id} at step '{current_step}' (use force=True to override)")
                return False
            
            # If forced and not at terminal step, update step to "end" before completing
            if force and current_step not in terminal_steps:
                log.warning(f"⚠️ Force completing session {session_id} at non-terminal step '{current_step}', updating to 'end'")
                current_state = session_data.get("state", {})
                current_state["step"] = "end"
                session_data["state"] = current_state
            
            now = datetime.datetime.utcnow()
            ttl = final_ttl_minutes or self.completion_ttl_minutes
            expires_at = now + datetime.timedelta(minutes=ttl)
            
            # Update session
            session_data["status"] = "completed"
            session_data["expires_at"] = expires_at.isoformat()
            session_data["last_heartbeat_at"] = now.isoformat()
            
            # Update state
            current_state = session_data.get("state", {})
            current_state["progress"] = 1.0
            current_state["last_updated"] = now.isoformat()
            session_data["state"] = current_state
            
            # Update in ChromaDB
            update_chat_session(
                session_id=session_id,
                session_data=session_data,
                metadata={
                    "status": "completed",
                    "completed_at": now.isoformat(),
                    "final_ttl": ttl
                }
            )
            
            # Log session completion
            audit_trail.log_session_event(
                event_type=AuditEventType.SESSION_COMPLETED,
                session_id=session_id,
                user_id=session_data.get("owner_id", "unknown"),
                action="session_completed",
                description="Session completed successfully",
                details={
                    "final_ttl_minutes": ttl,
                    "completion_time": now.isoformat()
                }
            )
            
            log.info(f"✅ Completed session {session_id} (expires in {ttl} minutes)")
            return True
            
        except Exception as e:
            log.error(f"❌ Error completing session {session_id}: {e}")
            return False
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: Session ID to retrieve
        
        Returns:
            Session object or None if not found
        """
        try:
            session_data = get_chat_session(session_id)
            if not session_data:
                log.warning(f"❌ No session data found for session_id: {session_id}")
                return None
            
            # Debug logs removed to reduce verbosity
            
            # Convert to Session object
            state_data = session_data.get("state", {})
            state = SessionState(
                step=state_data.get("step", ""),
                data=state_data.get("data", {}),
                progress=state_data.get("progress", 0.0),
                last_updated=state_data.get("last_updated", "")
            )
            
            return Session(
                session_id=session_data.get("session_id", session_id),  # Fallback to input session_id
                owner_type=session_data.get("owner_type", "candidate"),
                owner_id=session_data.get("owner_id", session_data.get("uid", "")),
                kind=session_data.get("kind", "candidate_pipeline"),
                state=state,
                started_at=session_data.get("started_at", session_data.get("timestamp", datetime.datetime.utcnow().isoformat())),
                last_heartbeat_at=session_data.get("last_heartbeat_at", session_data.get("timestamp", datetime.datetime.utcnow().isoformat())),
                expires_at=session_data.get("expires_at", (datetime.datetime.utcnow() + datetime.timedelta(hours=24)).isoformat()),
                status=session_data.get("status", "active"),
                metadata=session_data.get("metadata", {})
            )
            
        except Exception as e:
            log.error(f"❌ Error retrieving session {session_id}: {e}")
            return None
    
    def get_active_sessions(self, owner_id: str, kind: Optional[str] = None) -> List[Session]:
        """
        Get active sessions for an owner.
        
        Args:
            owner_id: Owner ID to query
            kind: Optional session kind filter
        
        Returns:
            List of active Session objects
        """
        try:
            # Query for sessions by owner_id using get method with where filter
            results = self.collection.get(
                where={"owner_id": owner_id}
            )
            
            sessions = []
            now = datetime.datetime.utcnow()
            
            # Get session IDs from the results
            session_ids = results.get('ids', [])
            if not session_ids:
                return sessions
            
            for session_id in session_ids:
                if isinstance(session_id, list):
                    session_id = session_id[0]
                
                session = self.get_session(session_id)
                if not session:
                    continue
                
                # Check if session is active and not expired
                if session.expires_at and session.expires_at.strip():
                    try:
                        expires_at = datetime.datetime.fromisoformat(session.expires_at)
                        if session.status == "active" and expires_at > now:
                            if kind is None or session.kind == kind:
                                sessions.append(session)
                    except ValueError as e:
                        log.warning(
                            f"⚠️ Invalid expires_at format for session {session.session_id}: {e}"
                        )
                        # Skip this session if date parsing fails
                        continue
                else:
                    # If no expires_at or empty string, consider it active
                    if session.status == "active":
                        if kind is None or session.kind == kind:
                            sessions.append(session)
            
            return sessions
            
        except Exception as e:
            log.error(f"❌ Error getting active sessions for {owner_id}: {e}")
            return []
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        try:
            # This would typically be a background job
            # For now, we'll just mark them as expired
            log.info("🧹 Session cleanup would be implemented as a background job")
            return 0
            
        except Exception as e:
            log.error(f"❌ Error cleaning up expired sessions: {e}")
            return 0
    
    def get_session_by_owner(self, owner_id: str, kind: str) -> Optional[Session]:
        """
        Get the most recent active session for an owner and kind.
        
        Args:
            owner_id: Owner ID
            kind: Session kind
        
        Returns:
            Most recent active Session or None
        """
        sessions = self.get_active_sessions(owner_id, kind)
        if not sessions:
            return None
        
        # Return the most recent session
        return max(sessions, key=lambda s: s.started_at)
    
    def get_or_reuse_session(
        self, 
        owner_id: str, 
        kind: str, 
        owner_type: str = "candidate",
        initial_step: str = "start",
        initial_data: Dict[str, Any] = None
    ) -> Session:
        """
        Get existing session or create new one if none exists.
        This ensures UIDs always reuse their existing session ID.
        
        Args:
            owner_id: Owner ID (UID)
            kind: Session kind
            owner_type: Owner type (default: "candidate")
            initial_step: Initial step if creating new session
            initial_data: Initial data if creating new session
        
        Returns:
            Existing or newly created Session
        """
        
        # Always check for existing session first
        # One-document-per-UID: upsert/get-or-create by UID
        existing = self.get_session(owner_id)
        if existing:
            log.info(f"✅ REUSING existing UID document {owner_id}")
            return existing
        
        log.info(f"🔄 No existing UID document, creating one for {owner_id}")
        new_session = self.create_session(
            owner_type=owner_type,
            owner_id=owner_id,
            kind=kind,
            initial_step=initial_step,
            initial_data=initial_data
        )
        log.info(f"✅ CREATED new session {new_session.session_id} for UID {owner_id}")
        return new_session
    
    def _migrate_session_data(self, owner_id: str, new_session_id: str):
        """
        Migrate important data from any existing sessions to the new session.
        This ensures continuity of data across session recreations.
        """
        try:
            from chroma import get_chat_session, update_chat_session
            
            # Get all sessions for this UID
            results = self.collection.query(
                query_texts=["session data"],
                n_results=100,
                where={"uid": owner_id}
            )
            
            if not results or not results.get('ids'):
                return
            
            session_ids = results['ids']
            if isinstance(session_ids[0], list):
                session_ids = session_ids[0]
            
            # Get the new session data
            new_session_data = get_chat_session(new_session_id)
            if not new_session_data:
                return
            
            migrated_data = {}
            
            # Check all existing sessions for important data to migrate
            for session_id in session_ids:
                if session_id == new_session_id:
                    continue  # Skip the new session itself
                
                old_session_data = get_chat_session(session_id)
                if not old_session_data:
                    continue
                
                # Debug: Log what's in the old session
                log.debug(
                    f"🔍 DEBUG: Old session {session_id} keys: {list(old_session_data.keys())}"
                )
                
                # Migrate structured_resume from resume_parser
                if "resume_parser" in old_session_data and "structured_resume" in old_session_data["resume_parser"]:
                    migrated_data["resume_parser"] = old_session_data["resume_parser"]
                elif "structured_resume" in old_session_data:
                    # Check if structured_resume is stored directly in the session
                    migrated_data["structured_resume"] = old_session_data["structured_resume"]
                else:
                    log.debug(
                        f"🔍 DEBUG: No resume_parser or structured_resume found in "
                        f"session {session_id}"
                    )
                
                # Migrate other important agent data
                for agent_name in ["personal_info_parser", "education_parser", "experience_parser", "gap_analyzer", "resume_assembler"]:
                    if agent_name in old_session_data and agent_name not in migrated_data:
                        migrated_data[agent_name] = old_session_data[agent_name]
            
            # Update the new session with migrated data
            if migrated_data:
                new_session_data.update(migrated_data)
                update_chat_session(
                    session_id=new_session_id,
                    session_data=new_session_data,
                    metadata={
                        "migrated_data": True,
                        "migrated_from_sessions": len(session_ids) - 1,
                        "uid": owner_id
                    }
                )
                log.info(
                    f"✅ MIGRATION COMPLETE: Migrated data to session {new_session_id}"
                )
            
        except Exception as e:
            log.error(f"❌ Error migrating session data for UID {owner_id}: {e}")
            import traceback
            log.error(f"Migration traceback: {traceback.format_exc()}")

# Create a single instance to be used across the application
session_manager = SessionManager()

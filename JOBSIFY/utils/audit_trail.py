#!/usr/bin/env python3
"""
Audit Trail System for Phase 3.
Implements comprehensive audit trails for all operations and system events.
"""

import datetime
import json
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

log = logging.getLogger("main")

class AuditEventType(Enum):
    """Types of audit events."""
    SESSION_CREATED = "session_created"
    SESSION_UPDATED = "session_updated"
    SESSION_COMPLETED = "session_completed"
    SESSION_EXPIRED = "session_expired"
    
    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_ACCESSED = "memory_accessed"
    
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_ACTION = "user_action"
    
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_INFO = "system_info"
    
    DATA_ACCESS = "data_access"
    DATA_MODIFIED = "data_modified"
    DATA_DELETED = "data_deleted"
    
    BACKGROUND_JOB_STARTED = "background_job_started"
    BACKGROUND_JOB_COMPLETED = "background_job_completed"
    BACKGROUND_JOB_FAILED = "background_job_failed"

class AuditSeverity(Enum):
    """Severity levels for audit events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Represents an audit event."""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: str
    user_id: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    description: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = None

class AuditTrail:
    """
    Comprehensive audit trail system for tracking all operations.
    """
    
    def __init__(self):
        self.audit_collection_name = "audit_trail"
        self.audit_collection = None
    
    def _get_collection(self):
        """Get or create the audit trail collection."""
        if self.audit_collection is None:
            from chroma import client, embedding_fn
            self.audit_collection = client.get_or_create_collection(
                name=self.audit_collection_name,
                embedding_function=embedding_fn
            )
        return self.audit_collection
    
    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        action: str,
        description: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an audit event.
        
        Args:
            event_type: Type of event
            severity: Severity level
            action: Action performed
            description: Human-readable description
            user_id: User who performed the action
            session_id: Session ID if applicable
            request_id: Request ID if applicable
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            details: Additional event details
            ip_address: IP address of the user
            user_agent: User agent string
            metadata: Additional metadata
        
        Returns:
            Event ID
        """
        try:
            event_id = f"audit_{int(datetime.datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
            
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                severity=severity,
                timestamp=datetime.datetime.utcnow().isoformat(),
                user_id=user_id,
                session_id=session_id,
                request_id=request_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                description=description,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent,
                metadata=metadata or {}
            )
            
            # Store audit event
            collection = self._get_collection()
            
            # Convert event to dict and handle enum serialization
            event_dict = asdict(event)
            event_dict['event_type'] = event_type.value
            event_dict['severity'] = severity.value
            
            collection.add(
                ids=[event_id],
                documents=[json.dumps(event_dict)],
                metadatas=[{
                    "event_type": event_type.value,
                    "severity": severity.value,
                    "user_id": user_id or "system",
                    "session_id": session_id or "none",
                    "timestamp": event.timestamp
                }]
            )
            
            log.info(f"📋 Audit logged: {event_type.value} - {action}")
            return event_id
            
        except Exception as e:
            log.error(f"❌ Error logging audit event: {e}")
            return None
    
    def log_session_event(
        self,
        event_type: AuditEventType,
        session_id: str,
        user_id: str,
        action: str,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a session-related audit event."""
        return self.log_event(
            event_type=event_type,
            severity=AuditSeverity.MEDIUM,
            action=action,
            description=description,
            user_id=user_id,
            session_id=session_id,
            resource_type="session",
            resource_id=session_id,
            details=details
        )
    
    def log_memory_event(
        self,
        event_type: AuditEventType,
        memory_id: str,
        user_id: str,
        action: str,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a memory-related audit event."""
        return self.log_event(
            event_type=event_type,
            severity=AuditSeverity.LOW,
            action=action,
            description=description,
            user_id=user_id,
            resource_type="memory",
            resource_id=memory_id,
            details=details
        )
    
    def log_agent_event(
        self,
        event_type: AuditEventType,
        agent_name: str,
        user_id: str,
        session_id: str,
        action: str,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log an agent-related audit event."""
        severity = AuditSeverity.HIGH if event_type == AuditEventType.AGENT_FAILED else AuditSeverity.MEDIUM
        
        return self.log_event(
            event_type=event_type,
            severity=severity,
            action=action,
            description=description,
            user_id=user_id,
            session_id=session_id,
            resource_type="agent",
            resource_id=agent_name,
            details=details
        )
    
    def log_system_event(
        self,
        event_type: AuditEventType,
        action: str,
        description: str,
        severity: AuditSeverity = AuditSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a system-related audit event."""
        return self.log_event(
            event_type=event_type,
            severity=severity,
            action=action,
            description=description,
            details=details
        )
    
    def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a data access event."""
        return self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.LOW,
            action=action,
            description=description,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details
        )
    
    def log_data_modification(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a data modification event."""
        return self.log_event(
            event_type=AuditEventType.DATA_MODIFIED,
            severity=AuditSeverity.MEDIUM,
            action=action,
            description=description,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details
        )
    
    def get_audit_events(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        severity: Optional[AuditSeverity] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit events with filtering.
        
        Args:
            user_id: Filter by user ID
            session_id: Filter by session ID
            event_type: Filter by event type
            severity: Filter by severity
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            limit: Maximum number of events to return
        
        Returns:
            List of audit events
        """
        try:
            collection = self._get_collection()
            
            # Build where clause
            where_clause = {}
            if user_id:
                where_clause["user_id"] = user_id
            if session_id:
                where_clause["session_id"] = session_id
            if event_type:
                where_clause["event_type"] = event_type.value
            if severity:
                where_clause["severity"] = severity.value
            
            # Query events
            results = collection.query(
                query_texts=["audit event"],
                n_results=limit,
                where=where_clause
            )
            
            events = []
            for i, event_id in enumerate(results.get('ids', [])):
                if isinstance(event_id, list):
                    event_id = event_id[0]
                
                event_data = json.loads(results['documents'][i])
                
                # Apply date filtering if specified
                if start_date or end_date:
                    event_timestamp = event_data.get('timestamp', '')
                    if start_date and event_timestamp < start_date:
                        continue
                    if end_date and event_timestamp > end_date:
                        continue
                
                events.append(event_data)
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return events
            
        except Exception as e:
            log.error(f"❌ Error retrieving audit events: {e}")
            return []
    
    def get_user_audit_trail(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get audit trail for a specific user."""
        return self.get_audit_events(user_id=user_id, limit=limit)
    
    def get_session_audit_trail(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get audit trail for a specific session."""
        return self.get_audit_events(session_id=session_id, limit=limit)
    
    def get_security_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get security-related audit events."""
        return self.get_audit_events(
            severity=AuditSeverity.HIGH,
            limit=limit
        ) + self.get_audit_events(
            event_type=AuditEventType.SYSTEM_ERROR,
            limit=limit
        )
    
    def get_data_access_log(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get data access log for a user."""
        return self.get_audit_events(
            user_id=user_id,
            event_type=AuditEventType.DATA_ACCESS,
            limit=limit
        )
    
    def get_agent_performance_log(self, agent_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get performance log for a specific agent."""
        events = self.get_audit_events(
            event_type=AuditEventType.AGENT_STARTED,
            limit=limit
        ) + self.get_audit_events(
            event_type=AuditEventType.AGENT_COMPLETED,
            limit=limit
        ) + self.get_audit_events(
            event_type=AuditEventType.AGENT_FAILED,
            limit=limit
        )
        
        # Filter by agent name
        return [event for event in events if event.get('resource_id') == agent_name]
    
    def generate_audit_report(
        self,
        start_date: str,
        end_date: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive audit report.
        
        Args:
            start_date: Start date for the report (ISO format)
            end_date: End date for the report (ISO format)
            user_id: Optional user ID to filter by
        
        Returns:
            Audit report data
        """
        try:
            # Get all events in the date range
            events = self.get_audit_events(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                limit=1000
            )
            
            # Generate statistics
            report = {
                "report_period": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "user_id": user_id
                },
                "total_events": len(events),
                "events_by_type": {},
                "events_by_severity": {},
                "events_by_user": {},
                "top_actions": {},
                "security_events": 0,
                "error_events": 0,
                "session_events": 0,
                "memory_events": 0,
                "agent_events": 0
            }
            
            # Analyze events
            for event in events:
                event_type = event.get('event_type', 'unknown')
                severity = event.get('severity', 'unknown')
                user = event.get('user_id', 'system')
                action = event.get('action', 'unknown')
                
                # Count by type
                report["events_by_type"][event_type] = report["events_by_type"].get(event_type, 0) + 1
                
                # Count by severity
                report["events_by_severity"][severity] = report["events_by_severity"].get(severity, 0) + 1
                
                # Count by user
                report["events_by_user"][user] = report["events_by_user"].get(user, 0) + 1
                
                # Count top actions
                report["top_actions"][action] = report["top_actions"].get(action, 0) + 1
                
                # Count specific event types
                if severity == "high" or event_type in ["system_error", "agent_failed"]:
                    report["security_events"] += 1
                
                if event_type in ["system_error", "agent_failed"]:
                    report["error_events"] += 1
                
                if "session" in event_type:
                    report["session_events"] += 1
                
                if "memory" in event_type:
                    report["memory_events"] += 1
                
                if "agent" in event_type:
                    report["agent_events"] += 1
            
            return report
            
        except Exception as e:
            log.error(f"❌ Error generating audit report: {e}")
            return {"error": str(e)}

# Create a single instance to be used across the application
audit_trail = AuditTrail()

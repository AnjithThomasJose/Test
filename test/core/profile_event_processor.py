"""
ProfileEventProcessor - Multi-source data ingestion with Pub/Sub simulation
Handles event-driven processing for candidate data from multiple sources
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .candidate_matching_models import (
    ProcessingEvent, CandidateProfile, DataSource, ProcessingStage,
    StructuredFact, DynamicTag, MultiViewEmbedding, ConfidenceScore,
    Provenance, FactType, TagCategory, EmbeddingView
)

log = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of processing events"""
    RESUME_UPLOADED = "resume_uploaded"
    CHAT_SESSION_CREATED = "chat_session_created"
    ASSESSMENT_COMPLETED = "assessment_completed"
    INTERVIEW_COMPLETED = "interview_completed"
    PROFILE_UPDATED = "profile_updated"
    DATA_SYNC = "data_sync"
    BATCH_PROCESSING = "batch_processing"


class EventPriority(int, Enum):
    """Event processing priorities"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class EventSubscription:
    """Event subscription configuration"""
    event_type: EventType
    handler: Callable[[ProcessingEvent], Awaitable[Dict[str, Any]]]
    priority: EventPriority = EventPriority.NORMAL
    retry_count: int = 3
    timeout_seconds: int = 30


@dataclass
class ProcessingQueue:
    """In-memory processing queue (simulating Pub/Sub)"""
    events: List[ProcessingEvent] = field(default_factory=list)
    processing: Dict[str, ProcessingEvent] = field(default_factory=dict)
    completed: List[ProcessingEvent] = field(default_factory=list)
    failed: List[ProcessingEvent] = field(default_factory=list)
    
    def add_event(self, event: ProcessingEvent):
        """Add event to queue with priority ordering"""
        self.events.append(event)
        # Sort by priority (higher priority first)
        self.events.sort(key=lambda e: e.priority, reverse=True)
    
    def get_next_event(self) -> Optional[ProcessingEvent]:
        """Get next event to process"""
        if self.events:
            return self.events.pop(0)
        return None
    
    def mark_processing(self, event: ProcessingEvent):
        """Mark event as being processed"""
        self.processing[event.event_id] = event
    
    def mark_completed(self, event: ProcessingEvent):
        """Mark event as completed"""
        if event.event_id in self.processing:
            del self.processing[event.event_id]
        self.completed.append(event)
    
    def mark_failed(self, event: ProcessingEvent):
        """Mark event as failed"""
        if event.event_id in self.processing:
            del self.processing[event.event_id]
        self.failed.append(event)


class ProfileEventProcessor:
    """
    Event-driven processor for multi-source candidate data
    Simulates Pub/Sub architecture for scalable processing
    """
    
    def __init__(self):
        self.queue = ProcessingQueue()
        self.subscriptions: Dict[EventType, List[EventSubscription]] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False
        self.max_concurrent_events = 10
        
        # Initialize default subscriptions
        self._setup_default_subscriptions()
    
    def _setup_default_subscriptions(self):
        """Setup default event subscriptions"""
        self.subscribe(
            EventType.RESUME_UPLOADED,
            self._handle_resume_upload,
            EventPriority.HIGH
        )
        
        self.subscribe(
            EventType.CHAT_SESSION_CREATED,
            self._handle_chat_session,
            EventPriority.NORMAL
        )
        
        self.subscribe(
            EventType.ASSESSMENT_COMPLETED,
            self._handle_assessment_completion,
            EventPriority.HIGH
        )
        
        self.subscribe(
            EventType.INTERVIEW_COMPLETED,
            self._handle_interview_completion,
            EventPriority.HIGH
        )
        
        self.subscribe(
            EventType.PROFILE_UPDATED,
            self._handle_profile_update,
            EventPriority.NORMAL
        )
    
    def subscribe(self, event_type: EventType, handler: Callable, 
                  priority: EventPriority = EventPriority.NORMAL,
                  retry_count: int = 3, timeout_seconds: int = 30):
        """Subscribe to an event type"""
        subscription = EventSubscription(
            event_type=event_type,
            handler=handler,
            priority=priority,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds
        )
        
        if event_type not in self.subscriptions:
            self.subscriptions[event_type] = []
        
        self.subscriptions[event_type].append(subscription)
        log.info(f"Subscribed to {event_type} with priority {priority}")
    
    async def publish_event(self, event_type: EventType, candidate_id: str, 
                          data: Dict[str, Any], priority: EventPriority = EventPriority.NORMAL):
        """Publish an event for processing"""
        event = ProcessingEvent(
            event_type=event_type.value,
            candidate_id=candidate_id,
            data=data,
            priority=priority.value
        )
        
        self.queue.add_event(event)
        log.info(f"Published {event_type} event for candidate {candidate_id}")
        
        # Start processing if not already running
        if not self.is_running:
            await self.start_processing()
    
    async def start_processing(self):
        """Start the event processing loop"""
        if self.is_running:
            return
        
        self.is_running = True
        log.info("Starting ProfileEventProcessor")
        
        # Start processing loop
        asyncio.create_task(self._processing_loop())
    
    async def stop_processing(self):
        """Stop the event processing loop"""
        self.is_running = False
        log.info("Stopping ProfileEventProcessor")
        
        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
        
        self.processing_tasks.clear()
    
    async def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Check if we can process more events
                if len(self.processing_tasks) >= self.max_concurrent_events:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next event
                event = self.queue.get_next_event()
                if not event:
                    await asyncio.sleep(0.1)
                    continue
                
                # Start processing task
                task = asyncio.create_task(self._process_event(event))
                self.processing_tasks[event.event_id] = task
                
                # Clean up completed tasks
                completed_tasks = [
                    event_id for event_id, task in self.processing_tasks.items()
                    if task.done()
                ]
                for event_id in completed_tasks:
                    del self.processing_tasks[event_id]
                
            except Exception as e:
                log.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_event(self, event: ProcessingEvent):
        """Process a single event"""
        try:
            self.queue.mark_processing(event)
            
            # Find handlers for this event type
            event_type = EventType(event.event_type)
            handlers = self.subscriptions.get(event_type, [])
            
            if not handlers:
                log.warning(f"No handlers for event type {event_type}")
                self.queue.mark_completed(event)
                return
            
            # Process with each handler
            results = []
            for subscription in handlers:
                try:
                    result = await asyncio.wait_for(
                        subscription.handler(event),
                        timeout=subscription.timeout_seconds
                    )
                    results.append(result)
                except asyncio.TimeoutError:
                    log.error(f"Handler timeout for event {event.event_id}")
                    if event.retry_count < subscription.retry_count:
                        event.retry_count += 1
                        self.queue.add_event(event)  # Retry
                        return
                except Exception as e:
                    log.error(f"Handler error for event {event.event_id}: {e}")
                    if event.retry_count < subscription.retry_count:
                        event.retry_count += 1
                        self.queue.add_event(event)  # Retry
                        return
            
            self.queue.mark_completed(event)
            log.info(f"Successfully processed event {event.event_id}")
            
        except Exception as e:
            log.error(f"Failed to process event {event.event_id}: {e}")
            self.queue.mark_failed(event)
    
    # ==================== EVENT HANDLERS ====================
    
    async def _handle_resume_upload(self, event: ProcessingEvent) -> Dict[str, Any]:
        """Handle resume upload event"""
        log.info(f"Processing resume upload for candidate {event.candidate_id}")
        
        resume_data = event.data.get("resume_data", {})
        
        # Extract structured facts from resume
        facts = await self._extract_resume_facts(resume_data, event.candidate_id)
        
        # Generate tags from resume
        tags = await self._generate_resume_tags(resume_data, event.candidate_id)
        
        # Create embeddings
        embeddings = await self._create_resume_embeddings(resume_data, event.candidate_id)
        
        return {
            "event_id": event.event_id,
            "candidate_id": event.candidate_id,
            "facts_extracted": len(facts),
            "tags_generated": len(tags),
            "embeddings_created": len(embeddings),
            "processing_stage": ProcessingStage.FACT_EXTRACTION.value
        }
    
    async def _handle_chat_session(self, event: ProcessingEvent) -> Dict[str, Any]:
        """Handle chat session event"""
        log.info(f"Processing chat session for candidate {event.candidate_id}")
        
        chat_data = event.data.get("chat_data", {})
        
        # Extract facts from chat
        facts = await self._extract_chat_facts(chat_data, event.candidate_id)
        
        # Generate tags from chat context
        tags = await self._generate_chat_tags(chat_data, event.candidate_id)
        
        # Create chat embeddings
        embeddings = await self._create_chat_embeddings(chat_data, event.candidate_id)
        
        return {
            "event_id": event.event_id,
            "candidate_id": event.candidate_id,
            "facts_extracted": len(facts),
            "tags_generated": len(tags),
            "embeddings_created": len(embeddings),
            "processing_stage": ProcessingStage.FACT_EXTRACTION.value
        }
    
    async def _handle_assessment_completion(self, event: ProcessingEvent) -> Dict[str, Any]:
        """Handle assessment completion event"""
        log.info(f"Processing assessment completion for candidate {event.candidate_id}")
        
        assessment_data = event.data.get("assessment_data", {})
        
        # Extract facts from assessment
        facts = await self._extract_assessment_facts(assessment_data, event.candidate_id)
        
        # Generate tags from assessment results
        tags = await self._generate_assessment_tags(assessment_data, event.candidate_id)
        
        # Create assessment embeddings
        embeddings = await self._create_assessment_embeddings(assessment_data, event.candidate_id)
        
        return {
            "event_id": event.event_id,
            "candidate_id": event.candidate_id,
            "facts_extracted": len(facts),
            "tags_generated": len(tags),
            "embeddings_created": len(embeddings),
            "processing_stage": ProcessingStage.FACT_EXTRACTION.value
        }
    
    async def _handle_interview_completion(self, event: ProcessingEvent) -> Dict[str, Any]:
        """Handle interview completion event"""
        log.info(f"Processing interview completion for candidate {event.candidate_id}")
        
        interview_data = event.data.get("interview_data", {})
        
        # Extract facts from interview
        facts = await self._extract_interview_facts(interview_data, event.candidate_id)
        
        # Generate tags from interview responses
        tags = await self._generate_interview_tags(interview_data, event.candidate_id)
        
        # Create interview embeddings
        embeddings = await self._create_interview_embeddings(interview_data, event.candidate_id)
        
        return {
            "event_id": event.event_id,
            "candidate_id": event.candidate_id,
            "facts_extracted": len(facts),
            "tags_generated": len(tags),
            "embeddings_created": len(embeddings),
            "processing_stage": ProcessingStage.FACT_EXTRACTION.value
        }
    
    async def _handle_profile_update(self, event: ProcessingEvent) -> Dict[str, Any]:
        """Handle profile update event"""
        log.info(f"Processing profile update for candidate {event.candidate_id}")
        
        # Trigger re-processing of all data sources
        await self.publish_event(
            EventType.DATA_SYNC,
            event.candidate_id,
            {"trigger": "profile_update"},
            EventPriority.NORMAL
        )
        
        return {
            "event_id": event.event_id,
            "candidate_id": event.candidate_id,
            "action": "triggered_data_sync",
            "processing_stage": ProcessingStage.MERGING.value
        }
    
    # ==================== FACT EXTRACTION HELPERS ====================
    
    async def _extract_resume_facts(self, resume_data: Dict[str, Any], candidate_id: str) -> List[StructuredFact]:
        """Extract structured facts from resume data"""
        facts = []
        
        # Extract skills
        skills = resume_data.get("Skills", [])
        for skill in skills:
            if isinstance(skill, dict):
                skill_name = skill.get("skill", skill.get("name", ""))
                if skill_name:
                    fact = StructuredFact(
                        fact_type=FactType.SKILL,
                        content={"skill_name": skill_name, "skill_data": skill},
                        confidence=ConfidenceScore(value=0.9, reasoning="Directly extracted from resume"),
                        provenance=Provenance(
                            source=DataSource.RESUME,
                            source_id=candidate_id,
                            extraction_method="resume_parser"
                        )
                    )
                    facts.append(fact)
        
        # Extract work experience
        work_exp = resume_data.get("WorkExperience", [])
        for exp in work_exp:
            if isinstance(exp, dict):
                fact = StructuredFact(
                    fact_type=FactType.EXPERIENCE,
                    content=exp,
                    confidence=ConfidenceScore(value=0.85, reasoning="Structured experience data"),
                    provenance=Provenance(
                        source=DataSource.RESUME,
                        source_id=candidate_id,
                        extraction_method="resume_parser"
                    )
                )
                facts.append(fact)
        
        # Extract education
        education = resume_data.get("Education", [])
        for edu in education:
            if isinstance(edu, dict):
                fact = StructuredFact(
                    fact_type=FactType.EDUCATION,
                    content=edu,
                    confidence=ConfidenceScore(value=0.9, reasoning="Structured education data"),
                    provenance=Provenance(
                        source=DataSource.RESUME,
                        source_id=candidate_id,
                        extraction_method="resume_parser"
                    )
                )
                facts.append(fact)
        
        return facts
    
    async def _extract_chat_facts(self, chat_data: Dict[str, Any], candidate_id: str) -> List[StructuredFact]:
        """Extract facts from chat session data"""
        facts = []
        
        # Extract interests and preferences from chat
        interests = chat_data.get("interests", [])
        for interest in interests:
            fact = StructuredFact(
                fact_type=FactType.INTEREST,
                content={"interest": interest},
                confidence=ConfidenceScore(value=0.7, reasoning="Extracted from chat conversation"),
                provenance=Provenance(
                    source=DataSource.CHAT_SESSION,
                    source_id=candidate_id,
                    extraction_method="chat_analysis"
                )
            )
            facts.append(fact)
        
        return facts
    
    async def _extract_assessment_facts(self, assessment_data: Dict[str, Any], candidate_id: str) -> List[StructuredFact]:
        """Extract facts from assessment data"""
        facts = []
        
        # Extract skill assessments
        skill_scores = assessment_data.get("skill_scores", {})
        for skill, score in skill_scores.items():
            fact = StructuredFact(
                fact_type=FactType.SKILL,
                content={"skill_name": skill, "assessment_score": score},
                confidence=ConfidenceScore(value=0.8, reasoning="Validated through assessment"),
                provenance=Provenance(
                    source=DataSource.ASSESSMENT,
                    source_id=candidate_id,
                    extraction_method="assessment_scoring"
                )
            )
            facts.append(fact)
        
        return facts
    
    async def _extract_interview_facts(self, interview_data: Dict[str, Any], candidate_id: str) -> List[StructuredFact]:
        """Extract facts from interview data"""
        facts = []
        
        # Extract soft skills from interview responses
        responses = interview_data.get("responses", [])
        for response in responses:
            if response.get("question_type") == "behavioral":
                fact = StructuredFact(
                    fact_type=FactType.SOFT_SKILL,
                    content={"response": response},
                    confidence=ConfidenceScore(value=0.75, reasoning="Inferred from interview response"),
                    provenance=Provenance(
                        source=DataSource.INTERVIEW,
                        source_id=candidate_id,
                        extraction_method="interview_analysis"
                    )
                )
                facts.append(fact)
        
        return facts
    
    # ==================== TAG GENERATION HELPERS ====================
    
    async def _generate_resume_tags(self, resume_data: Dict[str, Any], candidate_id: str) -> List[DynamicTag]:
        """Generate tags from resume data"""
        tags = []
        
        # Generate industry tags
        skills = resume_data.get("Skills", [])
        if any("python" in str(skill).lower() for skill in skills):
            tag = DynamicTag(
                category=TagCategory.TECHNICAL_SKILL,
                value="python_developer",
                confidence=ConfidenceScore(value=0.8, reasoning="Python skills detected"),
                provenance=Provenance(
                    source=DataSource.RESUME,
                    source_id=candidate_id,
                    extraction_method="skill_analysis"
                )
            )
            tags.append(tag)
        
        return tags
    
    async def _generate_chat_tags(self, chat_data: Dict[str, Any], candidate_id: str) -> List[DynamicTag]:
        """Generate tags from chat data"""
        tags = []
        
        # Generate work style tags from chat
        if chat_data.get("prefers_remote"):
            tag = DynamicTag(
                category=TagCategory.WORK_STYLE,
                value="remote_preference",
                confidence=ConfidenceScore(value=0.7, reasoning="Expressed in chat"),
                provenance=Provenance(
                    source=DataSource.CHAT_SESSION,
                    source_id=candidate_id,
                    extraction_method="preference_analysis"
                )
            )
            tags.append(tag)
        
        return tags
    
    async def _generate_assessment_tags(self, assessment_data: Dict[str, Any], candidate_id: str) -> List[DynamicTag]:
        """Generate tags from assessment data"""
        tags = []
        
        # Generate seniority tags based on assessment scores
        overall_score = assessment_data.get("overall_score", 0)
        if overall_score > 80:
            tag = DynamicTag(
                category=TagCategory.SENIORITY,
                value="senior_level",
                confidence=ConfidenceScore(value=0.8, reasoning="High assessment score"),
                provenance=Provenance(
                    source=DataSource.ASSESSMENT,
                    source_id=candidate_id,
                    extraction_method="score_analysis"
                )
            )
            tags.append(tag)
        
        return tags
    
    async def _generate_interview_tags(self, interview_data: Dict[str, Any], candidate_id: str) -> List[DynamicTag]:
        """Generate tags from interview data"""
        tags = []
        
        # Generate communication style tags
        if interview_data.get("communication_score", 0) > 7:
            tag = DynamicTag(
                category=TagCategory.SOFT_SKILL,
                value="strong_communicator",
                confidence=ConfidenceScore(value=0.75, reasoning="High communication score"),
                provenance=Provenance(
                    source=DataSource.INTERVIEW,
                    source_id=candidate_id,
                    extraction_method="interview_scoring"
                )
            )
            tags.append(tag)
        
        return tags
    
    # ==================== EMBEDDING CREATION HELPERS ====================
    
    async def _create_resume_embeddings(self, resume_data: Dict[str, Any], candidate_id: str) -> List[MultiViewEmbedding]:
        """Create embeddings from resume data"""
        embeddings = []
        
        # Create comprehensive embedding
        resume_text = json.dumps(resume_data, sort_keys=True)
        embedding = MultiViewEmbedding(
            view_type=EmbeddingView.RESUME_CONTENT,
            vector=[0.1] * 384,  # Placeholder vector
            confidence=ConfidenceScore(value=0.9, reasoning="Direct resume content"),
            source_data={"resume_data": resume_data},
            metadata={"candidate_id": candidate_id}
        )
        embeddings.append(embedding)
        
        return embeddings
    
    async def _create_chat_embeddings(self, chat_data: Dict[str, Any], candidate_id: str) -> List[MultiViewEmbedding]:
        """Create embeddings from chat data"""
        embeddings = []
        
        # Create chat context embedding
        chat_text = json.dumps(chat_data, sort_keys=True)
        embedding = MultiViewEmbedding(
            view_type=EmbeddingView.CHAT_CONTEXT,
            vector=[0.2] * 384,  # Placeholder vector
            confidence=ConfidenceScore(value=0.7, reasoning="Chat conversation context"),
            source_data={"chat_data": chat_data},
            metadata={"candidate_id": candidate_id}
        )
        embeddings.append(embedding)
        
        return embeddings
    
    async def _create_assessment_embeddings(self, assessment_data: Dict[str, Any], candidate_id: str) -> List[MultiViewEmbedding]:
        """Create embeddings from assessment data"""
        embeddings = []
        
        # Create assessment response embedding
        assessment_text = json.dumps(assessment_data, sort_keys=True)
        embedding = MultiViewEmbedding(
            view_type=EmbeddingView.ASSESSMENT_RESPONSES,
            vector=[0.3] * 384,  # Placeholder vector
            confidence=ConfidenceScore(value=0.8, reasoning="Assessment responses"),
            source_data={"assessment_data": assessment_data},
            metadata={"candidate_id": candidate_id}
        )
        embeddings.append(embedding)
        
        return embeddings
    
    async def _create_interview_embeddings(self, interview_data: Dict[str, Any], candidate_id: str) -> List[MultiViewEmbedding]:
        """Create embeddings from interview data"""
        embeddings = []
        
        # Create interview response embedding
        interview_text = json.dumps(interview_data, sort_keys=True)
        embedding = MultiViewEmbedding(
            view_type=EmbeddingView.CHAT_CONTEXT,  # Reuse chat context for interview
            vector=[0.4] * 384,  # Placeholder vector
            confidence=ConfidenceScore(value=0.75, reasoning="Interview responses"),
            source_data={"interview_data": interview_data},
            metadata={"candidate_id": candidate_id}
        )
        embeddings.append(embedding)
        
        return embeddings
    
    # ==================== UTILITY METHODS ====================
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "queue_size": len(self.queue.events),
            "processing_count": len(self.queue.processing),
            "completed_count": len(self.queue.completed),
            "failed_count": len(self.queue.failed),
            "is_running": self.is_running,
            "active_tasks": len(self.processing_tasks)
        }
    
    def get_event_history(self, candidate_id: Optional[str] = None) -> List[ProcessingEvent]:
        """Get event history for a candidate or all candidates"""
        all_events = (
            self.queue.completed + 
            self.queue.failed + 
            list(self.queue.processing.values())
        )
        
        if candidate_id:
            return [event for event in all_events if event.candidate_id == candidate_id]
        
        return all_events


# ==================== SINGLETON INSTANCE ====================

# Global instance for the application
profile_event_processor = ProfileEventProcessor()

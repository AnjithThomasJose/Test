#!/usr/bin/env python3
"""
Background Summarization Service for Phase 3.
Implements background summarization of interactions into high-importance summaries.
"""

import asyncio
import datetime
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import uuid

from .memory_entries import memory_entry_manager, MemoryType
from .memory_manager import memory_manager
from .session_manager import session_manager
from models.llm_invoker import invoke_llm

log = logging.getLogger(__name__)

@dataclass
class SummarizationJob:
    """Represents a background summarization job."""
    job_id: str
    owner_id: str
    job_type: str  # "session_summary", "interaction_summary", "memory_consolidation"
    status: str  # "pending", "processing", "completed", "failed"
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class BackgroundSummarizer:
    """
    Background summarization service that condenses interactions into high-importance summaries.
    """
    
    def __init__(self):
        self.jobs_collection_name = "summarization_jobs"
        self.events_collection_name = "system_events"
        self.jobs_collection = None
        self.events_collection = None
        self.is_running = False
    
    def _get_collections(self):
        """Get or create the required collections."""
        if self.jobs_collection is None or self.events_collection is None:
            from chroma import client, embedding_fn
            self.jobs_collection = client.get_or_create_collection(
                name=self.jobs_collection_name,
                embedding_function=embedding_fn
            )
            self.events_collection = client.get_or_create_collection(
                name=self.events_collection_name,
                embedding_function=embedding_fn
            )
        return self.jobs_collection, self.events_collection
    
    def _log_event(self, event_type: str, owner_id: str, details: Dict[str, Any]):
        """Log system events for observability."""
        try:
            jobs_collection, events_collection = self._get_collections()
            
            event_id = f"event_{int(datetime.datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
            event_data = {
                "event_id": event_id,
                "event_type": event_type,
                "owner_id": owner_id,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "details": details
            }
            
            events_collection.add(
                ids=[event_id],
                documents=[json.dumps(event_data)],
                metadatas=[{
                    "event_type": event_type,
                    "owner_id": owner_id,
                    "timestamp": event_data["timestamp"]
                }]
            )
            
            log.debug(f"📊 Logged event: {event_type} for {owner_id}")
            
        except Exception as e:
            log.error(f"❌ Error logging event: {e}")
    
    async def create_session_summary(self, owner_id: str, session_id: str) -> str:
        """
        Create a comprehensive summary of a completed session.
        
        Args:
            owner_id: Owner of the session
            session_id: Session ID to summarize
        
        Returns:
            Job ID for tracking
        """
        try:
            job_id = f"session_summary_{owner_id}_{int(datetime.datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
            
            job = SummarizationJob(
                job_id=job_id,
                owner_id=owner_id,
                job_type="session_summary",
                status="pending",
                created_at=datetime.datetime.utcnow().isoformat(),
                metadata={"session_id": session_id}
            )
            
            # Store job
            jobs_collection, _ = self._get_collections()
            jobs_collection.add(
                ids=[job_id],
                documents=[json.dumps(job.__dict__)],
                metadatas=[{
                    "owner_id": owner_id,
                    "job_type": "session_summary",
                    "status": "pending"
                }]
            )
            
            # Log event
            self._log_event("summarization_job_created", owner_id, {
                "job_id": job_id,
                "job_type": "session_summary",
                "session_id": session_id
            })
            
            # Start background processing
            asyncio.create_task(self._process_session_summary(job))
            
            return job_id
            
        except Exception as e:
            log.error(f"❌ Error creating session summary job: {e}")
            return None
    
    async def _process_session_summary(self, job: SummarizationJob):
        """Process a session summary job."""
        try:
            # Update job status
            await self._update_job_status(job.job_id, "processing", started_at=datetime.datetime.utcnow().isoformat())
            
            # Get session data
            from chroma import get_chat_session
            session_data = get_chat_session(job.metadata["session_id"])
            if not session_data:
                raise Exception("Session data not found")
            
            # Extract key information
            session_info = self._extract_session_info(session_data)
            
            # Generate summary using LLM
            summary_content = await self._generate_session_summary(session_info)
            
            # Create summary memory entry
            memory_entry_manager.create_summary_entry(
                owner_id=job.owner_id,
                content=summary_content,
                importance_score=0.9,
                tags=["session_summary", "completed"],
                context={
                    "session_id": job.metadata["session_id"],
                    "job_id": job.job_id,
                    "summary_type": "session_completion"
                }
            )
            
            # Update job status
            await self._update_job_status(
                job.job_id, 
                "completed", 
                completed_at=datetime.datetime.utcnow().isoformat()
            )
            
            # Log completion event
            self._log_event("summarization_job_completed", job.owner_id, {
                "job_id": job.job_id,
                "job_type": "session_summary",
                "summary_length": len(summary_content)
            })
            
            log.info(f"✅ Session summary completed for {job.owner_id}")
            
        except Exception as e:
            log.error(f"❌ Error processing session summary: {e}")
            await self._update_job_status(
                job.job_id, 
                "failed", 
                completed_at=datetime.datetime.utcnow().isoformat(),
                error_message=str(e)
            )
    
    def _extract_session_info(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from session data."""
        info = {
            "session_id": session_data.get("session_id"),
            "uid": session_data.get("uid"),
            "status": session_data.get("status"),
            "timestamp": session_data.get("timestamp"),
            "chat_messages": len(session_data.get("chat_history", [])),
            "agents_completed": []
        }
        
        # Extract agent outputs
        for agent_name in ["resume_parser", "gap_analyzer", "assessment_recommender", 
                          "assessment_question_generator", "assessment_evaluator", "report_generator"]:
            if agent_name in session_data:
                agent_data = session_data[agent_name]
                info["agents_completed"].append(agent_name)
                
                # Extract key metrics
                if agent_name == "resume_parser":
                    structured_resume = agent_data.get("structured_resume", {})
                    info["candidate_name"] = structured_resume.get("Name", "Unknown")
                    info["skills_count"] = len(structured_resume.get("skills", []))
                    info["experience_count"] = len(structured_resume.get("experience", []))
                
                elif agent_name == "gap_analyzer":
                    analysis = agent_data.get("raw_skill_gap_analysis_output", {})
                    info["missing_skills"] = analysis.get("missingSkills", [])
                    info["market_insights"] = analysis.get("marketAnalysis", [])
                
                elif agent_name == "assessment_recommender":
                    plan = agent_data.get("assessment_plan", [])
                    info["assessment_count"] = len(plan)
                    needs = agent_data.get("assessment_needs", {})
                    info["skills_to_assess"] = needs.get("skills_to_assess", [])
                
                elif agent_name == "assessment_question_generator":
                    questions_data = agent_data.get("generated_questions", {})
                    questions = questions_data.get("questions", {}).get("multi", {}).get("questions", [])
                    info["questions_generated"] = len(questions)
                
                elif agent_name == "assessment_evaluator":
                    results = agent_data.get("assessment_results", {})
                    info["assessment_score"] = results.get("total_score", 0)
                    info["max_score"] = results.get("max_score", 100)
                
                elif agent_name == "report_generator":
                    report = agent_data.get("report", {})
                    info["report_generated"] = True
                    info["report_summary"] = report.get("summary", "")
        
        return info
    
    async def _generate_session_summary(self, session_info: Dict[str, Any]) -> str:
        """Generate a comprehensive session summary using LLM."""
        try:
            prompt = f"""
            Generate a comprehensive summary of this candidate assessment session.
            
            Session Information:
            - Candidate: {session_info.get('candidate_name', 'Unknown')}
            - Session ID: {session_info.get('session_id', 'Unknown')}
            - Status: {session_info.get('status', 'Unknown')}
            - Chat Messages: {session_info.get('chat_messages', 0)}
            - Agents Completed: {', '.join(session_info.get('agents_completed', []))}
            
            Resume Analysis:
            - Skills Identified: {session_info.get('skills_count', 0)}
            - Work Experience: {session_info.get('experience_count', 0)} positions
            
            Gap Analysis:
            - Missing Skills: {', '.join(session_info.get('missing_skills', [])[:5])}
            - Market Insights: {len(session_info.get('market_insights', []))} insights
            
            Assessment Plan:
            - Assessments Recommended: {session_info.get('assessment_count', 0)}
            - Skills to Assess: {', '.join(session_info.get('skills_to_assess', [])[:5])}
            - Questions Generated: {session_info.get('questions_generated', 0)}
            
            Assessment Results:
            - Score: {session_info.get('assessment_score', 0)}/{session_info.get('max_score', 100)}
            - Report Generated: {session_info.get('report_generated', False)}
            
            Create a professional summary that captures:
            1. Overall session outcome
            2. Key findings and insights
            3. Candidate strengths and areas for improvement
            4. Recommendations for next steps
            5. Session completion status
            
            Keep it concise but comprehensive (2-3 paragraphs).
            """
            
            llm_response = await invoke_llm(
            prompt=prompt,
            task_type="text_generation",
            agent_name="background_summarizer"
        )
            content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            return content.strip()
            
        except Exception as e:
            log.error(f"❌ Error generating session summary: {e}")
            return f"Session completed with {len(session_info.get('agents_completed', []))} agents. Status: {session_info.get('status', 'Unknown')}"
    
    async def create_interaction_summary(self, owner_id: str, interaction_data: Dict[str, Any]) -> str:
        """
        Create a summary of a specific interaction.
        
        Args:
            owner_id: Owner of the interaction
            interaction_data: Interaction data to summarize
        
        Returns:
            Job ID for tracking
        """
        try:
            job_id = f"interaction_summary_{owner_id}_{int(datetime.datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
            
            job = SummarizationJob(
                job_id=job_id,
                owner_id=owner_id,
                job_type="interaction_summary",
                status="pending",
                created_at=datetime.datetime.utcnow().isoformat(),
                metadata=interaction_data
            )
            
            # Store job
            jobs_collection, _ = self._get_collections()
            jobs_collection.add(
                ids=[job_id],
                documents=[json.dumps(job.__dict__)],
                metadatas=[{
                    "owner_id": owner_id,
                    "job_type": "interaction_summary",
                    "status": "pending"
                }]
            )
            
            # Log event
            self._log_event("summarization_job_created", owner_id, {
                "job_id": job_id,
                "job_type": "interaction_summary"
            })
            
            # Start background processing
            asyncio.create_task(self._process_interaction_summary(job))
            
            return job_id
            
        except Exception as e:
            log.error(f"❌ Error creating interaction summary job: {e}")
            return None
    
    async def _process_interaction_summary(self, job: SummarizationJob):
        """Process an interaction summary job."""
        try:
            # Update job status
            await self._update_job_status(job.job_id, "processing", started_at=datetime.datetime.utcnow().isoformat())
            
            # Generate summary
            summary_content = await self._generate_interaction_summary(job.metadata)
            
            # Create summary memory entry
            memory_entry_manager.create_summary_entry(
                owner_id=job.owner_id,
                content=summary_content,
                importance_score=0.8,
                tags=["interaction_summary"],
                context={
                    "job_id": job.job_id,
                    "summary_type": "interaction"
                }
            )
            
            # Update job status
            await self._update_job_status(
                job.job_id, 
                "completed", 
                completed_at=datetime.datetime.utcnow().isoformat()
            )
            
            log.info(f"✅ Interaction summary completed for {job.owner_id}")
            
        except Exception as e:
            log.error(f"❌ Error processing interaction summary: {e}")
            await self._update_job_status(
                job.job_id, 
                "failed", 
                completed_at=datetime.datetime.utcnow().isoformat(),
                error_message=str(e)
            )
    
    async def _generate_interaction_summary(self, interaction_data: Dict[str, Any]) -> str:
        """Generate a summary of an interaction."""
        try:
            prompt = f"""
            Summarize this interaction for long-term memory:
            
            Interaction Data:
            {json.dumps(interaction_data, indent=2)}
            
            Create a concise summary that captures:
            1. Key points discussed
            2. Important decisions made
            3. Next steps or outcomes
            4. Any notable insights
            
            Keep it brief but informative (1-2 sentences).
            """
            
            llm_response = await invoke_llm(
            prompt=prompt,
            task_type="text_generation",
            agent_name="background_summarizer"
        )
            content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            return content.strip()
            
        except Exception as e:
            log.error(f"❌ Error generating interaction summary: {e}")
            return f"Interaction summary: {interaction_data.get('type', 'Unknown')} - {interaction_data.get('status', 'Completed')}"
    
    async def _update_job_status(
        self, 
        job_id: str, 
        status: str, 
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        """Update job status in the database."""
        try:
            jobs_collection, _ = self._get_collections()
            
            # Get current job data
            job_data = jobs_collection.get(ids=[job_id])
            if not job_data or not job_data.get('documents'):
                return
            
            job_dict = json.loads(job_data['documents'][0])
            job_dict['status'] = status
            
            if started_at:
                job_dict['started_at'] = started_at
            if completed_at:
                job_dict['completed_at'] = completed_at
            if error_message:
                job_dict['error_message'] = error_message
            
            # Update job
            jobs_collection.update(
                ids=[job_id],
                documents=[json.dumps(job_dict)],
                metadatas=[{
                    "owner_id": job_dict.get("owner_id"),
                    "job_type": job_dict.get("job_type"),
                    "status": status
                }]
            )
            
        except Exception as e:
            log.error(f"❌ Error updating job status: {e}")
    
    async def run_memory_consolidation(self, owner_id: str) -> str:
        """
        Run memory consolidation for a user.
        
        Args:
            owner_id: Owner to consolidate memories for
        
        Returns:
            Job ID for tracking
        """
        try:
            job_id = f"memory_consolidation_{owner_id}_{int(datetime.datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
            
            job = SummarizationJob(
                job_id=job_id,
                owner_id=owner_id,
                job_type="memory_consolidation",
                status="pending",
                created_at=datetime.datetime.utcnow().isoformat(),
                metadata={"consolidation_type": "scheduled"}
            )
            
            # Store job
            jobs_collection, _ = self._get_collections()
            jobs_collection.add(
                ids=[job_id],
                documents=[json.dumps(job.__dict__)],
                metadatas=[{
                    "owner_id": owner_id,
                    "job_type": "memory_consolidation",
                    "status": "pending"
                }]
            )
            
            # Log event
            self._log_event("summarization_job_created", owner_id, {
                "job_id": job_id,
                "job_type": "memory_consolidation"
            })
            
            # Start background processing
            asyncio.create_task(self._process_memory_consolidation(job))
            
            return job_id
            
        except Exception as e:
            log.error(f"❌ Error creating memory consolidation job: {e}")
            return None
    
    async def _process_memory_consolidation(self, job: SummarizationJob):
        """Process a memory consolidation job."""
        try:
            # Update job status
            await self._update_job_status(job.job_id, "processing", started_at=datetime.datetime.utcnow().isoformat())
            
            # Run memory consolidation
            memory_manager.consolidate_memories(job.owner_id, max_age_days=30)
            
            # Update job status
            await self._update_job_status(
                job.job_id, 
                "completed", 
                completed_at=datetime.datetime.utcnow().isoformat()
            )
            
            # Log completion event
            self._log_event("summarization_job_completed", job.owner_id, {
                "job_id": job.job_id,
                "job_type": "memory_consolidation"
            })
            
            log.info(f"✅ Memory consolidation completed for {job.owner_id}")
            
        except Exception as e:
            log.error(f"❌ Error processing memory consolidation: {e}")
            await self._update_job_status(
                job.job_id, 
                "failed", 
                completed_at=datetime.datetime.utcnow().isoformat(),
                error_message=str(e)
            )
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a summarization job."""
        try:
            jobs_collection, _ = self._get_collections()
            job_data = jobs_collection.get(ids=[job_id])
            
            if not job_data or not job_data.get('documents'):
                return None
            
            return json.loads(job_data['documents'][0])
            
        except Exception as e:
            log.error(f"❌ Error getting job status: {e}")
            return None
    
    def get_events(self, owner_id: str, event_type: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get system events for observability."""
        try:
            _, events_collection = self._get_collections()
            
            where_clause = {"owner_id": owner_id}
            if event_type:
                where_clause["event_type"] = event_type
            
            results = events_collection.query(
                query_texts=["system event"],
                n_results=limit,
                where=where_clause
            )
            
            events = []
            for i, event_id in enumerate(results.get('ids', [])):
                if isinstance(event_id, list):
                    event_id = event_id[0]
                
                event_data = json.loads(results['documents'][i])
                events.append(event_data)
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return events
            
        except Exception as e:
            log.error(f"❌ Error getting events: {e}")
            return []

# Create a single instance to be used across the application
background_summarizer = BackgroundSummarizer()

#!/usr/bin/env python3
"""
Background Job Scheduler for Phase 3.
Manages scheduled background tasks like memory consolidation and cleanup.
"""

import asyncio
import datetime
import json
import logging
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import uuid

from .background_summarizer import background_summarizer
from .memory_manager import memory_manager
from .session_manager import session_manager

log = logging.getLogger(__name__)

@dataclass
class ScheduledJob:
    """Represents a scheduled background job."""
    job_id: str
    job_type: str
    schedule_type: str  # "interval", "cron", "once"
    schedule_config: Dict[str, Any]
    is_active: bool
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    created_at: str = ""
    metadata: Dict[str, Any] = None

class JobScheduler:
    """
    Background job scheduler for managing periodic tasks.
    
    Jobs run in thread pools to prevent blocking the main event loop,
    allowing API requests to continue processing while background jobs execute.
    
    Architecture for Non-Blocking Execution:
    1. Scheduler loop runs in main event loop (checks every 60 seconds)
    2. When job is due, creates asyncio.create_task() - NON-BLOCKING
    3. Task runs _run_job_with_semaphore() which uses run_blocking_io()
    4. run_blocking_io() executes _run_job_in_thread() in ThreadPoolExecutor
    5. Thread creates its own event loop, completely isolated from main loop
    6. API requests continue processing normally in main event loop
    
    This ensures:
    - Reranking jobs never block API calls
    - Long-running jobs don't impact system responsiveness
    - Each job has isolated execution environment
    - Maximum 5 concurrent background jobs (configurable via semaphore)
    """
    
    def __init__(self):
        self.jobs_collection_name = "scheduled_jobs"
        self.jobs_collection = None
        self.is_running = False
        self.tasks = {}
        self.running_jobs = set()  # Track currently running jobs to prevent duplicates
        self.job_semaphore = None  # Will be initialized lazily in async context
        
        # Register default jobs
        self._register_default_jobs()
    
    def _get_job_semaphore(self):
        """Get or create the job semaphore (lazy initialization for async context)."""
        if self.job_semaphore is None:
            # Semaphore to limit concurrent background jobs (prevents overwhelming the system)
            # Allow up to 5 concurrent background jobs to run in parallel
            # Jobs run in thread pools, so they won't block the main event loop
            self.job_semaphore = asyncio.Semaphore(5)
        return self.job_semaphore
    
    def _get_collection(self):
        """Get or create the scheduled jobs collection."""
        if self.jobs_collection is None:
            from chroma import client, embedding_fn
            self.jobs_collection = client.get_or_create_collection(
                name=self.jobs_collection_name,
                embedding_function=embedding_fn
            )
        return self.jobs_collection
    
    def _register_default_jobs(self):
        """Register default background jobs."""
        # Memory consolidation job (every 24 hours)
        self.schedule_job(
            job_type="memory_consolidation",
            schedule_type="interval",
            schedule_config={"hours": 24},
            metadata={"description": "Consolidate old memories into summaries"}
        )
        
        # Session cleanup job (every 6 hours)
        self.schedule_job(
            job_type="session_cleanup",
            schedule_type="interval",
            schedule_config={"hours": 6},
            metadata={"description": "Clean up expired sessions"}
        )
        
        # Memory cleanup job (every 12 hours)
        self.schedule_job(
            job_type="memory_cleanup",
            schedule_type="interval",
            schedule_config={"hours": 12},
            metadata={"description": "Clean up old low-importance memories"}
        )
        
        # KB Auto-Population job (every 6 hours) - TEMPORARILY DISABLED
        # TOGGLE: Check if enabled via environment variable
        enable_kb_auto_population = os.getenv("ENABLE_KB_AUTO_POPULATION", "false").lower() == "true"
        
        if enable_kb_auto_population:
            self.schedule_job(
                job_type="kb_auto_population",
                schedule_type="interval",
                schedule_config={"hours": 6},
                metadata={
                    "description": "Automatically populate KB with materials for missing topics",
                    "max_topics": 20,
                    "max_materials_per_topic": 5
                }
            )
            log.info("✅ KB Auto-Population job scheduled (ENABLE_KB_AUTO_POPULATION=true)")
        else:
            log.info("⏸️ KB Auto-Population job disabled (ENABLE_KB_AUTO_POPULATION=false or not set)")
    
    def schedule_job(
        self,
        job_type: str,
        schedule_type: str,
        schedule_config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Schedule a background job.
        
        Args:
            job_type: Type of job to schedule
            schedule_type: "interval", "cron", or "once"
            schedule_config: Configuration for the schedule
            metadata: Additional metadata for the job
        
        Returns:
            Job ID
        """
        try:
            job_id = f"scheduled_{job_type}_{int(datetime.datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
            
            # Calculate next run time
            next_run = self._calculate_next_run(schedule_type, schedule_config)
            
            job = ScheduledJob(
                job_id=job_id,
                job_type=job_type,
                schedule_type=schedule_type,
                schedule_config=schedule_config,
                is_active=True,
                created_at=datetime.datetime.utcnow().isoformat(),
                next_run=next_run,
                metadata=metadata or {}
            )
            
            # Store job
            collection = self._get_collection()
            collection.add(
                ids=[job_id],
                documents=[json.dumps(job.__dict__)],
                metadatas=[{
                    "job_type": job_type,
                    "schedule_type": schedule_type,
                    "is_active": True
                }]
            )
            
            log.info(f"📅 Scheduled job: {job_type} (next run: {next_run})")
            return job_id
            
        except Exception as e:
            log.error(f"❌ Error scheduling job: {e}")
            return None
    
    def _calculate_next_run(self, schedule_type: str, schedule_config: Dict[str, Any]) -> str:
        """Calculate the next run time for a job."""
        now = datetime.datetime.utcnow()
        
        if schedule_type == "interval":
            if "hours" in schedule_config:
                next_run = now + datetime.timedelta(hours=schedule_config["hours"])
            elif "minutes" in schedule_config:
                next_run = now + datetime.timedelta(minutes=schedule_config["minutes"])
            elif "days" in schedule_config:
                next_run = now + datetime.timedelta(days=schedule_config["days"])
            else:
                next_run = now + datetime.timedelta(hours=1)  # Default to 1 hour
        elif schedule_type == "daily_at":
            # Support for daily at specific time (e.g., "23:00" for 11 PM UTC)
            # Also supports IST times converted to UTC
            target_time_str = schedule_config.get("time", "00:00")  # Default to midnight
            target_hour, target_minute = map(int, target_time_str.split(":"))
            
            # Calculate next run time in UTC
            target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
            
            # If target time has passed today, schedule for tomorrow
            if target_time <= now:
                target_time += datetime.timedelta(days=1)
            
            next_run = target_time
        elif schedule_type == "daily_at_times":
            # Support for multiple times per day (e.g., ["18:30", "22:30", "02:30", ...])
            times = schedule_config.get("times", [])
            if not times:
                # Fallback to default if no times specified
                next_run = now + datetime.timedelta(hours=1)
            else:
                # Parse all times and find the next one
                target_times = []
                for time_str in times:
                    try:
                        hour, minute = map(int, time_str.split(":"))
                        # Create time for today
                        target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        # If time has passed today, schedule for tomorrow
                        if target_time <= now:
                            target_time += datetime.timedelta(days=1)
                        target_times.append(target_time)
                    except (ValueError, AttributeError):
                        continue
                
                if target_times:
                    # Find the earliest upcoming time
                    next_run = min(target_times)
                else:
                    # Fallback if no valid times
                    next_run = now + datetime.timedelta(hours=1)
        elif schedule_type == "cron":
            # Simple cron-like scheduling (hourly at minute 0)
            next_run = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
        elif schedule_type == "once":
            # Run once in 1 minute
            next_run = now + datetime.timedelta(minutes=1)
        else:
            next_run = now + datetime.timedelta(hours=1)
        
        return next_run.isoformat()
    
    async def start_scheduler(self):
        """Start the background job scheduler."""
        if self.is_running:
            log.warning("⚠️ Scheduler is already running")
            return
        
        self.is_running = True
        log.info("🚀 Starting background job scheduler...")
        
        # Start the main scheduler loop
        asyncio.create_task(self._scheduler_loop())
    
    async def stop_scheduler(self):
        """Stop the background job scheduler."""
        self.is_running = False
        
        # Cancel all running tasks
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        
        self.tasks.clear()
        log.info("🛑 Background job scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.is_running:
            try:
                await self._check_and_run_jobs()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                log.error(f"❌ Error in scheduler loop: {e}")
                await asyncio.sleep(60)
    
    async def _check_and_run_jobs(self):
        """Check for jobs that need to be run."""
        try:
            collection = self._get_collection()
            
            # Get all active jobs
            results = collection.query(
                query_texts=["scheduled job"],
                n_results=100,
                where={"is_active": True}
            )
            
            if not results or not results.get('ids'):
                return
            
            now = datetime.datetime.utcnow()
            
            # Handle nested list structure from ChromaDB
            ids = results.get('ids', [])
            documents = results.get('documents', [])
            
            # Unwrap nested lists if needed
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if documents and isinstance(documents[0], list):
                documents = documents[0]
            
            for i, job_id in enumerate(ids):
                if isinstance(job_id, list):
                    job_id = job_id[0]
                
                if i >= len(documents):
                    continue
                
                # Get document - handle both string and list formats
                doc = documents[i]
                if isinstance(doc, list):
                    doc = doc[0] if doc else None
                
                if not doc:
                    continue
                
                # Parse JSON - handle both string and already-parsed dict
                if isinstance(doc, str):
                    job_data = json.loads(doc)
                elif isinstance(doc, dict):
                    job_data = doc
                else:
                    log.warning(f"⚠️ Unexpected document type for job {job_id}: {type(doc)}")
                    continue
                
                next_run_str = job_data.get('next_run')
                
                if not next_run_str:
                    continue
                
                next_run = datetime.datetime.fromisoformat(next_run_str)
                
                # Check if job should run now
                if now >= next_run:
                    # Check if job is already running (prevent duplicates)
                    if job_id not in self.running_jobs:
                        # Run the job in parallel (completely non-blocking)
                        # Uses asyncio.create_task to run in background without blocking
                        # The job will execute in a separate thread pool, isolated from API requests
                        task = asyncio.create_task(self._run_job_with_semaphore(job_id, job_data))
                        self.tasks[job_id] = task
                        # Don't await - let it run in background thread pool
                        # This ensures API requests continue processing normally
                    else:
                        log.debug(f"⏭️ Job {job_id} already running, skipping duplicate trigger")
                    
        except Exception as e:
            log.error(f"❌ Error checking jobs: {e}", exc_info=True)
    
    async def _run_job_with_semaphore(self, job_id: str, job_data: Dict[str, Any]):
        """
        Run a scheduled job with semaphore to limit concurrent jobs.
        
        This method ensures jobs run in a separate thread pool, completely
        non-blocking the main event loop and API requests.
        
        CRITICAL: This is fire-and-forget - the task is created but not awaited,
        ensuring the scheduler loop continues immediately without blocking.
        """
        # Acquire semaphore to limit concurrent background jobs
        semaphore = self._get_job_semaphore()
        async with semaphore:
            # Run job in thread pool to prevent blocking the main event loop
            # This allows API requests to continue processing while jobs run
            # The thread pool executor handles the async execution in isolation
            from core.utils import run_blocking_io
            try:
                # Run in thread pool - this is completely non-blocking
                # The thread has its own event loop, so async jobs can run properly
                # This call is awaited but runs in a separate thread, so it doesn't block
                # the main event loop or API request processing
                await run_blocking_io(self._run_job_in_thread, job_id, job_data)
            except Exception as e:
                log.error(f"❌ Error in job thread execution for {job_id}: {e}", exc_info=True)
                # Remove from running jobs even on error
                self.running_jobs.discard(job_id)
                self.tasks.pop(job_id, None)
                raise
    
    def _run_job_in_thread(self, job_id: str, job_data: Dict[str, Any]):
        """
        Synchronous wrapper that runs the async job in a separate event loop.
        
        This runs in a thread pool executor, completely isolated from the main event loop.
        Per-job loop is intentional for isolation; no shared state with main loop (Section 4 Issue 2).
        This ensures:
        - API requests are never blocked by background jobs
        - Each job has its own event loop in its own thread
        - Long-running jobs don't impact system responsiveness
        
        Args:
            job_id: Unique identifier for the job
            job_data: Job configuration and metadata
        """
        # Create a new event loop for this thread (isolated from main application event loop)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        job_timeout_seconds = int(os.getenv("JOB_RUN_TIMEOUT_SECONDS", "0"))  # 0 = no timeout
        try:
            coro = self._run_job(job_id, job_data)
            if job_timeout_seconds > 0:
                loop.run_until_complete(asyncio.wait_for(asyncio.shield(coro), timeout=job_timeout_seconds))
            else:
                loop.run_until_complete(coro)
        except asyncio.TimeoutError:
            log.error(f"❌ Job {job_id} timed out after {job_timeout_seconds}s")
            raise
        except Exception as e:
            log.error(f"❌ Error in job thread {job_id}: {e}", exc_info=True)
            raise
        finally:
            # Clean up the event loop to free resources; no references held after close
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            finally:
                loop.close()
    
    async def _run_job(self, job_id: str, job_data: Dict[str, Any]):
        """Run a scheduled job."""
        # Mark job as running
        self.running_jobs.add(job_id)
        
        try:
            job_type = job_data.get('job_type')
            
            # Execute the job based on type
            # Use asyncio.create_task for long-running jobs to ensure they run in parallel
            if job_type == "memory_consolidation":
                await self._run_memory_consolidation_job(job_id, job_data)
            elif job_type == "session_cleanup":
                await self._run_session_cleanup_job(job_id, job_data)
            elif job_type == "memory_cleanup":
                await self._run_memory_cleanup_job(job_id, job_data)
            elif job_type == "kb_auto_population":
                await self._run_kb_population_job(job_id, job_data)
            # REMOVED: Auto-rerank and auto matching jobs
            # These have been removed - manual API calls still work via /api/ranker and /api/job-matcher
            # Silently skip these job types if they exist in the database
            elif job_type in ["auto_rerank_all_jds", "daily_rerank_all_jds", "daily_rematch_all_candidates"]:
                log.debug(f"⏭️ Skipping removed job type: {job_type} (auto-rerank/auto matching disabled)")
                # Mark job as inactive so it won't run again
                try:
                    collection = self._get_collection()
                    job_data['is_active'] = False
                    collection.update(
                        ids=[job_id],
                        documents=[json.dumps(job_data)],
                        metadatas=[{
                            "job_type": job_type,
                            "schedule_type": job_data.get('schedule_type'),
                            "is_active": False
                        }]
                    )
                except Exception as e:
                    log.debug(f"Could not update job {job_id}: {e}")
                return
            else:
                log.warning(f"⚠️ Unknown job type: {job_type}")
                return
            
            # Update job last run time and calculate next run
            await self._update_job_after_run(job_id, job_data)
            
        except Exception as e:
            log.error(f"❌ Error running job {job_id}: {e}")
            background_summarizer._log_event("scheduled_job_failed", "system", {
                "job_id": job_id,
                "job_type": job_data.get('job_type'),
                "error": str(e)
            })
        finally:
            # Remove from running jobs set
            self.running_jobs.discard(job_id)
            # Clean up task reference
            self.tasks.pop(job_id, None)
    
    async def _run_memory_consolidation_job(self, job_id: str, job_data: Dict[str, Any]):
        """Run memory consolidation for all users."""
        try:
            # Get all unique user IDs from memory entries
            from .memory_entries import memory_entry_manager
            # This would need to be implemented in memory_entry_manager
            # For now, we'll just run consolidation for active sessions
            
            # Get active sessions and run consolidation
            # This is a simplified version - in production, you'd want to get all users
            pass
            
        except Exception as e:
            log.error(f"❌ Error in memory consolidation job: {e}")
    
    async def _run_session_cleanup_job(self, job_id: str, job_data: Dict[str, Any]):
        """Run session cleanup for expired sessions."""
        try:
            # Clean up expired sessions
            # This would need to be implemented in session_manager
            pass
            
        except Exception as e:
            log.error(f"❌ Error in session cleanup job: {e}")
    
    async def _run_memory_cleanup_job(self, job_id: str, job_data: Dict[str, Any]):
        """Run memory cleanup for old low-importance memories."""
        try:
            # Clean up old low-importance memories
            # This would need to be implemented in memory_entry_manager
            pass
            
        except Exception as e:
            log.error(f"❌ Error in memory cleanup job: {e}")
    
    async def _run_kb_population_job(self, job_id: str, job_data: Dict[str, Any]):
        """Run KB auto-population job."""
        try:
            # Import here to avoid circular dependencies
            from agents.kb_auto_populator import kb_population_job_handler
            
            # Get job metadata
            metadata = job_data.get('metadata', {})
            
            # Run the job handler (logging is handled inside)
            await kb_population_job_handler(metadata)
            
        except Exception as e:
            log.error(f"❌ Error in KB population job: {e}")
    
    # REMOVED: Auto-rerank and auto matching job handlers
    # These methods have been removed as auto matching has been disabled
    # Manual API calls still work via /api/ranker and /api/job-matcher endpoints
    # 
    # async def _run_daily_rerank_job(self, job_id: str, job_data: Dict[str, Any]):
    #     """Run daily re-ranking for all job descriptions."""
    #     ...
    #
    # async def _run_daily_rematch_job(self, job_id: str, job_data: Dict[str, Any]):
    #     """Run daily re-matching for all candidates."""
    #     ...
    
    async def _update_job_after_run(self, job_id: str, job_data: Dict[str, Any]):
        """Update job after it has been run."""
        try:
            collection = self._get_collection()
            
            # Update last run time
            job_data['last_run'] = datetime.datetime.utcnow().isoformat()
            
            # Calculate next run time
            next_run = self._calculate_next_run(
                job_data.get('schedule_type'),
                job_data.get('schedule_config', {})
            )
            job_data['next_run'] = next_run
            
            # Update job in database
            collection.update(
                ids=[job_id],
                documents=[json.dumps(job_data)],
                metadatas=[{
                    "job_type": job_data.get('job_type'),
                    "schedule_type": job_data.get('schedule_type'),
                    "is_active": job_data.get('is_active', True)
                }]
            )
            
        except Exception as e:
            log.error(f"❌ Error updating job after run: {e}")
    
    def get_scheduled_jobs(self) -> List[Dict[str, Any]]:
        """Get all scheduled jobs."""
        try:
            collection = self._get_collection()
            
            results = collection.query(
                query_texts=["scheduled job"],
                n_results=100
            )
            
            jobs = []
            for i, job_id in enumerate(results.get('ids', [])):
                if isinstance(job_id, list):
                    job_id = job_id[0]
                
                job_data = json.loads(results['documents'][i])
                jobs.append(job_data)
            
            return jobs
            
        except Exception as e:
            log.error(f"❌ Error getting scheduled jobs: {e}")
            return []
    
    def pause_job(self, job_id: str):
        """Pause a scheduled job."""
        try:
            collection = self._get_collection()
            job_data = collection.get(ids=[job_id])
            
            if not job_data or not job_data.get('documents'):
                log.warning(f"⚠️ Job {job_id} not found")
                return
            
            job_dict = json.loads(job_data['documents'][0])
            job_dict['is_active'] = False
            
            collection.update(
                ids=[job_id],
                documents=[json.dumps(job_dict)],
                metadatas=[{
                    "job_type": job_dict.get('job_type'),
                    "schedule_type": job_dict.get('schedule_type'),
                    "is_active": False
                }]
            )
            
            log.info(f"⏸️ Job {job_id} paused")
            
        except Exception as e:
            log.error(f"❌ Error pausing job: {e}")
    
    def pause_jobs_by_type(self, job_type: str):
        """Pause all jobs of a specific type."""
        try:
            collection = self._get_collection()
            
            # ChromaDB get() doesn't support multiple conditions, so use query() or get all and filter
            # Get all jobs of this type (ChromaDB where clause only supports single condition)
            results = collection.get(
                where={"job_type": job_type}
            )
            
            if not results or not results.get('ids'):
                log.info(f"ℹ️ No jobs of type '{job_type}' found to pause")
                return
            
            paused_count = 0
            for job_id in results['ids']:
                if isinstance(job_id, list):
                    job_id = job_id[0]
                
                job_data = collection.get(ids=[job_id])
                if not job_data or not job_data.get('documents'):
                    continue
                
                job_dict = json.loads(job_data['documents'][0])
                
                # Only pause if currently active
                if not job_dict.get('is_active', False):
                    continue
                
                job_dict['is_active'] = False
                
                collection.update(
                    ids=[job_id],
                    documents=[json.dumps(job_dict)],
                    metadatas=[{
                        "job_type": job_dict.get('job_type'),
                        "schedule_type": job_dict.get('schedule_type'),
                        "is_active": False
                    }]
                )
                paused_count += 1
            
            if paused_count > 0:
                log.info(f"⏸️ Paused {paused_count} job(s) of type '{job_type}'")
            
        except Exception as e:
            log.error(f"❌ Error pausing jobs by type: {e}")
    
    def resume_job(self, job_id: str):
        """Resume a paused job."""
        try:
            collection = self._get_collection()
            job_data = collection.get(ids=[job_id])
            
            if not job_data or not job_data.get('documents'):
                log.warning(f"⚠️ Job {job_id} not found")
                return
            
            job_dict = json.loads(job_data['documents'][0])
            job_dict['is_active'] = True
            
            # Recalculate next run time
            next_run = self._calculate_next_run(
                job_dict.get('schedule_type'),
                job_dict.get('schedule_config', {})
            )
            job_dict['next_run'] = next_run
            
            collection.update(
                ids=[job_id],
                documents=[json.dumps(job_dict)],
                metadatas=[{
                    "job_type": job_dict.get('job_type'),
                    "schedule_type": job_dict.get('schedule_type'),
                    "is_active": True
                }]
            )
            
            log.info(f"▶️ Job {job_id} resumed")
            
        except Exception as e:
            log.error(f"❌ Error resuming job: {e}")

# Create a single instance to be used across the application
job_scheduler = JobScheduler()

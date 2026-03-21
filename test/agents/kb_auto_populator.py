"""
KB Auto-Populator - Background service for discovering and adding materials to KB.
Processes the topic queue created by the topic tracker.
"""

import asyncio
import logging
import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from pathlib import Path

from agents.course_knowledge_base import CourseKnowledgeBase
from agents.market_and_course_recommender import _check_and_discover_materials
from agents.topic_tracker import get_topic_tracker
from core.logging_helpers import AgentLogger, create_log_context
from core.utils import run_blocking_io
from chroma import _get_collection

log = logging.getLogger(__name__)

class KBAutoPopulator:
    """Service for automated KB population from topic queue"""
    
    def __init__(self):
        self.kb = CourseKnowledgeBase()
        self.topic_tracker = get_topic_tracker()
        self.logs_dir = Path("logs/kb_population")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        # ChromaDB collection for storing daily logs
        self.logs_collection = _get_collection("kb_population_logs")
    
    async def process_topic_queue(
        self,
        max_topics: int = 20,
        max_materials_per_topic: int = 5,
        log_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process pending topics from the queue and discover materials.
        
        IMPORTANT: This runs as a BACKGROUND job and should NEVER block API responses.
        All errors are caught and logged, but won't propagate to API calls.
        
        Args:
            max_topics: Maximum number of topics to process in this run
            max_materials_per_topic: Maximum materials to discover per topic
            log_context: Logging context
        
        Returns:
            Dict with processing statistics
        """
        log_context = log_context or create_log_context("kb_auto_population", tenant_id="system")
        
        stats = {
            "topics_processed": 0,
            "topics_successful": 0,
            "topics_failed": 0,
            "total_materials_added": 0,
            "materials_by_type": {"course": 0, "book": 0, "paper": 0, "video": 0, "tutorial": 0}
        }
        
        try:
            # Get pending topics from queue
            pending_topics = await run_blocking_io(self.topic_tracker.get_pending_topics, limit=max_topics)
            
            if not pending_topics:
                return stats
            
            # Process each topic (silently, no progress logs)
            for topic_data in pending_topics:
                topic = topic_data["topic"]
                topic_id = topic_data["topic_id"]
                
                try:
                    # Wrap in try-catch to ensure one topic failure doesn't stop the whole job
                    discovered = await _check_and_discover_materials(
                        topic=topic,
                        material_types=["course", "book", "paper", "video", "tutorial"],
                        min_results_required=2,
                        min_similarity_threshold=0.3,
                        max_results_per_type=max_materials_per_topic,
                        log_context=log_context
                    )
                    
                    if discovered:
                        # Count materials by type
                        for material in discovered:
                            material_type = material.get("type", "unknown")
                            if material_type in stats["materials_by_type"]:
                                stats["materials_by_type"][material_type] += 1
                            stats["total_materials_added"] += 1
                        
                        stats["topics_successful"] += 1
                    else:
                        stats["topics_failed"] += 1
                    
                except Exception as topic_error:
                    # Log but don't stop processing
                    log.warning(f"⚠️ Failed to process topic '{topic}': {topic_error}")
                    stats["topics_failed"] += 1
                
                finally:
                    # Always delete topic from database after processing (regardless of success/failure)
                    try:
                        await run_blocking_io(self.topic_tracker.delete_topic, topic_id)
                    except Exception as delete_error:
                        log.error(f"❌ Failed to delete topic {topic_id}: {delete_error}")
                    
                    stats["topics_processed"] += 1
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(1)
            
            await run_blocking_io(self._write_daily_log, stats, log_context)
            
        except Exception as e:
            # Catch ANY error to prevent blocking API calls
            log.error(f"❌ KB auto-population failed (this won't affect API calls): {e}", exc_info=True)
            # Still write a log entry
            try:
                await run_blocking_io(self._write_daily_log, stats, log_context)
            except:
                pass  # Don't let logging errors crash the job
        
        return stats
    
    def _write_daily_log(self, stats: Dict[str, Any], log_context: Dict[str, Any]):
        """
        Write a daily log entry for KB auto-population runs.
        Stores in ChromaDB and also writes to JSON file as backup.
        """
        today = date.today()
        today_str = today.isoformat()
        timestamp = datetime.utcnow()
        
        try:
            # Create a unique run ID
            run_id = f"{today_str}_{timestamp.isoformat()}_{hashlib.md5(str(timestamp.timestamp()).encode()).hexdigest()[:8]}"
            
            # Store individual run in ChromaDB
            run_document = json.dumps({
                "run_id": run_id,
                "date": today_str,
                "timestamp": timestamp.isoformat(),
                "stats": stats,
                "status": "completed" if stats['topics_processed'] > 0 else "no_topics"
            }, ensure_ascii=False)
            
            # Store in ChromaDB with metadata for easy querying
            self.logs_collection.add(
                ids=[run_id],
                documents=[run_document],
                metadatas=[{
                    "date": today_str,
                    "timestamp": timestamp.isoformat(),
                    "status": "completed" if stats['topics_processed'] > 0 else "no_topics",
                    "topics_processed": stats["topics_processed"],
                    "topics_successful": stats["topics_successful"],
                    "topics_failed": stats["topics_failed"],
                    "total_materials_added": stats["total_materials_added"],
                    "materials_course": stats["materials_by_type"].get("course", 0),
                    "materials_book": stats["materials_by_type"].get("book", 0),
                    "materials_paper": stats["materials_by_type"].get("paper", 0),
                    "materials_video": stats["materials_by_type"].get("video", 0),
                    "materials_tutorial": stats["materials_by_type"].get("tutorial", 0),
                }]
            )
            
            log.info(f"📝 Daily log stored in ChromaDB: {run_id}")
            
        except Exception as e:
            log.error(f"❌ Error writing daily log to ChromaDB: {e}", exc_info=True)
        
        # Always write JSON backup (even if ChromaDB succeeded)
        try:
            self._write_json_backup(today_str, stats, timestamp)
        except Exception as e2:
            log.error(f"❌ Error writing JSON backup: {e2}", exc_info=True)
    
    def _write_json_backup(self, date_str: str, stats: Dict[str, Any], timestamp: datetime):
        """Write JSON backup file"""
        log_file = self.logs_dir / f"kb_population_{date_str}.json"
        
        # Read existing log if it exists
        daily_log = {
            "date": date_str,
            "runs": [],
            "summary": {
                "total_runs": 0,
                "total_topics_processed": 0,
                "total_topics_successful": 0,
                "total_topics_failed": 0,
                "total_materials_added": 0,
                "materials_by_type": {"course": 0, "book": 0, "paper": 0, "video": 0, "tutorial": 0}
            }
        }
        
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    daily_log = json.load(f)
            except Exception as e:
                log.warning(f"⚠️ Error reading existing daily log: {e}, creating new one")
        
        # Add this run to the log
        run_entry = {
            "timestamp": timestamp.isoformat(),
            "stats": stats,
            "status": "completed" if stats['topics_processed'] > 0 else "no_topics"
        }
        daily_log["runs"].append(run_entry)
        
        # Update summary
        daily_log["summary"]["total_runs"] = len(daily_log["runs"])
        daily_log["summary"]["total_topics_processed"] += stats["topics_processed"]
        daily_log["summary"]["total_topics_successful"] += stats["topics_successful"]
        daily_log["summary"]["total_topics_failed"] += stats["topics_failed"]
        daily_log["summary"]["total_materials_added"] += stats["total_materials_added"]
        
        for material_type, count in stats["materials_by_type"].items():
            daily_log["summary"]["materials_by_type"][material_type] += count
        
        # Write updated log
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(daily_log, f, indent=2, ensure_ascii=False)
    
    def get_daily_log(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        """
        Get the daily log for a specific date from ChromaDB.
        
        Args:
            target_date: Date to get log for (defaults to today)
        
        Returns:
            Daily log dictionary or None if not found
        """
        if target_date is None:
            target_date = date.today()
        
        date_str = target_date.isoformat()
        
        try:
            # Query ChromaDB for all runs on this date
            results = self.logs_collection.get(
                where={"date": date_str}
            )
            
            if not results.get("ids") or not results["ids"]:
                # Fallback to JSON file if ChromaDB has no data
                return self._get_daily_log_from_json(target_date)
            
            # Reconstruct daily log from ChromaDB results
            runs = []
            for i, run_id in enumerate(results["ids"]):
                document = results.get("documents", [""])[i] if results.get("documents") else ""
                metadata = results.get("metadatas", [{}])[i] if results.get("metadatas") else {}
                
                try:
                    run_data = json.loads(document) if document else {}
                except:
                    # If document parsing fails, reconstruct from metadata
                    run_data = {
                        "run_id": run_id,
                        "date": date_str,
                        "timestamp": metadata.get("timestamp", ""),
                        "stats": {
                            "topics_processed": metadata.get("topics_processed", 0),
                            "topics_successful": metadata.get("topics_successful", 0),
                            "topics_failed": metadata.get("topics_failed", 0),
                            "total_materials_added": metadata.get("total_materials_added", 0),
                            "materials_by_type": {
                                "course": metadata.get("materials_course", 0),
                                "book": metadata.get("materials_book", 0),
                                "paper": metadata.get("materials_paper", 0),
                                "video": metadata.get("materials_video", 0),
                                "tutorial": metadata.get("materials_tutorial", 0),
                            }
                        },
                        "status": metadata.get("status", "unknown")
                    }
                
                runs.append(run_data)
            
            # Sort runs by timestamp
            runs.sort(key=lambda x: x.get("timestamp", ""))
            
            # Calculate summary
            summary = {
                "total_runs": len(runs),
                "total_topics_processed": sum(r.get("stats", {}).get("topics_processed", 0) for r in runs),
                "total_topics_successful": sum(r.get("stats", {}).get("topics_successful", 0) for r in runs),
                "total_topics_failed": sum(r.get("stats", {}).get("topics_failed", 0) for r in runs),
                "total_materials_added": sum(r.get("stats", {}).get("total_materials_added", 0) for r in runs),
                "materials_by_type": {
                    "course": sum(r.get("stats", {}).get("materials_by_type", {}).get("course", 0) for r in runs),
                    "book": sum(r.get("stats", {}).get("materials_by_type", {}).get("book", 0) for r in runs),
                    "paper": sum(r.get("stats", {}).get("materials_by_type", {}).get("paper", 0) for r in runs),
                    "video": sum(r.get("stats", {}).get("materials_by_type", {}).get("video", 0) for r in runs),
                    "tutorial": sum(r.get("stats", {}).get("materials_by_type", {}).get("tutorial", 0) for r in runs),
                }
            }
            
            return {
                "date": date_str,
                "runs": runs,
                "summary": summary
            }
            
        except Exception as e:
            log.error(f"❌ Error reading daily log from ChromaDB for {target_date}: {e}")
            # Fallback to JSON
            return self._get_daily_log_from_json(target_date)
    
    def _get_daily_log_from_json(self, target_date: date) -> Optional[Dict[str, Any]]:
        """Fallback method to read from JSON file"""
        log_file = self.logs_dir / f"kb_population_{target_date.isoformat()}.json"
        
        if not log_file.exists():
            return None
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log.error(f"❌ Error reading daily log from JSON for {target_date}: {e}")
            return None
    
    def did_run_today(self, target_date: Optional[date] = None) -> bool:
        """
        Check if KB auto-population ran on a specific date.
        
        Args:
            target_date: Date to check (defaults to today)
        
        Returns:
            True if the job ran on that date, False otherwise
        """
        daily_log = self.get_daily_log(target_date)
        return daily_log is not None and len(daily_log.get("runs", [])) > 0
    
    def get_daily_summary(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Get a summary of KB auto-population activity for a specific date.
        
        Args:
            target_date: Date to get summary for (defaults to today)
        
        Returns:
            Summary dictionary with run statistics
        """
        daily_log = self.get_daily_log(target_date)
        
        if daily_log is None:
            return {
                "date": (target_date or date.today()).isoformat(),
                "ran": False,
                "message": "No runs recorded for this date"
            }
        
        summary = daily_log.get("summary", {})
        return {
            "date": daily_log.get("date", (target_date or date.today()).isoformat()),
            "ran": True,
            "total_runs": summary.get("total_runs", 0),
            "total_topics_processed": summary.get("total_topics_processed", 0),
            "total_topics_successful": summary.get("total_topics_successful", 0),
            "total_topics_failed": summary.get("total_topics_failed", 0),
            "total_materials_added": summary.get("total_materials_added", 0),
            "materials_by_type": summary.get("materials_by_type", {}),
            "runs": daily_log.get("runs", [])
        }


# Singleton instance
_kb_auto_populator: Optional[KBAutoPopulator] = None

def get_kb_auto_populator() -> KBAutoPopulator:
    """Get singleton KBAutoPopulator instance"""
    global _kb_auto_populator
    if _kb_auto_populator is None:
        _kb_auto_populator = KBAutoPopulator()
    return _kb_auto_populator

async def kb_population_job_handler(job_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Job handler for scheduled KB population.
    Called by JobScheduler.
    
    Args:
        job_metadata: Job metadata from scheduler
    
    Returns:
        Dict with job status and results
    """
    start_time = datetime.utcnow()
    try:
        populator = get_kb_auto_populator()
        
        # Get configuration from metadata or use defaults
        max_topics = job_metadata.get("max_topics", 20)
        max_materials_per_topic = job_metadata.get("max_materials_per_topic", 5)
        
        # Process topic queue
        stats = await populator.process_topic_queue(
            max_topics=max_topics,
            max_materials_per_topic=max_materials_per_topic
        )
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        result = {
            "status": "completed",
            "results": stats,
            "timestamp": end_time.isoformat(),
            "start_time": start_time.isoformat(),
            "duration_seconds": duration
        }
        
        # Only log completion summary
        log.info(f"✅ KB Population: {stats['topics_processed']} topics, {stats['total_materials_added']} materials added ({duration:.1f}s)")
        
        return result
        
    except Exception as e:
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        log.error(f"❌ KB Population Job Handler failed after {duration:.2f}s: {e}", exc_info=True)
        
        # Still write a log entry for failed runs
        try:
            populator = get_kb_auto_populator()
            await run_blocking_io(populator._write_daily_log, {
                "topics_processed": 0,
                "topics_successful": 0,
                "topics_failed": 0,
                "total_materials_added": 0,
                "materials_by_type": {"course": 0, "book": 0, "paper": 0, "video": 0, "tutorial": 0}
            }, {"agent": "kb_auto_population", "tenant_id": "system"})
        except:
            pass  # Don't fail if logging fails
        
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": end_time.isoformat(),
            "start_time": start_time.isoformat(),
            "duration_seconds": duration
        }


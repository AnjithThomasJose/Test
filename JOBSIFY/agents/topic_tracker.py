"""
Topic Tracker - Tracks missing topics for background KB population.
Stores topics that need materials discovered in a queue for background processing.
"""

import logging
from typing import List, Dict, Any, Set
from datetime import datetime
from chroma import client, _get_collection

log = logging.getLogger(__name__)

class TopicTracker:
    """Tracks topics that need materials discovered"""
    
    def __init__(self):
        self.collection = _get_collection("missing_topics_queue")
        self.kb = None  # Lazy-loaded KB instance
        log.info("✅ Topic Tracker initialized")
    
    def _get_kb(self):
        """Lazy-load KB instance to avoid circular imports"""
        if self.kb is None:
            from agents.course_knowledge_base import CourseKnowledgeBase
            self.kb = CourseKnowledgeBase()
        return self.kb
    
    def _topic_exists_in_kb(self, topic: str) -> bool:
        """
        Check if a topic already has materials in the KB.
        
        Args:
            topic: Topic/skill name to check
        
        Returns:
            True if topic has materials in KB, False otherwise
        """
        try:
            kb = self._get_kb()
            
            # Search for materials related to this topic
            results = kb.search_materials(
                query=topic,
                material_types=None,  # All types
                top_k=1,  # Just need to know if at least 1 exists
                similarity_threshold=0.5  # Reasonable threshold
            )
            
            # If we found at least 1 material, topic exists in KB
            return len(results) > 0
            
        except Exception as e:
            log.debug(f"Could not check KB for topic '{topic}': {e}")
            # On error, assume topic doesn't exist (add it to queue)
            return False
    
    def add_topic(self, topic: str, source: str = "skill_gap", priority: int = 1) -> bool:
        """
        Add a topic to the queue for background discovery.
        Only adds if the topic doesn't already have materials in the KB.
        
        Args:
            topic: Topic/skill name (e.g., "React.js", "Machine Learning")
            source: Where this topic came from (e.g., "skill_gap", "user_query")
            priority: Priority level (1=high, 2=medium, 3=low)
        
        Returns:
            True if added successfully, False if skipped or error
        """
        try:
            # ✅ CHECK: Only add if topic doesn't exist in KB
            if self._topic_exists_in_kb(topic):
                log.debug(f"⏭️ Topic '{topic}' already has materials in KB, skipping queue addition")
                return False
            
            topic_id = f"{topic.lower().strip()}_{source}"
            
            # Check if topic already exists in queue
            existing = self.collection.get(ids=[topic_id])
            if existing.get("ids"):
                # Update existing topic (increment request count, update timestamp)
                metadata = existing.get("metadatas", [{}])[0]
                request_count = metadata.get("request_count", 0) + 1
                
                # Build metadata without None values (ChromaDB doesn't accept None)
                update_metadata = {
                    "topic": topic,
                    "source": source,
                    "priority": min(priority, metadata.get("priority", priority)),
                    "request_count": request_count,
                    "last_requested": datetime.utcnow().isoformat(),
                    "status": "pending"
                }
                # Only include processed_at if it exists and is not None
                if metadata.get("processed_at"):
                    update_metadata["processed_at"] = metadata.get("processed_at")
                
                self.collection.update(
                    ids=[topic_id],
                    metadatas=[update_metadata]
                )
                log.debug(f"Updated topic in queue: {topic} (count: {request_count})")
            else:
                # Add new topic (don't include processed_at for new topics - ChromaDB doesn't accept None)
                self.collection.add(
                    ids=[topic_id],
                    documents=[f"Topic: {topic} from {source}"],
                    metadatas=[{
                        "topic": topic,
                        "source": source,
                        "priority": priority,
                        "request_count": 1,
                        "last_requested": datetime.utcnow().isoformat(),
                        "status": "pending"
                    }]
                )
                log.info(f"✅ Added topic to queue: {topic}")
            
            return True
        except Exception as e:
            log.error(f"❌ Error adding topic '{topic}': {e}")
            return False
    
    def add_topics_batch(self, topics: List[str], source: str = "skill_gap", priority: int = 1) -> int:
        """Add multiple topics at once"""
        added_count = 0
        for topic in topics:
            if self.add_topic(topic, source, priority):
                added_count += 1
        return added_count
    
    def get_pending_topics(self, limit: int = 50, priority: int = None) -> List[Dict[str, Any]]:
        """
        Get pending topics from the queue.
        
        Args:
            limit: Maximum number of topics to return
            priority: Filter by priority (None = all priorities)
        
        Returns:
            List of topic dictionaries
        """
        try:
            where_clause = {"status": "pending"}
            if priority:
                where_clause["priority"] = priority
            
            results = self.collection.query(
                query_texts=["pending topic"],
                n_results=limit,
                where=where_clause
            )
            
            # Handle both dict and non-dict results
            if not results:
                return []
            
            # ChromaDB returns nested lists, so we need to unwrap them
            ids = results.get("ids", []) if isinstance(results, dict) else []
            metadatas = results.get("metadatas", []) if isinstance(results, dict) else []
            
            # Unwrap nested lists if needed
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if metadatas and isinstance(metadatas[0], list):
                metadatas = metadatas[0]
            
            topics = []
            for i, topic_id in enumerate(ids):
                # Handle nested topic_id
                if isinstance(topic_id, list):
                    topic_id = topic_id[0] if topic_id else None
                
                if not topic_id:
                    continue
                
                # Get metadata for this topic
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                # Handle nested metadata
                if isinstance(metadata, list):
                    metadata = metadata[0] if metadata else {}
                
                topics.append({
                    "topic_id": topic_id,
                    "topic": metadata.get("topic", ""),
                    "source": metadata.get("source", ""),
                    "priority": metadata.get("priority", 3),
                    "request_count": metadata.get("request_count", 1),
                    "last_requested": metadata.get("last_requested", "")
                })
            
            # Sort by priority (1=high) and request_count (descending)
            topics.sort(key=lambda x: (x["priority"], -x["request_count"]))
            
            return topics[:limit]
        
        except Exception as e:
            log.error(f"❌ Error getting pending topics: {e}", exc_info=True)
            return []
    
    def mark_topic_processed(self, topic_id: str, success: bool = True) -> bool:
        """Mark a topic as processed"""
        try:
            existing = self.collection.get(ids=[topic_id])
            if not existing.get("ids"):
                return False
            
            metadata = existing.get("metadatas", [{}])[0]
            # Build updated metadata, ensuring no None values
            updated_metadata = {
                "topic": metadata.get("topic", ""),
                "source": metadata.get("source", ""),
                "priority": metadata.get("priority", 3),
                "request_count": metadata.get("request_count", 1),
                "last_requested": metadata.get("last_requested", datetime.utcnow().isoformat()),
                "status": "processed" if success else "failed",
                "processed_at": datetime.utcnow().isoformat()
            }
            
            self.collection.update(
                ids=[topic_id],
                metadatas=[updated_metadata]
            )
            return True
        except Exception as e:
            log.error(f"❌ Error marking topic as processed: {e}")
            return False
    
    def delete_topic(self, topic_id: str) -> bool:
        """
        Delete a topic from the queue after processing.
        
        Args:
            topic_id: ID of the topic to delete
        
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Check if topic exists
            existing = self.collection.get(ids=[topic_id])
            if not existing.get("ids"):
                return False
            
            # Delete the topic
            self.collection.delete(ids=[topic_id])
            return True
        except Exception as e:
            log.error(f"❌ Error deleting topic '{topic_id}': {e}")
            return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the topic queue"""
        try:
            all_topics = self.collection.get()
            total = len(all_topics.get("ids", []))
            
            pending = self.collection.query(
                query_texts=["pending"],
                n_results=1000,
                where={"status": "pending"}
            )
            pending_count = len(pending.get("ids", []) or [])
            
            return {
                "total_topics": total,
                "pending_topics": pending_count,
                "processed_topics": total - pending_count
            }
        except Exception as e:
            log.error(f"❌ Error getting queue stats: {e}")
            return {"total_topics": 0, "pending_topics": 0, "processed_topics": 0}


# Singleton instance
_topic_tracker: TopicTracker = None

def get_topic_tracker() -> TopicTracker:
    """Get singleton TopicTracker instance"""
    global _topic_tracker
    if _topic_tracker is None:
        _topic_tracker = TopicTracker()
    return _topic_tracker

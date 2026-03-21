#!/usr/bin/env python3
"""
Enhanced Memory Entry System for Phase 2.
Implements typed memory entries with importance scores, tags, and personalization.
"""

import datetime
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import logging

log = logging.getLogger("main")

class MemoryType(Enum):
    """Types of memory entries as specified in PDF requirements."""
    FACT = "fact"
    INTERACTION = "interaction"
    SUMMARY = "summary"

class MemoryImportance(Enum):
    """Importance levels for memory entries."""
    LOW = 0.1
    MEDIUM = 0.5
    HIGH = 0.8
    CRITICAL = 1.0

@dataclass
class MemoryEntry:
    """
    Enhanced memory entry with typing, importance scoring, and personalization.
    Matches PDF requirements for long-term context memory.
    """
    entry_id: str
    owner_id: str
    memory_type: MemoryType
    content: str
    importance_score: float  # 0.0 to 1.0
    tags: List[str]
    created_at: str
    last_accessed_at: str
    access_count: int = 0
    metadata: Dict[str, Any] = None
    related_entries: List[str] = None  # IDs of related memory entries
    context: Dict[str, Any] = None  # Additional context for personalization

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.related_entries is None:
            self.related_entries = []
        if self.context is None:
            self.context = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "entry_id": self.entry_id,
            "owner_id": self.owner_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "importance_score": self.importance_score,
            "tags": self.tags,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "metadata": self.metadata,
            "related_entries": self.related_entries,
            "context": self.context
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary."""
        return cls(
            entry_id=data["entry_id"],
            owner_id=data["owner_id"],
            memory_type=MemoryType(data["memory_type"]),
            content=data["content"],
            importance_score=data["importance_score"],
            tags=data.get("tags", []),
            created_at=data["created_at"],
            last_accessed_at=data["last_accessed_at"],
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
            related_entries=data.get("related_entries", []),
            context=data.get("context", {})
        )

class MemoryEntryManager:
    """
    Manages typed memory entries with importance scoring and personalization.
    """
    
    def __init__(self):
        self.collection_name = "memory_entries"
        # This will be initialized when ChromaDB is available
        self.collection = None
    
    def _get_collection(self):
        """Get or create the memory entries collection."""
        if self.collection is None:
            from chroma import client, embedding_fn
            self.collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_fn
            )
        return self.collection
    
    def create_fact_entry(
        self,
        owner_id: str,
        content: str,
        importance_score: float = 0.5,
        tags: List[str] = None,
        context: Dict[str, Any] = None
    ) -> MemoryEntry:
        """
        Create a fact-type memory entry.
        
        Args:
            owner_id: Owner of the memory
            content: Factual content to remember
            importance_score: Importance from 0.0 to 1.0
            tags: List of tags for categorization
            context: Additional context for personalization
        
        Returns:
            Created MemoryEntry
        """
        now = datetime.datetime.utcnow().isoformat()
        entry_id = f"fact_{owner_id}_{int(datetime.datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
        
        entry = MemoryEntry(
            entry_id=entry_id,
            owner_id=owner_id,
            memory_type=MemoryType.FACT,
            content=content,
            importance_score=max(0.0, min(1.0, importance_score)),
            tags=tags or [],
            created_at=now,
            last_accessed_at=now,
            context=context or {}
        )
        
        self._store_entry(entry)
        return entry
    
    def create_interaction_entry(
        self,
        owner_id: str,
        content: str,
        importance_score: float = 0.7,
        tags: List[str] = None,
        context: Dict[str, Any] = None
    ) -> MemoryEntry:
        """
        Create an interaction-type memory entry.
        
        Args:
            owner_id: Owner of the memory
            content: Interaction content to remember
            importance_score: Importance from 0.0 to 1.0
            tags: List of tags for categorization
            context: Additional context for personalization
        
        Returns:
            Created MemoryEntry
        """
        now = datetime.datetime.utcnow().isoformat()
        entry_id = f"interaction_{owner_id}_{int(datetime.datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
        
        entry = MemoryEntry(
            entry_id=entry_id,
            owner_id=owner_id,
            memory_type=MemoryType.INTERACTION,
            content=content,
            importance_score=max(0.0, min(1.0, importance_score)),
            tags=tags or [],
            created_at=now,
            last_accessed_at=now,
            context=context or {}
        )
        
        self._store_entry(entry)
        return entry
    
    def create_summary_entry(
        self,
        owner_id: str,
        content: str,
        importance_score: float = 0.9,
        tags: List[str] = None,
        context: Dict[str, Any] = None
    ) -> MemoryEntry:
        """
        Create a summary-type memory entry.
        
        Args:
            owner_id: Owner of the memory
            content: Summary content to remember
            importance_score: Importance from 0.0 to 1.0
            tags: List of tags for categorization
            context: Additional context for personalization
        
        Returns:
            Created MemoryEntry
        """
        now = datetime.datetime.utcnow().isoformat()
        entry_id = f"summary_{owner_id}_{int(datetime.datetime.utcnow().timestamp())}_{uuid.uuid4().hex[:8]}"
        
        entry = MemoryEntry(
            entry_id=entry_id,
            owner_id=owner_id,
            memory_type=MemoryType.SUMMARY,
            content=content,
            importance_score=max(0.0, min(1.0, importance_score)),
            tags=tags or [],
            created_at=now,
            last_accessed_at=now,
            context=context or {}
        )
        
        self._store_entry(entry)
        return entry
    
    def _store_entry(self, entry: MemoryEntry):
        """Store memory entry in ChromaDB."""
        try:
            collection = self._get_collection()
            collection.add(
                ids=[entry.entry_id],
                documents=[entry.content],
                metadatas=[{
                    "owner_id": entry.owner_id,
                    "memory_type": entry.memory_type.value,
                    "importance_score": entry.importance_score,
                    "tags": json.dumps(entry.tags),
                    "created_at": entry.created_at,
                    "last_accessed_at": entry.last_accessed_at,
                    "access_count": entry.access_count,
                    "metadata": json.dumps(entry.metadata),
                    "related_entries": json.dumps(entry.related_entries),
                    "context": json.dumps(entry.context)
                }]
            )
            log.info(
                f"✅ Stored memory entry: {entry.entry_id} ({entry.memory_type.value})"
            )
        except Exception as e:
            log.error(f"❌ Error storing memory entry: {e}")
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID."""
        try:
            collection = self._get_collection()
            results = collection.get(ids=[entry_id])
            
            if not results['ids']:
                return None
            
            metadata = results['metadatas'][0]
            return MemoryEntry(
                entry_id=entry_id,
                owner_id=metadata["owner_id"],
                memory_type=MemoryType(metadata["memory_type"]),
                content=results['documents'][0],
                importance_score=metadata["importance_score"],
                tags=json.loads(metadata.get("tags", "[]")),
                created_at=metadata["created_at"],
                last_accessed_at=metadata["last_accessed_at"],
                access_count=metadata.get("access_count", 0),
                metadata=json.loads(metadata.get("metadata", "{}")),
                related_entries=json.loads(metadata.get("related_entries", "[]")),
                context=json.loads(metadata.get("context", "{}"))
            )
        except Exception as e:
            log.error(f"❌ Error retrieving memory entry {entry_id}: {e}")
            return None
    
    def query_entries(
        self,
        owner_id: str,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
        max_importance: float = 1.0,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Query memory entries with filtering and personalization.
        
        Args:
            owner_id: Owner to query for
            memory_type: Optional memory type filter
            min_importance: Minimum importance score
            max_importance: Maximum importance score
            tags: Optional tags to filter by
            limit: Maximum number of entries to return
        
        Returns:
            List of matching MemoryEntry objects
        """
        try:
            collection = self._get_collection()
            
            # Build where clause for filtering
            where_clause = {"owner_id": owner_id}
            if memory_type:
                where_clause["memory_type"] = memory_type.value
            
            # Query with filters
            results = collection.query(
                query_texts=["memory entry"],
                n_results=limit,
                where=where_clause
            )
            
            entries = []
            for i, entry_id in enumerate(results.get('ids', [])):
                if isinstance(entry_id, list):
                    entry_id = entry_id[0]
                
                # Handle metadata as list of lists
                metadata = results['metadatas'][i]
                if isinstance(metadata, list):
                    metadata = metadata[0]
                
                importance = metadata.get("importance_score", 0.0)
                
                # Apply importance filter
                if importance < min_importance or importance > max_importance:
                    continue
                
                # Apply tags filter
                if tags:
                    entry_tags = json.loads(metadata.get("tags", "[]"))
                    if not any(tag in entry_tags for tag in tags):
                        continue
                
                # Handle documents as list of lists
                content = results['documents'][i]
                if isinstance(content, list):
                    content = content[0]
                
                entry = MemoryEntry(
                    entry_id=entry_id,
                    owner_id=metadata["owner_id"],
                    memory_type=MemoryType(metadata["memory_type"]),
                    content=content,
                    importance_score=importance,
                    tags=json.loads(metadata.get("tags", "[]")),
                    created_at=metadata["created_at"],
                    last_accessed_at=metadata["last_accessed_at"],
                    access_count=metadata.get("access_count", 0),
                    metadata=json.loads(metadata.get("metadata", "{}")),
                    related_entries=json.loads(metadata.get("related_entries", "[]")),
                    context=json.loads(metadata.get("context", "{}"))
                )
                entries.append(entry)
            
            # Sort by importance score (descending)
            entries.sort(key=lambda x: x.importance_score, reverse=True)
            return entries[:limit]
            
        except Exception as e:
            log.error(f"❌ Error querying memory entries: {e}")
            return []
    
    def update_entry_access(self, entry_id: str):
        """Update access count and last accessed time."""
        try:
            entry = self.get_entry(entry_id)
            if not entry:
                return
            
            entry.access_count += 1
            entry.last_accessed_at = datetime.datetime.utcnow().isoformat()
            
            # Update in ChromaDB
            collection = self._get_collection()
            collection.update(
                ids=[entry_id],
                metadatas=[{
                    "owner_id": entry.owner_id,
                    "memory_type": entry.memory_type.value,
                    "importance_score": entry.importance_score,
                    "tags": json.dumps(entry.tags),
                    "created_at": entry.created_at,
                    "last_accessed_at": entry.last_accessed_at,
                    "access_count": entry.access_count,
                    "metadata": json.dumps(entry.metadata),
                    "related_entries": json.dumps(entry.related_entries),
                    "context": json.dumps(entry.context)
                }]
            )
        except Exception as e:
            log.error(f"❌ Error updating entry access: {e}")
    
    def get_personalized_context(
        self,
        owner_id: str,
        context_type: str = "general",
        max_entries: int = 5
    ) -> str:
        """
        Get personalized context for LLM based on memory entries.
        
        Args:
            owner_id: Owner to get context for
            context_type: Type of context needed (general, technical, behavioral, etc.)
            max_entries: Maximum number of entries to include
        
        Returns:
            Formatted context string for LLM
        """
        try:
            # Query high-importance entries
            entries = self.query_entries(
                owner_id=owner_id,
                min_importance=0.6,
                limit=max_entries
            )
            
            if not entries:
                return ""
            
            context_parts = []
            for entry in entries:
                context_parts.append(f"[{entry.memory_type.value.upper()}] {entry.content}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            log.error(f"❌ Error getting personalized context: {e}")
            return ""
    
    def consolidate_memories(self, owner_id: str, max_age_days: int = 30):
        """
        Consolidate old memories into summaries.
        
        Args:
            owner_id: Owner to consolidate memories for
            max_age_days: Maximum age in days before consolidation
        """
        try:
            cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=max_age_days)
            
            # Get old low-importance entries
            old_entries = self.query_entries(
                owner_id=owner_id,
                memory_type=MemoryType.FACT,
                max_importance=0.4,
                limit=50
            )
            
            # Filter by age
            old_entries = [
                entry for entry in old_entries
                if datetime.datetime.fromisoformat(entry.created_at) < cutoff_date
            ]
            
            if len(old_entries) < 5:  # Need at least 5 entries to consolidate
                return
            
            # Create summary entry
            summary_content = f"Consolidated {len(old_entries)} low-importance facts from {max_age_days} days ago."
            summary_entry = self.create_summary_entry(
                owner_id=owner_id,
                content=summary_content,
                importance_score=0.7,
                tags=["consolidated", "old_memories"],
                context={"consolidated_count": len(old_entries), "period_days": max_age_days}
            )
            
            log.info(
                f"✅ Consolidated {len(old_entries)} old memories into summary: "
                f"{summary_entry.entry_id}"
            )
            
        except Exception as e:
            log.error(f"❌ Error consolidating memories: {e}")

# Create a single instance to be used across the application
memory_entry_manager = MemoryEntryManager()

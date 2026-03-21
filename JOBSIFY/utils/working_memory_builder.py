#!/usr/bin/env python3
"""
Working Memory Builder for Phase 2.
Implements token-bounded context selection, relevance scoring, and pre-LLM context optimization.
"""

import datetime
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json

from .memory_entries import MemoryEntry, MemoryType, memory_entry_manager

log = logging.getLogger(__name__)

@dataclass
class ContextItem:
    """Represents a context item with relevance score and token count."""
    content: str
    relevance_score: float
    token_count: int
    source: str  # "memory", "session", "chat_history"
    entry_id: Optional[str] = None
    importance_score: float = 0.0

class WorkingMemoryBuilder:
    """
    Builds optimized working memory for LLM context.
    Implements token-bounded context selection and relevance scoring.
    """
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.estimated_tokens_per_char = 0.25  # Rough estimation
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) * self.estimated_tokens_per_char)
    
    def build_working_memory(
        self,
        owner_id: str,
        current_context: str = "",
        context_type: str = "general",
        include_chat_history: bool = True,
        include_session_data: bool = True,
        include_memory_entries: bool = True,
        chat_history: List[Dict[str, Any]] = None
    ) -> str:
        """
        Build optimized working memory for LLM context.
        
        Args:
            owner_id: Owner to build context for
            current_context: Current context from the request
            context_type: Type of context needed (technical, behavioral, etc.)
            include_chat_history: Whether to include chat history
            include_session_data: Whether to include session data
            include_memory_entries: Whether to include memory entries
        
        Returns:
            Optimized context string for LLM
        """
        try:
            context_items = []
            remaining_tokens = self.max_tokens
            
            # Add current context first (highest priority)
            if current_context:
                current_tokens = self.estimate_tokens(current_context)
                context_items.append(ContextItem(
                    content=current_context,
                    relevance_score=1.0,
                    token_count=current_tokens,
                    source="current",
                    importance_score=1.0
                ))
                remaining_tokens -= current_tokens
            
            # Add chat history if requested
            if include_chat_history and remaining_tokens > 0:
                chat_items = self._get_chat_history_context(owner_id, remaining_tokens, chat_history)
                context_items.extend(chat_items)
                remaining_tokens -= sum(item.token_count for item in chat_items)
            
            # Add session data if requested
            if include_session_data and remaining_tokens > 0:
                session_items = self._get_session_context(owner_id, remaining_tokens)
                context_items.extend(session_items)
                remaining_tokens -= sum(item.token_count for item in session_items)
            
            # Add memory entries if requested
            if include_memory_entries and remaining_tokens > 0:
                memory_items = self._get_memory_entries_context(
                    owner_id, context_type, remaining_tokens
                )
                context_items.extend(memory_items)
                remaining_tokens -= sum(item.token_count for item in memory_items)
            
            # Optimize and format context
            optimized_context = self._optimize_context(context_items)
            
            log.debug(f"🧠 Built working memory: {len(context_items)} items, ~{self.max_tokens - remaining_tokens} tokens")
            return optimized_context
            
        except Exception as e:
            log.error(f"❌ Error building working memory: {e}")
            return current_context or ""
    
    def _get_chat_history_context(self, owner_id: str, max_tokens: int, chat_history: List[Dict[str, Any]] = None) -> List[ContextItem]:
        """Get relevant chat history context."""
        try:
            if not chat_history:
                return []
            
            items = []
            for msg in chat_history[-10:]:  # Last 10 messages
                content = f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                tokens = self.estimate_tokens(content)
                
                if tokens <= max_tokens:
                    items.append(ContextItem(
                        content=content,
                        relevance_score=0.8,
                        token_count=tokens,
                        source="chat_history",
                        importance_score=0.7
                    ))
                    max_tokens -= tokens
                else:
                    break
            
            return items
            
        except Exception as e:
            log.error(f"❌ Error getting chat history context: {e}")
            return []
    
    def _get_session_context(self, owner_id: str, max_tokens: int) -> List[ContextItem]:
        """Get relevant session data context."""
        try:
            from .session_manager import session_manager
            
            session = session_manager.get_session_by_owner(owner_id, "candidate_pipeline")
            if not session:
                return []
            
            # Get session data
            from chroma import get_chat_session
            session_data = get_chat_session(session.session_id)
            if not session_data:
                return []
            
            items = []
            
            # Add resume parser data
            if "resume_parser" in session_data:
                resume_data = session_data["resume_parser"]
                content = f"Resume: {resume_data.get('structured_resume', {}).get('Name', 'Candidate')} - {len(resume_data.get('structured_resume', {}).get('skills', []))} skills"
                tokens = self.estimate_tokens(content)
                
                if tokens <= max_tokens:
                    items.append(ContextItem(
                        content=content,
                        relevance_score=0.9,
                        token_count=tokens,
                        source="session",
                        importance_score=0.9
                    ))
                    max_tokens -= tokens
            
            # Add gap analysis data
            if "gap_analyzer" in session_data:
                gap_data = session_data["gap_analyzer"]
                analysis = gap_data.get("raw_skill_gap_analysis_output", {})
                content = f"Gap Analysis: {len(analysis.get('missingSkills', []))} missing skills identified"
                tokens = self.estimate_tokens(content)
                
                if tokens <= max_tokens:
                    items.append(ContextItem(
                        content=content,
                        relevance_score=0.8,
                        token_count=tokens,
                        source="session",
                        importance_score=0.8
                    ))
                    max_tokens -= tokens
            
            # Add assessment data
            if "assessment_recommender" in session_data:
                assessment_data = session_data["assessment_recommender"]
                content = f"Assessment Plan: {len(assessment_data.get('assessment_plan', []))} assessments recommended"
                tokens = self.estimate_tokens(content)
                
                if tokens <= max_tokens:
                    items.append(ContextItem(
                        content=content,
                        relevance_score=0.7,
                        token_count=tokens,
                        source="session",
                        importance_score=0.7
                    ))
                    max_tokens -= tokens
            
            return items
            
        except Exception as e:
            log.error(f"❌ Error getting session context: {e}")
            return []
    
    def _get_memory_entries_context(
        self, 
        owner_id: str, 
        context_type: str, 
        max_tokens: int
    ) -> List[ContextItem]:
        """Get relevant memory entries context."""
        try:
            # Query memory entries based on context type
            tags = self._get_context_tags(context_type)
            entries = memory_entry_manager.query_entries(
                owner_id=owner_id,
                min_importance=0.5,
                tags=tags,
                limit=10
            )
            
            items = []
            for entry in entries:
                content = f"[{entry.memory_type.value.upper()}] {entry.content}"
                tokens = self.estimate_tokens(content)
                
                if tokens <= max_tokens:
                    items.append(ContextItem(
                        content=content,
                        relevance_score=entry.importance_score,
                        token_count=tokens,
                        source="memory",
                        entry_id=entry.entry_id,
                        importance_score=entry.importance_score
                    ))
                    max_tokens -= tokens
                else:
                    break
            
            return items
            
        except Exception as e:
            log.error(f"❌ Error getting memory entries context: {e}")
            return []
    
    def _get_context_tags(self, context_type: str) -> List[str]:
        """Get relevant tags for context type."""
        tag_mapping = {
            "technical": ["technical", "skills", "coding", "programming"],
            "behavioral": ["behavioral", "soft_skills", "leadership", "teamwork"],
            "assessment": ["assessment", "evaluation", "scores", "performance"],
            "resume": ["resume", "experience", "education", "background"],
            "general": []
        }
        return tag_mapping.get(context_type, [])
    
    def _optimize_context(self, context_items: List[ContextItem]) -> str:
        """Optimize and format context for LLM."""
        if not context_items:
            return ""
        
        # Sort by relevance score (descending)
        context_items.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Group by source
        sections = {
            "current": [],
            "session": [],
            "memory": [],
            "chat_history": []
        }
        
        for item in context_items:
            if item.source in sections:
                sections[item.source].append(item)
        
        # Build formatted context
        context_parts = []
        
        # Current context (highest priority)
        if sections["current"]:
            context_parts.append("## Current Context")
            for item in sections["current"]:
                context_parts.append(item.content)
        
        # Session data
        if sections["session"]:
            context_parts.append("## Session Data")
            for item in sections["session"]:
                context_parts.append(item.content)
        
        # Memory entries
        if sections["memory"]:
            context_parts.append("## Relevant Memories")
            for item in sections["memory"]:
                context_parts.append(item.content)
        
        # Chat history
        if sections["chat_history"]:
            context_parts.append("## Recent Conversation")
            for item in sections["chat_history"]:
                context_parts.append(item.content)
        
        return "\n\n".join(context_parts)
    
    def create_memory_from_interaction(
        self,
        owner_id: str,
        interaction_content: str,
        context_type: str = "general",
        importance_score: float = 0.7
    ) -> Optional[str]:
        """
        Create a memory entry from an interaction.
        
        Args:
            owner_id: Owner of the memory
            interaction_content: Content of the interaction
            context_type: Type of context
            importance_score: Importance score for the memory
        
        Returns:
            Created memory entry ID or None
        """
        try:
            tags = self._get_context_tags(context_type)
            
            entry = memory_entry_manager.create_interaction_entry(
                owner_id=owner_id,
                content=interaction_content,
                importance_score=importance_score,
                tags=tags,
                context={"context_type": context_type}
            )
            
            log.info(f"✅ Created memory entry from interaction: {entry.entry_id}")
            return entry.entry_id
            
        except Exception as e:
            log.error(f"❌ Error creating memory from interaction: {e}")
            return None
    
    def create_memory_from_fact(
        self,
        owner_id: str,
        fact_content: str,
        context_type: str = "general",
        importance_score: float = 0.6
    ) -> Optional[str]:
        """
        Create a memory entry from a fact.
        
        Args:
            owner_id: Owner of the memory
            fact_content: Factual content
            context_type: Type of context
            importance_score: Importance score for the memory
        
        Returns:
            Created memory entry ID or None
        """
        try:
            tags = self._get_context_tags(context_type)
            
            entry = memory_entry_manager.create_fact_entry(
                owner_id=owner_id,
                content=fact_content,
                importance_score=importance_score,
                tags=tags,
                context={"context_type": context_type}
            )
            
            log.info(f"✅ Created memory entry from fact: {entry.entry_id}")
            return entry.entry_id
            
        except Exception as e:
            log.error(f"❌ Error creating memory from fact: {e}")
            return None

# Create a single instance to be used across the application
working_memory_builder = WorkingMemoryBuilder()

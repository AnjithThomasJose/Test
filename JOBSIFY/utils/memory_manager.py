import datetime
import logging
from typing import List, Dict, Any, Optional
from chroma import chat_sessions_collection
from .memory_entries import memory_entry_manager
from .working_memory_builder import working_memory_builder
from .session_manager import session_manager

log = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages persistent chat history using ChromaDB for consistency.
    """
    def __init__(self):
        self.collection = chat_sessions_collection

    def get_chat_history(self, uid: str) -> List[Dict[str, Any]]:
        """
        Retrieves the chat history for a given user ID from ChromaDB.
        """
        try:
            # Query for the most recent session for this UID
            results = self.collection.query(
                query_texts=["session data"],
                n_results=1,
                where={"uid": uid}
            )
            
            if results and results.get('ids') and len(results['ids']) > 0:
                # Get the most recent session
                session_id = results['ids'][0]
                # Ensure session_id is a string, not a list
                if isinstance(session_id, list):
                    session_id = session_id[0]
                session_data = self.collection.get(ids=[session_id])
                
                if session_data and session_data.get('documents'):
                    import json
                    session = json.loads(session_data['documents'][0])
                    return session.get("chat_history", [])
            return []
        except Exception as e:
            log.error(f"🚨 Error fetching chat history for UID {uid}: {e}")
            return []

    def update_chat_history(self, uid: str, new_messages: List[Dict[str, Any]]):
        """
        Updates chat history in the existing session in ChromaDB.
        """
        if not new_messages:
            return

        try:
            # Find existing session
            results = self.collection.query(
                query_texts=["session data"],
                n_results=1,
                where={"uid": uid}
            )
            
            if results and results.get('ids') and len(results['ids']) > 0:
                session_id = results['ids'][0]
                # Ensure session_id is a string, not a list
                if isinstance(session_id, list):
                    session_id = session_id[0]
                
                # Get current session data
                session_data = self.collection.get(ids=[session_id])
                if session_data and session_data.get('documents'):
                    import json
                    session = json.loads(session_data['documents'][0])
                    
                    # Add timestamps to new messages
                    timestamp = datetime.datetime.utcnow()
                    for msg in new_messages:
                        msg["timestamp"] = timestamp.isoformat()
                    
                    # Update chat history
                    if "chat_history" not in session:
                        session["chat_history"] = []
                    session["chat_history"].extend(new_messages)
                    
                    # Update the session in ChromaDB
                    self.collection.update(
                        ids=[session_id],
                        documents=[json.dumps(session)],
                        metadatas=[{
                            "agent": "memory_manager",
                            "uid": uid,
                            "last_updated": timestamp.isoformat()
                        }]
                    )
                    log.info(f"🧠 Memory updated for UID {uid} with {len(new_messages)} new message(s).")
            else:
                log.warning(f"⚠️ No existing session found for UID {uid}, cannot update chat history")
                
        except Exception as e:
            log.error(f"🚨 Error updating chat history for UID {uid}: {e}")

    def clear_chat_history(self, uid: str):
        """
        Clears the chat history for a given user ID.
        """
        try:
            # Find existing session
            results = self.collection.query(
                query_texts=["session data"],
                n_results=1,
                where={"uid": uid}
            )
            
            if results and results.get('ids') and len(results['ids']) > 0:
                session_id = results['ids'][0]
                # Ensure session_id is a string, not a list
                if isinstance(session_id, list):
                    session_id = session_id[0]
                
                # Get current session data
                session_data = self.collection.get(ids=[session_id])
                if session_data and session_data.get('documents'):
                    import json
                    session = json.loads(session_data['documents'][0])
                    
                    # Clear chat history
                    session["chat_history"] = []
                    
                    # Update the session in ChromaDB
                    self.collection.update(
                        ids=[session_id],
                        documents=[json.dumps(session)],
                        metadatas=[{
                            "agent": "memory_manager",
                            "uid": uid,
                            "last_updated": datetime.datetime.utcnow().isoformat()
                        }]
                    )
                    log.info(f"🧹 Chat history cleared for UID {uid}")
        except Exception as e:
            log.error(f"🚨 Error clearing chat history for UID {uid}: {e}")

    def format_chat_history(self, chat_history: List[Dict[str, Any]]) -> str:
        """
        Formats chat history into a readable string for LLM context.
        """
        if not chat_history:
            return ""
        
        history_lines = []
        for msg in chat_history:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            history_lines.append(f"{role} ({timestamp}): {content}")
        
        return "\n\nConversation History:\n" + "\n".join(history_lines)

    def get_rich_context_for_llm(self, uid: str, context_type: str = "general") -> str:
        """
        Get rich context for LLM using enhanced working memory builder.
        
        Args:
            uid: User ID to get context for
            context_type: Type of context needed (technical, behavioral, etc.)
        
        Returns:
            Formatted context string for LLM
        """
        try:
            # Get chat history for the working memory builder
            chat_history = self.get_chat_history(uid)
            
            # Use working memory builder for optimized context
            return working_memory_builder.build_working_memory(
                owner_id=uid,
                current_context="",
                context_type=context_type,
                include_chat_history=True,
                include_session_data=True,
                include_memory_entries=True,
                chat_history=chat_history
            )
            
        except Exception as e:
            log.error(f"❌ Error getting rich context for UID={uid}: {e}")
            return ""
    
    def create_memory_from_interaction(
        self, 
        uid: str, 
        content: str, 
        context_type: str = "general",
        importance_score: float = 0.7
    ) -> Optional[str]:
        """
        Create a memory entry from an interaction.
        
        Args:
            uid: User ID
            content: Interaction content
            context_type: Type of context
            importance_score: Importance score (0.0 to 1.0)
        
        Returns:
            Created memory entry ID or None
        """
        try:
            return working_memory_builder.create_memory_from_interaction(
                owner_id=uid,
                interaction_content=content,
                context_type=context_type,
                importance_score=importance_score
            )
        except Exception as e:
            log.error(f"❌ Error creating memory from interaction: {e}")
            return None
    
    def create_memory_from_fact(
        self, 
        uid: str, 
        content: str, 
        context_type: str = "general",
        importance_score: float = 0.6
    ) -> Optional[str]:
        """
        Create a memory entry from a fact.
        
        Args:
            uid: User ID
            content: Factual content
            context_type: Type of context
            importance_score: Importance score (0.0 to 1.0)
        
        Returns:
            Created memory entry ID or None
        """
        try:
            return working_memory_builder.create_memory_from_fact(
                owner_id=uid,
                fact_content=content,
                context_type=context_type,
                importance_score=importance_score
            )
        except Exception as e:
            log.error(f"❌ Error creating memory from fact: {e}")
            return None
    
    def get_personalized_context(self, uid: str, context_type: str = "general") -> str:
        """
        Get personalized context from memory entries.
        
        Args:
            uid: User ID
            context_type: Type of context needed
        
        Returns:
            Personalized context string
        """
        try:
            return memory_entry_manager.get_personalized_context(
                owner_id=uid,
                context_type=context_type,
                max_entries=5
            )
        except Exception as e:
            log.error(f"❌ Error getting personalized context: {e}")
            return ""
    
    def consolidate_memories(self, uid: str, max_age_days: int = 30):
        """
        Consolidate old memories into summaries.
        
        Args:
            uid: User ID
            max_age_days: Maximum age in days before consolidation
        """
        try:
            memory_entry_manager.consolidate_memories(uid, max_age_days)
        except Exception as e:
            log.error(f"❌ Error consolidating memories: {e}")

    def get_session_summary(self, uid: str) -> Dict[str, Any]:
        """
        Gets a summary of the current session for quick reference.
        """
        try:
            # Use a more generic query that will find any session with this UID
            results = self.collection.query(
                query_texts=["session data"],
                n_results=1,
                where={"uid": uid}
            )
            
            if not results or not results.get('ids') or len(results['ids']) == 0:
                return {"error": "No session found"}
            
            session_id = results['ids'][0]
            # Ensure session_id is a string, not a list
            if isinstance(session_id, list):
                session_id = session_id[0]
            session_data = self.collection.get(ids=[session_id])
            
            if not session_data or not session_data.get('documents'):
                return {"error": "Session data not found"}
            
            import json
            session = json.loads(session_data['documents'][0])
            
            return {
                "session_id": session.get('session_id'),
                "uid": session.get('uid'),
                "status": session.get('status'),
                "timestamp": session.get('timestamp'),
                "chat_messages": len(session.get("chat_history", [])),
                "agents_completed": [key for key in session.keys() if key not in ['session_id', 'uid', 'timestamp', 'status', 'chat_history']],
                "has_resume_data": "resume_parser" in session,
                "has_gap_analysis": "gap_analyzer" in session,
                "has_assessments": "assessment_recommender" in session,
                "has_questions": "assessment_question_generator" in session
            }
            
        except Exception as e:
            log.error(f"🚨 Error getting session summary for UID {uid}: {e}")
            return {"error": str(e)}

    def determine_resume_point(self, uid: str) -> Dict[str, Any]:
        """
        Determines where to resume the workflow based on completed agents.
        Returns the next step and any necessary data to resume from that point.
        """
        try:
            summary = self.get_session_summary(uid)
            
            if "error" in summary:
                return {
                    "resume_point": "validate_resume",
                    "reason": "No existing session",
                    "data": {}
                }
            
            # Check what's already completed and determine next step
            if summary.get("has_questions"):
                return {
                    "resume_point": "end",
                    "reason": "All steps already completed",
                    "data": {"session_summary": summary}
                }
            elif summary.get("has_assessments"):
                return {
                    "resume_point": "assessment_question_generator",
                    "reason": "Resume from question generation",
                    "data": {"session_summary": summary}
                }
            elif summary.get("has_gap_analysis"):
                return {
                    "resume_point": "assessment_recommender",
                    "reason": "Resume from assessment recommendation",
                    "data": {"session_summary": summary}
                }
            elif summary.get("has_resume_data"):
                return {
                    "resume_point": "gap_analyzer",
                    "reason": "Resume from gap analysis",
                    "data": {"session_summary": summary}
                }
            else:
                return {
                    "resume_point": "validate_resume",
                    "reason": "No completed steps found",
                    "data": {}
                }
                
        except Exception as e:
            log.error(f"🚨 Error determining resume point for UID {uid}: {e}")
            return {
                "resume_point": "validate_resume",
                "reason": f"Error: {str(e)}",
                "data": {}
            }

    def get_session_data_for_resume(self, uid: str, resume_point: str) -> Dict[str, Any]:
        """
        Gets the necessary session data to resume from a specific point.
        """
        try:
            results = self.collection.query(
                query_texts=["session data"],
                n_results=1,
                where={"uid": uid}
            )
            
            if not results or not results.get('ids') or len(results['ids']) == 0:
                return {}
            
            session_id = results['ids'][0]
            # Ensure session_id is a string, not a list
            if isinstance(session_id, list):
                session_id = session_id[0]
            session_data = self.collection.get(ids=[session_id])
            
            if not session_data or not session_data.get('documents'):
                return {}
            
            import json
            session = json.loads(session_data['documents'][0])
            
            # Extract relevant data based on resume point
            resume_data = {}
            
            if resume_point in ["gap_analyzer", "assessment_recommender", "assessment_question_generator"]:
                if "resume_parser" in session:
                    resume_data["structured_resume"] = session["resume_parser"].get("structured_resume", {})
                    resume_data["resume_text"] = session["resume_parser"].get("resume_text", "")
            
            if resume_point in ["assessment_recommender", "assessment_question_generator"]:
                if "gap_analyzer" in session:
                    resume_data["raw_skill_gap_analysis_output"] = session["gap_analyzer"].get("raw_skill_gap_analysis_output", {})
            
            if resume_point == "assessment_question_generator":
                if "assessment_recommender" in session:
                    resume_data["assessment_plan"] = session["assessment_recommender"].get("assessment_plan", [])
                    resume_data["assessment_needs"] = session["assessment_recommender"].get("assessment_needs", {})
            
            # Always include chat history
            resume_data["chat_history"] = session.get("chat_history", [])
            
            return resume_data
            
        except Exception as e:
            log.error(f"🚨 Error getting session data for resume for UID {uid}: {e}")
            return {}

# Create a single instance to be used across the application
memory_manager = MemoryManager()

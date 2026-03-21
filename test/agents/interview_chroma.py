"""
ChromaDB Integration for Interview Agent
Provides vector storage and retrieval for interview questions, responses, and evaluations
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

from chroma import client, embedding_fn, normalize_metadata

log = logging.getLogger(__name__)

class InterviewChromaManager:
    """Manages ChromaDB collections for interview-related data"""
    
    def __init__(self):
        self.client = client
        self.embedding_fn = embedding_fn
        
        # CRITICAL FIX: Lazy initialization to prevent blocking on import
        # Collections are created on first access, not at initialization time
        self._questions_collection = None
        self._responses_collection = None
        self._evaluations_collection = None
        self._sessions_collection = None
        self._question_templates_collection = None
        self._scenario_questions_collection = None
        self._collection_locks = {}  # Per-collection locks for thread safety
        
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection with timeout guard"""
        import threading
        if name not in self._collection_locks:
            self._collection_locks[name] = threading.Lock()
        
        # Use lock to prevent concurrent creation attempts
        with self._collection_locks[name]:
            try:
                # CRITICAL FIX: Direct call with error handling
                # Timeout is handled at the async layer (asyncio.to_thread with timeout)
                # This method is called from async code via asyncio.to_thread, so we keep it simple
                return self.client.get_or_create_collection(
                    name=name,
                    embedding_function=self.embedding_fn
                )
            except Exception as e:
                log.error(f"Error creating collection {name}: {e}")
                raise
    
    @property
    def questions_collection(self):
        """Lazy-loaded questions collection"""
        if self._questions_collection is None:
            self._questions_collection = self._get_or_create_collection("interview_questions")
        return self._questions_collection
    
    @property
    def responses_collection(self):
        """Lazy-loaded responses collection"""
        if self._responses_collection is None:
            self._responses_collection = self._get_or_create_collection("interview_responses")
        return self._responses_collection
    
    @property
    def evaluations_collection(self):
        """Lazy-loaded evaluations collection"""
        if self._evaluations_collection is None:
            self._evaluations_collection = self._get_or_create_collection("interview_evaluations")
        return self._evaluations_collection
    
    @property
    def sessions_collection(self):
        """Lazy-loaded sessions collection"""
        if self._sessions_collection is None:
            self._sessions_collection = self._get_or_create_collection("interview_sessions")
        return self._sessions_collection
    
    @property
    def question_templates_collection(self):
        """Lazy-loaded question templates collection"""
        if self._question_templates_collection is None:
            self._question_templates_collection = self._get_or_create_collection("question_templates")
        return self._question_templates_collection
    
    @property
    def scenario_questions_collection(self):
        """Lazy-loaded scenario questions collection"""
        if self._scenario_questions_collection is None:
            self._scenario_questions_collection = self._get_or_create_collection("interview_scenarios")
        return self._scenario_questions_collection
    
    # ==================== QUESTION MANAGEMENT ====================
    
    def store_question_template(self, question_id: str, question_text: str, 
                              question_type: str, difficulty: str, 
                              skills: List[str], metadata: Dict[str, Any] = None):
        """Store a reusable question template"""
        try:
            template_data = {
                "question_text": question_text,
                "question_type": question_type,
                "difficulty": difficulty,
                "skills": skills,
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                **(metadata or {})
            }
            
            # Prepare and normalize metadata
            metadata = {
                "question_type": question_type,
                "difficulty": difficulty,
                "skills": ",".join(skills) if isinstance(skills, list) else str(skills),
                "created_at": template_data["created_at"]
            }
            metadata = normalize_metadata(metadata)
            
            self.question_templates_collection.upsert(
                ids=[question_id],
                documents=[json.dumps(template_data)],
                metadatas=[metadata]
            )
            log.info(f"Stored question template: {question_id}")
            
        except Exception as e:
            log.error(f"Error storing question template {question_id}: {e}")
            raise
    
    def get_similar_questions(self, query_text: str, question_type: str = None, 
                            difficulty: str = None, skills: List[str] = None, 
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar questions based on semantic similarity"""
        try:
            where_clause = {}
            if question_type:
                where_clause["question_type"] = question_type
            if difficulty:
                where_clause["difficulty"] = difficulty
            
            results = self.question_templates_collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where_clause if where_clause else None
            )
            
            similar_questions = []
            for i, (doc, meta, doc_id, distance) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["ids"][0],
                results["distances"][0]
            )):
                try:
                    question_data = json.loads(doc)
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    # Filter by skills if specified
                    if skills:
                        question_skills = question_data.get("skills", [])
                        skill_overlap = len(set(skills) & set(question_skills))
                        if skill_overlap == 0:
                            continue
                    
                    similar_questions.append({
                        "question_id": doc_id,
                        "question_text": question_data["question_text"],
                        "question_type": question_data["question_type"],
                        "difficulty": question_data["difficulty"],
                        "skills": question_data["skills"],
                        "similarity_score": similarity_score,
                        "metadata": meta
                    })
                except json.JSONDecodeError:
                    log.warning(f"Failed to parse question data for {doc_id}")
                    continue
            
            return similar_questions
            
        except Exception as e:
            log.error(f"Error retrieving similar questions: {e}")
            return []
    
    def store_interview_question(self, session_id: str, question_text: str, 
                               question_type: str, context: Dict[str, Any] = None,
                               use_consolidated: bool = True):
        """
        Store a question asked during an interview
        
        Args:
            session_id: Session ID
            question_text: Question text
            question_type: Type of question
            context: Optional context
            use_consolidated: If True, embed in interview_sessions instead of separate collection
        """
        try:
            question_id = f"{session_id}_{int(time.time() * 1000)}"
            
            question_data = {
                "question_id": question_id,
                "session_id": session_id,
                "question_text": question_text,
                "question_type": question_type,
                "timestamp": datetime.now().isoformat(),
                "context": context or {}
            }
            
            if use_consolidated:
                # Embed in interview_sessions
                session_data = self.get_interview_session(session_id) or {}
                questions = session_data.get("questions", [])
                questions.append(question_data)
                session_data["questions"] = questions
                session_data["question_count"] = len(questions)
                session_data["consolidated"] = True
                self.store_interview_session(session_id, session_data)
                log.debug(f"Stored question in consolidated session: {session_id}")
            else:
                # Store in separate collection (legacy)
                metadata = {
                    "session_id": session_id,
                    "question_type": question_type,
                    "timestamp": question_data["timestamp"]
                }
                metadata = normalize_metadata(metadata)
                
                self.questions_collection.add(
                    ids=[question_id],
                    documents=[json.dumps(question_data)],
                    metadatas=[metadata]
                )
            
            return question_id
            
        except Exception as e:
            log.error(f"Error storing interview question: {e}")
            return None

    def store_scenario_question(
        self, 
        scenario_id: str,
        role_category: str,
        stage: str,
        difficulty: str,
        question_text: str,
        expected_depth: List[str],
        follow_up_triggers: List[str],
        metadata: Dict[str, Any] = None
    ):
        """Store a scenario-based interview question"""
        try:
            scenario_data = {
                "role_category": role_category,
                "stage": stage,
                "difficulty": difficulty,
                "question_text": question_text,
                "expected_depth": expected_depth,
                "follow_up_triggers": follow_up_triggers,
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                **(metadata or {})
            }
            
            # Prepare and normalize metadata
            metadata = {
                "role_category": role_category,
                "stage": stage,
                "difficulty": difficulty,
                "expected_depth": ",".join(expected_depth) if isinstance(expected_depth, list) else str(expected_depth),
                "follow_up_triggers": ",".join(follow_up_triggers) if isinstance(follow_up_triggers, list) else str(follow_up_triggers),
                "created_at": scenario_data["created_at"],
                "data": json.dumps(scenario_data)
            }
            metadata = normalize_metadata(metadata)
            
            self.scenario_questions_collection.upsert(
                ids=[scenario_id],
                documents=[question_text],
                metadatas=[metadata]
            )
            log.info(f"Stored scenario question: {scenario_id}")
            
        except Exception as e:
            log.error(f"Error storing scenario question {scenario_id}: {e}")
            raise

    def get_scenario_question(
        self,
        role_category: str,
        stage: str,
        difficulty: str = "intermediate",
        previous_scenario_ids: List[str] = None,
        top_k: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Retrieve scenario question from ChromaDB based on role, stage, and difficulty"""
        try:
            where_clause = {
                "role_category": role_category,
                "stage": stage,
                "difficulty": difficulty
            }
            
            results = self.scenario_questions_collection.query(
                query_texts=[f"{role_category} {stage} scenario"],
                n_results=top_k,
                where=where_clause
            )
            
            if not results["ids"][0]:
                return None
            
            # Filter out previously used scenarios
            available_scenarios = []
            for i, (doc_id, metadata) in enumerate(zip(results["ids"][0], results["metadatas"][0])):
                if not previous_scenario_ids or doc_id not in previous_scenario_ids:
                    scenario_data = json.loads(metadata.get("data", "{}"))
                    scenario_data["id"] = doc_id
                    available_scenarios.append(scenario_data)
            
            if available_scenarios:
                import random
                return random.choice(available_scenarios)
            
            return None
            
        except Exception as e:
            log.error(f"Error retrieving scenario question: {e}")
            return None

    def get_question_count(self, session_id: str) -> int:
        """
        Count stored questions for a session (server-side source of truth).
        Supports both consolidated (embedded) and legacy (separate collection) formats
        """
        try:
            # Try consolidated format first
            session_data = self.get_interview_session(session_id)
            if session_data:
                # Check if consolidated with embedded questions
                if session_data.get("consolidated") and session_data.get("questions"):
                    return len(session_data["questions"])
                # Check question_count from metadata
                if isinstance(session_data.get("question_count"), int):
                    return session_data["question_count"]

            # Fallback to legacy collection query
            results = self.questions_collection.query(
                query_texts=["interview questions"],
                n_results=300,  # bounded to avoid quota
                where={"session_id": session_id}
            )
            ids = results.get("ids") or []
            if ids and isinstance(ids[0], list):
                count = len(ids[0])
            else:
                count = len(ids)
            # If we hit the cap, treat as lower bound; return what we have
            return count
        except Exception as e:
            log.error(f"Error counting questions for session {session_id}: {e}")
            return 0

    def get_last_question(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent question for a session (id, text, timestamp).
        Supports both consolidated (embedded) and legacy (separate collection) formats
        """
        try:
            # Try consolidated format first
            session_data = self.get_interview_session(session_id)
            if session_data and session_data.get("consolidated") and session_data.get("questions"):
                questions = session_data["questions"]
                if questions:
                    # Sort by timestamp and return most recent
                    sorted_questions = sorted(questions, key=lambda x: x.get("timestamp", ""))
                    last_q = sorted_questions[-1]
                    return {
                        "question_id": last_q.get("question_id", ""),
                        "question_text": last_q.get("question_text", ""),
                        "timestamp": last_q.get("timestamp", "")
                    }
            
            # Fallback to legacy collection
            results = self.questions_collection.query(
                query_texts=["interview questions"],
                n_results=300,  # bounded to avoid quota
                where={"session_id": session_id}
            )
            if not results or not results.get("documents") or not results["documents"][0]:
                return None
            items = []
            for doc, meta, qid in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["ids"][0]
            ):
                try:
                    q = json.loads(doc)
                except json.JSONDecodeError:
                    continue
                ts = q.get("timestamp") or meta.get("timestamp")
                items.append({"question_id": qid, "question_text": q.get("question_text", ""), "timestamp": ts})
            if not items:
                return None
            items.sort(key=lambda x: x.get("timestamp", ""))
            return items[-1]
        except Exception as e:
            log.error(f"Error retrieving last question for session {session_id}: {e}")
            return None

    def is_similar_question(self, session_id: str, question_text: str, threshold: float = 0.92) -> Tuple[bool, float, Optional[str]]:
        """Check if question_text is too similar to any prior asked question in this session.

        Returns (is_similar, similarity_score, matched_question_text)
        """
        try:
            results = self.questions_collection.query(
                query_texts=[question_text],
                n_results=5,
                where={"session_id": session_id}
            )
            if not results or not results.get("documents") or not results["documents"][0]:
                return (False, 0.0, None)
            # distances ~ smaller is closer; similarity ~ 1 - distance
            top_idx = 0
            doc = results["documents"][0][top_idx]
            dist = results["distances"][0][top_idx]
            sim = max(0.0, 1.0 - dist)
            if sim >= threshold:
                try:
                    qd = json.loads(doc)
                    return (True, sim, qd.get("question_text", ""))
                except json.JSONDecodeError:
                    return (True, sim, None)
            return (False, sim, None)
        except Exception as e:
            log.error(f"Error similarity-checking question for session {session_id}: {e}")
            return (False, 0.0, None)
    
    # ==================== RESPONSE MANAGEMENT ====================
    
    def store_interview_response(self, session_id: str, question_id: str, 
                               response_text: str, response_analysis: Dict[str, Any],
                               candidate_info: Dict[str, Any] = None,
                               use_consolidated: bool = True):
        """
        Store a candidate's response with analysis
        
        Args:
            session_id: Session ID
            question_id: Question ID
            response_text: Response text
            response_analysis: Response analysis
            candidate_info: Optional candidate info
            use_consolidated: If True, embed in interview_sessions instead of separate collection
        """
        try:
            response_id = f"{session_id}_{question_id}_{int(time.time() * 1000)}"
            
            response_data = {
                "response_id": response_id,
                "session_id": session_id,
                "question_id": question_id,
                "response_text": response_text,
                "response_analysis": response_analysis,
                "candidate_info": candidate_info or {},
                "timestamp": datetime.now().isoformat()
            }
            
            if use_consolidated:
                # Embed in interview_sessions
                session_data = self.get_interview_session(session_id) or {}
                responses = session_data.get("responses", [])
                responses.append(response_data)
                session_data["responses"] = responses
                session_data["consolidated"] = True
                self.store_interview_session(session_id, session_data)
                log.debug(f"Stored response in consolidated session: {session_id}")
            else:
                # Store in separate collection (legacy)
                metadata = {
                    "session_id": session_id,
                    "question_id": question_id,
                    "response_type": response_analysis.get("type", "general"),
                    "engagement_score": response_analysis.get("engagement_score", 0.0),
                    "timestamp": response_data["timestamp"]
                }
                metadata = normalize_metadata(metadata)
                
                self.responses_collection.add(
                    ids=[response_id],
                    documents=[json.dumps(response_data)],
                    metadatas=[metadata]
                )
            
            return response_id
            
        except Exception as e:
            log.error(f"Error storing interview response: {e}")
            return None
    
    def get_session_responses(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all responses for a session
        Supports both consolidated (embedded) and legacy (separate collection) formats
        """
        try:
            # Try consolidated format first
            session_data = self.get_interview_session(session_id)
            if session_data and session_data.get("consolidated") and session_data.get("responses"):
                return sorted(session_data["responses"], key=lambda x: x.get("timestamp", ""))
            
            # Fallback to legacy collection
            results = self.responses_collection.query(
                query_texts=["interview responses"],
                n_results=100,
                where={"session_id": session_id}
            )
            
            responses = []
            for doc, meta, doc_id in zip(
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("ids", [[]])[0]
            ):
                try:
                    response_data = json.loads(doc)
                    responses.append({
                        "response_id": doc_id,
                        **response_data,
                        "metadata": meta
                    })
                except json.JSONDecodeError:
                    continue
            
            return sorted(responses, key=lambda x: x.get("timestamp", ""))
            
        except Exception as e:
            log.error(f"Error retrieving session responses: {e}")
            return []
    
    # ==================== EVALUATION MANAGEMENT ====================
    
    def store_interview_evaluation(self, session_id: str, evaluation_data: Dict[str, Any],
                                 candidate_info: Dict[str, Any] = None,
                                 use_consolidated: bool = True):
        """
        Store comprehensive interview evaluation
        
        Args:
            session_id: Session ID
            evaluation_data: Evaluation data
            candidate_info: Optional candidate info
            use_consolidated: If True, embed in interview_sessions instead of separate collection
        """
        try:
            evaluation_id = f"{session_id}_eval_{int(time.time() * 1000)}"
            
            evaluation_data["evaluation_id"] = evaluation_id
            evaluation_data["session_id"] = session_id
            evaluation_data["candidate_info"] = candidate_info or {}
            if "timestamp" not in evaluation_data:
                evaluation_data["timestamp"] = datetime.now().isoformat()
            
            if use_consolidated:
                # Embed in interview_sessions
                session_data = self.get_interview_session(session_id) or {}
                session_data["evaluation"] = evaluation_data
                session_data["consolidated"] = True
                self.store_interview_session(session_id, session_data)
                log.debug(f"Stored evaluation in consolidated session: {session_id}")
            else:
                # Store in separate collection (legacy)
                metadata = {
                    "session_id": session_id,
                    "overall_score": evaluation_data.get("overall_score", 0.0),
                    "status": evaluation_data.get("status", "unknown"),
                    "timestamp": evaluation_data["timestamp"]
                }
                metadata = normalize_metadata(metadata)
                
                self.evaluations_collection.add(
                    ids=[evaluation_id],
                    documents=[json.dumps(evaluation_data)],
                    metadatas=[metadata]
                )
            
            return evaluation_id
            
        except Exception as e:
            log.error(f"Error storing interview evaluation: {e}")
            return None
    
    def get_similar_evaluations(self, candidate_skills: List[str], 
                              job_requirements: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar interview evaluations for benchmarking"""
        try:
            query_text = f"Skills: {', '.join(candidate_skills)}. Requirements: {', '.join(job_requirements)}"
            
            results = self.evaluations_collection.query(
                query_texts=[query_text],
                n_results=top_k
            )
            
            similar_evaluations = []
            for doc, meta, doc_id, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["ids"][0],
                results["distances"][0]
            ):
                try:
                    eval_data = json.loads(doc)
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    similar_evaluations.append({
                        "evaluation_id": doc_id,
                        "overall_score": eval_data.get("overall_score", 0.0),
                        "status": eval_data.get("status", "unknown"),
                        "similarity_score": similarity_score,
                        "metadata": meta
                    })
                except json.JSONDecodeError:
                    continue
            
            return similar_evaluations
            
        except Exception as e:
            log.error(f"Error retrieving similar evaluations: {e}")
            return []
    
    # ==================== SESSION MANAGEMENT ====================
    
    def store_interview_session(self, session_id: str, session_data: Dict[str, Any]):
        """
        Store complete interview session data
        
        CRITICAL: This method merges with existing session data to preserve conversation_history
        and other important fields that might not be in the incoming session_data.
        
        Supports consolidated structure with embedded questions, responses, and evaluations:
        {
            "session_id": "...",
            "status": "ongoing",
            "questions": [...],      # Embedded from interview_questions
            "responses": [...],      # Embedded from interview_responses
            "evaluation": {...},     # Embedded from interview_evaluations
            "conversation_history": [...],  # CRITICAL: Preserved from existing data
            ...
        }
        """
        try:
            # CRITICAL FIX: Merge with existing session data to preserve conversation_history
            existing_data = self.get_interview_session(session_id) or {}
            
            # Preserve conversation_history from existing data if not in new data
            if "conversation_history" not in session_data and "conversation_history" in existing_data:
                session_data["conversation_history"] = existing_data["conversation_history"]
                log.debug(f"[CHROMA] Preserved conversation_history ({len(existing_data['conversation_history'])} messages) from existing session")
            
            # Merge existing data with new data (new data takes precedence for fields it provides)
            merged_data = {**existing_data, **session_data}
            
            merged_data["session_id"] = session_id
            if "timestamp" not in merged_data:
                merged_data["timestamp"] = datetime.now().isoformat()
            merged_data["updated_at"] = datetime.now().isoformat()
            
            # Update question_count if questions are embedded
            if "questions" in merged_data and isinstance(merged_data["questions"], list):
                merged_data["question_count"] = len(merged_data["questions"])
            
            # Prepare and normalize metadata
            metadata = {
                "session_id": session_id,
                "status": merged_data.get("status", "ongoing"),
                "question_count": merged_data.get("question_count", 0),
                "timestamp": merged_data["timestamp"],
                "updated_at": merged_data["updated_at"],
                "has_questions": bool(merged_data.get("questions")),
                "has_responses": bool(merged_data.get("responses")),
                "has_evaluation": bool(merged_data.get("evaluation")),
                "has_conversation_history": bool(merged_data.get("conversation_history")),
                "consolidated": merged_data.get("consolidated", False)
            }
            metadata = normalize_metadata(metadata)
            
            self.sessions_collection.upsert(
                ids=[session_id],
                documents=[json.dumps(merged_data)],
                metadatas=[metadata]
            )
            
            log.info(f"Stored interview session: {session_id} (consolidated: {merged_data.get('consolidated', False)}, history: {len(merged_data.get('conversation_history', []))} msgs)")
            
        except Exception as e:
            log.error(f"Error storing interview session: {e}")
            raise
    
    def get_interview_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve interview session data"""
        try:
            results = self.sessions_collection.get(ids=[session_id])
            
            if results["ids"]:
                session_data = json.loads(results["documents"][0])
                return session_data
            
            return None
            
        except Exception as e:
            log.error(f"Error retrieving session {session_id}: {e}")
            return None
    
    def get_similar_sessions(self, candidate_profile: Dict[str, Any], 
                           job_details: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
        """Find similar interview sessions for context"""
        try:
            query_text = f"Candidate: {candidate_profile.get('roles', '')} with skills {candidate_profile.get('skills', '')}. Job: {job_details.get('title', '')}"
            
            results = self.sessions_collection.query(
                query_texts=[query_text],
                n_results=top_k
            )
            
            similar_sessions = []
            for doc, meta, doc_id, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["ids"][0],
                results["distances"][0]
            ):
                try:
                    session_data = json.loads(doc)
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    similar_sessions.append({
                        "session_id": doc_id,
                        "similarity_score": similarity_score,
                        "session_data": session_data,
                        "metadata": meta
                    })
                except json.JSONDecodeError:
                    continue
            
            return similar_sessions
            
        except Exception as e:
            log.error(f"Error retrieving similar sessions: {e}")
            return []
    
    # ==================== PERSISTENT STATE MANAGEMENT ====================
    
    def store_session_state(self, session_id: str, state_data: Dict[str, Any]):
        """Store complete session state including fingerprints, intents, fallbacks, circuit breaker, and conversation context"""
        try:
            state_data["session_id"] = session_id
            state_data["timestamp"] = datetime.now().isoformat()
            state_data["updated_at"] = datetime.now().isoformat()
            
            # Prepare and normalize metadata
            metadata = {
                "session_id": session_id,
                "status": state_data.get("status", "active"),
                "timestamp": state_data["timestamp"],
                "updated_at": state_data["updated_at"]
            }
            metadata = normalize_metadata(metadata)
            
            # Use a dedicated collection for session state
            state_collection = self._get_or_create_collection("interview_session_state")
            state_collection.upsert(
                ids=[session_id],
                documents=[json.dumps(state_data)],
                metadatas=[metadata]
            )
            
            log.info(f"Stored session state for: {session_id}")
            
        except Exception as e:
            log.error(f"Error storing session state for {session_id}: {e}")
            raise
    
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete session state"""
        try:
            state_collection = self._get_or_create_collection("interview_session_state")
            results = state_collection.get(ids=[session_id])
            
            if results["ids"]:
                state_data = json.loads(results["documents"][0])
                return state_data
            
            return None
            
        except Exception as e:
            log.error(f"Error retrieving session state for {session_id}: {e}")
            return None
    
    def update_session_state_partial(self, session_id: str, partial_state: Dict[str, Any]):
        """Update only specific parts of session state (e.g., just fingerprints or intents)"""
        try:
            # Get existing state
            existing_state = self.get_session_state(session_id) or {}
            
            # Merge with partial update
            updated_state = {**existing_state, **partial_state}
            updated_state["updated_at"] = datetime.now().isoformat()
            
            # Store updated state
            self.store_session_state(session_id, updated_state)
            
        except Exception as e:
            log.error(f"Error updating partial session state for {session_id}: {e}")
            raise
    
    def store_question_fingerprints(self, session_id: str, fingerprints: List[List[str]]):
        """Store question fingerprints for a session"""
        try:
            self.update_session_state_partial(session_id, {
                "question_fingerprints": fingerprints,
                "fingerprints_updated_at": datetime.now().isoformat()
            })
        except Exception as e:
            log.error(f"Error storing question fingerprints for {session_id}: {e}")
    
    def get_question_fingerprints(self, session_id: str) -> List[List[str]]:
        """Retrieve question fingerprints for a session"""
        try:
            state = self.get_session_state(session_id)
            if state:
                return state.get("question_fingerprints", [])
            return []
        except Exception as e:
            log.error(f"Error retrieving question fingerprints for {session_id}: {e}")
            return []
    
    def store_negative_intent_history(self, session_id: str, intent_history: List[Dict[str, Any]]):
        """Store negative intent history for a session"""
        try:
            self.update_session_state_partial(session_id, {
                "negative_intent_history": intent_history,
                "intent_history_updated_at": datetime.now().isoformat()
            })
        except Exception as e:
            log.error(f"Error storing negative intent history for {session_id}: {e}")
    
    def get_negative_intent_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve negative intent history for a session"""
        try:
            state = self.get_session_state(session_id)
            if state:
                return state.get("negative_intent_history", [])
            return []
        except Exception as e:
            log.error(f"Error retrieving negative intent history for {session_id}: {e}")
            return []
    
    def clear_negative_intent_history(self, session_id: str) -> None:
        """Clear negative intent history for a session"""
        try:
            self.update_session_state_partial(session_id, {
                "negative_intent_history": [],
                "negative_intent_cleared_at": datetime.now().isoformat()
            })
            log.info(f"Cleared negative intent history for session: {session_id}")
        except Exception as e:
            log.error(f"Error clearing negative intent history for {session_id}: {e}")
            raise
    
    def store_recent_fallbacks(self, session_id: str, fallback_questions: List[str]):
        """Store recent fallback questions for a session"""
        try:
            self.update_session_state_partial(session_id, {
                "recent_fallbacks": fallback_questions,
                "fallbacks_updated_at": datetime.now().isoformat()
            })
        except Exception as e:
            log.error(f"Error storing recent fallbacks for {session_id}: {e}")
    
    def get_recent_fallbacks(self, session_id: str) -> List[str]:
        """Retrieve recent fallback questions for a session"""
        try:
            state = self.get_session_state(session_id)
            if state:
                return state.get("recent_fallbacks", [])
            return []
        except Exception as e:
            log.error(f"Error retrieving recent fallbacks for {session_id}: {e}")
            return []
    
    def store_circuit_breaker_state(self, session_id: str, cb_state: Dict[str, Any]):
        """Store circuit breaker state for a session"""
        try:
            self.update_session_state_partial(session_id, {
                "circuit_breaker": cb_state,
                "cb_updated_at": datetime.now().isoformat()
            })
        except Exception as e:
            log.error(f"Error storing circuit breaker state for {session_id}: {e}")
    
    def get_circuit_breaker_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve circuit breaker state for a session"""
        try:
            state = self.get_session_state(session_id)
            if state:
                return state.get("circuit_breaker")
            return None
        except Exception as e:
            log.error(f"Error retrieving circuit breaker state for {session_id}: {e}")
            return None
    
    def store_conversation_context(self, session_id: str, context_data: Dict[str, Any]):
        """Store conversation context state"""
        try:
            self.update_session_state_partial(session_id, {
                "conversation_context": context_data,
                "context_updated_at": datetime.now().isoformat()
            })
        except Exception as e:
            log.error(f"Error storing conversation context for {session_id}: {e}")
    
    def get_conversation_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation context state"""
        try:
            state = self.get_session_state(session_id)
            if state:
                return state.get("conversation_context")
            return None
        except Exception as e:
            log.error(f"Error retrieving conversation context for {session_id}: {e}")
            return None
    
    def store_conversation_history(self, session_id: str, history: List[Dict[str, str]]):
        """Store conversation history for a session"""
        try:
            # Get existing session data or create new
            existing_data = self.get_interview_session(session_id)
            log.debug(f"[CHROMA] Existing session_data for {session_id}: {type(existing_data)}, keys: {list(existing_data.keys()) if isinstance(existing_data, dict) else 'N/A'}")
            
            session_data = existing_data or {}
            
            # Update conversation_history in session data
            session_data["conversation_history"] = history
            session_data["session_id"] = session_id
            
            log.debug(f"[CHROMA] Storing session_data with conversation_history: {len(history)} messages, keys: {list(session_data.keys())}")
            
            # Store updated session
            self.store_interview_session(session_id, session_data)
            log.info(f"[CHROMA] Stored conversation_history: {len(history)} messages for session_id={session_id}")
            
            # Verify it was stored correctly
            verify_data = self.get_interview_session(session_id)
            if verify_data and isinstance(verify_data, dict):
                verify_history = verify_data.get("conversation_history")
                if isinstance(verify_history, list):
                    log.info(f"[CHROMA] Verified stored conversation_history: {len(verify_history)} messages")
                else:
                    log.error(f"[CHROMA] Verification failed: conversation_history is {type(verify_history)}")
        except Exception as e:
            log.error(f"Error storing conversation_history for {session_id}: {e}", exc_info=True)
            raise
    
    def get_conversation_history(self, session_id: str) -> Optional[List[Dict[str, str]]]:
        """Retrieve conversation history for a session"""
        try:
            session_data = self.get_interview_session(session_id)
            if session_data and isinstance(session_data, dict):
                log.debug(f"[CHROMA] Retrieved session_data for {session_id}, keys: {list(session_data.keys())}")
                history = session_data.get("conversation_history")
                log.debug(f"[CHROMA] conversation_history type: {type(history)}, value: {history}")
                if isinstance(history, list):
                    log.info(f"[CHROMA] Retrieved conversation_history: {len(history)} messages for session_id={session_id}")
                    return history
                else:
                    log.debug(f"[CHROMA] conversation_history is not a list (expected for new sessions): {type(history)}")
            else:
                log.debug(f"[CHROMA] Session data is None or not a dict for {session_id} (expected for new sessions): {type(session_data)}")
            return None
        except Exception as e:
            log.error(f"Error retrieving conversation_history for {session_id}: {e}", exc_info=True)
            return None
    
    def recover_session_state(self, session_id: str) -> Dict[str, Any]:
        """Recover all session state from ChromaDB and return as a dictionary for in-memory restoration"""
        try:
            state = self.get_session_state(session_id)
            if not state:
                return {}
            
            recovered = {
                "question_fingerprints": state.get("question_fingerprints", []),
                "negative_intent_history": state.get("negative_intent_history", []),
                "recent_fallbacks": state.get("recent_fallbacks", []),
                "circuit_breaker": state.get("circuit_breaker"),
                "conversation_context": state.get("conversation_context")
            }
            
            log.info(f"Recovered session state for: {session_id}")
            return recovered
            
        except Exception as e:
            log.error(f"Error recovering session state for {session_id}: {e}")
            return {}
    
    # ==================== ANALYTICS & INSIGHTS ====================
    
    def get_interview_analytics(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Get interview analytics for the specified time period"""
        try:
            cutoff_date = datetime.now().timestamp() - (time_period_days * 24 * 60 * 60)
            
            # Get all evaluations from the time period
            all_evaluations = self.evaluations_collection.get()
            
            analytics = {
                "total_interviews": 0,
                "average_score": 0.0,
                "hire_rate": 0.0,
                "common_skills": {},
                "question_effectiveness": {},
                "response_patterns": {}
            }
            
            scores = []
            hire_count = 0
            skill_counts = {}
            question_types = {}
            
            for doc, meta in zip(all_evaluations["documents"], all_evaluations["metadatas"]):
                try:
                    eval_data = json.loads(doc)
                    timestamp = datetime.fromisoformat(eval_data["timestamp"]).timestamp()
                    
                    if timestamp < cutoff_date:
                        continue
                    
                    analytics["total_interviews"] += 1
                    score = eval_data.get("overall_score", 0.0)
                    scores.append(score)
                    
                    if eval_data.get("status") in ["strong_hire", "hire"]:
                        hire_count += 1
                    
                    # Analyze skills
                    candidate_skills = eval_data.get("candidate_info", {}).get("skills", [])
                    for skill in candidate_skills:
                        skill_counts[skill] = skill_counts.get(skill, 0) + 1
                    
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            
            if scores:
                analytics["average_score"] = sum(scores) / len(scores)
            
            if analytics["total_interviews"] > 0:
                analytics["hire_rate"] = hire_count / analytics["total_interviews"]
            
            # Sort skills by frequency
            analytics["common_skills"] = dict(sorted(skill_counts.items(), 
                                                   key=lambda x: x[1], reverse=True)[:10])
            
            return analytics
            
        except Exception as e:
            log.error(f"Error generating analytics: {e}")
            return {}
    
    def get_question_effectiveness(self, question_type: str = None) -> Dict[str, Any]:
        """Analyze question effectiveness based on response quality"""
        try:
            where_clause = {"question_type": question_type} if question_type else None
            
            results = self.questions_collection.query(
                query_texts=["interview questions"],
                n_results=1000,
                where=where_clause
            )
            
            effectiveness = {
                "total_questions": 0,
                "average_engagement": 0.0,
                "high_engagement_questions": [],
                "low_engagement_questions": []
            }
            
            engagement_scores = []
            
            for doc, meta, doc_id in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["ids"][0]
            ):
                try:
                    question_data = json.loads(doc)
                    effectiveness["total_questions"] += 1
                    
                    # Get responses for this question
                    response_results = self.responses_collection.query(
                        query_texts=["responses"],
                        n_results=10,
                        where={"question_id": doc_id}
                    )
                    
                    question_engagements = []
                    for resp_doc in response_results["documents"][0]:
                        try:
                            resp_data = json.loads(resp_doc)
                            engagement = resp_data.get("response_analysis", {}).get("engagement_score", 0.0)
                            question_engagements.append(engagement)
                        except json.JSONDecodeError:
                            continue
                    
                    if question_engagements:
                        avg_engagement = sum(question_engagements) / len(question_engagements)
                        engagement_scores.append(avg_engagement)
                        
                        question_info = {
                            "question_id": doc_id,
                            "question_text": question_data["question_text"],
                            "average_engagement": avg_engagement,
                            "response_count": len(question_engagements)
                        }
                        
                        if avg_engagement >= 0.7:
                            effectiveness["high_engagement_questions"].append(question_info)
                        elif avg_engagement <= 0.3:
                            effectiveness["low_engagement_questions"].append(question_info)
                
                except json.JSONDecodeError:
                    continue
            
            if engagement_scores:
                effectiveness["average_engagement"] = sum(engagement_scores) / len(engagement_scores)
            
            return effectiveness
            
        except Exception as e:
            log.error(f"Error analyzing question effectiveness: {e}")
            return {}


# Global instance
interview_chroma = InterviewChromaManager()

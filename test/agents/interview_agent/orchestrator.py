# orchestrator.py
"""
Simplified, topic-focused orchestrator for LLM-first interview agent.

Design goals (Mode A - minimal & clean):
- Single responsibility: orchestrate flow (preprocess -> analyze -> generate -> persist -> return)
- Single analysis call per turn
- Topic-focused stages only
- Minimal, well-defined fallbacks
- No role classification, no portfolio digestion, no facts memory
- Keep anonymization, anti-repetition, and guardrails where needed
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from .state_manager import InterviewState, InterviewRequest, InterviewResponse
from .conversation_context import ConversationContext
from .response_analysis import IntelligentResponseAnalyzer
from .question_generator import _normalize_spacing, _normalize_question_grammar, _strip_leading_greeting, _greeting_name_from_candidate_info
from .aiinterview_question_gen import generate_question_nostream
from .anonymizer_module import PIIAnonymizer
from .session_manager import (
    save_conversation_context,
    save_session,
    load_conversation_context,
    save_conversation_history,
    load_conversation_history,
)
from .fallback_generator import get_default_question, get_natural_fallback_question
from .anti_repetition import check_question_similarity, track_question_fingerprint
from .summary_generator import generate_interview_summary
from .evaluator import InterviewEvaluator
from .history_utils import count_questions_in_history
from .intent_classifier import detect_intents
from utils.interview_utils import anonymize_conversation_history as anonymize_history_util
from utils.question_drift import is_drift
from utils.llm_telemetry import log_invoke_llm
from utils.safe_format import safe_format
from core.utils import schedule_background_task

log = logging.getLogger(__name__)


# -------------------------
# Helpers
# -------------------------
# Removed _ensure_metadata - metadata is passed via session dict, not on InterviewRequest


def _append_user_answer_to_history(interview_req: InterviewRequest, anonymized_answer: str) -> None:
    """
    Append the anonymized answer to both the anonymized_history (external) and
    to the interview_req.conversation_history (persisted canonical history).
    """
    interview_req.conversation_history = interview_req.conversation_history or []
    interview_req.conversation_history.append({"role": "user", "content": anonymized_answer})


def _compute_question_count(interview_req: InterviewRequest, server_question_count: Optional[int], anonymized_history: List[Dict]) -> int:
    if server_question_count:
        return server_question_count
    if interview_req.question_count:
        return interview_req.question_count
    return count_questions_in_history(anonymized_history)


def _determine_stage_from_count(question_count: int, analysis: Dict[str, Any]) -> str:
    """
    Deterministic baseline with simple adaptive override:
      - 0 -> topic_introduction
      - 1-2 -> topic_fundamentals
      - 3-5 -> topic_deep_dive
      - 6+ -> topic_feedback

    Adaptive: early deep dive if technical & deep; hold on fundamentals if brief or low engagement.
    """
    if question_count == 0:
        stage = "topic_introduction"
    elif question_count <= 2:
        stage = "topic_fundamentals"
    elif question_count <= 5:
        stage = "topic_deep_dive"
    else:
        stage = "topic_feedback"

    # Adaptive overrides
    if analysis:
        answer_type = analysis.get("type", "")
        engagement = float(analysis.get("engagement_score", 0.5) or 0.5)
        keywords = analysis.get("keywords", []) or []
        word_count = int(analysis.get("word_count", 0) or 0)

        if answer_type == "technical" and len(keywords) >= 4:
            stage = "topic_deep_dive"
        if answer_type == "brief" or word_count < 10:
            if question_count > 1:
                stage = "topic_fundamentals"
        if engagement < 0.3:
            stage = "topic_fundamentals"

    return stage


async def _analyze_response(
    interview_req: InterviewRequest,
    anonymized_history: List[Dict[str, Any]],
    candidate_info: Dict[str, Any],
    safe_job_details: Dict[str, Any],
    uid_context: Dict[str, Any],
    question_count: int
) -> Dict[str, Any]:
    """
    Analyze candidate response and return analysis dict.
    
    This is extracted from run_turn to be reusable for streaming flow.
    Includes exit intent detection to handle conversation-ending signals.
    """
    # CRITICAL: Check for exit intent FIRST before doing expensive analysis
    # This handles cases like "thank you" which should end the interview immediately
    if interview_req.answer and interview_req.answer.strip():
        try:
            from .intent_classifier import detect_intents_with_llm_fallback
            user_answer = interview_req.answer.strip()
            # Use hybrid approach: pattern-based first, LLM fallback for ambiguous cases
            intents = await detect_intents_with_llm_fallback(
                user_answer,
                interview_req.interview_topic or "",
                conversation_history=anonymized_history[-6:] if anonymized_history else None,
                use_llm_for_ambiguous=True
            )
            
            # Special handling for standalone "thank you" or "thanks" - high confidence exit
            user_answer_lower = user_answer.lower().strip()
            is_standalone_thanks = (
                user_answer_lower in ["thank you", "thanks", "thankyou"] or
                (len(user_answer_lower.split()) <= 3 and 
                 ("thank you" in user_answer_lower or "thanks" in user_answer_lower) and
                 not any(word in user_answer_lower for word in ["for", "about", "that", "this", "the", "a", "an"]))
            )
            
            if is_standalone_thanks:
                log.info(f"[EXIT_INTENT] Standalone 'thank you' detected - treating as polite exit signal")
                intents["exit_interview"] = True
                intents["confidence"] = 0.9  # High confidence for standalone thanks
            
            if intents.get("exit_interview"):
                log.info(f"[EXIT_INTENT] Exit intent detected in _analyze_response: '{user_answer[:100]}...'")
                
                # Determine exit reason
                exit_reason = None
                if "other task" in user_answer_lower or "other things" in user_answer_lower:
                    exit_reason = "other_commitments"
                elif "have to go" in user_answer_lower or "need to go" in user_answer_lower:
                    exit_reason = "time_constraint"
                elif "thank you" in user_answer_lower or "thanks" in user_answer_lower:
                    exit_reason = "polite_exit"
                
                # Generate appropriate exit message
                if exit_reason == "other_commitments":
                    exit_message = "Thank you for your time. I understand you have other commitments. We'll wrap up the interview now."
                elif exit_reason == "time_constraint":
                    exit_message = "Thank you for your time. We'll wrap up the interview now."
                else:
                    exit_message = "Thank you for your time. We'll wrap up the interview now."
                
                # Return exit intent analysis
                return {
                    "type": "exit_interview",
                    "engagement_score": 0.0,
                    "should_end_interview": True,
                    "exit_intent_detected": True,
                    "exit_message": exit_message,
                    "exit_reason": exit_reason,
                    "confidence": intents.get("confidence", 0.9)
                }
        except Exception as e:
            log.debug(f"Exit intent detection failed in _analyze_response: {e}")
            # Continue with normal analysis if intent detection fails
    
    response_analysis = {
        "type": "general",
        "confidence": 0.5,
        "keywords": [],
        "skills": [],
        "sentiment": "neutral",
        "engagement_score": 0.5,
        "word_count": 0,
        "context_understanding": ""
    }
    
    if interview_req.answer and interview_req.answer.strip():
        try:
            analysis = await IntelligentResponseAnalyzer.analyze_response_intelligently(
                text=interview_req.answer,
                history=anonymized_history,
                memory={"uid_context": uid_context},
                metadata={
                    "candidate_info": candidate_info,
                    "job_details": safe_job_details,
                    "is_first_request": question_count == 0,
                    "question_count": question_count
                }
            )
            # Convert to plain dict safely
            if hasattr(analysis, "dict"):
                analysis_dict = analysis.dict()
            else:
                analysis_dict = {
                    "type": getattr(analysis, "type", "general"),
                    "confidence": getattr(analysis, "confidence", 0.5),
                    "keywords": getattr(analysis, "keywords", []) or [],
                    "skills": getattr(analysis, "skills", []) or [],
                    "sentiment": getattr(analysis, "sentiment", "neutral"),
                    "engagement_score": getattr(analysis, "engagement_score", 0.5),
                    "word_count": getattr(analysis, "word_count", len(interview_req.answer.split())),
                    "evidence": getattr(analysis, "evidence", "")
                }
            # Map to legacy-lite fields used by generator
            response_analysis.update({
                "type": analysis_dict.get("type", "general"),
                "confidence": analysis_dict.get("confidence", 0.5),
                "keywords": analysis_dict.get("keywords", []) or [],
                "skills": analysis_dict.get("skills", []) or [],
                "sentiment": analysis_dict.get("sentiment", "neutral"),
                "engagement_score": float(analysis_dict.get("engagement_score", 0.5) or 0.5),
                "word_count": int(analysis_dict.get("word_count", len(interview_req.answer.split())) or 0),
                "context_understanding": analysis_dict.get("evidence", "") or ""
            })
        except Exception:
            log.exception("Response analysis failed; using defaults")
    
    # Stop-loss: Ensure analysis has minimum quality
    DEFAULT_SAFE_ANALYSIS = {
        "type": "general",
        "confidence": 0.5,
        "keywords": [],
        "skills": [],
        "sentiment": "neutral",
        "engagement_score": 0.5,
        "word_count": 0,
        "context_understanding": ""
    }
    
    if not response_analysis:
        response_analysis = DEFAULT_SAFE_ANALYSIS.copy()
    else:
        confidence = float(response_analysis.get("confidence", 0) or 0)
        engagement = float(response_analysis.get("engagement_score", 0) or 0)
        if confidence < 0.1 or engagement < 0.05:
            log.warning(f"Low-quality analysis detected (confidence={confidence}, engagement={engagement}); using safe defaults")
            response_analysis = DEFAULT_SAFE_ANALYSIS.copy()
    
    return response_analysis


# -------------------------
# Core orchestrator flow (minimal)
# -------------------------
async def run_turn(
    interview_req: InterviewRequest,
    uid_context: Optional[Dict[str, Any]] = None,
    candidate_info: Optional[Dict[str, Any]] = None,
    safe_job_details: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    server_question_count: Optional[int] = None,
    anonymizer: Optional[PIIAnonymizer] = None,
    persist: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Process a single turn:
      - recover/load context
      - anonymize & append candidate answer (if present)
      - run a single response analysis
      - select stage
      - call generate_question -> enforce guardrails internally
      - persist context & history
      - return (next_question, response_analysis)
    """
    start_ts = time.time()

    # Defensive defaults
    anonymizer = anonymizer or PIIAnonymizer()
    candidate_info = candidate_info or {}
    safe_job_details = safe_job_details or {}
    uid_context = uid_context or {}

    # Recover or load conversation context
    conversation_context = None
    if session_id:
        try:
            conv_ctx = await load_conversation_context(session_id)
            if conv_ctx:
                conversation_context = ConversationContext.from_dict(conv_ctx)
        except Exception:
            log.debug("No existing conversation context loaded")
    if conversation_context is None:
        conversation_context = ConversationContext()

    # Metadata is passed via session dict, not on InterviewRequest

    # Log incoming history and hydrate from persistence if needed
    incoming_history_len = len(interview_req.conversation_history or [])
    log.info(f"[RUN] incoming history len={incoming_history_len}")
    if session_id:
        try:
            loaded_history = await load_conversation_history(session_id)
            if loaded_history:
                log.info(f"[LOAD] history length={len(loaded_history)}")
                if len(interview_req.conversation_history or []) < len(loaded_history):
                    interview_req.conversation_history = loaded_history
        except Exception as load_exc:
            log.debug(f"[LOAD] Unable to hydrate conversation history: {load_exc}")

    # Step 1: anonymize existing history, then anonymize + append answer
    # We keep anonymized_history local to pass to LLM and analysis, but persist canonical interview_req.conversation_history.
    # LATENCY FIX: For question generation, we only need last 3 messages, but full history is still needed for evaluation
    try:
        # anonymize current canonical history using utility function
        # LATENCY FIX: Only anonymize last 3 messages for question gen hot path (audit fix #2)
        raw_history = interview_req.conversation_history or []
        messages_to_anonymize = raw_history[-3:] if len(raw_history) > 3 else raw_history
        anonymized_history = anonymize_history_util(
            messages_to_anonymize, 
            anonymizer, 
            candidate_info.get("names", []), 
            candidate_info.get("companies", [])
        )
    except Exception:
        # Fallback: manually anonymize each message
        # LATENCY FIX: Only anonymize last 3 messages for question gen (audit fix #2)
        # IMPROVEMENT: Batch anonymization for parallel processing
        raw_history = interview_req.conversation_history or []
        # Only anonymize last 3 messages for question generation hot path
        messages_to_anonymize = raw_history[-3:] if len(raw_history) > 3 else raw_history
        names = candidate_info.get("names", [])
        companies = candidate_info.get("companies", [])

        # Extract contents and roles
        contents = [msg.get("content", "") for msg in messages_to_anonymize]
        roles = [msg.get("role", "user") for msg in messages_to_anonymize]

        # Batch anonymize all contents in parallel, with safe fallback
        anonymized_history = []
        try:
            anonymized_contents = await anonymizer.anonymize_batch_async(contents, names, companies)
            for idx, (role, original_content, anonymized_content) in enumerate(
                zip(roles, contents, anonymized_contents)
            ):
                if isinstance(anonymized_content, Exception):
                    log.warning(
                        f"Anonymization failed for history message {idx}: {anonymized_content}"
                    )
                    anonymized_content = original_content
                anonymized_history.append({"role": role, "content": anonymized_content})
        except Exception as e:
            log.warning(
                f"Batch anonymization failed for history; falling back to sequential: {e}"
            )
            anonymized_history = []
            for msg in messages_to_anonymize:
                content = msg.get("content", "")
                role = msg.get("role", "user")
                try:
                    anonymized_content = await anonymizer.anonymize_text_async(
                        content, names, companies
                    )
                except Exception:
                    anonymized_content = content
                anonymized_history.append({"role": role, "content": anonymized_content})

    # FIX 1: ALWAYS append user answers to conversation_history (even if anonymization fails)
    # If there is an answer, anonymize it and append
    if interview_req.answer and interview_req.answer.strip():
        try:
            # FIX: Use async anonymization for large answers to prevent event loop blocking
            answer_text = interview_req.answer.strip()
            if len(answer_text) > 1000:
                anonymized_answer = await anonymizer.anonymize_text_async(
                    answer_text, 
                    candidate_info.get("names", []), 
                    candidate_info.get("companies", [])
                )
            else:
                anonymized_answer = anonymizer.anonymize_text(
                    answer_text, 
                    candidate_info.get("names", []), 
                    candidate_info.get("companies", [])
                )
        except Exception:
            anonymized_answer = interview_req.answer.strip()
        
        # Ensure anonymized_answer is not empty (fallback to raw answer if needed)
        if not anonymized_answer or not anonymized_answer.strip():
            anonymized_answer = interview_req.answer.strip()
        
        # append to anonymized_history for analysis/generation
        anonymized_history = (anonymized_history + [{"role": "user", "content": anonymized_answer}])[-3:]
        
        # CRITICAL: ALWAYS append to canonical history to persist (even if anonymization failed)
        # This ensures conversation_history always contains user answers for evaluation
        answer_to_store = anonymized_answer or interview_req.answer.strip()
        interview_req.conversation_history = interview_req.conversation_history or []
        interview_req.conversation_history.append({
            "role": "user",
            "content": answer_to_store
        })
        
        # Persist conversation_history after appending user answer (non-blocking)
        if persist and session_id:
            # Issue 6.1: Use schedule_background_task for proper exception logging
            schedule_background_task(
                save_conversation_history(session_id, interview_req.conversation_history),
                f"save_conversation_history:{session_id[:8]}"
            )
            log.info(f"[PERSIST] conversation_history save scheduled ({len(interview_req.conversation_history)} messages)")
            
            # NOTE: input_metadata is ephemeral session telemetry, not persisted to ChromaDB.
            # It remains available in the request context for the current interview session
            # and can be included in final reports, but is not stored as semantic memory.
    else:
        # ensure anonymized_history exists for first-turn prompts
        anonymized_history = anonymized_history or []

    # Step 2: compute question count (single source of truth)
    question_count = _compute_question_count(interview_req, server_question_count, anonymized_history)
    
    # FIX 3: Force question_count to increment on real answers
    # This guarantees that UI does not show "0 questions answered" when answers were actually provided
    if interview_req.answer and interview_req.answer.strip():
        question_count = max(question_count, (interview_req.question_count or 0)) + 1
        interview_req.question_count = question_count
    else:
        interview_req.question_count = question_count

    # Step 2b: Unified intent detection - handles all edge cases in one place
    # Early intent responses are stored in next_question and will get greeting at the end
    early_intent_response = None
    early_intent_analysis = None
    # Initialize response_analysis to ensure it's always defined
    response_analysis: Dict[str, Any] = {
        "type": "general",
        "confidence": 0.5,
        "keywords": [],
        "skills": [],
        "sentiment": "neutral",
        "engagement_score": 0.5,
        "word_count": 0,
        "context_understanding": ""
    }
    
    if interview_req.answer and interview_req.answer.strip():
        try:
            # Use hybrid intent detection with LLM fallback for robust exit detection
            from .intent_classifier import detect_intents_with_llm_fallback
            intents = await detect_intents_with_llm_fallback(
                interview_req.answer, 
                interview_req.interview_topic or "",
                conversation_history=interview_req.conversation_history[-6:] if interview_req.conversation_history else None,
                use_llm_for_ambiguous=True
            )
            
            # IMPROVEMENT: Also check recent conversation history for exit signals
            # This handles cases where user says "thank you" after asking to wrap up
            if not intents.get("exit_interview") and interview_req.conversation_history:
                # Check last 3 user messages for exit intent
                recent_user_messages = [
                    msg.get("content", "") 
                    for msg in interview_req.conversation_history[-6:]  # Last 6 messages (3 turns)
                    if msg.get("role") == "user"
                ]
                # FIX: Wrap intent detection for recent messages in asyncio.to_thread()
                for recent_msg in recent_user_messages[-2:]:  # Check last 2 user messages
                    if recent_msg:
                        recent_intents = await asyncio.to_thread(
                            detect_intents, 
                            recent_msg, 
                            interview_req.interview_topic or ""
                        )
                        if recent_intents.get("exit_interview"):
                            log.info(f"[EXIT_INTENT] Exit intent found in recent conversation history: '{recent_msg[:50]}...'")
                            intents["exit_interview"] = True
                            intents["confidence"] = 0.85  # Slightly lower confidence for historical detection
                            break
            
            # Exit interview - CRITICAL: Set flag to signal caller to conclude interview
            if intents.get("exit_interview"):
                log.info(f"[EXIT_INTENT] Exit intent detected from user message: '{interview_req.answer[:100]}...' (session: {session_id})")
                
                # IMPROVEMENT: Check if this is a repeated exit request (user said wrap up multiple times)
                exit_request_count = 0
                if interview_req.conversation_history:
                    # Issue 6.3: Limit concurrent intent detection to prevent thread pool exhaustion
                    MAX_CONCURRENT_INTENT_CHECKS = 3
                    INTENT_DETECTION_TIMEOUT = 5.0  # seconds per check
                    
                    user_messages = [
                        msg.get("content", "") 
                        for msg in interview_req.conversation_history[-6:]  # Reduced from 10 to 6
                        if msg.get("role") == "user"
                    ][-MAX_CONCURRENT_INTENT_CHECKS:]  # Only check last 3 user messages
                    
                    # Run intent detection in parallel with timeout protection
                    intent_tasks = [
                        asyncio.wait_for(
                            asyncio.to_thread(detect_intents, msg_content, interview_req.interview_topic or ""),
                            timeout=INTENT_DETECTION_TIMEOUT
                        )
                        for msg_content in user_messages if msg_content
                    ]
                    if intent_tasks:
                        msg_intents_list = await asyncio.gather(*intent_tasks, return_exceptions=True)
                        for idx, msg_intents in enumerate(msg_intents_list):
                            if isinstance(msg_intents, asyncio.TimeoutError):
                                log.warning(f"Intent detection timed out for message {idx}")
                                continue
                            if isinstance(msg_intents, Exception):
                                log.warning(f"Intent detection failed for message {idx}: {msg_intents}")
                                continue
                            if isinstance(msg_intents, dict) and msg_intents.get("exit_interview"):
                                exit_request_count += 1
                
                # IMPROVEMENT: Extract reason from user message for personalized response
                user_answer_lower = interview_req.answer.lower()
                exit_reason = None
                if "other task" in user_answer_lower or "other things" in user_answer_lower:
                    exit_reason = "other_commitments"
                elif "have to go" in user_answer_lower or "need to go" in user_answer_lower:
                    exit_reason = "time_constraint"
                elif "thank you" in user_answer_lower:
                    exit_reason = "polite_exit"
                
                # IMPROVEMENT: Personalized exit message based on context
                if exit_request_count > 1:
                    # User has asked to wrap up multiple times - be more direct and immediate
                    early_intent_response = "Understood. Thank you for your time. We'll conclude the interview now."
                elif exit_reason == "other_commitments":
                    early_intent_response = "Thank you for your time. I understand you have other commitments. We'll wrap up the interview now."
                elif exit_reason == "time_constraint":
                    early_intent_response = "Thank you for your time. We'll wrap up the interview now."
                else:
                    # Standard polite exit message
                    early_intent_response = "Thank you for your time. We'll wrap up the interview now."
                
                # Set response message and flag to indicate interview should end
                early_intent_analysis = {
                    "type": "exit_interview", 
                    "engagement_score": 0.0,
                    "should_end_interview": True,  # CRITICAL: Flag for caller to conclude interview
                    "exit_intent_detected": True,   # Additional flag for clarity
                    "exit_message": early_intent_response,  # Store message for history
                    "exit_request_count": exit_request_count,  # Track repeated requests
                    "exit_reason": exit_reason  # Store reason for analytics
                }
                
                # CRITICAL: Add exit message to conversation history immediately
                # This ensures the wrap-up message is preserved even if conclude_interview fails
                interview_req.conversation_history = interview_req.conversation_history or []
                interview_req.conversation_history.append({
                    "role": "assistant",
                    "content": early_intent_response
                })
                
                # Persist the exit message immediately (non-blocking)
                if persist and session_id:
                    # Issue 6.1: Use schedule_background_task for proper exception logging
                    schedule_background_task(
                        save_conversation_history(session_id, interview_req.conversation_history),
                        f"save_exit_history:{session_id[:8]}"
                    )
                    log.info(f"[EXIT_INTENT] Added exit message to conversation_history and scheduled persistence")
            
            # Topic switch - TEMPORARILY DISABLED
            # Original topic switch logic commented out for temporary disable:
            # elif intents.get("topic_switch") and intents.get("new_topic"):
            #     new_topic = intents["new_topic"].strip()
            #     if new_topic:
            #         old_topic = interview_req.interview_topic
            #         interview_req.interview_topic = new_topic
            #         
            #         # Reset stage to topic_introduction when topic switches
            #         conversation_context.interview_stage = "topic_introduction"
            #         conversation_context.interview_phase = "topic_introduction"
            #         
            #         # Add new topic to topics_discussed via update_context (centralized)
            #         # Note: This will be handled by update_context when response_analysis is processed
            #         log.info(
            #             f"Topic switch detected: '{old_topic}' -> '{new_topic}'. "
            #             f"Resetting stage to topic_introduction. Session: {session_id}"
            #         )
            #         
            #         # Persist the new topic immediately
            #         if persist and session_id:
            #             try:
            #                 await save_conversation_context(session_id, conversation_context.to_dict())
            #             except Exception:
            #                 log.debug("Failed to persist topic switch, continuing")
            #     # Continue normal flow after topic switch
            
            # Temporary replacement: block topic switch requests
            elif intents.get("topic_switch"):
                # Note: Greeting will be added at the end (Step 9) for question_count == 0
                early_intent_response = "Topic switch is not possible at the moment."
                early_intent_analysis = {"type": "topic_switch_blocked", "engagement_score": 0.5}
            
            # Refusal → ask simpler question
            elif intents.get("refusal"):
                early_intent_response = "No worries. Could you share anything you're comfortable with about this topic?"
                early_intent_analysis = {"type": "refusal", "engagement_score": 0.4}
            
            # Uncertainty → supportive follow-up
            elif intents.get("uncertainty"):
                early_intent_response = "That's okay. What part of this topic do you feel most comfortable discussing?"
                early_intent_analysis = {"type": "uncertainty", "engagement_score": 0.45}
            
            # Irrelevant → gently redirect (mode-aware) with adaptive question
            # Even when redirecting, try to build on what the candidate mentioned
            elif intents.get("irrelevant"):
                topic = interview_req.interview_topic or ""
                topic_lower = topic.lower()
                is_psychometric = any(kw in topic_lower for kw in PSYCHOMETRIC_KEYWORDS)
                is_personality = any(kw in topic_lower for kw in PERSONALITY_KEYWORDS)
                is_communication = any(kw in topic_lower for kw in COMMUNICATION_KEYWORDS)
                
                # Import adaptive context extraction and fallback builder
                from .question_generator import _build_topic_specific_fallback, _extract_last_answer_context
                stage = getattr(conversation_context, 'interview_stage', 'topic_introduction') or 'topic_introduction'
                
                # Extract what the candidate mentioned (even if answer was off-topic)
                # Use interview_req.conversation_history which contains the actual conversation
                history = interview_req.conversation_history or []
                answer_context = _extract_last_answer_context(history)
                key_claims = answer_context.get("key_claims", [])
                log.debug(f"Redirect path: extracted key_claims={key_claims} from answer")

                if is_psychometric or is_personality:
                    early_intent_response = (
                        "Can you describe a recent situation that shows how you "
                        "approached a difficult decision or high-pressure moment?"
                    )
                elif is_communication:
                    early_intent_response = (
                        "Can you tell me about a time you had to explain something complex "
                        "to someone unfamiliar with it, and how you adapted your message?"
                    )
                elif key_claims:
                    # Build adaptive redirect using what they mentioned
                    # Vary the template based on question count to avoid repetition detection
                    claim = key_claims[0]  # Use first detected claim
                    q_count = getattr(interview_req, 'question_count', 0) or 0
                    
                    # Rotate through different question templates
                    templates = [
                        f"You mentioned {claim} - can you walk me through a specific challenge you faced using it?",
                        f"Tell me more about how you've used {claim} in practice. What trade-offs did you encounter?",
                        f"What was a situation where {claim} was particularly useful or problematic for you?",
                        f"Can you give me a concrete example of applying {claim} to solve a real problem?",
                        f"How did you decide to use {claim} in that situation? What alternatives did you consider?",
                    ]
                    early_intent_response = templates[q_count % len(templates)]
                    log.debug(f"Adaptive redirect using candidate's claim: {claim}, template #{q_count % len(templates)}")
                else:
                    # Default: generate topic-specific redirect question
                    early_intent_response = _build_topic_specific_fallback(topic or "this topic", stage)

                early_intent_analysis = {"type": "redirect", "engagement_score": 0.5}
            
            # Meta requests
            elif intents.get("meta_request") == "repeat":
                early_intent_response = getattr(conversation_context, 'last_question', None) or "Let me repeat: Could you explain more about that?"
                early_intent_analysis = {"type": "meta_repeat", "engagement_score": 0.55}
            
            elif intents.get("meta_request") == "simplify":
                early_intent_response = "Sure — let's take it step by step. What's one simple example from your experience related to this topic?"
                early_intent_analysis = {"type": "meta_simplify", "engagement_score": 0.55}
            
            elif intents.get("meta_request") == "slower":
                early_intent_response = "Of course. I'll slow down. Can you tell me about your experience with this topic at your own pace?"
                early_intent_analysis = {"type": "meta_slower", "engagement_score": 0.55}
            
            elif intents.get("meta_request") == "faster":
                early_intent_response = "Got it. Let's move a bit quicker. What's your experience with this topic?"
                early_intent_analysis = {"type": "meta_faster", "engagement_score": 0.55}
                
        except Exception as e:
            # Non-fatal: continue normal path
            log.debug(f"Intent detection failed, continuing normal flow: {e}")

    # If early intent response was set, use it and skip normal question generation flow
    # Step 3: Single analysis call (if answer present). For first-turn w/o answer, keep defaults.
    # Skip normal flow if early intent response was already set
    if not early_intent_response:
        # Reset response_analysis to defaults (already initialized above)
        response_analysis = {
            "type": "general",
            "confidence": 0.5,
            "keywords": [],
            "skills": [],
            "sentiment": "neutral",
            "engagement_score": 0.5,
            "word_count": 0,
            "context_understanding": ""
        }

        if interview_req.answer and interview_req.answer.strip():
            try:
                # Note: analysis may benefit from full history, but for latency we use minimal
                # For now, use anonymized_history (last 3) for analysis too to keep it fast
                analysis = await IntelligentResponseAnalyzer.analyze_response_intelligently(
                    text=interview_req.answer,
                    history=anonymized_history,
                    memory={"uid_context": uid_context},
                    metadata={
                        "candidate_info": candidate_info,
                        "job_details": safe_job_details,
                        "is_first_request": question_count == 0,
                        "question_count": question_count
                    }
                )
                # Convert to plain dict safely
                if hasattr(analysis, "dict"):
                    analysis_dict = analysis.dict()
                else:
                    analysis_dict = {
                        "type": getattr(analysis, "type", "general"),
                        "confidence": getattr(analysis, "confidence", 0.5),
                        "keywords": getattr(analysis, "keywords", []) or [],
                        "skills": getattr(analysis, "skills", []) or [],
                        "sentiment": getattr(analysis, "sentiment", "neutral"),
                        "engagement_score": getattr(analysis, "engagement_score", 0.5),
                        "word_count": getattr(analysis, "word_count", len(interview_req.answer.split())),
                        "evidence": getattr(analysis, "evidence", "")
                    }
                # Map to legacy-lite fields used by generator
                response_analysis.update({
                    "type": analysis_dict.get("type", "general"),
                    "confidence": analysis_dict.get("confidence", 0.5),
                    "keywords": analysis_dict.get("keywords", []) or [],
                    "skills": analysis_dict.get("skills", []) or [],
                    "sentiment": analysis_dict.get("sentiment", "neutral"),
                    "engagement_score": float(analysis_dict.get("engagement_score", 0.5) or 0.5),
                    "word_count": int(analysis_dict.get("word_count", len(interview_req.answer.split())) or 0),
                    "context_understanding": analysis_dict.get("evidence", "") or ""
                })
            except Exception:
                log.exception("Response analysis failed; using defaults")
        
        # Stop-loss: Ensure analysis has minimum quality to prevent stage oscillation
        DEFAULT_SAFE_ANALYSIS = {
            "type": "general",
            "confidence": 0.5,
            "keywords": [],
            "skills": [],
            "sentiment": "neutral",
            "engagement_score": 0.5,
            "word_count": 0,
            "context_understanding": ""
        }
        
        if not response_analysis:
            response_analysis = DEFAULT_SAFE_ANALYSIS.copy()
        else:
            confidence = float(response_analysis.get("confidence", 0) or 0)
            engagement = float(response_analysis.get("engagement_score", 0) or 0)
            if confidence < 0.1 or engagement < 0.05:
                log.warning(f"Low-quality analysis detected (confidence={confidence}, engagement={engagement}); using safe defaults")
                response_analysis = DEFAULT_SAFE_ANALYSIS.copy()

        # Step 4: Update conversation context with analysis (engagement, coverage, keywords, etc.)
        # This centralizes all keyword/skill addition - no duplicate logic needed
        try:
            await conversation_context.update_context(response_analysis, question_count, persist=False)
        except Exception:
            log.debug("Failed to update conversation context, continuing")
        
        # Step 4b: Determine stage and update conversation_context
        stage = _determine_stage_from_count(question_count, response_analysis)
        conversation_context.interview_stage = stage
        conversation_context.interview_phase = stage

        # Step 5: System prompt rebuilding removed for low-latency QGen (audit fix #1)
        # Question generator now uses inline minimal prompts, so YAML loading and safe_format are unnecessary
        
        # Step 5b: DO NOT rewrite semantic content before generation. Use raw anonymized history for generation to avoid drift.
        # Rationale: rewriting before generation increases drift/hallucination risk and can change intent.
        # LATENCY FIX: Only pass last 3 messages to question generator (audit fix #3)
        rewritten_history = anonymized_history[-3:] if len(anonymized_history) > 3 else anonymized_history
        
        # Step 6: Generate question (aiinterview: default prompt or generic-test prompt for psychometric/personality/communication)
        next_question = ""
        fallback_used = False
        resume = getattr(interview_req, "resume", None) or getattr(interview_req, "structured_resume", None) or {}
        topic = (interview_req.interview_topic or "general").strip()
        # Use greeting name so LLM never sees CANDIDATE_A; empty/missing name becomes "there"
        cand_name = _greeting_name_from_candidate_info(candidate_info)
        try:
            qgen_result = await generate_question_nostream(
                conversation_history=rewritten_history,
                resume=resume,
                topic=topic,
                candidate_name=cand_name,
            )
            next_question = (qgen_result or {}).get("question", "") or ""
        except Exception:
            log.exception("Question generation failed; using fallback")
            fallback_used = True
            try:
                next_question = await get_natural_fallback_question(
                    conversation_context,
                    candidate_info,
                    session_id=session_id,
                    uid_context=uid_context,
                    interview_topic=interview_req.interview_topic
                )
            except Exception:
                next_question = await get_default_question(InterviewState.TOPIC_INTRODUCTION, interview_req.interview_topic)
        
        # CRITICAL FIX: Check if question is empty (even if no exception was raised)
        if not next_question or not next_question.strip():
            log.warning("Question generator returned empty question; using fallback")
            fallback_used = True
            try:
                next_question = await get_natural_fallback_question(
                    conversation_context,
                    candidate_info,
                    session_id=session_id,
                    uid_context=uid_context,
                    interview_topic=interview_req.interview_topic
                )
            except Exception:
                next_question = await get_default_question(InterviewState.TOPIC_INTRODUCTION, interview_req.interview_topic)
        
        # Check for question drift (off-topic or inappropriate questions)
        if next_question and interview_req.interview_topic:
            try:
                if is_drift(next_question, interview_req.interview_topic):
                    log.warning(f"Question drift detected; replacing with fallback. Topic: {interview_req.interview_topic}, Question: {next_question[:50]}...")
                    fallback_used = True
                    try:
                        next_question = await get_natural_fallback_question(
                            conversation_context,
                            candidate_info,
                            session_id=session_id,
                            uid_context=uid_context,
                            interview_topic=interview_req.interview_topic
                        )
                    except Exception:
                        next_question = await get_default_question(InterviewState.TOPIC_INTRODUCTION, interview_req.interview_topic)
            except Exception:
                log.debug("Question drift check failed; continuing with generated question")
        
        # First-turn hard override for generic tracks to avoid meta "what is a test?" questions
        if question_count == 0:
            topic_lower = (interview_req.interview_topic or "").strip().lower()
            if any(kw in topic_lower for kw in ["psychometric", "psychological", "behavioral", "aptitude", "personality"]):
                next_question = (
                    "Tell me about a time you had to make a tough decision with limited information. "
                    "What was your thought process and how did it turn out?"
                )
                log.info("[OVERRIDE] First-turn psychometric/personality question enforced.")
            elif any(kw in topic_lower for kw in ["communication", "communication test", "communication skills", "speaking", "presentation", "listening"]):
                next_question = (
                    "Describe a time you had to explain a complex idea to someone unfamiliar with it. "
                    "How did you adapt your message and what was the outcome?"
                )
                log.info("[OVERRIDE] First-turn communication question enforced.")
    else:
        # Early intent response was set - use it and skip question generation
        next_question = early_intent_response
        fallback_used = False
        # Use previous stage instead of forcing topic_introduction
        stage = conversation_context.interview_stage or "topic_introduction"
        conversation_context.interview_stage = stage
        conversation_context.interview_phase = stage
    
    # CRITICAL FIX: Ensure response_analysis is defined for early intents
    if early_intent_analysis:
        response_analysis = early_intent_analysis

    # Step 7: Anti-repetition tracking (best-effort) and persist fingerprint
    # CRITICAL FIX: Skip similarity check if we already used fallback to prevent infinite loops
    try:
        if session_id and next_question and not fallback_used:
            similar = await check_question_similarity(next_question, session_id)
            if similar:
                # replace with safe fallback and avoid storing fingerprint for duplicate
                # Mark as fallback to prevent re-checking
                fallback_used = True
                next_question = await get_natural_fallback_question(
                    conversation_context, 
                    candidate_info, 
                    session_id=session_id,
                    uid_context=uid_context,
                    interview_topic=interview_req.interview_topic
                )
                # DO NOT run check_question_similarity again on fallback to prevent infinite loop
            else:
                await track_question_fingerprint(next_question, session_id)
        elif session_id and next_question and fallback_used:
            # Still track fingerprint even for fallback (but don't check similarity again)
            await track_question_fingerprint(next_question, session_id)
    except Exception:
        log.debug("Anti-repetition check failed or skipped")
    
    # CRITICAL FIX 3: Reset stage to fundamentals ONLY if fallback was used AND conditions indicate need
    # Don't reset on every fallback - only if early stage, brief answer, or low engagement
    if fallback_used:
        should_reset = (
            question_count <= 4 or 
            response_analysis.get("type") == "brief" or 
            response_analysis.get("engagement_score", 0.5) < 0.3
        )
        if should_reset:
            stage = "topic_fundamentals"
            conversation_context.interview_stage = stage
            conversation_context.interview_phase = stage
            log.debug("Fallback used with reset conditions; resetting stage to topic_fundamentals")
        else:
            log.debug("Fallback used but keeping current stage (question_count=%d, type=%s, engagement=%.2f)", 
                        question_count, response_analysis.get("type"), response_analysis.get("engagement_score", 0.5))

    # Step 7b: Add greeting for first message only (BEFORE appending to history so history gets greeted question too)
    # Treat both "assistant" and "model" as assistant (clients may send role "model")
    def _is_assistant_msg(m):
        r = (m.get("role") or "").strip().lower()
        return r == "assistant" or r == "model"
    has_prior_assistant = any(_is_assistant_msg(msg) for msg in (interview_req.conversation_history or []))
    if (
        question_count == 0
        and not has_prior_assistant
        and next_question
        and next_question.strip()
        and not fallback_used
    ):
        # Always control the displayed name: strip any LLM greeting (e.g. "Hello CANDIDATE_A,") then prepend "Hello {name},"
        next_question = _strip_leading_greeting(next_question)
        name = _greeting_name_from_candidate_info(candidate_info)
        next_question = f"Hello {name}, " + next_question.lstrip()

    # Step 7c: On follow-up turns, strip any leading greeting so only the first question has it
    if (has_prior_assistant or question_count > 0) and next_question:
        next_question = _strip_leading_greeting(next_question)

    # Step 8: Update last_question for meta requests (e.g., "repeat")
    conversation_context.last_question = next_question

    # Append assistant question to conversation_history to ensure full Q/A pairs are preserved
    # CRITICAL: Skip if exit message was already added (to avoid duplicates)
    interview_req.conversation_history = interview_req.conversation_history or []
    if response_analysis and not response_analysis.get("exit_intent_detected"):
        # Only append if this is not an exit intent (exit message was already added earlier)
        interview_req.conversation_history.append({
            "role": "assistant",
            "content": next_question
        })
    elif response_analysis and response_analysis.get("exit_intent_detected"):
        # Exit intent: verify message is in history, but don't duplicate
        last_msg = interview_req.conversation_history[-1] if interview_req.conversation_history else None
        if not last_msg or last_msg.get("role") != "assistant" or next_question not in last_msg.get("content", ""):
            # Exit message missing, add it
            interview_req.conversation_history.append({
                "role": "assistant",
                "content": next_question
            })
            log.debug(f"[EXIT_INTENT] Added exit message to history (was missing)")
    else:
        # Normal case: append question
        interview_req.conversation_history.append({
            "role": "assistant",
            "content": next_question
        })

    # Persist conversation_history after appending assistant question (non-blocking)
    if persist and session_id:
        # Issue 6.1: Use schedule_background_task for proper exception logging
        schedule_background_task(
            save_conversation_history(session_id, interview_req.conversation_history),
            f"save_assistant_history:{session_id[:8]}"
        )
        log.info(f"[PERSIST] conversation_history save scheduled ({len(interview_req.conversation_history)} messages)")

    # Step 9: Persist conversation context and session (best-effort, non-blocking)
    # Batch persistence operations using asyncio.gather() for better performance
    try:
        if persist and session_id:
            tasks = [
                save_conversation_context(session_id, conversation_context.to_dict()),
                save_session(
                    {
                        "session_id": session_id,
                        "last_updated": time.time(),
                        "question_count": question_count,
                    }
                ),
            ]
            # Issue 6.1: Schedule both persistence calls together with proper exception logging
            async def _persist_background():
                await asyncio.gather(*tasks, return_exceptions=True)
            schedule_background_task(_persist_background(), f"persist_context_session:{session_id[:8]}")
    except Exception:
        log.exception("Failed to schedule persistence tasks")

    # Step 10: Force next_question to be a string before returning
    if not isinstance(next_question, str):
        next_question = str(next_question) if next_question else ""
    
    # Step 11b: Global spacing normalization (fixes "whereyou" -> "where you" and similar issues)
    if next_question:
        next_question = _normalize_spacing(next_question)
    
    # Ensure next_question is not empty (final safety check)
    if not next_question or not next_question.strip():
        log.warning("Final safety check: next_question is empty, using fallback")
        try:
            next_question = await get_natural_fallback_question(
                conversation_context,
                candidate_info,
                session_id=session_id,
                uid_context=uid_context,
                interview_topic=interview_req.interview_topic
            )
        except Exception:
            next_question = await get_default_question(InterviewState.TOPIC_INTRODUCTION, interview_req.interview_topic)
        # Ensure it's a string - use topic-specific fallback instead of generic
        if not next_question:
            topic = interview_req.interview_topic or "this area"
            next_question = f"Walk me through a specific project where you applied {topic} to solve a real problem."
        else:
            next_question = str(next_question)

    # Step 12: Return next question and analysis (caller will present question to candidate)
    latency_ms = int((time.time() - start_ts) * 1000)
    log.info(
        f"Interview Agent Response | Session: {session_id} | Question #{question_count} | Stage: {stage} | Latency: {latency_ms}ms"
    )
    log.info(f"Generated Question: {next_question[:200]}{'...' if len(next_question) > 200 else ''}")
    log.debug(f"Response Analysis: {response_analysis}")
    log.debug(f"Turn completed (qcount={question_count}, stage={stage}, latency_ms={latency_ms})")

    return str(next_question), response_analysis


# -------------------------
# Conclude interview (minimal)
# -------------------------
async def conclude_interview(
    interview_req: InterviewRequest,
    conversation_context: ConversationContext,
    session_id: Optional[str],
    candidate_info: Optional[Dict[str, Any]] = None,
    safe_job_details: Optional[Dict[str, Any]] = None,
    evaluator: Optional[InterviewEvaluator] = None,
) -> InterviewResponse:
    """
    Minimal, topic-focused conclusion:
    - Validate minimal inputs
    - Run full evaluation using InterviewEvaluator
    - Generate a detailed report via summary_generator
    - Fire callback via evaluator._trigger_callbacks (if configured)
    """
    candidate_info = candidate_info or {}
    safe_job_details = safe_job_details or {}
    evaluator = evaluator or InterviewEvaluator()

    # Safety net: If conversation_history is empty and session_id exists, try loading from persistence
    # CRITICAL: Always try to load if history is missing or too short (less than expected messages)
    final_history = interview_req.conversation_history or []
    if session_id:
        if not final_history or len(final_history) == 0:
            log.warning(f"[RECOVERY] conversation_history is empty, attempting to load from persistence for session_id={session_id}")
            loaded_history = await load_conversation_history(session_id)
            if loaded_history:
                final_history = loaded_history
                interview_req.conversation_history = loaded_history
                log.info(f"[RECOVERY] Loaded history for final eval: {len(loaded_history)} messages")
            else:
                log.error(f"[RECOVERY] Failed to load conversation_history from persistence for session_id={session_id}")
        else:
            log.info(f"[RECOVERY] conversation_history already present: {len(final_history)} messages")

    # Debug: Print conversation_history
    log.info("FINAL conversation_history DUMP:")
    for msg in final_history:
        content_preview = (msg.get("content") or "")[:80]
        log.info(f"{msg.get('role', 'unknown')}: {content_preview}")

    # Log final conversation_history length
    log.info(f"Final conversation_history length = {len(final_history)}")
    
    # Ensure interview_req has the final history
    interview_req.conversation_history = final_history

    # Build eval session/metadata - use final_history to ensure we have the loaded data
    eval_session = {
        "conversation_history": final_history,
        "metadata": {
            "candidate_name": candidate_info.get("name", ""),
            "job_title": safe_job_details.get("job_title", ""),
            "interview_topic": interview_req.interview_topic,
            "session_id": session_id
        },
        "session_id": session_id
    }
    
    eval_metadata = {
        "job_title": safe_job_details.get("job_title", ""),
        "domain": interview_req.interview_topic or "general",
        "candidate_info": candidate_info,
        "job_details": safe_job_details
    }
    
    # CRITICAL: Run evaluation BEFORE building report to ensure scores are available
    # This must execute before generate_interview_summary and build_final_report
    log.info(f"[EVAL] using history {len(final_history)} messages")
    
    # Extract actual answer count from conversation history for fallback
    def _count_answers_from_history(history: List[Dict[str, Any]]) -> int:
        """Count actual Q/A pairs in conversation history."""
        import re
        count = 0
        for msg in history or []:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if role in ("user", "candidate") and content and re.search(r"\w", content):
                count += 1
        return count
    
    actual_answer_count = _count_answers_from_history(final_history)
    
    try:
        evaluation_result = await evaluator.run_full_evaluation(
            eval_session, final_history, eval_metadata
        )
        # Ensure num_answers is set correctly even if evaluation partially failed
        if not evaluation_result.get("num_answers") and actual_answer_count > 0:
            evaluation_result["num_answers"] = actual_answer_count
            log.warning(
                "[EVAL] Corrected num_answers from 0 to %d based on conversation history",
                actual_answer_count
            )
    except Exception as e:
        log.exception("Full evaluation failed; using basic fallback")
        # Apply fallback scoring based on actual answer count
        fallback_scores = {
            "technical": 0.3 if actual_answer_count >= 2 else 0.2,
            "problem_solving": 0.3 if actual_answer_count >= 2 else 0.2,
            "communication": 0.4 if actual_answer_count >= 2 else 0.3,
            "experience_relevance": 0.3 if actual_answer_count >= 2 else 0.2,
            "cultural_fit": 0.3 if actual_answer_count >= 2 else 0.2,
        }
        fallback_overall = (
            0.40 * fallback_scores["technical"] +
            0.25 * fallback_scores["problem_solving"] +
            0.15 * fallback_scores["communication"] +
            0.10 * fallback_scores["experience_relevance"] +
            0.10 * fallback_scores["cultural_fit"]
        ) * 100.0
        
        evaluation_result = {
            "per_answer": [],
            "overall_score": fallback_overall,
            "num_answers": actual_answer_count,
            "category_scores": fallback_scores,
            "strengths": [],
            "weaknesses": [],
            "summary": f"Evaluation system encountered an error: {str(e)}. Applied fallback scoring based on {actual_answer_count} answer(s) provided.",
            "failed_evaluation": True,
            "error": f"evaluation_exception: {type(e).__name__}",
            "latency_seconds": 0.0,
        }
        log.warning(
            "[EVAL] Applied fallback evaluation: num_answers=%d, overall_score=%.2f",
            actual_answer_count,
            fallback_overall
        )

    # Log compact evaluation summary; minimal evaluator does not return per-answer details.
    failed_eval = evaluation_result.get("failed_evaluation", False)
    eval_error = evaluation_result.get("error", "")
    if failed_eval:
        log.error(
            "[EVAL] ⚠️ EVALUATION FAILED: num_answers=%s overall_score=%.2f error=%s",
            evaluation_result.get("num_answers", 0),
            float(evaluation_result.get("overall_score", 0.0) or 0.0),
            eval_error
        )
    else:
        log.info(
            "[EVAL] compact evaluation result: num_answers=%s overall_score=%.2f",
            evaluation_result.get("num_answers", 0),
            float(evaluation_result.get("overall_score", 0.0) or 0.0),
        )

    # Generate LLM-based narrative summary (summary_generator only produces text)
    # NOTE: This uses evaluation_result from above, so evaluation must complete first
    llm_summary_text = ""
    try:
        session_for_summary = {
            "conversation_history": final_history,
            "metadata": {
                "candidate_name": candidate_info.get("name", ""),
                "job_title": safe_job_details.get("job_title", ""),
                "interview_topic": interview_req.interview_topic
            }
        }
        conversation_ctx_dict = {
            "topics_discussed": conversation_context.topics_discussed,
            "engagement_level": conversation_context.engagement_level,
            "interview_phase": conversation_context.interview_phase
        }
        summary_result = await generate_interview_summary(
            session=session_for_summary,
            evaluation_summary=evaluation_result,
            conversation_context=conversation_ctx_dict,
            mode="interview_report"
        )
        # Extract full DetailedReport dict for candidate-facing fields
        llm_summary_dict = None
        llm_summary_text = ""
        
        if summary_result.get("ok") and summary_result.get("report"):
            # New format: DetailedReport with all candidate-facing fields
            llm_summary_dict = summary_result.get("report", {})
            llm_summary_text = llm_summary_dict.get("report", "")
        elif summary_result.get("ok") and summary_result.get("summary"):
            # Old format: just summary text
            llm_summary_text = summary_result.get("summary", {}).get("summary", "")
        else:
            llm_summary_text = f"Interview completed. Overall score: {evaluation_result.get('overall_score', 0.0)}."
    except Exception:
        log.exception("Summary generation failed; building basic summary")
        llm_summary_text = f"Interview completed. Overall score: {evaluation_result.get('overall_score', 0.0)}."
        llm_summary_dict = None
    
    # Generate structured interview report (UI-formatted)
    structured_report_result = None
    try:
        from .summary_generator import generate_structured_interview_report
        
        session_for_structured = {
            "conversation_history": final_history,
            "metadata": {
                "candidate_name": candidate_info.get("name", ""),
                "job_title": safe_job_details.get("job_title", ""),
                "interview_topic": interview_req.interview_topic
            }
        }
        
        structured_report_result = await generate_structured_interview_report(
            session=session_for_structured,
            evaluation_summary=evaluation_result,
            job_title=safe_job_details.get("job_title", ""),
            interview_topic=interview_req.interview_topic or ""
        )
        
        if structured_report_result.get("ok"):
            log.info("[ORCHESTRATOR] Generated structured interview report successfully")
        else:
            log.warning("[ORCHESTRATOR] Structured report generation failed, will use fallback")
    except Exception as e:
        log.exception("[ORCHESTRATOR] Failed to generate structured report: %s", e)
        structured_report_result = None
    
    # Build structured final report using report_builder
    try:
        from .report_builder import build_final_report
        
        metadata_for_report = {
            "candidate_info": candidate_info,
            "job_details": safe_job_details,
            "job_title": safe_job_details.get("job_title", ""),
            "interview_topic": interview_req.interview_topic,
            "debug_report_mode": False  # Can be enabled for debugging
        }
        
        # Extract structured report data if available
        structured_report_data = None
        if structured_report_result and structured_report_result.get("ok"):
            structured_report_data = structured_report_result.get("report", {})
        
        final_report = build_final_report(
            evaluation_result=evaluation_result,
            structured_report=structured_report_data,
            conversation_context=conversation_context.to_dict(),
            metadata=metadata_for_report
        )
        
        # Log the complete final report in a structured, readable format
        log.info("")
        log.info("=" * 100)
        log.info(" " * 30 + "FINAL INTERVIEW REPORT")
        log.info("=" * 100)
        log.info("")
        
        # Basic Information
        log.info("📋 BASIC INFORMATION")
        log.info("-" * 100)
        log.info(f"  Candidate Name: {final_report.get('candidate_name', 'N/A')}")
        log.info(f"  Job Title:      {final_report.get('job_title', 'N/A')}")
        log.info(f"  Position:       {final_report.get('position', 'N/A')}")
        log.info(f"  Interview Phase: {final_report.get('interview_phase', 'N/A')}")
        log.info("")
        
        # Overall Score (highlighted)
        overall_score = final_report.get('overall_score', 0.0)
        score_emoji = "🟢" if overall_score >= 80 else "🟡" if overall_score >= 60 else "🔴"
        log.info(f"📊 OVERALL SCORE: {score_emoji} {overall_score}/100")
        log.info("")
        
        # Session Metrics
        log.info("📈 SESSION METRICS")
        log.info("-" * 100)
        session_metrics = final_report.get("session_metrics", {})
        engagement = session_metrics.get('engagement', 0.0)
        ai_confidence = session_metrics.get('ai_confidence', 0.0)
        sentiment = session_metrics.get('sentiment', 'N/A')
        log.info(f"  Engagement Level:  {engagement:.2f} ({'High' if engagement >= 0.7 else 'Medium' if engagement >= 0.4 else 'Low'})")
        log.info(f"  AI Confidence:     {ai_confidence:.2f} ({'High' if ai_confidence >= 0.8 else 'Medium' if ai_confidence >= 0.6 else 'Low'})")
        log.info(f"  Sentiment:         {sentiment.upper()}")
        log.info("")
        
        # Detailed Breakdown
        log.info("📊 DETAILED BREAKDOWN")
        log.info("-" * 100)
        breakdown = final_report.get("detailed_breakdown", {})
        if breakdown:
            for category, score in breakdown.items():
                # Skip non-numeric entries (e.g., "_canonical" metadata)
                if not isinstance(score, (int, float)):
                    continue
                category_display = category.replace('_', ' ').title()
                bar_length = int(score / 2)  # Scale to 50 chars max
                bar = "█" * bar_length + "░" * (50 - bar_length)
                log.info(f"  {category_display:25s} {score:6.2f}/100  [{bar}]")
        else:
            log.info("  No breakdown data available")
        log.info("")
        
        # Strengths
        strengths = final_report.get("strengths", [])
        log.info("✅ STRENGTHS")
        log.info("-" * 100)
        if strengths:
            for i, strength in enumerate(strengths, 1):
                log.info(f"  {i}. {strength}")
        else:
            log.info("  No strengths listed")
        log.info("")
        
        # Concerns
        concerns = final_report.get("concerns", [])
        log.info("⚠️  CONCERNS / AREAS FOR IMPROVEMENT")
        log.info("-" * 100)
        if concerns:
            for i, concern in enumerate(concerns, 1):
                log.info(f"  {i}. {concern}")
        else:
            log.info("  No concerns listed")
        log.info("")
        
        # LLM Report Text
        llm_text = final_report.get("llm_report_text", "")
        log.info("📝 LLM-GENERATED NARRATIVE REPORT")
        log.info("-" * 100)
        if llm_text:
            # Log full text, but format it nicely
            lines = llm_text.split('\n')
            for line in lines:
                if line.strip():
                    log.info(f"  {line.strip()}")
                else:
                    log.info("")
            log.info("")
        else:
            log.info("  No narrative report available")
            log.info("")
        
        # Complete JSON structure for programmatic access
        log.info("🔧 COMPLETE REPORT (JSON)")
        log.info("-" * 100)
        try:
            # FIX: For large JSON objects, use asyncio.to_thread() to prevent blocking
            # Estimate size before serialization
            estimated_size = len(str(final_report))
            if estimated_size > 10000:
                report_json = await asyncio.to_thread(
                    json.dumps,
                    final_report,
                    indent=2,
                    ensure_ascii=False,
                )
            else:
                report_json = json.dumps(final_report, indent=2, ensure_ascii=False)
            # Log JSON in chunks if too long
            max_json_length = 5000
            if len(report_json) > max_json_length:
                log.info(report_json[:max_json_length])
                log.info(f"\n  ... [JSON truncated - total length: {len(report_json)} characters]")
                log.info("  [Use the structured fields above for full details]")
            else:
                log.info(report_json)
        except Exception as e:
            log.warning(f"  Could not serialize report to JSON: {e}")
        log.info("")
        log.info("=" * 100)
        log.info("")
    except Exception:
        log.exception("Report building failed; using evaluation result as fallback")
        # Fallback: use evaluation_result as report structure
        final_report = {
            "candidate_name": candidate_info.get("name", ""),
            "job_title": safe_job_details.get("job_title", ""),
            "position": safe_job_details.get("job_title", ""),
            "session_metrics": {
                "engagement": conversation_context.engagement_level,
                "ai_confidence": 0.5,
                "sentiment": "neutral"
            },
            "overall_score": evaluation_result.get("overall_score", 0.0),
            "detailed_breakdown": {},
            "strengths": evaluation_result.get("strengths", []),
            "concerns": evaluation_result.get("weaknesses", []),
            "interview_phase": conversation_context.interview_phase,
            "llm_report_text": llm_summary_text
        }
        # Log fallback report with same format
        log.warning("⚠️  Report building failed; using fallback structure")
        log.info("")
        log.info("=" * 100)
        log.info(" " * 30 + "FINAL INTERVIEW REPORT (FALLBACK)")
        log.info("=" * 100)
        log.info("")
        log.info("📋 BASIC INFORMATION")
        log.info("-" * 100)
        log.info(f"  Candidate Name: {final_report.get('candidate_name', 'N/A')}")
        log.info(f"  Job Title:      {final_report.get('job_title', 'N/A')}")
        log.info(f"  Interview Phase: {final_report.get('interview_phase', 'N/A')}")
        log.info("")
        overall_score = final_report.get('overall_score', 0.0)
        score_emoji = "🟢" if overall_score >= 80 else "🟡" if overall_score >= 60 else "🔴"
        log.info(f"📊 OVERALL SCORE: {score_emoji} {overall_score}/100")
        log.info("")
        log.info("⚠️  Note: This is a fallback report. Some details may be incomplete.")
        log.info("")
        log.info("🔧 COMPLETE REPORT (JSON)")
        log.info("-" * 100)
        try:
            # FIX: For large JSON objects, use asyncio.to_thread() to prevent blocking
            estimated_size = len(str(final_report))
            if estimated_size > 10000:
                report_json = await asyncio.to_thread(
                    json.dumps,
                    final_report,
                    indent=2,
                    ensure_ascii=False,
                )
            else:
                report_json = json.dumps(final_report, indent=2, ensure_ascii=False)
            max_json_length = 5000
            if len(report_json) > max_json_length:
                log.info(report_json[:max_json_length])
                log.info(f"\n  ... [JSON truncated - total length: {len(report_json)} characters]")
            else:
                log.info(report_json)
        except Exception as e:
            log.warning(f"  Could not serialize report to JSON: {e}")
        log.info("")
        log.info("=" * 100)
        log.info("")

    # Optionally persist final context and evaluation
    # CRITICAL FIX: Use fire-and-forget with timeout to prevent blocking subsequent requests
    if session_id:
        async def _persist_final_data():
            """Non-blocking persistence with timeout guard"""
            try:
                # Use asyncio.wait_for to prevent hanging on slow ChromaDB operations
                await asyncio.wait_for(
                    save_conversation_context(session_id, conversation_context.to_dict()),
                    timeout=5.0  # 5 second timeout for context save
                )
            except asyncio.TimeoutError:
                log.warning(f"[TIMEOUT] save_conversation_context exceeded 5s for session {session_id}")
            except Exception as e:
                log.debug(f"Failed to persist conversation context: {e}")
            
            try:
                await asyncio.wait_for(
                    save_session({
                        "session_id": session_id,
                        "status": "completed",
                        "last_updated": time.time(),
                        "evaluation": evaluation_result
                    }),
                    timeout=5.0  # 5 second timeout for session save
                )
            except asyncio.TimeoutError:
                log.warning(f"[TIMEOUT] save_session exceeded 5s for session {session_id}")
            except Exception as e:
                log.debug(f"Failed to persist final session: {e}")
        
        # Issue 6.1: Use schedule_background_task for proper exception logging
        schedule_background_task(_persist_final_data(), f"persist_final_data:{session_id[:8]}")
        log.debug("Scheduled non-blocking final persistence for session %s", session_id)

    # Build InterviewResponse with structured report - use final_history to ensure persistence
    log.info(f"[RESPONSE] Building InterviewResponse with conversation_history length: {len(final_history)}")
    assistant_questions = sum(1 for msg in final_history if (msg.get("role") or "").lower() == "assistant")
    interview_req.question_count = assistant_questions

    # Debug-only probe: log evaluation + summary artifacts before building InterviewResponse
    log.error("\n\n>>> EVAL DEBUG <<<")
    log.error(f"evaluation_result: {evaluation_result}")
    log.error(f"llm_summary_dict: {llm_summary_dict}")
    log.error(f"llm_summary_text: {llm_summary_text[:200]}")
    log.error(f"final_report keys: {list(final_report.keys())}")
    log.error(">>> END EVAL DEBUG <<<\n")

    # Send callback notification if callback URL is provided
    if interview_req.callback_url:
        try:
            from .callbacks import send_callback_notification
            await send_callback_notification(
                callback_url=interview_req.callback_url,
                uid=interview_req.uid,
                session_id=session_id,
                status="completed",
                evaluation_summary=evaluation_result,  # Keep for backward compatibility
                detailed_summary=llm_summary_text,
                structured_report=final_report,  # Send new UI-formatted report
                auth_token=interview_req.auth_token,
                conversation_context={
                    "topics_discussed": conversation_context.topics_discussed,
                    "engagement_level": conversation_context.engagement_level,
                    "interview_phase": conversation_context.interview_phase
                },
                response_analysis=None,
                session_metadata={
                    "session_id": session_id,
                    "question_count": interview_req.question_count or assistant_questions
                }
            )
            log.info(f"[CALLBACK] Sent structured report to {interview_req.callback_url}")
        except Exception as e:
            log.error(f"[CALLBACK] Error sending callback to {interview_req.callback_url}: {e}", exc_info=True)

    resp = InterviewResponse(
        conversation_history=final_history,
        question="Thank you for your time. We will be in touch with the next steps.",
        session_id=session_id,
        current_state=InterviewState.COMPLETED,
        evaluation_summary=evaluation_result,
        detailed_summary=llm_summary_text,
        final_summary=final_report,
        evaluation=evaluation_result,
        status="completed",
        question_count=interview_req.question_count or assistant_questions,
    )

    # Debug-only probe: log key fields after InterviewResponse is built
    log.error(f"INTERVIEW_RESPONSE.evaluation_summary: {resp.evaluation_summary}")
    log.error(f"INTERVIEW_RESPONSE.detailed_summary: {resp.detailed_summary}")
    log.error(f"INTERVIEW_RESPONSE.conversation_history len: {len(resp.conversation_history)}")

    # Log conclusion with evaluation status
    eval_status = "FAILED" if evaluation_result.get("failed_evaluation") else "OK"
    log.info(
        f"Interview concluded for session {session_id} | "
        f"Score: {evaluation_result.get('overall_score', 0.0)} | "
        f"Question Count: {interview_req.question_count or 0} | "
        f"Evaluation Status: {eval_status}"
    )
    log.info(f"Final Interview Response: {resp.question[:150]}{'...' if len(resp.question) > 150 else ''}")
    log.info(f"[RESPONSE] InterviewResponse conversation_history length: {len(resp.conversation_history)}")
    log.debug(f"Full InterviewResponse: {resp.dict()}")
    return resp

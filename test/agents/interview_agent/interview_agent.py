"""
Main interview agent entry point.

This module provides the main API endpoint for the interview agent,
using the refactored modular architecture.
"""

import logging
import time
import json
import asyncio
import uuid
from typing import Dict, List, Any, Tuple, AsyncIterator, Optional
from fastapi import Request
from fastapi.responses import StreamingResponse, Response
from settings import settings

# Import from new modular structure
from .request_parser import (
    parse_and_validate_request,
    build_interview_context_data,
)
from .anonymizer_module import PIIAnonymizer, sanitize_answer_for_prompt
from .conversation_context import ConversationContext
from .session_manager import save_conversation_context, load_conversation_context
from .state_manager import InterviewState, InterviewRequest, InterviewResponse, PersonaType
# Callbacks are now handled inside evaluator module
from .fallback_generator import get_natural_fallback_question, get_fallback_question_for_state, get_default_question
from .question_generator import (
    PSYCHOMETRIC_KEYWORDS,
    PERSONALITY_KEYWORDS,
    COMMUNICATION_KEYWORDS,
    _contains_banned_test_phrase,
    _build_safe_behavioral_question,
)
from .streaming_config import StreamingConfig
from .anti_repetition import check_question_similarity, track_question_fingerprint

# Import from simplified orchestrator module (core flow)
from .orchestrator import (
    run_turn,
    conclude_interview,
)

log = logging.getLogger(__name__)


async def ai_interview_agent_intelligent(request: Request) -> InterviewResponse:
    """
    Intelligent, context-aware interview agent with natural conversation flow.
    
    This is the main entry point for the interview agent API.
    It orchestrates the interview flow using the modular architecture.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        InterviewResponse with next question and conversation state
    """
    try:
        # Parse and validate request
        interview_req = await parse_and_validate_request(request)
        
        # Build context data (includes all processing - no need for duplicate process_uid_context)
        uid_context, candidate_info, safe_job_details, names, companies, session_id, start_time, anonymizer = build_interview_context_data(interview_req)
        
        # Rely on client-provided question_count, or default to 0
        server_question_count = interview_req.question_count or 0
        
        # Check if interview should end
        if interview_req.end_interview:
            # CRITICAL: Ensure conversation_history is loaded before concluding
            # request_parser should have loaded it, but add safety check here too
            if (not interview_req.conversation_history or len(interview_req.conversation_history) == 0) and session_id:
                from .session_manager import load_conversation_history
                try:
                    loaded_history = await load_conversation_history(session_id)
                    if loaded_history:
                        interview_req.conversation_history = loaded_history
                        log.info(f"[SAFETY] Loaded conversation_history before conclude: {len(loaded_history)} messages")
                except Exception as e:
                    log.warning(f"[SAFETY] Failed to load conversation_history before conclude: {e}")
            
            # Load conversation context for conclusion
            conversation_context = None
            if session_id:
                try:
                    conv_ctx = await load_conversation_context(session_id)
                    if conv_ctx:
                        conversation_context = ConversationContext.from_dict(conv_ctx)
                except Exception:
                    pass
            if conversation_context is None:
                conversation_context = ConversationContext()
            
            # Conclude interview
            from .evaluator import InterviewEvaluator
            evaluator = InterviewEvaluator()
            final_response = await conclude_interview(
                interview_req=interview_req,
                conversation_context=conversation_context,
                session_id=session_id,
                candidate_info=candidate_info,
                safe_job_details=safe_job_details,
                evaluator=evaluator
            )
            return final_response
        
        # Use simplified orchestrator to process turn
        # run_turn handles: context loading, anonymization, analysis, question generation, persistence
        next_question, response_analysis = await run_turn(
            interview_req=interview_req,
            uid_context=uid_context,
            candidate_info=candidate_info,
            safe_job_details=safe_job_details,
            session_id=session_id,
            server_question_count=server_question_count,
            anonymizer=anonymizer,
            persist=True
        )
        
        # CRITICAL: Check if exit intent was detected - if so, conclude interview immediately
        if response_analysis and (response_analysis.get("should_end_interview") or response_analysis.get("exit_intent_detected")):
            log.info(f"[EXIT_INTENT] Exit intent detected in response_analysis. Concluding interview for session: {session_id}")
            
            # Load conversation context for conclusion
            conversation_context = None
            if session_id:
                try:
                    conv_ctx = await load_conversation_context(session_id)
                    if conv_ctx:
                        conversation_context = ConversationContext.from_dict(conv_ctx)
                except Exception as e:
                    log.warning(f"[EXIT_INTENT] Failed to load conversation context: {e}")
            if conversation_context is None:
                conversation_context = ConversationContext()
            
            # Ensure conversation_history is loaded before concluding
            # Note: run_turn should have already added the exit message, but verify it's there
            if (not interview_req.conversation_history or len(interview_req.conversation_history) == 0) and session_id:
                from .session_manager import load_conversation_history
                try:
                    loaded_history = await load_conversation_history(session_id)
                    if loaded_history:
                        interview_req.conversation_history = loaded_history
                        log.info(f"[EXIT_INTENT] Loaded conversation_history before conclude: {len(loaded_history)} messages")
                except Exception as e:
                    log.warning(f"[EXIT_INTENT] Failed to load conversation_history before conclude: {e}")
            
            # Verify exit message is in conversation history (run_turn should have added it)
            exit_message = response_analysis.get("exit_message") or next_question or "Thank you for your time. We'll wrap up the interview now."
            if interview_req.conversation_history:
                # Check if last message is the exit message
                last_msg = interview_req.conversation_history[-1] if interview_req.conversation_history else None
                if not last_msg or last_msg.get("role") != "assistant" or exit_message not in last_msg.get("content", ""):
                    # Add exit message if it's missing
                    interview_req.conversation_history.append({
                        "role": "assistant",
                        "content": exit_message
                    })
                    log.info(f"[EXIT_INTENT] Added exit message to conversation_history (was missing)")
            
            # Conclude interview with error handling
            from .evaluator import InterviewEvaluator
            evaluator = InterviewEvaluator()
            try:
                final_response = await conclude_interview(
                    interview_req=interview_req,
                    conversation_context=conversation_context,
                    session_id=session_id,
                    candidate_info=candidate_info,
                    safe_job_details=safe_job_details,
                    evaluator=evaluator
                )
                log.info(f"[EXIT_INTENT] Interview concluded successfully for session: {session_id}")
                return final_response
            except Exception as e:
                log.error(f"[EXIT_INTENT] Failed to conclude interview: {e}", exc_info=True)
                # Return a fallback response indicating interview ended
                return InterviewResponse(
                    conversation_history=interview_req.conversation_history or [],
                    question=exit_message,
                    session_id=session_id,
                    current_state=InterviewState.COMPLETED,
                    status="completed",
                    session_metadata={
                        "error": f"Conclusion failed: {str(e)}",
                        "exit_intent_detected": True
                    },
                )
        
        # Load conversation context to get current state (run_turn already persisted it)
        conversation_context = None
        if session_id:
            try:
                conv_ctx = await load_conversation_context(session_id)
                if conv_ctx:
                    conversation_context = ConversationContext.from_dict(conv_ctx)
            except Exception:
                pass
        if conversation_context is None:
            conversation_context = ConversationContext()
        
        # Get question count from request (run_turn already updated it)
        question_count = interview_req.question_count or 0
        
        # NOTE: Do NOT append question to conversation_history here
        # run_turn already handles conversation history updates internally
        # Appending here would create duplicates: [sys, user, assist, assist]
        
        # Build response
        # Compute state directly from question_count (single source of truth)
        # This avoids stale data from async persistence
        if question_count == 0:
            current_state = InterviewState.TOPIC_INTRODUCTION
        elif question_count <= 2:
            current_state = InterviewState.TOPIC_FUNDAMENTALS
        elif question_count <= 5:
            current_state = InterviewState.TOPIC_DEEP_DIVE
        else:
            current_state = InterviewState.TOPIC_FEEDBACK
        
        # Override with conversation context if available and valid
        if hasattr(conversation_context, 'interview_stage') and conversation_context.interview_stage:
            stage = (conversation_context.interview_stage or "").strip().lower()
            if stage == "topic_fundamentals" and question_count >= 1:
                current_state = InterviewState.TOPIC_FUNDAMENTALS
            elif stage == "topic_deep_dive" and question_count >= 3:
                current_state = InterviewState.TOPIC_DEEP_DIVE
            elif stage == "topic_feedback" and question_count >= 6:
                current_state = InterviewState.TOPIC_FEEDBACK
        
        response = InterviewResponse(
            conversation_history=interview_req.conversation_history,
            question=next_question,
            session_id=session_id,
            current_state=current_state,
            question_count=question_count,  # Add to top-level response
            session_metadata={
                "question_count": question_count,
                "engagement_level": conversation_context.engagement_level,
                "topics_discussed": conversation_context.topics_discussed,
            },
        )
        
        # Log the final response being returned
        log.info(
            f"Interview Agent Final Response | Session: {session_id} | "
            f"Question Count: {question_count} | State: {current_state} | "
            f"Question: {next_question[:150]}{'...' if len(next_question) > 150 else ''}"
        )
        log.debug(f"Full InterviewResponse: question={next_question}, session_id={session_id}, state={current_state}")
        
        return response
        
    except Exception as e:
        log.error(f"Error in interview agent: {e}", exc_info=True)
        # Return error response
        return InterviewResponse(
            conversation_history=[],
            question="I apologize, but I encountered an error. Please try again.",
            session_id=None,
            current_state=None,
            session_metadata={"error": str(e)},
        )


# -------------------------
# Streaming Support
# -------------------------

async def _stream_question_generation(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.7
) -> AsyncIterator[str]:
    """
    Stream question generation using LangChain streaming.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name (optional, defaults to gemini-2.5-flash)
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation
    
    Yields:
        str: Individual tokens/chunks from the LLM
    """
    try:
        from dataclasses import replace
        from core.model_registry import get_model_for_task, TaskType
        from models.llm_invoker import _create_model_instance
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        
        # Get model config - use configurable model from settings
        model_name = model or getattr(settings, "INTERVIEW_QUESTION_MODEL", "gemini-2.5-flash")
        model_config = get_model_for_task(TaskType.TEXT_GENERATION, preferred_model=model_name)
        # Override temperature for interview question generation
        model_config = replace(model_config, temperature=temperature)
        
        # Use StreamingConfig for max_tokens if not provided (prevents early cut-off in streaming)
        token_limit = max_tokens or StreamingConfig.QUESTION_MAX_TOKENS
        log.info(f"Stream question generation: max_output_tokens={token_limit} (ensures full question, not a word limit)")
        
        # Create model instance
        model_instance = await _create_model_instance(
            model_config,
            response_mime_type=None,
            max_output_tokens=token_limit,
            system_instruction=None
        )
        
        # Convert messages to LangChain format
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))
        
        # Stream response
        # LangChain's astream yields AIMessage chunks, we need to extract content properly
        chunk_count = 0
        total_content_length = 0
        
        log.info(f"Starting stream for model {model_name} with {len(langchain_messages)} messages")
        
        # Use bound model so max_output_tokens is applied for this stream (constructor may be ignored in streaming)
        try:
            stream_model = model_instance.bind(max_output_tokens=token_limit)
        except TypeError:
            stream_model = model_instance
        
        try:
            async for chunk in stream_model.astream(langchain_messages):
                chunk_count += 1
                try:
                    content_to_yield = None
                    
                    # Log chunk type for debugging
                    chunk_type = type(chunk).__name__
                    log.debug(f"Received chunk #{chunk_count}: type={chunk_type}, hasattr content={hasattr(chunk, 'content')}")
                    
                    # Handle AIMessage objects (most common case)
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                        log.debug(f"Chunk #{chunk_count} content type: {type(content)}, value: {repr(content[:50]) if content else 'None'}")
                        
                        if content:  # Only yield non-empty content
                            if isinstance(content, str):
                                content_to_yield = content
                            elif isinstance(content, list):
                                # Handle list of content blocks (some models return this)
                                log.debug(f"Chunk #{chunk_count} has list content with {len(content)} items")
                                for idx, item in enumerate(content):
                                    if isinstance(item, str) and item:
                                        content_to_yield = item
                                        log.debug(f"Using list item #{idx} as string: '{item[:30]}...'")
                                        break
                                    elif hasattr(item, 'text') and item.text:
                                        content_to_yield = item.text
                                        log.debug(f"Using list item #{idx} text attribute: '{item.text[:30]}...'")
                                        break
                                if not content_to_yield:
                                    # Fallback: stringify all items
                                    content_to_yield = "".join(str(item) for item in content if item)
                                    log.debug(f"Fallback: joined all list items: '{content_to_yield[:30]}...'")
                            else:
                                content_to_yield = str(content) if content else None
                                log.debug(f"Converted non-string content to string: '{content_to_yield[:30] if content_to_yield else None}...'")
                    
                    # Handle direct string chunks
                    elif isinstance(chunk, str) and chunk.strip():
                        content_to_yield = chunk
                        log.debug(f"Chunk #{chunk_count} is direct string: '{chunk[:30]}...'")
                    
                    # Handle dict-like chunks
                    elif isinstance(chunk, dict):
                        content = chunk.get('content') or chunk.get('text')
                        if content:
                            content_to_yield = str(content) if not isinstance(content, str) else content
                            log.debug(f"Chunk #{chunk_count} is dict with content: '{content_to_yield[:30]}...'")
                    
                    # Yield the content if we have any
                    if content_to_yield:
                        total_content_length += len(content_to_yield)
                        log.info(f"Received chunk #{chunk_count}: '{content_to_yield[:50]}...' ({len(content_to_yield)} chars, total: {total_content_length} chars)")
                        
                        # Split large chunks into batches of up to STREAMING_WORDS_PER_CHUNK words per yield
                        words_per_chunk = getattr(StreamingConfig, "STREAMING_WORDS_PER_CHUNK", 48)
                        if len(content_to_yield) > 10:
                            words = content_to_yield.split()
                            if len(words) > 1:
                                for i in range(0, len(words), words_per_chunk):
                                    batch = words[i : i + words_per_chunk]
                                    chunk_piece = " ".join(batch) if i == 0 else " " + " ".join(batch)
                                    yield chunk_piece
                                    log.debug(f"  Yielding chunk {i // words_per_chunk + 1} ({len(batch)} words, total so far: {min(i + words_per_chunk, len(words))}/{len(words)} words in this API chunk)")
                            else:
                                yield content_to_yield
                        else:
                            yield content_to_yield
                    else:
                        log.warning(f"Skipping empty chunk #{chunk_count} (type: {chunk_type}, repr: {repr(chunk)[:100]})")
                        
                except Exception as chunk_error:
                    log.exception(f"Error processing chunk #{chunk_count}: {chunk_error}, chunk type: {type(chunk)}, chunk repr: {repr(chunk)[:100]}")
                    # Continue streaming even if one chunk fails
                    continue
            
            log.info(f"Streaming completed: {chunk_count} total chunks, {total_content_length} total chars")
            
            if chunk_count == 0:
                log.error("No chunks received from stream!")
            elif total_content_length < 10:
                log.warning(f"Very short content received: only {total_content_length} chars from {chunk_count} chunks")
                
        except Exception as stream_error:
            log.exception(f"Error in stream loop: {stream_error}")
            raise
                
    except Exception as e:
        log.exception(f"Error streaming question: {e}")
        # Don't yield empty string - let caller handle the error
        # The exception will propagate and be handled by the caller
        raise


async def _save_interview_turn(
    session_id: str,
    interview_req: InterviewRequest,
    question: str,
    response_analysis: Dict[str, Any],
    conversation_context: ConversationContext,
    question_count: int
):
    """Helper to save interview turn asynchronously (fire-and-forget for first questions)"""
    try:
        from .session_manager import save_conversation_history, save_conversation_context
        
        # Add question to conversation history
        if interview_req.conversation_history is None:
            interview_req.conversation_history = []
        
        # Add answer if provided
        if interview_req.answer and interview_req.answer.strip():
            interview_req.conversation_history.append({
                "role": "user",
                "content": interview_req.answer
            })
        
        # Add question
        interview_req.conversation_history.append({
            "role": "assistant",
            "content": question
        })
        
        # OPTIMIZATION: For first question (question_count=1), only save minimal data
        # Full context save can be deferred to subsequent turns
        if question_count == 1:
            # Minimal save for first question - just history, skip context
            asyncio.create_task(save_conversation_history(session_id, interview_req.conversation_history))
            # Skip context save for first question (will be saved on next turn)
            log.debug(f"[PERF] First question - deferred context save for session {session_id}")
        else:
            # Full save for subsequent questions
            await save_conversation_history(session_id, interview_req.conversation_history)
            conversation_context.question_count = question_count
            await save_conversation_context(session_id, conversation_context.to_dict())
        
    except Exception as e:
        log.warning(f"Error saving interview turn: {e}")


async def ai_interview_agent_intelligent_stream(request: Request) -> StreamingResponse:
    """
    Streaming version of the interview agent that yields SSE events.
    
    Returns:
        StreamingResponse: Server-Sent Events (SSE) stream
    """
    # Generate correlation ID for request tracing
    correlation_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Parse and validate request
        interview_req = await parse_and_validate_request(request)
        
        # Build context data
        uid_context, candidate_info, safe_job_details, names, companies, session_id, _, anonymizer = build_interview_context_data(interview_req)
        
        # Structured logging
        log.info(
            "Starting streaming interview",
            extra={
                "correlation_id": correlation_id,
                "session_id": session_id,
                "topic": interview_req.interview_topic,
                "question_count": interview_req.question_count or 0,
                "generate_avatar": getattr(interview_req, 'generate_avatar', False)
            }
        )
        
        # Use client-provided question_count, or default to 0
        server_question_count = interview_req.question_count or 0
        
        # Check if interview should end
        if interview_req.end_interview:
            # Load conversation context for conclusion
            conversation_context = None
            if session_id:
                try:
                    conv_ctx = await load_conversation_context(session_id)
                    if conv_ctx:
                        conversation_context = ConversationContext.from_dict(conv_ctx)
                except Exception:
                    pass
            if conversation_context is None:
                conversation_context = ConversationContext()
            
            # Conclude interview (non-streaming for conclusion)
            from .evaluator import InterviewEvaluator
            evaluator = InterviewEvaluator()
            final_response = await conclude_interview(
                interview_req=interview_req,
                conversation_context=conversation_context,
                session_id=session_id,
                candidate_info=candidate_info,
                safe_job_details=safe_job_details,
                evaluator=evaluator
            )
            
            # Return non-streaming JSON response with full report
            # This ensures the report is immediately available in the HTTP response
            
            response_data = {
                "status": final_response.status,
                "session_id": final_response.session_id,
                "question": final_response.question,
                "question_count": final_response.question_count,
                "conversation_history": final_response.conversation_history,
                "current_state": final_response.current_state.value if final_response.current_state else None,
                "evaluation_summary": final_response.evaluation_summary,
                # Include the full structured report (UI-formatted)
                "structured_report": final_response.final_summary,
                # Alias for backward compatibility
                "final_report": final_response.evaluation_summary,
                "narrative_report": final_response.detailed_summary,
                "detailed_summary": final_response.detailed_summary,
                "interview_completed": True
            }
            
            return Response(
                content=json.dumps(response_data, ensure_ascii=False),
                media_type="application/json",
                status_code=200
            )
        
        # Check avatar rate limiting before streaming
        generate_avatar = interview_req.generate_avatar if hasattr(interview_req, 'generate_avatar') else False
        avatar_rate_limited = False
        
        if generate_avatar:
            uid = interview_req.uid
            if uid:
                try:
                    # Access app state from request
                    app_state = request.app.state if hasattr(request, 'app') else None
                    if app_state:
                        from time import time as current_time
                        avatar_rl = getattr(app_state, "_avatar_rate_limits", {})
                        now = current_time()
                        user_bucket = avatar_rl.get(uid, {"count": 0, "reset_at": now + 3600})
                        
                        # Reset window if expired
                        if now > user_bucket["reset_at"]:
                            user_bucket = {"count": 0, "reset_at": now + 3600}
                        
                        user_bucket["count"] += 1
                        avatar_rl[uid] = user_bucket
                        app_state._avatar_rate_limits = avatar_rl
                        
                        if user_bucket["count"] > 10:  # 10 avatars per hour
                            log.warning(
                                f"Avatar rate limit exceeded for uid={uid}",
                                extra={"correlation_id": correlation_id}
                            )
                            avatar_rate_limited = True
                            generate_avatar = False
                except Exception as rl_error:
                    log.warning(f"Error checking avatar rate limit: {rl_error}")
        
        # Process turn with streaming question generation
        async def stream_interview_response():
            # Send initial event
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"
            
            # Send rate limit error if applicable
            if avatar_rate_limited:
                error_data = {
                    'type': 'avatar_error',
                    'message': 'Avatar generation rate limit exceeded. Please wait before requesting more avatars.'
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            
            # OPTIMIZATION: Parallelize all independent operations
            from .orchestrator import _analyze_response
            from .session_manager import load_conversation_context, load_conversation_history
            
            # Send progress indicator
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'loading_context', 'message': 'Loading conversation context...'})}\n\n"
            
            # Parallel session loading and anonymization (if needed)
            conversation_context = None
            loaded_history = None
            anonymized_history = interview_req.conversation_history or []
            
            # OPTIMIZATION: Skip ChromaDB loads for first question (new sessions)
            # New sessions have no history to load, so skip the expensive ChromaDB calls
            is_first_question = not interview_req.conversation_history or len(interview_req.conversation_history) == 0
            
            # Prepare tasks for parallel execution
            tasks = {}
            
            # Task 1: Load conversation context (skip for first question)
            if session_id and not is_first_question:
                tasks['context'] = load_conversation_context(session_id)
            
            # Task 2: Load conversation history (skip if not needed or first question)
            if session_id and not interview_req.conversation_history and not is_first_question:
                tasks['history'] = load_conversation_history(session_id)
            
            # Task 3: Anonymize history (if answer provided) - can run in parallel
            if interview_req.answer and interview_req.answer.strip():
                from utils.interview_utils import anonymize_conversation_history as anonymize_history_util
                # Anonymization is CPU-bound, but small enough to run async
                async def anonymize_task():
                    return anonymize_history_util(
                        interview_req.conversation_history or [],
                        anonymizer,
                        candidate_info.get("names", []),
                        candidate_info.get("companies", [])
                    ) if anonymizer else (interview_req.conversation_history or [])
                tasks['anonymize'] = anonymize_task()
            
            # Execute all tasks in parallel
            if tasks:
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                task_names = list(tasks.keys())
                
                for i, (task_name, result) in enumerate(zip(task_names, results)):
                    if isinstance(result, Exception):
                        log.warning(f"Error in parallel task '{task_name}': {result}")
                        continue
                    
                    if task_name == 'context' and result:
                        conversation_context = ConversationContext.from_dict(result)
                    elif task_name == 'history' and result:
                        loaded_history = result
                        if not interview_req.conversation_history:
                            interview_req.conversation_history = loaded_history
                    elif task_name == 'anonymize' and result:
                        anonymized_history = result
            
            if conversation_context is None:
                conversation_context = ConversationContext()
            
            # Analyze response (if answer provided) - now runs after parallel loading
            response_analysis = {}
            if interview_req.answer and interview_req.answer.strip():
                # Send progress indicator for analysis
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'analyzing', 'message': 'Analyzing response...'})}\n\n"
                
                response_analysis = await _analyze_response(
                    interview_req,
                    anonymized_history,
                    candidate_info,
                    safe_job_details,
                    uid_context,
                    server_question_count
                )
            
            # Check for exit intent
            if response_analysis.get("should_end_interview") or response_analysis.get("exit_intent_detected"):
                exit_message = response_analysis.get("exit_message") or "Thank you for your time. We'll wrap up the interview now."
                yield f"data: {json.dumps({'type': 'chunk', 'content': exit_message})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'question': exit_message, 'question_count': server_question_count, 'interview_completed': True})}\n\n"
                return
            
            # Update conversation context with analysis (if answer was provided)
            if response_analysis and interview_req.answer and interview_req.answer.strip():
                try:
                    await conversation_context.update_context(response_analysis, server_question_count, persist=False)
                except Exception:
                    log.debug("Failed to update conversation context, continuing")
            
            # Determine stage from question count and analysis
            from .orchestrator import _determine_stage_from_count
            stage = _determine_stage_from_count(server_question_count, response_analysis)
            conversation_context.interview_stage = stage
            conversation_context.interview_phase = stage
            
            # Generate question with streaming (aiinterview-style: professional interviewer, one question at a time)
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'generating_question', 'message': 'Generating question...'})}\n\n"
            
            truncate_count = StreamingConfig.HISTORY_TRUNCATE_MESSAGES
            history_for_generation = anonymized_history[-truncate_count:] if len(anonymized_history) > truncate_count else anonymized_history
            resume = getattr(interview_req, "resume", None) or getattr(interview_req, "structured_resume", None) or {}
            topic = (interview_req.interview_topic or "general").strip()
            from .question_generator import _greeting_name_from_candidate_info
            candidate_name = _greeting_name_from_candidate_info(candidate_info)
            from .aiinterview_question_gen import build_messages as build_aiinterview_messages
            messages = build_aiinterview_messages(history_for_generation, resume, topic, candidate_name)
            model_name = getattr(settings, "INTERVIEW_QUESTION_MODEL", "gemini-2.5-flash")
            
            # OPTIMIZATION: Start video generation in parallel with question generation
            video_updates_queue = asyncio.Queue()
            video_generation_started = False
            question_buffer = ""
            min_question_length_for_video = 20
            
            async def stream_video_updates(question_text: str):
                if generate_avatar and not avatar_rate_limited:
                    try:
                        from .video_utils import generate_avatar_video_streaming
                        async for avatar_update in generate_avatar_video_streaming(question_text.strip()):
                            await video_updates_queue.put(avatar_update)
                        await video_updates_queue.put(None)
                    except Exception as e:
                        log.exception(f"Error in video generation task: {e}")
                        await video_updates_queue.put({"type": "error", "message": str(e)})
            
            # Greeting only for the first message (no prior assistant in history; treat "model" as assistant)
            def _is_assistant_msg(m):
                r = (m.get("role") or "").strip().lower()
                return r == "assistant" or r == "model"
            is_first_message = not any(_is_assistant_msg(m) for m in (anonymized_history or []))
            if is_first_message:
                greeting_chunk = f"Hello {candidate_name}, "
                yield f"data: {json.dumps({'type': 'chunk', 'content': greeting_chunk})}\n\n"
            
            full_question = ""
            chunk_count = 0
            video_task = None
            # When we already sent "Hello there, ", strip same from LLM output to avoid duplicate/CANDIDATE_A
            from .question_generator import _strip_leading_greeting
            greeting_stripped = False
            llm_greeting_buffer = ""
            
            try:
                async for chunk in _stream_question_generation(
                    messages,
                    model=model_name,
                    max_tokens=StreamingConfig.QUESTION_MAX_TOKENS,
                    temperature=0.7
                ):
                    if chunk:  # Only process non-empty chunks
                        chunk_count += 1
                        # On first message, strip leading "Hello X, " from LLM so we don't duplicate or show CANDIDATE_A
                        if is_first_message and not greeting_stripped:
                            llm_greeting_buffer += chunk
                            if ", " in llm_greeting_buffer or ",\n" in llm_greeting_buffer:
                                stripped = _strip_leading_greeting(llm_greeting_buffer)
                                if len(stripped) < len(llm_greeting_buffer):
                                    chunk = stripped
                                    greeting_stripped = True
                                    llm_greeting_buffer = ""
                                else:
                                    chunk = None
                            elif len(llm_greeting_buffer) > 120:
                                chunk = _strip_leading_greeting(llm_greeting_buffer)
                                greeting_stripped = True
                                llm_greeting_buffer = ""
                            else:
                                chunk = None
                        if chunk:
                            # FIX: Avoid word concatenation across stream chunk boundaries (e.g. "...design" + "choice")
                            # If the last char of the current buffer is a letter and the first char of this chunk
                            # is also a letter, insert a space between them.
                            if full_question and chunk:
                                last_char = full_question[-1]
                                first_char = chunk[0]
                                if last_char.isalpha() and first_char.isalpha():
                                    chunk = " " + chunk
                            full_question = (full_question or "") + chunk
                            question_buffer += chunk  # Accumulate for video generation
                            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                        log.debug(f"Streaming chunk #{chunk_count}: '{chunk[:50]}...' (total so far: {len(full_question)} chars)")
                        
                        # Start video generation in parallel once we have enough text
                        if (generate_avatar and not avatar_rate_limited and 
                            not video_generation_started and 
                            len(question_buffer) >= min_question_length_for_video and
                            question_buffer.strip()):
                            video_generation_started = True
                            # Start video generation task in background with current question text
                            # Note: We'll update this with final processed question later if needed
                            video_task = asyncio.create_task(stream_video_updates(question_buffer.strip()))
                            log.info(
                                f"Started video generation in parallel (question length: {len(question_buffer)} chars)",
                                extra={"correlation_id": correlation_id}
                            )
                        
                        # Check for video updates (non-blocking)
                        try:
                            while not video_updates_queue.empty():
                                video_update = await asyncio.wait_for(
                                    video_updates_queue.get_nowait(), timeout=0.001
                                )
                                if video_update is None:
                                    break  # Video generation completed
                                # Yield video update
                                update_type = video_update.get("type")
                                if update_type == "status":
                                    status_data = {
                                        'type': 'avatar_status',
                                        'message': video_update.get('message'),
                                        'stage': video_update.get('stage'),
                                        'progress': video_update.get('progress'),
                                        'progress_percent': video_update.get('progress_percent'),
                                        'estimated_seconds_remaining': video_update.get('estimated_seconds_remaining')
                                    }
                                    yield f"data: {json.dumps(status_data)}\n\n"
                                elif update_type == "video":
                                    delivery_method = video_update.get("delivery_method", "base64_chunks")
                                    if delivery_method == "direct_url" or delivery_method == "direct_url_fallback":
                                        video_data = {
                                            'type': 'avatar_video_ready',
                                            'format': 'mp4',
                                            'video_url': video_update.get('video_url'),
                                            'video_id': video_update.get('video_id'),
                                            'size_bytes': video_update.get('size_bytes', 0),
                                            'delivery_method': delivery_method
                                        }
                                        yield f"data: {json.dumps(video_data)}\n\n"
                                    elif video_update.get("video_base64"):
                                        chunk_data = {
                                            'type': 'avatar_video_chunk',
                                            'chunk': video_update.get('video_base64', ''),
                                            'chunk_size': video_update.get('chunk_size', 0),
                                            'is_last': video_update.get('is_last', False)
                                        }
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                elif update_type == "error":
                                    error_data = {
                                        'type': 'avatar_error',
                                        'message': video_update.get('message'),
                                        'video_url': video_update.get('video_url')
                                    }
                                    yield f"data: {json.dumps(error_data)}\n\n"
                        except (asyncio.TimeoutError, asyncio.QueueEmpty):
                            pass  # No video updates yet, continue with question streaming
                
                log.info(f"Streaming completed: {chunk_count} chunks, {len(full_question)} total chars")
                
                # If we got very few chunks or very short question, log warning only (we don't micromanage; let the LLM give)
                if chunk_count < 3 or len(full_question) < 10:
                    log.warning(f"Streaming may have stopped early: only {chunk_count} chunks, {len(full_question)} chars")
                
                # After question streaming completes, process any remaining video updates
                if video_task and not video_task.done():
                    # Wait for video generation to complete and stream remaining updates
                    while True:
                        try:
                            video_update = await asyncio.wait_for(
                                video_updates_queue.get(), timeout=0.5
                            )
                            if video_update is None:
                                break  # Video generation completed
                            
                            # Yield video update
                            update_type = video_update.get("type")
                            if update_type == "status":
                                status_data = {
                                    'type': 'avatar_status',
                                    'message': video_update.get('message'),
                                    'stage': video_update.get('stage'),
                                    'progress': video_update.get('progress'),
                                    'progress_percent': video_update.get('progress_percent'),
                                    'estimated_seconds_remaining': video_update.get('estimated_seconds_remaining')
                                }
                                yield f"data: {json.dumps(status_data)}\n\n"
                            elif update_type == "video":
                                delivery_method = video_update.get("delivery_method", "base64_chunks")
                                if delivery_method == "direct_url" or delivery_method == "direct_url_fallback":
                                    video_data = {
                                        'type': 'avatar_video_ready',
                                        'format': 'mp4',
                                        'video_url': video_update.get('video_url'),
                                        'video_id': video_update.get('video_id'),
                                        'size_bytes': video_update.get('size_bytes', 0),
                                        'delivery_method': delivery_method
                                    }
                                    yield f"data: {json.dumps(video_data)}\n\n"
                                elif video_update.get("video_base64"):
                                    chunk_data = {
                                        'type': 'avatar_video_chunk',
                                        'chunk': video_update.get('video_base64', ''),
                                        'chunk_size': video_update.get('chunk_size', 0),
                                        'is_last': video_update.get('is_last', False)
                                    }
                                    yield f"data: {json.dumps(chunk_data)}\n\n"
                            elif update_type == "error":
                                error_data = {
                                    'type': 'avatar_error',
                                    'message': video_update.get('message'),
                                    'video_url': video_update.get('video_url')
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"
                        except asyncio.TimeoutError:
                            # Check if task is done
                            if video_task.done():
                                break
                            continue
                    
            except Exception as stream_error:
                log.exception(f"Error during streaming: {stream_error}")
                if not full_question.strip():
                    log.warning("Streaming failed, falling back to non-streaming (aiinterview-style)")
                    try:
                        from .aiinterview_question_gen import generate_question_nostream
                        qgen_result = await generate_question_nostream(
                            history_for_generation, resume, topic, candidate_name, model=model_name
                        )
                        full_question = (qgen_result or {}).get("question", "") or ""
                        if full_question:
                            yield f"data: {json.dumps({'type': 'chunk', 'content': full_question})}\n\n"
                    except Exception as fallback_error:
                        log.exception(f"Fallback generation also failed: {fallback_error}")
                        topic = interview_req.interview_topic or "your technical skills"
                        full_question = f"Walk me through a specific project where you applied {topic}."
            
            # Post-process question (same as non-streaming)
            # Log the raw question before processing
            log.debug(
                f"Raw streamed question (before processing): '{full_question}' ({len(full_question)} chars)",
                extra={"correlation_id": correlation_id}
            )
            
            # Let the LLM give: use streamed output with minimal post-process (no completeness/completion micromanagement)
            question = full_question.strip().split("\n")[0]
            question = question.strip('"').strip()
            from .question_generator import _normalize_spacing, _strip_leading_greeting
            question = _normalize_spacing(question)

            # Only fallback if empty or too short; otherwise trust the model output
            if len(question.strip()) < StreamingConfig.QUESTION_MIN_LENGTH:
                log.warning(f"Question too short: '{question}', using fallback")
                from .fallback_generator import get_natural_fallback_question
                try:
                    question = await get_natural_fallback_question(
                        conversation_context,
                        candidate_info,
                        session_id=session_id,
                        uid_context=uid_context,
                        interview_topic=interview_req.interview_topic
                    )
                except Exception:
                    topic = interview_req.interview_topic or "your technical skills"
                    question = f"Walk me through a specific project where you applied {topic}."
            elif not question.endswith("?") and not question.endswith("."):
                question = f"{question}?"

            if not question.strip():
                topic = interview_req.interview_topic or "your technical skills"
                question = f"Walk me through a specific project where you applied {topic}."

            # FINAL SAFETY NET for psychometric/personality/communication modes in streaming path:
            # If the final question still references tests/assessments/interviews,
            # override it with a deterministic behavioral question.
            topic = interview_req.interview_topic or ""
            topic_lower = topic.lower()
            is_psychometric = any(kw in topic_lower for kw in PSYCHOMETRIC_KEYWORDS)
            is_personality = any(kw in topic_lower for kw in PERSONALITY_KEYWORDS)
            is_communication = any(kw in topic_lower for kw in COMMUNICATION_KEYWORDS)
            if (is_psychometric or is_personality or is_communication) and _contains_banned_test_phrase(
                question
            ):
                log.warning(
                    "Streaming question for psychometric/personality/communication contained banned "
                    "test/meta phrase; replacing with safe behavioral template."
                )
                # Use same normalized behavioral topic as non-streaming generator
                if is_psychometric:
                    topic_for_question = (
                        "psychometric traits and decision-making patterns "
                        "(e.g., stress management, adaptability, collaboration)"
                    )
                elif is_personality:
                    topic_for_question = (
                        "personality traits and typical behavioral patterns at work "
                        "(e.g., collaboration style, conflict handling, motivation)"
                    )
                else:
                    topic_for_question = "communication skills in real work situations"
                question = _build_safe_behavioral_question(topic_for_question, stage)
            
            # Anti-repetition check for streaming path (same as non-streaming)
            # OPTIMIZATION: Skip for first question (no previous questions to compare)
            if session_id and question and server_question_count > 0:
                try:
                    similar = await check_question_similarity(question, session_id)
                    if similar:
                        log.warning(
                            f"Streaming: Question detected as similar to previous questions, using fallback",
                            extra={"correlation_id": correlation_id}
                        )
                        from .fallback_generator import get_natural_fallback_question
                        try:
                            question = await get_natural_fallback_question(
                                conversation_context,
                                candidate_info,
                                session_id=session_id,
                                uid_context=uid_context,
                                interview_topic=interview_req.interview_topic
                            )
                        except Exception:
                            topic = interview_req.interview_topic or "your technical skills"
                            question = f"What technical trade-offs have you encountered when working with {topic}?"
                    else:
                        # Track the question fingerprint if it's not similar
                        await track_question_fingerprint(question, session_id)
                except Exception as anti_rep_error:
                    log.debug(f"Anti-repetition check failed in streaming path: {anti_rep_error}")
                    # Best effort: still track the question even if check failed
                    if session_id:
                        try:
                            await track_question_fingerprint(question, session_id)
                        except Exception:
                            pass
            elif session_id and question and server_question_count == 0:
                # First question: just track fingerprint, skip similarity check
                log.debug(f"[PERF] First question - skipping anti-repetition check")
                try:
                    await track_question_fingerprint(question, session_id)
                except Exception:
                    pass
            
            # Ensure first message includes greeting (stream already sent it; align final payload)
            if is_first_message and question and not question.strip().lower().startswith(("hello ", "hi ", "hey ")):
                question = f"Hello {candidate_name}, " + question.lstrip()
            # On follow-up turns, strip any leading greeting so only the first question has it
            if not is_first_message and question:
                question = _strip_leading_greeting(question)
            
            log.info(f"Final processed question: '{question}' ({len(question)} chars)")
            
            # Note: Video generation is now handled in parallel during question streaming
            # If video generation wasn't started yet (e.g., question was too short), start it now
            # Also, if video was started with partial question, we may need to update it
            if (generate_avatar and question and question.endswith("?") and 
                not avatar_rate_limited):
                if not video_generation_started:
                    # Start video generation now
                    try:
                        from .video_utils import generate_avatar_video_streaming
                        log.info(
                            "Starting video generation (question was too short for parallel start)...",
                            extra={"correlation_id": correlation_id}
                        )
                        
                        # Stream avatar video generation
                        async for avatar_update in generate_avatar_video_streaming(question):
                            update_type = avatar_update.get("type")
                            
                            if update_type == "status":
                                # Enhanced progress indicators
                                status_data = {
                                    'type': 'avatar_status',
                                    'message': avatar_update.get('message'),
                                    'stage': avatar_update.get('stage'),
                                    'progress': avatar_update.get('progress'),
                                    'progress_percent': avatar_update.get('progress_percent'),
                                    'estimated_seconds_remaining': avatar_update.get('estimated_seconds_remaining')
                                }
                                yield f"data: {json.dumps(status_data)}\n\n"
                            elif update_type == "video":
                                delivery_method = avatar_update.get("delivery_method", "base64_chunks")
                                
                                # TRUE STREAMING: WebRTC Streaming API mode (real-time, low latency)
                                if delivery_method == "webrtc_streaming":
                                    video_data = {
                                        'type': 'avatar_video_ready',
                                        'format': 'webrtc',
                                        'delivery_method': 'webrtc_streaming',
                                        'session_id': avatar_update.get('session_id'),
                                        'task_id': avatar_update.get('task_id'),
                                        'webrtc': avatar_update.get('webrtc', {}),
                                        'is_streaming': True,
                                        'is_complete': True
                                    }
                                    yield f"data: {json.dumps(video_data)}\n\n"
                                    log.info(
                                        f"Avatar video streaming via WebRTC: session={avatar_update.get('session_id')}",
                                        extra={"correlation_id": correlation_id}
                                    )
                                # OPTIMIZED: Direct URL mode (browser handles streaming)
                                elif delivery_method == "direct_url" or delivery_method == "direct_url_fallback":
                                    avatar_duration = time.time() - avatar_start_time
                                    video_data = {
                                        'type': 'avatar_video_ready',
                                        'format': 'mp4',
                                        'video_url': avatar_update.get('video_url'),
                                        'video_id': avatar_update.get('video_id'),
                                        'size_bytes': avatar_update.get('size_bytes', 0),
                                        'delivery_method': delivery_method,
                                        'duration_seconds': round(avatar_duration, 2)
                                    }
                                    if avatar_update.get("warning"):
                                        video_data['warning'] = avatar_update.get("warning")
                                    yield f"data: {json.dumps(video_data)}\n\n"
                                    log.info(
                                        f"Avatar video ready (direct URL): {avatar_update.get('video_url')}",
                                        extra={"correlation_id": correlation_id}
                                    )
                                # Legacy: Base64 chunk streaming mode
                                elif avatar_update.get("is_streaming"):
                                    # Check if this is the start event
                                    if avatar_update.get("is_start") and not video_start_sent:
                                        # Send start event before first chunk
                                        start_data = {
                                            'type': 'avatar_video_start',
                                            'format': 'mp4',
                                            'video_url': avatar_update.get('video_url'),
                                            'size_bytes': avatar_update.get('size_bytes', 0),
                                            'is_streaming': True,
                                            'delivery_method': 'base64_chunks'
                                        }
                                        yield f"data: {json.dumps(start_data)}\n\n"
                                        video_start_sent = True
                                
                                    # Send chunk if it contains video_base64
                                    if avatar_update.get("video_base64"):
                                        chunk_data = {
                                            'type': 'avatar_video_chunk',
                                            'chunk': avatar_update.get('video_base64', ''),
                                            'chunk_size': avatar_update.get('chunk_size', 0),
                                            'is_last': avatar_update.get('is_last', False)
                                        }
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                
                                    # Send complete event if this is the completion event
                                    if avatar_update.get("is_complete"):
                                        avatar_duration = time.time() - avatar_start_time
                                        complete_data = {
                                            'type': 'avatar_video_complete',
                                            'format': 'mp4',
                                            'video_url': avatar_update.get('video_url'),
                                            'duration_seconds': round(avatar_duration, 2)
                                        }
                                        yield f"data: {json.dumps(complete_data)}\n\n"
                                else:
                                    # Non-streaming: chunk the base64 (fallback mode)
                                    video_base64 = avatar_update.get("video_base64", "")
                                    if video_base64:
                                        chunk_size = StreamingConfig.AVATAR_CHUNK_SIZE
                                        total_chunks = (len(video_base64) + chunk_size - 1) // chunk_size
                                    
                                        start_data = {
                                            'type': 'avatar_video_start',
                                            'total_chunks': total_chunks,
                                            'format': 'mp4',
                                            'video_url': avatar_update.get('video_url'),
                                            'size_bytes': avatar_update.get('size_bytes', 0),
                                            'delivery_method': 'base64_chunks'
                                        }
                                        yield f"data: {json.dumps(start_data)}\n\n"
                                    
                                        # Stream base64 in chunks
                                        for i in range(0, len(video_base64), chunk_size):
                                            chunk = video_base64[i:i + chunk_size]
                                            chunk_data = {
                                                'type': 'avatar_video_chunk',
                                                'chunk_index': i // chunk_size,
                                                'chunk': chunk,
                                                'is_last': (i + chunk_size) >= len(video_base64)
                                            }
                                            yield f"data: {json.dumps(chunk_data)}\n\n"
                                    
                                        avatar_duration = time.time() - avatar_start_time
                                        complete_data = {
                                            'type': 'avatar_video_complete',
                                            'format': 'mp4',
                                            'video_url': avatar_update.get('video_url'),
                                            'duration_seconds': round(avatar_duration, 2)
                                        }
                                        yield f"data: {json.dumps(complete_data)}\n\n"
                            elif update_type == "error":
                                error_data = {
                                    'type': 'avatar_error',
                                    'message': avatar_update.get('message'),
                                    'video_url': avatar_update.get('video_url'),  # Fallback URL if available
                                    'fallback_message': avatar_update.get('fallback_message')
                                }
                                yield f"data: {json.dumps(error_data)}\n\n"
                    except Exception as avatar_error:
                        log.exception(f"Error generating avatar video: {avatar_error}")
                        # Continue without avatar video - don't fail the interview
            
            # Save conversation and update state (non-blocking)
            asyncio.create_task(
                _save_interview_turn(
                    session_id,
                    interview_req,
                    question,
                    response_analysis,
                    conversation_context,
                    server_question_count + 1
                )
            )
            
            # Calculate metrics
            total_duration = time.time() - start_time
            
            # Log metrics (simple metrics - can be enhanced with proper metrics collector)
            log.info(
                f"Interview stream completed",
                extra={
                    "correlation_id": correlation_id,
                    "session_id": session_id,
                    "question_count": server_question_count + 1,
                    "chunk_count": chunk_count,
                    "duration_seconds": round(total_duration, 2),
                    "question_length": len(question),
                    "avatar_generated": generate_avatar
                }
            )
            
            # Send final event
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'question': question, 'question_count': server_question_count + 1})}\n\n"
        
        return StreamingResponse(
            stream_interview_response(),
            media_type="text/event-stream"
        )
        
    except Exception as exc:
        log.exception(
            f"Error in streaming interview agent: {exc}",
            extra={"correlation_id": correlation_id}
        )
        err_message = str(exc)

        async def stream_error():
            yield f"data: {json.dumps({'type': 'error', 'message': err_message, 'correlation_id': correlation_id})}\n\n"

        return StreamingResponse(
            stream_error(),
            media_type="text/event-stream"
        )


"""
History utilities for interview agent.

Provides functions for Q/A extraction, pairing, canonicalization, and conversation replay.
Pure parsing/structure operations with no side effects.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

log = logging.getLogger(__name__)


def get_previous_questions(conversation_history: List[Dict]) -> List[str]:
    """
    Extract all previous questions asked by the assistant from conversation history.
    
    Args:
        conversation_history: List of conversation messages
        
    Returns:
        List of previous questions (strings)
    """
    questions = []
    
    for message in conversation_history:
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if content and content.strip():
                # Check if it looks like a question
                if "?" in content or content.strip().endswith("?"):
                    questions.append(content.strip())
                # Also include statements that are clearly questions
                elif any(word in content.lower() for word in ["tell me", "can you", "what", "how", "why", "when", "where"]):
                    questions.append(content.strip())
    
    return questions


def pair_questions_with_answers(conversation_history: List[Dict]) -> List[Dict[str, Any]]:
    """
    Pair questions with their corresponding answers from conversation history.
    
    Args:
        conversation_history: List of conversation messages
        
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    pairs = []
    current_question = None
    
    for message in conversation_history:
        role = message.get("role", "")
        content = message.get("content", "").strip()
        
        if role == "assistant" and content:
            # This is a question
            if "?" in content or any(word in content.lower() for word in ["tell me", "can you", "what", "how", "why", "when", "where"]):
                current_question = content
        elif role == "user" and content and current_question:
            # This is an answer to the current question
            pairs.append({
                "question": current_question,
                "answer": content,
                "question_index": len(pairs)
            })
            current_question = None
    
    return pairs


def extract_qa_pairs(conversation_history: List[Dict]) -> List[Tuple[str, Optional[str]]]:
    """
    Extract question-answer pairs as tuples.
    
    Args:
        conversation_history: List of conversation messages
        
    Returns:
        List of (question, answer) tuples. Answer may be None if question hasn't been answered yet.
    """
    pairs = []
    current_question = None
    
    for message in conversation_history:
        role = message.get("role", "")
        content = message.get("content", "").strip()
        
        if role == "assistant" and content:
            # Check if it's a question
            if "?" in content or any(word in content.lower() for word in ["tell me", "can you", "what", "how", "why", "when", "where"]):
                # If we have a pending question without an answer, add it with None
                if current_question:
                    pairs.append((current_question, None))
                current_question = content
        elif role == "user" and content:
            # This is an answer
            if current_question:
                pairs.append((current_question, content))
                current_question = None
            # If there's no current question, this might be an unsolicited response
            # We'll skip it for now
    
    # Add final question if it hasn't been answered
    if current_question:
        pairs.append((current_question, None))
    
    return pairs


def canonicalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonicalize a message to a standard format.
    
    Args:
        message: Message dict with potentially varying structure
        
    Returns:
        Canonicalized message with 'role' and 'content' keys
    """
    canonical = {
        "role": message.get("role", "").lower(),
        "content": message.get("content", "").strip()
    }
    
    # Ensure role is one of the standard values
    if canonical["role"] not in ["system", "user", "assistant"]:
        # Try to infer from other fields
        if "question" in message or "assistant" in str(message).lower():
            canonical["role"] = "assistant"
        elif "answer" in message or "user" in str(message).lower():
            canonical["role"] = "user"
        else:
            canonical["role"] = "user"  # Default
    
    return canonical


def canonicalize_conversation_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Canonicalize an entire conversation history.
    
    Args:
        history: List of message dicts
        
    Returns:
        List of canonicalized message dicts
    """
    return [canonicalize_message(msg) for msg in history]


def replay_conversation(conversation_history: List[Dict], start_index: int = 0, end_index: Optional[int] = None) -> List[Dict]:
    """
    Replay a portion of conversation history.
    
    Args:
        conversation_history: Full conversation history
        start_index: Starting index (inclusive)
        end_index: Ending index (exclusive, None = to end)
        
    Returns:
        Subset of conversation history
    """
    if end_index is None:
        return conversation_history[start_index:]
    return conversation_history[start_index:end_index]


def get_last_n_turns(conversation_history: List[Dict], n: int = 1) -> List[Dict]:
    """
    Get the last N conversation turns (each turn = user + assistant message).
    
    Args:
        conversation_history: Full conversation history
        n: Number of turns to retrieve
        
    Returns:
        Last N turns of conversation
    """
    # Filter out system messages for turn counting
    non_system = [msg for msg in conversation_history if msg.get("role") != "system"]
    
    # Each turn is typically 2 messages (user + assistant), but could be more
    # We'll take the last 2*n messages as a safe approximation
    messages_per_turn = 2
    num_messages = n * messages_per_turn
    
    return non_system[-num_messages:] if len(non_system) > num_messages else non_system


def count_questions_in_history(conversation_history: List[Dict]) -> int:
    """
    Count the number of questions in conversation history.
    
    Args:
        conversation_history: List of conversation messages
        
    Returns:
        Number of questions asked
    """
    questions = get_previous_questions(conversation_history)
    return len(questions)


def get_last_question(conversation_history: List[Dict]) -> Optional[str]:
    """
    Get the last question asked in the conversation.
    
    Args:
        conversation_history: List of conversation messages
        
    Returns:
        Last question string, or None if no questions found
    """
    questions = get_previous_questions(conversation_history)
    return questions[-1] if questions else None


def get_last_answer(conversation_history: List[Dict]) -> Optional[str]:
    """
    Get the last answer provided by the user.
    
    Args:
        conversation_history: List of conversation messages
        
    Returns:
        Last answer string, or None if no answers found
    """
    for message in reversed(conversation_history):
        if message.get("role") == "user":
            content = message.get("content", "").strip()
            if content:
                return content
    return None


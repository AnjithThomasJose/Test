"""
Compression utilities for interview agent.

Provides functions for truncation, summarization, and token-aware compression
of conversation history. May call LLM via llm_utils for summarization.
"""

import logging
from typing import Dict, List, Any, Optional

log = logging.getLogger(__name__)

# Maximum number of recent turns to keep in conversation history
MAX_HISTORY_TURNS = 4


def compress_conversation_history(
    history: List[Dict],
    max_recent_turns: int = None,
    max_message_length: int = 300
) -> List[Dict]:
    """
    Compress conversation history for LLM prompts to prevent irrelevant questions.
    
    Strategy:
    - Keep first system message (if exists) for context
    - Keep last N turns (recent context) with full detail
    - Truncate long messages to prevent token bloat
    - Remove excessive system messages
    
    Args:
        history: Full conversation history
        max_recent_turns: Maximum number of recent turns to keep (default: MAX_HISTORY_TURNS)
        max_message_length: Maximum length per message before truncation
        
    Returns:
        Compressed conversation history suitable for LLM prompts
    """
    if max_recent_turns is None:
        max_recent_turns = MAX_HISTORY_TURNS
    
    if not history:
        return []
    
    # If history is short enough, just truncate long messages
    if len(history) <= (max_recent_turns * 2 + 1):  # +1 for system message
        compressed = []
        for msg in history:
            compressed_msg = msg.copy()
            content = compressed_msg.get("content", "")
            if len(content) > max_message_length:
                compressed_msg["content"] = content[:max_message_length] + "..."
            compressed.append(compressed_msg)
        return compressed
    
    # Separate system messages from conversation
    system_messages = []
    conversation = []
    
    for msg in history:
        if msg.get("role") == "system":
            system_messages.append(msg)
        else:
            conversation.append(msg)
    
    # Keep only the first system message
    compressed = system_messages[:1] if system_messages else []
    
    # Keep last N turns (each turn = user + assistant message)
    # Calculate how many messages to keep (N turns * 2 messages per turn)
    messages_to_keep = max_recent_turns * 2
    recent_messages = conversation[-messages_to_keep:] if len(conversation) > messages_to_keep else conversation
    
    # Truncate long messages
    for msg in recent_messages:
        compressed_msg = msg.copy()
        content = compressed_msg.get("content", "")
        if len(content) > max_message_length:
            compressed_msg["content"] = content[:max_message_length] + "..."
        compressed.append(compressed_msg)
    
    return compressed


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count for text (rough approximation).
    
    Args:
        text: Text to estimate tokens for
        chars_per_token: Average characters per token (default: 4.0)
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return int(len(text) / chars_per_token)


def truncate_to_token_limit(text: str, max_tokens: int, chars_per_token: float = 4.0) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        chars_per_token: Average characters per token (default: 4.0)
        
    Returns:
        Truncated text
    """
    if not text:
        return text
    
    max_chars = int(max_tokens * chars_per_token)
    if len(text) <= max_chars:
        return text
    
    # Truncate and add ellipsis
    return text[:max_chars - 3] + "..."


async def summarize_conversation_chunk(
    messages: List[Dict],
    max_tokens: int = 200,
    agent_name: str = "interview_agent_compression"
) -> str:
    """
    Summarize a chunk of conversation using LLM.
    
    Args:
        messages: List of messages to summarize
        max_tokens: Maximum tokens for summary
        agent_name: Agent name for LLM call
        
    Returns:
        Summarized text
    """
    from .llm_utils import invoke_llm
    
    if not messages:
        return ""
    
    # Build conversation text
    conversation_text = "\n".join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in messages
    ])
    
    prompt = f"""Summarize this conversation excerpt in {max_tokens} tokens or less, preserving key information:

{conversation_text}

Summary:"""
    
    try:
        summary = await invoke_llm(
            prompt=prompt,
            agent_name=agent_name,
            max_retries=1,
            timeout=10.0
        )
        
        if isinstance(summary, str):
            return summary.strip()
        return str(summary).strip()
        
    except Exception as e:
        log.warning(f"Failed to summarize conversation chunk: {e}")
        # Fallback to simple truncation
        return truncate_to_token_limit(conversation_text, max_tokens)


def chunk_conversation_history(
    history: List[Dict],
    chunk_size: int = 10
) -> List[List[Dict]]:
    """
    Split conversation history into chunks.
    
    Args:
        history: Full conversation history
        chunk_size: Number of messages per chunk
        
    Returns:
        List of message chunks
    """
    chunks = []
    for i in range(0, len(history), chunk_size):
        chunks.append(history[i:i + chunk_size])
    return chunks


async def compress_with_summarization(
    history: List[Dict],
    max_recent_turns: int = 4,
    summarize_older: bool = True,
    max_summary_tokens: int = 200
) -> List[Dict]:
    """
    Compress conversation history with LLM-based summarization for older messages.
    
    Args:
        history: Full conversation history
        max_recent_turns: Number of recent turns to keep in full
        summarize_older: Whether to summarize older messages
        max_summary_tokens: Maximum tokens for summary
        
    Returns:
        Compressed conversation history
    """
    if not history:
        return []
    
    # Separate system messages
    system_messages = [msg for msg in history if msg.get("role") == "system"]
    conversation = [msg for msg in history if msg.get("role") != "system"]
    
    # Keep recent turns in full
    messages_per_turn = 2
    recent_messages_count = max_recent_turns * messages_per_turn
    recent_messages = conversation[-recent_messages_count:] if len(conversation) > recent_messages_count else conversation
    older_messages = conversation[:-recent_messages_count] if len(conversation) > recent_messages_count else []
    
    compressed = system_messages[:1] if system_messages else []
    
    # Summarize older messages if requested
    if summarize_older and older_messages:
        try:
            summary_text = await summarize_conversation_chunk(
                older_messages,
                max_tokens=max_summary_tokens
            )
            if summary_text:
                compressed.append({
                    "role": "system",
                    "content": f"[Previous conversation summary: {summary_text}]"
                })
        except Exception as e:
            log.warning(f"Failed to summarize older messages: {e}, keeping truncated version")
            # Fallback to simple truncation
            for msg in older_messages[-2:]:  # Keep last 2 older messages
                compressed_msg = msg.copy()
                content = compressed_msg.get("content", "")
                if len(content) > 150:
                    compressed_msg["content"] = content[:150] + "..."
                compressed.append(compressed_msg)
    
    # Add recent messages
    compressed.extend(recent_messages)
    
    return compressed


def rehydrate_compressed_history(compressed: List[Dict]) -> List[Dict]:
    """
    Rehydrate compressed history (placeholder for future expansion).
    
    Currently just returns as-is, but could be extended to expand summaries.
    
    Args:
        compressed: Compressed conversation history
        
    Returns:
        Rehydrated conversation history
    """
    # For now, just return as-is
    # Future: Could expand summaries back into full messages if needed
    return compressed


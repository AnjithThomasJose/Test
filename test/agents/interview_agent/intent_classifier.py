"""
Hybrid intent classifier: fast pattern-based + robust LLM fallback.
This version includes:
- Fast pattern-based detection (microseconds) for obvious cases
- LLM-based fallback for ambiguous/edge cases (robust, scalable)
- Massive keyword/tech-domain guardrails to suppress false topic switches
- Soft semantic matching without embeddings
- Multi-intent handling with intensity-based confidence
- Advanced exit intent detection with false positive prevention:
  * Continuation indicator detection (e.g., "wrap up my explanation" ≠ exit)
  * Negation detection (e.g., "don't want to wrap up" ≠ exit)
  * Position-aware phrase matching
  * LLM context-aware detection for variations not in hardcoded list
"""

import logging
import re
import asyncio
from typing import Any, Dict, Set, Optional, List

log = logging.getLogger(__name__)

SAFE_DEFAULT = {
    "topic_switch": False,
    "new_topic": "",
    "exit_interview": False,
    "refusal": False,
    "uncertainty": False,
    "irrelevant": False,
    "meta_request": "",
    "confidence": 0.0,
}

_WORD_RE = re.compile(r"[A-Za-z0-9+#]+")

# ---------------------------------------------------------------------
# 1. HARD BLOCKLIST — prevents false topic-switch triggers
# ---------------------------------------------------------------------

TECH_DOMAIN_WORDS = {
    # Programming languages
    "java", "python", "golang", "go", "rust", "c++", "c", "typescript",
    "javascript", "js", "ts", "ruby", "php", "swift", "kotlin",

    # Backend systems
    "microservices", "architecture", "backend", "server", "api", "rest",
    "grpc", "scalability", "performance",

    # DevOps & infra
    "docker", "kubernetes", "k8s", "terraform", "ansible",
    "aws", "gcp", "azure", "devops", "ci", "cd", "linux",

    # Data & ML
    "ml", "machine learning", "ai", "nn", "cnn", "rnn", "lstm",
    "transformers", "data", "pandas", "numpy", "sql", "nosql",
    "postgres", "mysql", "mongodb", "redis", "elasticsearch",

    # Concurrency & distributed
    "threads", "threading", "concurrency", "parallelism", "distributed",
    "kafka", "message queues", "queues", "brokers",

    # Web/frontend
    "react", "redux", "nextjs", "angular", "vue", "frontend",

    # System design words
    "load balancing", "sharding", "replication", "indexing",
    "caching", "design", "system design"
}

# Words that indicate the candidate is referring to a technical detail,
# NOT issuing a topic-switch instruction.
TECH_CONTEXT_INDICATORS = {
    "in my experience", "i used", "i worked with", "i know", "i have experience",
    "i implemented", "i built", "i designed", "i handled",
}

# ---------------------------------------------------------------------
# 2. PATTERNS
# ---------------------------------------------------------------------
TOPIC_SWITCH_PATTERNS = [
    r"(?:switch|change|shift|move)\s+(?:the\s+)?(?:topic|conversation)\s*(?:to\s+)?(?P<topic>[A-Za-z0-9 .+#-]+)",
    r"(?:talk|discuss)\s+(?:about\s+)?(?P<topic>[A-Za-z0-9 .+#-]+)",
    r"(?:can\s+we|let's|lets)\s+(?:talk|discuss)\s+(?:about\s+)?(?P<topic>[A-Za-z0-9 .+#-]+)",
    r"(?:prefer|want)\s+(?:to\s+)?(?:talk|discuss)\s+(?:about\s+)?(?P<topic>[A-Za-z0-9 .+#-]+)",
]

# Explicit topic switch request patterns - these should ALWAYS trigger topic_switch
# even if the requested topic is a tech domain word
# Note: Include both straight apostrophe (') and curly apostrophe (') variants
EXPLICIT_TOPIC_SWITCH_PHRASES = {
    "can we talk about",
    "can we discuss",
    "let's talk about",
    "let's talk about",  # curly apostrophe variant
    "lets talk about",
    "i'd prefer to discuss",
    "i'd prefer to discuss",  # curly apostrophe variant
    "id prefer to discuss",
    "i would prefer to discuss",
    "i'd rather discuss",
    "i'd rather discuss",  # curly apostrophe variant
    "id rather discuss",
    "i want to talk about",
    "i'd like to discuss",
    "i'd like to discuss",  # curly apostrophe variant
    "id like to discuss",
    "can we switch to",
    "let's switch to",
    "let's switch to",  # curly apostrophe variant
    "lets switch to",
    "instead of this",
    "talk about something else",
    "discuss something else",
    "different topic",
    "change the topic",
    "change topic",
    "prefer to discuss",
    "rather discuss",
    "switch to discussing",
    "i prefer",
    "i'd prefer",
    "i'd prefer",  # curly apostrophe variant
}

EXIT_PHRASES = {
    # Direct exit requests
    "end interview", "end the interview", "end this interview",
    "stop the interview", "stop this interview", "stop interview",
    "quit", "leave", "exit",
    "bye", "goodbye", "good bye",
    "finish", "finish the interview", "finish this interview",
    "no more questions", "no more",
    
    # Wrap up variations
    "wrap up", "wrap up this interview", "wrap up the interview",
    "let's wrap up", "lets wrap up", "can we wrap up",
    "i need to wrap up", "i have to wrap up", "i should wrap up",
    "please wrap up", "wrap this up",
    
    # Conclude variations
    "conclude", "conclude the interview", "conclude this interview",
    
    # Let's end variations
    "let's end", "lets end", "let's end this", "lets end this",
    "let's end the interview", "lets end the interview",
    "let's end this interview", "lets end this interview",
    "can we end", "can we end this", "please end", "please end this",
    "i want to end", "i want to end this", "i'd like to end",
    
    # Done/finished variations
    "i'm done", "im done", "i am done",
    "i'm finished", "im finished", "i am finished",
    "done here", "finished here", "we're done", "were done",
    
    # That's all variations
    "that's all", "thats all", "that's it", "thats it",
    "that is all", "that will be all", "nothing more",
    
    # Polite exit patterns
    "thank you but", "thanks but",
    "thank you", "thanks",  # Standalone thank you is often a polite exit signal
    
    # Need to go/leave variations
    "i need to go", "i have to go", "i should go", "i gotta go",
    "i'm leaving", "im leaving", "i need to leave", "i have to leave",
    "i must go", "got to go", "have to go now",
    
    # Stop/end as standalone (careful with context)
    "stop", "end this", "end it",
}

# Phrases that might look like exit but are actually continuation
# (e.g., "I want to wrap up my explanation" = continue, not exit)
CONTINUATION_INDICATORS = {
    "wrap up my", "wrap up the", "finish my", "finish the",
    "conclude my", "conclude the", "end my", "end the",
    "done with my", "done with the",
}

# Negation words that indicate the user does NOT want to exit
NEGATION_WORDS = {
    "don't", "dont", "do not", "cannot", "can't", "cant",
    "won't", "wont", "will not", "not", "never", "no",
}

REFUSAL_PHRASES = {
    # Don't know variations
    "don't know", "do not know", "dont know", "i don't know",
    "i do not know", "no idea", "have no idea",
    
    # Can't/won't answer variations
    "cant answer", "can't answer", "cannot answer",
    "won't answer", "wont answer", "will not answer",
    "don't want to answer", "dont want to answer",
    "i don't want to answer", "i dont want to answer",
    "rather not answer", "i'd rather not", "id rather not",
    "prefer not to answer", "prefer not to say",
    
    # Skip variations
    "skip", "skip this", "skip this question", "next question",
    "can we skip", "let's skip", "lets skip",
    
    # Not sure variations
    "not sure", "i'm not sure", "im not sure",
    "uncertain", "unsure",
    
    # Refusal phrases
    "i refuse", "refuse to answer", "no comment", "pass",
    "rather not", "i'd rather not discuss", "don't want to discuss",
}

UNCERTAINTY_WORDS = {"maybe", "possibly", "perhaps", "unsure", "uncertain"}

META_REQUEST_MAP = {
    "repeat": {"repeat", "say again"},
    "simplify": {"simplify", "make it simple", "easier"},
    "slower": {"slow down", "slower"},
    "faster": {"faster", "speed up", "go faster"},
}

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "for", "on",
    "about", "can", "we", "let", "lets", "please", "you", "i",
}

# PERFORMANCE: Cache sorted exit phrases (they don't change, so compute once)
# Sort by length (longest first) to match "wrap up this interview" before "wrap up"
_SORTED_EXIT_PHRASES = tuple(sorted(EXIT_PHRASES, key=len, reverse=True))


def _tokenize(text: str) -> Set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)} if text else set()


def _semantic_overlap(message_words: Set[str], topic_words: Set[str]) -> float:
    if not message_words or not topic_words:
        return 0.0
    return len(message_words & topic_words) / len(topic_words)


# ---------------------------------------------------------------------
# MAIN CLASSIFIER (STRONG VERSION)
# ---------------------------------------------------------------------
def detect_intents(message: str, interview_topic: str = "") -> Dict[str, Any]:
    """
    Detect user intents from a message, with focus on exit intent detection.
    
    This function uses pattern matching to detect various user intents including:
    - Exit interview intent (wrap up, end, stop, etc.)
    - Topic switch requests
    - Refusal to answer
    - Uncertainty
    - Meta requests (repeat, simplify, etc.)
    
    The exit intent detection includes sophisticated false positive prevention:
    - Continuation indicators (e.g., "wrap up my explanation" is NOT an exit)
    - Negation detection (e.g., "don't want to wrap up" is NOT an exit)
    - Position-aware matching to avoid partial matches
    
    Args:
        message: User's message text
        interview_topic: Current interview topic (optional, for context)
    
    Returns:
        Dict with detected intents and confidence scores:
        {
            "exit_interview": bool,
            "topic_switch": bool,
            "refusal": bool,
            "uncertainty": bool,
            "meta_request": str,
            "confidence": float,
            ...
        }
    """
    result = SAFE_DEFAULT.copy()

    if not message or not message.strip():
        return result

    text = message.strip()
    text_lower = text.lower()
    confidence = 0.2

    # -------------------------------------------------------------
    # EXIT INTENT
    # Improved matching: Check for exit phrases with smart pattern matching
    # Includes false positive prevention via continuation indicators and negation detection
    # -------------------------------------------------------------
    
    # First check for continuation indicators to avoid false positives
    # (e.g., "I want to wrap up my explanation" should NOT trigger exit)
    has_continuation_indicator = any(ind in text_lower for ind in CONTINUATION_INDICATORS)
    
    # Strategy: Check multi-word phrases first (more specific), then single words
    # Sort phrases by length (longest first) to match "wrap up this interview" before "wrap up"
    # PERFORMANCE: Use pre-sorted cached phrases instead of sorting on each call
    sorted_phrases = _SORTED_EXIT_PHRASES
    
    for phrase in sorted_phrases:
        phrase_lower = phrase.lower()
        phrase_pos = text_lower.find(phrase_lower)
        
        if phrase_pos < 0:
            continue  # Phrase not found, skip to next
        
        # CRITICAL: Check if this phrase is part of a continuation indicator
        # e.g., if phrase is "wrap up" and text contains "wrap up my", block it
        phrase_blocked_by_continuation = False
        for cont_ind in CONTINUATION_INDICATORS:
            if phrase_lower in cont_ind:  # The exit phrase is part of a continuation indicator
                cont_pos = text_lower.find(cont_ind)
                if cont_pos >= 0:
                    # Check if the exit phrase position matches the start of the continuation indicator
                    # e.g., "wrap up" at position 10 and "wrap up my" at position 10 means they overlap
                    if phrase_pos == cont_pos:
                        # The exit phrase starts at the same position as continuation indicator - block it
                        phrase_blocked_by_continuation = True
                        log.debug(f"Exit intent blocked: '{phrase}' is part of continuation indicator '{cont_ind}' in '{text[:50]}...'")
                        break
                    # Also check if continuation indicator appears right after the phrase
                    # e.g., "wrap up" at 10, "wrap up my" at 10 means " my" comes right after
                    elif phrase_pos < cont_pos and cont_pos <= phrase_pos + len(phrase_lower) + 3:
                        # Continuation indicator appears right after or overlaps - block it
                        phrase_blocked_by_continuation = True
                        log.debug(f"Exit intent blocked: '{phrase}' followed by continuation indicator '{cont_ind}' in '{text[:50]}...'")
                        break
        
        if phrase_blocked_by_continuation:
            continue
        
        # Check for continuation indicators that appear before the phrase
        # (e.g., "my explanation" before "wrap up" in "i want to wrap up my explanation")
        if has_continuation_indicator:
            continuation_positions = [text_lower.find(ind) for ind in CONTINUATION_INDICATORS if ind in text_lower]
            if continuation_positions:
                continuation_pos = min(continuation_positions)
                if continuation_pos >= 0 and continuation_pos < phrase_pos:
                    # Continuation indicator comes before exit phrase - likely false positive
                    log.debug(f"Exit intent blocked by continuation indicator before phrase: '{phrase}' in '{text[:50]}...'")
                    continue
        
        # Check for negation before the phrase (e.g., "I don't want to wrap up")
        if phrase_pos > 0:
            before_phrase = text_lower[:phrase_pos].strip()
            # Check last 5 words before the phrase for negation (increased from 4)
            words_before = before_phrase.split()[-5:] if before_phrase else []
            has_negation = any(neg in words_before for neg in NEGATION_WORDS)
            
            if has_negation:
                log.debug(f"Exit intent blocked by negation: '{phrase}' in '{text[:50]}...'")
                continue
        
        # For multi-word phrases, use substring matching (they're already specific)
        if len(phrase.split()) > 1:
            result["exit_interview"] = True
            result["confidence"] = 0.95
            log.debug(f"Exit intent detected (multi-word): phrase '{phrase}' found in '{text[:50]}...'")
            return result
        else:
            # For single words, use word boundary matching to avoid false positives
            # e.g., "stop" should match "stop" but not "stopwatch"
            pattern = r'\b' + re.escape(phrase_lower) + r'\b'
            if re.search(pattern, text_lower):
                result["exit_interview"] = True
                result["confidence"] = 0.95
                log.debug(f"Exit intent detected (word boundary): phrase '{phrase}' found in '{text[:50]}...'")
                return result

    # -------------------------------------------------------------
    # REFUSAL
    # -------------------------------------------------------------
    if any(p in text_lower for p in REFUSAL_PHRASES):
        result["refusal"] = True
        result["confidence"] = 0.8
        return result

    # -------------------------------------------------------------
    # META REQUEST
    # -------------------------------------------------------------
    for meta, phrases in META_REQUEST_MAP.items():
        if any(p in text_lower for p in phrases):
            result["meta_request"] = meta
            result["confidence"] = 0.8
            return result

    # -------------------------------------------------------------
    # UNCERTAINTY
    # -------------------------------------------------------------
    if any(w in text_lower for w in UNCERTAINTY_WORDS):
        result["uncertainty"] = True
        confidence = 0.5

    # -------------------------------------------------------------
    # TOPIC SWITCH (strong rules)
    # -------------------------------------------------------------

    # 0. FIRST: Check for explicit topic switch phrases
    # These ALWAYS trigger topic_switch, even for tech domain topics
    # e.g., "Can we talk about JavaScript instead?" should be blocked
    for explicit_phrase in EXPLICIT_TOPIC_SWITCH_PHRASES:
        if explicit_phrase in text_lower:
            result["topic_switch"] = True
            result["new_topic"] = ""  # We don't allow the switch anyway
            result["confidence"] = 0.95
            log.debug(f"Explicit topic switch detected: '{explicit_phrase}' in '{text[:50]}...'")
            return result

    # 1. If message starts with technical-context indicators → NOT a topic switch
    if any(text_lower.startswith(prefix) for prefix in TECH_CONTEXT_INDICATORS):
        pass  # skip topic switch logic entirely

    else:
        # 2. Regex match for topic switch
        for pattern in TOPIC_SWITCH_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                candidate = (match.group("topic") or "").strip(" .?!")

                # 3. If candidate contains technical-domain keywords → NOT topic switch
                # (This handles cases where user mentions a tech in their answer,
                # not explicitly requesting to switch topics)
                cand_words = _tokenize(candidate)
                if cand_words & TECH_DOMAIN_WORDS:
                    continue  # skip false positives entirely

                # 4. Allow topic-switch only for human-adjacent topics (non-technical)
                if len(candidate.split()) <= 4:
                    result["topic_switch"] = True
                    result["new_topic"] = candidate
                    confidence = max(confidence, 0.9)
                break

    # -------------------------------------------------------------
    # IRRELEVANT DETECTION
    # -------------------------------------------------------------
    msg_words = _tokenize(text) - STOPWORDS
    topic_words = _tokenize(interview_topic)

    overlap = _semantic_overlap(msg_words, topic_words)

    if topic_words and overlap < 0.08:
        result["irrelevant"] = True
        confidence = max(confidence, 0.6)

    result["confidence"] = round(confidence, 3)
    return result


# ---------------------------------------------------------------------
# LLM-BASED EXIT INTENT DETECTION (ROBUST FALLBACK)
# ---------------------------------------------------------------------

async def detect_exit_intent_llm(
    message: str,
    conversation_history: Optional[List] = None,
    interview_topic: str = ""
) -> Dict[str, Any]:
    """
    Use LLM to detect if user wants to exit the interview.
    
    This is a robust fallback for cases where pattern matching is uncertain.
    The LLM can understand context, nuance, and variations we haven't hardcoded.
    
    Args:
        message: User's message text
        conversation_history: Optional recent conversation history for context
        interview_topic: Current interview topic
    
    Returns:
        Dict with:
        - exit_interview: bool
        - confidence: float (0-1)
        - reason: str (optional explanation)
    """
    if not message or not message.strip():
        return {"exit_interview": False, "confidence": 0.0}
    
    try:
        from . import llm_utils
        
        # Build context from recent conversation
        context_str = ""
        if conversation_history:
            recent = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
            context_str = "\nRecent conversation:\n" + "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')[:100]}"
                for msg in recent
            ])
        
        prompt = f"""Determine if the user wants to END or EXIT the interview conversation.

USER MESSAGE: "{message.strip()}"

{context_str}

INTERVIEW TOPIC: {interview_topic or "General"}

EXIT SIGNALS (user wants to end):
- Saying goodbye, thank you, or expressing gratitude (especially standalone)
- Asking to wrap up, finish, conclude, or end the interview
- Mentioning they need to go, have other commitments, or are done
- Short polite responses like "thank you", "thanks", "that's all", "i'm done"
- Any indication they want to stop the conversation

NOT EXIT SIGNALS (user wants to continue):
- "Thank you for..." followed by more content (gratitude within answer)
- "I want to wrap up my explanation" (completing a thought, not ending)
- Questions or requests for clarification
- Technical answers that happen to contain exit-related words

Consider the CONTEXT:
- If the message is very short (1-3 words) and contains gratitude, it's likely an exit
- If the message is part of a longer answer, it's likely NOT an exit
- If the user just answered a question and says "thank you", it's likely an exit

Return STRICT JSON only:
{{
    "exit_interview": true|false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation of why this is/isn't an exit signal"
}}"""
        
        messages = [
            {"role": "system", "content": "You are an expert at detecting when users want to end conversations. Be precise and context-aware."},
            {"role": "user", "content": prompt}
        ]
        
        result = await llm_utils.invoke_llm(
            messages,
            model="gemini-2.5-flash",  # Fast model for quick detection
            enforce_json=True,
            max_tokens=100,  # Short response needed
            temperature=0.1,  # Low temperature for consistent detection
            max_retries=1
        )
        
        if result.get("ok") and isinstance(result.get("json"), dict):
            llm_result = result["json"]
            exit_detected = llm_result.get("exit_interview", False)
            confidence = float(llm_result.get("confidence", 0.5))
            reason = llm_result.get("reason", "")
            
            log.debug(f"LLM exit intent detection: exit={exit_detected}, confidence={confidence}, reason={reason}")
            
            return {
                "exit_interview": exit_detected,
                "confidence": confidence,
                "reason": reason
            }
        else:
            log.warning(f"LLM exit intent detection failed: {result.get('error', 'Unknown error')}")
            return {"exit_interview": False, "confidence": 0.0}
            
    except Exception as e:
        log.warning(f"LLM exit intent detection error: {e}", exc_info=True)
        return {"exit_interview": False, "confidence": 0.0}


async def detect_intents_with_llm_fallback(
    message: str,
    interview_topic: str = "",
    conversation_history: Optional[List] = None,
    use_llm_for_ambiguous: bool = True
) -> Dict[str, Any]:
    """
    Detect user intents with LLM fallback for ambiguous cases.
    
    This hybrid approach:
    1. First tries fast pattern-based detection (instant)
    2. If pattern-based is uncertain or doesn't match, uses LLM (robust)
    
    Args:
        message: User's message text
        interview_topic: Current interview topic
        conversation_history: Optional conversation history for LLM context
        use_llm_for_ambiguous: Whether to use LLM when pattern matching is uncertain
    
    Returns:
        Dict with detected intents (same format as detect_intents)
    """
    # Step 1: Fast pattern-based detection
    pattern_result = detect_intents(message, interview_topic)
    
    # Step 2: If high-confidence intents detected (exit, topic_switch, refusal), return immediately
    if pattern_result.get("confidence", 0) >= 0.8:
        if (pattern_result.get("exit_interview") or 
            pattern_result.get("topic_switch") or 
            pattern_result.get("refusal")):
            return pattern_result
    
    # Step 3: If exit intent was detected with low confidence, or not detected but message is ambiguous,
    # use LLM for more robust detection
    if use_llm_for_ambiguous:
        # Check if message might be an exit signal but wasn't caught by patterns
        message_lower = message.lower().strip()
        is_ambiguous = (
            # Short messages that might be exit signals
            len(message_lower.split()) <= 5 or
            # Contains gratitude words but pattern didn't catch it
            any(word in message_lower for word in ["thank", "thanks", "appreciate", "grateful"]) or
            # Pattern detected exit but with low confidence
            (pattern_result.get("exit_interview") and pattern_result.get("confidence", 0) < 0.8)
        )
        
        if is_ambiguous:
            log.debug(f"Using LLM fallback for ambiguous exit intent detection: '{message[:50]}...'")
            llm_result = await detect_exit_intent_llm(message, conversation_history, interview_topic)
            
            # If LLM detected exit with reasonable confidence, use it
            if llm_result.get("exit_interview") and llm_result.get("confidence", 0) >= 0.7:
                pattern_result["exit_interview"] = True
                pattern_result["confidence"] = llm_result.get("confidence", 0.8)
                log.info(f"LLM confirmed exit intent: '{message[:50]}...' (confidence: {llm_result.get('confidence')})")
            elif not pattern_result.get("exit_interview") and llm_result.get("exit_interview"):
                # LLM detected exit but pattern didn't - trust LLM
                pattern_result["exit_interview"] = True
                pattern_result["confidence"] = llm_result.get("confidence", 0.8)
                log.info(f"LLM detected exit intent missed by patterns: '{message[:50]}...'")
    
    return pattern_result


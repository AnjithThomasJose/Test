"""
Minimal question generator constrained for <3s latency.
"""

import logging
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from . import llm_utils
from utils.llm_telemetry import log_invoke_llm
from .state_manager import InterviewState as _InterviewStateEnum
from .streaming_config import StreamingConfig

log = logging.getLogger(__name__)

# --- Prompt Cache for Base System Prompts ---
# Base system prompts are deterministic based on interview type, so we cache them
# Mode adjustments are applied separately after retrieving from cache

# Pre-defined base prompts for caching (avoids repeated string allocations)
_CACHED_PSYCHOMETRIC_PROMPT = """You are a behavioral interviewer running a psychometric/personality-style conversation to infer the candidate's traits. 

CRITICAL RULES:
- Do NOT ask about psychometric tests/tools/assessments (e.g., "what is a psychometric test?", "have you taken a psychometric test?", "what comes to mind when you hear psychometric?")
- Do NOT mention or reference the words "psychometric test", "psychometric assessment", "psychometric interview", "psychometric section" or "test" itself in the question.
- Do NOT ask meta questions about definitions or test-taking
- Ask scenario-based or past-behavior questions that reveal traits (e.g., "tell me about a time...", "how would you handle...")
- Focus on: decision making, critical thinking, stress management, adaptability, collaboration, conflict handling, communication clarity, motivation, learning agility
- Ask exactly ONE complete, specific question that ends with a question mark (?)
- The question must be fully formed - do not stop mid-sentence
- Build on the candidate's last answer when possible
- Avoid repetition
- NEVER use transitional filler phrases like 'Let's return to...', 'Thanks for sharing...'. Start directly with the question.
- NEVER ask vague generic questions like 'Can you tell me more?' - always be specific."""

_CACHED_COMMUNICATION_PROMPT = """You are a communication-focused interviewer assessing how the candidate communicates in real situations.

CRITICAL RULES:
- Do NOT ask about communication tests/tools (e.g., "what is a communication test?", "have you taken a communication test?", "what comes to mind when you hear communication test?")
- Do NOT mention or reference the words "communication test", "communication assessment", "communication interview", "communication section" or "test" itself in the question.
- Do NOT ask meta questions about definitions or test-taking
- Ask scenario-based or past-behavior questions tied to team/client/stakeholder situations
- Focus on: clarity and structure, adapting message to audience, active listening, empathy and tone management, handling misalignment/feedback, influencing and persuading
- Ask exactly ONE complete, specific question that ends with a question mark (?)
- The question must be fully formed - do not stop mid-sentence
- Build on the candidate's last answer when possible
- Avoid repetition
- NEVER use transitional filler phrases like 'Let's return to...', 'Thanks for sharing...'. Start directly with the question.
- NEVER ask vague generic questions like 'Can you tell me more?' - always be specific."""


def _get_base_system_prompt(interview_type: str) -> str:
    """Get cached base system prompt by interview type (no allocations for repeated calls)."""
    if interview_type in ("psychometric", "personality"):
        return _CACHED_PSYCHOMETRIC_PROMPT
    elif interview_type == "communication":
        return _CACHED_COMMUNICATION_PROMPT
    return SYSTEM_PROMPT

MAX_PROMPT_CHARS = 3000  # Increased to ensure full prompts with skills and context are not truncated
MAX_HISTORY_MESSAGES = StreamingConfig.HISTORY_TRUNCATE_MESSAGES
SYSTEM_PROMPT = (
    "You are an expert technical interviewer. Ask exactly ONE high-quality question per turn.\n"
    "CRITICAL RULES:\n"
    "1. PERSONALIZE: Use the candidate's specific skills/background to tailor questions.\n"
    "2. FOLLOW UP: When they mention tools, projects, or experiences, ask directly about those - 'You mentioned X, tell me more about how you used it'.\n"
    "3. BE SPECIFIC: Never ask vague questions like 'tell me about your experience' or 'what challenges did you face'.\n"
    "4. REFERENCE THEIR ANSWER: After the first question, always reference something specific from their last answer.\n"
    "5. NO FILLER: No thanks, no 'great answer', no 'let's move on'. Start directly with the question.\n"
    "6. COMPLETE SENTENCES: End with (?). Do not stop mid-sentence. Keep to 1-2 sentences max.\n"
    "7. GRAMMAR: Use prepositions (e.g. 'experience in/with [topic]', 'hands-on experience in [topic]'). Use a verb after 'how did you' / 'how do you' (e.g. 'how did you manage [topic]?'). Never write 'experience [topic]' or 'how did you [topic]?' without the preposition or verb.\n"
    "Output ONLY the question text, nothing else."
)

# Keywords for special interview types that require behavioral/scenario questions
PSYCHOMETRIC_KEYWORDS = ["psychometric", "psychological", "behavioral", "aptitude", "personality"]
PERSONALITY_KEYWORDS = ["personality", "personality test", "personality assessment", "trait", "behavioral"]
COMMUNICATION_KEYWORDS = ["communication", "communication test", "communication skills", "speaking", "presentation", "listening"]

# Phrases that must NOT appear in questions for psychometric/personality/communication modes
_BANNED_TEST_PHRASES = [
    "psychometric test",
    "psychometric assessment",
    "psychometric interview",
    "psychometric section",
    "communication test",
    "communication assessment",
    "communication interview",
    "communication section",
]

# Generic/filler phrases that indicate low-quality questions
_GENERIC_QUESTION_PATTERNS = [
    "can you tell me more about your experience",
    "tell me more about your experience",
    "can you elaborate on your experience",
    "let's return to",
    "let's go back to",
    "let's move on to",
    "thanks for sharing",
    "great answer",
    "good point",
    "interesting",
]

# Transitional filler prefixes to strip from questions
_FILLER_PREFIXES = [
    "thanks for sharing.",
    "thank you for sharing.",
    "great answer.",
    "good point.",
    "interesting.",
    "that's a clear explanation.",
    "that's a great explanation.",
    "that's a helpful explanation.",
    "that's a good explanation.",
    "i see.",
    "understood.",
    "got it.",
    "that's helpful.",
    "let's return to",
    "let's go back to",
    "let's move on to",
    "moving on,",
    "now,",
    "okay,",
    "alright,",
]


def _contains_banned_test_phrase(text: str) -> bool:
    """Return True if text contains any banned test/meta phrase."""
    lower = text.lower()
    return any(p in lower for p in _BANNED_TEST_PHRASES)


def _is_generic_question(text: str) -> bool:
    """Return True if question is too generic/vague."""
    lower = text.lower()
    return any(p in lower for p in _GENERIC_QUESTION_PATTERNS)


def _is_incomplete_or_redundant_question(text: str) -> bool:
    """
    Universal heuristic: question looks incomplete or redundant (e.g. ends with ", word?" with no real question).
    No topic- or scenario-specific logic.
    """
    if not text or not text.strip():
        return True
    t = text.strip().rstrip("?")
    if not t:
        return True
    # Ends with ", X" or ", X Y" where the part after the last comma is very short (1–2 words) → likely incomplete
    if "," in t:
        after_comma = t.split(",")[-1].strip()
        words = after_comma.split()
        if len(words) <= 2 and len(after_comma) < 25:
            return True
    return False


def _greeting_name_from_candidate_info(candidate_info: dict) -> str:
    """
    Get name for greeting; use 'there' when name is missing or a placeholder (e.g. CANDIDATE_A, Candidate).
    """
    name = candidate_info.get("name") or candidate_info.get("names") or "there"
    if isinstance(name, list):
        name = name[0] if name else "there"
    name = (name or "there").strip() or "there"
    if (
        not name
        or name.upper() == "CANDIDATE_A"
        or name.lower() in ("candidate", "applicant", "interviewee")
    ):
        return "there"
    return name


def _strip_leading_greeting(text: str) -> str:
    """
    Remove leading "Hello Name, " / "Hi Name, " / "Hey Name, " from question text.
    Use on follow-up questions so greeting appears only on the first question.
    Name may contain apostrophes and hyphens (e.g. O'Brien, Mary-Jane).
    """
    if not text or not text.strip():
        return text or ""
    # Match: (Hello|Hi|Hey) space + name (letters, spaces, apostrophe, hyphen) + optional comma + space
    stripped = re.sub(r"^(hello|hi|hey)\s+[\w\s'-]+,?\s*", "", text.strip(), count=1, flags=re.IGNORECASE)
    return stripped.strip()


def _strip_filler_prefix(text: str) -> str:
    """Remove transitional filler phrases from the start of a question."""
    result = text.strip()
    
    # Keep stripping until no more filler prefixes found
    changed = True
    max_iterations = 5  # Prevent infinite loops
    iterations = 0
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        lower = result.lower()
        
        for prefix in _FILLER_PREFIXES:
            if lower.startswith(prefix):
                # Find where the actual question starts (after the prefix)
                result = result[len(prefix):].strip()
                # Handle case where there's punctuation after the prefix
                if result and result[0] in ".,;:!":
                    result = result[1:].strip()
                changed = True
                break  # Restart loop to check for more prefixes
        
        # Also strip topic mentions like "Python." at the start
        if result and not changed:
            # Check if starts with a single word followed by period
            first_space = result.find(' ')
            first_period = result.find('.')
            if 0 < first_period < first_space or (first_period > 0 and first_space == -1):
                # Likely "Python. Can you..." pattern
                potential_topic = result[:first_period].strip()
                if len(potential_topic) < 30 and potential_topic[0].isupper():
                    result = result[first_period + 1:].strip()
                    changed = True
    
    # Capitalize first letter if needed
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    
    return result


def _build_topic_specific_fallback(topic: str, stage: str, skills: str = "", last_answer: str = "") -> str:
    """Build a topic-specific fallback question that's contextual rather than generic."""
    topic = topic or "this area"
    stage = (stage or "topic_introduction").lower()
    
    # Try to extract something specific from skills or last answer
    context_hint = ""
    if last_answer:
        # Look for specific technologies mentioned
        answer_lower = last_answer.lower()
        tech_found = []
        for tech in ["api", "database", "testing", "deployment", "async", "performance", "security", "docker", "kubernetes"]:
            if tech in answer_lower:
                tech_found.append(tech)
        if tech_found:
            context_hint = tech_found[0]
    
    if stage == "topic_introduction":
        if context_hint:
            return f"You mentioned {context_hint} - how do you typically handle that when working with {topic}?"
        return f"What specific projects have you built using {topic}, and what technical decisions did you make?"
    elif stage == "topic_fundamentals":
        if context_hint:
            return f"Can you give me a concrete example of applying {context_hint} with {topic}?"
        return f"Walk me through how you would design a solution for a real problem using {topic}."
    elif stage == "topic_deep_dive":
        if context_hint:
            return f"What trade-offs did you consider when implementing {context_hint} in your {topic} work?"
        return f"Describe a technically challenging situation with {topic} and how you resolved it."
    else:
        if context_hint:
            return f"What lessons did you learn from your experience with {context_hint} in {topic} projects?"
        return f"What would you do differently if you could redo one of your {topic} projects?"


def _extract_last_answer_context(conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract context from the candidate's last answer for adaptive follow-up.
    
    Returns:
        Dict with:
        - last_answer: The candidate's last response text
        - key_claims: List of specific claims/topics mentioned (tools, techniques, metrics)
        - follow_up_hooks: Suggested areas to probe deeper
    """
    if not conversation_history:
        return {"last_answer": "", "key_claims": [], "follow_up_hooks": []}
    
    # Find the last user (candidate) message
    last_answer = ""
    for msg in reversed(conversation_history):
        if msg.get("role") == "user":
            last_answer = (msg.get("content") or "").strip()
            break
    
    if not last_answer:
        return {"last_answer": "", "key_claims": [], "follow_up_hooks": []}
    
    # Extract key claims - look for specific technical terms, tools, metrics, experiences
    key_claims = []
    follow_up_hooks = []
    
    answer_lower = last_answer.lower()
    
    # Technical indicators that suggest depth to probe
    tech_patterns = [
        # Tools and frameworks
        ("django", "Django framework"),
        ("fastapi", "FastAPI"),
        ("flask", "Flask"),
        ("react", "React"),
        ("vue", "Vue.js"),
        ("angular", "Angular"),
        ("kubernetes", "Kubernetes"),
        ("docker", "Docker"),
        ("aws", "AWS"),
        ("gcp", "Google Cloud"),
        ("azure", "Azure"),
        ("postgresql", "PostgreSQL"),
        ("mongodb", "MongoDB"),
        ("redis", "Redis"),
        ("kafka", "Kafka"),
        ("rabbitmq", "RabbitMQ"),
        # Python-specific
        ("generators", "generators"),
        ("itertools", "itertools"),
        ("asyncio", "asyncio"),
        ("multiprocessing", "multiprocessing"),
        ("threading", "threading"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("pytest", "pytest"),
        ("unittest", "unittest"),
        ("decorators", "decorators"),
        ("context manager", "context managers"),
        ("metaclass", "metaclasses"),
        # Concepts
        ("microservices", "microservices"),
        ("api", "API design"),
        ("rest", "REST APIs"),
        ("graphql", "GraphQL"),
        ("caching", "caching strategies"),
        ("optimization", "optimization"),
        ("scalability", "scalability"),
        ("performance", "performance tuning"),
        ("testing", "testing approach"),
        ("ci/cd", "CI/CD pipeline"),
        ("agile", "Agile methodology"),
    ]
    
    for pattern, label in tech_patterns:
        if pattern in answer_lower:
            key_claims.append(label)
    
    # Look for quantitative claims (metrics, numbers, scale)
    import re
    metrics_patterns = [
        r'\d+%',  # percentages
        r'\d+x',  # multipliers
        r'\d+ (users|requests|transactions|records)',  # scale
        r'(million|thousand|billion)',  # large numbers
        r'(reduced|improved|increased) by',  # improvements
    ]
    for pattern in metrics_patterns:
        if re.search(pattern, answer_lower):
            follow_up_hooks.append("specific metrics or results they mentioned")
            break
    
    # Look for experience claims
    experience_patterns = [
        (r'i (built|created|developed|implemented|designed)', "something they built"),
        (r'(my team|our team|we)', "team collaboration"),
        (r'(production|deployed|live)', "production experience"),
        (r'(challenge|difficult|problem)', "challenges they faced"),
        (r'(learned|realized|discovered)', "lessons learned"),
    ]
    for pattern, hook in experience_patterns:
        if re.search(pattern, answer_lower):
            follow_up_hooks.append(hook)
    
    # Limit to top items
    key_claims = key_claims[:5]
    follow_up_hooks = follow_up_hooks[:3]
    
    return {
        "last_answer": last_answer[:300],  # Truncate for prompt size
        "key_claims": key_claims,
        "follow_up_hooks": follow_up_hooks,
    }


def _build_safe_behavioral_question(topic_for_question: str, stage: str) -> str:
    """
    Deterministic, non-meta behavioral questions for psychometric/personality/
    communication modes when the LLM accidentally references tests/assessments.
    """
    base = topic_for_question or "the way you typically behave at work"
    stage = (stage or "topic_introduction").lower()

    if stage == "topic_introduction":
        return (
            f"Tell me about a recent situation at work that best illustrates {base}. "
            "What happened and how did you handle it?"
        )
    if stage == "topic_fundamentals":
        return (
            f"Can you walk me through how you usually approach a challenging situation that tests {base}? "
            "What steps do you take and why?"
        )
    if stage == "topic_deep_dive":
        return (
            f"Describe a particularly difficult moment where {base} was critical to the outcome. "
            "What did you do, and what did you learn from it?"
        )
    # topic_feedback or any other stage
    return (
        f"Looking back on your past experiences with {base}, what patterns do you notice in how you respond, "
        "and is there anything you would like to improve?"
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    trimmed = text[:limit]
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return trimmed.strip()


def _normalize_spacing(text: str) -> str:
    """
    Global text normalization: ensure proper spacing between words.
    Fixes cases where words are concatenated without spaces (e.g., "whereyou" -> "where you").
    """
    if not text:
        return ""
    
    # Step 1: Fix missing spaces between words (e.g., "whereyou" -> "where you")
    # Match word boundaries: lowercase followed by uppercase, or word char followed by word char
    # This handles cases like "whereyou", "projectwhere", etc.
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # "whereYou" -> "where You"
    text = re.sub(r'(\w)([A-Z][a-z])', r'\1 \2', text)  # "whereYou" -> "where You" (more general)
    
    # Step 2: Normalize whitespace (multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Step 3: Fix spacing around punctuation (remove spaces before, ensure space after)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # "word ." -> "word."
    text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)  # "word.Next" -> "word. Next"

    # Step 4: Clean up awkward punctuation combos like ",?" -> "?"
    text = re.sub(r',\s*\?', '?', text)
    
    return text.strip()


def _normalize_question_grammar(question: str) -> str:
    """
    Universal grammar normalization: fix common LLM gaps so every question is grammatically correct.
    No topic- or scenario-specific logic (no hardcoded tech names or topics).
    - Missing preposition after "experience" / "hands-on experience".
    - Missing verb after "how did you" / "how do you".
    - Missing auxiliary "did": "how you approached/used X?" -> "how did you approach/use X?".
    - Redundant double verb: "how did you handle specifically leverage" -> "how did you specifically leverage".
    """
    if not question or not question.strip():
        return question or ""
    q = question.strip()
    # Fix missing preposition: "experience <topic>" -> "experience in <topic>"
    q = re.sub(r"\b(experience)\s+(?!in\s|with\s|that\s|of\s)(\S.*?)(\?|$)", r"\1 in \2\3", q, count=1, flags=re.IGNORECASE)
    # Fix "hands-on experience <topic>" -> "hands-on experience in <topic>"
    q = re.sub(r"\b(hands-on\s+experience)\s+(?!in\s|with\s)(\S.*?)(\?|$)", r"\1 in \2\3", q, count=1, flags=re.IGNORECASE)
    # Fix missing auxiliary: "how you approached/used X?" -> "how did you approach/use X?" (bare infinitive after "did"; universal verb set)
    _verb_to_base = {
        "approached": "approach", "approach": "approach", "used": "use", "use": "use",
        "applied": "apply", "apply": "apply", "handled": "handle", "handle": "handle",
        "managed": "manage", "manage": "manage", "leveraged": "leverage", "leverage": "leverage",
    }
    def _did_you(m):
        verb = m.group(1).lower()
        base = _verb_to_base.get(verb, verb)
        return "how did you " + base
    q = re.sub(
        r"\bhow\s+you\s+(approach(?:ed)?|use(?:d)?|apply|applied|handle(?:d)?|manage(?:d)?|leverage(?:d)?)\b",
        _did_you, q, count=1, flags=re.IGNORECASE
    )
    # Fix redundant double verb: "how did you handle specifically leverage" -> "how did you specifically leverage"
    q = re.sub(r"\b(how\s+did\s+you)\s+handle\s+(specifically\s+)(\w+)\b", r"\1 \2\3", q, count=1, flags=re.IGNORECASE)
    # "how did you handle <verb>" when next word is another verb -> "how did you <verb>"
    q = re.sub(r"\b(how\s+did\s+you)\s+handle\s+(leverage|use|apply|approach|manage)\b", r"\1 \2", q, count=1, flags=re.IGNORECASE)
    # Fix missing verb: "how did you <noun phrase>?" -> "how did you handle <noun phrase>?"
    q = re.sub(
        r"\b(how\s+did\s+you)\s+(?!manage|apply|handle|use|approach|demonstrate|improve|ensure|perform|apply|lead|support|implement|deliver)([^?]+)\?",
        r"\1 handle \2?",
        q,
        count=1,
        flags=re.IGNORECASE,
    )
    q = re.sub(
        r"\b(how\s+do\s+you)\s+(?!manage|apply|handle|use|approach|demonstrate|improve|ensure|perform|lead|support|implement|deliver)([^?]+)\?",
        r"\1 handle \2?",
        q,
        count=1,
        flags=re.IGNORECASE,
    )
    return q.strip()


def _truncate_to_two_sentences(text: str) -> str:
    """Keep at most 2 sentences (split on . ? !)."""
    if not (text or text.strip()):
        return text or ""
    # Normalize spacing first
    text = _normalize_spacing(text)
    # Split on sentence boundaries, keep non-empty parts
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    if len(parts) <= 2:
        return " ".join(parts).strip()
    return " ".join(parts[:2]).strip()


def _get_mode_aware_system_prompt(
    base_prompt: str,
    interview_mode: Optional[str] = None,
    interviewer_type: Optional[str] = None
) -> str:
    """
    Adjust system prompt tone based on interview mode and interviewer type.
    
    CRITICAL: Only affects prompt phrasing/tone, NOT state transitions or question logic.
    
    Args:
        base_prompt: Base system prompt (from topic detection or default)
        interview_mode: "practice" or "hiring"
        interviewer_type: "ai" or "human"
        
    Returns:
        Mode-aware system prompt with adjusted tone
    """
    if not interview_mode and not interviewer_type:
        return base_prompt
    
    mode = (interview_mode or "").lower()
    interviewer = (interviewer_type or "").lower()
    
    # Practice + AI: Coaching tone (supportive, encouraging)
    if mode == "practice" and interviewer == "ai":
        tone_addition = (
            "\n\nTONE: Be supportive and encouraging. This is a practice interview. "
            "Provide constructive feedback in your questions when appropriate. "
            "Help the candidate learn and improve."
        )
        return base_prompt + tone_addition
    
    # Hiring + Human: Copilot tone (formal, structured, AI assists human)
    elif mode == "hiring" and interviewer == "human":
        tone_addition = (
            "\n\nTONE: Be formal and structured. You are assisting a human interviewer. "
            "Ask clear, professional questions that help evaluate the candidate objectively. "
            "Maintain a neutral, business-appropriate tone."
        )
        return base_prompt + tone_addition
    
    # Hiring + AI: Conservative scoring tone (strict evaluation)
    elif mode == "hiring" and interviewer == "ai":
        tone_addition = (
            "\n\nTONE: Be professional and thorough. This is a hiring interview. "
            "Ask questions that help assess the candidate's fit for the role. "
            "Maintain a professional, evaluative tone."
        )
        return base_prompt + tone_addition
    
    # Practice + Human: Silent copilot (minimal intervention)
    elif mode == "practice" and interviewer == "human":
        tone_addition = (
            "\n\nTONE: Be supportive but minimal. You are assisting a human in a practice session. "
            "Ask questions that help the candidate practice, but let the human lead the conversation."
        )
        return base_prompt + tone_addition
    
    # Default: return base prompt unchanged
    return base_prompt


class QuestionGenerator:
    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or "gemini-2.5-flash"

    def _build_prompt(self, session: Dict[str, Any], interview_req: Any) -> List[Dict[str, str]]:
        metadata = session.get("metadata") or {}
        topic = (metadata.get("interview_topic") or getattr(interview_req, "interview_topic", "") or "").strip()
        topic = topic or "the topic"
        topic_lower = topic.lower()
        stage = (metadata.get("interview_stage") or "topic_introduction").strip() or "topic_introduction"

        raw_skills = metadata.get("candidate_skills") or []
        if isinstance(raw_skills, list):
            skills_text = ", ".join([str(s) for s in raw_skills if s])[:300]
        else:
            skills_text = str(raw_skills)[:300]

        history_lines: List[str] = []
        for msg in session.get("conversation_history", [])[-MAX_HISTORY_MESSAGES:]:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").strip().replace("\n", " ")
            if content:
                history_lines.append(f"{role}: {content[:200]}")
        history_text = "\n".join(history_lines)

        # CRITICAL FIX: Detect special interview types and apply appropriate system prompts
        is_psychometric = any(kw in topic_lower for kw in PSYCHOMETRIC_KEYWORDS)
        is_personality = any(kw in topic_lower for kw in PERSONALITY_KEYWORDS)
        is_communication = any(kw in topic_lower for kw in COMMUNICATION_KEYWORDS)

        # ✅ IMPORTANT:
        # For psychometric/personality topics the raw topic label is often
        # "Psychometric Test" or "Personality Test". If we pass this directly
        # into the prompt, the LLM tends to generate *meta* questions such as
        # "let's return to the psychometric test" instead of scenario-based,
        # behavioral questions. To avoid this, we:
        #   1. Normalize the topic we expose to the model so it talks about
        #      "psychometric traits" / "personality traits" rather than "tests".
        #   2. Strengthen the guardrails in the system prompt to explicitly
        #      forbid referencing the test itself in the generated question.
        topic_for_question = topic

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
        elif is_communication:
            # Keep communication label but strip explicit "test" wording if present
            topic_for_question = topic.replace("test", "").replace("Test", "").strip() or "communication skills"

        # Persist normalized topic in metadata so downstream logic (e.g. post-processing)
        # can reuse it without recomputing.
        metadata["_topic_for_question"] = topic_for_question
        session["metadata"] = metadata
        
        # OPTIMIZATION: Use cached base prompts to avoid repeated string allocations
        if is_psychometric or is_personality:
            interview_type = "psychometric"
        elif is_communication:
            interview_type = "communication"
        else:
            interview_type = "default"
        
        system_prompt = _get_base_system_prompt(interview_type)
        
        # Apply mode-aware tone adjustments (affects phrasing only, not logic)
        interview_mode = getattr(interview_req, "interview_mode", None) or metadata.get("interview_mode")
        interviewer_type = getattr(interview_req, "interviewer_type", None) or metadata.get("interviewer_type")
        system_prompt = _get_mode_aware_system_prompt(
            system_prompt,
            interview_mode=interview_mode,
            interviewer_type=interviewer_type
        )

        # Get question count from metadata
        question_count = metadata.get("question_count", 0) or 0
        
        # Extract context from candidate's last answer for adaptive follow-up
        conversation_history = session.get("conversation_history", [])
        answer_context = _extract_last_answer_context(conversation_history)
        
        # Build adaptive follow-up guidance based on what the candidate said
        follow_up_guidance = ""
        if answer_context["key_claims"]:
            claims_str = ", ".join(answer_context["key_claims"][:3])
            follow_up_guidance = f"\n- MUST FOLLOW UP on: {claims_str}. Ask HOW they used it, WHY they chose it, or for a concrete example."
        if answer_context["follow_up_hooks"]:
            hooks_str = ", ".join(answer_context["follow_up_hooks"][:2])
            follow_up_guidance += f"\n- PROBE DEEPER into: {hooks_str}."
        
        # Build skill-aware first question guidance
        first_question_guidance = ""
        if question_count == 0 and skills_text:
            # Parse skills to find relevant ones for the topic
            skills_list = [s.strip() for s in skills_text.split(",") if s.strip()]
            topic_related_skills = []
            topic_words = set(topic_lower.split())
            
            for skill in skills_list:
                skill_lower = skill.lower()
                # Check if skill is related to topic
                if any(tw in skill_lower for tw in topic_words) or skill_lower in topic_lower:
                    topic_related_skills.append(skill)
                # Check for common skill associations
                elif topic_lower == "python" and skill_lower in ["django", "fastapi", "flask", "pandas", "numpy", "machine learning", "ml", "data science", "tensorflow", "pytorch"]:
                    topic_related_skills.append(skill)
                elif topic_lower == "javascript" and skill_lower in ["react", "vue", "angular", "node", "typescript", "next.js", "express"]:
                    topic_related_skills.append(skill)
            
            if topic_related_skills:
                related_str = ", ".join(topic_related_skills[:3])
                first_question_guidance = (
                    f"PERSONALIZE based on their background: The candidate knows {related_str}. "
                    f"Ask about their experience applying {topic_for_question} with these specific skills/tools."
                )
            else:
                first_question_guidance = (
                    f"Ask about their practical experience with {topic_for_question} in real projects."
                )
        
        # If candidate gave a substantial answer, emphasize building on it
        adaptive_instruction = ""
        if len(answer_context["last_answer"]) > 50 and question_count > 0:
            # Extract the most specific thing they mentioned for direct follow-up
            last_answer_snippet = answer_context['last_answer'][:200]
            adaptive_instruction = (
                f"CANDIDATE'S LAST ANSWER: \"{last_answer_snippet}...\"\n"
                f">>> YOUR NEXT QUESTION MUST directly reference something specific from their answer above. "
                f"Ask them to elaborate, explain trade-offs, or give a concrete example of what they mentioned."
            )
        
        # Determine question guidance based on stage
        if question_count == 0:
            question_guidance = first_question_guidance or f"Ask about their hands-on experience with {topic_for_question}."
        else:
            question_guidance = f"Go DEEPER - directly follow up on their last answer. Ask HOW/WHY/trade-offs."
        
        user_payload = (
            f"Topic: {topic_for_question}\n"
            f"Stage: {stage}\n"
            f"Question #{question_count + 1}\n"
            f"Candidate Skills: {skills_text}\n\n"
            f"{adaptive_instruction}\n"
            f"INSTRUCTION: {question_guidance}"
            f"{follow_up_guidance}\n\n"
            f"Recent conversation:\n{history_text}\n\n"
            "RULES:\n"
            "- DO NOT ask generic questions like 'tell me about your experience' or 'what challenges have you faced'\n"
            "- DO reference specific things from their answer or skills\n"
            "- DO NOT repeat previous questions\n\n"
            f"Generate ONE specific question:"
        )
        allowed = max(200, MAX_PROMPT_CHARS - len(system_prompt) - 10)
        user_content = _truncate(user_payload, allowed)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    async def generate(
        self,
        session: Dict[str, Any],
        response_analysis: Optional[Dict[str, Any]] = None,
        interview_req: Any = None,
    ) -> Dict[str, Any]:
        messages = self._build_prompt(session, interview_req)
        metadata = session.get("metadata", {}) or {}
        session_id = session.get("session_id") or metadata.get("session_id")
        stage = metadata.get("interview_stage", "topic_introduction")
        start_ts = time.time()
        
        try:
            llm_result = await llm_utils.invoke_llm(
                messages,
                model=self.model,
                enforce_json=False,
                max_tokens=StreamingConfig.QUESTION_MAX_TOKENS,
                temperature=0.7,
                max_retries=1,
            )
        except Exception as exc:
            log.exception("Question generator LLM call failed: %s", exc)
            llm_result = {"ok": False, "raw": "", "json": None, "error": str(exc)}

        try:
            log_invoke_llm(
                session_id=session_id,
                stage=stage,
                model=self.model,
                start_ts=start_ts,
                ok=llm_result.get("ok", False),
                json_valid=False,
                fallback_used=not llm_result.get("ok", False),
            )
        except Exception:
            log.debug("Telemetry logging failed; continuing")
        
        if not llm_result.get("ok"):
            return await self._fallback_question(session, interview_req, llm_result.get("raw", ""))

        # Determine special interview mode flags and normalized topic
        interview_topic = (
            metadata.get("interview_topic")
            or getattr(interview_req, "interview_topic", "")
            or ""
        )
        topic_lower = interview_topic.lower()
        is_psychometric = any(kw in topic_lower for kw in PSYCHOMETRIC_KEYWORDS)
        is_personality = any(kw in topic_lower for kw in PERSONALITY_KEYWORDS)
        is_communication = any(kw in topic_lower for kw in COMMUNICATION_KEYWORDS)
        topic_for_question = metadata.get("_topic_for_question") or interview_topic or "the topic"

        raw = llm_result.get("raw", "") or ""
        question = raw.strip().split("\n")[0]
        question = _truncate_to_two_sentences(question)
        question = question.strip('"').strip()
        
        # Normalize spacing globally (fixes "whereyou" -> "where you" and similar issues)
        question = _normalize_spacing(question)
        
        # Strip filler prefixes like "Thanks for sharing. Let's return to..."
        question = _strip_filler_prefix(question)
        
        # Re-normalize spacing after filler stripping (in case stripping affected spacing)
        question = _normalize_spacing(question)
        
        # Global grammar fix: missing preposition after "experience", missing verb after "how did/do you"
        question = _normalize_question_grammar(question)
        
        if not question.endswith("?"):
            question = f"{question}?"
        
        # Replace empty, generic, or incomplete/redundant questions with topic-specific fallback
        if not question.strip() or _is_generic_question(question) or _is_incomplete_or_redundant_question(question):
            log.debug("Replacing empty/generic/incomplete question with topic-specific fallback")
            question = _build_topic_specific_fallback(topic_for_question, stage)

        # FINAL SAFETY NET for psychometric/personality/communication modes:
        # If the model still mentions tests/assessments/interviews explicitly,
        # override with a deterministic behavioral question that never references tests.
        if (is_psychometric or is_personality or is_communication) and _contains_banned_test_phrase(
            question
        ):
            log.warning(
                "Psychometric/Personality/Communication question contained banned test/meta phrase; "
                "replacing with safe behavioral template."
            )
            question = _build_safe_behavioral_question(topic_for_question, stage)

        return {
            "question": question,
            "analysis": {},
            "suggested_follow_ups": [],
            "confidence": 0.8,
            "reasoning": "",
            "raw": raw,
            "fallback": False,
        }

    async def _fallback_question(self, session: Dict[str, Any], interview_req: Any, raw: str) -> Dict[str, Any]:
        from .fallback_generator import get_default_question

        metadata = session.get("metadata") or {}
        stage_value = metadata.get("interview_stage", _InterviewStateEnum.TOPIC_INTRODUCTION.value)
        try:
            stage = _InterviewStateEnum(stage_value)
        except Exception:
            stage = _InterviewStateEnum.TOPIC_INTRODUCTION

        topic = metadata.get("interview_topic") or getattr(interview_req, "interview_topic", None)
        fallback_q = await get_default_question(state=stage, interview_topic=topic)
        return {
            "question": fallback_q,
            "analysis": {},
            "suggested_follow_ups": [],
            "confidence": 0.0,
            "reasoning": "",
            "raw": raw,
            "fallback": True,
        }


_default_qg: Optional[QuestionGenerator] = None


def _get_default_generator() -> QuestionGenerator:
    global _default_qg
    if _default_qg is None:
        _default_qg = QuestionGenerator()
    return _default_qg


async def generate_question(
    session: Dict[str, Any],
    response_analysis: Optional[Dict[str, Any]] = None,
    interview_req: Any = None,
) -> Dict[str, Any]:
    generator = _get_default_generator()
    return await generator.generate(session, response_analysis, interview_req)

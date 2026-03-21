"""
Conversation Analyzer - Extract insights from conversation history.

Extracts:
- Career aspirations
- Career goals
- Interests and priorities
- Conversation patterns
"""

import logging
import re
from typing import Dict, Any, List, Optional, Set
from models.llm_invoker import invoke_llm, TaskType

log = logging.getLogger(__name__)


# Common stop words to filter out (not professional topics)
_STOP_WORDS: Set[str] = {
    "i", "me", "my", "we", "our", "you", "your", "it", "its", "the", "a", "an",
    "and", "or", "but", "if", "then", "so", "as", "at", "by", "for", "in", "of",
    "on", "to", "with", "from", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "can", "may", "might", "must", "shall", "this", "that", "these", "those",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all", "any",
    "some", "no", "not", "only", "just", "also", "very", "too", "more", "most",
    "other", "such", "than", "into", "over", "after", "before", "between",
    "about", "want", "need", "like", "know", "think", "get", "make", "take",
    "see", "look", "find", "give", "tell", "ask", "use", "try", "come", "go",
    "work", "help", "show", "here", "there", "now", "today", "please", "thanks",
    "thank", "yes", "okay", "ok", "hi", "hello", "hey", "good", "great",
    "sure", "right", "well", "yeah", "yea", "something", "anything", "nothing",
    "everything", "someone", "anyone", "everyone", "thing", "things", "way",
    "time", "times", "day", "days", "year", "years", "lot", "lots", "bit",
    "really", "actually", "basically", "currently", "maybe", "probably"
}

# Pre-compiled regex patterns for performance (compiled once at module load)
_RE_QUOTED = re.compile(r'["\']([^"\']{2,30})["\']')
_RE_LEARNING = re.compile(
    r'\b(?:learn|study|practice|improve|master|understand|explore|courses?|training|certification|skills?\s+in|proficient\s+in|experience\s+(?:in|with))'
    r'\s+(?:about\s+|in\s+|for\s+|on\s+)?([a-z][a-z\s]{2,25}?)(?:\s|$|,|\.|\?)',
    re.IGNORECASE
)
_RE_JOB = re.compile(
    r'(?:([a-z][a-z\s]{2,25}?)\s+(?:jobs?|roles?|positions?|careers?|opportunities)|'
    r'(?:become|work\s+as|transition\s+to)\s+(?:a\s+)?([a-z][a-z\s]{2,25}?)(?:\s|$|,|\.|\?))',
    re.IGNORECASE
)
_RE_CAPITALIZED = re.compile(r'\b([A-Z][a-zA-Z]{2,15})\b')
_RE_COMPOUND = re.compile(r'\b([a-z]{3,15})\s+and\s+([a-z]{3,15})\b')
_RE_PUNCTUATION = re.compile(r'[.,;:!?]+$')


def _extract_topics_from_text(text: str) -> List[str]:
    """
    Extract meaningful professional topics from text without hardcoded lists.
    
    Performance optimized:
    - Pre-compiled regex patterns (no recompilation per call)
    - Early exit for short text
    - Limited iterations and output size
    
    Returns list of extracted topics (lowercase, deduplicated, max 10).
    """
    # Early exit for very short messages (not worth processing)
    if not text or len(text) < 10:
        return []
    
    # Limit input size to prevent slow processing on very long messages
    if len(text) > 500:
        text = text[:500]
    
    topics = []
    text_lower = text.lower()
    
    # Pattern 1: Quoted phrases (highest confidence - user explicitly mentioned)
    for match in _RE_QUOTED.findall(text_lower):
        phrase = match.strip()
        if len(phrase) > 2 and phrase not in _STOP_WORDS:
            topics.append(phrase)
            if len(topics) >= 10:
                break
    
    # Pattern 2: Learning/skill patterns
    if len(topics) < 10:
        for match in _RE_LEARNING.findall(text_lower):
            phrase = match.strip() if isinstance(match, str) else (match[0] or match[1] if isinstance(match, tuple) else "").strip()
            if phrase:
                words = [w for w in phrase.split() if w not in _STOP_WORDS and len(w) > 1]
                if words:
                    topics.append(" ".join(words[:3]))
                    if len(topics) >= 10:
                        break
    
    # Pattern 3: Job/role patterns
    if len(topics) < 10:
        for match in _RE_JOB.findall(text_lower):
            # Match is a tuple (group1, group2) - take whichever matched
            phrase = (match[0] or match[1]).strip() if isinstance(match, tuple) else match.strip()
            if phrase:
                words = [w for w in phrase.split() if w not in _STOP_WORDS and len(w) > 1]
                if words:
                    topics.append(" ".join(words[:3]))
                    if len(topics) >= 10:
                        break
    
    # Pattern 4: Capitalized terms (tools, technologies, proper nouns)
    if len(topics) < 10:
        for term in _RE_CAPITALIZED.findall(text)[:15]:  # Limit to first 15 matches
            term_lower = term.lower()
            if term_lower not in _STOP_WORDS and len(term_lower) > 2:
                topics.append(term_lower)
                if len(topics) >= 10:
                    break
    
    # Pattern 5: "X and Y" compound phrases (quick check)
    if len(topics) < 10:
        for w1, w2 in _RE_COMPOUND.findall(text_lower)[:5]:  # Limit to 5 matches
            if w1 not in _STOP_WORDS and len(w1) > 3:
                topics.append(w1)
            if w2 not in _STOP_WORDS and len(w2) > 3:
                topics.append(w2)
            if len(topics) >= 10:
                break
    
    # Fast deduplication and cleanup
    seen = set()
    cleaned = []
    for topic in topics:
        topic = _RE_PUNCTUATION.sub('', topic.strip().lower())
        if 3 <= len(topic) <= 30 and topic not in seen:
            words = topic.split()
            if not all(w in _STOP_WORDS for w in words):
                seen.add(topic)
                cleaned.append(topic)
                if len(cleaned) >= 10:
                    break
    
    return cleaned


async def extract_aspirations_from_conversation(
    conversation_history: List[Dict[str, str]],
    uid: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract career aspirations, goals, and interests from conversation history.
    
    Args:
        conversation_history: List of message dicts with 'role' and 'content'
        uid: Optional user ID for logging
        
    Returns:
        Dict with extracted aspirations, goals, interests, and confidence
    """
    if not conversation_history:
        return {
            "career_goals": [],
            "skills_of_interest": [],
            "career_paths_of_interest": [],
            "aspirations": [],
            "timeline": None,
            "confidence": 0.0
        }
    
    try:
        # Build conversation text (last 10 messages for context)
        recent_messages = conversation_history[-10:]
        conversation_text = "\n".join([
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in recent_messages
        ])
        
        # Build extraction prompt
        extraction_prompt = f"""Analyze this conversation and extract the user's career aspirations, goals, and interests.

Conversation:
{conversation_text}

Extract:
1. Career goals (e.g., "become a Lead Stylist", "start my own salon", "transition to Product Manager")
2. Skills they want to learn/improve
3. Career paths they're interested in
4. Any specific targets or timelines mentioned
5. General aspirations (what they want to achieve)

Return ONLY valid JSON (no markdown, no code blocks):
{{
    "career_goals": ["goal1", "goal2"],
    "skills_of_interest": ["skill1", "skill2"],
    "career_paths_of_interest": ["path1", "path2"],
    "aspirations": ["aspiration1", "aspiration2"],
    "timeline": "mentioned timeline if any, or null",
    "confidence": 0.0-1.0
}}
"""
        
        # Call LLM to extract
        response = await invoke_llm(
            prompt=extraction_prompt,
            task_type=TaskType.TEXT_GENERATION,
            agent_name="career_chatbot",
            max_output_tokens=500
        )
        
        # Parse response
        from core.utils import _to_text, _extract_json_from_response
        response_text = _to_text(response)
        
        # Extract JSON
        extracted_data = _extract_json_from_response(response_text)
        
        if extracted_data and isinstance(extracted_data, dict):
            log.info(f"✅ Extracted aspirations for uid={uid}: {len(extracted_data.get('career_goals', []))} goals, {len(extracted_data.get('skills_of_interest', []))} skills")
            return extracted_data
        
        # Fallback: Simple keyword-based extraction
        return _extract_aspirations_keywords(conversation_history)
        
    except Exception as e:
        log.warning(f"Error extracting aspirations: {e}, falling back to keyword extraction")
        return _extract_aspirations_keywords(conversation_history)


def _extract_aspirations_keywords(
    conversation_history: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Fallback: Simple keyword-based extraction of aspirations.
    """
    aspirations = []
    goals = []
    skills = []
    
    aspiration_keywords = ["want to", "aspire to", "goal is", "dream", "hope to", "plan to", "aim to", "become", "wish to"]
    goal_keywords = ["goal", "target", "objective", "aim"]
    skill_keywords = ["learn", "improve", "master", "skill", "course", "training"]
    
    for msg in conversation_history:
        if msg.get("role") == "user":
            content = msg.get("content", "").lower()
            
            # Extract aspirations
            for keyword in aspiration_keywords:
                if keyword in content:
                    aspirations.append(msg.get("content", ""))
                    break
            
            # Extract goals
            for keyword in goal_keywords:
                if keyword in content:
                    goals.append(msg.get("content", ""))
                    break
            
            # Extract skills
            for keyword in skill_keywords:
                if keyword in content:
                    skills.append(msg.get("content", ""))
                    break
    
    return {
        "career_goals": goals[:5],  # Limit to 5
        "skills_of_interest": skills[:5],
        "career_paths_of_interest": [],
        "aspirations": aspirations[:5],
        "timeline": None,
        "confidence": 0.3  # Low confidence for keyword-based
    }


async def extract_goals_from_message(
    user_message: str,
    uid: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract career goals from a single user message.
    
    Args:
        user_message: User's message text
        uid: Optional user ID for logging
        
    Returns:
        List of goal dicts with title, description, priority, etc.
    """
    if not user_message:
        return []
    
    try:
        # Check if message contains goal indicators
        goal_indicators = [
            "my goal is", "i want to", "i aim to", "i plan to",
            "my target is", "i aspire to", "my objective is",
            "i hope to", "i wish to", "i dream of"
        ]
        
        message_lower = user_message.lower()
        has_goal = any(indicator in message_lower for indicator in goal_indicators)
        
        if not has_goal:
            return []
        
        # Build extraction prompt
        extraction_prompt = f"""Extract career goals from this user message.

User Message: "{user_message}"

Extract any career goals mentioned. A goal should have:
- A clear title (what they want to achieve)
- A description (why or how)
- Optional: priority (high/medium/low), timeline

Return ONLY valid JSON array (no markdown, no code blocks):
[
    {{
        "title": "Goal title",
        "description": "Goal description",
        "priority": "high|medium|low",
        "target_date": "YYYY-MM-DD or null"
    }}
]

If no clear goal is mentioned, return empty array [].
"""
        
        # Call LLM to extract
        response = await invoke_llm(
            prompt=extraction_prompt,
            task_type=TaskType.TEXT_GENERATION,
            agent_name="career_chatbot",
            max_output_tokens=300
        )
        
        # Parse response
        from core.utils import _to_text, _extract_json_from_response
        response_text = _to_text(response)
        
        # Extract JSON
        extracted_goals = _extract_json_from_response(response_text)
        
        if isinstance(extracted_goals, list):
            log.info(f"✅ Extracted {len(extracted_goals)} goals from message for uid={uid}")
            return extracted_goals
        elif isinstance(extracted_goals, dict) and "goals" in extracted_goals:
            return extracted_goals["goals"]
        
        return []
        
    except Exception as e:
        log.warning(f"Error extracting goals from message: {e}")
        return []


async def analyze_conversation_patterns(
    conversation_history: List[Dict[str, str]],
    uid: Optional[str] = None,
    existing_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze conversation to extract patterns and insights.
    Builds conversation profile over time.
    
    Args:
        conversation_history: List of message dicts
        uid: Optional user ID for logging
        existing_profile: Optional existing conversation profile to merge with
    
    Returns:
        Dict with comprehensive conversation insights:
        - topics_discussed: Topics mentioned in conversation
        - interests: User interests detected
        - priorities: Most important topics (based on frequency)
        - question_types: Types of questions asked
        - engagement_level: How engaged the user is
        - conversation_themes: Recurring themes
        - preferred_topics: Topics user asks about most
        - interaction_patterns: How user interacts (frequency, depth, etc.)
        - learning_preferences: How user prefers to learn/engage
    """
    if not conversation_history:
        return {
            "topics_discussed": [],
            "interests": [],
            "priorities": [],
            "question_types": [],
            "engagement_level": 0.0,
            "conversation_themes": [],
            "preferred_topics": [],
            "interaction_patterns": {},
            "learning_preferences": {}
        }
    
    try:
        # Enhanced pattern analysis
        topics = []
        interests = []
        question_types = []
        themes = []
        topic_frequency = {}  # Track frequency of topics
        
        # Keywords for different categories (expanded)
        course_keywords = ["course", "learn", "training", "education", "certification", "tutorial", "study"]
        assessment_keywords = ["assessment", "test", "evaluate", "quiz", "exam", "validate", "check"]
        career_keywords = ["career", "job", "role", "position", "opportunity", "employment", "work"]
        skill_keywords = ["skill", "ability", "competency", "expertise", "proficiency", "capability"]
        goal_keywords = ["goal", "target", "objective", "aim", "plan", "aspire", "want to"]
        
        # Location preference keywords
        location_keywords = {
            "india": ["india", "indian", "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "chennai", "pune", "kolkata", "noida", "gurgaon"],
            "usa": ["usa", "united states", "america", "new york", "california", "texas", "seattle", "san francisco", "boston", "chicago"],
            "uk": ["uk", "united kingdom", "london", "manchester", "birmingham", "england", "british"],
            "uae": ["uae", "dubai", "abu dhabi", "emirates", "gulf"],
            "global": ["global", "worldwide", "international", "anywhere", "remote worldwide"],
            "remote": ["remote", "work from home", "wfh", "hybrid", "flexible location"]
        }
        
        # Work preference keywords
        work_type_keywords = {
            "remote": ["remote", "work from home", "wfh", "virtual", "telecommute"],
            "hybrid": ["hybrid", "flexible", "part remote"],
            "onsite": ["onsite", "on-site", "in-office", "office-based"]
        }
        
        # Specific technical topics to track
        specific_topics_tracked = []
        
        # Track interaction patterns
        user_messages = [m for m in conversation_history if m.get("role") == "user"]
        assistant_messages = [m for m in conversation_history if m.get("role") == "assistant"]
        total_user_messages = len(user_messages)
        total_assistant_messages = len(assistant_messages)
        
        # Analyze each user message
        for msg in user_messages:
            content = msg.get("content", "").lower()
            msg_length = len(msg.get("content", ""))
            
            # Categorize by keywords with frequency tracking
            if any(kw in content for kw in course_keywords):
                topics.append("courses")
                interests.append("learning")
                topic_frequency["courses"] = topic_frequency.get("courses", 0) + 1
                themes.append("learning_development")
            
            if any(kw in content for kw in assessment_keywords):
                topics.append("assessments")
                interests.append("skill evaluation")
                topic_frequency["assessments"] = topic_frequency.get("assessments", 0) + 1
                themes.append("skill_validation")
            
            if any(kw in content for kw in career_keywords):
                topics.append("career")
                interests.append("career development")
                topic_frequency["career"] = topic_frequency.get("career", 0) + 1
                themes.append("career_advancement")
            
            if any(kw in content for kw in skill_keywords):
                topics.append("skills")
                interests.append("skill development")
                topic_frequency["skills"] = topic_frequency.get("skills", 0) + 1
                themes.append("skill_development")
            
            if any(kw in content for kw in goal_keywords):
                topics.append("goals")
                interests.append("goal setting")
                topic_frequency["goals"] = topic_frequency.get("goals", 0) + 1
                themes.append("goal_oriented")
            
            # Detect question types with more detail
            if "?" in msg.get("content", ""):
                if any(kw in content for kw in ["how", "what", "why", "when", "where", "which"]):
                    question_types.append("information_seeking")
                elif any(kw in content for kw in ["should", "can", "could", "would", "may"]):
                    question_types.append("advice_seeking")
                elif any(kw in content for kw in ["is", "are", "do", "does", "did"]):
                    question_types.append("clarification_seeking")
            
            # Detect location preferences
            for region, keywords in location_keywords.items():
                if any(kw in content for kw in keywords):
                    topic_frequency[f"location_{region}"] = topic_frequency.get(f"location_{region}", 0) + 1
            
            # Detect work type preferences
            for work_type, keywords in work_type_keywords.items():
                if any(kw in content for kw in keywords):
                    topic_frequency[f"work_type_{work_type}"] = topic_frequency.get(f"work_type_{work_type}", 0) + 1
            
            # Extract meaningful phrases/topics dynamically from the message
            # Uses simple NLP-style extraction without hardcoded lists
            extracted_topics = _extract_topics_from_text(content)
            for topic in extracted_topics:
                specific_topics_tracked.append(topic)
                topic_frequency[f"topic_{topic}"] = topic_frequency.get(f"topic_{topic}", 0) + 1
        
        # Calculate engagement metrics
        total_messages = total_user_messages
        avg_length = sum(len(m.get("content", "")) for m in conversation_history) / max(len(conversation_history), 1)
        conversation_depth = avg_length / 100.0  # Normalize
        message_frequency = total_messages / max(len(conversation_history), 1)  # Messages per exchange
        
        # Engagement level: combination of frequency, depth, and consistency
        engagement_level = min(1.0, (
            (total_messages / 10.0) * 0.3 +  # Frequency component
            conversation_depth * 0.4 +  # Depth component
            message_frequency * 0.3  # Consistency component
        ))
        
        # Identify preferred topics (most frequently discussed)
        preferred_topics = sorted(
            topic_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        preferred_topics_list = [topic for topic, count in preferred_topics]
        
        # Identify conversation themes (recurring patterns)
        theme_frequency = {}
        for theme in themes:
            theme_frequency[theme] = theme_frequency.get(theme, 0) + 1
        conversation_themes = sorted(
            theme_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        conversation_themes_list = [theme for theme, count in conversation_themes]
        
        # Analyze interaction patterns
        interaction_patterns = {
            "message_count": total_user_messages,
            "avg_message_length": avg_length,
            "conversation_depth": conversation_depth,
            "question_frequency": len([m for m in user_messages if "?" in m.get("content", "")]) / max(total_user_messages, 1),
            "response_ratio": total_assistant_messages / max(total_user_messages, 1),
            "conversation_turns": len(conversation_history) // 2  # Approximate turns
        }
        
        # Infer learning preferences
        learning_preferences = {}
        if topic_frequency.get("courses", 0) > topic_frequency.get("assessments", 0):
            learning_preferences["preferred_learning_method"] = "structured_learning"
        elif topic_frequency.get("assessments", 0) > 0:
            learning_preferences["preferred_learning_method"] = "assessment_based"
        else:
            learning_preferences["preferred_learning_method"] = "exploratory"
        
        # Determine if user prefers detailed or concise responses
        if avg_length > 100:
            learning_preferences["response_preference"] = "detailed"
        elif avg_length > 50:
            learning_preferences["response_preference"] = "moderate"
        else:
            learning_preferences["response_preference"] = "concise"
        
        # Determine location preference (most frequently mentioned)
        location_preference = None
        location_counts = {k.replace("location_", ""): v for k, v in topic_frequency.items() if k.startswith("location_")}
        if location_counts:
            location_preference = max(location_counts.items(), key=lambda x: x[1])[0]
        
        # Determine work type preference
        work_type_preference = None
        work_type_counts = {k.replace("work_type_", ""): v for k, v in topic_frequency.items() if k.startswith("work_type_")}
        if work_type_counts:
            work_type_preference = max(work_type_counts.items(), key=lambda x: x[1])[0]
        
        # Get most discussed specific professional topics
        prof_topic_counts = {k.replace("topic_", ""): v for k, v in topic_frequency.items() if k.startswith("topic_")}
        frequently_discussed_topics = sorted(prof_topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        frequently_discussed_topics_list = [topic for topic, count in frequently_discussed_topics]
        
        # Build insights dict
        insights = {
            "topics_discussed": list(set(topics))[:10],
            "interests": list(set(interests))[:10],
            "priorities": preferred_topics_list[:5],  # Most frequent topics
            "question_types": list(set(question_types)),
            "engagement_level": round(engagement_level, 2),
            "conversation_themes": conversation_themes_list,
            "preferred_topics": preferred_topics_list,
            "interaction_patterns": interaction_patterns,
            "learning_preferences": learning_preferences,
            "topic_frequency": topic_frequency,
            "location_preference": location_preference,
            "work_type_preference": work_type_preference,
            "frequently_discussed_topics": frequently_discussed_topics_list,
            "specific_topics_tracked": list(set(specific_topics_tracked))[:10],
            "last_analyzed": None  # Will be set by caller
        }
        
        # Merge with existing profile if provided (build profile over time)
        if existing_profile:
            # Merge topics (accumulate over time)
            existing_topics = existing_profile.get("topics_discussed", [])
            merged_topics = list(set(existing_topics + insights["topics_discussed"]))[:15]
            insights["topics_discussed"] = merged_topics
            
            # Merge interests
            existing_interests = existing_profile.get("interests", [])
            merged_interests = list(set(existing_interests + insights["interests"]))[:15]
            insights["interests"] = merged_interests
            
            # Update engagement level (weighted average with existing)
            existing_engagement = existing_profile.get("engagement_level", 0.0)
            # Weight recent engagement more heavily
            insights["engagement_level"] = round((existing_engagement * 0.6 + engagement_level * 0.4), 2)
            
            # Merge preferred topics (accumulate frequency)
            existing_freq = existing_profile.get("topic_frequency", {})
            for topic, count in topic_frequency.items():
                existing_freq[topic] = existing_freq.get(topic, 0) + count
            insights["topic_frequency"] = existing_freq
            insights["preferred_topics"] = sorted(
                existing_freq.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            insights["preferred_topics"] = [topic for topic, count in insights["preferred_topics"]]
            
            # Merge conversation themes
            existing_themes = existing_profile.get("conversation_themes", [])
            merged_themes = list(set(existing_themes + insights["conversation_themes"]))[:10]
            insights["conversation_themes"] = merged_themes
            
            # Merge location preference (keep most recent if set, otherwise use existing)
            if not insights.get("location_preference"):
                insights["location_preference"] = existing_profile.get("location_preference")
            
            # Merge work type preference
            if not insights.get("work_type_preference"):
                insights["work_type_preference"] = existing_profile.get("work_type_preference")
            
            # Merge frequently discussed topics (recalculate from merged topic_frequency)
            merged_prof_topics = {k.replace("topic_", ""): v for k, v in insights["topic_frequency"].items() if k.startswith("topic_")}
            insights["frequently_discussed_topics"] = [t for t, c in sorted(merged_prof_topics.items(), key=lambda x: x[1], reverse=True)[:5]]
            
            # Merge specific topics tracked
            existing_specific = existing_profile.get("specific_topics_tracked", [])
            merged_specific = list(set(existing_specific + insights.get("specific_topics_tracked", [])))[:15]
            insights["specific_topics_tracked"] = merged_specific
        
        log.info(f"✅ Analyzed conversation patterns for uid={uid}: {len(insights['topics_discussed'])} topics, engagement={insights['engagement_level']}, location={insights.get('location_preference')}")
        
        return insights
        
    except Exception as e:
        log.warning(f"Error analyzing conversation patterns: {e}")
        return {
            "topics_discussed": [],
            "interests": [],
            "priorities": [],
            "question_types": [],
            "engagement_level": 0.0,
            "conversation_themes": [],
            "preferred_topics": [],
            "interaction_patterns": {},
            "learning_preferences": {},
            "location_preference": None,
            "work_type_preference": None,
            "frequently_discussed_topics": [],
            "specific_topics_tracked": []
        }




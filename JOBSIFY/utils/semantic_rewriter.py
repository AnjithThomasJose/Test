"""
Semantic Rewriter - Clean and normalize conversation history for LLM prompts.

This module provides utilities to sanitize, compact, and normalize conversation
turns before sending them to LLMs, improving prompt quality and reducing token usage.

Also provides semantic matching functionality using embeddings for skill matching.
"""

import re
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
import numpy as np

log = logging.getLogger(__name__)


def rewrite_recent_turns(history: List[Dict], max_turns: int = 8) -> List[Dict]:
    """
    Return a sanitized, compact list of the most recent turns for LLM prompts.
    
    Processing steps:
    - Keep last max_turns turns (counting Q/A pairs)
    - Normalize roles to 'assistant'/'user'/'system'
    - Trim each content to ~1000 chars, collapse long whitespace, remove greetings
    - Return list of dicts with normalized role/content pairs
    
    Args:
        history: List of message dicts with 'role' and 'content' keys
        max_turns: Maximum number of turns to keep (default: 8)
        
    Returns:
        List of normalized dicts: [{'role': 'assistant', 'content': '...'}, ...]
    """
    if not history:
        return []
    
    # Normalize roles and extract valid messages
    normalized = []
    for msg in history:
        role = msg.get("role", "").lower().strip()
        content = str(msg.get("content", "")).strip()
        
        if not content:
            continue
            
        # Normalize role
        if role in ("assistant", "ai", "model"):
            role = "assistant"
        elif role in ("user", "candidate", "human"):
            role = "user"
        elif role == "system":
            role = "system"
        else:
            # Default to user if unknown
            role = "user"
        
        normalized.append({"role": role, "content": content})
    
    # Keep only last max_turns messages
    if len(normalized) > max_turns:
        normalized = normalized[-max_turns:]
    
    # Process each message content
    processed = []
    for msg in normalized:
        content = msg["content"]
        
        # Remove common greetings (case-insensitive)
        greeting_patterns = [
            r'^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)),?\s*',
            r'^(thanks|thank\s+you),?\s*',
            r'^(sure|ok|okay|alright),?\s*',
        ]
        for pattern in greeting_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Collapse multiple whitespace to single space
        content = re.sub(r'\s+', ' ', content)
        
        # Trim to ~1000 characters (preserve word boundaries)
        if len(content) > 1000:
            # Find last space before 1000 chars
            trim_point = content[:1000].rfind(' ')
            if trim_point > 500:  # Only trim if we can keep substantial content
                content = content[:trim_point] + "..."
            else:
                content = content[:1000] + "..."
        
        content = content.strip()
        
        if content:  # Only add non-empty messages
            processed.append({
                "role": msg["role"],
                "content": content
            })
    
    return processed


# ==================== SEMANTIC MATCHING FUNCTIONS ====================

class SemanticMatchConfig:
    """Configuration for semantic matching behavior"""
    # Enable/disable semantic matching
    ENABLE_SEMANTIC_MATCHING = True
    # Minimum cosine similarity threshold for semantic matches
    SEMANTIC_SIMILARITY_THRESHOLD = 0.75
    # Maximum number of candidates to check for performance
    MAX_SEMANTIC_CANDIDATES = 20
    # Skip semantic matching if we already have a high-confidence match above this score
    SKIP_IF_SCORE_ABOVE = 0.85


def calculate_embedding_similarity(
    skill1: str,
    skill2: str
) -> Optional[float]:
    """
    Calculate semantic similarity between two skills using embeddings.
    
    Uses sentence transformers to create embeddings and cosine similarity
    to measure semantic relatedness. This can match skills that are
    semantically similar even if they don't share exact keywords.
    
    Examples:
        - "JavaScript" and "JS" -> high similarity
        - "Machine Learning" and "ML" -> high similarity
        - "Python" and "Django" -> moderate similarity (related technologies)
        - "React" and "Vue" -> moderate similarity (similar frameworks)
    
    Args:
        skill1: First skill string
        skill2: Second skill string
        
    Returns:
        Cosine similarity score (0-1) or None if embeddings fail.
        Higher scores indicate more semantic similarity.
    """
    if not SemanticMatchConfig.ENABLE_SEMANTIC_MATCHING:
        return None
    
    if not skill1 or not skill2:
        return None
    
    try:
        from chroma import embedding_fn
        
        # Note: This function is called from perform_skill_matching which already runs
        # in a thread pool (via run_cpu_intensive), so embedding operations won't block
        # the event loop. Progress bars are disabled in chroma.py initialization.
        # The embedding function is thread-safe and can handle concurrent calls.
        embeddings = embedding_fn([skill1, skill2])
        
        if not embeddings or len(embeddings) != 2:
            log.debug(f"Failed to generate embeddings for '{skill1}' and '{skill2}'")
            return None
        
        embedding1 = np.array(embeddings[0])
        embedding2 = np.array(embeddings[1])
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            # Ensure similarity is in valid range [-1, 1]
            similarity = max(-1.0, min(1.0, float(similarity)))
            return similarity
        
        return None
        
    except ImportError as e:
        log.debug(f"Embedding function not available: {e}")
        return None
    except Exception as e:
        log.debug(f"Semantic matching failed for '{skill1}' vs '{skill2}': {e}")
        return None


def find_semantic_matches(
    required_skill: str,
    candidate_skills: Set[str],
    current_best_score: float = 0.0
) -> Tuple[str, float]:
    """
    Find semantic matches using embeddings.
    
    This function searches through candidate skills to find semantically
    similar matches using embedding-based cosine similarity. It only checks
    a limited number of candidates for performance reasons.
    
    Args:
        required_skill: Required skill to match (from job description)
        candidate_skills: Set of candidate skills to search through
        current_best_score: Current best matching score from other methods.
                           Semantic matching will only return if it beats this.
        
    Returns:
        Tuple of (best_match, best_score) where:
        - best_match: The candidate skill with highest semantic similarity
        - best_score: The cosine similarity score (0-1)
        Returns ("", 0.0) if no good semantic match is found.
    """
    if not SemanticMatchConfig.ENABLE_SEMANTIC_MATCHING:
        return "", 0.0
    
    if not required_skill or not candidate_skills:
        return "", 0.0
    
    # Skip semantic matching if we already have a high-confidence match
    if current_best_score >= SemanticMatchConfig.SKIP_IF_SCORE_ABOVE:
        log.debug(
            f"Skipping semantic matching for '{required_skill}' - "
            f"already have high-confidence match (score: {current_best_score:.3f})"
        )
        return "", 0.0
    
    best_match = ""
    best_score = 0.0
    
    # Limit candidates for performance (check top N most promising)
    # Convert to list and limit to avoid performance issues with large skill sets
    candidates_list = list(candidate_skills)[:SemanticMatchConfig.MAX_SEMANTIC_CANDIDATES]
    
    log.debug(
        f"Checking semantic similarity for '{required_skill}' against "
        f"{len(candidates_list)} candidate skills"
    )
    
    for cand_skill in candidates_list:
        similarity = calculate_embedding_similarity(required_skill, cand_skill)
        
        if similarity and similarity >= SemanticMatchConfig.SEMANTIC_SIMILARITY_THRESHOLD:
            if similarity > best_score and similarity > current_best_score:
                best_match = cand_skill
                best_score = similarity
                log.debug(
                    f"  ✓ EMBEDDING SEMANTIC MATCH: '{required_skill}' <-> '{cand_skill}' "
                    f"(cosine similarity: {similarity:.3f})"
                )
    
    if best_match:
        log.info(
            f"Found semantic match: '{required_skill}' -> '{best_match}' "
            f"(score: {best_score:.3f})"
        )
    
    return best_match, best_score


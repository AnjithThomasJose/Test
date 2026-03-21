"""
LLM-based engagement analysis.

Analyzes engagement level, verbosity, coherence, and follow-up detection.
"""

import logging
from typing import Dict, List, Any, Optional

from ..llm_utils import invoke_llm

log = logging.getLogger(__name__)


async def analyze_engagement_comprehensive(
    text: str,
    history: List[Dict] = None,
    client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Analyze engagement comprehensively using LLM.
    
    Args:
        text: Response text to analyze
        history: Optional conversation history
        client: Optional custom LLM client
        
    Returns:
        Dict with:
        - engagement_score: float (0-1)
        - evidence: str (optional)
    """
    if not text or not text.strip():
        return {
            "engagement_score": 0.0,
            "evidence": "Empty response"
        }
    
    # Build context from history
    context_str = ""
    if history:
        recent = history[-3:] if len(history) > 3 else history
        context_str = f"\nRecent conversation:\n{str(recent)}"
    
    prompt = f"""Analyze the engagement level of this interview response.

RESPONSE TEXT:
"{text.strip()}"

{context_str}

Consider:
- Length and detail (too short = low engagement)
- Relevance to the question
- Enthusiasm and interest shown
- Follow-up questions or clarifications
- Depth of explanation

Return STRICT JSON only:
{{
    "engagement_score": 0.0 to 1.0 (0=low, 1=high),
    "evidence": "brief explanation of engagement level"
}}"""
    
    try:
        result = await invoke_llm(
            prompt=prompt,
            model=None,
            enforce_json=True,
            client=client
        )
        
        if result.get("ok") and result.get("json"):
            parsed = result.get("json", {})
            score = float(parsed.get("engagement_score", 0.0))
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
            return {
                "engagement_score": score,
                "evidence": str(parsed.get("evidence", "")).strip()
            }
        else:
            log.warning(f"LLM engagement analysis failed: {result.get('error')}")
            # Fallback: simple heuristic based on length
            word_count = len(text.split())
            fallback_score = min(1.0, word_count / 50.0)
            return {
                "engagement_score": fallback_score,
                "evidence": "Fallback: length-based heuristic"
            }
    
    except Exception as e:
        log.error(f"Error in analyze_engagement_comprehensive: {e}")
        # Fallback: simple heuristic
        word_count = len(text.split()) if text else 0
        fallback_score = min(1.0, word_count / 50.0)
        return {
            "engagement_score": fallback_score,
            "evidence": f"Error: {str(e)}"
        }

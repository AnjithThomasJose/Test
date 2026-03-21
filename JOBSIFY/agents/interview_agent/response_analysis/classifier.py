"""
LLM-based response type classifier.

Classifies response type: technical, behavioral, brief, or general.
"""

import logging
from typing import Dict, Any, Optional

from ..llm_utils import invoke_llm

log = logging.getLogger(__name__)


async def classify_response_type(
    text: str,
    context: dict = None,
    client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Classify the response type using LLM.
    
    Args:
        text: Response text to classify
        context: Optional context dict (conversation history, etc.)
        client: Optional custom LLM client
        
    Returns:
        Dict with:
        - type: "technical" | "behavioral" | "brief" | "general"
        - confidence: float (0-1)
        - evidence: str (optional)
    """
    if not text or not text.strip():
        return {
            "type": "general",
            "confidence": 0.0,
            "evidence": "Empty response"
        }
    
    # Build prompt
    context_str = ""
    if context:
        history = context.get("conversation_history", [])
        if history:
            recent = history[-3:] if len(history) > 3 else history
            context_str = f"\nRecent conversation context:\n{str(recent)}"
    
    prompt = f"""Classify this interview response by type.

RESPONSE TEXT:
"{text.strip()}"

{context_str}

TYPES:
- "technical": Discusses technical concepts, code, algorithms, systems, tools
- "behavioral": Discusses past experiences, situations, actions, outcomes
- "brief": Very short response (few words, minimal detail)
- "general": Doesn't fit other categories or is too vague

Return STRICT JSON only:
{{
    "type": "technical|behavioral|brief|general",
    "confidence": 0.0-1.0,
    "evidence": "brief explanation"
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
            return {
                "type": parsed.get("type", "general").lower().strip(),
                "confidence": float(parsed.get("confidence", 0.5)),
                "evidence": str(parsed.get("evidence", "")).strip()
            }
        else:
            log.warning(f"LLM classification failed: {result.get('error')}")
            return {
                "type": "general",
                "confidence": 0.0,
                "evidence": "LLM classification failed"
            }
    
    except Exception as e:
        log.error(f"Error in classify_response_type: {e}")
        return {
            "type": "general",
            "confidence": 0.0,
            "evidence": f"Error: {str(e)}"
        }

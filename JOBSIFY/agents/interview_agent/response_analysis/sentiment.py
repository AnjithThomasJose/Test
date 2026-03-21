"""
LLM-based sentiment analysis.

Analyzes sentiment (positive/neutral/negative) and polarity.
"""

import logging
from typing import Dict, Any, Optional

from ..llm_utils import invoke_llm

log = logging.getLogger(__name__)


async def analyze_sentiment(
    text: str,
    client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Analyze sentiment using LLM.
    
    Args:
        text: Response text to analyze
        client: Optional custom LLM client
        
    Returns:
        Dict with:
        - sentiment: "positive" | "neutral" | "negative"
        - sentiment_polarity: float (-1 to 1)
        - volatility: float (0-1, how much sentiment varies)
    """
    if not text or not text.strip():
        return {
            "sentiment": "neutral",
            "sentiment_polarity": 0.0,
            "volatility": 0.0
        }
    
    prompt = f"""Analyze the sentiment of this interview response.

RESPONSE TEXT:
"{text.strip()}"

Return STRICT JSON only:
{{
    "sentiment": "positive|neutral|negative",
    "sentiment_polarity": -1.0 to 1.0 (positive=1.0, neutral=0.0, negative=-1.0),
    "volatility": 0.0 to 1.0 (how much sentiment varies within the response)
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
            sentiment = parsed.get("sentiment", "neutral").lower().strip()
            if sentiment not in ["positive", "neutral", "negative"]:
                sentiment = "neutral"
            
            polarity = float(parsed.get("sentiment_polarity", 0.0))
            polarity = max(-1.0, min(1.0, polarity))  # Clamp to [-1, 1]
            
            volatility = float(parsed.get("volatility", 0.0))
            volatility = max(0.0, min(1.0, volatility))  # Clamp to [0, 1]
            
            return {
                "sentiment": sentiment,
                "sentiment_polarity": polarity,
                "volatility": volatility
            }
        else:
            log.warning(f"LLM sentiment analysis failed: {result.get('error')}")
            return {
                "sentiment": "neutral",
                "sentiment_polarity": 0.0,
                "volatility": 0.0
            }
    
    except Exception as e:
        log.error(f"Error in analyze_sentiment: {e}")
        return {
            "sentiment": "neutral",
            "sentiment_polarity": 0.0,
            "volatility": 0.0
        }

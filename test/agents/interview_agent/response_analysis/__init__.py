"""
Response analysis submodule for interview agent.

LLM-first architecture with parallel submodule execution.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from .models import ResponseAnalysis
from .classifier import classify_response_type
from .sentiment import analyze_sentiment
from .keywords import extract_keywords_llm, extract_skills_llm
from .engagement import analyze_engagement_comprehensive

log = logging.getLogger(__name__)


class IntelligentResponseAnalyzer:
    """
    LLM-powered response analysis orchestrator.
    
    Runs all analysis submodules in parallel and aggregates results
    into a validated ResponseAnalysis model.
    """
    
    @staticmethod
    async def analyze_response_intelligently(
        text: str,
        history: List[Dict] = None,
        memory: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        client: Optional[Any] = None
    ) -> ResponseAnalysis:
        """
        Comprehensive intelligent response analysis.
        
        Runs classifier, sentiment, keywords, skills, and engagement
        analysis in parallel, then merges results into ResponseAnalysis.
        
        Args:
            text: Response text to analyze
            history: Optional conversation history
            memory: Optional memory/context dict
            metadata: Optional metadata dict
            client: Optional custom LLM client
            
        Returns:
            ResponseAnalysis Pydantic model with all fields populated
        """
        if not text or not text.strip():
            # Return default for empty response
            return ResponseAnalysis(
                type="general",
                confidence=0.0,
                keywords=[],
                skills=[],
                sentiment="neutral",
                sentiment_polarity=0.0,
                volatility=0.0,
                engagement_score=0.0,
                evidence="Empty response"
            )
        
        # Build context dict for classifier
        context = {
            "conversation_history": history or [],
            "memory": memory or {},
            "metadata": metadata or {}
        }
        
        # Run all analysis modules in parallel
        try:
            classification_task = classify_response_type(text, context, client)
            sentiment_task = analyze_sentiment(text, client)
            keywords_task = extract_keywords_llm(text, client)
            skills_task = extract_skills_llm(text, client)
            engagement_task = analyze_engagement_comprehensive(text, history, client)
            
            # Wait for all tasks to complete
            classification, sentiment, keywords, skills, engagement = await asyncio.gather(
                classification_task,
                sentiment_task,
                keywords_task,
                skills_task,
                engagement_task,
                return_exceptions=True
            )
            
            # Handle exceptions from any task
            if isinstance(classification, Exception):
                log.error(f"Classification failed: {classification}")
                classification = {"type": "general", "confidence": 0.0, "evidence": "Classification error"}
            
            if isinstance(sentiment, Exception):
                log.error(f"Sentiment analysis failed: {sentiment}")
                sentiment = {"sentiment": "neutral", "sentiment_polarity": 0.0, "volatility": 0.0}
            
            if isinstance(keywords, Exception):
                log.error(f"Keyword extraction failed: {keywords}")
                keywords = []
            
            if isinstance(skills, Exception):
                log.error(f"Skill extraction failed: {skills}")
                skills = []
            
            if isinstance(engagement, Exception):
                log.error(f"Engagement analysis failed: {engagement}")
                engagement = {"engagement_score": 0.0, "evidence": "Engagement analysis error"}
            
            # Merge results into ResponseAnalysis
            # Build evidence string from all submodules
            evidence_parts = []
            if isinstance(classification, dict) and classification.get("evidence"):
                evidence_parts.append(f"Type: {classification.get('evidence')}")
            if isinstance(engagement, dict) and engagement.get("evidence"):
                evidence_parts.append(f"Engagement: {engagement.get('evidence')}")
            evidence = ". ".join(evidence_parts) if evidence_parts else ""
            
            # Extract values with fallbacks
            response_type = classification.get("type", "general") if isinstance(classification, dict) else "general"
            confidence = classification.get("confidence", 0.0) if isinstance(classification, dict) else 0.0
            
            sentiment_val = sentiment.get("sentiment", "neutral") if isinstance(sentiment, dict) else "neutral"
            sentiment_polarity = sentiment.get("sentiment_polarity", 0.0) if isinstance(sentiment, dict) else 0.0
            volatility = sentiment.get("volatility", 0.0) if isinstance(sentiment, dict) else 0.0
            
            keywords_list = keywords if isinstance(keywords, list) else []
            skills_list = skills if isinstance(skills, list) else []
            
            engagement_score = engagement.get("engagement_score", 0.0) if isinstance(engagement, dict) else 0.0
            
            # Create and return ResponseAnalysis model
            return ResponseAnalysis(
                type=response_type,
                confidence=confidence,
                keywords=keywords_list,
                skills=skills_list,
                sentiment=sentiment_val,
                sentiment_polarity=sentiment_polarity,
                volatility=volatility,
                engagement_score=engagement_score,
                evidence=evidence
            )
            
        except Exception as e:
            log.error(f"Error in analyze_response_intelligently: {e}")
            # Return safe fallback
            return ResponseAnalysis(
                type="general",
                confidence=0.0,
                keywords=[],
                skills=[],
                sentiment="neutral",
                sentiment_polarity=0.0,
                volatility=0.0,
                engagement_score=0.0,
                evidence=f"Analysis error: {str(e)}"
            )


# Legacy ResponseAnalyzer for backward compatibility
class ResponseAnalyzer:
    """Legacy response analyzer (deprecated - use IntelligentResponseAnalyzer)"""
    
    @staticmethod
    async def analyze_response(answer: str, is_first_request: bool = False) -> Dict[str, Any]:
        """
        Legacy analyze_response method.
        
        DEPRECATED: Use IntelligentResponseAnalyzer.analyze_response_intelligently instead.
        """
        log.warning("ResponseAnalyzer.analyze_response is deprecated. Use IntelligentResponseAnalyzer instead.")
        
        # Use new analyzer and convert to dict
        analysis = await IntelligentResponseAnalyzer.analyze_response_intelligently(
            text=answer,
            history=[],
            memory={},
            metadata={"is_first_request": is_first_request}
        )
        
        # Convert to legacy dict format
        return {
            "type": analysis.type,
            "confidence": analysis.confidence,
            "keywords": analysis.keywords,
            "sentiment": analysis.sentiment,
            "polarity": analysis.sentiment_polarity,
            "volatility": analysis.volatility,
            "length_score": 0.0,  # Not in new model
            "engagement_score": analysis.engagement_score,
            "word_count": len(answer.split()) if answer else 0
        }


# Re-export for convenience
__all__ = [
    "ResponseAnalysis",
    "IntelligentResponseAnalyzer",
    "ResponseAnalyzer",
    "classify_response_type",
    "analyze_sentiment",
    "extract_keywords_llm",
    "extract_skills_llm",
    "analyze_engagement_comprehensive",
]

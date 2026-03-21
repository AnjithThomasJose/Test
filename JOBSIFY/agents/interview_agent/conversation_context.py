"""
conversation_context.py

Lightweight, LLM-first compatible ConversationContext.

Responsibilities:
- Track minimal interview context state (topics, coverage counters, engagement)
- Provide to_dict() / from_dict() for safe persistence (session_manager expects dicts)
- Async update_context(...) which accepts a standardized response_analysis dict produced by response_analysis package
- No inline LLM calls, no direct skill extraction. Skill extraction is delegated to response_analysis modules.
- Uses await when calling session_manager.save_conversation_context(...)
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

log = logging.getLogger(__name__)

# Coverage bins we track by default
DEFAULT_COVERAGE_BINS = [
    "technical_skills",
    "experience",
    "problem_solving",
    "collaboration",
    "constraints",
    "outcomes",
]

class ConversationContext:
    """
    Simple, serializable conversation context.
    
    Minimal fields only:
    - topics_discussed: List of topics discussed
    - coverage: Dict of coverage bins (for evaluation summary only)
    - engagement_level: Current engagement score
    - conversation_depth: Question count
    - interview_stage: Current interview stage (authoritative, set by orchestrator)
    - interview_phase: Current interview phase (synced with stage)
    - session_id: Session identifier for persistence
    - last_question: Last question asked (for meta requests like "repeat")
    """
    def __init__(self) -> None:
        # Essential fields only
        self.topics_discussed: List[str] = []
        self.coverage: Dict[str, int] = {k: 0 for k in DEFAULT_COVERAGE_BINS}
        self.engagement_level: float = 0.5
        self.conversation_depth: int = 0
        self.interview_stage: str = "topic_introduction"
        self.interview_phase: str = "topic_introduction"
        self.session_id: Optional[str] = None
        self.last_question: Optional[str] = None

    # -------------------------
    # Serialization
    # -------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topics_discussed": list(self.topics_discussed),
            "coverage": dict(self.coverage),
            "engagement_level": float(self.engagement_level),
            "conversation_depth": int(self.conversation_depth),
            "interview_stage": str(self.interview_stage),
            "interview_phase": str(self.interview_phase),
            "session_id": self.session_id,
            "last_question": self.last_question,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        ctx = cls()
        ctx.topics_discussed = data.get("topics_discussed", []) or []
        ctx.coverage = data.get("coverage", {}) or {k: 0 for k in DEFAULT_COVERAGE_BINS}
        # ensure default bins exist
        for k in DEFAULT_COVERAGE_BINS:
            ctx.coverage.setdefault(k, 0)
        ctx.engagement_level = float(data.get("engagement_level", 0.5) or 0.5)
        ctx.conversation_depth = int(data.get("conversation_depth", 0) or 0)
        ctx.interview_stage = data.get("interview_stage", "topic_introduction") or "topic_introduction"
        ctx.interview_phase = data.get("interview_phase", "topic_introduction") or "topic_introduction"
        ctx.session_id = data.get("session_id")
        ctx.last_question = data.get("last_question")
        return ctx

    # -------------------------
    # Simple helpers
    # -------------------------
    def bump_coverage(self, bin_name: str, amount: int = 1) -> None:
        """Increment coverage counter for a bin."""
        if bin_name not in self.coverage:
            # add dynamic skill bin
            self.coverage[bin_name] = 0
        self.coverage[bin_name] = min(self.coverage.get(bin_name, 0) + amount, 9999)

    # -------------------------
    # Main async update entrypoint
    # -------------------------
    async def update_context(self, response_analysis: Dict[str, Any], question_count: int, persist: bool = True) -> None:
        """
        Update the conversation context using a standardized response_analysis dict.

        Expected fields in response_analysis (from response_analysis package):
            - keywords: list[str]
            - engagement_score: float (0..1)
            - sentiment: "positive"|"neutral"|"negative"
            - confidence: float (0..1)
            - word_count: int
            - type: "technical"|"behavioral"|"general"|"brief"
            - coverage: optional dict mapping coverage bins to booleans or scores
        
        NOTE: Coverage is ONLY for evaluation summary, NOT for state progression.
        Stage progression is handled by orchestrator._determine_stage_from_count().
        """
        try:
            ra = response_analysis or {}
            self.conversation_depth = question_count
            
            # engagement smoothing (simple moving average)
            new_eng = float(ra.get("engagement_score", self.engagement_level or 0.5) or 0.5)
            # simple smoothing
            self.engagement_level = (self.engagement_level * 0.6) + (new_eng * 0.4)

            # topics and keywords
            keywords = ra.get("keywords", []) or []
            for kw in keywords:
                if kw and kw not in self.topics_discussed:
                    self.topics_discussed.append(kw)

            # bump coverage from explicit coverage dict if provided
            # NOTE: Coverage is ONLY for evaluation summary, NOT for state progression.
            # Stage progression is handled by orchestrator._determine_stage_from_count().
            # Coverage must come from response_analysis - no fallback heuristics.
            cov = ra.get("coverage") or {}
            if isinstance(cov, dict) and cov:
                for k, v in cov.items():
                    try:
                        # accept boolean or numeric
                        if isinstance(v, bool) and v:
                            self.bump_coverage(k, 1)
                        elif isinstance(v, (int,float)) and v > 0:
                            self.bump_coverage(k, int(min(v, 5)))
                    except Exception:
                        continue

            # NOTE: interview_stage and interview_phase are set by orchestrator._determine_stage_from_count()
            # We do NOT update them here - orchestrator is authoritative

            # persist if requested (use session_manager's async API)
            if persist and self.session_id:
                try:
                    from . import session_manager
                    # session_manager.save_conversation_context expects a dict or serializable data
                    await session_manager.save_conversation_context(self.session_id, self.to_dict())
                except Exception:
                    log.exception("Failed to persist conversation context for session %s", self.session_id)

        except Exception:
            log.exception("update_context failed")

    # -------------------------
    # Utility: coverage score
    # -------------------------
    def get_coverage_score(self) -> float:
        """Return normalized coverage score between 0.0 and 1.0"""
        if not self.coverage:
            return 0.0
        vals = [min(v / 5.0, 1.0) for v in self.coverage.values()]
        return float(sum(vals) / len(vals)) if vals else 0.0

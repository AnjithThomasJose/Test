"""
Input Normalization Adapter

Converts inputs from all modalities (text, audio transcripts, avatar-mediated input)
into standardized format for the interview agent.

CRITICAL: This adapter does NOT:
- Track interview state
- Decide next questions
- Modify state transitions
- Use LLMs

It ONLY normalizes input format.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TranscriptChunk(BaseModel):
    """Single transcript chunk from Agora SDK or similar STT service"""
    text: str = Field(..., description="Transcript text fragment")
    timestamp_ms: int = Field(..., description="Timestamp in milliseconds when chunk was spoken")
    is_final: bool = Field(default=False, description="Whether this is a final chunk from STT")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="STT confidence score")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class NormalizedAnswer(BaseModel):
    """Complete normalized answer ready for interview agent"""
    answer_text: str = Field(..., description="Clean, normalized text answer")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input metadata (source, duration, hesitation markers, etc.)"
    )

    class Config:
        """Pydantic config"""
        extra = "forbid"


# ---------------------------------------------------------------------------
# Input Normalization Adapter
# ---------------------------------------------------------------------------

class InputNormalizationAdapter:
    """
    Converts streaming transcript chunks into complete answers.
    Works with text input (from Agora STT), not raw audio.
    
    Uses lightweight heuristics for boundary detection and text normalization.
    NO LLM calls are made in this adapter.
    """
    
    def __init__(
        self,
        silence_threshold_sec: float = 2.5,
        min_answer_length: int = 10,
        max_answer_length: int = 1000,
        default_source: str = "audio"
    ):
        """
        Initialize the adapter.
        
        Args:
            silence_threshold_sec: Seconds of silence before considering answer complete
            min_answer_length: Minimum characters before emitting an answer
            max_answer_length: Maximum characters (safety limit)
            default_source: Default input source modality ("text", "audio", "avatar")
        """
        self.silence_threshold = silence_threshold_sec
        self.min_length = min_answer_length
        self.max_length = max_answer_length
        self.default_source = default_source
        self._pending_chunks: Dict[str, List[TranscriptChunk]] = {}  # session_id -> chunks
        self._last_chunk_time: Dict[str, float] = {}  # session_id -> timestamp
        
        # NOTE: Session memory management
        # _pending_chunks and _last_chunk_time persist until flush_pending() is called.
        # For production, consider adding TTL cleanup or explicit session close hooks
        # to prevent memory leaks from disconnected sessions.
    
    async def process_chunk(
        self,
        chunk: TranscriptChunk,
        session_id: Optional[str] = None
    ) -> Optional[NormalizedAnswer]:
        """
        Process a transcript chunk.
        
        Args:
            chunk: Transcript chunk to process
            session_id: Session identifier (uses chunk.session_id if not provided)
            
        Returns:
            None if still accumulating, NormalizedAnswer when boundary detected
        """
        session_id = session_id or chunk.session_id
        if not session_id:
            log.warning("No session_id provided for chunk processing")
            return None
        
        # Accumulate chunk
        if session_id not in self._pending_chunks:
            self._pending_chunks[session_id] = []
        
        self._pending_chunks[session_id].append(chunk)
        
        # CRITICAL FIX: Store previous time BEFORE updating to compute silence correctly
        prev_time = self._last_chunk_time.get(session_id)
        current_time = chunk.timestamp_ms / 1000.0
        self._last_chunk_time[session_id] = current_time
        
        # Check for boundary (pass silence_duration for accurate detection)
        if prev_time is not None:
            silence_duration = current_time - prev_time
        else:
            silence_duration = 0.0
        
        if self._detect_boundary(chunk, session_id, silence_duration):
            return await self._emit_answer(session_id)
        
        return None
    
    def _detect_boundary(
        self,
        chunk: TranscriptChunk,
        session_id: str,
        silence_duration: float
    ) -> bool:
        """
        Detect if candidate has finished answering.
        Uses lightweight heuristics (no LLM).
        
        Args:
            chunk: Current chunk being processed
            session_id: Session identifier
            silence_duration: Duration of silence since last chunk (in seconds)
            
        Returns:
            True if answer boundary detected, False otherwise
        """
        chunks = self._pending_chunks.get(session_id, [])
        if not chunks:
            return False
        
        # Factor 1: Agora marked as final
        if chunk.is_final:
            return True
        
        # Factor 2: Silence threshold exceeded
        if silence_duration >= self.silence_threshold:
            # Check if we have minimum content
            total_text = " ".join(c.text for c in chunks)
            if len(total_text.strip()) >= self.min_length:
                return True
        
        # Factor 3: Sentence completion markers
        # Check both current chunk and last accumulated chunk for sentence end
        text = chunk.text.strip()
        last_chunk_text = chunks[-1].text.strip() if chunks else ""
        
        # Check if current chunk or last chunk ends with sentence marker
        has_sentence_end = (
            (text and text[-1] in ['.', '!', '?']) or
            (last_chunk_text and last_chunk_text[-1] in ['.', '!', '?'])
        )
        
        if has_sentence_end and silence_duration >= 1.0:  # Shorter threshold for sentence end
            return True
        
        # Factor 4: Maximum answer length (safety)
        # Check raw text length before normalization
        total_text = " ".join(c.text for c in chunks)
        if len(total_text) > self.max_length:
            return True
        
        return False
    
    async def _emit_answer(self, session_id: str) -> Optional[NormalizedAnswer]:
        """
        Combine chunks into normalized answer.
        
        Args:
            session_id: Session identifier
            
        Returns:
            NormalizedAnswer with clean text and metadata
        """
        chunks = self._pending_chunks.pop(session_id, [])
        self._last_chunk_time.pop(session_id, None)
        
        if not chunks:
            return None
        
        # Combine text
        raw_text = " ".join(c.text for c in chunks)
        
        # Truncate if exceeds max_length (safety limit)
        if len(raw_text) > self.max_length:
            raw_text = raw_text[:self.max_length].rsplit(' ', 1)[0]  # Truncate at word boundary
        
        # Normalize (lightweight, no LLM)
        normalized = self._normalize_text(raw_text)
        
        # Calculate metadata
        duration = (chunks[-1].timestamp_ms - chunks[0].timestamp_ms) / 1000.0
        hesitation_markers = self._detect_hesitation(normalized)
        avg_confidence = sum(c.confidence for c in chunks) / len(chunks) if chunks else 1.0
        
        return NormalizedAnswer(
            answer_text=normalized,
            metadata={
                "source": self.default_source,  # Use configurable source, not hard-coded "audio"
                "spoken_duration_sec": duration,
                "initial_silence_sec": 0,  # Could track if needed
                "hesitation_markers": hesitation_markers,
                "chunk_count": len(chunks),
                "avg_confidence": avg_confidence
            }
        )
    
    def _normalize_text(self, text: str) -> str:
        """
        Lightweight text normalization (no LLM).
        Removes common spoken artifacts.
        
        Args:
            text: Raw transcript text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Remove excessive filler words (keep first occurrence)
        fillers = ["um", "uh", "er", "ah", "like", "you know"]
        words = text.split()
        normalized_words = []
        prev_word = ""
        
        for word in words:
            word_lower = word.lower().strip(".,!?")
            if word_lower in fillers and word_lower == prev_word:
                continue  # Skip duplicate fillers
            normalized_words.append(word)
            prev_word = word_lower
        
        # Fix spacing around punctuation
        text = " ".join(normalized_words)
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # "word ." -> "word."
        text = re.sub(r'([.,!?])\s*([A-Z])', r'\1 \2', text)  # "word.Next" -> "word. Next"
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text.strip()
    
    def _detect_hesitation(self, text: str) -> List[str]:
        """
        Detect hesitation markers in normalized text.
        
        NOTE:
        Hesitation markers are heuristic hints only.
        They MUST NOT be used for scoring, ranking, or hiring decisions.
        These patterns may be culturally or contextually normal speech patterns.
        
        Args:
            text: Normalized text to analyze
            
        Returns:
            List of detected hesitation markers
        """
        hesitation_patterns = [
            r'\b(um|uh|er|ah)\b',
            r'\b(like|you know|sort of|kind of)\b',
            r'\b(maybe|perhaps|I think|I guess)\b'
        ]
        
        found = []
        for pattern in hesitation_patterns:
            matches = re.findall(pattern, text.lower())
            found.extend(matches)
        
        return list(set(found))  # Unique only
    
    async def flush_pending(self, session_id: str) -> Optional[NormalizedAnswer]:
        """
        Force emit any pending answer (e.g., on timeout or interview end).
        
        Args:
            session_id: Session identifier
            
        Returns:
            NormalizedAnswer if pending chunks exist, None otherwise
        """
        if session_id in self._pending_chunks and self._pending_chunks[session_id]:
            return await self._emit_answer(session_id)
        return None
    
    def normalize_text_input(
        self,
        text: str,
        source: str = "text"
    ) -> NormalizedAnswer:
        """
        Normalize plain text input (for typed text or pre-transcribed audio).
        
        Args:
            text: Plain text input
            source: Source modality ("text", "audio", "avatar")
            
        Returns:
            NormalizedAnswer with clean text and minimal metadata
        """
        normalized = self._normalize_text(text)
        return NormalizedAnswer(
            answer_text=normalized,
            metadata={
                "source": source,
                "spoken_duration_sec": None,
                "initial_silence_sec": None,
                "hesitation_markers": self._detect_hesitation(normalized)
            }
        )


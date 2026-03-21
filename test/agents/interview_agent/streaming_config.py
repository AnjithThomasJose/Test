"""
Streaming Configuration Constants
Centralized configuration for video interview streaming system.
"""


class StreamingConfig:
    """Configuration constants for streaming interview system."""
    
    # Question Generation Settings (generous so LLM can finish; prevents streaming cut-off)
    QUESTION_MAX_TOKENS: int = 1024
    QUESTION_MAX_LENGTH: int = 580
    QUESTION_MIN_LENGTH: int = 20
    QUESTION_MIN_WORDS: int = 6
    # Max words per streamed chunk (yield up to this many words per yield for smoother UX)
    STREAMING_WORDS_PER_CHUNK: int = 12
    
    # Avatar Video Settings
    AVATAR_CHUNK_SIZE: int = 10000  # bytes (~7.5KB per chunk)
    AVATAR_POLL_INTERVAL_INITIAL: float = 0.5  # initial polling interval (fast start)
    AVATAR_POLL_INTERVAL_MAX: float = 2.0  # maximum polling interval (exponential backoff)
    AVATAR_POLL_INTERVAL: float = 1.0  # legacy: default interval (kept for backward compat)
    AVATAR_MAX_ATTEMPTS: int = 60  # maximum polling attempts
    AVATAR_DEFAULT_TIMEOUT: int = 60  # seconds
    AVATAR_USE_DIRECT_URL: bool = True  # send URL directly instead of base64 (recommended)
    AVATAR_USE_STREAMING_API: bool = True  # use HeyGen Streaming API (WebRTC) instead of Video Generation API
    
    # History Truncation Settings
    HISTORY_TRUNCATE_MESSAGES: int = 3  # last N messages for question generation
    
    # Incomplete Question Detection Thresholds
    INCOMPLETE_QUESTION_MIN_LENGTH: int = 20  # characters
    INCOMPLETE_QUESTION_MIN_WORDS: int = 6  # words
    
    # Incomplete endings: articles, prepositions, question words (no adverb list; -ly handled by rule in question_completer)
    INCOMPLETE_ENDINGS: list = [
        " a", " the", " an", " of", " in", " on", " for", " with", " at", " to", " from",
        " about", " by", " during", " effective", " successful", " important", " critical",
        " essential", " key", " main", " primary", " significant", " you", " where", " when",
        " how", " what", " which", " who", " why", " that", " this", " these", " those",
    ]
    
    # Common adjectives that indicate incomplete questions
    COMMON_ADJECTIVES: list = [
        "effective", "successful", "important", "critical", "essential",
        "key", "main", "primary", "significant", "crucial", "vital"
    ]
    
    # Default avatar and voice IDs
    DEFAULT_AVATAR_ID: str = "Abigail_standing_office_front"
    DEFAULT_VOICE_ID: str = "1bd001e7e50f421d891986aad5158bc8"
    
    # Video dimensions
    AVATAR_VIDEO_WIDTH: int = 400
    AVATAR_VIDEO_HEIGHT: int = 400
    AVATAR_VIDEO_ASPECT_RATIO: str = "1:1"

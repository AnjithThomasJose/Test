import os
import logging
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Initialize logger for settings module
log = logging.getLogger(__name__)

log.debug(f"OS-->APP_ENV-------------->{os.getenv('APP_ENV')}<--------------")
# Detect current environment (normalize to lowercase for consistency)
APP_ENV = os.getenv("APP_ENV", "development").lower()
# Allow alias: dev -> development
if APP_ENV == "dev":
    APP_ENV = "development"
log.debug(f"Fall Back-->APP_ENV-------------->{APP_ENV}<--------------")

# Map environment names to .env file names
# production -> .env.prod, development -> .env.development, qa -> .env.qa
ENV_FILE_MAP = {
    "production": ".env.prod",
    "development": ".env.development",
    "dev": ".env.dev",
    "qa": ".env.qa",
    "demo": ".env.demo",
}

# Centralized Gemini model selection (same across all environments)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Determine which .env file to load
if APP_ENV in ENV_FILE_MAP:
    dotenv_file = ENV_FILE_MAP[APP_ENV]
else:
    dotenv_file = f".env.{APP_ENV}"

# Load the correct .env file
if os.path.exists(dotenv_file):
    load_dotenv(dotenv_file)
    log.info(f"✅ Loaded environment file: {dotenv_file}")
elif APP_ENV == "development" and os.path.exists(".env.dev"):
    # Fallback to .env.dev if present
    load_dotenv(".env.dev")
    log.info(f"✅ Loaded fallback environment file: .env.dev")
else:
    log.warning(f"⚠️  No .env file found for {APP_ENV} (looked for: {dotenv_file})")


class Settings(BaseSettings):
    # App environment
    APP_ENV: str = APP_ENV
    # Explicit mock toggle (default off)
    USE_MOCKS: bool = False
    INTERNAL_API_KEY: Optional[str] = None
    # API Keys
    TOKEN: str
    GOOGLE_API_KEY: str
    
    # Gemini Configuration - Using flash for better quality (was flash-lite for speed)
    GEMINI_MODEL: str = DEFAULT_GEMINI_MODEL
    
    # Groq Configuration - Fast Llama 3 model for resume parsing
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # OpenAI Configuration for embeddings
    OPENAI_API_KEY: Optional[str] = None

    # Database / Firebase
    DB_URL: str

    # ChromaDB Config
    CHROMA_API_KEY: str
    CHROMA_TENANT: str
    CHROMA_DATABASE: str

    # LangChain / LangSmith Settings
    LANGCHAIN_TRACING_V2: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: Optional[str] = None
    LANGSMITH_TRACING: Optional[str] = None
    LANGSMITH_ENDPOINT: Optional[str] = None
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: Optional[str] = None

    # Langfuse Settings (LLM observability / tracing)
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    LANGFUSE_BASE_URL: Optional[str] = None  # alias; LANGFUSE_HOST takes precedence
    LANGFUSE_ENABLED: bool = True
    LANGFUSE_MODEL_EVAL_ENABLED: bool = False  # Run LLM-as-judge scoring after pipeline
    LANGFUSE_TRACING_ENVIRONMENT: str = "default"  # e.g. development, qa, production — separates costs in Langfuse

    # LLM Cache Configuration
    USE_LLM_CACHE: bool = True
    LLM_CACHE_BACKEND: str = "sqlite"  # options: "redis" | "sqlite" | "memory"
    LLM_CACHE_TTL_SECONDS: int = 1800  # 30 minutes default
    REDIS_URL: Optional[str] = None
    SQLITE_CACHE_PATH: str = "/tmp/kafin_llm_cache.sqlite"
    
    # Centralized Cache Configuration (for all caches)
    CACHE_TTL_SECONDS: int = 3600  # 1 hour default for agent caches
    CACHE_TTL_MINUTES: int = 60  # 1 hour default (alternative format)
    MAX_CACHE_ENTRIES: int = 1000  # Default max entries for in-memory caches
    CACHE_CLEANUP_INTERVAL_SECONDS: int = 300  # 5 minutes - cleanup interval for SQLite
    
    # Resume Summary Cache Configuration (longer TTL since resumes rarely change)
    RESUME_SUMMARY_CACHE_TTL_SECONDS: int = 86400  # 24 hours - resumes are stable data

    # Google Search API Settings (loaded from environment for security)
    GOOGLE_CSE_ID: Optional[str] = None
    GOOGLE_SEARCH_API_KEY: Optional[str] = None
    
    # Security Configuration
    NODE_API_TOKEN: Optional[str] = None  # Token for verifying requests from Node API
    CALLBACK_WHITELIST_DOMAINS: Optional[str] = None  # Comma-separated list of whitelisted callback domains
    
    # Video Interview Configuration
    VIDEOSDK_API_KEY: Optional[str] = None  # VideoSDK API key for video meetings
    VIDEOSDK_API_SECRET: Optional[str] = None  # VideoSDK API secret for JWT token generation
    HEYGEN_API_KEY: Optional[str] = None  # HeyGen API key for avatar video generation
    
    # Avatar Video Generation Settings
    AVATAR_ENABLED: bool = True  # Enable/disable avatar video generation
    AVATAR_GENERATION_TIMEOUT: int = 60  # Maximum seconds to wait for avatar generation
    AVATAR_FALLBACK_TO_URL: bool = True  # Fallback to video URL if base64 conversion fails
    HEYGEN_MOCK_MODE: bool = False  # Use mock mode for testing (returns fake video chunks)
    
    # Interview Model Configuration
    INTERVIEW_QUESTION_MODEL: str = DEFAULT_GEMINI_MODEL  # Model for question generation
    INTERVIEW_ANALYSIS_MODEL: str = DEFAULT_GEMINI_MODEL  # Model for response analysis
    
    # Recruitment Search Configuration
    RECRUITMENT_DEFAULT_TOP_K: int = 50  # Default number of candidates to retrieve (1-100)
    RECRUITMENT_MAX_TOP_K: int = 100  # Maximum number of candidates allowed per request

    # Pipeline timeout (seconds) - job-matching flows need longer due to many LLM calls
    PIPELINE_TIMEOUT_SECONDS: int = 300  # Default for non-job-matching flows
    PIPELINE_TIMEOUT_JOB_MATCHING_SECONDS: int = 600  # Job-matching (career flow, compare flow)

    # Quota manager: when True, bypass internal quota/backoff checks and let requests reach the API.
    # Use when you have sufficient Gemini quota but hit false "Quota exceeded" from internal limits
    # (per_minute_request_limit=60, backoff after 429). API will still enforce its own limits.
    DISABLE_QUOTA_CHECK: bool = False

    # Gemini API tier: 1 = free/default (60 RPM), 2 = paid tier 2 (2000 RPM, 8M TPM).
    # Used by quota_manager when DISABLE_QUOTA_CHECK=False to set internal limits.
    GEMINI_TIER: int = 1

    class Config:
        # Use the same env file mapping logic
        env_file = ENV_FILE_MAP.get(APP_ENV, f".env.{APP_ENV}")
        case_sensitive = True
        extra = 'ignore' 

# Export a global instance with better error handling
try:
    settings = Settings()
    
    # Load Google Search API settings from environment (with fallback for backward compatibility)
    # These should be moved to environment variables in production
    if not settings.GOOGLE_CSE_ID:
        settings.GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "54e1ac8ae28004829")
    if not settings.GOOGLE_SEARCH_API_KEY:
        settings.GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY", "AIzaSyAtRjb_GbpmpGKmxZWPZCVQwIvRumbW6MU")
    
    # Warn if using hardcoded fallback values in production
    if APP_ENV in ["production", "qa"]:
        if os.getenv("GOOGLE_CSE_ID") is None:
            log.warning("⚠️  GOOGLE_CSE_ID not set in environment, using fallback value")
        if os.getenv("GOOGLE_SEARCH_API_KEY") is None:
            log.warning("⚠️  GOOGLE_SEARCH_API_KEY not set in environment, using fallback value")
            
except Exception as e:
    log.error(f"\n❌ ERROR: Failed to load settings!")
    log.error(f"Error type: {type(e).__name__}")
    log.error(f"Error details: {str(e)}\n")
    
    # Check which required fields are missing
    required_fields = ['TOKEN', 'GOOGLE_API_KEY', 'DB_URL', 'CHROMA_API_KEY', 'CHROMA_TENANT', 'CHROMA_DATABASE']
    missing_fields = []
    for field in required_fields:
        if not os.getenv(field):
            missing_fields.append(field)
    
    if missing_fields:
        env_file_name = ENV_FILE_MAP.get(APP_ENV, f".env.{APP_ENV}")
        log.warning(f"⚠️  Missing required environment variables: {', '.join(missing_fields)}")
        log.warning(f"💡 Make sure these are set in your environment or {env_file_name} file\n")
    
    raise
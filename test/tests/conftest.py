"""
Shared pytest fixtures for KA Agents test suite.
Issue 7.1: LangGraph workflow tests infrastructure.

NOTE: This conftest provides common fixtures. Individual test files may need
to handle their own module mocking depending on what they're testing.
"""
import pytest
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any

# Add agents directory to path for imports
agents_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, agents_dir)


@pytest.fixture
def minimal_state() -> Dict[str, Any]:
    """
    Minimal valid state for graph invocation.
    Contains only the required fields to start a graph run.
    """
    return {
        "body": {
            "uid": "test-user-123",
            "tenant_id": "test-tenant-456",
            "resume_url": "https://example.com/test-resume.pdf",
            "callback_url": "https://example.com/callback",
        },
        "uid": "test-user-123",
        "tenant_id": "test-tenant-456",
        "callback_url": "https://example.com/callback",
        "resume_url": "https://example.com/test-resume.pdf",
        "next": None,
    }


@pytest.fixture
def state_with_next() -> Dict[str, Any]:
    """State with a valid 'next' field set."""
    return {
        "body": {
            "uid": "test-user-123",
            "tenant_id": "test-tenant-456",
        },
        "uid": "test-user-123",
        "tenant_id": "test-tenant-456",
        "next": "validate_resume",
    }


@pytest.fixture
def state_with_invalid_next() -> Dict[str, Any]:
    """State with an invalid 'next' field."""
    return {
        "body": {
            "uid": "test-user-123",
            "tenant_id": "test-tenant-456",
        },
        "uid": "test-user-123",
        "tenant_id": "test-tenant-456",
        "next": "nonexistent_node_xyz",
    }


@pytest.fixture
def state_with_user_interests() -> Dict[str, Any]:
    """State for Flow 2 (second call with user interests)."""
    return {
        "body": {
            "uid": "test-user-123",
            "tenant_id": "test-tenant-456",
        },
        "uid": "test-user-123",
        "tenant_id": "test-tenant-456",
        "next": "validate_resume",
        "user_interests": [
            {"question": "What interests you?", "answer": "AI and ML"}
        ],
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for tests that don't need real API calls."""
    return '{"status": "success", "result": "mocked"}'


@pytest.fixture
def mock_structured_resume():
    """Mock structured resume data."""
    return {
        "personal_info": {
            "name": "Test User",
            "email": "test@example.com",
        },
        "experience": [
            {
                "company": "Test Corp",
                "title": "Software Engineer",
                "duration": "2 years",
            }
        ],
        "education": [
            {
                "institution": "Test University",
                "degree": "BS Computer Science",
            }
        ],
        "skills": ["Python", "JavaScript", "SQL"],
    }


@pytest.fixture
def mock_chroma_results():
    """Mock Chroma query results."""
    return {
        "ids": [["doc-1", "doc-2"]],
        "documents": [["Document 1 content", "Document 2 content"]],
        "metadatas": [[{"tenant_id": "test-tenant-456"}, {"tenant_id": "test-tenant-456"}]],
        "distances": [[0.1, 0.2]],
    }


@pytest.fixture
def mock_session_manager():
    """Mock session manager for tests."""
    mock = MagicMock()
    mock.get_session.return_value = {
        "session_id": "test-session-123",
        "uid": "test-user-123",
        "tenant_id": "test-tenant-456",
        "created_at": "2024-01-01T00:00:00Z",
    }
    mock.update_heartbeat.return_value = None
    mock.create_session.return_value = "test-session-123"
    return mock


@pytest.fixture
def mock_invoke_llm():
    """
    Fixture to patch invoke_llm across the codebase.
    Returns a mock that can be configured per test.
    """
    with patch("models.llm_invoker.invoke_llm") as mock:
        mock.return_value = '{"status": "success"}'
        yield mock


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client for vector store tests."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    mock_collection.add.return_value = None
    mock_collection.upsert.return_value = None
    mock_client.get_or_create_collection.return_value = mock_collection
    return mock_client


# Async fixtures for async tests
@pytest.fixture
def mock_async_invoke_llm():
    """Async mock for invoke_llm."""
    async_mock = AsyncMock(return_value='{"status": "success"}')
    return async_mock


@pytest.fixture
def graph_config():
    """Standard config for graph invocation."""
    from core.config import GRAPH_RECURSION_LIMIT
    return {"recursion_limit": GRAPH_RECURSION_LIMIT}


# Issue 7.4: Cleanup fixtures for global state and resources

@pytest.fixture(autouse=True)
def cleanup_global_state():
    """
    Ensure global state is cleaned up between tests.
    
    This fixture runs automatically for every test to prevent
    state leakage between tests.
    """
    yield
    
    try:
        from agents.agents.interview_agent.anti_repetition import _QUESTION_FINGERPRINTS
        _QUESTION_FINGERPRINTS.clear()
    except (ImportError, AttributeError):
        pass
    
    try:
        from agents.agents.interest_filler_agent import _context_cache
        _context_cache.clear()
    except (ImportError, AttributeError):
        pass
    
    try:
        from app import token_cache
        token_cache.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def thread_executor():
    """
    Provide a managed thread executor that auto-cleans up.
    
    Use this fixture instead of creating ThreadPoolExecutor directly
    in tests to ensure proper cleanup.
    """
    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=5, thread_name_prefix="test_executor")
    yield executor
    executor.shutdown(wait=True, cancel_futures=True)


@pytest.fixture
def event_loop_cleanup():
    """
    Fixture to ensure event loop cleanup after async tests.
    
    Use for tests that create their own event loops.
    """
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    
    try:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
        pass
    finally:
        loop.close()
        asyncio.set_event_loop(None)


@pytest.fixture
def mock_settings():
    """
    Provide mock settings for tests that need specific configuration.
    
    Patches the settings module to avoid loading real environment variables.
    """
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.GROQ_API_KEY = "test-groq-key"
    mock.GROQ_MODEL = "llama-3.3-70b-versatile"
    mock.GOOGLE_API_KEY = "test-google-key"
    mock.GEMINI_MODEL = "gemini-2.5-flash"
    mock.CHROMA_HOST = "localhost"
    mock.CHROMA_PORT = 8000
    mock.LANGFUSE_MODEL_EVAL_ENABLED = False
    
    with patch('settings.settings', mock):
        yield mock


@pytest.fixture(scope="session")
def session_cleanup():
    """
    Session-scoped cleanup that runs once at the end of all tests.
    
    Use for cleaning up resources that are expensive to create/destroy.
    """
    yield
    
    try:
        from core.supervisor_agent import _sync_wrapper_executor
        _sync_wrapper_executor.shutdown(wait=False, cancel_futures=True)
    except (ImportError, AttributeError):
        pass

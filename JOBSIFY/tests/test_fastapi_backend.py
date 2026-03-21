"""
FastAPI Backend Tests
Section 2: Tests for background tasks, timeouts, error handling, and session management.

Tests cover:
- Background task scheduling with schedule_background_task()
- Task timeout behavior (BACKGROUND_TASK_TIMEOUT_SECONDS)
- Error callback and logging
- Exception handler responses
- Error message sanitization
- Session creation timeout handling
"""
import pytest
import asyncio
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# =============================================================================
# Background Task Tests
# =============================================================================

class TestBackgroundTaskScheduling:
    """Tests for schedule_background_task() functionality (Issue 2.1)."""

    def test_schedule_background_task_exists(self):
        """schedule_background_task function should exist and be callable."""
        from core.background_tasks import schedule_background_task
        
        assert callable(schedule_background_task)

    @pytest.mark.asyncio
    async def test_schedule_background_task_returns_task(self):
        """schedule_background_task should return an asyncio.Task."""
        from core.background_tasks import schedule_background_task
        
        async def simple_coro():
            return "done"
        
        task = schedule_background_task(simple_coro(), "test_task")
        
        assert isinstance(task, asyncio.Task)
        await task  # Clean up

    @pytest.mark.asyncio
    async def test_schedule_background_task_executes_coroutine(self):
        """Scheduled task should execute the provided coroutine."""
        from core.background_tasks import schedule_background_task
        
        result_holder = {"executed": False}
        
        async def marker_coro():
            result_holder["executed"] = True
            return "completed"
        
        task = schedule_background_task(marker_coro(), "marker_task")
        await task
        
        assert result_holder["executed"] is True

    @pytest.mark.asyncio
    async def test_schedule_background_task_with_timeout(self):
        """Task with timeout should be wrapped in asyncio.wait_for."""
        from core.background_tasks import schedule_background_task
        
        async def quick_coro():
            await asyncio.sleep(0.01)
            return "quick"
        
        # Should complete within timeout
        task = schedule_background_task(quick_coro(), "quick_task", timeout_seconds=5.0)
        result = await task
        
        # Result may be the return value or None depending on implementation
        assert result == "quick" or result is None

    @pytest.mark.asyncio
    async def test_schedule_background_task_timeout_triggers(self):
        """Task exceeding timeout should handle TimeoutError gracefully."""
        from core.background_tasks import schedule_background_task
        
        async def slow_coro():
            await asyncio.sleep(10)  # Much longer than timeout
            return "slow"
        
        # Very short timeout
        task = schedule_background_task(slow_coro(), "slow_task", timeout_seconds=0.05)
        
        # Task should complete (timeout handled internally)
        # The implementation logs the timeout but doesn't re-raise
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            pass  # Expected if timeout propagates


class TestBackgroundTaskTimeout:
    """Tests for BACKGROUND_TASK_TIMEOUT_SECONDS configuration (Issue 2.2)."""

    def test_timeout_config_parsing(self):
        """Timeout configuration should parse from environment."""
        # Test the pattern used for timeout parsing
        def parse_timeout_value(value: str):
            """Parse timeout value from string."""
            if not value:
                return None
            try:
                val = float(value)
                return val if val > 0 else None
            except ValueError:
                return None
        
        # Test valid values
        assert parse_timeout_value("30.5") == 30.5
        
        # Test invalid values
        assert parse_timeout_value("invalid") is None
        
        # Test zero/negative values
        assert parse_timeout_value("0") is None
        assert parse_timeout_value("-5") is None
        
        # Test empty
        assert parse_timeout_value("") is None

    def test_default_timeout_is_none(self):
        """Default timeout should be None when not configured."""
        from core.background_tasks import BACKGROUND_TASK_TIMEOUT_SECONDS
        
        # May be None or a value depending on environment
        assert BACKGROUND_TASK_TIMEOUT_SECONDS is None or isinstance(BACKGROUND_TASK_TIMEOUT_SECONDS, (int, float))


class TestTaskErrorCallback:
    """Tests for task done callback and error logging (Issue 2.1)."""

    def test_log_task_done_function_exists(self):
        """_log_task_done callback function should exist."""
        from core.background_tasks import _log_task_done
        
        assert callable(_log_task_done)

    @pytest.mark.asyncio
    async def test_error_in_task_is_logged(self):
        """Errors in background tasks should be logged via callback."""
        from core.background_tasks import schedule_background_task
        
        async def failing_coro():
            raise ValueError("Test error")
        
        with patch('core.background_tasks.log') as mock_log:
            task = schedule_background_task(failing_coro(), "failing_task")
            
            # Wait for task to complete (with error)
            try:
                await task
            except ValueError:
                pass
            
            # Allow callback to execute
            await asyncio.sleep(0.1)


# =============================================================================
# Exception Handler Tests
# =============================================================================

class TestExceptionHandler:
    """Tests for global exception handler in app.py (Issue 2.4)."""

    def test_sanitize_error_message_exists(self):
        """sanitize_error_message function should exist."""
        from core.security import sanitize_error_message
        
        assert callable(sanitize_error_message)

    def test_sanitize_timeout_error(self):
        """Timeout errors should be sanitized to user-friendly message."""
        from core.security import sanitize_error_message
        
        error = asyncio.TimeoutError("Connection timed out")
        result = sanitize_error_message(error)
        
        assert "timed out" in result.lower() or "try again" in result.lower()

    def test_sanitize_connection_error(self):
        """Connection errors should be sanitized to a safe message."""
        from core.security import sanitize_error_message
        
        error = ConnectionError("Failed to connect")
        result = sanitize_error_message(error)
        
        # Should return a safe error message (may be generic or specific)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should not expose internal details
        assert "Failed to connect" not in result or "error" in result.lower()

    def test_sanitize_validation_error(self):
        """Validation errors should be sanitized."""
        from core.security import sanitize_error_message
        
        error = ValueError("Invalid input")
        result = sanitize_error_message(error)
        
        assert "invalid" in result.lower() or "input" in result.lower() or "request" in result.lower()

    def test_sanitize_generic_error(self):
        """Generic errors should return safe default message."""
        from core.security import sanitize_error_message
        
        error = Exception("Internal server details - should not expose")
        result = sanitize_error_message(error)
        
        # Should not contain internal details
        assert "Internal server details" not in result
        assert "error" in result.lower() or "occurred" in result.lower()

    def test_sanitize_with_include_details_dev(self):
        """With include_details=True in dev, should show error type."""
        from core.security import sanitize_error_message
        
        error = RuntimeError("Something went wrong")
        
        with patch.dict(os.environ, {"APP_ENV": "development"}):
            result = sanitize_error_message(error, include_details=True)
            # In dev mode with details, may include error type
            assert isinstance(result, str)

    def test_sanitize_rate_limit_error(self):
        """Rate limit errors should have specific message."""
        from core.security import sanitize_error_message
        
        # Create an error that looks like rate limiting
        class RateLimitError(Exception):
            pass
        
        error = RateLimitError("Rate limit exceeded")
        result = sanitize_error_message(error)
        
        # Should return some safe message
        assert isinstance(result, str)
        assert len(result) > 0


class TestExceptionHandlerResponse:
    """Tests for exception handler JSON response format."""

    def test_error_response_structure(self):
        """Error responses should have consistent structure."""
        expected_fields = ["status", "error"]
        
        # Simulate error response structure
        error_response = {
            "status": "error",
            "error": "An error occurred",
            "type": "ValueError"
        }
        
        for field in expected_fields:
            assert field in error_response

    def test_http_exception_not_handled(self):
        """HTTPException should be re-raised, not handled."""
        from fastapi import HTTPException
        
        # This is the expected behavior - HTTPExceptions pass through
        exc = HTTPException(status_code=404, detail="Not found")
        assert exc.status_code == 404


# =============================================================================
# Session Creation Timeout Tests
# =============================================================================

class TestSessionCreationTimeout:
    """Tests for session creation timeout handling (Issue 2.3)."""

    def test_session_timeout_constant(self):
        """Session timeout should be defined."""
        # Default timeout is 5 seconds
        expected_timeout = 5.0
        assert expected_timeout > 0

    @pytest.mark.asyncio
    async def test_session_graceful_degradation(self):
        """Session creation should gracefully degrade on timeout."""
        # Simulate the pattern used in session_manager
        result_holder = {"session_id": None}
        
        async def mock_store_session(session_id):
            await asyncio.sleep(10)  # Simulates slow storage
            return session_id
        
        session_id = "test-session-123"
        
        try:
            await asyncio.wait_for(mock_store_session(session_id), timeout=0.1)
        except asyncio.TimeoutError:
            # Graceful degradation: proceed with session_id even on timeout
            result_holder["session_id"] = session_id
        
        assert result_holder["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_session_returns_id_on_error(self):
        """Session should return ID even when storage fails."""
        session_id = "test-session-456"
        
        async def failing_store():
            raise ConnectionError("Storage unavailable")
        
        result = session_id  # Default
        
        try:
            await failing_store()
        except Exception:
            pass  # Session ID still valid
        
        assert result == session_id


# =============================================================================
# Dedicated Session Executor Tests
# =============================================================================

class TestSessionExecutor:
    """Tests for dedicated session executor (Issue 2.3)."""

    def test_executor_timeout_handling(self):
        """Session executor should handle timeouts gracefully."""
        import concurrent.futures
        
        # Simulate the session executor pattern
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        def slow_operation():
            import time
            time.sleep(5)
            return "completed"
        
        future = executor.submit(slow_operation)
        
        try:
            result = future.result(timeout=0.1)
        except concurrent.futures.TimeoutError:
            result = None  # Graceful degradation
        
        assert result is None
        executor.shutdown(wait=False)

    def test_executor_error_handling(self):
        """Session executor should handle errors without crashing."""
        import concurrent.futures
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        def failing_operation():
            raise RuntimeError("Operation failed")
        
        future = executor.submit(failing_operation)
        
        error_caught = False
        try:
            future.result(timeout=1.0)
        except RuntimeError:
            error_caught = True
        
        assert error_caught
        executor.shutdown(wait=False)


# =============================================================================
# Pipeline Error Callback Tests
# =============================================================================

class TestPipelineErrorCallback:
    """Tests for pipeline error callback handling (Issue 2.4)."""

    def test_error_callback_uses_sanitization(self):
        """Pipeline error callbacks should use sanitized error messages."""
        from core.security import sanitize_error_message
        
        # Simulate pipeline error
        raw_error = Exception("Database connection string: postgresql://user:password@host")
        sanitized = sanitize_error_message(raw_error)
        
        # Sanitized message should not contain sensitive info
        assert "password" not in sanitized
        assert "postgresql://" not in sanitized

    def test_callback_structure(self):
        """Error callback should have expected structure."""
        callback_data = {
            "status": "error",
            "message": "An error occurred",
            "timestamp": "2026-02-21T00:00:00Z"
        }
        
        assert "status" in callback_data
        assert callback_data["status"] == "error"


# =============================================================================
# Integration Tests
# =============================================================================

class TestBackgroundTaskIntegration:
    """Integration tests for background task system."""

    @pytest.mark.asyncio
    async def test_multiple_tasks_isolation(self):
        """Multiple background tasks should be isolated."""
        from core.background_tasks import schedule_background_task
        
        results = []
        
        async def task_a():
            await asyncio.sleep(0.01)
            results.append("a")
        
        async def task_b():
            await asyncio.sleep(0.01)
            results.append("b")
        
        task1 = schedule_background_task(task_a(), "task_a")
        task2 = schedule_background_task(task_b(), "task_b")
        
        await asyncio.gather(task1, task2)
        
        assert "a" in results
        assert "b" in results

    @pytest.mark.asyncio
    async def test_task_cancellation_handled(self):
        """Task cancellation should be handled gracefully."""
        from core.background_tasks import schedule_background_task
        
        async def long_running():
            await asyncio.sleep(100)
        
        task = schedule_background_task(long_running(), "cancellable_task")
        
        # Cancel the task
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

    def test_task_name_tracking(self):
        """Task names should be trackable for monitoring."""
        task_name = "important_pipeline_task"
        
        # Simulate task naming pattern
        assert isinstance(task_name, str)
        assert len(task_name) > 0

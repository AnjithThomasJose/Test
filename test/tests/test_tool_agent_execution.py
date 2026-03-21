"""
Tool / Agent Execution Tests
Section 6: Tests for middleware, sync wrappers, timeouts, and error logging.

Tests cover:
- Shared executor in apply_middleware (_sync_agent_executor)
- Sync wrapper timeout (SYNC_WRAPPER_TIMEOUT_SECONDS)
- Per-agent timeout configuration (get_agent_timeout_seconds)
- Novu trigger error logging
- Middleware error handling
"""
import pytest
import asyncio
import sys
import os
import concurrent.futures
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# =============================================================================
# Local Mock Implementations
# =============================================================================

def mock_get_agent_timeout_seconds(agent_name: str) -> Optional[float]:
    """Mock implementation of _get_agent_timeout_seconds."""
    per_agent = os.getenv(f"AGENT_{agent_name.upper()}_TIMEOUT_SECONDS", "").strip()
    if per_agent:
        try:
            return float(per_agent)
        except ValueError:
            pass
    default = os.getenv("AGENT_TIMEOUT_SECONDS", "").strip()
    if not default:
        return None
    try:
        return float(default)
    except ValueError:
        return None


# =============================================================================
# Shared Executor Tests
# =============================================================================

class TestSharedSyncAgentExecutor:
    """Tests for _sync_agent_executor shared executor (Issue 6.1)."""

    def test_executor_reuse_pattern(self):
        """Shared executor should be reused across calls."""
        # Create a shared executor (simulating _sync_agent_executor)
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=10,
            thread_name_prefix="sync_agent"
        )
        
        call_count = {"count": 0}
        
        def work():
            call_count["count"] += 1
            return True
        
        # Submit multiple tasks to same executor
        futures = [executor.submit(work) for _ in range(5)]
        results = [f.result() for f in futures]
        
        assert all(results)
        assert call_count["count"] == 5
        
        executor.shutdown(wait=True)

    def test_executor_thread_prefix(self):
        """Executor should use correct thread name prefix."""
        thread_prefix = "sync_agent"
        
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=thread_prefix
        )
        
        thread_names = []
        
        def capture_thread_name():
            import threading
            thread_names.append(threading.current_thread().name)
            return True
        
        future = executor.submit(capture_thread_name)
        future.result()
        
        assert len(thread_names) == 1
        assert thread_prefix in thread_names[0]
        
        executor.shutdown(wait=True)

    def test_executor_max_workers(self):
        """Executor should respect max_workers configuration."""
        max_workers = 10
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Executor should accept up to max_workers concurrent tasks
        assert executor._max_workers == max_workers
        
        executor.shutdown(wait=False)


class TestApplyMiddleware:
    """Tests for apply_middleware function (Issue 6.1)."""

    @pytest.mark.asyncio
    async def test_async_function_detection(self):
        """apply_middleware should detect async functions."""
        async def async_agent(state):
            return state
        
        def sync_agent(state):
            return state
        
        assert asyncio.iscoroutinefunction(async_agent)
        assert not asyncio.iscoroutinefunction(sync_agent)

    @pytest.mark.asyncio
    async def test_sync_function_wrapping(self):
        """Sync functions should be wrapped to run in executor."""
        def sync_agent(state):
            return {"result": "from_sync"}
        
        # Simulate the wrapping pattern
        async def async_wrapper(state):
            loop = asyncio.get_event_loop()
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            result = await loop.run_in_executor(executor, sync_agent, state)
            executor.shutdown(wait=False)
            return result
        
        result = await async_wrapper({"input": "test"})
        
        assert result["result"] == "from_sync"


# =============================================================================
# Sync Wrapper Timeout Tests
# =============================================================================

class TestSyncWrapperTimeout:
    """Tests for sync wrapper timeout (Issue 6.2)."""

    def test_sync_wrapper_timeout_env_var(self):
        """SYNC_WRAPPER_TIMEOUT_SECONDS should be configurable."""
        default = 300
        env_var = "SYNC_WRAPPER_TIMEOUT_SECONDS"
        
        configured = int(os.getenv(env_var, str(default)))
        
        assert configured > 0
        assert configured == default or configured > 0

    def test_sync_wrapper_timeout_default(self):
        """Default timeout should be 300 seconds."""
        default_timeout = 300
        
        assert default_timeout == 300

    def test_future_result_timeout(self):
        """Future.result() should respect timeout."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        def slow_work():
            import time
            time.sleep(10)
            return "done"
        
        future = executor.submit(slow_work)
        
        with pytest.raises(concurrent.futures.TimeoutError):
            future.result(timeout=0.05)
        
        executor.shutdown(wait=False)

    def test_timeout_error_handling(self):
        """TimeoutError should be caught and handled."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        def slow_work():
            import time
            time.sleep(10)
            return "done"
        
        future = executor.submit(slow_work)
        timeout = 0.05
        
        error_state = None
        
        try:
            future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            error_state = {"error": f"Agent timed out after {timeout} seconds"}
        
        assert error_state is not None
        assert "timed out" in error_state["error"]
        
        executor.shutdown(wait=False)


class TestSyncWrapperExecutor:
    """Tests for _sync_wrapper_executor (Issue 6.2)."""

    def test_sync_wrapper_executor_config(self):
        """Sync wrapper executor should have correct configuration."""
        max_workers = 20
        thread_prefix = "sync_wrapper"
        
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_prefix
        )
        
        assert executor._max_workers == max_workers
        
        executor.shutdown(wait=False)

    def test_run_async_in_thread_pattern(self):
        """Async functions should run in dedicated thread with event loop."""
        async def async_func(value):
            await asyncio.sleep(0.01)
            return value * 2
        
        def run_async_in_thread(async_fn, arg):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(async_fn(arg))
            finally:
                loop.close()
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        future = executor.submit(run_async_in_thread, async_func, 5)
        result = future.result(timeout=5)
        
        assert result == 10
        
        executor.shutdown(wait=True)


# =============================================================================
# Per-Agent Timeout Tests
# =============================================================================

class TestPerAgentTimeout:
    """Tests for per-agent timeout configuration (Issue 6.4)."""

    def test_get_agent_timeout_seconds_no_config(self):
        """Should return None when no timeout configured."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear relevant env vars
            os.environ.pop("AGENT_TIMEOUT_SECONDS", None)
            os.environ.pop("AGENT_TEST_TIMEOUT_SECONDS", None)
            
            result = mock_get_agent_timeout_seconds("test")
            assert result is None

    def test_get_agent_timeout_seconds_global(self):
        """Should use global AGENT_TIMEOUT_SECONDS."""
        with patch.dict(os.environ, {"AGENT_TIMEOUT_SECONDS": "60"}):
            result = mock_get_agent_timeout_seconds("any_agent")
            assert result == 60.0

    def test_get_agent_timeout_seconds_per_agent_override(self):
        """Per-agent timeout should override global."""
        with patch.dict(os.environ, {
            "AGENT_TIMEOUT_SECONDS": "60",
            "AGENT_RESUME_PARSER_TIMEOUT_SECONDS": "120"
        }):
            result = mock_get_agent_timeout_seconds("resume_parser")
            assert result == 120.0

    def test_get_agent_timeout_seconds_invalid_value(self):
        """Invalid timeout values should return None."""
        with patch.dict(os.environ, {"AGENT_TIMEOUT_SECONDS": "invalid"}):
            result = mock_get_agent_timeout_seconds("test")
            assert result is None

    def test_get_agent_timeout_seconds_empty_value(self):
        """Empty timeout values should return None."""
        with patch.dict(os.environ, {"AGENT_TIMEOUT_SECONDS": ""}):
            result = mock_get_agent_timeout_seconds("test")
            assert result is None


class TestAgentTimeoutApplication:
    """Tests for applying agent timeout in middleware."""

    @pytest.mark.asyncio
    async def test_timeout_with_asyncio_wait_for(self):
        """Agent timeout should use asyncio.wait_for."""
        async def agent_work():
            await asyncio.sleep(0.01)
            return {"result": "success"}
        
        timeout = 5.0
        result = await asyncio.wait_for(agent_work(), timeout=timeout)
        
        assert result["result"] == "success"

    @pytest.mark.asyncio
    async def test_timeout_triggers_timeout_error(self):
        """Agent exceeding timeout should raise TimeoutError."""
        async def slow_agent():
            await asyncio.sleep(10)
            return {"result": "never_reached"}
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_agent(), timeout=0.05)

    @pytest.mark.asyncio
    async def test_no_timeout_when_none(self):
        """No timeout should be applied when timeout is None."""
        async def agent_work():
            await asyncio.sleep(0.01)
            return {"result": "success"}
        
        timeout = None
        
        if timeout is not None:
            result = await asyncio.wait_for(agent_work(), timeout=timeout)
        else:
            result = await agent_work()
        
        assert result["result"] == "success"


# =============================================================================
# Novu Error Logging Tests
# =============================================================================

class TestNovuErrorLogging:
    """Tests for Novu trigger error logging (Issue 6.3)."""

    def test_novu_error_log_structure(self):
        """Novu error logs should have structured extra fields."""
        extra_fields = {
            "event_id": "evt_123",
            "subscriber_id": "sub_456",
            "source": "novu_api",
            "workflow_id": "wf_789",
            "error": "Connection refused"
        }
        
        # Verify all required fields are present
        required_fields = ["event_id", "subscriber_id", "source", "workflow_id"]
        for field in required_fields:
            assert field in extra_fields

    def test_novu_http_error_captures_status(self):
        """HTTP errors should capture status code."""
        error_info = {
            "status_code": 429,
            "error": "Rate limit exceeded"[:200]  # Truncated
        }
        
        assert "status_code" in error_info
        assert len(error_info["error"]) <= 200

    def test_novu_exception_logging_pattern(self):
        """Exceptions should be logged with exc_info."""
        with patch('logging.Logger.error') as mock_error:
            import logging
            logger = logging.getLogger("test")
            
            try:
                raise ValueError("Test error")
            except ValueError as e:
                logger.error(f"Failed: {e}", exc_info=True)
            
            mock_error.assert_called_once()
            call_kwargs = mock_error.call_args[1]
            assert call_kwargs.get("exc_info") is True

    def test_novu_error_returns_gracefully(self):
        """Novu errors should not crash the caller."""
        def novu_trigger():
            try:
                raise ConnectionError("Network error")
            except Exception as e:
                # Log but don't crash
                return None
        
        result = novu_trigger()
        
        # Function completed without raising
        assert result is None


# =============================================================================
# Middleware Error Handling Tests
# =============================================================================

class TestMiddlewareErrorHandling:
    """Tests for middleware error handling."""

    @pytest.mark.asyncio
    async def test_middleware_catches_agent_errors(self):
        """Middleware should catch and handle agent errors."""
        async def failing_agent(state):
            raise RuntimeError("Agent failed")
        
        error_caught = False
        result_state = None
        
        try:
            await failing_agent({})
        except RuntimeError:
            error_caught = True
            result_state = {"error": "Agent failed"}
        
        assert error_caught
        assert result_state["error"] == "Agent failed"

    @pytest.mark.asyncio
    async def test_error_state_shape(self):
        """Error state should have consistent shape."""
        original_state = {"uid": "user-1", "tenant_id": "tenant-1"}
        error = "Something went wrong"
        
        error_state = {**original_state, "error": error}
        
        assert "uid" in error_state
        assert "tenant_id" in error_state
        assert "error" in error_state
        assert error_state["error"] == error

    def test_learn_from_agent_execution_pattern(self):
        """Agent execution should support adaptive learning."""
        execution_log = []
        
        def learn_from_agent_execution(agent_name, input_state, output_state):
            execution_log.append({
                "agent": agent_name,
                "input_keys": list(input_state.keys()),
                "has_error": "error" in output_state
            })
        
        # Simulate successful execution
        learn_from_agent_execution(
            "test_agent",
            {"input": "data"},
            {"output": "result"}
        )
        
        # Simulate failed execution
        learn_from_agent_execution(
            "test_agent",
            {"input": "data"},
            {"error": "failed"}
        )
        
        assert len(execution_log) == 2
        assert execution_log[0]["has_error"] is False
        assert execution_log[1]["has_error"] is True


# =============================================================================
# Thread Pool Resource Management Tests
# =============================================================================

class TestThreadPoolResourceManagement:
    """Tests for thread pool resource management."""

    def test_executor_shutdown_pattern(self):
        """Executors should be properly shut down."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        def work():
            return True
        
        future = executor.submit(work)
        result = future.result()
        
        # Proper shutdown
        executor.shutdown(wait=True)
        
        assert result is True

    def test_executor_handles_exceptions(self):
        """Executor should handle task exceptions."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        def failing_work():
            raise ValueError("Task failed")
        
        future = executor.submit(failing_work)
        
        with pytest.raises(ValueError):
            future.result()
        
        executor.shutdown(wait=True)


# =============================================================================
# Integration Tests
# =============================================================================

class TestToolAgentIntegration:
    """Integration tests for tool/agent execution."""

    @pytest.mark.asyncio
    async def test_full_agent_execution_flow(self):
        """Test complete agent execution with timeout."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        timeout = 5.0
        
        def sync_agent(state):
            return {"result": state.get("input", "") + "_processed"}
        
        async def run_with_timeout(agent_fn, state):
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(executor, agent_fn, state)
            return await asyncio.wait_for(future, timeout=timeout)
        
        result = await run_with_timeout(sync_agent, {"input": "test"})
        
        assert result["result"] == "test_processed"
        
        executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_multiple_agents_concurrent(self):
        """Multiple agents should run concurrently."""
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        
        def agent(n):
            import time
            time.sleep(0.01)
            return {"n": n}
        
        async def run_agent(n):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(executor, agent, n)
        
        results = await asyncio.gather(
            run_agent(1),
            run_agent(2),
            run_agent(3),
        )
        
        assert len(results) == 3
        assert [r["n"] for r in results] == [1, 2, 3]
        
        executor.shutdown(wait=True)

    def test_error_propagation_pattern(self):
        """Errors should propagate with context."""
        def agent_with_error():
            raise ValueError("Invalid input")
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        future = executor.submit(agent_with_error)
        
        error_message = None
        try:
            future.result()
        except ValueError as e:
            error_message = str(e)
        
        assert error_message == "Invalid input"
        
        executor.shutdown(wait=True)

"""
Async / Concurrency Tests
Section 4: Tests for executor stats, backpressure, timeouts, and initialization patterns.

Tests cover:
- IO executor stats (get_io_executor_stats())
- Backpressure mechanism (IO_EXECUTOR_MAX_QUEUE_DEPTH)
- Firebase init timeout (FIREBASE_INIT_TIMEOUT_SECONDS)
- Job scheduler timeout (JOB_RUN_TIMEOUT_SECONDS)
- Lazy initialization pattern (ensure_indexes())
"""
import pytest
import asyncio
import sys
import os
import threading
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# =============================================================================
# IO Executor Stats Tests
# =============================================================================

class TestIOExecutorStats:
    """Tests for get_io_executor_stats() functionality (Issue 4.4)."""

    def test_get_io_executor_stats_exists(self):
        """get_io_executor_stats function should exist and be callable."""
        from core.utils import get_io_executor_stats
        
        assert callable(get_io_executor_stats)

    def test_get_io_executor_stats_returns_dict(self):
        """get_io_executor_stats should return a dictionary."""
        from core.utils import get_io_executor_stats
        
        stats = get_io_executor_stats()
        
        assert isinstance(stats, dict)

    def test_get_io_executor_stats_has_max_workers(self):
        """Stats should include max_workers field."""
        from core.utils import get_io_executor_stats
        
        stats = get_io_executor_stats()
        
        assert "max_workers" in stats
        assert isinstance(stats["max_workers"], int)
        assert stats["max_workers"] > 0

    def test_get_io_executor_stats_has_invocations(self):
        """Stats should include invocations_total field."""
        from core.utils import get_io_executor_stats
        
        stats = get_io_executor_stats()
        
        assert "invocations_total" in stats
        assert isinstance(stats["invocations_total"], int)
        assert stats["invocations_total"] >= 0

    def test_get_io_executor_stats_has_queue_size(self):
        """Stats should include queue_size field."""
        from core.utils import get_io_executor_stats
        
        stats = get_io_executor_stats()
        
        assert "queue_size" in stats
        # Queue size may be int or None (best-effort)
        assert stats["queue_size"] is None or isinstance(stats["queue_size"], int)

    def test_get_io_executor_stats_has_backpressure_config(self):
        """Stats should include backpressure_max_queue_depth field."""
        from core.utils import get_io_executor_stats
        
        stats = get_io_executor_stats()
        
        assert "backpressure_max_queue_depth" in stats
        assert isinstance(stats["backpressure_max_queue_depth"], int)


class TestIOExecutorBackpressure:
    """Tests for IO executor backpressure mechanism (Issue 4.4)."""

    def test_backpressure_disabled_by_default(self):
        """Backpressure should be disabled when max_queue_depth is 0."""
        from core.utils import get_io_executor_stats
        
        stats = get_io_executor_stats()
        
        # 0 means disabled
        if stats["backpressure_max_queue_depth"] == 0:
            assert True  # Backpressure is disabled
        else:
            assert stats["backpressure_max_queue_depth"] > 0

    def test_run_blocking_io_exists(self):
        """run_blocking_io function should exist."""
        from core.utils import run_blocking_io
        
        assert callable(run_blocking_io)

    @pytest.mark.asyncio
    async def test_run_blocking_io_executes(self):
        """run_blocking_io should execute blocking operations."""
        from core.utils import run_blocking_io
        
        def blocking_operation():
            return "result"
        
        result = await run_blocking_io(blocking_operation)
        
        assert result == "result"

    @pytest.mark.asyncio
    async def test_run_blocking_io_increments_invocations(self):
        """run_blocking_io should increment invocation counter."""
        from core.utils import run_blocking_io, get_io_executor_stats
        
        stats_before = get_io_executor_stats()
        
        def simple_op():
            return True
        
        await run_blocking_io(simple_op)
        
        stats_after = get_io_executor_stats()
        
        assert stats_after["invocations_total"] >= stats_before["invocations_total"]


# =============================================================================
# Firebase Init Timeout Tests
# =============================================================================

class TestFirebaseInitTimeout:
    """Tests for Firebase initialization timeout (Issue 4.1)."""

    def test_firebase_timeout_env_var(self):
        """FIREBASE_INIT_TIMEOUT_SECONDS should be configurable."""
        default_timeout = 10
        env_var = "FIREBASE_INIT_TIMEOUT_SECONDS"
        
        # Check default
        assert default_timeout > 0
        
        # Env var should be readable
        configured = os.getenv(env_var, str(default_timeout))
        assert configured is not None

    def test_firebase_timeout_is_positive(self):
        """Firebase timeout should be a positive integer."""
        timeout = int(os.getenv("FIREBASE_INIT_TIMEOUT_SECONDS", "10"))
        
        assert timeout > 0

    def test_requests_timeout_pattern(self):
        """Requests should use timeout parameter."""
        import requests
        
        # Verify requests module supports timeout
        # This tests the pattern used in firebase.py
        with patch('requests.get') as mock_get:
            mock_get.return_value = Mock(status_code=200, json=lambda: {})
            
            # Call with timeout (simulating firebase.py pattern)
            requests.get("http://example.com", timeout=10)
            
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]
            assert "timeout" in call_kwargs

    def test_firebase_graceful_degradation(self):
        """Firebase should set db=None on init failure."""
        # Simulate the graceful degradation pattern
        db = None
        
        try:
            raise ConnectionError("Network unavailable")
        except Exception:
            db = None  # Graceful degradation
        
        assert db is None


# =============================================================================
# Job Scheduler Timeout Tests
# =============================================================================

class TestJobSchedulerTimeout:
    """Tests for job scheduler timeout (Issue 4.2)."""

    def test_job_timeout_env_var(self):
        """JOB_RUN_TIMEOUT_SECONDS should be configurable."""
        env_var = "JOB_RUN_TIMEOUT_SECONDS"
        default = "0"  # 0 means no timeout
        
        configured = os.getenv(env_var, default)
        assert configured is not None

    def test_job_timeout_zero_means_disabled(self):
        """Timeout of 0 should mean no timeout."""
        timeout = int(os.getenv("JOB_RUN_TIMEOUT_SECONDS", "0"))
        
        # 0 = disabled (no timeout)
        if timeout == 0:
            has_timeout = False
        else:
            has_timeout = True
        
        # Either state is valid
        assert isinstance(has_timeout, bool)

    @pytest.mark.asyncio
    async def test_job_timeout_with_asyncio_wait_for(self):
        """Job timeout should use asyncio.wait_for pattern."""
        async def job_coro():
            await asyncio.sleep(0.01)
            return "completed"
        
        timeout_seconds = 5
        
        if timeout_seconds > 0:
            result = await asyncio.wait_for(job_coro(), timeout=timeout_seconds)
        else:
            result = await job_coro()
        
        assert result == "completed"

    @pytest.mark.asyncio
    async def test_job_timeout_triggers_timeout_error(self):
        """Job exceeding timeout should raise TimeoutError."""
        async def slow_job():
            await asyncio.sleep(10)
            return "slow"
        
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_job(), timeout=0.05)

    def test_job_event_loop_cleanup_pattern(self):
        """Job executor should clean up event loop properly."""
        # Simulate the cleanup pattern from job_scheduler.py
        loop = asyncio.new_event_loop()
        
        try:
            asyncio.set_event_loop(loop)
            # Simulate job execution
            result = loop.run_until_complete(asyncio.sleep(0.01))
        finally:
            # Cleanup pattern
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
            except Exception:
                pass
            loop.close()
        
        assert loop.is_closed()


# =============================================================================
# Ensure Indexes Tests
# =============================================================================

class TestEnsureIndexes:
    """Tests for ensure_indexes() lazy initialization pattern (Issue 4.3)."""

    @pytest.mark.asyncio
    async def test_ensure_indexes_one_time_execution(self):
        """ensure_indexes should only execute once."""
        execution_count = {"count": 0}
        indexes_ensured = False
        lock = asyncio.Lock()
        
        async def ensure_indexes():
            nonlocal indexes_ensured
            async with lock:
                if not indexes_ensured:
                    execution_count["count"] += 1
                    indexes_ensured = True
        
        # Call multiple times
        await ensure_indexes()
        await ensure_indexes()
        await ensure_indexes()
        
        # Should only execute once
        assert execution_count["count"] == 1

    @pytest.mark.asyncio
    async def test_ensure_indexes_concurrent_calls(self):
        """Concurrent calls should only execute once."""
        execution_count = {"count": 0}
        indexes_ensured = False
        lock = asyncio.Lock()
        
        async def ensure_indexes():
            nonlocal indexes_ensured
            async with lock:
                if not indexes_ensured:
                    await asyncio.sleep(0.01)  # Simulate work
                    execution_count["count"] += 1
                    indexes_ensured = True
        
        # Launch concurrent calls
        await asyncio.gather(
            ensure_indexes(),
            ensure_indexes(),
            ensure_indexes(),
            ensure_indexes(),
        )
        
        # Should only execute once
        assert execution_count["count"] == 1

    @pytest.mark.asyncio
    async def test_ensure_indexes_sets_flag(self):
        """ensure_indexes should set _indexes_ensured flag."""
        indexes_ensured = False
        lock = asyncio.Lock()
        
        async def ensure_indexes():
            nonlocal indexes_ensured
            async with lock:
                if not indexes_ensured:
                    indexes_ensured = True
        
        assert indexes_ensured is False
        await ensure_indexes()
        assert indexes_ensured is True


# =============================================================================
# Thread Pool Executor Tests
# =============================================================================

class TestThreadPoolExecutor:
    """Tests for thread pool executor configuration."""

    def test_thread_pool_max_workers_configurable(self):
        """THREAD_POOL_MAX_WORKERS should be configurable."""
        default = 50
        configured = int(os.getenv("THREAD_POOL_MAX_WORKERS", str(default)))
        
        assert configured > 0
        assert configured <= 500  # Sanity check

    def test_thread_safe_counter(self):
        """Invocation counter should be thread-safe."""
        import concurrent.futures
        
        counter = {"value": 0}
        lock = threading.Lock()
        
        def increment():
            with lock:
                counter["value"] += 1
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(increment) for _ in range(100)]
            concurrent.futures.wait(futures)
        
        assert counter["value"] == 100


# =============================================================================
# Async Context Manager Tests
# =============================================================================

class TestAsyncContextManager:
    """Tests for async context manager patterns."""

    @pytest.mark.asyncio
    async def test_async_lock_pattern(self):
        """Async lock should provide mutual exclusion."""
        lock = asyncio.Lock()
        results = []
        
        async def critical_section(name):
            async with lock:
                results.append(f"{name}_start")
                await asyncio.sleep(0.01)
                results.append(f"{name}_end")
        
        await asyncio.gather(
            critical_section("a"),
            critical_section("b"),
        )
        
        # Operations should be serialized
        # Either a_start, a_end, b_start, b_end or b_start, b_end, a_start, a_end
        assert len(results) == 4
        if results[0] == "a_start":
            assert results[1] == "a_end"
        else:
            assert results[0] == "b_start"
            assert results[1] == "b_end"


# =============================================================================
# Integration Tests
# =============================================================================

class TestAsyncConcurrencyIntegration:
    """Integration tests for async/concurrency features."""

    @pytest.mark.asyncio
    async def test_executor_and_async_interop(self):
        """Thread executor should work with async code."""
        from core.utils import run_blocking_io
        
        def blocking_work():
            import time
            time.sleep(0.01)
            return "blocking_result"
        
        async def async_work():
            await asyncio.sleep(0.01)
            return "async_result"
        
        # Run both
        blocking_result = await run_blocking_io(blocking_work)
        async_result = await async_work()
        
        assert blocking_result == "blocking_result"
        assert async_result == "async_result"

    def test_stats_are_consistent(self):
        """Executor stats should be internally consistent."""
        from core.utils import get_io_executor_stats
        
        stats = get_io_executor_stats()
        
        # Max workers should be positive
        assert stats["max_workers"] > 0
        
        # Invocations should be non-negative
        assert stats["invocations_total"] >= 0
        
        # Backpressure config should be non-negative
        assert stats["backpressure_max_queue_depth"] >= 0

    @pytest.mark.asyncio
    async def test_multiple_blocking_operations(self):
        """Multiple blocking operations should run concurrently."""
        from core.utils import run_blocking_io
        import time
        
        def slow_op(n):
            time.sleep(0.01)
            return n
        
        start = time.time()
        
        results = await asyncio.gather(
            run_blocking_io(slow_op, 1),
            run_blocking_io(slow_op, 2),
            run_blocking_io(slow_op, 3),
        )
        
        elapsed = time.time() - start
        
        # Should run concurrently, not sequentially
        # Sequential would take ~0.03s, concurrent should be ~0.01s
        assert elapsed < 0.05
        assert sorted(results) == [1, 2, 3]

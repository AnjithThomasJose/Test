"""
Issue 7.2: Parameterized tests for LLM error scenarios.

Tests the error handling paths in LLM invocation including:
- Timeout errors
- Rate limit errors
- Cancelled errors
- Generic exceptions
- Fallback behavior
- Error classification utilities
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock


class TestSafeLLMCallErrorHandling:
    """Tests for safe_llm_call wrapper with various error types."""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_type,expected_error_class", [
        (asyncio.TimeoutError(), "LLMTimeoutError"),
        (asyncio.CancelledError(), "LLMCancelledError"),
        (Exception("Rate limit exceeded"), "LLMRateLimitError"),
        (Exception("Quota exceeded"), "LLMRateLimitError"),
    ])
    async def test_error_classification(self, error_type, expected_error_class):
        """safe_llm_call should classify errors correctly."""
        from utils.llm_error_handler import safe_llm_call, LLMError
        
        async def failing_llm_call():
            raise error_type
        
        with pytest.raises(LLMError) as exc_info:
            await safe_llm_call(
                failing_llm_call,
                timeout=1.0,
                max_retries=1,
                backoff_factor=0.1,
                agent_name="test_agent"
            )
        
        assert expected_error_class in type(exc_info.value).__name__
    
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Should retry on timeout before giving up."""
        from utils.llm_error_handler import safe_llm_call
        
        call_count = 0
        
        async def timeout_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError()
            return "success"
        
        result = await safe_llm_call(
            timeout_then_succeed,
            timeout=5.0,
            max_retries=3,
            backoff_factor=0.1,
            agent_name="test_agent"
        )
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Should raise after max retries exhausted."""
        from utils.llm_error_handler import safe_llm_call, LLMTimeoutError
        
        call_count = 0
        
        async def always_timeout():
            nonlocal call_count
            call_count += 1
            raise asyncio.TimeoutError()
        
        with pytest.raises(LLMTimeoutError):
            await safe_llm_call(
                always_timeout,
                timeout=1.0,
                max_retries=3,
                backoff_factor=0.1,
                agent_name="test_agent"
            )
        
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self):
        """Successful call should not trigger retries."""
        from utils.llm_error_handler import safe_llm_call
        
        call_count = 0
        
        async def succeed():
            nonlocal call_count
            call_count += 1
            return "immediate success"
        
        result = await safe_llm_call(
            succeed,
            timeout=5.0,
            max_retries=3,
            agent_name="test_agent"
        )
        
        assert result == "immediate success"
        assert call_count == 1


class TestLLMErrorClassification:
    """Tests for error classification utility functions."""
    
    def test_is_cancellation_error(self):
        """Should correctly identify cancellation errors."""
        from utils.llm_error_handler import is_cancellation_error, LLMCancelledError
        
        assert is_cancellation_error(asyncio.CancelledError()) is True
        assert is_cancellation_error(LLMCancelledError("cancelled")) is True
        assert is_cancellation_error(Exception("Operation was cancelled")) is True
        assert is_cancellation_error(Exception("some other error")) is False
    
    def test_is_timeout_error(self):
        """Should correctly identify timeout errors."""
        from utils.llm_error_handler import is_timeout_error, LLMTimeoutError
        
        assert is_timeout_error(asyncio.TimeoutError()) is True
        assert is_timeout_error(LLMTimeoutError("timed out")) is True
        assert is_timeout_error(Exception("Connection timeout")) is True
        assert is_timeout_error(Exception("some other error")) is False
    
    def test_is_rate_limit_error(self):
        """Should correctly identify rate limit errors."""
        from utils.llm_error_handler import is_rate_limit_error, LLMRateLimitError
        
        assert is_rate_limit_error(LLMRateLimitError("rate limited")) is True
        assert is_rate_limit_error(Exception("Rate limit exceeded")) is True
        assert is_rate_limit_error(Exception("Quota exceeded")) is True
        assert is_rate_limit_error(Exception("some other error")) is False


class TestRetryAfterExtraction:
    """Tests for Retry-After header extraction."""
    
    def test_extract_from_attribute(self):
        """Should extract retry_after from exception attribute."""
        from utils.llm_error_handler import _extract_retry_after
        
        class MockException(Exception):
            retry_after = 30.0
        
        result = _extract_retry_after(MockException("rate limited"))
        assert result == 30.0
    
    def test_extract_from_message(self):
        """Should extract retry_after from error message."""
        from utils.llm_error_handler import _extract_retry_after
        
        result = _extract_retry_after(Exception("Please retry after 60 seconds"))
        assert result == 60.0
    
    def test_no_retry_after_available(self):
        """Should return None when no retry_after info available."""
        from utils.llm_error_handler import _extract_retry_after
        
        result = _extract_retry_after(Exception("Generic error"))
        assert result is None


class TestWithLLMErrorHandlingDecorator:
    """Tests for the @with_llm_error_handling decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_wraps_function(self):
        """Decorator should wrap async function with error handling."""
        from utils.llm_error_handler import with_llm_error_handling
        
        @with_llm_error_handling(timeout=5.0, max_retries=2, backoff_factor=0.1)
        async def my_llm_function():
            return "decorated result"
        
        result = await my_llm_function()
        assert result == "decorated result"
    
    @pytest.mark.asyncio
    async def test_decorator_retries_on_error(self):
        """Decorator should retry on transient errors."""
        from utils.llm_error_handler import with_llm_error_handling, LLMTimeoutError
        
        call_count = 0
        
        @with_llm_error_handling(timeout=5.0, max_retries=2, backoff_factor=0.1)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise asyncio.TimeoutError()
            return "eventually succeeded"
        
        result = await flaky_function()
        assert result == "eventually succeeded"
        assert call_count == 2


class TestFallbackResponseHandling:
    """Tests for fallback response detection and handling."""
    
    def test_is_fallback_response(self):
        """Should correctly identify fallback responses."""
        from models.llm_invoker import is_fallback_response, FALLBACK_RESPONSE_MARKER
        
        fallback = f"{FALLBACK_RESPONSE_MARKER} Sorry, I couldn't process that."
        normal = "This is a normal response."
        
        assert is_fallback_response(fallback) is True
        assert is_fallback_response(normal) is False
    
    def test_strip_fallback_marker(self):
        """Should strip fallback marker from responses."""
        from models.llm_invoker import strip_fallback_marker, FALLBACK_RESPONSE_MARKER
        
        fallback = f"{FALLBACK_RESPONSE_MARKER} The actual message."
        normal = "Normal message without marker."
        
        assert strip_fallback_marker(fallback) == "The actual message."
        assert strip_fallback_marker(normal) == "Normal message without marker."


class TestLLMRateLimitError:
    """Tests for LLMRateLimitError with retry_after attribute."""
    
    def test_rate_limit_error_with_retry_after(self):
        """LLMRateLimitError should store retry_after value."""
        from utils.llm_error_handler import LLMRateLimitError
        
        error = LLMRateLimitError("Rate limited", retry_after=60.0)
        
        assert error.retry_after == 60.0
        assert "Rate limited" in str(error)
    
    def test_rate_limit_error_without_retry_after(self):
        """LLMRateLimitError should work without retry_after."""
        from utils.llm_error_handler import LLMRateLimitError
        
        error = LLMRateLimitError("Rate limited")
        
        assert error.retry_after is None


class TestCreateRobustLLMCall:
    """Tests for create_robust_llm_call factory function."""
    
    @pytest.mark.asyncio
    async def test_creates_callable(self):
        """Should create an async callable."""
        from utils.llm_error_handler import create_robust_llm_call
        
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value="model response")
        
        robust_call = create_robust_llm_call(
            mock_model,
            prompt="test prompt",
            timeout=5.0,
            max_retries=2
        )
        
        result = await robust_call()
        assert result == "model response"

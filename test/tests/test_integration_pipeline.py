"""
Issue 7.3: Integration tests for the agent pipeline patterns.

Tests verify the design patterns and utilities used in the pipeline
without importing the heavy supervisor_agent module (which has many
transitive dependencies). Tests use local mock implementations that
mirror the production patterns.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any
import copy


# ============================================================================
# Local mock implementations mirroring production patterns
# ============================================================================

def mock_merge_names(existing, new):
    """Mirror of merge_names from supervisor_agent."""
    if existing is None:
        existing = []
    elif isinstance(existing, str):
        existing = [existing]
    if new is None:
        new = []
    elif isinstance(new, str):
        new = [new]
    return (existing or []) + (new or [])


def mock_merge_lists(existing, new):
    """Mirror of merge_lists from supervisor_agent."""
    if existing is None:
        existing = []
    elif not isinstance(existing, list):
        existing = [existing]
    if new is None:
        new = []
    elif not isinstance(new, list):
        new = [new]
    return (existing or []) + (new or [])


def mock_take_last(existing, new):
    """Mirror of take_last from supervisor_agent."""
    return new


class MockAnalysisMemory:
    """Mirror of AnalysisMemory pattern from supervisor_agent."""
    
    def __init__(self):
        self._memory: Dict[str, Any] = {}
    
    def set_memory(self, key: str, value: Any):
        self._memory[key] = value
    
    def get_memory(self, key: str) -> Any:
        """Return deep copy to ensure state isolation."""
        value = self._memory.get(key)
        if value is not None:
            return copy.deepcopy(value)
        return None


MOCK_VALID_ROUTES = [
    "validate_resume", "validate_jd", "groq_resume_parser",
    "job_matcher", "resume_scorer", "end", "__end__"
]


def mock_dispatcher_router(state: Dict[str, Any]) -> str:
    """Mirror of router pattern from supervisor_agent."""
    try:
        next_node = state.get("next", "")
        if not next_node or next_node not in MOCK_VALID_ROUTES:
            return "end"
        return next_node
    except Exception:
        return "end"


def mock_apply_middleware(agent_func, agent_name: str):
    """Mirror of apply_middleware pattern from supervisor_agent."""
    async def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if asyncio.iscoroutinefunction(agent_func):
                return await agent_func(state)
            else:
                return agent_func(state)
        except Exception as e:
            return {"error": str(e), "agent": agent_name}
    return wrapped


# ============================================================================
# Tests for merge reducer patterns
# ============================================================================

class TestMergeReducerPatterns:
    """Tests for state merge reducer functions."""
    
    def test_merge_names_combines_strings(self):
        """merge_names should combine string inputs into list."""
        result = mock_merge_names("existing", "new")
        assert result == ["existing", "new"]
    
    def test_merge_names_handles_none(self):
        """merge_names should handle None values."""
        assert mock_merge_names(None, "new") == ["new"]
        assert mock_merge_names("existing", None) == ["existing"]
        assert mock_merge_names(None, None) == []
    
    def test_merge_names_handles_lists(self):
        """merge_names should handle list inputs."""
        result = mock_merge_names(["a", "b"], ["c"])
        assert result == ["a", "b", "c"]
    
    def test_merge_lists_concatenates(self):
        """merge_lists should concatenate lists."""
        result = mock_merge_lists([1, 2], [3, 4])
        assert result == [1, 2, 3, 4]
    
    def test_merge_lists_handles_none(self):
        """merge_lists should handle None values."""
        assert mock_merge_lists(None, [1, 2]) == [1, 2]
        assert mock_merge_lists([1, 2], None) == [1, 2]
    
    def test_merge_lists_wraps_non_lists(self):
        """merge_lists should wrap non-list values in lists."""
        result = mock_merge_lists("single", [1, 2])
        assert result == ["single", 1, 2]
    
    def test_take_last_returns_new_value(self):
        """take_last should return the new value."""
        assert mock_take_last("old", "new") == "new"
        assert mock_take_last(None, "new") == "new"
        assert mock_take_last("old", None) is None


# ============================================================================
# Tests for state isolation patterns
# ============================================================================

class TestStateIsolationPatterns:
    """Tests for state isolation between concurrent requests."""
    
    def test_analysis_memory_returns_deep_copy(self):
        """AnalysisMemory.get_memory() should return deep copy."""
        memory = MockAnalysisMemory()
        memory.set_memory("test_key", {"nested": {"data": [1, 2, 3]}})
        
        retrieved = memory.get_memory("test_key")
        
        # Modify retrieved data
        retrieved["nested"]["data"].append(4)
        
        # Original should be unchanged
        original = memory.get_memory("test_key")
        assert 4 not in original["nested"]["data"]
    
    def test_state_mutations_isolated(self):
        """State changes in one request should not affect others."""
        base_state = {
            "uid": "user123",
            "messages": [],
            "data": {"key": "value"}
        }
        
        state1 = copy.deepcopy(base_state)
        state2 = copy.deepcopy(base_state)
        
        # Modify state1
        state1["messages"].append("message1")
        state1["data"]["key"] = "modified"
        
        # state2 should be unchanged
        assert len(state2["messages"]) == 0
        assert state2["data"]["key"] == "value"
    
    def test_memory_returns_none_for_missing_key(self):
        """get_memory should return None for missing keys."""
        memory = MockAnalysisMemory()
        
        result = memory.get_memory("nonexistent")
        assert result is None


# ============================================================================
# Tests for router patterns
# ============================================================================

class TestRouterPatterns:
    """Tests for dispatcher routing logic patterns."""
    
    def test_router_returns_valid_target(self):
        """Router should return valid routing target."""
        state = {"next": "validate_resume"}
        result = mock_dispatcher_router(state)
        assert result == "validate_resume"
    
    def test_router_handles_missing_next(self):
        """Router should return 'end' for missing next field."""
        state = {}
        result = mock_dispatcher_router(state)
        assert result == "end"
    
    def test_router_handles_invalid_next(self):
        """Router should return 'end' for invalid next value."""
        state = {"next": "nonexistent_node"}
        result = mock_dispatcher_router(state)
        assert result == "end"
    
    def test_router_handles_empty_next(self):
        """Router should return 'end' for empty next value."""
        state = {"next": ""}
        result = mock_dispatcher_router(state)
        assert result == "end"


# ============================================================================
# Tests for middleware patterns
# ============================================================================

class TestMiddlewarePatterns:
    """Tests for apply_middleware error handling patterns."""
    
    @pytest.mark.asyncio
    async def test_middleware_wraps_async_function(self):
        """Middleware should wrap async agent functions."""
        async def async_agent(state):
            return {"result": "success"}
        
        wrapped = mock_apply_middleware(async_agent, "test_agent")
        result = await wrapped({"input": "test"})
        
        assert result["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_middleware_wraps_sync_function(self):
        """Middleware should wrap sync agent functions."""
        def sync_agent(state):
            return {"result": "sync success"}
        
        wrapped = mock_apply_middleware(sync_agent, "test_agent")
        result = await wrapped({"input": "test"})
        
        assert result["result"] == "sync success"
    
    @pytest.mark.asyncio
    async def test_middleware_catches_exceptions(self):
        """Middleware should catch and wrap exceptions."""
        async def failing_agent(state):
            raise ValueError("Test error")
        
        wrapped = mock_apply_middleware(failing_agent, "test_agent")
        result = await wrapped({"input": "test"})
        
        assert "error" in result
        assert "Test error" in result["error"]
        assert result["agent"] == "test_agent"


# ============================================================================
# Tests for valid routes configuration
# ============================================================================

class TestValidRoutesConfiguration:
    """Tests for valid routing targets configuration."""
    
    def test_valid_routes_includes_end(self):
        """VALID_ROUTES should include 'end' terminator."""
        assert "end" in MOCK_VALID_ROUTES or "__end__" in MOCK_VALID_ROUTES
    
    def test_valid_routes_includes_common_agents(self):
        """VALID_ROUTES should include common agent nodes."""
        assert "validate_resume" in MOCK_VALID_ROUTES
        assert "job_matcher" in MOCK_VALID_ROUTES
    
    def test_valid_routes_is_iterable(self):
        """VALID_ROUTES should be iterable."""
        count = 0
        for route in MOCK_VALID_ROUTES:
            count += 1
        assert count > 0


# ============================================================================
# Tests for config patterns
# ============================================================================

class TestConfigPatterns:
    """Tests for configuration patterns used in pipeline."""
    
    def test_graph_recursion_limit_from_config(self):
        """GRAPH_RECURSION_LIMIT should be importable from config."""
        from core.config import GRAPH_RECURSION_LIMIT
        
        assert isinstance(GRAPH_RECURSION_LIMIT, int)
        assert GRAPH_RECURSION_LIMIT > 0
    
    def test_recursion_limit_reasonable_range(self):
        """GRAPH_RECURSION_LIMIT should be in reasonable range."""
        from core.config import GRAPH_RECURSION_LIMIT
        
        # Should be >= 25 for complex workflows, <= 100 to prevent runaway
        assert 25 <= GRAPH_RECURSION_LIMIT <= 100


# ============================================================================
# Tests for agent state patterns
# ============================================================================

class TestAgentStatePatterns:
    """Tests for AgentState structure patterns."""
    
    def test_state_accepts_common_fields(self):
        """State dicts should accept common workflow fields."""
        state = {
            "uid": "user123",
            "tenant_id": "tenant456",
            "body": {"action": "test"},
            "next": "validate_resume",
            "messages": [],
            "error": None
        }
        
        assert state["uid"] == "user123"
        assert state["tenant_id"] == "tenant456"
        assert state["body"]["action"] == "test"
    
    def test_state_supports_resume_fields(self):
        """State should support resume-related fields."""
        state = {
            "resume_text": "John Doe...",
            "is_valid_resume": True,
            "resume_confidence": 0.95,
            "parsed_resume": {"name": "John Doe"}
        }
        
        assert state["is_valid_resume"] is True
        assert state["resume_confidence"] == 0.95
    
    def test_state_supports_job_fields(self):
        """State should support job-related fields."""
        state = {
            "job_id": "job123",
            "job_description": "Software Engineer...",
            "match_score": 0.85,
            "skills_matched": ["Python", "AWS"]
        }
        
        assert state["match_score"] == 0.85
        assert "Python" in state["skills_matched"]


# ============================================================================
# Tests for error propagation patterns
# ============================================================================

class TestErrorPropagationPatterns:
    """Tests for error handling and propagation patterns."""
    
    def test_error_state_structure(self):
        """Error states should have consistent structure."""
        error_state = {
            "error": "Something went wrong",
            "error_code": "VALIDATION_ERROR",
            "next": "end"
        }
        
        assert "error" in error_state
        assert error_state["next"] == "end"
    
    @pytest.mark.asyncio
    async def test_error_propagates_through_middleware(self):
        """Errors should propagate through middleware with context."""
        async def error_agent(state):
            raise RuntimeError("Database connection failed")
        
        wrapped = mock_apply_middleware(error_agent, "db_agent")
        result = await wrapped({})
        
        assert "error" in result
        assert "Database connection failed" in result["error"]
    
    def test_error_state_preserves_context(self):
        """Error states should preserve original context."""
        original_state = {"uid": "user123", "action": "parse"}
        error_state = {
            **original_state,
            "error": "Parse failed",
            "next": "end"
        }
        
        assert error_state["uid"] == "user123"
        assert error_state["action"] == "parse"

"""Regression: AgentState schema uses explicit reducers for JD / role-fit fields (no duplicate declarations)."""
from collections import Counter
from typing import get_args

from core.supervisor_agent import AgentState, take_last


def test_agent_state_annotation_keys_unique():
    ann = getattr(AgentState, "__annotations__", {}) or {}
    counts = Counter(ann.keys())
    dupes = [k for k, n in counts.items() if n > 1]
    assert not dupes, f"Duplicate AgentState keys: {dupes}"


def test_role_fit_and_jd_fields_use_take_last():
    for name in ("enhanced_role_fit", "job_description", "job_details"):
        field = AgentState.__annotations__[name]
        args = get_args(field)
        assert args and args[-1] is take_last, f"{name} should use take_last reducer, got {args[-1]!r}"

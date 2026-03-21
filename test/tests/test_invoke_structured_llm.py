"""Tests for models.llm_invoker.invoke_structured_llm."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from unittest.mock import AsyncMock, patch, MagicMock

from core.quota_manager import QuotaStatus
from models.llm_invoker import invoke_structured_llm, _parse_structured_invoke_result


def _quota_patch_stack():
    return (
        patch(
            "models.llm_invoker.check_quota",
            return_value=QuotaStatus.AVAILABLE,
        ),
        patch("models.llm_invoker.is_in_backoff", return_value=False),
    )


class _SimpleSchema(BaseModel):
    """Minimal schema for structured output tests."""

    answer: str
    score: int = 0


def test_parse_structured_invoke_result_plain_model():
    sm = _SimpleSchema(answer="ok", score=3)
    out = _parse_structured_invoke_result(_SimpleSchema, sm, include_raw=False)
    assert isinstance(out, _SimpleSchema)
    assert out.answer == "ok"


def test_parse_structured_invoke_result_include_raw_dict():
    parsed = _SimpleSchema(answer="x", score=1)
    raw = MagicMock()
    raw.content = '{"answer":"x","score":1}'
    bundle = {"parsed": parsed, "raw": raw}
    out = _parse_structured_invoke_result(_SimpleSchema, bundle, include_raw=True)
    assert isinstance(out, tuple)
    p, r = out
    assert isinstance(p, _SimpleSchema)
    assert r is raw


@pytest.mark.asyncio
async def test_invoke_structured_llm_passthrough_safe_llm_call():
    qp, ib = _quota_patch_stack()
    with qp, ib, patch(
        "models.llm_invoker.safe_llm_call", new_callable=AsyncMock
    ) as mock_safe:
        mock_safe.return_value = _SimpleSchema(answer="from_mock", score=42)
        result = await invoke_structured_llm(
            "hello",
            _SimpleSchema,
            agent_name="test_agent_structured",
            raise_on_fallback=True,
            skip_cache=True,
        )
    assert isinstance(result, _SimpleSchema)
    assert result.answer == "from_mock"
    assert result.score == 42
    mock_safe.assert_awaited_once()


@pytest.mark.asyncio
async def test_invoke_structured_llm_cache_hit():
    cached_json = '{"answer":"cached","score":7}'
    qp, ib = _quota_patch_stack()
    with (
        qp,
        ib,
        patch("models.llm_invoker._llm_cache.get", return_value=cached_json),
        patch("models.llm_invoker.safe_llm_call", new_callable=AsyncMock) as mock_safe,
    ):
        result = await invoke_structured_llm(
            "prompt text",
            _SimpleSchema,
            agent_name="test_cache",
            raise_on_fallback=True,
            skip_cache=False,
        )
        mock_safe.assert_not_called()
    assert isinstance(result, _SimpleSchema)
    assert result.answer == "cached"
    assert result.score == 7

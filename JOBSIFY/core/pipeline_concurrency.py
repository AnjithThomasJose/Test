"""
Concurrency slots for heavy agent work on one process.

- ``PIPELINE_MAX_PARALLEL`` — applies only to **LangGraph** ``run_pipeline``
  (e.g. ``/analyze-resume-callback``, ``/compare-candidate-job``). Ranker and
  job-matcher API tasks **do not** use this slot, so a long **ranker** run does
  not block new **analyze resume** pipelines (unless you overload LLMs/CPU).

- ``RANKER_API_MAX_PARALLEL`` — optional separate cap for **async** ``/ranker``
  and ``/api/job-matcher`` background work (when ``callback_url`` is set).

Unset or ``0`` = no limit for that knob.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

log = logging.getLogger(__name__)

_init_lock = asyncio.Lock()
_sem: Optional[asyncio.Semaphore] = None
_active_limit: Optional[int] = None

_ranker_init_lock = asyncio.Lock()
_ranker_sem: Optional[asyncio.Semaphore] = None
_ranker_active_limit: Optional[int] = None


def pipeline_parallel_limit_from_env() -> int:
    raw = os.getenv("PIPELINE_MAX_PARALLEL", "").strip()
    if not raw:
        return 0
    try:
        n = int(raw)
        return n if n > 0 else 0
    except ValueError:
        log.warning("Invalid PIPELINE_MAX_PARALLEL=%r — ignoring limit", raw)
        return 0


def ranker_api_parallel_limit_from_env() -> int:
    raw = os.getenv("RANKER_API_MAX_PARALLEL", "").strip()
    if not raw:
        return 0
    try:
        n = int(raw)
        return n if n > 0 else 0
    except ValueError:
        log.warning("Invalid RANKER_API_MAX_PARALLEL=%r — ignoring limit", raw)
        return 0


def log_pipeline_concurrency_at_startup() -> None:
    lim = pipeline_parallel_limit_from_env()
    if lim <= 0:
        log.info(
            "Concurrency: PIPELINE_MAX_PARALLEL not set (or 0) — unlimited concurrent "
            "LangGraph pipelines (analyze-resume / compare); ranker does not take this slot."
        )
    else:
        log.info(
            "Concurrency: PIPELINE_MAX_PARALLEL=%s — extra graph pipelines wait for a free slot",
            lim,
        )
    rlim = ranker_api_parallel_limit_from_env()
    if rlim <= 0:
        log.info(
            "Concurrency: RANKER_API_MAX_PARALLEL not set (or 0) — unlimited async /ranker & /api/job-matcher runs."
        )
    else:
        log.info(
            "Concurrency: RANKER_API_MAX_PARALLEL=%s — extra ranker/job-matcher API tasks wait",
            rlim,
        )


async def _ensure_semaphore() -> Optional[asyncio.Semaphore]:
    global _sem, _active_limit
    limit = pipeline_parallel_limit_from_env()
    if limit <= 0:
        return None
    async with _init_lock:
        if _sem is None or _active_limit != limit:
            _sem = asyncio.Semaphore(limit)
            _active_limit = limit
            log.info(
                "Initialized pipeline semaphore: max %s concurrent run_pipeline executions",
                limit,
            )
        return _sem


@asynccontextmanager
async def pipeline_execution_slot():
    """
    Acquire a slot before running a full LangGraph pipeline (run_pipeline only).
    """
    sem = await _ensure_semaphore()
    if sem is None:
        yield
        return
    await sem.acquire()
    try:
        yield
    finally:
        sem.release()


async def _ensure_ranker_semaphore() -> Optional[asyncio.Semaphore]:
    global _ranker_sem, _ranker_active_limit
    limit = ranker_api_parallel_limit_from_env()
    if limit <= 0:
        return None
    async with _ranker_init_lock:
        if _ranker_sem is None or _ranker_active_limit != limit:
            _ranker_sem = asyncio.Semaphore(limit)
            _ranker_active_limit = limit
            log.info(
                "Initialized ranker API semaphore: max %s concurrent async ranker/job-matcher tasks",
                limit,
            )
        return _ranker_sem


@asynccontextmanager
async def ranker_api_execution_slot():
    """
    Optional slot for /ranker and /api/job-matcher background work only.
    Independent from pipeline_execution_slot so ranker does not block analyze-resume.
    """
    sem = await _ensure_ranker_semaphore()
    if sem is None:
        yield
        return
    await sem.acquire()
    try:
        yield
    finally:
        sem.release()

"""Shared fixtures + helpers for ``tests/unit/control/events/``."""

from __future__ import annotations

from typing import Any

from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.events.types.control_run import (
    RunCompletedEvent,
    RunCompletedPayload,
    RunFailedEvent,
    RunFailedPayload,
    RunStartedEvent,
    RunStartedPayload,
)


def make_started(
    *,
    run_id: str = "test-run",
    source: str = "control://orchestrator",
    offset: int = UNKNOWN_OFFSET,
    run_name: str = "run-1",
) -> RunStartedEvent:
    return RunStartedEvent(
        source=source,
        run_id=run_id,
        offset=offset,
        payload=RunStartedPayload(
            run_name=run_name,
            algorithm="sft",
            model_id="acme/test",
            dataset_id="default",
            config_hash="abc123",
        ),
    )


def make_completed(
    *,
    run_id: str = "test-run",
    source: str = "control://orchestrator",
    offset: int = UNKNOWN_OFFSET,
    duration_s: float = 1.0,
    status: str = "success",
) -> RunCompletedEvent:
    return RunCompletedEvent(
        source=source,
        run_id=run_id,
        offset=offset,
        payload=RunCompletedPayload(
            duration_s=duration_s,
            final_status=status,
            mlflow_run_id=None,
        ),
    )


def make_failed(
    *,
    run_id: str = "test-run",
    source: str = "control://orchestrator",
    offset: int = UNKNOWN_OFFSET,
    msg: str = "boom",
) -> RunFailedEvent:
    return RunFailedEvent(
        source=source,
        run_id=run_id,
        offset=offset,
        payload=RunFailedPayload(
            failing_stage="trainer",
            error_type="RuntimeError",
            message=msg,
            traceback_excerpt="<tb excerpt>",
        ),
    )


def event_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default-fill kwargs for :func:`make_started` and friends."""
    return overrides

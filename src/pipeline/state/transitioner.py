"""
State transition functions for pipeline stage lifecycle.

Pure mutators of ``PipelineAttemptState`` / ``PipelineState`` — each takes
a state object and mutates it in place with a specific semantic transition
(running / completed / failed / skipped / interrupted / finalized).

Lineage manipulation (``invalidate_from`` / ``restore_reused`` / the
per-stage wrappers around ``update_lineage``) lives in
:mod:`src.pipeline.state.lineage_manager` — this module keeps a single
concern: stage-run state transitions.
"""

from __future__ import annotations

from typing import Any

from src.pipeline.artifacts.base import utc_now_iso
from src.pipeline.state.models import (
    PipelineAttemptState,
    PipelineState,
    StageRunState,
)


def mark_stage_running(
    *,
    attempt: PipelineAttemptState,
    stage_name: str,
    started_at: str,
) -> None:
    """Mark a stage as running in the attempt state."""
    prev = attempt.stage_runs.get(stage_name)
    log_paths = dict(prev.log_paths) if prev else {}
    attempt.stage_runs[stage_name] = StageRunState(
        stage_name=stage_name,
        status=StageRunState.STATUS_RUNNING,
        execution_mode=StageRunState.MODE_EXECUTED,
        started_at=started_at,
        log_paths=log_paths,
    )


def mark_stage_completed(
    *,
    attempt: PipelineAttemptState,
    stage_name: str,
    outputs: dict[str, Any],
) -> None:
    """Mark a stage as completed with outputs."""
    stage_state = attempt.stage_runs.get(stage_name) or StageRunState(stage_name=stage_name)
    stage_state.status = StageRunState.STATUS_COMPLETED
    stage_state.execution_mode = StageRunState.MODE_EXECUTED
    stage_state.outputs = dict(outputs)
    stage_state.completed_at = utc_now_iso()
    stage_state.error = None
    stage_state.failure_kind = None
    stage_state.skip_reason = None
    attempt.stage_runs[stage_name] = stage_state


def mark_stage_failed(
    *,
    attempt: PipelineAttemptState,
    stage_name: str,
    error: str,
    failure_kind: str,
    outputs: dict[str, Any] | None = None,
) -> None:
    """Mark a stage as failed, propagating failure to the attempt."""
    stage_state = attempt.stage_runs.get(stage_name) or StageRunState(stage_name=stage_name)
    stage_state.status = StageRunState.STATUS_FAILED
    stage_state.execution_mode = StageRunState.MODE_EXECUTED
    stage_state.outputs = dict(outputs or {})
    stage_state.error = error
    stage_state.failure_kind = failure_kind
    stage_state.completed_at = utc_now_iso()
    attempt.stage_runs[stage_name] = stage_state
    attempt.status = StageRunState.STATUS_FAILED
    attempt.completed_at = utc_now_iso()
    attempt.error = error


def mark_stage_skipped(
    *,
    attempt: PipelineAttemptState,
    stage_name: str,
    reason: str,
    outputs: dict[str, Any] | None = None,
) -> None:
    """Mark a stage as skipped with a reason."""
    prev = attempt.stage_runs.get(stage_name)
    log_paths = dict(prev.log_paths) if prev else {}
    attempt.stage_runs[stage_name] = StageRunState(
        stage_name=stage_name,
        status=StageRunState.STATUS_SKIPPED,
        execution_mode=StageRunState.MODE_SKIPPED,
        outputs=dict(outputs or {}),
        skip_reason=reason,
        started_at=utc_now_iso(),
        completed_at=utc_now_iso(),
        log_paths=log_paths,
    )


def mark_stage_interrupted(
    *,
    attempt: PipelineAttemptState,
    stage_name: str,
    started_at: str,
) -> None:
    """Mark a stage as interrupted (e.g. by SIGINT)."""
    prev = attempt.stage_runs.get(stage_name)
    log_paths = dict(prev.log_paths) if prev else {}
    attempt.stage_runs[stage_name] = StageRunState(
        stage_name=stage_name,
        status=StageRunState.STATUS_INTERRUPTED,
        execution_mode=StageRunState.MODE_EXECUTED,
        started_at=started_at,
        completed_at=utc_now_iso(),
        log_paths=log_paths,
    )


def finalize_attempt_state(
    *,
    state: PipelineState,
    attempt: PipelineAttemptState,
    status: str,
    completed_at: str | None = None,
) -> None:
    """Finalize an attempt — set status, timestamp, and clear active_attempt_id."""
    attempt.status = status
    attempt.completed_at = completed_at or attempt.completed_at or utc_now_iso()
    state.pipeline_status = status
    if state.active_attempt_id == attempt.attempt_id:
        state.active_attempt_id = None


__all__ = [
    "finalize_attempt_state",
    "mark_stage_completed",
    "mark_stage_failed",
    "mark_stage_interrupted",
    "mark_stage_running",
    "mark_stage_skipped",
]

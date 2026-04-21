"""
State transition functions for pipeline stage lifecycle.

Extracted from PipelineOrchestrator to reduce orchestrator size and isolate
pure state mutations into testable functions.

Most functions operate on PipelineAttemptState / PipelineState objects
directly. ``restore_reused_context`` additionally mutates a pipeline context
dict and calls a sync-root callback — we keep it alongside the other
transitioners because its job is "materialise completed-and-reused stage
runs into attempt state".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.pipeline.artifacts.base import utc_now_iso
from src.pipeline.state import (
    PipelineAttemptState,
    PipelineState,
    StageLineageRef,
    StageRunState,
)

if TYPE_CHECKING:
    from collections.abc import Callable


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


def invalidate_lineage_from(
    *,
    lineage: dict[str, StageLineageRef],
    stage_names: list[str],
    start_stage_name: str,
) -> dict[str, StageLineageRef]:
    """Return a new lineage dict with all entries from ``start_stage_name`` onward removed.

    Used when a user restarts from a specific stage: every downstream stage's
    previous outputs must be invalidated so the pipeline recomputes them.

    Raises ``ValueError`` if ``start_stage_name`` is not in ``stage_names``.
    """
    if start_stage_name not in stage_names:
        raise ValueError(f"Unknown stage name: {start_stage_name}")
    start_idx = stage_names.index(start_stage_name)
    new_lineage = dict(lineage)
    for name in stage_names[start_idx:]:
        new_lineage.pop(name, None)
    return new_lineage


def restore_reused_context(
    *,
    attempt: PipelineAttemptState,
    lineage: dict[str, StageLineageRef],
    stage_names: list[str],
    start_stage_name: str,
    enabled_stage_names: list[str],
    context: dict[str, Any],
    sync_root_from_stage: Callable[[dict[str, Any], str, dict[str, Any]], None],
) -> None:
    """Materialise completed-and-reused stages into attempt state and context.

    For each stage strictly before ``start_stage_name`` that has a lineage
    entry:

    * If the stage is disabled in the current config, mark it as SKIPPED.
    * Otherwise copy its stored outputs into ``context[stage_name]``, invoke
      ``sync_root_from_stage`` so downstream root-level keys stay populated,
      and record the stage as COMPLETED/REUSED in attempt state.

    ``sync_root_from_stage`` is injected so this function stays independent of
    the ContextPropagator module.

    Raises ``ValueError`` if ``start_stage_name`` is not in ``stage_names``.
    """
    if start_stage_name not in stage_names:
        raise ValueError(f"Unknown stage name: {start_stage_name}")
    start_idx = stage_names.index(start_stage_name)
    for i, stage_name in enumerate(stage_names):
        if i >= start_idx:
            break
        ref = lineage.get(stage_name)
        if ref is None:
            continue
        if stage_name not in enabled_stage_names:
            mark_stage_skipped(
                attempt=attempt,
                stage_name=stage_name,
                reason="disabled_by_config",
                outputs=ref.outputs,
            )
            continue
        context[stage_name] = dict(ref.outputs)
        sync_root_from_stage(context, stage_name, ref.outputs)
        attempt.stage_runs[stage_name] = StageRunState(
            stage_name=stage_name,
            status=StageRunState.STATUS_COMPLETED,
            execution_mode=StageRunState.MODE_REUSED,
            outputs=dict(ref.outputs),
            started_at=utc_now_iso(),
            completed_at=utc_now_iso(),
            reuse_from={"attempt_id": ref.attempt_id, "stage_name": ref.stage_name},
        )


__all__ = [
    "finalize_attempt_state",
    "invalidate_lineage_from",
    "mark_stage_completed",
    "mark_stage_failed",
    "mark_stage_interrupted",
    "mark_stage_running",
    "mark_stage_skipped",
    "restore_reused_context",
]

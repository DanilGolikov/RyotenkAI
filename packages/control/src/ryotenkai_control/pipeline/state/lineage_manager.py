"""Pure-function lineage manipulation for the pipeline state.

Lineage (``PipelineState.current_output_lineage``) is the mapping
``stage_name → StageLineageRef`` that records "which stage's outputs are
currently considered reusable for restart". It is mutated at five distinct
moments in a pipeline run:

1. **Pre-loop invalidation**  — when the user restarts from stage *N*, every
   entry from *N* onward is dropped (we rerun them, so their previous
   outputs are no longer authoritative).
2. **Pre-loop restoration**   — for stages strictly *before* the restart
   point that had stored outputs, we re-materialise the context + attempt
   state so downstream logic can read them without re-execution.
3. **Stage completed**        — a stage's fresh outputs are written.
4. **Stage skipped**           — the entry is removed (we don't want the
   loop to incorrectly claim the skipped stage's outputs are available).
5. **Stage failed**            — same as skipped (remove, don't retain).

All five used to live inside ``PipelineOrchestrator``; moments 1 and 2 had
already moved to ``transitioner.py`` in an earlier refactor. This module
collects **all five** under one roof so future changes to the lineage
protocol (e.g. versioning, conflict resolution on parallel stages) need
touch only one file.

Invariant: every function here is **pure** — takes lineage + params, returns
a new lineage. No I/O, no state-store, no logger side effects. The caller
(``AttemptController``, coming in PR-A4) is responsible for persistence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.pipeline.artifacts.base import utc_now_iso
from src.pipeline.state.models import (
    PipelineAttemptState,
    StageLineageRef,
    StageRunState,
)
from src.pipeline.state.store import update_lineage
from src.pipeline.state.transitioner import mark_stage_skipped

if TYPE_CHECKING:
    from collections.abc import Callable

# -----------------------------------------------------------------------------
# Pre-loop: invalidation and restoration
# -----------------------------------------------------------------------------


def invalidate_from(
    *,
    lineage: dict[str, StageLineageRef],
    stage_names: list[str],
    start_stage_name: str,
) -> dict[str, StageLineageRef]:
    """Return a copy of ``lineage`` with every entry from ``start_stage_name`` onward removed.

    Used when the user restarts from a specific stage — every downstream
    stage's previous outputs must be invalidated so the pipeline recomputes
    them.

    Raises ``ValueError`` if ``start_stage_name`` is not in ``stage_names``.
    """
    if start_stage_name not in stage_names:
        raise ValueError(f"Unknown stage name: {start_stage_name}")
    start_idx = stage_names.index(start_stage_name)
    new_lineage = dict(lineage)
    for name in stage_names[start_idx:]:
        new_lineage.pop(name, None)
    return new_lineage


def restore_reused(
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

    ``sync_root_from_stage`` is injected so this function stays independent
    of the ContextPropagator module.

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


# -----------------------------------------------------------------------------
# Per-stage updates (semantic wrappers over update_lineage)
# -----------------------------------------------------------------------------


def after_stage_completed(
    lineage: dict[str, StageLineageRef],
    *,
    stage_name: str,
    attempt_id: str,
    outputs: dict[str, Any],
) -> dict[str, StageLineageRef]:
    """Record a successful stage's outputs into the lineage.

    Wrapper over :func:`update_lineage` that reads as intent rather than
    mechanics: readers see "after stage completed" instead of "update with
    outputs=..., remove=False".
    """
    return update_lineage(
        lineage,
        stage_name=stage_name,
        attempt_id=attempt_id,
        outputs=outputs,
    )


def after_stage_failed(
    lineage: dict[str, StageLineageRef],
    *,
    stage_name: str,
    attempt_id: str,
) -> dict[str, StageLineageRef]:
    """Drop a failed stage's entry from the lineage — its outputs are not authoritative."""
    return update_lineage(
        lineage,
        stage_name=stage_name,
        attempt_id=attempt_id,
        remove=True,
    )


def after_stage_skipped(
    lineage: dict[str, StageLineageRef],
    *,
    stage_name: str,
    attempt_id: str,
) -> dict[str, StageLineageRef]:
    """Drop a skipped stage's entry — same mechanics as failed, but conveys intent."""
    return update_lineage(
        lineage,
        stage_name=stage_name,
        attempt_id=attempt_id,
        remove=True,
    )


__all__ = [
    "after_stage_completed",
    "after_stage_failed",
    "after_stage_skipped",
    "invalidate_from",
    "restore_reused",
]

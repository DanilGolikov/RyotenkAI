"""Sole owner of ``PipelineState`` mutations for a single pipeline run.

Context
-------
Before this module existed, eight different orchestrator sites mutated
``_pipeline_state`` and two sites mutated ``_current_attempt``; each mutation
was followed (hopefully) by an ``_save_state()`` call. The invariant "every
state mutation is durable" was enforced by developer discipline, not by the
type system — forgetting a single ``_save_state`` silently discarded progress
on a crash.

``AttemptController`` collapses those mutation sites into a single class:

* There is **one** writer of ``PipelineState`` for the lifetime of a run.
* Every public mutator method auto-persists at the end via an injected
  ``save_fn: Callable[[PipelineState], None]``. Callers cannot forget to save
  because they cannot call ``state_store.save()`` directly anymore.
* Lineage updates and stage-state transitions (``mark_stage_*``) are
  encapsulated — the orchestrator no longer imports them.

Design choices
--------------
1. *Auto-save after each public method.* Invariant #9 of the refactor plan:
   "every public method of AttemptController produces a file-write". This is
   enforced structurally via ``_persist()`` in each mutator.

2. *save_fn Callable instead of owning the state_store.* The controller needs
   exactly one operation on the store — ``save(state)``. Injecting a callable
   (Interface Segregation + Dependency Inversion) keeps the controller
   decoupled from ``PipelineStateStore`` and trivially mockable
   (``save_fn=Mock()``).

3. *``snapshot()`` returns a deepcopy.* Red-flag #3 of the plan: "``snapshot()``
   returning a live reference silently breaks invariants". Downstream readers
   must not be able to mutate the controller's state.

4. *Pure collaborators.* ``lineage_manager`` functions are pure; transitioner
   ``mark_stage_*`` functions mutate the attempt dataclass in place. The
   controller composes them without re-implementing their logic.

This module is **free of I/O** apart from the injected ``save_fn``. No
logging, no MLflow, no signal handling. The orchestrator owns those concerns.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from src.pipeline.state import lineage_manager
from src.pipeline.state.models import (
    PipelineAttemptState,
    PipelineState,
    StageLineageRef,
    StageRunState,
)
from src.pipeline.state.transitioner import (
    finalize_attempt_state,
    mark_stage_completed,
    mark_stage_failed,
    mark_stage_interrupted,
    mark_stage_running,
    mark_stage_skipped,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class AttemptControllerError(RuntimeError):
    """Raised when controller invariants are violated (no state / no attempt)."""


class AttemptController:
    """Single-writer manager of ``PipelineState`` and the active ``PipelineAttemptState``.

    Lifecycle
    ---------
    1. Construct with ``save_fn`` and ``run_ctx``.
    2. Call ``adopt_state(state)`` after the state has been loaded or
       initialised by ``LaunchPreparator``.
    3. Call ``register_attempt(attempt)`` to mark an attempt as active.
    4. Use ``record_*`` / ``invalidate_lineage_from`` / ``restore_reused_context``
       throughout stage execution.
    5. Call ``finalize(status, completed_at)`` once the run is done.
    6. Read ``snapshot()`` (deep-copied) for downstream inspection.

    Any attempt to mutate without having called ``adopt_state`` raises
    ``AttemptControllerError`` — fail-fast rather than silently no-op.
    """

    __slots__ = (
        "_active_attempt",
        "_run_ctx",
        "_save_fn",
        "_state",
    )

    def __init__(
        self,
        *,
        save_fn: Callable[[PipelineState], None],
        run_ctx: Any,
    ) -> None:
        self._save_fn = save_fn
        self._run_ctx = run_ctx
        self._state: PipelineState | None = None
        self._active_attempt: PipelineAttemptState | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_state(self) -> PipelineState:
        if self._state is None:
            raise AttemptControllerError(
                "AttemptController: state has not been adopted. "
                "Call adopt_state() before mutating."
            )
        return self._state

    def _require_attempt(self) -> PipelineAttemptState:
        if self._active_attempt is None:
            raise AttemptControllerError(
                "AttemptController: no active attempt. "
                "Call register_attempt() before recording stage transitions."
            )
        return self._active_attempt

    def _persist(self) -> None:
        """Persist the state via the injected save_fn.

        Called at the end of every public mutator. This is the single point
        where durability is enforced — if the controller cannot persist, the
        caller's exception propagates (no silent failure).
        """
        if self._state is None:
            # Nothing to persist — adopt_state() was never called. The
            # invariant check in the mutator would have already raised, so
            # this branch should be unreachable, but we guard defensively.
            return
        self._save_fn(self._state)

    # ------------------------------------------------------------------
    # State adoption / lifecycle
    # ------------------------------------------------------------------

    @property
    def state(self) -> PipelineState:
        """Read-only access to the live state. Returns the live object.

        For downstream code that must not mutate, prefer ``snapshot()``.
        """
        return self._require_state()

    @property
    def active_attempt(self) -> PipelineAttemptState:
        """Read-only access to the live active attempt."""
        return self._require_attempt()

    @property
    def has_state(self) -> bool:
        """True iff ``adopt_state()`` has been called."""
        return self._state is not None

    @property
    def has_active_attempt(self) -> bool:
        """True iff an attempt is currently active."""
        return self._active_attempt is not None

    def adopt_state(self, state: PipelineState) -> None:
        """Accept a freshly loaded or initialised ``PipelineState``.

        Does NOT persist — the state is typically already on disk (just loaded)
        or has been persisted by ``state_store.init_state()``. Calling
        ``_persist()`` here would be redundant I/O.
        """
        self._state = state

    def register_attempt(self, attempt: PipelineAttemptState) -> None:
        """Append ``attempt`` to ``state.attempts`` and mark it active.

        Also flips ``pipeline_status`` to RUNNING (this is the canonical
        "pipeline has live work" signal). Persists at the end.
        """
        state = self._require_state()
        state.attempts.append(attempt)
        state.active_attempt_id = attempt.attempt_id
        state.pipeline_status = StageRunState.STATUS_RUNNING
        self._active_attempt = attempt
        self._persist()

    def record_rejected_attempt(
        self,
        *,
        attempt: PipelineAttemptState,
        status: str,
        completed_at: str,
    ) -> None:
        """Append a pre-rejected attempt and finalise it in one atomic persist.

        Unlike ``register_attempt`` followed by ``finalize``, this method
        never emits a transient ``pipeline_status=RUNNING`` snapshot: it
        appends the attempt with its final status (usually FAILED) and
        writes exactly once. Use this for launch-rejection paths where the
        attempt never actually starts executing stages.

        The attempt is NOT marked as the "active attempt" — it never went
        live — so subsequent ``record_*`` calls will fail fast rather than
        mutating a rejected record.
        """
        state = self._require_state()
        state.attempts.append(attempt)
        attempt.status = status
        attempt.completed_at = completed_at
        state.pipeline_status = status
        state.active_attempt_id = None
        self._persist()

    # ------------------------------------------------------------------
    # Stage transitions (auto-persist)
    # ------------------------------------------------------------------

    def record_running(self, *, stage_name: str, started_at: str) -> None:
        """Mark a stage as RUNNING in the active attempt."""
        attempt = self._require_attempt()
        mark_stage_running(attempt=attempt, stage_name=stage_name, started_at=started_at)
        self._persist()

    def record_completed(
        self,
        *,
        stage_name: str,
        outputs: dict[str, Any],
    ) -> None:
        """Mark a stage COMPLETED and record its outputs in the lineage."""
        state = self._require_state()
        attempt = self._require_attempt()
        mark_stage_completed(attempt=attempt, stage_name=stage_name, outputs=outputs)
        state.current_output_lineage = lineage_manager.after_stage_completed(
            state.current_output_lineage,
            stage_name=stage_name,
            attempt_id=attempt.attempt_id,
            outputs=outputs,
        )
        self._persist()

    def record_failed(
        self,
        *,
        stage_name: str,
        error: str,
        failure_kind: str,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        """Mark a stage FAILED and drop its entry from the lineage."""
        state = self._require_state()
        attempt = self._require_attempt()
        mark_stage_failed(
            attempt=attempt,
            stage_name=stage_name,
            error=error,
            failure_kind=failure_kind,
            outputs=outputs,
        )
        state.current_output_lineage = lineage_manager.after_stage_failed(
            state.current_output_lineage,
            stage_name=stage_name,
            attempt_id=attempt.attempt_id,
        )
        self._persist()

    def record_skipped(
        self,
        *,
        stage_name: str,
        reason: str,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        """Mark a stage SKIPPED and drop its entry from the lineage."""
        state = self._require_state()
        attempt = self._require_attempt()
        mark_stage_skipped(
            attempt=attempt,
            stage_name=stage_name,
            reason=reason,
            outputs=outputs,
        )
        state.current_output_lineage = lineage_manager.after_stage_skipped(
            state.current_output_lineage,
            stage_name=stage_name,
            attempt_id=attempt.attempt_id,
        )
        self._persist()

    def record_interrupted(self, *, stage_name: str, started_at: str) -> None:
        """Mark a stage INTERRUPTED (e.g. SIGINT)."""
        attempt = self._require_attempt()
        mark_stage_interrupted(
            attempt=attempt,
            stage_name=stage_name,
            started_at=started_at,
        )
        self._persist()

    def record_stage_log_paths(
        self,
        *,
        stage_name: str,
        log_paths: dict[str, str],
    ) -> None:
        """Attach per-stage log-file registry to the stage's StageRunState.

        Called right after ``record_running`` so the UI/reporter can surface
        log links for a still-running stage.
        """
        attempt = self._require_attempt()
        stage_state = attempt.stage_runs.get(stage_name)
        if stage_state is None:
            return
        stage_state.log_paths = dict(log_paths)
        self._persist()

    # ------------------------------------------------------------------
    # Lineage manipulation (pre-loop)
    # ------------------------------------------------------------------

    def invalidate_lineage_from(
        self,
        *,
        stage_names: list[str],
        start_stage_name: str,
    ) -> dict[str, StageLineageRef]:
        """Drop lineage entries from ``start_stage_name`` onward.

        Mutates ``state.current_output_lineage`` in place and persists.
        Returns the updated mapping (same object as ``state.current_output_lineage``).
        """
        state = self._require_state()
        state.current_output_lineage = lineage_manager.invalidate_from(
            lineage=state.current_output_lineage,
            stage_names=stage_names,
            start_stage_name=start_stage_name,
        )
        self._persist()
        return state.current_output_lineage

    def restore_reused_context(
        self,
        *,
        stage_names: list[str],
        start_stage_name: str,
        enabled_stage_names: list[str],
        context: dict[str, Any],
        sync_root_from_stage: Callable[[dict[str, Any], str, dict[str, Any]], None],
    ) -> None:
        """Materialise completed-and-reused stages into context + attempt state.

        Delegates to :func:`lineage_manager.restore_reused` — the controller
        adds the required ``attempt`` + ``lineage`` arguments and persists.
        """
        state = self._require_state()
        attempt = self._require_attempt()
        lineage_manager.restore_reused(
            attempt=attempt,
            lineage=state.current_output_lineage,
            stage_names=stage_names,
            start_stage_name=start_stage_name,
            enabled_stage_names=enabled_stage_names,
            context=context,
            sync_root_from_stage=sync_root_from_stage,
        )
        self._persist()

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(
        self,
        *,
        status: str,
        completed_at: str | None = None,
    ) -> None:
        """Finalize the active attempt and the pipeline status.

        Also clears ``active_attempt_id`` (the canonical "no live work" signal)
        via :func:`finalize_attempt_state`. If no attempt is active, only the
        top-level pipeline status is updated — this covers the
        "unexpected-error-before-attempt-built" branch.
        """
        state = self._require_state()
        if self._active_attempt is not None:
            finalize_attempt_state(
                state=state,
                attempt=self._active_attempt,
                status=status,
                completed_at=completed_at,
            )
        else:
            state.pipeline_status = status
        self._persist()

    def mark_attempt_completed_at(self, *, completed_at: str) -> None:
        """Stamp the active attempt's ``completed_at`` without persisting yet.

        Used by the interrupt/unexpected-error paths that need to set the
        timestamp before calling ``finalize`` (which then persists). Keeping
        this as a no-persist primitive avoids an extra write in the very
        common "stamp then finalize" pairing.
        """
        if self._active_attempt is not None:
            self._active_attempt.completed_at = completed_at

    def record_attempt_error(self, *, error: str) -> None:
        """Record an error message on the active attempt without persisting.

        Intended for ``_record_launch_rejection_attempt``-style flows that
        attach the rejection reason to the attempt before finalising it.
        """
        if self._active_attempt is not None:
            self._active_attempt.error = error

    def set_mlflow_run_ids(
        self,
        *,
        root_run_id: str | None = None,
        attempt_run_id: str | None = None,
    ) -> None:
        """Record MLflow run ids on state + active attempt.

        Kept as a dedicated mutator (rather than raw attribute writes) so
        persistence stays inside the controller. ``None`` values are ignored —
        pass only the ids you actually want to set.
        """
        state = self._require_state()
        if root_run_id is not None:
            state.root_mlflow_run_id = root_run_id
            if self._active_attempt is not None:
                self._active_attempt.root_mlflow_run_id = root_run_id
        if attempt_run_id is not None and self._active_attempt is not None:
            self._active_attempt.pipeline_attempt_mlflow_run_id = attempt_run_id
        self._persist()

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    def snapshot(self) -> PipelineState:
        """Return a deepcopy of the current state for safe downstream reads.

        Necessary because consumers (e.g. the execution loop) must not be able
        to mutate controller-owned state without going through the public API.
        """
        state = self._require_state()
        return copy.deepcopy(state)

    def active_attempt_id(self) -> str | None:
        """Convenience: the live ``attempt_id`` or ``None`` if no attempt is active."""
        return self._active_attempt.attempt_id if self._active_attempt is not None else None

    def save(self) -> None:
        """Explicit persist — for edge cases where state was mutated out-of-band.

        The orchestrator should NOT reach into the state dataclass directly,
        but a handful of legacy code paths (e.g. teardown-time final writes)
        still do. This method lets them persist without re-implementing the
        save logic, while keeping the save_fn owned solely by the controller.

        Prefer the semantic mutators (``record_*``, ``finalize``) over this.
        """
        self._persist()


__all__ = [
    "AttemptController",
    "AttemptControllerError",
]

"""Launch-time preparation for a pipeline run.

This module owns the sequence of operations that MUST happen before the
stage loop begins:

1. Resolve the on-disk run directory (or create it for fresh runs).
2. Create a :class:`PipelineStateStore` bound to that directory.
3. Load existing ``pipeline_state.json`` (resume / restart) or initialise a
   new one (fresh). Adopt the state into the injected ``AttemptController``
   so downstream mutations are routed through a single writer.
4. Derive ``requested_action`` / ``effective_action`` / ``start_stage_name``.
5. Validate config drift against the persisted state; raise
   :class:`LaunchPreparationError` on mismatch.
6. Compute ``start_idx`` / ``stop_idx`` / ``enabled_stage_names``.
7. Build a per-run :class:`PipelineAttemptState` (not yet registered — that's
   the orchestrator's call) and the matching attempt directory + log layout.
8. Stamp freshest config hashes onto ``PipelineState`` (they're the canonical
   source of truth for this run, overriding anything loaded from disk).

Everything is returned as a frozen :class:`PreparedAttempt` value object so
the orchestrator's launch code reads like assembly instead of another god
method with twelve local mutables.

**Deliberately out-of-scope** (these stay in the orchestrator):

* Acquiring the run lock — ``RunLockGuard`` is a cross-cutting concern with
  its own lifecycle tied to the orchestrator's ``finally``.
* Setting up MLflow — :class:`MLflowAttemptManager` (and its successor
  ``MLflowRunHierarchy``) live outside the preparation path.
* Forking the pipeline context — the orchestrator owns the live context
  object and the per-attempt fork.
* Registering the attempt into ``AttemptController.register_attempt`` and
  invalidating lineage — those are orchestrator calls that must happen
  AFTER the lock is acquired and BEFORE MLflow setup.

Why the split? Because a tight preparator is (a) side-effect minimal, so it
is trivial to test in isolation, and (b) reusable by the future web-API
"dry-run" path which needs to compute `start_stage_name` / drift checks
without actually starting a run.

.. note::
   On the rejection path (``prepare`` raises :class:`LaunchPreparationError`),
   the caller must call :meth:`record_launch_rejection` to emit a terminal
   failed-attempt record into state. See the method docstring for the exact
   invariants this guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.pipeline.artifacts.base import utc_now_iso
from src.pipeline.state import (
    AttemptController,
    PipelineAttemptState,
    PipelineState,
    PipelineStateLoadError,
    PipelineStateStore,
    StageRunState,
    build_attempt_state,
)
from src.utils.logger import init_run_logging, logger
from src.utils.logs_layout import LogLayout
from src.utils.result import AppError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.config.runtime import RuntimeSettings
    from src.pipeline.config_drift import ConfigDriftValidator
    from src.pipeline.state import RunContext
    from src.pipeline.execution import StagePlanner
    from src.pipeline.stages.base import PipelineStage

_SEPARATOR_CHAR = "="
_SEPARATOR_LINE_WIDTH = 80


class LaunchPreparationError(Exception):
    """Raised when preparation cannot produce a :class:`PreparedAttempt`.

    Carries the partially-computed context needed to record a rejected
    launch attempt (see :meth:`LaunchPreparator.record_launch_rejection`).
    Fields are optional because very-early failures (e.g. invalid run dir)
    may not have a state to attach.
    """

    def __init__(
        self,
        app_error: AppError,
        *,
        state: PipelineState | None = None,
        requested_action: str | None = None,
        effective_action: str | None = None,
        start_stage_name: str | None = None,
    ) -> None:
        super().__init__(app_error.message)
        self.app_error = app_error
        self.state = state
        self.requested_action = requested_action
        self.effective_action = effective_action
        self.start_stage_name = start_stage_name


@dataclass(frozen=True, slots=True)
class PreparedAttempt:
    """Immutable launch plan for one pipeline run.

    Returned by :meth:`LaunchPreparator.prepare` after all pre-loop work has
    completed successfully. The orchestrator uses it to:

    * Acquire the run lock (``state_store.lock_path``).
    * Fork the run-scoped context into an attempt-scoped context
      (``attempt_id``, ``attempt_no``, ``attempt_directory`` fields).
    * Register the attempt into the ``AttemptController``.
    * Drive the stage loop between ``start_idx`` and ``stop_idx``.

    All fields are read-only; mutations to any underlying object (state,
    attempt) must go through ``AttemptController``.
    """

    state: PipelineState
    attempt: PipelineAttemptState
    state_store: PipelineStateStore
    run_directory: Path
    attempt_directory: Path
    log_layout: LogLayout
    logical_run_id: str
    requested_action: str
    effective_action: str
    start_stage_name: str
    start_idx: int
    stop_idx: int
    enabled_stage_names: tuple[str, ...]
    forced_stage_names: frozenset[str] = field(default_factory=frozenset)


class LaunchPreparator:
    """Builder for :class:`PreparedAttempt` with rejection-recording support.

    The preparator is *one-shot per run* — it caches the state_store it
    created during ``prepare`` on ``self`` so the rejection-recording path
    (which runs after ``prepare`` raises) can reuse it. Reusing the same
    preparator across multiple runs is unsupported and will behave
    surprisingly; orchestrators should create a fresh instance per run.
    """

    __slots__ = (
        "_attempt_controller",
        "_config_drift",
        "_config_path",
        "_last_logical_run_id",
        "_last_run_directory",
        "_last_state_store",
        "_run_ctx",
        "_settings",
        "_stage_planner",
        "_stages",
        "_state_store_factory",
    )

    def __init__(
        self,
        *,
        config_path: Path,
        run_ctx: RunContext,
        settings: RuntimeSettings,
        stages: list[PipelineStage],
        stage_planner: StagePlanner,
        config_drift: ConfigDriftValidator,
        attempt_controller: AttemptController,
        state_store_factory: Callable[[Path], PipelineStateStore] = PipelineStateStore,
    ) -> None:
        self._config_path = config_path
        self._run_ctx = run_ctx
        self._settings = settings
        self._stages = stages
        self._stage_planner = stage_planner
        self._config_drift = config_drift
        self._attempt_controller = attempt_controller
        self._state_store_factory = state_store_factory

        # Populated during prepare() and read by record_launch_rejection()
        # if prepare() raised. Intentionally mutable: exception recovery
        # needs to consult the partial work.
        self._last_state_store: PipelineStateStore | None = None
        self._last_run_directory: Path | None = None
        self._last_logical_run_id: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(
        self,
        *,
        run_dir: Path | None,
        resume: bool,
        restart_from_stage: str | int | None,
        config_hashes: dict[str, str],
    ) -> PreparedAttempt:
        """Assemble a :class:`PreparedAttempt` or raise on rejection.

        Side effects:

        * Creates ``run_directory`` on disk for fresh runs.
        * Creates the state_store file for fresh runs; loads it for
          resume/restart.
        * Adopts the state into ``AttemptController`` (single-writer).
        * Initialises run-level logging pointed at the attempt directory.

        Raises:
          LaunchPreparationError: on config drift or unresumable state.
          PipelineStateLoadError: when resume/restart is requested but the
            target run directory has no ``pipeline_state.json``.
        """
        state, requested_action, effective_action, start_stage_name = self._bootstrap_state(
            run_dir=run_dir,
            resume=resume,
            restart_from_stage=restart_from_stage,
            config_hashes=config_hashes,
        )
        state_store = self._require_state_store()

        # Stamp canonical config hashes — the values we loaded from disk were
        # authoritative for the PREVIOUS run; this run's hashes are the truth.
        state.training_critical_config_hash = config_hashes["training_critical"]
        state.late_stage_config_hash = config_hashes["late_stage"]
        state.model_dataset_config_hash = config_hashes["model_dataset"]

        start_idx = self._stage_planner.get_stage_index(start_stage_name)
        stop_idx = len(self._stages)
        enabled_stage_names = self._stage_planner.compute_enabled_stage_names(
            start_stage_name=start_stage_name
        )
        forced_stage_names = self._stage_planner.forced_stage_names(
            start_stage_name=start_stage_name
        )

        attempt = build_attempt_state(
            state=state,
            run_ctx=self._run_ctx,
            requested_action=requested_action,
            effective_action=effective_action,
            restart_from_stage=start_stage_name,
            enabled_stage_names=enabled_stage_names,
            training_critical_config_hash=config_hashes["training_critical"],
            late_stage_config_hash=config_hashes["late_stage"],
            model_dataset_config_hash=config_hashes["model_dataset"],
        )

        attempt_directory = state_store.next_attempt_dir(attempt.attempt_no)
        log_layout = LogLayout(attempt_directory)
        init_run_logging(self._run_ctx.name, log_dir=attempt_directory)

        run_directory = self._require_run_directory()

        return PreparedAttempt(
            state=state,
            attempt=attempt,
            state_store=state_store,
            run_directory=run_directory,
            attempt_directory=attempt_directory,
            log_layout=log_layout,
            logical_run_id=self._require_logical_run_id(),
            requested_action=requested_action,
            effective_action=effective_action,
            start_stage_name=start_stage_name,
            start_idx=start_idx,
            stop_idx=stop_idx,
            enabled_stage_names=tuple(enabled_stage_names),
            forced_stage_names=frozenset(forced_stage_names),
        )

    def record_launch_rejection(
        self,
        *,
        launch_error: LaunchPreparationError,
        config_hashes: dict[str, str],
    ) -> None:
        """Persist a terminal failed-attempt record for a rejected launch.

        Invariants:

        * The state_store is consulted via the preparator's cache populated
          by ``prepare``. If ``prepare`` raised BEFORE creating the state
          store (very-early failures), this is a no-op — there is no state
          file to write into.
        * Writes exactly one snapshot (via
          :meth:`AttemptController.record_rejected_attempt`), never emitting
          an intermediate ``RUNNING`` state that could fool crash recovery.
        """
        if self._last_state_store is None or launch_error.state is None:
            return
        # Invariant: prepare() always fills these three together; the
        # state_store check above guards the happy path.
        assert launch_error.start_stage_name is not None
        assert launch_error.requested_action is not None
        assert launch_error.effective_action is not None

        enabled_stage_names = self._stage_planner.compute_enabled_stage_names(
            start_stage_name=launch_error.start_stage_name
        )
        attempt = build_attempt_state(
            state=launch_error.state,
            run_ctx=self._run_ctx,
            requested_action=launch_error.requested_action,
            effective_action=launch_error.effective_action,
            restart_from_stage=launch_error.start_stage_name,
            enabled_stage_names=enabled_stage_names,
            training_critical_config_hash=config_hashes["training_critical"],
            late_stage_config_hash=config_hashes["late_stage"],
            model_dataset_config_hash=config_hashes["model_dataset"],
        )
        attempt_directory = self._last_state_store.next_attempt_dir(attempt.attempt_no)
        init_run_logging(self._run_ctx.name, log_dir=attempt_directory)
        logger.info(_SEPARATOR_CHAR * _SEPARATOR_LINE_WIDTH)
        logger.error(f"Launch rejected before stage execution: {launch_error.app_error}")
        logger.info(_SEPARATOR_CHAR * _SEPARATOR_LINE_WIDTH)

        attempt.error = launch_error.app_error.message
        self._attempt_controller.record_rejected_attempt(
            attempt=attempt,
            status=StageRunState.STATUS_FAILED,
            completed_at=utc_now_iso(),
        )

    # ------------------------------------------------------------------
    # Cached-state accessors (populated in prepare(), read on rejection)
    # ------------------------------------------------------------------

    @property
    def last_state_store(self) -> PipelineStateStore | None:
        """The state_store created by the most recent ``prepare`` call, if any."""
        return self._last_state_store

    @property
    def last_run_directory(self) -> Path | None:
        """The resolved run directory from the most recent ``prepare`` call."""
        return self._last_run_directory

    @property
    def last_logical_run_id(self) -> str | None:
        """The logical_run_id from the most recent ``prepare`` call."""
        return self._last_logical_run_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bootstrap_state(
        self,
        *,
        run_dir: Path | None,
        resume: bool,
        restart_from_stage: str | int | None,
        config_hashes: dict[str, str],
    ) -> tuple[PipelineState, str, str, str]:
        """Load or create the pipeline state file; derive action + start stage.

        Caches the created ``state_store``, ``run_directory``, and
        ``logical_run_id`` onto the preparator so :meth:`record_launch_rejection`
        can reuse them if this method raises.
        """
        requested_run_dir = run_dir
        normalized_restart = (
            self._stage_planner.normalize_stage_ref(restart_from_stage)
            if restart_from_stage is not None
            else None
        )

        if requested_run_dir is not None:
            resolved_run_dir = requested_run_dir.expanduser().resolve()
        else:
            resolved_run_dir = (self._settings.runs_base_dir / self._run_ctx.name).resolve()

        state_store = self._state_store_factory(resolved_run_dir)

        # Cache EAGERLY — record_launch_rejection must see these even if we
        # raise further down in this method.
        self._last_state_store = state_store
        self._last_run_directory = resolved_run_dir

        if state_store.exists():
            state = state_store.load()
            self._attempt_controller.adopt_state(state)
            self._last_logical_run_id = state.logical_run_id

            requested_action = "resume" if resume else ("restart" if normalized_restart else "fresh")
            effective_action = (
                "auto_resume" if resume and normalized_restart is None else requested_action
            )
            start_stage_name = normalized_restart or (
                self._stage_planner.derive_resume_stage(state)
                if resume
                else self._stages[0].stage_name
            )
            if start_stage_name is None:
                raise LaunchPreparationError(
                    AppError(
                        message="No resumable stage found in pipeline_state.json",
                        code="RESUME_NOT_AVAILABLE",
                    ),
                    state=state,
                    requested_action=requested_action,
                    effective_action=effective_action,
                    start_stage_name=self._stages[0].stage_name,
                )
            drift_error = self._config_drift.validate_drift(
                state=state,
                start_stage_name=start_stage_name,
                config_hashes=config_hashes,
                resume=resume,
            )
            if drift_error is not None:
                raise LaunchPreparationError(
                    drift_error,
                    state=state,
                    requested_action=requested_action,
                    effective_action=effective_action,
                    start_stage_name=start_stage_name,
                )
            return state, requested_action, effective_action, start_stage_name

        if resume or normalized_restart is not None:
            raise PipelineStateLoadError(
                f"Missing pipeline_state.json in run directory: {resolved_run_dir}"
            )

        # Fresh run — derive logical_run_id from the requested directory name
        # when explicit (resume pattern), otherwise from the run context.
        logical_run_id = (
            resolved_run_dir.name if run_dir is not None else self._run_ctx.name
        )
        resolved_run_dir.mkdir(parents=True, exist_ok=True)
        state = state_store.init_state(
            logical_run_id=logical_run_id,
            config_path=str(self._config_path.expanduser().resolve()),
            training_critical_config_hash=config_hashes["training_critical"],
            late_stage_config_hash=config_hashes["late_stage"],
            model_dataset_config_hash=config_hashes["model_dataset"],
        )
        self._attempt_controller.adopt_state(state)
        self._last_logical_run_id = logical_run_id
        return state, "fresh", "fresh", self._stages[0].stage_name

    def _require_state_store(self) -> PipelineStateStore:
        if self._last_state_store is None:
            raise RuntimeError(
                "LaunchPreparator: state_store has not been created yet; "
                "prepare() must be called first"
            )
        return self._last_state_store

    def _require_run_directory(self) -> Path:
        if self._last_run_directory is None:
            raise RuntimeError(
                "LaunchPreparator: run_directory has not been resolved yet; "
                "prepare() must be called first"
            )
        return self._last_run_directory

    def _require_logical_run_id(self) -> str:
        if self._last_logical_run_id is None:
            raise RuntimeError(
                "LaunchPreparator: logical_run_id has not been resolved yet; "
                "prepare() must be called first"
            )
        return self._last_logical_run_id


__all__ = [
    "LaunchPreparationError",
    "LaunchPreparator",
    "PreparedAttempt",
]

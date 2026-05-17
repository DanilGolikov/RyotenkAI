"""Run-lifecycle coordinator — extracts emitter wiring out of the orchestrator.

Before this module existed, ``PipelineOrchestrator`` carried the full
event-emitter lifecycle inline: lazy construction once the run directory
resolved, ``EventEmitterRegistry`` register/deregister, the four
``_emit_run_*`` helpers, the :class:`MlflowFinalizer` upload in the
``finally`` block, and emitter close. After Phase 6.a/6.b the file had
grown past 1000 lines and the lifecycle was the largest cohesive group
of responsibilities that did not need to coexist with stage execution.

:class:`RunLifecycleCoordinator` owns that group end-to-end so the
orchestrator can shrink back to "wire stages and run them" and so the
lifecycle has a single object to test in isolation. The contract is
narrow and idempotent:

* :meth:`bind_run_directory` — lazy-construct the emitter once
  :class:`LaunchPreparator` resolves the canonical run directory.
  Safe to call multiple times; subsequent calls return the cached
  emitter.
* :meth:`emit_run_started` / :meth:`emit_run_completed` /
  :meth:`emit_run_failed` / :meth:`emit_run_cancelled` — terminal
  event emissions. No-op when the emitter was never built (e.g.
  the run aborted in launch-prep before the directory was resolved).
* :meth:`finalize` — flush + close emitter, run
  :class:`MlflowFinalizer` upload, deregister from
  :class:`EventEmitterRegistry`. Idempotent — calling it twice does
  nothing on the second invocation.

The coordinator does NOT own:

* Stage execution / stage emitter wiring — orchestrator still pushes
  ``set_emitter`` onto each stage (it needs the stage list).
* MLflow attempt lifecycle (run open / close / preflight) — that is
  :class:`MLflowAttemptManager`'s job.
* Cleanup of GPU / collectors / run-lock — orchestrator's ``finally``
  block still drives those.

Per-emission dynamic data (algorithm string, dataset id, active stage
name, shutdown signal) is supplied via callables passed at construction
time. The coordinator never reaches back into the orchestrator's
internals, which keeps the dependency arrow one-way and the unit tests
small.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

from ryotenkai_control.events import (
    ControlEventEmitter,
    EventEmitterRegistry,
    MlflowFinalizer,
)
from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.events.types.control_run import (
    RunCancelledEvent,
    RunCancelledPayload,
    RunCompletedEvent,
    RunCompletedPayload,
    RunFailedEvent,
    RunFailedPayload,
    RunStartedEvent,
    RunStartedPayload,
)
from ryotenkai_shared.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ryotenkai_shared.infrastructure.mlflow.protocol import IMLflowManager
    from ryotenkai_shared.pipeline_context import RunContext


__all__ = ["RunLifecycleCoordinator"]


class RunLifecycleCoordinator:
    """Coordinates the :class:`ControlEventEmitter` + :class:`MlflowFinalizer`
    + :class:`EventEmitterRegistry` lifecycle for a single pipeline run.

    Lifecycle:

    .. code-block:: python

       coord = RunLifecycleCoordinator(
           run_ctx=...,
           mlflow_manager_supplier=lambda: orch._mlflow_manager,
           ...
       )
       coord.bind_run_directory(run_dir)         # lazy emitter construction
       coord.emit_run_started(config_hashes=...)
       try:
           # ... stages execute ...
           coord.emit_run_completed(duration_s=..., status="success")
       except KeyboardInterrupt:
           coord.emit_run_cancelled(reason="user_interrupt")
       except RyotenkAIError as e:
           coord.emit_run_failed(e)
       finally:
           coord.finalize(pipeline_success=success)

    The coordinator is **single-use**: once :meth:`finalize` runs the
    emitter is closed and any further ``emit_run_*`` call is a silent
    no-op. The orchestrator's ``finally`` block is the canonical
    finalize-site.
    """

    def __init__(
        self,
        *,
        run_ctx: RunContext,
        algorithm_supplier: Callable[[], str],
        dataset_id_supplier: Callable[[], str],
        model_id_supplier: Callable[[], str],
        mlflow_run_id_supplier: Callable[[], str | None],
        active_stage_supplier: Callable[[], str | None],
        shutdown_signal_supplier: Callable[[], str | None],
        mlflow_manager_supplier: Callable[[], IMLflowManager | None],
        pre_built_emitter: ControlEventEmitter | None = None,
    ) -> None:
        """Construct the coordinator.

        ``run_ctx`` carries the run id (``run_ctx.name``) used for
        registry registration and as fallback MLflow run id. The
        ``*_supplier`` callables let the coordinator pull per-emission
        dynamic data from the orchestrator without holding a strong
        reference back to it. ``pre_built_emitter`` is the optional
        emitter the bootstrap already constructed when the caller
        supplied ``run_directory`` up-front.
        """
        self._run_ctx = run_ctx
        self._algorithm_supplier = algorithm_supplier
        self._dataset_id_supplier = dataset_id_supplier
        self._model_id_supplier = model_id_supplier
        self._mlflow_run_id_supplier = mlflow_run_id_supplier
        self._active_stage_supplier = active_stage_supplier
        self._shutdown_signal_supplier = shutdown_signal_supplier
        self._mlflow_manager_supplier = mlflow_manager_supplier

        self._emitter: ControlEventEmitter | None = None
        self._registered = False
        self._closed = False

        if pre_built_emitter is not None:
            self._emitter = pre_built_emitter
            self._register_emitter()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def emitter(self) -> ControlEventEmitter | None:
        """Return the emitter, or ``None`` until :meth:`bind_run_directory` runs."""
        return self._emitter

    @property
    def is_finalized(self) -> bool:
        """``True`` once :meth:`finalize` has run."""
        return self._closed

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def bind_run_directory(
        self, run_directory: Path,
    ) -> ControlEventEmitter | None:
        """Lazy-construct the :class:`ControlEventEmitter` for ``run_directory``.

        Idempotent: if the emitter was already constructed (either by
        the bootstrap with a user-supplied ``run_directory`` or by a
        prior call) the cached value is returned untouched.

        On construction failure the helper logs but does NOT raise —
        the orchestrator must continue even if event persistence is
        broken (stages still need to run; observability degrades
        gracefully).
        """
        if self._emitter is not None:
            return self._emitter
        try:
            self._emitter = ControlEventEmitter.for_run(
                run_id=self._run_ctx.name,
                run_directory=run_directory,
            )
        except Exception:
            logger.exception(
                "[RunLifecycleCoordinator] failed to build event emitter — "
                "control-side events for this run will not be persisted",
            )
            self._emitter = None
            return None
        self._register_emitter()
        return self._emitter

    def _register_emitter(self) -> None:
        """Register the active emitter with the process-wide
        :class:`EventEmitterRegistry`.

        Phase 6.a: publishes the emitter so the API SSE router
        (``/api/v1/runs/{id}/events/stream``) can subscribe to the
        in-memory bus without holding a reference to the orchestrator.
        Matching ``deregister`` lives in :meth:`finalize` so a crashed
        run still releases the slot.
        """
        if self._emitter is None or self._registered:
            return
        try:
            EventEmitterRegistry.instance().register(
                self._run_ctx.name, self._emitter,
            )
            self._registered = True
        except Exception:
            logger.exception(
                "[RunLifecycleCoordinator] failed to register emitter in "
                "EventEmitterRegistry — SSE live-tail will not see this run",
            )

    # ------------------------------------------------------------------
    # Run-terminal emissions
    # ------------------------------------------------------------------

    def emit_run_started(
        self,
        *,
        config_hashes: dict[str, str],
    ) -> None:
        """Emit :class:`RunStartedEvent` if the emitter is available.

        Pulls run-level metadata from the algorithm/dataset suppliers
        and the ``config_hashes`` snapshot. Best-effort: missing fields
        use sensible defaults so the envelope always validates.
        """
        if self._emitter is None or self._closed:
            return
        try:
            algorithm = self._algorithm_supplier()
            dataset_id = self._dataset_id_supplier()
            config_hash = (
                config_hashes.get("merged")
                or config_hashes.get("config")
                or next(iter(config_hashes.values()), "unknown")
            )
            # ``model_id`` is sourced via the algorithm path historically —
            # the orchestrator was reading ``self.config.model.name``. We
            # keep the same default here so the envelope stays identical.
            self._emitter.emit(
                RunStartedEvent(
                    source=self._emitter.source,
                    run_id=self._emitter.run_id,
                    offset=UNKNOWN_OFFSET,
                    payload=RunStartedPayload(
                        run_name=self._run_ctx.name,
                        algorithm=algorithm,
                        model_id=self._model_id_or_default(),
                        dataset_id=dataset_id,
                        config_hash=config_hash,
                    ),
                ),
            )
        except Exception:
            logger.exception(
                "[RunLifecycleCoordinator] RunStartedEvent emission failed",
            )

    def emit_run_completed(
        self,
        *,
        duration_s: float,
        status: str,
    ) -> None:
        """Emit :class:`RunCompletedEvent` if the emitter is available."""
        if self._emitter is None or self._closed:
            return
        try:
            mlflow_run_id = self._mlflow_run_id_supplier()
            self._emitter.emit(
                RunCompletedEvent(
                    source=self._emitter.source,
                    run_id=self._emitter.run_id,
                    offset=UNKNOWN_OFFSET,
                    payload=RunCompletedPayload(
                        duration_s=duration_s,
                        final_status=status,
                        mlflow_run_id=mlflow_run_id,
                    ),
                ),
            )
        except Exception:
            logger.exception(
                "[RunLifecycleCoordinator] RunCompletedEvent emission failed",
            )

    def emit_run_failed(self, exc: BaseException) -> None:
        """Emit :class:`RunFailedEvent` if the emitter is available."""
        if self._emitter is None or self._closed:
            return
        try:
            failing_stage = self._active_stage_supplier()
            tb = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__),
            )
            self._emitter.emit(
                RunFailedEvent(
                    source=self._emitter.source,
                    run_id=self._emitter.run_id,
                    offset=UNKNOWN_OFFSET,
                    payload=RunFailedPayload(
                        failing_stage=failing_stage or "unknown",
                        error_type=type(exc).__name__,
                        message=str(exc)[:2048],
                        traceback_excerpt=tb[:2048],
                    ),
                ),
            )
        except Exception:
            logger.exception(
                "[RunLifecycleCoordinator] RunFailedEvent emission failed",
            )

    def emit_run_cancelled(self, *, reason: str) -> None:
        """Emit :class:`RunCancelledEvent` if the emitter is available."""
        if self._emitter is None or self._closed:
            return
        try:
            self._emitter.emit(
                RunCancelledEvent(
                    source=self._emitter.source,
                    run_id=self._emitter.run_id,
                    offset=UNKNOWN_OFFSET,
                    payload=RunCancelledPayload(
                        reason=reason,
                        cancelled_at_stage=self._active_stage_supplier(),
                    ),
                ),
            )
        except Exception:
            logger.exception(
                "[RunLifecycleCoordinator] RunCancelledEvent emission failed",
            )

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self, *, pipeline_success: bool) -> None:
        """Flush + close emitter, upload journal to MLflow, deregister.

        Order (matches the pre-refactor orchestrator finally block):

        1. :meth:`ControlEventEmitter.close` — final fsync, sweeper stop.
        2. :class:`MlflowFinalizer` upload with cancellation/journal-
           completeness flags derived from the shutdown signal and the
           ``pipeline_success`` outcome.
        3. :class:`EventEmitterRegistry` deregister so a crashed run
           still releases its slot.

        Each step is wrapped in its own try/except so a failure in one
        does not skip the others. Idempotent — calling :meth:`finalize`
        a second time is a no-op.
        """
        if self._closed:
            return
        # Mark closed first so any racy emit_run_* call after finalize
        # short-circuits via the ``self._closed`` guard.
        self._closed = True
        emitter = self._emitter
        if emitter is None:
            # Even with no emitter, attempt registry cleanup defensively
            # (it's a no-op when the slot is absent).
            self._safe_deregister()
            return
        try:
            emitter.close()
        except Exception:
            logger.exception(
                "[RunLifecycleCoordinator] event emitter close failed",
            )
        try:
            self._finalize_events_to_mlflow(pipeline_success=pipeline_success)
        except Exception:
            logger.exception(
                "[RunLifecycleCoordinator] MLflow events finalize failed",
            )
        self._safe_deregister()

    def _safe_deregister(self) -> None:
        try:
            EventEmitterRegistry.instance().deregister(
                self._run_ctx.name,
            )
        except Exception:
            logger.exception(
                "[RunLifecycleCoordinator] EventEmitterRegistry.deregister failed",
            )

    def _finalize_events_to_mlflow(self, *, pipeline_success: bool) -> None:
        """Upload events.jsonl + manifest to MLflow via :class:`MlflowFinalizer`.

        Cancellation reason and the ``journal_complete`` flag are derived
        from the shutdown signal supplier (populated by the
        SIGINT/SIGTERM handlers / KeyboardInterrupt branch) and the
        ``pipeline_success`` outcome.
        """
        emitter = self._emitter
        if emitter is None:
            return
        manager = self._mlflow_manager_supplier()
        if manager is None:
            logger.debug(
                "[RunLifecycleCoordinator] MLflow manager unavailable; "
                "skipping events finalize",
            )
            return
        journal_path = emitter.journal.path
        if not journal_path.exists():
            logger.debug(
                "[RunLifecycleCoordinator] journal missing at finalize "
                "time; skipping",
            )
            return
        cancellation_reason: str | None = None
        journal_complete = True
        shutdown_signal = self._shutdown_signal_supplier()
        if shutdown_signal:
            cancellation_reason = f"signal:{shutdown_signal}"
            journal_complete = False
        elif not pipeline_success:
            # Failure case — journal IS complete (the orchestrator wrote
            # the RunFailedEvent before entering finalize), but consumers
            # may still want to distinguish success from failure uploads.
            cancellation_reason = None
        mlflow_run_id = (
            self._mlflow_run_id_supplier() or self._run_ctx.name
        )
        finalizer = MlflowFinalizer(manager)
        finalizer.upload(
            run_id=mlflow_run_id,
            journal_path=journal_path,
            cancellation_reason=cancellation_reason,
            journal_complete=journal_complete,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _model_id_or_default(self) -> str:
        """Resolve the ``model_id`` for :class:`RunStartedPayload`."""
        try:
            return self._model_id_supplier() or "unknown"
        except Exception:  # pragma: no cover — defensive
            return "unknown"

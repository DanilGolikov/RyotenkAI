"""HuggingFace ``TrainerCallback`` for terminal-state finalization.

Unified replacement for the historical ``CancellationCallback`` (Phase 9.B/9.C)
and ``CompletionCallback`` (Phase 11.A) -- the two callbacks shared ~80% of
their structure (marker write, telemetry emit, workspace resolution, run_id
pull) so we collapse them into a single parametric class driven by
``reason``.

The two operating modes:

* ``reason="cancel"`` -- the cooperative cancellation path. On
  ``on_step_end`` we pump the global :class:`ShutdownHandler` flag into
  HF's ``TrainerControl`` so the trainer saves + exits at the next
  checkpoint boundary. On ``on_train_end`` (only when the cancel flag
  was raised during this run) we drain the buffered MLflow records into
  the live run before HF closes it, then write
  ``<workspace>/cancelled.marker`` on flush-budget overrun for Mac-side
  reconciliation.
* ``reason="complete"`` -- the natural-completion path. ``on_step_end``
  is a no-op. On ``on_train_end`` (only when the cancel flag was NOT
  raised -- the cancellation owner handles its own terminal path) we
  drain the buffer and unconditionally write
  ``<workspace>/completion.marker`` so Mac-side reconciliation can
  finalize the upstream MLflow run if it was still ``RUNNING``.

Signal-safety contract is identical to the pre-merge callbacks: HF
Trainer calls these hooks from the main train loop, never from a
signal handler. ``ShutdownHandler``'s signal handler only flips a flag;
the callback reads it from the regular Python thread, free to call
HTTP / MLflow / whatever from a worker thread (the timeout helper).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from transformers import TrainerCallback

from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import TrainerControl, TrainerState, TrainingArguments


logger = get_logger(__name__)


__all__ = ["TerminalCallback"]


TerminalReason = Literal["cancel", "complete"]


class TerminalCallback(TrainerCallback):
    """Cooperative-cancellation / natural-completion finalization HF callback.

    Parametric over ``reason``:

    * ``reason="cancel"`` -- pumps the ShutdownHandler flag into
      HF ``TrainerControl`` on each ``on_step_end``; on ``on_train_end``
      drains the resilient MLflow transport buffer and writes
      ``cancelled.marker`` on flush-budget overrun.
    * ``reason="complete"`` -- on ``on_train_end`` (only when no
      cancellation was observed during the run) drains the MLflow
      buffer and unconditionally writes ``completion.marker``.

    Construction is fully dependency-injected so unit tests can pass
    stubbed shutdown handlers, MLflow managers, event publishers,
    and marker writers without registering real signal handlers or
    poking the filesystem.
    """

    #: Hard deadline for the on_train_end MLflow flush. MLflow client
    #: has its own internal retry up to ~60s -- without an explicit
    #: ceiling here the SIGKILL escalation (Supervisor's ``--grace``,
    #: default 30s) would pre-empt the flush. 5 seconds is enough for
    #: a typical buffered drain (small JSONL -> http POST per record)
    #: and leaves headroom for the rest of HF's on_train_end work.
    FLUSH_TIMEOUT_SECONDS: float = 5.0

    def __init__(
        self,
        *,
        reason: TerminalReason,
        shutdown_handler: object | None = None,
        mlflow_manager: object | None = None,
        flush_timeout_seconds: float | None = None,
        event_publisher: Callable[..., Any] | None = None,
        marker_writer: Callable[..., Any] | None = None,
        workspace_dir: object | None = None,
    ) -> None:
        """Build a TerminalCallback.

        :param reason: ``"cancel"`` or ``"complete"`` -- selects the
            on_train_end behaviour and the marker filename.
        :param shutdown_handler: object exposing ``.should_stop() -> bool``.
            When ``None`` (production default) we lazily resolve the
            global handler on first poll. Tests inject a stub.
        :param mlflow_manager: object exposing ``.flush_buffer() -> int``.
            When ``None`` we silently skip the flush -- local-mode
            trainings without MLflow tracking land here.
        :param flush_timeout_seconds: override the class-level
            ``FLUSH_TIMEOUT_SECONDS`` budget -- tests use a tiny value
            (e.g. 0.1s) to assert the timeout path quickly.
        :param event_publisher: callable ``(kind: str, payload: dict) -> None``
            that emits a structured event into the runner's EventBus.
            ``None`` -> no telemetry emission (e.g. local-mode trainings
            without runner attached).
        :param marker_writer: callable ``(payload: dict) -> Path | None``
            for custom marker write paths (testing seam). ``None`` ->
            resolve via :func:`atomic_write_text` against
            ``workspace_dir / <name>.marker``.
        :param workspace_dir: directory the trainer writes the marker
            file to. ``None`` -> read from ``HELIX_WORKSPACE`` env var
            (set by :class:`TrainingLauncher._build_job_env`).
        """
        if reason not in ("cancel", "complete"):
            raise ValueError(f"reason must be 'cancel' or 'complete', got {reason!r}")
        self._reason: TerminalReason = reason
        self._handler = shutdown_handler
        self._mlflow_manager = mlflow_manager
        self._flush_timeout = flush_timeout_seconds if flush_timeout_seconds is not None else self.FLUSH_TIMEOUT_SECONDS
        # Idempotency for cancellation logging (only log once per cancel)
        self._signalled = False
        self._event_publisher = event_publisher
        self._marker_writer = marker_writer
        self._workspace_dir = workspace_dir

    # ------------------------------------------------------------------
    # HF TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_step_end(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Pump the shutdown flag into HF ``TrainerControl`` (cancel mode only).

        Completion-reason callbacks treat this as a strict no-op -- they
        do not interact with the cancellation flag, the natural-end
        path runs purely on HF's own loop-exit logic.
        """
        if self._reason != "cancel":
            return

        handler = self._resolve_handler()
        try:
            should_stop = bool(handler.should_stop())  # type: ignore[attr-defined]
        except Exception as exc:
            # An error reading the handler's flag must never crash
            # training. Log once and behave as if no cancellation was
            # requested; the orchestrator-level shutdown path remains
            # the safety-net.
            logger.warning(
                "[CANCELLATION] failed to poll shutdown handler: %s -- continuing without cooperative stop",
                exc,
            )
            return

        if not should_stop:
            return

        # Order matters -- save first so HF checkpoints at the next
        # boundary, then exit AFTER the save lands.
        control.should_save = True
        control.should_training_stop = True

        if not self._signalled:
            self._signalled = True
            logger.info(
                "[CANCELLATION] shutdown flag observed at step=%d -- "
                "set should_save+should_training_stop; trainer will "
                "exit at the next checkpoint boundary",
                getattr(state, "global_step", -1),
            )

    def on_train_end(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Drain MLflow buffer + write marker on terminal exit.

        Cancel reason: only fires when ``self._signalled`` (i.e. a stop
        was observed). Writes ``cancelled.marker`` on flush-budget overrun.

        Complete reason: only fires when the shutdown flag is NOT
        raised (i.e. the cancellation owner is silent on this exit).
        Unconditionally writes ``completion.marker`` so Mac-side
        reconciliation can finalize the upstream MLflow run.
        """
        if self._reason == "cancel":
            if not self._signalled:
                # Clean exit -- cancellation flag was never raised, so
                # the cancellation callback is silent on the happy path.
                return
        else:  # complete
            if self._is_cancellation_active():
                logger.debug(
                    "[COMPLETION] on_train_end: cancellation flag raised -- deferring to cancel-mode callback"
                )
                return

        manager = self._mlflow_manager
        label = self._label()
        if manager is None:
            logger.debug(
                "[%s] on_train_end: no MLflow manager available; skipping flush",
                label,
            )
            return

        # Lazy import keeps this module slim-venv importable.
        from ryotenkai_pod.trainer.callbacks._flush_helper import (
            run_flush_with_deadline,
        )

        outcome = run_flush_with_deadline(
            manager.flush_buffer,  # type: ignore[attr-defined]
            timeout_seconds=self._flush_timeout,
            logger=logger,
            label=label,
        )

        if not outcome.timed_out and not outcome.raised:
            logger.info(
                "[%s] on_train_end: flushed %d buffered MLflow records before HF closes the run",
                label,
                outcome.drained_count,
            )

        marker_path: object | None = None
        if self._reason == "cancel":
            if outcome.timed_out:
                marker_path = self._write_marker(
                    run_id=self._safe_run_id(manager),
                    drained=outcome.drained_count,
                    flush_timed_out=True,
                )
        else:  # complete -- always write the marker
            marker_path = self._write_marker(
                run_id=self._safe_run_id(manager),
                drained=outcome.drained_count,
                flush_timed_out=outcome.timed_out,
            )

        self._emit_finalized_event(
            drained=outcome.drained_count,
            flush_timed_out=outcome.timed_out,
            marker_written=marker_path is not None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label(self) -> str:
        """Log prefix for this callback's emissions."""
        return "CANCELLATION" if self._reason == "cancel" else "COMPLETION"

    def _marker_filename(self) -> str:
        return "cancelled.marker" if self._reason == "cancel" else "completion.marker"

    def _telemetry_kind(self) -> str:
        # Lazy import to avoid pulling observability surface at module load.
        from ryotenkai_shared.observability.cancellation_telemetry import (
            CANCELLATION_FINALIZED,
            COMPLETION_FINALIZED,
        )

        return CANCELLATION_FINALIZED if self._reason == "cancel" else COMPLETION_FINALIZED

    def _resolve_handler(self) -> object:
        """Resolve the global handler on first use (cancel mode only)."""
        if self._handler is None:
            from ryotenkai_pod.trainer.orchestrator.shutdown_handler import (
                get_shutdown_handler,
            )

            self._handler = get_shutdown_handler()
        return self._handler

    def _is_cancellation_active(self) -> bool:
        """Defensive read of the cancellation flag (complete mode only)."""
        handler = self._handler
        if handler is None:
            try:
                from ryotenkai_pod.trainer.orchestrator.shutdown_handler import (
                    get_shutdown_handler,
                )

                handler = get_shutdown_handler()
                self._handler = handler
            except Exception as exc:
                logger.debug(
                    "[COMPLETION] failed to resolve shutdown handler: %s -- assuming not cancelled",
                    exc,
                )
                return False
        try:
            return bool(handler.should_stop())  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug(
                "[COMPLETION] handler.should_stop() raised: %s -- assuming not cancelled",
                exc,
            )
            return False

    def _safe_run_id(self, manager: object) -> str | None:
        """Pull active run_id defensively. ``None`` is acceptable."""
        return getattr(manager, "run_id", None)

    def _emit_finalized_event(
        self,
        *,
        drained: int,
        flush_timed_out: bool,
        marker_written: bool,
    ) -> None:
        """Best-effort telemetry -- never raises into the trainer's exit path."""
        publisher = self._event_publisher
        label = self._label()
        if publisher is None:
            logger.debug(
                "[%s] no event_publisher injected -- skipping finalized telemetry",
                label,
            )
            return
        try:
            kind = self._telemetry_kind()
            publisher(
                kind,
                {
                    "flushed_count": int(drained),
                    "flush_timed_out": bool(flush_timed_out),
                    "marker_written": bool(marker_written),
                    "flush_budget_seconds": float(self._flush_timeout),
                },
            )
        except Exception as exc:
            logger.debug(
                "[%s] failed to emit finalized event: %s",
                label,
                exc,
            )

    def _write_marker(
        self,
        *,
        run_id: str | None,
        drained: int,
        flush_timed_out: bool,
    ) -> object | None:
        """Write the terminal-state marker file to the workspace.

        ``cancel`` -> ``<workspace>/cancelled.marker`` (only on flush
        budget overrun).
        ``complete`` -> ``<workspace>/completion.marker`` (unconditional).

        Best-effort: returns the resulting path on success, ``None``
        when we couldn't determine a workspace or the write failed.
        Never raises -- the trainer is exiting and any failure here
        just means Mac-side reconciliation will skip; not catastrophic.
        """
        if self._marker_writer is not None:
            try:
                import time as _time

                return self._marker_writer(
                    {
                        "run_id": run_id,
                        "flushed_count": drained,
                        "flush_timed_out": flush_timed_out,
                        "ts_ms": _time.time() * 1000,
                    }
                )
            except Exception as exc:
                logger.debug(
                    "[%s] injected marker_writer failed: %s",
                    self._label(),
                    exc,
                )
                return None

        workspace = self._resolve_workspace_dir()
        if workspace is None:
            logger.debug(
                "[%s] no workspace dir resolved -- skipping marker write",
                self._label(),
            )
            return None

        try:
            import json
            import time

            from ryotenkai_shared.utils.atomic_fs import atomic_write_text

            target = Path(workspace) / self._marker_filename()
            if self._reason == "cancel":
                reason_str = "flush_budget_exceeded"
                payload_dict: dict[str, Any] = {
                    "run_id": run_id,
                    "flushed_count": int(drained),
                    "ts_ms": int(time.time() * 1000),
                    "reason": reason_str,
                }
            else:
                reason_str = "flush_budget_exceeded" if flush_timed_out else "natural_completion"
                payload_dict = {
                    "run_id": run_id,
                    "flushed_count": int(drained),
                    "flush_timed_out": bool(flush_timed_out),
                    "ts_ms": int(time.time() * 1000),
                    "reason": reason_str,
                }
            atomic_write_text(target, json.dumps(payload_dict, indent=2))
            level_log = logger.warning if (self._reason == "cancel" or flush_timed_out) else logger.info
            level_log(
                "[%s] wrote %s to %s (reason=%s); Mac-side reconciliation will pick this up",
                self._label(),
                self._marker_filename(),
                target,
                reason_str,
            )
            return target
        except Exception as exc:
            logger.warning(
                "[%s] failed to write %s: %s -- Mac-side reconciliation will skip",
                self._label(),
                self._marker_filename(),
                exc,
            )
            return None

    def _resolve_workspace_dir(self) -> str | Path | None:
        """Resolve workspace path: explicit injection > env var > None."""
        if isinstance(self._workspace_dir, (str, Path)):
            return self._workspace_dir
        import os

        env_value = os.environ.get("HELIX_WORKSPACE")
        if env_value:
            return env_value
        return None

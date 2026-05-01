"""HuggingFace ``TrainerCallback`` for cooperative cancellation.

Closes the gap where SIGTERM caught by :class:`ShutdownHandler` only
sets a thread-local flag — without this callback the HF Trainer never
sees the flag and keeps stepping until the orchestrator-level
``finally`` cleanup. That:

1. Wastes the post-SIGTERM grace window (Supervisor schedules SIGKILL
   after ``--grace`` seconds; trainer needs to checkpoint + exit
   cleanly *before* that).
2. Closes nested MLflow runs as ``FAILED`` instead of ``KILLED``
   because ``run_result.is_failure()`` doesn't distinguish "trainer
   never saw the signal" from "trainer crashed".
3. Loses any metrics buffered by ``ResilientMLflowTransport`` during
   an MLflow flap — those records sit on the pod's disk until the
   next live ``log_metric`` arrives, which after stop never happens.

Activation
----------

Lives in the trainer's callback list when the supervisor runs the
trainer subprocess. The supervisor sets ``RYOTENKAI_RUNNER_URL`` in
the spawn env (see ``TrainingLauncher._build_job_env``); the same
env var is the activation gate for :class:`RunnerEventCallback`.
:func:`TrainerFactory.create` inserts this callback at **index 0**
of the callback list — BEFORE HF Trainer's auto-registered MLflow
callback — so we set ``control.should_save`` and
``control.should_training_stop`` on the same step where HF will
decide whether to save and continue.

Local-mode runs (no runner attached) skip the callback entirely
because the env gate fails.

What it does on each HF hook
----------------------------

``on_step_end``
    Polls :func:`get_shutdown_handler().should_stop()`. On a fresh
    cancellation request:

    1. Sets ``control.should_save = True`` — HF saves a checkpoint
       at the next save boundary.
    2. Sets ``control.should_training_stop = True`` — HF exits the
       train loop after the save lands.
    3. Logs the transition once (idempotent across subsequent step
       calls; the underlying flag stays raised).

    Order matters: setting ``should_training_stop`` without
    ``should_save`` first would have HF skip the save and exit with
    a stale checkpoint.

``on_train_end``
    Phase 9.B — explicit MLflow buffer flush wrapped in a hard 5-second
    deadline via :func:`~src.training._concurrent_helpers.with_timeout`.
    The flush drains any records the resilient transport accumulated
    on disk during MLflow flap into the live MLflow run BEFORE HF
    Trainer's own MLflow callback runs ``end_run()``. Note: we never
    call ``end_run`` ourselves — that's HF's contract; double-close
    raises ``MlflowException``.

    Behaviour matrix:

    * Cancellation flag NOT raised (clean training end) → no-op.
      This callback is silent on the happy path.
    * Cancellation flag raised + flush succeeds within 5s → buffered
      metrics land in MLflow, HF closes the run normally.
    * Cancellation flag raised + flush exceeds 5s budget → the
      :class:`~src.training._concurrent_helpers.TimeoutExceededError`
      is caught and logged. Phase 9.C will write a
      ``cancelled.marker`` file in this branch so Mac-side
      reconciliation can force the upstream MLflow run status to
      ``KILLED``. In 9.B alone the operator just sees a warning in
      the trainer log.
    * MlflowManager unavailable (e.g. tracking disabled) → no-op.

Signal-safety contract
----------------------

The callback never executes inside a signal handler — HF Trainer
calls these hooks from the main train loop. ``ShutdownHandler``'s
signal handler only flips a flag (no I/O); the callback reads the
flag from the regular Python thread, free to call HTTP / MLflow /
whatever from a worker thread (the timeout helper).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import TrainerCallback

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import TrainerControl, TrainerState, TrainingArguments


logger = get_logger(__name__)


__all__ = ["CancellationCallback"]


class CancellationCallback(TrainerCallback):
    """Cooperative-cancellation HF callback driven by ShutdownHandler.

    See module docstring for the full contract.

    Construction is dependency-injected so unit tests can pass a fake
    handler with a deterministic ``should_stop()`` return value
    without registering real signal handlers (which pytest forbids
    in worker threads).
    """

    #: Hard deadline for the on_train_end MLflow flush. MLflow client
    #: has its own internal retry up to ~60s — without an explicit
    #: ceiling here the SIGKILL escalation (Supervisor's ``--grace``,
    #: default 30s) would pre-empt the flush. 5 seconds is enough for
    #: a typical buffered drain (small JSONL → http POST per record)
    #: and leaves headroom for the rest of HF's on_train_end work.
    FLUSH_TIMEOUT_SECONDS: float = 5.0

    def __init__(
        self,
        *,
        shutdown_handler: object | None = None,
        mlflow_manager: object | None = None,
        flush_timeout_seconds: float | None = None,
        event_publisher: Callable[..., Any] | None = None,
        marker_writer: Callable[..., Any] | None = None,
        workspace_dir: object | None = None,
    ) -> None:
        """Build a callback.

        Args:
            shutdown_handler: object exposing ``.should_stop() -> bool``.
                When ``None`` (production default) we reach into
                :func:`src.training.orchestrator.shutdown_handler.get_shutdown_handler`
                lazily on the first poll. Tests pass a stub.
            mlflow_manager: object exposing ``.flush_buffer() -> int``.
                When ``None`` (production default) we resolve the
                configured manager via the global
                :class:`MLflowManagerHolder` lazily on ``on_train_end``.
                Tests inject a stub.
            flush_timeout_seconds: override the class-level
                ``FLUSH_TIMEOUT_SECONDS`` budget — tests use a tiny
                value (e.g. 0.1s) to assert the timeout path
                without waiting in real time.
            event_publisher: callable ``(kind: str, payload: dict) -> None``
                that emits a structured event into the runner's
                EventBus. Phase 9.C — used to emit
                ``cancellation_finalized`` after the flush. Default
                ``None`` = no telemetry emission (e.g. local-mode
                trainings without runner attached).
            marker_writer: callable ``(payload: dict) -> Path | None``
                that writes the cancelled.marker file to the workspace
                when the flush deadline exceeds. Default ``None`` =
                lazy resolution via :func:`atomic_write_text` against
                ``workspace_dir / cancelled.marker``. Tests inject
                a stub returning a sentinel path.
            workspace_dir: directory the trainer writes
                ``cancelled.marker`` to when the flush exceeds budget.
                Default ``None`` = read from ``HELIX_WORKSPACE`` env
                var (set by ``TrainingLauncher._build_job_env``).
                Mac side scans ``attempts/<n>/cancelled.marker`` for
                reconciliation; the trainer's HELIX_WORKSPACE points
                at exactly that attempt directory in production.
        """
        self._handler = shutdown_handler
        self._mlflow_manager = mlflow_manager
        self._flush_timeout = flush_timeout_seconds if flush_timeout_seconds is not None else self.FLUSH_TIMEOUT_SECONDS
        self._signalled = False  # idempotency: we only log once per cancel
        # Phase 9.C plumbing — both nullable; production wires either
        # both or neither. Validated lazily on first use.
        self._event_publisher = event_publisher
        self._marker_writer = marker_writer
        self._workspace_dir = workspace_dir

    def _resolve_handler(self) -> object:
        """Resolve the global handler on first use.

        Lazy because importing the orchestrator package at module
        import time pulls heavy training deps into surfaces that
        only need the callback's class (e.g. unit tests of the
        TrainerFactory wiring layer).
        """
        if self._handler is None:
            from src.training.orchestrator.shutdown_handler import (
                get_shutdown_handler,
            )

            self._handler = get_shutdown_handler()
        return self._handler

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
        """Pump the shutdown flag into HF ``TrainerControl``.

        HF inspects ``control`` after each step to decide whether to
        save / evaluate / stop. We piggy-back on that same loop —
        no extra threads, no polling cost beyond a function call per
        step.
        """
        handler = self._resolve_handler()
        try:
            should_stop = bool(handler.should_stop())  # type: ignore[attr-defined]
        except Exception as exc:
            # An error reading the handler's flag must never crash
            # training. Log once and behave as if no cancellation
            # was requested; the orchestrator-level shutdown path
            # remains the safety-net.
            logger.warning(
                "[CANCELLATION] failed to poll shutdown handler: %s — " "continuing without cooperative stop",
                exc,
            )
            return

        if not should_stop:
            return

        # Order matters — see module docstring.
        # ``should_save = True`` first so HF saves a checkpoint at the
        # next save boundary; ``should_training_stop = True`` after so
        # HF exits AFTER the save (not before).
        control.should_save = True
        control.should_training_stop = True

        if not self._signalled:
            self._signalled = True
            logger.info(
                "[CANCELLATION] shutdown flag observed at step=%d — "
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
        """Phase 9.B — explicit MLflow buffer flush before HF closes the run.

        Silent on the happy path: only fires when cancellation was
        actually observed during the run (``self._signalled``).

        Drain ordering matters: HF Trainer's own MLflow callback runs
        ``end_run()`` on the same hook AFTER ours (we live at index
        0; HF's MLflow callback is auto-registered at the end of the
        list). So the sequence is:

            our.on_train_end → flush_buffer (drains backlog into live run)
            HF.on_train_end  → end_run("KILLED" / "FINISHED")

        We never call ``end_run`` ourselves — see module docstring.
        """
        if not self._signalled:
            # Clean exit (training completed naturally) — no flush
            # needed; the resilient transport drains on the
            # next-live-metric path during steady-state, and clean
            # exits don't accumulate backlog.
            return

        manager = self._resolve_mlflow_manager()
        if manager is None:
            # Tracking disabled or manager not yet initialised — no
            # buffer to flush. This is a normal state for local-mode
            # runs that happened to be wired with the cancellation
            # callback (shouldn't happen in practice — env gate in
            # factory.py prevents it — but defensive.)
            logger.debug(
                "[CANCELLATION] on_train_end: no MLflow manager available; " "skipping flush",
            )
            return

        # Phase 11.A — shared helper handles the deadline + branching
        # so CompletionCallback (natural-completion counterpart) gets
        # bit-identical behaviour for free. Helper never raises;
        # outcome carries ``timed_out`` / ``raised`` / ``drained_count``.
        from src.training.callbacks._flush_helper import (
            run_flush_with_deadline,
        )

        outcome = run_flush_with_deadline(
            manager.flush_buffer,  # type: ignore[attr-defined]
            timeout_seconds=self._flush_timeout,
            logger=logger,
            label="CANCELLATION",
        )

        if not outcome.timed_out and not outcome.raised:
            logger.info(
                "[CANCELLATION] on_train_end: flushed %d buffered MLflow " "records before HF closes the run",
                outcome.drained_count,
            )

        marker_path: object | None = None
        if outcome.timed_out:
            # The flush is still running in the background thread; we
            # just stopped waiting. HF Trainer's MLflow callback will
            # run next and close the run with whatever it knows. Phase
            # 9.C: write ``cancelled.marker`` so Mac-side
            # reconciliation can fix the upstream RunStatus if it
            # was left stuck on RUNNING.
            marker_path = self._write_cancelled_marker(
                run_id=self._safe_run_id(manager),
                drained=outcome.drained_count,
            )

        # Phase 9.C — emit ``cancellation_finalized`` so the Mac-side
        # consumer of the WS event stream knows the trainer's flush
        # is done (or timed out). Carries enough metadata for
        # operator forensics + Mac-side reconciliation:
        #
        # * ``flushed_count`` — buffered records that made it.
        # * ``flush_timed_out`` — primary signal for reconciliation.
        # * ``marker_written`` — bool; True ⇒ Mac will scan the
        #   marker file on the next poll.
        self._emit_finalized_event(
            drained=outcome.drained_count,
            flush_timed_out=outcome.timed_out,
            marker_written=marker_path is not None,
        )

    # ------------------------------------------------------------------
    # Phase 9.C helpers — telemetry event + cancelled.marker
    # ------------------------------------------------------------------

    def _emit_finalized_event(
        self,
        *,
        drained: int,
        flush_timed_out: bool,
        marker_written: bool,
    ) -> None:
        """Best-effort telemetry — never raises into the trainer's
        exit path."""
        publisher = self._event_publisher
        if publisher is None:
            logger.debug(
                "[CANCELLATION] no event_publisher injected — " "skipping cancellation_finalized telemetry",
            )
            return
        try:
            from src.runner.cancellation_telemetry import (
                CANCELLATION_FINALIZED,
            )

            publisher(
                CANCELLATION_FINALIZED,
                {
                    "flushed_count": int(drained),
                    "flush_timed_out": bool(flush_timed_out),
                    "marker_written": bool(marker_written),
                    "flush_budget_seconds": float(self._flush_timeout),
                },
            )
        except Exception as exc:
            logger.debug(
                "[CANCELLATION] failed to emit cancellation_finalized " "event: %s",
                exc,
            )

    def _safe_run_id(self, manager: object) -> str | None:
        """Pull the active run_id from the manager defensively.

        ``MlflowManager.run_id`` is the documented surface; tests may
        inject a stub without it. Defensive ``getattr`` keeps the
        marker path's payload meaningful when available, ``None``
        otherwise.
        """
        return getattr(manager, "run_id", None)

    def _write_cancelled_marker(
        self,
        *,
        run_id: str | None,
        drained: int,
    ) -> object | None:
        """Write ``<workspace>/cancelled.marker`` for Mac-side reconciliation.

        Best-effort: returns the resulting path on success, ``None``
        when we couldn't determine a workspace or the write failed.
        Never raises — the trainer is exiting and any failure here
        just means reconciliation will skip; not catastrophic.

        Defensive resolution order for the workspace dir:
        1. Constructor-injected ``workspace_dir`` (tests).
        2. ``HELIX_WORKSPACE`` env var (production — set by
           ``TrainingLauncher._build_job_env`` to the attempt
           directory's path on the pod's perspective).
        3. ``None`` — log-and-skip.
        """
        # Custom writer was injected — defer to it.
        if self._marker_writer is not None:
            try:
                return self._marker_writer(
                    {  # type: ignore[operator]
                        "run_id": run_id,
                        "flushed_count": drained,
                        "ts_ms": __import__("time").time() * 1000,
                    }
                )
            except Exception as exc:
                logger.debug(
                    "[CANCELLATION] injected marker_writer failed: %s",
                    exc,
                )
                return None

        workspace = self._resolve_workspace_dir()
        if workspace is None:
            logger.debug(
                "[CANCELLATION] no workspace dir resolved — " "skipping cancelled.marker write",
            )
            return None

        try:
            import json
            import time
            from pathlib import Path

            from src.utils.atomic_fs import atomic_write_text

            target = Path(workspace) / "cancelled.marker"
            payload = json.dumps(
                {
                    "run_id": run_id,
                    "flushed_count": int(drained),
                    "ts_ms": int(time.time() * 1000),
                    "reason": "flush_budget_exceeded",
                },
                indent=2,
            )
            atomic_write_text(target, payload)
            logger.warning(
                "[CANCELLATION] wrote cancelled.marker to %s — "
                "Mac-side reconciliation will pick this up to "
                "force-set MLflow run status to KILLED",
                target,
            )
            return target
        except Exception as exc:
            logger.warning(
                "[CANCELLATION] failed to write cancelled.marker: %s — " "Mac-side reconciliation will skip",
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

    def _resolve_mlflow_manager(self) -> object | None:
        """Return the injected MlflowManager (or ``None`` if unavailable).

        Production: ``TrainerFactory`` passes the live manager via
        the constructor — the same instance HF Trainer's MLflow
        callback uses, so a single ``flush_buffer`` call drains the
        backlog into the right run.

        Tests: inject a stub through the constructor.

        No global registry — there isn't one. Pass-through DI keeps
        ownership explicit and makes the callback trivially mockable.
        """
        return self._mlflow_manager

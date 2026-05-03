"""HuggingFace ``TrainerCallback`` for natural-completion finalization.

Counterpart to :class:`CancellationCallback` (Phase 9.B/9.C). Same
I/O shape, same hard deadline budget — but fires on the **happy
path** when training completes naturally (HF Trainer reaches
``max_steps`` / ``num_train_epochs`` without anyone pressing Stop).

Why a separate callback?
------------------------

Phase 9.B's :class:`CancellationCallback.on_train_end` only flushes
when ``self._signalled`` is True (i.e. user pressed Stop). Natural
completion never flips that flag, so any records the
:class:`ResilientMLflowTransport` accumulated during an upstream
MLflow flap stayed on the pod's disk and never reached MLflow.

The pre-Phase-11 architecture had two valid options:

1. Drop the ``_signalled`` gate from CancellationCallback ⇒
   minimal diff, but the class name lies (it's not just
   "Cancellation" anymore).
2. Add a separate CompletionCallback ⇒ +1 callback in the registry,
   honest naming, room for future natural-completion-specific work
   (e.g. HF Hub upload optimization).

We picked option 2 (SRP). The flush logic is shared via
:mod:`src.training.callbacks._flush_helper`, so the Cancellation
and Completion paths stay bit-identical — only the naming, marker
filename, and telemetry kind differ.

Activation
----------

Same env gate as :class:`CancellationCallback` and
:class:`RunnerEventCallback`: registered by
:class:`TrainerFactory` only when ``RYOTENKAI_RUNNER_URL`` is set in
the trainer's spawn env. Local-mode trainings (no in-pod runner)
skip the callback entirely.

Insertion order in the callback list:

* ``[0]`` — :class:`CancellationCallback` (so its on_step_end runs
  BEFORE HF's MLflow callback, see Phase 9 docstring)
* ``[1]`` — :class:`CompletionCallback` (this class). on_train_end
  ordering vs the cancellation callback doesn't matter — the
  callbacks are mutually exclusive (cancellation owns
  ``_signalled=True``, completion owns the inverse).
* ``[…]`` — TrainingEventsCallback / SystemMetricsCallback
* ``[end]`` — :class:`RunnerEventCallback`
* ``[after end]`` — HF Trainer's auto-registered MLflow callback
  (always runs LAST on on_train_end so it ``end_run``'s the live
  MLflow run after our flush has drained the buffer)

Behaviour matrix on ``on_train_end``
------------------------------------

* Cancellation flag raised (``shutdown_handler.should_stop()`` ==
  True) ⇒ NO-OP. CancellationCallback owns the cancel path; running
  flush twice would either be redundant (best case) or double-write
  the marker (worst case).
* Cancellation flag NOT raised + manager unavailable ⇒ NO-OP. No
  buffer to flush. Same defensive default as the cancellation
  callback's clean-exit branch.
* Cancellation flag NOT raised + flush within budget ⇒ write
  ``completion.marker`` with ``reason="natural_completion"``,
  ``flush_timed_out=False``. Mac-side reconciliation reads the
  marker and forces the MLflow run to ``FINISHED`` if it was still
  ``RUNNING`` (Mac was asleep when HF's MLflow callback ran).
* Cancellation flag NOT raised + flush exceeds 5s budget ⇒ write
  ``completion.marker`` with ``reason="flush_budget_exceeded"``,
  ``flush_timed_out=True``. Same Mac-side reconciliation, plus
  operator-visible warning that some metrics may be missing.

Why ``completion.marker`` always-write?
---------------------------------------

Unlike ``cancelled.marker`` (Phase 9.C — written only on flush
timeout), this marker is written on EVERY natural completion.
Reasoning: Mac on post-sleep resume needs to distinguish "trainer
exited naturally and the run is finalised" from "trainer is still
running, just no recent events". A 200-byte marker file is the
cheapest possible signal.

Signal-safety contract
----------------------

Same as CancellationCallback: HF Trainer calls these hooks from the
main train loop, never from a signal handler. The 5s deadline is
enforced by :func:`src.training._concurrent_helpers.with_timeout`
(``concurrent.futures``-based, portable, mock-friendly).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import TrainerCallback

from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers import TrainerControl, TrainerState, TrainingArguments


logger = get_logger(__name__)


__all__ = ["CompletionCallback"]


class CompletionCallback(TrainerCallback):
    """Natural-completion finalization counterpart to CancellationCallback.

    See module docstring for the full contract.
    """

    #: Hard deadline for the on_train_end MLflow flush. Symmetric with
    #: :attr:`CancellationCallback.FLUSH_TIMEOUT_SECONDS` so both paths
    #: behave identically when the upstream MLflow is slow.
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

        Args mirror :class:`CancellationCallback` for symmetry — the
        TrainerFactory wiring passes the same set of dependencies to
        both callbacks.

        Args:
            shutdown_handler: object exposing ``.should_stop() -> bool``.
                When ``None`` (production default) we lazily resolve the
                global handler on first ``on_train_end`` call. Tests
                inject a stub.
            mlflow_manager: object exposing ``.flush_buffer() -> int``.
                When ``None`` we skip the flush silently — local-mode
                trainings without MLflow tracking land here.
            flush_timeout_seconds: override the class-level
                :attr:`FLUSH_TIMEOUT_SECONDS` budget — tests use a tiny
                value to assert the timeout path quickly.
            event_publisher: callable ``(kind, payload) -> None`` for
                emitting ``completion_finalized`` telemetry events.
                ``None`` (production fallback when no runner attached)
                ⇒ skip the event silently.
            marker_writer: callable ``(payload) -> Path | None`` for
                custom marker write paths (testing seam). Default
                ``None`` ⇒ resolve via ``atomic_write_text`` against
                ``workspace_dir / completion.marker``.
            workspace_dir: directory the trainer writes
                ``completion.marker`` to. ``None`` ⇒ read from
                ``HELIX_WORKSPACE`` env (set by TrainingLauncher's
                spawn env).
        """
        self._handler = shutdown_handler
        self._mlflow_manager = mlflow_manager
        self._flush_timeout = flush_timeout_seconds if flush_timeout_seconds is not None else self.FLUSH_TIMEOUT_SECONDS
        self._event_publisher = event_publisher
        self._marker_writer = marker_writer
        self._workspace_dir = workspace_dir

    # ------------------------------------------------------------------
    # HF TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_train_end(  # type: ignore[override]
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Flush MetricsBuffer + write completion.marker on natural end.

        Strict no-op when the cancellation flag is raised — the
        cancellation callback owns that path. Strict no-op when no
        MLflow manager is attached (local-mode runs or tracking
        disabled).
        """
        if self._is_cancellation_active():
            # CancellationCallback.on_train_end will (or already did)
            # handle the flush + cancelled.marker write. Running ours
            # would either duplicate the work (best case) or stomp the
            # marker file (worst case).
            logger.debug(
                "[COMPLETION] on_train_end: cancellation flag raised — " "deferring to CancellationCallback",
            )
            return

        manager = self._mlflow_manager
        if manager is None:
            logger.debug(
                "[COMPLETION] on_train_end: no MLflow manager available; " "skipping flush",
            )
            return

        # Lazy imports keep this module slim-venv-importable.
        from ryotenkai_pod.trainer.callbacks._flush_helper import (
            run_flush_with_deadline,
        )

        outcome = run_flush_with_deadline(
            manager.flush_buffer,  # type: ignore[attr-defined]
            timeout_seconds=self._flush_timeout,
            logger=logger,
            label="COMPLETION",
        )

        if not outcome.timed_out and not outcome.raised:
            logger.info(
                "[COMPLETION] on_train_end: flushed %d buffered MLflow " "records before HF closes the run",
                outcome.drained_count,
            )

        # Always write the marker — Mac on wake reconciliation wants
        # to know "trainer finished naturally" regardless of whether
        # the flush succeeded. ``flush_timed_out`` field in the marker
        # payload tells reconciliation about partial data.
        marker_path = self._write_completion_marker(
            run_id=self._safe_run_id(manager),
            drained=outcome.drained_count,
            flush_timed_out=outcome.timed_out,
        )

        # Emit telemetry — operator dashboards key off this kind.
        self._emit_finalized_event(
            drained=outcome.drained_count,
            flush_timed_out=outcome.timed_out,
            marker_written=marker_path is not None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_cancellation_active(self) -> bool:
        """Defensive: check the global ShutdownHandler for the cancel flag.

        We do NOT cache the handler in the constructor like
        :class:`CancellationCallback` does — Completion callback only
        needs the flag once, on ``on_train_end``. Cheap to resolve
        lazily; avoids the slim-venv import burden of pulling in
        ``src.training.orchestrator`` at construction time when the
        production path doesn't need it (most runs are natural exits).
        """
        handler = self._handler
        if handler is None:
            try:
                from ryotenkai_pod.trainer.orchestrator.shutdown_handler import (
                    get_shutdown_handler,
                )

                handler = get_shutdown_handler()
                self._handler = handler  # cache for symmetry
            except Exception as exc:
                # Resolution failure ⇒ assume not cancelled. Fail-safe:
                # we'd rather double-flush than skip the natural-completion
                # marker.
                logger.debug(
                    "[COMPLETION] failed to resolve shutdown handler: %s " "— assuming not cancelled",
                    exc,
                )
                return False
        try:
            return bool(handler.should_stop())  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug(
                "[COMPLETION] handler.should_stop() raised: %s — " "assuming not cancelled",
                exc,
            )
            return False

    def _safe_run_id(self, manager: object) -> str | None:
        """Pull active run_id defensively. None is acceptable."""
        return getattr(manager, "run_id", None)

    def _emit_finalized_event(
        self,
        *,
        drained: int,
        flush_timed_out: bool,
        marker_written: bool,
    ) -> None:
        """Emit ``completion_finalized`` event. Best-effort; never raises."""
        publisher = self._event_publisher
        if publisher is None:
            logger.debug(
                "[COMPLETION] no event_publisher injected — " "skipping completion_finalized telemetry",
            )
            return
        try:
            from ryotenkai_pod.runner.cancellation_telemetry import (
                COMPLETION_FINALIZED,
            )

            publisher(
                COMPLETION_FINALIZED,
                {
                    "flushed_count": int(drained),
                    "flush_timed_out": bool(flush_timed_out),
                    "marker_written": bool(marker_written),
                    "flush_budget_seconds": float(self._flush_timeout),
                },
            )
        except Exception as exc:
            logger.debug(
                "[COMPLETION] failed to emit completion_finalized " "event: %s",
                exc,
            )

    def _write_completion_marker(
        self,
        *,
        run_id: str | None,
        drained: int,
        flush_timed_out: bool,
    ) -> object | None:
        """Write ``<workspace>/completion.marker`` for Mac-side reconciliation.

        Best-effort: returns the resulting path on success, ``None``
        when we couldn't determine a workspace or the write failed.
        Never raises — the trainer is exiting and any failure here
        just means reconciliation will skip; not catastrophic.

        The marker is written **always** on natural completion (success
        OR timeout). Mac-side reconciliation looks at
        ``flush_timed_out`` to decide whether to log a partial-data
        warning, but the act of writing the marker itself is the
        signal that "training finished naturally".
        """
        # Custom writer was injected — defer to it.
        if self._marker_writer is not None:
            try:
                return self._marker_writer(
                    {  # type: ignore[operator]
                        "run_id": run_id,
                        "flushed_count": drained,
                        "flush_timed_out": flush_timed_out,
                        "ts_ms": __import__("time").time() * 1000,
                    }
                )
            except Exception as exc:
                logger.debug(
                    "[COMPLETION] injected marker_writer failed: %s",
                    exc,
                )
                return None

        workspace = self._resolve_workspace_dir()
        if workspace is None:
            logger.debug(
                "[COMPLETION] no workspace dir resolved — " "skipping completion.marker write",
            )
            return None

        try:
            import json
            import time
            from pathlib import Path

            from ryotenkai_shared.utils.atomic_fs import atomic_write_text

            target = Path(workspace) / "completion.marker"
            reason = "flush_budget_exceeded" if flush_timed_out else "natural_completion"
            payload = json.dumps(
                {
                    "run_id": run_id,
                    "flushed_count": int(drained),
                    "flush_timed_out": bool(flush_timed_out),
                    "ts_ms": int(time.time() * 1000),
                    "reason": reason,
                },
                indent=2,
            )
            atomic_write_text(target, payload)
            level_log = logger.info if not flush_timed_out else logger.warning
            level_log(
                "[COMPLETION] wrote completion.marker to %s (reason=%s); "
                "Mac-side reconciliation will pick this up to finalize "
                "the upstream MLflow run if it's still RUNNING",
                target,
                reason,
            )
            return target
        except Exception as exc:
            logger.warning(
                "[COMPLETION] failed to write completion.marker: %s — " "Mac-side reconciliation will skip",
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

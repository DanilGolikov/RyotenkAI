"""Shared flush-with-deadline helper for the unified terminal callback.

The :class:`TerminalCallback` (reason="cancel" / "complete") calls
``mlflow_manager.flush_buffer()`` inside a hard 5-second deadline on
``on_train_end``. The deadline has the same shape on both reasons:

* Cancellation path — trainer about to exit because user pressed Stop.
  We want to drain the buffered MLflow records into the live run BEFORE
  HF's MLflow callback runs ``end_run("KILLED")``. If the flush takes
  too long, we bail and let HF close the run with whatever it knows;
  the missing records get reconciled later via ``cancelled.marker``.
* Completion path — trainer exited naturally (reached max_steps /
  num_train_epochs). The same drain happens, but reason is
  ``"natural_completion"``. If the flush times out, we still write
  ``completion.marker`` so Mac-side reconciliation sees the final state.

Extracting the shared shape into a helper avoids duplicating:

1. The ``with_timeout`` budget plumbing.
2. The success / timeout / exception branching.
3. The ``(drained_count, timed_out)`` tuple contract that both
   callbacks pass into their telemetry events.

Why not just call :func:`with_timeout` directly?
------------------------------------------------

We could — and the cancellation callback did, before this helper
landed. But each call site grew the same try/except around the same
result tuple. Two copies of branching logic are one too many; if a
future sub-phase adds a third terminal callback (PauseCallback,
maybe), it gets the same behaviour for free.

Keeping the helper this narrow (one function, two args, deterministic
return) is intentional — it's the SRP-clean home for the flush-budget
contract, not a general-purpose toolkit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging


__all__ = ["FlushOutcome", "run_flush_with_deadline"]


@dataclass(frozen=True)
class FlushOutcome:
    """Result of a deadline-bounded ``flush_buffer`` call.

    Both fields drive downstream behaviour:

    * ``drained_count`` — how many buffered MLflow records made it
      to the live run before we either succeeded or bailed. Reported
      in the ``cancellation_finalized`` / ``completion_finalized``
      telemetry events for operator dashboards.
    * ``timed_out`` — True iff the helper hit the deadline. Drives
      whether the marker file is written with
      ``reason="flush_budget_exceeded"`` (timed_out) or
      ``"natural_completion"`` / ``"cancellation"`` (success).
    * ``raised`` — True iff ``flush_fn`` raised an unexpected
      exception. Distinguishes "manager bug" from "MLflow upstream
      slow" for forensics. The exception itself is logged but not
      re-raised — terminal-path callbacks never bubble flush errors
      up to the trainer's exit path.
    """
    drained_count: int
    timed_out: bool
    raised: bool


def run_flush_with_deadline(
    flush_fn: Callable[[], int],
    *,
    timeout_seconds: float,
    logger: "logging.Logger | None" = None,
    label: str = "flush",
) -> FlushOutcome:
    """Run ``flush_fn`` with a hard deadline; never raise.

    The function is intentionally swallow-everything: by the time
    we're calling it, the trainer is already on its way out — a
    propagating exception here just delays HF's ``end_run`` and
    risks orphaning the MLflow run state.

    Args:
        flush_fn: Zero-arg callable returning ``int`` (count of
            drained records). In production this is bound to
            ``MlflowManager.flush_buffer``.
        timeout_seconds: Hard ceiling. Must be > 0; same constraint
            as :func:`src.training._concurrent_helpers.with_timeout`.
        logger: Optional logger for warning-level messages on
            timeout / exception. ``None`` ⇒ silent (caller is
            expected to publish telemetry separately).
        label: Short string included in warning messages — e.g.
            ``"completion_flush"`` / ``"cancellation_flush"``.
            Helps operator grep when both callbacks fire on the same
            run (rare but possible during a cancel/complete race).

    Returns:
        :class:`FlushOutcome` capturing what happened. The caller
        decides what to do with the tuple — typically: emit
        telemetry, write marker, log.
    """
    # Lazy import keeps this module slim-venv-importable even when
    # the heavy ``src.training`` package init is stubbed (the
    # cancellation callback's slim-venv test pattern).
    from ryotenkai_pod.trainer._concurrent_helpers import (
        TimeoutExceededError,
        with_timeout,
    )

    drained = 0
    timed_out = False
    raised = False

    try:
        drained = with_timeout(flush_fn, timeout_seconds=timeout_seconds)
    except TimeoutExceededError:
        timed_out = True
        if logger is not None:
            logger.warning(
                "[%s] MLflow flush exceeded %.1fs budget; proceeding "
                "to HF end_run regardless. Some buffered metrics may "
                "not have made it to the upstream run before exit.",
                label, timeout_seconds,
            )
    except Exception as exc:  # noqa: BLE001 — best-effort by contract
        raised = True
        if logger is not None:
            logger.warning(
                "[%s] flush_buffer raised unexpectedly: %s — "
                "proceeding to HF end_run", label, exc,
            )

    return FlushOutcome(
        drained_count=int(drained),
        timed_out=timed_out,
        raised=raised,
    )

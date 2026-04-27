"""Phase 9.C — observability constants for the cancellation chain.

A single source of truth for the event kinds the cancellation flow
emits. Spread across three processes:

* In-pod runner (Supervisor): ``cancellation_started`` /
  ``cancellation_completed`` from the FSM transitions.
* Trainer subprocess (CancellationCallback): ``cancellation_finalized``
  after the on_train_end MLflow flush.
* Mac control plane (TrainingMonitor): ``mlflow_reconciled_post_sigkill``
  /  ``cleanup_pod_failed`` from reconciliation + cleanup.

Defining the kind strings here keeps them grep-able across the
codebase. Operator dashboards / SLO alerts key off these strings —
renaming requires bumping every consumer at once, which makes the
visibility intentional.

Latency contract
----------------

Every event in the chain carries ``latency_ms`` measured against a
common anchor: the ``cancellation_started`` event's
``requested_at_ms`` field. Specifically:

* ``cancellation_started.requested_at_ms`` — anchor (epoch ms).
* ``cancellation_finalized.latency_ms`` — flush completion to
  anchor.
* ``cancellation_completed.total_latency_ms`` — FSM-terminal to
  anchor.

The clock is :func:`time.time` (epoch ms) rather than
:func:`time.monotonic` because we want timestamps to round-trip
through JSON / Mac log scrapes / MLflow run metadata. Monotonic
clocks aren't comparable across process boundaries; epoch is.
"""

from __future__ import annotations

import time

__all__ = [
    "CANCELLATION_REQUESTED",
    "CANCELLATION_STARTED",
    "CANCELLATION_FINALIZED",
    "CANCELLATION_COMPLETED",
    "MLFLOW_RECONCILED_POST_SIGKILL",
    "CLEANUP_POD_FAILED",
    "CANCELLATION_EVENT_KINDS",
    "now_ms",
    "latency_ms_since",
]


# ---------------------------------------------------------------------------
# Event kind constants
# ---------------------------------------------------------------------------

#: Operator pressed Stop (CLI / Web UI). Optional — Phase 9.C reserves
#: the kind for a future Mac-side event published the moment the
#: control plane receives the request, before the runner round-trip.
#: Not emitted in 9.C (placeholder so consumers know the schema).
CANCELLATION_REQUESTED = "cancellation_requested"

#: Supervisor caught the stop request and transitioned FSM to
#: ``stopping``. Carries:
#:   * ``requested_at_ms`` — anchor for downstream latency.
#:   * ``grace_seconds`` — supervisor's SIGKILL escalation budget.
#:   * ``reason`` — short identifier (``"sigterm"``, ``"idle_timeout"``,
#:     ``"max_lifetime"`` ...).
CANCELLATION_STARTED = "cancellation_started"

#: Trainer's :class:`CancellationCallback.on_train_end` finished its
#: MLflow flush (success OR timed out — distinguished by
#: ``flush_timed_out`` in payload). Carries:
#:   * ``latency_ms`` — wall time from cancellation_started.
#:   * ``flushed_count`` — records drained from the resilient buffer.
#:   * ``flush_timed_out`` — bool; True triggers Mac reconciliation.
#:   * ``marker_written`` — bool; True if cancelled.marker was
#:     written for Mac-side reconciliation pickup.
CANCELLATION_FINALIZED = "cancellation_finalized"

#: Supervisor reaped the trainer subprocess; FSM landed in terminal
#: state. Carries:
#:   * ``total_latency_ms`` — wall time from cancellation_started to
#:     reap.
#:   * ``terminal_state`` — ``"cancelled"`` / ``"failed"`` / ``"completed"``.
#:   * ``exit_code``, ``signal`` — for forensics.
CANCELLATION_COMPLETED = "cancellation_completed"

#: Mac-side reconciliation forced an MLflow run from ``RUNNING`` →
#: ``KILLED`` because the trainer SIGKILLed before ``end_run`` ran.
#: Carries:
#:   * ``run_id`` — MLflow run that was reconciled.
#:   * ``marker_path`` — where the cancelled.marker lived.
MLFLOW_RECONCILED_POST_SIGKILL = "mlflow_reconciled_post_sigkill"

#: Mac-side ``provider.cleanup_pod`` failed (RunPod GraphQL down,
#: SSH unreachable for single_node, etc.). Operator-visible signal
#: that pod billing may continue accruing — manual cleanup needed.
#: Carries:
#:   * ``provider`` — ``"runpod"`` / ``"single_node"``.
#:   * ``pod_id`` — when known.
#:   * ``error_code`` / ``message`` — structured for filtering.
CLEANUP_POD_FAILED = "cleanup_pod_failed"


#: Convenience set for grep / filter / dashboard wiring.
CANCELLATION_EVENT_KINDS: frozenset[str] = frozenset({
    CANCELLATION_REQUESTED,
    CANCELLATION_STARTED,
    CANCELLATION_FINALIZED,
    CANCELLATION_COMPLETED,
    MLFLOW_RECONCILED_POST_SIGKILL,
    CLEANUP_POD_FAILED,
})


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def now_ms() -> int:
    """Current epoch time in milliseconds.

    Used as the anchor (``requested_at_ms``) for the cancellation
    chain. Epoch (not monotonic) so payloads round-trip through JSON
    + Mac log scrapes + MLflow metadata across process boundaries.
    """
    return int(time.time() * 1000)


def latency_ms_since(start_ms: int) -> int:
    """Milliseconds elapsed since ``start_ms`` (epoch ms).

    Returns ``0`` when ``start_ms`` is in the future (clock skew on
    distributed deployments) so dashboards never see negative
    latencies — a future-anchor implies events arrived out of order
    and the operator should treat the latency as 'unknown', not a
    suspiciously fast value.
    """
    delta = now_ms() - start_ms
    return delta if delta > 0 else 0

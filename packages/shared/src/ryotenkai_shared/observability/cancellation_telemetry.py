"""Phase 9.C — observability constants for the cancellation chain.

A single source of truth for the event kinds the cancellation flow
emits. Spread across three processes:

* In-pod runner (Supervisor): ``cancellation_started`` /
  ``cancellation_completed`` from the FSM transitions.
* Trainer subprocess (TerminalCallback, reason="cancel"):
  ``cancellation_finalized`` after the on_train_end MLflow flush.
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
    "COMPLETION_FINALIZED",
    "MLFLOW_RECONCILED_POST_SIGKILL",
    "CLEANUP_POD_FAILED",
    # Phase 12.C — durability telemetry
    "EVENTS_DISK_PRESSURE",
    "EVENTS_ROTATED",
    "EVENTS_GC_RAN",
    "METRICS_BUFFER_RETRIEVED",
    "CANCELLATION_EVENT_KINDS",
    "DURABILITY_EVENT_KINDS",
    "TERMINAL_EVENT_KINDS",
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

#: Trainer's :class:`TerminalCallback` (reason="cancel") finished its
#: ``on_train_end`` MLflow flush (success OR timed out -- distinguished by
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

#: Trainer's :class:`TerminalCallback` (reason="complete") finished its
#: ``on_train_end`` MLflow flush on the **natural-completion** path --
#: training reached ``max_steps`` / ``num_train_epochs`` without
#: anyone pressing Stop. Counterpart to
#: :data:`CANCELLATION_FINALIZED`. Carries:
#:   * ``flushed_count`` — records drained from the resilient buffer.
#:   * ``flush_timed_out`` — bool; True ⇒ partial flush, marker
#:     payload's ``reason`` is ``"flush_budget_exceeded"``.
#:   * ``marker_written`` — bool; ``completion.marker`` is written
#:     **always** on natural end (per Phase 11 design), so this
#:     should be True except on disk-write failure.
#:   * ``flush_budget_seconds`` — the deadline used (debug aid).
COMPLETION_FINALIZED = "completion_finalized"

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


#: Phase 12.C — :class:`EventBus` failed to write a record to its
#: :class:`EventJournal`. Rate-limited 1/min per bus instance to
#: avoid feedback loops where the disk-pressure event itself
#: floods the journal. Carries:
#:   * ``error_code`` — usually ``"OSError"`` or ``"IOError"``.
#:   * ``total_bytes`` — current journal footprint at signalling time.
#:   * ``file_count`` — how many JSONL files exist.
EVENTS_DISK_PRESSURE = "events_disk_pressure"

#: Phase 12.C — :class:`EventJournal` advanced from
#: ``events.NNN.jsonl`` → ``events.NNN+1.jsonl`` because the cap was
#: hit. Carries:
#:   * ``from_seq`` — file just closed.
#:   * ``to_seq`` — new active file.
#:   * ``file_size_bytes`` — bytes written to the just-closed file.
#:   * ``oldest_remaining_seq`` — smallest seq still on disk after
#:     drop-oldest enforcement.
EVENTS_ROTATED = "events_rotated"

#: Phase 12.C — :class:`EventJournal` deleted at least one file
#: during init-time crash recovery (interrupted ``.tmp`` file or
#: stale rotation residue). Distinct from the per-rotate drop-oldest
#: which is implicit in :data:`EVENTS_ROTATED`. Carries:
#:   * ``deleted_seqs`` — list of seq numbers deleted.
#:   * ``deleted_bytes`` — cumulative size of removed files.
EVENTS_GC_RAN = "events_gc_ran"

#: Phase 12.C — :class:`ModelRetriever` finished its
#: ``metrics_buffer.jsonl`` retrieval + replay attempt (Phase 12.A.1).
#: Originally mirrored the legacy
#: ``ModelRetrieverEventCallbacks.on_metrics_buffer_retrieved`` callback
#: (removed in Phase 4 event-system unification); the bus kind name is
#: preserved so existing subscribers keep working. Carries:
#:   * ``replayed`` — int, count of metrics shipped to MLflow.
#:   * ``line_count`` — int, lines in the retrieved buffer file.
#:   * ``size_bytes`` — int, remote file size.
#:   * ``missing`` — bool, True ⇒ trainer drain succeeded; healthy.
#:   * ``oversized`` — bool, True ⇒ buffer over the 100 MiB safety cap.
METRICS_BUFFER_RETRIEVED = "metrics_buffer_retrieved"


#: Convenience set for grep / filter / dashboard wiring (cancellation
#: chain only).
CANCELLATION_EVENT_KINDS: frozenset[str] = frozenset({
    CANCELLATION_REQUESTED,
    CANCELLATION_STARTED,
    CANCELLATION_FINALIZED,
    CANCELLATION_COMPLETED,
    MLFLOW_RECONCILED_POST_SIGKILL,
    CLEANUP_POD_FAILED,
})

#: Phase 12.C — durability-related event kinds (journal +
#: metrics-buffer retrieval). Use this set when wiring "data
#: durability" dashboards or alerts. Disjoint from
#: :data:`CANCELLATION_EVENT_KINDS`.
DURABILITY_EVENT_KINDS: frozenset[str] = frozenset({
    EVENTS_DISK_PRESSURE,
    EVENTS_ROTATED,
    EVENTS_GC_RAN,
    METRICS_BUFFER_RETRIEVED,
})

#: Superset including the natural-completion path AND durability
#: events. Operator dashboards keying off "any terminal / durability
#: signal" should use this set; cancellation-only views (e.g. SLO
#: alerts on cancel latency) stay on :data:`CANCELLATION_EVENT_KINDS`.
TERMINAL_EVENT_KINDS: frozenset[str] = CANCELLATION_EVENT_KINDS | frozenset({
    COMPLETION_FINALIZED,
}) | DURABILITY_EVENT_KINDS


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

"""Cross-cutting observability constants.

Lived under ``ryotenkai_pod.runner.cancellation_telemetry`` until ADR
row 6 — both the runner-side ``Supervisor`` and the trainer-side
callbacks (cancellation, completion) emit events with these exact
kind strings. Hosting them in shared lets every party agree without
crossing the ``pod.trainer ↔ pod.runner`` boundary.
"""

from __future__ import annotations

from ryotenkai_shared.observability.cancellation_telemetry import (
    CANCELLATION_COMPLETED,
    CANCELLATION_FINALIZED,
    CANCELLATION_REQUESTED,
    CANCELLATION_STARTED,
    CLEANUP_POD_FAILED,
    COMPLETION_FINALIZED,
    EVENTS_DISK_PRESSURE,
    EVENTS_GC_RAN,
    EVENTS_ROTATED,
    METRICS_BUFFER_RETRIEVED,
    MLFLOW_RECONCILED_POST_SIGKILL,
    latency_ms_since,
    now_ms,
)

__all__ = [
    "CANCELLATION_COMPLETED",
    "CANCELLATION_FINALIZED",
    "CANCELLATION_REQUESTED",
    "CANCELLATION_STARTED",
    "CLEANUP_POD_FAILED",
    "COMPLETION_FINALIZED",
    "EVENTS_DISK_PRESSURE",
    "EVENTS_GC_RAN",
    "EVENTS_ROTATED",
    "METRICS_BUFFER_RETRIEVED",
    "MLFLOW_RECONCILED_POST_SIGKILL",
    "latency_ms_since",
    "now_ms",
]

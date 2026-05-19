"""
Phase 12.A — replay buffered MLflow metrics from a retrieved
``metrics_buffer.jsonl`` into the same MLflow run that was active
during training.

Background
----------
The trainer's :class:`~src.training.mlflow.resilient_transport.ResilientMLflowTransport`
buffers every ``mlflow.log_metric`` call to
``<workspace>/metrics_buffer.jsonl`` while the circuit breaker is
open (MLflow upstream unreachable — typically because the Mac is
asleep). The unified :class:`TerminalCallback` (reason="complete") drains
the buffer at end-of-training via its ``on_train_end`` hook, but only
succeeds when
MLflow upstream is reachable at that exact moment. If the Mac is
still asleep when training ends, the drain fails (no MLflow
connectivity) and the buffer file stays on disk. Without Phase 12
the file is then erased when the pod is terminated, and the metrics
are lost forever.

Phase 12.A.1 closes the gap by retrieving the buffer file during
:class:`~src.pipeline.stages.model_retriever.retriever.ModelRetriever`
(after the Mac wakes + pod resumes via Phase 11.C), and replaying
each record into the appropriate MLflow run via
:class:`mlflow.tracking.MlflowClient.log_metric`.

Idempotency by invariant
------------------------
The buffer file ONLY contains records that have NEVER been flushed —
:meth:`~src.training.mlflow.metrics_buffer.MetricsBuffer.flush`
removes drained entries before returning. So replay is "ship every
record exactly once" with no dedup logic needed: if a record is in
the file, MLflow has not seen it yet.

Edge cases
----------
* Empty / missing file → :class:`ReplayResult` with ``replayed=0``.
* Malformed JSONL line → skipped, error captured (capped at 10
  entries).
* MLflow client raises mid-stream → continue with remaining records,
  collect first 10 errors, return ``ReplayResult`` with
  ``failed > 0``.
* Buffer file left on disk in the local attempt directory after
  replay — the file is the canonical record of "what we tried to
  replay" and is preserved for forensics. Caller (ModelRetriever) is
  responsible for moving / archiving it as appropriate.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ryotenkai_shared.utils.logger import get_logger

logger = get_logger(__name__)

# Phase 12.A.1 — number of error messages to retain on failure path.
# Capped to avoid unbounded memory growth on a corrupt file with
# millions of bad lines.
_MAX_CAPTURED_ERRORS = 10


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of a buffer replay run.

    Attributes:
        replayed:     Number of entries successfully shipped to MLflow.
        failed:       Number of entries that raised on log_metric
                      (transport error, terminal run, etc.).
        skipped:      Number of malformed JSONL lines skipped on read.
        first_step:   Lowest ``step`` value encountered across replayed
                      entries; ``-1`` when ``replayed=0``.
        last_step:    Highest ``step`` value encountered across replayed
                      entries; ``-1`` when ``replayed=0``.
        duration_ms:  Wall-clock duration of the replay (read + log).
        errors:       First :data:`_MAX_CAPTURED_ERRORS` error messages
                      for forensics. Truncated silently after that.
    """

    replayed: int
    failed: int = 0
    skipped: int = 0
    first_step: int = -1
    last_step: int = -1
    duration_ms: int = 0
    errors: tuple[str, ...] = field(default_factory=tuple)


class _MlflowClientProtocol(Protocol):
    """Subset of :class:`mlflow.tracking.MlflowClient` that
    :class:`BufferedMetricsReplay` actually uses.

    Stated explicitly so the unit tests can pass a tiny stand-in
    without dragging the full ``mlflow`` package in (slim-venv
    pattern; same as ``test_resilient_transport_flush.py``).
    """

    def log_metric(  # noqa: D401  (Protocol method, not docstring)
        self,
        run_id: str,
        key: str,
        value: float,
        timestamp: int | None = ...,
        step: int | None = ...,
    ) -> Any: ...


class BufferedMetricsReplay:
    """Replay a JSONL metrics buffer into an MLflow run.

    Stateless aside from the injected client; safe to instantiate
    once per retrieval and call :meth:`replay` once.

    The class is deliberately decoupled from
    :class:`~src.training.mlflow.metrics_buffer.MetricsBuffer` —
    that module lives in the trainer side of the codebase
    (potentially on the pod), this one on the Mac. We share only
    the **on-disk format**: a JSONL file where each line is

        ``{"key": str, "value": float, "step": int, "timestamp": float}``

    Anything else (versioning, future ``run_id`` per record) is
    additive — readers ignore unknown keys and missing fields fall
    back to safe defaults.
    """

    def __init__(self, mlflow_client: _MlflowClientProtocol) -> None:
        self._client = mlflow_client

    def replay(self, *, buffer_path: Path, run_id: str) -> ReplayResult:
        """Read every record from ``buffer_path`` and replay into ``run_id``.

        Args:
            buffer_path: Path to a local copy of ``metrics_buffer.jsonl``
                          (already retrieved from the pod via SCP).
            run_id:      MLflow run ID to receive the metrics. Per the
                          plan, this is the parent / root run for the
                          attempt — multi-phase nested metrics still
                          land in the parent run with their original
                          ``key`` (e.g. ``"sapo/loss"`` ``"grpo/reward"``)
                          which makes them findable in the MLflow UI.

        Returns:
            :class:`ReplayResult` describing the outcome. Never raises:
            replay is best-effort and any error is captured in
            ``errors``.
        """
        started_at = time.monotonic()

        if not buffer_path.exists():
            logger.info(
                "[METRICS_REPLAY] buffer file not found: %s — nothing to replay",
                buffer_path,
            )
            return ReplayResult(replayed=0, duration_ms=0)

        # ------------------------------------------------------------------
        # 1. Read all entries; skip malformed lines without raising.
        # ------------------------------------------------------------------
        entries: list[dict[str, Any]] = []
        skipped = 0
        errors: list[str] = []

        try:
            with buffer_path.open(encoding="utf-8") as fh:
                for line_no, raw in enumerate(fh, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        skipped += 1
                        if len(errors) < _MAX_CAPTURED_ERRORS:
                            errors.append(
                                f"line {line_no}: malformed JSON ({exc})"
                            )
        except OSError as exc:
            errors.append(f"read failed: {exc}")
            return ReplayResult(
                replayed=0,
                skipped=skipped,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                errors=tuple(errors),
            )

        if not entries:
            logger.info(
                "[METRICS_REPLAY] buffer empty (skipped=%d) — nothing to replay",
                skipped,
            )
            return ReplayResult(
                replayed=0,
                skipped=skipped,
                duration_ms=int((time.monotonic() - started_at) * 1000),
                errors=tuple(errors),
            )

        # ------------------------------------------------------------------
        # 2. Sort by (step, timestamp) — same invariant as
        #    :meth:`MetricsBuffer.flush`. Guarantees monotonic step
        #    ordering in the MLflow UI even after the buffer has been
        #    appended to out-of-order across multiple breaker open/close
        #    cycles.
        # ------------------------------------------------------------------
        entries.sort(key=lambda e: (
            int(e.get("step", 0) or 0),
            float(e.get("timestamp", 0.0) or 0.0),
        ))

        # ------------------------------------------------------------------
        # 3. Replay one-by-one. MlflowClient.log_metric raises on
        #    transport failure; we capture and continue (best-effort).
        # ------------------------------------------------------------------
        replayed = 0
        failed = 0
        first_step = -1
        last_step = -1

        for entry in entries:
            try:
                key = entry["key"]
                value = float(entry["value"])
                step = int(entry.get("step", 0) or 0)
                ts_seconds = float(entry.get("timestamp", 0.0) or 0.0)
            except (KeyError, TypeError, ValueError) as exc:
                failed += 1
                if len(errors) < _MAX_CAPTURED_ERRORS:
                    errors.append(f"malformed entry {entry!r}: {exc}")
                continue

            ts_ms = int(ts_seconds * 1000) if ts_seconds > 0 else None

            try:
                self._client.log_metric(
                    run_id=run_id,
                    key=str(key),
                    value=value,
                    step=step,
                    timestamp=ts_ms,
                )
            except Exception as exc:  # noqa: BLE001 — best-effort by contract
                failed += 1
                if len(errors) < _MAX_CAPTURED_ERRORS:
                    errors.append(
                        f"log_metric({key}@{step}) failed: {type(exc).__name__}: {exc}"
                    )
                continue

            replayed += 1
            if first_step == -1 or step < first_step:
                first_step = step
            if step > last_step:
                last_step = step

        duration_ms = int((time.monotonic() - started_at) * 1000)

        if replayed > 0:
            logger.info(
                "[METRICS_REPLAY] replayed %d/%d metrics (run=%s, steps %d→%d, %dms)",
                replayed, len(entries), run_id, first_step, last_step, duration_ms,
            )
        if failed > 0:
            logger.warning(
                "[METRICS_REPLAY] %d/%d metric writes failed; first error: %s",
                failed, len(entries), errors[0] if errors else "(none captured)",
            )

        return ReplayResult(
            replayed=replayed,
            failed=failed,
            skipped=skipped,
            first_step=first_step,
            last_step=last_step,
            duration_ms=duration_ms,
            errors=tuple(errors),
        )


__all__ = ["BufferedMetricsReplay", "ReplayResult"]

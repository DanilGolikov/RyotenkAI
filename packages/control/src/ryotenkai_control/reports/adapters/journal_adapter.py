"""JournalReportAdapter — derive report event data from the typed journal.

Phase 7 replaces the legacy ``training_events.json`` adapter path (which
sniffed strings out of an MLflow artifact) with a typed reader for the
unified event journal written by
:class:`ryotenkai_control.events.journal_writer.JournalWriter` and
uploaded to MLflow under ``events/events.jsonl`` by
:class:`ryotenkai_control.events.mlflow_finalizer.MlflowFinalizer`.

Resolution order
----------------

1. ``workspace_dir`` (optional explicit path): if present, look for
   ``<workspace_dir>/events.jsonl``.
2. Workspace lookup is the local copy written by the live run; it is
   always preferred because it doesn't require an MLflow round-trip.
3. MLflow fallback: download the ``events/events.jsonl`` artifact via
   :class:`~ryotenkai_shared.infrastructure.mlflow.protocol.IMLflowManager`
   when the workspace copy is missing — runs finalized in a previous
   process may only have the uploaded copy left on the server.
4. Both missing → return :class:`ExperimentEventData` with empty
   fields and ``source="empty"``.

Per-event projection
--------------------

* ``StageStartedEvent``  → seeds a :class:`StageOutcome` (status "running").
* ``StageCompletedEvent`` → finalises the stage outcome (status
  "completed", ``duration_s``, ``completed_at``) and emits a single
  ``TimelineEntry``.
* ``StageFailedEvent``  → status "failed", carries the error message,
  ``severity="ERROR"`` timeline entry.
* ``StageSkippedEvent`` → status "skipped", info-severity timeline entry.
* ``StageInterruptedEvent`` → status "interrupted", warning-severity
  timeline entry.
* ``MemoryCacheClearedEvent`` → ``MemoryEvent(event_type="cache_clear")``
  with ``freed_mb`` derived from ``before_bytes - after_bytes`` (clamped
  ≥ 0).
* ``MemoryOOMDetectedEvent`` → ``MemoryEvent(event_type="oom")``.
* ``MemoryPressureWarningEvent`` → ``MemoryEvent(event_type="warning")``
  with ``utilization_percent``.
* :class:`~ryotenkai_shared.events.types.unknown.UnknownEvent` → skipped
  (forward-compat). Counter exposed via
  :attr:`ExperimentEventData.unknown_envelopes`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from ryotenkai_control.events.journal_reader import JournalReader
from ryotenkai_control.reports.domain.entities import MemoryEvent
from ryotenkai_control.reports.domain.event_data import (
    ExperimentEventData,
    StageOutcome,
    TimelineEntry,
)
from ryotenkai_shared.events import (
    BaseEvent,
    UnknownEvent,
)
from ryotenkai_shared.events.types.control_stage import (
    StageCompletedEvent,
    StageFailedEvent,
    StageInterruptedEvent,
    StageSkippedEvent,
    StageStartedEvent,
)
from ryotenkai_shared.events.types.pod_memory import (
    MemoryCacheClearedEvent,
    MemoryOOMDetectedEvent,
    MemoryPressureWarningEvent,
    MemoryThresholdReachedEvent,
)
from ryotenkai_shared.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ryotenkai_shared.infrastructure.mlflow.protocol import IMLflowManager


logger = get_logger(__name__)


# Default artifact path under the run; matches MlflowFinalizer's
# DEFAULT_ARTIFACT_PATH constant — kept inlined here to avoid importing
# the finalizer module just for a string constant.
MLFLOW_EVENTS_ARTIFACT_DIR = "events"
JOURNAL_FILENAME = "events.jsonl"

# Number of bytes per MB used to convert MemoryCacheCleared deltas
# (event payload is in bytes) to MemoryEvent.freed_mb. MiB (1024**2)
# matches the rest of the memory-manager surface.
_BYTES_PER_MB = 1024 * 1024


class JournalReportAdapter:
    """Reads typed events from a run's journal (workspace or MLflow).

    Workspace-first; MLflow artifact fallback. Errors are logged and
    coerced to an empty :class:`ExperimentEventData` so report rendering
    can still proceed (the typed journal is best-effort observability,
    not a hard run dependency).
    """

    def __init__(
        self,
        mlflow_manager: IMLflowManager | None = None,
        *,
        artifact_dir: str = MLFLOW_EVENTS_ARTIFACT_DIR,
        journal_filename: str = JOURNAL_FILENAME,
    ) -> None:
        self._mlflow = mlflow_manager
        self._artifact_dir = artifact_dir
        self._journal_filename = journal_filename

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        run_id: str,
        *,
        workspace_dir: Path | None = None,
    ) -> ExperimentEventData:
        """Return event-derived data for ``run_id``.

        Tries the workspace copy first, then the MLflow artifact, then
        returns an empty container.
        """
        # 1. Workspace lookup
        if workspace_dir is not None:
            journal_path = workspace_dir / self._journal_filename
            if journal_path.exists():
                return self._read_from_journal(journal_path, run_id=run_id, source="workspace")
            logger.debug(
                "[JournalAdapter] workspace journal absent: %s",
                journal_path,
            )

        # 2. MLflow fallback
        if self._mlflow is not None:
            downloaded = self._download_from_mlflow(run_id)
            if downloaded is not None:
                return self._read_from_journal(downloaded, run_id=run_id, source="mlflow")

        # 3. Empty
        logger.info(
            "[JournalAdapter] no journal available for run %s — returning empty data",
            run_id[:8] if run_id else "<unknown>",
        )
        return ExperimentEventData(run_id=run_id, source="empty")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_from_journal(
        self,
        journal_path: Path,
        *,
        run_id: str,
        source: str,
    ) -> ExperimentEventData:
        """Parse the journal at ``journal_path`` into ``ExperimentEventData``."""
        try:
            reader = JournalReader(journal_path)
            envelopes = list(reader.iter_envelopes())
        except OSError as exc:
            logger.warning(
                "[JournalAdapter] could not read %s: %s — returning empty data",
                journal_path,
                exc,
            )
            return ExperimentEventData(run_id=run_id, source=source)

        return self._project(envelopes, run_id=run_id, source=source)

    def _download_from_mlflow(self, run_id: str) -> Path | None:
        """Try to download ``events/events.jsonl`` from MLflow.

        Returns the local path when the download succeeded, ``None``
        otherwise. All errors are swallowed by design — events.jsonl is
        best-effort.
        """
        client = getattr(self._mlflow, "client", None)
        if client is None:
            return None

        artifact_path = f"{self._artifact_dir}/{self._journal_filename}"
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix=f"journal_{run_id[:8]}_"))
            local_path = client.download_artifacts(run_id, artifact_path, str(temp_dir))
            local = Path(local_path)
            if local.exists():
                return local
        except Exception as exc:
            logger.info(
                "[JournalAdapter] MLflow download of %s failed for run %s: %s",
                artifact_path,
                run_id[:8] if run_id else "<unknown>",
                exc,
            )
        return None

    def _project(
        self,
        envelopes: Iterable[BaseEvent],
        *,
        run_id: str,
        source: str,
    ) -> ExperimentEventData:
        """Project typed envelopes into :class:`ExperimentEventData`.

        Single linear pass over the journal. Stages are correlated by
        ``payload.stage_name``; out-of-order completion/failure events
        still resolve correctly because the started event seeds the
        record and the terminal event finalises it.
        """
        timeline_entries: list[TimelineEntry] = []
        memory_events: list[MemoryEvent] = []
        stage_outcomes: dict[str, StageOutcome] = {}
        total = 0
        unknown_count = 0

        for envelope in envelopes:
            total += 1
            if isinstance(envelope, UnknownEvent):
                unknown_count += 1
                continue

            # --- Stage envelopes ---------------------------------------
            if isinstance(envelope, StageStartedEvent):
                outcome = stage_outcomes.setdefault(
                    envelope.payload.stage_name,
                    StageOutcome(stage_name=envelope.payload.stage_name, status="running"),
                )
                outcome.started_at = envelope.time
                outcome.status = "running"
                continue

            if isinstance(envelope, StageCompletedEvent):
                outcome = stage_outcomes.setdefault(
                    envelope.payload.stage_name,
                    StageOutcome(stage_name=envelope.payload.stage_name, status="completed"),
                )
                outcome.status = "completed"
                outcome.completed_at = envelope.time
                outcome.duration_s = float(envelope.payload.duration_s)
                timeline_entries.append(
                    TimelineEntry(
                        timestamp=outcome.started_at or envelope.time,
                        kind="stage",
                        message=f"{envelope.payload.stage_name}: completed ({outcome.duration_s:.1f}s)",
                        severity="INFO",
                        source=envelope.source,
                        attributes={
                            "stage": envelope.payload.stage_name,
                            "status": "completed",
                            "duration_seconds": outcome.duration_s,
                        },
                    )
                )
                continue

            if isinstance(envelope, StageFailedEvent):
                outcome = stage_outcomes.setdefault(
                    envelope.payload.stage_name,
                    StageOutcome(stage_name=envelope.payload.stage_name, status="failed"),
                )
                outcome.status = "failed"
                outcome.completed_at = envelope.time
                outcome.error = envelope.payload.message
                if outcome.started_at is not None and outcome.duration_s == 0.0:
                    outcome.duration_s = (envelope.time - outcome.started_at).total_seconds()
                timeline_entries.append(
                    TimelineEntry(
                        timestamp=outcome.started_at or envelope.time,
                        kind="stage",
                        message=f"{envelope.payload.stage_name}: failed ({envelope.payload.error_type})",
                        severity="ERROR",
                        source=envelope.source,
                        attributes={
                            "stage": envelope.payload.stage_name,
                            "status": "failed",
                            "error": envelope.payload.message,
                            "error_type": envelope.payload.error_type,
                        },
                    )
                )
                continue

            if isinstance(envelope, StageSkippedEvent):
                outcome = stage_outcomes.setdefault(
                    envelope.payload.stage_name,
                    StageOutcome(stage_name=envelope.payload.stage_name, status="skipped"),
                )
                outcome.status = "skipped"
                outcome.completed_at = envelope.time
                timeline_entries.append(
                    TimelineEntry(
                        timestamp=envelope.time,
                        kind="stage",
                        message=f"{envelope.payload.stage_name}: skipped ({envelope.payload.reason})",
                        severity="INFO",
                        source=envelope.source,
                        attributes={
                            "stage": envelope.payload.stage_name,
                            "status": "skipped",
                            "reason": envelope.payload.reason,
                        },
                    )
                )
                continue

            if isinstance(envelope, StageInterruptedEvent):
                outcome = stage_outcomes.setdefault(
                    envelope.payload.stage_name,
                    StageOutcome(stage_name=envelope.payload.stage_name, status="interrupted"),
                )
                outcome.status = "interrupted"
                outcome.completed_at = envelope.time
                if outcome.started_at is not None and outcome.duration_s == 0.0:
                    outcome.duration_s = (envelope.time - outcome.started_at).total_seconds()
                timeline_entries.append(
                    TimelineEntry(
                        timestamp=outcome.started_at or envelope.time,
                        kind="stage",
                        message=f"{envelope.payload.stage_name}: interrupted (signal={envelope.payload.signal})",
                        severity="WARN",
                        source=envelope.source,
                        attributes={
                            "stage": envelope.payload.stage_name,
                            "status": "interrupted",
                            "signal": envelope.payload.signal,
                        },
                    )
                )
                continue

            # --- Memory envelopes --------------------------------------
            if isinstance(envelope, MemoryCacheClearedEvent):
                freed_bytes = max(
                    0,
                    envelope.payload.before_bytes - envelope.payload.after_bytes,
                )
                memory_events.append(
                    MemoryEvent(
                        timestamp=envelope.time,
                        event_type="cache_clear",
                        message=f"Cache cleared on {envelope.payload.device} ({envelope.payload.trigger})",
                        source=envelope.source or "MemoryManager",
                        freed_mb=freed_bytes // _BYTES_PER_MB if freed_bytes else 0,
                        operation=envelope.payload.trigger,
                    )
                )
                continue

            if isinstance(envelope, MemoryOOMDetectedEvent):
                memory_events.append(
                    MemoryEvent(
                        timestamp=envelope.time,
                        event_type="oom",
                        message=f"OOM detected on {envelope.payload.device}",
                        source=envelope.source or "MemoryManager",
                        operation="oom",
                    )
                )
                continue

            if isinstance(envelope, MemoryPressureWarningEvent):
                memory_events.append(
                    MemoryEvent(
                        timestamp=envelope.time,
                        event_type="warning",
                        message=(
                            f"Memory pressure on {envelope.payload.device} "
                            f"({envelope.payload.utilization_pct:.1f}% / "
                            f"threshold {envelope.payload.threshold_pct:.1f}%)"
                        ),
                        source=envelope.source or "MemoryManager",
                        utilization_percent=envelope.payload.utilization_pct,
                    )
                )
                continue

            if isinstance(envelope, MemoryThresholdReachedEvent):
                memory_events.append(
                    MemoryEvent(
                        timestamp=envelope.time,
                        event_type="warning",
                        message=(
                            f"Memory threshold on {envelope.payload.device}: "
                            f"{envelope.payload.metric}={envelope.payload.value:.2f} ≥ "
                            f"{envelope.payload.threshold:.2f} "
                            f"(action: {envelope.payload.action_taken})"
                        ),
                        source=envelope.source or "MemoryManager",
                        operation=envelope.payload.action_taken,
                    )
                )
                continue

            # All other typed events are intentionally ignored by the
            # report adapter at this phase; they may be surfaced by
            # purpose-specific plugins later.

        # Final invariant: timeline entries are sorted ascending by
        # timestamp (None first so the builder can still find ordering
        # within events whose started_at was lost).
        timeline_entries.sort(
            key=lambda e: (e.timestamp is None, e.timestamp)
        )

        return ExperimentEventData(
            run_id=run_id,
            timeline_entries=timeline_entries,
            memory_events=memory_events,
            stage_outcomes=stage_outcomes,
            total_envelopes=total,
            unknown_envelopes=unknown_count,
            source=source,
        )


__all__ = [
    "JOURNAL_FILENAME",
    "MLFLOW_EVENTS_ARTIFACT_DIR",
    "ExperimentEventData",
    "JournalReportAdapter",
]

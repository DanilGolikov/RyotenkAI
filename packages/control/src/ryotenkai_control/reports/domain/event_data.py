"""Domain dataclasses for journal-derived event data (Phase 7).

The report adapter pipeline used to parse the legacy
``training_events.json`` MLflow artifact. After Phase 6.a/b the typed
event journal (``events.jsonl``) is the single source of truth for
runtime events; this module holds the small, typed projection of that
journal that the report builder consumes:

* :class:`StageOutcome` — one record per stage envelope (started +
  completed/failed pair from ``ryotenkai.control.stage.*`` events).
* :class:`ExperimentEventData` — the aggregate bag a
  :class:`~ryotenkai_control.reports.adapters.journal_adapter.JournalReportAdapter`
  returns. Composed alongside the MLflow run metadata (params, tags,
  metric history) when the report is built.

Memory events reuse :class:`~ryotenkai_control.reports.domain.entities.MemoryEvent`
and the existing :class:`~ryotenkai_control.reports.models.report.TimelineEvent`
view model is built directly from the data here by the builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from ryotenkai_control.reports.domain.entities import MemoryEvent


@dataclass
class StageOutcome:
    """Derived per-stage outcome assembled from typed stage events.

    Mirrors the slim subset of
    :class:`ryotenkai_control.pipeline.artifacts.base.StageArtifactEnvelope`
    the report timeline needs: stage identity, status, timing, and a
    failure message when the stage failed.
    """

    stage_name: str
    status: str  # "running", "completed", "failed", "skipped", "interrupted"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_s: float = 0.0
    error: str | None = None


@dataclass
class TimelineEntry:
    """Timestamped event projection used by the report builder timeline.

    Kept narrow on purpose — the report's
    :class:`~ryotenkai_control.reports.models.report.TimelineEvent`
    view model is still the wire format the renderer consumes; this is
    just the journal-side carrier so adapters don't need to reach into
    the view layer.
    """

    timestamp: datetime | None
    kind: str  # "stage" | "memory" | "lifecycle" | ...
    message: str
    severity: str = "INFO"  # "INFO" | "WARN" | "ERROR"
    source: str = ""
    attributes: dict = field(default_factory=dict)


@dataclass
class ExperimentEventData:
    """Aggregate of all journal-derived data for a single run.

    Composed with MLflow metadata (params, tags, metric history) by the
    report builder. The fields here only come from ``events.jsonl``.
    """

    run_id: str
    timeline_entries: list[TimelineEntry] = field(default_factory=list)
    memory_events: list[MemoryEvent] = field(default_factory=list)
    stage_outcomes: dict[str, StageOutcome] = field(default_factory=dict)
    # Per-source counters useful for diagnostics / missing-artifact warnings.
    total_envelopes: int = 0
    unknown_envelopes: int = 0
    source: str = "workspace"  # "workspace" | "mlflow" | "empty"


__all__ = [
    "ExperimentEventData",
    "StageOutcome",
    "TimelineEntry",
]

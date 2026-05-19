"""Unit tests for :class:`JournalReportAdapter` (Phase 7).

7 classes mirror the policy contracts demanded by the test plan:
positive, negative, boundary, invariants, dependency errors,
regressions, and logic-specific behaviour.

Helpers build typed events directly (no JournalWriter) and serialize
them with :func:`ryotenkai_shared.events.codec.to_jsonl` so the tests
exercise the same wire format the production writer produces.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ryotenkai_control.reports.adapters.journal_adapter import (
    JOURNAL_FILENAME,
    JournalReportAdapter,
)
from ryotenkai_control.reports.domain.event_data import ExperimentEventData, StageOutcome
from ryotenkai_shared.events import to_jsonl
from ryotenkai_shared.events.types.control_stage import (
    StageCompletedEvent,
    StageCompletedPayload,
    StageFailedEvent,
    StageFailedPayload,
    StageInterruptedEvent,
    StageInterruptedPayload,
    StageSkippedEvent,
    StageSkippedPayload,
    StageStartedEvent,
    StageStartedPayload,
)
from ryotenkai_shared.events.types.pod_memory import (
    MemoryCacheClearedEvent,
    MemoryCacheClearedPayload,
    MemoryOOMDetectedEvent,
    MemoryOOMDetectedPayload,
    MemoryPressureWarningEvent,
    MemoryPressureWarningPayload,
)

RUN_ID = "run-phase-7-test"
T0 = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _stage_started(offset: int, stage: str, *, time: datetime | None = None) -> StageStartedEvent:
    return StageStartedEvent(
        source=f"control://orchestrator/{stage}",
        run_id=RUN_ID,
        offset=offset,
        time=time or T0,
        stage_id=stage,
        payload=StageStartedPayload(
            stage_name=stage,
            stage_index=0,
            total_stages=1,
            inputs_summary={},
        ),
    )


def _stage_completed(
    offset: int, stage: str, *, time: datetime | None = None, duration_s: float = 5.0
) -> StageCompletedEvent:
    return StageCompletedEvent(
        source=f"control://orchestrator/{stage}",
        run_id=RUN_ID,
        offset=offset,
        time=time or (T0 + timedelta(seconds=duration_s)),
        stage_id=stage,
        payload=StageCompletedPayload(
            stage_name=stage,
            duration_s=duration_s,
            outputs_summary={},
        ),
    )


def _stage_failed(offset: int, stage: str, *, time: datetime | None = None) -> StageFailedEvent:
    return StageFailedEvent(
        source=f"control://orchestrator/{stage}",
        run_id=RUN_ID,
        offset=offset,
        time=time or (T0 + timedelta(seconds=2)),
        stage_id=stage,
        payload=StageFailedPayload(
            stage_name=stage,
            error_type="RuntimeError",
            message="boom",
            traceback_excerpt="...",
        ),
    )


def _cache_cleared(offset: int, *, before: int, after: int) -> MemoryCacheClearedEvent:
    return MemoryCacheClearedEvent(
        source="pod://memory_manager",
        run_id=RUN_ID,
        offset=offset,
        time=T0 + timedelta(seconds=offset),
        payload=MemoryCacheClearedPayload(
            device="cuda:0",
            before_bytes=before,
            after_bytes=after,
            trigger="scheduled",
        ),
    )


def _oom(offset: int) -> MemoryOOMDetectedEvent:
    return MemoryOOMDetectedEvent(
        source="pod://memory_manager",
        run_id=RUN_ID,
        offset=offset,
        time=T0 + timedelta(seconds=offset),
        payload=MemoryOOMDetectedPayload(
            device="cuda:0",
            allocated_bytes=10 * 1024**3,
            reserved_bytes=12 * 1024**3,
        ),
    )


def _pressure_warning(offset: int, *, util_pct: float = 92.5) -> MemoryPressureWarningEvent:
    return MemoryPressureWarningEvent(
        source="pod://memory_manager",
        run_id=RUN_ID,
        offset=offset,
        time=T0 + timedelta(seconds=offset),
        payload=MemoryPressureWarningPayload(
            device="cuda:0",
            utilization_pct=util_pct,
            threshold_pct=90.0,
        ),
    )


def _skipped(offset: int, stage: str) -> StageSkippedEvent:
    return StageSkippedEvent(
        source=f"control://orchestrator/{stage}",
        run_id=RUN_ID,
        offset=offset,
        time=T0 + timedelta(seconds=offset),
        stage_id=stage,
        payload=StageSkippedPayload(stage_name=stage, reason="disabled"),
    )


def _interrupted(offset: int, stage: str) -> StageInterruptedEvent:
    return StageInterruptedEvent(
        source=f"control://orchestrator/{stage}",
        run_id=RUN_ID,
        offset=offset,
        time=T0 + timedelta(seconds=offset),
        stage_id=stage,
        payload=StageInterruptedPayload(stage_name=stage, signal=2, cleanup_completed=True),
    )


def _write_journal(path: Path, *events) -> Path:
    """Materialise ``events`` to a JSONL journal file (length-prefixed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as fh:
        for evt in events:
            fh.write(to_jsonl(evt).encode("utf-8"))
    return path


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestPositive:
    """Workspace journal exists → returns ExperimentEventData with all fields populated."""

    def test_load_workspace_journal_with_stage_and_memory_events(self, tmp_path: Path) -> None:
        journal = tmp_path / JOURNAL_FILENAME
        _write_journal(
            journal,
            _stage_started(0, "validate"),
            _stage_completed(1, "validate", duration_s=3.5),
            _cache_cleared(2, before=10 * 1024 * 1024, after=4 * 1024 * 1024),
        )

        adapter = JournalReportAdapter()
        data = adapter.load(RUN_ID, workspace_dir=tmp_path)

        assert isinstance(data, ExperimentEventData)
        assert data.source == "workspace"
        assert data.total_envelopes == 3
        assert data.unknown_envelopes == 0

        # Stage outcome
        assert "validate" in data.stage_outcomes
        outcome = data.stage_outcomes["validate"]
        assert isinstance(outcome, StageOutcome)
        assert outcome.status == "completed"
        assert outcome.duration_s == pytest.approx(3.5)

        # Timeline entry mirrors the completed stage
        assert len(data.timeline_entries) == 1
        entry = data.timeline_entries[0]
        assert entry.kind == "stage"
        assert "validate" in entry.message
        assert entry.severity == "INFO"

        # Memory event with freed_mb
        assert len(data.memory_events) == 1
        mem = data.memory_events[0]
        assert mem.event_type == "cache_clear"
        assert mem.freed_mb == 6  # 10 MiB - 4 MiB


class TestNegative:
    """Workspace missing → MLflow fallback succeeds → returns data."""

    def test_workspace_missing_then_mlflow_fallback_loads(self, tmp_path: Path) -> None:
        # Pretend MLflow has the journal artifact
        mlflow_local = tmp_path / "mlflow_download"
        mlflow_local.mkdir()
        journal_path = mlflow_local / JOURNAL_FILENAME
        _write_journal(journal_path, _stage_started(0, "deploy"), _stage_completed(1, "deploy"))

        class FakeClient:
            def download_artifacts(self, run_id, artifact_path, local_dir):
                # Simulate MLflow returning the path inside local_dir
                target = Path(local_dir) / JOURNAL_FILENAME
                target.write_bytes(journal_path.read_bytes())
                return str(target)

        # After the wide ``IMLflowManager`` retirement, the adapter
        # constructs an MlflowClient internally from ``tracking_uri``.
        # Patch the constructor at the import site so the test exercises
        # the new code path.
        import mlflow as _mlflow_mod

        original_client_cls = _mlflow_mod.MlflowClient
        try:
            _mlflow_mod.MlflowClient = lambda **_kwargs: FakeClient()  # type: ignore[assignment]
            adapter = JournalReportAdapter(tracking_uri="http://mlflow.test")
            data = adapter.load(RUN_ID, workspace_dir=tmp_path / "does-not-exist")
        finally:
            _mlflow_mod.MlflowClient = original_client_cls  # type: ignore[assignment]

        assert data.source == "mlflow"
        assert data.total_envelopes == 2
        assert "deploy" in data.stage_outcomes


class TestBoundary:
    """Empty journal → empty event data."""

    def test_empty_journal_returns_empty_data(self, tmp_path: Path) -> None:
        # Empty file but exists
        (tmp_path / JOURNAL_FILENAME).touch()
        adapter = JournalReportAdapter()
        data = adapter.load(RUN_ID, workspace_dir=tmp_path)

        assert data.source == "workspace"
        assert data.total_envelopes == 0
        assert data.timeline_entries == []
        assert data.memory_events == []
        assert data.stage_outcomes == {}

    def test_workspace_dir_missing_and_no_mlflow_returns_empty(self, tmp_path: Path) -> None:
        adapter = JournalReportAdapter()
        data = adapter.load(RUN_ID, workspace_dir=tmp_path / "nope")
        assert data.source == "empty"
        assert data.total_envelopes == 0


class TestInvariants:
    """Timeline entries sorted ascending by timestamp."""

    def test_timeline_sorted_by_timestamp(self, tmp_path: Path) -> None:
        journal = tmp_path / JOURNAL_FILENAME
        # Intentionally write stages out of strict time order
        _write_journal(
            journal,
            _stage_started(0, "a", time=T0 + timedelta(seconds=10)),
            _stage_completed(1, "a", time=T0 + timedelta(seconds=20), duration_s=10),
            _stage_started(2, "b", time=T0),
            _stage_completed(3, "b", time=T0 + timedelta(seconds=5), duration_s=5),
        )

        adapter = JournalReportAdapter()
        data = adapter.load(RUN_ID, workspace_dir=tmp_path)

        assert len(data.timeline_entries) == 2
        ts = [e.timestamp for e in data.timeline_entries]
        assert ts == sorted(ts)  # ascending


class TestDependencyErrors:
    """MLflow download fails → log + empty (or whatever workspace had)."""

    def test_mlflow_download_failure_yields_empty(self, tmp_path: Path) -> None:
        class FailingClient:
            def download_artifacts(self, *args, **kwargs):
                raise RuntimeError("artifact missing")

        import mlflow as _mlflow_mod

        original_client_cls = _mlflow_mod.MlflowClient
        try:
            _mlflow_mod.MlflowClient = lambda **_kwargs: FailingClient()  # type: ignore[assignment]
            adapter = JournalReportAdapter(tracking_uri="http://mlflow.test")
            data = adapter.load(RUN_ID, workspace_dir=tmp_path / "missing")
        finally:
            _mlflow_mod.MlflowClient = original_client_cls  # type: ignore[assignment]

        assert data.source == "empty"
        assert data.timeline_entries == []
        assert data.memory_events == []

    def test_no_tracking_uri_skips_fallback(self, tmp_path: Path) -> None:
        """When no tracking_uri is configured, the adapter skips the
        MLflow fallback (no client construction).
        """
        adapter = JournalReportAdapter(tracking_uri=None)
        data = adapter.load(RUN_ID, workspace_dir=tmp_path / "missing")
        assert data.source == "empty"


class TestRegressions:
    """Forward-compat: malformed lines + future event kinds become UnknownEvent
    and are counted but skipped without crashing the adapter."""

    def test_unknown_envelope_in_journal_skipped_and_counted(self, tmp_path: Path) -> None:
        journal = tmp_path / JOURNAL_FILENAME
        _write_journal(journal, _stage_started(0, "a"), _stage_completed(1, "a"))

        # Append a synthetic future-kind line manually. The codec maps
        # unknown discriminator values to UnknownEvent at parse time.
        future_line = '{"event_id":"00000000-0000-7000-8000-000000000099",' \
                      '"kind":"ryotenkai.future.unknown.v9",' \
                      '"source":"future","time":"2026-05-16T12:00:00Z",' \
                      '"run_id":"' + RUN_ID + '","offset":42,"schema_version":1,' \
                      '"severity":"info","payload":{}}\n'
        with journal.open("ab") as fh:
            fh.write(future_line.encode("utf-8"))

        adapter = JournalReportAdapter()
        data = adapter.load(RUN_ID, workspace_dir=tmp_path)

        # 2 typed envelopes + 1 unknown = 3 total
        assert data.total_envelopes == 3
        assert data.unknown_envelopes == 1
        # Known events still parsed correctly
        assert "a" in data.stage_outcomes
        assert data.stage_outcomes["a"].status == "completed"


class TestLogicSpecific:
    """Stage outcomes correctly derived from started+completed/failed pairs;
    memory event projections preserve typed fields."""

    def test_started_then_failed_pair_produces_failed_outcome(self, tmp_path: Path) -> None:
        journal = tmp_path / JOURNAL_FILENAME
        _write_journal(
            journal,
            _stage_started(0, "train"),
            _stage_failed(1, "train"),
        )
        adapter = JournalReportAdapter()
        data = adapter.load(RUN_ID, workspace_dir=tmp_path)

        assert data.stage_outcomes["train"].status == "failed"
        assert data.stage_outcomes["train"].error == "boom"
        assert data.stage_outcomes["train"].duration_s == pytest.approx(2.0)
        # One ERROR timeline entry
        assert len(data.timeline_entries) == 1
        assert data.timeline_entries[0].severity == "ERROR"

    def test_skipped_and_interrupted_outcomes(self, tmp_path: Path) -> None:
        journal = tmp_path / JOURNAL_FILENAME
        _write_journal(
            journal,
            _skipped(0, "eval"),
            _stage_started(1, "infer"),
            _interrupted(2, "infer"),
        )
        adapter = JournalReportAdapter()
        data = adapter.load(RUN_ID, workspace_dir=tmp_path)

        assert data.stage_outcomes["eval"].status == "skipped"
        assert data.stage_outcomes["infer"].status == "interrupted"
        # Severity mapping: skipped=INFO, interrupted=WARN
        sev_by_stage = {e.attributes["stage"]: e.severity for e in data.timeline_entries}
        assert sev_by_stage["eval"] == "INFO"
        assert sev_by_stage["infer"] == "WARN"

    def test_memory_event_projection_preserves_typed_payload(self, tmp_path: Path) -> None:
        journal = tmp_path / JOURNAL_FILENAME
        _write_journal(
            journal,
            _oom(0),
            _pressure_warning(1, util_pct=97.0),
        )
        adapter = JournalReportAdapter()
        data = adapter.load(RUN_ID, workspace_dir=tmp_path)

        types = [m.event_type for m in data.memory_events]
        assert types == ["oom", "warning"]
        warn = data.memory_events[1]
        assert warn.utilization_percent == pytest.approx(97.0)

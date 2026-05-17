"""Tests for :class:`ryotenkai_control.events.MlflowFinalizer` (Phase 6.a).

The finalizer composes:

* journal scan (sha256 + per-source offsets + type histogram)
* manifest writer (JSON sidecar)
* MLflow upload with exponential backoff retry

We exercise the seven canonical classes from
``docs/testing/mock_policy.md`` — chaos surfaces are the IMLflowManager
fake's ``fail_next_n_calls`` + ``raise_next_n`` and a deterministic
``sleep`` injection so the retry budget is enforced without wall-clock
waits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ryotenkai_control.events import (
    ControlEventEmitter,
    JournalWriter,
    MlflowFinalizer,
)
from ryotenkai_control.events.mlflow_finalizer import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
)
from tests.unit.control.events.conftest import (
    make_completed,
    make_failed,
    make_started,
)

# ---------------------------------------------------------------------------
# In-test MLflow recorder — no production code under test relies on the
# wider IMLflowManager surface (we only need ``log_artifact``).
# ---------------------------------------------------------------------------


@dataclass
class _ArtifactCall:
    local_path: str
    artifact_path: str | None
    run_id: str | None


@dataclass
class _ChaosMlflowRecorder:
    """Records ``log_artifact`` calls and supports staged exception injection."""

    calls: list[_ArtifactCall] = field(default_factory=list)
    fail_next: int = 0
    exception_type: type[BaseException] = RuntimeError

    def fail_next_n(self, n: int, exc_type: type[BaseException] = RuntimeError) -> None:
        self.fail_next = n
        self.exception_type = exc_type

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> bool:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise self.exception_type(f"chaos-fail({self.fail_next + 1})")
        self.calls.append(
            _ArtifactCall(
                local_path=local_path,
                artifact_path=artifact_path,
                run_id=run_id,
            ),
        )
        return True

    # Required for the IMLflowManager Protocol surface (we never call
    # any other methods in MlflowFinalizer, but mypy / runtime Protocol
    # may expect them).
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - defensive
        def _stub(*_a: Any, **_kw: Any) -> Any:
            raise AssertionError(f"MlflowFinalizer called unexpected method {name!r}")
        return _stub


def _seed_journal(path: Path) -> None:
    """Create a journal with three envelopes covering two sources."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = JournalWriter(path)
    writer.append(make_started(offset=0, source="control://orchestrator"))
    writer.append(make_completed(offset=1, source="control://orchestrator"))
    writer.append(
        make_failed(offset=0, source="pod://abc/trainer"),
    )
    writer.close()


def _no_sleep(_: float) -> None:
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPositive:
    def test_upload_success_writes_manifest_and_uploads_both_artifacts(
        self, tmp_path: Path,
    ) -> None:
        journal = tmp_path / "events.jsonl"
        _seed_journal(journal)
        mlflow = _ChaosMlflowRecorder()
        finalizer = MlflowFinalizer(mlflow, sleep=_no_sleep)

        ok = finalizer.upload(
            run_id="mlflow-run-1",
            journal_path=journal,
        )

        assert ok is True
        manifest_path = tmp_path / MANIFEST_FILENAME
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text("utf-8"))
        assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
        assert manifest["total_events"] == 3
        assert manifest["mlflow_uploaded"] is True
        assert manifest["events_sha256"] == hashlib.sha256(
            journal.read_bytes(),
        ).hexdigest()
        # Two calls: events.jsonl + events_manifest.json
        assert len(mlflow.calls) == 2
        paths = {c.local_path for c in mlflow.calls}
        assert str(journal) in paths
        assert str(manifest_path) in paths
        for call in mlflow.calls:
            assert call.artifact_path == "events"
            assert call.run_id == "mlflow-run-1"


class TestNegative:
    def test_all_attempts_fail_records_mlflow_uploaded_false(
        self, tmp_path: Path,
    ) -> None:
        journal = tmp_path / "events.jsonl"
        _seed_journal(journal)
        mlflow = _ChaosMlflowRecorder()
        mlflow.fail_next_n(99)  # never succeed
        finalizer = MlflowFinalizer(
            mlflow,
            retry_delays_s=(0.0, 0.0, 0.0),
            sleep=_no_sleep,
        )

        ok = finalizer.upload(
            run_id="mlflow-run-2",
            journal_path=journal,
        )

        assert ok is False
        manifest = json.loads(
            (tmp_path / MANIFEST_FILENAME).read_text("utf-8"),
        )
        assert manifest["mlflow_uploaded"] is False


class TestBoundary:
    def test_empty_journal_has_zero_events_and_empty_hash(
        self, tmp_path: Path,
    ) -> None:
        journal = tmp_path / "events.jsonl"
        journal.touch()
        mlflow = _ChaosMlflowRecorder()
        finalizer = MlflowFinalizer(mlflow, sleep=_no_sleep)

        ok = finalizer.upload(
            run_id="run-empty",
            journal_path=journal,
        )

        assert ok is True
        manifest = json.loads(
            (tmp_path / MANIFEST_FILENAME).read_text("utf-8"),
        )
        assert manifest["total_events"] == 0
        assert manifest["first_offset_per_source"] == {}
        assert manifest["last_offset_per_source"] == {}
        assert manifest["first_time"] is None
        assert manifest["last_time"] is None
        # Empty file SHA-256
        assert manifest["events_sha256"] == hashlib.sha256(b"").hexdigest()


class TestInvariants:
    def test_type_histogram_multi_source(self, tmp_path: Path) -> None:
        journal = tmp_path / "events.jsonl"
        _seed_journal(journal)
        mlflow = _ChaosMlflowRecorder()
        finalizer = MlflowFinalizer(mlflow, sleep=_no_sleep)

        finalizer.upload(run_id="run-1", journal_path=journal)

        manifest = json.loads(
            (tmp_path / MANIFEST_FILENAME).read_text("utf-8"),
        )
        histogram = manifest["type_histogram"]
        # One started + one completed + one failed
        assert histogram["ryotenkai.control.run.started"] == 1
        assert histogram["ryotenkai.control.run.completed"] == 1
        assert histogram["ryotenkai.control.run.failed"] == 1
        # Two distinct sources reflected in per-source maps
        assert set(manifest["first_offset_per_source"].keys()) == {
            "control://orchestrator",
            "pod://abc/trainer",
        }


class TestDependencyErrors:
    def test_journal_missing_returns_false_without_raising(
        self, tmp_path: Path,
    ) -> None:
        # No journal file exists.
        journal = tmp_path / "does-not-exist.jsonl"
        mlflow = _ChaosMlflowRecorder()
        finalizer = MlflowFinalizer(mlflow, sleep=_no_sleep)

        ok = finalizer.upload(
            run_id="run-1",
            journal_path=journal,
        )

        assert ok is False
        # MLflow was NOT contacted — no point uploading a missing file.
        assert mlflow.calls == []


class TestRegressions:
    def test_retry_policy_sleeps_in_correct_order(self, tmp_path: Path) -> None:
        """Two transient failures then success — sleep cadence matches policy."""
        journal = tmp_path / "events.jsonl"
        _seed_journal(journal)
        sleeps: list[float] = []

        def _record_sleep(d: float) -> None:
            sleeps.append(d)

        mlflow = _ChaosMlflowRecorder()
        # First 2 log_artifact calls fail (across attempts 1 & 2 — each
        # attempt issues 1 call for journal upload; on success it
        # issues a second call for the manifest, which we don't fail).
        mlflow.fail_next_n(2)

        finalizer = MlflowFinalizer(
            mlflow,
            retry_delays_s=(1.0, 5.0, 30.0),
            sleep=_record_sleep,
        )

        ok = finalizer.upload(
            run_id="run-1",
            journal_path=journal,
        )

        assert ok is True
        # Attempt 1 fails → sleep 5.0 before attempt 2.
        # Attempt 2 fails → sleep 30.0 before attempt 3.
        # Attempt 3 succeeds → no further sleeps.
        # The first delay (1.0) is consumed only on the first sleep
        # between attempts; we use list-index-based dispatch.
        assert sleeps == [5.0, 30.0]


class TestLogicSpecific:
    def test_cancellation_metadata_propagates(self, tmp_path: Path) -> None:
        journal = tmp_path / "events.jsonl"
        _seed_journal(journal)
        mlflow = _ChaosMlflowRecorder()
        finalizer = MlflowFinalizer(mlflow, sleep=_no_sleep)

        finalizer.upload(
            run_id="run-1",
            journal_path=journal,
            cancellation_reason="signal:SIGINT",
            journal_complete=False,
        )

        manifest = json.loads(
            (tmp_path / MANIFEST_FILENAME).read_text("utf-8"),
        )
        assert manifest["cancellation_reason"] == "signal:SIGINT"
        assert manifest["journal_complete"] is False

    def test_journal_built_via_emitter_close_then_uploaded(
        self, tmp_path: Path,
    ) -> None:
        """End-to-end: events emitted via ControlEventEmitter and then
        uploaded. The sha256 in the manifest must match the on-disk
        file AFTER ``emitter.close()`` (i.e. fsync completed).
        """
        run_dir = tmp_path / "run"
        emitter = ControlEventEmitter.for_run(
            run_id="run-1", run_directory=run_dir,
        )
        try:
            emitter.emit(make_started())
            emitter.emit(make_completed())
        finally:
            emitter.close()

        journal_path = run_dir / "events.jsonl"
        expected = hashlib.sha256(journal_path.read_bytes()).hexdigest()

        mlflow = _ChaosMlflowRecorder()
        finalizer = MlflowFinalizer(mlflow, sleep=_no_sleep)
        ok = finalizer.upload(run_id="run-1", journal_path=journal_path)
        assert ok is True
        manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text("utf-8"))
        assert manifest["events_sha256"] == expected
        assert manifest["total_events"] == 2

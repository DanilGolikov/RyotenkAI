"""Tests for the Phase H2 ``AttemptFailure`` dataclass + schema migration.

Coverage split by 7-class policy:

1. **Positive**          — happy-path roundtrip / from_exception / from_legacy.
2. **Negative**          — garbage values fall back safely.
3. **Boundary**          — None fields, empty strings, missing keys.
4. **Invariants**        — None failure on success / pending attempts.
5. **Regression**        — legacy ``error: str`` state files load with
                            auto-back-filled ``failure``.
6. **Schema migration**  — schema_version=1 → 2 upgrade path.
7. **Logic-specific**    — record_failure mutation semantics on AttemptController.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from ryotenkai_control.pipeline.state.attempt_controller import AttemptController
from ryotenkai_control.pipeline.state.models import (
    AttemptFailure,
    PipelineAttemptState,
    PipelineState,
    StageRunState,
)
from ryotenkai_control.pipeline.state.store import (
    SCHEMA_VERSION,
    PipelineStateLoadError,
    PipelineStateStore,
)
from ryotenkai_shared.errors import PipelineStageFailedError, RyotenkAIError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_attempt(status: str = StageRunState.STATUS_RUNNING) -> PipelineAttemptState:
    return PipelineAttemptState(
        attempt_id="attempt_1",
        attempt_no=1,
        runtime_name="rt",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=None,
        status=status,
        started_at="2026-05-17T00:00:00Z",
    )


def _make_state() -> PipelineState:
    return PipelineState(
        schema_version=SCHEMA_VERSION,
        logical_run_id="run-1",
        run_directory="/tmp/run-1",
        config_path="/tmp/c.yaml",
        active_attempt_id=None,
        pipeline_status=StageRunState.STATUS_PENDING,
        training_critical_config_hash="",
        late_stage_config_hash="",
    )


# ---------------------------------------------------------------------------
# 1. Positive — roundtrip serialisation
# ---------------------------------------------------------------------------


class TestPositive:
    def test_roundtrip_full_payload(self) -> None:
        original = AttemptFailure(
            code="PIPELINE_STAGE_FAILED",
            title="Pipeline stage failed",
            detail="no GPU",
            stage_name="gpu_deployer",
            stage_idx=2,
            stage_total=8,
            trace_id="t-abc",
            request_id="r-xyz",
            context={"reason": "no_quota"},
            failed_at="2026-05-17T10:00:00Z",
        )
        round_tripped = AttemptFailure.from_dict(original.to_dict())
        assert round_tripped == original

    def test_from_exception_pulls_stage_fields_from_context(self) -> None:
        exc = PipelineStageFailedError(
            detail="boom",
            context={"stage_name": "trainer", "stage_idx": 4, "stage_total": 7},
        )
        failure = AttemptFailure.from_exception(exc, request_id="r-1")
        assert failure.code == "PIPELINE_STAGE_FAILED"
        assert failure.detail == "boom"
        assert failure.stage_name == "trainer"
        assert failure.stage_idx == 4
        assert failure.stage_total == 7
        assert failure.request_id == "r-1"
        # Trace_id pulled from RyotenkAIError instance — None for raise-site
        # errors that don't go through ``from_problem``.
        assert failure.trace_id is None

    def test_from_exception_kwargs_override_context(self) -> None:
        exc = PipelineStageFailedError(
            detail="boom",
            context={"stage_name": "trainer", "stage_idx": 4, "stage_total": 7},
        )
        failure = AttemptFailure.from_exception(
            exc, stage_name="other", stage_idx=99, stage_total=99
        )
        assert failure.stage_name == "other"
        assert failure.stage_idx == 99
        assert failure.stage_total == 99

    def test_from_legacy_error_string(self) -> None:
        f = AttemptFailure.from_legacy_error_string(
            "rcloned died", failed_at="2026-05-17T01:01:01Z"
        )
        assert f.code == "LEGACY_ERROR"
        assert f.title == "Legacy attempt failure"
        assert f.detail == "rcloned died"
        assert f.failed_at == "2026-05-17T01:01:01Z"


# ---------------------------------------------------------------------------
# 2. Negative — garbage / malformed input
# ---------------------------------------------------------------------------


class TestNegative:
    def test_non_dict_context_dropped(self) -> None:
        d = {"code": "X", "title": "x", "context": "not a dict"}
        f = AttemptFailure.from_dict(d)
        assert f.context is None

    def test_non_int_stage_idx_falls_back_to_none(self) -> None:
        d = {"code": "X", "title": "x", "stage_idx": "garbage"}
        f = AttemptFailure.from_dict(d)
        assert f.stage_idx is None

    def test_missing_fields_use_defaults(self) -> None:
        f = AttemptFailure.from_dict({})
        assert f.code == ""
        assert f.title == ""
        assert f.detail is None
        assert f.context is None
        assert f.failed_at == ""


# ---------------------------------------------------------------------------
# 3. Boundary — None fields, empty strings
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_failure_field_omitted_from_serialised_attempt_when_none(self) -> None:
        attempt = _make_attempt()
        d = attempt.to_dict()
        assert "failure" not in d

    def test_failure_included_when_present(self) -> None:
        attempt = _make_attempt()
        attempt.failure = AttemptFailure(code="X", title="Y", failed_at="z")
        d = attempt.to_dict()
        assert d["failure"]["code"] == "X"
        assert d["failure"]["title"] == "Y"

    def test_empty_string_code(self) -> None:
        f = AttemptFailure(code="", title="")
        assert f.to_dict()["code"] == ""

    def test_explicit_none_context_serialises_as_none(self) -> None:
        f = AttemptFailure(code="X", title="T", context=None)
        assert f.to_dict()["context"] is None


# ---------------------------------------------------------------------------
# 4. Invariants — failure is None for pending / completed attempts
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_pending_attempt_default_failure_is_none(self) -> None:
        attempt = _make_attempt(status=StageRunState.STATUS_PENDING)
        assert attempt.failure is None

    def test_running_attempt_default_failure_is_none(self) -> None:
        attempt = _make_attempt(status=StageRunState.STATUS_RUNNING)
        assert attempt.failure is None

    def test_completed_attempt_serialised_without_failure(self) -> None:
        attempt = _make_attempt(status=StageRunState.STATUS_COMPLETED)
        assert "failure" not in attempt.to_dict()


# ---------------------------------------------------------------------------
# 5. Regression — legacy state files load with back-filled failure
# ---------------------------------------------------------------------------


class TestRegression:
    def test_legacy_failed_attempt_with_error_string_only_migrates(self) -> None:
        """Pre-H2 state.json: ``error: str`` but no ``failure`` block.

        Loader synthesises an ``AttemptFailure`` with
        ``code="LEGACY_ERROR"`` so resume tooling sees a typed
        record.
        """
        legacy = {
            "attempt_id": "a",
            "attempt_no": 1,
            "runtime_name": "x",
            "requested_action": "fresh",
            "effective_action": "fresh",
            "restart_from_stage": None,
            "status": StageRunState.STATUS_FAILED,
            "started_at": "2026-05-17T00:00:00Z",
            "completed_at": "2026-05-17T01:00:00Z",
            "error": "GPU unreachable",
        }
        attempt = PipelineAttemptState.from_dict(legacy)
        assert attempt.failure is not None
        assert attempt.failure.code == "LEGACY_ERROR"
        assert attempt.failure.detail == "GPU unreachable"
        assert attempt.failure.failed_at == "2026-05-17T01:00:00Z"
        # Legacy ``error`` field is preserved for tooling that reads
        # only the plain-string path.
        assert attempt.error == "GPU unreachable"

    def test_legacy_completed_attempt_does_not_synthesise_failure(self) -> None:
        legacy = {
            "attempt_id": "a",
            "attempt_no": 1,
            "runtime_name": "x",
            "requested_action": "fresh",
            "effective_action": "fresh",
            "restart_from_stage": None,
            "status": StageRunState.STATUS_COMPLETED,
            "started_at": "2026-05-17T00:00:00Z",
            "error": None,
        }
        attempt = PipelineAttemptState.from_dict(legacy)
        assert attempt.failure is None

    def test_legacy_failed_attempt_without_error_string(self) -> None:
        """Failed attempt with missing ``error`` ⇒ no synthesised failure."""
        legacy = {
            "attempt_id": "a",
            "attempt_no": 1,
            "runtime_name": "x",
            "requested_action": "fresh",
            "effective_action": "fresh",
            "restart_from_stage": None,
            "status": StageRunState.STATUS_FAILED,
            "started_at": "",
        }
        attempt = PipelineAttemptState.from_dict(legacy)
        assert attempt.failure is None


# ---------------------------------------------------------------------------
# 6. Schema migration — version=1 → version=2 with structured failure
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_load_legacy_state_v1_succeeds_and_bumps_version(self, tmp_path: Path) -> None:
        store = PipelineStateStore(tmp_path / "runs" / "legacy")
        store.state_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_payload = {
            "schema_version": 1,
            "logical_run_id": "legacy-run",
            "run_directory": str(tmp_path / "runs" / "legacy"),
            "config_path": "/tmp/c.yaml",
            "active_attempt_id": None,
            "pipeline_status": StageRunState.STATUS_FAILED,
            "training_critical_config_hash": "",
            "late_stage_config_hash": "",
            "attempts": [
                {
                    "attempt_id": "a",
                    "attempt_no": 1,
                    "runtime_name": "rt",
                    "requested_action": "fresh",
                    "effective_action": "fresh",
                    "restart_from_stage": None,
                    "status": StageRunState.STATUS_FAILED,
                    "started_at": "2026-05-17T00:00:00Z",
                    "completed_at": "2026-05-17T01:00:00Z",
                    "error": "GPU exploded",
                },
            ],
        }
        store.state_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

        loaded = store.load()
        assert loaded.schema_version == SCHEMA_VERSION
        assert len(loaded.attempts) == 1
        att = loaded.attempts[0]
        assert att.failure is not None
        assert att.failure.code == "LEGACY_ERROR"
        assert att.failure.detail == "GPU exploded"

    def test_load_v2_state_passes_through(self, tmp_path: Path) -> None:
        store = PipelineStateStore(tmp_path / "runs" / "v2")
        store.state_path.parent.mkdir(parents=True, exist_ok=True)
        store.state_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "logical_run_id": "v2-run",
                    "run_directory": str(tmp_path / "runs" / "v2"),
                    "config_path": "/tmp/c.yaml",
                    "active_attempt_id": None,
                    "pipeline_status": StageRunState.STATUS_PENDING,
                    "training_critical_config_hash": "",
                    "late_stage_config_hash": "",
                    "attempts": [],
                }
            ),
            encoding="utf-8",
        )
        loaded = store.load()
        assert loaded.schema_version == 2

    def test_unsupported_future_version_rejected(self, tmp_path: Path) -> None:
        store = PipelineStateStore(tmp_path / "runs" / "future")
        store.state_path.parent.mkdir(parents=True, exist_ok=True)
        store.state_path.write_text(
            json.dumps(
                {
                    "schema_version": 999,
                    "logical_run_id": "x",
                    "run_directory": str(tmp_path / "runs" / "future"),
                    "config_path": "",
                    "active_attempt_id": None,
                    "pipeline_status": "pending",
                    "training_critical_config_hash": "",
                    "late_stage_config_hash": "",
                    "attempts": [],
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(PipelineStateLoadError, match="schema_version=999"):
            store.load()


# ---------------------------------------------------------------------------
# 7. Logic-specific — AttemptController.record_failure semantics
# ---------------------------------------------------------------------------


class TestRecordFailureOnController:
    def test_record_failure_persists_and_backfills_error(self) -> None:
        save_fn = MagicMock()
        ctrl = AttemptController(save_fn=save_fn, run_ctx=SimpleNamespace(name="x"))
        ctrl.adopt_state(_make_state())
        attempt = _make_attempt()
        ctrl.register_attempt(attempt)
        save_fn.reset_mock()

        failure = AttemptFailure(
            code="PIPELINE_STAGE_FAILED",
            title="Pipeline stage failed",
            detail="explained",
            failed_at="2026-05-17T05:00:00Z",
        )
        ctrl.record_failure(failure)

        assert attempt.failure == failure
        # error: str field is also populated for legacy tooling.
        assert attempt.error == "explained"
        save_fn.assert_called_once()

    def test_record_failure_no_active_attempt_is_noop(self) -> None:
        save_fn = MagicMock()
        ctrl = AttemptController(save_fn=save_fn, run_ctx=SimpleNamespace(name="x"))
        ctrl.adopt_state(_make_state())
        ctrl.record_failure(AttemptFailure(code="X", title="T"))
        save_fn.assert_not_called()

    def test_record_failure_does_not_overwrite_existing_error(self) -> None:
        save_fn = MagicMock()
        ctrl = AttemptController(save_fn=save_fn, run_ctx=SimpleNamespace(name="x"))
        ctrl.adopt_state(_make_state())
        attempt = _make_attempt()
        attempt.error = "pre-existing"
        ctrl.register_attempt(attempt)
        save_fn.reset_mock()

        ctrl.record_failure(
            AttemptFailure(code="C", title="T", detail="new")
        )
        # The pre-existing legacy ``error`` is preserved — we only
        # back-fill when it's empty.
        assert attempt.error == "pre-existing"
        assert attempt.failure is not None
        assert attempt.failure.detail == "new"

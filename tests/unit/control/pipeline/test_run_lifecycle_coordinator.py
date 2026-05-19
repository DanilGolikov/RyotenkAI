"""Tests for :class:`ryotenkai_control.pipeline.run_lifecycle_coordinator.RunLifecycleCoordinator`.

Phase M7.2 — the legacy MLflow journal upload moved out of this
coordinator into the new
:class:`ryotenkai_control.pipeline.mlflow.lifecycle.coord.RunLifecycleCoord`
path (driven from the orchestrator's ``_teardown_mlflow_attempt``).
The events-side coordinator now owns only the event-emitter lifecycle
(emitter close + registry deregister); the MLflow manifest tests that
previously lived here moved with the implementation.

The remaining seven canonical classes from
``docs/testing/mock_policy.md`` exercised below:

* TestPositive — happy-path bind + emit
* TestNegative — finalize before bind is a no-op
* TestBoundary — double bind returns same emitter (idempotent)
* TestInvariants — finalize closes emitter + deregisters
* TestRegressions — emit after finalize is a no-op (no crash)
* TestLogicSpecific — active stage drives ``failing_stage``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from ryotenkai_control.events import (
    ControlEventEmitter,
    EventEmitterRegistry,
    JournalReader,
)
from ryotenkai_control.pipeline.run_lifecycle_coordinator import (
    RunLifecycleCoordinator,
)
from ryotenkai_shared.pipeline_context import RunContext


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Each test starts with a fresh process-wide registry."""
    EventEmitterRegistry.reset_instance()
    yield
    EventEmitterRegistry.reset_instance()


@dataclass
class _ArtifactCall:
    local_path: str
    artifact_path: str | None
    run_id: str | None


@dataclass
class _FakeMlflowManager:
    """Records ``log_artifact`` calls (no longer driven by the events coord
    after Phase M7.2; retained for tests that simulate the wider manager
    presence)."""

    calls: list[_ArtifactCall] = field(default_factory=list)
    fail_next: int = 0

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
        run_id: str | None = None,
    ) -> bool:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("simulated mlflow failure")
        self.calls.append(
            _ArtifactCall(
                local_path=local_path,
                artifact_path=artifact_path,
                run_id=run_id,
            ),
        )
        return True

    def __getattr__(self, name: str) -> Any:  # pragma: no cover — defensive
        def _stub(*_a: Any, **_kw: Any) -> Any:
            raise AssertionError(
                f"FakeMlflowManager: unexpected method {name!r}",
            )
        return _stub


def _make_coordinator(
    *,
    run_ctx: RunContext | None = None,
    mlflow_manager: Any = None,  # noqa: ARG001 -- kept for back-compat callers
    shutdown_signal: str | None = None,
    pre_built_emitter: ControlEventEmitter | None = None,
    active_stage: str | None = None,
    mlflow_run_id: str | None = "mlflow-run-1",
) -> RunLifecycleCoordinator:
    ctx = run_ctx if run_ctx is not None else RunContext.create()
    return RunLifecycleCoordinator(
        run_ctx=ctx,
        algorithm_supplier=lambda: "sft",
        dataset_id_supplier=lambda: "alpha",
        model_id_supplier=lambda: "gpt2",
        mlflow_run_id_supplier=lambda: mlflow_run_id,
        active_stage_supplier=lambda: active_stage,
        shutdown_signal_supplier=lambda: shutdown_signal,
        pre_built_emitter=pre_built_emitter,
    )


# ---------------------------------------------------------------------------
# 1. POSITIVE — happy-path bind + emit
# ---------------------------------------------------------------------------


class TestPositive:
    def test_bind_run_directory_constructs_emitter_and_registers(
        self, tmp_path: Path,
    ) -> None:
        ctx = RunContext.create()
        coord = _make_coordinator(run_ctx=ctx)
        run_dir = tmp_path / "r"
        emitter = coord.bind_run_directory(run_dir)
        assert emitter is not None
        assert coord.emitter is emitter
        assert (run_dir / "events.jsonl").parent.exists()
        # Registered with the process-wide registry.
        assert EventEmitterRegistry.instance().get(ctx.name) is emitter
        coord.finalize(pipeline_success=True)

    def test_emit_run_started_writes_to_journal(self, tmp_path: Path) -> None:
        coord = _make_coordinator()
        run_dir = tmp_path / "r"
        coord.bind_run_directory(run_dir)
        coord.emit_run_started(config_hashes={"merged": "deadbeef"})
        coord.finalize(pipeline_success=True)
        envelopes = list(
            JournalReader(run_dir / "events.jsonl").iter_envelopes(),
        )
        # First envelope is RunStarted; finalize may have appended none.
        assert envelopes[0].kind == "ryotenkai.control.run.started"
        assert envelopes[0].payload.config_hash == "deadbeef"
        assert envelopes[0].payload.algorithm == "sft"
        assert envelopes[0].payload.model_id == "gpt2"
        assert envelopes[0].payload.dataset_id == "alpha"


# ---------------------------------------------------------------------------
# 2. NEGATIVE — finalize without bind is a no-op
# ---------------------------------------------------------------------------


class TestNegative:
    def test_finalize_before_bind_does_not_raise(self) -> None:
        coord = _make_coordinator()
        # No bind_run_directory call; emitter is None.
        coord.finalize(pipeline_success=True)
        assert coord.is_finalized
        assert coord.emitter is None

    def test_emit_run_started_with_no_emitter_is_silent(self) -> None:
        coord = _make_coordinator()
        # No emitter — emit should be a silent no-op.
        coord.emit_run_started(config_hashes={"merged": "h"})
        coord.emit_run_completed(duration_s=1.0, status="success")
        coord.emit_run_failed(RuntimeError("x"))
        coord.emit_run_cancelled(reason="r")
        assert coord.emitter is None


# ---------------------------------------------------------------------------
# 3. BOUNDARY — double bind returns the same emitter
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_bind_run_directory_is_idempotent(self, tmp_path: Path) -> None:
        coord = _make_coordinator()
        first = coord.bind_run_directory(tmp_path / "r")
        second = coord.bind_run_directory(tmp_path / "r")
        assert first is second
        coord.finalize(pipeline_success=True)

    def test_pre_built_emitter_is_registered(self, tmp_path: Path) -> None:
        ctx = RunContext.create()
        emitter = ControlEventEmitter.for_run(
            run_id=ctx.name, run_directory=tmp_path / "r",
        )
        coord = _make_coordinator(
            run_ctx=ctx, pre_built_emitter=emitter,
        )
        assert coord.emitter is emitter
        assert EventEmitterRegistry.instance().get(ctx.name) is emitter
        coord.finalize(pipeline_success=True)


# ---------------------------------------------------------------------------
# 4. INVARIANTS — finalize closes emitter + deregisters
# ---------------------------------------------------------------------------


class TestInvariants:
    def test_finalize_closes_emitter_and_deregisters(self, tmp_path: Path) -> None:
        ctx = RunContext.create()
        coord = _make_coordinator(run_ctx=ctx)
        coord.bind_run_directory(tmp_path / "r")
        assert EventEmitterRegistry.instance().get(ctx.name) is coord.emitter
        coord.finalize(pipeline_success=True)
        # Emitter is marked closed.
        assert coord.emitter is not None
        assert coord.emitter.is_closed
        # Registry slot is released.
        assert EventEmitterRegistry.instance().get(ctx.name) is None

    def test_finalize_is_idempotent(self, tmp_path: Path) -> None:
        coord = _make_coordinator()
        coord.bind_run_directory(tmp_path / "r")
        coord.finalize(pipeline_success=True)
        # Second call: must not crash and must not re-close.
        coord.finalize(pipeline_success=True)
        assert coord.is_finalized


# ---------------------------------------------------------------------------
# 5. REGRESSIONS — emit after finalize must not crash
# ---------------------------------------------------------------------------


class TestRegressions:
    def test_emit_after_finalize_is_silent(self, tmp_path: Path) -> None:
        coord = _make_coordinator()
        coord.bind_run_directory(tmp_path / "r")
        coord.finalize(pipeline_success=False)
        # All four emit_run_* helpers must short-circuit silently.
        coord.emit_run_started(config_hashes={"merged": "h"})
        coord.emit_run_completed(duration_s=1.0, status="success")
        coord.emit_run_failed(RuntimeError("x"))
        coord.emit_run_cancelled(reason="r")
        # No additional envelopes appended after close.
        envelopes = list(
            JournalReader(tmp_path / "r" / "events.jsonl").iter_envelopes(),
        )
        assert envelopes == []


# ---------------------------------------------------------------------------
# 6. LOGIC-SPECIFIC — active stage drives ``failing_stage``
# ---------------------------------------------------------------------------


class TestLogicSpecific:
    def test_active_stage_supplier_drives_failing_stage_field(
        self, tmp_path: Path,
    ) -> None:
        coord = _make_coordinator(active_stage="dataset_validator")
        coord.bind_run_directory(tmp_path / "r")
        coord.emit_run_failed(RuntimeError("oops"))
        coord.finalize(pipeline_success=False)
        envelopes = list(
            JournalReader(tmp_path / "r" / "events.jsonl").iter_envelopes(),
        )
        failed = [
            e for e in envelopes if e.kind == "ryotenkai.control.run.failed"
        ]
        assert len(failed) == 1
        assert failed[0].payload.failing_stage == "dataset_validator"

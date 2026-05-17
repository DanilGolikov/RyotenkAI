"""Integration smoke tests for Phase 3 emitter wiring.

These tests cover the *wiring* between :class:`PipelineOrchestrator`,
:class:`PipelineBootstrap`, and :class:`ControlEventEmitter`. They do
NOT exercise the orchestrator's full ``run()`` flow (stage construction,
MLflow, run-lock acquisition all require heavyweight scaffolding that
the existing ``test_orchestrator_boundary.py`` suite covers). Instead
they assert the bits Phase 3 introduced:

* When the orchestrator is constructed with ``run_directory=...``,
  the bootstrap pre-builds a :class:`ControlEventEmitter` whose journal
  lives at ``<run_directory>/events.jsonl``.
* When the orchestrator's run-level helpers (``_emit_run_*``) fire,
  they write a single, well-typed envelope each.
* The journal contents satisfy the Phase 3 invariants: monotonic
  offsets, length-prefixed framing, valid kinds.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ryotenkai_control.events import JournalReader
from ryotenkai_control.pipeline.bootstrap.startup_validator import StartupValidator
from ryotenkai_control.pipeline.execution import StageRegistry
from ryotenkai_control.pipeline.orchestrator import PipelineOrchestrator


def _build_mock_config(*, source_path: Path | None = None) -> SimpleNamespace:
    cfg = SimpleNamespace()
    if source_path is not None:
        cfg._source_path = source_path
    return cfg


def _construct_orchestrator(
    tmp_path: Path,
    *,
    run_directory: Path | None = None,
) -> PipelineOrchestrator:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    config = _build_mock_config(source_path=config_path)
    secrets = SimpleNamespace()
    with (
        patch("ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
        patch.object(StartupValidator, "validate"),
        patch("ryotenkai_community.preflight.run_preflight", return_value=MagicMock(ok=True)),
        patch.object(StageRegistry, "_build_stages", return_value=[]),
    ):
        return PipelineOrchestrator(config=config, run_directory=run_directory)


class TestEmitterWiring:
    def test_emitter_built_when_run_directory_supplied(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "test-run"
        orch = _construct_orchestrator(tmp_path, run_directory=run_dir)
        assert orch.emitter is not None
        assert orch.emitter.journal.path == run_dir / "events.jsonl"
        orch.emitter.close()

    def test_emitter_none_when_run_directory_absent(self, tmp_path: Path) -> None:
        orch = _construct_orchestrator(tmp_path)
        assert orch.emitter is None

    def test_ensure_event_emitter_builds_lazily(self, tmp_path: Path) -> None:
        orch = _construct_orchestrator(tmp_path)
        orch.run_directory = tmp_path / "late-runs" / "r"
        orch._ensure_event_emitter()
        assert orch.emitter is not None
        assert (tmp_path / "late-runs" / "r" / "events.jsonl").parent.exists()
        orch.emitter.close()

    def test_ensure_event_emitter_is_idempotent(self, tmp_path: Path) -> None:
        orch = _construct_orchestrator(tmp_path, run_directory=tmp_path / "r")
        first = orch.emitter
        orch._ensure_event_emitter()
        assert orch.emitter is first
        first.close()


class TestRunEventEmissions:
    def test_emit_run_started_writes_to_journal(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "r"
        orch = _construct_orchestrator(tmp_path, run_directory=run_dir)
        orch._emit_run_started(config_hashes={"merged": "deadbeef"})
        orch.emitter.close()

        envelopes = list(JournalReader(run_dir / "events.jsonl").iter_envelopes())
        assert len(envelopes) == 1
        assert envelopes[0].kind == "ryotenkai.control.run.started"
        assert envelopes[0].run_id == orch.run_ctx.name
        # config_hash propagates from the caller-supplied dict.
        assert envelopes[0].payload.config_hash == "deadbeef"

    def test_emit_run_completed_writes_to_journal(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "r"
        orch = _construct_orchestrator(tmp_path, run_directory=run_dir)
        orch._emit_run_completed(duration_s=12.5, status="success")
        orch.emitter.close()

        envelopes = list(JournalReader(run_dir / "events.jsonl").iter_envelopes())
        assert len(envelopes) == 1
        assert envelopes[0].kind == "ryotenkai.control.run.completed"
        assert envelopes[0].payload.duration_s == 12.5
        assert envelopes[0].payload.final_status == "success"

    def test_emit_run_failed_records_error_metadata(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "r"
        orch = _construct_orchestrator(tmp_path, run_directory=run_dir)
        exc = RuntimeError("boom")
        orch._emit_run_failed(exc)
        orch.emitter.close()

        envelopes = list(JournalReader(run_dir / "events.jsonl").iter_envelopes())
        assert envelopes[0].kind == "ryotenkai.control.run.failed"
        assert envelopes[0].payload.error_type == "RuntimeError"
        assert "boom" in envelopes[0].payload.message

    def test_emit_run_cancelled_records_reason(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "r"
        orch = _construct_orchestrator(tmp_path, run_directory=run_dir)
        orch._emit_run_cancelled(reason="user_interrupt")
        orch.emitter.close()

        envelopes = list(JournalReader(run_dir / "events.jsonl").iter_envelopes())
        assert envelopes[0].kind == "ryotenkai.control.run.cancelled"
        assert envelopes[0].payload.reason == "user_interrupt"

    def test_emissions_are_monotonic(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "r"
        orch = _construct_orchestrator(tmp_path, run_directory=run_dir)
        orch._emit_run_started(config_hashes={"merged": "h"})
        orch._emit_run_completed(duration_s=1.0, status="success")
        orch.emitter.close()

        envelopes = list(JournalReader(run_dir / "events.jsonl").iter_envelopes())
        assert [e.offset for e in envelopes] == [0, 1]

    def test_helpers_no_op_when_emitter_missing(self, tmp_path: Path) -> None:
        """Defensive: when emitter is None the helpers must not raise."""
        orch = _construct_orchestrator(tmp_path)
        assert orch.emitter is None
        # Each call should be a silent no-op.
        orch._emit_run_started(config_hashes={"merged": "h"})
        orch._emit_run_completed(duration_s=1.0, status="success")
        orch._emit_run_failed(RuntimeError("x"))
        orch._emit_run_cancelled(reason="r")


class TestDerivers:
    def test_derive_algorithm_defaults_to_sft(self, tmp_path: Path) -> None:
        orch = _construct_orchestrator(tmp_path)
        # Mock config has no `.training` — helper must fall back gracefully.
        assert orch._derive_algorithm() == "sft"

    def test_derive_dataset_id_defaults_to_unknown(self, tmp_path: Path) -> None:
        orch = _construct_orchestrator(tmp_path)
        assert orch._derive_dataset_id() == "unknown"

    def test_derive_algorithm_picks_first_strategy(self, tmp_path: Path) -> None:
        orch = _construct_orchestrator(tmp_path)
        orch.config = SimpleNamespace(
            training=SimpleNamespace(
                strategies=[SimpleNamespace(strategy_type="dpo")],
            ),
        )
        assert orch._derive_algorithm() == "dpo"

    def test_derive_algorithm_rejects_unknown_strategy(self, tmp_path: Path) -> None:
        orch = _construct_orchestrator(tmp_path)
        orch.config = SimpleNamespace(
            training=SimpleNamespace(
                strategies=[SimpleNamespace(strategy_type="weird")],
            ),
        )
        # Falls back to sft when the strategy isn't in the Algorithm union.
        assert orch._derive_algorithm() == "sft"

    def test_derive_dataset_id_picks_first_key(self, tmp_path: Path) -> None:
        orch = _construct_orchestrator(tmp_path)
        orch.config = SimpleNamespace(datasets={"alpha": object(), "beta": object()})
        assert orch._derive_dataset_id() == "alpha"

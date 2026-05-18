"""Tests for the orchestrator-glue helpers in :mod:`...lifecycle.orchestrator_glue`.

Phase M7.2 — the orchestrator's MLflow setup/preflight/teardown bodies
moved out of ``orchestrator.py`` into ``orchestrator_glue.py`` to keep
the orchestrator under its 1000-line architectural guardrail. These
tests exercise the helpers in isolation so the orchestrator's own
tests don't have to spin up the full ``PipelineOrchestrator`` to verify
the side-effects.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ryotenkai_control.pipeline.mlflow.lifecycle.orchestrator_glue import (
    derive_engine_kind,
    derive_provider_gpu,
    derive_provider_kind,
    resolve_journal_for_upload,
    stamp_state_tracking_uri,
)


# ---------------------------------------------------------------------------
# 1. POSITIVE — derive_engine_kind / derive_provider_kind / derive_provider_gpu
# ---------------------------------------------------------------------------


class TestDeriveEngineKind:
    def test_returns_first_strategy_type_lowercased(self) -> None:
        cfg = MagicMock()
        strategy = MagicMock()
        strategy.strategy_type = "SFT"
        cfg.training.strategies = [strategy]
        assert derive_engine_kind(cfg) == "sft"

    def test_returns_unknown_when_no_strategies(self) -> None:
        cfg = MagicMock()
        cfg.training.strategies = []
        assert derive_engine_kind(cfg) == "unknown"

    def test_returns_unknown_when_training_missing(self) -> None:
        cfg = MagicMock(spec=[])  # no attributes
        assert derive_engine_kind(cfg) == "unknown"


class TestDeriveProviderKind:
    def test_returns_provider_kind_lowercased(self) -> None:
        cfg = MagicMock()
        cfg.provider.kind = "RunPod"
        assert derive_provider_kind(cfg) == "runpod"

    def test_returns_unknown_when_no_provider(self) -> None:
        cfg = MagicMock()
        cfg.provider = None
        assert derive_provider_kind(cfg) == "unknown"


class TestDeriveProviderGpu:
    def test_returns_gpu_type_string(self) -> None:
        cfg = MagicMock()
        cfg.provider.gpu_type = "H100-80GB"
        assert derive_provider_gpu(cfg) == "H100-80GB"

    def test_returns_unknown_when_gpu_type_none(self) -> None:
        cfg = MagicMock()
        cfg.provider.gpu_type = None
        assert derive_provider_gpu(cfg) == "unknown"

    def test_returns_unknown_when_provider_none(self) -> None:
        cfg = MagicMock()
        cfg.provider = None
        assert derive_provider_gpu(cfg) == "unknown"


# ---------------------------------------------------------------------------
# 2. POSITIVE — stamp_state_tracking_uri
# ---------------------------------------------------------------------------


class TestStampStateTrackingUri:
    def test_stamps_runtime_uri_and_ca_bundle(self) -> None:
        manager = MagicMock()
        manager.get_runtime_tracking_uri.return_value = "http://localhost:5002"
        cfg = MagicMock()
        cfg.integrations.mlflow.ca_bundle_path = "/tmp/ca.pem"
        state = MagicMock()
        stamp_state_tracking_uri(manager=manager, config=cfg, state=state)
        assert state.mlflow_runtime_tracking_uri == "http://localhost:5002"
        assert state.mlflow_ca_bundle_path == "/tmp/ca.pem"

    def test_stamps_none_when_uri_empty(self) -> None:
        manager = MagicMock()
        manager.get_runtime_tracking_uri.return_value = ""
        cfg = MagicMock()
        cfg.integrations.mlflow.ca_bundle_path = None
        state = MagicMock()
        stamp_state_tracking_uri(manager=manager, config=cfg, state=state)
        assert state.mlflow_runtime_tracking_uri is None
        assert state.mlflow_ca_bundle_path is None


# ---------------------------------------------------------------------------
# 3. POSITIVE / NEGATIVE — resolve_journal_for_upload
# ---------------------------------------------------------------------------


class TestResolveJournalForUpload:
    def test_returns_none_when_emitter_none(self) -> None:
        assert resolve_journal_for_upload(None) == (None, None)

    def test_returns_path_and_sha256_when_journal_exists(
        self, tmp_path: Path,
    ) -> None:
        journal = tmp_path / "events.jsonl"
        contents = b'{"a":1}\n{"b":2}\n'
        journal.write_bytes(contents)
        emitter = MagicMock()
        emitter.journal.path = journal
        path, sha = resolve_journal_for_upload(emitter)
        assert path == journal
        assert sha == hashlib.sha256(contents).hexdigest()

    def test_returns_path_and_none_sha_when_journal_missing(
        self, tmp_path: Path,
    ) -> None:
        journal = tmp_path / "missing.jsonl"
        emitter = MagicMock()
        emitter.journal.path = journal
        path, sha = resolve_journal_for_upload(emitter)
        assert path == journal
        assert sha is None

    def test_returns_none_when_emitter_journal_raises(self) -> None:
        emitter = MagicMock()
        type(emitter).journal = property(
            fget=lambda _self: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert resolve_journal_for_upload(emitter) == (None, None)

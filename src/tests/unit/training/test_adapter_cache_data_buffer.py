"""
Tests for DataBuffer.mark_phase_skipped() and new adapter-cache fields on PhaseState.

Coverage matrix
───────────────
PhaseState                   new fields defaults / to_dict / from_dict (backward compat)
DataBuffer.mark_phase_skipped  positive / negative / boundary / callbacks / persistence
Invariants                   status always SKIPPED; completed_at always set
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.training.managers.data_buffer import (
    DataBuffer,
    DataBufferEventCallbacks,
    PhaseState,
    PhaseStatus,
)
from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

pytestmark = pytest.mark.unit


def _mk_strategies() -> list[StrategyPhaseConfig]:
    return [
        StrategyPhaseConfig(
            strategy_type="sft",
            dataset="sft_data",
            hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=2e-4),
        ),
        StrategyPhaseConfig(
            strategy_type="dpo",
            dataset="pref_data",
            hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=2e-4, beta=0.1),
        ),
    ]


def _mk_buffer(tmp_path: Path, *, with_callbacks: DataBufferEventCallbacks | None = None) -> DataBuffer:
    buf = DataBuffer(base_output_dir=tmp_path, base_model_path="base-model", callbacks=with_callbacks)
    buf.init_pipeline(_mk_strategies(), force=True)
    return buf


# ─────────────────────────────────────────────
# PhaseState — new fields
# ─────────────────────────────────────────────


class TestPhaseStateAdapterCacheFields:
    def test_positive_defaults_are_false_and_none(self) -> None:
        phase = PhaseState(phase_idx=0, strategy_type="sft")
        assert phase.adapter_cache_hit is False
        assert phase.adapter_cache_tag is None
        assert phase.adapter_cache_upload_error is None

    def test_positive_to_dict_includes_cache_fields(self) -> None:
        phase = PhaseState(phase_idx=0, strategy_type="sft")
        phase.adapter_cache_hit = True
        phase.adapter_cache_tag = "phase-0-sft-ds1234567890"
        d = phase.to_dict()
        assert d["adapter_cache_hit"] is True
        assert d["adapter_cache_tag"] == "phase-0-sft-ds1234567890"
        assert d["adapter_cache_upload_error"] is None

    def test_positive_to_dict_upload_error_preserved(self) -> None:
        phase = PhaseState(phase_idx=1, strategy_type="dpo")
        phase.adapter_cache_upload_error = "Connection refused"
        d = phase.to_dict()
        assert d["adapter_cache_upload_error"] == "Connection refused"

    def test_positive_from_dict_round_trip(self) -> None:
        phase = PhaseState(phase_idx=0, strategy_type="sft")
        phase.adapter_cache_hit = True
        phase.adapter_cache_tag = "phase-0-sft-dsABC1234567"
        d = phase.to_dict()
        restored = PhaseState.from_dict(d)
        assert restored.adapter_cache_hit is True
        assert restored.adapter_cache_tag == "phase-0-sft-dsABC1234567"

    def test_invariant_backward_compat_missing_cache_fields_in_json(self) -> None:
        """
        Old JSON snapshots without adapter_cache fields should deserialize to defaults.
        """
        old_dict = {
            "phase_idx": 0,
            "strategy_type": "sft",
            "status": "completed",
        }
        phase = PhaseState.from_dict(old_dict)
        assert phase.adapter_cache_hit is False
        assert phase.adapter_cache_tag is None
        assert phase.adapter_cache_upload_error is None

    def test_boundary_unknown_fields_in_json_ignored(self) -> None:
        """Fields that exist in future versions are silently ignored (forward compat)."""
        d = {
            "phase_idx": 0,
            "strategy_type": "sft",
            "status": "pending",
            "adapter_cache_hit": False,
            "adapter_cache_tag": None,
            "adapter_cache_upload_error": None,
            "future_field": "some_value",
        }
        phase = PhaseState.from_dict(d)
        assert phase.phase_idx == 0


# ─────────────────────────────────────────────
# DataBuffer.mark_phase_skipped
# ─────────────────────────────────────────────


class TestMarkPhaseSkipped:
    # ── Positive ──────────────────────────────

    def test_positive_sets_status_to_skipped(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        buf.mark_phase_skipped(0, reason="adapter_cache_hit: repo@tag")
        assert buf.state.phases[0].status == PhaseStatus.SKIPPED

    def test_positive_sets_completed_at(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        buf.mark_phase_skipped(0, reason="test")
        assert buf.state.phases[0].completed_at is not None

    def test_positive_state_is_saved(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        buf.mark_phase_skipped(0, reason="test")
        # Load raw JSON from disk to verify persistence
        state_file = tmp_path / DataBuffer.STATE_FILENAME
        raw = json.loads(state_file.read_text())
        phases = raw["phases"]
        assert any(p["status"] == "skipped" for p in phases)

    def test_positive_fires_on_phase_completed_callback(self, tmp_path: Path) -> None:
        cb = MagicMock()
        callbacks = DataBufferEventCallbacks(on_phase_completed=cb)
        buf = _mk_buffer(tmp_path, with_callbacks=callbacks)
        buf.mark_phase_skipped(0, reason="cache_hit")
        cb.assert_called_once_with(0, "sft", "skipped")

    def test_positive_with_checkpoint_path(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        buf.mark_phase_skipped(0, reason="cache_hit", checkpoint_path="/tmp/ckpt")
        assert buf.state.phases[0].checkpoint_path == "/tmp/ckpt"

    def test_positive_last_phase_can_be_skipped(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        last_idx = len(_mk_strategies()) - 1
        buf.mark_phase_skipped(last_idx, reason="cache_hit")
        assert buf.state.phases[last_idx].status == PhaseStatus.SKIPPED

    # ── Negative ──────────────────────────────

    def test_negative_out_of_range_positive_idx_raises(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        with pytest.raises(IndexError):
            buf.mark_phase_skipped(99, reason="out-of-range")

    def test_negative_out_of_range_negative_idx_raises(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        with pytest.raises(IndexError):
            buf.mark_phase_skipped(-1, reason="negative")

    # ── Boundary ──────────────────────────────

    def test_boundary_phase_0_is_valid_first_index(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        buf.mark_phase_skipped(0, reason="boundary-first")
        assert buf.state.phases[0].status == PhaseStatus.SKIPPED

    def test_boundary_checkpoint_path_none_does_not_override(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        original_checkpoint = buf.state.phases[0].checkpoint_path
        buf.mark_phase_skipped(0, reason="no-ckpt", checkpoint_path=None)
        assert buf.state.phases[0].checkpoint_path == original_checkpoint

    def test_boundary_empty_reason_is_accepted(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        buf.mark_phase_skipped(0, reason="")  # must not raise
        assert buf.state.phases[0].status == PhaseStatus.SKIPPED

    # ── Invariants ────────────────────────────

    def test_invariant_status_is_always_skipped(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        for i in range(len(_mk_strategies())):
            buf.mark_phase_skipped(i, reason=f"cache_{i}")
            assert buf.state.phases[i].status == PhaseStatus.SKIPPED

    def test_invariant_completed_at_always_set(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        buf.mark_phase_skipped(0, reason="test")
        assert buf.state.phases[0].completed_at is not None

    def test_invariant_other_phases_not_affected(self, tmp_path: Path) -> None:
        buf = _mk_buffer(tmp_path)
        buf.mark_phase_skipped(0, reason="only-first")
        assert buf.state.phases[1].status == PhaseStatus.PENDING

    # ── Regression ────────────────────────────

    def test_regression_skipped_status_value_persists_in_json(self, tmp_path: Path) -> None:
        """Status 'skipped' must survive a serialization / deserialization round-trip."""
        buf = _mk_buffer(tmp_path)
        buf.mark_phase_skipped(0, reason="cache_hit")

        state_file = tmp_path / DataBuffer.STATE_FILENAME
        raw = json.loads(state_file.read_text())
        # Find phase 0 in the phases list
        phase_raw = next(p for p in raw["phases"] if p["phase_idx"] == 0)
        assert phase_raw["status"] == "skipped"

    def test_regression_adapter_cache_hit_field_persists(self, tmp_path: Path) -> None:
        """adapter_cache_hit=True must survive a to_dict round-trip."""
        buf = _mk_buffer(tmp_path)
        buf.state.phases[0].adapter_cache_hit = True
        buf.state.phases[0].adapter_cache_tag = "phase-0-sft-dsXXXXXXXXXX"
        buf.mark_phase_skipped(0, reason="cache_hit")  # triggers save

        state_file = tmp_path / DataBuffer.STATE_FILENAME
        raw = json.loads(state_file.read_text())
        phase_raw = next(p for p in raw["phases"] if p["phase_idx"] == 0)
        assert phase_raw["adapter_cache_hit"] is True
        assert phase_raw["adapter_cache_tag"] == "phase-0-sft-dsXXXXXXXXXX"

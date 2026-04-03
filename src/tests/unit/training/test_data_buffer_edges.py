from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.training.managers import data_buffer as db
from src.training.managers.data_buffer import (
    DataBuffer,
    DataBufferEventCallbacks,
    FaultSimulator,
    PhaseState,
    PhaseStatus,
    PipelineState,
    list_available_runs,
)
from src.utils.config import PhaseHyperparametersConfig, StrategyPhaseConfig

pytestmark = pytest.mark.unit


def _mk_strategies(*, epochs: int | None = None, learning_rate: float | None = None) -> list[StrategyPhaseConfig]:
    return [
        StrategyPhaseConfig(
            strategy_type="sft",
            dataset="sft_data",
            hyperparams=PhaseHyperparametersConfig(epochs=epochs, learning_rate=learning_rate),
        ),
        StrategyPhaseConfig(
            strategy_type="dpo",
            dataset="pref_data",
            hyperparams=PhaseHyperparametersConfig(epochs=1, learning_rate=2e-4, beta=0.1),
        ),
    ]


class TestSanitizeAndCompatHelpers:
    def test_sanitize_metrics_recurses_and_handles_arrays_lists_and_fallback(self) -> None:
        class Scalar:
            def __init__(self, v: int):
                self._v = v

            def item(self) -> int:
                return self._v

        class Arr:
            def tolist(self) -> list[int]:
                return [1, 2]

        out = db._sanitize_metrics(
            {
                "scalar": Scalar(5),
                "arr": Arr(),
                "nested": {"x": Scalar(1)},
                "lst": [Scalar(2), "ok"],
                "fallback": SimpleNamespace(a=1),
            }
        )
        assert out["scalar"] == 5
        assert out["arr"] == [1, 2]
        assert out["nested"]["x"] == 1
        assert out["lst"] == [2, "ok"]
        assert isinstance(out["fallback"], str)

    def test_phase_state_from_dict_unknown_fields_and_unknown_status_fallback(self) -> None:
        phase = PhaseState.from_dict({"phase_idx": 0, "strategy_type": "sft", "status": "weird", "new": 1})
        assert phase.status == PhaseStatus.PENDING

    def test_pipeline_state_from_dict_phases_none_and_unknown_fields(self) -> None:
        state = PipelineState.from_dict(
            {
                "run_id": "r",
                "base_output_dir": "/tmp",
                "base_model_path": "m",
                "total_phases": 0,
                "phases": None,
                "completed_at": "2025-01-01T00:00:00",
                "unknown": 123,
            }
        )
        assert state.phases == []
        assert state.completed_at is not None

    def test_pipeline_state_failed_phases_and_progress_percent_zero_total(self) -> None:
        st = PipelineState(run_id="r", base_output_dir="/tmp", base_model_path="m", total_phases=0, phases=[])
        assert st.progress_percent == 0.0

        p0 = PhaseState(phase_idx=0, strategy_type="sft")
        p0.status = PhaseStatus.FAILED
        st = PipelineState(run_id="r", base_output_dir="/tmp", base_model_path="m", total_phases=1, phases=[p0])
        assert st.failed_phases == [p0]

    def test_extract_checkpoint_step_unknown_format_returns_minus_one(self, tmp_path: Path) -> None:
        assert db._extract_checkpoint_step(tmp_path / "checkpoint-bad") == -1


class TestDataBufferInitAndPathsEdges:
    def test_state_property_raises_when_not_initialized(self, tmp_path: Path) -> None:
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m")
        with pytest.raises(RuntimeError, match="Call init_pipeline"):
            _ = buffer.state

    def test_init_pipeline_rejects_none_strategy(self, tmp_path: Path) -> None:
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m")
        with pytest.raises(ValueError, match="cannot contain None"):
            buffer.init_pipeline([None])  # type: ignore[list-item]

    def test_init_pipeline_epochs_default_1_and_learning_rate_from_global(self, tmp_path: Path) -> None:
        cb = DataBufferEventCallbacks(on_pipeline_initialized=MagicMock())
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m", callbacks=cb)

        strategies = _mk_strategies(epochs=None, learning_rate=None)
        global_hp = PhaseHyperparametersConfig(epochs=None, learning_rate=1e-4)
        buffer.init_pipeline(strategies, global_hyperparams=global_hp, force=True)

        assert buffer.state.phases[0].epochs == 1
        assert buffer.state.phases[0].learning_rate == 1e-4
        cb.on_pipeline_initialized.assert_called_once()

    def test_load_existing_missing_corrupt_and_invalid_format(self, tmp_path: Path) -> None:
        # NOTE: load_existing() loads state from a specific output directory (no nested run_id folder).
        missing_out = tmp_path / "missing_output"
        with pytest.raises(FileNotFoundError):
            DataBuffer.load_existing(base_output_dir=missing_out, run_id="missing")

        # Corrupt JSON
        corrupt_out = tmp_path / "corrupt_output"
        corrupt_out.mkdir()
        (corrupt_out / DataBuffer.STATE_FILENAME).write_text("{not-json", encoding="utf-8")
        with pytest.raises(ValueError, match="Corrupted state file"):
            DataBuffer.load_existing(base_output_dir=corrupt_out, run_id="run_bad")

        # Invalid format (not dict)
        list_out = tmp_path / "list_output"
        list_out.mkdir()
        (list_out / DataBuffer.STATE_FILENAME).write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError, match="Expected dict"):
            DataBuffer.load_existing(base_output_dir=list_out, run_id="run_list")

    def test_get_phase_output_dir_fallback_when_output_dir_missing(self, tmp_path: Path) -> None:
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m")
        buffer.init_pipeline(_mk_strategies(), force=True)

        buffer.state.phases[0].output_dir = None
        out = buffer.get_phase_output_dir(0)
        assert "phase_0_sft" in out
        assert Path(out).exists()

    def test_get_model_path_for_phase_fallbacks_and_simulation(self, tmp_path: Path) -> None:
        strategies = _mk_strategies()

        # Simulation: missing checkpoint -> base model
        sim = FaultSimulator(missing_checkpoint=True)
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m", _fault_simulator=sim)
        buffer.init_pipeline(strategies, force=True)
        assert buffer.get_model_path_for_phase(1) == "m"

        # checkpoint-final fallback when no explicit checkpoint_path
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m")
        buffer.init_pipeline(strategies, force=True)
        prev_dir = Path(buffer.get_phase_output_dir(0))
        (prev_dir / "checkpoint-final").mkdir(parents=True)
        buffer.state.phases[0].checkpoint_path = None
        assert "checkpoint-final" in buffer.get_model_path_for_phase(1)

        # latest checkpoint-N fallback (and include invalid checkpoint name -> _extract_checkpoint_step ValueError path)
        (prev_dir / "checkpoint-bad").mkdir()
        (prev_dir / "checkpoint-50").mkdir()
        (prev_dir / "checkpoint-100").mkdir()
        (prev_dir / "checkpoint-final").rmdir()
        model_path = buffer.get_model_path_for_phase(1)
        assert model_path.endswith("checkpoint-100")

        # no checkpoints -> warn + base model
        for p in prev_dir.glob("checkpoint-*"):
            if p.is_dir():
                for sub in p.glob("*"):
                    if sub.is_file():
                        sub.unlink()
                p.rmdir()
        assert buffer.get_model_path_for_phase(1) == "m"

        with pytest.raises(IndexError):
            buffer.get_model_path_for_phase(999)


class TestPhaseCallbacksAndStateIOEdges:
    def test_phase_callbacks_and_invalid_indices(self, tmp_path: Path) -> None:
        cb = DataBufferEventCallbacks(
            on_phase_started=MagicMock(),
            on_phase_completed=MagicMock(),
            on_state_saved=MagicMock(),
            on_checkpoint_cleanup=MagicMock(),
        )
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m", callbacks=cb)
        buffer.init_pipeline(_mk_strategies(), force=True)

        buffer.mark_phase_started(0)
        cb.on_phase_started.assert_called_once()

        buffer.mark_phase_completed(0)
        buffer.mark_phase_failed(1, "boom")
        buffer.mark_phase_interrupted(0, "sigint")
        assert cb.on_phase_completed.call_count >= 3

        with pytest.raises(IndexError):
            buffer.mark_phase_started(999)
        with pytest.raises(IndexError):
            buffer.mark_phase_completed(999)
        with pytest.raises(IndexError):
            buffer.mark_phase_failed(999, "x")
        with pytest.raises(IndexError):
            buffer.mark_phase_interrupted(999, "x")

        # init_pipeline/save_state should have emitted on_state_saved at least once
        assert cb.on_state_saved.call_count >= 1

    def test_save_state_skips_when_state_none(self, tmp_path: Path) -> None:
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m")
        buffer.save_state()  # should no-op

    def test_save_state_cleans_tmp_file_on_error(self, tmp_path: Path) -> None:
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m")
        buffer.init_pipeline(_mk_strategies(), force=True)

        with patch("src.training.managers.data_buffer.json.dump", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                buffer.save_state()

        assert not any(buffer.run_dir.glob(".pipeline_state_*.tmp"))

    def test_save_state_is_atomic_and_does_not_corrupt_existing_state_file_on_error(self, tmp_path: Path) -> None:
        """
        Production contract for atomic save:
        if save_state fails while writing the tmp file, existing pipeline_state.json
        must stay valid and unchanged (resume still possible).
        """
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m")
        buffer.init_pipeline(_mk_strategies(), force=True)

        before_text = buffer.state_file.read_text(encoding="utf-8")
        before_obj = json.loads(before_text)
        assert isinstance(before_obj, dict)

        # Mutate in-memory state so a successful save would change the file.
        buffer.state.current_phase = 123

        with patch("src.training.managers.data_buffer.json.dump", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                buffer.save_state()

        after_text = buffer.state_file.read_text(encoding="utf-8")
        after_obj = json.loads(after_text)

        assert after_text == before_text
        assert after_obj == before_obj
        assert buffer.state_file.exists()

    def test_load_state_ignores_leftover_tmp_files_from_crash(self, tmp_path: Path) -> None:
        """
        Crash simulation: atomic-save tmp file left in run_dir.
        load_state() must still work because it only reads pipeline_state.json.
        """
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m")
        buffer.init_pipeline(_mk_strategies(), force=True)

        # Leftover tmp file (e.g. process died before cleanup).
        (buffer.run_dir / ".pipeline_state_leftover.tmp").write_text('{"junk":true', encoding="utf-8")

        assert buffer.load_state() is True

    def test_load_state_slow_io_simulation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m")
        buffer.init_pipeline(_mk_strategies(), force=True)

        buffer._fault_simulator = FaultSimulator(slow_io_delay_ms=10)
        monkeypatch.setattr("src.training.managers.data_buffer.time.sleep", lambda _s: None)

        assert buffer.load_state() is True

    def test_load_state_returns_false_on_corrupted_json(self, tmp_path: Path) -> None:
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m", run_id="rid_corrupt_json")
        buffer.run_dir.mkdir(parents=True, exist_ok=True)
        buffer.state_file.write_text("{not-json", encoding="utf-8")

        assert buffer.load_state() is False
        assert buffer.is_initialized is False

    def test_load_state_returns_false_on_valid_json_but_not_dict(self, tmp_path: Path) -> None:
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m", run_id="rid_not_dict")
        buffer.run_dir.mkdir(parents=True, exist_ok=True)
        buffer.state_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        assert buffer.load_state() is False
        assert buffer.is_initialized is False

    def test_get_resume_phase_interrupted_and_state_none(self, tmp_path: Path) -> None:
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m")
        assert buffer.get_resume_phase() is None

        buffer.init_pipeline(_mk_strategies(), force=True)
        buffer.mark_phase_started(0)
        buffer.mark_phase_interrupted(0, "sigint", checkpoint_path="/tmp/ckpt")
        assert buffer.get_resume_phase() == 0

    def test_get_resume_checkpoint_edges(self, tmp_path: Path) -> None:
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m")
        buffer.init_pipeline(_mk_strategies(), force=True)

        assert buffer.get_resume_checkpoint(-1) is None
        assert buffer.get_resume_checkpoint(999) is None

        ckpt = Path(buffer.get_phase_output_dir(0)) / "checkpoint-final"
        ckpt.mkdir(parents=True)
        buffer.state.phases[0].checkpoint_path = str(ckpt)
        assert buffer.get_resume_checkpoint(0) == str(ckpt)

        buffer.state.phases[0].checkpoint_path = "/definitely/missing"
        ckpt.rmdir()
        assert buffer.get_resume_checkpoint(0) is None

    def test_cleanup_old_checkpoints_skip_no_state_and_permission_errors_and_callback(self, tmp_path: Path) -> None:
        cb = DataBufferEventCallbacks(on_checkpoint_cleanup=MagicMock())
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m", callbacks=cb)
        assert buffer.cleanup_old_checkpoints() == []

        buffer.init_pipeline(_mk_strategies(), force=True)
        phase_dir = Path(buffer.get_phase_output_dir(0))
        (phase_dir / "checkpoint-0").mkdir()
        (phase_dir / "checkpoint-100").mkdir()
        (phase_dir / "checkpoint-final").mkdir()

        # PermissionError + OSError are swallowed
        with patch(
            "src.training.managers.data_buffer.shutil.rmtree",
            side_effect=[PermissionError("no"), OSError("nope")],
        ):
            _ = buffer.cleanup_old_checkpoints(keep_last=0, dry_run=False)

        # Happy path triggers callback when something deleted
        deleted = buffer.cleanup_old_checkpoints(keep_last=0, dry_run=False)
        assert deleted
        cb.on_checkpoint_cleanup.assert_called_once()

    def test_get_summary_and_repr_not_initialized(self, tmp_path: Path) -> None:
        buffer = DataBuffer(base_output_dir=tmp_path, base_model_path="m")
        assert buffer.get_summary()["status"] == "not_initialized"
        assert "not_initialized" in repr(buffer)


class TestListAvailableRunsEdges:
    def test_list_available_runs_edges(self, tmp_path: Path) -> None:
        base = tmp_path / "runs"
        assert list_available_runs(base) == []

        base.mkdir()
        (base / "not_a_dir.txt").write_text("x", encoding="utf-8")

        # Run with unreadable/corrupt state -> should be skipped with warning
        run_dir = base / "run1"
        run_dir.mkdir()
        (run_dir / DataBuffer.STATE_FILENAME).write_text("{bad", encoding="utf-8")

        runs = list_available_runs(base)
        assert runs == []

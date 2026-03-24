from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.training.orchestrator.resume_manager import ResumeManager


def _mk_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.training.hyperparams = MagicMock()
    cfg.model.name = "base-model"
    return cfg


class TestSetupBuffer:
    def test_setup_buffer_fresh_initializes_pipeline_force_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        rm = ResumeManager(cfg)

        buf = MagicMock()

        def mk_buffer(**kwargs):
            # Ensure args are wired correctly
            assert kwargs["base_output_dir"] == "output"
            assert kwargs["base_model_path"] == cfg.model.name
            assert kwargs["run_id"] == "rid"
            return buf

        monkeypatch.setattr("src.training.orchestrator.resume_manager.DataBuffer", mk_buffer)

        strategies = [MagicMock()]
        out_buf, start_phase, should_load = rm.setup_buffer(strategies, resume=False, run_id="rid")

        assert out_buf is buf
        assert start_phase == 0
        assert should_load is False
        buf.init_pipeline.assert_called_once()
        kwargs = buf.init_pipeline.call_args.kwargs
        assert kwargs["global_hyperparams"] is cfg.training.hyperparams
        assert kwargs["force"] is True

    def test_setup_buffer_resume_state_loaded_and_resume_phase(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        rm = ResumeManager(cfg)

        buf = MagicMock()
        buf.load_state.return_value = True
        buf.get_resume_phase.return_value = 2
        monkeypatch.setattr("src.training.orchestrator.resume_manager.DataBuffer", lambda **kw: buf)

        strategies = [MagicMock()]
        out_buf, start_phase, should_load = rm.setup_buffer(strategies, resume=True, run_id="rid")
        assert out_buf is buf
        assert start_phase == 2
        assert should_load is True
        buf.init_pipeline.assert_not_called()

    def test_setup_buffer_resume_phase_zero_does_not_require_checkpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        rm = ResumeManager(cfg)

        buf = MagicMock()
        buf.load_state.return_value = True
        buf.get_resume_phase.return_value = 0
        monkeypatch.setattr("src.training.orchestrator.resume_manager.DataBuffer", lambda **kw: buf)

        out_buf, start_phase, should_load = rm.setup_buffer([MagicMock()], resume=True, run_id="rid")
        assert out_buf is buf
        assert start_phase == 0
        assert should_load is False

    def test_setup_buffer_resume_all_complete_does_not_init_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        rm = ResumeManager(cfg)

        buf = MagicMock()
        buf.load_state.return_value = True
        buf.get_resume_phase.return_value = None
        monkeypatch.setattr("src.training.orchestrator.resume_manager.DataBuffer", lambda **kw: buf)

        out_buf, start_phase, should_load = rm.setup_buffer([MagicMock()], resume=True, run_id="rid")
        assert out_buf is buf
        assert start_phase == 0
        assert should_load is False
        buf.init_pipeline.assert_not_called()

    def test_setup_buffer_resume_state_missing_initializes_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _mk_cfg()
        rm = ResumeManager(cfg)

        buf = MagicMock()
        buf.load_state.return_value = False
        monkeypatch.setattr("src.training.orchestrator.resume_manager.DataBuffer", lambda **kw: buf)

        strategies = [MagicMock()]
        out_buf, start_phase, should_load = rm.setup_buffer(strategies, resume=True, run_id="rid")
        assert out_buf is buf
        assert start_phase == 0
        assert should_load is False
        buf.init_pipeline.assert_called_once_with(strategies, global_hyperparams=cfg.training.hyperparams, force=True)


class TestCheckpointPath:
    def test_phase_zero_returns_none(self) -> None:
        rm = ResumeManager(_mk_cfg())
        assert rm.get_checkpoint_path_for_phase(MagicMock(), phase_idx=0) is None

    def test_nonzero_delegates_to_buffer(self) -> None:
        rm = ResumeManager(_mk_cfg())
        buf = MagicMock()
        buf.get_model_path_for_phase.return_value = "/ckpt"
        assert rm.get_checkpoint_path_for_phase(buf, phase_idx=2) == "/ckpt"
        buf.get_model_path_for_phase.assert_called_once_with(2)


class TestLoadModelFromCheckpoint:
    def test_checkpoint_not_found_returns_err(self, tmp_path: Path) -> None:
        rm = ResumeManager(_mk_cfg())
        base_model = MagicMock()
        res = rm.load_model_from_checkpoint(str(tmp_path / "missing"), base_model)
        assert res.is_failure()

    def test_peft_adapter_checkpoint_uses_peft(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")

        base_model = MagicMock()
        peft_model = object()

        class _PeftModel:
            @staticmethod
            def from_pretrained(model, path):
                assert model is base_model
                assert path == str(ckpt)
                return peft_model

        monkeypatch.setitem(__import__("sys").modules, "peft", SimpleNamespace(PeftModel=_PeftModel))

        rm = ResumeManager(_mk_cfg())
        res = rm.load_model_from_checkpoint(str(ckpt), base_model)
        assert res.is_success()
        assert res.unwrap() is peft_model

    def test_full_model_checkpoint_returns_base_model(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        base_model = MagicMock()
        rm = ResumeManager(_mk_cfg())
        res = rm.load_model_from_checkpoint(str(ckpt), base_model)
        assert res.is_success()
        assert res.unwrap() is base_model

    def test_exception_returns_err(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")

        base_model = MagicMock()

        class _PeftModel:
            @staticmethod
            def from_pretrained(model, path):
                raise RuntimeError("boom")

        monkeypatch.setitem(__import__("sys").modules, "peft", SimpleNamespace(PeftModel=_PeftModel))

        rm = ResumeManager(_mk_cfg())
        res = rm.load_model_from_checkpoint(str(ckpt), base_model)
        assert res.is_failure()
        assert "Failed to load model from checkpoint" in str(res.unwrap_err())


class TestResumeSignals:
    def test_can_resume_false_when_buffer_none(self) -> None:
        rm = ResumeManager(_mk_cfg())
        assert rm.can_resume(None) is False

    def test_can_resume_delegates(self) -> None:
        rm = ResumeManager(_mk_cfg())
        buf = MagicMock()
        buf.can_resume.return_value = True
        assert rm.can_resume(buf) is True

    def test_is_all_complete_true_when_resume_phase_none(self) -> None:
        rm = ResumeManager(_mk_cfg())
        buf = MagicMock()
        buf.get_resume_phase.return_value = None
        assert rm.is_all_complete(buf) is True

    def test_was_interrupted_false_when_state_none(self) -> None:
        rm = ResumeManager(_mk_cfg())
        buf = MagicMock()
        buf._state = None
        assert rm.was_interrupted(buf) is False

    def test_was_interrupted_true_when_global_status_interrupted(self) -> None:
        rm = ResumeManager(_mk_cfg())
        buf = MagicMock()
        buf._state = object()
        buf.state.status = "interrupted"
        buf.state.phases = []
        assert rm.was_interrupted(buf) is True

    def test_was_interrupted_true_when_phase_interrupted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rm = ResumeManager(_mk_cfg())
        buf = MagicMock()
        buf._state = object()
        buf.state.status = "running"

        # Use the real PhaseStatus enum from module
        from src.training.managers.data_buffer import PhaseStatus

        buf.state.phases = [SimpleNamespace(status=PhaseStatus.INTERRUPTED, phase_idx=1)]
        assert rm.was_interrupted(buf) is True

    def test_get_interrupt_info_returns_details(self) -> None:
        rm = ResumeManager(_mk_cfg())
        buf = MagicMock()

        from src.training.managers.data_buffer import PhaseStatus

        phase = SimpleNamespace(
            status=PhaseStatus.INTERRUPTED,
            phase_idx=1,
            strategy_type="sft",
            error_message="x",
            checkpoint_path="/ckpt",
            completed_at=None,
        )
        buf.state.phases = [phase]

        info = rm.get_interrupt_info(buf)
        assert info is not None
        assert info["phase_idx"] == 1
        assert info["checkpoint_path"] == "/ckpt"

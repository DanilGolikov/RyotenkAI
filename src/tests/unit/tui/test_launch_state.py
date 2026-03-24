from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.tui.launch import RestartPointOption
from src.tui.launch_state import build_submittable_launch_request, prepare_launch_mode_state


def test_prepare_launch_mode_state_for_new_run_is_pure_ui_state() -> None:
    state = prepare_launch_mode_state("new_run", None, None)

    assert state.mode_title == "new run"
    assert "new logical run directory" in state.info_markup
    assert state.restart_points == ()
    assert state.restart_options == ()
    assert state.restart_selector_disabled is True


def test_prepare_launch_mode_state_for_resume_uses_resume_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_config = tmp_path / "config.yaml"
    monkeypatch.setattr(
        "src.tui.launch_state.validate_resume_run",
        lambda run_dir, config_path=None: (resolved_config, "Inference Deployer"),
    )

    state = prepare_launch_mode_state("resume", tmp_path / "runs" / "existing", None)

    assert state.mode_title == "resume"
    assert "Inference Deployer" in state.info_markup
    assert state.resolved_config_path == resolved_config
    assert state.restart_options == ()


def test_prepare_launch_mode_state_for_resume_hides_absolute_pipeline_state_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaking_path = tmp_path / "runs" / "empty" / "pipeline_state.json"
    monkeypatch.setattr(
        "src.tui.launch_state.validate_resume_run",
        lambda run_dir, config_path=None: (_ for _ in ()).throw(ValueError(f"Missing pipeline state: {leaking_path}")),
    )

    with pytest.raises(ValueError, match="Selected run does not contain pipeline_state.json yet."):
        prepare_launch_mode_state("resume", tmp_path / "runs" / "empty", None)


def test_prepare_launch_mode_state_for_restart_without_run_dir_shows_warning() -> None:
    state = prepare_launch_mode_state("restart", None, None)

    assert state.mode_title == "restart"
    assert "requires an existing run directory" in state.info_markup
    assert "[yellow]" in state.info_markup


def test_prepare_launch_mode_state_for_restart_builds_restart_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_config = tmp_path / "config.yaml"
    monkeypatch.setattr(
        "src.tui.launch_state.load_restart_point_options",
        lambda run_dir, config_path=None: (
            resolved_config,
            [
                RestartPointOption(
                    stage="Inference Deployer",
                    available=True,
                    mode="fresh_or_resume",
                    reason="restart_allowed",
                ),
                RestartPointOption(
                    stage="Model Evaluator",
                    available=False,
                    mode="live_runtime_only",
                    reason="missing_inference_outputs",
                ),
            ],
        ),
    )

    state = prepare_launch_mode_state("restart", tmp_path / "runs" / "existing", None)

    assert state.mode_title == "restart | 1 available"
    assert state.resolved_config_path == resolved_config
    assert state.restart_options == (("Inference Deployer", "Inference Deployer"),)
    assert state.selected_restart_stage == "Inference Deployer"
    assert "1 available" in state.info_markup
    assert "1 blocked" in state.info_markup


def test_build_submittable_launch_request_resolves_resume_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved_config = tmp_path / "config.yaml"
    monkeypatch.setattr(
        "src.tui.launch_state.validate_resume_run",
        lambda run_dir, config_path=None: (resolved_config, "Inference Deployer"),
    )

    request, prepared_config = build_submittable_launch_request(
        mode="resume",
        run_dir=tmp_path / "runs" / "existing",
        config_path=None,
        restart_stage=None,
        log_level="DEBUG",
    )

    assert request.mode == "resume"
    assert request.config_path == resolved_config.resolve()
    assert request.log_level == "DEBUG"
    assert prepared_config == resolved_config


def test_build_submittable_launch_request_blocks_fresh_without_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MagicMock()
    store.exists.return_value = False
    monkeypatch.setattr("src.tui.launch_state.PipelineStateStore", lambda _run_dir: store)

    with pytest.raises(ValueError, match="Fresh attempt requires an existing run directory"):
        build_submittable_launch_request(
            mode="fresh",
            run_dir=tmp_path / "runs" / "missing",
            config_path=tmp_path / "config.yaml",
            restart_stage=None,
        )

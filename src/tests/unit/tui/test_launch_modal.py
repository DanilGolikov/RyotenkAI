from __future__ import annotations

from pathlib import Path

from src.tui.screens.launch_modal import (
    LaunchModal,
    _LOG_LEVEL_OPTIONS,
    _restart_section_visible,
    _run_dir_value_for_mode,
)


def test_run_dir_value_for_mode_keeps_generated_path_only_for_new_run() -> None:
    generated_run_dir = Path("runs/run_20260324_144527_dt55g")

    assert (
        _run_dir_value_for_mode("new_run", context_run_dir=None, new_run_dir=generated_run_dir)
        == "runs/run_20260324_144527_dt55g"
    )
    assert _run_dir_value_for_mode("resume", context_run_dir=None, new_run_dir=generated_run_dir) == ""
    assert _run_dir_value_for_mode("fresh", context_run_dir=None, new_run_dir=generated_run_dir) == ""
    assert _run_dir_value_for_mode("restart", context_run_dir=None, new_run_dir=generated_run_dir) == ""


def test_run_dir_value_for_mode_uses_context_run_for_existing_run_actions() -> None:
    context_run_dir = Path("runs/run_existing")
    generated_run_dir = Path("runs/run_generated")

    assert (
        _run_dir_value_for_mode("resume", context_run_dir=context_run_dir, new_run_dir=generated_run_dir)
        == "runs/run_existing"
    )


def test_restart_section_visible_only_for_restart_mode() -> None:
    assert _restart_section_visible("restart") is True
    assert _restart_section_visible("new_run") is False
    assert _restart_section_visible("fresh") is False
    assert _restart_section_visible("resume") is False


def test_launch_modal_exposes_structured_config_binding() -> None:
    bindings = {binding.action: binding.key for binding in LaunchModal.BINDINGS}

    assert bindings["browse_config"] == "ctrl+b"


def test_launch_modal_exposes_log_level_options() -> None:
    assert _LOG_LEVEL_OPTIONS == [("INFO", "INFO"), ("DEBUG", "DEBUG")]


def test_launch_modal_body_is_scrollable_for_taller_modes() -> None:
    assert "max-height: 80vh;" in LaunchModal.DEFAULT_CSS
    assert "overflow-y: auto;" in LaunchModal.DEFAULT_CSS

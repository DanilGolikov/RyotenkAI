from __future__ import annotations

from pathlib import Path

from src.pipeline.launch_queries import (
    RestartPointOption,
    load_restart_point_options as query_load_restart_point_options,
    pick_default_launch_mode as query_pick_default_launch_mode,
    resolve_config_path_for_run as query_resolve_config_path_for_run,
    validate_resume_run as query_validate_resume_run,
)


def resolve_config_path_for_run(run_dir: Path, config_path: Path | None = None) -> Path:
    return query_resolve_config_path_for_run(run_dir, config_path)


def load_restart_point_options(run_dir: Path, config_path: Path | None = None) -> tuple[Path, list[RestartPointOption]]:
    return query_load_restart_point_options(run_dir, config_path)


def pick_default_launch_mode(run_dir: Path) -> str:
    return query_pick_default_launch_mode(run_dir)


def validate_resume_run(run_dir: Path, config_path: Path | None = None) -> tuple[Path, str]:
    return query_validate_resume_run(run_dir, config_path)

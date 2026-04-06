from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.tui.adapters.state import run_state_exists
from src.tui.launch import (
    MODE_FRESH,
    MODE_NEW_RUN,
    MODE_RESTART,
    MODE_RESUME,
    LaunchRequest,
    RestartPointOption,
    load_restart_point_options,
    resolve_config_path_for_run,
    validate_resume_run,
)

if TYPE_CHECKING:
    from pathlib import Path

_MODE_LABELS = {
    MODE_NEW_RUN: "new run",
    MODE_FRESH: "fresh attempt",
    MODE_RESUME: "resume",
    MODE_RESTART: "start from stage",
}

_RESTART_MODE_LABELS = {
    "fresh_only": "fresh only",
    "fresh_or_resume": "fresh or resume",
    "reconnect_only": "reconnect only",
    "live_runtime_only": "live runtime only",
}

_RESTART_REASON_LABELS = {
    "restart_allowed": "restart allowed",
    "missing_gpu_deployer_outputs": "missing GPU deployer outputs",
    "missing_model_retriever_outputs": "missing model retriever outputs",
    "missing_inference_outputs": "missing inference outputs",
    "inference_runtime_not_healthy": "inference runtime not healthy",
    "training_critical_config_changed": "training-critical config changed",
    "late_stage_config_changed": "late-stage config changed",
}


def _user_facing_prepare_error(message: str) -> str:
    if "pipeline_state.json" in message:
        return "Selected run does not contain pipeline_state.json yet."
    return message


@dataclass(frozen=True, slots=True)
class LaunchModeState:
    mode_title: str
    info_markup: str
    restart_points: tuple[RestartPointOption, ...] = ()
    restart_options: tuple[tuple[str, str], ...] = ()
    selected_restart_stage: str | None = None
    resolved_config_path: Path | None = None

    @property
    def restart_selector_disabled(self) -> bool:
        return not self.restart_options


def mode_label(value: str) -> str:
    return _MODE_LABELS.get(value, value.replace("_", " "))


def restart_mode_label(value: str) -> str:
    return _RESTART_MODE_LABELS.get(value, value.replace("_", " "))


def restart_reason_label(value: str) -> str:
    return _RESTART_REASON_LABELS.get(value, value.replace("_", " "))


def prepare_launch_mode_state(mode: str, run_dir: Path | None, config_path: Path | None) -> LaunchModeState:
    if mode == MODE_NEW_RUN:
        return LaunchModeState(
            mode_title="new run",
            info_markup="[dim]New run: a new logical run directory will be created automatically.[/dim]",
        )

    if mode == MODE_FRESH:
        if run_dir is None or not run_state_exists(run_dir):
            info_markup = "[yellow]Fresh attempt requires an existing run with pipeline_state.json.[/yellow]"
            return LaunchModeState(mode_title="fresh attempt", info_markup=info_markup)
        info_markup = (
            "[dim]Fresh attempt will restart the selected run from stage 1 and create a new attempt.[/dim]"
        )
        resolved_config = resolve_config_path_for_run(run_dir.expanduser().resolve(), config_path)
        return LaunchModeState(
            mode_title="fresh attempt",
            info_markup=info_markup,
            resolved_config_path=resolved_config,
        )

    if mode == MODE_RESUME:
        if run_dir is None:
            return LaunchModeState(
                mode_title="resume",
                info_markup="[yellow]Resume requires an existing run directory.[/yellow]",
            )
        try:
            resolved_config, resume_stage = validate_resume_run(run_dir, config_path)
        except Exception as exc:
            raise ValueError(_user_facing_prepare_error(str(exc))) from exc
        return LaunchModeState(
            mode_title="resume",
            info_markup=f"[dim]Resume will continue automatically from [bold]{resume_stage}[/bold].[/dim]",
            resolved_config_path=resolved_config,
        )

    if run_dir is None:
        return LaunchModeState(
            mode_title="restart",
            info_markup="[yellow]Start from stage requires an existing run directory.[/yellow]",
        )

    try:
        resolved_config, restart_points = load_restart_point_options(run_dir, config_path)
    except Exception as exc:
        raise ValueError(_user_facing_prepare_error(str(exc))) from exc
    restart_options = tuple((item.stage, item.stage) for item in restart_points if item.available)
    blocked_count = len(restart_points) - len(restart_options)
    lines = [
        f"[bold]Restart points[/bold]  [green]{len(restart_options)} available[/green]  [yellow]{blocked_count} blocked[/yellow]"
    ]
    for item in restart_points:
        if item.available:
            lines.append(f"  [green]+[/green] {item.stage}  [dim]{restart_mode_label(item.mode)}[/dim]")
        else:
            lines.append(f"  [yellow]-[/yellow] {item.stage}  [dim]{restart_reason_label(item.reason)}[/dim]")
    return LaunchModeState(
        mode_title=f"restart | {len(restart_options)} available",
        info_markup="\n".join(lines),
        restart_points=tuple(restart_points),
        restart_options=restart_options,
        selected_restart_stage=restart_options[0][1] if restart_options else None,
        resolved_config_path=resolved_config,
    )


def build_submittable_launch_request(
    *,
    mode: str,
    run_dir: Path | None,
    config_path: Path | None,
    restart_stage: str | None,
    log_level: str = "INFO",
) -> tuple[LaunchRequest, Path | None]:
    if run_dir is None:
        raise ValueError("Run directory is required.")

    resolved_config_path = config_path
    if mode == MODE_RESUME:
        resolved_config_path, _resume_stage = validate_resume_run(run_dir, config_path)

    request = LaunchRequest(
        mode=mode,  # type: ignore[arg-type]
        run_dir=run_dir,
        config_path=resolved_config_path,
        restart_from_stage=restart_stage,
        log_level=log_level,  # type: ignore[arg-type]
    ).validate()

    if mode == MODE_FRESH and not run_state_exists(run_dir):
        raise ValueError("Fresh attempt requires an existing run directory with pipeline_state.json.")

    return request, resolved_config_path

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.pipeline.restart_points import compute_config_hashes, list_restart_points
from src.pipeline.state import PipelineStateStore, StageRunState
from src.utils.config import load_config


@dataclass(frozen=True, slots=True)
class RestartPointOption:
    stage: str
    available: bool
    mode: str
    reason: str


def resolve_config_path_for_run(run_dir: Path, config_path: Path | None = None) -> Path:
    if config_path is not None:
        return config_path.expanduser().resolve()
    state = PipelineStateStore(run_dir.expanduser().resolve()).load()
    if not state.config_path:
        raise ValueError("Existing run has no config_path in pipeline_state.json")
    return Path(state.config_path).expanduser().resolve()


def load_restart_point_options(run_dir: Path, config_path: Path | None = None) -> tuple[Path, list[RestartPointOption]]:
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_config = resolve_config_path_for_run(resolved_run_dir, config_path)
    config = load_config(resolved_config)
    points = list_restart_points(resolved_run_dir, config)
    return resolved_config, [RestartPointOption(**point) for point in points]


def pick_default_launch_mode(run_dir: Path) -> str:
    try:
        state = PipelineStateStore(run_dir.expanduser().resolve()).load()
    except Exception:
        return "restart"
    return "resume" if derive_resume_stage(state) is not None else "restart"


def validate_resume_run(run_dir: Path, config_path: Path | None = None) -> tuple[Path, str]:
    resolved_run_dir = run_dir.expanduser().resolve()
    state = PipelineStateStore(resolved_run_dir).load()
    resolved_config = resolve_config_path_for_run(resolved_run_dir, config_path)
    config = load_config(resolved_config)
    config_hashes = compute_config_hashes(config)

    if state.model_dataset_config_hash:
        if state.model_dataset_config_hash != config_hashes["model_dataset"]:
            raise ValueError("training_critical config changed for existing logical run; resume is blocked")
    elif state.training_critical_config_hash != config_hashes["training_critical"]:
        raise ValueError("training_critical config changed for existing logical run; resume is blocked")

    if state.late_stage_config_hash != config_hashes["late_stage"]:
        raise ValueError(
            "late_stage config changed; resume is blocked. Use manual restart from Inference Deployer or Model Evaluator"
        )

    start_stage = derive_resume_stage(state)
    if start_stage is None:
        latest_status = state.attempts[-1].status if state.attempts else state.pipeline_status
        raise ValueError(f"Nothing to resume: latest attempt is already {latest_status}. Use restart or new run.")
    return resolved_config, start_stage


def derive_resume_stage(state) -> str | None:
    if not state.attempts:
        return None
    latest = state.attempts[-1]
    for stage_name in latest.enabled_stage_names or latest.stage_runs:
        stage_state = latest.stage_runs.get(stage_name)
        if stage_state is None:
            return stage_name
        if stage_state.status in {
            StageRunState.STATUS_FAILED,
            StageRunState.STATUS_INTERRUPTED,
            StageRunState.STATUS_PENDING,
            StageRunState.STATUS_RUNNING,
            StageRunState.STATUS_STALE,
        }:
            return stage_name
    return None

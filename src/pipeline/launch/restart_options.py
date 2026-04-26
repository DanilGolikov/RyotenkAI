"""Lightweight restart-point queries for launch flows.

This module powers the "what can I do with this saved run?" surface
exposed to the CLI (``ryotenkai list-restart-points``) and to the web
backend's launch endpoints. It is intentionally side-effect-light — no
orchestrator construction, no stage instantiation, no provider secrets,
no network probes — so importing it stays cheap enough for CLI startup.

Decision rules are shared with the orchestrator-driven path
(:class:`src.pipeline.execution.RestartPointsInspector`) via
:func:`src.pipeline.launch.restart_rules.compute_restart_points`. The
single difference: the inspector probes the live inference endpoint, this
module skips the probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.pipeline.config_drift import compute_config_hashes
from src.pipeline.launch.restart_rules import compute_restart_points
from src.pipeline.stages.constants import CANONICAL_STAGE_ORDER
from src.pipeline.state import PipelineStateStore
from src.pipeline.state.queries import first_unfinished_stage
from src.utils.config import load_config

if TYPE_CHECKING:
    from pathlib import Path

    from src.pipeline.state import PipelineState
    from src.utils.config import PipelineConfig


@dataclass(frozen=True, slots=True)
class RestartPointOption:
    stage: str
    available: bool
    mode: str
    reason: str


def list_restart_points(run_dir: Path, config: PipelineConfig) -> list[dict[str, Any]]:
    """Per-stage restart availability without a runtime health probe.

    The health probe is a network call with timeout — callers that just
    want a quick "can I restart from here?" answer skip it. The
    orchestrator's :class:`RestartPointsInspector` performs the probe
    when actually preparing a run.
    """
    store = PipelineStateStore(run_dir.expanduser().resolve())
    state = store.load()
    config_hashes = compute_config_hashes(config)
    return compute_restart_points(
        state=state,
        stage_names=CANONICAL_STAGE_ORDER,
        config_hashes=config_hashes,
        # Probe intentionally omitted — runtime check belongs in
        # _validate_stage_prerequisites at launch time.
        inference_health_checker=lambda _ctx: True,
    )


# ---------------------------------------------------------------------------
# Resume / restart helpers consumed by main.py and the web launch service
# ---------------------------------------------------------------------------


def resolve_config_path_for_run(run_dir: Path, config_path: Path | None = None) -> Path:
    if config_path is not None:
        return config_path.expanduser().resolve()
    state = PipelineStateStore(run_dir.expanduser().resolve()).load()
    if not state.config_path:
        raise ValueError("Existing run has no config_path in pipeline_state.json")
    return Path(state.config_path).expanduser().resolve()


def load_restart_point_options(
    run_dir: Path, config_path: Path | None = None
) -> tuple[Path, list[RestartPointOption]]:
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


def derive_resume_stage(state: PipelineState) -> str | None:
    """First not-successfully-completed stage in the latest attempt.

    Iterates over the attempt's saved stage list (``enabled_stage_names``
    if present, otherwise the keys of ``stage_runs``) — that's the source
    of truth for "what was scheduled to run". The orchestrator-side
    ``StagePlanner.derive_resume_stage`` answers the same question against
    the *live* stage list; both delegate iteration to the shared
    :func:`first_unfinished_stage` helper.
    """
    if not state.attempts:
        return None
    latest = state.attempts[-1]
    return first_unfinished_stage(
        latest.enabled_stage_names or latest.stage_runs,
        latest.stage_runs,
    )


__all__ = [
    "RestartPointOption",
    "derive_resume_stage",
    "list_restart_points",
    "load_restart_point_options",
    "pick_default_launch_mode",
    "resolve_config_path_for_run",
    "validate_resume_run",
]

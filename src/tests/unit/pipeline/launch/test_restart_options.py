from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.pipeline.launch.restart_options import (
    RestartPointOption,
    derive_resume_stage,
    load_restart_point_options,
    pick_default_launch_mode,
    resolve_config_path_for_run,
    validate_resume_run,
)
from src.pipeline.state import PipelineAttemptState, PipelineState, StageRunState


def _build_state(
    *,
    config_path: str = "configs/pipeline.yaml",
    training_critical_config_hash: str = "train_hash",
    late_stage_config_hash: str = "late_hash",
    model_dataset_config_hash: str = "",
    attempts: list[PipelineAttemptState] | None = None,
    pipeline_status: str = StageRunState.STATUS_FAILED,
) -> PipelineState:
    return PipelineState(
        schema_version=1,
        logical_run_id="run_1",
        run_directory="runs/run_1",
        config_path=config_path,
        active_attempt_id=None,
        pipeline_status=pipeline_status,
        training_critical_config_hash=training_critical_config_hash,
        late_stage_config_hash=late_stage_config_hash,
        model_dataset_config_hash=model_dataset_config_hash,
        attempts=attempts or [],
    )


def _build_attempt(
    *,
    status: str,
    enabled_stage_names: list[str] | None = None,
    stage_runs: dict[str, StageRunState] | None = None,
) -> PipelineAttemptState:
    return PipelineAttemptState(
        attempt_id="attempt-1",
        attempt_no=1,
        runtime_name="runtime",
        requested_action="resume",
        effective_action="resume",
        restart_from_stage=None,
        status=status,
        started_at="2026-03-30T00:00:00+00:00",
        enabled_stage_names=enabled_stage_names or ["Dataset Validator"],
        stage_runs=stage_runs or {},
    )


def test_resolve_config_path_for_run_uses_explicit_path(tmp_path: Path) -> None:
    explicit_path = tmp_path / "configs" / "pipeline.yaml"

    result = resolve_config_path_for_run(tmp_path / "runs" / "existing", explicit_path)

    assert result == explicit_path.resolve()


def test_resolve_config_path_for_run_raises_when_state_has_no_config_path(monkeypatch, tmp_path: Path) -> None:
    state = _build_state(config_path="")
    monkeypatch.setattr("src.pipeline.launch.restart_options.PipelineStateStore.load", lambda self: state)

    with pytest.raises(ValueError, match="no config_path"):
        resolve_config_path_for_run(tmp_path / "runs" / "existing")


def test_load_restart_point_options_maps_plain_dicts_to_dataclass(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_obj = MagicMock()
    restart_points = [
        {
            "stage": "Inference Deployer",
            "available": True,
            "mode": "fresh_or_resume",
            "reason": "restart_allowed",
        },
        {
            "stage": "Model Evaluator",
            "available": False,
            "mode": "live_runtime_only",
            "reason": "missing_inference_outputs",
        },
    ]

    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.resolve_config_path_for_run",
        lambda run_dir, provided_config_path=None: provided_config_path,
    )
    monkeypatch.setattr("src.pipeline.launch.restart_options.load_config", lambda _path: config_obj)
    monkeypatch.setattr("src.pipeline.launch.restart_options.list_restart_points", lambda run_dir, config: restart_points)

    resolved_path, points = load_restart_point_options(tmp_path / "runs" / "existing", config_path)

    assert resolved_path == config_path
    assert points == [
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
    ]


def test_load_restart_point_options_propagates_load_config_error(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.resolve_config_path_for_run",
        lambda run_dir, provided_config_path=None: provided_config_path,
    )
    monkeypatch.setattr("src.pipeline.launch.restart_options.load_config", lambda _path: (_ for _ in ()).throw(RuntimeError("bad config")))

    with pytest.raises(RuntimeError, match="bad config"):
        load_restart_point_options(tmp_path / "runs" / "existing", config_path)


def test_load_restart_point_options_propagates_restart_point_query_error(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.resolve_config_path_for_run",
        lambda run_dir, provided_config_path=None: provided_config_path,
    )
    monkeypatch.setattr("src.pipeline.launch.restart_options.load_config", lambda _path: MagicMock())
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.list_restart_points",
        lambda run_dir, config: (_ for _ in ()).throw(RuntimeError("query failed")),
    )

    with pytest.raises(RuntimeError, match="query failed"):
        load_restart_point_options(tmp_path / "runs" / "existing", config_path)


def test_pick_default_launch_mode_returns_restart_when_state_load_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.PipelineStateStore.load",
        lambda self: (_ for _ in ()).throw(FileNotFoundError("missing state")),
    )

    assert pick_default_launch_mode(tmp_path / "runs" / "missing") == "restart"


def test_pick_default_launch_mode_returns_resume_when_resume_stage_exists(monkeypatch, tmp_path: Path) -> None:
    state = _build_state(
        attempts=[
            _build_attempt(
                status=StageRunState.STATUS_FAILED,
                stage_runs={"Dataset Validator": StageRunState(stage_name="Dataset Validator", status=StageRunState.STATUS_FAILED)},
            )
        ]
    )
    monkeypatch.setattr("src.pipeline.launch.restart_options.PipelineStateStore.load", lambda self: state)

    assert pick_default_launch_mode(tmp_path / "runs" / "existing") == "resume"


@pytest.mark.parametrize(
    ("stage_status", "expected_stage"),
    [
        (StageRunState.STATUS_FAILED, "Dataset Validator"),
        (StageRunState.STATUS_INTERRUPTED, "Dataset Validator"),
        (StageRunState.STATUS_PENDING, "Dataset Validator"),
        (StageRunState.STATUS_RUNNING, "Dataset Validator"),
        (StageRunState.STATUS_STALE, "Dataset Validator"),
        (StageRunState.STATUS_COMPLETED, None),
        (StageRunState.STATUS_SKIPPED, None),
    ],
)
def test_derive_resume_stage_status_matrix(stage_status: str, expected_stage: str | None) -> None:
    state = _build_state(
        attempts=[
            _build_attempt(
                status=stage_status,
                stage_runs={"Dataset Validator": StageRunState(stage_name="Dataset Validator", status=stage_status)},
            )
        ]
    )

    assert derive_resume_stage(state) == expected_stage


def test_derive_resume_stage_returns_first_missing_stage_in_enabled_order() -> None:
    state = _build_state(
        attempts=[
            _build_attempt(
                status=StageRunState.STATUS_FAILED,
                enabled_stage_names=["Dataset Validator", "GPU Deployer"],
                stage_runs={
                    "Dataset Validator": StageRunState(
                        stage_name="Dataset Validator",
                        status=StageRunState.STATUS_COMPLETED,
                    )
                },
            )
        ]
    )

    assert derive_resume_stage(state) == "GPU Deployer"


def test_validate_resume_run_allows_provider_only_hash_drift_when_model_dataset_hash_matches(monkeypatch, tmp_path: Path) -> None:
    state = _build_state(
        training_critical_config_hash="old_training_hash",
        late_stage_config_hash="late_hash",
        model_dataset_config_hash="model_hash",
        attempts=[
            _build_attempt(
                status=StageRunState.STATUS_FAILED,
                stage_runs={"Dataset Validator": StageRunState(stage_name="Dataset Validator", status=StageRunState.STATUS_FAILED)},
            )
        ],
    )
    config_path = tmp_path / "config.yaml"
    config_obj = MagicMock()

    monkeypatch.setattr("src.pipeline.launch.restart_options.PipelineStateStore.load", lambda self: state)
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.resolve_config_path_for_run",
        lambda run_dir, provided_config_path=None: provided_config_path,
    )
    monkeypatch.setattr("src.pipeline.launch.restart_options.load_config", lambda _path: config_obj)
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.compute_config_hashes",
        lambda _config: {
            "training_critical": "new_training_hash",
            "late_stage": "late_hash",
            "model_dataset": "model_hash",
        },
    )

    resolved_config, start_stage = validate_resume_run(tmp_path / "runs" / "existing", config_path)

    assert resolved_config == config_path
    assert start_stage == "Dataset Validator"


def test_validate_resume_run_blocks_when_model_dataset_hash_changes(monkeypatch, tmp_path: Path) -> None:
    state = _build_state(
        training_critical_config_hash="old_training_hash",
        late_stage_config_hash="late_hash",
        model_dataset_config_hash="old_model_hash",
        attempts=[
            _build_attempt(
                status=StageRunState.STATUS_FAILED,
                stage_runs={"Dataset Validator": StageRunState(stage_name="Dataset Validator", status=StageRunState.STATUS_FAILED)},
            )
        ],
    )
    config_path = tmp_path / "config.yaml"
    config_obj = MagicMock()

    monkeypatch.setattr("src.pipeline.launch.restart_options.PipelineStateStore.load", lambda self: state)
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.resolve_config_path_for_run",
        lambda run_dir, provided_config_path=None: provided_config_path,
    )
    monkeypatch.setattr("src.pipeline.launch.restart_options.load_config", lambda _path: config_obj)
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.compute_config_hashes",
        lambda _config: {
            "training_critical": "new_training_hash",
            "late_stage": "late_hash",
            "model_dataset": "new_model_hash",
        },
    )

    with pytest.raises(ValueError, match="training_critical config changed"):
        validate_resume_run(tmp_path / "runs" / "existing", config_path)


def test_validate_resume_run_uses_legacy_training_hash_when_model_dataset_hash_missing(monkeypatch, tmp_path: Path) -> None:
    state = _build_state(
        training_critical_config_hash="old_training_hash",
        late_stage_config_hash="late_hash",
        model_dataset_config_hash="",
        attempts=[
            _build_attempt(
                status=StageRunState.STATUS_FAILED,
                stage_runs={"Dataset Validator": StageRunState(stage_name="Dataset Validator", status=StageRunState.STATUS_FAILED)},
            )
        ],
    )
    config_path = tmp_path / "config.yaml"
    config_obj = MagicMock()

    monkeypatch.setattr("src.pipeline.launch.restart_options.PipelineStateStore.load", lambda self: state)
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.resolve_config_path_for_run",
        lambda run_dir, provided_config_path=None: provided_config_path,
    )
    monkeypatch.setattr("src.pipeline.launch.restart_options.load_config", lambda _path: config_obj)
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.compute_config_hashes",
        lambda _config: {
            "training_critical": "new_training_hash",
            "late_stage": "late_hash",
            "model_dataset": "ignored",
        },
    )

    with pytest.raises(ValueError, match="training_critical config changed"):
        validate_resume_run(tmp_path / "runs" / "existing", config_path)


def test_validate_resume_run_blocks_on_late_stage_hash_drift(monkeypatch, tmp_path: Path) -> None:
    state = _build_state(
        model_dataset_config_hash="model_hash",
        late_stage_config_hash="old_late_hash",
        attempts=[
            _build_attempt(
                status=StageRunState.STATUS_FAILED,
                stage_runs={"Dataset Validator": StageRunState(stage_name="Dataset Validator", status=StageRunState.STATUS_FAILED)},
            )
        ],
    )
    config_path = tmp_path / "config.yaml"
    config_obj = MagicMock()

    monkeypatch.setattr("src.pipeline.launch.restart_options.PipelineStateStore.load", lambda self: state)
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.resolve_config_path_for_run",
        lambda run_dir, provided_config_path=None: provided_config_path,
    )
    monkeypatch.setattr("src.pipeline.launch.restart_options.load_config", lambda _path: config_obj)
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.compute_config_hashes",
        lambda _config: {
            "training_critical": "train_hash",
            "late_stage": "new_late_hash",
            "model_dataset": "model_hash",
        },
    )

    with pytest.raises(ValueError, match="late_stage config changed"):
        validate_resume_run(tmp_path / "runs" / "existing", config_path)


def test_validate_resume_run_raises_when_nothing_is_resumable(monkeypatch, tmp_path: Path) -> None:
    state = _build_state(
        model_dataset_config_hash="model_hash",
        attempts=[
            _build_attempt(
                status=StageRunState.STATUS_COMPLETED,
                stage_runs={"Dataset Validator": StageRunState(stage_name="Dataset Validator", status=StageRunState.STATUS_COMPLETED)},
            )
        ],
    )
    config_path = tmp_path / "config.yaml"
    config_obj = MagicMock()

    monkeypatch.setattr("src.pipeline.launch.restart_options.PipelineStateStore.load", lambda self: state)
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.resolve_config_path_for_run",
        lambda run_dir, provided_config_path=None: provided_config_path,
    )
    monkeypatch.setattr("src.pipeline.launch.restart_options.load_config", lambda _path: config_obj)
    monkeypatch.setattr(
        "src.pipeline.launch.restart_options.compute_config_hashes",
        lambda _config: {
            "training_critical": "train_hash",
            "late_stage": "late_hash",
            "model_dataset": "model_hash",
        },
    )

    with pytest.raises(ValueError, match="Nothing to resume"):
        validate_resume_run(tmp_path / "runs" / "existing", config_path)

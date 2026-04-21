from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.orchestrator import LaunchPreparationError, PipelineOrchestrator
from src.pipeline.state import PipelineAttemptState, PipelineState, PipelineStateLoadError, PipelineStateStore, StageLineageRef, StageRunState
from src.pipeline.stages.constants import StageNames
from src.utils.result import Ok


def _build_mock_config() -> MagicMock:
    config = MagicMock()
    config.model.name = "gpt2"
    config.model.model_dump.return_value = {"name": "gpt2"}
    config.training.type = "sft"
    config.training.strategies = []
    config.training.model_dump.return_value = {"type": "sft"}
    config.training.get_strategy_chain.return_value = []
    config.training.get_effective_load_in_4bit.return_value = False
    config.training.hyperparams.per_device_train_batch_size = 4

    dataset_cfg = MagicMock()
    dataset_cfg.model_dump.return_value = {"path": "data/train.jsonl"}
    config.datasets = {"default": dataset_cfg}

    config.get_active_provider_name.return_value = "single_node"
    config.get_provider_config.return_value = {"cleanup": {"on_interrupt": True}}
    config.experiment_tracking.mlflow = MagicMock(
        tracking_uri="http://localhost:5002",
        system_metrics_callback_enabled=False,
    )

    config.inference.enabled = False
    config.inference.model_dump.return_value = {"enabled": False}
    config.inference.common.keep_inference_after_eval = False

    config.evaluation.enabled = False
    config.evaluation.model_dump.return_value = {"enabled": False}
    return config


def _build_orchestrator(config_path: Path, config: MagicMock, *, run_directory: Path | None = None) -> PipelineOrchestrator:
    secrets = MagicMock()
    secrets.hf_token = "test-token"
    stages = [
        MagicMock(stage_name=StageNames.DATASET_VALIDATOR),
        MagicMock(stage_name=StageNames.GPU_DEPLOYER),
        MagicMock(stage_name=StageNames.TRAINING_MONITOR),
        MagicMock(stage_name=StageNames.MODEL_RETRIEVER),
        MagicMock(stage_name=StageNames.INFERENCE_DEPLOYER),
        MagicMock(stage_name=StageNames.MODEL_EVALUATOR),
    ]
    with (
        patch("src.pipeline.orchestrator.load_config", return_value=config),
        patch("src.pipeline.orchestrator.load_secrets", return_value=secrets),
        patch("src.pipeline.orchestrator.validate_strategy_chain", return_value=Ok(None)),
        patch.object(PipelineOrchestrator, "_init_stages", return_value=stages),
    ):
        orchestrator = PipelineOrchestrator(config_path, run_directory=run_directory)
    orchestrator._setup_mlflow_for_attempt = MagicMock(return_value=None)
    orchestrator._ensure_mlflow_preflight = MagicMock(return_value=None)
    return orchestrator


def _make_attempt(stage_states: dict[str, StageRunState], *, status: str = StageRunState.STATUS_FAILED) -> PipelineAttemptState:
    return PipelineAttemptState(
        attempt_id="logical_run_1:attempt:1",
        attempt_no=1,
        runtime_name="runtime_attempt",
        requested_action="fresh",
        effective_action="fresh",
        restart_from_stage=StageNames.DATASET_VALIDATOR,
        status=status,
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at=datetime.now(timezone.utc).isoformat(),
        training_critical_config_hash="train_hash",
        late_stage_config_hash="late_hash",
        enabled_stage_names=list(stage_states),
        stage_runs=stage_states,
    )


def test_normalize_stage_ref_supports_indices_and_names(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    orchestrator = _build_orchestrator(config_path, _build_mock_config())

    # 1-based human index (1 = first, 6 = last)
    assert orchestrator._normalize_stage_ref(1) == StageNames.DATASET_VALIDATOR
    assert orchestrator._normalize_stage_ref("1") == StageNames.DATASET_VALIDATOR
    assert orchestrator._normalize_stage_ref(5) == StageNames.INFERENCE_DEPLOYER
    assert orchestrator._normalize_stage_ref("6") == StageNames.MODEL_EVALUATOR
    # name-based (case and underscore-insensitive)
    assert orchestrator._normalize_stage_ref("Inference Deployer") == StageNames.INFERENCE_DEPLOYER
    assert orchestrator._normalize_stage_ref("model_evaluator") == StageNames.MODEL_EVALUATOR


def test_normalize_stage_ref_rejects_unknown_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    orchestrator = _build_orchestrator(config_path, _build_mock_config())

    with pytest.raises(ValueError, match="Unknown stage reference"):
        orchestrator._normalize_stage_ref("unknown_stage")
    with pytest.raises(ValueError, match="out of range"):
        orchestrator._normalize_stage_ref("0")
    with pytest.raises(ValueError, match="out of range"):
        orchestrator._normalize_stage_ref("7")


def test_derive_resume_stage_prefers_first_missing_or_failed_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    orchestrator = _build_orchestrator(config_path, _build_mock_config())

    attempt = _make_attempt(
        {
            StageNames.DATASET_VALIDATOR: StageRunState(
                stage_name=StageNames.DATASET_VALIDATOR,
                status=StageRunState.STATUS_COMPLETED,
            ),
            StageNames.GPU_DEPLOYER: StageRunState(
                stage_name=StageNames.GPU_DEPLOYER,
                status=StageRunState.STATUS_COMPLETED,
            ),
            StageNames.TRAINING_MONITOR: StageRunState(
                stage_name=StageNames.TRAINING_MONITOR,
                status=StageRunState.STATUS_FAILED,
            ),
        }
    )
    state = PipelineState(
        schema_version=1,
        logical_run_id="logical_run_1",
        run_directory=str(tmp_path / "runs" / "logical_run_1"),
        config_path=str(config_path),
        active_attempt_id=attempt.attempt_id,
        pipeline_status=StageRunState.STATUS_FAILED,
        training_critical_config_hash="train_hash",
        late_stage_config_hash="late_hash",
        attempts=[attempt],
        current_output_lineage={},
    )

    assert orchestrator._derive_resume_stage(state) == StageNames.TRAINING_MONITOR


def test_bootstrap_pipeline_state_fresh_creates_new_state(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    orchestrator = _build_orchestrator(config_path, _build_mock_config(), run_directory=run_dir)

    state, requested_action, effective_action, start_stage = orchestrator._bootstrap_pipeline_state(
        run_dir=run_dir,
        resume=False,
        restart_from_stage=None,
        config_hashes={"training_critical": "train_hash", "late_stage": "late_hash", "model_dataset": "md_hash"},
    )

    assert requested_action == "fresh"
    assert effective_action == "fresh"
    assert start_stage == StageNames.DATASET_VALIDATOR
    assert state.logical_run_id == "logical_run_1"
    assert (run_dir / "pipeline_state.json").exists()


def test_bootstrap_pipeline_state_resume_requires_existing_state(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    orchestrator = _build_orchestrator(config_path, _build_mock_config(), run_directory=run_dir)

    with pytest.raises(PipelineStateLoadError, match="Missing pipeline_state.json"):
        orchestrator._bootstrap_pipeline_state(
            run_dir=run_dir,
            resume=True,
            restart_from_stage=None,
            config_hashes={"training_critical": "train_hash", "late_stage": "late_hash", "model_dataset": "md_hash"},
        )


def test_bootstrap_pipeline_state_resume_uses_failed_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    store = PipelineStateStore(run_dir)
    state = store.init_state(
        logical_run_id="logical_run_1",
        config_path=str(config_path),
        training_critical_config_hash="train_hash",
        late_stage_config_hash="late_hash",
    )
    attempt = _make_attempt(
        {
            StageNames.DATASET_VALIDATOR: StageRunState(stage_name=StageNames.DATASET_VALIDATOR, status="completed"),
            StageNames.GPU_DEPLOYER: StageRunState(stage_name=StageNames.GPU_DEPLOYER, status="failed"),
        }
    )
    state.attempts.append(attempt)
    state.active_attempt_id = attempt.attempt_id
    store.save(state)

    orchestrator = _build_orchestrator(config_path, _build_mock_config(), run_directory=run_dir)

    loaded_state, requested_action, effective_action, start_stage = orchestrator._bootstrap_pipeline_state(
        run_dir=run_dir,
        resume=True,
        restart_from_stage=None,
        config_hashes={"training_critical": "train_hash", "late_stage": "late_hash", "model_dataset": "md_hash"},
    )

    assert loaded_state.logical_run_id == "logical_run_1"
    assert requested_action == "resume"
    assert effective_action == "auto_resume"
    assert start_stage == StageNames.GPU_DEPLOYER


def test_bootstrap_pipeline_state_blocks_training_critical_drift(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    store = PipelineStateStore(run_dir)
    state = store.init_state(
        logical_run_id="logical_run_1",
        config_path=str(config_path),
        training_critical_config_hash="old_hash",
        late_stage_config_hash="late_hash",
    )
    store.save(state)

    orchestrator = _build_orchestrator(config_path, _build_mock_config(), run_directory=run_dir)

    with pytest.raises(LaunchPreparationError, match="training_critical config changed"):
        orchestrator._bootstrap_pipeline_state(
            run_dir=run_dir,
            resume=False,
            restart_from_stage=StageNames.MODEL_RETRIEVER,
            config_hashes={"training_critical": "new_hash", "late_stage": "late_hash", "model_dataset": "md_hash"},
        )


def test_build_dataset_validation_state_outputs_compacts_summary(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    orchestrator = _build_orchestrator(config_path, _build_mock_config())
    orchestrator._validation_artifact_mgr._validation_accumulator = {
        "/data/train.jsonl": {
            "name": "default",
            "path": "/data/train.jsonl",
            "sample_count": 10,
            "status": "failed",
            "critical_failures": 1,
            "plugins": [],
        }
    }

    outputs = orchestrator._validation_artifact_mgr.build_dataset_validation_state_outputs(
        stage_ctx={
            "validation_status": "failed",
            "warnings": ["w1", "w2"],
            "message": "All datasets failed validation",
        }
    )

    assert outputs["validation_artifact_ref"] == "dataset_validator_results.json"
    assert outputs["validation_status"] == "failed"
    assert outputs["datasets_validated"] == 1
    assert outputs["datasets_failed"] == 1
    assert outputs["failed_datasets"] == ["default"]
    assert outputs["validation_warning_count"] == 2


@pytest.mark.parametrize(
    ("inference_enabled", "evaluation_enabled", "start_stage", "expected_enabled"),
    [
        (False, False, StageNames.DATASET_VALIDATOR, {StageNames.DATASET_VALIDATOR, StageNames.GPU_DEPLOYER, StageNames.TRAINING_MONITOR, StageNames.MODEL_RETRIEVER}),
        (True, False, StageNames.DATASET_VALIDATOR, {StageNames.DATASET_VALIDATOR, StageNames.GPU_DEPLOYER, StageNames.TRAINING_MONITOR, StageNames.MODEL_RETRIEVER, StageNames.INFERENCE_DEPLOYER}),
        (True, True, StageNames.DATASET_VALIDATOR, {StageNames.DATASET_VALIDATOR, StageNames.GPU_DEPLOYER, StageNames.TRAINING_MONITOR, StageNames.MODEL_RETRIEVER, StageNames.INFERENCE_DEPLOYER, StageNames.MODEL_EVALUATOR}),
        (False, False, StageNames.INFERENCE_DEPLOYER, {StageNames.DATASET_VALIDATOR, StageNames.GPU_DEPLOYER, StageNames.TRAINING_MONITOR, StageNames.MODEL_RETRIEVER, StageNames.INFERENCE_DEPLOYER}),
        (False, False, StageNames.MODEL_EVALUATOR, {StageNames.DATASET_VALIDATOR, StageNames.GPU_DEPLOYER, StageNames.TRAINING_MONITOR, StageNames.MODEL_RETRIEVER, StageNames.MODEL_EVALUATOR}),
    ],
)
def test_compute_enabled_stage_names_covers_optional_stage_combinations(
    tmp_path: Path,
    inference_enabled: bool,
    evaluation_enabled: bool,
    start_stage: str,
    expected_enabled: set[str],
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    config = _build_mock_config()
    config.inference.enabled = inference_enabled
    config.inference.model_dump.return_value = {"enabled": inference_enabled}
    config.evaluation.enabled = evaluation_enabled
    config.evaluation.model_dump.return_value = {"enabled": evaluation_enabled}
    orchestrator = _build_orchestrator(config_path, config)

    enabled = set(orchestrator._compute_enabled_stage_names(start_stage_name=start_stage))

    assert enabled == expected_enabled


def test_validate_stage_prerequisites_for_training_monitor_requires_workspace(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    orchestrator = _build_orchestrator(config_path, _build_mock_config())
    orchestrator.context[StageNames.GPU_DEPLOYER] = {
        "ssh_host": "127.0.0.1",
        "ssh_port": 22,
    }

    error = orchestrator._validate_stage_prerequisites(
        stage_name=StageNames.TRAINING_MONITOR,
        start_stage_name=StageNames.TRAINING_MONITOR,
    )

    assert error is not None
    assert error.code == "MISSING_TRAINING_MONITOR_PREREQUISITES"


def test_validate_stage_prerequisites_for_evaluator_requires_live_runtime(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    orchestrator = _build_orchestrator(config_path, _build_mock_config())

    with patch.object(orchestrator, "_is_inference_runtime_healthy", return_value=False):
        error = orchestrator._validate_stage_prerequisites(
            stage_name=StageNames.MODEL_EVALUATOR,
            start_stage_name=StageNames.MODEL_EVALUATOR,
        )

    assert error is not None
    assert error.code == "INFERENCE_RUNTIME_NOT_HEALTHY"


def test_is_inference_runtime_healthy_handles_success_and_failure(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    orchestrator = _build_orchestrator(config_path, _build_mock_config())
    orchestrator.context[StageNames.INFERENCE_DEPLOYER] = {
        "endpoint_info": {"health_url": "http://localhost/health"},
    }

    response = MagicMock()
    response.__enter__.return_value = MagicMock(status=200)
    response.__exit__.return_value = None
    with patch("src.pipeline.executor.stage_planner.urlopen", return_value=response):
        assert orchestrator._is_inference_runtime_healthy() is True

    with patch("src.pipeline.executor.stage_planner.urlopen", side_effect=RuntimeError("boom")):
        assert orchestrator._is_inference_runtime_healthy() is False


def test_list_restart_points_reports_block_reasons(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    store = PipelineStateStore(run_dir)
    state = store.init_state(
        logical_run_id="logical_run_1",
        config_path=str(config_path),
        training_critical_config_hash="train_hash",
        late_stage_config_hash="late_hash",
    )
    state.current_output_lineage = {
        StageNames.GPU_DEPLOYER: StageLineageRef(
            attempt_id="logical_run_1:attempt:1",
            stage_name=StageNames.GPU_DEPLOYER,
            outputs={"ssh_host": "host", "ssh_port": 22},
        )
    }
    store.save(state)

    config = _build_mock_config()
    config.evaluation.enabled = True
    config.evaluation.model_dump.return_value = {"enabled": True}
    orchestrator = _build_orchestrator(config_path, config, run_directory=run_dir)
    config_hashes = orchestrator._build_config_hashes()
    state.training_critical_config_hash = config_hashes["training_critical"]
    state.late_stage_config_hash = config_hashes["late_stage"]
    store.save(state)

    with patch.object(orchestrator, "_is_inference_runtime_healthy", return_value=False):
        points = orchestrator.list_restart_points(run_dir)

    by_stage = {item["stage"]: item for item in points}
    assert by_stage[StageNames.TRAINING_MONITOR]["available"] is False
    assert by_stage[StageNames.TRAINING_MONITOR]["reason"] == "missing_gpu_deployer_outputs"
    assert by_stage[StageNames.INFERENCE_DEPLOYER]["reason"] == "missing_model_retriever_outputs"
    assert by_stage[StageNames.MODEL_EVALUATOR]["reason"] == "missing_inference_outputs"


def test_list_restart_points_marks_late_stage_config_drift(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    store = PipelineStateStore(run_dir)
    state = store.init_state(
        logical_run_id="logical_run_1",
        config_path=str(config_path),
        training_critical_config_hash="placeholder",
        late_stage_config_hash="old_late_hash",
    )
    state.current_output_lineage = {
        StageNames.MODEL_RETRIEVER: StageLineageRef(
            attempt_id="logical_run_1:attempt:1",
            stage_name=StageNames.MODEL_RETRIEVER,
            outputs={"hf_repo_id": "repo/model"},
        )
    }
    store.save(state)

    config = _build_mock_config()
    config.inference.enabled = True
    config.inference.model_dump.return_value = {"enabled": True, "port": 8000}
    orchestrator = _build_orchestrator(config_path, config, run_directory=run_dir)
    config_hashes = orchestrator._build_config_hashes()
    state.training_critical_config_hash = config_hashes["training_critical"]
    store.save(state)

    points = orchestrator.list_restart_points(run_dir)
    by_stage = {item["stage"]: item for item in points}

    assert by_stage[StageNames.MODEL_RETRIEVER]["available"] is False
    assert by_stage[StageNames.MODEL_RETRIEVER]["reason"] == "late_stage_config_changed"
    assert by_stage[StageNames.INFERENCE_DEPLOYER]["available"] is True

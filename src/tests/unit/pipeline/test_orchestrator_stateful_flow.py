from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.execution import StageRegistry
from src.pipeline.state import PipelineStateStore, StageRunState
from src.pipeline.stages.base import PipelineStage
from src.pipeline.stages.constants import StageNames
from src.utils.result import AppError, Err, Failure, Ok, Result, Success


def _build_mock_config(*, inference_enabled: bool, evaluation_enabled: bool) -> MagicMock:
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

    config.inference.enabled = inference_enabled
    config.inference.model_dump.return_value = {"enabled": inference_enabled}
    config.inference.common.keep_inference_after_eval = False

    config.evaluation.enabled = evaluation_enabled
    config.evaluation.model_dump.return_value = {"enabled": evaluation_enabled}
    return config


class ControlledStage(PipelineStage):
    def __init__(
        self,
        config: MagicMock,
        stage_name: str,
        *,
        calls: list[str],
        behavior: Any,
    ) -> None:
        super().__init__(config, stage_name)
        self._calls = calls
        self._behavior = behavior
        self.cleanup_calls = 0

    def execute(self, context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
        self._calls.append(self.stage_name)
        result = self._behavior(context)
        if isinstance(result, (Success, Failure)):
            return result
        return Ok(self.update_context(context, result))

    def cleanup(self) -> None:
        self.cleanup_calls += 1


def _make_stage_behaviors(*, fail_stage: str | None = None, interrupt_stage: str | None = None) -> dict[str, Any]:
    def dataset_validator(_context: dict[str, Any]) -> dict[str, Any]:
        return {"sample_count": 10, "validation_mode": "plugin"}

    def gpu_deployer(_context: dict[str, Any]) -> dict[str, Any]:
        return {
            "resource_id": "pod-1",
            "ssh_host": "127.0.0.1",
            "ssh_port": 22,
            "ssh_user": "root",
            "ssh_key_path": "/tmp/key",
            "workspace_path": "/workspace",
            "provider_name": "single_node",
            "provider_type": "local",
            "provider_info": {},
        }

    def training_monitor(_context: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "completed",
            "training_duration_seconds": 12.5,
            "training_info": {"runtime_seconds": 12.5},
        }

    def model_retriever(_context: dict[str, Any]) -> dict[str, Any]:
        return {
            "hf_repo_id": "repo/model",
            "local_model_path": "/tmp/model",
            "hf_uploaded": True,
            "model_size_mb": 123.0,
        }

    def inference_deployer(_context: dict[str, Any]) -> dict[str, Any]:
        return {
            "endpoint_url": "http://localhost:8000/v1",
            "inference_endpoint_url": "http://localhost:8000/v1",
            "inference_model_name": "repo/model",
            "endpoint_info": {"health_url": "http://localhost:8000/health"},
        }

    def model_evaluator(_context: dict[str, Any]) -> dict[str, Any]:
        return {
            "eval_passed": True,
            "eval_summary": {"score": 1.0},
        }

    behaviors = {
        StageNames.DATASET_VALIDATOR: dataset_validator,
        StageNames.GPU_DEPLOYER: gpu_deployer,
        StageNames.TRAINING_MONITOR: training_monitor,
        StageNames.MODEL_RETRIEVER: model_retriever,
        StageNames.INFERENCE_DEPLOYER: inference_deployer,
        StageNames.MODEL_EVALUATOR: model_evaluator,
    }

    if fail_stage is not None:
        def _failed(_context: dict[str, Any]) -> Result[dict[str, Any], AppError]:
            return Err(AppError(message=f"{fail_stage} failed", code=f"{fail_stage}_FAILED"))

        behaviors[fail_stage] = _failed

    if interrupt_stage is not None:
        def _interrupt(_context: dict[str, Any]) -> dict[str, Any]:
            raise KeyboardInterrupt()

        behaviors[interrupt_stage] = _interrupt

    return behaviors


def _build_orchestrator(
    config_path: Path,
    config: MagicMock,
    *,
    run_directory: Path,
    stage_behaviors: dict[str, Any],
    calls: list[str],
) -> PipelineOrchestrator:
    secrets = MagicMock()
    secrets.hf_token = "test-token"
    stages = [
        ControlledStage(config, StageNames.DATASET_VALIDATOR, calls=calls, behavior=stage_behaviors[StageNames.DATASET_VALIDATOR]),
        ControlledStage(config, StageNames.GPU_DEPLOYER, calls=calls, behavior=stage_behaviors[StageNames.GPU_DEPLOYER]),
        ControlledStage(config, StageNames.TRAINING_MONITOR, calls=calls, behavior=stage_behaviors[StageNames.TRAINING_MONITOR]),
        ControlledStage(config, StageNames.MODEL_RETRIEVER, calls=calls, behavior=stage_behaviors[StageNames.MODEL_RETRIEVER]),
        ControlledStage(config, StageNames.INFERENCE_DEPLOYER, calls=calls, behavior=stage_behaviors[StageNames.INFERENCE_DEPLOYER]),
        ControlledStage(config, StageNames.MODEL_EVALUATOR, calls=calls, behavior=stage_behaviors[StageNames.MODEL_EVALUATOR]),
    ]
    with (
        patch("src.pipeline.bootstrap.pipeline_bootstrap.load_config", return_value=config),
        patch("src.pipeline.bootstrap.pipeline_bootstrap.load_secrets", return_value=secrets),
        patch("src.pipeline.bootstrap.startup_validator.validate_strategy_chain", return_value=Ok(None)),
        patch.object(StageRegistry, "_build_stages", return_value=stages),
        patch.object(PipelineOrchestrator, "_setup_mlflow", return_value=None),
    ):
        orchestrator = PipelineOrchestrator(config_path, run_directory=run_directory)
    orchestrator._setup_mlflow_for_attempt = MagicMock(return_value=None)
    orchestrator._ensure_mlflow_preflight = MagicMock(return_value=None)
    return orchestrator


def test_stateful_fresh_run_persists_completed_and_skipped_stages(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    config = _build_mock_config(inference_enabled=False, evaluation_enabled=False)
    calls: list[str] = []
    orchestrator = _build_orchestrator(
        config_path,
        config,
        run_directory=run_dir,
        stage_behaviors=_make_stage_behaviors(),
        calls=calls,
    )

    result = orchestrator.run(run_dir=run_dir)

    assert result.is_success()
    assert calls == [
        StageNames.DATASET_VALIDATOR,
        StageNames.GPU_DEPLOYER,
        StageNames.TRAINING_MONITOR,
        StageNames.MODEL_RETRIEVER,
    ]

    state = PipelineStateStore(run_dir).load()
    assert len(state.attempts) == 1
    attempt = state.attempts[0]
    assert attempt.status == StageRunState.STATUS_COMPLETED
    assert state.pipeline_status == StageRunState.STATUS_COMPLETED
    assert state.active_attempt_id is None
    assert attempt.stage_runs[StageNames.INFERENCE_DEPLOYER].status == StageRunState.STATUS_SKIPPED
    assert attempt.stage_runs[StageNames.MODEL_EVALUATOR].status == StageRunState.STATUS_SKIPPED
    assert StageNames.MODEL_RETRIEVER in state.current_output_lineage


def test_stateful_resume_reuses_completed_stages_and_appends_attempt(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    config = _build_mock_config(inference_enabled=True, evaluation_enabled=True)

    first_calls: list[str] = []
    first_orchestrator = _build_orchestrator(
        config_path,
        config,
        run_directory=run_dir,
        stage_behaviors=_make_stage_behaviors(fail_stage=StageNames.INFERENCE_DEPLOYER),
        calls=first_calls,
    )
    first_result = first_orchestrator.run(run_dir=run_dir)

    assert first_result.is_failure()
    assert first_calls == [
        StageNames.DATASET_VALIDATOR,
        StageNames.GPU_DEPLOYER,
        StageNames.TRAINING_MONITOR,
        StageNames.MODEL_RETRIEVER,
        StageNames.INFERENCE_DEPLOYER,
    ]

    second_calls: list[str] = []
    second_orchestrator = _build_orchestrator(
        config_path,
        config,
        run_directory=run_dir,
        stage_behaviors=_make_stage_behaviors(),
        calls=second_calls,
    )

    with patch(
        "src.pipeline.executor.stage_planner.is_inference_runtime_healthy",
        return_value=True,
    ):
        second_result = second_orchestrator.run(run_dir=run_dir, resume=True)

    assert second_result.is_success()
    assert second_calls == [StageNames.INFERENCE_DEPLOYER, StageNames.MODEL_EVALUATOR]

    state = PipelineStateStore(run_dir).load()
    assert len(state.attempts) == 2
    assert state.attempts[0].status == StageRunState.STATUS_FAILED
    assert state.attempts[1].status == StageRunState.STATUS_COMPLETED
    assert state.pipeline_status == StageRunState.STATUS_COMPLETED
    assert state.active_attempt_id is None
    assert state.attempts[1].stage_runs[StageNames.MODEL_RETRIEVER].execution_mode == StageRunState.MODE_REUSED
    assert state.attempts[1].stage_runs[StageNames.INFERENCE_DEPLOYER].execution_mode == StageRunState.MODE_EXECUTED


def test_stateful_manual_restart_forces_disabled_inference_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    config = _build_mock_config(inference_enabled=False, evaluation_enabled=False)

    seed_calls: list[str] = []
    seed_orchestrator = _build_orchestrator(
        config_path,
        config,
        run_directory=run_dir,
        stage_behaviors=_make_stage_behaviors(),
        calls=seed_calls,
    )
    assert seed_orchestrator.run(run_dir=run_dir).is_success()

    restart_calls: list[str] = []
    restart_orchestrator = _build_orchestrator(
        config_path,
        config,
        run_directory=run_dir,
        stage_behaviors=_make_stage_behaviors(),
        calls=restart_calls,
    )
    restart_result = restart_orchestrator.run(run_dir=run_dir, restart_from_stage="Inference Deployer")

    assert restart_result.is_success()
    assert restart_calls == [StageNames.INFERENCE_DEPLOYER]

    state = PipelineStateStore(run_dir).load()
    assert len(state.attempts) == 2
    second_attempt = state.attempts[1]
    assert second_attempt.stage_runs[StageNames.INFERENCE_DEPLOYER].status == StageRunState.STATUS_COMPLETED
    assert second_attempt.stage_runs[StageNames.MODEL_EVALUATOR].status == StageRunState.STATUS_SKIPPED


def test_stateful_interrupt_marks_attempt_and_stage_as_interrupted(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    config = _build_mock_config(inference_enabled=False, evaluation_enabled=False)
    calls: list[str] = []
    orchestrator = _build_orchestrator(
        config_path,
        config,
        run_directory=run_dir,
        stage_behaviors=_make_stage_behaviors(interrupt_stage=StageNames.TRAINING_MONITOR),
        calls=calls,
    )

    result = orchestrator.run(run_dir=run_dir)

    assert result.is_failure()
    state = PipelineStateStore(run_dir).load()
    attempt = state.attempts[0]
    assert attempt.status == StageRunState.STATUS_INTERRUPTED
    assert attempt.stage_runs[StageNames.TRAINING_MONITOR].status == StageRunState.STATUS_INTERRUPTED
    assert state.pipeline_status == StageRunState.STATUS_INTERRUPTED
    assert state.active_attempt_id is None


def test_stateful_resume_with_corrupted_state_returns_pipeline_state_error(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pipeline_state.json").write_text("{broken json", encoding="utf-8")

    config = _build_mock_config(inference_enabled=False, evaluation_enabled=False)
    calls: list[str] = []
    orchestrator = _build_orchestrator(
        config_path,
        config,
        run_directory=run_dir,
        stage_behaviors=_make_stage_behaviors(),
        calls=calls,
    )

    result = orchestrator.run(run_dir=run_dir, resume=True)

    assert result.is_failure()
    assert result.unwrap_err().code == "PIPELINE_STATE_ERROR"  # type: ignore[union-attr]
    assert calls == []


def test_stateful_config_drift_creates_failed_attempt_with_visible_error(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    store = PipelineStateStore(run_dir)
    store.init_state(
        logical_run_id="logical_run_1",
        config_path=str(config_path),
        training_critical_config_hash="old_hash",
        late_stage_config_hash="late_hash",
    )

    config = _build_mock_config(inference_enabled=False, evaluation_enabled=False)
    calls: list[str] = []
    orchestrator = _build_orchestrator(
        config_path,
        config,
        run_directory=run_dir,
        stage_behaviors=_make_stage_behaviors(),
        calls=calls,
    )

    result = orchestrator.run(run_dir=run_dir, restart_from_stage=StageNames.MODEL_RETRIEVER)

    assert result.is_failure()
    assert result.unwrap_err().code == "CONFIG_DRIFT"  # type: ignore[union-attr]
    assert calls == []

    state = PipelineStateStore(run_dir).load()
    assert len(state.attempts) == 1
    attempt = state.attempts[0]
    assert attempt.status == StageRunState.STATUS_FAILED
    assert "training_critical config changed" in (attempt.error or "")
    assert state.pipeline_status == StageRunState.STATUS_FAILED
    assert state.active_attempt_id is None

    log_path = store.next_attempt_dir(attempt.attempt_no) / "logs" / "pipeline.log"
    assert log_path.exists()
    assert "Launch rejected before stage execution" in log_path.read_text(encoding="utf-8")


def test_stateful_run_fails_when_run_lock_is_already_held(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    config = _build_mock_config(inference_enabled=False, evaluation_enabled=False)
    calls: list[str] = []
    orchestrator = _build_orchestrator(
        config_path,
        config,
        run_directory=run_dir,
        stage_behaviors=_make_stage_behaviors(),
        calls=calls,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.lock").write_text("pid=123\n", encoding="utf-8")

    result = orchestrator.run(run_dir=run_dir)

    assert result.is_failure()
    assert "already locked" in str(result.unwrap_err())  # type: ignore[union-attr]
    assert calls == []


def test_stateful_dataset_validation_failure_persists_compact_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    run_dir = tmp_path / "runs" / "logical_run_1"
    config = _build_mock_config(inference_enabled=False, evaluation_enabled=False)
    calls: list[str] = []
    orchestrator = _build_orchestrator(
        config_path,
        config,
        run_directory=run_dir,
        stage_behaviors=_make_stage_behaviors(fail_stage=StageNames.DATASET_VALIDATOR),
        calls=calls,
    )
    orchestrator._validation_artifact_mgr._validation_accumulator = {
        "/data/train.jsonl": {
            "name": "default",
            "path": "/data/train.jsonl",
            "sample_count": 1,
            "status": "failed",
            "critical_failures": 1,
            "plugins": [],
        }
    }

    result = orchestrator.run(run_dir=run_dir)

    assert result.is_failure()
    state = PipelineStateStore(run_dir).load()
    attempt = state.attempts[0]
    stage_state = attempt.stage_runs[StageNames.DATASET_VALIDATOR]
    assert stage_state.status == StageRunState.STATUS_FAILED
    assert stage_state.outputs["validation_status"] == "failed"
    assert stage_state.outputs["datasets_validated"] == 1
    assert stage_state.outputs["datasets_failed"] == 1
    assert stage_state.outputs["failed_datasets"] == ["default"]
    assert stage_state.outputs["validation_artifact_ref"] == "dataset_validator_results.json"

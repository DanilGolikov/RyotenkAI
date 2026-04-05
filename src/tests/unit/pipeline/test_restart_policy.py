from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.state import PipelineState, StageRunState
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
    config.datasets = {}
    config.get_active_provider_name.return_value = "single_node"
    config.get_provider_config.return_value = {}
    config.experiment_tracking.mlflow = None
    config.inference.enabled = False
    config.inference.model_dump.return_value = {"enabled": False}
    config.evaluation.enabled = False
    config.evaluation.model_dump.return_value = {"enabled": False}
    return config


def _build_orchestrator(config_path: Path, config: MagicMock) -> PipelineOrchestrator:
    secrets = MagicMock()
    secrets.hf_token = "test-token"
    with (
        patch("src.pipeline.orchestrator.load_config", return_value=config),
        patch("src.pipeline.orchestrator.load_secrets", return_value=secrets),
        patch("src.pipeline.orchestrator.validate_strategy_chain", return_value=Ok(None)),
        patch("src.pipeline.orchestrator.DatasetValidator"),
        patch("src.pipeline.orchestrator.GPUDeployer"),
        patch("src.pipeline.orchestrator.TrainingMonitor"),
        patch("src.pipeline.orchestrator.ModelRetriever"),
        patch("src.pipeline.orchestrator.InferenceDeployer"),
        patch("src.pipeline.orchestrator.ModelEvaluator"),
    ):
        return PipelineOrchestrator(config_path)


def test_forced_disabled_inference_stage_is_enabled_for_manual_restart(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    orchestrator = _build_orchestrator(config_path, _build_mock_config())

    enabled = orchestrator._compute_enabled_stage_names(start_stage_name=StageNames.INFERENCE_DEPLOYER)

    assert StageNames.INFERENCE_DEPLOYER in enabled
    assert StageNames.MODEL_EVALUATOR not in enabled


def test_late_stage_config_drift_allowed_only_for_late_stages(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: gpt2\n")
    orchestrator = _build_orchestrator(config_path, _build_mock_config())

    state = PipelineState(
        schema_version=1,
        logical_run_id="logical_run_1",
        run_directory=str(tmp_path / "runs" / "logical_run_1"),
        config_path=str(config_path),
        active_attempt_id=None,
        pipeline_status=StageRunState.STATUS_FAILED,
        training_critical_config_hash="train_hash",
        late_stage_config_hash="old_late_hash",
        attempts=[],
        current_output_lineage={},
    )

    drift_error = orchestrator._validate_config_drift(
        state=state,
        start_stage_name=StageNames.INFERENCE_DEPLOYER,
        config_hashes={"training_critical": "train_hash", "late_stage": "new_late_hash", "model_dataset": "md_hash"},
        resume=False,
    )
    assert drift_error is None

    drift_error = orchestrator._validate_config_drift(
        state=state,
        start_stage_name=StageNames.MODEL_RETRIEVER,
        config_hashes={"training_critical": "train_hash", "late_stage": "new_late_hash", "model_dataset": "md_hash"},
        resume=False,
    )
    assert drift_error is not None
    assert "late_stage config changed" in drift_error.message


"""Unit tests for StageInfoLogger.

Verify that each stage type emits the correct MLflow calls with the correct
arguments, plus edge cases (no manager, missing context, non-dict context,
legacy vs plugin validation modes).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipeline.context.stage_info_logger import StageInfoLogger
from src.pipeline.stages import StageNames


@pytest.fixture
def logger_under_test() -> StageInfoLogger:
    return StageInfoLogger()


@pytest.fixture
def mgr() -> MagicMock:
    return MagicMock()


# -----------------------------------------------------------------------------
# No-op paths
# -----------------------------------------------------------------------------


def test_no_op_when_manager_is_none(logger_under_test: StageInfoLogger) -> None:
    logger_under_test.log(
        mlflow_manager=None,
        context={StageNames.GPU_DEPLOYER: {"provider_name": "x"}},
        stage_name=StageNames.GPU_DEPLOYER,
    )
    # No crash, nothing to assert


def test_no_op_when_stage_not_in_context(logger_under_test: StageInfoLogger, mgr: MagicMock) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={},
        stage_name=StageNames.GPU_DEPLOYER,
    )
    mgr.log_provider_info.assert_not_called()
    mgr.log_event_info.assert_not_called()


def test_unknown_stage_is_skipped(logger_under_test: StageInfoLogger, mgr: MagicMock) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={"SomeStage": {"key": "val"}},
        stage_name="SomeStage",
    )
    mgr.log_provider_info.assert_not_called()
    mgr.log_params.assert_not_called()


def test_non_dict_context_is_skipped(logger_under_test: StageInfoLogger, mgr: MagicMock) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={StageNames.GPU_DEPLOYER: "garbage"},
        stage_name=StageNames.GPU_DEPLOYER,
    )
    mgr.log_provider_info.assert_not_called()


# -----------------------------------------------------------------------------
# GPU Deployer
# -----------------------------------------------------------------------------


def test_gpu_deployer_logs_provider_info_and_durations(
    logger_under_test: StageInfoLogger, mgr: MagicMock
) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={
            StageNames.GPU_DEPLOYER: {
                "provider_name": "runpod",
                "provider_type": "cloud",
                "gpu_type": "A100",
                "resource_id": "r1",
                "upload_duration_seconds": 12.3,
                "deps_duration_seconds": 4.5,
            }
        },
        stage_name=StageNames.GPU_DEPLOYER,
    )
    mgr.log_provider_info.assert_called_once_with(
        provider_name="runpod",
        provider_type="cloud",
        gpu_type="A100",
        resource_id="r1",
    )
    assert mgr.log_event_info.call_count == 2


def test_gpu_deployer_skips_missing_durations(
    logger_under_test: StageInfoLogger, mgr: MagicMock
) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={
            StageNames.GPU_DEPLOYER: {
                "provider_name": "runpod",
                "provider_type": "cloud",
            }
        },
        stage_name=StageNames.GPU_DEPLOYER,
    )
    mgr.log_provider_info.assert_called_once()
    mgr.log_event_info.assert_not_called()


def test_gpu_deployer_uses_unknown_defaults(
    logger_under_test: StageInfoLogger, mgr: MagicMock
) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={StageNames.GPU_DEPLOYER: {}},
        stage_name=StageNames.GPU_DEPLOYER,
    )
    mgr.log_provider_info.assert_called_once()
    args, kwargs = mgr.log_provider_info.call_args
    assert kwargs["provider_name"]  # unknown-default (non-empty)
    assert kwargs["provider_type"]  # unknown-default


# -----------------------------------------------------------------------------
# Dataset Validator
# -----------------------------------------------------------------------------


def test_dataset_validator_plugin_mode_logs_prefixed_metrics(
    logger_under_test: StageInfoLogger, mgr: MagicMock
) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={
            StageNames.DATASET_VALIDATOR: {
                "validation_mode": "plugin",
                "metrics": {"sample_count": 1000, "avg_length": 45.2, "source": "jsonl"},
            }
        },
        stage_name=StageNames.DATASET_VALIDATOR,
    )
    mgr.log_params.assert_called_once()
    params = mgr.log_params.call_args.args[0]
    assert params["dataset.sample_count"] == 1000.0
    assert params["dataset.avg_length"] == 45.2
    assert params["dataset.source"] == "jsonl"  # non-numeric becomes str


def test_dataset_validator_legacy_mode_fixed_params(
    logger_under_test: StageInfoLogger, mgr: MagicMock
) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={
            StageNames.DATASET_VALIDATOR: {
                "sample_count": 1000,
                "metrics": {"avg_length": 50.0, "empty_ratio": 0.01, "diversity_score": 0.9},
            }
        },
        stage_name=StageNames.DATASET_VALIDATOR,
    )
    mgr.log_params.assert_called_once_with(
        {
            "dataset.sample_count": 1000,
            "dataset.avg_length": 50.0,
            "dataset.empty_ratio": 0.01,
            "dataset.diversity_score": 0.9,
        }
    )


def test_dataset_validator_no_metrics_is_skipped(
    logger_under_test: StageInfoLogger, mgr: MagicMock
) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={StageNames.DATASET_VALIDATOR: {"metrics": {}}},
        stage_name=StageNames.DATASET_VALIDATOR,
    )
    mgr.log_params.assert_not_called()


# -----------------------------------------------------------------------------
# Training Monitor
# -----------------------------------------------------------------------------


def test_training_monitor_logs_duration_event(
    logger_under_test: StageInfoLogger, mgr: MagicMock
) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={
            StageNames.TRAINING_MONITOR: {
                "training_duration_seconds": 600.5,
            }
        },
        stage_name=StageNames.TRAINING_MONITOR,
    )
    mgr.log_event_info.assert_called_once()
    assert "600.5s" in mgr.log_event_info.call_args.args[0]


def test_training_monitor_logs_training_info_metrics(
    logger_under_test: StageInfoLogger, mgr: MagicMock
) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={
            StageNames.TRAINING_MONITOR: {
                "training_info": {
                    "runtime_seconds": 500,
                    "final_loss": 0.05,
                    "final_accuracy": 0.95,
                    "total_steps": 100,
                }
            }
        },
        stage_name=StageNames.TRAINING_MONITOR,
    )
    mgr.log_metrics.assert_called_once()
    metrics = mgr.log_metrics.call_args.args[0]
    assert metrics["training.runtime_seconds"] == 500
    assert metrics["training.final_loss"] == 0.05
    assert metrics["training.final_accuracy"] == 0.95
    assert metrics["training.total_steps"] == 100.0


def test_training_monitor_empty_info_skips_metrics(
    logger_under_test: StageInfoLogger, mgr: MagicMock
) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={StageNames.TRAINING_MONITOR: {"training_info": {}}},
        stage_name=StageNames.TRAINING_MONITOR,
    )
    mgr.log_metrics.assert_not_called()


# -----------------------------------------------------------------------------
# Model Retriever
# -----------------------------------------------------------------------------


def test_model_retriever_logs_model_size_event(
    logger_under_test: StageInfoLogger, mgr: MagicMock
) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={StageNames.MODEL_RETRIEVER: {"model_size_mb": 1500.0}},
        stage_name=StageNames.MODEL_RETRIEVER,
    )
    mgr.log_event_info.assert_called_once()
    assert "1500.0 MB" in mgr.log_event_info.call_args.args[0]


def test_model_retriever_logs_upload_when_hf_uploaded(
    logger_under_test: StageInfoLogger, mgr: MagicMock
) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={
            StageNames.MODEL_RETRIEVER: {
                "model_size_mb": 100.0,
                "hf_uploaded": True,
                "upload_duration_seconds": 10.0,
                "hf_repo_id": "org/m",
            }
        },
        stage_name=StageNames.MODEL_RETRIEVER,
    )
    # Two events: model size + hf upload
    assert mgr.log_event_info.call_count == 2
    upload_call = mgr.log_event_info.call_args_list[1]
    assert "org/m" in upload_call.args[0]


def test_model_retriever_skip_upload_when_not_uploaded(
    logger_under_test: StageInfoLogger, mgr: MagicMock
) -> None:
    logger_under_test.log(
        mlflow_manager=mgr,
        context={
            StageNames.MODEL_RETRIEVER: {
                "model_size_mb": 100.0,
                "hf_uploaded": False,
                "upload_duration_seconds": 5.0,
            }
        },
        stage_name=StageNames.MODEL_RETRIEVER,
    )
    assert mgr.log_event_info.call_count == 1  # only model_size

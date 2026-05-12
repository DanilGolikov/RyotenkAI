"""L2 component tests for :class:`StageInfoLogger` using FakeMLflowManager.

Demonstrates the canonical-fakes pattern: stateless logger gets a
fully working in-memory IMLflowManager, drives a per-stage call, and
verifies side effects on the fake's recorded run.
"""

from __future__ import annotations

import pytest

from ryotenkai_control.pipeline.context.stage_info_logger import StageInfoLogger
from ryotenkai_control.pipeline.stages import StageNames
from tests._fakes.mlflow import FakeMLflowManager

pytestmark = [
    pytest.mark.component,
    pytest.mark.uses_fake("FakeMLflowManager"),
    pytest.mark.exercises_protocol("IMLflowManager"),
]


@pytest.fixture
def fake_mlflow(manual_clock) -> FakeMLflowManager:
    fake = FakeMLflowManager(clock=manual_clock)
    fake.setup()
    fake.start_run("stage-info-logger-test")
    return fake


@pytest.fixture
def logger() -> StageInfoLogger:
    return StageInfoLogger()


class TestStageInfoLoggerGpuDeployer:
    def test_logs_provider_info_with_full_context(
        self, logger: StageInfoLogger, fake_mlflow: FakeMLflowManager,
    ) -> None:
        context = {
            StageNames.GPU_DEPLOYER: {
                "provider_name": "runpod",
                "provider_type": "cloud",
                "gpu_type": "A100",
                "resource_id": "pod-123",
                "upload_duration_seconds": 12.5,
                "deps_duration_seconds": 30.0,
            },
        }
        logger.log(
            mlflow_manager=fake_mlflow,
            context=context,
            stage_name=StageNames.GPU_DEPLOYER,
        )
        run_id = fake_mlflow.runs_for("stage-info-logger-test")[0]
        tags = fake_mlflow.get_run(run_id).tags
        assert tags["provider.name"] == "runpod"
        assert tags["provider.gpu_type"] == "A100"
        assert tags["provider.resource_id"] == "pod-123"
        # Two events emitted: one for upload duration, one for deps.
        assert tags["event.info.last"].startswith("Dependencies installed")

    def test_zero_upload_duration_still_emits_event(
        self, logger: StageInfoLogger, fake_mlflow: FakeMLflowManager,
    ) -> None:
        # Boundary: upload_duration_seconds=0 is a valid (instant) upload —
        # the comment in stage_info_logger.py explicitly forbids treating
        # it as "missing".
        context = {
            StageNames.GPU_DEPLOYER: {
                "provider_name": "single_node",
                "provider_type": "host",
                "upload_duration_seconds": 0.0,
            },
        }
        logger.log(
            mlflow_manager=fake_mlflow,
            context=context,
            stage_name=StageNames.GPU_DEPLOYER,
        )
        run_id = fake_mlflow.runs_for("stage-info-logger-test")[0]
        tags = fake_mlflow.get_run(run_id).tags
        assert "Files uploaded" in tags["event.info.last"]

    def test_no_op_when_mlflow_manager_is_none(self, logger: StageInfoLogger) -> None:
        # Negative path: ``mlflow_manager=None`` must short-circuit.
        # Without a fake to assert against, the test relies on no exception.
        logger.log(
            mlflow_manager=None,
            context={StageNames.GPU_DEPLOYER: {"provider_name": "x", "provider_type": "y"}},
            stage_name=StageNames.GPU_DEPLOYER,
        )

    def test_unknown_stage_does_not_call_mlflow(
        self, logger: StageInfoLogger, fake_mlflow: FakeMLflowManager,
    ) -> None:
        # Negative path: an unrecognised stage name produces no log calls.
        snapshot_before = fake_mlflow.snapshot()
        logger.log(
            mlflow_manager=fake_mlflow,
            context={"NotARealStage": {"foo": "bar"}},
            stage_name="NotARealStage",
        )
        # Run state untouched — no params / events added.
        run_id = fake_mlflow.runs_for("stage-info-logger-test")[0]
        run_after = fake_mlflow.snapshot()["runs"][run_id]
        assert run_after == snapshot_before["runs"][run_id]


class TestStageInfoLoggerDatasetValidator:
    def test_legacy_metrics_logged_as_params(
        self, logger: StageInfoLogger, fake_mlflow: FakeMLflowManager,
    ) -> None:
        context = {
            StageNames.DATASET_VALIDATOR: {
                "validation_mode": "legacy",
                "sample_count": 1000,
                "metrics": {"avg_length": 50.5, "empty_ratio": 0.05, "diversity_score": 0.9},
            },
        }
        logger.log(
            mlflow_manager=fake_mlflow,
            context=context,
            stage_name=StageNames.DATASET_VALIDATOR,
        )
        run_id = fake_mlflow.runs_for("stage-info-logger-test")[0]
        params = fake_mlflow.get_run(run_id).params
        assert params["dataset.sample_count"] == 1000
        assert params["dataset.avg_length"] == 50.5
        assert params["dataset.diversity_score"] == 0.9

    def test_plugin_metrics_coerce_to_float_or_str(
        self, logger: StageInfoLogger, fake_mlflow: FakeMLflowManager,
    ) -> None:
        context = {
            StageNames.DATASET_VALIDATOR: {
                "validation_mode": "plugin",
                "metrics": {"score": "0.95", "label": "good"},
            },
        }
        logger.log(
            mlflow_manager=fake_mlflow,
            context=context,
            stage_name=StageNames.DATASET_VALIDATOR,
        )
        run_id = fake_mlflow.runs_for("stage-info-logger-test")[0]
        params = fake_mlflow.get_run(run_id).params
        assert params["dataset.score"] == pytest.approx(0.95)
        assert params["dataset.label"] == "good"

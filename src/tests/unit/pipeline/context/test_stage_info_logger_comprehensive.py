"""Comprehensive tests for StageInfoLogger.

Focuses particularly on the falsy-value regressions (loss=0 / accuracy=0 /
duration=0) that the pre-refactor code silently dropped.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipeline.context.stage_info_logger import StageInfoLogger
from src.pipeline.stages import StageNames


@pytest.fixture
def slog() -> StageInfoLogger:
    return StageInfoLogger()


@pytest.fixture
def mgr() -> MagicMock:
    return MagicMock()


# =============================================================================
# 1. POSITIVE
# =============================================================================


class TestPositive:
    def test_gpu_deployer_happy(self, slog: StageInfoLogger, mgr: MagicMock) -> None:
        slog.log(
            mlflow_manager=mgr,
            context={
                StageNames.GPU_DEPLOYER: {
                    "provider_name": "runpod",
                    "provider_type": "cloud",
                    "upload_duration_seconds": 10.0,
                    "deps_duration_seconds": 5.0,
                }
            },
            stage_name=StageNames.GPU_DEPLOYER,
        )
        mgr.log_provider_info.assert_called_once()
        assert mgr.log_event_info.call_count == 2

    def test_training_monitor_full_info(self, slog: StageInfoLogger, mgr: MagicMock) -> None:
        slog.log(
            mlflow_manager=mgr,
            context={
                StageNames.TRAINING_MONITOR: {
                    "training_duration_seconds": 60.0,
                    "training_info": {
                        "runtime_seconds": 50,
                        "final_loss": 0.1,
                        "final_accuracy": 0.9,
                        "total_steps": 100,
                    },
                }
            },
            stage_name=StageNames.TRAINING_MONITOR,
        )
        mgr.log_event_info.assert_called_once()
        mgr.log_metrics.assert_called_once()
        metrics = mgr.log_metrics.call_args.args[0]
        assert len(metrics) == 4


# =============================================================================
# 2. NEGATIVE
# =============================================================================


class TestNegative:
    def test_none_manager_no_op(self, slog: StageInfoLogger) -> None:
        slog.log(mlflow_manager=None, context={}, stage_name=StageNames.GPU_DEPLOYER)

    def test_stage_missing_in_context(self, slog: StageInfoLogger, mgr: MagicMock) -> None:
        slog.log(mlflow_manager=mgr, context={}, stage_name=StageNames.GPU_DEPLOYER)
        mgr.log_provider_info.assert_not_called()

    def test_context_value_wrong_type(self, slog: StageInfoLogger, mgr: MagicMock) -> None:
        for bad in [None, "string", 42, [1, 2, 3]]:
            mgr.reset_mock()
            slog.log(
                mlflow_manager=mgr,
                context={StageNames.GPU_DEPLOYER: bad},  # type: ignore[dict-item]
                stage_name=StageNames.GPU_DEPLOYER,
            )
            mgr.log_provider_info.assert_not_called()

    def test_unknown_stage_name_no_op(self, slog: StageInfoLogger, mgr: MagicMock) -> None:
        slog.log(
            mlflow_manager=mgr,
            context={"Unknown": {"foo": "bar"}},
            stage_name="Unknown",
        )
        mgr.log_provider_info.assert_not_called()
        mgr.log_params.assert_not_called()


# =============================================================================
# 3. BOUNDARY
# =============================================================================


class TestBoundary:
    def test_gpu_deployer_with_empty_context_dict(self, slog: StageInfoLogger, mgr: MagicMock) -> None:
        slog.log(
            mlflow_manager=mgr,
            context={StageNames.GPU_DEPLOYER: {}},
            stage_name=StageNames.GPU_DEPLOYER,
        )
        # Provider info still logged (with unknown defaults)
        mgr.log_provider_info.assert_called_once()
        # No event durations
        mgr.log_event_info.assert_not_called()

    def test_dataset_validator_metrics_dict_is_empty(self, slog: StageInfoLogger, mgr: MagicMock) -> None:
        slog.log(
            mlflow_manager=mgr,
            context={StageNames.DATASET_VALIDATOR: {"metrics": {}}},
            stage_name=StageNames.DATASET_VALIDATOR,
        )
        mgr.log_params.assert_not_called()

    def test_training_monitor_only_duration_no_info(self, slog: StageInfoLogger, mgr: MagicMock) -> None:
        slog.log(
            mlflow_manager=mgr,
            context={StageNames.TRAINING_MONITOR: {"training_duration_seconds": 123.0}},
            stage_name=StageNames.TRAINING_MONITOR,
        )
        mgr.log_event_info.assert_called_once()
        mgr.log_metrics.assert_not_called()

    def test_training_info_empty_dict(self, slog: StageInfoLogger, mgr: MagicMock) -> None:
        slog.log(
            mlflow_manager=mgr,
            context={StageNames.TRAINING_MONITOR: {"training_info": {}}},
            stage_name=StageNames.TRAINING_MONITOR,
        )
        mgr.log_metrics.assert_not_called()


# =============================================================================
# 4. INVARIANTS
# =============================================================================


class TestInvariants:
    def test_log_never_raises_on_shape_variations(self, slog: StageInfoLogger, mgr: MagicMock) -> None:
        """Invariant: log() tolerates any context shape without exceptions."""
        for ctx in [
            {},
            {StageNames.GPU_DEPLOYER: None},
            {StageNames.GPU_DEPLOYER: []},
            {StageNames.TRAINING_MONITOR: {"training_info": "not a dict"}},
            {StageNames.MODEL_RETRIEVER: {"hf_uploaded": None}},
        ]:
            slog.log(mlflow_manager=mgr, context=ctx, stage_name=StageNames.GPU_DEPLOYER)

    def test_log_idempotent_wrt_context(self, slog: StageInfoLogger, mgr: MagicMock) -> None:
        """Invariant: log() does not mutate the provided context dict."""
        ctx = {StageNames.GPU_DEPLOYER: {"provider_name": "x", "provider_type": "local"}}
        snapshot = dict(ctx[StageNames.GPU_DEPLOYER])
        slog.log(mlflow_manager=mgr, context=ctx, stage_name=StageNames.GPU_DEPLOYER)
        assert ctx[StageNames.GPU_DEPLOYER] == snapshot


# =============================================================================
# 5. DEPENDENCY ERRORS
# =============================================================================


class TestDependencyErrors:
    def test_log_provider_info_failure_bubbles_up(self, slog: StageInfoLogger) -> None:
        """If MLflow manager errors out, the logger does NOT swallow it —
        the orchestrator decides what to do."""
        mgr = MagicMock()
        mgr.log_provider_info.side_effect = RuntimeError("mlflow down")
        with pytest.raises(RuntimeError, match="mlflow down"):
            slog.log(
                mlflow_manager=mgr,
                context={StageNames.GPU_DEPLOYER: {"provider_name": "x"}},
                stage_name=StageNames.GPU_DEPLOYER,
            )

    def test_log_params_mlflow_exception_propagates(self, slog: StageInfoLogger) -> None:
        mgr = MagicMock()
        mgr.log_params.side_effect = RuntimeError("params error")
        with pytest.raises(RuntimeError):
            slog.log(
                mlflow_manager=mgr,
                context={
                    StageNames.DATASET_VALIDATOR: {
                        "metrics": {"avg_length": 1},
                        "validation_mode": "legacy",
                    }
                },
                stage_name=StageNames.DATASET_VALIDATOR,
            )


# =============================================================================
# 6. REGRESSIONS — falsy-value bugs the review caught
# =============================================================================


class TestRegressions:
    """These tests would all have FAILED with the pre-fix code."""

    def test_regression_training_duration_zero_still_logged(
        self, slog: StageInfoLogger, mgr: MagicMock
    ) -> None:
        """REGRESSION: training_duration=0.0 used to be dropped by ``if training_dur:``."""
        slog.log(
            mlflow_manager=mgr,
            context={StageNames.TRAINING_MONITOR: {"training_duration_seconds": 0.0}},
            stage_name=StageNames.TRAINING_MONITOR,
        )
        mgr.log_event_info.assert_called_once()

    def test_regression_final_loss_zero_still_logged(
        self, slog: StageInfoLogger, mgr: MagicMock
    ) -> None:
        """REGRESSION: final_loss=0.0 means 'perfect convergence' — must still be logged."""
        slog.log(
            mlflow_manager=mgr,
            context={StageNames.TRAINING_MONITOR: {"training_info": {"final_loss": 0.0}}},
            stage_name=StageNames.TRAINING_MONITOR,
        )
        mgr.log_metrics.assert_called_once()
        metrics = mgr.log_metrics.call_args.args[0]
        assert metrics["training.final_loss"] == 0.0

    def test_regression_final_accuracy_zero_still_logged(
        self, slog: StageInfoLogger, mgr: MagicMock
    ) -> None:
        """REGRESSION: accuracy=0.0 (classification failure) must be visible in MLflow."""
        slog.log(
            mlflow_manager=mgr,
            context={StageNames.TRAINING_MONITOR: {"training_info": {"final_accuracy": 0.0}}},
            stage_name=StageNames.TRAINING_MONITOR,
        )
        metrics = mgr.log_metrics.call_args.args[0]
        assert metrics["training.final_accuracy"] == 0.0

    def test_regression_total_steps_zero_still_logged(
        self, slog: StageInfoLogger, mgr: MagicMock
    ) -> None:
        """REGRESSION: total_steps=0 (no-op phase) — logged as 0.0."""
        slog.log(
            mlflow_manager=mgr,
            context={StageNames.TRAINING_MONITOR: {"training_info": {"total_steps": 0}}},
            stage_name=StageNames.TRAINING_MONITOR,
        )
        metrics = mgr.log_metrics.call_args.args[0]
        assert metrics["training.total_steps"] == 0.0

    def test_regression_runtime_seconds_zero_still_logged(
        self, slog: StageInfoLogger, mgr: MagicMock
    ) -> None:
        """REGRESSION: runtime_seconds=0 (cached run) — logged."""
        slog.log(
            mlflow_manager=mgr,
            context={StageNames.TRAINING_MONITOR: {"training_info": {"runtime_seconds": 0}}},
            stage_name=StageNames.TRAINING_MONITOR,
        )
        metrics = mgr.log_metrics.call_args.args[0]
        assert metrics["training.runtime_seconds"] == 0

    def test_regression_model_size_zero_still_logged(
        self, slog: StageInfoLogger, mgr: MagicMock
    ) -> None:
        """REGRESSION: model_size=0 (empty model, test edge case) — emits event."""
        slog.log(
            mlflow_manager=mgr,
            context={StageNames.MODEL_RETRIEVER: {"model_size_mb": 0.0}},
            stage_name=StageNames.MODEL_RETRIEVER,
        )
        mgr.log_event_info.assert_called_once()

    def test_regression_upload_duration_zero_still_logged(
        self, slog: StageInfoLogger, mgr: MagicMock
    ) -> None:
        """REGRESSION: upload_duration=0 (local-copy shortcut) — emits event."""
        slog.log(
            mlflow_manager=mgr,
            context={
                StageNames.GPU_DEPLOYER: {
                    "provider_name": "local",
                    "upload_duration_seconds": 0.0,
                }
            },
            stage_name=StageNames.GPU_DEPLOYER,
        )
        # One provider_info + one upload event (deps skipped → None)
        assert mgr.log_event_info.call_count == 1

    def test_regression_hf_upload_event_zero_duration(
        self, slog: StageInfoLogger, mgr: MagicMock
    ) -> None:
        """REGRESSION: HF upload with 0s duration (instant cache-hit) — emits event."""
        slog.log(
            mlflow_manager=mgr,
            context={
                StageNames.MODEL_RETRIEVER: {
                    "hf_uploaded": True,
                    "upload_duration_seconds": 0.0,
                    "hf_repo_id": "org/m",
                }
            },
            stage_name=StageNames.MODEL_RETRIEVER,
        )
        # model_size event NOT logged (model_size_mb is None), but HF upload IS.
        assert mgr.log_event_info.call_count == 1
        assert "org/m" in mgr.log_event_info.call_args.args[0]


# =============================================================================
# 7. COMBINATORIAL
# =============================================================================


# Cross-product: (stage_name, context_shape, expects_mlflow_call_count)
@pytest.mark.parametrize(
    ("stage_name", "ctx", "expected_log_calls"),
    [
        # Positive: dataset validator, plugin vs legacy mode
        (
            StageNames.DATASET_VALIDATOR,
            {
                StageNames.DATASET_VALIDATOR: {
                    "metrics": {"avg_length": 10, "count": 1000},
                    "validation_mode": "plugin",
                }
            },
            {"log_params": 1},
        ),
        (
            StageNames.DATASET_VALIDATOR,
            {
                StageNames.DATASET_VALIDATOR: {
                    "sample_count": 1000,
                    "metrics": {"avg_length": 10},
                }
            },
            {"log_params": 1},
        ),
        # Negative: empty metrics
        (
            StageNames.DATASET_VALIDATOR,
            {StageNames.DATASET_VALIDATOR: {"metrics": {}}},
            {"log_params": 0},
        ),
        # Boundary: GPU deployer with no durations
        (
            StageNames.GPU_DEPLOYER,
            {StageNames.GPU_DEPLOYER: {"provider_name": "x"}},
            {"log_event_info": 0, "log_provider_info": 1},
        ),
        # Model retriever without HF upload
        (
            StageNames.MODEL_RETRIEVER,
            {StageNames.MODEL_RETRIEVER: {"model_size_mb": 100.0, "hf_uploaded": False}},
            {"log_event_info": 1},
        ),
    ],
)
def test_combinatorial_mlflow_call_counts(
    slog: StageInfoLogger,
    mgr: MagicMock,
    stage_name: str,
    ctx: dict,
    expected_log_calls: dict[str, int],
) -> None:
    slog.log(mlflow_manager=mgr, context=ctx, stage_name=stage_name)
    for method, count in expected_log_calls.items():
        actual = getattr(mgr, method).call_count
        assert actual == count, f"{method}: expected {count}, got {actual}"

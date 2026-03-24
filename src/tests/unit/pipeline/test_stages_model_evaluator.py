"""
Unit tests for the new ModelEvaluator stage.

Tests the new plugin-based evaluation architecture.
Old perplexity/generation tests are removed (that logic was deleted).
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.stages.model_evaluator import ModelEvaluator


def _mk_cfg(*, eval_enabled: bool = True) -> MagicMock:
    """Create a minimal PipelineConfig mock for ModelEvaluator tests."""
    cfg = MagicMock()
    cfg.model.name = "test-model"

    eval_cfg = MagicMock()
    eval_cfg.enabled = eval_enabled

    cfg.evaluation = eval_cfg
    cfg.inference.engine = "vllm"
    cfg.experiment_tracking.mlflow = None  # MLflow disabled

    return cfg


class TestModelEvaluatorSkipCases:
    """Tests for cases where evaluation is skipped."""

    def test_skip_when_evaluation_disabled(self) -> None:
        """When evaluation.enabled=false, stage returns Ok with skipped=True."""
        stage = ModelEvaluator(_mk_cfg(eval_enabled=False))
        res = cast("Any", stage.execute({}))
        assert res.is_success()
        out = res.unwrap()["Model Evaluator"]
        assert out["evaluation_skipped"] is True
        assert "evaluation.enabled=false" in out["reason"]

    def test_skip_when_no_endpoint_url_in_context(self) -> None:
        """When no endpoint_url in context, evaluation is skipped (provider doesn't support eval)."""
        stage = ModelEvaluator(_mk_cfg(eval_enabled=True))
        res = cast("Any", stage.execute({}))
        assert res.is_success()
        out = res.unwrap()["Model Evaluator"]
        assert out["evaluation_skipped"] is True
        assert "endpoint_url not available" in out["reason"]

    def test_skip_when_eval_config_is_none(self) -> None:
        """When config.evaluation is None, stage returns skipped."""
        cfg = MagicMock()
        cfg.model.name = "test-model"
        cfg.evaluation = None
        cfg.experiment_tracking.mlflow = None

        stage = ModelEvaluator(cfg)
        res = cast("Any", stage.execute({}))
        assert res.is_success()
        out = res.unwrap()["Model Evaluator"]
        assert out["evaluation_skipped"] is True


class TestModelEvaluatorHappyPath:
    """Tests for successful evaluation flow."""

    def test_run_succeeds_with_passing_summary(self) -> None:
        """When EvaluationRunner returns passed=True, stage returns success with metrics."""
        from src.evaluation.runner import RunSummary

        mock_summary = RunSummary(
            overall_passed=True,
            sample_count=10,
            duration_seconds=1.5,
        )
        mock_summary.plugin_results = {
            "helixql_syntax": MagicMock(
                passed=True,
                metrics={"valid_ratio": 0.9, "valid_count": 9, "total_count": 10},
            )
        }

        stage = ModelEvaluator(_mk_cfg(eval_enabled=True))

        with (
            patch("src.evaluation.runner.EvaluationRunner") as MockRunner,
            patch("src.evaluation.model_client.factory.ModelClientFactory.create", return_value=MagicMock()),
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.return_value = mock_summary
            MockRunner.return_value = mock_runner_instance

            context: dict[str, Any] = {
                "endpoint_url": "http://127.0.0.1:8000/v1",
                "inference_model_name": "test-model",
            }
            res = cast("Any", stage.execute(context))

        assert res.is_success()
        out = res.unwrap()["Model Evaluator"]
        assert out["eval_passed"] is True
        assert "eval_summary" in out

    def test_run_with_failing_summary_still_returns_ok(self) -> None:
        """Even when evaluation fails, stage returns Ok (failure is in eval_passed)."""
        from src.evaluation.runner import RunSummary

        mock_summary = RunSummary(
            overall_passed=False,
            sample_count=5,
            duration_seconds=0.8,
        )
        mock_summary.plugin_results = {
            "helixql_syntax": MagicMock(
                passed=False,
                metrics={"valid_ratio": 0.4, "valid_count": 2, "total_count": 5},
                errors=["valid_ratio below threshold"],
                recommendations=["Review training data"],
            )
        }

        stage = ModelEvaluator(_mk_cfg(eval_enabled=True))

        with (
            patch("src.evaluation.runner.EvaluationRunner") as MockRunner,
            patch("src.evaluation.model_client.factory.ModelClientFactory.create", return_value=MagicMock()),
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.return_value = mock_summary
            MockRunner.return_value = mock_runner_instance

            context: dict[str, Any] = {
                "endpoint_url": "http://127.0.0.1:8000/v1",
            }
            res = cast("Any", stage.execute(context))

        assert res.is_success()
        out = res.unwrap()["Model Evaluator"]
        assert out["eval_passed"] is False

    def test_mlflow_metrics_are_flattened_correctly(self) -> None:
        """MLflow metrics follow eval.{plugin_name}.{metric_key} naming convention."""
        from src.evaluation.plugins.base import EvalResult
        from src.evaluation.runner import RunSummary

        plugin_result = EvalResult(
            plugin_name="helixql_syntax",
            passed=True,
            metrics={"valid_ratio": 0.9, "valid_count": 9, "total_count": 10},
            sample_count=10,
        )

        mock_summary = RunSummary(overall_passed=True, sample_count=10)
        mock_summary.plugin_results = {"helixql_syntax": plugin_result}

        metrics = ModelEvaluator._build_mlflow_metrics(mock_summary.plugin_results)

        assert "eval.helixql_syntax.valid_ratio" in metrics
        assert metrics["eval.helixql_syntax.valid_ratio"] == pytest.approx(0.9)
        assert "eval.helixql_syntax.passed" in metrics
        assert metrics["eval.helixql_syntax.passed"] == 1.0

    def test_uses_model_name_from_config_when_not_in_context(self) -> None:
        """Uses config.model.name when inference_model_name is not in context."""
        from src.evaluation.runner import RunSummary

        mock_summary = RunSummary(overall_passed=True, sample_count=0)

        stage = ModelEvaluator(_mk_cfg(eval_enabled=True))

        captured_model: list[str] = []

        def fake_client_init(base_url, model, **kwargs):
            captured_model.append(model)
            return MagicMock()

        with (
            patch("src.evaluation.runner.EvaluationRunner") as MockRunner,
            patch(
                "src.evaluation.model_client.factory.ModelClientFactory.create",
                side_effect=lambda engine, base_url, model, **kw: fake_client_init(base_url, model, **kw),
            ),
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run.return_value = mock_summary
            MockRunner.return_value = mock_runner_instance

            context: dict[str, Any] = {
                "endpoint_url": "http://127.0.0.1:8000/v1",
                # no inference_model_name
            }
            stage.execute(context)

        assert len(captured_model) == 1
        assert captured_model[0] == "test-model"

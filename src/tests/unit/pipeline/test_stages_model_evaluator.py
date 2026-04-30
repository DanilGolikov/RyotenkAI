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


@pytest.fixture
def mock_preflight_ok():
    """Stub the pre-flight ping so happy-path tests don't need a live HTTP server."""
    from src.utils.result import Ok

    with patch(
        "src.pipeline.stages.model_evaluator._preflight_check_endpoint",
        return_value=Ok(None),
    ) as m:
        yield m


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

    def test_endpoint_url_missing_returns_err(self) -> None:
        """When no endpoint_url in context, stage now returns Err — no
        more silent skip. Prior behaviour produced "successful" eval
        runs with empty answers; see plan §1."""
        stage = ModelEvaluator(_mk_cfg(eval_enabled=True))
        res = cast("Any", stage.execute({}))
        assert res.is_err()
        assert res.unwrap_err().code == "EVAL_ENDPOINT_MISSING"

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

    def test_run_succeeds_with_passing_summary(self, mock_preflight_ok) -> None:
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

    def test_run_with_failing_summary_still_returns_ok(self, mock_preflight_ok) -> None:
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

    def test_uses_model_name_from_config_when_not_in_context(self, mock_preflight_ok) -> None:
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


# ---------------------------------------------------------------------------
# Pre-flight ping — strict reachability check (C4 of plan)
# ---------------------------------------------------------------------------


class TestPreflight:
    """Direct tests for ``_preflight_check_endpoint``."""

    def _mk_response(self, status: int):
        resp = MagicMock()
        resp.status_code = status
        return resp

    def test_returns_ok_on_2xx(self) -> None:
        from src.pipeline.stages.model_evaluator import _preflight_check_endpoint

        with patch("src.pipeline.stages.model_evaluator.httpx.get") as mock_get:
            mock_get.return_value = self._mk_response(200)
            res = _preflight_check_endpoint("http://example/v1")
        assert res.is_success()
        mock_get.assert_called_once_with("http://example/v1/models", timeout=5.0)

    def test_returns_ok_on_204_no_content(self) -> None:
        from src.pipeline.stages.model_evaluator import _preflight_check_endpoint

        with patch("src.pipeline.stages.model_evaluator.httpx.get") as mock_get:
            mock_get.return_value = self._mk_response(204)
            res = _preflight_check_endpoint("http://example/v1")
        assert res.is_success()

    def test_returns_ok_on_404(self) -> None:
        """404 means the server answered → endpoint is reachable. The /models
        contract check is left to the eval client itself; pre-flight only
        cares about TCP/HTTP-level reachability."""
        from src.pipeline.stages.model_evaluator import _preflight_check_endpoint

        with patch("src.pipeline.stages.model_evaluator.httpx.get") as mock_get:
            mock_get.return_value = self._mk_response(404)
            res = _preflight_check_endpoint("http://example/v1")
        assert res.is_success()

    def test_returns_err_on_5xx(self) -> None:
        from src.pipeline.stages.model_evaluator import _preflight_check_endpoint

        with patch("src.pipeline.stages.model_evaluator.httpx.get") as mock_get, \
             patch("src.pipeline.stages.model_evaluator.time.sleep"):
            mock_get.return_value = self._mk_response(503)
            res = _preflight_check_endpoint("http://example/v1")
        assert res.is_err()
        assert res.unwrap_err().code == "EVAL_ENDPOINT_UNREACHABLE"

    def test_returns_err_on_connect_error(self) -> None:
        import httpx as _httpx
        from src.pipeline.stages.model_evaluator import _preflight_check_endpoint

        with patch("src.pipeline.stages.model_evaluator.httpx.get") as mock_get, \
             patch("src.pipeline.stages.model_evaluator.time.sleep"):
            mock_get.side_effect = _httpx.ConnectError("refused")
            res = _preflight_check_endpoint("http://example/v1")
        assert res.is_err()
        assert res.unwrap_err().code == "EVAL_ENDPOINT_UNREACHABLE"

    def test_returns_err_on_timeout(self) -> None:
        import httpx as _httpx
        from src.pipeline.stages.model_evaluator import _preflight_check_endpoint

        with patch("src.pipeline.stages.model_evaluator.httpx.get") as mock_get, \
             patch("src.pipeline.stages.model_evaluator.time.sleep"):
            mock_get.side_effect = _httpx.ReadTimeout("slow")
            res = _preflight_check_endpoint("http://example/v1")
        assert res.is_err()
        assert res.unwrap_err().code == "EVAL_ENDPOINT_UNREACHABLE"

    def test_retries_once_then_fails(self) -> None:
        import httpx as _httpx
        from src.pipeline.stages.model_evaluator import _preflight_check_endpoint

        call_count = {"n": 0}

        def _raise(*args, **kwargs):
            call_count["n"] += 1
            raise _httpx.ConnectError("nope")

        with patch("src.pipeline.stages.model_evaluator.httpx.get", side_effect=_raise), \
             patch("src.pipeline.stages.model_evaluator.time.sleep"):
            res = _preflight_check_endpoint("http://example/v1")
        # One initial + one retry = 2 attempts
        assert call_count["n"] == 2
        assert res.is_err()

    def test_retry_succeeds_on_second_attempt(self) -> None:
        import httpx as _httpx
        from src.pipeline.stages.model_evaluator import _preflight_check_endpoint

        call_count = {"n": 0}
        good_resp = MagicMock()
        good_resp.status_code = 200

        def _flaky(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _httpx.ConnectError("nope")
            return good_resp

        with patch("src.pipeline.stages.model_evaluator.httpx.get", side_effect=_flaky), \
             patch("src.pipeline.stages.model_evaluator.time.sleep"):
            res = _preflight_check_endpoint("http://example/v1")
        assert res.is_success()
        assert call_count["n"] == 2

    def test_strips_trailing_slash_from_endpoint(self) -> None:
        from src.pipeline.stages.model_evaluator import _preflight_check_endpoint

        with patch("src.pipeline.stages.model_evaluator.httpx.get") as mock_get:
            mock_get.return_value = self._mk_response(200)
            _preflight_check_endpoint("http://example/v1/")
        # No double slash before /models
        called_url = mock_get.call_args[0][0]
        assert called_url == "http://example/v1/models"


class TestEvaluatorPreflightIntegration:
    """End-to-end: stage.execute() must surface pre-flight failures."""

    def test_stage_returns_err_when_preflight_fails(self) -> None:
        """Unreachable endpoint → Err(EVAL_ENDPOINT_UNREACHABLE), no eval runs."""
        import httpx as _httpx

        stage = ModelEvaluator(_mk_cfg(eval_enabled=True))

        with patch("src.pipeline.stages.model_evaluator.httpx.get") as mock_get, \
             patch("src.pipeline.stages.model_evaluator.time.sleep"), \
             patch("src.evaluation.runner.EvaluationRunner") as MockRunner:
            mock_get.side_effect = _httpx.ConnectError("refused")
            res = cast("Any", stage.execute({
                "endpoint_url": "http://127.0.0.1:8000/v1",
            }))
            # Eval must NOT have been built — pre-flight aborts first.
            MockRunner.assert_not_called()

        assert res.is_err()
        assert res.unwrap_err().code == "EVAL_ENDPOINT_UNREACHABLE"

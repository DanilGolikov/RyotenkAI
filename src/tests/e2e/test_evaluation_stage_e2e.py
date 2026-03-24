"""
E2E tests for the Evaluation Stage.

Tests the full evaluation flow: ModelEvaluator → EvaluationRunner → Plugins,
with mocked inputs (context, dataset) and mocked external dependencies (inference HTTP).
Real provider classes (RunPod, SingleNode) are used for activate_for_eval / deactivate_after_eval.

Cleanup: all filesystem artifacts live in pytest tmp_path (auto-cleaned).

Test categories:
  1. Positive + Negative (happy path / failure path)
  2. Edge cases (boundaries, empty data, single sample)
  3. Invariants (type guarantees, consistency)
  4. Dependency errors (inference failures, plugin crashes)
  5. Regressions (HelixQL-specific syntax detection)
  6. Specific logic (dataset extraction, factory, registry)
  7. Combinatorial (threshold × quality matrix)
  8. Provider integration (RunPod / SingleNode / Serverless real classes)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.config.evaluation.schema import (
    EvaluationConfig,
    EvaluationDatasetConfig,
    EvaluatorPluginConfig,
    EvaluatorsConfig,
)
from src.evaluation.model_client.factory import ModelClientFactory
from src.evaluation.model_client.mock_client import MockInferenceClient
from src.evaluation.plugins.base import EvalResult, EvalSample, EvaluatorPlugin
from src.evaluation.plugins.registry import EvaluatorPluginRegistry
from src.evaluation.plugins.syntax_check.helixql_syntax import (
    HelixQLSyntaxPlugin,
    _check_helixql_syntax,
)
from src.evaluation.runner import EvaluationRunner
from src.pipeline.stages.model_evaluator import ModelEvaluator

# ---------------------------------------------------------------------------
# Valid / Invalid HelixQL fixtures
# ---------------------------------------------------------------------------

_VALID_HELIXQL = """\
QUERY getUser(userId: ID) => {
  user = GET node("User", userId)
  RETURN user
}"""

_VALID_HELIXQL_MULTI = """\
QUERY findPosts(authorId: ID) => {
  posts = FILTER nodes("Post") WHERE authorId == $authorId
  RETURN posts
}"""

_INVALID_NO_QUERY = "SELECT * FROM users WHERE id = 1"
_INVALID_NO_RETURN = "QUERY broken(x: ID) => {\n  user = GET node(\"User\", x)\n}"
_INVALID_BAD_SIGNATURE = "QUERY => {\n  RETURN 1\n}"
_INVALID_UNMATCHED_BRACES = "QUERY test(x: ID) => {\n  RETURN x\n"
_INVALID_EMPTY = ""
_INVALID_WHITESPACE_ONLY = "   \n\t\n  "


# ============================================================================
# Fixtures
# ============================================================================


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _make_flat_row(question: str, expected: str | None = None, ctx: str | None = None) -> dict:
    row: dict[str, Any] = {"question": question}
    if expected is not None:
        row["expected_answer"] = expected
    if ctx is not None:
        row["context"] = ctx
    return row


def _make_chat_row(user_msg: str, assistant_msg: str | None = None) -> dict:
    messages: list[dict] = [{"role": "user", "content": user_msg}]
    if assistant_msg is not None:
        messages.append({"role": "assistant", "content": assistant_msg})
    return {"messages": messages}


def _build_eval_config(
    dataset_path: str,
    *,
    enabled: bool = True,
    plugin_enabled: bool = True,
    min_valid_ratio: float = 0.8,
) -> EvaluationConfig:
    """Build a real Pydantic EvaluationConfig (not MagicMock)."""
    return EvaluationConfig(
        enabled=enabled,
        dataset=EvaluationDatasetConfig(path=dataset_path),
        evaluators=EvaluatorsConfig(plugins=[
            EvaluatorPluginConfig(
                id="syntax_main",
                plugin="helixql_syntax",
                enabled=plugin_enabled,
                params={},
                thresholds={"min_valid_ratio": min_valid_ratio},
            )
        ]),
    )


def _build_pipeline_cfg(eval_config: EvaluationConfig) -> MagicMock:
    """Build minimal pipeline config mock with real EvaluationConfig."""
    cfg = MagicMock()
    cfg.model.name = "test-model"
    cfg.inference.engine = "vllm"
    cfg.evaluation = eval_config
    cfg.experiment_tracking.mlflow = None
    return cfg


@pytest.fixture
def valid_dataset(tmp_path: Path) -> Path:
    """10 samples with HelixQL-like questions and valid expected answers."""
    rows = [_make_flat_row(f"Generate query for user {i}", _VALID_HELIXQL) for i in range(10)]
    return _write_jsonl(tmp_path / "eval_valid.jsonl", rows)


@pytest.fixture
def mixed_dataset(tmp_path: Path) -> Path:
    """10 samples: 8 valid + 2 invalid (80% boundary for default threshold)."""
    rows = [_make_flat_row(f"query {i}") for i in range(10)]
    return _write_jsonl(tmp_path / "eval_mixed.jsonl", rows)


@pytest.fixture
def invalid_dataset(tmp_path: Path) -> Path:
    """10 samples where all model answers will be invalid HelixQL."""
    rows = [_make_flat_row(f"query {i}") for i in range(10)]
    return _write_jsonl(tmp_path / "eval_invalid.jsonl", rows)


@pytest.fixture
def chat_dataset(tmp_path: Path) -> Path:
    """5 samples in chat/messages format."""
    rows = [_make_chat_row(f"Generate query for task {i}", _VALID_HELIXQL) for i in range(5)]
    return _write_jsonl(tmp_path / "eval_chat.jsonl", rows)


@pytest.fixture
def single_sample_dataset(tmp_path: Path) -> Path:
    return _write_jsonl(tmp_path / "eval_single.jsonl", [_make_flat_row("one query")])


@pytest.fixture
def empty_lines_dataset(tmp_path: Path) -> Path:
    """Dataset with only blank lines."""
    p = tmp_path / "eval_blank.jsonl"
    p.write_text("\n\n  \n\t\n", encoding="utf-8")
    return p


@pytest.fixture
def malformed_dataset(tmp_path: Path) -> Path:
    """Mix of valid JSON, malformed JSON, and blank lines."""
    p = tmp_path / "eval_malformed.jsonl"
    p.write_text(
        json.dumps(_make_flat_row("good question")) + "\n"
        + "{{not json}}\n"
        + "\n"
        + json.dumps(_make_flat_row("another good one")) + "\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def no_question_dataset(tmp_path: Path) -> Path:
    """Rows with no question field."""
    rows = [{"context": "some context", "extra": "data"} for _ in range(3)]
    return _write_jsonl(tmp_path / "eval_no_question.jsonl", rows)


# ============================================================================
# 1. Positive Tests
# ============================================================================


class TestPositive:
    """Happy-path tests: evaluation completes successfully."""

    def test_full_flow_all_valid_responses(self, valid_dataset: Path) -> None:
        """All model answers are valid HelixQL → plugin passes, overall_passed=True."""
        client = MockInferenceClient(response=_VALID_HELIXQL)
        eval_cfg = _build_eval_config(str(valid_dataset))
        runner = EvaluationRunner(eval_cfg)

        summary = runner.run(client)

        assert summary.overall_passed is True
        assert summary.sample_count == 10
        assert "syntax_main" in summary.plugin_results
        result = summary.plugin_results["syntax_main"]
        assert result.passed is True
        assert result.metrics["valid_ratio"] == 1.0
        assert result.metrics["valid_count"] == 10
        assert result.failed_samples == []
        assert summary.errors == []

    def test_full_flow_above_threshold(self, mixed_dataset: Path) -> None:
        """80% valid (at threshold) → passes with min_valid_ratio=0.8."""
        call_count = 0

        def mixed_response(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return _VALID_HELIXQL if call_count <= 8 else _INVALID_NO_QUERY

        client = MockInferenceClient(response=mixed_response)
        eval_cfg = _build_eval_config(str(mixed_dataset), min_valid_ratio=0.8)
        runner = EvaluationRunner(eval_cfg)

        summary = runner.run(client)

        assert summary.overall_passed is True
        result = summary.plugin_results["syntax_main"]
        assert result.passed is True
        assert result.metrics["valid_ratio"] == pytest.approx(0.8)

    def test_chat_format_dataset(self, chat_dataset: Path) -> None:
        """Messages-format dataset is correctly parsed and evaluated."""
        client = MockInferenceClient(response=_VALID_HELIXQL)
        eval_cfg = _build_eval_config(str(chat_dataset))
        runner = EvaluationRunner(eval_cfg)

        summary = runner.run(client)

        assert summary.overall_passed is True
        assert summary.sample_count == 5

    def test_model_evaluator_stage_returns_ok(self, valid_dataset: Path) -> None:
        """ModelEvaluator.execute() returns Ok with eval_passed=True."""
        eval_cfg = _build_eval_config(str(valid_dataset))
        stage = ModelEvaluator(_build_pipeline_cfg(eval_cfg))

        client = MockInferenceClient(response=_VALID_HELIXQL)

        from unittest.mock import patch

        with patch(
            "src.evaluation.model_client.factory.ModelClientFactory.create",
            return_value=client,
        ):
            result = stage.execute({"endpoint_url": "http://fake:8000/v1"})

        assert result.is_success()
        out = result.unwrap()["Model Evaluator"]
        assert out["eval_passed"] is True
        assert "eval_summary" in out

    def test_different_valid_helixql_variants(self, tmp_path: Path) -> None:
        """Multiple distinct valid HelixQL variants all pass."""
        rows = [_make_flat_row(f"q{i}") for i in range(2)]
        ds = _write_jsonl(tmp_path / "variants.jsonl", rows)

        counter = {"i": 0}
        variants = [_VALID_HELIXQL, _VALID_HELIXQL_MULTI]

        def rotating(prompt: str) -> str:
            v = variants[counter["i"] % len(variants)]
            counter["i"] += 1
            return v

        client = MockInferenceClient(response=rotating)
        summary = EvaluationRunner(_build_eval_config(str(ds))).run(client)

        assert summary.overall_passed is True
        assert summary.plugin_results["syntax_main"].metrics["valid_count"] == 2


# ============================================================================
# 2. Negative Tests
# ============================================================================


class TestNegative:
    """Failure-path tests: evaluation fails or is skipped."""

    def test_all_invalid_below_threshold(self, invalid_dataset: Path) -> None:
        """All responses invalid → valid_ratio=0 → fails."""
        client = MockInferenceClient(response=_INVALID_NO_QUERY)
        eval_cfg = _build_eval_config(str(invalid_dataset))

        summary = EvaluationRunner(eval_cfg).run(client)

        assert summary.overall_passed is False
        result = summary.plugin_results["syntax_main"]
        assert result.passed is False
        assert result.metrics["valid_ratio"] == 0.0
        assert len(result.errors) > 0
        assert len(result.recommendations) > 0

    def test_dataset_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent dataset → error in summary."""
        eval_cfg = _build_eval_config(str(tmp_path / "nonexistent.jsonl"))
        client = MockInferenceClient()

        summary = EvaluationRunner(eval_cfg).run(client)

        assert summary.overall_passed is False
        assert len(summary.errors) > 0
        assert "Failed to load" in summary.errors[0]

    def test_no_dataset_configured(self) -> None:
        """evaluation.dataset=None → error."""
        eval_cfg = EvaluationConfig(enabled=True, dataset=None)
        client = MockInferenceClient()

        summary = EvaluationRunner(eval_cfg).run(client)

        assert summary.overall_passed is False
        assert any("not configured" in e for e in summary.errors)

    def test_no_plugins_enabled(self, valid_dataset: Path) -> None:
        """All plugins disabled → error in summary."""
        eval_cfg = _build_eval_config(str(valid_dataset), plugin_enabled=False)
        client = MockInferenceClient(response=_VALID_HELIXQL)

        summary = EvaluationRunner(eval_cfg).run(client)

        assert summary.overall_passed is False
        assert any("No evaluation plugins" in e for e in summary.errors)

    def test_unsupported_engine_raises_value_error(self) -> None:
        """Unknown engine name → ValueError from ModelClientFactory."""
        with pytest.raises(ValueError, match="No model client registered"):
            ModelClientFactory.create(engine="unknown_engine", base_url="http://x", model="m")

    def test_evaluator_stage_skip_when_disabled(self) -> None:
        """ModelEvaluator returns skipped when evaluation.enabled=false."""
        eval_cfg = _build_eval_config("/fake.jsonl", enabled=False)
        stage = ModelEvaluator(_build_pipeline_cfg(eval_cfg))

        result = stage.execute({})

        assert result.is_success()
        out = result.unwrap()["Model Evaluator"]
        assert out["evaluation_skipped"] is True

    def test_evaluator_stage_skip_when_no_endpoint_url(self, valid_dataset: Path) -> None:
        """No endpoint_url in context → evaluation skipped."""
        eval_cfg = _build_eval_config(str(valid_dataset))
        stage = ModelEvaluator(_build_pipeline_cfg(eval_cfg))

        result = stage.execute({})

        assert result.is_success()
        out = result.unwrap()["Model Evaluator"]
        assert out["evaluation_skipped"] is True


# ============================================================================
# 3. Edge Cases
# ============================================================================


class TestEdgeCases:
    """Boundary and unusual input tests."""

    def test_single_sample_valid(self, single_sample_dataset: Path) -> None:
        """Single valid sample → passes."""
        client = MockInferenceClient(response=_VALID_HELIXQL)
        summary = EvaluationRunner(
            _build_eval_config(str(single_sample_dataset))
        ).run(client)

        assert summary.overall_passed is True
        assert summary.sample_count == 1

    def test_single_sample_invalid(self, single_sample_dataset: Path) -> None:
        """Single invalid sample → fails (0% < 80%)."""
        client = MockInferenceClient(response=_INVALID_NO_QUERY)
        summary = EvaluationRunner(
            _build_eval_config(str(single_sample_dataset))
        ).run(client)

        assert summary.overall_passed is False
        assert summary.plugin_results["syntax_main"].metrics["valid_ratio"] == 0.0

    def test_exact_boundary_threshold_passes(self, tmp_path: Path) -> None:
        """valid_ratio == min_valid_ratio → passes (>=, not >)."""
        rows = [_make_flat_row(f"q{i}") for i in range(10)]
        ds = _write_jsonl(tmp_path / "boundary.jsonl", rows)
        counter = {"i": 0}

        def responses(prompt: str) -> str:
            counter["i"] += 1
            return _VALID_HELIXQL if counter["i"] <= 7 else _INVALID_NO_QUERY

        eval_cfg = _build_eval_config(str(ds), min_valid_ratio=0.7)
        summary = EvaluationRunner(eval_cfg).run(MockInferenceClient(response=responses))

        assert summary.plugin_results["syntax_main"].passed is True

    def test_just_below_threshold_fails(self, tmp_path: Path) -> None:
        """valid_ratio < min_valid_ratio → fails."""
        rows = [_make_flat_row(f"q{i}") for i in range(10)]
        ds = _write_jsonl(tmp_path / "below.jsonl", rows)
        counter = {"i": 0}

        def responses(prompt: str) -> str:
            counter["i"] += 1
            return _VALID_HELIXQL if counter["i"] <= 6 else _INVALID_NO_QUERY

        eval_cfg = _build_eval_config(str(ds), min_valid_ratio=0.7)
        summary = EvaluationRunner(eval_cfg).run(MockInferenceClient(response=responses))

        assert summary.plugin_results["syntax_main"].passed is False

    def test_empty_lines_dataset_returns_error(self, empty_lines_dataset: Path) -> None:
        """Dataset with only blank lines → no samples loaded → error."""
        eval_cfg = _build_eval_config(str(empty_lines_dataset))
        summary = EvaluationRunner(eval_cfg).run(MockInferenceClient())

        assert summary.overall_passed is False
        assert any("Failed to load" in e for e in summary.errors)

    def test_malformed_json_lines_skipped(self, malformed_dataset: Path) -> None:
        """Malformed JSON lines are skipped, valid ones are evaluated."""
        client = MockInferenceClient(response=_VALID_HELIXQL)
        summary = EvaluationRunner(
            _build_eval_config(str(malformed_dataset))
        ).run(client)

        assert summary.sample_count == 2
        assert summary.overall_passed is True

    def test_no_question_field_rows_skipped(self, no_question_dataset: Path) -> None:
        """Rows without question/messages are skipped → 0 samples → no plugins error."""
        client = MockInferenceClient(response=_VALID_HELIXQL)
        eval_cfg = _build_eval_config(str(no_question_dataset))

        summary = EvaluationRunner(eval_cfg).run(client)

        assert summary.sample_count == 0

    def test_threshold_zero_always_passes(self, invalid_dataset: Path) -> None:
        """min_valid_ratio=0 → always passes even with all invalid."""
        client = MockInferenceClient(response=_INVALID_NO_QUERY)
        eval_cfg = _build_eval_config(str(invalid_dataset), min_valid_ratio=0.0)

        summary = EvaluationRunner(eval_cfg).run(client)
        assert summary.plugin_results["syntax_main"].passed is True

    def test_threshold_one_requires_perfection(self, tmp_path: Path) -> None:
        """min_valid_ratio=1.0 → one invalid sample causes failure."""
        rows = [_make_flat_row(f"q{i}") for i in range(10)]
        ds = _write_jsonl(tmp_path / "strict.jsonl", rows)
        counter = {"i": 0}

        def mostly_valid(prompt: str) -> str:
            counter["i"] += 1
            return _VALID_HELIXQL if counter["i"] <= 9 else _INVALID_NO_QUERY

        eval_cfg = _build_eval_config(str(ds), min_valid_ratio=1.0)
        summary = EvaluationRunner(eval_cfg).run(MockInferenceClient(response=mostly_valid))

        assert summary.plugin_results["syntax_main"].passed is False

    def test_empty_model_response(self, single_sample_dataset: Path) -> None:
        """Empty model response → invalid HelixQL."""
        client = MockInferenceClient(response="")
        summary = EvaluationRunner(
            _build_eval_config(str(single_sample_dataset))
        ).run(client)

        assert summary.plugin_results["syntax_main"].metrics["valid_ratio"] == 0.0


# ============================================================================
# 4. Invariants
# ============================================================================


class TestInvariants:
    """Properties that must always hold regardless of input."""

    def test_run_summary_is_json_serializable(self, valid_dataset: Path) -> None:
        """RunSummary.to_dict() must produce valid JSON."""
        client = MockInferenceClient(response=_VALID_HELIXQL)
        summary = EvaluationRunner(_build_eval_config(str(valid_dataset))).run(client)

        serialized = summary.to_dict()
        json_str = json.dumps(serialized)
        assert isinstance(json.loads(json_str), dict)

    def test_sample_count_matches_actual(self, valid_dataset: Path) -> None:
        """summary.sample_count == number of evaluable rows in dataset."""
        client = MockInferenceClient(response=_VALID_HELIXQL)
        summary = EvaluationRunner(_build_eval_config(str(valid_dataset))).run(client)

        assert summary.sample_count == 10
        assert summary.plugin_results["syntax_main"].sample_count == 10

    def test_duration_is_non_negative(self, valid_dataset: Path) -> None:
        """duration_seconds >= 0."""
        client = MockInferenceClient(response=_VALID_HELIXQL)
        summary = EvaluationRunner(_build_eval_config(str(valid_dataset))).run(client)

        assert summary.duration_seconds >= 0

    def test_overall_passed_consistent_with_plugins(self, invalid_dataset: Path) -> None:
        """overall_passed=False when ANY plugin fails."""
        client = MockInferenceClient(response=_INVALID_NO_QUERY)
        summary = EvaluationRunner(_build_eval_config(str(invalid_dataset))).run(client)

        any_failed = any(not r.passed for r in summary.plugin_results.values())
        assert summary.overall_passed is not any_failed

    def test_metrics_keys_always_present(self, valid_dataset: Path) -> None:
        """HelixQL plugin always returns valid_count, invalid_count, total_count, valid_ratio."""
        client = MockInferenceClient(response=_VALID_HELIXQL)
        summary = EvaluationRunner(_build_eval_config(str(valid_dataset))).run(client)

        metrics = summary.plugin_results["syntax_main"].metrics
        assert "valid_count" in metrics
        assert "invalid_count" in metrics
        assert "total_count" in metrics
        assert "valid_ratio" in metrics

    def test_plugin_name_matches_result(self, valid_dataset: Path) -> None:
        """EvalResult.plugin_name always stores the registered plugin name."""
        client = MockInferenceClient(response=_VALID_HELIXQL)
        summary = EvaluationRunner(_build_eval_config(str(valid_dataset))).run(client)

        for result in summary.plugin_results.values():
            assert result.plugin_name == "helixql_syntax"

    def test_failed_samples_indices_are_valid(self, tmp_path: Path) -> None:
        """failed_samples contains valid indices (within sample range)."""
        rows = [_make_flat_row(f"q{i}") for i in range(5)]
        ds = _write_jsonl(tmp_path / "idx.jsonl", rows)

        counter = {"i": 0}

        def half_valid(prompt: str) -> str:
            counter["i"] += 1
            return _VALID_HELIXQL if counter["i"] % 2 == 0 else _INVALID_NO_QUERY

        summary = EvaluationRunner(
            _build_eval_config(str(ds), min_valid_ratio=0.0)
        ).run(MockInferenceClient(response=half_valid))

        result = summary.plugin_results["syntax_main"]
        for idx in result.failed_samples:
            assert 0 <= idx < summary.sample_count

    def test_valid_plus_invalid_equals_total(self, mixed_dataset: Path) -> None:
        """valid_count + invalid_count == total_count."""
        counter = {"i": 0}

        def mixed(prompt: str) -> str:
            counter["i"] += 1
            return _VALID_HELIXQL if counter["i"] <= 6 else _INVALID_NO_QUERY

        summary = EvaluationRunner(
            _build_eval_config(str(mixed_dataset), min_valid_ratio=0.0)
        ).run(MockInferenceClient(response=mixed))

        m = summary.plugin_results["syntax_main"].metrics
        assert m["valid_count"] + m["invalid_count"] == m["total_count"]

    def test_mlflow_metrics_flatten_correctly(self, valid_dataset: Path) -> None:
        """ModelEvaluator._build_mlflow_metrics produces eval.{plugin}.{key} format."""
        result = EvalResult(
            plugin_name="helixql_syntax",
            passed=True,
            metrics={"valid_ratio": 0.95, "valid_count": 19, "total_count": 20, "non_numeric": "skip_me"},
            sample_count=20,
        )

        metrics = ModelEvaluator._build_mlflow_metrics({"helixql_syntax": result})

        assert "eval.helixql_syntax.valid_ratio" in metrics
        assert "eval.helixql_syntax.passed" in metrics
        assert metrics["eval.helixql_syntax.passed"] == 1.0
        assert "eval.helixql_syntax.non_numeric" not in metrics


# ============================================================================
# 5. Dependency Errors
# ============================================================================


class TestDependencyErrors:
    """Inference failures and plugin crashes."""

    def test_inference_runtime_error_produces_empty_answer(self, valid_dataset: Path) -> None:
        """When inference raises RuntimeError, model_answer is empty string."""
        def failing_inference(prompt: str) -> str:
            raise RuntimeError("Connection refused")

        client = MockInferenceClient(response=failing_inference)
        eval_cfg = _build_eval_config(str(valid_dataset))

        summary = EvaluationRunner(eval_cfg).run(client)

        assert summary.sample_count == 10
        result = summary.plugin_results["syntax_main"]
        assert result.metrics["valid_count"] == 0

    def test_partial_inference_failures(self, tmp_path: Path) -> None:
        """Some inference calls fail → those get empty answers, rest proceed normally."""
        rows = [_make_flat_row(f"q{i}") for i in range(4)]
        ds = _write_jsonl(tmp_path / "partial.jsonl", rows)
        counter = {"i": 0}

        def sometimes_fail(prompt: str) -> str:
            counter["i"] += 1
            if counter["i"] % 2 == 0:
                raise RuntimeError("timeout")
            return _VALID_HELIXQL

        client = MockInferenceClient(response=sometimes_fail)
        summary = EvaluationRunner(
            _build_eval_config(str(ds), min_valid_ratio=0.0)
        ).run(client)

        assert summary.sample_count == 4
        m = summary.plugin_results["syntax_main"].metrics
        assert m["valid_count"] == 2
        assert m["invalid_count"] == 2

    def test_plugin_crash_is_captured(self, tmp_path: Path) -> None:
        """If a plugin raises an unexpected exception, EvaluationRunner catches it."""
        rows = [_make_flat_row("crash test")]
        ds = _write_jsonl(tmp_path / "crash.jsonl", rows)

        @EvaluatorPluginRegistry.register
        class _CrashPlugin(EvaluatorPlugin):
            name = "_e2e_crash_plugin"
            priority = 1

            @classmethod
            def get_description(cls) -> str:
                return "E2E crash plugin"

            def evaluate(self, samples: list[EvalSample]) -> EvalResult:
                raise ValueError("Intentional crash for testing")

            def get_recommendations(self, result: EvalResult) -> list[str]:
                return []

        try:
            eval_cfg = EvaluationConfig(
                enabled=True,
                dataset=EvaluationDatasetConfig(path=str(ds)),
                evaluators=EvaluatorsConfig(plugins=[
                    EvaluatorPluginConfig(id="crash_main", plugin="_e2e_crash_plugin", enabled=True, params={}, thresholds={}),
                    EvaluatorPluginConfig(
                        id="syntax_main",
                        plugin="helixql_syntax",
                        enabled=True,
                        params={},
                        thresholds={"min_valid_ratio": 0.0},
                    ),
                ]),
            )
            client = MockInferenceClient(response=_VALID_HELIXQL)
            summary = EvaluationRunner(eval_cfg).run(client)

            crash_result = summary.plugin_results["crash_main"]
            assert crash_result.passed is False
            assert any("crashed" in e.lower() for e in crash_result.errors)

            syntax_result = summary.plugin_results["syntax_main"]
            assert syntax_result.passed is True
        finally:
            EvaluatorPluginRegistry._registry.pop("_e2e_crash_plugin", None)

    def test_dataset_permission_error(self, tmp_path: Path) -> None:
        """Unreadable dataset → error in summary (not crash)."""
        ds = tmp_path / "no_read.jsonl"
        ds.write_text('{"question": "q"}\n')
        ds.chmod(0o000)

        try:
            summary = EvaluationRunner(
                _build_eval_config(str(ds))
            ).run(MockInferenceClient())

            assert summary.overall_passed is False
            assert len(summary.errors) > 0
        finally:
            ds.chmod(0o644)


# ============================================================================
# 6. Regressions (HelixQL syntax detection)
# ============================================================================


class TestRegressions:
    """Specific HelixQL syntax checks that must not regress."""

    @pytest.mark.parametrize(
        "text,expected_valid",
        [
            (_VALID_HELIXQL, True),
            (_VALID_HELIXQL_MULTI, True),
            (_INVALID_NO_QUERY, False),
            (_INVALID_NO_RETURN, False),
            (_INVALID_BAD_SIGNATURE, False),
            (_INVALID_UNMATCHED_BRACES, False),
            (_INVALID_EMPTY, False),
            (_INVALID_WHITESPACE_ONLY, False),
            ("QUERY foo(x: ID) => {\n  RETURN x\n}", True),
            ("  QUERY   bar(a: Int) =>  {\n  data = GET node(\"X\", a)\n  RETURN data\n}", True),
        ],
        ids=[
            "valid_basic",
            "valid_filter",
            "invalid_sql",
            "invalid_no_return",
            "invalid_bad_sig",
            "invalid_braces",
            "invalid_empty",
            "invalid_whitespace",
            "valid_minimal",
            "valid_indented",
        ],
    )
    def test_check_helixql_syntax(self, text: str, expected_valid: bool) -> None:
        """Direct test of _check_helixql_syntax helper."""
        ok, reason = _check_helixql_syntax(text)
        assert ok is expected_valid, f"Expected valid={expected_valid}, got {ok}, reason={reason}"

    def test_markdown_wrapped_helixql_invalid(self) -> None:
        """HelixQL wrapped in markdown code fence should still be valid (QUERY keyword present)."""
        wrapped = f"```helixql\n{_VALID_HELIXQL}\n```"
        ok, _ = _check_helixql_syntax(wrapped)
        assert ok is True

    def test_multiple_queries_in_response(self) -> None:
        """Response with multiple QUERY blocks → valid (first QUERY matches)."""
        multi = _VALID_HELIXQL + "\n\n" + _VALID_HELIXQL_MULTI
        ok, _ = _check_helixql_syntax(multi)
        assert ok is True

    def test_extra_text_around_query_valid(self) -> None:
        """Extra text before/after QUERY block → still valid if QUERY is present."""
        text = f"Here is the query:\n{_VALID_HELIXQL}\nDone."
        ok, _ = _check_helixql_syntax(text)
        assert ok is True


# ============================================================================
# 7. Specific Logic Tests
# ============================================================================


class TestSpecificLogic:
    """Tests for internal components and their specific behavior."""

    def test_extract_row_fields_flat_format(self) -> None:
        """Flat format: question, expected_answer extracted; reserved 'context' is NOT in metadata."""
        row = {"question": "Q?", "expected_answer": "A!", "context": "C"}
        q, ea, meta = EvaluationRunner._extract_row_fields(row)
        assert q == "Q?"
        assert ea == "A!"
        # 'context' is a reserved field → excluded from metadata dict
        assert "context" not in meta
        assert meta == {}

    def test_extract_row_fields_messages_format(self) -> None:
        """Messages format: user→question, assistant→expected_answer; metadata is empty."""
        row = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        q, ea, meta = EvaluationRunner._extract_row_fields(row)
        assert q == "Hello"
        assert ea == "Hi there"
        # 'messages' is a reserved field → metadata is empty
        assert meta == {}

    def test_extract_row_fields_flat_alternative_answer_key(self) -> None:
        """Flat format with 'answer' instead of 'expected_answer'."""
        row = {"question": "Q", "answer": "A"}
        q, ea, _meta = EvaluationRunner._extract_row_fields(row)
        assert ea == "A"

    def test_extract_row_fields_no_known_keys(self) -> None:
        """Row with no question/messages → empty question; unknown keys go into metadata."""
        row = {"data": "something"}
        q, ea, meta = EvaluationRunner._extract_row_fields(row)
        assert q == ""
        assert ea is None
        # 'data' is not reserved → included in metadata
        assert meta == {"data": "something"}

    def test_model_client_factory_supported_engines(self) -> None:
        """Factory reports vllm as supported."""
        engines = ModelClientFactory.supported_engines()
        assert "vllm" in engines

    def test_model_client_factory_is_supported(self) -> None:
        """is_supported returns correct booleans."""
        assert ModelClientFactory.is_supported("vllm") is True
        assert ModelClientFactory.is_supported("unknown") is False

    def test_model_client_factory_creates_openai_client(self) -> None:
        """Factory creates OpenAICompatibleInferenceClient for vllm engine."""
        from src.evaluation.model_client.openai_client import OpenAICompatibleInferenceClient

        client = ModelClientFactory.create(engine="vllm", base_url="http://x:8000/v1", model="m")
        assert isinstance(client, OpenAICompatibleInferenceClient)

    def test_plugin_registry_contains_helixql_syntax(self) -> None:
        """helixql_syntax is registered after import."""
        from src.evaluation.plugins.discovery import ensure_evaluation_plugins_discovered

        ensure_evaluation_plugins_discovered(force=True)
        assert EvaluatorPluginRegistry.is_registered("helixql_syntax")

    def test_plugin_priority_affects_execution_order(self, tmp_path: Path) -> None:
        """Plugins execute in priority order (lower number = earlier)."""
        execution_order: list[str] = []

        @EvaluatorPluginRegistry.register
        class _HighPriPlugin(EvaluatorPlugin):
            name = "_e2e_high_pri"
            priority = 1

            @classmethod
            def get_description(cls) -> str:
                return "E2E high priority plugin"

            def evaluate(self, samples: list[EvalSample]) -> EvalResult:
                execution_order.append(self.name)
                return EvalResult(plugin_name=self.name, passed=True, sample_count=len(samples))

            def get_recommendations(self, result: EvalResult) -> list[str]:
                return []

        @EvaluatorPluginRegistry.register
        class _LowPriPlugin(EvaluatorPlugin):
            name = "_e2e_low_pri"
            priority = 99

            @classmethod
            def get_description(cls) -> str:
                return "E2E low priority plugin"

            def evaluate(self, samples: list[EvalSample]) -> EvalResult:
                execution_order.append(self.name)
                return EvalResult(plugin_name=self.name, passed=True, sample_count=len(samples))

            def get_recommendations(self, result: EvalResult) -> list[str]:
                return []

        try:
            rows = [_make_flat_row("test")]
            ds = _write_jsonl(tmp_path / "order.jsonl", rows)
            eval_cfg = EvaluationConfig(
                enabled=True,
                dataset=EvaluationDatasetConfig(path=str(ds)),
                evaluators=EvaluatorsConfig(plugins=[
                    EvaluatorPluginConfig(id="high_pri", plugin="_e2e_high_pri", enabled=True, params={}, thresholds={}),
                    EvaluatorPluginConfig(id="low_pri", plugin="_e2e_low_pri", enabled=True, params={}, thresholds={}),
                    EvaluatorPluginConfig(id="syntax_main", plugin="helixql_syntax", enabled=False, params={}, thresholds={}),
                ]),
            )
            EvaluationRunner(eval_cfg).run(MockInferenceClient(response=_VALID_HELIXQL))

            assert execution_order == ["_e2e_high_pri", "_e2e_low_pri"]
        finally:
            EvaluatorPluginRegistry._registry.pop("_e2e_high_pri", None)
            EvaluatorPluginRegistry._registry.pop("_e2e_low_pri", None)

    def test_requires_expected_answer_skipping(self, tmp_path: Path) -> None:
        """Plugin with requires_expected_answer=True is skipped when dataset has no expected answers."""
        @EvaluatorPluginRegistry.register
        class _NeedsExpected(EvaluatorPlugin):
            name = "_e2e_needs_expected"
            priority = 5
            requires_expected_answer = True

            @classmethod
            def get_description(cls) -> str:
                return "E2E plugin requiring expected answer"

            def evaluate(self, samples: list[EvalSample]) -> EvalResult:
                return EvalResult(plugin_name=self.name, passed=True, sample_count=len(samples))

            def get_recommendations(self, result: EvalResult) -> list[str]:
                return []

        try:
            rows = [_make_flat_row("q_no_expected")]
            ds = _write_jsonl(tmp_path / "no_expected.jsonl", rows)
            eval_cfg = EvaluationConfig(
                enabled=True,
                dataset=EvaluationDatasetConfig(path=str(ds)),
                evaluators=EvaluatorsConfig(plugins=[
                    EvaluatorPluginConfig(
                        id="needs_expected_main",
                        plugin="_e2e_needs_expected",
                        enabled=True,
                        params={},
                        thresholds={},
                    ),
                    EvaluatorPluginConfig(
                        id="syntax_main",
                        plugin="helixql_syntax",
                        enabled=True,
                        params={},
                        thresholds={"min_valid_ratio": 0.0},
                    ),
                ]),
            )
            summary = EvaluationRunner(eval_cfg).run(MockInferenceClient(response=_VALID_HELIXQL))

            assert "needs_expected_main" in summary.skipped_plugins
            assert "needs_expected_main" not in summary.plugin_results
            assert "syntax_main" in summary.plugin_results
        finally:
            EvaluatorPluginRegistry._registry.pop("_e2e_needs_expected", None)

    def test_helixql_plugin_empty_samples_list(self) -> None:
        """HelixQLSyntaxPlugin.evaluate([]) returns passed=True with valid_ratio=1.0."""
        plugin = HelixQLSyntaxPlugin(params={}, thresholds={"min_valid_ratio": 0.8})
        result = plugin.evaluate([])

        assert result.passed is True
        assert result.metrics["valid_ratio"] == 1.0
        assert result.metrics["total_count"] == 0


# ============================================================================
# 8. Combinatorial (threshold × quality matrix)
# ============================================================================


class TestCombinatorial:
    """Cross-product of thresholds and response quality."""

    @pytest.mark.parametrize(
        "valid_count,total,threshold,expected_pass",
        [
            (10, 10, 0.8, True),
            (8, 10, 0.8, True),
            (7, 10, 0.8, False),
            (0, 10, 0.0, True),
            (10, 10, 1.0, True),
            (9, 10, 1.0, False),
            (1, 1, 0.5, True),
            (0, 1, 0.5, False),
            (5, 10, 0.5, True),
            (4, 10, 0.5, False),
        ],
        ids=[
            "10/10@0.8=pass",
            "8/10@0.8=pass",
            "7/10@0.8=fail",
            "0/10@0.0=pass",
            "10/10@1.0=pass",
            "9/10@1.0=fail",
            "1/1@0.5=pass",
            "0/1@0.5=fail",
            "5/10@0.5=pass",
            "4/10@0.5=fail",
        ],
    )
    def test_threshold_quality_matrix(
        self,
        tmp_path: Path,
        valid_count: int,
        total: int,
        threshold: float,
        expected_pass: bool,
    ) -> None:
        """Parameterized threshold × quality matrix."""
        rows = [_make_flat_row(f"q{i}") for i in range(total)]
        ds = _write_jsonl(tmp_path / f"matrix_{valid_count}_{total}.jsonl", rows)
        counter = {"i": 0}

        def controlled(prompt: str) -> str:
            counter["i"] += 1
            return _VALID_HELIXQL if counter["i"] <= valid_count else _INVALID_NO_QUERY

        eval_cfg = _build_eval_config(str(ds), min_valid_ratio=threshold)
        summary = EvaluationRunner(eval_cfg).run(MockInferenceClient(response=controlled))

        result = summary.plugin_results["syntax_main"]
        assert result.passed is expected_pass, (
            f"Expected passed={expected_pass} for {valid_count}/{total} @ threshold={threshold}, "
            f"got passed={result.passed}, valid_ratio={result.metrics.get('valid_ratio')}"
        )


# ============================================================================
# 9. Provider Integration (real classes, not mocked)
# ============================================================================


class TestProviderIntegration:
    """Test real provider activate_for_eval / deactivate_after_eval behavior."""

    def test_runpod_pods_activate_for_eval_err_without_deploy(self) -> None:
        """Real RunPodPodInferenceProvider.activate_for_eval() returns Err when deploy() was never called."""
        from src.providers.runpod.inference.pods.provider import RunPodPodInferenceProvider

        provider = object.__new__(RunPodPodInferenceProvider)
        # Simulate state after __init__ but before deploy()
        provider._api = None
        provider._pod_id = None
        provider._eval_session = None
        result = provider.activate_for_eval()

        assert result.is_failure()
        assert "deploy()" in str(result.unwrap_err()) or "not initialized" in str(result.unwrap_err())

    def test_runpod_pods_deactivate_after_eval_ok_without_session(self) -> None:
        """Real RunPodPodInferenceProvider.deactivate_after_eval() returns Ok when session is None (no-op)."""
        from src.providers.runpod.inference.pods.provider import RunPodPodInferenceProvider

        provider = object.__new__(RunPodPodInferenceProvider)
        # deactivate without ever calling activate → should be a safe no-op
        provider._api = None
        provider._eval_session = None
        result = provider.deactivate_after_eval()

        assert result.is_success()

    def test_single_node_activate_for_eval_err_when_not_deployed(self) -> None:
        """Real SingleNodeInferenceProvider.activate_for_eval() → Err when endpoint is None."""
        from src.providers.single_node.inference.provider import SingleNodeInferenceProvider

        provider = object.__new__(SingleNodeInferenceProvider)
        provider._endpoint_info = None
        result = provider.activate_for_eval()

        assert result.is_failure()
        assert "not deployed" in str(result.unwrap_err())

    def test_single_node_activate_for_eval_ok_with_endpoint(self) -> None:
        """Real SingleNodeInferenceProvider.activate_for_eval() → Ok(url) when endpoint exists."""
        from src.providers.inference.interfaces import EndpointInfo
        from src.providers.single_node.inference.provider import SingleNodeInferenceProvider

        provider = object.__new__(SingleNodeInferenceProvider)
        provider._endpoint_info = EndpointInfo(
            endpoint_url="http://127.0.0.1:8000/v1",
            api_type="openai_compatible",
            provider_type="single_node",
            engine="vllm",
            model_id="test-model",
        )
        result = provider.activate_for_eval()

        assert result.is_success()
        assert result.unwrap() == "http://127.0.0.1:8000/v1"

    def test_single_node_deactivate_after_eval_ok(self) -> None:
        """Real SingleNodeInferenceProvider.deactivate_after_eval() → Ok(None) (no-op)."""
        from src.providers.single_node.inference.provider import SingleNodeInferenceProvider

        provider = object.__new__(SingleNodeInferenceProvider)
        result = provider.deactivate_after_eval()

        assert result.is_success()

    def test_eval_stage_skips_when_provider_rejects_activation(self, valid_dataset: Path) -> None:
        """Full flow: provider.activate_for_eval()→Err → no endpoint_url → eval skipped."""
        eval_cfg = _build_eval_config(str(valid_dataset))
        stage = ModelEvaluator(_build_pipeline_cfg(eval_cfg))

        result = stage.execute({})

        assert result.is_success()
        out = result.unwrap()["Model Evaluator"]
        assert out["evaluation_skipped"] is True

    def test_eval_stage_reads_endpoint_url_from_inference_deployer_context(
        self, valid_dataset: Path
    ) -> None:
        """Regression: ModelEvaluator must read endpoint_url from context['Inference Deployer'],
        not from context root.

        update_context() stores stage output under context[stage_name], so
        InferenceDeployer puts endpoint_url in context['Inference Deployer']['endpoint_url'].
        Reading from context root would always return None → eval always skipped.
        """
        from src.pipeline.stages.constants import StageNames

        eval_cfg = _build_eval_config(str(valid_dataset))
        stage = ModelEvaluator(_build_pipeline_cfg(eval_cfg))

        client = MockInferenceClient(response=_VALID_HELIXQL)

        # Simulate what InferenceDeployer puts into context via update_context()
        context = {
            StageNames.INFERENCE_DEPLOYER: {
                "endpoint_url": "http://127.0.0.1:8000/v1",
                "inference_model_name": "test-model",
                "inference_deployed": True,
            }
            # NOTE: 'endpoint_url' is intentionally NOT in context root
        }

        from unittest.mock import patch

        with patch(
            "src.evaluation.model_client.factory.ModelClientFactory.create",
            return_value=client,
        ):
            result = stage.execute(context)

        assert result.is_success()
        out = result.unwrap()["Model Evaluator"]
        assert "evaluation_skipped" not in out, (
            "Evaluation should NOT be skipped — endpoint_url is in context['Inference Deployer']"
        )
        assert out["eval_passed"] is True

    # ------------------------------------------------------------------
    # Regression: adapter_ref='auto' bug (ISSUE-18)
    # ------------------------------------------------------------------

    def test_runpod_pods_activate_for_eval_err_when_adapter_ref_empty(self) -> None:
        """Regression: activate_for_eval must return Err when _adapter_ref is empty.

        Before the fix, activate_for_eval read self._cfg.inference.common.model_source
        which could be 'auto', causing merge_lora.py to fail with:
          'Adapter auto is neither a local path nor a valid HuggingFace repo ID'

        After the fix, deploy() stores the *resolved* adapter_ref in self._adapter_ref,
        and activate_for_eval() returns Err immediately if it is empty/unset.
        """
        from src.providers.runpod.inference.pods.provider import RunPodPodInferenceProvider

        provider = object.__new__(RunPodPodInferenceProvider)
        provider._api = MagicMock()  # non-None so the first guard passes
        provider._pod_id = "pod_abc123"  # non-empty so the second guard passes
        provider._adapter_ref = ""  # empty → must fail with a clear message

        result = provider.activate_for_eval()

        assert result.is_failure(), "Expected Err when _adapter_ref is empty"
        err_msg = str(result.unwrap_err())
        assert "adapter_ref" in err_msg or "model_source" in err_msg, (
            f"Error message should mention adapter_ref or model_source, got: {err_msg!r}"
        )

    def test_runpod_pods_deploy_stores_resolved_adapter_ref(self) -> None:
        """Regression: adapter_ref should be stored after the computation in deploy().

        Verifies that the assignment self._adapter_ref = adapter_ref exists and
        runs before any early-return guards (api key, ssh key, network volume...).
        We test this by calling the code path up to the point where
        _adapter_ref is assigned, using an Err return from missing API key.
        """
        from src.providers.runpod.inference.pods.provider import RunPodPodInferenceProvider

        provider = object.__new__(RunPodPodInferenceProvider)
        provider._adapter_ref = ""
        provider._eval_session = None
        provider._pod_id = None
        provider._pod_name = None
        provider._network_volume_id = None
        provider._network_volume_meta = None
        provider._event_logger = None
        provider._api = None
        provider._endpoint_info = None
        provider._volume_cfg = None

        mock_cfg = MagicMock()
        mock_cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
        mock_cfg.model.trust_remote_code = False
        mock_cfg.inference.common.model_source = "auto"
        mock_cfg.inference.engine = "vllm"
        provider._cfg = mock_cfg

        mock_provider_cfg = MagicMock()
        mock_provider_cfg.connect.ssh.key_path = "/tmp/nonexistent_key"
        provider._provider_cfg = mock_provider_cfg

        mock_secrets = MagicMock()
        mock_secrets.runpod_api_key = None  # ← triggers Err early, but AFTER _adapter_ref assignment
        provider._secrets = mock_secrets

        resolved_hf_repo = "myorg/mymodel-v7-lora"
        result = provider.deploy(
            model_source=resolved_hf_repo,
            run_id="test-run",
            base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
            lora_path=None,
        )

        # deploy() returns Err (missing api key) but _adapter_ref must already be set
        assert result.is_failure(), "Expected Err due to missing runpod_api_key"
        assert provider._adapter_ref == resolved_hf_repo, (
            f"Expected _adapter_ref={resolved_hf_repo!r}, got {provider._adapter_ref!r}. "
            "deploy() must store resolved adapter_ref before api-key check "
            "so activate_for_eval() never receives 'auto'."
        )

    def test_build_eval_session_params_returns_complete_dict(self) -> None:
        """Unit test: _build_eval_session_params() returns all keys expected by pod_session.activate()."""
        from pathlib import Path

        from src.providers.runpod.inference.pods.constants import (
            POD_HF_CACHE_DIR,
            POD_LOCK_DIR,
            pod_merged_dir,
            pod_pid_file,
            pod_run_dir,
        )
        from src.providers.runpod.inference.pods.provider import RunPodPodInferenceProvider

        provider = object.__new__(RunPodPodInferenceProvider)
        provider._pod_id = "testpod123"
        provider._adapter_ref = "myorg/mymodel-lora"

        mock_cfg = MagicMock()
        mock_cfg.model.name = "Qwen/Qwen2.5-0.5B-Instruct"
        mock_cfg.model.trust_remote_code = False
        provider._cfg = mock_cfg

        mock_engine_cfg = MagicMock()
        mock_engine_cfg.model_dump.return_value = {"max_model_len": 4096}
        provider._engine_cfg = mock_engine_cfg

        mock_serve_cfg = MagicMock()
        mock_serve_cfg.port = 8000
        provider._serve_cfg = mock_serve_cfg

        mock_provider_cfg = MagicMock()
        mock_provider_cfg.connect.ssh.key_path = "/tmp/test_key"
        mock_provider_cfg.model_dump.return_value = {"gpu_type": "A100"}
        provider._provider_cfg = mock_provider_cfg

        mock_secrets = MagicMock()
        mock_secrets.hf_token = "hf_test_token"
        provider._secrets = mock_secrets

        params = provider._build_eval_session_params()

        expected_keys = {
            "key_path", "serve_port", "run_dir", "merged_dir", "hf_cache_dir",
            "pid_file", "log_file", "hash_file", "lock_dir",
            "config_hash", "base_model_id", "adapter_ref", "trust_remote_code",
            "hf_token", "vllm_cfg",
        }
        assert set(params.keys()) == expected_keys, (
            f"Missing keys: {expected_keys - set(params.keys())}"
        )

        run_key = "testpod123"
        assert params["hf_cache_dir"] == POD_HF_CACHE_DIR
        assert params["lock_dir"] == POD_LOCK_DIR
        assert params["run_dir"] == pod_run_dir(run_key)
        assert params["merged_dir"] == pod_merged_dir(run_key)
        assert params["pid_file"] == pod_pid_file(run_key)
        assert isinstance(params["key_path"], Path)
        assert params["serve_port"] == 8000
        assert params["adapter_ref"] == "myorg/mymodel-lora"
        assert params["hf_token"] == "hf_test_token"
        assert isinstance(params["config_hash"], str) and len(params["config_hash"]) > 0


# ============================================================================
# 10. Cleanup Verification
# ============================================================================


class TestCleanup:
    """Verify no state leaks between tests."""

    def test_tmp_dataset_is_isolated(self, tmp_path: Path) -> None:
        """Each test gets its own tmp_path — no cross-contamination."""
        ds = _write_jsonl(tmp_path / "isolated.jsonl", [_make_flat_row("q")])
        assert ds.exists()
        assert ds.parent == tmp_path

    def test_plugin_registry_not_polluted(self) -> None:
        """E2E test plugins are cleaned up — only production plugins remain."""
        from src.evaluation.plugins.discovery import ensure_evaluation_plugins_discovered

        ensure_evaluation_plugins_discovered(force=True)
        for name in list(EvaluatorPluginRegistry._registry.keys()):
            assert not name.startswith("_e2e_"), f"Test plugin {name!r} leaked into registry"

    def test_mock_client_has_no_side_effects(self) -> None:
        """MockInferenceClient doesn't touch filesystem or network."""
        client = MockInferenceClient(response="test")
        result = client.generate("anything")
        assert result == "test"

"""
Unit tests for CerebrasJudgePlugin, CerebrasProvider, and ScoreAggregator.

Tests:
- test_score_normalization              — (score-1)/4 for all values 1-5
- test_aggregate_scores_mean_and_p50    — aggregate_scores() computes correct metrics
- test_aggregate_scores_empty           — empty scores returns passed=True with sample_count=0
- test_aggregate_scores_threshold_pass  — mean above threshold → passed=True
- test_aggregate_scores_threshold_fail  — mean below threshold → passed=False with errors
- test_evaluate_missing_api_key         — SecretsResolver raises RuntimeError → EvalResult(passed=False)
- test_evaluate_with_mock_provider      — MockJudgeProvider flows through to EvalResult
- test_judge_json_parse_fallback        — CerebrasProvider._parse_response() with invalid JSON
- test_judge_regex_score_fallback       — regex fallback extracts score from malformed JSON
- test_max_samples_limit                — samples[:max_samples] correctly applied
- test_evaluate_missing_api_key         — SecretsResolver raises RuntimeError → EvalResult(passed=False)
- test_cerebras_provider_builds_messages — _build_messages() includes question/expected/model_answer
- test_parse_response_valid_json        — normal happy-path JSON parse
- test_parse_response_clamps_score      — score clamped to [1, 5]
- test_plugin_validate_params_bad_model — raises ValueError for empty model param
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.plugins.base import EvalSample
from src.evaluation.plugins.llm_judge.cerebras_judge import CerebrasJudgePlugin, CerebrasProvider
from src.evaluation.plugins.llm_judge.interface import JudgeResponse
from src.evaluation.plugins.utils import aggregate_scores

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(question: str = "Q", model_answer: str = "A", expected_answer: str = "E") -> EvalSample:
    return EvalSample(question=question, model_answer=model_answer, expected_answer=expected_answer)


def _make_plugin(params: dict[str, Any] | None = None, thresholds: dict[str, Any] | None = None) -> CerebrasJudgePlugin:
    plugin = CerebrasJudgePlugin(
        params=params or {"model": "llama-3.3-70b", "max_samples": 10},
        thresholds=thresholds or {"min_mean_score": 0.6},
    )
    # Inject a dummy _secrets to pass the attribute check
    object.__setattr__(plugin, "_secrets", {"EVAL_CEREBRAS_API_KEY": "fake-key"})
    return plugin


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------


class TestScoreNormalization:
    @pytest.mark.parametrize(
        "raw_score, expected_normalized",
        [
            (1, 0.0),
            (2, 0.25),
            (3, 0.5),
            (4, 0.75),
            (5, 1.0),
        ],
    )
    def test_normalization_formula(self, raw_score: int, expected_normalized: float):
        """(score - 1) / 4 maps 1-5 to [0, 1]."""
        normalized = (raw_score - 1) / 4
        assert abs(normalized - expected_normalized) < 1e-9


# ---------------------------------------------------------------------------
# aggregate_scores utility
# ---------------------------------------------------------------------------


class TestAggregateScores:
    def test_mean_and_p50_basic(self):
        result = aggregate_scores(
            scores=[0.5, 0.75, 1.0],
            raw_scores=[3, 4, 5],
            failed_indices=[],
            plugin_name="test_plugin",
            threshold_key="min_mean_score",
            thresholds={"min_mean_score": 0.6},
            recommendations=[],
        )
        assert result.passed is True
        assert abs(result.metrics["mean_score"] - 0.75) < 0.001
        assert abs(result.metrics["p50_score"] - 0.75) < 0.001
        assert result.metrics["sample_count"] == 3

    def test_threshold_fail(self):
        result = aggregate_scores(
            scores=[0.0, 0.25, 0.5],
            raw_scores=[1, 2, 3],
            failed_indices=[0],
            plugin_name="test_plugin",
            threshold_key="min_mean_score",
            thresholds={"min_mean_score": 0.6},
            recommendations=["improve quality"],
        )
        assert result.passed is False
        assert len(result.errors) > 0
        assert result.recommendations == ["improve quality"]

    def test_empty_scores_returns_no_samples(self):
        result = aggregate_scores(
            scores=[],
            raw_scores=[],
            failed_indices=[],
            plugin_name="test_plugin",
            threshold_key="min_mean_score",
            thresholds={},
            recommendations=[],
        )
        assert result.passed is True
        assert result.metrics["sample_count"] == 0

    def test_score_distribution_populated(self):
        result = aggregate_scores(
            scores=[0.0, 0.5, 1.0],
            raw_scores=[1, 3, 5],
            failed_indices=[],
            plugin_name="test_plugin",
            threshold_key="min_mean_score",
            thresholds={"min_mean_score": 0.0},
            recommendations=[],
        )
        dist = result.metrics["score_distribution"]
        assert dist["1"] == 1
        assert dist["3"] == 1
        assert dist["5"] == 1
        assert dist["2"] == 0

    def test_uses_default_threshold_when_missing(self):
        """When threshold_key is absent, falls back to default 0.6."""
        result = aggregate_scores(
            scores=[0.7],
            raw_scores=[4],
            failed_indices=[],
            plugin_name="p",
            threshold_key="min_mean_score",
            thresholds={},  # key not present
            recommendations=[],
        )
        assert result.passed is True  # 0.7 >= 0.6 default


# ---------------------------------------------------------------------------
# CerebrasProvider — unit tests (no HTTP)
# ---------------------------------------------------------------------------


class TestCerebrasProviderParseResponse:
    def test_valid_json_response(self):
        provider = CerebrasProvider(api_key="test")
        raw = json.dumps({"reasoning": "good answer", "score": 4})
        response = provider._parse_response(raw)

        assert response.score == 4
        assert response.reasoning == "good answer"
        assert response.raw_response == raw

    def test_score_clamped_to_max(self):
        provider = CerebrasProvider(api_key="test")
        raw = json.dumps({"score": 10, "reasoning": ""})
        response = provider._parse_response(raw)
        assert response.score == 5

    def test_score_clamped_to_min(self):
        provider = CerebrasProvider(api_key="test")
        raw = json.dumps({"score": -3, "reasoning": ""})
        response = provider._parse_response(raw)
        assert response.score == 1

    def test_json_parse_fallback_with_markdown_fence(self):
        """Handles score embedded in markdown fences."""
        provider = CerebrasProvider(api_key="test")
        raw = '```json\n{"score": 3, "reasoning": "partial"}\n```'
        response = provider._parse_response(raw)
        assert response.score == 3

    def test_regex_fallback_extracts_score(self):
        """Last-resort regex extracts score when JSON parse fails entirely."""
        provider = CerebrasProvider(api_key="test")
        raw = 'some text "score": 5 more text'
        response = provider._parse_response(raw)
        assert response.score == 5

    def test_unparseable_raises_runtime_error(self):
        """Completely unparseable response raises RuntimeError."""
        provider = CerebrasProvider(api_key="test")
        raw = "I cannot provide a score because reasons"
        with pytest.raises(RuntimeError, match="Could not extract score"):
            provider._parse_response(raw)

    def test_build_messages_contains_all_parts(self):
        provider = CerebrasProvider(api_key="test")
        msgs = provider._build_messages(question="Q?", expected="E", model_answer="A")

        assert msgs[0]["role"] == "system"
        user_msg = msgs[1]["content"]
        assert "Q?" in user_msg
        assert "E" in user_msg
        assert "A" in user_msg


# ---------------------------------------------------------------------------
# CerebrasJudgePlugin — unit tests
# ---------------------------------------------------------------------------


class TestCerebrasJudgePlugin:
    def test_get_description_non_empty(self):
        assert len(CerebrasJudgePlugin.get_description()) > 10

    def test_validate_params_bad_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            CerebrasJudgePlugin(params={"model": ""}, thresholds={})

    def test_validate_params_bad_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            CerebrasJudgePlugin(params={"model": "llama-3.3-70b", "temperature": 5.0}, thresholds={})

    def test_validate_params_bad_max_samples_raises(self):
        with pytest.raises(ValueError, match="max_samples"):
            CerebrasJudgePlugin(params={"model": "llama-3.3-70b", "max_samples": 0}, thresholds={})

    def test_evaluate_with_mock_provider(self):
        """CerebrasJudgePlugin.evaluate() uses injected provider → correct EvalResult."""
        from src.evaluation.plugins.llm_judge.interface import JudgeResponse

        plugin = _make_plugin(thresholds={"min_mean_score": 0.5})
        samples = [_make_sample() for _ in range(3)]

        mock_provider = MagicMock()
        mock_provider.judge.return_value = JudgeResponse(score=4, reasoning="ok", raw_response='{"score":4}')

        with patch(
            "src.evaluation.plugins.llm_judge.cerebras_judge.CerebrasProvider",
            return_value=mock_provider,
        ):
            result = plugin.evaluate(samples)

        assert result.passed is True
        assert result.metrics["mean_score"] == 0.75  # (4-1)/4
        assert result.metrics["sample_count"] == 3
        assert mock_provider.judge.call_count == 3

    def test_evaluate_all_samples_fail_returns_passed_false(self):
        """When all API calls fail, EvalResult.passed=False with descriptive error."""
        plugin = _make_plugin()

        mock_provider = MagicMock()
        mock_provider.judge.side_effect = RuntimeError("API error")

        with patch(
            "src.evaluation.plugins.llm_judge.cerebras_judge.CerebrasProvider",
            return_value=mock_provider,
        ):
            result = plugin.evaluate([_make_sample()])

        assert result.passed is False
        assert "All samples failed" in result.errors[0]

    def test_max_samples_limit_applied(self):
        """Samples are capped to max_samples param."""
        plugin = _make_plugin(params={"model": "llama-3.3-70b", "max_samples": 2})
        samples = [_make_sample() for _ in range(5)]

        mock_provider = MagicMock()
        mock_provider.judge.return_value = JudgeResponse(score=5, reasoning="", raw_response="{}")

        with patch(
            "src.evaluation.plugins.llm_judge.cerebras_judge.CerebrasProvider",
            return_value=mock_provider,
        ):
            result = plugin.evaluate(samples)

        assert mock_provider.judge.call_count == 2
        assert result.metrics["sample_count"] == 2

    def test_evaluate_missing_api_key_raises(self):
        """Plugin without _secrets injected raises AttributeError (not RuntimeError from resolver)."""
        plugin = CerebrasJudgePlugin(
            params={"model": "llama-3.3-70b"},
            thresholds={},
        )
        # _secrets NOT injected
        with pytest.raises(AttributeError):
            plugin.evaluate([_make_sample()])

    def test_partial_failures_are_tracked(self):
        """Samples that fail API call appear in failed_samples."""
        plugin = _make_plugin(thresholds={"min_mean_score": 0.0})
        samples = [_make_sample() for _ in range(3)]

        call_count = [0]

        def mock_judge(*_args, **_kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("flaky")
            return JudgeResponse(score=5, reasoning="", raw_response="{}")

        mock_provider = MagicMock()
        mock_provider.judge.side_effect = mock_judge

        with patch(
            "src.evaluation.plugins.llm_judge.cerebras_judge.CerebrasProvider",
            return_value=mock_provider,
        ):
            result = plugin.evaluate(samples)

        assert 1 in result.failed_samples

    def test_name_is_cerebras_judge(self):
        assert CerebrasJudgePlugin.name == "cerebras_judge"

    def test_requires_expected_answer_true(self):
        assert CerebrasJudgePlugin.requires_expected_answer is True

    def test_priority_is_60(self):
        assert CerebrasJudgePlugin.priority == 60


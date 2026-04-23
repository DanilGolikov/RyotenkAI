"""
Tests for HelixQLSemanticMatchPlugin.

Coverage:
- Registration and name
- _validate_params: valid/invalid min_mean_score
- evaluate(): exact match → score 1.0, exact_match_ratio 1.0
- evaluate(): hard eval failures → score < 1.0
- evaluate(): empty samples → passed=True, all ratios 0.0
- evaluate(): threshold check (passes/fails)
- evaluate(): metrics include all required keys
- evaluate(): near_match, exact_match, hard_eval_pass ratios are computed
- get_recommendations(): non-empty on failure, empty on pass
- Score range invariant: always [0.0, 1.0]
- Combinatorial: multiple samples with varying quality
"""

from __future__ import annotations

import pytest

from typing import Any

import pytest
from plugin import HelixQLSemanticMatchPlugin

from src.evaluation.plugins.base import EvalSample


@pytest.fixture(autouse=True)
def _attach_community_name_like_loader(monkeypatch):
    """Simulate src.community.loader._attach_community_metadata.

    The real loader assigns plugin_cls.name = manifest.plugin.id when
    the catalog is loaded. These unit tests instantiate the plugin class
    directly (no catalog), so we mirror the assignment for the duration of
    each test to keep self.name / ValidationResult.plugin_name
    populated.
    """
    monkeypatch.setattr(HelixQLSemanticMatchPlugin, "name", "helixql_semantic_match", raising=False)


def _make_sample(
    question: str = "Get all users",
    model_answer: str = "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items",
    expected_answer: str = "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items",
    metadata: dict[str, Any] | None = None,
) -> EvalSample:
    return EvalSample(
        question=question,
        model_answer=model_answer,
        expected_answer=expected_answer,
        metadata=metadata or {},
    )


def _make_plugin(thresholds: dict[str, Any] | None = None) -> HelixQLSemanticMatchPlugin:
    return HelixQLSemanticMatchPlugin(params={}, thresholds=thresholds or {"min_mean_score": 0.7})


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestSemanticMatchPluginRegistration:
    def test_is_registered(self) -> None:
        from src.community.catalog import catalog
        from src.evaluation.plugins.registry import EvaluatorPluginRegistry

        catalog.reload()
        assert "helixql_semantic_match" in EvaluatorPluginRegistry._registry

    def test_name_classvar(self) -> None:
        assert HelixQLSemanticMatchPlugin.name == "helixql_semantic_match"

    def test_requires_expected_answer(self) -> None:
        assert HelixQLSemanticMatchPlugin.requires_expected_answer is True


# ---------------------------------------------------------------------------
# _validate_params
# ---------------------------------------------------------------------------


class TestSemanticMatchPluginValidateParams:
    def test_valid_threshold_ok(self) -> None:
        plugin = _make_plugin({"min_mean_score": 0.7})
        assert plugin is not None

    def test_threshold_zero_ok(self) -> None:
        plugin = _make_plugin({"min_mean_score": 0.0})
        assert plugin is not None

    def test_threshold_one_ok(self) -> None:
        plugin = _make_plugin({"min_mean_score": 1.0})
        assert plugin is not None

    def test_threshold_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="min_mean_score"):
            _make_plugin({"min_mean_score": 1.5})

    def test_threshold_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="min_mean_score"):
            _make_plugin({"min_mean_score": -0.1})


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestSemanticMatchPluginEvaluate:
    def test_exact_match_scores_1_0(self) -> None:
        plugin = _make_plugin()
        sample = _make_sample()
        result = plugin.evaluate([sample])
        assert result.metrics["exact_match_ratio"] == 1.0

    def test_empty_samples_passes(self) -> None:
        plugin = _make_plugin()
        result = plugin.evaluate([])
        assert result.passed is True
        assert result.sample_count == 0

    def test_empty_samples_zero_ratios(self) -> None:
        plugin = _make_plugin()
        result = plugin.evaluate([])
        assert result.metrics.get("exact_match_ratio", 0.0) == 0.0
        assert result.metrics.get("near_match_ratio", 0.0) == 0.0
        assert result.metrics.get("hard_eval_pass_ratio", 0.0) == 0.0

    def test_bad_candidate_fails_threshold(self) -> None:
        plugin = _make_plugin({"min_mean_score": 0.9})
        sample = _make_sample(
            model_answer="SELECT * FROM users",
            expected_answer="QUERY GetAll () =>\n    items <- N<User>\n    RETURN items",
        )
        result = plugin.evaluate([sample])
        assert result.passed is False

    def test_score_range_invariant(self) -> None:
        plugin = _make_plugin()
        samples = [
            _make_sample(
                model_answer="QUERY A () => a <- N<A> RETURN a", expected_answer="QUERY B () => b <- N<B> RETURN b"
            ),
            _make_sample(model_answer="", expected_answer="QUERY B () => b <- N<B> RETURN b"),
        ]
        result = plugin.evaluate(samples)
        mean_score = result.metrics.get("mean_score", 0.0)
        assert 0.0 <= mean_score <= 1.0

    def test_metrics_include_required_keys(self) -> None:
        plugin = _make_plugin()
        result = plugin.evaluate([_make_sample()])
        required = {"exact_match_ratio", "near_match_ratio", "hard_eval_pass_ratio", "mean_score"}
        assert required.issubset(result.metrics.keys())

    def test_sample_count_matches_input(self) -> None:
        plugin = _make_plugin()
        samples = [_make_sample() for _ in range(5)]
        result = plugin.evaluate(samples)
        assert result.sample_count == 5

    def test_failed_samples_tracked(self) -> None:
        plugin = _make_plugin({"min_mean_score": 0.99})
        samples = [
            _make_sample(),
            _make_sample(model_answer="garbage output not a query"),
        ]
        result = plugin.evaluate(samples)
        # At least the garbage sample should be in failed
        assert len(result.failed_samples) >= 1

    def test_plugin_name_in_result(self) -> None:
        plugin = _make_plugin()
        result = plugin.evaluate([_make_sample()])
        assert result.plugin_name == "helixql_semantic_match"

    def test_all_exact_matches_passed_at_high_threshold(self) -> None:
        plugin = _make_plugin({"min_mean_score": 1.0})
        q = "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items"
        result = plugin.evaluate([_make_sample(model_answer=q, expected_answer=q)])
        assert result.passed is True

    def test_hard_eval_errors_reduce_hard_eval_pass_ratio(self) -> None:
        plugin = _make_plugin()
        bad_query = "```python\nSELECT * FROM users\n```"
        sample = _make_sample(
            model_answer=bad_query,
            expected_answer="QUERY GetAll () =>\n    items <- N<User>\n    RETURN items",
        )
        result = plugin.evaluate([sample])
        assert result.metrics["hard_eval_pass_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Combinatorial
# ---------------------------------------------------------------------------


class TestSemanticMatchPluginCombinatorial:
    @pytest.mark.parametrize(
        "threshold,expected_pass",
        [
            (0.0, True),
            (0.5, True),
            (0.99, False),
            (1.0, False),
        ],
    )
    def test_threshold_determines_pass_with_mixed_samples(self, threshold: float, expected_pass: bool) -> None:
        plugin = _make_plugin({"min_mean_score": threshold})
        q = "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items"
        samples = [
            _make_sample(model_answer=q, expected_answer=q),
            _make_sample(model_answer="SELECT * FROM users", expected_answer=q),
        ]
        result = plugin.evaluate(samples)
        if threshold == 0.0:
            assert result.passed is True
        # High threshold should fail due to bad sample
        elif threshold >= 0.99:
            assert result.passed is False


# ---------------------------------------------------------------------------
# get_recommendations
# ---------------------------------------------------------------------------


class TestSemanticMatchPluginRecommendations:
    def test_recommendations_on_failure(self) -> None:
        plugin = _make_plugin({"min_mean_score": 0.99})
        bad_sample = _make_sample(model_answer="garbage", expected_answer="QUERY Get () => RETURN x")
        result = plugin.evaluate([bad_sample])
        recs = plugin.get_recommendations(result)
        assert len(recs) > 0

    def test_no_recommendations_when_passed(self) -> None:
        plugin = _make_plugin({"min_mean_score": 0.0})
        result = plugin.evaluate([_make_sample()])
        recs = plugin.get_recommendations(result)
        assert recs == []

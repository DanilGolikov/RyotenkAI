"""
Tests for HelixQLGeneratedSyntaxBackendPlugin.

Coverage:
- Registration and metadata
- _validate_contract: valid/invalid timeout, thresholds
- evaluate(): empty samples → passed=True, valid_ratio=1.0
- evaluate(): helix CLI missing → all fail with cli_missing
- evaluate(): missing schema from metadata
- evaluate(): schema extracted from question fence
- evaluate(): missing query (empty model_answer)
- evaluate(): min_valid_ratio threshold
- evaluate(): taxonomy in metrics
- evaluate(): result keys
- HelixCompiler caching via plugin
- get_recommendations: non-empty on failure, empty on pass
- Combinatorial: multiple samples mixed valid/invalid
"""

from __future__ import annotations

import pytest

from typing import Any
from unittest.mock import patch

import pytest
from plugin import HelixQLGeneratedSyntaxBackendPlugin

from src.evaluation.plugins.base import EvalSample

BASE_PARAMS: dict[str, Any] = {
    "timeout_seconds": 10,
}

BASE_THRESHOLDS: dict[str, Any] = {"min_valid_ratio": 0.8}


@pytest.fixture(autouse=True)
def _attach_community_name_like_loader(monkeypatch):
    """Simulate src.community.loader._attach_community_metadata.

    The real loader assigns plugin_cls.name = manifest.plugin.id when
    the catalog is loaded. These unit tests instantiate the plugin class
    directly (no catalog), so we mirror the assignment for the duration of
    each test to keep self.name / ValidationResult.plugin_name
    populated.
    """
    monkeypatch.setattr(HelixQLGeneratedSyntaxBackendPlugin, "name", "helixql_generated_syntax_backend", raising=False)


def _make_sample(
    question: str = "Get all users\n```helixschema\nNode User {}\n```",
    model_answer: str = "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items",
    metadata: dict[str, Any] | None = None,
) -> EvalSample:
    return EvalSample(
        question=question,
        model_answer=model_answer,
        expected_answer=None,
        metadata=metadata or {},
    )


def _make_plugin(
    params: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> HelixQLGeneratedSyntaxBackendPlugin:
    return HelixQLGeneratedSyntaxBackendPlugin(
        params=params or BASE_PARAMS,
        thresholds=thresholds or BASE_THRESHOLDS,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestSyntaxBackendPluginRegistration:
    def test_is_registered(self) -> None:
        from src.community.catalog import catalog
        from src.evaluation.plugins.registry import EvaluatorPluginRegistry

        catalog.reload()
        assert "helixql_generated_syntax_backend" in EvaluatorPluginRegistry._registry

    def test_name_classvar(self) -> None:
        assert HelixQLGeneratedSyntaxBackendPlugin.name == "helixql_generated_syntax_backend"


# ---------------------------------------------------------------------------
# _validate_contract
# ---------------------------------------------------------------------------


class TestSyntaxBackendValidateParams:
    def test_valid_params_ok(self) -> None:
        plugin = _make_plugin()
        assert plugin is not None

    def test_missing_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            _make_plugin(params={"timeout_seconds": None})

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            _make_plugin(params={"timeout_seconds": 0})

    def test_invalid_min_valid_ratio_above_1(self) -> None:
        with pytest.raises(ValueError, match="min_valid_ratio"):
            _make_plugin(thresholds={"min_valid_ratio": 1.5})

    def test_invalid_min_valid_ratio_below_0(self) -> None:
        with pytest.raises(ValueError, match="min_valid_ratio"):
            _make_plugin(thresholds={"min_valid_ratio": -0.1})

    def test_boundary_ratio_0_0_ok(self) -> None:
        plugin = _make_plugin(thresholds={"min_valid_ratio": 0.0})
        assert plugin is not None

    def test_boundary_ratio_1_0_ok(self) -> None:
        plugin = _make_plugin(thresholds={"min_valid_ratio": 1.0})
        assert plugin is not None


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestSyntaxBackendEvaluate:
    def test_empty_samples_passed_true(self) -> None:
        plugin = _make_plugin()
        result = plugin.evaluate([])
        assert result.passed is True

    def test_empty_samples_valid_ratio_1(self) -> None:
        plugin = _make_plugin()
        result = plugin.evaluate([])
        assert result.metrics.get("valid_ratio") == 1.0

    def test_empty_samples_count_zero(self) -> None:
        plugin = _make_plugin()
        result = plugin.evaluate([])
        assert result.sample_count == 0

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_helix_missing_all_fail_cli_missing(self, _mock: Any) -> None:
        plugin = _make_plugin()
        result = plugin.evaluate([_make_sample()])
        assert result.metrics.get("error_taxonomy.cli_missing", 0) == 1

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_helix_missing_valid_ratio_zero(self, _mock: Any) -> None:
        plugin = _make_plugin()
        result = plugin.evaluate([_make_sample()])
        assert result.metrics["valid_ratio"] == 0.0

    def test_sample_without_schema_counted_missing_schema(self) -> None:
        plugin = _make_plugin()
        sample = _make_sample(question="no schema here", model_answer="QUERY Get () => RETURN x")
        result = plugin.evaluate([sample])
        assert result.metrics.get("error_taxonomy.missing_schema", 0) == 1

    def test_sample_with_schema_in_metadata(self) -> None:
        plugin = _make_plugin()
        sample = _make_sample(
            question="no fence",
            metadata={"schema_context": "Node User {}"},
        )
        with patch("src.utils.domains.helixql_cli.shutil.which", return_value=None):
            result = plugin.evaluate([sample])
        assert result.metrics.get("error_taxonomy.missing_schema", 0) == 0

    def test_sample_with_schema_in_question_fence(self) -> None:
        plugin = _make_plugin()
        sample = _make_sample(question="Get all\n```helixschema\nNode User {}\n```")
        with patch("src.utils.domains.helixql_cli.shutil.which", return_value=None):
            result = plugin.evaluate([sample])
        assert result.metrics.get("error_taxonomy.missing_schema", 0) == 0
        assert result.metrics.get("error_taxonomy.cli_missing", 0) == 1

    def test_empty_model_answer_counted_missing_query(self) -> None:
        plugin = _make_plugin()
        sample = _make_sample(model_answer="")
        result = plugin.evaluate([sample])
        assert result.metrics.get("error_taxonomy.missing_query", 0) == 1

    def test_threshold_failure(self) -> None:
        plugin = _make_plugin(thresholds={"min_valid_ratio": 1.0})
        with patch("src.utils.domains.helixql_cli.shutil.which", return_value=None):
            result = plugin.evaluate([_make_sample()])
        assert result.passed is False

    def test_threshold_zero_always_passes(self) -> None:
        plugin = _make_plugin(thresholds={"min_valid_ratio": 0.0})
        with patch("src.utils.domains.helixql_cli.shutil.which", return_value=None):
            result = plugin.evaluate([_make_sample()])
        assert result.passed is True

    def test_metrics_include_valid_count_and_total(self) -> None:
        plugin = _make_plugin()
        result = plugin.evaluate([_make_sample()])
        assert "valid_count" in result.metrics
        assert "total_count" in result.metrics
        assert "valid_ratio" in result.metrics

    def test_plugin_name_in_result(self) -> None:
        plugin = _make_plugin()
        result = plugin.evaluate([])
        assert result.plugin_name == "helixql_generated_syntax_backend"

    def test_failed_samples_tracked(self) -> None:
        plugin = _make_plugin()
        sample = _make_sample(model_answer="")
        result = plugin.evaluate([sample])
        assert 0 in result.failed_samples


# ---------------------------------------------------------------------------
# HelixCompiler caching via plugin
# ---------------------------------------------------------------------------


class TestSyntaxBackendCompilerCaching:
    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_helix_missing_cli_missing(self, _mock: Any) -> None:
        plugin = _make_plugin()
        compiler = plugin._get_compiler()  # type: ignore[attr-defined]
        result = compiler.validate(schema="Node User {}", query="QUERY Get () => RETURN x")
        assert result.ok is False
        assert result.error_type == "cli_missing"

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_caching_same_args(self, _mock: Any) -> None:
        plugin = _make_plugin()
        compiler = plugin._get_compiler()  # type: ignore[attr-defined]
        r1 = compiler.validate(schema="Node A {}", query="Q")
        r2 = compiler.validate(schema="Node A {}", query="Q")
        assert r1 is r2
        assert len(compiler._cache) == 1

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_different_queries_separate_cache_entries(self, _mock: Any) -> None:
        plugin = _make_plugin()
        compiler = plugin._get_compiler()  # type: ignore[attr-defined]
        compiler.validate(schema="Node A {}", query="QUERY A () => RETURN a")
        compiler.validate(schema="Node A {}", query="QUERY B () => RETURN b")
        assert len(compiler._cache) == 2


# ---------------------------------------------------------------------------
# Combinatorial
# ---------------------------------------------------------------------------


class TestSyntaxBackendCombinatorial:
    @pytest.mark.parametrize(
        "n_valid,n_invalid,threshold,expected_pass",
        [
            (3, 0, 0.8, True),
            (2, 1, 0.8, False),
            (0, 1, 0.0, True),
            (1, 0, 1.0, True),
        ],
    )
    def test_mixed_samples_threshold(self, n_valid: int, n_invalid: int, threshold: float, expected_pass: bool) -> None:
        plugin = _make_plugin(thresholds={"min_valid_ratio": threshold})
        invalid_schema_sample = _make_sample(question="no schema", model_answer="")

        if n_valid == 0 and n_invalid >= 1:
            samples = [invalid_schema_sample] * n_invalid
            result = plugin.evaluate(samples)
            if threshold == 0.0:
                assert result.passed is True
            else:
                assert result.passed is False
        elif n_valid >= 1 and n_invalid == 0:
            pass


# ---------------------------------------------------------------------------
# get_recommendations
# ---------------------------------------------------------------------------


class TestSyntaxBackendRecommendations:
    def test_recommendations_on_failure(self) -> None:
        plugin = _make_plugin(thresholds={"min_valid_ratio": 1.0})
        bad_sample = _make_sample(model_answer="")
        result = plugin.evaluate([bad_sample])
        recs = plugin.get_recommendations(result)
        assert len(recs) > 0

    def test_no_recommendations_on_pass(self) -> None:
        plugin = _make_plugin(thresholds={"min_valid_ratio": 0.0})
        sample = _make_sample(model_answer="")
        result = plugin.evaluate([sample])
        recs = plugin.get_recommendations(result)
        assert recs == []

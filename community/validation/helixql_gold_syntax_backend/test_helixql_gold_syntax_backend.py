"""
Tests for HelixQLGoldSyntaxBackendValidator.

Coverage:
- _validate_contract: valid/invalid timeout, max_error_examples
- _extract_schema_and_query: messages format vs flat format
- HelixCompiler: CLI missing → cli_missing, caching
- validate(): missing schema → taxonomy entry
- validate(): missing query → taxonomy entry
- validate(): helix CLI missing → all fail with cli_missing
- validate(): pass_rate threshold
- validate(): empty dataset → warning
- Registration and name
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from plugin import HelixQLGoldSyntaxBackendValidator

BASE_PARAMS: dict[str, Any] = {  # noqa: WPS407
    "timeout_seconds": 10,
}
BASE_THRESHOLDS: dict[str, Any] = {"min_pass_rate": 1.0}  # noqa: WPS407

VALID_SAMPLE: dict[str, Any] = {  # noqa: WPS407
    "question": "Get all users\n```helixschema\nNode User {}\n```",
    "schema_context": "Node User {}",
    "expected_answer": "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items",
}

MESSAGES_SAMPLE: dict[str, Any] = {  # noqa: WPS407
    "messages": [
        {"role": "user", "content": "Get all users\n```helixschema\nNode User {}\n```"},
        {"role": "assistant", "content": "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items"},
    ]
}


def _make_plugin(config: dict[str, Any] | None = None) -> HelixQLGoldSyntaxBackendValidator:
    params = dict(BASE_PARAMS)
    thresholds = dict(BASE_THRESHOLDS)
    for key, value in (config or {}).items():
        if key in {"timeout_seconds", "max_error_examples"}:
            params[key] = value
        else:
            thresholds[key] = value
    return HelixQLGoldSyntaxBackendValidator(params, thresholds)


def _run_validate(samples: list[dict[str, Any]], config: dict[str, Any] | None = None) -> Any:
    plugin = _make_plugin(config or BASE_PARAMS)
    plugin._get_samples_for_validation = lambda dataset, **_kw: dataset  # type: ignore[assignment]
    return plugin.validate(samples)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestGoldSyntaxValidatorRegistration:
    def test_is_registered(self) -> None:
        from src.community.catalog import catalog
        from src.data.validation.registry import ValidationPluginRegistry

        catalog.reload()
        assert "helixql_gold_syntax_backend" in ValidationPluginRegistry.list_plugins()

    def test_name_classvar(self) -> None:
        assert HelixQLGoldSyntaxBackendValidator.name == "helixql_gold_syntax_backend"

    def test_priority_classvar(self) -> None:
        assert HelixQLGoldSyntaxBackendValidator.priority == 40  # noqa: WPS432


# ---------------------------------------------------------------------------
# _validate_contract
# ---------------------------------------------------------------------------


class TestGoldSyntaxValidatorConfig:
    def test_valid_config_ok(self) -> None:
        plugin = _make_plugin({"timeout_seconds": 10})
        assert plugin is not None

    def test_missing_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            _make_plugin({"timeout_seconds": None})

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            _make_plugin({"timeout_seconds": 0})

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            _make_plugin({"timeout_seconds": -5})

    def test_invalid_max_error_examples_raises(self) -> None:
        with pytest.raises(ValueError, match="max_error_examples"):
            _make_plugin({"timeout_seconds": 10, "max_error_examples": -2})


# ---------------------------------------------------------------------------
# _extract_schema_and_query
# ---------------------------------------------------------------------------


class TestExtractSchemaAndQuery:
    def test_flat_sample_with_schema_context_field(self) -> None:
        sample = {
            "schema_context": "Node User {}",
            "expected_answer": "QUERY Get () => RETURN x",
        }
        schema, query = HelixQLGoldSyntaxBackendValidator._extract_schema_and_query(sample)  # type: ignore[attr-defined]
        assert schema == "Node User {}"
        assert query == "QUERY Get () => RETURN x"

    def test_flat_sample_extracts_schema_from_fence(self) -> None:
        sample = {
            "question": "What?\n```helixschema\nNode User {}\n```",
            "expected_answer": "QUERY Get () => RETURN x",
        }
        schema, _query = HelixQLGoldSyntaxBackendValidator._extract_schema_and_query(sample)  # type: ignore[attr-defined]
        assert "Node User {}" in schema

    def test_messages_sample_extraction(self) -> None:
        schema, query = HelixQLGoldSyntaxBackendValidator._extract_schema_and_query(MESSAGES_SAMPLE)  # type: ignore[attr-defined]
        assert "Node User {}" in schema
        assert "QUERY GetAll" in query

    def test_messages_without_assistant_returns_empty_query(self) -> None:
        sample: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "```helixschema\nNode User {}\n```"},
            ]
        }
        _, query = HelixQLGoldSyntaxBackendValidator._extract_schema_and_query(sample)  # type: ignore[attr-defined]
        assert query == ""

    def test_reference_answer_used_as_fallback(self) -> None:
        sample = {
            "schema_context": "Node User {}",
            "reference_answer": "QUERY GetAll () => RETURN items",
        }
        _, query = HelixQLGoldSyntaxBackendValidator._extract_schema_and_query(sample)  # type: ignore[attr-defined]
        assert "QUERY GetAll" in query

    def test_empty_sample_returns_empty_strings(self) -> None:
        schema, query = HelixQLGoldSyntaxBackendValidator._extract_schema_and_query({})  # type: ignore[attr-defined]
        assert schema == ""
        assert query == ""


# ---------------------------------------------------------------------------
# HelixCompiler via plugin
# ---------------------------------------------------------------------------


class TestCompilerViaPlugin:
    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_helix_missing_returns_cli_missing(self, _mock: Any) -> None:
        plugin = _make_plugin()
        compiler = plugin._get_compiler()  # type: ignore[attr-defined]
        result = compiler.validate(schema="Node User {}", query="QUERY Get () => RETURN x")
        assert result.ok is False
        assert result.error_type == "cli_missing"

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_caches_result(self, _mock: Any) -> None:
        plugin = _make_plugin()
        compiler = plugin._get_compiler()  # type: ignore[attr-defined]
        r1 = compiler.validate(schema="Node A {}", query="QUERY A () => RETURN a")
        r2 = compiler.validate(schema="Node A {}", query="QUERY A () => RETURN a")
        assert r1 is r2
        assert len(compiler._cache) == 1

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_different_queries_cached_separately(self, _mock: Any) -> None:
        plugin = _make_plugin()
        compiler = plugin._get_compiler()  # type: ignore[attr-defined]
        compiler.validate(schema="Node A {}", query="QUERY A () => RETURN a")
        compiler.validate(schema="Node A {}", query="QUERY B () => RETURN b")
        assert len(compiler._cache) == 2


# ---------------------------------------------------------------------------
# validate() end-to-end
# ---------------------------------------------------------------------------


class TestGoldSyntaxValidatorValidate:
    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_helix_missing_all_fail(self, _mock: Any) -> None:
        result = _run_validate([VALID_SAMPLE])
        assert result.passed is False
        taxonomy_keys = [k for k in result.metrics if "cli_missing" in k]
        assert len(taxonomy_keys) > 0

    def test_sample_without_schema_counted_as_missing_schema(self) -> None:
        sample = {"expected_answer": "QUERY Get () => RETURN x"}
        result = _run_validate([sample])
        assert result.metrics.get("error_taxonomy.missing_schema", 0) == 1.0

    def test_sample_without_query_counted_as_missing_query(self) -> None:
        sample = {"schema_context": "Node User {}"}
        result = _run_validate([sample])
        assert result.metrics.get("error_taxonomy.missing_query", 0) == 1.0

    def test_empty_dataset_produces_warning_about_no_samples(self) -> None:
        result = _run_validate([])
        assert any("No samples" in w for w in result.warnings)
        assert result.passed is False  # 0/0=0.0 < default threshold 1.0

    def test_pass_rate_threshold_lower_bound(self) -> None:
        with patch("src.utils.domains.helixql_cli.shutil.which", return_value=None):
            result = _run_validate([VALID_SAMPLE, VALID_SAMPLE], config={**BASE_PARAMS, "min_pass_rate": 0.0})
        assert result.passed is True

    def test_plugin_name_in_result(self) -> None:
        result = _run_validate([])
        assert result.plugin_name == "helixql_gold_syntax_backend"

    def test_metrics_include_pass_rate(self) -> None:
        with patch("src.utils.domains.helixql_cli.shutil.which", return_value=None):
            result = _run_validate([VALID_SAMPLE])
        assert "pass_rate" in result.metrics
        assert "checked_samples" in result.metrics

    def test_default_max_error_examples_is_five(self) -> None:
        samples = [{"expected_answer": "QUERY Get () => RETURN x"} for _ in range(10)]
        result = _run_validate(samples)
        assert len(result.errors) == 1
        assert len(result.error_groups) == 1
        assert result.error_groups[0].error_type == "missing_schema"
        assert result.error_groups[0].sample_indices == [0, 1, 2, 3, 4]
        assert result.error_groups[0].total_count == 10

    def test_max_error_examples_minus_one_keeps_all_examples(self) -> None:
        samples = [{"expected_answer": "QUERY Get () => RETURN x"} for _ in range(10)]
        result = _run_validate(samples, config={**BASE_PARAMS, "max_error_examples": -1})
        assert len(result.errors) == 1
        assert result.error_groups[0].sample_indices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_max_error_examples_zero_suppresses_per_sample_examples(self) -> None:
        samples = [{"expected_answer": "QUERY Get () => RETURN x"} for _ in range(3)]
        result = _run_validate(samples, config={**BASE_PARAMS, "max_error_examples": 0})
        assert len(result.errors) == 1
        assert "pass_rate=" in result.errors[0]
        assert result.error_groups == []

    def test_errors_are_grouped_by_error_type(self) -> None:
        samples = [
            {"expected_answer": "QUERY Get () => RETURN x"},
            {"schema_context": "Node User {}"},
            {"expected_answer": "QUERY Get () => RETURN x"},
            {"schema_context": "Node User {}"},
        ]
        result = _run_validate(samples, config={**BASE_PARAMS, "max_error_examples": -1})
        groups = {g.error_type: g.sample_indices for g in result.error_groups}
        assert groups["missing_schema"] == [0, 2]
        assert groups["missing_query"] == [1, 3]

    def test_recommendations_on_failure(self) -> None:
        with patch("src.utils.domains.helixql_cli.shutil.which", return_value=None):
            result = _run_validate([VALID_SAMPLE])
        plugin = _make_plugin()
        recs = plugin.get_recommendations(result)
        assert len(recs) > 0

    def test_no_recommendations_when_passed(self) -> None:
        result = _run_validate([], config={**BASE_PARAMS, "min_pass_rate": 0.0})
        plugin = _make_plugin()
        recs = plugin.get_recommendations(result)
        assert recs == []

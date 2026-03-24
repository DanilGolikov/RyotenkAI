"""
Tests for HelixQLPreferenceSemanticsValidator.

Coverage:
- _validate_contract: valid/invalid timeout
- _validate_pair: missing schema → issue
- _validate_pair: missing chosen → issue
- _validate_pair: missing rejected → issue
- _validate_pair: helix missing → chosen_cli_missing
- _validate_pair: chosen passes, rejected fails → None (valid pair)
- _validate_pair: chosen score ≤ rejected score → chosen_not_better_than_rejected
- validate(): min_valid_ratio threshold
- validate(): taxonomy in metrics
- Caching behavior
- Registration under correct name
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from src.data.validation.plugins.dpo.helixql_preference_semantics import HelixQLPreferenceSemanticsValidator
from src.utils.domains.helixql_cli import CompileResult

BASE_PARAMS: dict[str, Any] = {  # noqa: WPS407
    "timeout_seconds": 10,
}
BASE_THRESHOLDS: dict[str, Any] = {"min_valid_ratio": 0.95}  # noqa: WPS407

VALID_PAIR_SAMPLE: dict[str, Any] = {  # noqa: WPS407
    "prompt": "Get all users\n```helixschema\nNode User {}\n```",
    "schema_context": "Node User {}",
    "reference_answer": "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items",
    "chosen": "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items",
    "rejected": "SELECT * FROM users",
}


def _make_plugin() -> HelixQLPreferenceSemanticsValidator:
    return HelixQLPreferenceSemanticsValidator(BASE_PARAMS, BASE_THRESHOLDS)


def _run_validate(samples: list[dict[str, Any]], config: dict[str, Any] | None = None) -> Any:
    params = dict(BASE_PARAMS)
    thresholds = dict(BASE_THRESHOLDS)
    if config:
        for key, value in config.items():
            if key in {"timeout_seconds", "max_error_examples"}:
                params[key] = value
            else:
                thresholds[key] = value
    plugin = HelixQLPreferenceSemanticsValidator(params, thresholds)
    plugin._get_samples_for_validation = lambda dataset, **_kw: dataset  # type: ignore[assignment]
    return plugin.validate(samples)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestDPOValidatorRegistration:
    def test_is_registered(self) -> None:
        from src.data.validation.discovery import ensure_validation_plugins_discovered
        from src.data.validation.registry import ValidationPluginRegistry

        ensure_validation_plugins_discovered(force=True)
        assert "helixql_preference_semantics" in ValidationPluginRegistry.list_plugins()

    def test_name_classvar(self) -> None:
        assert HelixQLPreferenceSemanticsValidator.name == "helixql_preference_semantics"

    def test_priority_classvar(self) -> None:
        assert HelixQLPreferenceSemanticsValidator.priority == 35  # noqa: WPS432


# ---------------------------------------------------------------------------
# _validate_contract
# ---------------------------------------------------------------------------


class TestDPOValidatorConfig:
    def test_valid_config_ok(self) -> None:
        plugin = HelixQLPreferenceSemanticsValidator(BASE_PARAMS, BASE_THRESHOLDS)
        assert plugin is not None

    def test_missing_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            HelixQLPreferenceSemanticsValidator({"timeout_seconds": None})

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            HelixQLPreferenceSemanticsValidator({"timeout_seconds": 0})


# ---------------------------------------------------------------------------
# _validate_pair
# ---------------------------------------------------------------------------


class TestValidatePair:
    def test_missing_schema_context(self) -> None:
        plugin = _make_plugin()
        sample = {**VALID_PAIR_SAMPLE, "schema_context": "", "prompt": "no schema"}
        result = plugin._validate_pair(sample=sample)  # type: ignore[attr-defined]
        assert result == "missing_schema_context"

    def test_missing_chosen(self) -> None:
        plugin = _make_plugin()
        sample = {**VALID_PAIR_SAMPLE, "chosen": ""}
        result = plugin._validate_pair(sample=sample)  # type: ignore[attr-defined]
        assert result == "missing_chosen"

    def test_missing_rejected(self) -> None:
        plugin = _make_plugin()
        sample = {**VALID_PAIR_SAMPLE, "rejected": ""}
        result = plugin._validate_pair(sample=sample)  # type: ignore[attr-defined]
        assert result == "missing_rejected"

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_helix_missing_chosen_fails_with_cli_missing(self, _mock: Any) -> None:
        plugin = _make_plugin()
        result = plugin._validate_pair(sample=VALID_PAIR_SAMPLE)  # type: ignore[attr-defined]
        assert result is not None
        assert "cli_missing" in result

    def test_exact_same_chosen_and_rejected_returns_not_better(self) -> None:
        plugin = _make_plugin()
        compiler = plugin._get_compiler()  # type: ignore[attr-defined]
        same_query = "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items"
        compiler._cache[("Node User {}", same_query)] = CompileResult(ok=True, error_type="ok")
        sample = {**VALID_PAIR_SAMPLE, "chosen": same_query, "rejected": same_query}
        result = plugin._validate_pair(sample=sample)  # type: ignore[attr-defined]
        assert result == "chosen_not_better_than_rejected"

    def test_chosen_better_than_rejected_returns_none(self) -> None:
        plugin = _make_plugin()
        compiler = plugin._get_compiler()  # type: ignore[attr-defined]
        chosen_q = "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items"
        rejected_q = "SELECT * FROM users"
        compiler._cache[("Node User {}", chosen_q)] = CompileResult(ok=True, error_type="ok")
        compiler._cache[("Node User {}", rejected_q)] = CompileResult(ok=False, error_type="compiler_error")
        result = plugin._validate_pair(sample=VALID_PAIR_SAMPLE)  # type: ignore[attr-defined]
        assert result is None


# ---------------------------------------------------------------------------
# validate() end-to-end
# ---------------------------------------------------------------------------


class TestDPOValidatorValidate:
    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_helix_missing_all_invalid(self, _mock: Any) -> None:
        result = _run_validate([VALID_PAIR_SAMPLE])
        assert result.passed is False

    def test_empty_dataset_with_zero_threshold_passes(self) -> None:
        result = _run_validate([], config={**BASE_PARAMS, "min_valid_ratio": 0.0})
        assert result.passed is True

    def test_malformed_sample_not_counted(self) -> None:
        result = _run_validate(["not_a_dict"])  # type: ignore[list-item]
        taxonomy_keys = [k for k in result.metrics if "malformed_sample" in k]
        assert len(taxonomy_keys) > 0

    def test_valid_ratio_below_threshold_fails(self) -> None:
        with patch("src.utils.domains.helixql_cli.shutil.which", return_value=None):
            result = _run_validate(
                [VALID_PAIR_SAMPLE, VALID_PAIR_SAMPLE], config={**BASE_PARAMS, "min_valid_ratio": 1.0}
            )
        assert result.passed is False

    def test_valid_ratio_with_zero_threshold_always_passes(self) -> None:
        result = _run_validate([], config={**BASE_PARAMS, "min_valid_ratio": 0.0})
        assert result.passed is True

    def test_plugin_name_in_result(self) -> None:
        result = _run_validate([])
        assert result.plugin_name == "helixql_preference_semantics"

    def test_metrics_include_valid_ratio(self) -> None:
        result = _run_validate([])
        assert "valid_ratio" in result.metrics

    def test_taxonomy_captured_in_metrics(self) -> None:
        with patch("src.utils.domains.helixql_cli.shutil.which", return_value=None):
            result = _run_validate([VALID_PAIR_SAMPLE])
        taxonomy_keys = [k for k in result.metrics if k.startswith("error_taxonomy.")]
        assert len(taxonomy_keys) > 0

    def test_error_groups_collect_indices(self) -> None:
        with patch("src.utils.domains.helixql_cli.shutil.which", return_value=None):
            result = _run_validate([VALID_PAIR_SAMPLE], config={**BASE_PARAMS, "max_error_examples": -1})
        assert len(result.error_groups) >= 1
        assert result.error_groups[0].sample_indices == [0]

    def test_recommendations_on_failure(self) -> None:
        with patch("src.utils.domains.helixql_cli.shutil.which", return_value=None):
            result = _run_validate([VALID_PAIR_SAMPLE])
        plugin = _make_plugin()
        recs = plugin.get_recommendations(result)
        assert len(recs) > 0

    def test_no_recommendations_when_passed(self) -> None:
        result = _run_validate([], config={**BASE_PARAMS, "min_valid_ratio": 0.0})
        plugin = _make_plugin()
        recs = plugin.get_recommendations(result)
        assert recs == []

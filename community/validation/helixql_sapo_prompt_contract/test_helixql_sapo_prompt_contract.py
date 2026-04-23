"""
Tests for HelixQLSAPOPromptContractValidator.

Coverage:
- Valid RL sample with all required fields → pass
- Missing prompt → issue
- Missing schema_context → issue (both as field and in fence)
- Missing reference_answer → issue
- Messages-based sample: extraction from messages list
- Schema extracted from fence in prompt
- pass_rate threshold check
- Empty dataset
- Malformed samples are skipped
- Multiple issues combined
- Recommendations on failure
- Registration under correct name
"""

from __future__ import annotations

from typing import Any

from plugin import HelixQLSAPOPromptContractValidator

VALID_SAMPLE: dict[str, Any] = {  # noqa: WPS407
    "prompt": "What are all users?\n```helixschema\nNode User {}\n```",
    "schema_context": "Node User {}",
    "reference_answer": "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items",
}

VALID_MESSAGES_SAMPLE: dict[str, Any] = {  # noqa: WPS407
    "messages": [
        {"role": "user", "content": "What are all users?\n```helixschema\nNode User {}\n```"},
        {"role": "assistant", "content": "QUERY GetAll () =>\n    items <- N<User>\n    RETURN items"},
    ],
}


def _make_dataset(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return samples


def _run_validate(samples: list[dict[str, Any]], config: dict[str, Any] | None = None) -> Any:
    params: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    for key, value in (config or {}).items():
        if key == "max_error_examples":
            params[key] = value
        else:
            thresholds[key] = value
    plugin = HelixQLSAPOPromptContractValidator(params=params, thresholds=thresholds)
    ds = _make_dataset(samples)
    # Patch _get_samples_for_validation to return list directly
    plugin._get_samples_for_validation = lambda dataset, **_kw: dataset  # type: ignore[assignment]
    return plugin.validate(ds)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestSAPOValidatorRegistration:
    def test_is_registered(self) -> None:
        from src.community.catalog import catalog
        from src.data.validation.registry import ValidationPluginRegistry

        catalog.reload()
        assert "helixql_sapo_prompt_contract" in ValidationPluginRegistry.list_plugins()

    def test_name_classvar(self) -> None:
        """``name`` is attached by the community loader from manifest.plugin.id."""
        from src.community.catalog import catalog
        from src.data.validation.registry import ValidationPluginRegistry

        catalog.reload()
        registered = ValidationPluginRegistry._plugins["helixql_sapo_prompt_contract"]
        assert registered.name == "helixql_sapo_prompt_contract"


# ---------------------------------------------------------------------------
# _sample_issues — unit level
# ---------------------------------------------------------------------------


class TestSAPOValidatorSampleIssues:
    def test_valid_sample_no_issues(self) -> None:
        issues = HelixQLSAPOPromptContractValidator._sample_issues(VALID_SAMPLE)  # type: ignore[attr-defined]
        assert issues == []

    def test_missing_prompt_field_returns_missing_prompt(self) -> None:
        sample = {**VALID_SAMPLE, "prompt": ""}
        issues = HelixQLSAPOPromptContractValidator._sample_issues(sample)  # type: ignore[attr-defined]
        assert "missing_prompt" in issues

    def test_missing_schema_context_returns_issue(self) -> None:
        sample = {
            "prompt": "no schema here",
            "schema_context": "",
            "reference_answer": "QUERY Get () => RETURN x",
        }
        issues = HelixQLSAPOPromptContractValidator._sample_issues(sample)  # type: ignore[attr-defined]
        assert "missing_schema_context" in issues

    def test_schema_in_fence_satisfies_schema_context(self) -> None:
        sample = {
            "prompt": "What are users?\n```helixschema\nNode User {}\n```",
            "schema_context": "",
            "reference_answer": "QUERY Get () => items <- N<User> RETURN items",
        }
        issues = HelixQLSAPOPromptContractValidator._sample_issues(sample)  # type: ignore[attr-defined]
        assert "missing_schema_context" not in issues

    def test_missing_reference_answer_returns_issue(self) -> None:
        sample = {
            "prompt": "What?\n```helixschema\nNode User {}\n```",
            "schema_context": "Node User {}",
            "reference_answer": "",
        }
        issues = HelixQLSAPOPromptContractValidator._sample_issues(sample)  # type: ignore[attr-defined]
        assert "missing_reference_answer" in issues

    def test_multiple_issues_all_returned(self) -> None:
        sample: dict[str, Any] = {}
        issues = HelixQLSAPOPromptContractValidator._sample_issues(sample)  # type: ignore[attr-defined]
        assert "missing_prompt" in issues
        assert "missing_schema_context" in issues
        assert "missing_reference_answer" in issues

    def test_messages_based_sample_valid(self) -> None:
        issues = HelixQLSAPOPromptContractValidator._sample_issues(VALID_MESSAGES_SAMPLE)  # type: ignore[attr-defined]
        assert issues == []

    def test_messages_missing_assistant_returns_reference_issue(self) -> None:
        sample: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": "What?\n```helixschema\nNode User {}\n```"},
            ]
        }
        issues = HelixQLSAPOPromptContractValidator._sample_issues(sample)  # type: ignore[attr-defined]
        assert "missing_reference_answer" in issues

    def test_messages_missing_user_returns_prompt_issue(self) -> None:
        sample: dict[str, Any] = {
            "messages": [
                {"role": "assistant", "content": "QUERY Get () => RETURN x"},
            ]
        }
        issues = HelixQLSAPOPromptContractValidator._sample_issues(sample)  # type: ignore[attr-defined]
        assert "missing_prompt" in issues

    def test_malformed_messages_entries_are_skipped(self) -> None:
        sample: dict[str, Any] = {
            "messages": ["not_a_dict", 42, None],
        }
        # Should not crash, may have issues
        issues = HelixQLSAPOPromptContractValidator._sample_issues(sample)  # type: ignore[attr-defined]
        assert isinstance(issues, list)


# ---------------------------------------------------------------------------
# validate() — end-to-end
# ---------------------------------------------------------------------------


class TestSAPOValidatorValidate:
    def test_all_valid_samples_passes(self) -> None:
        result = _run_validate([VALID_SAMPLE, VALID_SAMPLE])
        assert result.passed is True
        assert result.metrics["pass_rate"] == 1.0

    def test_all_invalid_fails(self) -> None:
        result = _run_validate([{"prompt": "", "schema_context": "", "reference_answer": ""}])
        assert result.passed is False
        assert result.metrics["pass_rate"] == 0.0

    def test_empty_dataset_with_zero_threshold_passes(self) -> None:
        # With 0 checked samples, pass_rate=0.0. Only passes when min_pass_rate=0.0.
        result = _run_validate([], config={"min_pass_rate": 0.0})
        assert result.passed is True

    def test_empty_dataset_default_threshold_fails(self) -> None:
        # Default min_pass_rate=1.0, 0/0 = 0.0 → fails
        result = _run_validate([])
        assert result.passed is False

    def test_pass_rate_threshold_honored(self) -> None:
        samples = [VALID_SAMPLE, VALID_SAMPLE, {"prompt": "", "schema_context": "", "reference_answer": ""}]
        result = _run_validate(samples, config={"min_pass_rate": 0.8})
        # 2/3 ≈ 0.667, below 0.8
        assert result.passed is False

    def test_pass_rate_threshold_passes_with_lower_requirement(self) -> None:
        samples = [VALID_SAMPLE, VALID_SAMPLE, {"prompt": "", "schema_context": "", "reference_answer": ""}]
        result = _run_validate(samples, config={"min_pass_rate": 0.5})
        # 2/3 ≈ 0.667, above 0.5
        assert result.passed is True

    def test_malformed_non_mapping_samples_skipped(self) -> None:
        result = _run_validate(["not_a_dict", 42, None])  # type: ignore[list-item]
        # Malformed samples go to taxonomy but checked=0 → pass_rate=0.0 fails at threshold 1.0
        taxonomy_keys = [k for k in result.metrics if "malformed_sample" in k]
        assert len(taxonomy_keys) > 0

    def test_plugin_name_in_result(self) -> None:
        result = _run_validate([VALID_SAMPLE])
        assert result.plugin_name == "helixql_sapo_prompt_contract"

    def test_metrics_include_required_keys(self) -> None:
        result = _run_validate([VALID_SAMPLE])
        assert "pass_rate" in result.metrics
        assert "checked_samples" in result.metrics
        assert "invalid_count" in result.metrics

    def test_taxonomy_errors_in_metrics(self) -> None:
        invalid_sample = {"prompt": "", "schema_context": "", "reference_answer": ""}
        result = _run_validate([invalid_sample])
        # At least one taxonomy entry present
        taxonomy_keys = [k for k in result.metrics if k.startswith("contract_issues.")]
        assert len(taxonomy_keys) > 0

    def test_error_groups_collect_indices(self) -> None:
        invalid_sample = {"prompt": "", "schema_context": "", "reference_answer": ""}
        result = _run_validate([invalid_sample], config={"max_error_examples": -1})
        groups = {g.error_type: g.sample_indices for g in result.error_groups}
        assert groups["missing_prompt"] == [0]
        assert groups["missing_schema_context"] == [0]
        assert groups["missing_reference_answer"] == [0]


# ---------------------------------------------------------------------------
# get_recommendations
# ---------------------------------------------------------------------------


class TestSAPOValidatorRecommendations:
    def test_recommendations_on_failed_result(self) -> None:
        result = _run_validate([{"prompt": "", "schema_context": "", "reference_answer": ""}])
        plugin = HelixQLSAPOPromptContractValidator({})
        recs = plugin.get_recommendations(result)
        assert len(recs) > 0

    def test_no_recommendations_when_passed(self) -> None:
        result = _run_validate([VALID_SAMPLE])
        plugin = HelixQLSAPOPromptContractValidator({})
        recs = plugin.get_recommendations(result)
        assert recs == []

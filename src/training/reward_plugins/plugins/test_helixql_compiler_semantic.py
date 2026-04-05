"""
Tests for HelixQLCompilerSemanticRewardPlugin.

Coverage:
- _validate_params: no longer checks backend (always compile)
- build_config_kwargs: returns reward_weights with correct values and count
- build_trainer_kwargs: missing required dataset fields
- build_trainer_kwargs: returns reward_funcs only (no reward_weights)
- compiler_reward: helix CLI missing → -1.0 for all
- compiler_reward: empty schema/query → -1.0
- compiler_reward: columns coercion from list/scalar/missing
- semantic_reward: helix CLI missing → 0.0 for all
- semantic_reward: empty schema/query → 0.0
- HelixCompiler: caching behavior
- _coerce_column: edge cases (None, list, scalar, short list)
- Registration: plugin is registered under correct name
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.training.reward_plugins.plugins.helixql_compiler_semantic import (
    HelixQLCompilerSemanticRewardPlugin,
    _coerce_column,
)
from src.training.reward_plugins.registry import RewardPluginRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset_with_features(*fields: str) -> MagicMock:
    ds = MagicMock()
    ds.features = {f: None for f in fields}
    return ds


_DEFAULT_PARAMS: dict[str, Any] = {}  # noqa: WPS407


def _make_plugin(params: dict[str, Any] | None = None) -> HelixQLCompilerSemanticRewardPlugin:
    return HelixQLCompilerSemanticRewardPlugin(_DEFAULT_PARAMS if params is None else params)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestHelixQLCompilerSemanticRewardPluginRegistration:
    def test_is_registered_under_correct_name(self) -> None:
        import src.training.reward_plugins.plugins  # noqa: F401  # type: ignore[import]

        assert "helixql_compiler_semantic" in RewardPluginRegistry._registry

    def test_registered_class_is_correct(self) -> None:
        import src.training.reward_plugins.plugins  # noqa: F401  # type: ignore[import]

        cls = RewardPluginRegistry._registry.get("helixql_compiler_semantic")
        assert cls is HelixQLCompilerSemanticRewardPlugin


# ---------------------------------------------------------------------------
# _validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    def test_empty_params_ok(self) -> None:
        plugin = HelixQLCompilerSemanticRewardPlugin({})
        assert plugin is not None

    def test_custom_timeout_ok(self) -> None:
        plugin = _make_plugin({"timeout_seconds": 30})  # noqa: WPS432
        assert plugin.params["timeout_seconds"] == 30  # noqa: WPS432


# ---------------------------------------------------------------------------
# build_config_kwargs
# ---------------------------------------------------------------------------


class TestBuildConfigKwargs:
    def test_returns_reward_weights(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer", "schema_context")
        result = plugin.build_config_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        assert "reward_weights" in result

    def test_reward_weights_are_1_0(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer", "schema_context")
        result = plugin.build_config_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        assert all(w == 1.0 for w in result["reward_weights"])

    def test_reward_weights_count_matches_reward_funcs(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer", "schema_context")
        config_result = plugin.build_config_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        trainer_result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        assert len(config_result["reward_weights"]) == len(trainer_result["reward_funcs"])

    def test_does_not_contain_reward_funcs(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer")
        result = plugin.build_config_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        assert "reward_funcs" not in result


# ---------------------------------------------------------------------------
# build_trainer_kwargs
# ---------------------------------------------------------------------------


class TestBuildTrainerKwargs:
    def test_returns_reward_funcs(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer", "schema_context")
        result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        assert "reward_funcs" in result

    def test_does_not_contain_reward_weights(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer", "schema_context")
        result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        assert "reward_weights" not in result

    def test_reward_funcs_has_two_functions(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer", "schema_context")
        result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        assert len(result["reward_funcs"]) == 2
        assert all(callable(f) for f in result["reward_funcs"])

    def test_missing_prompt_field_raises(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("reference_answer")
        with pytest.raises(ValueError, match="Missing"):
            plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())

    def test_missing_reference_answer_field_raises(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt")
        with pytest.raises(ValueError, match="Missing"):
            plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())

    def test_both_fields_missing_raises_with_both_listed(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("schema_context")
        with pytest.raises(ValueError) as exc_info:
            plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        assert "prompt" in str(exc_info.value)
        assert "reference_answer" in str(exc_info.value)

    def test_reward_funcs_have_correct_names(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer")
        result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        names = [f.__name__ for f in result["reward_funcs"]]
        assert "compiler_reward" in names
        assert "semantic_reward" in names


# ---------------------------------------------------------------------------
# compiler_reward function behavior
# ---------------------------------------------------------------------------


class TestCompilerRewardFunction:
    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_helix_missing_returns_minus_one(self, _mock_which: Any) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer")
        result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        compiler_reward = result["reward_funcs"][0]
        scores = compiler_reward(
            completions=["QUERY Get () => items <- N<User> RETURN items"],
            prompt=["```helixschema\nNode User {}\n```"],
        )
        assert scores == [-1.0]

    def test_empty_completions_list(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer")
        result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        compiler_reward = result["reward_funcs"][0]
        scores = compiler_reward(completions=[])
        assert scores == []

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_empty_schema_returns_minus_one(self, _mock_which: Any) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer")
        result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        compiler_reward = result["reward_funcs"][0]
        scores = compiler_reward(
            completions=["QUERY Get () => items <- N<User> RETURN items"],
            prompt=["no schema here"],
        )
        assert scores == [-1.0]

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_empty_query_returns_minus_one(self, _mock_which: Any) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer")
        result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        compiler_reward = result["reward_funcs"][0]
        scores = compiler_reward(
            completions=[""],
            prompt=["```helixschema\nNode User {}\n```"],
        )
        assert scores == [-1.0]

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_multiple_completions_all_get_scored(self, _mock_which: Any) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer")
        result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        compiler_reward = result["reward_funcs"][0]
        completions = [
            "QUERY A () => x <- N<User> RETURN x",
            "QUERY B () => y <- N<Order> RETURN y",
        ]
        prompts = [
            "```helixschema\nNode User {}\n```",
            "```helixschema\nNode Order {}\n```",
        ]
        scores = compiler_reward(completions=completions, prompt=prompts)
        assert len(scores) == 2


# ---------------------------------------------------------------------------
# semantic_reward function behavior
# ---------------------------------------------------------------------------


class TestSemanticRewardFunction:
    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_helix_missing_returns_zero(self, _mock_which: Any) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer")
        result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        semantic_reward = result["reward_funcs"][1]
        scores = semantic_reward(
            completions=["QUERY Get () => items <- N<User> RETURN items"],
            prompt=["```helixschema\nNode User {}\n```"],
            reference_answer=["QUERY Get () => items <- N<User> RETURN items"],
        )
        assert scores == [0.0]

    def test_empty_completions_returns_empty(self) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer")
        result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        semantic_reward = result["reward_funcs"][1]
        scores = semantic_reward(completions=[])
        assert scores == []

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_empty_schema_returns_zero(self, _mock_which: Any) -> None:
        plugin = _make_plugin()
        ds = _make_dataset_with_features("prompt", "reference_answer")
        result = plugin.build_trainer_kwargs(train_dataset=ds, phase_config=MagicMock(), pipeline_config=MagicMock())
        semantic_reward = result["reward_funcs"][1]
        scores = semantic_reward(
            completions=["QUERY Get () => x <- N<User> RETURN x"],
            prompt=["no schema here at all"],
            reference_answer=["QUERY Get () => x <- N<User> RETURN x"],
        )
        assert scores == [0.0]


# ---------------------------------------------------------------------------
# HelixCompiler caching
# ---------------------------------------------------------------------------


class TestCompilerCaching:
    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_same_args_returns_cached_result(self, _mock_which: Any) -> None:
        plugin = _make_plugin()
        compiler = plugin._get_compiler()  # type: ignore[attr-defined]
        r1 = compiler.validate(schema="Node User {}", query="QUERY Get () => x <- N<User> RETURN x")
        r2 = compiler.validate(schema="Node User {}", query="QUERY Get () => x <- N<User> RETURN x")
        assert r1 is r2

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_different_queries_not_cached_together(self, _mock_which: Any) -> None:
        plugin = _make_plugin()
        compiler = plugin._get_compiler()  # type: ignore[attr-defined]
        compiler.validate(schema="Node User {}", query="QUERY A () => x <- N<User> RETURN x")
        compiler.validate(schema="Node User {}", query="QUERY B () => y <- N<Order> RETURN y")
        assert len(compiler._cache) == 2

    @patch("src.utils.domains.helixql_cli.shutil.which", return_value=None)
    def test_helix_missing_sets_cli_missing_error_type(self, _mock_which: Any) -> None:
        plugin = _make_plugin()
        compiler = plugin._get_compiler()  # type: ignore[attr-defined]
        result = compiler.validate(schema="Node User {}", query="QUERY Get () => x <- N<User> RETURN x")
        assert result.ok is False
        assert result.error_type == "cli_missing"


# ---------------------------------------------------------------------------
# _coerce_column
# ---------------------------------------------------------------------------


class TestCoerceColumn:
    def test_none_value_returns_empty_strings(self) -> None:
        result = _coerce_column({}, "missing_key", 3)
        assert result == ["", "", ""]

    def test_list_value_returned_as_is(self) -> None:
        result = _coerce_column({"key": ["a", "b"]}, "key", 2)
        assert result == ["a", "b"]

    def test_list_shorter_than_size_is_padded(self) -> None:
        result = _coerce_column({"key": ["a"]}, "key", 3)
        assert len(result) == 3
        assert result[0] == "a"
        assert result[1] == ""
        assert result[2] == ""

    def test_scalar_string_broadcast_to_size(self) -> None:
        result = _coerce_column({"key": "schema_text"}, "key", 3)
        assert len(result) == 3
        assert all(v == "schema_text" for v in result)

    def test_zero_size_returns_empty_list(self) -> None:
        result = _coerce_column({}, "key", 0)
        assert result == []

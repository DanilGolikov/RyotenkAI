"""
Tests for RewardPlugin.build_config_kwargs() — the new config-routing method.

Coverage categories:
- Positive: default implementation returns empty dict
- Positive: subclass override returns expected payload
- Positive: build_config_kwargs and build_trainer_kwargs are independent
- Boundary: all signature params are keyword-only (positional call raises)
- Boundary: default returns {} (not None, not a new dict identity each call)
- Invariant: build_config_kwargs is NOT abstract — concrete subclass need not override
- Invariant: build_trainer_kwargs IS abstract — subclass MUST override it
- Invariant: returned dict is a plain dict (not a subclass)
- Regression: old single-method contract (everything via build_trainer_kwargs) no longer holds
- Logic: override can merge with superclass default via super()
- Combinatorial: plugin with config_kwargs only / trainer_kwargs only / both / neither
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from src.training.reward_plugins.base import RewardPlugin


# ---------------------------------------------------------------------------
# Minimal concrete implementations
# ---------------------------------------------------------------------------


class _NoOverridePlugin(RewardPlugin):
    """Does not override build_config_kwargs — exercises the default."""

    name = "_no_override"

    def build_trainer_kwargs(
        self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
    ) -> dict[str, Any]:
        return {"reward_funcs": ["fn_a"]}


class _ConfigOnlyPlugin(RewardPlugin):
    """Only overrides build_config_kwargs (config-level params, no trainer extras)."""

    name = "_config_only"

    def build_config_kwargs(
        self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
    ) -> dict[str, Any]:
        return {"reward_weights": [0.5, 0.5]}

    def build_trainer_kwargs(
        self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
    ) -> dict[str, Any]:
        return {}


class _TrainerOnlyPlugin(RewardPlugin):
    """Overrides only build_trainer_kwargs (no config-level params)."""

    name = "_trainer_only"

    def build_trainer_kwargs(
        self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
    ) -> dict[str, Any]:
        return {"reward_funcs": ["fn_x", "fn_y"]}


class _BothPlugin(RewardPlugin):
    """Overrides both methods — the typical full-featured plugin."""

    name = "_both"

    def build_config_kwargs(
        self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
    ) -> dict[str, Any]:
        return {"reward_weights": [1.0, 1.0]}

    def build_trainer_kwargs(
        self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
    ) -> dict[str, Any]:
        return {"reward_funcs": ["compiler", "semantic"]}


class _NeitherPlugin(RewardPlugin):
    """No extra config or trainer kwargs (empty on both sides)."""

    name = "_neither"

    def build_trainer_kwargs(
        self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
    ) -> dict[str, Any]:
        return {}


class _SuperCallingPlugin(RewardPlugin):
    """Merges own keys with the parent default via super()."""

    name = "_super_calling"

    def build_config_kwargs(
        self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
    ) -> dict[str, Any]:
        base = super().build_config_kwargs(
            train_dataset=train_dataset,
            phase_config=phase_config,
            pipeline_config=pipeline_config,
        )
        return {**base, "extra_key": "extra_val"}

    def build_trainer_kwargs(
        self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
    ) -> dict[str, Any]:
        return {}


# Shared dummy args — build_config_kwargs ignores them in the base
_DS = object()
_PC = object()
_PL = object()


def _call_config(plugin: RewardPlugin) -> dict[str, Any]:
    return plugin.build_config_kwargs(train_dataset=_DS, phase_config=_PC, pipeline_config=_PL)


def _call_trainer(plugin: RewardPlugin) -> dict[str, Any]:
    return plugin.build_trainer_kwargs(train_dataset=_DS, phase_config=_PC, pipeline_config=_PL)


# ---------------------------------------------------------------------------
# Default implementation (no override)
# ---------------------------------------------------------------------------


class TestDefaultBuildConfigKwargs:
    def test_returns_empty_dict(self) -> None:
        result = _call_config(_NoOverridePlugin({}))
        assert result == {}

    def test_returns_plain_dict_type(self) -> None:
        result = _call_config(_NoOverridePlugin({}))
        assert type(result) is dict  # noqa: E721

    def test_does_not_return_none(self) -> None:
        result = _call_config(_NoOverridePlugin({}))
        assert result is not None

    def test_independent_calls_return_equal_empty_dicts(self) -> None:
        plugin = _NoOverridePlugin({})
        r1 = _call_config(plugin)
        r2 = _call_config(plugin)
        assert r1 == r2 == {}

    def test_mutating_result_does_not_affect_next_call(self) -> None:
        plugin = _NoOverridePlugin({})
        r1 = _call_config(plugin)
        r1["injected"] = "value"
        r2 = _call_config(plugin)
        assert "injected" not in r2


# ---------------------------------------------------------------------------
# Boundary — signature enforcement
# ---------------------------------------------------------------------------


class TestBuildConfigKwargsSignature:
    def test_all_params_are_keyword_only(self) -> None:
        sig = inspect.signature(RewardPlugin.build_config_kwargs)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"Parameter '{name}' must be keyword-only"
            )

    def test_positional_call_raises_type_error(self) -> None:
        plugin = _NoOverridePlugin({})
        with pytest.raises(TypeError):
            plugin.build_config_kwargs(_DS, _PC, _PL)  # type: ignore[call-arg]

    def test_symmetrical_signature_with_build_trainer_kwargs(self) -> None:
        config_params = set(inspect.signature(RewardPlugin.build_config_kwargs).parameters) - {"self"}
        trainer_params = set(inspect.signature(RewardPlugin.build_trainer_kwargs).parameters) - {"self"}
        assert config_params == trainer_params


# ---------------------------------------------------------------------------
# Invariants — abstractness
# ---------------------------------------------------------------------------


class TestAbstractnessinvariants:
    def test_build_config_kwargs_is_not_abstract(self) -> None:
        abstract_methods = RewardPlugin.__abstractmethods__
        assert "build_config_kwargs" not in abstract_methods

    def test_build_trainer_kwargs_is_abstract(self) -> None:
        abstract_methods = RewardPlugin.__abstractmethods__
        assert "build_trainer_kwargs" in abstract_methods

    def test_subclass_without_build_trainer_kwargs_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            class IncompletePlugin(RewardPlugin):  # type: ignore[abstract]
                name = "_incomplete"

            IncompletePlugin({})  # type: ignore[abstract]

    def test_subclass_without_build_config_kwargs_override_is_valid(self) -> None:
        # _NoOverridePlugin does NOT override build_config_kwargs — must succeed
        plugin = _NoOverridePlugin({})
        assert plugin is not None


# ---------------------------------------------------------------------------
# Override behaviour
# ---------------------------------------------------------------------------


class TestBuildConfigKwargsOverride:
    def test_config_only_plugin_returns_reward_weights(self) -> None:
        result = _call_config(_ConfigOnlyPlugin({}))
        assert "reward_weights" in result

    def test_config_only_plugin_reward_weights_values(self) -> None:
        result = _call_config(_ConfigOnlyPlugin({}))
        assert result["reward_weights"] == [0.5, 0.5]

    def test_config_only_trainer_kwargs_empty(self) -> None:
        result = _call_trainer(_ConfigOnlyPlugin({}))
        assert result == {}

    def test_trainer_only_config_kwargs_empty(self) -> None:
        result = _call_config(_TrainerOnlyPlugin({}))
        assert result == {}

    def test_trainer_only_trainer_kwargs_populated(self) -> None:
        result = _call_trainer(_TrainerOnlyPlugin({}))
        assert result["reward_funcs"] == ["fn_x", "fn_y"]

    def test_super_calling_plugin_merges_with_base_empty(self) -> None:
        result = _call_config(_SuperCallingPlugin({}))
        assert result["extra_key"] == "extra_val"

    def test_super_calling_plugin_no_unexpected_keys(self) -> None:
        result = _call_config(_SuperCallingPlugin({}))
        # Only the key explicitly added — base returns {}
        assert set(result.keys()) == {"extra_key"}


# ---------------------------------------------------------------------------
# Combinatorial: config_kwargs vs trainer_kwargs separation
# ---------------------------------------------------------------------------


class TestConfigTrainerSeparation:
    def test_config_only_has_no_reward_funcs(self) -> None:
        result = _call_config(_ConfigOnlyPlugin({}))
        assert "reward_funcs" not in result

    def test_trainer_only_has_no_reward_weights(self) -> None:
        result = _call_trainer(_TrainerOnlyPlugin({}))
        assert "reward_weights" not in result

    def test_both_plugin_config_has_only_config_keys(self) -> None:
        result = _call_config(_BothPlugin({}))
        assert "reward_weights" in result
        assert "reward_funcs" not in result

    def test_both_plugin_trainer_has_only_trainer_keys(self) -> None:
        result = _call_trainer(_BothPlugin({}))
        assert "reward_funcs" in result
        assert "reward_weights" not in result

    def test_neither_plugin_both_empty(self) -> None:
        assert _call_config(_NeitherPlugin({})) == {}
        assert _call_trainer(_NeitherPlugin({})) == {}

    def test_both_results_are_independent_objects(self) -> None:
        plugin = _BothPlugin({})
        config_result = _call_config(plugin)
        trainer_result = _call_trainer(plugin)
        assert config_result is not trainer_result

    @pytest.mark.parametrize("plugin_cls", [_ConfigOnlyPlugin, _TrainerOnlyPlugin, _BothPlugin, _NeitherPlugin, _NoOverridePlugin])
    def test_config_kwargs_is_always_dict(self, plugin_cls) -> None:
        result = _call_config(plugin_cls({}))
        assert isinstance(result, dict)

    @pytest.mark.parametrize("plugin_cls", [_ConfigOnlyPlugin, _TrainerOnlyPlugin, _BothPlugin, _NeitherPlugin, _NoOverridePlugin])
    def test_trainer_kwargs_is_always_dict(self, plugin_cls) -> None:
        result = _call_trainer(plugin_cls({}))
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Regression: old single-method contract no longer valid
# ---------------------------------------------------------------------------


class TestRegression:
    def test_build_trainer_kwargs_no_longer_returns_reward_weights_in_typical_plugin(self) -> None:
        """
        Before the fix, build_trainer_kwargs returned both reward_funcs AND reward_weights.
        After the fix, build_trainer_kwargs must NOT return reward_weights.
        """
        result = _call_trainer(_BothPlugin({}))
        assert "reward_weights" not in result, (
            "reward_weights must NOT appear in build_trainer_kwargs — it belongs in build_config_kwargs"
        )

    def test_build_config_kwargs_does_not_contain_reward_funcs_in_typical_plugin(self) -> None:
        result = _call_config(_BothPlugin({}))
        assert "reward_funcs" not in result, (
            "reward_funcs must NOT appear in build_config_kwargs — it belongs in build_trainer_kwargs"
        )

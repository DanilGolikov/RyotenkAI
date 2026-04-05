"""
Tests for RewardPluginResult NamedTuple and build_reward_plugin_result().

Coverage categories:
- Positive: happy path with registered plugin returns correct split result
- Negative: missing reward_plugin key raises ValueError
- Negative: empty-string reward_plugin raises ValueError
- Negative: whitespace-only reward_plugin raises ValueError
- Negative: None reward_plugin raises ValueError
- Negative: unknown plugin name raises KeyError (plugin not in registry)
- Boundary: reward_plugin param is stripped of surrounding whitespace
- Boundary: reward_params missing from phase_config → defaults to {}
- Boundary: reward_params is None → falls back to {}
- Boundary: reward_params is non-dict type → falls back to {}
- Boundary: reward_params is empty dict → passed as-is
- Invariant: RewardPluginResult is a NamedTuple (immutable, indexable)
- Invariant: both fields are always dicts
- Invariant: config_kwargs and trainer_kwargs are separate objects
- Invariant: calling twice returns equal (but not same-identity) results
- Dependency: ensure_reward_plugins_discovered() is called once per invocation
- Dependency: plugin.build_config_kwargs() called with correct kwargs
- Dependency: plugin.build_trainer_kwargs() called with correct kwargs
- Regression: build_reward_plugin_kwargs name no longer exported from the package
- Logic: config_kwargs comes from build_config_kwargs, trainer_kwargs from build_trainer_kwargs
- Combinatorial: plugin with non-empty config_kwargs + non-empty trainer_kwargs
- Combinatorial: plugin with empty config_kwargs + non-empty trainer_kwargs
- Combinatorial: plugin with non-empty config_kwargs + empty trainer_kwargs
- Combinatorial: plugin with both empty
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from src.training.reward_plugins.factory import RewardPluginResult, build_reward_plugin_result
from src.training.reward_plugins.registry import RewardPluginRegistry


# ---------------------------------------------------------------------------
# Helpers — isolated registry per test
# ---------------------------------------------------------------------------


def _save_and_clear_registry() -> dict[str, Any]:
    original = dict(RewardPluginRegistry._registry)
    RewardPluginRegistry._registry.clear()
    return original


def _restore_registry(original: dict[str, Any]) -> None:
    RewardPluginRegistry._registry.clear()
    RewardPluginRegistry._registry.update(original)


@pytest.fixture()
def clean_registry():
    original = _save_and_clear_registry()
    yield
    _restore_registry(original)


def _make_phase_config(
    plugin_name: str | None = "test_plugin",
    reward_params: Any = None,
    strategy_type: str = "grpo",
) -> MagicMock:
    cfg = MagicMock()
    cfg.strategy_type = strategy_type
    params: dict[str, Any] = {}
    if plugin_name is not None:
        params["reward_plugin"] = plugin_name
    if reward_params is not None:
        params["reward_params"] = reward_params
    cfg.params = params
    return cfg


def _register_plugin(
    name: str,
    config_kwargs: dict[str, Any] | None = None,
    trainer_kwargs: dict[str, Any] | None = None,
) -> None:
    """Register a minimal plugin under `name` that returns fixed dicts."""
    _config = config_kwargs or {}
    _trainer = trainer_kwargs or {}

    from src.training.reward_plugins.base import RewardPlugin

    class _FixedPlugin(RewardPlugin):
        def build_config_kwargs(
            self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
        ) -> dict[str, Any]:
            return dict(_config)

        def build_trainer_kwargs(
            self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
        ) -> dict[str, Any]:
            return dict(_trainer)

    _FixedPlugin.name = name  # type: ignore[assignment]
    RewardPluginRegistry._registry[name] = _FixedPlugin


_DS = MagicMock()
_PL = MagicMock()


# ---------------------------------------------------------------------------
# RewardPluginResult structure
# ---------------------------------------------------------------------------


class TestRewardPluginResultStructure:
    def test_is_named_tuple(self) -> None:
        result = RewardPluginResult(config_kwargs={"a": 1}, trainer_kwargs={"b": 2})
        assert isinstance(result, tuple)

    def test_fields_accessible_by_name(self) -> None:
        result = RewardPluginResult(config_kwargs={"a": 1}, trainer_kwargs={"b": 2})
        assert result.config_kwargs == {"a": 1}
        assert result.trainer_kwargs == {"b": 2}

    def test_fields_accessible_by_index(self) -> None:
        result = RewardPluginResult(config_kwargs={"a": 1}, trainer_kwargs={"b": 2})
        assert result[0] == {"a": 1}
        assert result[1] == {"b": 2}

    def test_is_immutable(self) -> None:
        result = RewardPluginResult(config_kwargs={}, trainer_kwargs={})
        with pytest.raises(AttributeError):
            result.config_kwargs = {"new": "val"}  # type: ignore[misc]

    def test_unpacking_works(self) -> None:
        config_kwargs, trainer_kwargs = RewardPluginResult(config_kwargs={"x": 1}, trainer_kwargs={"y": 2})
        assert config_kwargs == {"x": 1}
        assert trainer_kwargs == {"y": 2}

    def test_equality_by_value(self) -> None:
        r1 = RewardPluginResult(config_kwargs={"w": [1.0]}, trainer_kwargs={"funcs": []})
        r2 = RewardPluginResult(config_kwargs={"w": [1.0]}, trainer_kwargs={"funcs": []})
        assert r1 == r2

    def test_has_exactly_two_fields(self) -> None:
        assert len(RewardPluginResult._fields) == 2
        assert RewardPluginResult._fields == ("config_kwargs", "trainer_kwargs")


# ---------------------------------------------------------------------------
# Positive: happy path
# ---------------------------------------------------------------------------


class TestBuildRewardPluginResultHappyPath:
    def test_returns_reward_plugin_result_instance(self, clean_registry: None) -> None:
        _register_plugin("happy_plugin", config_kwargs={"reward_weights": [1.0]}, trainer_kwargs={"reward_funcs": ["f"]})
        pc = _make_phase_config("happy_plugin")
        result = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert isinstance(result, RewardPluginResult)

    def test_config_kwargs_contains_plugin_config_output(self, clean_registry: None) -> None:
        _register_plugin("cfg_plugin", config_kwargs={"reward_weights": [0.7, 0.3]})
        pc = _make_phase_config("cfg_plugin")
        result = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert result.config_kwargs == {"reward_weights": [0.7, 0.3]}

    def test_trainer_kwargs_contains_plugin_trainer_output(self, clean_registry: None) -> None:
        _register_plugin("trn_plugin", trainer_kwargs={"reward_funcs": ["fn_a", "fn_b"]})
        pc = _make_phase_config("trn_plugin")
        result = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert result.trainer_kwargs == {"reward_funcs": ["fn_a", "fn_b"]}

    def test_config_and_trainer_are_separate_dicts(self, clean_registry: None) -> None:
        _register_plugin("sep_plugin", config_kwargs={"reward_weights": [1.0]}, trainer_kwargs={"reward_funcs": []})
        pc = _make_phase_config("sep_plugin")
        result = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert result.config_kwargs is not result.trainer_kwargs

    def test_result_fields_are_plain_dicts(self, clean_registry: None) -> None:
        _register_plugin("plain_plugin")
        pc = _make_phase_config("plain_plugin")
        result = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert type(result.config_kwargs) is dict  # noqa: E721
        assert type(result.trainer_kwargs) is dict  # noqa: E721


# ---------------------------------------------------------------------------
# Negative: missing / empty / None plugin name
# ---------------------------------------------------------------------------


class TestBuildRewardPluginResultMissingPlugin:
    def test_no_reward_plugin_key_raises_value_error(self, clean_registry: None) -> None:
        pc = _make_phase_config(plugin_name=None)
        with pytest.raises(ValueError, match="reward_plugin"):
            build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)

    def test_empty_string_plugin_name_raises_value_error(self, clean_registry: None) -> None:
        pc = _make_phase_config(plugin_name="")
        with pytest.raises(ValueError, match="reward_plugin"):
            build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)

    def test_whitespace_only_plugin_name_raises_value_error(self, clean_registry: None) -> None:
        pc = _make_phase_config(plugin_name="   ")
        with pytest.raises(ValueError, match="reward_plugin"):
            build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)

    def test_none_plugin_name_raises_value_error(self, clean_registry: None) -> None:
        pc = MagicMock()
        pc.strategy_type = "grpo"
        pc.params = {"reward_plugin": None}
        with pytest.raises(ValueError, match="reward_plugin"):
            build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)

    def test_value_error_mentions_strategy_type(self, clean_registry: None) -> None:
        pc = _make_phase_config(plugin_name="", strategy_type="sapo")
        with pytest.raises(ValueError) as exc_info:
            build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert "SAPO" in str(exc_info.value)

    def test_unknown_plugin_name_raises_key_error(self, clean_registry: None) -> None:
        pc = _make_phase_config("totally_unknown_plugin_xyz")
        with pytest.raises(KeyError):
            build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)


# ---------------------------------------------------------------------------
# Boundary: plugin_name stripping and reward_params coercion
# ---------------------------------------------------------------------------


class TestBuildRewardPluginResultBoundary:
    def test_plugin_name_is_stripped_of_whitespace(self, clean_registry: None) -> None:
        _register_plugin("padded_plugin")
        pc = _make_phase_config("  padded_plugin  ")
        # Must not raise — whitespace stripped before registry lookup
        result = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert isinstance(result, RewardPluginResult)

    def test_reward_params_missing_uses_empty_dict(self, clean_registry: None) -> None:
        received_params: list[dict] = []

        from src.training.reward_plugins.base import RewardPlugin

        class _ParamCapture(RewardPlugin):
            name = "param_capture"

            def __init__(self, params: dict) -> None:
                received_params.append(params)
                super().__init__(params)

            def build_trainer_kwargs(self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any) -> dict:
                return {}

        RewardPluginRegistry._registry["param_capture"] = _ParamCapture

        pc = _make_phase_config("param_capture", reward_params=None)
        # Remove reward_params from params entirely to test the "missing" path
        pc.params = {"reward_plugin": "param_capture"}
        build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert received_params[-1] == {}

    def test_reward_params_none_value_falls_back_to_empty_dict(self, clean_registry: None) -> None:
        received_params: list[dict] = []

        from src.training.reward_plugins.base import RewardPlugin

        class _ParamCapture2(RewardPlugin):
            name = "param_capture_2"

            def __init__(self, params: dict) -> None:
                received_params.append(params)
                super().__init__(params)

            def build_trainer_kwargs(self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any) -> dict:
                return {}

        RewardPluginRegistry._registry["param_capture_2"] = _ParamCapture2

        pc = MagicMock()
        pc.strategy_type = "grpo"
        pc.params = {"reward_plugin": "param_capture_2", "reward_params": None}
        build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert received_params[-1] == {}

    def test_reward_params_non_dict_falls_back_to_empty_dict(self, clean_registry: None) -> None:
        received_params: list[dict] = []

        from src.training.reward_plugins.base import RewardPlugin

        class _ParamCapture3(RewardPlugin):
            name = "param_capture_3"

            def __init__(self, params: dict) -> None:
                received_params.append(params)
                super().__init__(params)

            def build_trainer_kwargs(self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any) -> dict:
                return {}

        RewardPluginRegistry._registry["param_capture_3"] = _ParamCapture3

        pc = MagicMock()
        pc.strategy_type = "grpo"
        pc.params = {"reward_plugin": "param_capture_3", "reward_params": "not_a_dict"}
        build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert received_params[-1] == {}

    def test_reward_params_empty_dict_passed_as_is(self, clean_registry: None) -> None:
        received_params: list[dict] = []

        from src.training.reward_plugins.base import RewardPlugin

        class _ParamCapture4(RewardPlugin):
            name = "param_capture_4"

            def __init__(self, params: dict) -> None:
                received_params.append(params)
                super().__init__(params)

            def build_trainer_kwargs(self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any) -> dict:
                return {}

        RewardPluginRegistry._registry["param_capture_4"] = _ParamCapture4

        pc = MagicMock()
        pc.strategy_type = "grpo"
        pc.params = {"reward_plugin": "param_capture_4", "reward_params": {}}
        build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert received_params[-1] == {}


# ---------------------------------------------------------------------------
# Dependency: correct args forwarded to plugin methods
# ---------------------------------------------------------------------------


class TestBuildRewardPluginResultDependencies:
    def test_ensure_plugins_discovered_is_called(self, clean_registry: None) -> None:
        _register_plugin("disc_plugin")
        pc = _make_phase_config("disc_plugin")

        with patch(
            "src.training.reward_plugins.factory.ensure_reward_plugins_discovered"
        ) as mock_disc:
            # Re-register after patch because discovery may have already run
            _register_plugin("disc_plugin")
            build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
            mock_disc.assert_called_once()

    def test_build_config_kwargs_receives_correct_args(self, clean_registry: None) -> None:
        from src.training.reward_plugins.base import RewardPlugin

        recorded: list[dict] = []

        class _RecordPlugin(RewardPlugin):
            name = "record_cfg"

            def build_config_kwargs(self, *, train_dataset, phase_config, pipeline_config) -> dict:
                recorded.append({"ds": train_dataset, "pc": phase_config, "pl": pipeline_config})
                return {}

            def build_trainer_kwargs(self, *, train_dataset, phase_config, pipeline_config) -> dict:
                return {}

        RewardPluginRegistry._registry["record_cfg"] = _RecordPlugin

        ds = object()
        pl = object()
        pc = _make_phase_config("record_cfg")
        build_reward_plugin_result(train_dataset=ds, phase_config=pc, pipeline_config=pl)

        assert len(recorded) == 1
        assert recorded[0]["ds"] is ds
        assert recorded[0]["pc"] is pc
        assert recorded[0]["pl"] is pl

    def test_build_trainer_kwargs_receives_correct_args(self, clean_registry: None) -> None:
        from src.training.reward_plugins.base import RewardPlugin

        recorded: list[dict] = []

        class _RecordPlugin2(RewardPlugin):
            name = "record_trn"

            def build_config_kwargs(self, *, train_dataset, phase_config, pipeline_config) -> dict:
                return {}

            def build_trainer_kwargs(self, *, train_dataset, phase_config, pipeline_config) -> dict:
                recorded.append({"ds": train_dataset, "pc": phase_config, "pl": pipeline_config})
                return {}

        RewardPluginRegistry._registry["record_trn"] = _RecordPlugin2

        ds = object()
        pl = object()
        pc = _make_phase_config("record_trn")
        build_reward_plugin_result(train_dataset=ds, phase_config=pc, pipeline_config=pl)

        assert len(recorded) == 1
        assert recorded[0]["ds"] is ds
        assert recorded[0]["pc"] is pc
        assert recorded[0]["pl"] is pl


# ---------------------------------------------------------------------------
# Invariant: idempotency
# ---------------------------------------------------------------------------


class TestBuildRewardPluginResultInvariants:
    def test_two_calls_return_equal_results(self, clean_registry: None) -> None:
        _register_plugin("idem_plugin", config_kwargs={"w": [1.0]}, trainer_kwargs={"funcs": []})
        pc = _make_phase_config("idem_plugin")
        r1 = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        r2 = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert r1 == r2

    def test_mutating_result_config_kwargs_does_not_affect_next_call(self, clean_registry: None) -> None:
        _register_plugin("mut_plugin", config_kwargs={"reward_weights": [1.0]})
        pc = _make_phase_config("mut_plugin")
        r1 = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        r1.config_kwargs["injected"] = "bad"
        r2 = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert "injected" not in r2.config_kwargs


# ---------------------------------------------------------------------------
# Regression: old API removed
# ---------------------------------------------------------------------------


class TestRegression:
    def test_build_reward_plugin_kwargs_not_exported_from_package(self) -> None:
        import src.training.reward_plugins as pkg

        assert not hasattr(pkg, "build_reward_plugin_kwargs"), (
            "build_reward_plugin_kwargs was replaced by build_reward_plugin_result — "
            "it must not be exported anymore"
        )

    def test_build_reward_plugin_kwargs_not_in_factory_module(self) -> None:
        import src.training.reward_plugins.factory as fac

        assert not hasattr(fac, "build_reward_plugin_kwargs"), (
            "build_reward_plugin_kwargs was removed — factory must not expose the old name"
        )

    def test_result_is_namedtuple_not_plain_dict(self, clean_registry: None) -> None:
        _register_plugin("legacy_check_plugin")
        pc = _make_phase_config("legacy_check_plugin")
        result = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert not isinstance(result, dict), "result must be RewardPluginResult, not a plain dict"


# ---------------------------------------------------------------------------
# Combinatorial
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize(
        "config_out,trainer_out",
        [
            ({"reward_weights": [1.0, 1.0]}, {"reward_funcs": ["f1", "f2"]}),
            ({}, {"reward_funcs": ["f1"]}),
            ({"reward_weights": [0.5]}, {}),
            ({}, {}),
        ],
    )
    def test_result_fields_match_plugin_output(
        self,
        clean_registry: None,
        config_out: dict,
        trainer_out: dict,
    ) -> None:
        _register_plugin("combo_plugin", config_kwargs=config_out, trainer_kwargs=trainer_out)
        pc = _make_phase_config("combo_plugin")
        result = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert result.config_kwargs == config_out
        assert result.trainer_kwargs == trainer_out

    @pytest.mark.parametrize("strategy_type", ["grpo", "sapo"])
    def test_works_for_all_rl_strategy_types(self, clean_registry: None, strategy_type: str) -> None:
        _register_plugin(f"rl_{strategy_type}_plugin")
        pc = _make_phase_config(f"rl_{strategy_type}_plugin", strategy_type=strategy_type)
        result = build_reward_plugin_result(train_dataset=_DS, phase_config=pc, pipeline_config=_PL)
        assert isinstance(result, RewardPluginResult)

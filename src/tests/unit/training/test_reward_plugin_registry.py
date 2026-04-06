"""
Tests for RewardPluginRegistry and RewardPlugin ABC.

Coverage:
- Pattern B registration: cls.name read by decorator
- Subclass with empty name raises ValueError
- Duplicate registration overwrites
- create() happy path
- create() unknown name raises KeyError
- RewardPlugin.__init__ calls _validate_params
- _validate_params override raises on bad params
- build_trainer_kwargs is abstract
- Isolation between test cases (clean registry state)
"""

from __future__ import annotations

from typing import Any

import pytest

from src.training.reward_plugins.base import RewardPlugin
from src.training.reward_plugins.registry import RewardPluginRegistry

# ---------------------------------------------------------------------------
# Helpers — fresh registry per test
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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRewardPluginRegistration:
    def test_register_reads_name_from_classvar(self, clean_registry: None) -> None:
        @RewardPluginRegistry.register
        class DummyPlugin(RewardPlugin):
            name = "dummy_reward"

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {}

        assert "dummy_reward" in RewardPluginRegistry._registry

    def test_register_empty_name_raises_value_error(self, clean_registry: None) -> None:
        with pytest.raises(ValueError, match="non-empty 'name' ClassVar"):

            @RewardPluginRegistry.register
            class NoNamePlugin(RewardPlugin):
                name = ""

                def build_trainer_kwargs(
                    self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
                ) -> dict[str, Any]:
                    return {}

    def test_register_returns_class_unchanged(self, clean_registry: None) -> None:
        class Original(RewardPlugin):
            name = "original_reward"

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {}

        returned = RewardPluginRegistry.register(Original)
        assert returned is Original

    def test_duplicate_registration_overwrites(self, clean_registry: None) -> None:
        class PluginV1(RewardPlugin):
            name = "versioned_reward"

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {"v": 1}

        class PluginV2(RewardPlugin):
            name = "versioned_reward"

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {"v": 2}

        RewardPluginRegistry.register(PluginV1)
        RewardPluginRegistry.register(PluginV2)
        assert RewardPluginRegistry._registry["versioned_reward"] is PluginV2

    def test_multiple_plugins_registered_independently(self, clean_registry: None) -> None:
        @RewardPluginRegistry.register
        class PluginA(RewardPlugin):
            name = "plugin_a_reward"

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {}

        @RewardPluginRegistry.register
        class PluginB(RewardPlugin):
            name = "plugin_b_reward"

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {}

        assert "plugin_a_reward" in RewardPluginRegistry._registry
        assert "plugin_b_reward" in RewardPluginRegistry._registry


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


class TestRewardPluginRegistryCreate:
    def test_create_returns_plugin_instance(self, clean_registry: None) -> None:
        @RewardPluginRegistry.register
        class SimplePlugin(RewardPlugin):
            name = "simple_reward"

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {}

        instance = RewardPluginRegistry.create("simple_reward", {})
        assert isinstance(instance, SimplePlugin)

    def test_create_passes_params_to_plugin(self, clean_registry: None) -> None:
        @RewardPluginRegistry.register
        class ParamPlugin(RewardPlugin):
            name = "param_reward"

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {}

        params = {"timeout": 30, "mode": "strict"}
        instance = RewardPluginRegistry.create("param_reward", params)
        assert instance.params == params

    def test_create_unknown_name_raises_key_error(self, clean_registry: None) -> None:
        with pytest.raises(KeyError, match="not registered"):
            RewardPluginRegistry.create("nonexistent_plugin", {})

    def test_key_error_message_lists_available_plugins(self, clean_registry: None) -> None:
        @RewardPluginRegistry.register
        class KnownPlugin(RewardPlugin):
            name = "known_reward"

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {}

        with pytest.raises(KeyError) as exc_info:
            RewardPluginRegistry.create("unknown_reward", {})
        assert "known_reward" in str(exc_info.value)


# ---------------------------------------------------------------------------
# RewardPlugin ABC
# ---------------------------------------------------------------------------


class TestRewardPluginABC:
    def test_cannot_instantiate_abstract_base_directly(self) -> None:
        with pytest.raises(TypeError):
            RewardPlugin({})  # type: ignore[abstract]

    def test_validate_params_called_during_init(self, clean_registry: None) -> None:
        called = []

        class ValidatedPlugin(RewardPlugin):
            name = "validated_reward"

            def _validate_params(self) -> None:
                called.append(True)

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {}

        ValidatedPlugin({"key": "val"})
        assert called == [True]

    def test_validate_params_can_raise_value_error(self, clean_registry: None) -> None:
        class StrictPlugin(RewardPlugin):
            name = "strict_reward"

            def _validate_params(self) -> None:
                if "required" not in self.params:
                    raise ValueError("'required' param is missing")

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {}

        with pytest.raises(ValueError, match="required"):
            StrictPlugin({})

    def test_params_stored_on_instance(self, clean_registry: None) -> None:
        class StoredPlugin(RewardPlugin):
            name = "stored_reward"

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {}

        params = {"a": 1, "b": "x"}
        instance = StoredPlugin(params)
        assert instance.params == params

    def test_build_trainer_kwargs_is_abstract(self) -> None:
        import inspect

        assert inspect.isabstract(RewardPlugin)
        abstract_methods = RewardPlugin.__abstractmethods__
        assert "build_trainer_kwargs" in abstract_methods

    def test_plugin_inherits_base_plugin_classvars(self, clean_registry: None) -> None:
        class FullPlugin(RewardPlugin):
            name = "full_reward"
            priority = 10
            version = "3.0.0"

            def build_trainer_kwargs(
                self, *, train_dataset: Any, phase_config: Any, pipeline_config: Any
            ) -> dict[str, Any]:
                return {}

        assert FullPlugin.name == "full_reward"
        assert FullPlugin.priority == 10
        assert FullPlugin.version == "3.0.0"

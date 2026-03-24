"""
Tests for src.utils.plugin_base.BasePlugin mixin.

Coverage:
- Default ClassVar values
- Subclass override
- MRO safety with ABC
- Cooperative use with other ABCs (ValidationPlugin, RewardPlugin)
- Invariants: name/priority/version are ClassVar not instance vars
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pytest

from src.utils.plugin_base import BasePlugin

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestBasePluginDefaults:
    def test_name_default_is_empty_string(self) -> None:
        assert BasePlugin.name == ""

    def test_priority_default_is_50(self) -> None:
        assert BasePlugin.priority == 50  # noqa: WPS432

    def test_version_default_is_1_0_0(self) -> None:
        assert BasePlugin.version == "1.0.0"

    def test_name_is_classvar_not_instance_attr(self) -> None:
        # ClassVar should be accessible on the class itself
        class ConcretePlugin(BasePlugin):
            name: ClassVar[str] = "test"

        assert ConcretePlugin.name == "test"
        instance = ConcretePlugin()
        assert instance.name == "test"

    def test_no_init_required(self) -> None:
        # BasePlugin should be safely instantiable with no args
        plugin = BasePlugin()
        assert plugin is not None


# ---------------------------------------------------------------------------
# Subclass overrides
# ---------------------------------------------------------------------------


class TestBasePluginSubclass:
    def test_subclass_can_override_name(self) -> None:
        class MyPlugin(BasePlugin):
            name: ClassVar[str] = "my_plugin"

        assert MyPlugin.name == "my_plugin"

    def test_subclass_can_override_priority(self) -> None:
        class HighPriorityPlugin(BasePlugin):
            priority: ClassVar[int] = 10

        assert HighPriorityPlugin.priority == 10

    def test_subclass_can_override_version(self) -> None:
        class VersionedPlugin(BasePlugin):
            version: ClassVar[str] = "2.5.0"

        assert VersionedPlugin.version == "2.5.0"

    def test_subclass_inherits_defaults_if_not_overridden(self) -> None:
        class MinimalPlugin(BasePlugin):
            name: ClassVar[str] = "minimal"

        assert MinimalPlugin.priority == 50  # noqa: WPS432
        assert MinimalPlugin.version == "1.0.0"

    def test_different_subclasses_have_independent_classvars(self) -> None:
        class PluginA(BasePlugin):
            name: ClassVar[str] = "plugin_a"
            priority: ClassVar[int] = 10

        class PluginB(BasePlugin):
            name: ClassVar[str] = "plugin_b"
            priority: ClassVar[int] = 90

        assert PluginA.name != PluginB.name
        assert PluginA.priority != PluginB.priority


# ---------------------------------------------------------------------------
# MRO safety with ABC
# ---------------------------------------------------------------------------


class TestBasePluginMRO:
    def test_compatible_with_abc_mixin(self) -> None:
        class ConcreteABC(BasePlugin, ABC):
            name: ClassVar[str] = "concrete_abc"

            @abstractmethod
            def run(self) -> None: ...

        class ConcreteImpl(ConcreteABC):
            def run(self) -> None:
                pass

        instance = ConcreteImpl()
        assert instance.name == "concrete_abc"

    def test_does_not_introduce_abstractmethods(self) -> None:
        # BasePlugin should not require any method implementations
        class Simple(BasePlugin):
            name: ClassVar[str] = "simple"

        # Should not raise
        instance = Simple()
        assert instance is not None

    def test_no_mro_conflict_with_multiple_bases(self) -> None:
        class Mixin:
            def extra(self) -> str:
                return "extra"

        class CombinedPlugin(BasePlugin, Mixin):
            name: ClassVar[str] = "combined"

        instance = CombinedPlugin()
        assert instance.name == "combined"
        assert instance.extra() == "extra"


# ---------------------------------------------------------------------------
# Integration: BasePlugin used by real plugin ABCs
# ---------------------------------------------------------------------------


class TestBasePluginInRealABCs:
    def test_validation_plugin_abc_uses_base_plugin(self) -> None:
        from src.data.validation.base import ValidationPlugin

        assert issubclass(ValidationPlugin, BasePlugin)

    def test_evaluator_plugin_abc_uses_base_plugin(self) -> None:
        from src.evaluation.plugins.base import EvaluatorPlugin

        assert issubclass(EvaluatorPlugin, BasePlugin)

    def test_reward_plugin_abc_uses_base_plugin(self) -> None:
        from src.training.reward_plugins.base import RewardPlugin

        assert issubclass(RewardPlugin, BasePlugin)

    def test_validation_plugin_inherits_default_version(self) -> None:
        from src.data.validation.base import ValidationPlugin

        # Concrete plugins should define their own name; base class version is inherited
        assert hasattr(ValidationPlugin, "version")


# ---------------------------------------------------------------------------
# Boundary: priority edge cases
# ---------------------------------------------------------------------------


class TestBasePluginPriorityBoundaries:
    @pytest.mark.parametrize("priority", [0, 1, 50, 99, 100])  # noqa: WPS432
    def test_valid_priority_values(self, priority: int) -> None:
        class P(BasePlugin):
            name: ClassVar[str] = "p"

        P.priority = priority  # type: ignore[assignment]
        assert P.priority == priority

    def test_base_default_priority_is_midpoint(self) -> None:
        # 50 is semantically the "neutral" middle priority
        assert BasePlugin.priority == 50  # noqa: WPS432

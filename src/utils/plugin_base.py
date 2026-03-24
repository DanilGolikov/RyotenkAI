"""
BasePlugin — lightweight mixin for all plugin systems in this project.

Provides a shared set of ClassVar metadata fields that every plugin type
(validation, evaluation, reward, report) should carry.

Usage:
    class MyPluginBase(BasePlugin, ABC):
        @abstractmethod
        def run(self) -> Result: ...

    class ConcretePlugin(MyPluginBase):
        name = "my_concrete_plugin"
        priority = 10
        version = "1.2.0"
"""

from __future__ import annotations

from typing import ClassVar


class BasePlugin:
    """
    Mixin that enforces a common metadata contract across all plugin ABCs.

    All plugin systems in this project share three invariants:
      - ``name``     — unique string key used by registries for lookup.
      - ``priority`` — execution order hint (lower = runs earlier).
      - ``version``  — semver string, useful for compatibility checks.

    This class intentionally carries NO abstract methods and NO ``__init__``
    so it can be inserted into any ABC/Protocol hierarchy without MRO conflicts.

    Convention:
        Every concrete plugin class MUST set a non-empty ``name``.
        Registries SHOULD assert ``plugin_cls.name`` is non-empty on registration.
    """

    name: ClassVar[str] = ""
    priority: ClassVar[int] = 50
    version: ClassVar[str] = "1.0.0"


__all__ = ["BasePlugin"]

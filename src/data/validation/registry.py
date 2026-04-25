"""Registry for validation plugins loaded from the community catalogue.

Thin subclass over :class:`PluginRegistry` — the shared base owns lookup,
secret injection, error formatting, and book-keeping. The only kind-
specific bit is :meth:`_make_init_kwargs`, which adapts the uniform
``instantiate(plugin_id, params=…, thresholds=…)`` call into the
``ValidationPlugin.__init__(params, thresholds)`` signature.

Module-level singleton :data:`validation_registry` is what the rest of
the codebase imports — direct class access is reserved for tests that
need fresh state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.community.registry_base import PluginRegistry

if TYPE_CHECKING:
    from src.data.validation.base import ValidationPlugin


class ValidationPluginRegistry(PluginRegistry["ValidationPlugin"]):
    """Validation-kind registry. Plugin ctor expects ``(params, thresholds)``."""

    _kind: ClassVar[str] = "validation"

    def _make_init_kwargs(self, init_kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "params": dict(init_kwargs.get("params") or {}),
            "thresholds": dict(init_kwargs.get("thresholds") or {}),
        }


#: The singleton other modules import. Tests can construct fresh instances
#: but production wiring (catalog → registry → call sites) goes through
#: this one so secret/env injection state stays consistent.
validation_registry = ValidationPluginRegistry()


__all__ = ["ValidationPluginRegistry", "validation_registry"]

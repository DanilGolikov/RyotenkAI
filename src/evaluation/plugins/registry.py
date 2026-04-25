"""Registry for evaluation plugins loaded from the community catalogue.

Thin subclass over :class:`PluginRegistry`. The plugin constructor takes
``(params, thresholds)`` (same shape as validation), so the kwargs adapter
is identical — but the kind label differs for log/error messages, and
the eval-side resolver uses the ``EVAL_*`` namespace rather than ``DTST_*``.

Module-level singleton :data:`evaluator_registry` is what the rest of
the codebase imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from src.community.registry_base import PluginRegistry

if TYPE_CHECKING:
    from src.evaluation.plugins.base import EvaluatorPlugin


class EvaluatorPluginRegistry(PluginRegistry["EvaluatorPlugin"]):
    """Evaluation-kind registry. Plugin ctor expects ``(params, thresholds)``."""

    _kind: ClassVar[str] = "evaluator"

    def _make_init_kwargs(self, init_kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "params": dict(init_kwargs.get("params") or {}),
            "thresholds": dict(init_kwargs.get("thresholds") or {}),
        }


evaluator_registry = EvaluatorPluginRegistry()


__all__ = ["EvaluatorPluginRegistry", "evaluator_registry"]

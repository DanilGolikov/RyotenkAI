"""Plugin catalogue service.

Aggregates manifests from the three plugin registries (reward, validation,
evaluation) and returns them in UI-friendly shape. Triggers lazy discovery
so freshly started API processes see every registered plugin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.schemas.plugin import PluginKind, PluginListResponse, PluginManifest

if TYPE_CHECKING:
    pass


def _reward_manifests() -> list[dict]:
    from src.training.reward_plugins.discovery import ensure_reward_plugins_discovered
    from src.training.reward_plugins.registry import RewardPluginRegistry

    ensure_reward_plugins_discovered()
    return RewardPluginRegistry.list_manifests()


def _validation_manifests() -> list[dict]:
    from src.data.validation.discovery import ensure_validation_plugins_discovered
    from src.data.validation.registry import ValidationPluginRegistry

    ensure_validation_plugins_discovered()
    return ValidationPluginRegistry.list_manifests()


def _evaluation_manifests() -> list[dict]:
    from src.evaluation.plugins.discovery import ensure_evaluation_plugins_discovered
    from src.evaluation.plugins.registry import EvaluatorPluginRegistry

    ensure_evaluation_plugins_discovered()
    return EvaluatorPluginRegistry.list_manifests()


_COLLECTORS = {
    "reward": _reward_manifests,
    "validation": _validation_manifests,
    "evaluation": _evaluation_manifests,
}


def list_plugins(kind: PluginKind) -> PluginListResponse:
    if kind not in _COLLECTORS:
        raise ValueError(f"unknown plugin kind: {kind!r}")
    raw = _COLLECTORS[kind]()
    plugins = [PluginManifest(**item) for item in raw]
    plugins.sort(key=lambda m: (m.category or "~", m.id))
    return PluginListResponse(kind=kind, plugins=plugins)


__all__ = ["list_plugins"]

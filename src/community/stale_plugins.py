"""Detect plugin references in a pipeline config that no longer match
any registered community plugin.

The Configure modal already prevents users from picking unknown plugins
in the catalog dropdown, but a YAML edit, a manifest deletion, or a
catalog reload can leave behind references whose target id has gone
away. This module walks the same per-kind enumerators the preflight
gate uses (see :mod:`src.community.preflight`) and surfaces any
config entry whose ``plugin`` field is not in the registered set.

Surfaced in two places:

- ``GET /api/v1/projects/{id}/config`` — embedded in the response so
  the UI can render a "Remove from config" button per stale row.
- :func:`find_stale_plugins` directly — for callers (CLI, tests) that
  want the list without going through the API layer.

The check is *purely* read-only: we never modify the config or the
catalog. Removal is the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.community.manifest import PluginKind
    from src.utils.config import PipelineConfig


@dataclass(frozen=True, slots=True)
class StalePluginRef:
    """One config entry referencing a plugin id absent from the catalog.

    ``instance_id`` is the YAML-level id (the ``id`` field on the
    config row, or — for reports — the plugin id itself, which doubles
    as instance id since ``reports.sections`` lists plugin ids
    directly). ``location`` is a dotted path the UI can use to
    deep-link the user into the offending row.
    """

    plugin_kind: PluginKind
    plugin_name: str
    instance_id: str
    location: str


def find_stale_plugins(config: PipelineConfig) -> list[StalePluginRef]:
    """Return every plugin reference in ``config`` whose target id is
    not currently registered in the community catalog.

    Empty list ⇒ every reference resolves cleanly. Non-empty ⇒ caller
    decides whether to surface (UI banner) or block (e.g. preflight,
    though the preflight gate has its own "unknown plugin" handling
    that's already silent for the same triple — keep this strictly
    informational).
    """
    from src.community.catalog import catalog

    catalog.ensure_loaded()

    # Reuse the preflight enumerator but only consume the identity
    # triple — params/thresholds aren't relevant for staleness. Keeping
    # the enumerators in one place means stale-detection automatically
    # picks up any new plugin kinds added later.
    from src.community.preflight import (
        _enabled_evaluation_plugins,
        _enabled_report_plugins,
        _enabled_reward_plugins,
        _enabled_validation_plugins,
    )

    stale: list[StalePluginRef] = []

    for ref in _enabled_validation_plugins(config):
        if not _is_registered(ref.kind, ref.plugin_name):
            ds_id, _, instance = ref.instance_id.partition(".")
            stale.append(StalePluginRef(
                plugin_kind=ref.kind,
                plugin_name=ref.plugin_name,
                instance_id=instance or ref.instance_id,
                location=f"datasets.{ds_id}.validations.plugins[{instance}]",
            ))

    for ref in _enabled_evaluation_plugins(config):
        if not _is_registered(ref.kind, ref.plugin_name):
            stale.append(StalePluginRef(
                plugin_kind=ref.kind,
                plugin_name=ref.plugin_name,
                instance_id=ref.instance_id,
                location=f"evaluation.evaluators.plugins[{ref.instance_id}]",
            ))

    for ref in _enabled_reward_plugins(config):
        if not _is_registered(ref.kind, ref.plugin_name):
            stale.append(StalePluginRef(
                plugin_kind=ref.kind,
                plugin_name=ref.plugin_name,
                instance_id=ref.instance_id,
                location=f"training.strategies[{ref.instance_id}].params.reward_plugin",
            ))

    for ref in _enabled_report_plugins(config):
        if not _is_registered(ref.kind, ref.plugin_name):
            stale.append(StalePluginRef(
                plugin_kind=ref.kind,
                plugin_name=ref.plugin_name,
                instance_id=ref.instance_id,
                location=f"reports.sections[{ref.plugin_name}]",
            ))

    return stale


def _is_registered(kind: PluginKind, plugin_name: str) -> bool:
    """Per-kind registration check.

    Each kind has its own registry singleton. Walking through
    ``catalog.get(kind, plugin_name)`` would also work but raises
    KeyError on the unknown path — the per-kind ``is_registered``
    method is the cleaner shape for boolean tests.
    """
    if kind == "validation":
        from src.data.validation.registry import validation_registry

        return validation_registry.is_registered(plugin_name)
    if kind == "evaluation":
        from src.evaluation.plugins.registry import evaluator_registry

        return evaluator_registry.is_registered(plugin_name)
    if kind == "reward":
        from src.training.reward_plugins.registry import reward_registry

        return reward_registry.is_registered(plugin_name)
    if kind == "reports":
        from src.reports.plugins.registry import report_registry

        return report_registry.is_registered(plugin_name)
    return False


__all__ = ["StalePluginRef", "find_stale_plugins"]

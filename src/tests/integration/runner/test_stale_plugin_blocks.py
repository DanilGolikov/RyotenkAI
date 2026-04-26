"""Integration: stale-plugin reference blocks the submit chain.

A "stale" plugin is one referenced by ``config.yaml`` (`reward.plugin`,
`validation.plugins[*].id`, etc.) but absent from the on-disk
``community/`` catalog. Submitting against a stale ref would have
the trainer fail at runtime with a missing-import — surfacing it
*before* the SSH tunnel opens is a much better UX.

:func:`src.community.stale_plugins.find_stale_plugins` walks the
config and returns one :class:`StalePluginRef` per orphan reference.
The pipeline bootstrap calls it alongside preflight; non-empty list
→ launch refused.

The 8 unit tests in :mod:`src.tests.unit.community.test_stale_plugins`
exercise the walker over varied configs. This module pins the
contract the rest of the launch chain depends on: the *shape* of
the row, the inheritance of the gate's decision, and the fact that
``find_stale_plugins`` returns an empty list for a clean config —
the equivalence of "everything is fine, proceed".
"""

from __future__ import annotations

from src.community.stale_plugins import StalePluginRef


def test_stale_plugin_ref_carries_kind_and_id() -> None:
    """Pin the row shape: plugin_kind + plugin_name + instance_id + dotted location.

    The Web UI renders this as the per-row "Remove from config" button;
    the CLI surfaces it in the launch failure message. A future
    refactor that drops one of these fields silently breaks both
    surfaces.
    """
    ref = StalePluginRef(
        plugin_kind="reward",
        plugin_name="ghost_reward",
        instance_id="ghost_reward",
        location="strategies[0].params.reward_plugin",
    )
    assert ref.plugin_kind == "reward"
    assert ref.plugin_name == "ghost_reward"
    assert ref.instance_id == "ghost_reward"
    assert ref.location == "strategies[0].params.reward_plugin"


def test_stale_ref_is_hashable_for_dedupe() -> None:
    """The bootstrap dedupes via a set when a config references the
    same stale plugin from multiple locations. Pin hashability so
    the dedupe path keeps working."""
    ref = StalePluginRef(
        plugin_kind="validation",
        plugin_name="missing",
        instance_id="missing-1",
        location="datasets.train.validations.plugins[0].id",
    )
    {ref}  # must not raise
    assert hash(ref) == hash(ref)


def test_stale_refs_compare_by_value() -> None:
    """Two refs with the same fields are equal — supports set-based
    dedupe across the same orphan referenced more than once."""
    a = StalePluginRef(
        plugin_kind="reward", plugin_name="x", instance_id="x", location="a",
    )
    b = StalePluginRef(
        plugin_kind="reward", plugin_name="x", instance_id="x", location="a",
    )
    assert a == b

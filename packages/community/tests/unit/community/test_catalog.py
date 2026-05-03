"""Unit tests for CommunityCatalog."""

from __future__ import annotations

import textwrap
from pathlib import Path

from src.community.catalog import CommunityCatalog


def _make_plugin(root: Path, kind: str, plugin_id: str, class_name: str) -> None:
    plugin_dir = root / kind / plugin_id
    plugin_dir.mkdir(parents=True)
    # Reward plugins must declare ``supported_strategies`` — the schema
    # rejects empty for kind="reward" (see PluginSpec._check_supported_strategies).
    extra_spec = ""
    if kind == "reward":
        extra_spec = '\nsupported_strategies = ["grpo", "sapo"]'
    (plugin_dir / "manifest.toml").write_text(
        textwrap.dedent(f"""
            [plugin]
            id = "{plugin_id}"
            kind = "{kind}"{extra_spec}

            [plugin.entry_point]
            module = "plugin"
            class = "{class_name}"
        """).strip()
    )
    (plugin_dir / "plugin.py").write_text(f"class {class_name}:\n    pass\n")


def test_catalog_ensure_loaded_is_lazy_and_idempotent(tmp_path: Path) -> None:
    _make_plugin(tmp_path, "validation", "alpha", "Alpha")
    catalog = CommunityCatalog(root=tmp_path)
    catalog.ensure_loaded()
    first = catalog.plugins("validation")
    catalog.ensure_loaded()  # second call is a no-op
    second = catalog.plugins("validation")
    assert [entry.manifest.plugin.id for entry in first] == ["alpha"]
    assert [entry.manifest.plugin.id for entry in second] == ["alpha"]


def test_catalog_reload_replaces_contents(tmp_path: Path) -> None:
    _make_plugin(tmp_path, "reward", "alpha", "Alpha")
    catalog = CommunityCatalog(root=tmp_path)
    catalog.ensure_loaded()
    assert [e.manifest.plugin.id for e in catalog.plugins("reward")] == ["alpha"]

    _make_plugin(tmp_path, "reward", "beta", "Beta")
    catalog.reload()
    ids = sorted(e.manifest.plugin.id for e in catalog.plugins("reward"))
    assert ids == ["alpha", "beta"]


def test_catalog_get_unknown_id_raises(tmp_path: Path) -> None:
    catalog = CommunityCatalog(root=tmp_path)
    catalog.ensure_loaded()
    try:
        catalog.get("evaluation", "missing")
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_catalog_auto_reloads_when_new_plugin_appears(tmp_path: Path) -> None:
    """``ensure_loaded`` re-scans when a new plugin folder is added
    without requiring a manual ``reload()`` — this is what powers the
    web UI "add a preset → show up on refresh" flow."""
    _make_plugin(tmp_path, "validation", "alpha", "Alpha")
    catalog = CommunityCatalog(root=tmp_path)
    catalog.ensure_loaded()
    assert [e.manifest.plugin.id for e in catalog.plugins("validation")] == ["alpha"]

    # Simulate a user dropping a new plugin into community/validation/
    _make_plugin(tmp_path, "validation", "beta", "Beta")

    # No explicit reload() — just another call, which must pick up the change.
    catalog.ensure_loaded()
    ids = sorted(e.manifest.plugin.id for e in catalog.plugins("validation"))
    assert ids == ["alpha", "beta"]


def test_catalog_auto_reloads_when_manifest_edited(tmp_path: Path) -> None:
    """Editing an existing manifest updates its mtime and must also
    trigger a reload — otherwise the web UI wouldn't pick up renamed
    fields, changed descriptions, etc."""
    import time

    _make_plugin(tmp_path, "evaluation", "dummy", "DummyPlugin")
    catalog = CommunityCatalog(root=tmp_path)
    catalog.ensure_loaded()
    first = catalog.plugins("evaluation")
    assert first[0].manifest.plugin.description == ""

    # Re-write manifest with a new description field. Sleep briefly so
    # the mtime ticks on filesystems with low-resolution mtimes.
    time.sleep(0.01)
    (tmp_path / "evaluation" / "dummy" / "manifest.toml").write_text(
        textwrap.dedent("""
            [plugin]
            id = "dummy"
            kind = "evaluation"
            description = "edited"

            [plugin.entry_point]
            module = "plugin"
            class = "DummyPlugin"
        """).strip()
    )

    catalog.ensure_loaded()
    second = catalog.plugins("evaluation")
    assert second[0].manifest.plugin.description == "edited"


def test_catalog_no_reload_when_tree_unchanged(tmp_path: Path) -> None:
    """Cache-hit path: repeated ``ensure_loaded`` on an unchanged tree
    must return the same LoadedPlugin instances (no reload, no rebuild)."""
    _make_plugin(tmp_path, "reward", "alpha", "Alpha")
    catalog = CommunityCatalog(root=tmp_path)
    catalog.ensure_loaded()
    first = catalog.plugins("reward")
    catalog.ensure_loaded()
    second = catalog.plugins("reward")
    # Same class objects — reload would have re-imported and produced new ones.
    assert first[0].plugin_cls is second[0].plugin_cls

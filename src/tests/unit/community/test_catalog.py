"""Unit tests for CommunityCatalog."""

from __future__ import annotations

import textwrap
from pathlib import Path

from src.community.catalog import CommunityCatalog


def _make_plugin(root: Path, kind: str, plugin_id: str, class_name: str) -> None:
    plugin_dir = root / kind / plugin_id
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "manifest.toml").write_text(
        textwrap.dedent(f"""
            [plugin]
            id = "{plugin_id}"
            kind = "{kind}"

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

"""Tests for the loose / strict load modes (A4 in cozy-booping-walrus).

Default loose: broken plugins are captured as :class:`LoadFailure`
records and skipped, so the rest of the catalog keeps loading.

Strict (``COMMUNITY_STRICT=1`` env or ``strict=True`` kwarg): the
original exception re-raises, turning silent loader bugs into hard
test failures during dev / CI.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.community.loader import load_plugins


def _write_broken_manifest(tmp_root: Path) -> None:
    """Drop a plugin folder whose manifest is missing required fields."""
    plugin_dir = tmp_root / "validation" / "broken"
    plugin_dir.mkdir(parents=True)
    # No ``[plugin.entry_point]`` block — model_validate raises.
    (plugin_dir / "manifest.toml").write_text(
        textwrap.dedent("""
            [plugin]
            id = "broken"
            kind = "validation"
        """).strip()
    )
    (plugin_dir / "plugin.py").write_text("class Whatever:\n    pass\n")


def _write_broken_import(tmp_root: Path) -> None:
    """Drop a plugin folder whose plugin.py raises at import."""
    plugin_dir = tmp_root / "evaluation" / "explodes"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "manifest.toml").write_text(
        textwrap.dedent("""
            [plugin]
            id = "explodes"
            kind = "evaluation"

            [plugin.entry_point]
            module = "plugin"
            class = "ExplodingPlugin"
        """).strip()
    )
    (plugin_dir / "plugin.py").write_text("raise RuntimeError('boom at import')\n")


def test_loose_mode_captures_manifest_failures(tmp_path: Path) -> None:
    _write_broken_manifest(tmp_path)
    result = load_plugins("validation", root=tmp_path, strict=False)
    assert list(result) == []
    assert len(result.failures) == 1
    f = result.failures[0]
    assert f.error_type == "manifest_parse"
    assert f.entry_name == "broken"
    assert f.plugin_id is None  # manifest didn't parse
    assert f.traceback  # full traceback retained for UI drilldown


def test_loose_mode_captures_import_failures(tmp_path: Path) -> None:
    _write_broken_import(tmp_path)
    result = load_plugins("evaluation", root=tmp_path, strict=False)
    assert list(result) == []
    assert len(result.failures) == 1
    f = result.failures[0]
    assert f.error_type == "import_error"
    assert f.plugin_id == "explodes"
    assert "boom at import" in f.message


def test_strict_mode_reraises_manifest_error(tmp_path: Path) -> None:
    _write_broken_manifest(tmp_path)
    with pytest.raises(Exception, match=r"PluginManifest"):
        load_plugins("validation", root=tmp_path, strict=True)


def test_strict_mode_reraises_import_error(tmp_path: Path) -> None:
    _write_broken_import(tmp_path)
    with pytest.raises(RuntimeError, match="boom at import"):
        load_plugins("evaluation", root=tmp_path, strict=True)


def test_env_var_triggers_strict_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``COMMUNITY_STRICT=1`` switches the default to strict."""
    _write_broken_import(tmp_path)
    monkeypatch.setenv("COMMUNITY_STRICT", "1")
    with pytest.raises(RuntimeError, match="boom at import"):
        load_plugins("evaluation", root=tmp_path)


@pytest.mark.parametrize("value", ["0", "", "false", "no"])
def test_env_var_falsy_keeps_loose_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    """Falsy / unrecognised values for the env var are treated as loose."""
    _write_broken_import(tmp_path)
    monkeypatch.setenv("COMMUNITY_STRICT", value)
    result = load_plugins("evaluation", root=tmp_path)
    assert len(result.failures) == 1


def test_explicit_strict_overrides_env_var(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``strict=False`` from the caller wins even when the env says strict."""
    _write_broken_import(tmp_path)
    monkeypatch.setenv("COMMUNITY_STRICT", "1")
    result = load_plugins("evaluation", root=tmp_path, strict=False)
    assert len(result.failures) == 1


def test_catalog_surfaces_failures_to_api_service(
    tmp_path: Path,
) -> None:
    """The catalog stores failures so plugin_service.list_plugins can surface them."""
    _write_broken_import(tmp_path)
    from src.community.catalog import CommunityCatalog

    cat = CommunityCatalog(root=tmp_path)
    cat.reload()
    failures = cat.failures("evaluation")
    assert len(failures) == 1
    assert failures[0].error_type == "import_error"

"""Unit tests for the community loader (scan + import)."""

from __future__ import annotations

import textwrap
import zipfile
from pathlib import Path

import pytest

from src.community.loader import load_plugins, load_presets


def _build_plugin_folder(root: Path, kind: str, plugin_id: str, *, class_name: str) -> Path:
    plugin_dir = root / kind / plugin_id
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "manifest.toml").write_text(
        textwrap.dedent(f"""
            [plugin]
            id = "{plugin_id}"
            kind = "{kind}"
            version = "1.0.0"
            priority = 42

            [plugin.entry_point]
            module = "plugin"
            class = "{class_name}"

            [secrets]
            required = ["EVAL_DUMMY"]
        """).strip()
    )
    (plugin_dir / "plugin.py").write_text(
        textwrap.dedent(f"""
            class {class_name}:
                greeting = "hi"
        """).strip()
    )
    return plugin_dir


def test_load_plugins_imports_class_and_attaches_metadata(tmp_path: Path) -> None:
    _build_plugin_folder(tmp_path, "evaluation", "dummy", class_name="DummyPlugin")
    loaded = load_plugins("evaluation", root=tmp_path)
    assert len(loaded) == 1
    entry = loaded[0]
    assert entry.manifest.plugin.id == "dummy"
    cls = entry.plugin_cls
    assert cls.name == "dummy"
    assert cls.priority == 42
    assert cls._required_secrets == ("EVAL_DUMMY",)
    assert cls.greeting == "hi"


def test_load_plugins_duplicate_id_raises(tmp_path: Path) -> None:
    for folder_name in ("a", "b"):
        plugin_dir = tmp_path / "validation" / folder_name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "manifest.toml").write_text(
            textwrap.dedent("""
                [plugin]
                id = "same_id"
                kind = "validation"

                [plugin.entry_point]
                module = "plugin"
                class = "Dummy"
            """).strip()
        )
        (plugin_dir / "plugin.py").write_text("class Dummy:\n    pass\n")

    with pytest.raises(ValueError, match="duplicate plugin id"):
        load_plugins("validation", root=tmp_path)


def test_load_plugins_kind_mismatch_is_skipped(tmp_path: Path, caplog) -> None:
    plugin_dir = tmp_path / "validation" / "x"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "manifest.toml").write_text(
        textwrap.dedent("""
            [plugin]
            id = "x"
            kind = "evaluation"

            [plugin.entry_point]
            module = "plugin"
            class = "Dummy"
        """).strip()
    )
    (plugin_dir / "plugin.py").write_text("class Dummy:\n    pass\n")

    loaded = load_plugins("validation", root=tmp_path)
    assert loaded == []


def test_load_plugins_archive(tmp_path: Path) -> None:
    kind_dir = tmp_path / "reward"
    kind_dir.mkdir(parents=True)
    archive = kind_dir / "zipped.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "manifest.toml",
            textwrap.dedent("""
                [plugin]
                id = "zipped"
                kind = "reward"

                [plugin.entry_point]
                module = "plugin"
                class = "Zipped"
            """).strip(),
        )
        zf.writestr("plugin.py", "class Zipped:\n    origin = 'archive'\n")

    loaded = load_plugins("reward", root=tmp_path)
    assert len(loaded) == 1
    assert loaded[0].plugin_cls.origin == "archive"


def test_load_presets(tmp_path: Path) -> None:
    preset_dir = tmp_path / "presets" / "starter"
    preset_dir.mkdir(parents=True)
    (preset_dir / "manifest.toml").write_text(
        textwrap.dedent("""
            [preset]
            id = "starter"
            name = "Starter"
            description = "demo"
            size_tier = "small"

            [preset.entry_point]
            file = "preset.yaml"
        """).strip()
    )
    (preset_dir / "preset.yaml").write_text("model:\n  name: demo\n")

    loaded = load_presets(root=tmp_path)
    assert len(loaded) == 1
    assert loaded[0].manifest.preset.id == "starter"
    assert "demo" in loaded[0].yaml_text


def test_load_plugins_same_module_name_across_plugins(tmp_path: Path) -> None:
    """Two plugins with identical `plugin.py` must not collide in sys.modules."""
    _build_plugin_folder(tmp_path, "evaluation", "a_plugin", class_name="A")
    _build_plugin_folder(tmp_path, "evaluation", "b_plugin", class_name="B")
    loaded = load_plugins("evaluation", root=tmp_path)
    ids = {entry.manifest.plugin.id for entry in loaded}
    assert ids == {"a_plugin", "b_plugin"}


def test_folder_wins_over_coexisting_zip(tmp_path: Path, caplog) -> None:
    """When ``my_plugin/`` and ``my_plugin.zip`` sit side by side in the same
    kind directory, the folder is authoritative and the zip is skipped with
    a warning. This is the dev-time default — the folder is the source of
    truth while the zip is a stale pack artefact."""
    folder = _build_plugin_folder(tmp_path, "validation", "my_plugin", class_name="Impl")

    # Produce a *different* class inside the zip so the test can tell which one won.
    zip_path = tmp_path / "validation" / "my_plugin.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "my_plugin/manifest.toml",
            textwrap.dedent("""
                [plugin]
                id = "my_plugin"
                kind = "validation"

                [plugin.entry_point]
                module = "plugin"
                class = "ShouldNotWin"
            """).strip(),
        )
        zf.writestr("my_plugin/plugin.py", "class ShouldNotWin:\n    pass\n")

    with caplog.at_level("WARNING", logger="ryotenkai"):
        loaded = load_plugins("validation", root=tmp_path)

    # Only the folder's plugin loads, and the class comes from the folder.
    assert len(loaded) == 1
    assert loaded[0].plugin_cls.__name__ == "Impl"
    assert loaded[0].source_path == folder

    # The overlap is surfaced to the author, not silently hidden.
    assert any("shadows" in record.message for record in caplog.records)


def test_archive_only_plugin_still_loads(tmp_path: Path) -> None:
    """A stem with no matching folder still works from the zip alone."""
    zip_path = tmp_path / "reward" / "only_zip.zip"
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "only_zip/manifest.toml",
            textwrap.dedent("""
                [plugin]
                id = "only_zip"
                kind = "reward"

                [plugin.entry_point]
                module = "plugin"
                class = "ZippedOnly"
            """).strip(),
        )
        zf.writestr("only_zip/plugin.py", "class ZippedOnly:\n    pass\n")

    loaded = load_plugins("reward", root=tmp_path)
    assert [entry.manifest.plugin.id for entry in loaded] == ["only_zip"]

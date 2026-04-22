"""Unit tests for :mod:`src.community.pack`."""

from __future__ import annotations

import textwrap
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.community.pack import pack_community_folder

_VALID_PLUGIN_MANIFEST = textwrap.dedent('''
    [plugin]
    id = "my_plugin"
    kind = "validation"
    name = "MyValidator"
    version = "1.0.0"
    description = "ok"

    [plugin.entry_point]
    module = "plugin"
    class = "MyValidator"
''')

_VALID_PRESET_MANIFEST = textwrap.dedent('''
    [preset]
    id = "starter"
    name = "Starter"
    description = "."
    size_tier = "small"
    version = "0.1.0"

    [preset.entry_point]
    file = "preset.yaml"
''')


def _make_plugin_dir(tmp_path: Path, *, manifest: str = _VALID_PLUGIN_MANIFEST) -> Path:
    d = tmp_path / "my_plugin"
    d.mkdir()
    (d / "plugin.py").write_text("class MyValidator:\n    pass\n")
    (d / "manifest.toml").write_text(manifest)
    return d


def test_pack_creates_zip_next_to_folder(tmp_path: Path) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    result = pack_community_folder(plugin_dir)

    # archive lives alongside the source folder — same parent, `.zip` suffix
    assert result.archive_path == plugin_dir.parent / "my_plugin.zip"
    assert result.archive_path.is_file()
    assert zipfile.is_zipfile(result.archive_path)


def test_archive_contains_expected_files(tmp_path: Path) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    result = pack_community_folder(plugin_dir)

    with zipfile.ZipFile(result.archive_path) as zf:
        names = set(zf.namelist())
    assert names == {"my_plugin/manifest.toml", "my_plugin/plugin.py"}


def test_pack_excludes_caches_and_compiled(tmp_path: Path) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    (plugin_dir / "__pycache__").mkdir()
    (plugin_dir / "__pycache__" / "plugin.cpython-313.pyc").write_bytes(b"\x00")
    (plugin_dir / ".pytest_cache").mkdir()
    (plugin_dir / ".pytest_cache" / "CACHEDIR.TAG").write_text("x")
    (plugin_dir / ".DS_Store").write_text("x")

    result = pack_community_folder(plugin_dir)
    with zipfile.ZipFile(result.archive_path) as zf:
        names = set(zf.namelist())
    assert names == {"my_plugin/manifest.toml", "my_plugin/plugin.py"}


def test_pack_refuses_if_archive_exists(tmp_path: Path) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    (plugin_dir.parent / "my_plugin.zip").write_bytes(b"old")
    with pytest.raises(FileExistsError):
        pack_community_folder(plugin_dir)


def test_pack_force_overwrites(tmp_path: Path) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    archive = plugin_dir.parent / "my_plugin.zip"
    archive.write_bytes(b"stale")
    result = pack_community_folder(plugin_dir, force=True)
    assert result.archive_path == archive
    assert zipfile.is_zipfile(archive)


def test_pack_requires_manifest(tmp_path: Path) -> None:
    broken = tmp_path / "no_manifest"
    broken.mkdir()
    (broken / "plugin.py").write_text("x = 1\n")
    with pytest.raises(FileNotFoundError, match=r"no manifest\.toml"):
        pack_community_folder(broken)


def test_pack_validates_manifest(tmp_path: Path) -> None:
    """Invalid manifest blocks pack — we don't ship broken archives."""
    bad_manifest = textwrap.dedent('''
        [plugin]
        id = "x"
        kind = "invalid_kind"

        [plugin.entry_point]
        module = "plugin"
        class = "X"
    ''')
    plugin_dir = _make_plugin_dir(tmp_path, manifest=bad_manifest)
    with pytest.raises(ValidationError):
        pack_community_folder(plugin_dir)


def test_pack_dry_run_returns_file_list_without_writing(tmp_path: Path) -> None:
    plugin_dir = _make_plugin_dir(tmp_path)
    result = pack_community_folder(plugin_dir, dry_run=True)
    assert not result.archive_path.exists()
    assert "my_plugin/plugin.py" in result.files
    assert "my_plugin/manifest.toml" in result.files


def test_pack_preset(tmp_path: Path) -> None:
    """Pack works for presets too — the kind is auto-detected from the manifest."""
    preset_dir = tmp_path / "starter"
    preset_dir.mkdir()
    (preset_dir / "manifest.toml").write_text(_VALID_PRESET_MANIFEST)
    (preset_dir / "preset.yaml").write_text("model: {}\n")

    result = pack_community_folder(preset_dir)
    with zipfile.ZipFile(result.archive_path) as zf:
        assert set(zf.namelist()) == {"starter/manifest.toml", "starter/preset.yaml"}


def test_pack_includes_package_layout(tmp_path: Path) -> None:
    """Multi-file plugin (``plugin/`` package) is zipped with its siblings."""
    plugin_dir = tmp_path / "big_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "manifest.toml").write_text(
        _VALID_PLUGIN_MANIFEST.replace('id = "my_plugin"', 'id = "big_plugin"').replace(
            "MyValidator", "BigValidator"
        )
    )
    pkg = plugin_dir / "plugin"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("from .main import BigValidator\n")
    (pkg / "main.py").write_text("class BigValidator:\n    pass\n")
    (pkg / "util.py").write_text("HELPER = 1\n")

    result = pack_community_folder(plugin_dir)
    with zipfile.ZipFile(result.archive_path) as zf:
        names = set(zf.namelist())
    assert names == {
        "big_plugin/manifest.toml",
        "big_plugin/plugin/__init__.py",
        "big_plugin/plugin/main.py",
        "big_plugin/plugin/util.py",
    }


def test_pack_rejects_file_path(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("x")
    with pytest.raises(NotADirectoryError):
        pack_community_folder(file_path)

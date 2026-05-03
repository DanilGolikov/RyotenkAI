"""Phase 6.2 — :class:`PluginUnpacker` contract.

The unpacker is the receiving end of the
:class:`PluginPacker` ↔ runner round-trip. We assert:

- TestEmptyPayload         no plugins → no-op, dirs still exist
- TestHappyPath            single + multi plugin extraction
- TestForceOverwrite       repeated unpack replaces existing
- TestPathTraversal        ``../etc/passwd`` rejected
- TestSymlinkRejection     symlink entries rejected
- TestUnknownKindSkipped   loose / unknown-kind entries listed in skipped
- TestZipBomb              uncompressed cap enforced
- TestCorruptArchive       non-ZIP bytes raise PluginUnpackError
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from ryotenkai_pod.runner.plugin_unpacker import (
    PluginUnpackError,
    PluginUnpacker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zip(entries: dict[str, bytes]) -> bytes:
    """Build a ZIP from {arcname: bytes} pairs and return the bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _make_zip_with_symlink(workspace: Path) -> bytes:
    """Build a ZIP that declares a symlink entry — replicates what
    a malicious client could craft. We need to set the external_attr
    to mark it as a symlink (0xA1ED → 0o120755 in POSIX).
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("reward/evil/link")
        info.external_attr = (0o120755 << 16) | 0x80
        zf.writestr(info, "/etc/passwd")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Empty payload
# ---------------------------------------------------------------------------


class TestEmptyPayload:
    def test_empty_bytes_is_noop_with_dir_created(self, tmp_path: Path) -> None:
        unpacker = PluginUnpacker(tmp_path)
        result = unpacker.unpack(b"")
        assert result.installed == ()
        assert result.skipped == ()
        assert result.total_bytes == 0
        # The community root must exist even with no plugins, so the
        # trainer's catalog walk doesn't crash.
        assert (tmp_path / "community").is_dir()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_single_plugin_extracted(self, tmp_path: Path) -> None:
        zip_bytes = _make_zip({
            "reward/plugin_a/manifest.toml": b"[plugin]\nid='plugin_a'\nkind='reward'\n",
            "reward/plugin_a/plugin.py": b"# trainer code",
        })
        unpacker = PluginUnpacker(tmp_path)
        result = unpacker.unpack(zip_bytes)

        assert result.installed == ("reward/plugin_a",)
        plugin_dir = tmp_path / "community" / "reward" / "plugin_a"
        assert (plugin_dir / "manifest.toml").read_text().startswith("[plugin]")
        assert (plugin_dir / "plugin.py").read_bytes() == b"# trainer code"
        assert result.total_bytes == len(b"[plugin]\nid='plugin_a'\nkind='reward'\n") + len(b"# trainer code")

    def test_multiple_plugins_extracted(self, tmp_path: Path) -> None:
        zip_bytes = _make_zip({
            "reward/plugin_a/manifest.toml": b"a",
            "reward/plugin_a/plugin.py": b"a",
            "reward/plugin_b/manifest.toml": b"b",
            "reward/plugin_b/nested/helper.py": b"helper",
        })
        unpacker = PluginUnpacker(tmp_path)
        result = unpacker.unpack(zip_bytes)
        # Sorted by (kind, plugin_id).
        assert result.installed == ("reward/plugin_a", "reward/plugin_b")
        assert (tmp_path / "community" / "reward" / "plugin_b" / "nested" / "helper.py").read_text() == "helper"


# ---------------------------------------------------------------------------
# force=True overwrite
# ---------------------------------------------------------------------------


class TestForceOverwrite:
    def test_unpack_replaces_existing(self, tmp_path: Path) -> None:
        # Pre-populate target dir with stale content.
        plugin_dir = tmp_path / "community" / "reward" / "plugin_a"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "stale.txt").write_text("OLD")

        zip_bytes = _make_zip({
            "reward/plugin_a/manifest.toml": b"new",
            "reward/plugin_a/plugin.py": b"new",
        })
        unpacker = PluginUnpacker(tmp_path)
        unpacker.unpack(zip_bytes, force=True)

        # Stale file must be gone (force=True removes the dir first).
        assert not (plugin_dir / "stale.txt").exists()
        assert (plugin_dir / "manifest.toml").read_text() == "new"

    def test_force_false_blocks_overwrite(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "community" / "reward" / "plugin_a"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "manifest.toml").write_text("existing")

        zip_bytes = _make_zip({
            "reward/plugin_a/manifest.toml": b"new",
        })
        unpacker = PluginUnpacker(tmp_path)
        with pytest.raises(PluginUnpackError, match="already installed"):
            unpacker.unpack(zip_bytes, force=False)


# ---------------------------------------------------------------------------
# Defensive checks
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_dotdot_path_rejected(self, tmp_path: Path) -> None:
        zip_bytes = _make_zip({
            "reward/plugin_a/../../../etc/evil": b"pwned",
        })
        unpacker = PluginUnpacker(tmp_path)
        with pytest.raises(PluginUnpackError, match="traversal"):
            unpacker.unpack(zip_bytes)

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        zip_bytes = _make_zip({
            "/etc/passwd": b"pwned",
        })
        unpacker = PluginUnpacker(tmp_path)
        with pytest.raises(PluginUnpackError, match="traversal"):
            unpacker.unpack(zip_bytes)


class TestSymlinkRejection:
    def test_symlink_entry_rejected(self, tmp_path: Path) -> None:
        zip_bytes = _make_zip_with_symlink(tmp_path)
        unpacker = PluginUnpacker(tmp_path)
        with pytest.raises(PluginUnpackError, match="symlink"):
            unpacker.unpack(zip_bytes)


class TestUnknownKindSkipped:
    def test_unknown_kind_appears_in_skipped(self, tmp_path: Path) -> None:
        zip_bytes = _make_zip({
            "validation/foo/manifest.toml": b"x",  # kind not in RECOGNISED_KINDS
            "reward/plugin_a/manifest.toml": b"valid",
        })
        unpacker = PluginUnpacker(tmp_path)
        result = unpacker.unpack(zip_bytes)
        assert result.installed == ("reward/plugin_a",)
        assert "validation/foo/manifest.toml" in result.skipped

    def test_loose_root_file_skipped(self, tmp_path: Path) -> None:
        zip_bytes = _make_zip({
            "README.md": b"docs",  # not under <kind>/<id>/
            "reward/plugin_a/manifest.toml": b"valid",
        })
        unpacker = PluginUnpacker(tmp_path)
        result = unpacker.unpack(zip_bytes)
        assert result.installed == ("reward/plugin_a",)
        assert "README.md" in result.skipped


class TestZipBomb:
    def test_oversize_uncompressed_rejected(self, tmp_path: Path) -> None:
        # Set a tiny cap so the test stays fast — 100 bytes.
        zip_bytes = _make_zip({
            "reward/plugin_a/manifest.toml": b"x" * 50,
            "reward/plugin_a/big": b"y" * 200,  # pushes us over 100
        })
        unpacker = PluginUnpacker(tmp_path, max_uncompressed_bytes=100)
        with pytest.raises(PluginUnpackError, match="zip-bomb"):
            unpacker.unpack(zip_bytes)


class TestCorruptArchive:
    def test_non_zip_bytes_raises(self, tmp_path: Path) -> None:
        unpacker = PluginUnpacker(tmp_path)
        with pytest.raises(PluginUnpackError, match="valid ZIP"):
            unpacker.unpack(b"this is not a zip file")


class TestLibs:
    """The ``libs`` top-level dir must be recognised so reward plugins
    declaring ``[[lib_requirements]]`` find their imports at trainer
    startup."""

    def test_libs_extracted_to_community_libs_dir(self, tmp_path: Path) -> None:
        zip_bytes = _make_zip({
            "reward/plugin_a/manifest.toml": b"plugin",
            "reward/plugin_a/plugin.py": b"# stub\n",
            "libs/helixql/manifest.toml": b"lib",
            "libs/helixql/__init__.py": b'__version__ = "1.0.0"\n',
            "libs/helixql/extract.py": b"def extract(): ...\n",
        })
        unpacker = PluginUnpacker(tmp_path)
        result = unpacker.unpack(zip_bytes)

        assert "reward/plugin_a" in result.installed
        assert "libs/helixql" in result.installed
        # Lib body lands at <workspace>/community/libs/<id>/.
        community = tmp_path / "community"
        assert (community / "libs" / "helixql" / "manifest.toml").exists()
        assert (community / "libs" / "helixql" / "__init__.py").exists()
        assert (community / "libs" / "helixql" / "extract.py").exists()

    def test_libs_only_payload_extracts(self, tmp_path: Path) -> None:
        # Edge case: a payload with libs but no plugins. Realistically
        # the launcher always pairs them, but the unpacker shouldn't
        # care — it processes whatever recognised top-level dirs are
        # present.
        zip_bytes = _make_zip({
            "libs/helixql/manifest.toml": b"lib",
            "libs/helixql/__init__.py": b"\n",
        })
        unpacker = PluginUnpacker(tmp_path)
        result = unpacker.unpack(zip_bytes)
        assert result.installed == ("libs/helixql",)

    def test_lib_with_nested_subdir(self, tmp_path: Path) -> None:
        # Libs can have arbitrary subtrees (e.g. submodules); make
        # sure the routing doesn't truncate them.
        zip_bytes = _make_zip({
            "libs/helixql/__init__.py": b"\n",
            "libs/helixql/sub/__init__.py": b"\n",
            "libs/helixql/sub/util.py": b"def helper(): ...\n",
        })
        unpacker = PluginUnpacker(tmp_path)
        unpacker.unpack(zip_bytes)
        community = tmp_path / "community"
        assert (community / "libs" / "helixql" / "sub" / "util.py").exists()

    def test_libs_force_overwrite(self, tmp_path: Path) -> None:
        # Pin: force=True replaces an existing libs/<id>/ folder
        # (mirrors plugin behaviour). Single-tenant pod always wants
        # the freshest payload.
        community = tmp_path / "community"
        existing = community / "libs" / "helixql"
        existing.mkdir(parents=True)
        (existing / "stale.py").write_text("old content")

        zip_bytes = _make_zip({
            "libs/helixql/__init__.py": b"new\n",
        })
        unpacker = PluginUnpacker(tmp_path)
        unpacker.unpack(zip_bytes, force=True)

        assert not (existing / "stale.py").exists()
        assert (existing / "__init__.py").read_bytes() == b"new\n"

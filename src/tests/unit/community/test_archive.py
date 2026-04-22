"""Unit tests for community archive extraction."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.community.archive import ensure_extracted, resolve_extraction_root


def _make_zip(tmp: Path, name: str, files: dict[str, str]) -> Path:
    path = tmp / name
    with zipfile.ZipFile(path, "w") as zf:
        for rel, content in files.items():
            zf.writestr(rel, content)
    return path


def test_ensure_extracted_extracts_files(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    archive = _make_zip(tmp_path, "plugin.zip", {"manifest.toml": "x", "plugin.py": "y"})
    target = ensure_extracted(archive, cache_dir=cache)
    assert (target / "manifest.toml").read_text() == "x"
    assert (target / ".extracted").exists()


def test_ensure_extracted_is_idempotent(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    archive = _make_zip(tmp_path, "plugin.zip", {"manifest.toml": "x"})
    first = ensure_extracted(archive, cache_dir=cache)
    second = ensure_extracted(archive, cache_dir=cache)
    assert first == second


def test_ensure_extracted_rejects_non_zip(tmp_path: Path) -> None:
    broken = tmp_path / "not_a_zip.zip"
    broken.write_bytes(b"plain text")
    with pytest.raises(ValueError):
        ensure_extracted(broken, cache_dir=tmp_path / "cache")


def test_ensure_extracted_refreshes_on_content_change(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    archive = tmp_path / "plugin.zip"
    _make_zip(tmp_path, "plugin.zip", {"manifest.toml": "v1"})
    first = ensure_extracted(archive, cache_dir=cache)
    # Rewrite archive with different content — sha256 changes, new cache entry.
    archive.unlink()
    _make_zip(tmp_path, "plugin.zip", {"manifest.toml": "v2"})
    second = ensure_extracted(archive, cache_dir=cache)
    assert first != second
    assert (second / "manifest.toml").read_text() == "v2"


def test_resolve_extraction_root_descends_into_single_subdir(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    inner = outer / "nested"
    inner.mkdir(parents=True)
    (inner / "manifest.toml").write_text("x")
    assert resolve_extraction_root(outer) == inner


def test_resolve_extraction_root_keeps_flat_layout(tmp_path: Path) -> None:
    (tmp_path / "manifest.toml").write_text("x")
    (tmp_path / "plugin.py").write_text("y")
    assert resolve_extraction_root(tmp_path) == tmp_path

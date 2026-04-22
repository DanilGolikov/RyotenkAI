"""Pack a plugin or preset folder into a distributable zip.

The archive is placed **next to** the source folder (inside the kind
directory), so the community loader picks it up automatically if the
source folder is later removed. This matches the two equivalent forms
the loader already understands: ``community/<kind>/<id>/`` (folder) and
``community/<kind>/<id>.zip`` (archive).
"""

from __future__ import annotations

import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

from src.community.manifest import PluginManifest, PresetManifest

# Patterns inside the plugin folder that are excluded from the archive.
_EXCLUDED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
_EXCLUDED_SUFFIXES = (".pyc", ".pyo")
_EXCLUDED_NAMES = frozenset({".DS_Store", "Thumbs.db"})


@dataclass(frozen=True, slots=True)
class PackResult:
    archive_path: Path
    files: tuple[str, ...]     # archive-relative paths included in the zip
    total_bytes: int           # uncompressed total of included files


def _is_excluded(path: Path, *, root: Path) -> bool:
    rel = path.relative_to(root)
    for part in rel.parts:
        if part in _EXCLUDED_DIRS:
            return True
    if path.name in _EXCLUDED_NAMES:
        return True
    return any(path.name.endswith(suf) for suf in _EXCLUDED_SUFFIXES)


def _collect_files(source: Path) -> list[tuple[Path, str]]:
    """Return (absolute_path, archive_name) tuples for every file to include.

    ``archive_name`` is always prefixed with the source folder's name, so
    the zip unpacks to a single top-level directory — the loader's
    ``resolve_extraction_root`` happily descends into that.
    """
    out: list[tuple[Path, str]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if _is_excluded(path, root=source):
            continue
        rel = path.relative_to(source)
        # Keep the folder name as the archive root so the zip layout
        # mirrors the unpacked layout (easier for humans to inspect).
        archive_name = f"{source.name}/{rel.as_posix()}"
        out.append((path, archive_name))
    return out


def _validate_manifest(source: Path) -> None:
    manifest_path = source / "manifest.toml"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no manifest.toml in {source} — run `ryotenkai community scaffold` first"
        )
    doc = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    if "plugin" in doc:
        PluginManifest.model_validate(doc)
    elif "preset" in doc:
        PresetManifest.model_validate(doc)
    else:
        raise ValueError(
            f"{manifest_path} has neither [plugin] nor [preset] section — cannot pack"
        )


def pack_community_folder(
    source: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> PackResult:
    """Zip ``source`` into ``<source>.zip`` next to it.

    The manifest is validated first — a broken manifest blocks pack rather
    than shipping a broken archive. Build caches (``__pycache__`` etc.)
    are filtered out.
    """
    source = source.resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"source must be a directory: {source}")
    _validate_manifest(source)

    archive_path = source.parent / f"{source.name}.zip"
    if archive_path.exists() and not force:
        raise FileExistsError(
            f"{archive_path} already exists; pass --force to overwrite"
        )

    files = _collect_files(source)
    if not files:
        raise ValueError(f"no files to pack under {source}")

    total_bytes = sum(abs_path.stat().st_size for abs_path, _ in files)

    if dry_run:
        return PackResult(
            archive_path=archive_path,
            files=tuple(name for _, name in files),
            total_bytes=total_bytes,
        )

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for abs_path, archive_name in files:
            zf.write(abs_path, archive_name)

    return PackResult(
        archive_path=archive_path,
        files=tuple(name for _, name in files),
        total_bytes=total_bytes,
    )


__all__ = ["PackResult", "pack_community_folder"]

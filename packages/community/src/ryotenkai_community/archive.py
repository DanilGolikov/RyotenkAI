"""ZIP archive handling for community plugins and presets.

Each archive is extracted into ``community/.cache/<sha256>/`` once, keyed by
the content hash of the ZIP file. Subsequent loads reuse the extracted path;
when the archive content changes its sha256 changes, so a fresh cache entry
is created (old entries are pruned lazily).
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

from src.community.constants import CACHE_DIR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_extracted(archive_path: Path, *, cache_dir: Path = CACHE_DIR) -> Path:
    """Extract ``archive_path`` into cache and return the extraction root.

    The extraction is idempotent: if the expected ``<cache_dir>/<sha256>/``
    directory already exists and is non-empty, it is reused as-is. When a
    previous extraction exists for the same archive stem but with a different
    hash, the old directory is removed.
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"not a valid zip archive: {archive_path}")

    digest = _sha256(archive_path)
    target = cache_dir / digest
    marker = target / ".extracted"

    if marker.exists():
        return target

    # Clean up stale extractions for the same archive stem.
    stem = archive_path.stem
    if cache_dir.exists():
        for entry in cache_dir.iterdir():
            if not entry.is_dir() or entry.name == digest:
                continue
            stem_marker = entry / f".source-{stem}"
            if stem_marker.exists():
                shutil.rmtree(entry, ignore_errors=True)

    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target)

    (target / f".source-{stem}").touch()
    marker.touch()
    return target


def resolve_extraction_root(extracted_dir: Path) -> Path:
    """Some archives ship their files nested under a single top-level folder.

    If the cache directory contains exactly one sub-directory and no manifest
    at its root, descend into it.
    """
    entries = [
        entry
        for entry in extracted_dir.iterdir()
        if not entry.name.startswith(".")
    ]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted_dir


__all__ = ["ensure_extracted", "resolve_extraction_root"]

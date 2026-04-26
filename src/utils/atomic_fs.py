"""Shared filesystem helpers for pipeline settings stores.

Used by ``src/pipeline/project/`` and ``src/pipeline/settings/providers/``
to keep atomic-write + snapshot-per-save semantics in one place.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    """Second-precision UTC ISO-8601 string with trailing Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def snapshot_filename(ts: str | None = None) -> str:
    """Snapshot filename ``<YYYY-MM-DDTHH-MM-SSZ>.yaml`` — colon-free."""
    return (ts or utc_now_iso()).replace(":", "-") + ".yaml"


def created_at_from_filename(filename: str) -> str:
    """Recover an ISO-8601 timestamp from ``<YYYY-MM-DDTHH-MM-SSZ[-N]>.yaml``."""
    stem = Path(filename).stem
    if "Z-" in stem:
        stem = stem.split("Z-")[0] + "Z"
    if "T" in stem:
        date_part, _, time_part = stem.partition("T")
        time_iso = time_part.replace("-", ":", 2)
        return f"{date_part}T{time_iso}"
    return stem


def unique_snapshot_path(history_dir: Path) -> Path:
    """Return a snapshot path that does not yet exist in ``history_dir``.

    Resolves same-second collisions by appending ``-N`` before the extension.
    """
    base = snapshot_filename()
    candidate = history_dir / base
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    for n in range(1, 10_000):
        alt = history_dir / f"{stem}-{n}.yaml"
        if not alt.exists():
            return alt
    raise RuntimeError("exhausted snapshot filename attempts")


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file + atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, encoding="utf-8", newline="\n"
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    Path(tmp_name).replace(path)


def atomic_write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


__all__ = [
    "atomic_write_json",
    "atomic_write_text",
    "created_at_from_filename",
    "snapshot_filename",
    "unique_snapshot_path",
    "utc_now_iso",
]

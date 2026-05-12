"""Pytest plumbing for ``tests/golden/`` snapshot tests.

Configures syrupy to anchor snapshots at ``tests/golden/_snapshots/``
so they're committed alongside the test sources rather than scattered
under pytest's auto-discovered tree.

Provides a ``scrubbed`` fixture that normalises timestamps / IDs /
absolute paths in a dict before snapshotting — these are the three
common sources of non-determinism on real-system snapshots.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# WHY: syrupy.SnapshotAssertion lives at the package root in 4.x. Re-import
# here just so the override fixture below can name it explicitly.
from syrupy.assertion import SnapshotAssertion
from syrupy.extensions.amber import AmberSnapshotExtension


class _AnchoredAmber(AmberSnapshotExtension):
    """Amber serializer rooted at ``tests/golden/_snapshots/`` regardless of test path."""

    @classmethod
    def dirname(cls, *, test_location: Any) -> str:
        # WHY: amber default colocates ``__snapshots__`` next to each test;
        # we want a single ``_snapshots`` folder under tests/golden/ so the
        # tree is easy to review at PR time.
        return str(Path(test_location.filepath).resolve().parent / "_snapshots")


@pytest.fixture
def snapshot_anchored(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """``snapshot`` with the anchored amber extension."""
    return snapshot.use_extension(_AnchoredAmber)


_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
)
_ISO_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?")
_R_RUNID_RE = re.compile(r"r-\d{4,}")
_T_RUNID_RE = re.compile(r"t-\d{4,}")
_ABS_PATH_RE = re.compile(r"/(?:Users|home|tmp|private)/[^\s\"]+")


def _scrub_string(value: str) -> str:
    value = _UUID_RE.sub("<UUID>", value)
    value = _ISO_TS_RE.sub("<ISO_TS>", value)
    value = _R_RUNID_RE.sub("<RUN_ID>", value)
    value = _T_RUNID_RE.sub("<TRAINER_ID>", value)
    value = _ABS_PATH_RE.sub("<ABS_PATH>", value)
    return value


def _scrub_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return _scrub_string(obj)
    if isinstance(obj, dict):
        return {k: _scrub_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_scrub_obj(v) for v in obj)
    if isinstance(obj, float):
        # Quantise floats to 4 decimal places — clock-derived timestamps
        # vary in the last digit across platforms.
        if obj != obj or obj in (float("inf"), float("-inf")):  # NaN/inf
            return str(obj)
        return round(obj, 4)
    return obj


@pytest.fixture
def scrub() -> Callable[[Any], Any]:
    """Return a function that recursively scrubs non-deterministic fields."""
    return _scrub_obj

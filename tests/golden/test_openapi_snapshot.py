"""Snapshot test for OpenAPI spec hash.

Provides historical visibility of how the API evolves. Redundant with
the schemathesis-driven drift gate but useful at PR review time: a
diff against the snapshot tells a reviewer exactly which fields moved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.golden]


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC_PATH = _REPO_ROOT / "web" / "src" / "api" / "openapi.json"


def test_openapi_hash_snapshot(snapshot_anchored) -> None:  # type: ignore[no-untyped-def]
    """Snapshot a stable digest of the OpenAPI spec.

    We deliberately snapshot the *hash + top-level shape* rather than
    the entire spec — a full snapshot would be tens of thousands of
    lines and dominate code review.
    """
    spec_raw = _SPEC_PATH.read_text()
    spec = json.loads(spec_raw)

    summary = {
        "openapi": spec.get("openapi"),
        "version": spec.get("info", {}).get("version"),
        "title": spec.get("info", {}).get("title"),
        "paths": sorted(spec.get("paths", {}).keys()),
        "components_schemas": sorted(
            spec.get("components", {}).get("schemas", {}).keys(),
        ),
        "sha256_first16": hashlib.sha256(spec_raw.encode("utf-8")).hexdigest()[:16],
    }
    assert summary == snapshot_anchored

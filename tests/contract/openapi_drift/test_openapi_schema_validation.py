"""Static validation of the committed OpenAPI spec.

Catches malformed specs before any runtime fuzzing — a fast failure
mode (<100 ms) so PR review surfaces the issue immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC_PATH = _REPO_ROOT / "web" / "src" / "api" / "openapi.json"


def _load_spec() -> dict:
    return json.loads(_SPEC_PATH.read_text())


def test_spec_file_exists() -> None:
    assert _SPEC_PATH.is_file(), f"spec file missing: {_SPEC_PATH}"


def test_spec_is_valid_json() -> None:
    _load_spec()  # raises if not


def test_spec_is_openapi_3() -> None:
    spec = _load_spec()
    version = spec.get("openapi", "")
    assert version.startswith("3."), f"expected OpenAPI 3.x, got {version!r}"


def test_spec_has_required_top_level_keys() -> None:
    spec = _load_spec()
    for key in ("openapi", "info", "paths"):
        assert key in spec, f"missing required key {key!r}"


def test_spec_info_has_title_and_version() -> None:
    spec = _load_spec()
    info = spec["info"]
    assert info.get("title")
    assert info.get("version")


def test_spec_loads_via_schemathesis() -> None:
    """Schemathesis owns the deeper schema validation. If it can't
    parse the spec, the live fuzz lane will trip — surface here first
    so the failure is fast and self-contained."""
    import schemathesis

    schema = schemathesis.openapi.from_path(str(_SPEC_PATH))
    # Some operations should exist; an empty spec passes JSON parsing
    # but has zero contractual surface and shouldn't ship.
    assert schema.statistic.operations.total > 0

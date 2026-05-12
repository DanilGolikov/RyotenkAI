"""OpenAPI drift gate (no Stack — pure import-time check).

We re-build the OpenAPI spec from a freshly-constructed control app
via ``ryotenkai_control.api.main:create_app().openapi()`` and compare
its sha256 against the committed snapshot. Any drift fires with a
human-readable remediation hint.

Runs in <300 ms locally. Belongs in the PR-blocking lane.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.contract]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC_PATH = _REPO_ROOT / "web" / "src" / "api" / "openapi.json"
_REMEDIATION = (
    "OpenAPI drift detected. Run "
    "`make sync-openapi && make regen-zod && make regen-msw` "
    "(or `make regen-fe-contracts`) and commit the regenerated artifacts."
)


def _canonicalise(spec: dict) -> str:
    """Stable JSON serialisation that mirrors ``openapi_dump.py``.

    The dump CLI writes with ``ensure_ascii=False`` + ``indent=2`` +
    ``sort_keys=False`` and a single trailing newline. Reproducing
    those exact options is what makes drift detection meaningful —
    otherwise we'd false-positive on every Cyrillic / em-dash docstring.
    """
    return json.dumps(spec, indent=2, sort_keys=False, ensure_ascii=False) + "\n"


def _build_live_spec() -> dict:
    # We deliberately don't reuse the dump CLI; the gate must be
    # importable + self-contained.
    from ryotenkai_control.api.main import create_app

    app = create_app()
    return app.openapi()


def test_committed_spec_matches_live_app() -> None:
    """The on-disk ``openapi.json`` mirrors what ``create_app()`` emits."""
    live = _build_live_spec()
    committed = json.loads(_SPEC_PATH.read_text())

    if live == committed:
        return

    # When they don't match, present a useful diff hint without
    # dumping the entire spec into pytest output (CI logs are noisy
    # enough already).
    live_hash = hashlib.sha256(_canonicalise(live).encode()).hexdigest()[:16]
    on_disk_hash = hashlib.sha256(_canonicalise(committed).encode()).hexdigest()[:16]

    diff_summary: list[str] = []
    live_paths = set(live.get("paths", {}))
    on_disk_paths = set(committed.get("paths", {}))
    if live_paths != on_disk_paths:
        only_live = sorted(live_paths - on_disk_paths)
        only_disk = sorted(on_disk_paths - live_paths)
        if only_live:
            diff_summary.append(f"  paths only in live: {only_live[:5]}")
        if only_disk:
            diff_summary.append(f"  paths only in committed: {only_disk[:5]}")

    live_schemas = set(live.get("components", {}).get("schemas", {}))
    on_disk_schemas = set(committed.get("components", {}).get("schemas", {}))
    if live_schemas != on_disk_schemas:
        only_live = sorted(live_schemas - on_disk_schemas)
        only_disk = sorted(on_disk_schemas - live_schemas)
        if only_live:
            diff_summary.append(f"  schemas only in live: {only_live[:5]}")
        if only_disk:
            diff_summary.append(f"  schemas only in committed: {only_disk[:5]}")

    pytest.fail(
        "\n".join(
            [
                _REMEDIATION,
                f"live sha256[:16]: {live_hash}",
                f"committed sha256[:16]: {on_disk_hash}",
                *diff_summary,
            ],
        ),
    )


def test_committed_spec_is_canonical_dump() -> None:
    """Catch indentation drift — the dump must be stable so subsequent
    regenerations produce zero-diff output unless real schema changes."""
    raw = _SPEC_PATH.read_text()
    parsed = json.loads(raw)
    expected = _canonicalise(parsed)
    if raw == expected:
        return
    # Same data, different formatting — still drift; remediation is
    # the same.
    pytest.fail(
        f"openapi.json formatting drift — re-run `make sync-openapi`. "
        f"Hint: indent=2, no trailing whitespace, single trailing newline.",
    )

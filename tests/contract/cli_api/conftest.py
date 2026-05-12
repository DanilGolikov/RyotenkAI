"""Fixtures for CLI ↔ API parity tests.

Greenfield migration of
``packages/control/tests/contract/conftest.py``.

Wires:

- An in-process FastAPI ``TestClient`` using ``ApiSettings(runs_dir=tmp_path)``
  so neither side touches the real ``runs/`` directory.
- A ``CliRunner`` from ``typer.testing`` for invoking the CLI through
  the same root Typer the user runs.
- An autouse fixture that clears the process-local ``state_cache``
  before every test. Without this, the API's mtime-keyed cache can
  return a snapshot the CLI's raw read won't see — ``runs ls`` would
  diff against an outdated ``GET /runs`` payload.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from ryotenkai_control.api.config import ApiSettings
from ryotenkai_control.api.dependencies import get_settings
from ryotenkai_control.api.main import create_app
from ryotenkai_control.api.state_cache import clear_cache
from ryotenkai_control.cli.app import app as cli_app


@pytest.fixture
def _clean_state_cache() -> Iterator[None]:
    """Clear the API's process-local mtime cache before AND after each test.

    Mandatory for any contract test that compares CLI raw reads against
    API responses — the alternative is a stale cache returning a
    snapshot from a previous test fixture.
    """
    clear_cache()
    try:
        yield
    finally:
        clear_cache()


@pytest.fixture
def _clean_catalog() -> Iterator[None]:
    """Reload the singleton CommunityCatalog before AND after each test.

    Tests must not see plugin state leaked from a prior run.
    ``catalog.reload()`` is cheap (just resets internal lists +
    clears the loaded flag).
    """
    from ryotenkai_community.catalog import catalog

    catalog.reload()
    try:
        yield
    finally:
        catalog.reload()


@pytest.fixture
def parity_runs_dir(tmp_path: Path) -> Path:
    target = tmp_path / "runs"
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture
def parity_projects_root(tmp_path: Path) -> Path:
    target = tmp_path / "ryotenkai_home"
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture
def parity_settings(
    parity_runs_dir: Path, parity_projects_root: Path,
) -> ApiSettings:
    return ApiSettings(
        runs_dir=parity_runs_dir,
        projects_root=parity_projects_root,
        serve_spa=False,
        cors_origins=["http://localhost:5173"],
    )


@pytest.fixture
def parity_api_client(
    parity_settings: ApiSettings,
    _clean_state_cache: None,
    _clean_catalog: None,
) -> Iterator[TestClient]:
    """In-process FastAPI client tied to the same settings the CLI uses."""
    api_app = create_app(parity_settings)
    api_app.dependency_overrides[get_settings] = lambda: parity_settings
    with TestClient(api_app) as client:
        yield client


@pytest.fixture
def parity_cli_runner(
    parity_runs_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> CliRunner:
    """``CliRunner`` for the root Typer.

    Tests pass ``str(parity_runs_dir)`` explicitly to commands like
    ``runs ls``; we still set ``RYOTENKAI_RUNS_DIR`` for any
    ENV-aware path.
    """
    monkeypatch.setenv("RYOTENKAI_RUNS_DIR", str(parity_runs_dir))
    return CliRunner()


@pytest.fixture
def parity_cli_app():
    """Expose the root Typer for ``runner.invoke(parity_cli_app, ...)``."""
    return cli_app


__all__ = [
    "parity_api_client",
    "parity_cli_app",
    "parity_cli_runner",
    "parity_projects_root",
    "parity_runs_dir",
    "parity_settings",
]

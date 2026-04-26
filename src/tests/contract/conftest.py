"""Common fixtures for contract tests (CLI ↔ API parity).

Wires:

- An in-process FastAPI ``TestClient`` (or pure ASGI when needed) using
  ``ApiSettings(runs_dir=tmp_path)`` so neither side touches the real
  ``runs/`` directory.
- A ``CliRunner`` from ``typer.testing`` for invoking the CLI through
  the same root Typer the user runs.
- An autouse fixture that clears the process-local ``state_cache``
  before every test (NR-02 in plan B). Without this, the API's mtime-
  keyed cache can return a snapshot the CLI's raw read won't see —
  ``runs ls`` would diff against an outdated ``GET /runs`` payload.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from src.api.config import ApiSettings
from src.api.dependencies import get_settings
from src.api.main import create_app
from src.api.state_cache import clear_cache
from src.cli.app import app as cli_app


@pytest.fixture(autouse=True)
def _clean_state_cache() -> Iterator[None]:
    """Clear the API's process-local mtime cache before AND after each test.

    Mandatory for any contract test that compares CLI raw reads against
    API responses — see plan B NR-02 for the failure mode this prevents.
    """
    clear_cache()
    try:
        yield
    finally:
        clear_cache()


@pytest.fixture(autouse=True)
def _clean_catalog() -> Iterator[None]:
    """Reload the singleton CommunityCatalog before AND after each test.

    Plan B Q-24: tests must not see plugin state leaked from a prior
    run. ``catalog.reload()`` is cheap (just resets internal lists +
    clears the loaded flag).
    """
    from src.community.catalog import catalog

    catalog.reload()
    try:
        yield
    finally:
        catalog.reload()


@pytest.fixture()
def runs_dir(tmp_path: Path) -> Path:
    target = tmp_path / "runs"
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture()
def projects_root(tmp_path: Path) -> Path:
    target = tmp_path / "ryotenkai_home"
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture()
def settings(runs_dir: Path, projects_root: Path) -> ApiSettings:
    return ApiSettings(
        runs_dir=runs_dir,
        projects_root=projects_root,
        serve_spa=False,
        cors_origins=["http://localhost:5173"],
    )


@pytest.fixture()
def api_client(settings: ApiSettings) -> Iterator[TestClient]:
    """In-process FastAPI client tied to the same ``settings`` the CLI uses."""
    api_app = create_app(settings)
    api_app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(api_app) as client:
        yield client


@pytest.fixture()
def cli_runner(
    runs_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> CliRunner:
    """``CliRunner`` for the root Typer.

    Tests pass ``str(runs_dir)`` explicitly to commands like ``runs ls``;
    we still set ``RYOTENKAI_RUNS_DIR`` for any ENV-aware path.
    """
    monkeypatch.setenv("RYOTENKAI_RUNS_DIR", str(runs_dir))
    return CliRunner()


@pytest.fixture()
def cli_app_obj():
    """Expose the root Typer for ``runner.invoke(cli_app_obj, ...)``."""
    return cli_app


__all__ = [
    "api_client",
    "cli_app_obj",
    "cli_runner",
    "projects_root",
    "runs_dir",
    "settings",
]

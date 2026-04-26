"""Shared fixtures for runner unit tests.

Each test gets a fresh app and a temp workspace so the FSM state
files don't leak between tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from src.runner.main import create_app

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def runner_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> "Iterator[TestClient]":
    """``TestClient`` rooted in an isolated workspace directory."""
    monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
    with TestClient(create_app()) as client:
        yield client

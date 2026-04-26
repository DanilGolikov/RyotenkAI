"""Shared fixtures for runner unit tests.

Phase 0 ships only the FastAPI ``TestClient`` factory — Phase 1 adds
fixtures for FSM (in tmp dir), event bus (small ring buffer), and
mock supervisor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from src.runner.main import create_app

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def runner_client() -> "Iterator[TestClient]":
    """Build a fresh app + ``TestClient`` per test.

    Each test gets an isolated FastAPI instance — the factory pattern
    in :func:`src.runner.main.create_app` makes this cheap.
    """
    with TestClient(create_app()) as client:
        yield client

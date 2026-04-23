"""End-to-end tests for ETag / Last-Modified on state-reading endpoints.

These cover what the browser / React Query actually sees:
* first request returns 200 with ``ETag`` and ``Last-Modified`` set,
* a follow-up with ``If-None-Match`` returns ``304 Not Modified`` and an
  empty body,
* writing a new state bumps the ETag and the follow-up returns 200 again.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.pipeline.state import PipelineStateStore
from src.pipeline.state.cache import clear_cache

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from fastapi.testclient import TestClient


def _bump_state_mtime(run_dir: Path) -> None:
    """Simulate a writer saving a new state without touching the pydantic models."""
    state_path = PipelineStateStore(run_dir).state_path
    current = state_path.stat().st_mtime_ns
    future = current + 1_000_000_000
    os.utime(state_path, ns=(future, future))


def test_get_run_emits_cache_headers(
    client: TestClient,
    seed_completed_run: Callable[..., Path],
) -> None:
    clear_cache()
    seed_completed_run("run_cache_1")

    response = client.get("/api/v1/runs/run_cache_1")
    assert response.status_code == 200
    assert response.headers["ETag"].startswith('W/"')
    assert response.headers["Last-Modified"].endswith("GMT")
    assert response.headers["Cache-Control"] == "no-cache"


def test_get_run_returns_304_on_matching_etag(
    client: TestClient,
    seed_completed_run: Callable[..., Path],
) -> None:
    clear_cache()
    seed_completed_run("run_cache_2")
    first = client.get("/api/v1/runs/run_cache_2")
    etag = first.headers["ETag"]

    second = client.get("/api/v1/runs/run_cache_2", headers={"If-None-Match": etag})
    assert second.status_code == 304
    # 304 responses must not carry a body per RFC 7232 §4.1.
    assert second.content == b""
    # Validators refreshed so the next poll stays cheap.
    assert second.headers["ETag"] == etag


def test_get_run_returns_200_after_state_changes(
    client: TestClient,
    runs_dir: Path,
    seed_completed_run: Callable[..., Path],
) -> None:
    clear_cache()
    run_dir = seed_completed_run("run_cache_3")
    first = client.get("/api/v1/runs/run_cache_3")
    etag = first.headers["ETag"]

    _bump_state_mtime(run_dir)

    second = client.get("/api/v1/runs/run_cache_3", headers={"If-None-Match": etag})
    assert second.status_code == 200
    assert second.headers["ETag"] != etag


def test_get_attempt_detail_emits_cache_headers(
    client: TestClient,
    seed_completed_run: Callable[..., Path],
) -> None:
    clear_cache()
    seed_completed_run("run_cache_4")

    response = client.get("/api/v1/runs/run_cache_4/attempts/1")
    assert response.status_code == 200
    assert "ETag" in response.headers


def test_get_attempt_detail_304_on_match(
    client: TestClient,
    seed_completed_run: Callable[..., Path],
) -> None:
    clear_cache()
    seed_completed_run("run_cache_5")
    first = client.get("/api/v1/runs/run_cache_5/attempts/1")
    etag = first.headers["ETag"]

    second = client.get(
        "/api/v1/runs/run_cache_5/attempts/1",
        headers={"If-None-Match": etag},
    )
    assert second.status_code == 304


def test_get_attempt_stages_304_on_match(
    client: TestClient,
    seed_completed_run: Callable[..., Path],
) -> None:
    clear_cache()
    seed_completed_run("run_cache_6")
    first = client.get("/api/v1/runs/run_cache_6/attempts/1/stages")
    etag = first.headers["ETag"]

    second = client.get(
        "/api/v1/runs/run_cache_6/attempts/1/stages",
        headers={"If-None-Match": etag},
    )
    assert second.status_code == 304

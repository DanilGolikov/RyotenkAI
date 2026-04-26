"""Phase 0 smoke tests — verifies the runner package boots cleanly.

These tests pin the contract every later phase has to keep:
- ``src.runner`` package imports without side effects.
- The FastAPI factory produces a working app.
- Liveness / readiness / version probes respond.
- Sub-routers are mounted at the canonical prefixes.

Phase 1+ replaces the ``_skeleton`` placeholder in ``api/jobs.py``
with real endpoints; the corresponding test below will be deleted
in that phase.
"""

from __future__ import annotations

import re

import pytest

from src.runner import RUNTIME_IMAGE, create_app
from src.runner.api import events as events_api
from src.runner.api import internal as internal_api
from src.runner.api import jobs as jobs_api
from src.runner.event_bus import Event
from src.runner.main import API_V1_PREFIX
from src.runner.state import JobSnapshot, JobState


class TestPackageSurface:
    """Phase 0 contract: the public surface listed in ``__init__``."""

    def test_runtime_image_is_pinned(self) -> None:
        """``RUNTIME_IMAGE`` must follow the `<repo>:<tag>` shape."""
        assert re.match(r"^[\w./-]+:[\w.-]+$", RUNTIME_IMAGE)

    def test_runtime_image_does_not_use_override(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """When the override env is unset, the resolved image matches the package default.

        Probes the override mechanism through its public effect — re-importing
        ``__about__`` with the env set yields a different value, and clearing
        the env restores the canonical pin. This guards both directions
        without poking at the underscore-prefixed default.
        """
        import importlib

        from src.runner import __about__ as about_module

        monkeypatch.delenv("RYOTENKAI_RUNTIME_IMAGE_OVERRIDE", raising=False)
        baseline = importlib.reload(about_module).RUNTIME_IMAGE
        assert baseline == RUNTIME_IMAGE  # current import equals fresh reload — no env in effect

        monkeypatch.setenv("RYOTENKAI_RUNTIME_IMAGE_OVERRIDE", "ghcr.io/example/dev:0.0.0")
        overridden = importlib.reload(about_module).RUNTIME_IMAGE
        assert overridden == "ghcr.io/example/dev:0.0.0"
        assert overridden != baseline

        # Clean up: reload again with no env so module-level state stays canonical.
        monkeypatch.delenv("RYOTENKAI_RUNTIME_IMAGE_OVERRIDE", raising=False)
        importlib.reload(about_module)

    def test_create_app_is_idempotent(self) -> None:
        """Two calls to the factory build independent apps."""
        app1 = create_app()
        app2 = create_app()
        assert app1 is not app2


class TestStateModel:
    """``JobState`` enum invariants — Phase 1 will add transition rules."""

    def test_terminal_states(self) -> None:
        terminal = {s for s in JobState if s.is_terminal}
        assert terminal == {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}

    def test_non_terminal_states(self) -> None:
        non_terminal = {s for s in JobState if not s.is_terminal}
        assert non_terminal == {JobState.PREPARING, JobState.RUNNING, JobState.STOPPING}

    def test_snapshot_is_immutable(self) -> None:
        snap = JobSnapshot(job_id="j-1", state=JobState.RUNNING)
        with pytest.raises(AttributeError):  # frozen dataclass
            snap.state = JobState.COMPLETED  # type: ignore[misc]


class TestEventBusContract:
    """``Event`` dataclass — placeholder until Phase 1 lands the bus."""

    def test_event_is_frozen(self) -> None:
        e = Event(offset=0, timestamp="2026-04-26T00:00:00Z", kind="test", payload={"k": "v"})
        with pytest.raises(AttributeError):
            e.offset = 1  # type: ignore[misc]


class TestHTTPSurface:
    """The FastAPI app must boot and answer probes in Phase 0."""

    def test_healthz(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        r = runner_client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_readyz(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        r = runner_client.get("/readyz")
        assert r.status_code == 200
        assert r.json() == {"status": "ready"}

    def test_version(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        r = runner_client.get("/version")
        assert r.status_code == 200
        assert r.json() == {"image": RUNTIME_IMAGE}

    def test_jobs_skeleton_is_mounted(self, runner_client) -> None:  # type: ignore[no-untyped-def]
        """Phase 1 deletes this — confirms the router is mounted now."""
        r = runner_client.get(f"{API_V1_PREFIX}/jobs/_skeleton")
        assert r.status_code == 200
        assert r.json() == {"status": "skeleton"}

    def test_routers_mounted(self) -> None:
        """Each sub-router is a real FastAPI APIRouter, not a stub."""
        from fastapi import APIRouter

        for module in (jobs_api, internal_api, events_api):
            assert isinstance(module.router, APIRouter)

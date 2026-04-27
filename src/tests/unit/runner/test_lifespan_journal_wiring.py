"""Phase 12.B — :func:`_lifespan` journal wiring contract.

Pin that the FastAPI lifespan constructs a journal under
``<workspace>/.runner/events/``, attaches it to the bus, and stashes
it on ``app.state.journal``. Failure to construct the journal must
fall back to journal-less behaviour without crashing the runner boot.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Positive — journal wired on lifespan
# ---------------------------------------------------------------------------


class TestPositive:
    def test_journal_attached_to_bus(self, runner_client) -> None:
        # The fixture's TestClient triggers lifespan startup; once
        # any HTTP request returns, app.state is populated.
        # ``runner_client`` is from ``conftest.runner_client``.
        runner_client.get("/healthz")

        app = runner_client.app
        assert app.state.journal is not None
        assert app.state.bus.journal is app.state.journal

    def test_journal_root_dir_under_workspace(
        self, runner_client, tmp_path: Path,
    ) -> None:
        runner_client.get("/healthz")
        app = runner_client.app
        assert app.state.journal.root_dir == tmp_path / ".runner" / "events"


# ---------------------------------------------------------------------------
# 2. Robustness — fallback when journal init fails
# ---------------------------------------------------------------------------


class TestFallback:
    def test_runner_boots_when_journal_init_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi.testclient import TestClient

        from src.runner.main import create_app
        from src.tests.unit.runner.conftest import MockSupervisor

        # Force EventJournal construction to raise.
        from src.runner import main as main_mod

        class _BrokenJournal:
            def __init__(self, *args, **kwargs):
                raise OSError("simulated EROFS")

        monkeypatch.setattr(main_mod, "EventJournal", _BrokenJournal)
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))

        # Lifespan should NOT raise; runner boots in journal-less mode.
        with TestClient(create_app(supervisor_factory=MockSupervisor)) as client:
            r = client.get("/healthz")
            assert r.status_code == 200
            assert client.app.state.journal is None
            assert client.app.state.bus.journal is None


# ---------------------------------------------------------------------------
# 3. Persistence — published events end up on disk
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_published_events_visible_on_disk(
        self, runner_client, tmp_path: Path
    ) -> None:
        runner_client.get("/healthz")
        bus = runner_client.app.state.bus
        bus.publish("test_event", {"value": 42})

        # Read back via the same journal.
        journal = runner_client.app.state.journal
        records = list(journal.iter_records(since=0))
        assert any(
            r.kind == "test_event" and r.payload.get("value") == 42
            for r in records
        )

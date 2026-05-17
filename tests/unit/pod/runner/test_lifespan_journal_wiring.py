"""Phase 12.B — :func:`_lifespan` journal wiring contract.

Pin that the FastAPI lifespan constructs a journal under
``<workspace>/events/`` (per :class:`PodLayout`), attaches it to the
bus, and stashes it on ``app.state.journal``. Failure to construct
the journal must fall back to journal-less behaviour without
crashing the runner boot.
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
        assert app.state.journal.root_dir == tmp_path / "events"


# ---------------------------------------------------------------------------
# 2. Robustness — fallback when journal init fails
# ---------------------------------------------------------------------------


class TestFallback:
    def test_runner_boots_when_journal_init_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Batch 6b: ``tests/`` is not a Python package (no __init__.py), so the
        # original ``from tests.unit.pod.runner.conftest import MockSupervisor``
        # fails under any import-mode. Load conftest via importlib — same
        # pattern as test_phase_14e_srp_fixes.py.
        import importlib.util as _ilu
        import pathlib as _pathlib

        from fastapi.testclient import TestClient

        from ryotenkai_pod.runner.main import create_app
        _conftest_path = _pathlib.Path(__file__).resolve().parent / "conftest.py"
        _spec = _ilu.spec_from_file_location(
            "_pod_runner_conftest_for_lifespan_journal_wiring",
            str(_conftest_path),
        )
        assert _spec is not None and _spec.loader is not None
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        MockSupervisor = _mod.MockSupervisor

        # Force EventJournal construction to raise.
        from ryotenkai_pod.runner import main as main_mod

        class _BrokenJournal:
            def __init__(self, *args, **kwargs):
                raise OSError("simulated EROFS")

        monkeypatch.setattr(main_mod, "EventJournal", _BrokenJournal)
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        # Phase 14.B — provide a default runtime provider so the
        # lifespan can boot. Single-node = NoOp client, no creds.
        monkeypatch.setenv("RYOTENKAI_RUNTIME_PROVIDER", "single_node")

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
        from ryotenkai_pod.runner.event_bus import legacy_kind_for

        runner_client.get("/healthz")
        bus = runner_client.app.state.bus
        bus.publish_legacy("test_event", {"value": 42})

        # Read back via the same journal.
        journal = runner_client.app.state.journal
        envelopes = list(journal.iter_envelopes())
        matched = False
        for ev in envelopes:
            if legacy_kind_for(ev) != "test_event":
                continue
            raw = getattr(ev, "raw_payload", None)
            if isinstance(raw, dict) and raw.get("value") == 42:
                matched = True
                break
        assert matched

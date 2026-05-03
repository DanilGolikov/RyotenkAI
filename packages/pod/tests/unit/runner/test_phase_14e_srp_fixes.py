"""Phase 14.E — coverage for the SRP/SoC fixes.

Single test module covering V1 (rotation binding), V3 (heartbeat
helper), V6 (validate_journal_config), V8 (heartbeat constants
re-export). V5 (ExceptionClassifier) lives separately because it's
in the training/mlflow package and the tests there require
``datasets`` which slim CI venvs don't have.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Stub `runpod` so any provider imports during lifespan boot are safe.
if "runpod" not in sys.modules:
    _stub = types.ModuleType("runpod")
    _stub.api_key = ""
    _stub.create_pod = MagicMock()
    _stub.get_pod = MagicMock()
    _stub.stop_pod = MagicMock()
    _stub.resume_pod = MagicMock()
    _stub.terminate_pod = MagicMock()
    sys.modules["runpod"] = _stub


from fastapi.testclient import TestClient  # noqa: E402

from ryotenkai_pod.runner.event_bus import EventBus  # noqa: E402
from ryotenkai_pod.runner.event_journal import (  # noqa: E402
    EventJournal,
    validate_journal_config,
)
from ryotenkai_pod.runner.api._activity import (  # noqa: E402
    mark_heartbeat_if_present,
    send_ws_with_activity,
)
from ryotenkai_pod.runner.heartbeat import (  # noqa: E402
    EXPLICIT_HEARTBEAT_TTL_SECONDS,
    HEARTBEAT_TTL_SECONDS,
    MacHeartbeat,
)
from ryotenkai_pod.runner.main import create_app  # noqa: E402
# Same-folder conftest provides MockSupervisor; load via importlib (see
# test_main_lifespan_bootstrap for rationale).
import importlib.util as _ilu  # noqa: E402
import pathlib as _pathlib  # noqa: E402

_conftest_path = _pathlib.Path(__file__).resolve().parent / "conftest.py"
_spec = _ilu.spec_from_file_location("_pod_runner_conftest_for_phase_14e", str(_conftest_path))
assert _spec is not None and _spec.loader is not None
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MockSupervisor = _mod.MockSupervisor  # noqa: E402


# ---------------------------------------------------------------------------
# V1 — rotation binding via deferred attach
# ---------------------------------------------------------------------------


class TestRotationBinding:
    def test_attach_journal_rotation_listener_registers_callback(
        self, tmp_path: Path,
    ) -> None:
        # Pre-14.E this required the lifespan's mutable cell hack.
        # Post-14.E: bus.attach() wires the callback after both
        # objects exist.
        journal = EventJournal(root_dir=tmp_path / "events")
        bus = EventBus(journal=journal)
        bus.attach_journal_rotation_listener()
        # Bound method equality (not identity — fresh bound method
        # objects on each access).
        assert journal._on_rotate == bus._publish_rotation_event

    def test_attach_with_no_journal_is_noop(self) -> None:
        # Bus constructed without a journal → attach is a no-op.
        bus = EventBus(journal=None)
        bus.attach_journal_rotation_listener()  # must not raise

    def test_publish_rotation_event_emits_events_rotated(
        self, tmp_path: Path,
    ) -> None:
        from ryotenkai_pod.runner.cancellation_telemetry import EVENTS_ROTATED

        journal = EventJournal(root_dir=tmp_path / "events")
        bus = EventBus(journal=journal)
        bus.attach_journal_rotation_listener()

        # Manually invoke the callback (simulates a journal rotation).
        bus._publish_rotation_event(
            from_seq=1, to_seq=2, file_size_bytes=1024,
            oldest_remaining_seq=0,
        )

        # Newest event in the bus buffer is EVENTS_ROTATED with the payload.
        latest = bus._buffer[-1]
        assert latest.kind == EVENTS_ROTATED
        assert latest.payload["from_seq"] == 1
        assert latest.payload["to_seq"] == 2

    def test_lifespan_uses_deferred_binding_pattern(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # End-to-end: lifespan boots, journal exists, bus attaches.
        monkeypatch.setenv("RYOTENKAI_WORKSPACE", str(tmp_path))
        monkeypatch.setenv("RYOTENKAI_RUNTIME_PROVIDER", "single_node")
        with TestClient(create_app(supervisor_factory=MockSupervisor)) as client:
            assert client.app.state.journal is not None
            # _on_rotate should be the bus's internal publisher,
            # NOT the legacy circular-binding closure.
            assert (
                client.app.state.journal._on_rotate
                == client.app.state.bus._publish_rotation_event
            )


# ---------------------------------------------------------------------------
# V3 — heartbeat helper
# ---------------------------------------------------------------------------


class _AppStateStub:
    def __init__(self, heartbeat: Any = None) -> None:
        self.heartbeat = heartbeat


class TestMarkHeartbeatIfPresent:
    def test_marks_active_when_heartbeat_present(self) -> None:
        hb = MagicMock()
        mark_heartbeat_if_present(_AppStateStub(heartbeat=hb))
        hb.mark_active.assert_called_once()

    def test_noop_when_heartbeat_absent(self) -> None:
        # No-op, no exception.
        mark_heartbeat_if_present(_AppStateStub(heartbeat=None))

    def test_noop_when_app_state_is_none(self) -> None:
        # Defensive: app_state itself None → still no-op.
        mark_heartbeat_if_present(None)

    def test_noop_when_app_state_lacks_heartbeat_attr(self) -> None:
        class _Bare: ...
        mark_heartbeat_if_present(_Bare())  # must not raise


class TestSendWsWithActivity:
    @pytest.mark.asyncio
    async def test_sends_then_marks_in_order(self) -> None:
        order: list[str] = []
        ws = MagicMock()

        async def _fake_send(payload: dict) -> None:
            order.append("send")

        ws.send_json = _fake_send
        hb = MagicMock()
        hb.mark_active.side_effect = lambda: order.append("mark")

        await send_ws_with_activity(ws, {"x": 1}, _AppStateStub(heartbeat=hb))
        assert order == ["send", "mark"]

    @pytest.mark.asyncio
    async def test_send_failure_skips_mark(self) -> None:
        ws = MagicMock()

        async def _fake_send(payload: dict) -> None:
            raise ConnectionError("Mac asleep")

        ws.send_json = _fake_send
        hb = MagicMock()

        with pytest.raises(ConnectionError):
            await send_ws_with_activity(
                ws, {"x": 1}, _AppStateStub(heartbeat=hb),
            )
        # mark_active NOT called on failed send — pre-14.E inline
        # code had the same property; pin it.
        hb.mark_active.assert_not_called()


# ---------------------------------------------------------------------------
# V6 — validate_journal_config
# ---------------------------------------------------------------------------


class TestValidateJournalConfig:
    def test_valid_params_pass(self) -> None:
        validate_journal_config(
            file_size_cap=1024, max_files=5,
            fsync_batch=10, fsync_interval_ms=100,
        )  # must not raise

    def test_zero_file_size_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="file_size_cap"):
            validate_journal_config(
                file_size_cap=0, max_files=1,
                fsync_batch=1, fsync_interval_ms=0,
            )

    def test_zero_max_files_raises(self) -> None:
        with pytest.raises(ValueError, match="max_files"):
            validate_journal_config(
                file_size_cap=1, max_files=0,
                fsync_batch=1, fsync_interval_ms=0,
            )

    def test_zero_fsync_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="fsync_batch"):
            validate_journal_config(
                file_size_cap=1, max_files=1,
                fsync_batch=0, fsync_interval_ms=0,
            )

    def test_negative_fsync_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="fsync_interval_ms"):
            validate_journal_config(
                file_size_cap=1, max_files=1,
                fsync_batch=1, fsync_interval_ms=-1,
            )

    def test_pure_function_no_filesystem_touch(self, tmp_path: Path) -> None:
        # Pin: validation does NOT create directories.
        # (EventJournal __init__ creates root_dir; validation alone
        # must not.)
        validate_journal_config(
            file_size_cap=1, max_files=1,
            fsync_batch=1, fsync_interval_ms=0,
        )
        # tmp_path's children should be empty.
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# V8 — heartbeat constants re-exported at module level
# ---------------------------------------------------------------------------


class TestHeartbeatConstants:
    def test_module_level_aliases_match_class_attrs(self) -> None:
        # Phase 14.E (V8) — operators import the constants directly
        # without pulling the class.
        assert HEARTBEAT_TTL_SECONDS == MacHeartbeat.HEARTBEAT_TTL_SECONDS
        assert (
            EXPLICIT_HEARTBEAT_TTL_SECONDS
            == MacHeartbeat.EXPLICIT_HEARTBEAT_TTL_SECONDS
        )

    def test_explicit_default_120s(self) -> None:
        # Pin the value — operator dashboards may grep on it.
        assert EXPLICIT_HEARTBEAT_TTL_SECONDS == 120.0

    def test_implicit_default_60s(self) -> None:
        assert HEARTBEAT_TTL_SECONDS == 60.0

"""Unit tests for :class:`PodEventForwarder`.

The forwarder was extracted from :class:`TrainingMonitor` in
2026-05-17 to keep ``training_monitor.py`` under the 1500-line
architectural guardrail. This module exercises the forwarder in
isolation — no monitor fixture indirection — so behaviour can be
verified without dragging in the full pipeline / WS stack.

Coverage:

* :meth:`PodEventForwarder.dispatch_event` — milestone logging,
  health-snapshot side effects, trainer_exited / FSM-state routing
  through the injected callbacks.
* :meth:`PodEventForwarder.is_new_envelope_shape` — detection of new
  vs legacy wire shape.
* :meth:`PodEventForwarder.forward_pod_envelope` — emitter wiring,
  legacy-shape skip, ``kind_dotted`` / ``timestamp`` normalisation.
* :meth:`PodEventForwarder.maybe_log_status` — rate-limit window,
  field-rendering fallbacks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from ryotenkai_control.pipeline.stages.training_monitor_pod_event_forwarder import (
    PodEventForwarder,
)
from ryotenkai_pod.runner.event_bus import envelope_to_wire
from ryotenkai_shared.events.types.pod_training import (
    TrainingStartedEvent,
    TrainingStartedPayload,
)

from tests._fakes.event_emitter import FakeEventEmitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


@dataclass
class _FakeMonitorState:
    """Concrete impl of the forwarder's ``MonitorState`` Protocol.

    Plain dataclass — the forwarder reads + mutates these attributes by
    name, so a dataclass with the right field set is enough. Tests can
    inspect / pin them directly.
    """

    _first_event_logged: bool = False
    _trainer_started_logged: bool = False
    _last_event_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _last_status_log_time: float = 0.0
    _last_offset: int = 0
    _training_start_time: float = 0.0


def _make_forwarder(
    *,
    emitter: FakeEventEmitter | None = None,
    state: _FakeMonitorState | None = None,
    on_exit: Any = None,
    on_state: Any = None,
    on_fallback: Any = None,
) -> tuple[PodEventForwarder, _FakeMonitorState, FakeEventEmitter]:
    """Build a forwarder with stub callbacks and shared state.

    All callbacks default to "raise if invoked" so a stray dispatch
    that should have been silent is caught loudly. Tests that need a
    callback can override.
    """
    state = state or _FakeMonitorState()
    emitter = emitter if emitter is not None else FakeEventEmitter()

    def _raise(*a: Any, **kw: Any) -> dict[str, Any]:
        raise AssertionError("unexpected callback invocation")

    async def _raise_async(*a: Any, **kw: Any) -> dict[str, Any]:
        raise AssertionError("unexpected fallback invocation")

    fwd = PodEventForwarder(
        state=state,  # type: ignore[arg-type]  # dataclass satisfies the Protocol structurally
        emitter=emitter,
        handle_trainer_exited=on_exit or _raise,
        terminal_from_state=on_state or _raise,
        fallback_to_status=on_fallback or _raise_async,
        terminal_states=_TERMINAL_STATES,
        log_status_interval=15,
    )
    return fwd, state, emitter


# ---------------------------------------------------------------------------
# Envelope-shape detection
# ---------------------------------------------------------------------------


class TestIsNewEnvelopeShape:
    def test_new_shape_with_kind_dotted_returns_true(self) -> None:
        fwd, _, _ = _make_forwarder()
        assert fwd.is_new_envelope_shape(
            {"kind_dotted": "ryotenkai.pod.training.started", "payload": {}},
        )

    def test_new_shape_with_event_id_returns_true(self) -> None:
        fwd, _, _ = _make_forwarder()
        assert fwd.is_new_envelope_shape({"event_id": "abc", "payload": {}})

    def test_new_shape_with_schema_version_returns_true(self) -> None:
        fwd, _, _ = _make_forwarder()
        assert fwd.is_new_envelope_shape({"schema_version": 1, "payload": {}})

    def test_legacy_shape_returns_false(self) -> None:
        fwd, _, _ = _make_forwarder()
        # Pre-Phase-2 ``Event.to_dict()`` shape.
        assert not fwd.is_new_envelope_shape(
            {"offset": 0, "timestamp": "2026-05-17T00:00:00Z", "kind": "health_snapshot", "payload": {}},
        )


# ---------------------------------------------------------------------------
# Dispatch — milestone & routing
# ---------------------------------------------------------------------------


class TestDispatchMilestones:
    def test_first_event_sets_logged_flag(self) -> None:
        fwd, state, _ = _make_forwarder()
        assert not state._first_event_logged

        fwd.dispatch_event({"kind": "anything", "payload": {}})

        assert state._first_event_logged

    def test_trainer_spawned_sets_started_flag(self) -> None:
        fwd, state, _ = _make_forwarder()
        assert not state._trainer_started_logged

        fwd.dispatch_event({"kind": "trainer_spawned", "payload": {"pid": 1234}})

        assert state._trainer_started_logged

    def test_trainer_spawned_idempotent(self) -> None:
        fwd, state, _ = _make_forwarder()
        fwd.dispatch_event({"kind": "trainer_spawned", "payload": {"pid": 1}})
        # Should not raise even though flag already set.
        fwd.dispatch_event({"kind": "trainer_spawned", "payload": {"pid": 2}})
        assert state._trainer_started_logged


class TestDispatchRouting:
    def test_health_snapshot_refreshes_last_event_at(self) -> None:
        fwd, state, _ = _make_forwarder()
        before = state._last_event_at

        fwd.dispatch_event({
            "kind": "health_snapshot",
            "payload": {"gpu_util_percent": 50.0},
        })

        assert state._last_event_at >= before

    def test_health_snapshot_returns_none(self) -> None:
        fwd, _, _ = _make_forwarder()
        result = fwd.dispatch_event({
            "kind": "health_snapshot",
            "payload": {"gpu_util_percent": 50.0},
        })
        assert result is None

    def test_trainer_exited_invokes_handler(self) -> None:
        captured: list[dict[str, Any]] = []

        def _on_exit(payload: dict[str, Any]) -> dict[str, Any]:
            captured.append(payload)
            return {"status": "completed", "duration_seconds": 0.0}

        fwd, _, _ = _make_forwarder(on_exit=_on_exit)

        result = fwd.dispatch_event({
            "kind": "trainer_exited",
            "payload": {"exit_code": 0},
        })

        assert captured == [{"exit_code": 0}]
        assert result == {"status": "completed", "duration_seconds": 0.0}

    def test_bare_terminal_state_invokes_state_handler(self) -> None:
        captured: list[tuple[str, dict[str, Any]]] = []

        def _on_state(state: str, payload: dict[str, Any]) -> dict[str, Any]:
            captured.append((state, payload))
            return {"status": state}

        fwd, _, _ = _make_forwarder(on_state=_on_state)

        result = fwd.dispatch_event({
            "kind": "state_changed",
            "payload": {"state": "completed", "extra": "x"},
        })

        assert captured == [("completed", {"state": "completed", "extra": "x"})]
        assert result == {"status": "completed"}

    def test_unknown_kind_returns_none(self) -> None:
        fwd, _, _ = _make_forwarder()
        result = fwd.dispatch_event({"kind": "future_unknown_event", "payload": {}})
        assert result is None


# ---------------------------------------------------------------------------
# Envelope forwarding
# ---------------------------------------------------------------------------


def _wire_started(offset: int = 0) -> dict[str, Any]:
    """Build a real wire-shape envelope via the production helper."""
    event = TrainingStartedEvent(
        source="pod://runner/trainer",
        run_id="run-fwd-unit",
        offset=offset,
        payload=TrainingStartedPayload(
            max_steps=10,
            num_train_epochs=1,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            algorithm="sft",
        ),
    )
    return envelope_to_wire(event)


class TestForwardPodEnvelope:
    def test_typed_envelope_reaches_emitter(self) -> None:
        emitter = FakeEventEmitter()
        fwd, _, _ = _make_forwarder(emitter=emitter)

        fwd.forward_pod_envelope(_wire_started(offset=0))

        assert len(emitter.received_remote) == 1
        assert emitter.received_remote[0].kind == "ryotenkai.pod.training.started"

    def test_legacy_shape_is_not_forwarded(self) -> None:
        emitter = FakeEventEmitter()
        fwd, _, _ = _make_forwarder(emitter=emitter)

        fwd.forward_pod_envelope({
            "offset": 0,
            "timestamp": "2026-05-17T00:00:00Z",
            "kind": "health_snapshot",
            "payload": {"gpu_util_percent": 10.0},
        })

        assert emitter.received_remote == []

    def test_no_emitter_is_a_silent_noop(self) -> None:
        fwd, _, _ = _make_forwarder(emitter=None)
        fwd.set_emitter(None)
        # Must not raise.
        fwd.forward_pod_envelope(_wire_started(offset=0))

    def test_malformed_envelope_is_dropped_not_raised(self) -> None:
        emitter = FakeEventEmitter()
        fwd, _, _ = _make_forwarder(emitter=emitter)

        # Looks new-shape (kind_dotted present) but payload is invalid
        # for the declared kind — adapter rejects it.
        fwd.forward_pod_envelope({
            "kind_dotted": "ryotenkai.pod.training.started",
            "kind": "training_started",
            "event_id": "00000000-0000-0000-0000-000000000000",
            "schema_version": 1,
            "time": "2026-05-17T00:00:00Z",
            "offset": 0,
            "run_id": "x",
            "source": "pod://x",
            "payload": {"this": "is", "not": "a valid started payload"},
        })

        assert emitter.received_remote == []


# ---------------------------------------------------------------------------
# Status logging
# ---------------------------------------------------------------------------


class TestMaybeLogStatus:
    def test_first_call_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        fwd, _, _ = _make_forwarder()
        with caplog.at_level("INFO"):
            fwd.maybe_log_status({"gpu_util_percent": 80.0})
        running_lines = [r for r in caplog.records if "running" in r.getMessage()]
        assert len(running_lines) >= 1

    def test_rate_limit_within_window_suppresses_repeat(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        fwd, _, _ = _make_forwarder()
        fake_now = [1000.0]
        monkeypatch.setattr(time, "time", lambda: fake_now[0])

        with caplog.at_level("INFO"):
            fwd.maybe_log_status({"gpu_util_percent": 80.0})
            # Same instant — suppressed.
            fwd.maybe_log_status({"gpu_util_percent": 80.0})

        running_lines = [r for r in caplog.records if "running" in r.getMessage()]
        assert len(running_lines) == 1

    def test_window_expiry_re_emits(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        fwd, _, _ = _make_forwarder()
        fake_now = [1000.0]
        monkeypatch.setattr(time, "time", lambda: fake_now[0])

        with caplog.at_level("INFO"):
            fwd.maybe_log_status({"gpu_util_percent": 80.0})
            fake_now[0] += 16  # past the 15 s window
            fwd.maybe_log_status({"gpu_util_percent": 80.0})

        running_lines = [r for r in caplog.records if "running" in r.getMessage()]
        assert len(running_lines) == 2

    def test_missing_fields_render_as_dash(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        fwd, _, _ = _make_forwarder()
        with caplog.at_level("INFO"):
            fwd.maybe_log_status({})

        running_lines = [r for r in caplog.records if "running" in r.getMessage()]
        assert running_lines
        rendered = running_lines[0].getMessage()
        # GPU / VRAM / Temp / CPU / RAM all fall back to em-dash when
        # the snapshot is empty.
        assert rendered.count("—") >= 4

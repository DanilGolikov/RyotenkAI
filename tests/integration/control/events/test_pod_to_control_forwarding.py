"""Integration: pod WS events forwarded into control's ``events.jsonl``.

Closes the post-Phase-10 gap where :meth:`TrainingMonitor._dispatch_event`
consumed event dicts from :meth:`JobClient.subscribe_events` but never
called :meth:`IEventEmitter.emit_remote`. The downstream effect was
that every pod-side typed envelope produced by Phase 2's
``RunnerEventCallback`` (training started, step, epoch boundaries,
checkpoints, terminal disposition, memory and health snapshots, etc.)
was effectively orphaned: it never reached the control-side journal,
the live SSE fan-out, ``MlflowFinalizer.upload``, or the
``JournalReportAdapter`` training timeline.

The fix wraps each event dict from the WS stream in a sanitising
heuristic — wire shape that carries ``kind_dotted`` /
``schema_version`` / ``event_id`` is "new envelope" and gets
re-validated through :data:`EVENT_ADAPTER` then forwarded via
``emitter.emit_remote``; the legacy ``Event.to_dict`` shape (only
``offset`` + ``timestamp`` + ``kind`` + ``payload``) is left
untouched so the inline operator-facing callbacks keep running.

Coverage:

* Smoke — typed envelopes from the WS stream land on the emitter's
  ``received_remote`` list in the same order, with all identity
  fields preserved.
* Edge cases — malformed envelope (validation failure) is dropped
  with a warning; legacy wire shape is NOT forwarded; duplicate
  envelopes are deduped by the emitter so the journal does not grow
  with reruns.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ryotenkai_pod.runner.event_bus import envelope_to_wire
from ryotenkai_shared.events.types.pod_training import (
    TrainingStartedEvent,
    TrainingStartedPayload,
    TrainingStepEvent,
    TrainingStepPayload,
)

from tests._fakes.event_emitter import FakeEventEmitter
from tests.unit.control.pipeline.test_training_monitor_v2 import (
    _ctx_with_handles,
    _make_monitor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_started_envelope(*, offset: int, run_id: str = "run-fwd") -> dict[str, Any]:
    """Build a wire-shape envelope for a typed pod training_started event.

    Uses the real pod-side ``envelope_to_wire`` so the test fakes the
    exact bytes the production WS handler would push.
    """
    event = TrainingStartedEvent(
        source="pod://runner/trainer",
        run_id=run_id,
        offset=offset,
        payload=TrainingStartedPayload(
            max_steps=100,
            num_train_epochs=1,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            algorithm="sft",
        ),
    )
    return envelope_to_wire(event)


def _make_step_envelope(*, offset: int, step: int, run_id: str = "run-fwd") -> dict[str, Any]:
    event = TrainingStepEvent(
        source="pod://runner/trainer",
        run_id=run_id,
        offset=offset,
        payload=TrainingStepPayload(
            step=step,
            loss=0.5,
            learning_rate=1e-4,
        ),
    )
    return envelope_to_wire(event)


def _make_client(events: list[dict[str, Any]]) -> Any:
    """Fake JobClient whose subscribe_events yields the given wire dicts."""
    client = MagicMock()

    async def _stream(_job_id: str, *, since: int = 0, **_kw: Any) -> Any:
        for ev in events:
            yield ev

    client.subscribe_events = _stream
    client.aclose = AsyncMock(return_value=None)
    client.get_status = AsyncMock(return_value={"state": "completed"})
    return client


# ---------------------------------------------------------------------------
# Smoke — typed envelopes survive the WS → emitter forwarding
# ---------------------------------------------------------------------------


class TestForwardingSmoke:
    def test_pod_typed_events_reach_emitter_in_order(self) -> None:
        emitter = FakeEventEmitter()
        monitor = _make_monitor(emitter=emitter)

        wire_events = [
            _make_started_envelope(offset=10),
            _make_step_envelope(offset=11, step=1),
            _make_step_envelope(offset=12, step=2),
            # Terminal — trainer exited 0 (also a typed envelope; here we
            # use the legacy minimal dict so the monitor's existing
            # ``_handle_trainer_exited`` path closes the run cleanly).
            {
                "offset": 13,
                "kind": "trainer_exited",
                "payload": {
                    "exit_code": 0,
                    "signal": None,
                    "cancellation_requested": False,
                },
            },
        ]
        client = _make_client(wire_events)

        result = monitor.execute(_ctx_with_handles(client))
        assert result["status"] == "completed"

        # The two typed pod events were forwarded; the trainer_exited
        # legacy dict (no ``kind_dotted``) is intentionally NOT forwarded.
        forwarded_kinds = [ev.kind for ev in emitter.received_remote]
        assert forwarded_kinds == [
            "ryotenkai.pod.training.started",
            "ryotenkai.pod.training.step",
            "ryotenkai.pod.training.step",
        ]
        # Identity fields preserved verbatim (offset / run_id / source).
        assert [ev.offset for ev in emitter.received_remote] == [10, 11, 12]
        assert {ev.run_id for ev in emitter.received_remote} == {"run-fwd"}
        assert {ev.source for ev in emitter.received_remote} == {"pod://runner/trainer"}

    def test_missing_emitter_does_not_crash_loop(self) -> None:
        """No emitter wired → forwarding silently no-ops, terminal still ok."""
        monitor = _make_monitor(emitter=None)

        wire_events = [
            _make_started_envelope(offset=0),
            {
                "offset": 1,
                "kind": "trainer_exited",
                "payload": {
                    "exit_code": 0,
                    "signal": None,
                    "cancellation_requested": False,
                },
            },
        ]
        client = _make_client(wire_events)
        result = monitor.execute(_ctx_with_handles(client))
        assert result["status"] == "completed"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestForwardingEdgeCases:
    def test_malformed_envelope_dropped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A wire dict that looks new-shape but fails validation must
        be skipped — never crash the WS consumer loop."""
        emitter = FakeEventEmitter()
        monitor = _make_monitor(emitter=emitter)

        # Looks new-shape (has kind_dotted) but the payload doesn't
        # match the schema — ``max_steps`` is required.
        malformed: dict[str, Any] = {
            "event_id": "01234567-89ab-cdef-0123-456789abcdef",
            "kind": "ryotenkai.pod.training.started",
            "kind_dotted": "ryotenkai.pod.training.started",
            "source": "pod://runner/trainer",
            "time": "2026-05-17T00:00:00+00:00",
            "timestamp": "2026-05-17T00:00:00+00:00",
            "run_id": "run-fwd",
            "stage_id": None,
            "offset": 0,
            "schema_version": 1,
            "severity": "info",
            "payload": {"this": "is", "not": "valid"},
        }

        wire_events = [
            malformed,
            _make_started_envelope(offset=1),
            {
                "offset": 2,
                "kind": "trainer_exited",
                "payload": {
                    "exit_code": 0,
                    "signal": None,
                    "cancellation_requested": False,
                },
            },
        ]
        client = _make_client(wire_events)
        result = monitor.execute(_ctx_with_handles(client))
        assert result["status"] == "completed"

        # Only the well-formed envelope was forwarded; the malformed
        # one was dropped without taking down the loop.
        assert len(emitter.received_remote) == 1
        assert emitter.received_remote[0].offset == 1

    def test_legacy_wire_shape_not_forwarded(self) -> None:
        """The pre-Phase-2 ``Event.to_dict()`` shape (no envelope fields)
        is intentionally NOT forwarded — inline callbacks still run,
        but ``emit_remote`` is bypassed so we don't materialize an
        ``UnknownEvent`` for every legacy frame."""
        emitter = FakeEventEmitter()
        monitor = _make_monitor(emitter=emitter)

        wire_events = [
            # Pure legacy shape — only the 4 keys the old pod emitted.
            {"offset": 0, "kind": "trainer_spawned", "payload": {"pid": 1234}},
            {"offset": 1, "kind": "health_snapshot", "payload": {"cpu_pct": 12.0}},
            {
                "offset": 2,
                "kind": "trainer_exited",
                "payload": {
                    "exit_code": 0,
                    "signal": None,
                    "cancellation_requested": False,
                },
            },
        ]
        client = _make_client(wire_events)
        result = monitor.execute(_ctx_with_handles(client))
        assert result["status"] == "completed"

        # Legacy shape → no remote-emit calls landed.
        assert emitter.received_remote == []

    def test_duplicate_envelope_deduped(self) -> None:
        """If pod re-sends the same envelope (e.g. after a WS reconnect
        with overlapping ``since``), the emitter's dedup drops the
        second copy — the journal does not grow on retry."""
        emitter = FakeEventEmitter()
        monitor = _make_monitor(emitter=emitter)

        first = _make_started_envelope(offset=0)
        # Same wire dict — same (run_id, source, offset) tuple, so the
        # FakeEventEmitter (mirroring real ``ControlEventEmitter`` dedup)
        # will drop the second copy.
        duplicate = json.loads(json.dumps(first))

        wire_events = [
            first,
            duplicate,
            {
                "offset": 1,
                "kind": "trainer_exited",
                "payload": {
                    "exit_code": 0,
                    "signal": None,
                    "cancellation_requested": False,
                },
            },
        ]
        client = _make_client(wire_events)
        result = monitor.execute(_ctx_with_handles(client))
        assert result["status"] == "completed"

        assert len(emitter.received_remote) == 1
        assert emitter.dropped_remote == 1

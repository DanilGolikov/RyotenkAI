"""Phase 2 — :class:`PodEventEmitter` contract.

Coverage matrix:

* TestPositive       — emit/emit_remote happy path through the bus.
* TestNegative       — emit_remote preserves caller-supplied identity.
* TestBoundary       — concurrent emit threads cannot collide on offset.
* TestInvariants     — emit() never raises even when the bus rejects.
* TestDependencyErrors — bus.publish failure is swallowed + logged.
* TestRegressions    — runtime ``isinstance`` check against
                        :class:`IEventEmitter` Protocol.
* TestLogicSpecific  — :meth:`stage_scope` uses ContextVar (per-task
                        isolation).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ryotenkai_pod.runner.event_bus import EventBus
from ryotenkai_pod.runner.event_emitter import PodEventEmitter
from ryotenkai_shared.events import UNKNOWN_OFFSET, IEventEmitter
from ryotenkai_shared.events.types.pod_lifecycle import (
    StopRequestedEvent,
    StopRequestedPayload,
)


def _make_event(*, source: str = "pod://test/runner", offset: int = UNKNOWN_OFFSET) -> StopRequestedEvent:
    return StopRequestedEvent(
        source=source,
        run_id="test",
        offset=offset,
        payload=StopRequestedPayload(grace_seconds=30.0),
    )


class TestPositive:
    def test_emit_publishes_to_bus(self) -> None:
        bus = EventBus(capacity=10)
        emitter = PodEventEmitter(bus, source="pod://test/runner")
        emitter.emit(_make_event())
        assert bus.next_offset == 1

    def test_emit_remote_publishes_to_bus(self) -> None:
        bus = EventBus(capacity=10)
        emitter = PodEventEmitter(bus, source="pod://test/runner")
        ev = _make_event(offset=42)
        emitter.emit_remote(ev)
        stored = next(iter(list(bus.iter_buffered_envelopes())))
        # Caller-supplied offset is preserved.
        assert stored.offset == 42


class TestNegative:
    def test_emit_remote_keeps_event_id_and_time(self) -> None:
        bus = EventBus(capacity=10)
        emitter = PodEventEmitter(bus, source="pod://test/runner")
        ev = _make_event(offset=5)
        original_id = ev.event_id
        original_time = ev.time
        emitter.emit_remote(ev)
        stored = next(iter(list(bus.iter_buffered_envelopes())))
        assert stored.event_id == original_id
        assert stored.time == original_time


class TestBoundary:
    def test_concurrent_emits_get_distinct_offsets(self) -> None:
        bus = EventBus(capacity=1000)
        emitter = PodEventEmitter(bus, source="pod://test/runner")
        # 50 threads × 10 emits each — without the bus' threading.Lock
        # we'd see collisions.
        N = 50

        def _spam() -> None:
            for _ in range(10):
                emitter.emit(_make_event())

        with ThreadPoolExecutor(max_workers=N) as exe:
            for _ in range(N):
                exe.submit(_spam)
        # Each emit assigned a fresh offset.
        offsets = sorted(e.offset for e in bus.iter_buffered_envelopes())
        assert offsets == list(range(N * 10))


class TestInvariants:
    def test_offsets_are_monotonic(self) -> None:
        bus = EventBus(capacity=10)
        emitter = PodEventEmitter(bus, source="pod://test/runner")
        for _ in range(5):
            emitter.emit(_make_event())
        offsets = [e.offset for e in bus.iter_buffered_envelopes()]
        assert offsets == sorted(offsets)
        assert offsets == [0, 1, 2, 3, 4]


class TestDependencyErrors:
    def test_emit_never_raises_when_bus_publish_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bus = EventBus(capacity=10)
        emitter = PodEventEmitter(bus, source="pod://test/runner")

        def _broken(event):  # noqa: ANN001
            raise RuntimeError("simulated bus failure")

        monkeypatch.setattr(bus, "publish", _broken)
        # Must not raise even when the bus is broken.
        emitter.emit(_make_event())


class TestRegressions:
    def test_satisfies_runtime_protocol(self) -> None:
        bus = EventBus(capacity=10)
        emitter = PodEventEmitter(bus, source="pod://test/runner")
        assert isinstance(emitter, IEventEmitter)


class TestLogicSpecific:
    def test_stage_scope_is_context_var(self) -> None:
        bus = EventBus(capacity=10)
        emitter = PodEventEmitter(bus, source="pod://test/runner")
        with emitter.stage_scope("outer"):
            with emitter.stage_scope("inner"):
                emitter.emit(_make_event())
        stored = next(iter(list(bus.iter_buffered_envelopes())))
        assert stored.stage_id == "inner"

    def test_stage_scope_unaffected_after_exit(self) -> None:
        bus = EventBus(capacity=10)
        emitter = PodEventEmitter(bus, source="pod://test/runner")
        with emitter.stage_scope("temporary"):
            pass
        emitter.emit(_make_event())
        stored = next(iter(list(bus.iter_buffered_envelopes())))
        assert stored.stage_id is None


_ = threading

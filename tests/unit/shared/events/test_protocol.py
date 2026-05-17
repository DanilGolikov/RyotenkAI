"""Unit tests: :mod:`ryotenkai_shared.events.protocol`.

The :class:`IEventEmitter` Protocol is the boundary every concrete
emitter must satisfy. Mocking it is forbidden by sentinel
:mod:`tests._lint.test_no_protocol_mocking`; this file exists to ensure
the canonical fake :class:`FakeEventEmitter` is the right substitute.
"""

from __future__ import annotations

from typing import Literal

import pytest

from ryotenkai_shared.events.envelope import BaseEvent
from ryotenkai_shared.events.protocol import IEventEmitter
from ryotenkai_shared.events.severity import Severity  # noqa: TC001 — Pydantic field type
from tests._fakes.event_emitter import FakeEventEmitter


class _ScopedEvent(BaseEvent):
    kind: Literal["test.scoped"] = "test.scoped"
    severity: Severity = "info"


def _make_event(**overrides) -> _ScopedEvent:
    defaults = {"source": "control://orchestrator", "run_id": "run-1", "offset": 0}
    defaults.update(overrides)
    return _ScopedEvent(**defaults)


# ===========================================================================
# 1. Positive
# ===========================================================================


class TestPositive:
    def test_fake_satisfies_runtime_checkable_protocol(self) -> None:
        emitter = FakeEventEmitter()
        assert isinstance(emitter, IEventEmitter)

    def test_emit_appends_event_to_emitted_list(self) -> None:
        emitter = FakeEventEmitter()
        event = _make_event()
        emitter.emit(event)
        assert len(emitter.emitted) == 1
        assert emitter.emitted[0].kind == "test.scoped"

    def test_emit_remote_appends_when_no_dedup(self) -> None:
        emitter = FakeEventEmitter()
        event = _make_event(offset=5)
        emitter.emit_remote(event)
        assert emitter.received_remote == [event]


# ===========================================================================
# 2. Negative
# ===========================================================================


class TestNegative:
    def test_emit_with_chaos_failure_drops_silently(self) -> None:
        emitter = FakeEventEmitter()
        emitter.inject_emit_failure(2)
        emitter.emit(_make_event())
        emitter.emit(_make_event())
        # Both dropped under the chaos counter.
        assert emitter.emitted == []
        emitter.emit(_make_event())
        # Third call goes through.
        assert len(emitter.emitted) == 1

    def test_emit_remote_with_validation_failure_increments_invalid(self) -> None:
        emitter = FakeEventEmitter()
        emitter.inject_validation_failure(1)
        emitter.emit_remote(_make_event(offset=10))
        assert emitter.received_remote == []
        assert emitter.invalid_remote == 1


# ===========================================================================
# 3. Invariants
# ===========================================================================


class TestInvariants:
    def test_emit_assigns_monotonic_offsets_per_source(self) -> None:
        emitter = FakeEventEmitter()
        a = _make_event(source="control://orchestrator")
        b = _make_event(source="control://orchestrator")
        c = _make_event(source="pod://run/trainer")
        emitter.emit(a)
        emitter.emit(b)
        emitter.emit(c)
        offsets = [e.offset for e in emitter.emitted]
        assert offsets == [0, 1, 0]  # per-source counter

    def test_stage_scope_fills_stage_id_for_emitted_events(self) -> None:
        emitter = FakeEventEmitter()
        with emitter.stage_scope("dataset_validator"):
            emitter.emit(_make_event())
        assert emitter.emitted[0].stage_id == "dataset_validator"

    def test_stage_scope_does_not_leak_after_exit(self) -> None:
        emitter = FakeEventEmitter()
        with emitter.stage_scope("a"):
            pass
        emitter.emit(_make_event())
        assert emitter.emitted[0].stage_id is None

    def test_emit_remote_preserves_identity_fields_verbatim(self) -> None:
        emitter = FakeEventEmitter()
        event = _make_event(source="pod://r/trainer", offset=7)
        emitter.emit_remote(event)
        assert emitter.received_remote[0].offset == 7
        assert emitter.received_remote[0].source == "pod://r/trainer"

    def test_emit_remote_dedups_silently(self) -> None:
        emitter = FakeEventEmitter()
        event = _make_event(source="pod://r/trainer", offset=7)
        emitter.emit_remote(event)
        emitter.emit_remote(event)  # duplicate (run_id, source, offset)
        assert len(emitter.received_remote) == 1
        assert emitter.dropped_remote == 1


# ===========================================================================
# 4. Dependency errors
# ===========================================================================


class TestDependencyErrors:
    def test_inject_emit_failure_rejects_negative_n(self) -> None:
        emitter = FakeEventEmitter()
        with pytest.raises(ValueError):
            emitter.inject_emit_failure(-1)

    def test_inject_validation_failure_rejects_negative_n(self) -> None:
        emitter = FakeEventEmitter()
        with pytest.raises(ValueError):
            emitter.inject_validation_failure(-1)


# ===========================================================================
# 5. Logic-specific
# ===========================================================================


class TestLogicSpecific:
    def test_nested_stage_scopes_override_outer(self) -> None:
        emitter = FakeEventEmitter()
        with emitter.stage_scope("outer"), emitter.stage_scope("inner"):
            emitter.emit(_make_event())
        assert emitter.emitted[0].stage_id == "inner"

    def test_clear_resets_all_state(self) -> None:
        emitter = FakeEventEmitter()
        emitter.emit(_make_event())
        emitter.emit_remote(_make_event(offset=42))
        emitter.inject_emit_failure(1)
        emitter.clear()
        assert emitter.emitted == []
        assert emitter.received_remote == []
        assert emitter._emit_failures_remaining == 0

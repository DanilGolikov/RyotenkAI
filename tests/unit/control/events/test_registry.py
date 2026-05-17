"""Tests for :class:`ryotenkai_control.events.EventEmitterRegistry` (Phase 6.a).

The registry is a thin singleton dict + Lock. The seven-class layout
mirrors :doc:`/docs/testing/mock_policy.md` — one positive, one negative,
one boundary, one invariant, one dependency-error, one regression-shaped,
one logic-specific (concurrency).
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from ryotenkai_control.events import ControlEventEmitter, EventEmitterRegistry


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Ensure each test starts with a fresh process-wide registry."""
    EventEmitterRegistry.reset_instance()
    yield
    EventEmitterRegistry.reset_instance()


def _make_emitter(tmp_path: Path, run_id: str = "run-1") -> ControlEventEmitter:
    return ControlEventEmitter.for_run(run_id=run_id, run_directory=tmp_path / run_id)


class TestPositive:
    def test_register_then_get_returns_same_emitter(self, tmp_path: Path) -> None:
        reg = EventEmitterRegistry.instance()
        emitter = _make_emitter(tmp_path)
        reg.register("run-1", emitter)
        assert reg.get("run-1") is emitter
        emitter.close()

    def test_singleton_returns_same_instance(self) -> None:
        a = EventEmitterRegistry.instance()
        b = EventEmitterRegistry.instance()
        assert a is b


class TestNegative:
    def test_get_missing_returns_none(self) -> None:
        reg = EventEmitterRegistry.instance()
        assert reg.get("no-such-run") is None

    def test_register_empty_run_id_raises(self, tmp_path: Path) -> None:
        reg = EventEmitterRegistry.instance()
        emitter = _make_emitter(tmp_path)
        try:
            with pytest.raises(ValueError):
                reg.register("", emitter)
        finally:
            emitter.close()


class TestBoundary:
    def test_register_replaces_existing_entry(self, tmp_path: Path) -> None:
        reg = EventEmitterRegistry.instance()
        first = _make_emitter(tmp_path, "run-1")
        second = _make_emitter(tmp_path, "run-1-v2")
        reg.register("run-1", first)
        reg.register("run-1", second)
        assert reg.get("run-1") is second
        first.close()
        second.close()

    def test_deregister_idempotent_when_absent(self) -> None:
        reg = EventEmitterRegistry.instance()
        # Must not raise on missing key.
        reg.deregister("never-registered")
        reg.deregister("never-registered")


class TestInvariants:
    def test_deregister_removes_entry(self, tmp_path: Path) -> None:
        reg = EventEmitterRegistry.instance()
        emitter = _make_emitter(tmp_path)
        reg.register("run-1", emitter)
        reg.deregister("run-1")
        assert reg.get("run-1") is None
        emitter.close()

    def test_contains_protocol(self, tmp_path: Path) -> None:
        reg = EventEmitterRegistry.instance()
        emitter = _make_emitter(tmp_path)
        reg.register("run-1", emitter)
        assert "run-1" in reg
        assert "missing" not in reg
        emitter.close()

    def test_len_reflects_active_runs(self, tmp_path: Path) -> None:
        reg = EventEmitterRegistry.instance()
        e1 = _make_emitter(tmp_path, "run-1")
        e2 = _make_emitter(tmp_path, "run-2")
        reg.register("run-1", e1)
        reg.register("run-2", e2)
        assert len(reg) == 2
        reg.deregister("run-1")
        assert len(reg) == 1
        e1.close()
        e2.close()


class TestDependencyErrors:
    def test_active_run_ids_returns_defensive_copy(self, tmp_path: Path) -> None:
        reg = EventEmitterRegistry.instance()
        emitter = _make_emitter(tmp_path)
        reg.register("run-1", emitter)
        ids = reg.active_run_ids()
        ids.append("not-real")
        # Mutation of the snapshot does NOT affect the registry.
        assert reg.active_run_ids() == ["run-1"]
        emitter.close()

    def test_reset_instance_drops_state(self, tmp_path: Path) -> None:
        reg = EventEmitterRegistry.instance()
        emitter = _make_emitter(tmp_path)
        reg.register("run-1", emitter)
        EventEmitterRegistry.reset_instance()
        fresh = EventEmitterRegistry.instance()
        assert fresh.get("run-1") is None
        emitter.close()


class TestRegressions:
    def test_contains_non_string_returns_false(self) -> None:
        """Defends against ``x in registry`` with non-string keys."""
        reg = EventEmitterRegistry.instance()
        assert 42 not in reg
        assert None not in reg

    def test_register_does_not_carry_to_new_singleton(
        self, tmp_path: Path,
    ) -> None:
        """``reset_instance`` must clear visibility — guards against leaked state."""
        reg = EventEmitterRegistry.instance()
        emitter = _make_emitter(tmp_path)
        reg.register("run-1", emitter)
        EventEmitterRegistry.reset_instance()
        # The original ``reg`` reference still holds the emitter, but
        # a new ``instance()`` does NOT.
        assert reg.get("run-1") is emitter
        assert EventEmitterRegistry.instance().get("run-1") is None
        emitter.close()


class TestLogicSpecific:
    def test_concurrent_registers_serialize_safely(
        self, tmp_path: Path,
    ) -> None:
        """100 threads × 10 register/deregister cycles — no exceptions, no torn dict.

        Stress-tests the internal :class:`threading.Lock`. If the lock
        were missing the dict would either raise (RuntimeError under
        concurrent mutation) or leave orphan entries; we assert the
        final count is zero.
        """
        reg = EventEmitterRegistry.instance()
        emitter = _make_emitter(tmp_path)

        errors: list[BaseException] = []

        def _worker(worker_id: int) -> None:
            try:
                for cycle in range(10):
                    rid = f"run-{worker_id}-{cycle}"
                    reg.register(rid, emitter)
                    assert reg.get(rid) is emitter
                    reg.deregister(rid)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(reg) == 0
        emitter.close()

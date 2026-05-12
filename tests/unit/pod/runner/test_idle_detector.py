"""Phase 4.1 — :class:`IdleDetector` contract.

The detector replaces a 165-LOC bash script (``watchdog.sh``) with
the same threshold semantics — so the test matrix mirrors the bash
script's branches:

- TestStartupGrace      no triggers during the first STARTUP_GRACE seconds
- TestGPUIdleTrigger    sustained idle past IDLE_THRESHOLD → request_stop
- TestGPUResumes        GPU activity resets the idle timer
- TestMaxLifetime       hard kill switch fires regardless of GPU
- TestSupervisorGone    FSM terminal → loop exits cleanly
- TestProviderFailure   ``None`` from provider treated as "not idle"
- TestLifecycle         start/stop, idempotency, restart-after-stop

We use a :class:`TickClock` that doubles as the clock and the
sleep stub: every ``await tick_clock.sleep(...)`` advances the
virtual time AND bumps an iteration counter. Tests then drive the
detector "until it has done N polls" via :func:`_run_until_ticks`,
which yields control until the counter reaches the target. This
avoids the trap of "yield N times in the test" — it makes the
test robust to changes in the detector's internal await topology.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

import pytest

from ryotenkai_pod.runner.event_bus import EventBus
from ryotenkai_pod.runner.idle_detector import (
    DEFAULT_GPU_MEM_MAX_PCT,
    DEFAULT_GPU_UTIL_MAX,
    GPUMetrics,
    IdleDetector,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class TickClock:
    """Clock + sleep stub that counts polling iterations.

    Combining the roles means a test can drive the detector "until N
    polls have completed" without guessing how many event-loop yields
    that takes — the detector itself bumps :attr:`ticks` every time
    it calls :meth:`sleep`. Slight changes to the detector's await
    topology won't silently break tests this way.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.ticks = 0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        self.ticks += 1
        # One real yield so the test body can observe the increment
        # and feed in the next batch of side effects (advance metrics,
        # etc.).
        await asyncio.sleep(0)


class FakeSupervisor:
    """Stub of :class:`Supervisor` for the detector to talk to.

    The detector only touches:
    - ``is_running`` (bool property)
    - ``request_stop(grace_seconds=…)`` (async, called once on trigger)
    """

    def __init__(self) -> None:
        self.is_running = True
        self.stop_calls: list[float] = []

    async def request_stop(self, *, grace_seconds: float) -> None:
        self.stop_calls.append(grace_seconds)
        # Mimic the production supervisor: after a stop request the
        # FSM eventually transitions to a terminal state, the
        # subprocess exits, and ``is_running`` flips False.
        self.is_running = False


def _scripted_provider(values: Iterable[GPUMetrics]):
    """Return a metrics provider that yields values in order, then
    sticks on the last value (or ``None`` after exhaustion)."""
    values = list(values)

    async def _read() -> GPUMetrics:
        if not values:
            return None
        if len(values) > 1:
            return values.pop(0)
        return values[0]

    return _read


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def bus() -> EventBus:
    return EventBus(capacity=100)


@pytest.fixture
def supervisor() -> FakeSupervisor:
    return FakeSupervisor()


@pytest.fixture
def tick_clock() -> TickClock:
    return TickClock()


def _build_detector(
    bus: EventBus,
    supervisor: FakeSupervisor,
    tick_clock: TickClock,
    *,
    metrics: Iterable[GPUMetrics] | _AsyncProvider,
    startup_grace: float = 2.0,
    idle_threshold: float = 5.0,
    max_lifetime: float = 10_000.0,
    poll_interval: float = 1.0,
) -> IdleDetector:
    """One-liner constructor used across the test file."""
    if callable(metrics):
        provider = metrics  # passed as already-an-async-callable
    else:
        provider = _scripted_provider(metrics)
    return IdleDetector(
        supervisor=supervisor,  # type: ignore[arg-type]
        bus=bus,
        startup_grace=startup_grace,
        idle_threshold=idle_threshold,
        max_lifetime=max_lifetime,
        poll_interval=poll_interval,
        metrics_provider=provider,
        clock=tick_clock,
        sleep=tick_clock.sleep,
    )


_AsyncProvider = type(lambda: None)  # placeholder for typing


async def _run_until_ticks(
    tick_clock: TickClock, *, target: int, max_yields: int = 1000,
) -> None:
    """Yield until the detector has done ``target`` polling iterations
    or the bail-out cap is hit (prevents an infinite test if the
    detector exits earlier than expected)."""
    yields = 0
    while tick_clock.ticks < target and yields < max_yields:
        await asyncio.sleep(0)
        yields += 1


# ---------------------------------------------------------------------------
# Startup grace
# ---------------------------------------------------------------------------


class TestStartupGrace:
    async def test_no_trigger_within_grace(
        self,
        bus: EventBus,
        supervisor: FakeSupervisor,
        tick_clock: TickClock,
    ) -> None:
        # startup_grace=10s, poll_interval=1s, idle_threshold=5s.
        # Run for 5 polls → uptime=5s, still inside grace.
        detector = _build_detector(
            bus, supervisor, tick_clock,
            metrics=[(0, 0)],  # always idle
            startup_grace=10.0,
            idle_threshold=5.0,
            poll_interval=1.0,
        )
        detector.start()
        await _run_until_ticks(tick_clock, target=5)
        await detector.stop()
        assert supervisor.stop_calls == []
        kinds = [e.kind for e in list(bus._buffer)]
        assert "idle_detector_triggered" not in kinds


# ---------------------------------------------------------------------------
# GPU-idle trigger
# ---------------------------------------------------------------------------


class TestGPUIdleTrigger:
    async def test_sustained_idle_triggers_stop(
        self,
        bus: EventBus,
        supervisor: FakeSupervisor,
        tick_clock: TickClock,
    ) -> None:
        # poll_interval=1s, startup_grace=2s, idle_threshold=5s.
        # 3 ticks past grace → idle_since=3, then idle for 5 more
        # ticks → trigger at uptime=10s.
        detector = _build_detector(
            bus, supervisor, tick_clock,
            metrics=[(0, 0)],
            startup_grace=2.0,
            idle_threshold=5.0,
            poll_interval=1.0,
        )
        detector.start()
        await _run_until_ticks(tick_clock, target=12)
        await detector.stop()
        assert len(supervisor.stop_calls) == 1
        assert "idle_detector_triggered" in [e.kind for e in list(bus._buffer)]

    async def test_gpu_busy_no_trigger(
        self,
        bus: EventBus,
        supervisor: FakeSupervisor,
        tick_clock: TickClock,
    ) -> None:
        detector = _build_detector(
            bus, supervisor, tick_clock,
            metrics=[(95, 80)],  # GPU under heavy load
            startup_grace=1.0,
            idle_threshold=5.0,
            poll_interval=1.0,
        )
        detector.start()
        await _run_until_ticks(tick_clock, target=10)
        await detector.stop()
        assert supervisor.stop_calls == []


# ---------------------------------------------------------------------------
# GPU resumes — idle timer resets
# ---------------------------------------------------------------------------


class TestGPUResumes:
    async def test_idle_timer_resets_when_busy(
        self,
        bus: EventBus,
        supervisor: FakeSupervisor,
        tick_clock: TickClock,
    ) -> None:
        # Idle for 3 ticks → busy 1 tick → idle 3 ticks (sticks on
        # last). With idle_threshold=5, the second idle stretch (3
        # ticks) does NOT trigger.
        detector = _build_detector(
            bus, supervisor, tick_clock,
            metrics=[(0, 0), (0, 0), (0, 0), (50, 50), (0, 0), (0, 0), (0, 0)],
            startup_grace=0.0,
            idle_threshold=5.0,
            poll_interval=1.0,
        )
        detector.start()
        await _run_until_ticks(tick_clock, target=7)
        await detector.stop()
        assert supervisor.stop_calls == []
        kinds = [e.kind for e in list(bus._buffer)]
        assert "gpu_idle_started" in kinds
        assert "gpu_idle_cleared" in kinds


# ---------------------------------------------------------------------------
# Max-lifetime kill switch
# ---------------------------------------------------------------------------


class TestMaxLifetime:
    async def test_max_lifetime_triggers_regardless_of_gpu(
        self,
        bus: EventBus,
        supervisor: FakeSupervisor,
        tick_clock: TickClock,
    ) -> None:
        detector = _build_detector(
            bus, supervisor, tick_clock,
            metrics=[(99, 99)],  # busy
            startup_grace=10_000.0,  # never reach GPU-idle path
            idle_threshold=10_000.0,
            max_lifetime=3.0,  # very short for testing
            poll_interval=1.0,
        )
        detector.start()
        await _run_until_ticks(tick_clock, target=5)
        await detector.stop()
        assert len(supervisor.stop_calls) == 1
        triggered = next(
            e for e in list(bus._buffer) if e.kind == "idle_detector_triggered"
        )
        assert triggered.payload["reason"] == "max_lifetime"


# ---------------------------------------------------------------------------
# Supervisor reaped externally
# ---------------------------------------------------------------------------


class TestSupervisorGone:
    async def test_loop_exits_when_supervisor_done(
        self,
        bus: EventBus,
        supervisor: FakeSupervisor,
        tick_clock: TickClock,
    ) -> None:
        # Trainer completes naturally (FSM terminal) — detector loop
        # should exit cleanly without any trigger.
        supervisor.is_running = False
        detector = _build_detector(
            bus, supervisor, tick_clock,
            metrics=[(0, 0)],
            startup_grace=0.0,
            idle_threshold=1.0,
            poll_interval=0.1,
        )
        detector.start()
        await _run_until_ticks(tick_clock, target=1)
        # The internal task should have returned.
        assert detector._task is not None
        await asyncio.wait_for(detector._task, timeout=1.0)
        assert supervisor.stop_calls == []


# ---------------------------------------------------------------------------
# Provider failure handling
# ---------------------------------------------------------------------------


class TestProviderFailure:
    async def test_none_metrics_treated_as_not_idle(
        self,
        bus: EventBus,
        supervisor: FakeSupervisor,
        tick_clock: TickClock,
    ) -> None:
        detector = _build_detector(
            bus, supervisor, tick_clock,
            metrics=[None],  # nvidia-smi missing
            startup_grace=0.0,
            idle_threshold=2.0,
            poll_interval=1.0,
        )
        detector.start()
        await _run_until_ticks(tick_clock, target=10)
        await detector.stop()
        assert supervisor.stop_calls == []

    async def test_provider_exception_treated_as_not_idle(
        self,
        bus: EventBus,
        supervisor: FakeSupervisor,
        tick_clock: TickClock,
    ) -> None:
        async def _bad() -> GPUMetrics:
            raise RuntimeError("nvidia-smi crashed")

        detector = _build_detector(
            bus, supervisor, tick_clock,
            metrics=_bad,
            startup_grace=0.0,
            idle_threshold=2.0,
            poll_interval=1.0,
        )
        detector.start()
        await _run_until_ticks(tick_clock, target=10)
        await detector.stop()
        assert supervisor.stop_calls == []


# ---------------------------------------------------------------------------
# Lifecycle / idempotency
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_start_idempotent(
        self,
        bus: EventBus,
        supervisor: FakeSupervisor,
        tick_clock: TickClock,
    ) -> None:
        detector = _build_detector(
            bus, supervisor, tick_clock,
            metrics=[(50, 50)],
            poll_interval=0.01,
        )
        detector.start()
        first_task = detector._task
        detector.start()  # no-op
        assert detector._task is first_task
        await detector.stop()

    async def test_stop_safe_when_never_started(
        self,
        bus: EventBus,
        supervisor: FakeSupervisor,
    ) -> None:
        detector = IdleDetector(
            supervisor=supervisor,  # type: ignore[arg-type]
            bus=bus,
        )
        await detector.stop()  # must not raise

    async def test_restart_after_stop(
        self,
        bus: EventBus,
        supervisor: FakeSupervisor,
        tick_clock: TickClock,
    ) -> None:
        detector = _build_detector(
            bus, supervisor, tick_clock,
            metrics=[(50, 50)],
            poll_interval=0.01,
        )
        detector.start()
        await detector.stop()
        detector.start()
        assert detector.is_running
        await detector.stop()


# ---------------------------------------------------------------------------
# Defaults match watchdog.sh
# ---------------------------------------------------------------------------


class TestThresholdParity:
    """Defaults must match watchdog.sh exactly. If watchdog.sh ever
    gets retuned, sync these numbers — both ship together."""

    def test_gpu_thresholds_match_legacy(self) -> None:
        assert DEFAULT_GPU_UTIL_MAX == 5
        assert DEFAULT_GPU_MEM_MAX_PCT == 30

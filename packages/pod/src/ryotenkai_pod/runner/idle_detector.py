"""GPU-idle / max-lifetime watcher — Python replacement for ``watchdog.sh``.

Background
----------
The legacy bash ``watchdog.sh`` was an independent process inside the
pod that polled GPU utilisation and a ``.pipeline_heartbeat`` file
written by the Mac. If the pipeline disappeared (laptop sleep) AND
the GPU stayed idle for too long, watchdog.sh sent a stop request
to the RunPod GraphQL endpoint. Brittle: 165 lines of bash, no
tests, depends on file-system heartbeat invariant.

This module folds the same logic into the Python runner. Now the
in-pod ``Supervisor`` itself decides when the run is dead (no Mac
heartbeat, no GPU activity, exceeded max lifetime) and drives a
graceful stop through the FSM. The actual pod removal / sleep
happens via :mod:`src.runner.pod_terminator` (Phase 11.B decision
matrix; user-initiated stop ⇒ ``podTerminate``, natural completion
⇒ ``podStop`` to preserve ``/workspace`` for resume).

Thresholds (mirror watchdog.sh exactly):

==================== ============================== ====
Constant             Meaning                        Default
==================== ============================== ====
``STARTUP_GRACE``    Seconds the GPU is allowed to  300
                     show "idle" right after spawn
                     (model loading, dataset
                     prefetch — legitimately low-
                     utilisation work).
``IDLE_THRESHOLD``   Continuous GPU-idle window     1200
                     before triggering stop.
``MAX_LIFETIME``     Hard kill switch — run         172800 (48 h)
                     longer than this and we stop
                     regardless of GPU state.
``POLL_INTERVAL``    Seconds between metric polls.  30.0
``GPU_UTIL_MAX``     util% under which a GPU is     5
                     considered idle.
``GPU_MEM_MAX_PCT``  memory% under which a GPU is   30
                     considered idle.
==================== ============================== ====

Both ``util`` and ``mem_pct`` must be below their thresholds to
count as idle — the same belt-and-suspenders condition the bash
script used.

GPU metrics provider:
The detector accepts an injectable metrics callable so tests can
supply deterministic values without a GPU. The default reader tries
``pynvml`` first and falls back to ``nvidia-smi`` over subprocess
when pynvml is missing or NVML is mis-installed (common with
mismatched CUDA). Both paths return the same ``(max_util, max_mem_pct)``
tuple across all visible GPUs, or ``None`` if neither works (the
detector then treats the missing reading as "not idle" so we never
SIGTERM a working trainer because of monitoring failure).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from time import monotonic
from typing import TYPE_CHECKING

from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.events.types.pod_health import (
    HealthIdleDetectedEvent,
    HealthIdleDetectedPayload,
    HealthMaxLifetimeExceededEvent,
    HealthMaxLifetimeExceededPayload,
)

if TYPE_CHECKING:
    from ryotenkai_pod.runner.event_bus import EventBus
    from ryotenkai_pod.runner.supervisor import Supervisor


__all__ = [
    "DEFAULT_GPU_MEM_MAX_PCT",
    "DEFAULT_GPU_UTIL_MAX",
    "DEFAULT_GRACE_BEFORE_STOP",
    "DEFAULT_IDLE_THRESHOLD",
    "DEFAULT_MAX_LIFETIME",
    "DEFAULT_POLL_INTERVAL",
    "DEFAULT_STARTUP_GRACE",
    "ENV_IDLE_THRESHOLD_MINUTES",
    "ENV_MAX_LIFETIME_HOURS",
    "GPUMetricsProvider",
    "IdleDetector",
    "resolve_thresholds_from_env",
]


DEFAULT_STARTUP_GRACE = 300.0  # 5 min
DEFAULT_IDLE_THRESHOLD = 1200.0  # 20 min
DEFAULT_MAX_LIFETIME = 172_800.0  # 48 h
DEFAULT_POLL_INTERVAL = 30.0
DEFAULT_GPU_UTIL_MAX = 5  # percent
DEFAULT_GPU_MEM_MAX_PCT = 30  # percent
DEFAULT_GRACE_BEFORE_STOP = 30.0  # SIGTERM grace passed to supervisor.request_stop


# Env vars the Mac-side training launcher sets so the pod's IdleDetector
# inherits the user-configured ``pod_lifecycle`` thresholds. Absent /
# unparseable values fall back to the in-pod defaults above (E-СРЕД fix).
ENV_MAX_LIFETIME_HOURS = "RYOTENKAI_POD_MAX_LIFETIME_HOURS"
ENV_IDLE_THRESHOLD_MINUTES = "RYOTENKAI_POD_IDLE_THRESHOLD_MINUTES"


def resolve_thresholds_from_env(
    env: dict[str, str] | None = None,
) -> tuple[float, float]:
    """Read ``(max_lifetime_seconds, idle_threshold_seconds)`` from env.

    Used by the runner's lifespan when constructing the
    :class:`IdleDetector`. Each env var is parsed independently:
    bad / missing values fall through to the in-pod default. ``env``
    defaults to :data:`os.environ`; tests pass a tailored mapping.
    """
    source = dict(env) if env is not None else dict(os.environ)

    def _float_or(default: float, *, name: str, scale: float) -> float:
        raw = source.get(name, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        if value <= 0:
            return default
        return value * scale

    max_lifetime_s = _float_or(
        DEFAULT_MAX_LIFETIME, name=ENV_MAX_LIFETIME_HOURS, scale=3600.0,
    )
    idle_threshold_s = _float_or(
        DEFAULT_IDLE_THRESHOLD, name=ENV_IDLE_THRESHOLD_MINUTES, scale=60.0,
    )
    return max_lifetime_s, idle_threshold_s


# ``None`` means "couldn't read metrics" — treat as "not idle".
GPUMetrics = tuple[int, int] | None
GPUMetricsProvider = Callable[[], Awaitable[GPUMetrics]]


# ---------------------------------------------------------------------------
# Default metrics readers
# ---------------------------------------------------------------------------


async def _read_via_pynvml() -> GPUMetrics:
    """Fast path — pynvml direct. ``None`` on any error so caller can fall back."""
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        max_util = 0
        max_mem_pct = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util_obj = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_obj = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = int(util_obj.gpu)
            mem_total = int(mem_obj.total)
            mem_used = int(mem_obj.used)
            mem_pct = (mem_used * 100 // mem_total) if mem_total > 0 else 0
            max_util = max(max_util, util)
            max_mem_pct = max(max_mem_pct, mem_pct)
        return max_util, max_mem_pct
    except Exception:
        return None
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


async def _read_via_smi() -> GPUMetrics:
    """Slow-path fallback — invokes ``nvidia-smi``. ``None`` if missing."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
    except TimeoutError:
        with contextlib.suppress(Exception):
            proc.kill()
        return None

    if proc.returncode != 0:
        return None

    max_util = 0
    max_mem_pct = 0
    for raw in stdout.decode("utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 3:
            continue
        try:
            util = int(parts[0])
            mem_used = int(parts[1])
            mem_total = int(parts[2])
        except ValueError:
            continue
        if mem_total <= 0:
            continue
        mem_pct = mem_used * 100 // mem_total
        max_util = max(max_util, util)
        max_mem_pct = max(max_mem_pct, mem_pct)
    return max_util, max_mem_pct


async def default_gpu_metrics() -> GPUMetrics:
    """Production reader: pynvml → nvidia-smi → ``None``."""
    result = await _read_via_pynvml()
    if result is not None:
        return result
    return await _read_via_smi()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class IdleDetector:
    """Async polling watcher for GPU idle and max-lifetime breaches.

    Construct once per ``Supervisor`` (a 1:1 binding); call
    :meth:`start` once the supervisor has spawned a trainer; call
    :meth:`stop` from the lifespan exit. Re-startable: after the
    detector triggers a stop and exits, calling ``start()`` again
    spins up a fresh polling loop with a reset clock.

    Thresholds (``startup_grace`` / ``idle_threshold`` /
    ``max_lifetime`` / ``poll_interval`` / ``gpu_util_max`` /
    ``gpu_mem_max_pct``) are constructor parameters — the defaults
    mirror watchdog.sh exactly. Tests pass aggressive overrides
    (e.g. ``poll_interval=0.01``) plus deterministic clocks and
    metrics providers.
    """

    def __init__(
        self,
        supervisor: "Supervisor",
        bus: "EventBus",
        *,
        startup_grace: float = DEFAULT_STARTUP_GRACE,
        idle_threshold: float = DEFAULT_IDLE_THRESHOLD,
        max_lifetime: float = DEFAULT_MAX_LIFETIME,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        gpu_util_max: int = DEFAULT_GPU_UTIL_MAX,
        gpu_mem_max_pct: int = DEFAULT_GPU_MEM_MAX_PCT,
        grace_before_stop: float = DEFAULT_GRACE_BEFORE_STOP,
        metrics_provider: GPUMetricsProvider | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._supervisor = supervisor
        self._bus = bus
        self._startup_grace = startup_grace
        self._idle_threshold = idle_threshold
        self._max_lifetime = max_lifetime
        self._poll_interval = poll_interval
        self._gpu_util_max = gpu_util_max
        self._gpu_mem_max_pct = gpu_mem_max_pct
        self._grace_before_stop = grace_before_stop
        self._metrics_provider: GPUMetricsProvider = metrics_provider or default_gpu_metrics
        self._clock = clock or monotonic
        # Injecting ``sleep`` lets tests advance time deterministically:
        # production runs use ``asyncio.sleep`` (real wall-clock), tests
        # supply a stub that yields once and returns immediately so a
        # test can simulate hours of polling in milliseconds.
        self._sleep: Callable[[float], Awaitable[None]] = sleep or asyncio.sleep

        self._task: asyncio.Task[None] | None = None
        self._started_at: float | None = None
        # Wall-clock start time for the typed max-lifetime event.
        # ``_started_at`` uses the injectable monotonic ``_clock`` for
        # uptime math (tests advance it deterministically); this field
        # records the real datetime so consumers see when the trainer
        # actually began.
        self._started_wall_clock_at: datetime | None = None

    # --- read-only accessors ----------------------------------------

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    # --- lifecycle ---------------------------------------------------

    def start(self) -> None:
        """Spawn the polling task. Idempotent — a second call while
        running is a no-op."""
        if self.is_running:
            return
        self._started_at = self._clock()
        self._started_wall_clock_at = datetime.now(UTC)
        self._task = asyncio.create_task(self._loop(), name="idle_detector.loop")

    async def stop(self) -> None:
        """Cancel the polling task; safe even if never started."""
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._started_at = None
        self._started_wall_clock_at = None

    # --- internals ---------------------------------------------------

    async def _loop(self) -> None:
        idle_since: float | None = None

        while True:
            try:
                await self._sleep(self._poll_interval)
            except asyncio.CancelledError:
                return

            # If the supervisor already reaped the trainer (FSM in
            # terminal state), our work is done — no point hammering
            # nvidia-smi any longer.
            if not self._supervisor.is_running:
                return

            assert self._started_at is not None
            uptime = self._clock() - self._started_at

            # Hard kill switch — no metrics needed, just the clock.
            if uptime >= self._max_lifetime:
                await self._trigger(
                    reason="max_lifetime",
                    payload={"uptime_seconds": uptime},
                )
                return

            # GPU idle detection only fires AFTER the startup grace —
            # gives bitsandbytes / model load time without false-
            # alarming.
            if uptime < self._startup_grace:
                continue

            metrics = await self._safely_read_metrics()
            if metrics is None:
                # Unreadable metrics → assume not idle. Better a
                # leaked trainer than a spuriously cancelled run
                # because nvidia-smi was momentarily wedged.
                idle_since = None
                continue

            util, mem_pct = metrics
            is_idle = (
                util < self._gpu_util_max and mem_pct < self._gpu_mem_max_pct
            )

            if is_idle:
                if idle_since is None:
                    idle_since = self._clock()
                    self._bus.publish_legacy(
                        "gpu_idle_started",
                        {"util": util, "mem_pct": mem_pct},
                        source="pod://runner/idle_detector",
                    )
                elif self._clock() - idle_since >= self._idle_threshold:
                    await self._trigger(
                        reason="gpu_idle",
                        payload={
                            "util": util,
                            "mem_pct": mem_pct,
                            "idle_seconds": self._clock() - idle_since,
                        },
                    )
                    return
            elif idle_since is not None:
                # GPU work resumed — clear the idle timer.
                self._bus.publish_legacy(
                    "gpu_idle_cleared",
                    {"util": util, "mem_pct": mem_pct},
                    source="pod://runner/idle_detector",
                )
                idle_since = None

    async def _safely_read_metrics(self) -> GPUMetrics:
        """Wrap the provider so a programming error in pynvml /
        a nvidia-smi crash doesn't kill the loop."""
        try:
            return await self._metrics_provider()
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    async def _trigger(self, *, reason: str, payload: dict[str, object]) -> None:
        """Publish a structured event and ask the supervisor to stop.

        Emits the typed :class:`HealthIdleDetectedEvent` for the
        ``gpu_idle`` reason. The ``max_lifetime`` reason now emits the
        typed :class:`HealthMaxLifetimeExceededEvent` (E-СРЕД fix —
        replaces the legacy free-form payload). The legacy
        ``publish_legacy`` shim is kept for backward-compat consumers
        until a dedicated migration phase removes it.
        """
        if reason == "gpu_idle":
            raw_idle = payload.get("idle_seconds")
            idle_seconds = float(raw_idle) if isinstance(raw_idle, (int, float)) else 0.0
            self._bus.publish(
                HealthIdleDetectedEvent(
                    source="pod://runner/idle_detector",
                    run_id="unknown",
                    offset=UNKNOWN_OFFSET,
                    payload=HealthIdleDetectedPayload(
                        idle_duration_s=idle_seconds,
                        last_activity_at=datetime.now(UTC),
                    ),
                ),
            )
        elif reason == "max_lifetime":
            raw_uptime = payload.get("uptime_seconds")
            uptime_seconds = (
                float(raw_uptime)
                if isinstance(raw_uptime, (int, float))
                else 0.0
            )
            started_at = self._started_wall_clock_at or datetime.now(UTC)
            self._bus.publish(
                HealthMaxLifetimeExceededEvent(
                    source="pod://runner/idle_detector",
                    run_id="unknown",
                    offset=UNKNOWN_OFFSET,
                    payload=HealthMaxLifetimeExceededPayload(
                        started_at=started_at,
                        max_lifetime_s=float(self._max_lifetime),
                        actual_runtime_s=uptime_seconds,
                    ),
                ),
            )
            # Keep the legacy publish for back-compat with any consumer
            # still watching ``idle_detector_triggered`` (tests + a
            # handful of dashboards). The typed event is authoritative.
            self._bus.publish_legacy(
                "idle_detector_triggered",
                {"reason": reason, **payload},
                source="pod://runner/idle_detector",
            )
        else:
            self._bus.publish_legacy(
                "idle_detector_triggered",
                {"reason": reason, **payload},
                source="pod://runner/idle_detector",
            )
        await self._supervisor.request_stop(grace_seconds=self._grace_before_stop)

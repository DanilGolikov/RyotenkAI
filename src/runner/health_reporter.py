"""Periodic GPU / RAM / CPU snapshot publisher — Phase 4.2.

Sits next to :class:`IdleDetector`, sharing the same metrics
provider concept but used for a different purpose:

- IdleDetector: yes/no decision — is the GPU idle long enough to
  warrant a stop?
- HealthReporter: continuous telemetry — emit a ``health_snapshot``
  event every N seconds so the Mac client / Web UI can render a
  live load chart.

The two could share GPU readings, but keeping them separate avoids
coupling: a slow nvidia-smi shouldn't delay the idle check, and a
broken health metric shouldn't disable idle detection. They are
both cheap (one ``nvidia-smi`` invocation per poll).

CPU / RAM are read with ``psutil`` when available; missing psutil
makes those fields ``None`` rather than crashing — production
images ship psutil, but the dev test environment may not.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from src.runner.idle_detector import default_gpu_metrics

if TYPE_CHECKING:
    from src.runner.event_bus import EventBus

__all__ = [
    "DEFAULT_HEALTH_INTERVAL",
    "HealthReporter",
    "HealthSnapshot",
    "HealthSnapshotProvider",
]


DEFAULT_HEALTH_INTERVAL = 30.0


HealthSnapshot = dict[str, float | int | None]
HealthSnapshotProvider = Callable[[], Awaitable[HealthSnapshot]]


async def _read_psutil() -> tuple[float | None, float | None, float | None]:
    """Return (cpu_percent, ram_used_gb, ram_total_gb) — ``None`` triple
    if psutil is missing."""
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return None, None, None
    try:
        cpu = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        used_gb = vm.used / (1024**3)
        total_gb = vm.total / (1024**3)
        return cpu, used_gb, total_gb
    except Exception:
        return None, None, None


async def default_health_snapshot() -> HealthSnapshot:
    """Production snapshot reader — combines GPU + CPU + RAM."""
    gpu = await default_gpu_metrics()
    cpu_pct, ram_used, ram_total = await _read_psutil()
    return {
        "gpu_util_percent": gpu[0] if gpu else None,
        "gpu_memory_percent": gpu[1] if gpu else None,
        "cpu_percent": cpu_pct,
        "ram_used_gb": ram_used,
        "ram_total_gb": ram_total,
    }


class HealthReporter:
    """Publishes a ``health_snapshot`` event every ``interval`` seconds.

    Constructor is symmetric with :class:`IdleDetector` — same
    injectable ``snapshot_provider`` / ``sleep`` so tests can drive
    it deterministically without a real GPU.
    """

    def __init__(
        self,
        bus: "EventBus",
        *,
        interval: float = DEFAULT_HEALTH_INTERVAL,
        snapshot_provider: HealthSnapshotProvider | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._bus = bus
        self._interval = interval
        self._provider: HealthSnapshotProvider = (
            snapshot_provider or default_health_snapshot
        )
        self._sleep: Callable[[float], Awaitable[None]] = sleep or asyncio.sleep
        self._task: asyncio.Task[None] | None = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        """Spawn the polling task. Idempotent."""
        if self.is_running:
            return
        self._task = asyncio.create_task(self._loop(), name="health_reporter.loop")

    async def stop(self) -> None:
        if self._task is None:
            return
        if not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self._sleep(self._interval)
            except asyncio.CancelledError:
                return
            try:
                snapshot = await self._provider()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Defensive — never crash the reporter loop on a
                # transient psutil / nvidia-smi error.
                continue
            self._bus.publish("health_snapshot", dict(snapshot))

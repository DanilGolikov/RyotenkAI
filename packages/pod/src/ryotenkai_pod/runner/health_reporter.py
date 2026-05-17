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
from pathlib import Path
from typing import TYPE_CHECKING

from ryotenkai_shared.events import UNKNOWN_OFFSET
from ryotenkai_shared.events.types.pod_health import (
    GPUSnapshot,
    HealthSnapshotEvent,
    HealthSnapshotPayload,
)

from ryotenkai_pod.runner.idle_detector import default_gpu_metrics

if TYPE_CHECKING:
    from ryotenkai_pod.runner.event_bus import EventBus

__all__ = [
    "DEFAULT_HEALTH_INTERVAL",
    "HealthReporter",
    "HealthSnapshot",
    "HealthSnapshotProvider",
]


DEFAULT_HEALTH_INTERVAL = 30.0


# Container-aware memory reading. ``psutil.virtual_memory`` reports the
# HOST's RAM inside a container — on a shared GPU host this can be
# 500+ GB even though the container itself has a much smaller cgroup
# limit, so the operator sees nonsense like "RAM: 41.9/504 GB" while
# their pod is allocated 32 GB. Read the cgroup files directly when
# available, fall back to psutil for non-containerised dev environments.
_CGROUP_V2_MEM_MAX = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V2_MEM_CURRENT = Path("/sys/fs/cgroup/memory.current")
_CGROUP_V1_MEM_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_CGROUP_V1_MEM_USAGE = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
#: Cgroup v1 reports near-LLONG_MAX when no limit is set; cgroup v2
#: uses the literal string "max". 10 TiB is well above any realistic
#: container limit but well below the "no limit" sentinel, so anything
#: bigger gets treated as "unlimited" and we fall back to psutil.
_NO_LIMIT_THRESHOLD_BYTES = 10 * (1024 ** 4)


HealthSnapshot = dict[str, float | int | None]
HealthSnapshotProvider = Callable[[], Awaitable[HealthSnapshot]]


def _read_cgroup_memory() -> tuple[int, int] | None:
    """Read (used_bytes, limit_bytes) from cgroup v2, then v1.

    Returns ``None`` when no usable container limit is found — caller
    must then fall back to host-wide ``psutil.virtual_memory``. A
    "no limit" sentinel (cgroup v2 string ``"max"``, cgroup v1
    near-LLONG_MAX value) is treated as "no usable limit".
    """
    # cgroup v2 (modern Docker / k8s on 5.x kernels)
    try:
        max_str = _CGROUP_V2_MEM_MAX.read_text().strip()
        if max_str != "max":
            limit = int(max_str)
            if 0 < limit < _NO_LIMIT_THRESHOLD_BYTES:
                used = int(_CGROUP_V2_MEM_CURRENT.read_text().strip())
                return used, limit
    except (OSError, ValueError):
        pass
    # cgroup v1 (older Docker / k8s)
    try:
        limit = int(_CGROUP_V1_MEM_LIMIT.read_text().strip())
        if 0 < limit < _NO_LIMIT_THRESHOLD_BYTES:
            used = int(_CGROUP_V1_MEM_USAGE.read_text().strip())
            return used, limit
    except (OSError, ValueError):
        pass
    return None


async def _read_psutil() -> tuple[float | None, float | None, float | None]:
    """Return (cpu_percent, ram_used_gb, ram_total_gb) — ``None`` triple
    if psutil is missing.

    Memory is read from cgroup files when available so the operator
    sees the container's allocated RAM, not the host's full RAM.
    """
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        return None, None, None
    try:
        cpu = float(psutil.cpu_percent(interval=None))
    except Exception:
        cpu = None

    cgroup = _read_cgroup_memory()
    if cgroup is not None:
        used_bytes, total_bytes = cgroup
        return cpu, used_bytes / (1024**3), total_bytes / (1024**3)

    try:
        vm = psutil.virtual_memory()
        return cpu, vm.used / (1024**3), vm.total / (1024**3)
    except Exception:
        return cpu, None, None


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
            # Build a typed envelope when the snapshot reader supplied
            # at least the required CPU / RAM fields. The legacy
            # snapshot is a free-form dict (kept for the older WS
            # consumer); we mirror its values in the typed payload so
            # downstream typed consumers see the same numbers.
            cpu_pct = float(snapshot.get("cpu_percent") or 0.0)
            ram_used_gb = snapshot.get("ram_used_gb")
            ram_bytes = int((ram_used_gb or 0.0) * (1024**3))
            disk_free = int(snapshot.get("disk_free_bytes") or 0)
            gpu_util = snapshot.get("gpu_util_percent")
            gpu_mem = snapshot.get("gpu_memory_percent")
            gpus: list[GPUSnapshot] = []
            if gpu_util is not None or gpu_mem is not None:
                gpus.append(
                    GPUSnapshot(
                        device="gpu0",
                        utilization_pct=float(gpu_util or 0.0),
                        memory_used_bytes=0,
                        memory_total_bytes=0,
                    ),
                )
            try:
                self._bus.publish(
                    HealthSnapshotEvent(
                        source="pod://runner/health_reporter",
                        run_id="unknown",
                        offset=UNKNOWN_OFFSET,
                        payload=HealthSnapshotPayload(
                            cpu_pct=cpu_pct,
                            ram_bytes=ram_bytes,
                            gpu=gpus,
                            disk_free_bytes=disk_free,
                        ),
                    ),
                )
            except Exception:
                # Bus might be closed during shutdown — best-effort.
                continue

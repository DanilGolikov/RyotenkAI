"""Phase 11.E — Mac-side "control plane is active" heartbeat.

Background
----------
Phase 11.B implicit heartbeat (driven by WS yields + REST GETs) goes
stale during long Mac-side work that bypasses the runner FastAPI —
notably ``ModelRetriever.download_directory`` which streams adapters
through a separate SSH ``tar | ssh ... | tar`` pipeline. After ~60 s
of pure SCP, the implicit heartbeat would expire and trigger
``podStop`` mid-download.

The fix is an explicit, periodic "process is alive" ping: while the
Mac orchestrator process is alive, this service POSTs to
``/api/v1/control/heartbeat`` every 30 s. The runner's
:class:`MacHeartbeat` is refreshed with a 120 s TTL (2× the ping
interval) so a single missed cycle doesn't immediately stale the
heartbeat. Combined with the retry loop in :class:`PodTerminator`,
a transient stall on the Mac side cannot trigger an unintended
``podStop``.

Lifetime contract
-----------------
* :meth:`start` — spawn the background task. Idempotent: a second
  ``start()`` while running is a no-op.
* :meth:`stop` — cancel + await the task. Idempotent: a second
  ``stop()`` is a no-op. Always called from the orchestrator's
  cleanup path (``finally`` block in
  ``TrainingLauncher.start_training``).

The task itself uses ``asyncio.shield`` semantics implicitly via
:meth:`asyncio.wait_for` so a cancellation cleanly drops the
in-flight ping rather than hanging the whole shutdown.

Failure tolerance
-----------------
Transient errors (httpx connect/timeout, runner not yet ready, SSH
tunnel briefly closed) are logged at DEBUG and retried on the next
interval. We never raise into the orchestrator — the orchestrator
treats this service as fire-and-forget. The pod-side retry loop
(:class:`PodTerminator._check_heartbeat_with_retries`) is the
second line of defence for the case where multiple consecutive
pings fail.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.api.clients.job_client import JobClient


__all__ = [
    "DEFAULT_PING_INTERVAL_SECONDS",
    "DEFAULT_TTL_SECONDS",
    "ControlPlaneHeartbeat",
]


logger = logging.getLogger("ryotenkai.control_plane_heartbeat")


#: How often to ping the runner. Half of the runner's default
#: explicit TTL (120 s), so a single dropped ping doesn't stale the
#: heartbeat. Tunable via constructor kwarg for tests.
DEFAULT_PING_INTERVAL_SECONDS = 30.0

#: TTL we ask the runner to apply on each ping. ``None`` = use
#: runner's explicit-default (120 s). Override for short tests.
DEFAULT_TTL_SECONDS: float | None = None


class ControlPlaneHeartbeat:
    """Background ping service that keeps the in-pod heartbeat fresh.

    Owns a single :class:`asyncio.Task`. Construction does not start
    the task — call :meth:`start` from the orchestrator after the
    SSH tunnel + :class:`JobClient` are ready.

    Args:
        client: An open :class:`JobClient` connected through the
            current SSH tunnel. The service does NOT close the
            client on stop — that's the caller's lifecycle.
        ping_interval_seconds: Cadence between pings.
            ``DEFAULT_PING_INTERVAL_SECONDS`` (30 s) by default.
        ttl_seconds: TTL to ask the runner to apply on each ping.
            ``None`` ⇒ runner's default (120 s). Tests override
            for compressed timing.
        on_error: Optional callable invoked on each failed ping with
            the exception (or ``None`` for non-200 responses). Tests
            use it to assert error paths; production passes ``None``
            (errors logged at DEBUG only).
    """

    def __init__(
        self,
        client: JobClient,
        *,
        ping_interval_seconds: float = DEFAULT_PING_INTERVAL_SECONDS,
        ttl_seconds: float | None = DEFAULT_TTL_SECONDS,
        on_error: Callable[..., object] | None = None,
    ) -> None:
        self._client = client
        self._interval = max(1.0, float(ping_interval_seconds))
        self._ttl = ttl_seconds
        self._on_error = on_error

        # Task + state. Created lazily in :meth:`start`.
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        # Telemetry counters — useful for tests + operator dashboards.
        self._ping_success_count = 0
        self._ping_failure_count = 0

    @property
    def is_running(self) -> bool:
        """``True`` between :meth:`start` and :meth:`stop`."""
        return self._task is not None and not self._task.done()

    @property
    def ping_success_count(self) -> int:
        return self._ping_success_count

    @property
    def ping_failure_count(self) -> int:
        return self._ping_failure_count

    async def start(self) -> None:
        """Spawn the background ping task. Idempotent.

        Sends an immediate ping on start so the runner sees the
        orchestrator before the first interval elapses; this also
        verifies the tunnel is reachable.
        """
        if self.is_running:
            return
        self._stop_event = asyncio.Event()
        # Synchronous immediate ping (best-effort — failure logged but
        # doesn't block startup; the task picks up the cadence).
        await self._send_one(initial=True)
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancel + await the task. Idempotent.

        Always safe to call from a ``finally`` block; the task's
        :meth:`asyncio.CancelledError` is swallowed and the method
        returns once the task has fully unwound.
        """
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is None:
            return
        if self._task.done():
            self._task = None
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task
        self._task = None

    async def _run(self) -> None:
        """Ping-then-sleep loop. Exits when ``_stop_event`` is set or
        the task is cancelled.

        We use ``asyncio.wait_for`` on the stop event so a stop
        request unblocks the sleep immediately rather than waiting
        for the next interval.
        """
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._interval,
                )
                # Stop event fired during the wait — exit cleanly.
                return
            except TimeoutError:
                # Normal path — interval elapsed without stop signal.
                pass
            except asyncio.CancelledError:
                return

            await self._send_one(initial=False)

    async def _send_one(self, *, initial: bool) -> None:
        """Send a single heartbeat ping. Best-effort.

        ``initial=True`` only changes the log message — useful for
        operator-side debugging ("did the start ping go through?").
        Returns nothing; counters track success / failure.
        """
        try:
            ok = await self._client.send_heartbeat(ttl_seconds=self._ttl)
        except Exception as exc:
            self._ping_failure_count += 1
            logger.debug(
                "[CP-HEARTBEAT] %s ping raised: %s",
                "initial" if initial else "scheduled",
                exc,
            )
            if self._on_error is not None:
                with contextlib.suppress(Exception):
                    self._on_error(exc)
            return

        if ok:
            self._ping_success_count += 1
            logger.debug(
                "[CP-HEARTBEAT] %s ping ok",
                "initial" if initial else "scheduled",
            )
        else:
            self._ping_failure_count += 1
            logger.debug(
                "[CP-HEARTBEAT] %s ping returned non-200",
                "initial" if initial else "scheduled",
            )
            if self._on_error is not None:
                with contextlib.suppress(Exception):
                    self._on_error(None)

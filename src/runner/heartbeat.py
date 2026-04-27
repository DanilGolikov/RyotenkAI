"""Phase 11.B — Mac control-plane heartbeat ledger.

Tracks the timestamp of the last successful in-pod ↔ Mac interaction.
Used by :class:`PodTerminator` (`pod_terminator.py`) to decide between
two natural-completion paths:

* **Mac alive** — give the orchestrator a 60-second grace window so
  ``ModelRetriever`` can SCP adapters off the pod before we ``podStop``.
* **Mac asleep** — no point waiting; ``podStop`` immediately so GPU
  billing drops to zero.

Why a heartbeat instead of an explicit handshake?
-------------------------------------------------

* The runner already streams events over WebSocket (Phase 1) and serves
  REST GETs on ``/jobs/{id}`` (Phase 1, used by ``ryotenkai job status``
  and ``ModelRetriever``). Both of those touch the runner only when the
  Mac is awake. Piggy-backing on them gives a free, zero-round-trip
  signal.
* Adding a dedicated ``POST /heartbeat`` endpoint would duplicate the
  signal that's already implicit in WS keepalive + REST traffic. YAGNI.
* The decision is binary (alive vs not), the latency budget is loose
  (we tolerate up to ``HEARTBEAT_TTL_SECONDS`` of staleness), and the
  cost of a wrong answer is small (false-positive = pod waits 60s
  unnecessarily, false-negative = pod sleeps and Mac resumes via the
  Phase 11.C resume flow).

Threading
---------

The runner is single-process FastAPI. Updates come from:

* WS handler (``src/runner/api/events.py``) — after each successful
  ``ws.send_json(...)``.
* REST handler (``src/runner/api/jobs.py``) — on each successful
  ``GET /jobs/{id}`` reply (so a long ``ModelRetriever`` poll keeps
  the heartbeat fresh even after the WS subscription drops).

All callers run on the FastAPI event loop. We don't need a lock — the
single integer write is atomic in CPython, and a slightly stale read
in the ``PodTerminator`` decision is acceptable. We expose the time
source as a kwarg for tests so they can dial ``monotonic`` deterministically.

Clock choice
------------

:func:`time.monotonic` (NOT :func:`time.time`). The heartbeat is a
within-process freshness signal — it never leaves the runner. Monotonic
is immune to wall-clock skew (NTP corrections, Mac timezone changes
in flight) which would otherwise let a backwards jump make a stale
heartbeat look fresh.
"""

from __future__ import annotations

import time
from collections.abc import Callable


__all__ = ["MacHeartbeat"]


class MacHeartbeat:
    """Ledger of last successful Mac↔pod interaction.

    Constructed once on FastAPI lifespan start (`src/runner/main.py::_lifespan`),
    shared via ``app.state.heartbeat`` to every request handler that needs it.

    Threading: single-writer (FastAPI event loop), best-effort reader
    semantics. No locks needed — one integer write is atomic in CPython.

    Test seam: pass a custom ``clock`` callable returning
    :func:`time.monotonic`-compatible seconds-as-float. Tests use
    deterministic counters; production uses :func:`time.monotonic`.
    """

    #: Default TTL for implicit pings (WS yields, REST GETs). Picked
    #: to cover typical OS sleep-detection grace (Mac power-nap can
    #: keep TCP alive for ~30-45s) plus a margin.
    HEARTBEAT_TTL_SECONDS: float = 60.0

    #: Default TTL for **explicit** control-plane pings (Phase 11.E).
    #: Set to 2× the recommended client ping interval (30 s) so a
    #: single missed ping doesn't immediately stale the heartbeat —
    #: the orchestrator gets one full retry cycle of slack.
    EXPLICIT_HEARTBEAT_TTL_SECONDS: float = 120.0

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        """Build a heartbeat ledger.

        Args:
            clock: Zero-arg callable returning monotonic seconds-as-float.
                ``None`` ⇒ :func:`time.monotonic`. Tests inject a
                deterministic counter.
            ttl_seconds: Override the class-level
                :attr:`HEARTBEAT_TTL_SECONDS`. Mostly for tests; in
                production the class attribute is the operational
                contract.
        """
        self._clock = clock or time.monotonic
        self._default_ttl = (
            ttl_seconds if ttl_seconds is not None
            else self.HEARTBEAT_TTL_SECONDS
        )
        # ``None`` = never seen the Mac (fresh runner, no subscriber
        # yet). Distinct from "Mac was here but went quiet" (any
        # ``float`` value, including 0.0 when monotonic clock starts
        # there). The ``is_alive()`` semantics treat "never seen" as
        # asleep — when the runner just booted and no one's
        # connected, the safest assumption is to ``podStop`` if a
        # terminal state arrives early (rare race).
        #
        # We use ``None`` rather than a sentinel float (e.g. ``-1``)
        # because ``time.monotonic`` is allowed to return ``0`` on
        # startup, and tests inject deterministic clocks that often
        # start at ``0.0``. A sentinel float would silently misclassify
        # a legitimate 0-timestamp mark as "never seen".
        self._last_active_s: float | None = None
        # Phase 11.E — TTL of the most recent ``mark_active`` call.
        # Implicit pings (WS / REST) use the default; explicit
        # control-plane pings use a longer TTL so a single missed
        # cycle doesn't stale the heartbeat. We track the most-recent
        # TTL because a fresh long-TTL ping should reset the freshness
        # contract, not be capped by an earlier short-TTL value.
        self._last_ttl_s: float = self._default_ttl

    def mark_active(self, *, ttl_seconds: float | None = None) -> None:
        """Record that we just had a successful interaction with the Mac.

        Called from:

        * WS event yield handler — after :meth:`WebSocket.send_json`
          returns. TCP backpressure ⇒ if ``send_json`` blocks /
          fails, ``mark_active`` is not called ⇒ heartbeat goes
          stale ⇒ correct.
        * REST GET handler — after the response is serialised.
        * **Phase 11.E** — explicit ``POST /api/v1/control/heartbeat``
          from the Mac orchestrator's ``ControlPlaneHeartbeat`` service.
          This is the **process-active** signal: while the Mac
          orchestrator process is alive, it pings every 30 s so the
          pod knows long-running work (e.g. ``ModelRetriever`` SCP'ing
          model adapters via a separate SSH stream) is still in flight,
          even when no WS / REST traffic is hitting the runner.

        Args:
            ttl_seconds: Override the default TTL for THIS mark.
                ``None`` ⇒ use the constructor's default (typically
                :attr:`HEARTBEAT_TTL_SECONDS` = 60s).
                Phase 11.E callers pass
                :attr:`EXPLICIT_HEARTBEAT_TTL_SECONDS` (120 s) so a
                missed ping doesn't immediately flip the heartbeat.

        Idempotent: every call updates the timestamp to "now" and
        the TTL to the new value. Multiple calls within the same
        loop iteration cost nothing.
        """
        self._last_active_s = self._clock()
        self._last_ttl_s = (
            ttl_seconds if ttl_seconds is not None else self._default_ttl
        )

    def is_alive(self) -> bool:
        """Did the Mac touch us within the most-recent TTL window?

        Returns:
            True iff a positive timestamp exists AND it's no older
            than the TTL of the most recent ``mark_active`` call.
            False on a fresh runner (never seen anyone) OR after a
            silent gap longer than the TTL.

        ``PodTerminator`` reads this on every terminal-hook decision.
        Failure mode of a stale read is bounded: at worst the pod
        stops or grace-waits unnecessarily — never destroys data.
        """
        last = self._last_active_s
        if last is None:
            return False
        return (self._clock() - last) < self._last_ttl_s

    def age_seconds(self) -> float | None:
        """How long since the last interaction.

        Returns:
            ``None`` when the Mac has never been seen (fresh runner).
            Otherwise the seconds-as-float gap between now and the
            last :meth:`mark_active` call.

        Used by telemetry: ``pod_terminal_decision`` events include
        the heartbeat age so operator dashboards can debug
        "why did this run go straight to ``stopped_for_resume`` —
        was the Mac actually asleep?".
        """
        last = self._last_active_s
        if last is None:
            return None
        return self._clock() - last

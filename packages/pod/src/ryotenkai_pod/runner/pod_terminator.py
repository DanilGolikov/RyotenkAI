"""Phase 11.B / 14.B — Provider-agnostic terminal-hook decision matrix.

Replaces Phase 9.A's :class:`PodStopper` with a smarter handler that
picks between four actions based on terminal state, Mac heartbeat,
and volume kind:

* **Terminate** — pod fully deleted. For user-stop (cancelled),
  failed runs without ``KEEP_ON_ERROR``, or any pod with a network
  volume (cloud-provider constraint: network-volume pods cannot be
  stopped, only terminated).
* **Stop (sleep)** — pod paused. GPU billing→0, ``/workspace``
  preserved, recoverable via the provider's resume call. Default for
  natural completion + Mac asleep (artifacts wait on disk for the
  user to wake up and resume) and for failed runs when Mac is
  asleep (operator may want to debug the checkpoint).
* **Stop with grace** — same as stop, but wait while Mac heartbeat
  stays alive (capped at 10 min). For natural completion + Mac alive:
  give the orchestrator's ``ModelRetriever`` time to SCP adapters off
  the pod, then stop. ``ModelRetriever`` GET requests refresh the
  heartbeat → grace effectively extends to cover the whole download.
* **Keep alive** — no-op. For ``failed`` + ``KEEP_ON_ERROR=true``,
  reserved for SSH-forensics (Phase 9.A carry-over).

Decision matrix (Phase 11 § 11.1)
---------------------------------

| terminal | mac_alive | volume     | keep_err | outcome                       | action      |
|----------|-----------|------------|----------|-------------------------------|-------------|
| cancel   | *         | *          | *        | terminated_user_stop          | terminate   |
| failed   | *         | *          | true     | kept_alive_for_debug          | none        |
| failed   | true      | persistent | false    | terminated_safety             | terminate   |
| failed   | false     | persistent | false    | stopped_for_resume            | pause       |
| failed   | *         | network    | false    | terminated_safety             | terminate   |
| complete | true      | persistent | *        | stopped_for_resume_short_grace| grace+pause |
| complete | false     | persistent | *        | stopped_for_resume            | pause       |
| complete | *         | network    | *        | terminated_safety             | terminate   |

Phase 14.B migration
--------------------

Pre-14.B this module owned RunPod GraphQL transport directly
(``DEFAULT_RUNPOD_GRAPHQL_URL``, ``_ALREADY_GONE_RE``,
``_call_mutation`` / ``_call_terminate`` / ``_call_stop``). Phase
14.B § 1.5 extracted those into
:mod:`src.providers.runpod.runtime.lifecycle_client` behind the
:class:`~src.runner.runtime.lifecycle_client.IPodLifecycleClient`
Protocol. The terminator now dispatches to ``self._client`` —
RunPod, single-node, or any future provider all work through the
same interface.

Phase 14.B § 1.4 also moved env reads (``RUNPOD_VOLUME_KIND``,
``RUNPOD_KEEP_ON_ERROR``, ``RUNPOD_POD_ID``) from per-call into the
constructor. Lifespan reads env once at boot
(:mod:`src.runner.runtime.provider_registry`); the terminator
inherits that snapshot. Missing creds become :class:`BootstrapConfigError`
at boot, not silent ``SKIPPED`` outcomes 4 hours later.

Idempotency
-----------

Both ``terminate`` and ``pause`` are idempotent: already-gone /
already-stopped pods return ``ALREADY_TERMINATED`` /
``ALREADY_STOPPED`` from the provider client. Mac's
:meth:`provider.cleanup_pod` (canonical Mac-side path) and our
in-pod terminator can run concurrently for the same pod — both
orderings converge to "pod gone or sleeping safely".
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ryotenkai_pod.runner.heartbeat import MacHeartbeat
    from ryotenkai_shared.infrastructure.lifecycle import IPodLifecycleClient


__all__ = [
    "PodTerminalOutcome",
    "PodTerminator",
    "decide_terminal_outcome",
    "run_terminal_hook",
]


# ---------------------------------------------------------------------------
# Outcome strings
# ---------------------------------------------------------------------------


# PodTerminalOutcome moved to
# ``ryotenkai_shared.infrastructure.lifecycle.outcomes`` (ADR row 7,
# Phase C drift fix). Re-exported here so existing callers keep
# working while we sweep imports — the long-term home is shared.
from ryotenkai_shared.infrastructure.lifecycle.outcomes import (  # noqa: E402
    PodTerminalOutcome,
)


# ---------------------------------------------------------------------------
# Pure decision function
# ---------------------------------------------------------------------------


def decide_terminal_outcome(
    *,
    terminal_state: str,
    mac_alive: bool,
    volume_kind: str,
    keep_on_error: bool,
) -> str:
    """Pure decision per § 11.1 table — easy to test exhaustively.

    Args:
        terminal_state: ``"completed"`` / ``"failed"`` / ``"cancelled"``.
            Anything else returns :data:`KEPT_ALIVE_FOR_DEBUG` as a
            safe fallback (we don't recognise the state, don't risk
            destroying data).
        mac_alive: :meth:`MacHeartbeat.is_alive` snapshot.
        volume_kind: ``"persistent"`` (default RunPod training volume,
            stop-able) or ``"network"`` (network volume, terminate-only
            per RunPod constraint). Anything else ⇒ treated as
            ``persistent``.
        keep_on_error: ``True`` ⇒ ``RUNPOD_KEEP_ON_ERROR=true`` was
            set. Honoured **only** on ``failed``; user-stop always
            terminates regardless.

    Returns:
        One of the decision-stage :class:`PodTerminalOutcome` strings.
        Action dispatch is the caller's job.
    """
    # User-stop always terminates — explicit user action overrides
    # everything (including KEEP_ON_ERROR, which is for *automatic*
    # failures only).
    if terminal_state == "cancelled":
        return PodTerminalOutcome.TERMINATED_USER_STOP

    # KEEP_ON_ERROR honoured only on automatic failures.
    if terminal_state == "failed" and keep_on_error:
        return PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG

    # Network volume: cannot be stopped. Always terminate.
    if volume_kind == "network":
        return PodTerminalOutcome.TERMINATED_SAFETY

    # ``persistent`` (default) volume: stop-vs-terminate based on
    # state + heartbeat.
    if terminal_state == "failed":
        # PR-C — failed run + Mac alive: brief diagnostic grace so the
        # post-mortem SCP completes before resource cleanup races it.
        # Pre PR-C this returned TERMINATED_SAFETY (0 grace), causing
        # the 15-crash-incident postmortems to consistently see
        # ``<<MISSING>>`` because the pod was gone before LogManager
        # could pull. Failed + Mac asleep keeps the legacy semantics:
        # pause to keep checkpoint accessible.
        if mac_alive:
            return PodTerminalOutcome.TERMINATED_AFTER_DIAGNOSTIC_GRACE
        return PodTerminalOutcome.STOPPED_FOR_RESUME

    if terminal_state == "completed":
        # Natural completion + Mac alive: give retriever time to grab
        # adapters before sleeping the pod.
        # Natural completion + Mac asleep: artifacts wait on disk;
        # user resumes on wake.
        if mac_alive:
            return PodTerminalOutcome.STOPPED_FOR_RESUME_SHORT_GRACE
        return PodTerminalOutcome.STOPPED_FOR_RESUME

    # Unknown state — keep alive defensively.
    return PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG


# ---------------------------------------------------------------------------
# Terminator (action dispatch)
# ---------------------------------------------------------------------------


class PodTerminator:
    """Phase 14.B — provider-agnostic action dispatcher.

    Construct once in the lifespan with the pre-resolved
    :class:`~src.runner.runtime.lifecycle_client.IPodLifecycleClient` +
    lifespan-static config (:attr:`_resource_id`, :attr:`_volume_kind`,
    :attr:`_keep_on_error`). Invoke :meth:`decide_and_act` from the
    supervisor's reap path.

    Test seams:
        * ``client`` — inject a fake :class:`IPodLifecycleClient`
          stub. Tests dispatch against the stub; HTTP-level coverage
          lives next to the RunPod impl in
          :mod:`src.tests.unit.providers.runpod.runtime`.
        * ``sleep`` — swap so retry / grace timings don't actually
          wait in unit tests.
    """

    #: Base grace before pause on the SHORT_GRACE path. ModelRetriever
    #: typically completes in 30-60s for a few-GB adapter; 60s is the
    #: minimum window. If retriever takes longer, its GET requests
    #: refresh the heartbeat and the grace loop keeps extending.
    GRACE_BASE_SECONDS: float = 60.0

    #: Hard cap on the SHORT_GRACE wait. Even if the heartbeat keeps
    #: refreshing forever, we eventually pause after this. Protects
    #: against runaway "Mac is alive but never finishes downloading"
    #: cases.
    GRACE_MAX_SECONDS: float = 600.0

    #: Polling interval inside the grace loop. Smaller = faster
    #: response when heartbeat dies; larger = less event-bus chatter.
    GRACE_TICK_SECONDS: float = 10.0

    #: Phase 11.E — number of retry checks before declaring Mac dead
    #: on the natural-completion path. Each retry sleeps
    #: :attr:`HEARTBEAT_RETRY_TICK_SECONDS` seconds before re-reading
    #: the heartbeat. With defaults of 3 retries × 10 s = 30 s total
    #: window, the orchestrator's 30 s ping cadence is guaranteed to
    #: land at least once during the retry loop, so any momentary
    #: heartbeat staleness (e.g. the Mac's just-now-asleep window
    #: waiting for the next ping) self-corrects.
    HEARTBEAT_RETRY_ATTEMPTS: int = 3

    #: Sleep between heartbeat retries. Picked to bracket the Mac's
    #: 30 s ping interval with 3× safety: 3 retries × 10 s = 30 s
    #: round-trip.
    HEARTBEAT_RETRY_TICK_SECONDS: float = 10.0

    #: PR-C — diagnostic grace for ``failed + mac_alive`` decisions.
    #: One Mac-side SCP roundtrip + tail decode is ~2-5 s; we give 30 s
    #: to cover slower SSH connections and the periodic-pull window
    #: (LOG_DOWNLOAD_INTERVAL_DEFAULT=5 s × ~6 ticks). Aborted early
    #: when the heartbeat dies so the cost is paid only when Mac is
    #: actually trying to read.
    DIAGNOSTIC_GRACE_SECONDS: float = 30.0

    #: Polling interval for the diagnostic-grace heartbeat watcher.
    #: 2 s is a balance between snappy abort (when Mac dies) and event
    #: bus chatter — 2 s × 15 ticks = 30 s in the typical no-abort case.
    DIAGNOSTIC_GRACE_TICK_SECONDS: float = 2.0

    def __init__(
        self,
        *,
        client: "IPodLifecycleClient",
        resource_id: str,
        volume_kind: str,
        keep_on_error: bool,
        sleep: "Callable[[float], Awaitable[None]] | None" = None,
        grace_base_seconds: float | None = None,
        grace_max_seconds: float | None = None,
        grace_tick_seconds: float | None = None,
        heartbeat_retry_attempts: int | None = None,
        heartbeat_retry_tick_seconds: float | None = None,
        diagnostic_grace_seconds: float | None = None,
        diagnostic_grace_tick_seconds: float | None = None,
    ) -> None:
        self._client = client
        self._resource_id = resource_id
        # Defensive normalisation — pre-14.B caller passed env raw and
        # we clamped invalid values to "persistent". Phase 14.B's
        # provider_registry already does this clamping at boot, but
        # we keep the safety net so direct callers (e.g. tests) get
        # the same behaviour.
        self._volume_kind = (
            volume_kind if volume_kind in ("persistent", "network")
            else "persistent"
        )
        self._keep_on_error = keep_on_error
        self._sleep = sleep or asyncio.sleep
        self._grace_base = (
            grace_base_seconds if grace_base_seconds is not None
            else self.GRACE_BASE_SECONDS
        )
        self._grace_max = (
            grace_max_seconds if grace_max_seconds is not None
            else self.GRACE_MAX_SECONDS
        )
        self._grace_tick = (
            grace_tick_seconds if grace_tick_seconds is not None
            else self.GRACE_TICK_SECONDS
        )
        # Phase 11.E — retry knobs.
        self._heartbeat_retry_attempts = (
            heartbeat_retry_attempts if heartbeat_retry_attempts is not None
            else self.HEARTBEAT_RETRY_ATTEMPTS
        )
        self._heartbeat_retry_tick = (
            heartbeat_retry_tick_seconds
            if heartbeat_retry_tick_seconds is not None
            else self.HEARTBEAT_RETRY_TICK_SECONDS
        )
        # PR-C — diagnostic grace knobs. Tests pass small values to
        # keep wall-clock time bounded; production uses class defaults.
        self._diagnostic_grace_seconds = (
            diagnostic_grace_seconds if diagnostic_grace_seconds is not None
            else self.DIAGNOSTIC_GRACE_SECONDS
        )
        self._diagnostic_grace_tick = (
            diagnostic_grace_tick_seconds
            if diagnostic_grace_tick_seconds is not None
            else self.DIAGNOSTIC_GRACE_TICK_SECONDS
        )

    async def decide_and_act(
        self,
        *,
        terminal_state: str,
        heartbeat: "MacHeartbeat",
        bus_publish: "Callable[[str, dict[str, Any]], Any]",
    ) -> dict[str, Any]:
        """Run the decision matrix + dispatch to the right action.

        Returns a dict with ``decision`` (intent) and ``action`` (what
        actually happened on the provider call, or ``None`` for no-op
        outcomes). Caller publishes events from inside this method —
        we emit ``pod_terminal_decision`` after deciding and
        ``pod_stop_attempt`` after acting (Phase 9.A naming kept for
        backwards compat with operator dashboards).
        """
        # Phase 11.E — retry the heartbeat check before declaring Mac
        # dead. Without retries, a single missed control-plane ping
        # (or a Mac-side SCP-stream that briefly starved the
        # orchestrator's heartbeat thread) would flip ``mac_alive`` to
        # False and trigger ``STOPPED_FOR_RESUME`` mid-ModelRetriever.
        # The retry loop polls the ledger N times with
        # ``heartbeat_retry_tick`` seconds between attempts; the Mac's
        # 30 s ping cadence is guaranteed to land at least once during
        # the default 3×10 s = 30 s window.
        mac_alive, retry_attempts_used = await self._check_heartbeat_with_retries(
            heartbeat=heartbeat,
            terminal_state=terminal_state,
            bus_publish=bus_publish,
        )
        heartbeat_age = heartbeat.age_seconds()

        decision = decide_terminal_outcome(
            terminal_state=terminal_state,
            mac_alive=mac_alive,
            volume_kind=self._volume_kind,
            keep_on_error=self._keep_on_error,
        )

        # Publish decision regardless of action — operators see
        # "this is what we picked and why".
        bus_publish(
            "pod_terminal_decision",
            {
                "decision": decision,
                "terminal_state": terminal_state,
                "mac_alive": mac_alive,
                "heartbeat_age_seconds": heartbeat_age,
                "heartbeat_retry_attempts_used": retry_attempts_used,
                "volume_kind": self._volume_kind,
                "keep_on_error": self._keep_on_error,
            },
        )

        # No-op decisions: nothing to dispatch.
        if decision == PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG:
            return {"decision": decision, "action": None}

        # Dispatch through the provider client.
        if decision in (
            PodTerminalOutcome.TERMINATED_USER_STOP,
            PodTerminalOutcome.TERMINATED_SAFETY,
        ):
            client_result = await self._client.terminate(
                resource_id=self._resource_id,
            )
        elif decision == PodTerminalOutcome.TERMINATED_AFTER_DIAGNOSTIC_GRACE:
            # PR-C — wait briefly so Mac can pull the post-mortem
            # before we tear the pod down. Heartbeat-aware: if Mac
            # disconnects during the grace, abort early — no point
            # waiting for a reader that left.
            await self._wait_diagnostic_grace(
                heartbeat=heartbeat, bus_publish=bus_publish,
            )
            client_result = await self._client.terminate(
                resource_id=self._resource_id,
            )
        elif decision == PodTerminalOutcome.STOPPED_FOR_RESUME:
            client_result = await self._client.pause(
                resource_id=self._resource_id,
            )
        elif decision == PodTerminalOutcome.STOPPED_FOR_RESUME_SHORT_GRACE:
            await self._wait_grace(
                heartbeat=heartbeat, bus_publish=bus_publish,
            )
            client_result = await self._client.pause(
                resource_id=self._resource_id,
            )
        else:  # pragma: no cover — defensive; decision matrix exhaustive
            return {"decision": decision, "action": None}

        action = client_result.outcome
        bus_publish(
            "pod_stop_attempt",
            {
                "terminal_state": terminal_state,
                "decision": decision,
                "action": action,
                # Phase 9.A `outcome` carry — old dashboards parse this.
                # We promote ``action`` to be the canonical field.
                "outcome": action,
                # Phase 14.B — per-call retry pressure. Surfaces "we
                # got there but it took 3 tries" to operator dashboards.
                "attempts_made": client_result.attempts_made,
                "provider": self._client.provider_name,
            },
        )
        return {"decision": decision, "action": action}

    # --- heartbeat retry --------------------------------------------------

    async def _check_heartbeat_with_retries(
        self,
        *,
        heartbeat: "MacHeartbeat",
        terminal_state: str,
        bus_publish: "Callable[[str, dict[str, Any]], Any]",
    ) -> tuple[bool, int]:
        """Poll the heartbeat ledger up to N times before declaring Mac dead.

        Phase 11.E. The first read is free; if it returns ``True`` we
        skip the loop entirely (fast path, no extra latency on the
        common case where the orchestrator has been actively pinging).
        Otherwise we sleep ``heartbeat_retry_tick`` seconds and re-read
        up to ``heartbeat_retry_attempts`` more times.

        Why retry only when first read is False:
            We never want to delay the terminal hook unnecessarily. The
            retry loop's cost is bounded — only paid when the heartbeat
            looks stale, which is the only case where a wrong decision
            would actually hurt. When Mac is clearly alive (orchestrator
            actively pinging), we get the SHORT_GRACE path with zero
            retry latency.

        Why NOT retry on cancelled/failed:
            User-stop and automatic-failure paths have well-defined
            outcomes regardless of heartbeat (terminate / honour
            KEEP_ON_ERROR). Heartbeat retries only matter for natural
            completion, where the wrong call (mac_alive=False when in
            fact the orchestrator is just heads-down on SCP) costs us
            data.

        Returns:
            ``(mac_alive, attempts_used)`` — ``attempts_used`` is the
            number of retries we actually performed (0 when first read
            already returned True, up to ``heartbeat_retry_attempts``
            when we exhausted the loop). Surfaced in the
            ``pod_terminal_decision`` event so operators see how
            close we came to a wrong call.
        """
        # Fast path — first read decides on positive answer.
        if heartbeat.is_alive():
            return True, 0

        # Retry path — only meaningful for natural completion.
        if terminal_state != "completed":
            return False, 0

        bus_publish(
            "pod_terminal_heartbeat_retry_started",
            {
                "max_attempts": self._heartbeat_retry_attempts,
                "tick_seconds": self._heartbeat_retry_tick,
            },
        )

        for attempt in range(1, self._heartbeat_retry_attempts + 1):
            await self._sleep(self._heartbeat_retry_tick)
            if heartbeat.is_alive():
                bus_publish(
                    "pod_terminal_heartbeat_retry_recovered",
                    {
                        "attempt": attempt,
                        "max_attempts": self._heartbeat_retry_attempts,
                    },
                )
                return True, attempt

        bus_publish(
            "pod_terminal_heartbeat_retry_exhausted",
            {"attempts": self._heartbeat_retry_attempts},
        )
        return False, self._heartbeat_retry_attempts

    # --- PR-C diagnostic grace --------------------------------------------

    async def _wait_diagnostic_grace(
        self,
        *,
        heartbeat: "MacHeartbeat",
        bus_publish: "Callable[[str, dict[str, Any]], Any]",
    ) -> None:
        """Wait up to :attr:`DIAGNOSTIC_GRACE_SECONDS` so Mac can SCP
        the post-mortem (trainer.stdio.log + runner.log) before pod
        teardown.

        Distinct from :meth:`_wait_grace` (the SHORT_GRACE path):

        * SHORT_GRACE waits for the **happy** path — ModelRetriever
          fetching adapters after a successful run. Up to 600 s, polled
          every 10 s.
        * DIAGNOSTIC_GRACE waits for the **failure** post-mortem —
          tail SCP, dmesg/nvidia-smi probes. 30 s default, polled
          every 2 s. Aborted early when heartbeat dies (RP8).

        Always emits ``pod_terminal_diagnostic_grace_started`` /
        ``…ended`` events so operator dashboards can plot how often we
        wait the full window vs abort early.
        """
        bus_publish(
            "pod_terminal_diagnostic_grace_started",
            {
                "max_seconds": self._diagnostic_grace_seconds,
                "tick_seconds": self._diagnostic_grace_tick,
            },
        )

        elapsed = 0.0
        reason = "max_budget_reached"
        while elapsed < self._diagnostic_grace_seconds:
            await self._sleep(self._diagnostic_grace_tick)
            elapsed += self._diagnostic_grace_tick

            if not heartbeat.is_alive():
                # Mac went away mid-grace — no reader left, no point
                # waiting. Abort early so resource cleanup doesn't
                # bill for an empty wait.
                reason = "heartbeat_lost"
                break

        bus_publish(
            "pod_terminal_diagnostic_grace_ended",
            {"reason": reason, "elapsed_s": elapsed},
        )

    # --- grace loop -------------------------------------------------------

    async def _wait_grace(
        self,
        *,
        heartbeat: "MacHeartbeat",
        bus_publish: "Callable[[str, dict[str, Any]], Any]",
    ) -> None:
        """Wait while Mac heartbeat is alive, capped at ``GRACE_MAX_SECONDS``.

        The base grace window is :attr:`GRACE_BASE_SECONDS`; each loop
        iteration sleeps :attr:`GRACE_TICK_SECONDS`. We extend
        indefinitely as long as the heartbeat keeps refreshing —
        ModelRetriever's GET polls hit ``MacHeartbeat.mark_active``
        through the runner's REST handler, so a long retriever
        download keeps the grace alive automatically.

        Two exit conditions:
        1. Heartbeat goes stale (Mac asleep) → break, pause now.
        2. ``GRACE_MAX_SECONDS`` reached → hard cap, pause now.
        """
        bus_publish(
            "pod_terminal_grace_started",
            {
                "base_seconds": self._grace_base,
                "max_seconds": self._grace_max,
                "tick_seconds": self._grace_tick,
            },
        )

        elapsed = 0.0
        while True:
            await self._sleep(self._grace_tick)
            elapsed += self._grace_tick

            if elapsed >= self._grace_max:
                bus_publish(
                    "pod_terminal_grace_ended",
                    {"reason": "max_budget_exceeded", "elapsed_s": elapsed},
                )
                return

            if elapsed >= self._grace_base and not heartbeat.is_alive():
                # Past base grace + Mac went silent → done waiting.
                # We don't break out earlier than ``GRACE_BASE_SECONDS``
                # even if heartbeat dies briefly; that gives Mac a
                # chance to reconnect after a transient blip.
                bus_publish(
                    "pod_terminal_grace_ended",
                    {"reason": "heartbeat_lost", "elapsed_s": elapsed},
                )
                return


# ---------------------------------------------------------------------------
# Convenience wrapper — used as the supervisor's terminal_hook closure
# ---------------------------------------------------------------------------


async def run_terminal_hook(
    terminator: PodTerminator,
    *,
    terminal_state: str,
    heartbeat: "MacHeartbeat",
    bus_publish: "Callable[[str, dict[str, Any]], Any]",
) -> None:
    """Wrap :meth:`PodTerminator.decide_and_act` for use as a
    ``terminal_hook`` callback on :class:`Supervisor`.

    Errors are swallowed — a failed terminal action must never
    prevent the FSM from reaching its terminal state. The wrapper
    publishes a structured error event for forensics on any
    unexpected exception in the decision/action chain.
    """
    try:
        await terminator.decide_and_act(
            terminal_state=terminal_state,
            heartbeat=heartbeat,
            bus_publish=bus_publish,
        )
    except Exception as exc:  # pragma: no cover — defensive
        bus_publish(
            "pod_stop_error",
            {"terminal_state": terminal_state, "error": repr(exc)},
        )

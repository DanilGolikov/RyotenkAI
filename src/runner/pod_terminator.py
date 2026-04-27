"""Phase 11.B — RunPod terminal-hook with stop-vs-terminate decision matrix.

Replaces Phase 9.A's :class:`PodStopper` (which always called
``podTerminate``) with a smarter handler that picks between four
actions based on terminal state, Mac heartbeat, and volume kind:

* **Terminate** — pod fully deleted (``podTerminate``). For user-stop
  (cancelled), failed runs without ``KEEP_ON_ERROR``, or any pod with
  a network volume (RunPod constraint: network-volume pods cannot be
  stopped, only terminated).
* **Stop (sleep)** — pod paused (``podStop``). GPU billing→0,
  ``/workspace`` preserved, recoverable via ``podResume``. The default
  for natural completion + Mac asleep (artifacts wait on disk for the
  user to wake up and resume) and for failed runs when Mac is asleep
  (operator may want to debug the checkpoint).
* **Stop with grace** — same as stop, but wait while Mac heartbeat
  stays alive (capped at 10 min). For natural completion + Mac alive:
  give the orchestrator's ``ModelRetriever`` time to SCP adapters off
  the pod, then podStop. ``ModelRetriever`` GET requests refresh the
  heartbeat → grace effectively extends to cover the whole download.
* **Keep alive** — no-op. For ``failed`` + ``KEEP_ON_ERROR=true``,
  reserved for SSH-forensics (Phase 9.A carry-over).

Decision matrix (Phase 11 § 11.1)
---------------------------------

| terminal | mac_alive | volume     | keep_err | outcome                       | action      |
|----------|-----------|------------|----------|-------------------------------|-------------|
| cancel   | *         | *          | *        | terminated_user_stop          | podTerminate|
| failed   | *         | *          | true     | kept_alive_for_debug          | none        |
| failed   | true      | persistent | false    | terminated_safety             | podTerminate|
| failed   | false     | persistent | false    | stopped_for_resume            | podStop     |
| failed   | *         | network    | false    | terminated_safety             | podTerminate|
| complete | true      | persistent | *        | stopped_for_resume_short_grace| grace+podStop|
| complete | false     | persistent | *        | stopped_for_resume            | podStop     |
| complete | *         | network    | *        | terminated_safety             | podTerminate|

Migration from Phase 9.A
------------------------

* ``RUNPOD_AUTO_STOP`` env is **removed**. The new default behaviour
  is "always act on terminal" (per user mandate § 11.1 — no toggle).
  Operators who previously relied on ``AUTO_STOP=false`` to keep
  pods alive for debugging now use ``RUNPOD_KEEP_ON_ERROR=true`` for
  failed runs, or accept that successful runs go to ``stopped_for_resume``
  (recoverable with ``podResume``, costs only storage).
* ``RUNPOD_KEEP_ON_ERROR`` is **kept** (Phase 9.A carry-over). Honours
  failed-only — user-stop always terminates.
* ``RUNPOD_VOLUME_KIND`` is **new**. Set by ``TrainingLauncher._build_job_env``
  to ``"network"`` when the training pod uses a RunPod network volume,
  else ``"persistent"``. Default unset ⇒ treat as ``persistent``
  (current training-flow behaviour).

Idempotency
-----------

Both ``podTerminate`` and ``podStop`` are idempotent on RunPod:
already-gone / already-stopped pods return error strings we recognise
as success (``_ALREADY_GONE_RE`` regex). The decision matrix doesn't
peek at current pod state — it dispatches based on terminal info and
the GraphQL call sorts out "already done" cases.

When Mac calls ``provider.cleanup_pod()`` (canonical Mac-side path)
and our in-pod terminator runs concurrently for the same pod:

* Mac → ``podTerminate`` first → our subsequent ``podStop``/``podTerminate``
  hits the already-gone path → outcome ``ALREADY_TERMINATED``.
* Our → ``podStop`` first → Mac's ``podTerminate`` succeeds (stopped
  pods can still be terminated).

Both orderings converge to "pod gone or sleeping safely".

Provider-agnostic surface
-------------------------

This module is RunPod-specific by design — other providers (Lambda,
single_node) have their own terminal hooks. The decision-function
:func:`decide_terminal_outcome` is pure and provider-agnostic, but
the action dispatch (:meth:`PodTerminator._call_terminate`,
:meth:`PodTerminator._call_stop`) is RunPod GraphQL. Adding a second
provider = sibling module + extracted ``ProviderTerminalHook`` Protocol.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.runner.heartbeat import MacHeartbeat


__all__ = [
    "DEFAULT_RUNPOD_GRAPHQL_URL",
    "PodTerminalOutcome",
    "PodTerminator",
    "decide_terminal_outcome",
    "run_terminal_hook",
]


DEFAULT_RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

#: Regex matching RunPod error fragments that mean "the pod is already
#: in the goal state" (gone, stopped, not running). Treated as success
#: because the pod's intent is satisfied.
_ALREADY_GONE_RE = re.compile(
    r"already.*(stop|exit|terminat)|"
    r"not\s+running|not\s+found|does\s+not\s+exist|no\s+such\s+pod",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Outcome strings
# ---------------------------------------------------------------------------


class PodTerminalOutcome:
    """Outcomes published on the bus for operator dashboards.

    Strings rather than an Enum so payloads round-trip through JSON
    cleanly. Operator alerts grep for these by name.

    Decision-stage outcomes (what we *intended* to do):
    """

    TERMINATED_USER_STOP = "terminated_user_stop"
    """User clicked Stop → ``podTerminate`` (irreversible)."""

    TERMINATED_SAFETY = "terminated_safety"
    """Failed/network-volume → ``podTerminate`` (no resume path)."""

    STOPPED_FOR_RESUME = "stopped_for_resume"
    """Mac asleep → ``podStop`` immediately (artifacts wait for resume)."""

    STOPPED_FOR_RESUME_SHORT_GRACE = "stopped_for_resume_short_grace"
    """Mac alive → wait for retriever, then ``podStop``."""

    KEPT_ALIVE_FOR_DEBUG = "kept_alive_for_debug"
    """``RUNPOD_KEEP_ON_ERROR=true`` on failed → no-op (SSH-forensics)."""

    DISABLED = "disabled"
    """Reserved — currently unused. Phase 11 removes ``AUTO_STOP``,
    so 'disabled' as a decision is gone. Kept in the enum for
    forward-compat (e.g. future ``RUNPOD_TERMINAL_OFF`` toggle)."""

    SKIPPED = "skipped"
    """Decision wanted to act but creds missing (no API key / pod_id)."""

    #: Action-stage outcomes (what actually happened on the GraphQL call).
    #: Reported alongside the decision outcome so operators can see
    #: "we wanted to terminate, the call returned already-gone".

    TERMINATED = "terminated"
    """``podTerminate`` GraphQL returned success."""

    ALREADY_TERMINATED = "already_terminated"
    """``podTerminate`` returned an already-gone marker; idempotent."""

    STOPPED = "stopped"
    """``podStop`` GraphQL returned success."""

    ALREADY_STOPPED = "already_stopped"
    """``podStop`` returned an already-stopped/gone marker; idempotent."""

    FAILED = "failed"
    """All retry attempts exhausted on transient errors."""


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
        # Failed run + Mac alive: nothing to recover, terminate.
        # Failed run + Mac asleep: keep checkpoint accessible, stop.
        if mac_alive:
            return PodTerminalOutcome.TERMINATED_SAFETY
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
    """Calls RunPod GraphQL to terminate / stop the pod, with retries.

    Construct once (env-driven config), inject from the lifespan,
    invoke :meth:`decide_and_act` from the supervisor's reap path.

    Test seams: ``http_client_factory`` and ``sleep`` swappable so
    retry timings don't actually wait in unit tests.
    """

    #: Base grace before podStop on the SHORT_GRACE path. ModelRetriever
    #: typically completes in 30-60s for a few-GB adapter; 60s is the
    #: minimum window. If retriever takes longer, its GET requests
    #: refresh the heartbeat and the grace loop keeps extending.
    GRACE_BASE_SECONDS: float = 60.0

    #: Hard cap on the SHORT_GRACE wait. Even if the heartbeat keeps
    #: refreshing forever, we eventually podStop after this. Protects
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

    def __init__(
        self,
        *,
        graphql_url: str = DEFAULT_RUNPOD_GRAPHQL_URL,
        request_timeout: float = 30.0,
        max_attempts: int = 3,
        http_client_factory: "Callable[[], httpx.AsyncClient] | None" = None,
        sleep: "Callable[[float], Awaitable[None]] | None" = None,
        grace_base_seconds: float | None = None,
        grace_max_seconds: float | None = None,
        grace_tick_seconds: float | None = None,
        heartbeat_retry_attempts: int | None = None,
        heartbeat_retry_tick_seconds: float | None = None,
    ) -> None:
        self._url = graphql_url
        self._timeout = request_timeout
        self._max_attempts = max_attempts
        self._http_factory = http_client_factory or (
            lambda: httpx.AsyncClient(timeout=request_timeout)
        )
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

    async def decide_and_act(
        self,
        *,
        terminal_state: str,
        heartbeat: "MacHeartbeat",
        bus_publish: "Callable[[str, dict[str, Any]], Any]",
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run the decision matrix + dispatch to the right action.

        Returns a dict with ``decision`` (intent) and ``action`` (what
        actually happened on the GraphQL call, or ``None`` for no-op
        outcomes). Caller publishes events from inside this method —
        we emit ``pod_terminal_decision`` after deciding and
        ``pod_stop_attempt`` after acting (Phase 9.A naming kept for
        backwards compat with operator dashboards).
        """
        e = env if env is not None else os.environ
        keep_on_error = (e.get("RUNPOD_KEEP_ON_ERROR") or "").lower() == "true"
        volume_kind = (e.get("RUNPOD_VOLUME_KIND") or "persistent").lower()
        if volume_kind not in ("persistent", "network"):
            # Unknown value ⇒ assume persistent (safe default).
            volume_kind = "persistent"

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
            volume_kind=volume_kind,
            keep_on_error=keep_on_error,
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
                "volume_kind": volume_kind,
                "keep_on_error": keep_on_error,
            },
        )

        # No-op decisions: nothing to dispatch.
        if decision == PodTerminalOutcome.KEPT_ALIVE_FOR_DEBUG:
            return {"decision": decision, "action": None}

        # Decisions that need GraphQL — check creds first.
        api_key = e.get("RUNPOD_API_KEY")
        pod_id = e.get("RUNPOD_POD_ID")
        if not api_key or not pod_id:
            # Decision wanted to act but can't — surface as SKIPPED.
            bus_publish(
                "pod_stop_attempt",
                {
                    "terminal_state": terminal_state,
                    "decision": decision,
                    "action": PodTerminalOutcome.SKIPPED,
                    # Phase 9.A `outcome` field — keep for backwards
                    # compatibility with old dashboards / parsers
                    # (Phase 11 plan § 11.6 — sub-phase 11.B).
                    "outcome": PodTerminalOutcome.SKIPPED,
                },
            )
            return {"decision": decision, "action": PodTerminalOutcome.SKIPPED}

        # Dispatch.
        if decision in (
            PodTerminalOutcome.TERMINATED_USER_STOP,
            PodTerminalOutcome.TERMINATED_SAFETY,
        ):
            action = await self._call_terminate(api_key=api_key, pod_id=pod_id)
        elif decision == PodTerminalOutcome.STOPPED_FOR_RESUME:
            action = await self._call_stop(api_key=api_key, pod_id=pod_id)
        elif decision == PodTerminalOutcome.STOPPED_FOR_RESUME_SHORT_GRACE:
            await self._wait_grace(
                heartbeat=heartbeat, bus_publish=bus_publish,
            )
            action = await self._call_stop(api_key=api_key, pod_id=pod_id)
        else:  # pragma: no cover — defensive; decision matrix exhaustive
            return {"decision": decision, "action": None}

        bus_publish(
            "pod_stop_attempt",
            {
                "terminal_state": terminal_state,
                "decision": decision,
                "action": action,
                # Phase 9.A `outcome` carry — old dashboards parse this.
                # We promote ``action`` to be the canonical field.
                "outcome": action,
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
        1. Heartbeat goes stale (Mac asleep) → break, podStop now.
        2. ``GRACE_MAX_SECONDS`` reached → hard cap, podStop now.
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

    # --- GraphQL calls ----------------------------------------------------

    async def _call_terminate(
        self, *, api_key: str, pod_id: str,
    ) -> str:
        """Send ``podTerminate`` mutation with retries."""
        return await self._call_mutation(
            api_key=api_key,
            pod_id=pod_id,
            mutation_name="podTerminate",
            success_outcome=PodTerminalOutcome.TERMINATED,
            already_outcome=PodTerminalOutcome.ALREADY_TERMINATED,
        )

    async def _call_stop(
        self, *, api_key: str, pod_id: str,
    ) -> str:
        """Send ``podStop`` mutation with retries.

        Phase 11.B: switched from the Phase 9.A always-terminate path
        to a context-sensitive stop on natural completion / Mac asleep.
        ``podStop`` preserves ``/workspace`` so adapters and
        ``MetricsBuffer.jsonl`` remain fetchable on resume.
        """
        return await self._call_mutation(
            api_key=api_key,
            pod_id=pod_id,
            mutation_name="podStop",
            success_outcome=PodTerminalOutcome.STOPPED,
            already_outcome=PodTerminalOutcome.ALREADY_STOPPED,
        )

    async def _call_mutation(
        self,
        *,
        api_key: str,
        pod_id: str,
        mutation_name: str,
        success_outcome: str,
        already_outcome: str,
    ) -> str:
        """Generic GraphQL mutation caller — retries + idempotency.

        Both ``podTerminate`` and ``podStop`` follow the same RunPod
        envelope shape: HTTP 200 + ``"data":{"<mutation_name>":...}``
        on success, ``"errors":[...]`` on failure (where the error
        message may indicate idempotent already-done).
        """
        mutation = (
            f'mutation{{{mutation_name}(input:{{podId:"{pod_id}"}})}}'
        )
        payload = {"query": mutation}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        _last_error: str = ""  # noqa: F841 — kept for future enrichment
        async with self._http_factory() as client:
            for attempt in range(1, self._max_attempts + 1):
                try:
                    response = await client.post(
                        self._url, json=payload, headers=headers,
                    )
                    text = response.text
                except Exception as exc:
                    _last_error = repr(exc)
                else:
                    # GraphQL success: 200 + ``"<mutation_name>"`` token
                    # in body + no ``"errors"`` key.
                    if (
                        response.status_code == 200
                        and f'"{mutation_name}"' in text
                        and '"errors"' not in text
                    ):
                        return success_outcome
                    # Idempotency: "already terminated" / "not running"
                    # / "does not exist" → goal state matches intent.
                    if _ALREADY_GONE_RE.search(text):
                        return already_outcome
                    _last_error = (
                        f"http_status={response.status_code} "
                        f"body={text[:300]}"
                    )

                if attempt < self._max_attempts:
                    # Exponential-ish backoff: 5 s, 10 s, 15 s. Same
                    # as Phase 9.A retry shape.
                    await self._sleep(attempt * 5.0)

        return PodTerminalOutcome.FAILED


# ---------------------------------------------------------------------------
# Convenience wrapper — used as the supervisor's terminal_hook closure
# ---------------------------------------------------------------------------


async def run_terminal_hook(
    terminator: PodTerminator,
    *,
    terminal_state: str,
    heartbeat: "MacHeartbeat",
    bus_publish: "Callable[[str, dict[str, Any]], Any]",
    env: dict[str, str] | None = None,
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
            env=env,
        )
    except Exception as exc:  # pragma: no cover — defensive
        bus_publish(
            "pod_stop_error",
            {"terminal_state": terminal_state, "error": repr(exc)},
        )

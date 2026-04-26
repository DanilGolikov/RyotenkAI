"""RunPod self-stop hook — Python replacement for ``runpod_stop_pod.sh``.

After a job reaches a terminal FSM state (``completed`` / ``failed`` /
``cancelled``), the pod has finished its work and any further uptime
is pure cost. This module sends the RunPod GraphQL ``podStop``
mutation from inside the pod itself — no Mac round-trip needed
(useful when the user closes their laptop right after launch).

The hook honours the same env-var contract as the legacy bash
wrapper so deployment configs don't need to change:

==================================== ====================================
``RUNPOD_AUTO_STOP``                  Master toggle. ``"true"`` → stop on
                                       terminal; anything else → never
                                       stop. Default: ``"true"``.
``RUNPOD_API_KEY``                    GraphQL auth token. Missing →
                                       hook logs and skips (no crash).
``RUNPOD_POD_ID``                     Pod identifier for the mutation.
                                       Missing → skip.
``RUNPOD_KEEP_ON_ERROR``              ``"true"`` keeps the pod alive
                                       on ``failed`` so on-call can
                                       SSH in. ``"false"`` (default)
                                       stops in all terminal states.
==================================== ====================================

Provider-agnostic:
The hook is RunPod-specific by design — other providers (Lambda Labs,
single_node) don't have a "stop the pod" concept that makes sense
in the same shape. When we add another provider with auto-stop,
we'll add a sibling module (``lambdalabs_stopper.py``, etc.) and
let the lifespan choose by env. Phase 5+ may extract a
``ProviderStopHook`` Protocol if a second implementation arrives.

Idempotency:
``podStop`` on RunPod is idempotent — already-stopped pods return
an error string we recognise as success. The retry loop (3 attempts
with exponential backoff) handles transient network errors but
treats "already stopped" as success on any attempt.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = [
    "DEFAULT_RUNPOD_GRAPHQL_URL",
    "PodStopOutcome",
    "PodStopper",
    "should_stop_pod",
]


DEFAULT_RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"

# Recognised "already stopped / not running" error fragments — we
# treat them as success because the goal state matches our intent.
# Lifted verbatim from the legacy bash regex.
_ALREADY_STOPPED_RE = re.compile(
    r"already.*(stop|exit)|not\s+running",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def should_stop_pod(
    *,
    auto_stop: str | None,
    failed: bool,
    keep_on_error: str | None,
) -> bool:
    """Pure decision function — separated so it's trivially testable.

    Mirrors the legacy bash logic:
    - ``RUNPOD_AUTO_STOP != "true"`` → never stop.
    - terminal in ``failed`` AND ``RUNPOD_KEEP_ON_ERROR == "true"``
      → keep alive for debug.
    - otherwise → stop.
    """
    if (auto_stop or "").lower() != "true":
        return False
    return not (failed and (keep_on_error or "false").lower() == "true")


# ---------------------------------------------------------------------------
# Stopper
# ---------------------------------------------------------------------------


class PodStopOutcome:
    """Result categories for :meth:`PodStopper.stop`. Strings rather
    than an enum so they round-trip cleanly through the bus payload."""

    DISABLED = "disabled"  # AUTO_STOP=false or KEEP_ON_ERROR matched
    SKIPPED = "skipped"  # missing API key / pod id
    STOPPED = "stopped"  # GraphQL podStop succeeded
    ALREADY_STOPPED = "already_stopped"
    FAILED = "failed"  # all retries exhausted


class PodStopper:
    """Calls ``podStop`` GraphQL mutation when the FSM lands in terminal.

    Construct lazily (env-driven config), inject from the lifespan,
    invoke :meth:`stop_if_needed` from the supervisor's reap path
    or from ``main.shutdown`` — never from a request handler (the
    HTTP response would block on the GraphQL round-trip).

    The ``http_client_factory`` and ``sleep`` parameters are test
    seams: production runs use ``httpx.AsyncClient`` and
    ``asyncio.sleep``; tests inject mocks so the retry loop runs
    deterministically without 30-s waits.
    """

    def __init__(
        self,
        *,
        graphql_url: str = DEFAULT_RUNPOD_GRAPHQL_URL,
        request_timeout: float = 30.0,
        max_attempts: int = 3,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._url = graphql_url
        self._timeout = request_timeout
        self._max_attempts = max_attempts
        self._http_factory = http_client_factory or (
            lambda: httpx.AsyncClient(timeout=request_timeout)
        )
        self._sleep = sleep or asyncio.sleep

    async def stop_if_needed(
        self,
        *,
        terminal_state: str,
        env: dict[str, str] | None = None,
    ) -> str:
        """Inspect env + terminal state, decide whether to stop, do it.

        Returns one of :class:`PodStopOutcome` so the caller can
        publish a structured event. ``terminal_state`` is the
        ``JobState`` value as a string (``"completed"`` / ``"failed"``
        / ``"cancelled"``); only ``"failed"`` interacts with the
        ``RUNPOD_KEEP_ON_ERROR`` short-circuit.
        """
        e = env if env is not None else os.environ
        failed = terminal_state == "failed"

        if not should_stop_pod(
            auto_stop=e.get("RUNPOD_AUTO_STOP"),
            failed=failed,
            keep_on_error=e.get("RUNPOD_KEEP_ON_ERROR"),
        ):
            return PodStopOutcome.DISABLED

        api_key = e.get("RUNPOD_API_KEY")
        pod_id = e.get("RUNPOD_POD_ID")
        if not api_key or not pod_id:
            return PodStopOutcome.SKIPPED

        return await self._call_graphql(api_key=api_key, pod_id=pod_id)

    async def _call_graphql(self, *, api_key: str, pod_id: str) -> str:
        """Send podStop with retries; return one of the OUTCOME values."""
        # Build the mutation the way the legacy script did — the
        # backend tolerates both inline-quoted and variable-style
        # queries, but we keep the wire shape identical for parity.
        mutation = (
            'mutation{podStop(input:{podId:"' + pod_id + '"}){id}}'
        )
        payload = {"query": mutation}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        # ``last_error`` is intentionally unused-after-loop today —
        # PodStopOutcome doesn't carry the failure message yet. Keeping
        # the local makes future enrichment (publish failure detail
        # to the bus) a one-line change. Mark with `_` to satisfy lint
        # while preserving the doc.
        _last_error: str = ""
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
                    # Success: GraphQL returns ``{"data":{"podStop":{"id":"..."}}}``.
                    if response.status_code == 200 and '"id"' in text:
                        return PodStopOutcome.STOPPED
                    # Idempotency: "already stopped" / "not running"
                    # also counts as success.
                    if _ALREADY_STOPPED_RE.search(text):
                        return PodStopOutcome.ALREADY_STOPPED
                    _last_error = (
                        f"http_status={response.status_code} body={text[:300]}"
                    )

                if attempt < self._max_attempts:
                    # Exponential backoff: 5 s, 10 s, 15 s — same shape
                    # as the legacy bash script's ``sleep $((attempt * 5))``.
                    await self._sleep(attempt * 5.0)

        # All attempts exhausted.
        return PodStopOutcome.FAILED


# ---------------------------------------------------------------------------
# Convenience: one-shot wrapper used by the supervisor's reap path
# ---------------------------------------------------------------------------


async def stop_pod_on_terminal(
    stopper: PodStopper,
    *,
    terminal_state: str,
    bus_publish: Callable[[str, dict[str, Any]], Any],
    env: dict[str, str] | None = None,
) -> None:
    """Call :meth:`PodStopper.stop_if_needed` and publish the outcome.

    Convenience layer used from :class:`Supervisor._reap` so the
    supervisor doesn't carry the env-lookup / event-shape boilerplate.
    Errors are swallowed — a failed pod-stop attempt must not prevent
    the FSM from reaching its terminal state.
    """
    try:
        outcome = await stopper.stop_if_needed(
            terminal_state=terminal_state, env=env,
        )
    except Exception as exc:  # pragma: no cover — defensive
        bus_publish(
            "pod_stop_error",
            {"terminal_state": terminal_state, "error": repr(exc)},
        )
        return

    bus_publish(
        "pod_stop_attempt",
        {"terminal_state": terminal_state, "outcome": outcome},
    )

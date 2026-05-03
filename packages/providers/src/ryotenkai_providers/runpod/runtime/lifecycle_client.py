"""Phase 14.B — RunPod GraphQL impl of :class:`IPodLifecycleClient`.

Owns RunPod-specific transport details that previously lived inside
:mod:`src.runner.pod_terminator`:

* :data:`DEFAULT_RUNPOD_GRAPHQL_URL` — the upstream API endpoint.
* :data:`_ALREADY_GONE_RE` — regex that detects
  RunPod's idempotency markers ("already terminated", "not running",
  "does not exist") in a GraphQL response body.
* The async ``_call_mutation`` retry loop — verbatim move from
  ``pod_terminator.PodTerminator._call_mutation`` (Phase 9.A
  shape preserved bit-for-bit).

Two semantic changes vs the pre-14.B version:

1. **Return type.** Pre-14.B returned a bare ``str`` outcome.
   Phase 14.B returns
   :class:`~src.runner.runtime.lifecycle_client.LifecycleActionResult`
   so retry pressure (``attempts_made``) and forensics
   (``raw_response_excerpt``, ``last_error``) are surfaced to the
   bus event payload.
2. **API key scoping.** Pre-14.B accepted ``api_key=...`` per call.
   Phase 14.B captures it on the client instance — the lifespan
   builds the client once with the key and reuses it across
   terminal hooks. Resource id is per-call (a single client could
   in principle act on multiple pods, though today we only have
   one pod per runner).

Phase 14.B keeps the HTTP client lifecycle short-lived (a fresh
:class:`~httpx.AsyncClient` per call) — matches existing behaviour
exactly. Connection-reuse is an optimization for a later phase.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Final

import httpx

from ryotenkai_shared.constants import PROVIDER_RUNPOD
from ryotenkai_shared.infrastructure.lifecycle import PodTerminalOutcome
from ryotenkai_shared.infrastructure.lifecycle import (
    IPodLifecycleClient,
    LifecycleActionResult,
)

__all__ = [
    "DEFAULT_RUNPOD_GRAPHQL_URL",
    "RunPodPodLifecycleClient",
]


#: RunPod public GraphQL endpoint. Override via constructor for tests
#: that hit a local mock server.
DEFAULT_RUNPOD_GRAPHQL_URL: Final[str] = "https://api.runpod.io/graphql"


#: Regex for RunPod's idempotency markers — same expression that lived
#: inside :mod:`src.runner.pod_terminator` pre-14.B. Matches:
#:
#:   * ``already terminated`` / ``already exited`` / ``already stopped``
#:   * ``not running``
#:   * ``not found`` / ``does not exist`` / ``no such pod``
#:
#: Case-insensitive — RunPod's wording varies across deployments.
_ALREADY_GONE_RE: Final[re.Pattern[str]] = re.compile(
    r"already.*(stop|exit|terminat)|"
    r"not\s+running|not\s+found|does\s+not\s+exist|no\s+such\s+pod",
    re.IGNORECASE,
)


# Phase 14.B § 1.6 — outcome strings for ``resume`` are NOT promoted
# to :class:`PodTerminalOutcome` (yet). They live here as private
# constants so the impl is consistent and tests can pin the
# vocabulary. See § OQ-2 of the plan.
_RESUME_SUCCESS_OUTCOME: Final[str] = "resumed"
_RESUME_ALREADY_OUTCOME: Final[str] = "already_running"


class RunPodPodLifecycleClient:
    """RunPod GraphQL client — async, retry-aware, idempotent-marker-aware.

    Conforms to
    :class:`~src.runner.runtime.lifecycle_client.IPodLifecycleClient`.
    Used by :class:`~src.runner.pod_terminator.PodTerminator` after
    Phase 14.B's lifespan rewrite picks the right client based on
    :data:`~src.constants.RUNTIME_PROVIDER_ENV_VAR`.

    Test seams:
        * ``http_client_factory`` — swap :class:`~httpx.AsyncClient`
          for a stub. Default factory creates a fresh client per
          mutation call (matches pre-14.B behaviour).
        * ``sleep`` — swap ``asyncio.sleep`` so retry backoffs don't
          actually wait in unit tests.
    """

    def __init__(
        self,
        *,
        api_key: str,
        graphql_url: str = DEFAULT_RUNPOD_GRAPHQL_URL,
        request_timeout: float = 30.0,
        max_attempts: int = 3,
        http_client_factory: Callable[[], httpx.AsyncClient] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._api_key = api_key
        self._url = graphql_url
        self._timeout = request_timeout
        self._max_attempts = max_attempts
        self._http_factory = http_client_factory or (
            lambda: httpx.AsyncClient(timeout=request_timeout)
        )
        self._sleep = sleep or asyncio.sleep

    @property
    def provider_name(self) -> str:
        return PROVIDER_RUNPOD

    async def terminate(self, *, resource_id: str) -> LifecycleActionResult:
        """Send ``podTerminate`` mutation with retries."""
        return await self._call_mutation(
            pod_id=resource_id,
            mutation_name="podTerminate",
            success_outcome=PodTerminalOutcome.TERMINATED,
            already_outcome=PodTerminalOutcome.ALREADY_TERMINATED,
        )

    async def pause(self, *, resource_id: str) -> LifecycleActionResult:
        """Send ``podStop`` mutation with retries.

        Phase 11.B introduced the stop-vs-terminate distinction —
        ``podStop`` preserves ``/workspace`` so adapters and the
        ``MetricsBuffer.jsonl`` remain fetchable on resume.
        """
        return await self._call_mutation(
            pod_id=resource_id,
            mutation_name="podStop",
            success_outcome=PodTerminalOutcome.STOPPED,
            already_outcome=PodTerminalOutcome.ALREADY_STOPPED,
        )

    async def resume(self, *, resource_id: str) -> LifecycleActionResult:
        """Send ``podResume`` mutation with retries.

        Note:
            The runner does NOT self-resume today — Mac wakes the pod
            after operator interaction. This method exists for
            Protocol completeness; landing the contract is cheap and
            unlocks future "self-resume after capacity reservation"
            scenarios. Outcomes ``"resumed"`` / ``"already_running"``
            are RunPod-private until promoted to
            :class:`PodTerminalOutcome` in a later phase
            (Phase 14.B § 1.2 + § OQ-2).
        """
        return await self._call_mutation(
            pod_id=resource_id,
            mutation_name="podResume",
            success_outcome=_RESUME_SUCCESS_OUTCOME,
            already_outcome=_RESUME_ALREADY_OUTCOME,
        )

    async def _call_mutation(
        self,
        *,
        pod_id: str,
        mutation_name: str,
        success_outcome: str,
        already_outcome: str,
    ) -> LifecycleActionResult:
        """Generic GraphQL mutation caller — retries + idempotency.

        ``podTerminate`` / ``podStop`` / ``podResume`` all follow
        RunPod's single-mutation envelope: HTTP 200 +
        ``"data":{"<mutation_name>":...}`` on success,
        ``"errors":[...]`` on failure (where the message may indicate
        an idempotent already-done).

        Returns:
            :class:`LifecycleActionResult` with ``outcome`` set to
            either ``success_outcome``, ``already_outcome``, or
            :data:`PodTerminalOutcome.FAILED` if all retries exhaust.
        """
        mutation = (
            f'mutation{{{mutation_name}(input:{{podId:"{pod_id}"}})}}'
        )
        payload = {"query": mutation}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        last_error: str | None = None
        last_excerpt: str | None = None
        attempts_made = 0

        async with self._http_factory() as client:
            for attempt in range(1, self._max_attempts + 1):
                attempts_made = attempt
                try:
                    response = await client.post(
                        self._url, json=payload, headers=headers,
                    )
                    text = response.text
                except Exception as exc:
                    last_error = repr(exc)
                else:
                    last_excerpt = text
                    # GraphQL success: 200 + ``"<mutation_name>"`` token
                    # in body + no ``"errors"`` key.
                    if (
                        response.status_code == 200
                        and f'"{mutation_name}"' in text
                        and '"errors"' not in text
                    ):
                        return LifecycleActionResult(
                            outcome=success_outcome,
                            attempts_made=attempts_made,
                            raw_response_excerpt=text,
                        )
                    # Idempotency: "already terminated" / "not running"
                    # / "does not exist" → goal state matches intent.
                    if _ALREADY_GONE_RE.search(text):
                        return LifecycleActionResult(
                            outcome=already_outcome,
                            attempts_made=attempts_made,
                            raw_response_excerpt=text,
                        )
                    last_error = (
                        f"http_status={response.status_code} "
                        f"body={text[:300]}"
                    )

                if attempt < self._max_attempts:
                    # Exponential-ish backoff: 5 s, 10 s, 15 s. Same
                    # as Phase 9.A retry shape.
                    await self._sleep(attempt * 5.0)

        return LifecycleActionResult(
            outcome=PodTerminalOutcome.FAILED,
            attempts_made=attempts_made,
            last_error=last_error,
            raw_response_excerpt=last_excerpt,
        )


# Static guarantee — fail fast at module import if we ever drift from
# the Protocol shape (e.g. forgot to add ``resume`` after refactor).
_runtime_check: IPodLifecycleClient = RunPodPodLifecycleClient(api_key="_static_check_only")  # noqa: F841

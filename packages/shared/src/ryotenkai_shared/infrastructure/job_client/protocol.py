"""Phase 4 — Provider-agnostic ``IJobClient`` Protocol.

Extracted additively from
:class:`ryotenkai_shared.utils.clients.job_client.JobClient`. The
existing concrete class IS the HTTP+WS client that Mac talks to via
SSH tunnel. The Protocol exists so component / property / chaos tests
can program HTTP-shape failure modes (timeout, 429, 4xx, network
partition) deterministically without spinning up the runner uvicorn.

Scope: only the **REST** subset that's actually called from Mac-side
control plane today (health, submit, status, request_stop, heartbeat).
WS subscription is intentionally excluded — it's a separate concern
covered by Phase 5 ``WSReconnectStorm`` chaos work and the existing
:class:`JobClient` event-iterator already lives in production.

**Definition-only**: no production class implements ``IJobClient``
yet. ``JobClient`` happens to expose the same method names; the
compliance test parametrizes over ``[fake, real]`` and ``real`` is
``pytest.skip``-ed until a thin adapter lands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class JobClientProtocolError(RuntimeError):
    """Base error from :class:`IJobClient` operations (mirrors :class:`JobClientError`)."""


class JobClientNotFoundError(JobClientProtocolError):
    """Server returned 404 for the requested job id."""


class JobClientRateLimitedError(JobClientProtocolError):
    """Server returned 429 — caller should back off."""


class JobClientNetworkError(JobClientProtocolError):
    """Transport-level failure (connection reset / timeout / DNS / etc.)."""


@dataclass(frozen=True)
class JobSubmissionResult:
    """Server response on ``submit_job``."""

    job_id: str
    sequence: int = 0
    offset: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class IJobClient(Protocol):
    """Async REST surface for the in-pod runner."""

    async def health_check(self) -> bool:
        """Return whether the runner is reachable and healthy."""
        ...

    async def submit_job(
        self,
        job_spec: dict[str, Any],
        *,
        plugins_payload: bytes | None = None,
        timeout: float | None = None,
    ) -> JobSubmissionResult:
        """Submit a job; runner replies 202 with sequence/offset.

        Raises:
            JobClientRateLimitedError: server returned 429.
            JobClientNetworkError: transport failure.
            JobClientProtocolError: any other non-2xx response.
        """
        ...

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """Fetch the latest snapshot for ``job_id``.

        Raises:
            JobClientNotFoundError: server returned 404.
            JobClientRateLimitedError: server returned 429.
            JobClientNetworkError: transport failure.
        """
        ...

    async def request_stop(
        self,
        job_id: str,
        *,
        grace_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Ask the runner to stop ``job_id`` gracefully.

        Raises:
            JobClientNotFoundError: server returned 404.
            JobClientRateLimitedError: server returned 429.
            JobClientNetworkError: transport failure.
        """
        ...

    async def send_heartbeat(self, *, ttl_seconds: float | None = None) -> bool:
        """Tell the runner the Mac control-plane is still active.

        Returns ``True`` on a 200 response, ``False`` on transport
        error or any non-200 status (fire-and-forget contract).
        """
        ...


__all__ = [
    "IJobClient",
    "JobClientNetworkError",
    "JobClientNotFoundError",
    "JobClientProtocolError",
    "JobClientRateLimitedError",
    "JobSubmissionResult",
]

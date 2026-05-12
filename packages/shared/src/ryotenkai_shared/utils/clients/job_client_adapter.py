"""Phase 5 — real :class:`IJobClient` adapter wrapping :class:`JobClient`.

Additive only. :class:`ryotenkai_shared.utils.clients.job_client.JobClient`
is shaped close to :class:`IJobClient` (same method names, async, REST)
but :meth:`JobClient.submit_job` returns a raw dict while the Protocol
returns a typed :class:`JobSubmissionResult`. This adapter is a thin
wrapper that normalises the return shape and translates the underlying
exceptions onto the typed Protocol error hierarchy.

The compliance test (``tests/contract/protocol_compliance/test_job_client_compliance.py``)
parametrizes ``real`` over this adapter when ``RYOTENKAI_LIVE=1`` and
the runner URL is reachable.
"""

from __future__ import annotations

from typing import Any

from ryotenkai_shared.infrastructure.job_client import (
    IJobClient,
    JobClientNetworkError,
    JobClientNotFoundError,
    JobClientProtocolError,
    JobSubmissionResult,
)
from ryotenkai_shared.utils.clients.job_client import (
    JobClient,
    JobClientError,
    JobNotFoundError,
)


class HTTPJobClientAdapter:
    """Adapter that satisfies :class:`IJobClient` by delegating to :class:`JobClient`."""

    def __init__(self, *, base_url: str, request_timeout: float = 30.0) -> None:
        self._inner = JobClient(base_url=base_url, request_timeout=request_timeout)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def health_check(self) -> bool:
        return await self._inner.health_check()

    async def submit_job(
        self,
        job_spec: dict[str, Any],
        *,
        plugins_payload: bytes | None = None,
        timeout: float | None = None,
    ) -> JobSubmissionResult:
        try:
            body = await self._inner.submit_job(
                job_spec, plugins_payload=plugins_payload, timeout=timeout,
            )
        except JobClientError as exc:
            raise JobClientNetworkError(str(exc)) from exc
        return JobSubmissionResult(
            job_id=str(body.get("job_id", "")),
            sequence=int(body.get("sequence", 0)),
            offset=int(body.get("offset", 0)),
            extras={
                k: v for k, v in body.items()
                if k not in {"job_id", "sequence", "offset"}
            },
        )

    async def get_status(self, job_id: str) -> dict[str, Any]:
        try:
            return await self._inner.get_status(job_id)
        except JobNotFoundError as exc:
            raise JobClientNotFoundError(str(exc)) from exc
        except JobClientError as exc:
            raise JobClientNetworkError(str(exc)) from exc

    async def request_stop(
        self, job_id: str, *, grace_seconds: float | None = None,
    ) -> dict[str, Any]:
        try:
            return await self._inner.request_stop(job_id, grace_seconds=grace_seconds)
        except JobNotFoundError as exc:
            raise JobClientNotFoundError(str(exc)) from exc
        except JobClientError as exc:
            raise JobClientNetworkError(str(exc)) from exc

    async def send_heartbeat(self, *, ttl_seconds: float | None = None) -> bool:
        try:
            return await self._inner.send_heartbeat(ttl_seconds=ttl_seconds)
        except JobClientError:
            return False


# Static guarantee.
_runtime_check: IJobClient = HTTPJobClientAdapter(base_url="http://127.0.0.1:0")


__all__ = ["HTTPJobClientAdapter"]

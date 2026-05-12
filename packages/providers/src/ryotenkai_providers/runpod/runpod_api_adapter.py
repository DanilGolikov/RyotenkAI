"""Phase 5 — real :class:`IRunPodAPI` adapter (HTTP transport).

Additive only. The adapter is a thin httpx wrapper that speaks the
REST shape the ``fake-runpod`` sidecar exposes
(``GET /api/pods``, ``GET /api/pods/{id}``, ``POST /api/pods/{id}/{verb}``).
This lets the live-protocol-compliance lane drive the real adapter
against the hermetic stack's sidecar without a real RunPod account —
the highest-leverage Phase 5 real-adapter wiring.

Production code keeps using the SDK-backed
:class:`ryotenkai_providers.runpod.training.api_client.RunPodAPIClient`.
This adapter is parallel infrastructure: nothing imports it today
except the compliance test under
``tests/contract/protocol_compliance/test_runpod_api_compliance.py``
when ``RYOTENKAI_LIVE=1``.
"""

from __future__ import annotations

from typing import Any

import httpx

from ryotenkai_shared.infrastructure.runpod_api import (
    IRunPodAPI,
    RunPodInfo,
    RunPodLifecycleResponse,
    RunPodPartialResponseError,
    RunPodRateLimitedError,
    RunPodTransientError,
)


def _decode_info(payload: dict[str, Any]) -> RunPodInfo:
    return RunPodInfo(
        pod_id=str(payload["pod_id"]),
        desired_status=str(payload["desired_status"]),
        runtime_status=payload.get("runtime_status"),
        ssh_host=payload.get("ssh_host"),
        ssh_port=payload.get("ssh_port"),
        cost_per_hr=payload.get("cost_per_hr"),
        machine_id=payload.get("machine_id"),
        extras=payload.get("extras", {}) or {},
    )


def _decode_lifecycle(payload: dict[str, Any]) -> RunPodLifecycleResponse:
    return RunPodLifecycleResponse(
        outcome=str(payload.get("outcome", "ok")),
        message=str(payload.get("message", "")),
    )


def _raise_for_status(response: httpx.Response) -> None:
    """Map sidecar status codes onto :class:`RunPodAPIError` subclasses."""
    if response.status_code == 429:
        raise RunPodRateLimitedError(response.text or "rate limited")
    if response.status_code == 502:
        raise RunPodPartialResponseError(response.text or "partial response")
    if response.status_code == 503:
        raise RunPodTransientError(response.text or "transient")


class HTTPRunPodAPIAdapter:
    """Thin REST adapter that satisfies :class:`IRunPodAPI`.

    Constructed with a ``base_url`` (e.g. the fake-runpod sidecar URL
    in tests, or a future real proxy URL in production) and an
    ``httpx.AsyncClient``-compatible timeout.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        client = self._client or httpx.AsyncClient(timeout=self._timeout)
        try:
            response = await client.request(
                method, self._base_url + path, params=params, json=json,
            )
        finally:
            if self._owns_client and self._client is None:
                await client.aclose()
        _raise_for_status(response)
        return response

    async def find_pod(self, pod_id: str) -> RunPodInfo | None:
        response = await self._request("GET", f"/api/pods/{pod_id}")
        if response.status_code == 404:
            return None
        return _decode_info(response.json())

    async def list_pods(self) -> list[RunPodInfo]:
        response = await self._request("GET", "/api/pods")
        return [_decode_info(item) for item in response.json()]

    async def stop_pod(self, pod_id: str) -> RunPodLifecycleResponse:
        response = await self._request("POST", f"/api/pods/{pod_id}/stop")
        return _decode_lifecycle(response.json())

    async def terminate_pod(self, pod_id: str) -> RunPodLifecycleResponse:
        response = await self._request("POST", f"/api/pods/{pod_id}/terminate")
        return _decode_lifecycle(response.json())

    async def resume_pod(self, pod_id: str) -> RunPodLifecycleResponse:
        response = await self._request("POST", f"/api/pods/{pod_id}/resume")
        return _decode_lifecycle(response.json())


# Static guarantee — fail fast at module import if the adapter drifts
# from the Protocol shape.
_runtime_check: IRunPodAPI = HTTPRunPodAPIAdapter("http://127.0.0.1:0")


__all__ = ["HTTPRunPodAPIAdapter"]

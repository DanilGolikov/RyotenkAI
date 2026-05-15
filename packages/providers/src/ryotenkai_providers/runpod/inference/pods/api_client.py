"""
RunPod client for inference Pods + legacy Network Volumes.

Scope:
- Pod lifecycle via official Python SDK
- Legacy network volume operations via REST
- No business logic (no naming, no lifecycle policy decisions)

Phase A2 Batch 11 (2026-05-15): migrated from ``Result[T, ProviderError]``
to raise-based typed exceptions. The public
``RunPodPodInferenceProvider`` facade catches typed errors and
translates them back into ``Result`` for
:class:`IInferenceProvider` Protocol conformance.
"""

from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ryotenkai_providers.constants import (
    HTTP_GET,
    HTTP_POST,
    HTTP_STATUS_ERROR_THRESHOLD,
    RETRY_BACKOFF_FACTOR,
    RETRY_TOTAL_ATTEMPTS,
    TIMEOUT_REQUEST_DEFAULT,
    TIMEOUT_REQUEST_LONG,
    TIMEOUT_REQUEST_SHORT,
)
from ryotenkai_providers.runpod.sdk_adapter import RunPodSDKClient
from ryotenkai_shared.errors import ProviderUnavailableError
from ryotenkai_shared.utils.logger import logger

from .constants import RUNPOD_REST_API_BASE_URL

_UNEXPECTED_RESPONSE_CODE = "RUNPOD_UNEXPECTED_RESPONSE"


class RunPodPodsRESTClient:
    """SDK-backed Pod client + REST-backed legacy Network Volume client.

    Phase A2 Batch 11: raise-based contract. Methods return ``T`` and
    raise :class:`ProviderUnavailableError` (with the legacy code on
    the exception's ``context`` dict) on failure.
    """

    def __init__(self, *, api_key: str, api_base_url: str = RUNPOD_REST_API_BASE_URL):
        self.api_base = str(api_base_url).rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._sdk = RunPodSDKClient(api_key=api_key)

        self.session = requests.Session()
        retry_strategy = Retry(
            total=RETRY_TOTAL_ATTEMPTS,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            # IMPORTANT:
            # - We intentionally do NOT retry POST automatically to avoid duplicating non-idempotent creates
            #   (network volumes / pods). Provider-level code implements safer "retry + re-list" logic.
            allowed_methods=[HTTP_GET, "PATCH", "DELETE"],
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry_strategy))

        logger.debug(f"🔗 RunPodPodsRESTClient initialized: {self.api_base} (retry: 3 attempts, backoff: 2.0s)")

    # ---------------------------------------------------------------------
    # Generic request helper
    # ---------------------------------------------------------------------

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        timeout_seconds: int = TIMEOUT_REQUEST_DEFAULT,
    ) -> Any:
        url = f"{self.api_base}{path}"
        try:
            resp = self.session.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=payload,
                timeout=timeout_seconds,
            )
        except Exception as exc:
            raise ProviderUnavailableError(
                detail=f"RunPod REST request failed: {method} {url}: {exc!s}",
                context={"code": "RUNPOD_REST_REQUEST_FAILED", "method": method, "url": url},
                cause=exc,
            ) from exc

        # Try to parse body even on errors.
        text = (resp.text or "").strip()
        data: Any = None
        if text:
            try:
                data = resp.json()
            except Exception:
                data = text

        if resp.status_code >= HTTP_STATUS_ERROR_THRESHOLD:
            raise ProviderUnavailableError(
                detail=f"RunPod REST HTTP {resp.status_code} for {method} {url}: {data!r}",
                context={
                    "code": "RUNPOD_REST_HTTP_ERROR",
                    "status_code": resp.status_code,
                    "method": method,
                    "url": url,
                },
            )

        return data

    # ---------------------------------------------------------------------
    # Network volumes
    # ---------------------------------------------------------------------

    def list_network_volumes(self) -> list[dict[str, Any]]:
        val = self._request_json(HTTP_GET, "/networkvolumes", timeout_seconds=TIMEOUT_REQUEST_DEFAULT)
        if isinstance(val, list):
            return val
        raise ProviderUnavailableError(
            detail=f"Unexpected networkvolumes response type: {type(val).__name__}",
            context={"code": _UNEXPECTED_RESPONSE_CODE},
        )

    def get_network_volume(self, *, network_volume_id: str) -> dict[str, Any]:
        val = self._request_json(
            HTTP_GET, f"/networkvolumes/{network_volume_id}", timeout_seconds=TIMEOUT_REQUEST_DEFAULT
        )
        if isinstance(val, dict):
            return val
        raise ProviderUnavailableError(
            detail=f"Unexpected get_network_volume response type: {type(val).__name__}",
            context={"code": _UNEXPECTED_RESPONSE_CODE},
        )

    def create_network_volume(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        val = self._request_json(HTTP_POST, "/networkvolumes", payload=payload, timeout_seconds=TIMEOUT_REQUEST_SHORT)
        if isinstance(val, dict):
            return val
        raise ProviderUnavailableError(
            detail=f"Unexpected create_network_volume response type: {type(val).__name__}",
            context={"code": _UNEXPECTED_RESPONSE_CODE},
        )

    # ---------------------------------------------------------------------
    # Pods (SDK-backed)
    # ---------------------------------------------------------------------

    def list_pods(self, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._sdk.list_pods(params=params)

    def get_pod(self, *, pod_id: str) -> dict[str, Any]:
        return self._sdk.get_pod(pod_id=pod_id)

    def create_pod(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        return self._sdk.create_pod_from_payload(payload=payload)

    def start_pod(self, *, pod_id: str) -> None:
        self._sdk.start_pod(pod_id=pod_id)

    def stop_pod(self, *, pod_id: str) -> None:
        self._sdk.stop_pod(pod_id=pod_id)

    def delete_pod(self, *, pod_id: str) -> None:
        self._sdk.delete_pod(pod_id=pod_id)


__all__ = [
    "RunPodPodsRESTClient",
]

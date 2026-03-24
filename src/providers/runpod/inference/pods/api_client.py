"""
RunPod REST API client for Pods + Network Volumes.

Scope:
- Low-level HTTP requests with retries and timeouts
- No business logic (no naming, no lifecycle policy decisions)
"""

from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.providers.constants import (
    HTTP_GET,
    HTTP_POST,
    HTTP_STATUS_ERROR_THRESHOLD,
    RETRY_BACKOFF_FACTOR,
    RETRY_TOTAL_ATTEMPTS,
    TIMEOUT_REQUEST_DEFAULT,
    TIMEOUT_REQUEST_LONG,
    TIMEOUT_REQUEST_SHORT,
)
from src.utils.logger import logger
from src.utils.result import Err, Ok, ProviderError, Result

from .constants import RUNPOD_REST_API_BASE_URL

_UNEXPECTED_RESPONSE_CODE = "RUNPOD_UNEXPECTED_RESPONSE"


class RunPodPodsRESTClient:
    """REST client for `https://rest.runpod.io/v1` (Pods + Network Volumes)."""

    def __init__(self, *, api_key: str, api_base_url: str = RUNPOD_REST_API_BASE_URL):
        self.api_base = str(api_base_url).rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

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
    ) -> Result[Any, ProviderError]:
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
        except Exception as e:
            return Err(
                ProviderError(
                    message=f"RunPod REST request failed: {method} {url}: {e!s}",
                    code="RUNPOD_REST_REQUEST_FAILED",
                    details={"method": method, "url": url},
                )
            )

        # Try to parse body even on errors.
        text = (resp.text or "").strip()
        data: Any = None
        if text:
            try:
                data = resp.json()
            except Exception:
                data = text

        if resp.status_code >= HTTP_STATUS_ERROR_THRESHOLD:
            return Err(
                ProviderError(
                    message=f"RunPod REST HTTP {resp.status_code} for {method} {url}: {data!r}",
                    code="RUNPOD_REST_HTTP_ERROR",
                    details={"status_code": resp.status_code, "method": method, "url": url},
                )
            )

        return Ok(data)

    # ---------------------------------------------------------------------
    # Network volumes
    # ---------------------------------------------------------------------

    def list_network_volumes(self) -> Result[list[dict[str, Any]], ProviderError]:
        res = self._request_json(HTTP_GET, "/networkvolumes", timeout_seconds=TIMEOUT_REQUEST_DEFAULT)
        if res.is_failure():
            return res  # type: ignore[return-value]
        val = res.unwrap()
        if isinstance(val, list):
            return Ok(val)
        return Err(
            ProviderError(
                message=f"Unexpected networkvolumes response type: {type(val).__name__}",
                code=_UNEXPECTED_RESPONSE_CODE,
            )
        )

    def get_network_volume(self, *, network_volume_id: str) -> Result[dict[str, Any], ProviderError]:
        res = self._request_json(
            HTTP_GET, f"/networkvolumes/{network_volume_id}", timeout_seconds=TIMEOUT_REQUEST_DEFAULT
        )
        if res.is_failure():
            return res  # type: ignore[return-value]
        val = res.unwrap()
        if isinstance(val, dict):
            return Ok(val)
        return Err(
            ProviderError(
                message=f"Unexpected get_network_volume response type: {type(val).__name__}",
                code=_UNEXPECTED_RESPONSE_CODE,
            )
        )

    def create_network_volume(self, *, payload: dict[str, Any]) -> Result[dict[str, Any], ProviderError]:
        res = self._request_json(HTTP_POST, "/networkvolumes", payload=payload, timeout_seconds=TIMEOUT_REQUEST_SHORT)
        if res.is_failure():
            return res  # type: ignore[return-value]
        val = res.unwrap()
        if isinstance(val, dict):
            return Ok(val)
        return Err(
            ProviderError(
                message=f"Unexpected create_network_volume response type: {type(val).__name__}",
                code=_UNEXPECTED_RESPONSE_CODE,
            )
        )

    # ---------------------------------------------------------------------
    # Pods
    # ---------------------------------------------------------------------

    def list_pods(self, *, params: dict[str, Any] | None = None) -> Result[list[dict[str, Any]], ProviderError]:
        res = self._request_json(HTTP_GET, "/pods", params=params, timeout_seconds=TIMEOUT_REQUEST_DEFAULT)
        if res.is_failure():
            return res  # type: ignore[return-value]
        val = res.unwrap()
        if isinstance(val, list):
            return Ok(val)
        return Err(
            ProviderError(
                message=f"Unexpected pods response type: {type(val).__name__}",
                code=_UNEXPECTED_RESPONSE_CODE,
            )
        )

    def get_pod(self, *, pod_id: str) -> Result[dict[str, Any], ProviderError]:
        res = self._request_json(HTTP_GET, f"/pods/{pod_id}", timeout_seconds=TIMEOUT_REQUEST_DEFAULT)
        if res.is_failure():
            return res  # type: ignore[return-value]
        val = res.unwrap()
        if isinstance(val, dict):
            return Ok(val)
        return Err(
            ProviderError(
                message=f"Unexpected get_pod response type: {type(val).__name__}",
                code=_UNEXPECTED_RESPONSE_CODE,
            )
        )

    def create_pod(self, *, payload: dict[str, Any]) -> Result[dict[str, Any], ProviderError]:
        res = self._request_json(HTTP_POST, "/pods", payload=payload, timeout_seconds=TIMEOUT_REQUEST_LONG)
        if res.is_failure():
            return res  # type: ignore[return-value]
        val = res.unwrap()
        if isinstance(val, dict):
            return Ok(val)
        return Err(
            ProviderError(
                message=f"Unexpected create_pod response type: {type(val).__name__}",
                code=_UNEXPECTED_RESPONSE_CODE,
            )
        )

    def start_pod(self, *, pod_id: str) -> Result[None, ProviderError]:
        res = self._request_json(HTTP_POST, f"/pods/{pod_id}/start", timeout_seconds=TIMEOUT_REQUEST_SHORT)
        if res.is_failure():
            return Err(res.unwrap_err())  # type: ignore[union-attr]
        return Ok(None)

    def stop_pod(self, *, pod_id: str) -> Result[None, ProviderError]:
        res = self._request_json(HTTP_POST, f"/pods/{pod_id}/stop", timeout_seconds=TIMEOUT_REQUEST_SHORT)
        if res.is_failure():
            return Err(res.unwrap_err())  # type: ignore[union-attr]
        return Ok(None)

    def delete_pod(self, *, pod_id: str) -> Result[None, ProviderError]:
        res = self._request_json("DELETE", f"/pods/{pod_id}", timeout_seconds=TIMEOUT_REQUEST_SHORT)
        if res.is_failure():
            return Err(res.unwrap_err())  # type: ignore[union-attr]
        return Ok(None)


__all__ = [
    "RunPodPodsRESTClient",
]

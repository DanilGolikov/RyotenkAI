"""
RunPod control layer.

Training backend policy:
- create_pod: SDK-backed API
- query_pod: SDK-backed API
- terminate_pod: SDK-backed API
- get_ssh_info: SDK-backed API via query_pod + PodSnapshot parsing

Inference backend policy:
- start/stop/delete/get: SDK-backed Pod API
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol

from src.providers.runpod.models import PodSnapshot
from src.utils.logger import logger
from src.utils.result import Err, Ok, ProviderError, Result

_CREATE_POD_MAX_RETRIES = 3
_CREATE_POD_RETRY_DELAY_S = 10

_TRANSIENT_MARKERS = (
    "no longer any instances available",
    "no instances available",
    "does not have the resources",
    "try again",
    "rate limit",
    "timeout",
    "503",
    "502",
)

if TYPE_CHECKING:
    from src.config.providers.runpod import RunPodProviderConfig


class _TrainingApiProtocol(Protocol):
    def create_pod(
        self,
        config: RunPodProviderConfig,
        *,
        pod_name: str | None = None,
    ) -> Result[dict[str, Any], ProviderError]: ...

    def query_pod(self, pod_id: str) -> Result[dict[str, Any], ProviderError]: ...

    def terminate_pod(self, pod_id: str) -> Result[None, ProviderError]: ...

    def get_ssh_info(self, pod_id: str) -> Result[dict[str, Any], ProviderError]: ...

    def extract_exposed_ssh_info(
        self,
        pod_data: dict[str, Any] | None,
        *,
        pod_id: str | None = None,
    ) -> Result[dict[str, Any], ProviderError]: ...


class _InferenceApiProtocol(Protocol):
    def get_pod(self, *, pod_id: str) -> Result[dict[str, Any], ProviderError]: ...

    def start_pod(self, *, pod_id: str) -> Result[None, ProviderError]: ...

    def stop_pod(self, *, pod_id: str) -> Result[None, ProviderError]: ...

    def delete_pod(self, *, pod_id: str) -> Result[None, ProviderError]: ...


class RunPodTrainingPodControl:
    """SDK-backed control for training pods."""

    def __init__(self, *, api: _TrainingApiProtocol):
        self._api = api

    def create_pod(
        self,
        *,
        config: RunPodProviderConfig,
        pod_name: str,
    ) -> Result[dict[str, Any], ProviderError]:
        """Create pod with retries for transient capacity errors."""
        last_err: ProviderError | None = None
        for attempt in range(1, _CREATE_POD_MAX_RETRIES + 1):
            result = self._api.create_pod(config=config, pod_name=pod_name)
            if result.is_success():
                return result

            last_err = result.unwrap_err()  # type: ignore[union-attr]
            if not self._is_transient_error(last_err):
                return result

            if attempt < _CREATE_POD_MAX_RETRIES:
                logger.warning(
                    "[POD_CONTROL] Create pod failed (attempt %d/%d), retrying in %ds: %s",
                    attempt,
                    _CREATE_POD_MAX_RETRIES,
                    _CREATE_POD_RETRY_DELAY_S,
                    last_err.message,
                )
                time.sleep(_CREATE_POD_RETRY_DELAY_S)

        logger.error("[POD_CONTROL] Create pod failed after %d attempts", _CREATE_POD_MAX_RETRIES)
        return Err(last_err)  # type: ignore[arg-type]

    @staticmethod
    def _is_transient_error(err: ProviderError) -> bool:
        msg = err.message.lower()
        return any(marker in msg for marker in _TRANSIENT_MARKERS)

    def query_pod(self, pod_id: str) -> Result[dict[str, Any], ProviderError]:
        return self._api.query_pod(pod_id)

    def query_pod_snapshot(self, pod_id: str) -> Result[PodSnapshot, ProviderError]:
        """Query pod and return a typed snapshot."""
        result = self._api.query_pod(pod_id)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]
        return Ok(PodSnapshot.from_graphql(result.unwrap()))

    def extract_exposed_ssh_info(
        self,
        pod_data: dict[str, Any] | None,
        *,
        pod_id: str | None = None,
    ) -> Result[dict[str, Any], ProviderError]:
        return self._api.extract_exposed_ssh_info(pod_data, pod_id=pod_id)

    def get_ssh_info(self, pod_id: str) -> Result[dict[str, Any], ProviderError]:
        snapshot_result = self.query_pod_snapshot(pod_id)
        if snapshot_result.is_success():
            snapshot = snapshot_result.unwrap()
            if snapshot.ssh_endpoint is not None:
                return Ok({"host": snapshot.ssh_endpoint.host, "port": snapshot.ssh_endpoint.port})
        return self._api.get_ssh_info(pod_id)

    def terminate_pod(self, pod_id: str) -> Result[None, ProviderError]:
        return self._api.terminate_pod(pod_id)


class RunPodInferencePodControl:
    """SDK-backed control for inference pod start/stop/delete operations."""

    def __init__(self, *, api: _InferenceApiProtocol):
        self._api = api

    def start_pod(self, *, pod_id: str) -> Result[None, ProviderError]:
        return self._api.start_pod(pod_id=pod_id)

    def get_pod(self, *, pod_id: str) -> Result[dict[str, Any], ProviderError]:
        return self._api.get_pod(pod_id=pod_id)

    def stop_pod(self, *, pod_id: str) -> Result[None, ProviderError]:
        return self._api.stop_pod(pod_id=pod_id)

    def delete_pod(self, *, pod_id: str) -> Result[None, ProviderError]:
        return self._api.delete_pod(pod_id=pod_id)


__all__ = [
    "RunPodInferencePodControl",
    "RunPodTrainingPodControl",
]

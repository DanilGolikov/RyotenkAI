"""
RunPod control layer.

Training backend policy:
- create_pod: SDK-backed API
- query_pod: SDK-backed API
- terminate_pod: SDK-backed API
- get_ssh_info: SDK-backed API via query_pod + PodSnapshot parsing

Inference backend policy:
- start/stop/delete/get: SDK-backed Pod API

Phase A2 Batch 11 (2026-05-15): migrated from ``Result[T, ProviderError]``
to raise-based typed exceptions. The public RunPod provider facade
catches typed exceptions and returns ``Result`` for Protocol
conformance.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol

from ryotenkai_providers.runpod.models import PodSnapshot
from ryotenkai_shared.errors import ProviderUnavailableError
from ryotenkai_shared.utils.logger import logger

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
    from ryotenkai_shared.config.providers.runpod import RunPodProviderConfig


class _TrainingApiProtocol(Protocol):
    def create_pod(
        self,
        config: RunPodProviderConfig,
        *,
        pod_name: str | None = None,
    ) -> dict[str, Any]: ...

    def query_pod(self, pod_id: str) -> dict[str, Any]: ...

    def terminate_pod(self, pod_id: str) -> None: ...

    def stop_pod(self, pod_id: str) -> None: ...

    def resume_pod(self, pod_id: str) -> None: ...

    def get_ssh_info(self, pod_id: str) -> dict[str, Any]: ...

    def extract_exposed_ssh_info(
        self,
        pod_data: dict[str, Any] | None,
        *,
        pod_id: str | None = None,
    ) -> dict[str, Any]: ...


class _InferenceApiProtocol(Protocol):
    def get_pod(self, *, pod_id: str) -> dict[str, Any]: ...

    def start_pod(self, *, pod_id: str) -> None: ...

    def stop_pod(self, *, pod_id: str) -> None: ...

    def delete_pod(self, *, pod_id: str) -> None: ...


class RunPodTrainingPodControl:
    """SDK-backed control for training pods.

    Phase A2 Batch 11: raise-based contract. Methods return ``T`` and
    raise typed exceptions (subclasses of :class:`RyotenkAIError`) on
    failure. ``create_pod`` retries on transient capacity errors but
    re-raises after exhaustion.
    """

    def __init__(self, *, api: _TrainingApiProtocol):
        self._api = api

    def create_pod(
        self,
        *,
        config: RunPodProviderConfig,
        pod_name: str,
    ) -> dict[str, Any]:
        """Create pod with retries for transient capacity errors.

        Re-raises the last exception when all attempts are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _CREATE_POD_MAX_RETRIES + 1):
            try:
                return self._api.create_pod(config=config, pod_name=pod_name)
            except Exception as exc:
                last_exc = exc
                if not self._is_transient_error_str(str(exc)):
                    raise

                if attempt < _CREATE_POD_MAX_RETRIES:
                    logger.warning(
                        "[POD_CONTROL] Create pod failed (attempt %d/%d), retrying in %ds: %s",
                        attempt,
                        _CREATE_POD_MAX_RETRIES,
                        _CREATE_POD_RETRY_DELAY_S,
                        exc,
                    )
                    time.sleep(_CREATE_POD_RETRY_DELAY_S)

        logger.error("[POD_CONTROL] Create pod failed after %d attempts", _CREATE_POD_MAX_RETRIES)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _is_transient_error_str(msg: str) -> bool:
        lower = msg.lower()
        return any(marker in lower for marker in _TRANSIENT_MARKERS)

    def query_pod(self, pod_id: str) -> dict[str, Any]:
        return self._api.query_pod(pod_id)

    def query_pod_snapshot(self, pod_id: str) -> PodSnapshot:
        """Query pod and return a typed snapshot."""
        pod_data = self._api.query_pod(pod_id)
        return PodSnapshot.from_graphql(pod_data)

    def extract_exposed_ssh_info(
        self,
        pod_data: dict[str, Any] | None,
        *,
        pod_id: str | None = None,
    ) -> dict[str, Any]:
        return self._api.extract_exposed_ssh_info(pod_data, pod_id=pod_id)

    def get_ssh_info(self, pod_id: str) -> dict[str, Any]:
        try:
            snapshot = self.query_pod_snapshot(pod_id)
        except Exception:
            return self._api.get_ssh_info(pod_id)
        if snapshot.ssh_endpoint is not None:
            return {"host": snapshot.ssh_endpoint.host, "port": snapshot.ssh_endpoint.port}
        return self._api.get_ssh_info(pod_id)

    def terminate_pod(self, pod_id: str) -> None:
        self._api.terminate_pod(pod_id)

    def stop_pod(self, *, pod_id: str) -> None:
        """Pause the pod (RunPod ``podStop``). Phase 14.A pause/resume path."""
        self._api.stop_pod(pod_id)

    def start_pod(self, *, pod_id: str) -> None:
        """Resume a stopped pod (RunPod ``podResume``). Phase 14.A path.

        Named ``start_pod`` for symmetry with the inference-side
        ``RunPodInferencePodControl.start_pod`` so callers can treat
        the two control surfaces uniformly.
        """
        self._api.resume_pod(pod_id)


class RunPodInferencePodControl:
    """SDK-backed control for inference pod start/stop/delete operations.

    Phase A2 Batch 11: raise-based contract.
    """

    def __init__(self, *, api: _InferenceApiProtocol):
        self._api = api

    def start_pod(self, *, pod_id: str) -> None:
        self._api.start_pod(pod_id=pod_id)

    def get_pod(self, *, pod_id: str) -> dict[str, Any]:
        return self._api.get_pod(pod_id=pod_id)

    def query_pod_snapshot(self, pod_id: str) -> PodSnapshot:
        """Inference-side analogue of ``RunPodTrainingPodControl.query_pod_snapshot``.

        Both control surfaces now expose the same one-method readiness
        contract, so the canonical ``PodSshWaiter`` (introduced in a
        follow-up commit) can poll either one through a single
        ``PodQuery`` Protocol — no per-side branches.

        Raises ``ProviderUnavailableError`` with
        ``context['code'] == 'RUNPOD_POD_DATA_MISSING'`` when the SDK
        reports the pod doesn't exist (matches training's behavior so
        the waiter's terminal-class check is symmetric).
        """
        pod_data = self._api.get_pod(pod_id=pod_id)
        if not pod_data:
            raise ProviderUnavailableError(
                detail="No pod data received",
                context={"code": "RUNPOD_POD_DATA_MISSING", "pod_id": pod_id},
            )
        return PodSnapshot.from_graphql(pod_data)

    def stop_pod(self, *, pod_id: str) -> None:
        self._api.stop_pod(pod_id=pod_id)

    def delete_pod(self, *, pod_id: str) -> None:
        self._api.delete_pod(pod_id=pod_id)


__all__ = [
    "RunPodInferencePodControl",
    "RunPodTrainingPodControl",
]

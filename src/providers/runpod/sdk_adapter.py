"""
RunPod Python SDK adapter for pod lifecycle.

This module centralizes interaction with the official ``runpod`` package while
preserving project-level contracts:
- Result-based return values
- structured ProviderError objects
- explicit payload normalization for existing provider code
"""

from __future__ import annotations

import contextlib
import io
import threading
from typing import Any

import runpod

from src.utils.result import Err, Ok, ProviderError, Result

_RUNPOD_SDK_CALL_FAILED = "RUNPOD_SDK_CALL_FAILED"
_RUNPOD_SDK_VALIDATION_ERROR = "RUNPOD_SDK_VALIDATION_ERROR"
_RUNPOD_SDK_UNEXPECTED_RESPONSE = "RUNPOD_SDK_UNEXPECTED_RESPONSE"

_CAPACITY_MARKERS = (
    "no longer any instances available",
    "no instances available",
    "does not have the resources",
    "try again",
    "rate limit",
    "timeout",
    "503",
    "502",
    "could not find any pods with required specifications",
    "there are no instances currently available",
    "no available datacenter with requested resources",
)

_SDK_LOCK = threading.Lock()


def _coerce_ports(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(items) or None
    return None


def _is_capacity_error(err: ProviderError) -> bool:
    msg = err.message.lower()
    return any(marker in msg for marker in _CAPACITY_MARKERS)


class RunPodSDKClient:
    """Thin Result-based adapter over the official ``runpod`` Python SDK."""

    def __init__(self, *, api_key: str):
        self._api_key = api_key

    def _call_sdk(self, fn_name: str, *args: Any, **kwargs: Any) -> Result[Any, ProviderError]:
        fn = getattr(runpod, fn_name, None)
        if fn is None:
            return Err(
                ProviderError(
                    message=f"runpod SDK function is not available: {fn_name}",
                    code=_RUNPOD_SDK_CALL_FAILED,
                    details={"function": fn_name},
                )
            )

        previous_api_key = getattr(runpod, "api_key", None)
        try:
            with _SDK_LOCK:
                runpod.api_key = self._api_key
                # The SDK prints some internals (e.g. create_pod raw_response).
                with contextlib.redirect_stdout(io.StringIO()):
                    result = fn(*args, **kwargs)
            return Ok(result)
        except ValueError as exc:
            return Err(
                ProviderError(
                    message=f"runpod SDK validation failed in {fn_name}: {exc}",
                    code=_RUNPOD_SDK_VALIDATION_ERROR,
                    details={"function": fn_name},
                )
            )
        except Exception as exc:
            return Err(
                ProviderError(
                    message=f"runpod SDK call failed in {fn_name}: {exc}",
                    code=_RUNPOD_SDK_CALL_FAILED,
                    details={"function": fn_name},
                )
            )
        finally:
            with contextlib.suppress(Exception):
                runpod.api_key = previous_api_key

    def get_pod(self, *, pod_id: str) -> Result[dict[str, Any], ProviderError]:
        result = self._call_sdk("get_pod", pod_id)
        if result.is_failure():
            return result  # type: ignore[return-value]
        value = result.unwrap()
        if isinstance(value, dict):
            return Ok(value)
        return Err(
            ProviderError(
                message=f"Unexpected runpod SDK get_pod response type: {type(value).__name__}",
                code=_RUNPOD_SDK_UNEXPECTED_RESPONSE,
            )
        )

    def list_pods(self, *, params: dict[str, Any] | None = None) -> Result[list[dict[str, Any]], ProviderError]:
        result = self._call_sdk("get_pods")
        if result.is_failure():
            return result  # type: ignore[return-value]
        value = result.unwrap()
        if not isinstance(value, list):
            return Err(
                ProviderError(
                    message=f"Unexpected runpod SDK get_pods response type: {type(value).__name__}",
                    code=_RUNPOD_SDK_UNEXPECTED_RESPONSE,
                )
            )
        if not params:
            return Ok(value)
        return Ok([pod for pod in value if self._pod_matches_filters(pod, params)])

    @staticmethod
    def _pod_matches_filters(pod: dict[str, Any], params: dict[str, Any]) -> bool:
        if not isinstance(pod, dict):
            return False
        for key, expected in params.items():
            if expected is None:
                continue
            if key == "computeType":
                # The current callers only request GPU pods. SDK responses may omit
                # this field, so treat missing value as non-blocking.
                actual = pod.get("computeType")
                if actual is not None and str(actual) != str(expected):
                    return False
                continue
            actual = pod.get(key)
            if actual is None:
                return False
            if str(actual) != str(expected):
                return False
        return True

    def create_pod(self, **sdk_kwargs: Any) -> Result[dict[str, Any], ProviderError]:
        result = self._call_sdk("create_pod", **sdk_kwargs)
        if result.is_failure():
            return result  # type: ignore[return-value]
        value = result.unwrap()
        if isinstance(value, dict):
            return Ok(value)
        return Err(
            ProviderError(
                message=f"Unexpected runpod SDK create_pod response type: {type(value).__name__}",
                code=_RUNPOD_SDK_UNEXPECTED_RESPONSE,
            )
        )

    def create_pod_from_payload(self, *, payload: dict[str, Any]) -> Result[dict[str, Any], ProviderError]:
        gpu_type_ids_raw = payload.get("gpuTypeIds")
        gpu_type_ids = [str(item).strip() for item in gpu_type_ids_raw or [] if str(item).strip()]

        sdk_kwargs: dict[str, Any] = {
            "name": str(payload.get("name") or "").strip(),
            "image_name": str(payload.get("imageName") or "").strip(),
            "cloud_type": str(payload.get("cloudType") or "ALL").strip() or "ALL",
            "support_public_ip": bool(payload.get("supportPublicIp", True)),
            "start_ssh": bool(payload.get("startSsh", True)),
            "data_center_id": payload.get("dataCenterId"),
            "country_code": payload.get("countryCode"),
            "gpu_count": int(payload.get("gpuCount") or 1),
            "volume_in_gb": int(payload.get("volumeInGb") or 0),
            "container_disk_in_gb": payload.get("containerDiskInGb"),
            "min_vcpu_count": int(payload.get("minVcpuCount") or 1),
            "min_memory_in_gb": int(payload.get("minMemoryInGb") or 1),
            "docker_args": str(payload.get("dockerArgs") or "").strip(),
            "ports": _coerce_ports(payload.get("ports")),
            "volume_mount_path": str(payload.get("volumeMountPath") or "/runpod-volume").strip() or "/runpod-volume",
            "env": payload.get("env"),
            "template_id": payload.get("templateId"),
            "network_volume_id": payload.get("networkVolumeId"),
            "allowed_cuda_versions": payload.get("allowedCudaVersions"),
            "min_download": payload.get("minDownloadMbps"),
            "min_upload": payload.get("minUploadMbps"),
            "instance_id": payload.get("instanceId"),
        }
        sdk_kwargs = {key: value for key, value in sdk_kwargs.items() if value is not None}

        if not gpu_type_ids:
            return self.create_pod(**sdk_kwargs)

        last_err: ProviderError | None = None
        for gpu_type_id in gpu_type_ids:
            attempt_result = self.create_pod(gpu_type_id=gpu_type_id, **sdk_kwargs)
            if attempt_result.is_success():
                return attempt_result

            err = attempt_result.unwrap_err()  # type: ignore[union-attr]
            last_err = err
            if not _is_capacity_error(err):
                return attempt_result

        return Err(
            last_err
            or ProviderError(
                message="runpod SDK create_pod failed for all requested GPU types",
                code=_RUNPOD_SDK_CALL_FAILED,
            )
        )

    def stop_pod(self, *, pod_id: str) -> Result[None, ProviderError]:
        result = self._call_sdk("stop_pod", pod_id)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]
        return Ok(None)

    def start_pod(self, *, pod_id: str) -> Result[None, ProviderError]:
        pod_result = self.get_pod(pod_id=pod_id)
        if pod_result.is_failure():
            return Err(pod_result.unwrap_err())  # type: ignore[union-attr]

        pod = pod_result.unwrap()
        gpu_count_raw = pod.get("gpuCount")
        try:
            gpu_count = int(gpu_count_raw) if gpu_count_raw is not None else 1
        except (TypeError, ValueError):
            gpu_count = 1

        result = self._call_sdk("resume_pod", pod_id, gpu_count)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]
        return Ok(None)

    def delete_pod(self, *, pod_id: str) -> Result[None, ProviderError]:
        result = self._call_sdk("terminate_pod", pod_id)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]
        return Ok(None)


__all__ = ["RunPodSDKClient"]

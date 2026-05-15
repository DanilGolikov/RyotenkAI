"""
RunPod Python SDK adapter for pod lifecycle.

This module centralizes interaction with the official ``runpod`` package while
preserving project-level contracts. Phase A2 Batch 11 (2026-05-15):
migrated from Result-based returns to raise-based typed exceptions
(:class:`ProviderUnavailableError` / :class:`ProviderRateLimitedError` /
:class:`ProviderAuthFailedError`). The public RunPod provider facade
catches these and translates back into ``Result[T, ProviderError]`` so
:class:`IGPUProvider` / :class:`IInferenceProvider` Protocol contracts
remain unchanged until Batch 12 migrates the Protocols themselves.
"""

from __future__ import annotations

import contextlib
import io
import threading
from typing import Any

import runpod

from ryotenkai_shared.errors import (
    ProviderAuthFailedError,
    ProviderRateLimitedError,
    ProviderUnavailableError,
)

_RUNPOD_SDK_CALL_FAILED = "RUNPOD_SDK_CALL_FAILED"
_RUNPOD_SDK_VALIDATION_ERROR = "RUNPOD_SDK_VALIDATION_ERROR"
_RUNPOD_SDK_UNEXPECTED_RESPONSE = "RUNPOD_SDK_UNEXPECTED_RESPONSE"

#: Phase 11.C-1+: substrings RunPod returns when capacity is
#: temporarily unavailable. Used by ``create_pod_from_payload`` retry
#: loop AND by Phase 11.C-2's ``resume_pod_with_retry`` to decide
#: whether an error is transient (retry with backoff) vs fatal.
#: Public symbol — Phase 11 callers import it for capacity detection.
CAPACITY_MARKERS = (
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
# Backwards-compat alias for any local consumer that still grep's
# the leading underscore. Will be removed when Phase 12 lands.
_CAPACITY_MARKERS = CAPACITY_MARKERS

_AUTH_MARKERS = (
    "unauthorized",
    "401",
    "403",
    "invalid api key",
    "authentication failed",
    "permission denied",
)

_RATE_LIMIT_MARKERS = ("rate limit", "429")

_SDK_LOCK = threading.Lock()


def _coerce_ports(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(items) or None
    return None


def is_capacity_error_message(msg: str) -> bool:
    """Phase 11.C-2 — public matcher used by ``resume_pod_with_retry``.

    Pure string check against :data:`CAPACITY_MARKERS`. Callers pass
    just the message text (not a ProviderError) so they don't need to
    import the error class to use it.
    """
    lower = msg.lower()
    return any(marker in lower for marker in CAPACITY_MARKERS)


def _is_capacity_error_str(msg: str) -> bool:
    """Internal helper for ``create_pod_from_payload`` retry loop."""
    return is_capacity_error_message(msg)


def _classify_sdk_exception(exc: BaseException, fn_name: str) -> Exception:
    """Map an SDK exception to a typed :class:`RyotenkAIError`.

    Inspects the message text against ``_AUTH_MARKERS`` /
    ``_RATE_LIMIT_MARKERS`` / ``CAPACITY_MARKERS``. Falls through to
    :class:`ProviderUnavailableError` (the catch-all transient
    classification — caller can retry).
    """
    msg = str(exc).lower()
    detail = f"runpod SDK call failed in {fn_name}: {exc}"
    context = {"function": fn_name}
    if any(marker in msg for marker in _AUTH_MARKERS):
        return ProviderAuthFailedError(
            detail=detail,
            context={**context, "code": _RUNPOD_SDK_CALL_FAILED},
            cause=exc if isinstance(exc, Exception) else None,
        )
    if any(marker in msg for marker in _RATE_LIMIT_MARKERS):
        return ProviderRateLimitedError(
            detail=detail,
            context={**context, "code": _RUNPOD_SDK_CALL_FAILED},
            cause=exc if isinstance(exc, Exception) else None,
        )
    return ProviderUnavailableError(
        detail=detail,
        context={**context, "code": _RUNPOD_SDK_CALL_FAILED},
        cause=exc if isinstance(exc, Exception) else None,
    )


class RunPodSDKClient:
    """Thin raise-based adapter over the official ``runpod`` Python SDK.

    Phase A2 Batch 11: methods return ``T`` and raise
    :class:`ProviderUnavailableError` / :class:`ProviderRateLimitedError`
    / :class:`ProviderAuthFailedError` on SDK failure. The public
    provider facade (``RunPodProvider`` / ``RunPodPodInferenceProvider``)
    catches these and wraps them as ``Err(ProviderError(...))`` for
    Protocol conformance.
    """

    def __init__(self, *, api_key: str):
        self._api_key = api_key

    def _call_sdk(self, fn_name: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(runpod, fn_name, None)
        if fn is None:
            raise ProviderUnavailableError(
                detail=f"runpod SDK function is not available: {fn_name}",
                context={"function": fn_name, "code": _RUNPOD_SDK_CALL_FAILED},
            )

        previous_api_key = getattr(runpod, "api_key", None)
        try:
            with _SDK_LOCK:
                runpod.api_key = self._api_key
                # The SDK prints some internals (e.g. create_pod raw_response).
                with contextlib.redirect_stdout(io.StringIO()):
                    return fn(*args, **kwargs)
        except ValueError as exc:
            raise ProviderUnavailableError(
                detail=f"runpod SDK validation failed in {fn_name}: {exc}",
                context={"function": fn_name, "code": _RUNPOD_SDK_VALIDATION_ERROR},
                cause=exc,
            ) from exc
        except Exception as exc:
            raise _classify_sdk_exception(exc, fn_name) from exc
        finally:
            with contextlib.suppress(Exception):
                runpod.api_key = previous_api_key

    def get_pod(self, *, pod_id: str) -> dict[str, Any]:
        value = self._call_sdk("get_pod", pod_id)
        if isinstance(value, dict):
            return value
        raise ProviderUnavailableError(
            detail=f"Unexpected runpod SDK get_pod response type: {type(value).__name__}",
            context={"code": _RUNPOD_SDK_UNEXPECTED_RESPONSE},
        )

    def list_pods(self, *, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        value = self._call_sdk("get_pods")
        if not isinstance(value, list):
            raise ProviderUnavailableError(
                detail=f"Unexpected runpod SDK get_pods response type: {type(value).__name__}",
                context={"code": _RUNPOD_SDK_UNEXPECTED_RESPONSE},
            )
        if not params:
            return value
        return [pod for pod in value if self._pod_matches_filters(pod, params)]

    @staticmethod
    def _pod_matches_filters(pod: dict[str, Any], params: dict[str, Any]) -> bool:
        # Defensive: the SDK's typing claims dict but real responses have
        # been observed wrapping unexpected shapes during partial outages.
        if not isinstance(pod, dict):  # type: ignore[unreachable]
            return False  # type: ignore[unreachable]
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

    def create_pod(self, **sdk_kwargs: Any) -> dict[str, Any]:
        value = self._call_sdk("create_pod", **sdk_kwargs)
        if isinstance(value, dict):
            return value
        raise ProviderUnavailableError(
            detail=f"Unexpected runpod SDK create_pod response type: {type(value).__name__}",
            context={"code": _RUNPOD_SDK_UNEXPECTED_RESPONSE},
        )

    def create_pod_from_payload(self, *, payload: dict[str, Any]) -> dict[str, Any]:
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

        last_exc: Exception | None = None
        for gpu_type_id in gpu_type_ids:
            try:
                return self.create_pod(gpu_type_id=gpu_type_id, **sdk_kwargs)
            except Exception as exc:
                last_exc = exc
                if not _is_capacity_error_str(str(exc)):
                    raise

        if last_exc is not None:
            raise last_exc
        raise ProviderUnavailableError(
            detail="runpod SDK create_pod failed for all requested GPU types",
            context={"code": _RUNPOD_SDK_CALL_FAILED},
        )

    def stop_pod(self, *, pod_id: str) -> None:
        self._call_sdk("stop_pod", pod_id)

    def start_pod(self, *, pod_id: str) -> None:
        pod = self.get_pod(pod_id=pod_id)
        gpu_count_raw = pod.get("gpuCount")
        try:
            gpu_count = int(gpu_count_raw) if gpu_count_raw is not None else 1
        except (TypeError, ValueError):
            gpu_count = 1

        self._call_sdk("resume_pod", pod_id, gpu_count)

    def delete_pod(self, *, pod_id: str) -> None:
        self._call_sdk("terminate_pod", pod_id)


__all__ = ["RunPodSDKClient"]

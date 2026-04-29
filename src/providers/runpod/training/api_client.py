"""
RunPod training client backed by the official Python SDK.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.providers.runpod.models import PodSnapshot, read_ssh_public_key
from src.providers.runpod.sdk_adapter import RunPodSDKClient
from src.runner.__about__ import RUNTIME_IMAGE
from src.utils.logger import logger
from src.utils.result import Err, Ok, ProviderError, Result

_POD_ID_KEY = "pod_id"

_POD_NAME_MAX_LEN_UI = 80
_VOLUME_MOUNT = "/workspace"

if TYPE_CHECKING:
    from src.providers.runpod.training.config import RunPodProviderConfig


# Empty docker_args = use the image's CMD.
#
# Why this is empty (was: a long bash bootstrap that ended in
# ``sleep infinity``):
#
#   When RunPod is given a non-empty ``docker_args``, the SDK passes
#   it as Docker's CMD and our image's
#   ``ENTRYPOINT ["/usr/bin/dumb-init", "--", "/entrypoint.sh"]``
#   forwards those args to entrypoint.sh, which has a "custom command
#   path" that ``exec``s the args verbatim (used by integration tests
#   and `docker run image bash`). The previous bootstrap ended in
#   ``sleep infinity`` — pod stayed alive indefinitely, sshd was
#   running, but uvicorn was NEVER launched. /healthz never answered.
#   Every "runner /healthz did not return 200 within 30s" timeout
#   we hit traced back to this.
#
#   The only thing the old bootstrap did that we still need is
#   ``echo $PUBLIC_KEY >> ~/.ssh/authorized_keys``. That now happens
#   inside entrypoint.sh itself (see the PUBLIC_KEY block) — fewer
#   moving parts and provider-agnostic.
RUNPOD_DOCKER_ARGS: str = ""


def build_pod_launch_kwargs(
    config: RunPodProviderConfig,
    pod_name: str | None,
    public_key: str | None,
) -> dict[str, Any]:
    """Build SDK kwargs for ``runpod.create_pod``."""
    train_cfg = config.training
    name_val = (pod_name or f"ryotenkai-training-{int(time.time())}").replace('"', "").strip()
    if len(name_val) > _POD_NAME_MAX_LEN_UI:
        name_val = name_val[:_POD_NAME_MAX_LEN_UI]

    env: dict[str, str] = {}
    if public_key:
        env["PUBLIC_KEY"] = public_key

    return {
        "name": name_val,
        # Image is pinned in src.runner.__about__ — no longer a user
        # config field as of Phase 6.6 (versions tied to release).
        "image_name": RUNTIME_IMAGE,
        "gpu_type_id": train_cfg.gpu_type,
        "cloud_type": train_cfg.cloud_type,
        "support_public_ip": True,
        "start_ssh": True,
        "gpu_count": 1,
        "volume_in_gb": train_cfg.volume_disk_gb,
        "container_disk_in_gb": train_cfg.container_disk_gb,
        "docker_args": RUNPOD_DOCKER_ARGS,
        "ports": train_cfg.ports,
        "volume_mount_path": _VOLUME_MOUNT,
        "env": env or None,
        "template_id": train_cfg.template_id,
    }


class RunPodAPIClient:
    """Training pod lifecycle client backed by ``runpod`` SDK."""

    def __init__(self, api_base_url: str, api_key: str):
        self.api_base = api_base_url
        self.api_key = api_key
        self._sdk = RunPodSDKClient(api_key=api_key)
        logger.debug(f"🔗 RunPodAPIClient initialized via SDK: {api_base_url}")

    def create_pod(
        self,
        config: RunPodProviderConfig,
        *,
        pod_name: str | None = None,
    ) -> Result[dict[str, Any], ProviderError]:
        """Create a new RunPod training pod using the SDK."""
        logger.info("📦 Creating RunPod pod via RunPod SDK...")

        public_key = read_ssh_public_key(config.connect.ssh.key_path)
        sdk_kwargs = build_pod_launch_kwargs(config, pod_name, public_key)

        train_cfg = config.training
        logger.debug(f"[RUNPOD:CONFIG] image={RUNTIME_IMAGE}, template_id={train_cfg.template_id}")
        logger.info(f"📦 Using Docker image: {RUNTIME_IMAGE}")

        result = self._sdk.create_pod(**sdk_kwargs)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]

        pod_data = result.unwrap()
        if not pod_data or not pod_data.get("id"):
            return Err(ProviderError(message=f"Failed to create pod: {pod_data}", code="RUNPOD_POD_DATA_MISSING"))

        pod_id = str(pod_data["id"])
        enriched = self._sdk.get_pod(pod_id=pod_id)
        if enriched.is_ok() and enriched.unwrap():
            pod_data = enriched.unwrap()

        return Ok(self._log_and_build_create_result(pod_data, train_cfg))

    @staticmethod
    def _log_and_build_create_result(pod_data: dict[str, Any], train_cfg: Any) -> dict[str, Any]:
        pod_id = str(pod_data["id"])
        gpu_type = train_cfg.gpu_type
        pod_host_id = pod_data.get("machine", {}).get("podHostId")
        cost = pod_data.get("costPerHr")

        def _fmt_cost(v: Any) -> str:
            try:
                return f"${float(v):.3f}/hr"
            except Exception:
                return "unknown"

        logger.info(f"✅ Pod created: {pod_id} (status={pod_data.get('desiredStatus')})")
        logger.info(
            f"   GPU: {pod_data.get('gpuCount')} x {gpu_type} | "
            f"vCPU: {pod_data.get('vcpuCount')} | RAM: {pod_data.get('memoryInGb')}GB | Cost: {_fmt_cost(cost)}"
        )
        logger.info(f"   Image: {pod_data.get('imageName') or RUNTIME_IMAGE}")
        logger.info(
            f"   Disks: container={train_cfg.container_disk_gb}GB, volume={train_cfg.volume_disk_gb}GB, "
            f"mount={_VOLUME_MOUNT} | Ports: {train_cfg.ports}"
        )
        logger.info(f"   Machine: podHostId={pod_host_id}, machineId={pod_data.get('machineId')}")

        return {
            _POD_ID_KEY: pod_id,
            "machine": pod_host_id,
            "gpu_count": pod_data.get("gpuCount"),
            "cost_per_hr": cost,
            "gpu_type": gpu_type,
        }

    def query_pod(self, pod_id: str) -> Result[dict[str, Any], ProviderError]:
        """Query pod status using the SDK."""
        result = self._sdk.get_pod(pod_id=pod_id)
        if result.is_failure():
            err = result.unwrap_err()  # type: ignore[union-attr]
            return Err(
                ProviderError(
                    message=f"Failed to query pod: {err.message}",
                    code=err.code,
                    details={_POD_ID_KEY: pod_id},
                )
            )

        pod_data = result.unwrap()
        if not pod_data:
            return Err(
                ProviderError(
                    message="No pod data received", code="RUNPOD_POD_DATA_MISSING", details={_POD_ID_KEY: pod_id}
                )
            )
        return Ok(pod_data)

    def terminate_pod(self, pod_id: str) -> Result[None, ProviderError]:
        """Terminate (delete) pod using the SDK."""
        logger.info(f"🗑️ Terminating pod {pod_id}...")
        result = self._sdk.delete_pod(pod_id=pod_id)
        if result.is_failure():
            err = result.unwrap_err()  # type: ignore[union-attr]
            logger.error(f"Failed to terminate pod {pod_id}: {err}")
            return Err(
                ProviderError(
                    message=f"Failed to terminate pod: {err.message}",
                    code=err.code,
                    details={_POD_ID_KEY: pod_id},
                )
            )
        logger.info(f"✅ Pod {pod_id} terminated")
        return Ok(None)

    def stop_pod(self, pod_id: str) -> Result[None, ProviderError]:
        """Phase 11.C — pause pod (sleep), preserves /workspace.

        Mirrors the in-pod runner's :class:`PodTerminator` ``podStop``
        path but called from Mac side. Useful when the orchestrator
        wants to clean up a stopped pod after the retriever finished
        (rare path — usually the in-pod terminator already stopped
        the pod). Idempotent: SDK returns success even if pod is
        already sleeping / gone.
        """
        logger.info(f"⏸️  Stopping (sleeping) pod {pod_id}...")
        result = self._sdk.stop_pod(pod_id=pod_id)
        if result.is_failure():
            err = result.unwrap_err()  # type: ignore[union-attr]
            logger.error(f"Failed to stop pod {pod_id}: {err}")
            return Err(
                ProviderError(
                    message=f"Failed to stop pod: {err.message}",
                    code=err.code,
                    details={_POD_ID_KEY: pod_id},
                )
            )
        logger.info(f"✅ Pod {pod_id} stopped (sleeping)")
        return Ok(None)

    def resume_pod(self, pod_id: str) -> Result[None, ProviderError]:
        """Phase 11.C — resume pod from sleep state.

        Wakes a previously stopped pod (Phase 11.B's ``podStop``
        path) so the orchestrator can re-attach SSH and run
        :class:`ModelRetriever` to pull artifacts off ``/workspace``.

        Failure modes:
        * Capacity exhausted ⇒ ProviderError with RUNPOD_NO_GPU
          marker (caller's :func:`resume_pod_with_retry` matches
          ``_CAPACITY_MARKERS`` to schedule retries).
        * Pod already running ⇒ SDK returns success — no-op,
          idempotent.
        * Pod terminated ⇒ ProviderError with "not found" / "does
          not exist" message (caller surfaces as POD_GONE).
        """
        logger.info(f"▶️  Resuming pod {pod_id}...")
        result = self._sdk.start_pod(pod_id=pod_id)
        if result.is_failure():
            err = result.unwrap_err()  # type: ignore[union-attr]
            logger.error(f"Failed to resume pod {pod_id}: {err}")
            return Err(
                ProviderError(
                    message=f"Failed to resume pod: {err.message}",
                    code=err.code,
                    details={_POD_ID_KEY: pod_id},
                )
            )
        logger.info(f"✅ Pod {pod_id} resume request accepted")
        return Ok(None)

    def get_ssh_info(self, pod_id: str) -> Result[dict[str, Any], ProviderError]:
        """Get SSH connection info for a pod."""
        pod_result = self.query_pod(pod_id)
        if pod_result.is_failure():
            return Err(pod_result.unwrap_err())  # type: ignore[union-attr]
        ssh_info = self.extract_exposed_ssh_info(pod_result.unwrap(), pod_id=pod_id)
        if ssh_info.is_failure():
            return Err(ssh_info.unwrap_err())  # type: ignore[union-attr]
        return Ok(ssh_info.unwrap())

    @staticmethod
    def extract_exposed_ssh_info(
        pod_data: dict[str, Any] | None,
        *,
        pod_id: str | None = None,
    ) -> Result[dict[str, Any], ProviderError]:
        """Extract automation-grade SSH endpoint from RunPod pod data."""
        details = {_POD_ID_KEY: pod_id} if pod_id else None

        if not pod_data or not pod_data.get("runtime"):
            return Err(
                ProviderError(
                    message="Pod runtime info not available",
                    code="RUNPOD_RUNTIME_NOT_AVAILABLE",
                    details=details,
                )
            )

        snapshot = PodSnapshot.from_graphql(pod_data)
        if snapshot.ssh_endpoint is not None:
            return Ok({"host": snapshot.ssh_endpoint.host, "port": snapshot.ssh_endpoint.port})

        return Err(
            ProviderError(
                message="SSH over exposed TCP is not available on pod",
                code="RUNPOD_SSH_PORT_UNAVAILABLE",
                details=details,
            )
        )


__all__ = ["RUNPOD_DOCKER_ARGS", "RunPodAPIClient", "build_pod_launch_kwargs"]

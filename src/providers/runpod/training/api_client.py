"""
RunPod training client backed by the official Python SDK.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.providers.runpod.models import PodSnapshot, read_ssh_public_key
from src.providers.runpod.sdk_adapter import RunPodSDKClient
from src.utils.logger import logger
from src.utils.result import Err, Ok, ProviderError, Result

_POD_ID_KEY = "pod_id"

_POD_NAME_MAX_LEN_UI = 80
_VOLUME_MOUNT = "/workspace"

if TYPE_CHECKING:
    from src.providers.runpod.training.config import RunPodProviderConfig


def build_ssh_bootstrap_cmd() -> str:
    """Shell command that ensures sshd is running inside a RunPod container."""
    return (
        "bash -c '"
        "set -e; "
        "if ! command -v sshd >/dev/null 2>&1; then "
        "apt update; "
        "DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y; "
        "fi; "
        "mkdir -p ~/.ssh; "
        "chmod 700 ~/.ssh; "
        "echo $PUBLIC_KEY >> ~/.ssh/authorized_keys; "
        "chmod 600 ~/.ssh/authorized_keys; "
        "mkdir -p /run/sshd || true; "
        "/usr/sbin/sshd || service ssh start || true; "
        "sleep infinity"
        "'"
    )


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
        "image_name": train_cfg.image_name,
        "gpu_type_id": train_cfg.gpu_type,
        "cloud_type": train_cfg.cloud_type,
        "support_public_ip": True,
        "start_ssh": True,
        "gpu_count": 1,
        "volume_in_gb": train_cfg.volume_disk_gb,
        "container_disk_in_gb": train_cfg.container_disk_gb,
        "docker_args": build_ssh_bootstrap_cmd(),
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
        logger.debug(f"[RUNPOD:CONFIG] image_name={train_cfg.image_name}, template_id={train_cfg.template_id}")
        logger.info(f"📦 Using Docker image: {train_cfg.image_name}")

        result = self._sdk.create_pod(**sdk_kwargs)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]

        pod_data = result.unwrap()
        if not pod_data or not pod_data.get("id"):
            return Err(ProviderError(message=f"Failed to create pod: {pod_data}", code="RUNPOD_POD_DATA_MISSING"))

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
        logger.info(f"   Image: {pod_data.get('imageName') or train_cfg.image_name}")
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


__all__ = ["RunPodAPIClient", "build_pod_launch_kwargs", "build_ssh_bootstrap_cmd"]

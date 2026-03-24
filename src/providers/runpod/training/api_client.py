"""
RunPod API Client - Pure GraphQL API wrapper.

Handles all GraphQL interactions with RunPod API.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.providers.constants import (
    HTTP_STATUS_OK,
    KEY_ERRORS,
    KEY_QUERY,
    RETRY_BACKOFF_FACTOR,
    RETRY_TOTAL_ATTEMPTS,
    SSH_PORT_DEFAULT,
    TIMEOUT_REQUEST_DEFAULT,
)
from src.utils.logger import logger
from src.utils.result import Err, Ok, ProviderError, Result

_API_REQUEST_FAILED_CODE = "RUNPOD_API_REQUEST_FAILED"
_POD_ID_KEY = "pod_id"

_POD_NAME_MAX_LEN_UI = 80
_QUERY_TIMEOUT = 10

if TYPE_CHECKING:
    from src.providers.runpod.training.config import RunPodProviderConfig


class RunPodAPIClient:
    """
    Pure GraphQL API client for RunPod.

    Responsibilities:
    - Create pods via GraphQL mutations
    - Query pod status via GraphQL queries
    - Terminate pods via GraphQL mutations

    Does NOT handle:
    - Business logic
    - SSH operations
    - File uploads
    - Training setup
    """

    def __init__(self, api_base_url: str, api_key: str):
        """
        Initialize RunPod API client.

        Args:
            api_base_url: RunPod API base URL (e.g., "https://api.runpod.io")
            api_key: RunPod API key
        """
        self.api_base = api_base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Configure session with automatic retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=RETRY_TOTAL_ATTEMPTS,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],  # retry on these HTTP codes
            allowed_methods=["POST", "GET"],  # retry POST for GraphQL mutations
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

        logger.debug(f"🔗 RunPodAPIClient initialized: {api_base_url} (retry: 3 attempts, backoff: 2.0s)")

    def create_pod(
        self,
        config: RunPodProviderConfig,
        *,
        pod_name: str | None = None,
    ) -> Result[dict[str, Any], ProviderError]:
        """
        Create a new RunPod pod using GraphQL API.

        Args:
            config: RunPod provider configuration
            pod_name: Optional user-friendly pod name (for debugging in RunPod UI)

        Returns:
            Result with pod info (pod_id, machine, gpu_count, cost_per_hr) or structured provider error
        """
        logger.info("📦 Creating RunPod pod via GraphQL API...")

        # Build environment variables in GraphQL format
        env_items: list[tuple[str, str]] = []

        # RunPod full SSH over exposed TCP requires an sshd inside the container and an authorized_keys entry.
        # Official templates provide this, but custom images must bootstrap it.
        #
        # We try to provide PUBLIC_KEY explicitly (public data) if a .pub file exists next to the configured key.
        # This makes SSH bootstrap deterministic even if the platform doesn't inject $PUBLIC_KEY automatically.
        try:
            key_path_raw = config.connect.ssh.key_path
            key_path = Path(str(key_path_raw)).expanduser()
            pub_path = Path(str(key_path) + ".pub")
            if pub_path.exists():
                public_key = pub_path.read_text(encoding="utf-8").strip()
                if public_key:
                    env_items.append(("PUBLIC_KEY", public_key))
        except OSError:
            # Best-effort: never fail pod creation due to local public key reading issues.
            pass

        def _esc(v: str) -> str:
            return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "").replace("\r", "")

        if env_items:
            env_vars_graphql = "[" + ", ".join(f'{{key: "{_esc(k)}", value: "{_esc(v)}"}}' for k, v in env_items) + "]"
        else:
            env_vars_graphql = "[]"

        # Extract config values
        train_cfg = config.training
        cloud_type = train_cfg.cloud_type
        gpu_type = train_cfg.gpu_type
        timestamp = int(time.time())
        name_val = (pod_name or f"ryotenkai-training-{timestamp}").replace('"', "").strip()
        # RunPod UI name is primarily for humans; keep it short and safe.
        if len(name_val) > _POD_NAME_MAX_LEN_UI:
            name_val = name_val[:_POD_NAME_MAX_LEN_UI]
        container_disk = train_cfg.container_disk_gb
        volume_disk = train_cfg.volume_disk_gb
        ports = train_cfg.ports
        image_name = train_cfg.image_name
        # Workspace/volume mount is intentionally hardcoded for this provider.
        volume_mount = "/workspace"

        logger.debug(f"[RUNPOD:CONFIG] image_name={image_name}, template_id={train_cfg.template_id}")
        logger.info(f"📦 Using Docker image: {image_name}")

        # Bootstrap SSH for custom images (required for exposed TCP SSH to work reliably).
        # This follows RunPod docs: https://docs.runpod.io/pods/configuration/use-ssh
        docker_args = (
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

        mutation = f"""
        mutation {{
            podFindAndDeployOnDemand(input: {{
                cloudType: {cloud_type}
                gpuTypeId: "{gpu_type}"
                gpuCount: 1
                name: "{name_val}"
                imageName: "{image_name}"
                dockerArgs: "{_esc(docker_args)}"
                containerDiskInGb: {container_disk}
                volumeInGb: {volume_disk}
                ports: "{ports}"
                volumeMountPath: "{volume_mount}"
                startSsh: true
                startJupyter: false
                env: {env_vars_graphql}
            }}) {{
                id
                desiredStatus
                imageName
                gpuCount
                vcpuCount
                memoryInGb
                costPerHr
                machine {{
                    podHostId
                }}
            }}
        }}
        """

        payload = {KEY_QUERY: mutation}
        logger.debug("[RUNPOD:REQUEST] POST /graphql")

        try:
            response = self.session.post(
                f"{self.api_base}/graphql",
                headers=self.headers,
                json=payload,
                timeout=TIMEOUT_REQUEST_DEFAULT,
            )

            logger.debug(f"[RUNPOD:RESPONSE] status={response.status_code}")
            if response.status_code != HTTP_STATUS_OK:
                logger.error(f"Response body: {response.text}")

            response.raise_for_status()
            result = response.json()

            # Check for GraphQL errors
            if KEY_ERRORS in result:
                error_msg = result[KEY_ERRORS][0].get("message", "Unknown GraphQL error")
                logger.error(f"GraphQL error: {error_msg}")
                return Err(ProviderError(message=f"Failed to create pod: {error_msg}", code="RUNPOD_GRAPHQL_ERROR"))

            # Extract pod data
            pod_data = result.get("data", {}).get("podFindAndDeployOnDemand")
            if not pod_data or not pod_data.get("id"):
                return Err(ProviderError(message=f"Failed to create pod: {result}", code="RUNPOD_POD_DATA_MISSING"))

            pod_id = str(pod_data["id"])
            desired_status = pod_data.get("desiredStatus")
            image_name_out = pod_data.get("imageName") or image_name
            gpu_count_out = pod_data.get("gpuCount")
            vcpu = pod_data.get("vcpuCount")
            mem_gb = pod_data.get("memoryInGb")
            machine_id = pod_data.get("machineId")
            pod_host_id = pod_data.get("machine", {}).get("podHostId")
            cost = pod_data.get("costPerHr")

            def _fmt_cost(v: Any) -> str:
                try:
                    return f"${float(v):.3f}/hr"
                except Exception:
                    return "unknown"

            cost_str = _fmt_cost(cost)

            logger.info(f"✅ Pod created: {pod_id} (status={desired_status})")
            logger.info(f"   GPU: {gpu_count_out} x {gpu_type} | vCPU: {vcpu} | RAM: {mem_gb}GB | Cost: {cost_str}")
            logger.info(f"   Image: {image_name_out}")
            logger.info(
                f"   Disks: container={container_disk}GB, volume={volume_disk}GB, mount={volume_mount} | Ports: {ports}"
            )
            logger.info(f"   Machine: podHostId={pod_host_id}, machineId={machine_id}")

            return Ok(
                {
                    _POD_ID_KEY: pod_id,
                    "machine": pod_host_id,
                    "gpu_count": pod_data.get("gpuCount"),
                    "cost_per_hr": pod_data.get("costPerHr"),
                    "gpu_type": gpu_type,
                }
            )

        except requests.RequestException as e:
            logger.error(f"RunPod API error: {e}")
            return Err(ProviderError(message=f"Failed to create pod: {e}", code=_API_REQUEST_FAILED_CODE))

    def query_pod(self, pod_id: str) -> Result[dict[str, Any], ProviderError]:
        """
        Query pod status using GraphQL API.

        Args:
            pod_id: Pod ID to query

        Returns:
            Result with pod data (id, desiredStatus, runtime) or structured provider error
        """
        query = f"""
        query {{
            pod(input: {{podId: "{pod_id}"}}) {{
                id
                desiredStatus
                runtime {{
                    uptimeInSeconds
                    ports {{
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                    }}
                }}
            }}
        }}
        """

        logger.debug("[RUNPOD:REQUEST] POST /graphql (query_pod)")

        try:
            response = self.session.post(
                f"{self.api_base}/graphql",
                headers=self.headers,
                json={KEY_QUERY: query},
                timeout=_QUERY_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()

            if KEY_ERRORS in result:
                error_msg = result[KEY_ERRORS][0].get("message", "Unknown error")
                return Err(ProviderError(message=f"GraphQL error: {error_msg}", code="RUNPOD_GRAPHQL_ERROR"))

            pod_data = result.get("data", {}).get("pod")
            if not pod_data:
                return Err(
                    ProviderError(
                        message="No pod data received", code="RUNPOD_POD_DATA_MISSING", details={_POD_ID_KEY: pod_id}
                    )
                )

            return Ok(pod_data)

        except requests.RequestException as e:
            return Err(
                ProviderError(
                    message=f"Failed to query pod: {e}", code=_API_REQUEST_FAILED_CODE, details={_POD_ID_KEY: pod_id}
                )
            )

    def terminate_pod(self, pod_id: str) -> Result[None, ProviderError]:
        """
        Terminate (delete) pod using GraphQL API.

        Args:
            pod_id: Pod ID to terminate

        Returns:
            Result with None on success or structured provider error
        """
        logger.info(f"🗑️ Terminating pod {pod_id}...")

        mutation = f"""
        mutation {{
            podTerminate(input: {{podId: "{pod_id}"}})
        }}
        """

        try:
            response = self.session.post(
                f"{self.api_base}/graphql",
                headers=self.headers,
                json={KEY_QUERY: mutation},
                timeout=_QUERY_TIMEOUT,
            )
            response.raise_for_status()

            logger.info(f"✅ Pod {pod_id} terminated")
            return Ok(None)

        except requests.RequestException as e:
            logger.error(f"Failed to terminate pod {pod_id}: {e}")
            return Err(
                ProviderError(
                    message=f"Failed to terminate pod: {e}",
                    code=_API_REQUEST_FAILED_CODE,
                    details={_POD_ID_KEY: pod_id},
                )
            )

    def get_ssh_info(self, pod_id: str) -> Result[dict[str, Any], ProviderError]:
        """
        Get SSH connection info for a pod.

        Args:
            pod_id: Pod ID

        Returns:
            Result with SSH info (host, port) or structured provider error
        """
        query = f"""
        query {{
            pod(input: {{podId: "{pod_id}"}}) {{
                id
                runtime {{
                    ports {{
                        ip
                        privatePort
                        publicPort
                    }}
                }}
            }}
        }}
        """

        try:
            response = self.session.post(
                f"{self.api_base}/graphql",
                headers=self.headers,
                json={KEY_QUERY: query},
                timeout=_QUERY_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()

            pod_data = result.get("data", {}).get("pod")
            if not pod_data or not pod_data.get("runtime"):
                return Err(
                    ProviderError(
                        message="Pod runtime info not available",
                        code="RUNPOD_RUNTIME_NOT_AVAILABLE",
                        details={_POD_ID_KEY: pod_id},
                    )
                )

            ports = pod_data.get("runtime", {}).get("ports", [])

            for port in ports:
                if port.get("privatePort") == SSH_PORT_DEFAULT:
                    host = port.get("ip")
                    public_port = port.get("publicPort")
                    return Ok({"host": host, "port": public_port})

            return Err(
                ProviderError(
                    message="SSH port not available on pod",
                    code="RUNPOD_SSH_PORT_UNAVAILABLE",
                    details={_POD_ID_KEY: pod_id},
                )
            )

        except requests.RequestException as e:
            return Err(
                ProviderError(
                    message=f"Failed to get SSH info: {e}", code=_API_REQUEST_FAILED_CODE, details={_POD_ID_KEY: pod_id}
                )
            )


__all__ = ["RunPodAPIClient"]

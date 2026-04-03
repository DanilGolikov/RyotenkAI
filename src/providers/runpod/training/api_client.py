"""
RunPod API Client - Pure GraphQL API wrapper.

Handles all GraphQL interactions with RunPod API.
"""

from __future__ import annotations

import time
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
    TIMEOUT_REQUEST_DEFAULT,
)
from src.providers.runpod.models import PodSnapshot, read_ssh_public_key
from src.utils.logger import logger
from src.utils.result import Err, Ok, ProviderError, Result

_API_REQUEST_FAILED_CODE = "RUNPOD_API_REQUEST_FAILED"
_POD_ID_KEY = "pod_id"

_POD_NAME_MAX_LEN_UI = 80
_QUERY_TIMEOUT = 10
_VOLUME_MOUNT = "/workspace"

if TYPE_CHECKING:
    from src.providers.runpod.training.config import RunPodProviderConfig


def _esc_graphql(v: str) -> str:
    """Escape a string value for inline GraphQL string literals."""
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "").replace("\r", "")


def build_ssh_bootstrap_cmd() -> str:
    """Shell command that ensures sshd is running inside a RunPod container.

    Follows RunPod docs: https://docs.runpod.io/pods/configuration/use-ssh
    """
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


def build_pod_launch_mutation(
    config: RunPodProviderConfig,
    pod_name: str | None,
    public_key: str | None,
) -> str:
    """Build the GraphQL ``podFindAndDeployOnDemand`` mutation string."""
    env_items: list[tuple[str, str]] = []
    if public_key:
        env_items.append(("PUBLIC_KEY", public_key))

    if env_items:
        env_graphql = "[" + ", ".join(f'{{key: "{_esc_graphql(k)}", value: "{_esc_graphql(v)}"}}' for k, v in env_items) + "]"
    else:
        env_graphql = "[]"

    train_cfg = config.training
    name_val = (pod_name or f"ryotenkai-training-{int(time.time())}").replace('"', "").strip()
    if len(name_val) > _POD_NAME_MAX_LEN_UI:
        name_val = name_val[:_POD_NAME_MAX_LEN_UI]

    docker_args = build_ssh_bootstrap_cmd()

    return f"""
    mutation {{
        podFindAndDeployOnDemand(input: {{
            cloudType: {train_cfg.cloud_type}
            gpuTypeId: "{train_cfg.gpu_type}"
            gpuCount: 1
            name: "{name_val}"
            imageName: "{train_cfg.image_name}"
            dockerArgs: "{_esc_graphql(docker_args)}"
            containerDiskInGb: {train_cfg.container_disk_gb}
            volumeInGb: {train_cfg.volume_disk_gb}
            ports: "{train_cfg.ports}"
            volumeMountPath: "{_VOLUME_MOUNT}"
            startSsh: true
            startJupyter: false
            env: {env_graphql}
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
        """Create a new RunPod pod using GraphQL API."""
        logger.info("📦 Creating RunPod pod via GraphQL API...")

        public_key = read_ssh_public_key(config.connect.ssh.key_path)
        mutation = build_pod_launch_mutation(config, pod_name, public_key)

        train_cfg = config.training
        logger.debug(f"[RUNPOD:CONFIG] image_name={train_cfg.image_name}, template_id={train_cfg.template_id}")
        logger.info(f"📦 Using Docker image: {train_cfg.image_name}")

        try:
            response = self.session.post(
                f"{self.api_base}/graphql",
                headers=self.headers,
                json={KEY_QUERY: mutation},
                timeout=TIMEOUT_REQUEST_DEFAULT,
            )

            logger.debug(f"[RUNPOD:RESPONSE] status={response.status_code}")
            if response.status_code != HTTP_STATUS_OK:
                logger.error(f"Response body: {response.text}")

            response.raise_for_status()
            result = response.json()

            if KEY_ERRORS in result:
                error_msg = result[KEY_ERRORS][0].get("message", "Unknown GraphQL error")
                logger.error(f"GraphQL error: {error_msg}")
                return Err(ProviderError(message=f"Failed to create pod: {error_msg}", code="RUNPOD_GRAPHQL_ERROR"))

            pod_data = result.get("data", {}).get("podFindAndDeployOnDemand")
            if not pod_data or not pod_data.get("id"):
                return Err(ProviderError(message=f"Failed to create pod: {result}", code="RUNPOD_POD_DATA_MISSING"))

            return Ok(self._log_and_build_create_result(pod_data, train_cfg))

        except requests.RequestException as e:
            logger.error(f"RunPod API error: {e}")
            return Err(ProviderError(message=f"Failed to create pod: {e}", code=_API_REQUEST_FAILED_CODE))

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
            ssh_info = self.extract_exposed_ssh_info(pod_data, pod_id=pod_id)
            if ssh_info.is_failure():
                return Err(ssh_info.unwrap_err())  # type: ignore[union-attr]
            return Ok(ssh_info.unwrap())

        except requests.RequestException as e:
            return Err(
                ProviderError(
                    message=f"Failed to get SSH info: {e}", code=_API_REQUEST_FAILED_CODE, details={_POD_ID_KEY: pod_id}
                )
            )

    @staticmethod
    def extract_exposed_ssh_info(
        pod_data: dict[str, Any] | None,
        *,
        pod_id: str | None = None,
    ) -> Result[dict[str, Any], ProviderError]:
        """Extract automation-grade SSH endpoint from RunPod pod data.

        Delegates to ``PodSnapshot.from_graphql`` for parsing, then converts
        the typed ``SshEndpoint`` back to the dict format expected by callers.
        """
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


__all__ = ["RunPodAPIClient"]

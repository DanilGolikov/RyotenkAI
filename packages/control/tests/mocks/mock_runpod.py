"""
Mock RunPod API and SSH classes for testing without network calls.
"""

from __future__ import annotations


class MockRunPodAPI:
    """
    Mock RunPod API client.

    Simulates GraphQL API operations without network calls.
    """

    def __init__(self):
        self.created_pods: list[dict] = []
        self.terminated_pods: list[str] = []
        self.queries: list[str] = []

        # Configuration
        self._next_pod_id = "pod_test_12345"
        self._pod_status = "RUNNING"
        self._cost_per_hr = 0.5
        self._should_fail_create = False
        self._should_fail_query = False
        self._should_fail_terminate = False

    def create_pod(
        self,
        name: str = "test-pod",
        image_name: str = "runpod/pytorch",
        gpu_type: str = "NVIDIA RTX 4060",
        **kwargs,
    ) -> dict:
        """
        Mock pod creation.

        Returns pod info dict similar to real API.
        """
        if self._should_fail_create:
            raise RuntimeError("Mock: Failed to create pod")

        pod = {
            "id": self._next_pod_id,
            "name": name,
            "status": self._pod_status,
            "imageName": image_name,
            "machine": {
                "gpu": gpu_type,
            },
            "runtime": {
                "gpus": [{"id": "gpu-0", "gpuUtilPercent": 0}],
                "ports": [{"ip": "185.1.2.3", "publicPort": 22169, "privatePort": 22, "type": "tcp"}],
            },
            "costPerHr": self._cost_per_hr,
            **kwargs,
        }

        self.created_pods.append(pod)
        return pod

    def query_pod(self, pod_id: str) -> dict | None:
        """
        Mock pod query.

        Returns pod info or None if not found.
        """
        if self._should_fail_query:
            raise RuntimeError("Mock: Failed to query pod")

        self.queries.append(pod_id)

        for pod in self.created_pods:
            if pod["id"] == pod_id:
                pod["status"] = self._pod_status
                return pod

        return None

    def terminate_pod(self, pod_id: str) -> bool:
        """
        Mock pod termination.

        Returns True on success.
        """
        if self._should_fail_terminate:
            raise RuntimeError("Mock: Failed to terminate pod")

        self.terminated_pods.append(pod_id)

        # Remove from created pods
        self.created_pods = [p for p in self.created_pods if p["id"] != pod_id]

        return True

    # Test helpers
    def set_pod_status(self, status: str) -> None:
        """Set status for all pods (RUNNING, EXITED, etc.)."""
        self._pod_status = status
        for pod in self.created_pods:
            pod["status"] = status

    def set_should_fail(
        self,
        create: bool = False,
        query: bool = False,
        terminate: bool = False,
    ) -> None:
        """Configure API failures for testing error handling."""
        self._should_fail_create = create
        self._should_fail_query = query
        self._should_fail_terminate = terminate

    def reset(self) -> None:
        """Reset mock state."""
        self.created_pods = []
        self.terminated_pods = []
        self.queries = []
        self._pod_status = "RUNNING"
        self._should_fail_create = False
        self._should_fail_query = False
        self._should_fail_terminate = False


class MockRunPodSSH:
    """
    Mock SSH client for RunPod.

    Simulates SSH operations without network connections.
    """

    def __init__(
        self,
        host: str = "185.1.2.3",
        port: int = 22169,
        user: str = "root",
    ):
        self.host = host
        self.port = port
        self.user = user

        # Tracking
        self.commands_executed: list[str] = []
        self.files_uploaded: list[tuple[str, str]] = []
        self.files_downloaded: list[tuple[str, str]] = []
        self.connection_tests: int = 0

        # Configuration
        self._is_connected = False
        self._should_fail_connect = False
        self._should_fail_command = False
        self._should_fail_upload = False
        self._should_fail_download = False
        self._command_outputs: dict[str, str] = {}

    def test_connection(self) -> tuple[bool, str]:
        """
        Test SSH connection.

        Returns (success, error_message).
        """
        self.connection_tests += 1

        if self._should_fail_connect:
            return False, "Mock: Connection refused"

        self._is_connected = True
        return True, ""

    def exec_command(self, cmd: str) -> tuple[bool, str]:
        """
        Execute command via SSH.

        Returns (success, output_or_error).
        """
        if self._should_fail_command:
            return False, "Mock: Command failed"

        self.commands_executed.append(cmd)

        # Return configured output or default
        output = self._command_outputs.get(cmd, f"Mock output for: {cmd}")
        return True, output

    def upload_file(self, local_path: str, remote_path: str) -> tuple[bool, str]:
        """
        Upload file via SCP.

        Returns (success, error_message).
        """
        if self._should_fail_upload:
            return False, "Mock: Upload failed"

        self.files_uploaded.append((local_path, remote_path))
        return True, ""

    def download_directory(self, remote_path: str, local_path: str) -> tuple[bool, str]:
        """
        Download directory via SCP.

        Returns (success, error_message).
        """
        if self._should_fail_download:
            return False, "Mock: Download failed"

        self.files_downloaded.append((remote_path, local_path))
        return True, ""

    def download_file(self, remote_path: str, local_path: str) -> tuple[bool, str]:
        """Download single file."""
        return self.download_directory(remote_path, local_path)

    # Test helpers
    def set_should_fail(
        self,
        connect: bool = False,
        command: bool = False,
        upload: bool = False,
        download: bool = False,
    ) -> None:
        """Configure failures for testing error handling."""
        self._should_fail_connect = connect
        self._should_fail_command = command
        self._should_fail_upload = upload
        self._should_fail_download = download

    def set_command_output(self, cmd: str, output: str) -> None:
        """Set specific output for a command."""
        self._command_outputs[cmd] = output

    def reset(self) -> None:
        """Reset mock state."""
        self.commands_executed = []
        self.files_uploaded = []
        self.files_downloaded = []
        self.connection_tests = 0
        self._is_connected = False
        self._should_fail_connect = False
        self._should_fail_command = False
        self._should_fail_upload = False
        self._should_fail_download = False
        self._command_outputs = {}


__all__ = ["MockRunPodAPI", "MockRunPodSSH"]

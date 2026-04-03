"""
Runpodctl client for local automation.

This module wraps the local `runpodctl` binary and exposes a small Result-based
API for pod lifecycle and file transfer operations.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from src.providers.runpod.models import read_ssh_public_key
from src.utils.result import Err, Ok, ProviderError, Result

_RUNPODCTL_NOT_AVAILABLE = "RUNPODCTL_NOT_AVAILABLE"
_RUNPODCTL_COMMAND_FAILED = "RUNPODCTL_COMMAND_FAILED"
_RUNPODCTL_OUTPUT_INVALID = "RUNPODCTL_OUTPUT_INVALID"


def resolve_runpodctl_binary(binary: str | None = None, settings_path: str = "") -> str:
    """Resolve runpodctl binary path with repo-local preference.

    Args:
        binary: Explicit path (highest priority).
        settings_path: Path from RuntimeSettings.runpodctl_path.
    """
    if binary:
        return binary

    if settings_path:
        return settings_path

    repo_root = Path(__file__).resolve().parents[3]
    local_binary = repo_root / "runpodctl"
    if local_binary.exists() and os.access(local_binary, os.X_OK):
        return str(local_binary)

    return "runpodctl"


class RunPodCtlClient:
    """Thin wrapper around the local runpodctl binary."""

    def __init__(self, *, api_key: str, api_url: str | None = None, binary: str | None = None):
        self._api_key = api_key
        self._api_url = api_url
        self._binary = resolve_runpodctl_binary(binary)

    def is_available(self) -> bool:
        """Whether runpodctl is installed on the local machine."""
        return shutil.which(self._binary) is not None

    def create_training_pod(
        self,
        *,
        gpu_type: str,
        image_name: str,
        pod_name: str,
        cloud_type: str,
        container_disk_gb: int,
        volume_disk_gb: int,
        ports: str,
        public_key: str | None = None,
    ) -> Result[dict[str, Any], ProviderError]:
        """Create a training pod via runpodctl."""
        args = [
            "pod",
            "create",
            "--output",
            "json",
            "--gpu-id",
            gpu_type,
            "--image",
            image_name,
            "--name",
            pod_name,
            "--container-disk-in-gb",
            str(container_disk_gb),
            "--volume-in-gb",
            str(volume_disk_gb),
            "--volume-mount-path",
            "/workspace",
            "--ports",
            ports,
            "--ssh",
            "--cloud-type",
            cloud_type.upper(),
        ]

        if public_key:
            args.extend(["--env", json.dumps({"PUBLIC_KEY": public_key.strip()})])

        return self._run_json(args, timeout=180)

    def get_pod(self, pod_id: str) -> Result[dict[str, Any], ProviderError]:
        """Fetch pod details via runpodctl."""
        return self._run_json(["pod", "get", pod_id, "--output", "json"], timeout=60)

    def start_pod(self, pod_id: str) -> Result[None, ProviderError]:
        """Start a pod via runpodctl."""
        result = self._run_json(["pod", "start", pod_id, "--output", "json"], timeout=60)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]
        return Ok(None)

    def stop_pod(self, pod_id: str) -> Result[None, ProviderError]:
        """Stop a pod via runpodctl."""
        result = self._run_json(["pod", "stop", pod_id, "--output", "json"], timeout=60)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]
        return Ok(None)

    def remove_pod(self, pod_id: str) -> Result[None, ProviderError]:
        """Remove a pod via runpodctl."""
        result = self._run_json(["pod", "delete", pod_id, "--output", "json"], timeout=60)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]
        return Ok(None)

    def send(self, *, local_path: str, code: str) -> Result[None, ProviderError]:
        """Send a file or directory via runpodctl."""
        result = self._run_text(["send", local_path, "--code", code], timeout=600)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]
        return Ok(None)

    def _run_json(self, args: list[str], *, timeout: int) -> Result[dict[str, Any], ProviderError]:
        result = self._run(args, timeout=timeout)
        if result.is_failure():
            return Err(result.unwrap_err())  # type: ignore[union-attr]

        stdout = result.unwrap().strip()
        if not stdout:
            return Ok({})

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return Err(
                ProviderError(
                    message=f"runpodctl returned invalid JSON: {exc}",
                    code=_RUNPODCTL_OUTPUT_INVALID,
                    details={"stdout": stdout[:1000]},
                )
            )

        if isinstance(data, dict):
            return Ok(data)

        return Err(
            ProviderError(
                message=f"runpodctl returned unexpected JSON type: {type(data).__name__}",
                code=_RUNPODCTL_OUTPUT_INVALID,
            )
        )

    def _run_text(self, args: list[str], *, timeout: int) -> Result[str, ProviderError]:
        return self._run(args, timeout=timeout)

    def _run(self, args: list[str], *, timeout: int) -> Result[str, ProviderError]:
        binary_path = shutil.which(self._binary)
        if binary_path is None:
            return Err(
                ProviderError(
                    message="runpodctl is not installed on the control machine",
                    code=_RUNPODCTL_NOT_AVAILABLE,
                )
            )

        env = os.environ.copy()
        env["RUNPOD_API_KEY"] = self._api_key
        if self._api_url:
            env["RUNPOD_API_URL"] = self._api_url

        cmd = [binary_path, *args]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return Err(
                ProviderError(
                    message=f"runpodctl command timed out after {timeout}s: {' '.join(args)}",
                    code=_RUNPODCTL_COMMAND_FAILED,
                )
            )
        except OSError as exc:
            return Err(
                ProviderError(
                    message=f"Failed to execute runpodctl: {exc}",
                    code=_RUNPODCTL_COMMAND_FAILED,
                )
            )

        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            return Err(
                ProviderError(
                    message=f"runpodctl command failed: {' '.join(args)} :: {stderr}",
                    code=_RUNPODCTL_COMMAND_FAILED,
                    details={"args": args, "stderr": stderr[:1000]},
                )
            )

        return Ok(completed.stdout or "")

    @staticmethod
    def read_public_key(key_path: str) -> str | None:
        """Read a sibling .pub file if it exists."""
        return read_ssh_public_key(key_path)


__all__ = ["RunPodCtlClient"]

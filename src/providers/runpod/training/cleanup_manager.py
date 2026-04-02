"""
RunPod Cleanup Manager - Manages pod cleanup and orphaned pods tracking.

Handles pod lifecycle cleanup and maintains registry of active pods.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from src.utils.logger import logger

if TYPE_CHECKING:
    from src.utils.result import ProviderError, Result


class _PodTerminateControl(Protocol):
    def terminate_pod(self, pod_id: str) -> Result[None, ProviderError]: ...


class RunPodCleanupManager:
    """
    Manages pod cleanup and orphaned pods tracking.

    Responsibilities:
    - Register active pods in JSON file (for recovery/debugging)
    - Unregister pods after successful cleanup
    - Cleanup single pod on demand
    """

    def __init__(self, api_client: _PodTerminateControl):
        """
        Initialize cleanup manager.

        Args:
            api_client: RunPodAPIClient instance for API operations
        """
        self.api_client = api_client
        # Keep the registry inside the project runs directory (control-plane friendly).
        project_root = Path(__file__).resolve().parents[4]
        self.registry_file = project_root / "runs" / "ryotenkai_active_pods.json"
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"🗑️ CleanupManager initialized: {self.registry_file}")

    def register_pod(self, pod_id: str, *, api_base: str) -> None:
        """
        Register an active pod for cleanup tracking.

        IMPORTANT (security):
        - We intentionally do NOT persist API keys/tokens on disk.
        - The control plane must provide RunPod credentials via environment/secrets at runtime.
        """
        try:
            pods = {}
            if self.registry_file.exists():
                try:
                    pods = json.loads(self.registry_file.read_text())
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted registry file, starting fresh: {self.registry_file}")
                    pods = {}

            pods[pod_id] = {
                "created_at": time.time(),
                "api_base": api_base,
            }

            self.registry_file.write_text(json.dumps(pods, indent=2))
            logger.debug(f"Registered pod {pod_id} for cleanup")
        except OSError as e:
            logger.warning(f"Failed to register pod for cleanup: {e}")

    def unregister_pod(self, pod_id: str) -> None:
        """Unregister a pod after successful cleanup."""
        try:
            if not self.registry_file.exists():
                return

            pods = json.loads(self.registry_file.read_text())
            if pod_id in pods:
                del pods[pod_id]
                self.registry_file.write_text(json.dumps(pods, indent=2))
                logger.debug(f"Unregistered pod {pod_id}")
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to unregister pod: {e}")

    def cleanup_pod(self, pod_id: str) -> Result[None, ProviderError]:
        """Cleanup single pod (terminate + unregister)."""
        logger.warning(f"🗑️ Cleaning up pod {pod_id}...")

        result = self.api_client.terminate_pod(pod_id)

        if result.is_success():
            self.unregister_pod(pod_id)

        return result

    def list_registered_pods(self) -> list[str]:
        """List all registered pod IDs (for debugging/manual cleanup)."""
        try:
            if not self.registry_file.exists():
                return []
            pods = json.loads(self.registry_file.read_text())
            return list(pods.keys())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to list registered pods: {e}")
            return []


def create_cleanup_manager(api_base: str, api_key: str) -> RunPodCleanupManager:
    """
    Create a cleanup manager instance.

    Args:
        api_base: RunPod API base URL
        api_key: RunPod API key

    Returns:
        RunPodCleanupManager instance
    """
    from src.providers.runpod.training.api_client import RunPodAPIClient

    api_client = RunPodAPIClient(api_base_url=api_base, api_key=api_key)
    return RunPodCleanupManager(api_client)


__all__ = ["RunPodCleanupManager", "create_cleanup_manager"]

"""
RunPod Cleanup Manager - Terminates RunPod training pods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.utils.logger import logger

if TYPE_CHECKING:
    from src.utils.result import ProviderError, Result


class _PodTerminateControl(Protocol):
    def terminate_pod(self, pod_id: str) -> Result[None, ProviderError]: ...


class RunPodCleanupManager:
    """Thin wrapper over the RunPod API that terminates pods on demand."""

    def __init__(self, api_client: _PodTerminateControl):
        self.api_client = api_client

    def cleanup_pod(self, pod_id: str) -> Result[None, ProviderError]:
        """Terminate a single pod."""
        logger.warning(f"🗑️ Cleaning up pod {pod_id}...")
        return self.api_client.terminate_pod(pod_id)


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

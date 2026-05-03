"""
RunPod Provider - cloud GPU via RunPod API.

For training on RunPod cloud instances (A40, H100, etc.)

Features:
    - GraphQL API for pod management
    - Automatic pod creation and termination
    - SSH access to running pods
    - Health checks and monitoring

Example config:
    providers:
      runpod:
        connect:
          ssh:
            key_path: ~/.ssh/id_ed25519_runpod
        cleanup:
          auto_delete_pod: true
        training:
          gpu_type: "NVIDIA A40"
          cloud_type: ALL
          image_name: ryotenkai/ryotenkai-training-runtime:latest
          container_disk_gb: 100
          volume_disk_gb: 20
          ports: "8888/http,22/tcp"

Usage:
    from src.providers.runpod.training import RunPodProvider

    provider = RunPodProvider(config=provider_config, secrets=secrets)
    result = provider.connect()  # Creates pod, waits for ready

    # ... deploy training ...

    provider.disconnect()  # Terminates pod
"""

# Auto-register with factory
from src.constants import PROVIDER_RUNPOD
from src.providers.training.factory import GPUProviderFactory

from .api_client import RunPodAPIClient
from .cleanup_manager import RunPodCleanupManager, create_cleanup_manager
from .config import RunPodProviderConfig
from .lifecycle_manager import PodLifecycleManager
from .provider import RunPodProvider

if not GPUProviderFactory.is_registered(PROVIDER_RUNPOD):
    GPUProviderFactory.register(PROVIDER_RUNPOD, RunPodProvider)

__all__ = [
    "PodLifecycleManager",
    "RunPodAPIClient",
    "RunPodCleanupManager",
    "RunPodProvider",
    "RunPodProviderConfig",
    "create_cleanup_manager",
]

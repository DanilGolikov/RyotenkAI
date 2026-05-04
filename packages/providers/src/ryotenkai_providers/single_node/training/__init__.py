"""
SingleNode Provider - local PC via SSH.

For training on personal GPU servers (RTX 4060, 4090, etc.)

Features:
    - Direct SSH connection (no cloud API)
    - Auto GPU detection via nvidia-smi
    - MemoryManager integration for auto batch_size
    - Health checks before training

Example config:
    providers:
      single_node:
        connect:
          ssh:
            alias: pc                        # Use SSH alias from ~/.ssh/config
        cleanup:
          cleanup_workspace: false
          keep_on_error: true
          on_interrupt: true
        training:
          workspace_path: /home/user/workspace
          docker_image: my/training-runtime:latest
          gpu_type: "RTX 4060"               # Optional, for logging

Usage:
    from ryotenkai_providers.single_node.training import SingleNodeProvider

    provider = SingleNodeProvider(config=provider_config, secrets=secrets)
    result = provider.connect()
"""

# Registration is manifest-driven (provider.toml). PR-1.11 removed the
# legacy GPUProviderFactory.register call; the ProviderRegistry walks
# the on-disk manifests at process start.
from .config import SingleNodeProviderConfig
from .health_check import SingleNodeHealthCheck
from .provider import SingleNodeProvider

__all__ = [
    "SingleNodeProviderConfig",
    "SingleNodeHealthCheck",
    "SingleNodeProvider",
]

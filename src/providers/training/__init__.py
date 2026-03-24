"""
Training provider contracts and factory.

Canonical locations:
- Interfaces: `src.providers.training.interfaces`
- Factory: `src.providers.training.factory`
- SSH client: `src.utils.ssh_client`
"""

from .factory import GPUProviderFactory, auto_register_providers
from .interfaces import (
    GPUInfo,
    IGPUProvider,
    ProviderCapabilities,
    ProviderFactory,
    ProviderStatus,
    SSHConnectionInfo,
)

# Importing this package should behave similarly to legacy `src.pipeline.providers`:
# it triggers best-effort auto-registration of built-in providers.
auto_register_providers()

__all__ = [
    "GPUInfo",
    "GPUProviderFactory",
    "IGPUProvider",
    "ProviderCapabilities",
    "ProviderFactory",
    "ProviderStatus",
    "SSHConnectionInfo",
    "auto_register_providers",
]

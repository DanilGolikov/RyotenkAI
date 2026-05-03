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

# Importing this package triggers best-effort auto-registration of built-in
# providers — that's why the cross-validator imports it dynamically when
# checking that the YAML's `training.provider` is a known name.
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

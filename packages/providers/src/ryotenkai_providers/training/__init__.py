"""Training-side provider Protocols + base class.

The :class:`GPUProviderFactory` and ``auto_register_providers`` were
removed in the manifest-driven registry migration (PR-1.11). Use
:class:`ryotenkai_providers.registry.ProviderRegistry` instead — auto-
discovers providers from ``provider.toml`` manifests on disk.
"""

from .interfaces import (
    GPUInfo,
    ICapacityErrorClassifier,
    IGPUProvider,
    IRecoveryProbeProvider,
    ITerminalActionProvider,
    ProviderBase,
    ProviderCapabilities,
    ProviderStatus,
    SSHConnectionInfo,
)

__all__ = [
    "GPUInfo",
    "ICapacityErrorClassifier",
    "IGPUProvider",
    "IRecoveryProbeProvider",
    "ITerminalActionProvider",
    "ProviderBase",
    "ProviderCapabilities",
    "ProviderStatus",
    "SSHConnectionInfo",
]

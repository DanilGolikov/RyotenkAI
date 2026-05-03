"""
Provider-specific configuration schemas.

Goal:
- Keep provider config schemas in ONE place (this package).
- Runtime provider implementations (`src/providers/<provider>/*`) import schemas from here.
- Provider modules may keep thin re-export wrappers for backward compatibility.
"""

from .runpod import (
    RunPodCleanupConfig,
    RunPodConnectConfig,
    RunPodInferencePodConfig,
    RunPodInferenceServeConfig,
    RunPodNetworkVolumeConfig,
    RunPodPodsInferenceConfig,
    RunPodProviderConfig,
    RunPodSSHConfig,
    RunPodTrainingConfig,
)
from .single_node import (
    SingleNodeCleanupConfig,
    SingleNodeConfig,
    SingleNodeConnectConfig,
    SingleNodeInferenceConfig,
    SingleNodeTrainingConfig,
)
from .ssh import SSHConfig, SSHConnectSettings
from .template import TemplateProviderConfig

__all__ = [
    "RunPodCleanupConfig",
    "RunPodConnectConfig",
    "RunPodInferencePodConfig",
    "RunPodInferenceServeConfig",
    "RunPodNetworkVolumeConfig",
    "RunPodPodsInferenceConfig",
    "RunPodProviderConfig",
    "RunPodSSHConfig",
    "RunPodTrainingConfig",
    "SSHConfig",
    "SSHConnectSettings",
    "SingleNodeCleanupConfig",
    "SingleNodeConfig",
    "SingleNodeConnectConfig",
    "SingleNodeInferenceConfig",
    "SingleNodeTrainingConfig",
    "TemplateProviderConfig",
]

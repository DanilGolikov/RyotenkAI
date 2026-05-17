"""
Centralized configuration schema package.

This package hosts Pydantic models and helpers for pipeline configuration.
`src/utils/config.py` remains a backward-compatible facade during migration.
"""

from .base import StrictBaseModel
from .datasets import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceHF,
    DatasetSourceLocal,
    DatasetSourceUnion,
    DatasetValidationPluginConfig,
    DatasetValidationsConfig,
)
from .inference import (
    InferenceChatUIConfig,
    InferenceCommonConfig,
    InferenceConfig,
    InferenceHealthCheckConfig,
    InferenceLoRAConfig,
    InferenceSingleNodeServeConfig,
)
from .integrations import (
    IntegrationsConfig,
    HuggingFaceConfig,
    HuggingFaceHubConfig,
    MLflowConfig,
)
from .model import ModelConfig
from .pipeline import PipelineConfig
from .pod_lifecycle import PodLifecycleConfig
from .providers.ssh import SSHConfig, SSHConnectSettings
from .reports import ReportsConfig
from .runtime import RuntimeSettings, load_runtime_settings
from .secrets import Secrets, load_secrets
from .training import (
    VALID_START_STRATEGIES,
    VALID_STRATEGY_TRANSITIONS,
    AdaLoraConfig,
    AdapterConfigUnion,
    GlobalHyperparametersConfig,
    LoRAConfig,
    LoraConfig,
    PhaseHyperparametersConfig,
    QLoRAConfig,
    QloraConfig,
    StrategyPhaseConfig,
    TrainingConfig,
    TrainingOnlyConfig,
    validate_strategy_chain,
)

__all__ = [
    "VALID_START_STRATEGIES",
    "VALID_STRATEGY_TRANSITIONS",
    "AdaLoraConfig",
    "AdapterConfigUnion",
    "DatasetConfig",
    "DatasetLocalPaths",
    "DatasetSourceHF",
    "DatasetSourceLocal",
    "DatasetSourceUnion",
    "DatasetValidationPluginConfig",
    "DatasetValidationsConfig",
    "IntegrationsConfig",
    "GlobalHyperparametersConfig",
    "HuggingFaceConfig",
    "HuggingFaceHubConfig",
    "InferenceChatUIConfig",
    "InferenceCommonConfig",
    "InferenceConfig",
    "InferenceHealthCheckConfig",
    "InferenceLoRAConfig",
    "InferenceSingleNodeServeConfig",
    "LoRAConfig",
    "LoraConfig",
    "MLflowConfig",
    "ModelConfig",
    "PhaseHyperparametersConfig",
    "PipelineConfig",
    "PodLifecycleConfig",
    "QLoRAConfig",
    "QloraConfig",
    "ReportsConfig",
    "RuntimeSettings",
    "SSHConfig",
    "SSHConnectSettings",
    "Secrets",
    "StrategyPhaseConfig",
    "StrictBaseModel",
    "TrainingConfig",
    "TrainingOnlyConfig",
    "load_runtime_settings",
    "load_secrets",
    "validate_strategy_chain",
]

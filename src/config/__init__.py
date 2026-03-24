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
    DatasetValidationPluginConfig,
    DatasetValidationsConfig,
)
from .inference import (
    InferenceChatUIConfig,
    InferenceCommonConfig,
    InferenceConfig,
    InferenceEnginesConfig,
    InferenceHealthCheckConfig,
    InferenceLoRAConfig,
    InferenceSingleNodeServeConfig,
    InferenceVLLMEngineConfig,
)
from .integrations import (
    ExperimentTrackingConfig,
    HuggingFaceConfig,
    HuggingFaceHubConfig,
    MLflowConfig,
)
from .model import ModelConfig
from .pipeline import PipelineConfig, load_config
from .providers.ssh import SSHConfig, SSHConnectSettings
from .secrets import Secrets, load_secrets
from .training import (
    VALID_START_STRATEGIES,
    VALID_STRATEGY_TRANSITIONS,
    AdaLoraConfig,
    GlobalHyperparametersConfig,
    LoRAConfig,
    LoraConfig,
    PhaseHyperparametersConfig,
    QLoRAConfig,
    StrategyPhaseConfig,
    TrainingConfig,
    TrainingOnlyConfig,
    validate_strategy_chain,
)

__all__ = [
    "VALID_START_STRATEGIES",
    "VALID_STRATEGY_TRANSITIONS",
    "AdaLoraConfig",
    "DatasetConfig",
    "DatasetLocalPaths",
    "DatasetSourceHF",
    "DatasetSourceLocal",
    "DatasetValidationPluginConfig",
    "DatasetValidationsConfig",
    "ExperimentTrackingConfig",
    "GlobalHyperparametersConfig",
    "HuggingFaceConfig",
    "HuggingFaceHubConfig",
    "InferenceChatUIConfig",
    "InferenceCommonConfig",
    "InferenceConfig",
    "InferenceEnginesConfig",
    "InferenceHealthCheckConfig",
    "InferenceLoRAConfig",
    "InferenceSingleNodeServeConfig",
    "InferenceVLLMEngineConfig",
    "LoRAConfig",
    "LoraConfig",
    "MLflowConfig",
    "ModelConfig",
    "PhaseHyperparametersConfig",
    "PipelineConfig",
    "QLoRAConfig",
    "SSHConfig",
    "SSHConnectSettings",
    "Secrets",
    "StrategyPhaseConfig",
    "StrictBaseModel",
    "TrainingConfig",
    "TrainingOnlyConfig",
    "load_config",
    "load_secrets",
    "validate_strategy_chain",
]

"""Generic, side-side-agnostic utilities.

After Phase A.7 of monorepo packagization (plan §A.7) the trainer-only
``container`` and ``memory_manager`` modules moved to
:mod:`src.training` — Mac-side code (pipeline, api, cli) does not pull
GPU-introspection / DI-container code anymore. Import them from
``src.training.{container,memory_manager}`` directly.
"""

from ryotenkai_shared.config import PipelineConfig, Secrets, load_secrets
from .environment import EnvironmentReporter, EnvironmentSnapshot
from .logger import console, logger
from .result import (
    AppError,
    ConfigError,
    DatasetError,
    Err,
    Failure,
    InferenceError,
    ModelError,
    Ok,
    OOMError,
    ProviderError,
    Result,
    StrategyError,
    Success,
    TrainingError,
    err,
    ok,
)

__all__ = [
    "AppError",
    "ConfigError",
    "DatasetError",
    "EnvironmentReporter",
    "EnvironmentSnapshot",
    "Err",
    "Failure",
    "InferenceError",
    "ModelError",
    "OOMError",
    "Ok",
    "PipelineConfig",
    "ProviderError",
    "Result",
    "Secrets",
    "StrategyError",
    "Success",
    "TrainingError",
    "console",
    "err",
    "load_secrets",
    "logger",
    "ok",
]

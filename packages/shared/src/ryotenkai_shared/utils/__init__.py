"""Generic, side-side-agnostic utilities.

After Phase A.7 of monorepo packagization (plan §A.7) the trainer-only
``container`` and ``memory_manager`` modules moved to
:mod:`src.training` — Mac-side code (pipeline, api, cli) does not pull
GPU-introspection / DI-container code anymore. Import them from
``src.training.{container,memory_manager}`` directly.

Phase A2 finale (2026-05-16): the legacy ``Result[T, AppError]`` module
(``ryotenkai_shared.utils.result``) is deleted. All error handling
goes through :mod:`ryotenkai_shared.errors` (RyotenkAIError hierarchy).
"""

from ryotenkai_shared.config import PipelineConfig, Secrets, load_secrets
from .environment import EnvironmentReporter, EnvironmentSnapshot
from .logger import console, logger

__all__ = [
    "EnvironmentReporter",
    "EnvironmentSnapshot",
    "PipelineConfig",
    "Secrets",
    "console",
    "load_secrets",
    "logger",
]

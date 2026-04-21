"""Fail-fast startup validation for the pipeline.

The bootstrap package exposes :class:`StartupValidator` — a pure validator
that inspects the (config, secrets) pair for missing credentials and
incompatible strategy chains before any stage runs. Extracting this from
``PipelineOrchestrator.__init__`` keeps construction concerns tight and
makes startup validation test-in-isolation.
"""

from src.pipeline.bootstrap.startup_validator import (
    StartupValidationError,
    StartupValidator,
)

__all__ = ["StartupValidationError", "StartupValidator"]

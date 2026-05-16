"""Fail-fast startup validation for the pipeline.

The bootstrap package exposes :class:`StartupValidator` — a pure validator
that inspects the (config, secrets) pair for missing credentials and
incompatible strategy chains before any stage runs. Extracting this from
``PipelineOrchestrator.__init__`` keeps construction concerns tight and
makes startup validation test-in-isolation.

Validation failures raise typed :class:`RyotenkAIError` subclasses
(``ProviderAuthFailedError`` for missing credentials,
``StrategyChainInvalidError`` for chain rejects) directly — the legacy
``StartupValidationError(ValueError)`` wrapper was removed in the
worker-rendering bug fix (2026-05-16) because it stripped typed
semantics, causing the worker subprocess to print raw Python tracebacks
instead of the unified kubectl-style CLI output.
"""

from ryotenkai_control.pipeline.bootstrap.pipeline_bootstrap import BootstrapResult, PipelineBootstrap
from ryotenkai_control.pipeline.bootstrap.startup_validator import StartupValidator

__all__ = [
    "BootstrapResult",
    "PipelineBootstrap",
    "StartupValidator",
]

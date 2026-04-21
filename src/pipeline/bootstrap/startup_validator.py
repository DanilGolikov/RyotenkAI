"""Startup-time validation of secrets and strategy chains.

This module owns every fail-fast check that must pass before the
orchestrator builds collaborators / stages. Running them in one place
means:

* Construction-time failures are distinguishable from runtime failures
  (distinct exception type, distinct test surface).
* Tests do not need to instantiate a full orchestrator to verify a
  single secret rule — they call ``StartupValidator.validate`` directly.
* The orchestrator ``__init__`` stays a thin coordinator.

Validation covers:

1. ``HF_TOKEN`` — always set in the environment (every HF integration
   depends on it).
2. Training-provider secrets — currently RunPod requires ``RUNPOD_API_KEY``.
3. Inference-provider secrets — the inference-specific provider check.
4. Evaluation plugin secrets — delegated to :func:`validate_eval_plugin_secrets`.
5. Training strategy chain — delegated to :func:`validate_strategy_chain`.

All failures raise :class:`StartupValidationError` so callers can
distinguish "bad input" from unexpected runtime errors.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.config.validators.runtime import validate_eval_plugin_secrets
from src.constants import PROVIDER_RUNPOD
from src.utils.config import validate_strategy_chain
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.utils.config import PipelineConfig, Secrets


class StartupValidationError(ValueError):
    """Raised when startup validation detects a fatal misconfiguration.

    Subclasses :class:`ValueError` for backward compatibility with the
    previous orchestrator code that used bare ``ValueError``.
    """


class StartupValidator:
    """Pure validator for (config, secrets) pairs.

    Stateless — all methods are static-style but attached to the class for
    discoverability. Call :meth:`validate` for the full suite, or the
    individual ``check_*`` methods for finer-grained testing.
    """

    @classmethod
    def validate(cls, *, config: PipelineConfig, secrets: Secrets) -> None:
        """Run every startup check. Raises on the first failure.

        Side effects:
        * Sets ``HF_TOKEN`` in ``os.environ`` (every HuggingFace integration
          reads that variable). Mirrors the pre-refactor behaviour.
        """
        cls.set_hf_token_env(secrets=secrets)
        cls.check_training_provider_secrets(config=config, secrets=secrets)
        cls.check_inference_provider_secrets(config=config, secrets=secrets)
        cls.check_eval_plugin_secrets(config=config, secrets=secrets)
        cls.check_strategy_chain(config=config)

    # ------------------------------------------------------------------
    # Individual checks (test-in-isolation surface)
    # ------------------------------------------------------------------

    @staticmethod
    def set_hf_token_env(*, secrets: Secrets) -> None:
        """Expose ``HF_TOKEN`` through ``os.environ`` (str-cast for MagicMock)."""
        os.environ["HF_TOKEN"] = str(secrets.hf_token)

    @staticmethod
    def check_training_provider_secrets(
        *, config: PipelineConfig, secrets: Secrets
    ) -> None:
        """Fail when the active training provider needs a secret we don't have."""
        try:
            active_provider = config.get_active_provider_name()
        except (ValueError, AttributeError):
            # Config may be incomplete in tests or during initialization —
            # treat "cannot determine provider" as "no provider-specific
            # secrets required" rather than a validation failure.
            return

        if active_provider == PROVIDER_RUNPOD and not getattr(
            secrets, "runpod_api_key", None
        ):
            raise StartupValidationError(
                f"RUNPOD_API_KEY is required when using provider {PROVIDER_RUNPOD!r}. "
                "Set it via environment variable RUNPOD_API_KEY or in config/secrets.env."
            )

    @staticmethod
    def check_inference_provider_secrets(
        *, config: PipelineConfig, secrets: Secrets
    ) -> None:
        """Fail when an enabled inference provider needs a secret we don't have."""
        inference_cfg = getattr(config, "inference", None)
        if (
            getattr(inference_cfg, "enabled", False) is True
            and getattr(inference_cfg, "provider", None) in {PROVIDER_RUNPOD}
            and not getattr(secrets, "runpod_api_key", None)
        ):
            raise StartupValidationError(
                f"RUNPOD_API_KEY is required when using inference.provider="
                f"{getattr(inference_cfg, 'provider', None)!r}. "
                "Set it via environment variable RUNPOD_API_KEY or in config/secrets.env."
            )

    @staticmethod
    def check_eval_plugin_secrets(
        *, config: PipelineConfig, secrets: Secrets
    ) -> None:
        """Validate that every enabled eval plugin's secrets are present."""
        # Delegates to the shared validator; propagates its exception class
        # unchanged so existing tests that patch it keep working.
        validate_eval_plugin_secrets(config, secrets)

    @staticmethod
    def check_strategy_chain(*, config: PipelineConfig) -> None:
        """Fail when the configured training strategy chain is invalid."""
        strategies = config.training.strategies
        if not strategies:
            return
        validation = validate_strategy_chain(strategies)
        chain_str = " -> ".join(s.strategy_type.upper() for s in strategies)
        if validation.is_failure():
            error = validation.unwrap_err()
            logger.error(f"Invalid strategy chain: {chain_str}")
            logger.error(f"   Error: {error}")
            raise StartupValidationError(f"Invalid strategy chain: {error}")
        logger.info(f"Strategy chain checked: {chain_str}")


__all__ = ["StartupValidationError", "StartupValidator"]

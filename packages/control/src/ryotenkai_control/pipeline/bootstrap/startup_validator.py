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

from ryotenkai_control.pipeline.validators.runtime import validate_eval_plugin_secrets
from ryotenkai_shared.constants import PROVIDER_RUNPOD, PROVIDER_SINGLE_NODE
from ryotenkai_shared.config import validate_strategy_chain
from ryotenkai_shared.utils.logger import logger


def _resolve_required_secrets_for_provider(
    provider_name: str,
) -> tuple[str, ...]:
    """Manifest-driven secret requirements (Phase 14.D+F final form).

    Delegates to :meth:`ProviderRegistry.required_secrets` — the
    provider's ``provider.toml`` is the single source of truth.
    Replaces the legacy hardcoded ``if provider_name == PROVIDER_RUNPOD``
    string-dispatch chain. Adding a third provider = drop a
    ``provider.toml`` next to the new package; no code change here.

    Returns ``()`` for unknown ids — the validator surfaces a "not
    registered" error elsewhere with a clearer message.
    """
    from ryotenkai_providers.registry import get_registry

    try:
        return get_registry().required_secrets(
            provider_name, role="training",
        )
    except KeyError:
        return ()

if TYPE_CHECKING:
    from ryotenkai_shared.config import PipelineConfig, Secrets


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
        """Fail when the active training provider needs a secret we don't have.

        Phase 14.D+F: provider-driven validation. Pre-14.D this hardcoded
        ``active_provider == PROVIDER_RUNPOD`` and read ``runpod_api_key``
        directly. Now we resolve the active provider's class and iterate
        :meth:`IGPUProvider.required_secrets` — adding a third provider
        is a one-line registry change, not a string-check edit here.
        """
        try:
            active_provider = config.get_active_provider_name()
        except (ValueError, AttributeError):
            # Config may be incomplete in tests or during initialization —
            # treat "cannot determine provider" as "no provider-specific
            # secrets required" rather than a validation failure.
            return

        # Map provider name → required secret tuple. Lazy-resolved
        # so test-time stubs of provider modules don't trip on
        # heavy imports. The map is closed (matches the registry in
        # :mod:`src.runner.runtime.provider_registry`); adding a
        # provider = update both registries.
        required_secrets = _resolve_required_secrets_for_provider(
            active_provider,
        )
        for secret_name in required_secrets:
            attr_name = secret_name.lower()
            if not getattr(secrets, attr_name, None):
                raise StartupValidationError(
                    f"{secret_name} is required when using provider "
                    f"{active_provider!r}. Set it via environment "
                    f"variable {secret_name} or in config/secrets.env.",
                )

    @staticmethod
    def check_inference_provider_secrets(
        *, config: PipelineConfig, secrets: Secrets
    ) -> None:
        """Fail when an enabled inference provider needs a secret we don't have.

        Phase 14.D+F: same provider-driven pattern as
        :meth:`check_training_provider_secrets`. Inference reuses
        the training provider's secret list — both surfaces hit
        the same RunPod API key.
        """
        inference_cfg = getattr(config, "inference", None)
        if not getattr(inference_cfg, "enabled", False):
            return

        inference_provider = getattr(inference_cfg, "provider", None)
        if not isinstance(inference_provider, str):
            return

        required_secrets = _resolve_required_secrets_for_provider(
            inference_provider,
        )
        for secret_name in required_secrets:
            attr_name = secret_name.lower()
            if not getattr(secrets, attr_name, None):
                raise StartupValidationError(
                    f"{secret_name} is required when using "
                    f"inference.provider={inference_provider!r}. "
                    f"Set it via environment variable {secret_name} "
                    f"or in config/secrets.env.",
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
        from ryotenkai_shared.errors import StrategyChainInvalidError

        strategies = config.training.strategies
        if not strategies:
            return
        chain_str = " -> ".join(s.strategy_type.upper() for s in strategies)
        try:
            validate_strategy_chain(strategies)
        except StrategyChainInvalidError as exc:
            logger.error(f"Invalid strategy chain: {chain_str}")
            logger.error(f"   Error: {exc}")
            raise StartupValidationError(f"Invalid strategy chain: {exc}") from exc
        logger.info(f"Strategy chain checked: {chain_str}")


__all__ = ["StartupValidationError", "StartupValidator"]

"""
GPU Provider Factory - creates provider instances from configuration.

Factory pattern implementation for GPU providers.
Providers are registered by name (e.g., "single_node", "runpod").
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from src.utils.result import Failure, ProviderError, Success, err

if TYPE_CHECKING:
    from src.providers.training.interfaces import IGPUProvider
    from src.utils.config import Secrets

logger = logging.getLogger("ryotenkai")


class ProviderConstructor(Protocol):
    """Callable that constructs an IGPUProvider from (config, secrets)."""

    def __call__(self, config: dict[str, Any], secrets: Secrets) -> IGPUProvider: ...


class GPUProviderFactory:
    """
    Factory for creating GPU provider instances.

    Providers are registered by name and created using that name.
    The provider name in YAML config = identifier for Factory.

    Built-in providers:
        - "single_node": SingleNodeProvider (local PC via SSH)
        - "runpod": RunPodProvider (RunPod cloud)

    Example:
        # Registration (done automatically in provider modules)
        GPUProviderFactory.register("single_node", SingleNodeProvider)
        GPUProviderFactory.register("runpod", RunPodProvider)

        # Usage
        provider = GPUProviderFactory.create(
            provider_name="single_node",
            provider_config=config.providers["single_node"],
            secrets=secrets,
        )
    """

    # Registry: provider name → provider class
    _providers: ClassVar[dict[str, ProviderConstructor]] = {}

    @classmethod
    def register(cls, name: str, provider_class: ProviderConstructor) -> None:
        """
        Register a provider by name.

        Args:
            name: Provider name (e.g., "single_node", "runpod")
            provider_class: Provider class implementing IGPUProvider
        """
        provider_display_name = getattr(provider_class, "__name__", type(provider_class).__name__)

        if name in cls._providers:
            logger.warning(
                f"[FACTORY:OVERRIDE] Provider '{name}' already registered, overriding with {provider_display_name}"
            )

        cls._providers[name] = provider_class
        logger.debug(f"[FACTORY:REGISTERED] name={name}, class={provider_display_name}")

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a provider.

        Args:
            name: Provider name to unregister

        Returns:
            True if unregistered, False if not found
        """
        if name in cls._providers:
            del cls._providers[name]
            logger.debug(f"[FACTORY:UNREGISTERED] name={name}")
            return True
        return False

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if provider is registered."""
        return name in cls._providers

    @classmethod
    def create(
        cls,
        provider_name: str,
        provider_config: dict[str, Any],
        secrets: Secrets,
    ) -> Success[IGPUProvider] | Failure[ProviderError]:
        """
        Create provider instance by name.

        Args:
            provider_name: Provider name (key from providers: block in YAML)
            provider_config: Provider configuration dict
            secrets: Secrets object with API keys and credentials

        Returns:
            Ok(IGPUProvider): Provider instance
            Err(ProviderError): Structured error if provider is unknown or instantiation fails
        """
        if provider_name not in cls._providers:
            available = cls.get_available_providers()
            return err(
                ProviderError(
                    message=(
                        f"Unknown provider: '{provider_name}'. "
                        f"Available providers: {available}. "
                        f"Did you forget to import the provider module?"
                    ),
                    code="PROVIDER_NOT_REGISTERED",
                    details={"provider_name": provider_name, "available": available},
                )
            )

        provider_factory = cls._providers[provider_name]
        provider_display_name = getattr(provider_factory, "__name__", type(provider_factory).__name__)

        logger.info(f"[FACTORY:CREATE] name={provider_name}, class={provider_display_name}")

        try:
            return Success(provider_factory(provider_config, secrets))
        except TypeError as e:
            return err(
                ProviderError(
                    message=(
                        f"Failed to create provider '{provider_name}': {e}. "
                        f"Check that {provider_display_name}.__init__ accepts (config, secrets) arguments."
                    ),
                    code="PROVIDER_INIT_FAILED",
                    details={"provider_name": provider_name, "exception": str(e)},
                )
            )

    @classmethod
    def create_from_config(
        cls,
        provider_name: str,
        providers_config: dict[str, dict[str, Any]],
        secrets: Secrets,
    ) -> Success[IGPUProvider] | Failure[ProviderError]:
        """
        Create provider from providers registry config.

        Convenience method for creating provider from pipeline config.

        Args:
            provider_name: Name in providers config (e.g., "single_node", "runpod")
            providers_config: Dict of all provider configs
            secrets: Secrets object

        Returns:
            Ok(IGPUProvider): Provider instance
            Err(ProviderError): Structured error if config key missing or provider unknown
        """
        if provider_name not in providers_config:
            available = list(providers_config.keys())
            return err(
                ProviderError(
                    message=f"Provider '{provider_name}' not found in config. Available: {available}",
                    code="PROVIDER_CONFIG_MISSING",
                    details={"provider_name": provider_name, "available": available},
                )
            )

        provider_config = providers_config[provider_name]
        return cls.create(
            provider_name=provider_name,
            provider_config=provider_config,
            secrets=secrets,
        )


def auto_register_providers() -> None:
    """
    Auto-register all built-in providers.

    Called when providers package is imported.
    Import order matters - providers must be imported after factory.
    """
    # Import here to avoid circular imports
    # These imports will register providers via their module-level code
    try:
        importlib.import_module("src.providers.single_node.training")
        logger.debug("[FACTORY:AUTO_REGISTER] single_node provider loaded")
    except ImportError as e:
        logger.debug(f"[FACTORY:AUTO_REGISTER] single_node not available: {e}")

    try:
        importlib.import_module("src.providers.runpod.training")
        logger.debug("[FACTORY:AUTO_REGISTER] runpod provider loaded")
    except ImportError as e:
        logger.debug(f"[FACTORY:AUTO_REGISTER] runpod not available: {e}")

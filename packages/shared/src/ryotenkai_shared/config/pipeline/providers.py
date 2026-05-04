from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PipelineProviderMixin:
    """
    Provider-related helper methods for PipelineConfig.

    NOTE: This mixin must be used with a Pydantic model that defines:
    - self.providers: dict[str, Any]   # YAML dict OR provider-specific BaseModel
    - self.training.provider: str | None

    After ``PipelineConfig._validate_provider_blocks_against_manifests``
    runs, ``self.providers[id]`` is the provider's typed Pydantic
    schema instance (e.g. ``RunPodProviderConfig``). Modular runtimes
    or unit tests that bypass the validator may still see raw dicts.
    Callers that explicitly need a dict should use
    :meth:`get_provider_config_as_dict`.
    """

    providers: dict[str, Any]
    training: Any

    def get_provider_config(self, name: str | None = None) -> Any:
        """
        Get provider config block by name.

        Returns the post-validation typed Pydantic model for known
        providers; raw dict for unknown ones (modular runtime / tests).

        Args:
            name: Provider id from registry. None = use training.provider.

        Raises:
            ValueError: If no provider specified or not found.
        """
        # Local import to avoid heavy side-effects at module import time.
        from ryotenkai_shared.utils.logger import logger

        if name is None:
            name = self.training.provider

        if name is None:
            raise ValueError(
                "No provider specified. Set 'training.provider' in config or pass provider name explicitly."
            )

        if name not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(
                f"Provider '{name}' not found. Available: {available}. Add it to 'providers:' section in config."
            )

        logger.debug(f"[CFG:PROVIDER] Resolved provider: {name}")
        return self.providers[name]

    def get_provider_config_as_dict(self, name: str | None = None) -> dict[str, Any]:
        """Same as :meth:`get_provider_config` but always returns dict.

        Used by callers that hash, serialize, or otherwise treat the
        block as a plain mapping (config-drift hashing,
        markdown/JSON reports). Pydantic models are flattened via
        ``model_dump(mode="json")``.
        """
        block = self.get_provider_config(name)
        if isinstance(block, BaseModel):
            return block.model_dump(mode="json")
        if isinstance(block, dict):
            return block
        raise TypeError(
            f"Provider block for {name!r} has unexpected type: {type(block).__name__}"
        )

    def get_provider_training_config(self, name: str | None = None) -> Any:
        """
        Get a training-scoped view of provider config.

        Returns the typed ``training`` sub-block (e.g.
        ``RunPodTrainingConfig``) when the parent block is typed,
        or the raw ``training`` dict for legacy / modular-runtime
        configs.

        Raises:
            ValueError: If the provider block has no ``training`` section.
        """
        provider_cfg = self.get_provider_config(name)

        if isinstance(provider_cfg, BaseModel):
            training_cfg = getattr(provider_cfg, "training", None)
            if training_cfg is None:
                raise ValueError(
                    f"Provider config '{name}' is missing the 'training' block."
                )
            return training_cfg

        if isinstance(provider_cfg, dict):
            training_cfg = provider_cfg.get("training")
            if not isinstance(training_cfg, dict):
                raise ValueError(
                    f"Provider config must contain 'training:' block (dict). Got: {type(training_cfg).__name__}"
                )
            return training_cfg

        raise ValueError(
            f"Provider config has unexpected type: {type(provider_cfg).__name__}"
        )

    def get_active_provider_name(self) -> str:
        """
        Get the active provider name.

        Returns:
            Provider name from training.provider

        Raises:
            ValueError: If no provider specified
        """
        if not self.training.provider:
            raise ValueError("No provider specified in training.provider")
        return self.training.provider


__all__ = [
    "PipelineProviderMixin",
]

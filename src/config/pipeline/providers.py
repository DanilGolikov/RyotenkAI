from __future__ import annotations

from typing import Any


class PipelineProviderMixin:
    """
    Provider-related helper methods for PipelineConfig.

    NOTE: This mixin must be used with a Pydantic model that defines:
    - self.providers: dict[str, dict[str, Any]]
    - self.training.provider: str | None
    """

    providers: dict[str, dict[str, Any]]
    training: Any

    def get_provider_config(self, name: str | None = None) -> dict[str, Any]:
        """
        Get provider config by name.

        Args:
            name: Provider name from registry. None = use training.provider.

        Returns:
            Provider configuration dict

        Raises:
            KeyError: If provider not found
            ValueError: If no provider specified
        """
        # Local import to avoid heavy side-effects at module import time.
        from src.utils.logger import logger

        # Resolve name
        if name is None:
            name = self.training.provider

        if name is None:
            raise ValueError(
                "No provider specified. Set 'training.provider' in config or pass provider name explicitly."
            )

        # Look up in providers registry
        if name not in self.providers:
            available = list(self.providers.keys())
            raise ValueError(
                f"Provider '{name}' not found. Available: {available}. Add it to 'providers:' section in config."
            )

        logger.debug(f"[CFG:PROVIDER] Resolved provider: {name}")
        return self.providers[name]

    def get_provider_training_config(self, name: str | None = None) -> dict[str, Any]:
        """
        Get a training-scoped view of provider config.

        Motivation:
        - New provider schemas (e.g. single_node v3+, runpod vX) group training settings under `training:`.
        - Some runtime components only need training keys like `training_start_timeout`, `mock_mode`, `gpu_type`.

        Behavior:
        - Provider configs must expose a dict `training:` block. We return it.
        - No legacy/flat provider formats are supported.
        """
        provider_cfg = self.get_provider_config(name)
        training_cfg = provider_cfg.get("training")
        if not isinstance(training_cfg, dict):
            raise ValueError(
                f"Provider config must contain 'training:' block (dict). Got: {type(training_cfg).__name__}"
            )
        return training_cfg

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

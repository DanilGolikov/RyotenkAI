"""
ModelClientFactory — creates IModelInference clients based on inference engine name.

Architecture:
    Provider (single_node / runpod) deploys an endpoint with a specific engine API.
    Client communicates with that engine API. Client depends on ENGINE, not PROVIDER.

    ModelEvaluator → ModelClientFactory.create(engine, url, model) → IModelInference

Adding a new engine:
    1. Create a new client class implementing IModelInference.generate(prompt) -> str
    2. Add an entry to _ENGINE_BUILDERS dict in this module.
    3. Done — OCP satisfied, no other files need changing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from src.constants import INFERENCE_ENGINE_VLLM
from src.utils.logger import logger

if TYPE_CHECKING:
    from .interfaces import IModelInference


def _build_openai_compatible(base_url: str, model: str, **kwargs: Any) -> IModelInference:
    from .openai_client import OpenAICompatibleInferenceClient

    return OpenAICompatibleInferenceClient(base_url=base_url, model=model, **kwargs)


ClientBuilder = Callable[..., "IModelInference"]

_ENGINE_BUILDERS: dict[str, ClientBuilder] = {
    INFERENCE_ENGINE_VLLM: _build_openai_compatible,
}


class ModelClientFactory:
    """
    Factory that resolves an inference engine name to the appropriate IModelInference client.

    Follows Open/Closed Principle:
        - Closed for modification: ModelEvaluator never changes when a new engine is added.
        - Open for extension: add entry to _ENGINE_BUILDERS + client class.

    Thread-safety note: _ENGINE_BUILDERS is populated at import time and read-only at runtime.
    """

    @staticmethod
    def create(
        engine: str,
        base_url: str,
        model: str,
        **kwargs: Any,
    ) -> IModelInference:
        """
        Build an IModelInference client for the given engine.

        Args:
            engine:   Engine name from config (e.g. "vllm", "tgi", "ollama").
            base_url: Base URL of the deployed inference endpoint.
            model:    Model identifier for API requests.
            **kwargs: Extra engine-specific params (timeout, temperature, etc.).

        Returns:
            IModelInference implementation ready to call .generate().

        Raises:
            ValueError: if the engine has no registered builder.
        """
        builder = _ENGINE_BUILDERS.get(engine)
        if builder is None:
            supported = ", ".join(sorted(_ENGINE_BUILDERS.keys()))
            raise ValueError(
                f"No model client registered for engine '{engine}'. "
                f"Supported engines: [{supported}]. "
                f"Register a new builder in {__name__}._ENGINE_BUILDERS."
            )

        logger.debug(f"[EVAL] Creating model client for engine '{engine}' → {base_url}")
        return builder(base_url, model, **kwargs)

    @staticmethod
    def supported_engines() -> list[str]:
        """Return a sorted list of engine names with registered client builders."""
        return sorted(_ENGINE_BUILDERS.keys())

    @staticmethod
    def is_supported(engine: str) -> bool:
        """Check if an engine has a registered client builder."""
        return engine in _ENGINE_BUILDERS


__all__ = ["ModelClientFactory"]

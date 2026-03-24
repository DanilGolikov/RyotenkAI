"""
Inference provider factory.

Unlike training providers (GPUProviderFactory), inference providers are scoped to serving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.constants import (
    PROVIDER_RUNPOD,
    PROVIDER_SINGLE_NODE,
    SUPPORTED_INFERENCE_PROVIDERS,
)
from src.utils.result import Failure, ProviderError, Success, err

if TYPE_CHECKING:
    from src.providers.inference.interfaces import IInferenceProvider
    from src.utils.config import PipelineConfig, Secrets


class InferenceProviderFactory:
    """Create inference providers based on `config.inference`."""

    @staticmethod
    def create(*, config: PipelineConfig, secrets: Secrets) -> Success[IInferenceProvider] | Failure[ProviderError]:
        """
        Create inference provider instance.

        Returns:
            Ok(IInferenceProvider): Provider instance
            Err(ProviderError): If provider type is unsupported
        """
        provider: str = config.inference.provider

        if provider == PROVIDER_SINGLE_NODE:
            from src.providers.single_node.inference.provider import SingleNodeInferenceProvider

            return Success(SingleNodeInferenceProvider(config=config, secrets=secrets))

        if provider == PROVIDER_RUNPOD:
            from src.providers.runpod.inference.pods.provider import RunPodPodInferenceProvider

            return Success(RunPodPodInferenceProvider(config=config, secrets=secrets))

        return err(
            ProviderError(
                message=(
                    f"Unsupported inference provider: '{provider}'. "
                    f"Supported: {', '.join(repr(p) for p in SUPPORTED_INFERENCE_PROVIDERS)}."
                ),
                code="INFERENCE_PROVIDER_UNSUPPORTED",
                details={"provider": provider, "supported": list(SUPPORTED_INFERENCE_PROVIDERS)},
            )
        )

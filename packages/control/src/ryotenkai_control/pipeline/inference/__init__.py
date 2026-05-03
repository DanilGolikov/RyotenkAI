"""Inference deployment domain (Provider × Engine).

After Phase A.3 (monorepo packagization, plan §A.3) this package only
re-exports from :mod:`src.providers.inference` — engine helpers (vLLM,
…) and provider factory both live under ``src.providers.inference``.
The shim is removed at the start of Phase B as part of the codemod.
"""

from ryotenkai_providers.inference.factory import InferenceProviderFactory
from ryotenkai_providers.inference.interfaces import EndpointInfo, IInferenceProvider, InferenceCapabilities
from ryotenkai_providers.inference.vllm.engine import VLLMEngine

__all__ = [
    "EndpointInfo",
    "IInferenceProvider",
    "InferenceCapabilities",
    "InferenceProviderFactory",
    "VLLMEngine",
]

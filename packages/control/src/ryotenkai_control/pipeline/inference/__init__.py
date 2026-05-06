"""Inference deployment domain (Provider × Engine).

Re-exports from :mod:`ryotenkai_providers.inference` — engine runtimes live
in :mod:`ryotenkai_engines` (engine_id-keyed registry). The legacy
``VLLMEngine`` class was deleted in PR-14; consumers go through
``IInferenceEngine`` / ``LaunchSpec``.
"""

from ryotenkai_providers.inference.factory import InferenceProviderFactory
from ryotenkai_providers.inference.interfaces import EndpointInfo, IInferenceProvider, InferenceCapabilities

__all__ = [
    "EndpointInfo",
    "IInferenceProvider",
    "InferenceCapabilities",
    "InferenceProviderFactory",
]

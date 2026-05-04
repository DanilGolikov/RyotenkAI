"""Inference-side provider Protocols.

The :class:`InferenceProviderFactory` was removed in the manifest-driven
registry migration (PR-1.11). Use
:meth:`ryotenkai_providers.registry.ProviderRegistry.create_inference`
instead.
"""

from .interfaces import EndpointInfo, IInferenceProvider, InferenceCapabilities

__all__ = [
    "EndpointInfo",
    "IInferenceProvider",
    "InferenceCapabilities",
]

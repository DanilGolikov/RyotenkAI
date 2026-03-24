"""
Inference deployment domain (Provider × Engine).

This package is intentionally separate from training GPU providers:
- training providers manage training infrastructure lifecycle
- inference providers manage serving/inference lifecycle

Source of truth for all providers: `src/providers/`
- training: `src/providers/<provider>/training`
- inference: `src/providers/<provider>/inference`

This package contains:
- `pipeline.inference.engines`: how we serve (runtime: vLLM/TGI/Ollama/...)
"""

from src.providers.inference.factory import InferenceProviderFactory
from src.providers.inference.interfaces import EndpointInfo, IInferenceProvider, InferenceCapabilities

__all__ = [
    "EndpointInfo",
    "IInferenceProvider",
    "InferenceCapabilities",
    "InferenceProviderFactory",
]

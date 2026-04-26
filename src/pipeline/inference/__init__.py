"""
Inference deployment domain (Provider × Engine).

This package is intentionally separate from training GPU providers:
- training providers manage training infrastructure lifecycle
- inference providers manage serving/inference lifecycle

Source of truth for all providers: `src/providers/`
- training: `src/providers/<provider>/training`
- inference: `src/providers/<provider>/inference`

Engine modules (vllm.py, ...) carry runtime-specific start/stop commands and
health-check strategies. Currently only vLLM exists; sibling engines (sglang,
lmdeploy, TGI) can land here as flat modules without re-introducing a sub-package.
"""

from src.pipeline.inference.vllm import VLLMEngine
from src.providers.inference.factory import InferenceProviderFactory
from src.providers.inference.interfaces import EndpointInfo, IInferenceProvider, InferenceCapabilities

__all__ = [
    "EndpointInfo",
    "IInferenceProvider",
    "InferenceCapabilities",
    "InferenceProviderFactory",
    "VLLMEngine",
]

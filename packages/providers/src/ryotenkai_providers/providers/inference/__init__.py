"""
Inference providers (infrastructure layer).

Provider responsibilities:
- connect to target infrastructure (SSH / cloud API)
- prepare runtime workspace
- start/stop inference runtime (delegating engine-specific commands to `engines/*`)
- implement health checks and best-effort cleanup
"""

from .factory import InferenceProviderFactory
from .interfaces import EndpointInfo, IInferenceProvider, InferenceCapabilities

__all__ = [
    "EndpointInfo",
    "IInferenceProvider",
    "InferenceCapabilities",
    "InferenceProviderFactory",
]

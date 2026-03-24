"""
Inference engines (runtime layer).

Engine responsibilities:
- provide engine-specific start/stop commands
- define health check strategy/endpoints (OpenAI-compatible, custom, etc.)
"""

from .vllm import VLLMEngine

__all__ = [
    "VLLMEngine",
]

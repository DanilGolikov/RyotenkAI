"""vLLM engine — first concrete engine in the plugin system.

Public API:
  * :class:`VLLMEngineConfig` — Pydantic config (kind="vllm").
  * :class:`VLLMEngineRuntime` — IInferenceEngine implementation.

Loaded by ``EngineRegistry`` via ``engine.toml`` in this folder.
Generic code (providers/control/shared) MUST NOT import these classes
directly — go through the registry. The importlinter contract
``generic code must not import concrete engine modules`` enforces.
"""

from __future__ import annotations

from ryotenkai_engines.vllm.config import VLLMEngineConfig
from ryotenkai_engines.vllm.runtime import VLLMEngineRuntime

__all__ = (
    "VLLMEngineConfig",
    "VLLMEngineRuntime",
)

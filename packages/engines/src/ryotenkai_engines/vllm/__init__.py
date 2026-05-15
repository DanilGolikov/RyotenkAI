"""vLLM engine — first concrete engine in the plugin system.

Public API (lazy):
  * :class:`VLLMEngineConfig` — Pydantic config (kind="vllm").
  * :class:`VLLMEngineRuntime` — IInferenceEngine implementation.

Both are imported lazily on attribute access to avoid the
``shared → engines → shared`` circular at module init: ``runtime.py``
transitively pulls ``ryotenkai_shared.config`` — and during a fresh
import of ``ryotenkai_shared.config.inference``, the loader is mid-init
and the import would fail.

Direct submodule imports work as expected:
    from ryotenkai_engines.vllm.config import VLLMEngineConfig
    from ryotenkai_engines.vllm.runtime import VLLMEngineRuntime
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ryotenkai_engines.vllm.config import VLLMEngineConfig
    from ryotenkai_engines.vllm.runtime import VLLMEngineRuntime


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access."""
    if name == "VLLMEngineConfig":
        from ryotenkai_engines.vllm.config import VLLMEngineConfig as _Cls

        return _Cls
    if name == "VLLMEngineRuntime":
        from ryotenkai_engines.vllm.runtime import VLLMEngineRuntime as _Cls

        return _Cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "VLLMEngineConfig",
    "VLLMEngineRuntime",
)

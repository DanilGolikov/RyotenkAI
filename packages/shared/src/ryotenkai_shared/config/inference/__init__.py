from .common import (
    InferenceChatUIConfig,
    InferenceCommonConfig,
    InferenceHealthCheckConfig,
    InferenceLoRAConfig,
)
from .schema import InferenceConfig, _InferenceEnginesProxy
from .single_node import InferenceSingleNodeServeConfig

# ---------------------------------------------------------------------------
# Backward-compat aliases (deleted in PR-9)
# ---------------------------------------------------------------------------


def _get_vllm_engine_config_cls():  # type: ignore[no-untyped-def]
    """Lazy import to avoid the engines→shared circular at module init."""
    # ``ryotenkai_engines.vllm.config`` imports only stdlib + pydantic +
    # ryotenkai_engines.interfaces — no transitive dep back into shared.
    # We import-and-cache on first access.
    global _vllm_engine_config_cls
    if _vllm_engine_config_cls is None:
        # Import the leaf module directly (NOT via ryotenkai_engines.vllm
        # which has __init__ that drags runtime.py and its utils.result import).
        import importlib

        mod = importlib.import_module("ryotenkai_engines.vllm.config")
        _vllm_engine_config_cls = mod.VLLMEngineConfig
    return _vllm_engine_config_cls


_vllm_engine_config_cls = None  # type: ignore[assignment]


class _LazyAliasMeta(type):
    """Metaclass that resolves the class on first attribute access."""
    def __getattr__(cls, name):  # type: ignore[no-untyped-def]
        return getattr(_get_vllm_engine_config_cls(), name)

    def __call__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        return _get_vllm_engine_config_cls()(*args, **kwargs)

    def __instancecheck__(cls, instance):  # type: ignore[no-untyped-def]
        return isinstance(instance, _get_vllm_engine_config_cls())


class InferenceVLLMEngineConfig(metaclass=_LazyAliasMeta):
    """Backward-compat alias for ``ryotenkai_engines.vllm.config.VLLMEngineConfig``.

    New code imports ``VLLMEngineConfig`` from ``ryotenkai_engines`` directly.
    Lazy resolution avoids the ``shared → engines → shared`` circular at
    module init time.
    """


#: Pre-discriminated-union shape — the registry of engine-specific configs.
#: New code uses ``cfg.inference.engine`` directly (typed via the
#: discriminated union). Kept as a duck-typed proxy so legacy code that
#: did ``cfg.inference.engines.vllm`` keeps working.
InferenceEnginesConfig = _InferenceEnginesProxy

__all__ = [
    "InferenceChatUIConfig",
    "InferenceCommonConfig",
    "InferenceConfig",
    "InferenceEnginesConfig",  # backward-compat alias
    "InferenceHealthCheckConfig",
    "InferenceLoRAConfig",
    "InferenceSingleNodeServeConfig",
    "InferenceVLLMEngineConfig",  # backward-compat alias
]

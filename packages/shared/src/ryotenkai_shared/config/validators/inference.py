from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.constants import SUPPORTED_INFERENCE_PROVIDERS

if TYPE_CHECKING:
    from ..inference.schema import InferenceConfig


def validate_inference_enabled_is_supported(cfg: InferenceConfig) -> None:
    """Fail-fast guard: inference stage is feature-flagged, but not all
    combinations are implemented in the runtime code yet.

    Post-discriminated-unions: the engine kind comes from
    ``cfg.engine.kind`` (the discriminator). The engine registry is the
    single source of truth for which kinds are supported (replaces the
    legacy ``SUPPORTED_INFERENCE_ENGINES`` constant).
    """
    if not cfg.enabled:
        return

    # Provider check (unchanged).
    if cfg.provider not in SUPPORTED_INFERENCE_PROVIDERS:
        raise ValueError(
            f"inference.enabled=true but inference.provider='{cfg.provider}' is not supported yet. "
            f"Supported: {', '.join(repr(p) for p in SUPPORTED_INFERENCE_PROVIDERS)}."
        )

    # Engine kind check — engine.kind must be in the registry.
    from ryotenkai_engines import get_registry

    registry = get_registry()
    engine_kind = cfg.engine.kind
    if engine_kind not in registry.list():
        raise ValueError(
            f"inference.enabled=true but inference.engine.kind={engine_kind!r} "
            f"is not registered in EngineRegistry. "
            f"Known engines: {sorted(registry.list())}."
        )

    # Belt-and-braces: registered engine MUST resolve to a runtime class
    # without LoadFailure (catches manifest/runtime drift at config load).
    failures = {f.engine_id: f.reason for f in registry.failures()}
    if engine_kind in failures:
        raise ValueError(
            f"inference.engine.kind={engine_kind!r} has a manifest/runtime load "
            f"failure: {failures[engine_kind]}"
        )


__all__ = [
    "validate_inference_enabled_is_supported",
]

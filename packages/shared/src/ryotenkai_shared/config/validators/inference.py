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

    # Provider × engine compatibility — cross-check against the provider's
    # ``[capabilities.supported_engines]`` whitelist. Empty whitelist
    # (default) is a no-op (provider accepts any engine the registry
    # knows about). Populated, the whitelist is enforced.
    #
    # Dynamic import to keep the ``shared has no internal deps`` importlinter
    # contract green; the same pattern is used in cross-validators that
    # cross-check against the providers registry.
    import importlib

    try:
        registry_mod = importlib.import_module("ryotenkai_providers.registry")
    except ImportError:
        return  # modular runtimes without ryotenkai_providers — best-effort.

    try:
        provider_registry = registry_mod.ProviderRegistry.from_filesystem()
        provider_manifest = provider_registry.get_manifest(cfg.provider)
    except Exception:  # noqa: BLE001 — defensive; never block config load on registry hiccups
        return

    supported = getattr(provider_manifest.capabilities, "supported_engines", ())
    if supported and engine_kind not in supported:
        raise ValueError(
            f"provider {cfg.provider!r} does not support engine "
            f"{engine_kind!r}. Supported by this provider: "
            f"{list(supported)}."
        )


__all__ = [
    "validate_inference_enabled_is_supported",
]

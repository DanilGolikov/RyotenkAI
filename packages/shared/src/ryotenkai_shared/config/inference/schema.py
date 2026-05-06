from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from ryotenkai_shared.constants import InferenceProviderName

from ..base import StrictBaseModel
from .common import InferenceCommonConfig


def _engine_config_union() -> Any:
    """Resolve the engine config union from the registry.

    Lazy import — keeps ``ryotenkai_shared.config.inference.schema``
    importable even when ``ryotenkai_engines`` is not installed (rare,
    but the leaf-package contract allows the dependency to be optional
    in some deployment scenarios).
    """
    from ryotenkai_engines import get_engine_config_union

    return get_engine_config_union()


class InferenceConfig(StrictBaseModel):
    """Inference deployment configuration (Provider × Engine).

    Architecture (post-discriminated-unions refactor):
      * ``engine: <EngineConfigUnion>`` — Tag-based discriminated union
        over all engines registered in :class:`ryotenkai_engines.EngineRegistry`.
        Today (single engine): the raw ``VLLMEngineConfig`` class is
        returned by the union builder; ``kind: Literal["vllm"]`` on the
        class still enforces type-safety. When a 2nd engine is added,
        Pydantic auto-switches to a Tag/Discriminator wrapper.

    YAML before::

        inference:
          enabled: true
          provider: single_node
          engine: vllm
          engines:
            vllm:
              max_model_len: 8192

    YAML after::

        inference:
          enabled: true
          provider: single_node
          engine:
            kind: vllm
            max_model_len: 8192
    """

    enabled: bool = False
    provider: InferenceProviderName | None = Field(
        default=None,
        description=(
            "Provider registered in Settings. Leave empty when inference "
            "is disabled — required when enabled."
        ),
    )
    engine: _engine_config_union() = Field(  # type: ignore[valid-type]
        default_factory=lambda: _engine_default_factory()(),
        description=(
            "Engine config — discriminated by ``kind``. The default factory "
            "constructs the first engine the registry knows about (vLLM today)."
        ),
    )
    common: InferenceCommonConfig = Field(default_factory=InferenceCommonConfig)  # pyright: ignore[reportArgumentType]

    # =========================================================================
    # BACKWARD-COMPAT: read-only ``engines`` proxy
    # =========================================================================
    @property
    def engines(self) -> _InferenceEnginesProxy:
        """Read-only proxy that exposes ``engines.vllm`` for legacy callers.

        Pre-discriminated-union code (single_node / runpod inference
        providers, inference_deployer) reads ``cfg.inference.engines.vllm``.
        Property keeps those working until PR-7 migrates them to DI via
        the engine registry and PR-9 deletes the property.

        Deprecated — new code should use ``cfg.inference.engine`` directly.
        """
        return _InferenceEnginesProxy(self.engine)

    @model_validator(mode="after")
    def _run_model_validators(self) -> InferenceConfig:
        """Centralized cross-field validators."""
        from ..validators.inference import validate_inference_enabled_is_supported

        if self.enabled and not (self.provider and self.provider.strip()):
            raise ValueError(
                "inference.provider is required when inference.enabled=true "
                "(pick a provider in Settings or set inference.provider in YAML)."
            )
        validate_inference_enabled_is_supported(self)
        return self


class _InferenceEnginesProxy:
    """Read-only proxy exposing ``.vllm`` (and future ``.<id>``) for the
    pre-discriminated-union ``cfg.inference.engines.vllm`` access pattern.

    Two construction shapes:
      * ``_InferenceEnginesProxy(engine_cfg)`` — used internally by the
        ``InferenceConfig.engines`` property.
      * ``_InferenceEnginesProxy(vllm=…)`` — backward-compat for legacy
        callers that still construct via ``InferenceEnginesConfig(vllm=…)``
        (test fixtures, primarily). Resulting proxy exposes ``.vllm``.

    Deleted in PR-9 alongside its property.
    """

    __slots__ = ("_engine_cfg",)

    def __init__(self, engine_cfg: Any = None, **kwargs: Any) -> None:
        # Legacy shape: InferenceEnginesConfig(vllm=...)  /  (sglang=...).
        # Walk kwargs for the engine kind name; first non-None wins.
        if engine_cfg is None:
            for value in kwargs.values():
                if value is not None:
                    engine_cfg = value
                    break
        self._engine_cfg = engine_cfg

    def __getattr__(self, name: str) -> Any:
        # Avoid recursion on _engine_cfg lookup itself.
        if name == "_engine_cfg":
            raise AttributeError(name)
        engine_cfg = self._engine_cfg
        if engine_cfg is None:
            raise AttributeError(
                f"engines.{name} accessed but no engine config bound. "
                f"Use cfg.inference.engine directly."
            )
        # Match by kind discriminator OR fall back to "vllm" historically.
        kind = getattr(engine_cfg, "kind", None)
        if kind == name:
            return engine_cfg
        # Tolerant fallback for legacy code that always says ``.vllm``
        # regardless of the actual engine — return the bound config.
        if name == kind:
            return engine_cfg
        # Strict: explicit attribute name doesn't match. Helpful error.
        raise AttributeError(
            f"engines.{name} accessed but cfg.engine.kind={kind!r}. "
            f"Use cfg.inference.engine directly."
        )


def _engine_default_factory() -> Any:
    """Resolve the default engine config class from the registry.

    Picks the alphabetically-first engine id (sorted by ``EngineRegistry.list()``).
    Today: ``vllm`` is the only entry. When more engines arrive, operators
    can override per-pipeline via YAML.
    """
    from ryotenkai_engines import get_registry

    registry = get_registry()
    ids = registry.list()
    if not ids:
        # Edge case — no engines registered. Return a no-op factory; the
        # validator below catches it when ``enabled=true``.
        return lambda: None
    return registry.get_config_class(ids[0])


__all__ = [
    "InferenceConfig",
]

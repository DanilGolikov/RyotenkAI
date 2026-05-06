"""Contract: every shipped engine satisfies ``IInferenceEngine`` AND its
runtime ``get_capabilities()`` exactly matches the manifest's
``[capabilities]`` block.

This is the runtime mirror of ``scripts/check_engine_manifests.py`` —
fires on the same drift, but in pytest where failures get nicer
attribution and run on every PR by default.
"""

from __future__ import annotations

import pytest

from ryotenkai_engines.interfaces import BaseEngineConfig, IInferenceEngine
from ryotenkai_engines.registry import EngineRegistry

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def registry() -> EngineRegistry:
    """Use the default in-tree registry (the shipped engines)."""
    return EngineRegistry.from_filesystem()


def test_at_least_one_engine_registered(registry: EngineRegistry) -> None:
    assert registry.list(), (
        "No engines discovered — check packages/engines/src/ryotenkai_engines/*/engine.toml"
    )


def test_no_load_failures(registry: EngineRegistry) -> None:
    failures = registry.failures()
    assert not failures, (
        "Engines with load failures:\n"
        + "\n".join(f"  - {f.engine_id}: {f.exc_type}: {f.reason}" for f in failures)
    )


@pytest.mark.parametrize("engine_id", EngineRegistry.from_filesystem().list())
class TestEngineParity:
    """Run for every shipped engine."""

    def test_runtime_implements_protocol(
        self, registry: EngineRegistry, engine_id: str
    ) -> None:
        runtime_cls = registry.get_runtime(engine_id)
        assert isinstance(runtime_cls(), IInferenceEngine), (
            f"{engine_id}: {runtime_cls.__name__} does not satisfy IInferenceEngine"
        )

    def test_runtime_engine_id_classvar(
        self, registry: EngineRegistry, engine_id: str
    ) -> None:
        runtime_cls = registry.get_runtime(engine_id)
        assert runtime_cls.engine_id == engine_id

    def test_config_class_subclasses_base(
        self, registry: EngineRegistry, engine_id: str
    ) -> None:
        cfg_cls = registry.get_config_class(engine_id)
        assert issubclass(cfg_cls, BaseEngineConfig)

    def test_config_class_kind_literal_matches(
        self, registry: EngineRegistry, engine_id: str
    ) -> None:
        cfg_cls = registry.get_config_class(engine_id)
        kind_field = cfg_cls.model_fields.get("kind")
        assert kind_field is not None, f"{cfg_cls.__name__} has no ``kind`` field"
        # Literal["..."] arguments
        args = getattr(kind_field.annotation, "__args__", None)
        assert args is not None and len(args) == 1
        assert args[0] == engine_id, (
            f"{cfg_cls.__name__}.kind = {args[0]!r}, expected {engine_id!r}"
        )

    def test_capabilities_match_manifest(
        self, registry: EngineRegistry, engine_id: str
    ) -> None:
        manifest_caps = registry.get_manifest(engine_id).capabilities
        runtime_caps = registry.get_runtime(engine_id)().get_capabilities()
        assert runtime_caps == manifest_caps, (
            f"capability drift for {engine_id!r}:\n"
            f"  manifest: {manifest_caps.model_dump()}\n"
            f"  runtime:  {runtime_caps.model_dump()}"
        )

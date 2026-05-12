"""Smoke test for the shipped ``packages/engines/.../vllm/engine.toml``.

Confirms the manifest loads via ``EngineRegistry.from_filesystem``,
the runtime + config classes resolve, and they're consistent with
each other.
"""

from __future__ import annotations

import pytest

from ryotenkai_engines.registry import EngineRegistry

pytestmark = pytest.mark.unit


def test_vllm_manifest_in_default_registry() -> None:
    """Default registry walk discovers vLLM."""
    registry = EngineRegistry.from_filesystem()
    assert "vllm" in registry.list()
    assert registry.failures() == ()


def test_vllm_manifest_fields() -> None:
    registry = EngineRegistry.from_filesystem()
    manifest = registry.get_manifest("vllm")
    assert manifest.engine.id == "vllm"
    assert manifest.engine.name == "vLLM"
    assert manifest.engine.version == "1.0.0"
    assert manifest.engine.upstream_version == "0.7.0"
    assert manifest.engine.stability == "stable"
    assert manifest.image is None  # convention default applies


def test_vllm_image_resolves_via_convention() -> None:
    registry = EngineRegistry.from_filesystem()
    image = registry.get_image("vllm", env={})
    assert image == "ryotenkai/inference-vllm:1.0.0"


def test_vllm_runtime_resolves() -> None:
    registry = EngineRegistry.from_filesystem()
    runtime_cls = registry.get_runtime("vllm")
    from ryotenkai_engines.vllm.runtime import VLLMEngineRuntime

    assert runtime_cls is VLLMEngineRuntime


def test_vllm_config_class_resolves() -> None:
    registry = EngineRegistry.from_filesystem()
    cfg_cls = registry.get_config_class("vllm")
    from ryotenkai_engines.vllm.config import VLLMEngineConfig

    assert cfg_cls is VLLMEngineConfig


def test_vllm_capabilities_match_manifest() -> None:
    """Drift gate — runtime.get_capabilities() and manifest [capabilities]
    must produce the same EngineCapabilities object."""
    registry = EngineRegistry.from_filesystem()
    manifest = registry.get_manifest("vllm")
    runtime_caps = registry.get_runtime("vllm")().get_capabilities()
    assert runtime_caps == manifest.capabilities

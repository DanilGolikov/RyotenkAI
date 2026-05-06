"""Tag-based discriminated union builder.

Verifies the union dispatches correctly to the per-engine config class
based on the ``kind`` discriminator. Uses synthetic engine configs +
synthetic registry — does NOT depend on PR-3's vLLM port.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from ryotenkai_engines._config_union import (
    DISCRIMINATOR_FIELD,
    _NoEnginesPlaceholder,
    build_engine_config_union,
)
from ryotenkai_engines.interfaces import BaseEngineConfig
from ryotenkai_engines.registry import EngineRegistry

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helper — synthesize a registry with N fake config classes
# ---------------------------------------------------------------------------


_MANIFEST_TEMPLATE = """
schema_version = 1

[engine]
id = "{eid}"
name = "{eid}"
version = "1.0.0"

[capabilities]
api_dialect              = "openai_compatible"
supports_lora            = true
supports_quantization    = false
supports_streaming       = true
supports_tensor_parallel = true
supported_dtypes         = ["bfloat16"]
default_port             = 8000

[entry_points.runtime]
module = "{runtime_mod}"
class  = "{runtime_cls}"

[entry_points.config_schema]
module = "{config_mod}"
class  = "{config_cls}"
"""


# Module-level fake configs that the synthetic manifests reference.
class FakeAlphaRuntime:
    engine_id = "alpha"
    config_class: type[BaseEngineConfig]

    def get_capabilities(self):  # type: ignore[no-untyped-def]
        return None
    def build_launch_spec(self, **kwargs):  # type: ignore[no-untyped-def]
        return None
    def build_healthcheck_command(self, **kwargs):  # type: ignore[no-untyped-def]
        return ""
    def build_default_endpoint_url(self, **kwargs):  # type: ignore[no-untyped-def]
        return ""
    def validate_config(self, cfg):  # type: ignore[no-untyped-def]
        return None


class AlphaConfig(BaseEngineConfig):
    kind: Literal["alpha"] = "alpha"
    alpha_setting: int = 1


FakeAlphaRuntime.config_class = AlphaConfig


class FakeBetaRuntime:
    engine_id = "beta"
    config_class: type[BaseEngineConfig]

    def get_capabilities(self):  # type: ignore[no-untyped-def]
        return None
    def build_launch_spec(self, **kwargs):  # type: ignore[no-untyped-def]
        return None
    def build_healthcheck_command(self, **kwargs):  # type: ignore[no-untyped-def]
        return ""
    def build_default_endpoint_url(self, **kwargs):  # type: ignore[no-untyped-def]
        return ""
    def validate_config(self, cfg):  # type: ignore[no-untyped-def]
        return None


class BetaConfig(BaseEngineConfig):
    kind: Literal["beta"] = "beta"
    beta_setting: str = "x"


FakeBetaRuntime.config_class = BetaConfig


def _setup_test_module() -> None:
    """Hack — register this module under a stable importable name so the
    manifests' entry_points can resolve our fake classes."""
    import sys

    sys.modules.setdefault("tests", type(sys)("tests"))  # type: ignore[arg-type]
    sys.modules.setdefault("tests.unit", type(sys)("tests.unit"))  # type: ignore[arg-type]
    sys.modules["tests.unit.test_config_union"] = sys.modules[__name__]


def _write_manifest(root: Path, eid: str, runtime_cls: str, config_cls: str) -> None:
    folder = root / eid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "engine.toml").write_text(
        _MANIFEST_TEMPLATE.format(
            eid=eid,
            runtime_mod="tests.unit.test_config_union",
            runtime_cls=runtime_cls,
            config_mod="tests.unit.test_config_union",
            config_cls=config_cls,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Single-member union (today's reality: only vLLM exists)
# ---------------------------------------------------------------------------


class TestSingleMember:
    """Pydantic v2 quirk: single-member union can't use Discriminator. We
    return the raw config class instead. ``kind: Literal[…]`` on the class
    still enforces type-safety. Once a 2nd engine arrives, the Tag-based
    union activates (see ``TestMultiMember``)."""

    def test_single_engine_returns_raw_class(self, tmp_path: Path) -> None:
        _setup_test_module()
        _write_manifest(tmp_path, "alpha", "FakeAlphaRuntime", "AlphaConfig")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        union = build_engine_config_union(registry)

        # In the degenerate single-engine case we return the class itself.
        assert union is AlphaConfig

    def test_single_engine_class_validates_correctly(self, tmp_path: Path) -> None:
        _setup_test_module()
        _write_manifest(tmp_path, "alpha", "FakeAlphaRuntime", "AlphaConfig")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        union = build_engine_config_union(registry)

        cfg = union.model_validate({"kind": "alpha", "alpha_setting": 42})
        assert isinstance(cfg, AlphaConfig)
        assert cfg.alpha_setting == 42

    def test_single_engine_rejects_wrong_kind_via_literal(
        self, tmp_path: Path
    ) -> None:
        _setup_test_module()
        _write_manifest(tmp_path, "alpha", "FakeAlphaRuntime", "AlphaConfig")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        union = build_engine_config_union(registry)

        # Wrong kind — Pydantic's ``Literal["alpha"]`` rejects "beta".
        with pytest.raises(ValidationError):
            union.model_validate({"kind": "beta"})

    def test_single_engine_kind_can_default_to_literal(
        self, tmp_path: Path
    ) -> None:
        """The ``kind`` field has a default value matching the Literal —
        omitting it is allowed (Pydantic fills it)."""
        _setup_test_module()
        _write_manifest(tmp_path, "alpha", "FakeAlphaRuntime", "AlphaConfig")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        union = build_engine_config_union(registry)

        cfg = union.model_validate({"alpha_setting": 1})
        assert cfg.kind == "alpha"


# ---------------------------------------------------------------------------
# Multi-member union (when 2+ engines exist)
# ---------------------------------------------------------------------------


class TestMultiMember:
    def test_two_engines_union_dispatches_by_kind(self, tmp_path: Path) -> None:
        _setup_test_module()
        _write_manifest(tmp_path, "alpha", "FakeAlphaRuntime", "AlphaConfig")
        _write_manifest(tmp_path, "beta", "FakeBetaRuntime", "BetaConfig")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        union = build_engine_config_union(registry)

        adapter = TypeAdapter(union)
        a = adapter.validate_python({"kind": "alpha", "alpha_setting": 7})
        b = adapter.validate_python({"kind": "beta", "beta_setting": "hi"})
        assert isinstance(a, AlphaConfig)
        assert isinstance(b, BetaConfig)
        assert a.alpha_setting == 7
        assert b.beta_setting == "hi"

    def test_extra_field_rejected(self, tmp_path: Path) -> None:
        """Crossfield: kind=alpha but a beta-only field present → extra='forbid' rejects."""
        _setup_test_module()
        _write_manifest(tmp_path, "alpha", "FakeAlphaRuntime", "AlphaConfig")
        _write_manifest(tmp_path, "beta", "FakeBetaRuntime", "BetaConfig")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        union = build_engine_config_union(registry)

        adapter = TypeAdapter(union)
        with pytest.raises(ValidationError, match="extra|Extra|unexpected"):
            adapter.validate_python({"kind": "alpha", "beta_setting": "x"})


# ---------------------------------------------------------------------------
# Empty registry — the placeholder branch
# ---------------------------------------------------------------------------


class TestEmptyRegistry:
    def test_empty_registry_returns_placeholder(self, tmp_path: Path) -> None:
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        assert registry.list() == ()
        result = build_engine_config_union(registry)
        assert result is _NoEnginesPlaceholder


# ---------------------------------------------------------------------------
# Discriminator field name — public contract
# ---------------------------------------------------------------------------


class TestDiscriminatorContract:
    def test_discriminator_field_is_kind(self) -> None:
        assert DISCRIMINATOR_FIELD == "kind"


# ---------------------------------------------------------------------------
# Wrapping the union in a parent BaseModel — the actual usage pattern
# ---------------------------------------------------------------------------


class TestWrappedInBaseModel:
    def test_union_works_inside_basemodel_field(self, tmp_path: Path) -> None:
        """Mirrors how InferenceConfig.engine will use the union (PR-6)."""
        _setup_test_module()
        _write_manifest(tmp_path, "alpha", "FakeAlphaRuntime", "AlphaConfig")
        _write_manifest(tmp_path, "beta", "FakeBetaRuntime", "BetaConfig")
        registry = EngineRegistry.from_filesystem(roots=[tmp_path])
        union = build_engine_config_union(registry)

        class Wrapper(BaseModel):
            engine: union  # type: ignore[valid-type]

        w = Wrapper.model_validate({"engine": {"kind": "beta", "beta_setting": "z"}})
        assert isinstance(w.engine, BetaConfig)
        assert w.engine.beta_setting == "z"

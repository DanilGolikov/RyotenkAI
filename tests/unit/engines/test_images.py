"""``resolve_image`` — convention default + override chain.

Categories: positive, negative, boundary, invariant, logic-specific
(resolution-order), combinatorial.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import pytest

from ryotenkai_engines.images import (
    DEFAULT_IMAGE_REGISTRY,
    ENV_IMAGE_OVERRIDE_PATTERN,
    ENV_IMAGE_REGISTRY,
    _convention_image_name,
    resolve_image,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers — synthetic manifests with the minimum surface resolve_image needs.
# ---------------------------------------------------------------------------


class _FakeEngineSpec(NamedTuple):
    id: str
    version: str


class _FakeImageSpec(NamedTuple):
    default: str


def _fake_manifest(
    *,
    engine_id: str,
    version: str,
    explicit_image: str | None = None,
) -> SimpleNamespace:
    """Build a duck-typed manifest object with the fields ``resolve_image``
    actually reads."""
    return SimpleNamespace(
        engine=_FakeEngineSpec(id=engine_id, version=version),
        image=_FakeImageSpec(default=explicit_image) if explicit_image else None,
    )


# ---------------------------------------------------------------------------
# Convention default (case 4 of the chain)
# ---------------------------------------------------------------------------


class TestConventionDefault:
    def test_basic_convention_default(self) -> None:
        out = _convention_image_name(
            engine_id="vllm",
            engine_version="1.0.0",
            env={},
        )
        assert out == "ryotenkai/inference-vllm:1.0.0"

    def test_env_overrides_registry_prefix(self) -> None:
        out = _convention_image_name(
            engine_id="vllm",
            engine_version="1.0.0",
            env={ENV_IMAGE_REGISTRY: "ghcr.io/myorg"},
        )
        assert out == "ghcr.io/myorg/inference-vllm:1.0.0"

    def test_resolve_falls_through_to_convention_when_nothing_else(self) -> None:
        manifest = _fake_manifest(engine_id="vllm", version="1.0.0")
        out = resolve_image(engine_id="vllm", manifest=manifest, env={})
        assert out == "ryotenkai/inference-vllm:1.0.0"


# ---------------------------------------------------------------------------
# Override chain — each level wins over the next
# ---------------------------------------------------------------------------


class TestOverrideChain:
    def test_env_override_wins_over_provider(self) -> None:
        manifest = _fake_manifest(
            engine_id="vllm",
            version="1.0.0",
            explicit_image="manifest/explicit:1",
        )
        provider_overrides = {"vllm": _FakeImageSpec(default="provider/override:1")}

        env = {
            ENV_IMAGE_OVERRIDE_PATTERN.format(engine_upper="VLLM"): "env/override:1",
        }
        out = resolve_image(
            engine_id="vllm",
            manifest=manifest,
            provider_overrides=provider_overrides,
            env=env,
        )
        assert out == "env/override:1"

    def test_provider_override_wins_over_manifest(self) -> None:
        manifest = _fake_manifest(
            engine_id="vllm",
            version="1.0.0",
            explicit_image="manifest/explicit:1",
        )
        # Provider override shape: object with .image attribute.
        prov_obj = SimpleNamespace(image="provider/override:1")
        out = resolve_image(
            engine_id="vllm",
            manifest=manifest,
            provider_overrides={"vllm": prov_obj},
            env={},
        )
        assert out == "provider/override:1"

    def test_provider_override_dict_shape(self) -> None:
        """Provider overrides also accept dict shape, not just attr access."""
        manifest = _fake_manifest(engine_id="vllm", version="1.0.0")
        out = resolve_image(
            engine_id="vllm",
            manifest=manifest,
            provider_overrides={"vllm": {"image": "from/dict:1"}},
            env={},
        )
        assert out == "from/dict:1"

    def test_manifest_explicit_wins_over_convention(self) -> None:
        manifest = _fake_manifest(
            engine_id="vllm",
            version="1.0.0",
            explicit_image="manifest/explicit:1",
        )
        out = resolve_image(engine_id="vllm", manifest=manifest, env={})
        assert out == "manifest/explicit:1"

    def test_provider_override_for_other_engine_is_ignored(self) -> None:
        """Override for engine 'sglang' must not affect 'vllm' resolution."""
        manifest = _fake_manifest(engine_id="vllm", version="1.0.0")
        out = resolve_image(
            engine_id="vllm",
            manifest=manifest,
            provider_overrides={"sglang": _FakeImageSpec(default="x")},
            env={},
        )
        assert out == "ryotenkai/inference-vllm:1.0.0"

    def test_empty_provider_override_image_falls_through(self) -> None:
        """Provider declares an override but image is empty/None — fall through."""
        manifest = _fake_manifest(
            engine_id="vllm",
            version="1.0.0",
            explicit_image="manifest/explicit:1",
        )
        prov_obj = SimpleNamespace(image=None)
        out = resolve_image(
            engine_id="vllm",
            manifest=manifest,
            provider_overrides={"vllm": prov_obj},
            env={},
        )
        assert out == "manifest/explicit:1"


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_engine_id_mismatch_raises(self) -> None:
        manifest = _fake_manifest(engine_id="vllm", version="1.0.0")
        with pytest.raises(ValueError, match="does not match"):
            resolve_image(engine_id="sglang", manifest=manifest, env={})


# ---------------------------------------------------------------------------
# Logic-specific — env override case-sensitivity
# ---------------------------------------------------------------------------


class TestEnvCase:
    def test_env_override_uses_uppercase_engine_id(self) -> None:
        manifest = _fake_manifest(engine_id="vllm", version="1.0.0")
        # Lowercase env key — must NOT match.
        out = resolve_image(
            engine_id="vllm",
            manifest=manifest,
            env={"RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_vllm": "lower/case:1"},
        )
        assert out == "ryotenkai/inference-vllm:1.0.0"

        # Uppercase env key — wins.
        out = resolve_image(
            engine_id="vllm",
            manifest=manifest,
            env={"RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_VLLM": "upper/case:1"},
        )
        assert out == "upper/case:1"


# ---------------------------------------------------------------------------
# Constants are stable parts of the public env contract
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_registry_prefix(self) -> None:
        assert DEFAULT_IMAGE_REGISTRY == "ryotenkai"

    def test_env_var_name(self) -> None:
        assert ENV_IMAGE_REGISTRY == "RYOTENKAI_INFERENCE_IMAGE_REGISTRY"

    def test_override_pattern_format(self) -> None:
        formatted = ENV_IMAGE_OVERRIDE_PATTERN.format(engine_upper="VLLM")
        assert formatted == "RYOTENKAI_INFERENCE_IMAGE_OVERRIDE_VLLM"

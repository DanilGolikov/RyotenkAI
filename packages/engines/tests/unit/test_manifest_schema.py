"""``EngineManifest`` schema validation — all 8 categories.

* Positive — well-formed manifests parse correctly.
* Negative — common malformations are rejected with a clear message.
* Boundary — extreme but valid values (port 1, port 65535, empty strings, …).
* Invariant — cross-field rules ([engine].id format, capability-quantization
  parity, dtype/quantization disjointness).
* Dependency-error — manifest references missing modules / classes are out
  of scope here (registry tests).
* Regression — schema_version gate.
* Logic-specific — image floating-tag rejection.
* Combinatorial — every (api_dialect × supports_*) combination round-trips.
"""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from ryotenkai_engines.manifest import (
    LATEST_ENGINE_SCHEMA_VERSION,
    EngineManifest,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_manifest_data(**overrides) -> dict:  # type: ignore[no-untyped-def]
    """A manifest dict that PASSES validation; tests override individual keys."""
    base = {
        "schema_version": 1,
        "engine": {
            "id": "vllm",
            "name": "vLLM",
            "version": "1.0.0",
        },
        "capabilities": {
            "api_dialect": "openai_compatible",
            "supports_lora": True,
            "supports_quantization": True,
            "supports_streaming": True,
            "supports_tensor_parallel": True,
            "supported_quantizations": ["awq", "gptq"],
            "supported_dtypes": ["bfloat16", "float16"],
            "default_port": 8000,
        },
        "entry_points": {
            "runtime": {
                "module": "ryotenkai_engines.vllm.runtime",
                "class": "VLLMEngineRuntime",
            },
            "config_schema": {
                "module": "ryotenkai_engines.vllm.config",
                "class": "VLLMEngineConfig",
            },
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_minimal_manifest_parses(self) -> None:
        manifest = EngineManifest.model_validate(_minimal_manifest_data())
        assert manifest.engine.id == "vllm"
        assert manifest.capabilities.api_dialect == "openai_compatible"
        assert manifest.image is None  # convention default applies

    def test_with_explicit_image(self) -> None:
        data = _minimal_manifest_data(image={"default": "ryotenkai/inference-vllm:1.0.0"})
        manifest = EngineManifest.model_validate(data)
        assert manifest.image is not None
        assert manifest.image.default == "ryotenkai/inference-vllm:1.0.0"

    def test_with_upstream_version(self) -> None:
        data = _minimal_manifest_data()
        data["engine"]["upstream_version"] = "0.7.0"
        manifest = EngineManifest.model_validate(data)
        assert manifest.engine.upstream_version == "0.7.0"


# ---------------------------------------------------------------------------
# 2. Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_missing_engine_block(self) -> None:
        data = _minimal_manifest_data()
        del data["engine"]
        with pytest.raises(ValidationError):
            EngineManifest.model_validate(data)

    def test_missing_capabilities_block(self) -> None:
        data = _minimal_manifest_data()
        del data["capabilities"]
        with pytest.raises(ValidationError):
            EngineManifest.model_validate(data)

    def test_missing_entry_points(self) -> None:
        data = _minimal_manifest_data()
        del data["entry_points"]
        with pytest.raises(ValidationError):
            EngineManifest.model_validate(data)

    def test_unknown_top_level_field(self) -> None:
        data = _minimal_manifest_data(unexpected_field="oops")
        with pytest.raises(ValidationError, match="Extra inputs"):
            EngineManifest.model_validate(data)

    def test_uppercase_engine_id_rejected(self) -> None:
        data = _minimal_manifest_data()
        data["engine"]["id"] = "VLLM"
        with pytest.raises(ValidationError, match="snake_case"):
            EngineManifest.model_validate(data)

    def test_engine_id_with_hyphen_rejected(self) -> None:
        data = _minimal_manifest_data()
        data["engine"]["id"] = "vllm-fork"
        with pytest.raises(ValidationError, match="snake_case"):
            EngineManifest.model_validate(data)

    def test_invalid_version_format_rejected(self) -> None:
        data = _minimal_manifest_data()
        data["engine"]["version"] = "not-a-version"
        with pytest.raises(ValidationError, match="not a valid PEP 440"):
            EngineManifest.model_validate(data)

    def test_runtime_entry_point_missing(self) -> None:
        data = _minimal_manifest_data()
        del data["entry_points"]["runtime"]
        with pytest.raises(ValidationError):
            EngineManifest.model_validate(data)

    def test_config_schema_entry_point_missing(self) -> None:
        data = _minimal_manifest_data()
        del data["entry_points"]["config_schema"]
        with pytest.raises(ValidationError):
            EngineManifest.model_validate(data)


# ---------------------------------------------------------------------------
# 3. Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_default_port_min(self) -> None:
        data = _minimal_manifest_data()
        data["capabilities"]["default_port"] = 1
        manifest = EngineManifest.model_validate(data)
        assert manifest.capabilities.default_port == 1

    def test_default_port_max(self) -> None:
        data = _minimal_manifest_data()
        data["capabilities"]["default_port"] = 65535
        manifest = EngineManifest.model_validate(data)
        assert manifest.capabilities.default_port == 65535

    def test_default_port_zero_rejected(self) -> None:
        data = _minimal_manifest_data()
        data["capabilities"]["default_port"] = 0
        with pytest.raises(ValidationError):
            EngineManifest.model_validate(data)

    def test_default_port_too_high_rejected(self) -> None:
        data = _minimal_manifest_data()
        data["capabilities"]["default_port"] = 65536
        with pytest.raises(ValidationError):
            EngineManifest.model_validate(data)

    def test_engine_id_single_char(self) -> None:
        data = _minimal_manifest_data()
        data["engine"]["id"] = "a"
        # Single-char IDs are technically valid but unusual; rather than
        # add a min-length rule we accept and let the drift detector flag.
        manifest = EngineManifest.model_validate(data)
        assert manifest.engine.id == "a"

    def test_supports_quantization_false_with_empty_list(self) -> None:
        data = _minimal_manifest_data()
        data["capabilities"]["supports_quantization"] = False
        data["capabilities"]["supported_quantizations"] = []
        manifest = EngineManifest.model_validate(data)
        assert manifest.capabilities.supports_quantization is False
        assert manifest.capabilities.supported_quantizations == ()


# ---------------------------------------------------------------------------
# 4. Invariant
# ---------------------------------------------------------------------------


class TestInvariant:
    def test_quantization_parity_violation_rejected(self) -> None:
        """supports_quantization=False with non-empty list ⇒ rejected."""
        data = _minimal_manifest_data()
        data["capabilities"]["supports_quantization"] = False
        data["capabilities"]["supported_quantizations"] = ["awq"]
        with pytest.raises(ValidationError, match="non-empty"):
            EngineManifest.model_validate(data)

    def test_supported_quantizations_duplicates_rejected(self) -> None:
        data = _minimal_manifest_data()
        data["capabilities"]["supported_quantizations"] = ["awq", "awq", "gptq"]
        with pytest.raises(ValidationError, match="duplicates"):
            EngineManifest.model_validate(data)

    def test_supported_dtypes_duplicates_rejected(self) -> None:
        data = _minimal_manifest_data()
        data["capabilities"]["supported_dtypes"] = ["bfloat16", "bfloat16"]
        with pytest.raises(ValidationError, match="duplicates"):
            EngineManifest.model_validate(data)

    def test_supported_dtypes_empty_rejected(self) -> None:
        data = _minimal_manifest_data()
        data["capabilities"]["supported_dtypes"] = []
        with pytest.raises(ValidationError, match="at least one dtype"):
            EngineManifest.model_validate(data)

    def test_dtype_quantization_overlap_rejected(self) -> None:
        """Cross-block invariant: supported_quantizations and supported_dtypes
        must not overlap."""
        data = _minimal_manifest_data()
        data["capabilities"]["supported_quantizations"] = ["fp8"]
        data["capabilities"]["supported_dtypes"] = ["bfloat16", "fp8"]
        with pytest.raises(ValidationError, match="must not overlap"):
            EngineManifest.model_validate(data)


# ---------------------------------------------------------------------------
# 6. Regression — schema_version gate
# ---------------------------------------------------------------------------


class TestSchemaVersionGate:
    def test_too_low_schema_version_rejected(self) -> None:
        data = _minimal_manifest_data(schema_version=0)
        with pytest.raises(ValidationError, match=">= 1"):
            EngineManifest.model_validate(data)

    def test_too_high_schema_version_rejected(self) -> None:
        data = _minimal_manifest_data(
            schema_version=LATEST_ENGINE_SCHEMA_VERSION + 1,
        )
        with pytest.raises(ValidationError, match="newer than this"):
            EngineManifest.model_validate(data)

    def test_current_schema_version_accepted(self) -> None:
        data = _minimal_manifest_data(schema_version=LATEST_ENGINE_SCHEMA_VERSION)
        manifest = EngineManifest.model_validate(data)
        assert manifest.schema_version == LATEST_ENGINE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 7. Logic-specific — image floating-tag rejection
# ---------------------------------------------------------------------------


class TestImageFloatingTags:
    @pytest.mark.parametrize(
        "tag",
        [":latest", ":dev", ":main", ":master", ":nightly", ":edge"],
    )
    def test_floating_tag_rejected(self, tag: str) -> None:
        data = _minimal_manifest_data(image={"default": f"ryotenkai/inference-vllm{tag}"})
        with pytest.raises(ValidationError, match="floating tag"):
            EngineManifest.model_validate(data)

    def test_pinned_semver_tag_accepted(self) -> None:
        data = _minimal_manifest_data(image={"default": "ryotenkai/inference-vllm:1.0.0"})
        manifest = EngineManifest.model_validate(data)
        assert manifest.image is not None
        assert manifest.image.default.endswith(":1.0.0")

    def test_empty_image_default_rejected(self) -> None:
        data = _minimal_manifest_data(image={"default": ""})
        with pytest.raises(ValidationError, match="must not be empty"):
            EngineManifest.model_validate(data)


# ---------------------------------------------------------------------------
# 8. Combinatorial — capability flag matrix
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize(
        "lora,quant,stream,tp",
        list(itertools.product([True, False], repeat=4)),
    )
    def test_capability_flag_matrix(
        self, lora: bool, quant: bool, stream: bool, tp: bool
    ) -> None:
        """Every of the 16 combinations of supports_* flags either parses
        cleanly OR violates the quantization invariant — no other failures."""
        data = _minimal_manifest_data()
        data["capabilities"]["supports_lora"] = lora
        data["capabilities"]["supports_quantization"] = quant
        data["capabilities"]["supports_streaming"] = stream
        data["capabilities"]["supports_tensor_parallel"] = tp
        # Match the invariant: when supports_quantization=False, the list
        # MUST be empty.
        if not quant:
            data["capabilities"]["supported_quantizations"] = []

        manifest = EngineManifest.model_validate(data)
        assert manifest.capabilities.supports_lora is lora
        assert manifest.capabilities.supports_quantization is quant
        assert manifest.capabilities.supports_streaming is stream
        assert manifest.capabilities.supports_tensor_parallel is tp

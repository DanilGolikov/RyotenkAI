from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import InferenceConfig, InferenceEnginesConfig, InferenceVLLMEngineConfig

pytestmark = pytest.mark.unit


def _engines_cfg() -> InferenceEnginesConfig:
    # Phase 6.6: image fields removed (pinned via INFERENCE_IMAGES);
    # vllm engine config is now empty by default — defaults are valid.
    return InferenceEnginesConfig(vllm=InferenceVLLMEngineConfig())


class TestInferenceConfigValidators:
    def test_positive_disabled_inference_allows_any_supported_provider_engine(self) -> None:
        # Boundary: when inference.enabled=false we allow any schema-supported provider/engine pair.
        _ = InferenceConfig(
            enabled=False,
            provider="single_node",
            engine="vllm",
            engines=_engines_cfg(),
        )

    def test_negative_invalid_provider_value(self) -> None:
        with pytest.raises(ValidationError) as e:
            _ = InferenceConfig(
                enabled=True,
                provider="unknown_provider",
                engine="vllm",
                engines=_engines_cfg(),
            )
        assert any((err.get("loc") or ("",))[0] == "provider" for err in e.value.errors())

    def test_negative_invalid_engine_value(self) -> None:
        with pytest.raises(ValidationError) as e:
            _ = InferenceConfig(
                enabled=True,
                provider="single_node",
                engine="unknown_engine",
                engines=_engines_cfg(),
            )
        assert any(tuple(err.get("loc") or ()) == ("engine",) for err in e.value.errors())

    def test_positive_enabled_inference_supported_combo(self) -> None:
        _ = InferenceConfig(
            enabled=True,
            provider="single_node",
            engine="vllm",
            engines=_engines_cfg(),
        )

    def test_combinatorial_invalid_provider_and_engine_reports_both(self) -> None:
        with pytest.raises(ValidationError) as e:
            _ = InferenceConfig(
                enabled=True,
                provider="unknown_provider",
                engine="unknown_engine",
                engines=_engines_cfg(),
            )
        errors = e.value.errors()
        assert any((err.get("loc") or ("",))[0] == "provider" for err in errors)
        assert any((err.get("loc") or ("",))[0] == "engine" for err in errors)

    def test_positive_enabled_runpod_allows_missing_images(self) -> None:
        # Images are not required for runpod Pods (unified image lives under providers.runpod.inference.pod.image_name).
        _ = InferenceConfig(
            enabled=True,
            provider="runpod",
            engine="vllm",
        )

    # Phase 6.6 deletion: image fields are no longer user-facing
    # config (they're pinned in src.inference.__about__.INFERENCE_IMAGES),
    # so the legacy "merge_image / serve_image required for single_node"
    # validator is gone. Test deleted with the validator.


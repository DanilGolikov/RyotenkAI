"""Inference config validators — post discriminated-unions.

After PR-6:
  * ``InferenceConfig.engine`` is a typed Pydantic model (``VLLMEngineConfig``
    today; future engines via the EngineConfigUnion).
  * ``InferenceConfig.engines`` is a backward-compat property exposing
    ``.<kind>`` access for legacy callers.
  * The validator cross-checks ``engine.kind`` against the engine
    registry — engine string was replaced with the discriminator.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from ryotenkai_engines.vllm.config import VLLMEngineConfig

from ryotenkai_shared.config import InferenceConfig

pytestmark = pytest.mark.unit


class TestInferenceConfigValidators:
    def test_positive_disabled_inference_allows_default_engine(self) -> None:
        cfg = InferenceConfig(enabled=False, provider="single_node")
        assert cfg.engine.kind == "vllm"

    def test_negative_invalid_provider_value(self) -> None:
        with pytest.raises(ValidationError) as e:
            _ = InferenceConfig(
                enabled=True,
                provider="unknown_provider",
                engine=VLLMEngineConfig(),
            )
        assert any((err.get("loc") or ("",))[0] == "provider" for err in e.value.errors())

    def test_negative_invalid_engine_kind(self) -> None:
        """Unknown engine kind ⇒ Pydantic discriminator rejects."""
        with pytest.raises(ValidationError):
            _ = InferenceConfig(
                enabled=True,
                provider="single_node",
                engine={"kind": "unknown_engine"},
            )

    def test_positive_enabled_inference_supported_combo(self) -> None:
        cfg = InferenceConfig(
            enabled=True,
            provider="single_node",
            engine=VLLMEngineConfig(),
        )
        assert cfg.engine.kind == "vllm"

    def test_positive_enabled_runpod_allows_missing_images(self) -> None:
        """Images auto-derive via convention; no manifest tweaking needed."""
        cfg = InferenceConfig(enabled=True, provider="runpod")
        assert cfg.engine.kind == "vllm"

    def test_inference_config_explicit_provider_required_when_enabled(self) -> None:
        with pytest.raises(ValidationError, match="inference.provider is required"):  # noqa: RUF043
            _ = InferenceConfig(enabled=True, provider=None)

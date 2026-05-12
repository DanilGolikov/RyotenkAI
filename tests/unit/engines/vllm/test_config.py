"""``VLLMEngineConfig`` — Pydantic validation.

Categories: positive, negative, boundary, invariant, regression
(porting from legacy ``VLLMEngineConfig``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ryotenkai_engines.vllm.config import (
    GPU_MEMORY_UTILIZATION_DEFAULT,
    MAX_MODEL_LEN_DEFAULT,
    MAX_MODEL_LEN_MIN,
    VLLMEngineConfig,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Positive
# ---------------------------------------------------------------------------


class TestPositive:
    def test_defaults_parse(self) -> None:
        cfg = VLLMEngineConfig()
        assert cfg.kind == "vllm"
        assert cfg.merge_before_deploy is True
        assert cfg.tensor_parallel_size == 1
        assert cfg.max_model_len == MAX_MODEL_LEN_DEFAULT
        assert cfg.gpu_memory_utilization == GPU_MEMORY_UTILIZATION_DEFAULT
        assert cfg.quantization is None
        assert cfg.enforce_eager is False

    def test_full_override(self) -> None:
        cfg = VLLMEngineConfig(
            tensor_parallel_size=4,
            max_model_len=8192,
            gpu_memory_utilization=0.85,
            quantization="awq",
            enforce_eager=True,
        )
        assert cfg.tensor_parallel_size == 4
        assert cfg.max_model_len == 8192
        assert cfg.gpu_memory_utilization == 0.85
        assert cfg.quantization == "awq"
        assert cfg.enforce_eager is True

    def test_kind_field_is_literal_vllm(self) -> None:
        cfg = VLLMEngineConfig.model_validate({"kind": "vllm"})
        assert cfg.kind == "vllm"


# ---------------------------------------------------------------------------
# Negative
# ---------------------------------------------------------------------------


class TestNegative:
    def test_wrong_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VLLMEngineConfig.model_validate({"kind": "sglang"})

    def test_unknown_field_rejected(self) -> None:
        """``extra='forbid'`` from BaseEngineConfig propagates."""
        with pytest.raises(ValidationError, match="Extra inputs"):
            VLLMEngineConfig.model_validate({"unexpected_field": True})

    def test_negative_tensor_parallel_size_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VLLMEngineConfig(tensor_parallel_size=0)

    def test_too_short_max_model_len_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VLLMEngineConfig(max_model_len=MAX_MODEL_LEN_MIN - 1)

    def test_zero_gpu_memory_utilization_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VLLMEngineConfig(gpu_memory_utilization=0.0)

    def test_above_one_gpu_memory_utilization_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VLLMEngineConfig(gpu_memory_utilization=1.01)


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_min_tensor_parallel_size(self) -> None:
        cfg = VLLMEngineConfig(tensor_parallel_size=1)
        assert cfg.tensor_parallel_size == 1

    def test_min_max_model_len(self) -> None:
        cfg = VLLMEngineConfig(max_model_len=MAX_MODEL_LEN_MIN)
        assert cfg.max_model_len == MAX_MODEL_LEN_MIN

    def test_min_gpu_memory_utilization(self) -> None:
        cfg = VLLMEngineConfig(gpu_memory_utilization=0.001)
        assert cfg.gpu_memory_utilization == 0.001

    def test_max_gpu_memory_utilization(self) -> None:
        cfg = VLLMEngineConfig(gpu_memory_utilization=1.0)
        assert cfg.gpu_memory_utilization == 1.0


# ---------------------------------------------------------------------------
# Invariant — kind discriminator can default
# ---------------------------------------------------------------------------


class TestInvariant:
    def test_kind_default_omitted(self) -> None:
        """Omitting kind is fine — Literal default fills it."""
        cfg = VLLMEngineConfig.model_validate({})
        assert cfg.kind == "vllm"

    def test_subclass_of_base_engine_config(self) -> None:
        from ryotenkai_engines.interfaces import BaseEngineConfig

        assert issubclass(VLLMEngineConfig, BaseEngineConfig)

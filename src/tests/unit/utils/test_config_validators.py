"""
Validator tests for config schema (strict, no backward compatibility).

Focus:
- Required fields are enforced (fail-fast)
- Field validators accept/reject expected values
- Legacy flat fields are forbidden
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.utils.config import (
    AdaLoraConfig,
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    GlobalHyperparametersConfig,
    LoraConfig,
    ModelConfig,
    PhaseHyperparametersConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)


def _model_cfg(**overrides) -> ModelConfig:
    data = {
        "name": "test-model",
        "torch_dtype": "bfloat16",
        "trust_remote_code": False,
    }
    data.update(overrides)
    return ModelConfig(**data)


def _lora_cfg(**overrides) -> LoraConfig:
    data = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": "all-linear",
        "use_dora": False,
        "use_rslora": False,
        "init_lora_weights": "gaussian",
    }
    data.update(overrides)
    return LoraConfig(**data)


def _hp_global_cfg(**overrides) -> GlobalHyperparametersConfig:
    data = {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.0,
        "epochs": 1,
    }
    data.update(overrides)
    return GlobalHyperparametersConfig(**data)


def _training_cfg(**overrides) -> TrainingOnlyConfig:
    data = {
        "type": "qlora",
        "lora": _lora_cfg(),
        "hyperparams": _hp_global_cfg(),
        "strategies": [StrategyPhaseConfig(strategy_type="sft")],
    }
    data.update(overrides)
    return TrainingOnlyConfig(**data)


class TestModelConfig:
    @pytest.mark.parametrize("dtype", ["auto", "bfloat16", "float16", "float32"])
    def test_torch_dtype_valid(self, dtype: str) -> None:
        cfg = _model_cfg(torch_dtype=dtype)
        assert cfg.torch_dtype == dtype

    @pytest.mark.parametrize("dtype", ["int8", "fp8", "", "invalid"])
    def test_torch_dtype_invalid(self, dtype: str) -> None:
        with pytest.raises(ValidationError, match="torch_dtype must be one of"):
            _ = _model_cfg(torch_dtype=dtype)

    def test_required_fields_enforced(self) -> None:
        with pytest.raises(ValidationError, match="Field required"):
            _ = ModelConfig(name="x")  # missing torch_dtype + trust_remote_code


class TestLoraConfig:
    def test_bias_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError, match="bias must be one of"):
            _ = _lora_cfg(bias="invalid")

    def test_target_modules_none_becomes_default(self) -> None:
        cfg = _lora_cfg(target_modules=None)
        assert cfg.target_modules == "all-linear"

    def test_init_lora_weights_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError, match="init_lora_weights must be one of"):
            _ = _lora_cfg(init_lora_weights="nope")

    def test_dora_incompatible_with_loftq(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            _ = _lora_cfg(use_dora=True, init_lora_weights="loftq")


class TestStrategyPhaseConfig:
    def test_legacy_flat_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _ = StrategyPhaseConfig(strategy_type="sft", epochs=5)  # type: ignore[call-arg]

    def test_phase_hyperparams_block_allowed(self) -> None:
        phase = StrategyPhaseConfig(
            strategy_type="sft",
            hyperparams=PhaseHyperparametersConfig(epochs=3, learning_rate=2e-4),
        )
        assert phase.hyperparams.epochs == 3


class TestTrainingOnlyConfig:
    @pytest.mark.parametrize("training_type", ["qlora", "lora", "adalora"])
    def test_type_valid(self, training_type: str) -> None:
        if training_type == "adalora":
            cfg = _training_cfg(
                type="adalora",
                adalora=AdaLoraConfig(
                    init_r=16,
                    target_r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    bias="none",
                    target_modules="all-linear",
                ),
            )
            assert cfg.type == "adalora"
            assert cfg.adalora is not None
            return

        cfg = _training_cfg(type=training_type)
        assert cfg.type == training_type

    def test_type_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Training type must be one of"):
            _ = _training_cfg(type="full_ft")

    def test_adalora_requires_block(self) -> None:
        with pytest.raises(ValidationError, match="requires 'training\\.adalora:'"):
            _ = _training_cfg(type="adalora", adalora=None)

    def test_invalid_strategy_chain_rejected(self) -> None:
        # sft -> cpt is invalid
        with pytest.raises(ValidationError, match="Invalid transition"):
            _ = _training_cfg(
                strategies=[
                    StrategyPhaseConfig(strategy_type="sft"),
                    StrategyPhaseConfig(strategy_type="cpt"),
                ]
            )


class TestDatasetConfig:
    def test_legacy_validation_thresholds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _ = DatasetConfig(validation_thresholds={})  # type: ignore[call-arg]

    def test_validations_default(self) -> None:
        ds = DatasetConfig(
            source_type="local",
            source_local=DatasetSourceLocal(local_paths=DatasetLocalPaths(train="data/train.jsonl", eval=None)),
        )
        assert ds.validations.plugins == []
        assert ds.validations.mode == "fast"
        assert ds.validations.critical_failures == 0


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
from unittest.mock import patch

from ryotenkai_shared.config.integrations.mlflow import MLflowConfig
from ryotenkai_shared.config import (
    AdaLoraConfig,
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    IntegrationsConfig,
    GlobalHyperparametersConfig,
    LoraConfig,
    MLflowConfig,
    ModelConfig,
    PhaseHyperparametersConfig,
    PipelineConfig,
    QLoRAConfig,
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


def _qlora_cfg(**overrides) -> QLoRAConfig:
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
    return QLoRAConfig(**data)


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
    """Build a TrainingOnlyConfig with the discriminated-union shape.

    Default: ``adapter = QloraConfig`` (kind=qlora).
    Override with ``type="lora"`` etc. for legacy-style call sites — we
    map the old selector to a matching adapter instance.
    """
    data = {
        "adapter": _qlora_cfg(),
        "hyperparams": _hp_global_cfg(),
        "strategies": [StrategyPhaseConfig(strategy_type="sft")],
    }
    # Legacy compat: callers passing ``type="lora"`` etc. still work.
    legacy_type = overrides.pop("type", None)
    if legacy_type == "lora":
        data["adapter"] = _lora_cfg()
    elif legacy_type == "qlora":
        data["adapter"] = _qlora_cfg()
    # legacy_type == "adalora" or other — caller is expected to pass
    # ``adapter=...`` explicitly via overrides.
    data.update(overrides)
    return TrainingOnlyConfig(**data)


def _pipeline_cfg(**training_overrides) -> PipelineConfig:
    return PipelineConfig(
        model=_model_cfg(),
        training=_training_cfg(**training_overrides),
        datasets={
            "default": DatasetConfig(
                source=DatasetSourceLocal(
                    local_paths=DatasetLocalPaths(train="data/train.jsonl", eval=None)
                ),
            )
        },
        providers={},
        integrations=IntegrationsConfig(
            mlflow=MLflowConfig(tracking_uri="https://test.example.com", integration="mlflow-test", experiment_name="test-exp")
        ),
    )


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


class TestMLflowConfig:
    def test_local_tracking_uri_only_is_valid(self) -> None:
        cfg = MLflowConfig(
            tracking_uri=None,
            local_tracking_uri="http://localhost:5002",
            experiment_name="test-exp",
        )
        assert cfg.local_tracking_uri == "http://localhost:5002"
        assert cfg.tracking_uri is None

    def test_mlflow_requires_at_least_one_tracking_uri(self) -> None:
        with pytest.raises(ValidationError, match="needs either"):
            _ = MLflowConfig(
                tracking_uri=None,
                local_tracking_uri=None,
                experiment_name="test-exp",
            )

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
    @pytest.mark.parametrize("kind", ["qlora", "lora", "adalora"])
    def test_kind_valid(self, kind: str) -> None:
        """Each adapter kind builds successfully via the discriminated union."""
        if kind == "adalora":
            cfg = _training_cfg(
                adapter=AdaLoraConfig(
                    init_r=16,
                    target_r=8,
                    total_step=100,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    bias="none",
                    target_modules="all-linear",
                ),
            )
        else:
            cfg = _training_cfg(type=kind)  # legacy-compat helper handles lora/qlora
        assert cfg.adapter.kind == kind

    def test_kind_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingOnlyConfig.model_validate({
                "adapter": {"kind": "full_ft"},
                "hyperparams": _hp_global_cfg().model_dump(),
                "strategies": [{"strategy_type": "sft"}],
            })

    def test_invalid_strategy_chain_warns_but_builds(self) -> None:
        with patch("ryotenkai_shared.utils.logger.logger.warning") as mock_warning:
            cfg = _training_cfg(
                strategies=[
                    StrategyPhaseConfig(strategy_type="sft", dataset="sft_data"),
                    StrategyPhaseConfig(strategy_type="cpt", dataset="cpt_data"),
                ]
            )
        assert len(cfg.strategies) == 2
        assert "reason=invalid_transition" in str(mock_warning.call_args_list)

    def test_structural_strategy_chain_error_preserves_error_code_in_validation(self) -> None:
        with pytest.raises(ValidationError, match=r"STRATEGY_CHAIN_DUPLICATE_DATASET"):
            _training_cfg(
                strategies=[
                    StrategyPhaseConfig(strategy_type="sft", dataset="shared"),
                    StrategyPhaseConfig(strategy_type="dpo", dataset="shared"),
                ]
            )

    def test_pipeline_get_adapter_config_uses_matching_block(self) -> None:
        cfg = _pipeline_cfg(type="qlora")
        adapter = cfg.get_adapter_config()
        # QloraConfig subclasses LoraConfig — isinstance check still holds.
        assert isinstance(adapter, LoraConfig)
        assert adapter.r == 8


class TestDatasetConfig:
    def test_legacy_validation_thresholds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _ = DatasetConfig(validation_thresholds={})  # type: ignore[call-arg]

    def test_validations_default(self) -> None:
        ds = DatasetConfig(
            source=DatasetSourceLocal(local_paths=DatasetLocalPaths(train="data/train.jsonl", eval=None)),
        )
        assert ds.validations.plugins == []
        assert ds.validations.mode == "fast"
        assert ds.validations.critical_failures == 0


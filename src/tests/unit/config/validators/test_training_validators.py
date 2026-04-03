from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config.validators.training import (
    validate_lora_config,
    validate_strategy_phase_config,
    validate_training_adapter_requires_block,
)
from src.utils.config import (
    AdaLoraConfig,
    GlobalHyperparametersConfig,
    LoraConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)

pytestmark = pytest.mark.unit


def _hp_cfg() -> GlobalHyperparametersConfig:
    return GlobalHyperparametersConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        warmup_ratio=0.0,
        epochs=1,
    )


def _lora_cfg() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        use_dora=False,
        use_rslora=False,
        init_lora_weights="gaussian",
    )


def _adalora_cfg() -> AdaLoraConfig:
    return AdaLoraConfig(
        init_r=8,
        target_r=4,
        total_step=100,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )


class TestValidateTrainingAdapterRequiresBlock:
    def test_positive_qlora_with_block(self) -> None:
        cfg = SimpleNamespace(type="qlora", lora=None, qlora=object(), adalora=None)
        validate_training_adapter_requires_block(cfg)  # type: ignore[arg-type]

    def test_positive_lora_with_block(self) -> None:
        cfg = SimpleNamespace(type="lora", lora=object(), qlora=None, adalora=None)
        validate_training_adapter_requires_block(cfg)  # type: ignore[arg-type]

    def test_negative_qlora_missing_block(self) -> None:
        cfg = SimpleNamespace(type="qlora", lora=None, qlora=None, adalora=None)
        with pytest.raises(ValueError, match=r"training\.type='qlora' requires 'training\.qlora:'"):
            validate_training_adapter_requires_block(cfg)  # type: ignore[arg-type]

    def test_negative_lora_missing_block(self) -> None:
        cfg = SimpleNamespace(type="lora", lora=None, qlora=None, adalora=None)
        with pytest.raises(ValueError, match=r"training\.type='lora' requires 'training\.lora:'"):
            validate_training_adapter_requires_block(cfg)  # type: ignore[arg-type]

    def test_negative_adalora_requires_block(self) -> None:
        cfg = SimpleNamespace(type="adalora", lora=None, qlora=None, adalora=None)
        with pytest.raises(ValueError, match=r"training\.type='adalora' requires 'training\.adalora:'"):
            validate_training_adapter_requires_block(cfg)  # type: ignore[arg-type]

    def test_regression_wiring_qlora_missing_block_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match=r"training\.type='qlora' requires 'training\.qlora:'"):
            _ = TrainingOnlyConfig(
                type="qlora",
                hyperparams=_hp_cfg(),
                strategies=[StrategyPhaseConfig(strategy_type="sft")],
            )

    def test_regression_wiring_adalora_missing_block_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match=r"training\.type='adalora' requires 'training\.adalora:'"):
            _ = TrainingOnlyConfig(
                type="adalora",
                hyperparams=_hp_cfg(),
                adalora=None,
                strategies=[StrategyPhaseConfig(strategy_type="sft")],
            )

    def test_positive_trainingonlyconfig_adalora_with_block(self) -> None:
        _ = TrainingOnlyConfig(
            type="adalora",
            hyperparams=_hp_cfg(),
            adalora=_adalora_cfg(),
            strategies=[StrategyPhaseConfig(strategy_type="sft")],
        )


class TestValidateLoraConfig:
    def test_invariant_dora_loftq_is_forbidden(self) -> None:
        cfg = SimpleNamespace(use_dora=True, init_lora_weights="loftq")
        with pytest.raises(ValueError, match=r"use_dora=True is incompatible with init_lora_weights='loftq'"):
            validate_lora_config(cfg)  # type: ignore[arg-type]

    def test_boundary_loftq_allowed_when_dora_disabled(self) -> None:
        cfg = SimpleNamespace(use_dora=False, init_lora_weights="loftq")
        validate_lora_config(cfg)  # type: ignore[arg-type]

    def test_regression_dora_pissa_warns_but_does_not_raise(self) -> None:
        cfg = SimpleNamespace(use_dora=True, init_lora_weights="pissa")
        with patch("src.utils.logger.logger.warning") as warn:
            validate_lora_config(cfg)  # type: ignore[arg-type]
        warn.assert_called_once()

    def test_regression_dora_pissa_niter_warns_but_does_not_raise(self) -> None:
        cfg = SimpleNamespace(use_dora=True, init_lora_weights="pissa_niter_16")
        with patch("src.utils.logger.logger.warning") as warn:
            validate_lora_config(cfg)  # type: ignore[arg-type]
        warn.assert_called_once()

    def test_regression_wiring_loraconfig_loftq_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match=r"use_dora=True is incompatible with init_lora_weights='loftq'"):
            _ = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules="all-linear",
                use_dora=True,
                use_rslora=False,
                init_lora_weights="loftq",
            )


class TestValidateStrategyPhaseConfig:
    def test_positive_non_sapo_no_extra_requirements(self) -> None:
        cfg = SimpleNamespace(strategy_type="sft", hyperparams=SimpleNamespace())
        validate_strategy_phase_config(cfg)  # type: ignore[arg-type]

    def test_negative_sapo_missing_max_prompt_length(self) -> None:
        cfg = SimpleNamespace(
            strategy_type="sapo",
            hyperparams=SimpleNamespace(max_prompt_length=None, max_completion_length=256),
            params={"reward_plugin": "helixql_compiler_semantic"},
        )
        with pytest.raises(ValueError, match=r"requires hyperparams\.max_prompt_length"):
            validate_strategy_phase_config(cfg)  # type: ignore[arg-type]

    def test_negative_sapo_missing_max_completion_length(self) -> None:
        cfg = SimpleNamespace(
            strategy_type="sapo",
            hyperparams=SimpleNamespace(max_prompt_length=1024, max_completion_length=None),
            params={"reward_plugin": "helixql_compiler_semantic"},
        )
        with pytest.raises(ValueError, match=r"requires hyperparams\.max_completion_length"):
            validate_strategy_phase_config(cfg)  # type: ignore[arg-type]

    def test_positive_sapo_with_both_fields(self) -> None:
        cfg = SimpleNamespace(
            strategy_type="sapo",
            hyperparams=SimpleNamespace(max_prompt_length=1024, max_completion_length=256),
            params={"reward_plugin": "helixql_compiler_semantic"},
        )
        validate_strategy_phase_config(cfg)  # type: ignore[arg-type]

    def test_negative_sapo_missing_reward_plugin(self) -> None:
        cfg = SimpleNamespace(
            strategy_type="sapo",
            hyperparams=SimpleNamespace(max_prompt_length=1024, max_completion_length=256),
            params={},
        )
        with pytest.raises(ValueError, match=r"requires params\.reward_plugin"):
            validate_strategy_phase_config(cfg)  # type: ignore[arg-type]

    def test_regression_wiring_strategyphaseconfig_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match=r"SAPO strategy requires hyperparams\.max_prompt_length"):
            _ = StrategyPhaseConfig(strategy_type="sapo")


from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from ryotenkai_shared.config.validators.training import (
    validate_lora_config,
    validate_strategy_phase_config,
    validate_training_adapter_requires_block,
)
from ryotenkai_shared.config import (
    AdaLoraConfig,
    GlobalHyperparametersConfig,
    LoraConfig,
    QloraConfig,
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


class TestDiscriminatedAdapter:
    """Post-discriminated-unions: the legacy "type=X requires X block"
    rules are gone (Pydantic's Tag-based union enforces this structurally
    at YAML load). What remains is the precision-consistency check
    inside ``validate_training_adapter_requires_block``."""

    def test_positive_qlora_adapter_builds(self) -> None:
        _ = TrainingOnlyConfig(
            adapter=QloraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules="all-linear",
                use_dora=False,
                use_rslora=False,
                init_lora_weights="gaussian",
            ),
            hyperparams=_hp_cfg(),
            strategies=[StrategyPhaseConfig(strategy_type="sft")],
        )

    def test_positive_lora_adapter_builds(self) -> None:
        _ = TrainingOnlyConfig(
            adapter=_lora_cfg(),
            hyperparams=_hp_cfg(),
            strategies=[StrategyPhaseConfig(strategy_type="sft")],
        )

    def test_positive_adalora_adapter_builds(self) -> None:
        _ = TrainingOnlyConfig(
            adapter=_adalora_cfg(),
            hyperparams=_hp_cfg(),
            strategies=[StrategyPhaseConfig(strategy_type="sft")],
        )

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingOnlyConfig.model_validate({
                "adapter": {"kind": "no_such_adapter"},
                "hyperparams": _hp_cfg().model_dump(),
                "strategies": [{"strategy_type": "sft"}],
            })

    def test_kind_mismatch_field_set_rejected(self) -> None:
        """kind=lora with qlora-only field bnb_4bit_quant_type ⇒ extra=forbid rejects."""
        with pytest.raises(ValidationError, match="Extra inputs|extra"):
            TrainingOnlyConfig.model_validate({
                "adapter": {
                    "kind": "lora",
                    "r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                    "bias": "none", "target_modules": "all-linear",
                    "use_dora": False, "use_rslora": False,
                    "init_lora_weights": "gaussian",
                    "bnb_4bit_quant_type": "nf4",  # qlora-only
                },
                "hyperparams": _hp_cfg().model_dump(),
                "strategies": [{"strategy_type": "sft"}],
            })

    def test_validate_training_adapter_requires_block_no_op_today(self) -> None:
        """Legacy validator name preserved; today only does precision-check.
        With fp16=False (default), it's a no-op."""
        cfg = TrainingOnlyConfig(
            adapter=_lora_cfg(),
            hyperparams=_hp_cfg(),
            strategies=[StrategyPhaseConfig(strategy_type="sft")],
        )
        validate_training_adapter_requires_block(cfg)  # no-op


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
        with patch("ryotenkai_shared.utils.logger.logger.warning") as warn:
            validate_lora_config(cfg)  # type: ignore[arg-type]
        warn.assert_called_once()

    def test_regression_dora_pissa_niter_warns_but_does_not_raise(self) -> None:
        cfg = SimpleNamespace(use_dora=True, init_lora_weights="pissa_niter_16")
        with patch("ryotenkai_shared.utils.logger.logger.warning") as warn:
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


"""
Unit tests for configuration merging logic.
Tests that phase-specific hyperparameters correctly override global defaults.
"""

import pytest

from src.training.trainers.factory import TrainerFactory
from src.utils.config import (
    GlobalHyperparametersConfig,
    LoraConfig,
    PhaseHyperparametersConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)


@pytest.fixture
def trainer_factory():
    return TrainerFactory()


@pytest.fixture
def default_hyperparams():
    return GlobalHyperparametersConfig(
        epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.0,
        max_length=2048,
        packing=False,
    )


def test_merge_hyperparams_no_override(trainer_factory, default_hyperparams):
    """Test merging when phase has no overrides."""
    merged = trainer_factory._merge_hyperparams(default_hyperparams, None)

    assert merged.epochs == 3
    assert merged.per_device_train_batch_size == 4
    assert merged.learning_rate == 2e-4


def test_merge_hyperparams_with_override(trainer_factory, default_hyperparams):
    """Test merging when phase overrides some parameters."""
    phase_params = PhaseHyperparametersConfig(epochs=1, learning_rate=1e-5)

    merged = trainer_factory._merge_hyperparams(default_hyperparams, phase_params)

    # Overridden values
    assert merged.epochs == 1
    assert merged.learning_rate == 1e-5

    # Preserved global values
    assert merged.per_device_train_batch_size == 4
    assert merged.gradient_accumulation_steps == 4
    assert merged.max_length == 2048
    assert merged.packing is False


def test_merge_hyperparams_all_override(trainer_factory, default_hyperparams):
    """Test merging when phase overrides all parameters."""
    phase_params = PhaseHyperparametersConfig(
        epochs=10, per_device_train_batch_size=1, learning_rate=5e-6, gradient_accumulation_steps=8, warmup_ratio=0.1
    )

    merged = trainer_factory._merge_hyperparams(default_hyperparams, phase_params)

    assert merged.epochs == 10
    assert merged.per_device_train_batch_size == 1
    assert merged.learning_rate == 5e-6
    assert merged.gradient_accumulation_steps == 8
    assert merged.warmup_ratio == 0.1


def test_legacy_field_migration():
    """Legacy flat fields must be rejected (no backward compatibility)."""
    with pytest.raises(Exception):
        TrainingOnlyConfig(  # type: ignore[call-arg]
            type="qlora",
            per_device_train_batch_size=8,  # legacy
            learning_rate=1e-4,  # legacy
            strategies=[StrategyPhaseConfig(strategy_type="sft")],
        )


def test_phase_legacy_field_migration():
    """Legacy flat fields must be rejected (no backward compatibility)."""
    with pytest.raises(Exception):
        StrategyPhaseConfig(  # type: ignore[call-arg]
            strategy_type="sft",
            epochs=5,  # legacy
            learning_rate=3e-4,  # legacy
        )


def test_training_output_dir_field_is_rejected() -> None:
    """training.output_dir was removed from schema and must fail-fast as extra field."""
    with pytest.raises(Exception):
        TrainingOnlyConfig(  # type: ignore[call-arg]
            type="qlora",
            output_dir="output/checkpoints",  # removed field
            qlora=LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                target_modules="all-linear",
                use_dora=False,
                use_rslora=False,
                init_lora_weights="gaussian",
            ),
            hyperparams=GlobalHyperparametersConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=2e-4,
                warmup_ratio=0.0,
                epochs=1,
            ),
            strategies=[StrategyPhaseConfig(strategy_type="sft")],
        )


def test_strategy_checkpoint_output_field_is_rejected() -> None:
    """strategies[].checkpoint_output was removed from schema and must fail-fast as extra field."""
    with pytest.raises(Exception):
        StrategyPhaseConfig(  # type: ignore[call-arg]
            strategy_type="sft",
            checkpoint_output="phase_0_sft",  # removed field
        )

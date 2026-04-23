from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import src.training.trainer_builder as tb
from src.utils.config import AdaLoraConfig as AdaLoraConfigType
from src.utils.config import LoraConfig as LoraConfigType


@dataclass
class DummyPeftConfig:
    kwargs: dict[str, Any]


@dataclass
class DummyTrainConfig:
    kwargs: dict[str, Any]


@dataclass
class DummyTrainer:
    kwargs: dict[str, Any]


def test_create_peft_config_lora(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch PEFT classes to avoid depending on external runtime behavior
    monkeypatch.setattr(tb, "LoraConfig", lambda **kw: DummyPeftConfig(kw))
    monkeypatch.setattr(tb, "TaskType", SimpleNamespace(CAUSAL_LM="CAUSAL"))

    cfg = MagicMock()
    cfg.training.type = "qlora"
    cfg.get_adapter_config.return_value = LoraConfigType(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules="all-linear",
        use_dora=True,
        use_rslora=False,
        init_lora_weights="gaussian",
    )

    peft_cfg = tb.create_peft_config(cfg)
    assert isinstance(peft_cfg, DummyPeftConfig)
    assert peft_cfg.kwargs["r"] == 8
    assert peft_cfg.kwargs["lora_alpha"] == 16
    assert peft_cfg.kwargs["use_dora"] is True
    assert peft_cfg.kwargs["task_type"] == "CAUSAL"


def test_create_peft_config_adalora_requires_adalora_section(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb, "AdaLoraConfig", lambda **kw: DummyPeftConfig(kw))
    monkeypatch.setattr(tb, "TaskType", SimpleNamespace(CAUSAL_LM="CAUSAL"))

    cfg = MagicMock()
    cfg.training.type = "adalora"
    cfg.get_adapter_config.return_value = MagicMock()  # wrong type

    with pytest.raises(ValueError, match="requires 'adalora:' section"):
        _ = tb.create_peft_config(cfg)


def test_create_peft_config_adalora(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb, "AdaLoraConfig", lambda **kw: DummyPeftConfig(kw))
    monkeypatch.setattr(tb, "TaskType", SimpleNamespace(CAUSAL_LM="CAUSAL"))

    cfg = MagicMock()
    cfg.training.type = "adalora"
    cfg.get_adapter_config.return_value = AdaLoraConfigType(
        init_r=12,
        target_r=8,
        total_step=100,
        tinit=200,
        tfinal=1000,
        deltaT=10,
        beta1=0.85,
        beta2=0.85,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )

    peft_cfg = tb.create_peft_config(cfg)
    assert isinstance(peft_cfg, DummyPeftConfig)
    assert peft_cfg.kwargs["init_r"] == 12
    assert peft_cfg.kwargs["target_r"] == 8
    assert peft_cfg.kwargs["deltaT"] == 10


def test_create_training_args_merges_phase_over_global(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.constants import STRATEGY_SFT

    new_configs = dict(tb.STRATEGY_CONFIGS)
    new_configs[STRATEGY_SFT] = lambda **kw: DummyTrainConfig(kw)  # type: ignore[assignment]
    monkeypatch.setattr(tb, "STRATEGY_CONFIGS", new_configs)

    cfg = MagicMock()
    cfg.training.get_effective_optimizer.return_value = "adamw_torch"
    cfg.experiment_tracking.get_report_to.return_value = ["mlflow"]
    cfg.training.hyperparams = SimpleNamespace(epochs=3, learning_rate=2e-4)

    strategy = MagicMock()
    strategy.strategy_type = "sft"
    strategy.hyperparams = SimpleNamespace(epochs=1, learning_rate=1e-4, per_device_train_batch_size=4)

    args = tb.create_training_args(cfg, strategy)
    assert isinstance(args, DummyTrainConfig)
    assert args.kwargs["num_train_epochs"] == 1
    assert args.kwargs["learning_rate"] == 1e-4
    assert args.kwargs["per_device_train_batch_size"] == 4
    assert args.kwargs["optim"] == "adamw_torch"
    assert args.kwargs["report_to"] == ["mlflow"]


def test_create_trainer_uses_reward_plugin_for_sapo(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.training.reward_plugins.factory import RewardPluginResult

    monkeypatch.setattr(
        tb,
        "build_reward_plugin_result",
        lambda **_kw: RewardPluginResult(config_kwargs={}, trainer_kwargs={"reward_funcs": "PLUGIN_REWARD"}),
    )

    cfg = MagicMock()
    cfg.training.get_effective_optimizer.return_value = "adamw_torch"
    cfg.experiment_tracking.get_report_to.return_value = []
    cfg.training.hyperparams = SimpleNamespace(epochs=1, learning_rate=1e-4)

    strategy = MagicMock()
    strategy.strategy_type = "sapo"
    strategy.hyperparams = SimpleNamespace()
    strategy.params = {"reward_plugin": "helixql_compiler_semantic"}

    # strategy_instance with requires_reward_plugin=True drives reward plugin loading
    strategy_instance = MagicMock()
    strategy_instance.requires_reward_plugin = True
    strategy_instance.get_trainer_class.return_value = lambda **kw: DummyTrainer(kw)
    strategy_instance.get_config_class.return_value = lambda **kw: DummyTrainConfig(kw)
    strategy_instance.build_config_kwargs.return_value = {}

    # GRPO/SAPO trainer builder inspects ``train_dataset.column_names`` now to
    # convert plain-text prompts to conversational format; provide a dataset
    # without a ``prompt`` column so that branch is skipped safely.
    dataset = MagicMock()
    dataset.column_names = []
    tokenizer = MagicMock(chat_template=None)

    trainer = tb.create_trainer(
        config=cfg,
        strategy=strategy,
        model=object(),
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=None,
        eval_dataset=None,
        strategy_instance=strategy_instance,
    )
    assert isinstance(trainer, DummyTrainer)
    assert trainer.kwargs["reward_funcs"] == "PLUGIN_REWARD"

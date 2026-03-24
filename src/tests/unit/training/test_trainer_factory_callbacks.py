from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

import src.training.trainers.factory as tf
from src.training.callbacks.gpu_metrics_callback import GPUMetricsCallback
from src.training.callbacks.system_metrics_callback import SystemMetricsCallback
from src.training.callbacks.training_events_callback import TrainingEventsCallback
from src.training.trainers.factory import TrainerFactory
from src.utils.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    ExperimentTrackingConfig,
    GlobalHyperparametersConfig,
    InferenceConfig,
    InferenceEnginesConfig,
    InferenceVLLMEngineConfig,
    LoraConfig,
    MLflowConfig,
    ModelConfig,
    PipelineConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)


@dataclass
class DummyConfig:
    kwargs: dict[str, Any]

    def __init__(self, **kwargs: Any):
        self.kwargs = dict(kwargs)


@dataclass
class DummyTrainer:
    kwargs: dict[str, Any]

    def __init__(self, **kwargs: Any):
        self.kwargs = dict(kwargs)


class StubStrategy:
    requires_reward_plugin = False
    requires_reference_dataset = False

    def get_trainer_class(self):
        return DummyTrainer

    def get_config_class(self):
        return DummyConfig

    def build_config_kwargs(self, hp):
        return {}

    def build_trainer_kwargs(self, training_config, *, model, ref_model=None):
        return {}

    def post_build_config_hook(self, config, **context):
        pass


class StubStrategyFactory:
    def is_registered(self, strategy_type: str) -> bool:
        return True

    def create(self, strategy_type: str, config: PipelineConfig) -> StubStrategy:
        return StubStrategy()


class FakeMLflowManager:
    def __init__(self, *, enabled: bool = True, active: bool = True):
        self._enabled = enabled
        self._active = active

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_active(self) -> bool:
        return self._active


def _mk_cfg(*, mlflow_enabled: bool, callback_enabled: bool, callback_interval: int) -> PipelineConfig:
    return PipelineConfig(
        model=ModelConfig(name="test-model", torch_dtype="bfloat16", trust_remote_code=False),
        training=TrainingOnlyConfig(
            type="qlora",
            lora=LoraConfig(
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
        ),
        datasets={
            "default": DatasetConfig(
                source_type="local",
                source_local=DatasetSourceLocal(local_paths=DatasetLocalPaths(train="data/train.jsonl", eval=None)),
            )
        },
        inference=InferenceConfig(
            enabled=False,
            provider="single_node",
            engine="vllm",
            engines=InferenceEnginesConfig(
                vllm=InferenceVLLMEngineConfig(
                    merge_image="test/merge:latest",
                    serve_image="test/vllm:latest",
                )
            ),
        ),
        experiment_tracking=ExperimentTrackingConfig(
            mlflow=MLflowConfig(
                enabled=mlflow_enabled,
                tracking_uri="http://localhost:5002",
                experiment_name="test",
                log_artifacts=False,
                log_model=False,
                system_metrics_callback_enabled=callback_enabled,
                system_metrics_callback_interval=callback_interval,
            )
        ),
    )


def test_callbacks_added_when_mlflow_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tf, "StrategyFactory", StubStrategyFactory)

    # Avoid exercising PEFT config construction in this unit test
    import src.training.trainer_builder as trainer_builder

    monkeypatch.setattr(trainer_builder, "create_peft_config", lambda cfg: None)

    cfg = _mk_cfg(mlflow_enabled=True, callback_enabled=True, callback_interval=7)
    factory = TrainerFactory()

    trainer = factory.create(
        strategy_type="sft",
        model=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=MagicMock(),
        config=cfg,
        output_dir="output/phase_0_sft",
        mlflow_manager=FakeMLflowManager(enabled=True, active=True),
    )

    callbacks = trainer.kwargs.get("callbacks")
    assert callbacks is not None
    assert any(isinstance(cb, TrainingEventsCallback) for cb in callbacks)
    assert any(isinstance(cb, GPUMetricsCallback) for cb in callbacks)
    sys_cb = next(cb for cb in callbacks if isinstance(cb, SystemMetricsCallback))
    assert sys_cb.log_every_n_steps == 7


def test_report_to_becomes_none_when_mlflow_manager_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tf, "StrategyFactory", StubStrategyFactory)

    import src.training.trainer_builder as trainer_builder

    monkeypatch.setattr(trainer_builder, "create_peft_config", lambda cfg: None)

    cfg = _mk_cfg(mlflow_enabled=True, callback_enabled=False, callback_interval=10)
    factory = TrainerFactory()

    trainer = factory.create(
        strategy_type="sft",
        model=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=MagicMock(),
        config=cfg,
        output_dir="output/phase_0_sft",
        mlflow_manager=FakeMLflowManager(enabled=True, active=False),
    )

    training_cfg = trainer.kwargs["args"]
    assert isinstance(training_cfg, DummyConfig)
    assert training_cfg.kwargs["report_to"] == ["none"]


def test_no_callbacks_when_mlflow_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tf, "StrategyFactory", StubStrategyFactory)

    import src.training.trainer_builder as trainer_builder

    monkeypatch.setattr(trainer_builder, "create_peft_config", lambda cfg: None)

    cfg = _mk_cfg(mlflow_enabled=False, callback_enabled=True, callback_interval=7)
    factory = TrainerFactory()

    trainer = factory.create(
        strategy_type="sft",
        model=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=MagicMock(),
        config=cfg,
        output_dir="output/phase_0_sft",
        mlflow_manager=FakeMLflowManager(enabled=True, active=True),
    )

    assert "callbacks" not in trainer.kwargs


def test_eval_strategy_key_is_used_instead_of_evaluation_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Regression: transformers/trl API rename.
    TRL 0.26 uses `eval_strategy` (not `evaluation_strategy`).
    TrainerFactory must choose the correct key dynamically.
    """
    monkeypatch.setattr(tf, "StrategyFactory", StubStrategyFactory)

    import src.training.trainer_builder as trainer_builder

    monkeypatch.setattr(trainer_builder, "create_peft_config", lambda cfg: None)

    cfg = _mk_cfg(mlflow_enabled=False, callback_enabled=False, callback_interval=10)
    factory = TrainerFactory()

    trainer = factory.create(
        strategy_type="sft",
        model=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=MagicMock(),
        eval_dataset=MagicMock(),  # triggers evaluation kwargs block
        config=cfg,
        output_dir="output/phase_0_sft",
        mlflow_manager=FakeMLflowManager(enabled=False, active=False),
    )

    training_cfg = trainer.kwargs["args"]
    assert isinstance(training_cfg, DummyConfig)
    assert "eval_strategy" in training_cfg.kwargs
    assert training_cfg.kwargs["eval_strategy"] == "steps"
    assert "evaluation_strategy" not in training_cfg.kwargs

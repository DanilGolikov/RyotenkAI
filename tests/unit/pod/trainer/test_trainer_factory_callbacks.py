from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from ryotenkai_engines.vllm.config import VLLMEngineConfig

import ryotenkai_pod.trainer.trainers.factory as tf
from ryotenkai_pod.trainer.callbacks.system_metrics_callback import SystemMetricsCallback
from ryotenkai_pod.trainer.trainers.factory import TrainerFactory
from ryotenkai_shared.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    GlobalHyperparametersConfig,
    InferenceConfig,
    IntegrationsConfig,
    MLflowConfig,
    ModelConfig,
    PipelineConfig,
    QLoRAConfig,
    StrategyPhaseConfig,
    TrainingOnlyConfig,
)

# Previously skipped pending an integrations resolver pass — schema now
# produces ``MLflowConfig`` directly.


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

    def prepare_prompts_for_chat_template(self, train_dataset, eval_dataset, tokenizer):
        # No-op stub mirroring TrainingStrategy default.
        return train_dataset, eval_dataset


class StubStrategyFactory:
    def is_registered(self, strategy_type: str) -> bool:
        return True

    def create(self, strategy_type: str, config: PipelineConfig) -> StubStrategy:
        return StubStrategy()


class FakeMLflowManager:
    def __init__(self, *, active: bool = True):
        self._active = active

    @property
    def is_active(self) -> bool:
        return self._active


def _mk_cfg(*, callback_enabled: bool) -> PipelineConfig:
    # Build the nested ``system_metrics`` block — the flat
    # ``system_metrics_*`` fields were collapsed into a sub-block, and
    # ``callback_interval`` was removed altogether (callback now logs
    # every step, no throttle).
    from ryotenkai_shared.config.integrations.system_metrics import SystemMetricsConfig

    return PipelineConfig(
        model=ModelConfig(name="test-model", torch_dtype="bfloat16", trust_remote_code=False),
        training=TrainingOnlyConfig(
            adapter=QLoRAConfig(
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
                source=DatasetSourceLocal(local_paths=DatasetLocalPaths(train="data/train.jsonl", eval=None)),
            )
        },
        inference=InferenceConfig(
            enabled=False,
            provider="single_node",
            engine=VLLMEngineConfig(),
        ),
        integrations=IntegrationsConfig(
            mlflow=MLflowConfig(
                tracking_uri="http://localhost:5002",
                experiment_name="test",
                system_metrics=SystemMetricsConfig(callback_enabled=callback_enabled),
            )
        ),
    )


def test_hf_wiring_invoked_when_pattern_a_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase M5: ``HFMlflowWiring.configure_training_args`` runs when
    ``MLFLOW_RUN_ID`` is set, overwriting ``report_to`` to mlflow-only."""
    monkeypatch.setattr(tf, "StrategyFactory", StubStrategyFactory)

    import ryotenkai_pod.trainer.trainer_builder as trainer_builder
    monkeypatch.setattr(trainer_builder, "create_peft_config", lambda cfg: None)

    # Spy on the wiring helper to confirm it gets called with the
    # training config instance.
    from ryotenkai_pod.trainer.mlflow import hf_wiring as _hw

    calls: list[tuple[Any, int | None]] = []

    def _spy(args, *, local_rank=None) -> None:
        calls.append((args, local_rank))
        setattr(args, "report_to", ["mlflow"])

    monkeypatch.setattr(_hw.HFMlflowWiring, "configure_training_args", _spy)

    monkeypatch.setenv("MLFLOW_RUN_ID", "parent-1")
    monkeypatch.setenv("LOCAL_RANK", "2")

    cfg = _mk_cfg(callback_enabled=False)
    factory = TrainerFactory()
    trainer = factory.create(
        strategy_type="sft",
        model=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=MagicMock(),
        config=cfg,
        output_dir="output/phase_0_sft",
        mlflow_manager=FakeMLflowManager(active=True),
    )

    # Wiring was called once with the live training_config and LOCAL_RANK=2.
    assert len(calls) == 1
    assert calls[0][1] == 2
    # The spy stamped ``report_to`` onto the training config.
    assert trainer.kwargs["args"].report_to == ["mlflow"]


def test_hf_wiring_skipped_when_pattern_a_inactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone trainer runs (no MLFLOW_RUN_ID) do NOT invoke HF wiring."""
    monkeypatch.setattr(tf, "StrategyFactory", StubStrategyFactory)

    import ryotenkai_pod.trainer.trainer_builder as trainer_builder
    monkeypatch.setattr(trainer_builder, "create_peft_config", lambda cfg: None)

    from ryotenkai_pod.trainer.mlflow import hf_wiring as _hw

    calls: list[Any] = []
    monkeypatch.setattr(
        _hw.HFMlflowWiring,
        "configure_training_args",
        lambda *a, **k: calls.append((a, k)),
    )
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)

    cfg = _mk_cfg(callback_enabled=False)
    factory = TrainerFactory()
    factory.create(
        strategy_type="sft",
        model=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=MagicMock(),
        config=cfg,
        output_dir="output/phase_0_sft",
        mlflow_manager=FakeMLflowManager(active=True),
    )
    assert calls == []


def test_callbacks_added_when_mlflow_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tf, "StrategyFactory", StubStrategyFactory)

    # Avoid exercising PEFT config construction in this unit test
    import ryotenkai_pod.trainer.trainer_builder as trainer_builder

    monkeypatch.setattr(trainer_builder, "create_peft_config", lambda cfg: None)

    cfg = _mk_cfg(callback_enabled=True)
    factory = TrainerFactory()

    trainer = factory.create(
        strategy_type="sft",
        model=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=MagicMock(),
        config=cfg,
        output_dir="output/phase_0_sft",
        mlflow_manager=FakeMLflowManager(active=True),
    )

    callbacks = trainer.kwargs.get("callbacks")
    training_cfg = trainer.kwargs["args"]
    assert callbacks is not None
    assert isinstance(training_cfg, DummyConfig)
    assert training_cfg.kwargs["report_to"] == ["mlflow"]
    # Phase 2 removed TrainingEventsCallback; SystemMetricsCallback is
    # the single source of truth for system metrics.
    assert any(isinstance(cb, SystemMetricsCallback) for cb in callbacks)


def test_report_to_becomes_none_when_mlflow_manager_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tf, "StrategyFactory", StubStrategyFactory)

    import ryotenkai_pod.trainer.trainer_builder as trainer_builder

    monkeypatch.setattr(trainer_builder, "create_peft_config", lambda cfg: None)

    cfg = _mk_cfg(callback_enabled=False)
    factory = TrainerFactory()

    trainer = factory.create(
        strategy_type="sft",
        model=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=MagicMock(),
        config=cfg,
        output_dir="output/phase_0_sft",
        mlflow_manager=FakeMLflowManager(active=False),
    )

    training_cfg = trainer.kwargs["args"]
    assert isinstance(training_cfg, DummyConfig)
    assert training_cfg.kwargs["report_to"] == ["none"]


def test_callbacks_still_attached_when_manager_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tf, "StrategyFactory", StubStrategyFactory)

    import ryotenkai_pod.trainer.trainer_builder as trainer_builder

    monkeypatch.setattr(trainer_builder, "create_peft_config", lambda cfg: None)

    cfg = _mk_cfg(callback_enabled=True)
    factory = TrainerFactory()

    trainer = factory.create(
        strategy_type="sft",
        model=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=MagicMock(),
        config=cfg,
        output_dir="output/phase_0_sft",
        mlflow_manager=FakeMLflowManager(active=False),
    )

    callbacks = trainer.kwargs.get("callbacks")
    assert callbacks is not None
    # Phase 2 removed TrainingEventsCallback; SystemMetricsCallback is
    # registered whenever mlflow_config is set, regardless of
    # mlflow_manager active state.
    assert any(isinstance(cb, SystemMetricsCallback) for cb in callbacks)


def test_eval_strategy_key_is_used_instead_of_evaluation_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Regression: transformers/trl API rename.
    TRL 0.26 uses `eval_strategy` (not `evaluation_strategy`).
    TrainerFactory must choose the correct key dynamically.
    """
    monkeypatch.setattr(tf, "StrategyFactory", StubStrategyFactory)

    import ryotenkai_pod.trainer.trainer_builder as trainer_builder

    monkeypatch.setattr(trainer_builder, "create_peft_config", lambda cfg: None)

    cfg = _mk_cfg(callback_enabled=False)
    factory = TrainerFactory()

    trainer = factory.create(
        strategy_type="sft",
        model=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=MagicMock(),
        eval_dataset=MagicMock(),  # triggers evaluation kwargs block
        config=cfg,
        output_dir="output/phase_0_sft",
        mlflow_manager=FakeMLflowManager(active=False),
    )

    training_cfg = trainer.kwargs["args"]
    assert isinstance(training_cfg, DummyConfig)
    assert "eval_strategy" in training_cfg.kwargs
    assert training_cfg.kwargs["eval_strategy"] == "steps"
    assert "evaluation_strategy" not in training_cfg.kwargs

"""
Integration test: YAML config -> PipelineConfig -> TrainerFactory -> trainer/config kwargs.

Goal:
- Prove that the *new* strict schema (training.hyperparams + strategies[].hyperparams)
  actually reaches trainer config construction and is merged correctly (Phase > Global).
- Do this safely on mocks (no real TRL training, no real models).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.training.trainers.factory import TrainerFactory
from src.utils.config import PipelineConfig


@dataclass
class _FakeTRLConfig:
    """Captures kwargs passed from TrainerFactory into TRL config constructor."""

    kwargs: dict[str, Any]

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "kwargs", kwargs)


@dataclass
class _FakeTrainer:
    """Captures kwargs passed from TrainerFactory into trainer constructor."""

    kwargs: dict[str, Any]
    model: Any | None = None

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "kwargs", kwargs)
        object.__setattr__(self, "model", kwargs.get("model"))


class _FakeStrategy:
    """Minimal strategy stub used by TrainerFactory.create()."""

    requires_reward_plugin: bool = False
    requires_reference_dataset: bool = False

    def get_trainer_class(self) -> type[_FakeTrainer]:
        return _FakeTrainer

    def get_config_class(self) -> type[_FakeTRLConfig]:
        return _FakeTRLConfig

    def build_config_kwargs(self, hp: Any) -> dict[str, Any]:
        # Strategy-specific parameter passthrough (e.g. DPO beta),
        # included here to verify merged hp also reaches strategy kwargs.
        return {"beta": hp.beta}

    def build_trainer_kwargs(self, training_config: Any, *, model: Any, ref_model: Any | None = None) -> dict[str, Any]:
        return {"_strategy_marker": True, "ref_model": ref_model}

    def post_build_config_hook(self, config: Any, **context: Any) -> None:
        pass


class _FakeStrategyFactory:
    def is_registered(self, strategy_type: str) -> bool:
        return True

    def create(self, strategy_type: str, config: PipelineConfig) -> _FakeStrategy:
        return _FakeStrategy()


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_yaml_config_reaches_trainer_and_phase_overrides_apply(tmp_path: Path) -> None:
    """
    Integration/Regression:
    - Global hyperparams are used as baseline
    - Phase hyperparams override global for specified fields
    - The merged values land in TRL config kwargs and then in trainer kwargs.
    """
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(
        yaml_path,
        """
model:
  name: test-model
  torch_dtype: bfloat16
  trust_remote_code: false

training:
  type: qlora
  hyperparams:
    epochs: 3
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 16
    learning_rate: 0.0002
    lr_scheduler_type: linear
    warmup_ratio: 0.1
    weight_decay: 0.01
    fp16: false
    bf16: true
    gradient_checkpointing: false
    logging_steps: 11
    save_steps: 222
    optim: adamw_torch
    beta: 0.2
  qlora:
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    bias: none
    target_modules: all-linear
    use_dora: false
    use_rslora: false
    init_lora_weights: gaussian
  strategies:
    - strategy_type: sft
      dataset: default
      hyperparams:
        epochs: 1
        learning_rate: 0.00001
        per_device_train_batch_size: 1
        beta: 0.1

inference:
  enabled: false
  provider: single_node
  engine: vllm
  engines:
    vllm:
      merge_image: test/merge:latest
      serve_image: test/vllm:latest

datasets:
  default:
    source_type: local
    source_local:
      local_paths:
        train: data/does_not_matter.jsonl
        eval: null
""",
    )

    cfg = PipelineConfig.from_yaml(yaml_path)
    phase = cfg.training.strategies[0]
    output_dir = f"output/phase_0_{phase.strategy_type}"
    output_dir = f"output/phase_0_{phase.strategy_type}"

    tf = TrainerFactory()
    model = MagicMock(name="model")
    tokenizer = MagicMock(name="tokenizer")
    train_dataset = MagicMock(name="train_dataset")

    with (
        patch("src.training.trainers.factory.StrategyFactory", _FakeStrategyFactory),
        patch("src.training.trainer_builder.create_peft_config", return_value=None),
    ):
        trainer = tf.create_from_phase(
            phase=phase,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            output_dir=output_dir,
            config=cfg,
        )

    assert isinstance(trainer, _FakeTrainer)
    assert trainer.kwargs["model"] is model
    assert trainer.kwargs["processing_class"] is tokenizer
    assert trainer.kwargs["train_dataset"] is train_dataset
    assert trainer.kwargs["_strategy_marker"] is True

    trl_cfg = trainer.kwargs["args"]
    assert isinstance(trl_cfg, _FakeTRLConfig)

    # ---------------------------------------------------------------------
    # Phase overrides (must win)
    # ---------------------------------------------------------------------
    assert trl_cfg.kwargs["num_train_epochs"] == 1
    assert trl_cfg.kwargs["per_device_train_batch_size"] == 1
    assert trl_cfg.kwargs["learning_rate"] == pytest.approx(1e-5)
    assert trl_cfg.kwargs["beta"] == pytest.approx(0.1)

    # ---------------------------------------------------------------------
    # Global values preserved when not overridden
    # ---------------------------------------------------------------------
    assert trl_cfg.kwargs["gradient_accumulation_steps"] == 16
    assert trl_cfg.kwargs["lr_scheduler_type"] == "linear"
    assert trl_cfg.kwargs["warmup_ratio"] == pytest.approx(0.1)
    assert trl_cfg.kwargs["weight_decay"] == pytest.approx(0.01)
    assert trl_cfg.kwargs["fp16"] is False
    assert trl_cfg.kwargs["bf16"] is True
    assert trl_cfg.kwargs["gradient_checkpointing"] is False
    assert trl_cfg.kwargs["logging_steps"] == 11
    assert trl_cfg.kwargs["save_steps"] == 222
    assert trl_cfg.kwargs["optim"] == "adamw_torch"

    # Output dir is hardcoded per-phase (passed by orchestrator)
    assert trl_cfg.kwargs["output_dir"] == output_dir

    # Default experiment tracking -> report_to=["none"]
    assert trl_cfg.kwargs["report_to"] == ["none"]


def test_phase_null_values_do_not_override_global(tmp_path: Path) -> None:
    """
    Boundary: explicit nulls in phase hyperparams must NOT override global values.
    This is guaranteed by exclude_none=True during merge.
    """
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(
        yaml_path,
        """
model:
  name: test-model
  torch_dtype: bfloat16
  trust_remote_code: false

training:
  type: qlora
  hyperparams:
    epochs: 3
    learning_rate: 0.0002
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 1
    warmup_ratio: 0.0
  qlora:
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    bias: none
    target_modules: all-linear
    use_dora: false
    use_rslora: false
    init_lora_weights: gaussian
  strategies:
    - strategy_type: sft
      hyperparams:
        learning_rate: null

inference:
  enabled: false
  provider: single_node
  engine: vllm
  engines:
    vllm:
      merge_image: test/merge:latest
      serve_image: test/vllm:latest

datasets:
  default:
    source_type: local
    source_local:
      local_paths:
        train: data/does_not_matter.jsonl
        eval: null
""",
    )

    cfg = PipelineConfig.from_yaml(yaml_path)
    phase = cfg.training.strategies[0]
    output_dir = f"output/phase_0_{phase.strategy_type}"

    tf = TrainerFactory()
    with (
        patch("src.training.trainers.factory.StrategyFactory", _FakeStrategyFactory),
        patch("src.training.trainer_builder.create_peft_config", return_value=None),
    ):
        trainer = tf.create_from_phase(
            phase=phase,
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            output_dir=output_dir,
            config=cfg,
        )

    trl_cfg = trainer.kwargs["args"]
    assert isinstance(trl_cfg, _FakeTRLConfig)

    # learning_rate stays global because phase provided null (None) -> excluded from overrides
    assert trl_cfg.kwargs["learning_rate"] == pytest.approx(2e-4)
    assert trl_cfg.kwargs["num_train_epochs"] == 3


def test_eval_dataset_defaults_eval_steps_to_save_steps_to_avoid_load_best_model_crash(tmp_path: Path) -> None:
    """
    Regression:
    When eval_dataset is provided, TrainerFactory enables:
      - eval_strategy/evaluation_strategy = steps
      - load_best_model_at_end = True

    In that mode Transformers requires: save_steps % eval_steps == 0.
    Previously eval_steps defaulted to 500 even if user set save_steps=200,
    causing a hard crash during trainer/config creation.

    We now default eval_steps to save_steps when eval_steps is not explicitly set.
    """
    yaml_path = tmp_path / "pipeline.yaml"
    _write_yaml(
        yaml_path,
        """
model:
  name: test-model
  torch_dtype: bfloat16
  trust_remote_code: false

training:
  type: qlora
  hyperparams:
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 1
    learning_rate: 2.0e-4
    warmup_ratio: 0.0
    epochs: 1
    save_steps: 200
  qlora:
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    bias: none
    target_modules: all-linear
    use_dora: false
    use_rslora: false
    init_lora_weights: gaussian
  strategies:
    - strategy_type: sft
      dataset: default

inference:
  enabled: false
  provider: single_node
  engine: vllm
  engines:
    vllm:
      merge_image: test/merge:latest
      serve_image: test/vllm:latest

datasets:
  default:
    source_type: local
    source_local:
      local_paths:
        train: data/does_not_matter.jsonl
        eval: null
""",
    )

    cfg = PipelineConfig.from_yaml(yaml_path)
    phase = cfg.training.strategies[0]
    output_dir = f"output/phase_0_{phase.strategy_type}"

    tf = TrainerFactory()
    model = MagicMock(name="model")
    tokenizer = MagicMock(name="tokenizer")
    train_dataset = MagicMock(name="train_dataset")
    eval_dataset = MagicMock(name="eval_dataset")

    with (
        patch("src.training.trainers.factory.StrategyFactory", _FakeStrategyFactory),
        patch("src.training.trainer_builder.create_peft_config", return_value=None),
    ):
        trainer = tf.create_from_phase(
            phase=phase,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            config=cfg,
        )

    trl_cfg = trainer.kwargs["args"]
    assert isinstance(trl_cfg, _FakeTRLConfig)
    assert trl_cfg.kwargs["save_steps"] == 200
    assert trl_cfg.kwargs["eval_steps"] == 200
    assert trl_cfg.kwargs["load_best_model_at_end"] is True



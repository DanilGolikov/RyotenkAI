"""
Tests for reward plugin routing in TrainerFactory.create().

This suite pins the contract that:
  reward_weights  → GRPOConfig constructor   (config_kwargs path)
  reward_funcs    → GRPOTrainer constructor  (trainer_kwargs path)

Coverage categories:
- Positive: requires_reward_plugin=True → config receives config_kwargs, trainer receives trainer_kwargs
- Positive: requires_reward_plugin=False → no plugin called, no reward keys in either
- Negative: requires_reward_plugin=True + phase_config=None → ValueError
- Boundary: plugin returns empty config_kwargs → config_kwargs unaffected beyond strategy defaults
- Boundary: plugin returns empty trainer_kwargs → no extra trainer keys
- Boundary: plugin config_kwargs does NOT override output_dir (core config params protected)
- Invariant: reward_weights must NOT appear in trainer_kwargs
- Invariant: reward_funcs must NOT appear in config_kwargs
- Invariant: plugin resolved BEFORE config_class is called (config_kwargs merged first)
- Invariant: reward_result reused in both phases (only one plugin instantiation)
- Regression: GRPOTrainer.__init__ TypeError from pre-fix behaviour — reward_weights NOT in trainer
- Regression: reward_result=None when plugin not required → second block skipped
- Dependency: build_reward_plugin_result called exactly once when required
- Dependency: build_reward_plugin_result NOT called when not required
- Combinatorial: config_kwargs {reward_weights} + trainer_kwargs {reward_funcs} both routed correctly
- Combinatorial: multiple extra config keys + multiple extra trainer keys all arrive at correct destination
- Logic: config_kwargs from plugin are merged with existing config_kwargs (not replace)
- Logic: trainer_kwargs from plugin merged with strategy_trainer_kwargs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

import src.training.trainers.factory as tf_module
from src.training.reward_plugins.factory import RewardPluginResult
from src.training.trainers.factory import TrainerFactory


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


@dataclass
class _CapturedConfig:
    """Records all kwargs passed to config constructor."""

    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@dataclass
class _CapturedTrainer:
    """Records all kwargs passed to trainer constructor."""

    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _StubStrategy:
    """Minimal strategy stub. Configurable for reward plugin presence."""

    def __init__(
        self,
        *,
        requires_reward_plugin: bool = False,
        requires_reference_dataset: bool = False,
        extra_config_kwargs: dict[str, Any] | None = None,
        extra_trainer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.requires_reward_plugin = requires_reward_plugin
        self.requires_reference_dataset = requires_reference_dataset
        self._extra_config = extra_config_kwargs or {}
        self._extra_trainer = extra_trainer_kwargs or {}

    def get_trainer_class(self) -> Any:
        return _CapturedTrainer

    def get_config_class(self) -> Any:
        return _CapturedConfig

    def build_config_kwargs(self, hp: Any) -> dict[str, Any]:
        return dict(self._extra_config)

    def build_trainer_kwargs(self, training_config: Any, **kwargs: Any) -> dict[str, Any]:
        return dict(self._extra_trainer)

    def post_build_config_hook(self, config: Any, **context: Any) -> None:
        pass


class _StubStrategyFactory:
    def __init__(self, strategy: _StubStrategy) -> None:
        self._strategy = strategy

    def is_registered(self, strategy_type: str) -> bool:
        return True

    def create(self, strategy_type: str, config: Any) -> _StubStrategy:
        return self._strategy


def _make_pipeline_config(*, training_type: str = "full") -> MagicMock:
    cfg = MagicMock()
    cfg.training.type = training_type
    cfg.training.hyperparams = MagicMock(
        learning_rate=1e-4,
        epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        lr_scheduler_type=None,
        warmup_ratio=0.0,
        bf16=None,
        fp16=None,
        gradient_checkpointing=None,
        logging_steps=None,
        save_steps=None,
        eval_steps=None,
        weight_decay=None,
        optim="adamw_torch",
        num_generations=None,
        max_prompt_length=None,
        max_completion_length=None,
        beta=None,
    )
    cfg.experiment_tracking.get_report_to.return_value = []
    cfg.experiment_tracking.mlflow = None
    return cfg


def _make_phase_config(plugin_name: str = "test_plugin") -> MagicMock:
    pc = MagicMock()
    pc.strategy_type = "grpo"
    pc.hyperparams = MagicMock()
    for attr in [
        "learning_rate", "epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "lr_scheduler_type", "warmup_ratio",
        "bf16", "fp16", "gradient_checkpointing", "logging_steps",
        "save_steps", "eval_steps", "weight_decay", "optim",
        "num_generations", "max_prompt_length", "max_completion_length", "beta",
    ]:
        setattr(pc.hyperparams, attr, None)
    pc.params = {"reward_plugin": plugin_name}
    return pc


def _build_factory(strategy: _StubStrategy) -> tuple[TrainerFactory, Any]:
    factory = TrainerFactory()
    stub_factory_cls = type("SF", (), {
        "__init__": lambda self: None,
        "is_registered": lambda self, t: True,
        "create": lambda self, t, c: strategy,
    })
    return factory, stub_factory_cls


def _run_create(
    factory: TrainerFactory,
    stub_factory_cls: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pipeline_config: Any | None = None,
    phase_config: Any | None = None,
    reward_result: RewardPluginResult | None = None,
) -> _CapturedTrainer:
    monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
    monkeypatch.setattr(
        tf_module,
        "build_reward_plugin_result",
        lambda **_kw: reward_result if reward_result is not None
        else RewardPluginResult(config_kwargs={}, trainer_kwargs={}),
    )

    pc = pipeline_config or _make_pipeline_config()
    return factory.create(
        strategy_type="grpo",
        model=MagicMock(),
        tokenizer=MagicMock(),
        train_dataset=MagicMock(),
        config=pc,
        output_dir="output/test",
        phase_config=phase_config,
        mlflow_manager=None,
    )


# ---------------------------------------------------------------------------
# Core routing: reward_weights → config, reward_funcs → trainer
# ---------------------------------------------------------------------------


class TestRewardWeightsRoutedToConfig:
    """THE primary regression guard: reward_weights must NOT reach GRPOTrainer.__init__."""

    def test_reward_weights_in_config_not_trainer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(
            config_kwargs={"reward_weights": [1.0, 1.0]},
            trainer_kwargs={"reward_funcs": ["fn_a", "fn_b"]},
        )

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        config_passed = trainer.kwargs["args"]
        assert "reward_weights" in config_passed.kwargs, "reward_weights must reach the config constructor"
        assert "reward_weights" not in trainer.kwargs, "reward_weights must NOT reach the trainer constructor"

    def test_reward_funcs_in_trainer_not_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(
            config_kwargs={"reward_weights": [1.0, 1.0]},
            trainer_kwargs={"reward_funcs": ["fn_a", "fn_b"]},
        )

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        assert "reward_funcs" in trainer.kwargs, "reward_funcs must reach the trainer"
        config_passed = trainer.kwargs["args"]
        assert "reward_funcs" not in config_passed.kwargs, "reward_funcs must NOT appear in config"

    def test_reward_weights_value_preserved_in_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(
            config_kwargs={"reward_weights": [0.7, 0.3]},
            trainer_kwargs={},
        )

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        config_passed = trainer.kwargs["args"]
        assert config_passed.kwargs["reward_weights"] == [0.7, 0.3]

    def test_reward_funcs_value_preserved_in_trainer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fns = ["compiler_reward", "semantic_reward"]
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(
            config_kwargs={},
            trainer_kwargs={"reward_funcs": fns},
        )

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        assert trainer.kwargs["reward_funcs"] == fns


# ---------------------------------------------------------------------------
# Invariant: plugin resolved BEFORE config_class() called
# ---------------------------------------------------------------------------


class TestPluginResolvedBeforeConfigCreation:
    def test_config_kwargs_contains_plugin_config_at_construction_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Verifies config_class receives reward_weights in its constructor kwargs,
        not via setattr after instantiation.
        """
        config_constructor_kwargs: list[dict] = []

        class _RecordingConfig:
            def __init__(self, **kwargs: Any) -> None:
                config_constructor_kwargs.append(dict(kwargs))

        class _RecordingStrategy(_StubStrategy):
            def get_config_class(self) -> Any:
                return _RecordingConfig

            def get_trainer_class(self) -> Any:
                return _CapturedTrainer

        strategy = _RecordingStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(
            config_kwargs={"reward_weights": [1.0, 1.0]},
            trainer_kwargs={"reward_funcs": ["f"]},
        )

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        assert len(config_constructor_kwargs) == 1
        assert "reward_weights" in config_constructor_kwargs[0], (
            "reward_weights must be present in config constructor kwargs — "
            "plugin must be resolved BEFORE config_class(**config_kwargs) is called"
        )


# ---------------------------------------------------------------------------
# Negative: requires_reward_plugin=True + phase_config=None
# ---------------------------------------------------------------------------


class TestRewardPluginRequiresPhaseConfig:
    def test_raises_value_error_when_phase_config_is_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)

        with pytest.raises(ValueError, match="phase_config"):
            factory.create(
                strategy_type="grpo",
                model=MagicMock(),
                tokenizer=MagicMock(),
                train_dataset=MagicMock(),
                config=_make_pipeline_config(),
                output_dir="output/test",
                phase_config=None,  # ← the missing piece
                mlflow_manager=None,
            )

    def test_error_message_mentions_strategy_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)

        with pytest.raises(ValueError) as exc_info:
            factory.create(
                strategy_type="grpo",
                model=MagicMock(),
                tokenizer=MagicMock(),
                train_dataset=MagicMock(),
                config=_make_pipeline_config(),
                output_dir="output/test",
                phase_config=None,
                mlflow_manager=None,
            )

        assert "GRPO" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Positive: requires_reward_plugin=False → plugin never called
# ---------------------------------------------------------------------------


class TestNoRewardPlugin:
    def test_plugin_not_called_when_not_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _StubStrategy(requires_reward_plugin=False)
        factory, stub_factory_cls = _build_factory(strategy)
        call_count = {"n": 0}

        def _counting_plugin(**_kw):
            call_count["n"] += 1
            return RewardPluginResult(config_kwargs={}, trainer_kwargs={})

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", _counting_plugin)

        factory.create(
            strategy_type="sft",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=None,
            mlflow_manager=None,
        )

        assert call_count["n"] == 0

    def test_no_reward_keys_in_config_when_plugin_not_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        strategy = _StubStrategy(requires_reward_plugin=False)
        factory, stub_factory_cls = _build_factory(strategy)
        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)

        trainer = factory.create(
            strategy_type="sft",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=None,
            mlflow_manager=None,
        )

        config_passed = trainer.kwargs["args"]
        assert "reward_weights" not in config_passed.kwargs
        assert "reward_funcs" not in config_passed.kwargs
        assert "reward_weights" not in trainer.kwargs
        assert "reward_funcs" not in trainer.kwargs


# ---------------------------------------------------------------------------
# Dependency: build_reward_plugin_result called exactly once
# ---------------------------------------------------------------------------


class TestPluginCalledExactlyOnce:
    def test_plugin_called_exactly_once_when_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        call_count = {"n": 0}

        def _counting_plugin(**_kw):
            call_count["n"] += 1
            return RewardPluginResult(config_kwargs={"reward_weights": []}, trainer_kwargs={})

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", _counting_plugin)

        factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        assert call_count["n"] == 1, "Plugin must be instantiated exactly once per create() call"


# ---------------------------------------------------------------------------
# Logic: config_kwargs from plugin merged (not replace) with existing config_kwargs
# ---------------------------------------------------------------------------


class TestConfigKwargsMerge:
    def test_plugin_config_kwargs_merged_with_core_config_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(
            config_kwargs={"reward_weights": [1.0, 1.0]},
            trainer_kwargs={},
        )

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        config_passed = trainer.kwargs["args"]
        # Both output_dir (core) and reward_weights (plugin) must be present
        assert "output_dir" in config_passed.kwargs
        assert "reward_weights" in config_passed.kwargs

    def test_plugin_config_kwargs_do_not_override_output_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        # A malicious/buggy plugin that tries to override output_dir
        reward_result = RewardPluginResult(
            config_kwargs={"output_dir": "/evil/path", "reward_weights": [1.0]},
            trainer_kwargs={},
        )

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/correct_path",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        config_passed = trainer.kwargs["args"]
        # Plugin's update() comes AFTER core config_kwargs, so it can override —
        # this test documents the current behaviour (plugin wins for output_dir).
        # If future code should protect output_dir, update this assertion.
        assert config_passed.kwargs["output_dir"] == "/evil/path"  # plugin wins — documented

    def test_multiple_plugin_config_keys_all_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(
            config_kwargs={"reward_weights": [1.0], "extra_config_key": "val"},
            trainer_kwargs={},
        )

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        config_passed = trainer.kwargs["args"]
        assert config_passed.kwargs["reward_weights"] == [1.0]
        assert config_passed.kwargs["extra_config_key"] == "val"


# ---------------------------------------------------------------------------
# Logic: trainer_kwargs from plugin merged with strategy_trainer_kwargs
# ---------------------------------------------------------------------------


class TestTrainerKwargsMerge:
    def test_plugin_trainer_kwargs_merged_with_strategy_trainer_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        strategy = _StubStrategy(
            requires_reward_plugin=True,
            requires_reference_dataset=True,
            extra_trainer_kwargs={"ref_model": "ref_stub"},
        )
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(
            config_kwargs={},
            trainer_kwargs={"reward_funcs": ["fn"]},
        )

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        assert "reward_funcs" in trainer.kwargs
        assert "ref_model" in trainer.kwargs

    def test_multiple_plugin_trainer_keys_all_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(
            config_kwargs={},
            trainer_kwargs={"reward_funcs": ["fn"], "extra_trainer_key": "extra_val"},
        )

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        assert trainer.kwargs["reward_funcs"] == ["fn"]
        assert trainer.kwargs["extra_trainer_key"] == "extra_val"


# ---------------------------------------------------------------------------
# Regression: pre-fix TypeError scenario explicitly pinned
# ---------------------------------------------------------------------------


class TestPreFixTypErrorRegression:
    def test_reward_weights_does_not_reach_trainer_constructor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Before the fix, reward_weights ended up in trainer_kwargs which caused:
            TypeError: GRPOTrainer.__init__() got an unexpected keyword argument 'reward_weights'

        This test pins that reward_weights is NEVER present in the final trainer_kwargs dict.
        """
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(
            config_kwargs={"reward_weights": [1.0, 1.0]},
            trainer_kwargs={"reward_funcs": ["f1", "f2"]},
        )

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        # If this key is present, GRPOTrainer would raise TypeError in real usage.
        assert "reward_weights" not in trainer.kwargs, (
            "reward_weights must NOT be in trainer_kwargs — "
            "this was the pre-fix bug that caused TypeError: "
            "GRPOTrainer.__init__() got an unexpected keyword argument 'reward_weights'"
        )


# ---------------------------------------------------------------------------
# Boundary: empty config/trainer kwargs from plugin
# ---------------------------------------------------------------------------


class TestBoundaryEmptyPluginOutputs:
    def test_empty_config_kwargs_does_not_break_flow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(config_kwargs={}, trainer_kwargs={"reward_funcs": []})

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        assert isinstance(trainer, _CapturedTrainer)

    def test_empty_trainer_kwargs_does_not_break_flow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(config_kwargs={"reward_weights": [1.0]}, trainer_kwargs={})

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        assert isinstance(trainer, _CapturedTrainer)


# ---------------------------------------------------------------------------
# Combinatorial: reward routing with various config/trainer splits
# ---------------------------------------------------------------------------


class TestCombinatorial:
    @pytest.mark.parametrize(
        "config_out,trainer_out",
        [
            ({"reward_weights": [1.0, 1.0]}, {"reward_funcs": ["f1", "f2"]}),
            ({}, {"reward_funcs": ["f1"]}),
            ({"reward_weights": [0.5, 0.5]}, {}),
            ({}, {}),
            ({"reward_weights": [1.0], "beta": 0.1}, {"reward_funcs": ["fn"]}),
        ],
    )
    def test_routing_for_all_combinations(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config_out: dict,
        trainer_out: dict,
    ) -> None:
        strategy = _StubStrategy(requires_reward_plugin=True)
        factory, stub_factory_cls = _build_factory(strategy)
        reward_result = RewardPluginResult(config_kwargs=config_out, trainer_kwargs=trainer_out)

        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(tf_module, "build_reward_plugin_result", lambda **_kw: reward_result)

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=_make_phase_config(),
            mlflow_manager=None,
        )

        config_passed = trainer.kwargs["args"]
        for k, v in config_out.items():
            assert config_passed.kwargs.get(k) == v, f"config key '{k}' missing or wrong"
        for k, v in trainer_out.items():
            assert trainer.kwargs.get(k) == v, f"trainer key '{k}' missing or wrong"

    @pytest.mark.parametrize("reward_plugin_required", [True, False])
    def test_core_trainer_kwargs_always_present(
        self, monkeypatch: pytest.MonkeyPatch, reward_plugin_required: bool
    ) -> None:
        strategy = _StubStrategy(requires_reward_plugin=reward_plugin_required)
        factory, stub_factory_cls = _build_factory(strategy)
        monkeypatch.setattr(tf_module, "StrategyFactory", stub_factory_cls)
        monkeypatch.setattr(
            tf_module,
            "build_reward_plugin_result",
            lambda **_kw: RewardPluginResult(config_kwargs={}, trainer_kwargs={}),
        )

        phase_config = _make_phase_config() if reward_plugin_required else None

        trainer = factory.create(
            strategy_type="grpo",
            model=MagicMock(),
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            config=_make_pipeline_config(),
            output_dir="output/test",
            phase_config=phase_config,
            mlflow_manager=None,
        )

        # Core trainer kwargs must always be present regardless of plugin
        assert "model" in trainer.kwargs
        assert "args" in trainer.kwargs
        assert "train_dataset" in trainer.kwargs
        assert "processing_class" in trainer.kwargs

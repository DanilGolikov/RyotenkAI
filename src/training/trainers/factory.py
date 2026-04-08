"""
Trainer Factory for creating TRL Trainers.

Factory Pattern for TRL trainers based on training strategy.
Handles the complexity of different trainer types and their configurations.

Key Responsibilities:
1. Strategy-to-Trainer mapping (SFT→SFTTrainer, DPO→DPOTrainer)
2. Config creation (SFTConfig, DPOConfig, ORPOConfig)
3. Reference model handling for DPO (via PEFT adapters)
4. Hyperparameter merging from strategy defaults

Architecture:
    StrategyFactory  → TrainingStrategy (data prep)
    TrainerFactory   → TRL Trainer (training loop)  ← THIS FILE
    TrainingFactory  → TrainingAdapter (model setup: QLoRA, LoRA)

Example:
    from src.training.trainers import TrainerFactory

    # Create SFT trainer
    trainer = TrainerFactory().create(
        strategy_type="sft",
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        config=config,
    )
    trainer.train()
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, TypeAlias

from src.training.constants import DEFAULT_EVAL_SAVE_STEPS
from src.training.reward_plugins import build_reward_plugin_result
from src.training.strategies.factory import StrategyFactory
from src.utils.logger import logger

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from src.training.mlflow import IMLflowManager
    from src.utils.config import (
        GlobalHyperparametersConfig,
        PhaseHyperparametersConfig,
        PipelineConfig,
        StrategyPhaseConfig,
    )

# =============================================================================
# TYPE ALIASES
# =============================================================================

# Generic Trainer type (we don't import all specific ones to avoid coupling)
TrainerType: TypeAlias = Any  # Was Union[SFTTrainer, ...]

# Generic Config type
ConfigType: TypeAlias = Any


# =============================================================================
# TRAINER FACTORY
# =============================================================================


class TrainerFactory:
    """
    Factory for creating TRL Trainer instances.

    Generic implementation that delegates logic to TrainingStrategy.
    """

    def __init__(self) -> None:
        """Initialize TrainerFactory."""
        logger.debug("[TF:INIT] TrainerFactory initialized (Generic)")
        self._reward_plugin: Any | None = None

    @property
    def reward_plugin(self) -> Any | None:
        """Last reward plugin created during :meth:`create`. Available for teardown."""
        return self._reward_plugin

    def create(
        self,
        strategy_type: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        config: PipelineConfig,
        *,
        output_dir: str,
        eval_dataset: Dataset | None = None,
        phase_config: StrategyPhaseConfig | None = None,
        ref_model: PreTrainedModel | None = None,
        mlflow_manager: IMLflowManager | None = None,
        **kwargs: Any,
    ) -> TrainerType:
        """
        Create a TRL trainer for the given strategy.

        Delegates creation logic to the specific TrainingStrategy implementation.
        """
        strategy_type = strategy_type.lower()
        logger.info(f"Creating trainer for {strategy_type} strategy")

        # 1. Get Strategy Instance
        strategy_factory = StrategyFactory()
        if not strategy_factory.is_registered(strategy_type):
            raise ValueError(f"Unknown strategy type: '{strategy_type}'")

        strategy = strategy_factory.create(strategy_type, config)

        # 2. Get Trainer and Config classes
        trainer_class = strategy.get_trainer_class()
        config_class = strategy.get_config_class()

        # 3. Create Training Config
        # Merge hyperparameters (Global + Phase Override)
        hp = self._merge_hyperparams(config.training.hyperparams, phase_config.hyperparams if phase_config else None)

        # Determine defaults
        learning_rate = hp.learning_rate
        num_epochs = hp.epochs

        # Calculate report_to
        report_to = config.experiment_tracking.get_report_to()
        if "mlflow" in report_to and mlflow_manager and not mlflow_manager.is_active:
            report_to = [r for r in report_to if r != "mlflow"]
            if not report_to:
                report_to = ["none"]

        # Base config kwargs
        config_kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": hp.per_device_train_batch_size,
            "gradient_accumulation_steps": hp.gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "lr_scheduler_type": hp.lr_scheduler_type,
            "warmup_ratio": hp.warmup_ratio,
            "bf16": hp.bf16,
            "fp16": hp.fp16,
            "gradient_checkpointing": hp.gradient_checkpointing,
            "logging_steps": hp.logging_steps,
            "save_steps": hp.save_steps,
            "weight_decay": hp.weight_decay,
            "optim": hp.optim,
            "report_to": report_to,
        }
        # Filter None
        config_kwargs = {k: v for k, v in config_kwargs.items() if v is not None}

        # Delegate strategy-specific config params
        strategy_config_kwargs = strategy.build_config_kwargs(hp)
        config_kwargs.update(strategy_config_kwargs)

        # Resolve reward plugin config kwargs before instantiating the TRL config.
        # reward_weights (and any future config-level plugin params) must reach the
        # config constructor directly — not via setattr after the fact.
        reward_result = None
        self._reward_plugin = None
        if strategy.requires_reward_plugin:
            if phase_config is None:
                raise ValueError(
                    f"{strategy_type.upper()} strategy requires explicit phase_config for reward plugin resolution"
                )
            reward_result = build_reward_plugin_result(
                train_dataset=train_dataset,
                phase_config=phase_config,
                pipeline_config=config,
            )
            self._reward_plugin = reward_result.plugin
            config_kwargs.update(reward_result.config_kwargs)

        # Enable evaluation only when eval_dataset is provided
        if eval_dataset is not None:
            # Default eval cadence must be compatible with save cadence when load_best_model_at_end=True.
            # If user overrides save_steps but not eval_steps, using a hard default (500) can crash:
            #   "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation steps"
            # So we default eval_steps to save_steps (or 500 if both are unset).
            eval_steps = (
                hp.eval_steps
                if hp.eval_steps is not None
                else (hp.save_steps if hp.save_steps is not None else DEFAULT_EVAL_SAVE_STEPS)
            )
            save_steps = hp.save_steps if hp.save_steps is not None else DEFAULT_EVAL_SAVE_STEPS

            # Ensure compatibility: save_steps must be a multiple of eval_steps.
            # If user provided an incompatible combination, auto-fix by rounding save_steps up.
            if eval_steps > 0 and save_steps % eval_steps != 0:
                adjusted_save_steps = ((save_steps // eval_steps) + 1) * eval_steps
                logger.warning(
                    "[TF:EVAL] save_steps=%s is not a multiple of eval_steps=%s while load_best_model_at_end=True. "
                    "Adjusting save_steps -> %s.",
                    save_steps,
                    eval_steps,
                    adjusted_save_steps,
                )
                save_steps = adjusted_save_steps
                config_kwargs["save_steps"] = save_steps
            # transformers/trl API compatibility:
            # - Newer versions renamed `evaluation_strategy` -> `eval_strategy`
            # - Some config classes accept **kwargs (tests/mocks)
            sig = inspect.signature(config_class.__init__)
            params = sig.parameters
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

            if "eval_strategy" in params or ("evaluation_strategy" not in params and has_varkw):
                config_kwargs["eval_strategy"] = "steps"
            elif "evaluation_strategy" in params:
                config_kwargs["evaluation_strategy"] = "steps"
            else:
                # If neither is supported, skip setting strategy to avoid crashing.
                logger.warning(
                    f"[TF:EVAL] Config class {getattr(config_class, '__name__', str(config_class))} "
                    "does not support eval_strategy/evaluation_strategy; skipping evaluation strategy param."
                )

            config_kwargs.update(
                {
                    "eval_steps": eval_steps,
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_loss",
                    "greater_is_better": False,
                }
            )

        # Instantiate Config
        training_config = config_class(**config_kwargs)

        logger.debug(f"[TF:CONFIG_CREATED] config={config_class.__name__}, lr={learning_rate}, epochs={num_epochs}")

        # 4. Build Trainer Kwargs
        trainer_kwargs = {
            "model": model,
            "args": training_config,
            "train_dataset": train_dataset,
            "processing_class": tokenizer,
        }

        # Add PEFT config (Common logic)
        # All strategies need adapters when using QLoRA/LoRA
        # Note: We could move this to strategy too, but it seems generic for the pipeline
        if hasattr(config.training, "type") and config.training.type in ("qlora", "lora", "adalora"):
            from src.training.trainer_builder import create_peft_config

            peft_config = create_peft_config(config)
            if peft_config is not None:
                trainer_kwargs["peft_config"] = peft_config

        if eval_dataset is not None:
            trainer_kwargs["eval_dataset"] = eval_dataset

        # Delegate strategy-specific trainer args (e.g. ref_model, reward_funcs)
        # Pass model and ref_model in kwargs context
        strategy_kwargs: dict[str, Any] = {
            "model": model,
            "ref_model": ref_model,
        }
        if strategy.requires_reference_dataset:
            strategy_kwargs["train_dataset"] = train_dataset

        # Apply any post-build config mutations (e.g. DPO PEFT adapter names)
        strategy.post_build_config_hook(training_config, model=model, ref_model=ref_model)

        strategy_trainer_kwargs = strategy.build_trainer_kwargs(
            training_config,
            **strategy_kwargs,
        )
        if reward_result is not None:
            strategy_trainer_kwargs.update(reward_result.trainer_kwargs)
        trainer_kwargs.update(strategy_trainer_kwargs)

        # 5. Add Callbacks (Common Logic)
        mlflow_config = config.experiment_tracking.mlflow if config.experiment_tracking else None
        callbacks = trainer_kwargs.get("callbacks", []) or []

        if mlflow_config:
            from src.training.callbacks import TrainingEventsCallback

            callbacks.append(TrainingEventsCallback(mlflow_manager=mlflow_manager))

            from src.training.callbacks.gpu_metrics_callback import GPUMetricsCallback

            callbacks.append(GPUMetricsCallback(mlflow_manager=mlflow_manager))

            if mlflow_config.system_metrics_callback_enabled:
                from src.training.callbacks import SystemMetricsCallback

                callback_interval = mlflow_config.system_metrics_callback_interval
                callbacks.append(SystemMetricsCallback(log_every_n_steps=callback_interval))

        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        # Merge extra kwargs
        trainer_kwargs.update(kwargs)

        # 6. Instantiate Trainer
        trainer = trainer_class(**trainer_kwargs)
        logger.info(f"✅ {trainer_class.__name__} created successfully")

        return trainer

    def create_from_phase(
        self,
        phase: StrategyPhaseConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        config: PipelineConfig,
        *,
        output_dir: str,
        mlflow_manager: IMLflowManager | None = None,
        **kwargs: Any,
    ) -> TrainerType:
        """
        Create trainer from a strategy phase configuration.
        """
        return self.create(
            strategy_type=phase.strategy_type,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            config=config,
            output_dir=output_dir,
            phase_config=phase,
            mlflow_manager=mlflow_manager,
            **kwargs,
        )

    @staticmethod
    def _merge_hyperparams(
        global_params: GlobalHyperparametersConfig, phase_params: PhaseHyperparametersConfig | None
    ) -> GlobalHyperparametersConfig:
        """Merge global hyperparameters with phase-specific overrides."""
        merged = global_params.model_copy()
        if phase_params:
            overrides = phase_params.model_dump(exclude_unset=True, exclude_none=True)
            for k, v in overrides.items():
                setattr(merged, k, v)
        return merged

    def get_trainer_class(self, strategy_type: str) -> type[TrainerType]:
        """Get the trainer class for a strategy type (via StrategyFactory)."""
        # Create dummy config just to get strategy instance
        # This is a bit awkward but Strategy requires config init.
        # Alternatively we could make get_trainer_class static in strategy but that breaks polymorphism if it depends on config?
        # Ideally get_trainer_class should be class method or we instantiate strategy.
        # For now, let's just error or Mock config?
        # Actually this method is rarely used outside tests or info.
        # Let's try to get it from registry directly if possible, or instantiate.
        # But Strategy init requires config.
        # Assuming we can't easily get it without config.
        # Let's raise NotImplementedError or try best effort.
        raise NotImplementedError("get_trainer_class is deprecated. Use StrategyFactory to get strategy instance.")

    def get_config_class(self, strategy_type: str) -> type[ConfigType]:
        raise NotImplementedError("get_config_class is deprecated.")

    @staticmethod
    def list_supported_strategies() -> list[str]:
        """List all supported strategy types."""
        return list(StrategyFactory().list_available().keys())

    @staticmethod
    def get_trainer_info(strategy_type: str) -> dict[str, str | bool]:
        """Get information about trainer for a strategy."""
        # This needs config to instantiate strategy.
        # Maybe we can just list from metadata?
        strategy_factory = StrategyFactory()
        if not strategy_factory.is_registered(strategy_type):
            raise ValueError(f"Unknown strategy: {strategy_type}")

        # We can't get class without instance easily if we strictly follow the new pattern.
        # But for info, we can look at metadata.
        return {"strategy_type": strategy_type, "info": "See StrategyMetadata"}


__all__ = ["TrainerFactory", "TrainerType"]

"""
Unified trainer builder - replaces training adapters.

Creates TRL trainers with PEFT config directly from configuration.
No adapter classes needed - TRL handles LoRA/QLoRA natively.

Key simplification:
- TRL SFTTrainer accepts peft_config parameter
- PEFT auto-applies LoRA to model
- No manual model.setup() or prepare_model_for_kbit_training()

Example:
    from src.training.trainer_builder import create_peft_config, create_trainer

    peft_config = create_peft_config(config)
    trainer = create_trainer(config, strategy, model, tokenizer, dataset, peft_config)
    trainer.train()
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from peft import AdaLoraConfig, LoraConfig, TaskType
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, ORPOConfig, ORPOTrainer, SFTConfig, SFTTrainer

from src.constants import (
    DEFAULT_LEARNING_RATES,
    STRATEGY_COT,
    STRATEGY_CPT,
    STRATEGY_DPO,
    STRATEGY_GRPO,
    STRATEGY_ORPO,
    STRATEGY_SAPO,
    STRATEGY_SFT,
)
from src.training.constants import (
    DEFAULT_MAX_COMPLETION_LENGTH,
    HP_MAX_COMPLETION_LENGTH,
    HP_MAX_LENGTH,
)
from src.training.reward_plugins import build_reward_plugin_result
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from src.training.strategies.base import TrainingStrategy
    from src.utils.config import PipelineConfig, StrategyPhaseConfig

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Strategy → Trainer / Config lookup tables
#
# These mappings are kept for backward-compatibility and informational use.
# New call-sites should use strategy.get_trainer_class() / strategy.get_config_class()
# and strategy.build_config_kwargs(hp) instead of reaching into these dicts.
# ---------------------------------------------------------------------------

STRATEGY_TRAINERS: MappingProxyType[str, type] = MappingProxyType(
    {
        STRATEGY_CPT: SFTTrainer,
        STRATEGY_SFT: SFTTrainer,
        STRATEGY_COT: SFTTrainer,
        STRATEGY_DPO: DPOTrainer,
        STRATEGY_ORPO: ORPOTrainer,
        STRATEGY_GRPO: GRPOTrainer,
        STRATEGY_SAPO: GRPOTrainer,
    }
)

STRATEGY_CONFIGS: MappingProxyType[str, type] = MappingProxyType(
    {
        STRATEGY_CPT: SFTConfig,
        STRATEGY_SFT: SFTConfig,
        STRATEGY_COT: SFTConfig,
        STRATEGY_DPO: DPOConfig,
        STRATEGY_ORPO: ORPOConfig,
        STRATEGY_GRPO: GRPOConfig,
        STRATEGY_SAPO: GRPOConfig,
    }
)


def create_peft_config(config: PipelineConfig) -> LoraConfig | AdaLoraConfig:
    """
    Create PEFT config from pipeline configuration.

    Supports:
    - type: "lora"    → LoraConfig (full precision + LoRA)
    - type: "qlora"   → LoraConfig (4-bit quantization + LoRA, same config)
    - type: "adalora" → AdaLoraConfig (adaptive rank allocation)

    Advanced LoRA variants (DoRA, rsLoRA) are enabled via flags in lora config.

    Args:
        config: Pipeline configuration

    Returns:
        LoraConfig or AdaLoraConfig for PEFT

    Example:
        peft_config = create_peft_config(config)
        trainer = SFTTrainer(model, ..., peft_config=peft_config)
    """
    training_type = config.training.type.lower()
    adapter_cfg = config.get_adapter_config()

    # Handle target_modules
    target_modules = adapter_cfg.target_modules
    if isinstance(target_modules, str) and target_modules == "all-linear":
        target_modules = "all-linear"
        logger.info("PEFT target_modules: all-linear (auto-detect)")
    elif isinstance(target_modules, list):
        logger.info(f"PEFT target_modules: {target_modules}")

    # AdaLoRA: Adaptive Low-Rank Adaptation
    if training_type == "adalora":
        from src.utils.config import AdaLoraConfig as AdaLoraConfigType

        if not isinstance(adapter_cfg, AdaLoraConfigType):
            raise ValueError("type='adalora' requires 'adalora:' section in config")

        logger.info(
            f"AdaLoRA config: init_r={adapter_cfg.init_r}, target_r={adapter_cfg.target_r}, "
            f"total_step={adapter_cfg.total_step}"
        )
        return AdaLoraConfig(
            init_r=adapter_cfg.init_r,
            target_r=adapter_cfg.target_r,
            total_step=adapter_cfg.total_step,
            tinit=adapter_cfg.tinit,
            tfinal=adapter_cfg.tfinal,
            deltaT=adapter_cfg.delta_t,
            beta1=adapter_cfg.beta1,
            beta2=adapter_cfg.beta2,
            lora_alpha=adapter_cfg.lora_alpha,
            lora_dropout=adapter_cfg.lora_dropout,
            bias=adapter_cfg.bias,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )

    # LoRA / QLoRA (both use LoraConfig, difference is in model quantization)
    from src.utils.config import LoraConfig as LoraConfigType

    if not isinstance(adapter_cfg, LoraConfigType):
        raise ValueError(f"type='{training_type}' requires 'lora:' section in config")

    init_lora_weights: Any
    raw_init = adapter_cfg.init_lora_weights
    # PEFT stubs are overly strict here (they don't model dynamic values like "pissa_niter_16"),
    # while runtime PEFT supports them. We keep runtime correctness and isolate `Any` to this param only.
    if raw_init in ("gaussian", "true"):
        init_lora_weights = True
    elif raw_init == "false":
        init_lora_weights = False
    else:
        init_lora_weights = raw_init

    peft_config = LoraConfig(
        r=adapter_cfg.r,
        lora_alpha=adapter_cfg.lora_alpha,
        lora_dropout=adapter_cfg.lora_dropout,
        bias=adapter_cfg.bias,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        # Advanced LoRA variants
        use_dora=adapter_cfg.use_dora,
        use_rslora=adapter_cfg.use_rslora,
        init_lora_weights=init_lora_weights,
    )

    # Log variant info
    variants = []
    if adapter_cfg.use_dora:
        variants.append("DoRA")
    if adapter_cfg.use_rslora:
        variants.append("rsLoRA")
    variant_str = f" ({', '.join(variants)})" if variants else ""

    logger.info(
        f"LoRA config created: r={adapter_cfg.r}, alpha={adapter_cfg.lora_alpha}, type={training_type}{variant_str}"
    )
    logger.debug(
        f"[TRAINER:PEFT] r={adapter_cfg.r}, alpha={adapter_cfg.lora_alpha}, dropout={adapter_cfg.lora_dropout}"
    )

    return peft_config


def create_training_args(
    config: PipelineConfig,
    strategy: StrategyPhaseConfig,
    *,
    output_dir: str | None = None,
    strategy_instance: TrainingStrategy | None = None,
    extra_config_kwargs: dict[str, Any] | None = None,
) -> Any:
    """
    Create TRL training arguments from config and strategy.

    Hyperparameter resolution:
    - Phase-specific (strategy.hyperparams) overrides Global (config.training.hyperparams)
    - Merge priority: Phase > Global

    When ``strategy_instance`` is provided, strategy-specific config kwargs are built
    by delegating to ``strategy_instance.build_config_kwargs(hp)`` — no if-chains.
    When omitted, falls back to the legacy STRATEGY_CONFIGS lookup + if-chains.

    Args:
        config: Pipeline configuration
        strategy: Current strategy phase config
        output_dir: Override output directory (defaults to ``output/phase_0_{type}``)
        strategy_instance: Optional pre-created TrainingStrategy instance. When present,
            used for config class lookup and strategy-specific kwargs instead of the
            static STRATEGY_CONFIGS / if-chains fallback path.

    Returns:
        TRL training config (SFTConfig, DPOConfig, ORPOConfig, GRPOConfig, or any
        config class returned by strategy_instance.get_config_class()).
    """
    strategy_type = strategy.strategy_type.lower()

    # Get optimizer based on training type
    optimizer = config.training.get_effective_optimizer()

    # Get experiment tracking reporters
    report_to = config.experiment_tracking.get_report_to()

    # Get effective hyperparameters (Phase Override > Global)
    def get_hp(name: str, default: Any = None) -> Any:
        """Phase > Global > Default."""
        val = getattr(strategy.hyperparams, name, None)
        if val is not None:
            return val
        val = getattr(config.training.hyperparams, name, None)
        if val is not None:
            return val
        return default

    learning_rate = get_hp("learning_rate")
    num_epochs = get_hp("epochs")
    per_device_train_batch_size = get_hp("per_device_train_batch_size")
    gradient_accumulation_steps = get_hp("gradient_accumulation_steps")
    warmup_ratio = get_hp("warmup_ratio")

    if output_dir is None:
        output_dir = f"output/phase_0_{strategy_type}"

    args: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lr_scheduler_type": get_hp("lr_scheduler_type"),
        "warmup_ratio": warmup_ratio,
        "weight_decay": get_hp("weight_decay"),
        "optim": optimizer,
        "bf16": get_hp("bf16"),
        "fp16": get_hp("fp16"),
        "gradient_checkpointing": get_hp("gradient_checkpointing"),
        "logging_steps": get_hp("logging_steps"),
        "save_steps": get_hp("save_steps"),
        "report_to": report_to,
        "include_num_input_tokens_seen": True,
        "logging_nan_inf_filter": False,
    }

    if strategy_instance is not None:
        # Clean path: delegate strategy-specific kwargs to the strategy object.
        config_class = strategy_instance.get_config_class()

        # Synthesise a lightweight hp proxy from get_hp so build_config_kwargs works
        # with the same phase-priority semantics.
        class _HpProxy:
            def __getattr__(self, name: str) -> Any:
                return get_hp(name)

        strategy_specific = strategy_instance.build_config_kwargs(_HpProxy())
        # Remove duplicates: base args take precedence for the common fields;
        # strategy_specific fills in strategy-only fields.
        for k, v in strategy_specific.items():
            if k not in args or v is not None:
                args[k] = v
    else:
        # Legacy fallback: static lookup + if-chains.
        config_class = STRATEGY_CONFIGS[strategy_type]

        if strategy_type in (STRATEGY_CPT, STRATEGY_SFT, STRATEGY_COT):
            args["packing"] = get_hp("packing")
            args[HP_MAX_LENGTH] = get_hp(HP_MAX_LENGTH)

        if strategy_type in (STRATEGY_DPO, STRATEGY_ORPO):
            args["beta"] = get_hp("beta", 0.1)
            args[HP_MAX_LENGTH] = get_hp(HP_MAX_LENGTH)

        if strategy_type in (STRATEGY_GRPO, STRATEGY_SAPO):
            args["loss_type"] = strategy_type
            args["num_generations"] = get_hp("num_generations", 4)
            args["max_prompt_length"] = get_hp("max_prompt_length")
            args[HP_MAX_COMPLETION_LENGTH] = get_hp(HP_MAX_COMPLETION_LENGTH, DEFAULT_MAX_COMPLETION_LENGTH)
            if strategy_type == STRATEGY_SAPO:
                args["sapo_temperature_pos"] = get_hp("sapo_temperature_pos", 1.0)
                args["sapo_temperature_neg"] = get_hp("sapo_temperature_neg", 1.0)

    if extra_config_kwargs:
        args.update(extra_config_kwargs)

    training_args = config_class(**args)

    logger.debug(
        "[TRAINER:ARGS] strategy=%s, lr=%s, epochs=%s, optimizer=%s",
        strategy_type,
        learning_rate,
        num_epochs,
        optimizer,
    )
    logger.info("Experiment tracking: %s", report_to)

    return training_args


def create_trainer(
    config: PipelineConfig,
    strategy: StrategyPhaseConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    peft_config: LoraConfig | None,
    eval_dataset: Dataset | None = None,
    *,
    strategy_instance: TrainingStrategy | None = None,
) -> Any:
    """
    Create TRL trainer for a strategy phase.

    TRL automatically applies PEFT adapters when peft_config is provided.

    When ``strategy_instance`` is provided, trainer class resolution and
    reward-plugin detection are delegated to the strategy object — no if-chains.
    When omitted, falls back to the legacy STRATEGY_TRAINERS lookup.

    Args:
        config: Pipeline configuration
        strategy: Current strategy phase config
        model: Base model (PEFT applied by TRL if peft_config is set)
        tokenizer: Tokenizer
        train_dataset: Training dataset
        peft_config: Optional PEFT config (LoRA / QLoRA / AdaLoRA)
        eval_dataset: Optional evaluation dataset
        strategy_instance: Optional pre-created TrainingStrategy. When present,
            ``get_trainer_class()`` and ``requires_reward_plugin`` are used
            instead of the static STRATEGY_TRAINERS / if-chains fallback.

    Returns:
        Configured TRL trainer
    """
    strategy_type = strategy.strategy_type.lower()

    if strategy_instance is not None:
        trainer_class = strategy_instance.get_trainer_class()
    else:
        trainer_class = STRATEGY_TRAINERS[strategy_type]

    needs_reward_plugin = (
        strategy_instance.requires_reward_plugin
        if strategy_instance is not None
        else strategy_type in (STRATEGY_GRPO, STRATEGY_SAPO)
    )

    reward_result = None
    if needs_reward_plugin:
        reward_result = build_reward_plugin_result(
            train_dataset=train_dataset,
            phase_config=strategy,
            pipeline_config=config,
        )
        logger.info("[TRAINER:%s] Using configured reward plugin", strategy_type.upper())

    training_args = create_training_args(
        config,
        strategy,
        strategy_instance=strategy_instance,
        extra_config_kwargs=reward_result.config_kwargs if reward_result is not None else None,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
    }

    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    if reward_result is not None:
        trainer_kwargs.update(reward_result.trainer_kwargs)
        logger.info("[TRAINER:%s] Reward funcs applied", strategy_type.upper())

    trainer = trainer_class(**trainer_kwargs)

    logger.info("Created %s for %s", trainer_class.__name__, strategy_type)
    logger.debug("[TRAINER:CREATED] class=%s, strategy=%s", trainer_class.__name__, strategy_type)

    return trainer


__all__ = [
    "DEFAULT_LEARNING_RATES",
    "STRATEGY_CONFIGS",
    "STRATEGY_TRAINERS",
    "create_peft_config",
    "create_trainer",
    "create_training_args",
]

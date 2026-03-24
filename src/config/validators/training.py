from __future__ import annotations

from typing import TYPE_CHECKING

from src.constants import STRATEGY_GRPO, STRATEGY_SAPO

if TYPE_CHECKING:
    from ..training.lora.lora import LoraConfig
    from ..training.schema import TrainingOnlyConfig
    from ..training.strategies import StrategyPhaseConfig


def validate_training_adalora_requires_block(cfg: TrainingOnlyConfig) -> None:
    """Cross-field rules for TrainingOnlyConfig adapter blocks."""

    # If training.type == 'adalora' → require explicit training.adalora block.
    # Fail-fast: no auto-creation of missing blocks.
    if cfg.type == "adalora" and cfg.adalora is None:
        raise ValueError("training.type='adalora' requires 'training.adalora:' section in config")


def validate_lora_config(cfg: LoraConfig) -> None:
    """Cross-field rules for training.lora (LoraConfig)."""

    # DoRA incompatibilities (from PEFT documentation)
    if cfg.use_dora:
        # LoftQ does not work with DoRA
        if cfg.init_lora_weights == "loftq":
            raise ValueError(
                "use_dora=True is incompatible with init_lora_weights='loftq'. "
                "LoftQ initialization does not currently work with DoRA."
            )

        # PiSSA has its own initialization logic, not recommended with DoRA
        if cfg.init_lora_weights == "pissa" or cfg.init_lora_weights.startswith("pissa_niter_"):
            # Local import to avoid heavy side-effects at module import time.
            from src.utils.logger import logger

            logger.warning(
                "[CFG:LORA_WARNING] use_dora=True with init_lora_weights='pissa' is experimental. "
                "PiSSA has its own initialization logic that may conflict with DoRA."
            )


def validate_strategy_phase_config(cfg: StrategyPhaseConfig) -> None:
    """Cross-field rules for a single training strategy phase (StrategyPhaseConfig)."""

    strategy_type = cfg.strategy_type.lower()

    # GRPO-family strategies require explicit prompt/completion limits.
    if strategy_type in {STRATEGY_GRPO, STRATEGY_SAPO}:
        if cfg.hyperparams.max_prompt_length is None:
            raise ValueError(
                f"{strategy_type.upper()} strategy requires hyperparams.max_prompt_length.\n"
                "Example: max_prompt_length: 1024"
            )
        if cfg.hyperparams.max_completion_length is None:
            raise ValueError(
                f"{strategy_type.upper()} strategy requires hyperparams.max_completion_length.\n"
                "Example: max_completion_length: 512"
            )
        params = getattr(cfg, "params", {}) or {}
        reward_plugin = params.get("reward_plugin") if isinstance(params, dict) else None
        if not reward_plugin:
            raise ValueError(
                f"{strategy_type.upper()} strategy requires params.reward_plugin.\n"
                "Example: params: {reward_plugin: helixql_compiler_semantic}"
            )


__all__ = [
    "validate_lora_config",
    "validate_strategy_phase_config",
    "validate_training_adalora_requires_block",
]

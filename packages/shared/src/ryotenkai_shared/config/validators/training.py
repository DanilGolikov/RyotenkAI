from __future__ import annotations

from typing import TYPE_CHECKING

from ryotenkai_shared.constants import STRATEGY_GRPO, STRATEGY_SAPO

if TYPE_CHECKING:
    from ..training.lora.lora import LoraConfig
    from ..training.schema import TrainingOnlyConfig
    from ..training.strategies import StrategyPhaseConfig


def validate_training_adapter_requires_block(cfg: TrainingOnlyConfig) -> None:
    """Precision-consistency validator (post-discriminated-unions).

    The legacy "type=X requires X block" rules are now redundant — Pydantic's
    discriminated union enforces structural correctness at YAML load
    (kind=qlora MUST be a QloraConfig payload, etc.).

    What remains is the precision-conflict check: fp16 AMP cannot run with
    a bfloat16 LoRA compute path. We keep this as a model-level cross-check
    because it spans hyperparams + adapter — no schema-level place to put it.

    Backward-compat: name preserved so any external scripts that import this
    helper keep working until PR-9.
    """
    _validate_precision_consistency(cfg)


def _validate_precision_consistency(cfg: TrainingOnlyConfig) -> None:
    """Detect fp16 + adapter conflicts.

    TRL's SFTTrainer creates LoRA parameters in the model's *native* dtype
    (usually bfloat16 for modern models like Qwen, Llama 3, etc.).
    PyTorch GradScaler (fp16 AMP) crashes on bfloat16 gradients with:
        "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'

    Two cases:
    1. QLoRA with explicit bnb_4bit_compute_dtype=bfloat16 + fp16=True ⇒ raise.
    2. Any adapter kind with fp16=True ⇒ warning (bf16=True is recommended).
    """
    hp = getattr(cfg, "hyperparams", None)
    if hp is None:
        return
    uses_fp16 = getattr(hp, "fp16", False) or False
    if not uses_fp16:
        return

    adapter = cfg.adapter
    if adapter.kind == "qlora":
        compute_dtype = getattr(adapter, "bnb_4bit_compute_dtype", "bfloat16")
        if compute_dtype == "bfloat16":
            raise ValueError(
                "Precision conflict: hyperparams.fp16=true with bnb_4bit_compute_dtype='bfloat16'. "
                "GradScaler (fp16 AMP) cannot operate on BFloat16 tensors. "
                "Fix: use bf16: true instead of fp16: true."
            )

    from ryotenkai_shared.utils.logger import logger

    logger.warning(
        "[CFG:PRECISION] fp16=true with %s adapter is risky: TRL/SFTTrainer creates "
        "LoRA params in the model's native dtype (often bfloat16), which crashes "
        "GradScaler. Recommended: use bf16: true instead of fp16: true.",
        adapter.kind,
    )


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
            from ryotenkai_shared.utils.logger import logger

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

    # Adapter cache validation.
    cache = getattr(cfg, "adapter_cache", None)
    if cache is not None and getattr(cache, "enabled", False):
        if not cache.repo_id:
            raise ValueError(
                "adapter_cache.repo_id is required when adapter_cache.enabled=true.\n"
                "Example: adapter_cache:\n  enabled: true\n  repo_id: org/my-adapters-cache"
            )
        if not cfg.dataset:
            raise ValueError(
                "adapter_cache.enabled=true requires dataset to be set.\n"
                "The dataset is used to compute a fingerprint for cache invalidation.\n"
                "Example: dataset: sft_data"
            )


__all__ = [
    "validate_lora_config",
    "validate_strategy_phase_config",
    "validate_training_adapter_requires_block",
]

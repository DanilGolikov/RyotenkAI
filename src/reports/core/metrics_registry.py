"""
Registry for metric definitions and strategy profiles.

Centralizes knowledge about:
1. What metrics exist and what they mean (descriptions).
2. Which metrics are relevant for which strategy (profiles).
"""

from types import MappingProxyType
from typing import Final

# ============================================================================
# METRIC DESCRIPTIONS (detailed)
# ============================================================================

METRIC_DESCRIPTIONS: Final[MappingProxyType[str, tuple[str, str]]] = MappingProxyType(
    {
        # --- Universal Metrics ---
        "train_loss": (
            "Training Loss",
            "Loss on the training set. Shows how wrong the model is on the data it currently sees. "
            "It should decrease steadily. If it does not, training is not progressing. "
            "If it spikes, the learning rate is likely too high.",
        ),
        "eval_loss": (
            "Eval Loss",
            "Loss on the validation (evaluation) set. Tracks quality on data the model does not use "
            "to update weights. If train_loss falls while eval_loss rises, overfitting is likely.",
        ),
        "mean_token_accuracy": (
            "Token Accuracy",
            "Accuracy of next-token prediction. In SFT tasks it should increase. "
            "Reflects the fraction of cases where the model predicted the token correctly.",
        ),
        "entropy": (
            "Entropy",
            "Entropy of the probability distribution—a measure of uncertainty. "
            "Decreasing entropy usually means the model is more confident in its outputs.",
        ),
        "loss": (
            "Loss",
            "Overall loss (may include regularization). It should decrease. Sharp spikes "
            "indicate unstable training (e.g., a bad data batch).",
        ),
        "learning_rate": (
            "Learning Rate",
            "Current learning rate, controlled by the scheduler. Typical pattern: warmup → peak → gradual decay.",
        ),
        "grad_norm": (
            "Gradient Norm",
            "L2 norm of gradients across all parameters—shows the strength of weight updates. "
            "Spikes >10× suggest instability (gradient explosion). Very small values (<0.01) may indicate vanishing gradients.",
        ),
        "train_samples_per_second": (
            "Train Samples/s",
            "Training samples processed per second—a throughput metric. Helps spot regressions "
            "(I/O bottleneck, throttling, slow dataloader). Sharp drops may indicate data or hardware issues.",
        ),
        "epoch": (
            "Epoch",
            "Current training epoch (how many full passes over the dataset). Useful for correlating "
            "metrics with training progress and comparing runs.",
        ),
        "total_flos": (
            "Total FLOPs",
            "Estimated total FLOPs spent on training—used to assess compute cost; the value "
            "usually increases monotonically during training.",
        ),
        # --- DPO/ORPO Metrics ---
        "rewards/accuracies": (
            "Reward Accuracy",
            "Fraction of cases where the model preferred the chosen response over the rejected one. "
            "For DPO/ORPO this is the main quality signal. Ideal is typically > 0.6–0.7.",
        ),
        "rewards/margins": (
            "Reward Margin",
            "Difference in log-probabilities between chosen and rejected. Should increase "
            "(the model confidently prefers the better response).",
        ),
        "logps/chosen": (
            "LogP (Chosen)",
            "Log-probability of the preferred response. Often decreases as the model adapts to the reference, "
            "but margin should still increase.",
        ),
        "logps/rejected": (
            "LogP (Rejected)",
            "Log-probability of the dispreferred response. Should fall faster than chosen.",
        ),
        # --- SAPO / GRPO Metrics ---
        "reward": (
            "Reward",
            "Average reward per generation—the main RL metric. Should increase.",
        ),
        "reward_std": (
            "Reward Std",
            "Standard deviation of reward—reflects variability in response quality.",
        ),
        "kl": (
            "KL Divergence",
            "Divergence between the trainable model and the reference model. "
            "Too rapid growth suggests mode collapse (the model drifts away from fluent language).",
        ),
        "completion_length": (
            "Completion Length",
            "Average length of generated completions—an informative auxiliary metric.",
        ),
    }
)

# ==============================================================================
# STRATEGY PROFILES
# ============================================================================

# Default metrics for any unknown strategy
DEFAULT_METRICS: Final[tuple[str, ...]] = (
    "loss",
    "train_loss",
    "eval_loss",
    "learning_rate",
    "grad_norm",
    "train_samples_per_second",
    "epoch",
)

# Metrics specific to SFT/CPT/CoT
SFT_METRICS: Final[tuple[str, ...]] = (
    *DEFAULT_METRICS,
    "mean_token_accuracy",
    "entropy",
)

# Metrics specific to DPO/ORPO (Preference Learning)
DPO_METRICS: Final[tuple[str, ...]] = (
    *DEFAULT_METRICS,
    "rewards/accuracies",
    "rewards/margins",
    "logps/chosen",
    "logps/rejected",
)

# Metrics specific to SAPO/GRPO (Online RL)
SAPO_METRICS: Final[tuple[str, ...]] = (
    *DEFAULT_METRICS,
    "reward",  # Often just 'reward' or 'rewards/mean'
    "reward_std",
    "kl",  # KL divergence
    "completion_length",
)

# Mapping strategy name -> metric list
STRATEGY_METRIC_PROFILES: Final[MappingProxyType[str, tuple[str, ...]]] = MappingProxyType(
    {
        "sft": SFT_METRICS,
        "cpt": SFT_METRICS,  # CPT is basically SFT on raw text
        "cot": SFT_METRICS,  # CoT is SFT with specific data
        "dpo": DPO_METRICS,
        "orpo": DPO_METRICS,
        "sapo": SAPO_METRICS,
        # Aliases if needed
        "ppo": SAPO_METRICS,  # PPO is similar to SAPO in metrics
        "grpo": SAPO_METRICS,
    }
)


def get_metrics_for_strategy(strategy_type: str) -> tuple[str, ...]:
    """Get list of metrics to analyze for a given strategy."""
    return STRATEGY_METRIC_PROFILES.get(strategy_type.lower(), DEFAULT_METRICS)

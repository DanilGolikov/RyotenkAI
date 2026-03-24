"""
Project-wide constants, literals and defaults.

Goal:
- Eliminate magic strings/numbers scattered across the codebase
- Keep cross-cutting identifiers (provider names, engine names, filenames) in ONE place

Notes:
- Keep this module dependency-free (no imports from project packages) to avoid circular imports.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Literal

# ---------------------------------------------------------------------------
# Provider keys (YAML + factories + runtime identifiers)
# ---------------------------------------------------------------------------

type ProviderSingleNodeName = Literal["single_node"]
type ProviderRunPodName = Literal["runpod"]

PROVIDER_SINGLE_NODE: Final[ProviderSingleNodeName] = "single_node"
PROVIDER_RUNPOD: Final[ProviderRunPodName] = "runpod"

type TrainingProviderName = ProviderSingleNodeName | ProviderRunPodName
type InferenceProviderName = ProviderSingleNodeName | ProviderRunPodName

# Providers implemented in runtime code today (feature flags may still gate usage).
SUPPORTED_INFERENCE_PROVIDERS: Final[tuple[InferenceProviderName, ...]] = (
    PROVIDER_SINGLE_NODE,
    PROVIDER_RUNPOD,
)

# ---------------------------------------------------------------------------
# Inference engines
# ---------------------------------------------------------------------------

type InferenceEngineVllm = Literal["vllm"]

INFERENCE_ENGINE_VLLM: Final[InferenceEngineVllm] = "vllm"

type InferenceEngineName = InferenceEngineVllm

# Engines implemented for inference deployment today.
SUPPORTED_INFERENCE_ENGINES: Final[tuple[InferenceEngineName, ...]] = (INFERENCE_ENGINE_VLLM,)

# ---------------------------------------------------------------------------
# Training strategy types (YAML + factory + trainer_builder)
# ---------------------------------------------------------------------------

STRATEGY_CPT: Final[str] = "cpt"
STRATEGY_SFT: Final[str] = "sft"
STRATEGY_COT: Final[str] = "cot"
STRATEGY_DPO: Final[str] = "dpo"
STRATEGY_ORPO: Final[str] = "orpo"
STRATEGY_GRPO: Final[str] = "grpo"
STRATEGY_SAPO: Final[str] = "sapo"

ALL_STRATEGIES: Final[tuple[str, ...]] = (
    STRATEGY_CPT,
    STRATEGY_SFT,
    STRATEGY_COT,
    STRATEGY_DPO,
    STRATEGY_ORPO,
    STRATEGY_GRPO,
    STRATEGY_SAPO,
)

# Strategy human-readable descriptions
STRATEGY_DESCRIPTIONS: MappingProxyType[str, str] = MappingProxyType(
    {
        STRATEGY_CPT: "Continual Pre-Training for domain adaptation",
        STRATEGY_SFT: "Supervised Fine-Tuning for instruction following",
        STRATEGY_COT: "Chain-of-Thought for reasoning abilities",
        STRATEGY_DPO: "Direct Preference Optimization for alignment",
        STRATEGY_ORPO: "Odds Ratio Preference Optimization (combined SFT+DPO)",
        STRATEGY_GRPO: "Group Relative Policy Optimization (Online RL)",
        STRATEGY_SAPO: "Soft Adaptive Policy Optimization (Online RL via GRPO)",
    }
)

# Default learning rates per strategy — research-backed (Nature 2025 / HuggingFace TRL)
DEFAULT_LEARNING_RATES: MappingProxyType[str, float] = MappingProxyType(
    {
        STRATEGY_CPT: 1e-5,
        STRATEGY_SFT: 2e-4,
        STRATEGY_COT: 2e-5,
        STRATEGY_DPO: 5e-6,
        STRATEGY_ORPO: 1e-5,
        STRATEGY_GRPO: 2e-4,
        STRATEGY_SAPO: 2e-4,
    }
)

# Default epochs per strategy
DEFAULT_EPOCHS: MappingProxyType[str, int] = MappingProxyType(
    {
        STRATEGY_CPT: 1,
        STRATEGY_SFT: 3,
        STRATEGY_COT: 2,
        STRATEGY_DPO: 1,
        STRATEGY_ORPO: 2,
        STRATEGY_GRPO: 3,
        STRATEGY_SAPO: 3,
    }
)

# Default batch sizes per strategy
DEFAULT_BATCH_SIZES: MappingProxyType[str, int] = MappingProxyType(
    {
        STRATEGY_CPT: 8,
        STRATEGY_SFT: 4,
        STRATEGY_COT: 2,
        STRATEGY_DPO: 4,
        STRATEGY_ORPO: 4,
        STRATEGY_GRPO: 4,
        STRATEGY_SAPO: 4,
    }
)

# ---------------------------------------------------------------------------
# Inference deployer: generated artifacts
# ---------------------------------------------------------------------------

INFERENCE_DIRNAME: Final[str] = "inference"
INFERENCE_MANIFEST_FILENAME: Final[str] = "inference_manifest.json"
INFERENCE_CHAT_SCRIPT_FILENAME: Final[str] = "chat_inference.py"
INFERENCE_README_FILENAME: Final[str] = "inference_README.md"

# Single-node inference: container naming (used by provider + scripts)
VLLM_INFERENCE_CONTAINER_NAME: Final[str] = "ryotenkai-inference-vllm"

# ---------------------------------------------------------------------------
# Shared timeouts, intervals, display (used by pipeline + config)
# ---------------------------------------------------------------------------

SSH_PORT_DEFAULT: Final[int] = 22
SSH_CMD_TIMEOUT: Final[int] = 30  # Timeout for SSH command execution (seconds)
ERROR_MESSAGE_TRUNCATE: Final[int] = 200  # Max chars for error messages in logs
CONSOLE_LINE_WIDTH: Final[int] = 80  # Width for console separators/boxes
TRAINING_START_TIMEOUT_DEFAULT: Final[int] = 120  # Default timeout waiting for training to start
LOG_DOWNLOAD_INTERVAL_DEFAULT: Final[int] = 30  # Default interval for downloading training logs (seconds)

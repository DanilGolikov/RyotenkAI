"""
Project-wide constants, literals and defaults.

Goal:
- Eliminate magic strings/numbers scattered across the codebase
- Keep cross-cutting identifiers (provider names, engine names, filenames) in ONE place

Notes:
- Keep this module dependency-free (no imports from project packages) to avoid circular imports.
"""

from __future__ import annotations

import os
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

# Phase 14.A — runtime-side provider identity contract.
#
# The Mac launcher writes this env var into the JobSpec (value =
# ``provider.provider_name``) so the in-pod runner's bootstrap
# registry can pick the right :class:`IPodLifecycleClient` impl
# without guessing from the presence/absence of provider-specific
# vars (e.g. ``RUNPOD_API_KEY``). Single source of truth for
# "which provider booted me?" inside the runner.
#
# Phase 14.B is what wires the runner-side registry to read this;
# Phase 14.A only adds the constant + has each provider populate it
# in :meth:`required_runtime_env_vars`.
RUNTIME_PROVIDER_ENV_VAR: Final[str] = "RYOTENKAI_RUNTIME_PROVIDER"

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
# PR-B: was 30s pre-2026-05-02. Lowered to 5s so a trainer that crashes
# at T+11s (the 15-crash incident) gets at least one mid-flight pull
# before postmortem. Cost is negligible: ``stat -c%s`` on the remote
# is ~10ms; 12×/min × 60min = ~720 calls aggregating ~7s of SSH per
# hour. PR-B's push-tail in ``trainer_exited`` covers the case where
# even 5s is too long (sub-5-sec crashes still get the tail via WS).
LOG_DOWNLOAD_INTERVAL_DEFAULT: Final[int] = 5  # Default interval for downloading training logs (seconds)

# ---------------------------------------------------------------------------
# HuggingFace Hub upload policy (shared: adapter_cache + model_retriever)
# ---------------------------------------------------------------------------

HF_UPLOAD_RETRIES: Final[int] = 3
HF_UPLOAD_RETRY_DELAY_S: Final[int] = 10
# Per-attempt ceiling; matches legacy MR_UPLOAD_TIMEOUT in pipeline/constants.py.
HF_UPLOAD_TIMEOUT_S: Final[int] = 1800

# Files that constitute a LoRA adapter checkpoint.
# Used by adapter_cache (allow_patterns) and model_retriever (copy_cmds).
LORA_CHECKPOINT_PATTERNS: Final[tuple[str, ...]] = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "config.json",
)


# ---------------------------------------------------------------------------
# Runtime image — single source of truth for the docker image the Mac
# control plane provisions on RunPod / single_node hosts.
#
# Lived in ``src.runner.__about__`` until Phase A.4 of monorepo
# packagization (plan §A.4); kept here so pipeline-side dependency
# installers stop reaching across the runner boundary (root cause of
# the ``pipeline → runner`` cycle, plan §2.4-C).
#
# Override via ``RYOTENKAI_RUNTIME_IMAGE_OVERRIDE`` env, intended for
# CI smoke tests / dev iteration only — not a user-facing config.
#
# Image semver:
#   * v1.x — baked-in ``src/`` baseline at ``/opt/ryotenkai``;
#     retired but kept on Docker Hub for emergency rollback via
#     ``RYOTENKAI_RUNTIME_IMAGE_OVERRIDE``.
#   * v2.x — thin image (env-only): no ``src/`` in the image; the
#     Mac control plane rsyncs ``src/runner`` and its deps into the
#     run-scoped workspace, then SSH-execs uvicorn from there.
#     Wire-incompatible with v1.x clients.
# ---------------------------------------------------------------------------

# Bumped in lock-step with the docker image published by
# ``docker/training/build_and_push.sh``. Repo path
# ``${DOCKER_USERNAME}/ryotenkai-training-runtime`` resolves to the
# doubled-prefix path below — kept as-is to match the publish script.
_DEFAULT_RUNTIME_IMAGE: Final[str] = (
    "ryotenkai/ryotenkai-training-runtime:v0.1.1"
)


def _resolve_runtime_image() -> str:
    override = os.environ.get("RYOTENKAI_RUNTIME_IMAGE_OVERRIDE", "").strip()
    return override or _DEFAULT_RUNTIME_IMAGE


RUNTIME_IMAGE: Final[str] = _resolve_runtime_image()

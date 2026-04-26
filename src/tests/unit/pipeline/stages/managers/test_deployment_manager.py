"""
Unit tests for TrainingDeploymentManager (deployment_manager.py).

Focus:
- Result semantics (Ok/Err) and branching
- Regressions: multi-dataset upload from config.datasets
- Invariants: remote config path is fixed to config/pipeline_config.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.config.integrations.mlflow import MLflowTrackingRef
from src.pipeline.stages.managers.deployment_manager import TrainingDeploymentManager
from src.utils.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    ExperimentTrackingConfig,
    GlobalHyperparametersConfig,
    InferenceConfig,
    InferenceEnginesConfig,
    InferenceVLLMEngineConfig,
    ModelConfig,
    PipelineConfig,
    QLoRAConfig,
    TrainingOnlyConfig,
)

pytestmark = pytest.mark.unit

# NOTE:
# Unit tests must not depend on repo-root `data/` or `config/` folders.
# Keep all test fixtures under `src/tests/fixtures`.
DATASET_CHAT_FIXTURE = "src/tests/fixtures/datasets/test_chat.jsonl"
DATASET_INSTRUCTION_FIXTURE = "src/tests/fixtures/datasets/test_instruction.jsonl"
CONFIG_FIXTURE = "src/tests/fixtures/configs/test_pipeline.yaml"

# Minimal valid provider config for training.provider='single_node'.
SINGLE_NODE_PROVIDER_CFG: dict[str, Any] = {
    "connect": {"ssh": {"alias": "pc"}},
    "training": {"workspace_path": "/tmp/workspace", "docker_image": "test/training-runtime:latest"},
}

RUNPOD_PROVIDER_CFG: dict[str, Any] = {
    "connect": {"ssh": {"key_path": "/tmp/id_ed25519"}},
    "cleanup": {},
    "training": {"image_name": "test/training-runtime:latest"},
    "inference": {},
}


def _mk_experiment_tracking() -> ExperimentTrackingConfig:
    return ExperimentTrackingConfig(
        mlflow=MLflowTrackingRef(
            integration="mlflow-test",
            experiment_name="test-exp",
        )
    )


@dataclass(frozen=True)
class DummySecrets:
    """Minimal Secrets-like object for TrainingDeploymentManager."""

    hf_token: str = "hf_test_token"


@pytest.fixture
def secrets() -> DummySecrets:
    return DummySecrets(hf_token="hf_test_token")


@pytest.fixture
def base_config() -> PipelineConfig:
    return PipelineConfig(
        model=ModelConfig(name="gpt2", torch_dtype="bfloat16", trust_remote_code=False),
        providers={"single_node": SINGLE_NODE_PROVIDER_CFG},
        training=TrainingOnlyConfig(
            provider="single_node",
            type="qlora",
            qlora=QLoRAConfig(
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
        ),
        datasets={
            "default": DatasetConfig(
                source_type="local",
                source_local=DatasetSourceLocal(local_paths=DatasetLocalPaths(train=DATASET_CHAT_FIXTURE, eval=None)),
            )
        },
        inference=InferenceConfig(
            enabled=False,
            provider="single_node",
            engine="vllm",
            engines=InferenceEnginesConfig(
                vllm=InferenceVLLMEngineConfig(
                    merge_image="test/merge:latest",
                    serve_image="test/vllm:latest",
                )
            ),
        ),
        experiment_tracking=_mk_experiment_tracking(),
    )


@pytest.fixture
def config_multi_dataset() -> PipelineConfig:
    """
    Config with multiple local dataset paths (to cover multi-dataset logic).
    """

    dataset_a = DATASET_CHAT_FIXTURE
    dataset_b = DATASET_INSTRUCTION_FIXTURE
    assert Path(dataset_a).exists(), f"Expected fixture dataset to exist: {dataset_a}"
    assert Path(dataset_b).exists(), f"Expected fixture dataset to exist: {dataset_b}"

    return PipelineConfig(
        model=ModelConfig(name="gpt2", torch_dtype="bfloat16", trust_remote_code=False),
        providers={"single_node": SINGLE_NODE_PROVIDER_CFG},
        training=TrainingOnlyConfig(
            provider="single_node",
            type="qlora",
            qlora=QLoRAConfig(
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
            strategies=[
                # Ensure BOTH datasets are referenced -> must be uploaded
                # (deploy_files uploads datasets used by strategies)
                {"strategy_type": "sft", "dataset": "default"},
                {"strategy_type": "dpo", "dataset": "secondary"},
            ],
        ),
        datasets={
            "default": DatasetConfig(
                source_type="local",
                source_local={
                    "local_paths": {"train": dataset_a, "eval": None},
                },
            ),
            "secondary": DatasetConfig(
                source_type="local",
                source_local={
                    "local_paths": {"train": dataset_b, "eval": None},
                },
            ),
        },
        inference=InferenceConfig(
            enabled=False,
            provider="single_node",
            engine="vllm",
            engines=InferenceEnginesConfig(
                vllm=InferenceVLLMEngineConfig(
                    merge_image="test/merge:latest",
                    serve_image="test/vllm:latest",
                )
            ),
        ),
        experiment_tracking=_mk_experiment_tracking(),
    )


@pytest.fixture
def manager(base_config: PipelineConfig, secrets: DummySecrets) -> TrainingDeploymentManager:
    return TrainingDeploymentManager(config=base_config, secrets=secrets)


def test_set_workspace_sets_workspace(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace/run_123")
    assert manager._workspace == "/workspace/run_123"

    manager.set_workspace(workspace_path="/workspace/run_456")
    assert manager._workspace == "/workspace/run_456"



"""Facade smoke tests for TrainingDeploymentManager.

After Wave 3 decomposition the deployment concern is split into four
components living under ``deployment/``. This file keeps only smoke
tests for the facade itself: instantiation wiring, workspace
propagation, and that the public API delegates to the right
components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.stages.managers.deployment.code_syncer import CodeSyncer
from src.pipeline.stages.managers.deployment.dependency_installer import DependencyInstaller
from src.pipeline.stages.managers.deployment.file_uploader import FileUploader
from src.pipeline.stages.managers.deployment.training_launcher import TrainingLauncher
from src.pipeline.stages.managers.deployment_manager import TrainingDeploymentManager
from src.utils.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    GlobalHyperparametersConfig,
    InferenceConfig,
    InferenceEnginesConfig,
    InferenceVLLMEngineConfig,
    ModelConfig,
    PipelineConfig,
    QLoRAConfig,
    TrainingOnlyConfig,
)
from src.utils.result import Ok

pytestmark = pytest.mark.unit


DATASET_CHAT_FIXTURE = "src/tests/fixtures/datasets/test_chat.jsonl"

SINGLE_NODE_PROVIDER_CFG: dict[str, Any] = {
    "connect": {"ssh": {"alias": "pc"}},
    "training": {"workspace_path": "/tmp/workspace", "docker_image": "test/training-runtime:latest"},
}


@dataclass(frozen=True)
class DummySecrets:
    hf_token: str = "hf_test_token"


@pytest.fixture
def secrets() -> DummySecrets:
    return DummySecrets()


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
    )


@pytest.fixture
def manager(base_config: PipelineConfig, secrets: DummySecrets) -> TrainingDeploymentManager:
    return TrainingDeploymentManager(config=base_config, secrets=secrets)


def test_manager_constructs_all_four_components(manager: TrainingDeploymentManager):
    """Facade owns one of each component, all wired through the same config."""
    assert isinstance(manager._code_syncer, CodeSyncer)
    assert isinstance(manager._file_uploader, FileUploader)
    assert isinstance(manager._deps_installer, DependencyInstaller)
    assert isinstance(manager._launcher, TrainingLauncher)
    # FileUploader must have received the same CodeSyncer (cross-component DI).
    assert manager._file_uploader._code_syncer is manager._code_syncer
    # TrainingLauncher must have received the same DependencyInstaller (cross-component DI).
    assert manager._launcher._deps_installer is manager._deps_installer


def test_set_workspace_propagates_to_every_component(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace/run_123")
    assert manager._workspace == "/workspace/run_123"
    assert manager._code_syncer.workspace == "/workspace/run_123"
    assert manager._file_uploader.workspace == "/workspace/run_123"
    assert manager._deps_installer.workspace == "/workspace/run_123"
    assert manager._launcher.workspace == "/workspace/run_123"

    manager.set_workspace(workspace_path="/workspace/run_456")
    assert manager._workspace == "/workspace/run_456"
    assert manager._launcher.workspace == "/workspace/run_456"


def test_workspace_property_reflects_current_value(manager: TrainingDeploymentManager):
    assert manager.workspace == TrainingDeploymentManager.DEFAULT_WORKSPACE
    manager.set_workspace(workspace_path="/workspace/custom")
    assert manager.workspace == "/workspace/custom"


def test_deploy_files_delegates_to_file_uploader(manager: TrainingDeploymentManager):
    ssh = MagicMock()
    ctx = {"config_path": "x"}
    with patch.object(manager._file_uploader, "deploy_files", return_value=Ok(None)) as mock:
        result = manager.deploy_files(ssh, ctx)
    assert result.is_ok()
    mock.assert_called_once_with(ssh, ctx)


def test_install_dependencies_delegates_to_deps_installer(manager: TrainingDeploymentManager):
    ssh = MagicMock()
    with patch.object(manager._deps_installer, "install", return_value=Ok(None)) as mock:
        result = manager.install_dependencies(ssh)
    assert result.is_ok()
    mock.assert_called_once_with(ssh)


def test_start_training_delegates_to_launcher(manager: TrainingDeploymentManager):
    ssh = MagicMock()
    ctx: dict[str, Any] = {}
    with patch.object(manager._launcher, "start_training", return_value=Ok({"mode": "docker"})) as mock:
        result = manager.start_training(ssh, ctx)
    assert result.is_ok()
    mock.assert_called_once_with(ssh, ctx, None)

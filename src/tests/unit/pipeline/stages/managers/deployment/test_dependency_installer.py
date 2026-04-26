"""Unit tests for src.pipeline.stages.managers.deployment.dependency_installer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.stages.managers.deployment.dependency_installer import DependencyInstaller
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
from src.utils.result import Failure, Ok, ProviderError

pytestmark = pytest.mark.unit

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

DATASET_CHAT_FIXTURE = "src/tests/fixtures/datasets/test_chat.jsonl"


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
def installer(base_config: PipelineConfig, secrets: DummySecrets) -> DependencyInstaller:
    return DependencyInstaller(config=base_config, secrets=secrets)


def test_install_single_node_uses_runtime_image_verify(installer: DependencyInstaller):
    """single_node: install() must verify runtime image via docker on host."""
    ssh_client = MagicMock()

    with patch.object(installer, "_verify_single_node_docker_runtime", return_value=Ok(None)) as mock_verify:
        result = installer.install(ssh_client)

    assert result.is_ok()
    mock_verify.assert_called_once()


def test_install_cloud_verify_ok_skips_install(base_config: PipelineConfig, secrets: DummySecrets):
    """runpod/cloud: if deps already present in image, we only verify."""
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    installer = DependencyInstaller(config=cfg, secrets=secrets)
    ssh_client = MagicMock()

    with patch.object(DependencyInstaller, "verify_prebuilt_dependencies", return_value=Ok(None)) as mock_verify:
        result = installer.install(ssh_client)

    assert result.is_ok()
    mock_verify.assert_called_once()


def test_install_cloud_verify_fail_returns_err(base_config: PipelineConfig, secrets: DummySecrets):
    """runpod/cloud: if deps missing in the image, we FAIL (no fallback install)."""
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    installer = DependencyInstaller(config=cfg, secrets=secrets)
    ssh_client = MagicMock()

    with patch.object(
        DependencyInstaller,
        "verify_prebuilt_dependencies",
        return_value=Failure(ProviderError(message="missing packages", code="DEPS_MISSING")),
    ):
        result = installer.install(ssh_client)

    assert result.is_err()


def test_verify_prebuilt_dependencies_success():
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "OK\nversion=1.0", "")

    result = DependencyInstaller.verify_prebuilt_dependencies(ssh_client)
    assert result.is_ok()


def test_verify_prebuilt_dependencies_failure_returns_err():
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (False, "", "ImportError")

    result = DependencyInstaller.verify_prebuilt_dependencies(ssh_client)
    assert result.is_err()
    assert "Runtime contract check failed" in str(result.unwrap_err())


def test_verify_single_node_docker_runtime_no_image_returns_config_error(installer: DependencyInstaller):
    """_verify_single_node_docker_runtime returns ConfigError when docker_image is absent."""
    ssh_client = MagicMock()

    with patch(
        "src.pipeline.stages.managers.deployment.dependency_installer.get_single_node_training_cfg",
        return_value={"workspace_path": "/tmp/w"},
    ):
        result = installer._verify_single_node_docker_runtime(ssh_client)

    assert result.is_err()
    err = result.unwrap_err()
    assert "docker_image is required" in str(err)


def test_verify_single_node_docker_runtime_pull_failure_returns_error(installer: DependencyInstaller):
    """Propagates pull failure from _ensure_docker_image_present."""
    ssh_client = MagicMock()
    pull_err = ProviderError(message="pull failed", code="DOCKER_PULL_FAILED")

    with patch.object(installer, "_ensure_docker_image_present", return_value=Failure(pull_err)):
        result = installer._verify_single_node_docker_runtime(ssh_client)

    assert result.is_err()
    assert "pull failed" in str(result.unwrap_err())


def test_verify_single_node_docker_runtime_check_failed_no_ok_in_stdout(installer: DependencyInstaller):
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "no marker", "")

    with patch.object(installer, "_ensure_docker_image_present", return_value=Ok(None)):
        result = installer._verify_single_node_docker_runtime(ssh_client)

    assert result.is_err()
    assert "missing required packages" in str(result.unwrap_err())


def test_verify_single_node_docker_runtime_check_failed_exec_returns_false(installer: DependencyInstaller):
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (False, "", "")

    with patch.object(installer, "_ensure_docker_image_present", return_value=Ok(None)):
        result = installer._verify_single_node_docker_runtime(ssh_client)

    assert result.is_err()


def test_set_workspace_propagates(installer: DependencyInstaller):
    installer.set_workspace("/tmp/run_42")
    assert installer.workspace == "/tmp/run_42"

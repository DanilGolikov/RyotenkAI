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
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.config.integrations.mlflow import MLflowConfig, MLflowTrackingRef
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
from src.utils.result import ConfigError, Failure, Ok, ProviderError

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


def test_start_training_fast_completion_marker_returns_ok(base_config: PipelineConfig, secrets: DummySecrets):
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    manager = TrainingDeploymentManager(config=cfg, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()
    executed_commands: list[str] = []

    def exec_side_effect(command: str, **kwargs: Any):
        executed_commands.append(command)
        if command.startswith("if [ -f /workspace/TRAINING_COMPLETE ]"):
            return True, "STATUS=COMPLETE\n", ""
        return True, "", ""

    ssh_client.exec_command.side_effect = exec_side_effect

    with (
        patch.object(manager, "_create_env_file", return_value=Ok("/workspace/.env")),
        patch("src.pipeline.stages.managers.deployment_manager.time.sleep", return_value=None),
    ):
        result = manager.start_training(ssh_client, {"config_path": "config/local_dev.yaml"})

    assert result.is_ok()
    assert result.unwrap().get("mode") == "docker"
    assert any("--config config/pipeline_config.yaml" in cmd for cmd in executed_commands)


def test_start_training_failure_marker_returns_err(base_config: PipelineConfig, secrets: DummySecrets):
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    manager = TrainingDeploymentManager(config=cfg, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()

    def exec_side_effect(command: str, **kwargs: Any):
        if command.startswith("if [ -f /workspace/TRAINING_COMPLETE ]"):
            return True, "STATUS=FAILED\nTraceback: boom\n", ""
        return True, "", ""

    ssh_client.exec_command.side_effect = exec_side_effect

    with (
        patch.object(manager, "_create_env_file", return_value=Ok("/workspace/.env")),
        patch("src.pipeline.stages.managers.deployment_manager.time.sleep", return_value=None),
    ):
        result = manager.start_training(ssh_client, {"config_path": "config/local_dev.yaml"})

    assert result.is_err()
    assert "Training failed:" in str(result.unwrap_err())


def test_start_training_docker_generates_docker_run_script_contains_mount_and_name(
    base_config: PipelineConfig, secrets: DummySecrets
):
    cfg = base_config.model_copy(deep=True)
    cfg.providers["single_node"] = {
        "training": {
            "execution_mode": "docker",
            "docker_image": "test/runtime:latest",
            "docker_shm_size": "16g",
            "docker_container_name_prefix": "helix_training",
        }
    }

    manager = TrainingDeploymentManager(config=cfg, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace/run_123")

    ssh_client = MagicMock()
    executed: list[str] = []

    def exec_side_effect(command: str, **kwargs: Any):
        executed.append(command)
        if command.startswith("test -f ") and "TRAINING_COMPLETE" in command:
            return False, "", ""
        if command.startswith("test -f ") and "TRAINING_FAILED" in command:
            return False, "", ""
        if command.startswith("docker ps"):
            return True, "RUNNING", ""
        return True, "", ""

    ssh_client.exec_command.side_effect = exec_side_effect

    with (
        patch.object(manager, "_ensure_docker_image_present", return_value=Ok(None)),
        patch.object(manager, "_create_env_file", return_value=Ok("/workspace/run_123/.env")),
        patch("src.pipeline.stages.managers.deployment_manager.time.sleep", return_value=None),
    ):
        result = manager.start_training(ssh_client, {"run": SimpleNamespace(name="my_run")})

    assert result.is_ok()
    create_cmd = next(cmd for cmd in executed if cmd.startswith("cat > /workspace/run_123/start_training.sh"))
    assert "docker run --rm --detach" in create_cmd
    assert "--name helix_training_my_run" in create_cmd
    assert "-v /workspace/run_123:/workspace" in create_cmd
    assert "-w /workspace" in create_cmd
    assert "test/runtime:latest" in create_cmd
    assert "python3 -m src.training.run_training --config config/pipeline_config.yaml" in create_cmd


@pytest.mark.skip(reason=(
    "Requires experiment_tracking resolver: _create_env_file assumes "
    "experiment_tracking.mlflow is a resolved MLflowConfig, but per PR3 "
    "the project YAML only carries a MLflowTrackingRef. Unskip once "
    "src/config/integrations/resolver.py lands and load_config merges "
    "the integration payload into MLflowConfig."
))
def test_create_env_file_docker_mode_sets_workspace_to_container_path(base_config: PipelineConfig, secrets: DummySecrets):
    deployment = TrainingDeploymentManager(config=base_config, secrets=secrets)
    deployment.set_workspace(workspace_path="/workspace/run_123")

    ssh_client = MagicMock()
    recorded: list[str] = []

    def exec_side_effect(command: str, **kwargs: Any):
        recorded.append(command)
        return True, "", ""

    ssh_client.exec_command.side_effect = exec_side_effect

    result = deployment._create_env_file(ssh_client, context={})
    assert result.is_ok()

    create_cmd = recorded[0]
    assert 'export HELIX_WORKSPACE="/workspace"' in create_cmd
    assert 'export PYTHONPATH="/workspace"' in create_cmd


@pytest.mark.skip(reason=(
    "Requires experiment_tracking resolver: _create_env_file assumes "
    "experiment_tracking.mlflow is a resolved MLflowConfig, but per PR3 "
    "the project YAML only carries a MLflowTrackingRef. Unskip once "
    "src/config/integrations/resolver.py lands and load_config merges "
    "the integration payload into MLflowConfig."
))
def test_create_env_file_includes_hf_token_and_mlflow_vars(secrets: DummySecrets):
    mlflow_cfg = MLflowConfig(
        tracking_uri="https://public.example.ts.net",
        local_tracking_uri="http://localhost:5002",
        ca_bundle_path="certs/mlflow-ca.pem",
        experiment_name="test-exp",
    )
    config = PipelineConfig(
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
        experiment_tracking=ExperimentTrackingConfig(mlflow=mlflow_cfg),
    )
    deployment = TrainingDeploymentManager(config=config, secrets=secrets)
    deployment.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()
    recorded: list[str] = []

    def exec_side_effect(command: str, **kwargs: Any):
        recorded.append(command)
        return True, "", ""

    ssh_client.exec_command.side_effect = exec_side_effect

    result = deployment._create_env_file(ssh_client, context={"mlflow_parent_run_id": "parent_123"})
    assert result.is_ok()
    assert result.unwrap() == "/workspace/.env"

    create_cmd = recorded[0]
    assert 'export HF_TOKEN="hf_test_token"' in create_cmd
    assert 'export MLFLOW_TRACKING_URI="https://public.example.ts.net"' in create_cmd
    assert 'export MLFLOW_PARENT_RUN_ID="parent_123"' in create_cmd
    assert 'export MLFLOW_HTTP_REQUEST_TIMEOUT="15"' in create_cmd
    assert 'export MLFLOW_HTTP_REQUEST_MAX_RETRIES="2"' in create_cmd
    assert 'export REQUESTS_CA_BUNDLE="certs/mlflow-ca.pem"' in create_cmd
    assert 'export SSL_CERT_FILE="certs/mlflow-ca.pem"' in create_cmd


@pytest.mark.skip(reason=(
    "Requires experiment_tracking resolver: _create_env_file assumes "
    "experiment_tracking.mlflow is a resolved MLflowConfig, but per PR3 "
    "the project YAML only carries a MLflowTrackingRef. Unskip once "
    "src/config/integrations/resolver.py lands and load_config merges "
    "the integration payload into MLflowConfig."
))
def test_create_env_file_mlflow_remote_falls_back_to_local_tracking_uri(secrets: DummySecrets):
    mlflow_cfg = MLflowConfig(
        tracking_uri=None,
        local_tracking_uri="http://localhost:5002",
        experiment_name="test-exp",
    )
    config = PipelineConfig(
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
        experiment_tracking=ExperimentTrackingConfig(mlflow=mlflow_cfg),
    )
    deployment = TrainingDeploymentManager(config=config, secrets=secrets)
    deployment.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()
    recorded: list[str] = []

    def exec_side_effect(command: str, **kwargs: Any):
        recorded.append(command)
        return True, "", ""

    ssh_client.exec_command.side_effect = exec_side_effect

    result = deployment._create_env_file(ssh_client, context={})
    assert result.is_ok()
    assert 'export MLFLOW_TRACKING_URI="http://localhost:5002"' in recorded[0]


def test_start_training_ps_detects_running_process_returns_ok(base_config: PipelineConfig, secrets: DummySecrets):
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    manager = TrainingDeploymentManager(config=cfg, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()

    def exec_side_effect(command: str, **kwargs: Any):
        if command.startswith("if [ -f /workspace/TRAINING_COMPLETE ]"):
            return True, "STATUS=RUNNING\n", ""
        return True, "", ""

    ssh_client.exec_command.side_effect = exec_side_effect

    with (
        patch.object(manager, "_create_env_file", return_value=Ok("/workspace/.env")),
        patch("src.pipeline.stages.managers.deployment_manager.time.sleep", return_value=None),
    ):
        result = manager.start_training(ssh_client, {"config_path": "config/local.yaml"})

    assert result.is_ok()
    assert result.unwrap().get("mode") == "docker"


def test_start_training_log_file_exists_returns_ok(base_config: PipelineConfig, secrets: DummySecrets):
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    manager = TrainingDeploymentManager(config=cfg, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()

    def exec_side_effect(command: str, **kwargs: Any):
        if command.startswith("if [ -f /workspace/TRAINING_COMPLETE ]"):
            return True, "STATUS=LOG_EXISTS\n", ""
        return True, "", ""

    ssh_client.exec_command.side_effect = exec_side_effect

    with (
        patch.object(manager, "_create_env_file", return_value=Ok("/workspace/.env")),
        patch("src.pipeline.stages.managers.deployment_manager.time.sleep", return_value=None),
    ):
        result = manager.start_training(ssh_client, {"config_path": "config/local.yaml"})

    assert result.is_ok()


def test_start_training_no_process_includes_log_details_in_err(base_config: PipelineConfig, secrets: DummySecrets):
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    manager = TrainingDeploymentManager(config=cfg, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()

    def exec_side_effect(command: str, **kwargs: Any):
        if command.startswith("if [ -f /workspace/TRAINING_COMPLETE ]"):
            return True, "STATUS=NONE\n", ""
        if command.startswith("tail -n 80 /workspace/training.log"):
            return True, "Traceback: something bad happened\n", ""
        return True, "", ""

    ssh_client.exec_command.side_effect = exec_side_effect

    from itertools import count

    # start_training now uses a wall-clock deadline; advance time deterministically to avoid real waits.
    time_counter = count(start=0, step=60)

    with (
        patch.object(manager, "_create_env_file", return_value=Ok("/workspace/.env")),
        patch("src.pipeline.stages.managers.deployment_manager.time.sleep", return_value=None),
        patch("src.pipeline.stages.managers.deployment_manager.time.time", side_effect=lambda: next(time_counter)),
    ):
        result = manager.start_training(ssh_client, {"config_path": "config/local.yaml"})

    assert result.is_err()
    err = str(result.unwrap_err())
    assert "Training failed to start" in err
    assert "Log content:" in err


@pytest.mark.skip(reason=(
    "Requires experiment_tracking resolver: _create_env_file assumes "
    "experiment_tracking.mlflow is a resolved MLflowConfig, but per PR3 "
    "the project YAML only carries a MLflowTrackingRef. Unskip once "
    "src/config/integrations/resolver.py lands and load_config merges "
    "the integration payload into MLflowConfig."
))
def test_create_env_file_exec_fails_returns_err(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (False, "", "boom")

    result = manager._create_env_file(ssh_client, context=None)
    assert result.is_err()
    assert "Failed to create .env file" in str(result.unwrap_err())


def test_start_training_returns_err_when_env_file_creation_fails(base_config: PipelineConfig, secrets: DummySecrets):
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    manager = TrainingDeploymentManager(config=cfg, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace")
    ssh_client = MagicMock()

    with patch.object(manager, "_create_env_file", return_value=Failure(ProviderError(message="env failed", code="ENV_FAILED"))):
        result = manager.start_training(ssh_client, {})

    assert result.is_err()
    assert "env failed" in str(result.unwrap_err())


def test_start_training_create_script_chmod_and_launch_failures(base_config: PipelineConfig, secrets: DummySecrets):
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    manager = TrainingDeploymentManager(config=cfg, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace")

    # 1) create script fails
    ssh_client = MagicMock()
    ssh_client.exec_command.side_effect = [
        (False, "", "nope"),  # create start script
    ]
    with patch.object(manager, "_create_env_file", return_value=Ok("/workspace/.env")):
        result = manager.start_training(ssh_client, {})
    assert result.is_err()
    assert "Failed to create start script" in str(result.unwrap_err())

    # 2) chmod fails
    ssh_client = MagicMock()
    ssh_client.exec_command.side_effect = [
        (True, "", ""),  # create script ok
        (False, "", "nope"),  # chmod
    ]
    with patch.object(manager, "_create_env_file", return_value=Ok("/workspace/.env")):
        result = manager.start_training(ssh_client, {})
    assert result.is_err()
    assert "Failed to chmod start script" in str(result.unwrap_err())

    # 3) launch fails
    ssh_client = MagicMock()
    ssh_client.exec_command.side_effect = [
        (True, "", ""),  # create script ok
        (True, "", ""),  # chmod ok
        (False, "", "nope"),  # launch
    ]
    with patch.object(manager, "_create_env_file", return_value=Ok("/workspace/.env")):
        result = manager.start_training(ssh_client, {})
    assert result.is_err()
    assert "Failed to start training" in str(result.unwrap_err())


def test_start_training_docker_training_complete_marker_returns_ok(
    base_config: PipelineConfig, secrets: DummySecrets
):
    """_start_training_docker returns Ok when TRAINING_COMPLETE marker found immediately."""
    manager = TrainingDeploymentManager(config=base_config, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace/run_docker")

    ssh_client = MagicMock()
    executed: list[str] = []

    def exec_side_effect(command: str, **kwargs: Any):
        executed.append(command)
        if "TRAINING_COMPLETE" in command and command.startswith("test -f"):
            return True, "SUCCESS", ""
        return True, "", ""

    ssh_client.exec_command.side_effect = exec_side_effect

    with (
        patch.object(manager, "_ensure_docker_image_present", return_value=Ok(None)),
        patch.object(manager, "_create_env_file", return_value=Ok("/workspace/run_docker/.env")),
        patch("src.pipeline.stages.managers.deployment_manager.time.sleep", return_value=None),
    ):
        result = manager.start_training(ssh_client, {})

    assert result.is_ok()
    assert result.unwrap().get("mode") == "docker"


# =============================================================================
# Lines 1093-1097 – _start_training_docker: log file exists marker
# =============================================================================

def test_start_training_docker_log_file_exists_returns_ok(
    base_config: PipelineConfig, secrets: DummySecrets
):
    """_start_training_docker returns Ok when training.log exists on host (container wrote into mount)."""
    manager = TrainingDeploymentManager(config=base_config, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace/run_docker")

    ssh_client = MagicMock()
    executed: list[str] = []

    def exec_side_effect(command: str, **kwargs: Any):
        executed.append(command)
        # TRAINING_COMPLETE check → not found
        if "TRAINING_COMPLETE" in command and command.startswith("test -f"):
            return False, "", ""
        # TRAINING_FAILED check → not found
        if "TRAINING_FAILED" in command and command.startswith("test -f"):
            return False, "", ""
        # docker ps (container check) → not running
        if command.startswith("docker ps"):
            return False, "", ""
        # log file exists check → present
        if "training.log" in command and command.startswith("test -f"):
            return True, "EXISTS", ""
        return True, "", ""

    ssh_client.exec_command.side_effect = exec_side_effect

    with (
        patch.object(manager, "_ensure_docker_image_present", return_value=Ok(None)),
        patch.object(manager, "_create_env_file", return_value=Ok("/workspace/run_docker/.env")),
        patch("src.pipeline.stages.managers.deployment_manager.docker_is_container_running", return_value=False),
        patch("src.pipeline.stages.managers.deployment_manager.time.sleep", return_value=None),
    ):
        result = manager.start_training(ssh_client, {})

    assert result.is_ok()
    assert result.unwrap().get("mode") == "docker"


# =============================================================================
# Lines 1103-1108 – _start_training_docker: TRAINING_START_TIMEOUT
# =============================================================================

def test_start_training_docker_timeout_returns_provider_error(
    base_config: PipelineConfig, secrets: DummySecrets
):
    """_start_training_docker returns ProviderError TRAINING_START_TIMEOUT when all polls exhaust."""
    manager = TrainingDeploymentManager(config=base_config, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace/run_docker")

    ssh_client = MagicMock()

    def exec_side_effect(command: str, **kwargs: Any):
        # Narrow to marker-check commands only (script body also contains these strings)
        if command.startswith("test -f") and "TRAINING_COMPLETE" in command:
            return False, "", ""
        if command.startswith("test -f") and "TRAINING_FAILED" in command:
            return False, "", ""
        if command.startswith("test -f") and "training.log" in command:
            return False, "", ""
        # script creation, chmod, launch, log cat → all succeed
        return True, "", ""

    ssh_client.exec_command.side_effect = exec_side_effect

    # Force only 1 attempt so the test doesn't spin
    with (
        patch.object(manager, "_ensure_docker_image_present", return_value=Ok(None)),
        patch.object(manager, "_create_env_file", return_value=Ok("/workspace/run_docker/.env")),
        patch("src.pipeline.stages.managers.deployment_manager.docker_is_container_running", return_value=False),
        patch("src.pipeline.stages.managers.deployment_manager.time.sleep", return_value=None),
        patch("src.pipeline.stages.managers.deployment_manager.DEPLOYMENT_TRAINING_START_TIMEOUT", 1),
    ):
        result = manager.start_training(ssh_client, {})

    assert result.is_err()
    err = result.unwrap_err()
    assert isinstance(err, ProviderError)
    assert err.code == "TRAINING_START_TIMEOUT"

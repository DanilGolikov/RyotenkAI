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

from src.pipeline.stages.managers.deployment_manager import TrainingDeploymentManager
from src.utils.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceHF,
    DatasetSourceLocal,
    ExperimentTrackingConfig,
    GlobalHyperparametersConfig,
    InferenceConfig,
    InferenceEnginesConfig,
    InferenceVLLMEngineConfig,
    LoraConfig,
    MLflowConfig,
    ModelConfig,
    PipelineConfig,
    TrainingOnlyConfig,
)
from src.utils.result import ConfigError, Err, Failure, Ok, ProviderError

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
        mlflow=MLflowConfig(
            tracking_uri="http://127.0.0.1:5002",
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
            qlora=LoraConfig(
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
            qlora=LoraConfig(
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


def test_build_ssh_opts_alias_mode(manager: TrainingDeploymentManager):
    ssh_client = MagicMock()
    ssh_client.ssh_base_opts = None  # no real opts → legacy path
    ssh_client._is_alias_mode = True

    assert manager._build_ssh_opts(ssh_client) == "-o StrictHostKeyChecking=no"


def test_build_ssh_opts_explicit_mode(manager: TrainingDeploymentManager):
    ssh_client = MagicMock()
    ssh_client.ssh_base_opts = None  # no real opts → legacy path
    ssh_client._is_alias_mode = False
    ssh_client.key_path = "/tmp/test_key"
    ssh_client.port = 2222

    assert manager._build_ssh_opts(ssh_client) == "-i /tmp/test_key -p 2222 -o StrictHostKeyChecking=no"


def test_build_ssh_opts_reuses_base_opts_from_ssh_client(manager: TrainingDeploymentManager):
    ssh_client = MagicMock()
    ssh_client.ssh_base_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ControlMaster=auto"]
    ssh_client.key_path = "/tmp/key"
    ssh_client.port = 3333

    result = manager._build_ssh_opts(ssh_client)
    assert "-i /tmp/key" in result
    assert "-p 3333" in result
    assert "ControlMaster=auto" in result


def test_set_workspace_sets_workspace(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace/run_123")
    assert manager._workspace == "/workspace/run_123"

    manager.set_workspace(workspace_path="/workspace/run_456")
    assert manager._workspace == "/workspace/run_456"


def test_deploy_files_multiple_datasets_from_config(config_multi_dataset: PipelineConfig, secrets: DummySecrets):
    deployment = TrainingDeploymentManager(config=config_multi_dataset, secrets=secrets)

    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(deployment, "_upload_files_batch", return_value=Ok(None)) as mock_batch,
        patch.object(deployment, "_sync_source_code", return_value=Ok(None)),
    ):
        assert Path(CONFIG_FIXTURE).exists(), f"Expected fixture config to exist: {CONFIG_FIXTURE}"
        context = {"config_path": CONFIG_FIXTURE}
        result = deployment.deploy_files(ssh_client, context)

    assert result.is_ok()

    # Verify batch uploader received both datasets + config
    files_to_upload: list[tuple[str, str]] = mock_batch.call_args[0][1]
    assert (CONFIG_FIXTURE, "config/pipeline_config.yaml") in files_to_upload
    assert (str(Path(DATASET_CHAT_FIXTURE).resolve()), "data/sft/test_chat.jsonl") in files_to_upload
    assert (
        str(Path(DATASET_INSTRUCTION_FIXTURE).resolve()),
        "data/dpo/test_instruction.jsonl",
    ) in files_to_upload


def test_deploy_files_dataset_not_found_returns_err(secrets: DummySecrets):
    missing_path = "src/tests/fixtures/datasets/definitely_missing_dataset.jsonl"
    assert not Path(missing_path).exists(), f"Test expects missing path: {missing_path}"

    config = PipelineConfig(
        model=ModelConfig(name="gpt2", torch_dtype="bfloat16", trust_remote_code=False),
        providers={"single_node": SINGLE_NODE_PROVIDER_CFG},
        training=TrainingOnlyConfig(
            provider="single_node",
            type="qlora",
            qlora=LoraConfig(
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
                source_local={
                    "local_paths": {"train": missing_path, "eval": None},
                },
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
    deployment = TrainingDeploymentManager(config=config, secrets=secrets)

    ssh_client = MagicMock()
    result = deployment.deploy_files(ssh_client, {"dataset_path": missing_path})

    assert result.is_err()
    assert "Dataset file not found" in str(result.unwrap_err())


def test_deploy_files_batch_failure_falls_back_to_individual(
    config_multi_dataset: PipelineConfig,
    secrets: DummySecrets,
):
    deployment = TrainingDeploymentManager(config=config_multi_dataset, secrets=secrets)

    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(deployment, "_upload_files_batch", return_value=Failure(ProviderError(message="batch failed", code="BATCH_FAILED"))),
        patch.object(deployment, "_upload_files_individual", return_value=Ok(None)) as mock_individual,
        patch.object(deployment, "_sync_source_code", return_value=Ok(None)),
    ):
        result = deployment.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_ok()
    mock_individual.assert_called_once()


def test_upload_files_batch_no_existing_files_returns_err(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()
    result = manager._upload_files_batch(
        ssh_client,
        files_to_upload=[
            ("definitely_missing_1.txt", "a.txt"),
            ("definitely_missing_2.txt", "b.txt"),
        ],
    )

    assert result.is_err()
    assert "No files to upload" in str(result.unwrap_err())


def test_upload_files_batch_rejects_absolute_remote_name(manager: TrainingDeploymentManager):
    """
    Regression/invariant:
    remote_name must be relative; absolute remote_name would bypass tmpdir in Path(tmpdir)/remote_name
    and can cause SameFileError or write outside staging directory.
    """
    manager.set_workspace(workspace_path="/workspace")

    class SSHClientStub:
        _is_alias_mode = True
        key_path = ""
        port = 22
        ssh_target = "pc"

        def exec_command(self, command: str, background: bool = False, timeout: int = 30, silent: bool = False):
            return True, "EXISTS", ""

    ssh_client = SSHClientStub()

    result = manager._upload_files_batch(
        ssh_client,
        files_to_upload=[
            ("requirements.txt", "/abs/requirements.txt"),
        ],
    )
    assert result.is_err()
    assert "must be relative" in str(result.unwrap_err())


def test_deploy_files_skips_unused_datasets(secrets: DummySecrets):
    """
    Invariant: deploy_files uploads only datasets referenced by training.strategies.
    """
    dataset_a = DATASET_CHAT_FIXTURE
    dataset_b = DATASET_INSTRUCTION_FIXTURE
    assert Path(dataset_a).exists()
    assert Path(dataset_b).exists()

    cfg = PipelineConfig(
        model=ModelConfig(name="gpt2", torch_dtype="bfloat16", trust_remote_code=False),
        providers={"single_node": SINGLE_NODE_PROVIDER_CFG},
        training=TrainingOnlyConfig(
            provider="single_node",
            type="qlora",
            qlora=LoraConfig(
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
            strategies=[{"strategy_type": "sft", "dataset": "default"}],
        ),
        datasets={
            "default": DatasetConfig(
                source_type="local",
                source_local={
                    "local_paths": {"train": dataset_a, "eval": None},
                },
            ),
            # Not referenced by strategies -> must not be uploaded
            "unused": DatasetConfig(
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

    deployment = TrainingDeploymentManager(config=cfg, secrets=secrets)

    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(deployment, "_upload_files_batch", return_value=Ok(None)) as mock_batch,
        patch.object(deployment, "_sync_source_code", return_value=Ok(None)),
    ):
        result = deployment.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_ok()
    files_to_upload: list[tuple[str, str]] = mock_batch.call_args[0][1]
    assert (str(Path(dataset_a).resolve()), "data/sft/test_chat.jsonl") in files_to_upload
    assert not any(remote == "data/cot/test_instruction.jsonl" for _local, remote in files_to_upload)


def test_upload_files_batch_ownership_warning_is_non_critical(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace")

    class SSHClientStub:
        _is_alias_mode = True
        key_path = ""
        port = 22
        ssh_target = "pc"

        def exec_command(self, command: str, background: bool = False, timeout: int = 30, silent: bool = False):
            # Used only for remote verification in this test.
            return True, "EXISTS", ""

    ssh_client = SSHClientStub()

    # Make subprocess.run fail with an ownership warning (should be treated as non-critical).
    mocked_completed = MagicMock()
    mocked_completed.returncode = 1
    mocked_completed.stdout = ""
    mocked_completed.stderr = "tar: Cannot change ownership to uid 1000, gid 1000: Operation not permitted"

    with patch("src.pipeline.stages.managers.deployment_manager.subprocess.run", return_value=mocked_completed):
        result = manager._upload_files_batch(
            ssh_client,
            files_to_upload=[
                ("requirements.txt", "requirements.txt"),
            ],
        )

    assert result.is_ok()


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


def test_sync_source_code_success(manager: TrainingDeploymentManager):
    # Ensure REQUIRED_MODULES exist in repo (this test should run from repo root).
    for module in manager.REQUIRED_MODULES:
        assert Path(module).exists(), f"Expected REQUIRED_MODULES entry to exist locally: {module}"

    manager.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()
    ssh_client.ssh_base_opts = None
    ssh_client._is_alias_mode = True
    ssh_client.ssh_target = "pc"
    ssh_client.key_path = ""
    ssh_client.port = 22
    ssh_client.exec_command.return_value = (True, "", "")

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ""
    completed.stderr = ""

    with patch("src.pipeline.stages.managers.deployment_manager.subprocess.run", return_value=completed) as mock_run:
        result = manager._sync_source_code(ssh_client)

    assert result.is_ok()
    # Single batch rsync call for all modules
    assert mock_run.call_count == 1
    rsync_cmd = mock_run.call_args[0][0]
    assert "rsync" in rsync_cmd
    for module in manager.REQUIRED_MODULES:
        assert module in rsync_cmd


def test_sync_source_code_rsync_failure_tar_fallback(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()
    ssh_client.ssh_base_opts = None
    ssh_client._is_alias_mode = True
    ssh_client.ssh_target = "pc"
    ssh_client.key_path = ""
    ssh_client.port = 22
    ssh_client.exec_command.return_value = (True, "", "")

    failing = MagicMock()
    failing.returncode = 1
    failing.stdout = ""
    failing.stderr = "rsync failed"

    with (
        patch("src.pipeline.stages.managers.deployment_manager.subprocess.run", return_value=failing),
        patch.object(manager, "_sync_module_tar", return_value=Ok(None)) as mock_tar,
    ):
        result = manager._sync_source_code(ssh_client)

    assert result.is_ok()
    # Batch rsync fails → per-module tar fallback for each module
    assert mock_tar.call_count == len(manager.REQUIRED_MODULES)


def test_sync_module_tar_dir_verify_exists_on_failure_returns_ok(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace")

    module = "src/training"
    assert Path(module).exists()

    ssh_client = MagicMock()
    ssh_client._is_alias_mode = True
    ssh_client.ssh_target = "pc"
    ssh_client.key_path = ""
    ssh_client.port = 22
    ssh_client.exec_command.return_value = (True, "EXISTS", "")

    failing = MagicMock()
    failing.returncode = 1
    failing.stdout = ""
    failing.stderr = "tar failed"

    with patch("src.pipeline.stages.managers.deployment_manager.subprocess.run", return_value=failing):
        result = manager._sync_module_tar(ssh_client, module=module, ssh_opts="-o StrictHostKeyChecking=no")

    assert result.is_ok()


def test_sync_module_tar_dir_verify_missing_returns_err(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace")

    module = "src/training"
    assert Path(module).exists()

    ssh_client = MagicMock()
    ssh_client._is_alias_mode = True
    ssh_client.ssh_target = "pc"
    ssh_client.key_path = ""
    ssh_client.port = 22
    ssh_client.exec_command.return_value = (False, "", "")

    failing = MagicMock()
    failing.returncode = 1
    failing.stdout = ""
    failing.stderr = "tar failed"

    with patch("src.pipeline.stages.managers.deployment_manager.subprocess.run", return_value=failing):
        result = manager._sync_module_tar(ssh_client, module=module, ssh_opts="-o StrictHostKeyChecking=no")

    assert result.is_err()
    assert f"Failed to sync {module}" in str(result.unwrap_err())


def test_install_dependencies_single_node_uses_runtime_image_verify(manager: TrainingDeploymentManager):
    """single_node: install_dependencies() must verify runtime image via docker on host."""
    ssh_client = MagicMock()

    with patch.object(manager, "_verify_single_node_docker_runtime", return_value=Ok(None)) as mock_verify:
        result = manager.install_dependencies(ssh_client)

    assert result.is_ok()
    mock_verify.assert_called_once()


def test_install_dependencies_cloud_verify_ok_skips_install(base_config: PipelineConfig, secrets: DummySecrets):
    """runpod/cloud: if deps already present in image, we only verify."""
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    manager = TrainingDeploymentManager(config=cfg, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()

    with patch.object(TrainingDeploymentManager, "_verify_prebuilt_dependencies", return_value=Ok(None)) as mock_verify:
        result = manager.install_dependencies(ssh_client)

    assert result.is_ok()
    mock_verify.assert_called_once()


def test_install_dependencies_cloud_verify_fail_returns_err(base_config: PipelineConfig, secrets: DummySecrets):
    """runpod/cloud: if deps missing in the image, we FAIL (no fallback install)."""
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    manager = TrainingDeploymentManager(config=cfg, secrets=secrets)
    manager.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()

    with patch.object(TrainingDeploymentManager, "_verify_prebuilt_dependencies", return_value=Failure(ProviderError(message="missing packages", code="DEPS_MISSING"))):
        result = manager.install_dependencies(ssh_client)

    assert result.is_err()


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
            qlora=LoraConfig(
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
            qlora=LoraConfig(
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


def test_deploy_files_individual_fallback_failure_is_returned(
    config_multi_dataset: PipelineConfig, secrets: DummySecrets
):
    deployment = TrainingDeploymentManager(config=config_multi_dataset, secrets=secrets)

    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(deployment, "_upload_files_batch", return_value=Failure(ProviderError(message="batch failed", code="BATCH_FAILED"))),
        patch.object(deployment, "_upload_files_individual", return_value=Failure(ProviderError(message="individual failed", code="INDIVIDUAL_FAILED"))),
        patch.object(deployment, "_sync_source_code", return_value=Ok(None)) as mock_sync,
    ):
        result = deployment.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_err()
    assert "individual failed" in str(result.unwrap_err())
    assert mock_sync.call_count == 0


def test_deploy_files_sync_source_code_failure_is_returned(config_multi_dataset: PipelineConfig, secrets: DummySecrets):
    deployment = TrainingDeploymentManager(config=config_multi_dataset, secrets=secrets)

    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(deployment, "_upload_files_batch", return_value=Ok(None)),
        patch.object(deployment, "_sync_source_code", return_value=Failure(ProviderError(message="sync failed", code="SYNC_FAILED"))),
    ):
        result = deployment.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_err()
    assert "sync failed" in str(result.unwrap_err())


def test_upload_files_batch_hard_failure_returns_err(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace")

    class SSHClientStub:
        _is_alias_mode = True
        key_path = ""
        port = 22
        ssh_target = "pc"

        def exec_command(self, command: str, background: bool = False, timeout: int = 30, silent: bool = False):
            return True, "EXISTS", ""

    ssh_client = SSHClientStub()

    completed = MagicMock()
    completed.returncode = 1
    completed.stdout = ""
    completed.stderr = "Permission denied"

    with patch("src.pipeline.stages.managers.deployment_manager.subprocess.run", return_value=completed):
        result = manager._upload_files_batch(ssh_client, files_to_upload=[("requirements.txt", "requirements.txt")])

    assert result.is_err()
    assert "Batch upload failed:" in str(result.unwrap_err())


def test_upload_files_individual_success_uploads_dataset_and_aux_files(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace")

    dataset_files = [
        (DATASET_CHAT_FIXTURE, DATASET_CHAT_FIXTURE),
        (DATASET_INSTRUCTION_FIXTURE, DATASET_INSTRUCTION_FIXTURE),
    ]
    for p, _remote in dataset_files:
        assert Path(p).exists(), f"Expected fixture dataset to exist: {p}"
    assert Path(CONFIG_FIXTURE).exists(), f"Expected fixture config to exist: {CONFIG_FIXTURE}"

    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")
    ssh_client.upload_file.return_value = (True, "")

    result = manager._upload_files_individual(
        ssh_client,
        dataset_files=dataset_files,
        config_path=CONFIG_FIXTURE,
    )

    assert result.is_ok()
    # 2 datasets + config
    assert ssh_client.upload_file.call_count == len(dataset_files) + 1


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


def test_verify_prebuilt_dependencies_success(manager: TrainingDeploymentManager):
    ssh_client = MagicMock()
    ssh_client.exec_command.side_effect = [
        (True, "OK\n", ""),  # verify_cmd
        (True, "PyTorch: 2.x\nTransformers: 4.x\nTRL: 0.x\nPEFT: 0.x\n", ""),  # version_cmd
    ]

    result = manager._verify_prebuilt_dependencies(ssh_client)
    assert result.is_ok()


def test_verify_prebuilt_dependencies_failure_returns_err(manager: TrainingDeploymentManager):
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (False, "", "ImportError")

    result = manager._verify_prebuilt_dependencies(ssh_client)
    assert result.is_err()
    assert "Runtime contract check failed" in str(result.unwrap_err())


def test_sync_source_code_skips_missing_module_and_still_ok(manager: TrainingDeploymentManager, monkeypatch):
    manager.set_workspace(workspace_path="/workspace")

    missing_module = "src/definitely_missing_module_xyz"
    assert not Path(missing_module).exists()

    # Force REQUIRED_MODULES to contain only a missing entry to hit the warning branch.
    monkeypatch.setattr(TrainingDeploymentManager, "REQUIRED_MODULES", [missing_module])

    ssh_client = MagicMock()
    ssh_client._is_alias_mode = True
    ssh_client.ssh_target = "pc"
    ssh_client.key_path = ""
    ssh_client.port = 22
    ssh_client.exec_command.return_value = (True, "", "")

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ""
    completed.stderr = ""

    with patch("src.pipeline.stages.managers.deployment_manager.subprocess.run", return_value=completed) as mock_run:
        result = manager._sync_source_code(ssh_client)

    assert result.is_ok()
    # No module exists locally -> should never attempt rsync/tar.
    assert mock_run.call_count == 0


def test_sync_source_code_tar_fallback_failure_is_returned(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace")

    ssh_client = MagicMock()
    ssh_client._is_alias_mode = True
    ssh_client.ssh_target = "pc"
    ssh_client.key_path = ""
    ssh_client.port = 22
    ssh_client.exec_command.return_value = (True, "", "")

    failing = MagicMock()
    failing.returncode = 1
    failing.stdout = ""
    failing.stderr = "rsync failed"

    with (
        patch("src.pipeline.stages.managers.deployment_manager.subprocess.run", return_value=failing),
        patch.object(manager, "_sync_module_tar", return_value=Failure(ProviderError(message="tar failed", code="TAR_FAILED"))),
    ):
        result = manager._sync_source_code(ssh_client)

    assert result.is_err()
    assert "tar failed" in str(result.unwrap_err())


def test_sync_module_tar_file_scp_success_returns_ok(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace")

    module = "src/__init__.py"
    assert Path(module).exists()

    ssh_client = MagicMock()
    ssh_client._is_alias_mode = True
    ssh_client.ssh_target = "pc"
    ssh_client.key_path = ""
    ssh_client.port = 22

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ""
    completed.stderr = ""

    with patch("src.pipeline.stages.managers.deployment_manager.subprocess.run", return_value=completed):
        result = manager._sync_module_tar(ssh_client, module=module, ssh_opts="-o StrictHostKeyChecking=no")

    assert result.is_ok()


    # NOTE: no runtime dependency installation in docker-only mode


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


def test_upload_files_batch_verification_missing_is_non_fatal(manager: TrainingDeploymentManager):
    manager.set_workspace(workspace_path="/workspace")

    class SSHClientStub:
        _is_alias_mode = True
        key_path = ""
        port = 22
        ssh_target = "pc"

        def exec_command(self, command: str, background: bool = False, timeout: int = 30, silent: bool = False):
            # Verification fails for test -f
            if command.startswith("test -f"):
                return False, "", ""
            return True, "", ""

    ssh_client = SSHClientStub()

    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = ""
    completed.stderr = ""

    with patch("src.pipeline.stages.managers.deployment_manager.subprocess.run", return_value=completed):
        result = manager._upload_files_batch(ssh_client, files_to_upload=[("requirements.txt", "requirements.txt")])

    assert result.is_ok()


def test_upload_files_individual_failure_and_warning_paths(
    manager: TrainingDeploymentManager, monkeypatch: pytest.MonkeyPatch
):
    manager.set_workspace(workspace_path="/workspace")

    # mkdir fails -> warning, then upload_file fails -> Err
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (False, "", "mkdir failed")
    ssh_client.upload_file.return_value = (False, "upload failed")

    result = manager._upload_files_individual(
        ssh_client,
        dataset_files=[(DATASET_CHAT_FIXTURE, "data/x.jsonl")],
        config_path=CONFIG_FIXTURE,
    )
    assert result.is_err()
    assert "Failed to upload dataset" in str(result.unwrap_err())

    # requirements missing + config missing -> both warnings, still Ok
    orig_exists = Path.exists

    def exists(p: Path) -> bool:
        if str(p) == "requirements.txt":
            return False
        return orig_exists(p)

    monkeypatch.setattr("pathlib.Path.exists", exists)
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")
    ssh_client.upload_file.return_value = (True, "")

    result = manager._upload_files_individual(
        ssh_client,
        dataset_files=[(DATASET_CHAT_FIXTURE, "data/x.jsonl")],
        config_path="definitely_missing.yaml",
    )
    assert result.is_ok()

    # requirements upload fails -> warning, but still Ok
    monkeypatch.setattr("pathlib.Path.exists", orig_exists)
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")
    ssh_client.upload_file.side_effect = [
        (True, ""),  # dataset ok
        (False, "req failed"),  # requirements
    ]
    result = manager._upload_files_individual(
        ssh_client,
        dataset_files=[(DATASET_CHAT_FIXTURE, "data/x.jsonl")],
        config_path="definitely_missing.yaml",  # avoid needing config upload
    )
    assert result.is_ok()

    # config mkdir fails + config upload fails -> warnings, still Ok
    ssh_client = MagicMock()
    ssh_client.exec_command.side_effect = [
        (True, "", ""),  # mkdir dataset dir
        (False, "", "mkdir config failed"),  # mkdir config dir
    ]
    ssh_client.upload_file.side_effect = [
        (True, ""),  # dataset ok
        (True, ""),  # requirements ok
        (False, "config upload failed"),  # config upload
    ]
    result = manager._upload_files_individual(
        ssh_client,
        dataset_files=[(DATASET_CHAT_FIXTURE, "data/x.jsonl")],
        config_path=CONFIG_FIXTURE,
    )
    assert result.is_ok()


# =============================================================================
# HF-only datasets should not fail deploy_files
# =============================================================================

def test_deploy_files_huggingface_only_uploads_config_and_syncs_code(secrets: DummySecrets):
    """HF-only datasets skip local file uploads but still upload config and sync source code."""
    cfg = PipelineConfig(
        model=ModelConfig(name="gpt2", torch_dtype="bfloat16", trust_remote_code=False),
        providers={"single_node": SINGLE_NODE_PROVIDER_CFG},
        training=TrainingOnlyConfig(
            provider="single_node",
            type="qlora",
            qlora=LoraConfig(
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
                source_type="huggingface",
                source_hf=DatasetSourceHF(train_id="org/dataset"),
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
    deployment = TrainingDeploymentManager(config=cfg, secrets=secrets)

    ssh_client = MagicMock()
    with (
        patch.object(deployment, "_upload_files_batch", return_value=Ok(None)) as mock_batch,
        patch.object(deployment, "_sync_source_code", return_value=Ok(None)),
    ):
        result = deployment.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_ok()
    mock_batch.assert_called_once()
    uploaded_files = mock_batch.call_args.args[1]
    assert uploaded_files == [(CONFIG_FIXTURE, "config/pipeline_config.yaml")]


# =============================================================================
# Lines 473-475 – FILE_UPLOAD_FAILED (OSError in deploy_files)
# =============================================================================

def test_deploy_files_os_error_in_batch_upload_returns_provider_error(
    base_config: PipelineConfig, secrets: DummySecrets
):
    """deploy_files wraps OSError from _upload_files_batch into ProviderError FILE_UPLOAD_FAILED."""
    deployment = TrainingDeploymentManager(config=base_config, secrets=secrets)

    ssh_client = MagicMock()

    with patch.object(deployment, "_upload_files_batch", side_effect=OSError("disk full")):
        result = deployment.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_err()
    err = result.unwrap_err()
    assert isinstance(err, ProviderError)
    assert err.code == "FILE_UPLOAD_FAILED"
    assert "disk full" in err.message


# =============================================================================
# Lines 491-496 – _get_active_provider_name fallback paths
# =============================================================================

def test_get_active_provider_name_returns_string_from_training_provider(secrets: DummySecrets):
    """Falls back to training.provider string when get_active_provider_name() raises."""
    cfg = MagicMock()
    cfg.get_active_provider_name.side_effect = RuntimeError("not impl")
    cfg.training.provider = "my_custom_provider"

    mgr = TrainingDeploymentManager.__new__(TrainingDeploymentManager)
    mgr.config = cfg

    assert mgr._get_active_provider_name() == "my_custom_provider"


def test_get_active_provider_name_non_string_provider_returns_single_node(secrets: DummySecrets):
    """Falls back to PROVIDER_SINGLE_NODE when training.provider is not a string."""
    from src.constants import PROVIDER_SINGLE_NODE

    cfg = MagicMock()
    cfg.get_active_provider_name.side_effect = RuntimeError("not impl")
    cfg.training.provider = 42  # not a string

    mgr = TrainingDeploymentManager.__new__(TrainingDeploymentManager)
    mgr.config = cfg

    assert mgr._get_active_provider_name() == PROVIDER_SINGLE_NODE


def test_get_active_provider_name_no_training_attr_returns_single_node():
    """Falls back to PROVIDER_SINGLE_NODE when config has no training attribute."""
    from src.constants import PROVIDER_SINGLE_NODE

    class _Cfg:
        def get_active_provider_name(self) -> str:
            raise RuntimeError("not impl")

    mgr = TrainingDeploymentManager.__new__(TrainingDeploymentManager)
    mgr.config = _Cfg()

    assert mgr._get_active_provider_name() == PROVIDER_SINGLE_NODE


# =============================================================================
# Lines 618-626 – _verify_single_node_docker_runtime: no docker_image
# =============================================================================

def test_verify_single_node_docker_runtime_no_image_returns_config_error(
    manager: TrainingDeploymentManager,
):
    """_verify_single_node_docker_runtime returns ConfigError when docker_image is absent."""
    ssh_client = MagicMock()

    # Patch training cfg to have no docker_image key
    with patch.object(manager, "_get_single_node_training_cfg", return_value={"workspace_path": "/tmp/w"}):
        result = manager._verify_single_node_docker_runtime(ssh_client)

    assert result.is_err()
    err = result.unwrap_err()
    assert isinstance(err, ConfigError)
    assert err.code == "DOCKER_IMAGE_NOT_CONFIGURED"


# =============================================================================
# Lines 631-632 – _verify_single_node_docker_runtime: image pull failure
# =============================================================================

def test_verify_single_node_docker_runtime_pull_failure_returns_error(
    manager: TrainingDeploymentManager,
):
    """_verify_single_node_docker_runtime propagates pull failure from _ensure_docker_image_present."""
    ssh_client = MagicMock()
    pull_err = ProviderError(message="image not found", code="DOCKER_PULL_FAILED")

    with patch.object(manager, "_ensure_docker_image_present", return_value=Failure(pull_err)):
        result = manager._verify_single_node_docker_runtime(ssh_client)

    assert result.is_err()
    assert result.unwrap_err().code == "DOCKER_PULL_FAILED"


# =============================================================================
# Lines 640-650 – DOCKER_RUNTIME_CHECK_FAILED
# =============================================================================

def test_verify_single_node_docker_runtime_check_failed_no_ok_in_stdout(
    manager: TrainingDeploymentManager,
):
    """DOCKER_RUNTIME_CHECK_FAILED when exec_command succeeds but stdout contains no 'OK'."""
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "FAILED: missing packages", "")

    with patch.object(manager, "_ensure_docker_image_present", return_value=Ok(None)):
        result = manager._verify_single_node_docker_runtime(ssh_client)

    assert result.is_err()
    err = result.unwrap_err()
    assert isinstance(err, ProviderError)
    assert err.code == "DOCKER_RUNTIME_CHECK_FAILED"


def test_verify_single_node_docker_runtime_check_failed_exec_returns_false(
    manager: TrainingDeploymentManager,
):
    """DOCKER_RUNTIME_CHECK_FAILED when exec_command itself returns success=False."""
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (False, "", "container error")

    with patch.object(manager, "_ensure_docker_image_present", return_value=Ok(None)):
        result = manager._verify_single_node_docker_runtime(ssh_client)

    assert result.is_err()
    err = result.unwrap_err()
    assert isinstance(err, ProviderError)
    assert err.code == "DOCKER_RUNTIME_CHECK_FAILED"


# =============================================================================
# Lines 1046-1049 – _start_training_docker: TRAINING_COMPLETE marker
# =============================================================================

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

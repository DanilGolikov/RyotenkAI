"""Unit tests for src.pipeline.stages.managers.deployment.code_syncer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.stages.managers.deployment.code_syncer import CodeSyncer
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


DATASET_CHAT_FIXTURE = "src/tests/fixtures/datasets/test_chat.jsonl"

SINGLE_NODE_PROVIDER_CFG: dict = {
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
def syncer(base_config: PipelineConfig, secrets: DummySecrets) -> CodeSyncer:
    s = CodeSyncer(config=base_config, secrets=secrets)
    s.set_workspace("/workspace")
    return s


def test_sync_success(syncer: CodeSyncer):
    for module in CodeSyncer.REQUIRED_MODULES:
        assert Path(module).exists(), f"Expected REQUIRED_MODULES entry to exist locally: {module}"

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

    with patch("src.pipeline.stages.managers.deployment.code_syncer.subprocess.run", return_value=completed) as mock_run:
        result = syncer.sync(ssh_client)

    assert result.is_ok()
    assert mock_run.call_count == 1
    rsync_cmd = mock_run.call_args[0][0]
    assert "rsync" in rsync_cmd
    for module in CodeSyncer.REQUIRED_MODULES:
        assert module in rsync_cmd


def test_sync_rsync_failure_tar_fallback(syncer: CodeSyncer):
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
        patch("src.pipeline.stages.managers.deployment.code_syncer.subprocess.run", return_value=failing),
        patch.object(syncer, "_sync_module_tar", return_value=Ok(None)) as mock_tar,
    ):
        result = syncer.sync(ssh_client)

    assert result.is_ok()
    assert mock_tar.call_count == len(CodeSyncer.REQUIRED_MODULES)


def test_sync_module_tar_dir_verify_exists_on_failure_returns_ok(syncer: CodeSyncer):
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

    with patch("src.pipeline.stages.managers.deployment.code_syncer.subprocess.run", return_value=failing):
        result = syncer._sync_module_tar(ssh_client, module=module, ssh_opts="-o StrictHostKeyChecking=no")

    assert result.is_ok()


def test_sync_module_tar_dir_verify_missing_returns_err(syncer: CodeSyncer):
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

    with patch("src.pipeline.stages.managers.deployment.code_syncer.subprocess.run", return_value=failing):
        result = syncer._sync_module_tar(ssh_client, module=module, ssh_opts="-o StrictHostKeyChecking=no")

    assert result.is_err()
    assert f"Failed to sync {module}" in str(result.unwrap_err())


def test_sync_skips_missing_module_and_still_ok(syncer: CodeSyncer, monkeypatch):
    missing_module = "src/definitely_missing_module_xyz"
    assert not Path(missing_module).exists()

    monkeypatch.setattr(CodeSyncer, "REQUIRED_MODULES", [missing_module])

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

    with patch("src.pipeline.stages.managers.deployment.code_syncer.subprocess.run", return_value=completed) as mock_run:
        result = syncer.sync(ssh_client)

    assert result.is_ok()
    assert mock_run.call_count == 0


def test_sync_tar_fallback_failure_is_returned(syncer: CodeSyncer):
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
        patch("src.pipeline.stages.managers.deployment.code_syncer.subprocess.run", return_value=failing),
        patch.object(syncer, "_sync_module_tar", return_value=Failure(ProviderError(message="tar failed", code="TAR_FAILED"))),
    ):
        result = syncer.sync(ssh_client)

    assert result.is_err()
    assert "tar failed" in str(result.unwrap_err())


def test_sync_module_tar_file_success_returns_ok(syncer: CodeSyncer):
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

    with patch("src.pipeline.stages.managers.deployment.code_syncer.subprocess.run", return_value=completed):
        result = syncer._sync_module_tar(ssh_client, module=module, ssh_opts="-o StrictHostKeyChecking=no")

    assert result.is_ok()


def test_set_workspace_propagates():
    syncer = CodeSyncer(config=MagicMock(), secrets=MagicMock())
    assert syncer.workspace == "/workspace"
    syncer.set_workspace("/tmp/run_42")
    assert syncer.workspace == "/tmp/run_42"

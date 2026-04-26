"""Unit tests for src.pipeline.stages.managers.deployment.file_uploader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.stages.managers.deployment.code_syncer import CodeSyncer
from src.pipeline.stages.managers.deployment.file_uploader import FileUploader
from src.utils.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceHF,
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
DATASET_INSTRUCTION_FIXTURE = "src/tests/fixtures/datasets/test_instruction.jsonl"
CONFIG_FIXTURE = "src/tests/fixtures/configs/test_pipeline.yaml"

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
def config_multi_dataset() -> PipelineConfig:
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
                {"strategy_type": "sft", "dataset": "default"},
                {"strategy_type": "dpo", "dataset": "secondary"},
            ],
        ),
        datasets={
            "default": DatasetConfig(
                source_type="local",
                source_local={"local_paths": {"train": DATASET_CHAT_FIXTURE, "eval": None}},
            ),
            "secondary": DatasetConfig(
                source_type="local",
                source_local={"local_paths": {"train": DATASET_INSTRUCTION_FIXTURE, "eval": None}},
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
    )


def _make_uploader(config: PipelineConfig, secrets: DummySecrets, *, workspace: str = "/workspace") -> FileUploader:
    code_syncer = CodeSyncer(config=config, secrets=secrets)
    uploader = FileUploader(config=config, secrets=secrets, code_syncer=code_syncer)
    code_syncer.set_workspace(workspace)
    uploader.set_workspace(workspace)
    return uploader


@pytest.fixture
def uploader(base_config: PipelineConfig, secrets: DummySecrets) -> FileUploader:
    return _make_uploader(base_config, secrets)


# ============================================================================
# deploy_files orchestration
# ============================================================================


def test_deploy_files_multiple_datasets_from_config(config_multi_dataset: PipelineConfig, secrets: DummySecrets):
    uploader = _make_uploader(config_multi_dataset, secrets)

    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(uploader, "_upload_files_batch", return_value=Ok(None)) as mock_batch,
        patch.object(uploader._code_syncer, "sync", return_value=Ok(None)),
    ):
        assert Path(CONFIG_FIXTURE).exists(), f"Expected fixture config to exist: {CONFIG_FIXTURE}"
        result = uploader.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_ok()

    files_to_upload: list[tuple[str, str]] = mock_batch.call_args[0][1]
    assert (CONFIG_FIXTURE, "config/pipeline_config.yaml") in files_to_upload
    assert (str(Path(DATASET_CHAT_FIXTURE).resolve()), "data/sft/test_chat.jsonl") in files_to_upload
    assert (
        str(Path(DATASET_INSTRUCTION_FIXTURE).resolve()),
        "data/dpo/test_instruction.jsonl",
    ) in files_to_upload


def test_deploy_files_dataset_not_found_returns_err(secrets: DummySecrets):
    missing_path = "src/tests/fixtures/datasets/definitely_missing_dataset.jsonl"
    assert not Path(missing_path).exists()

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
                source_local={"local_paths": {"train": missing_path, "eval": None}},
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
    uploader = _make_uploader(config, secrets)

    ssh_client = MagicMock()
    result = uploader.deploy_files(ssh_client, {"dataset_path": missing_path})

    assert result.is_err()
    assert "Dataset file not found" in str(result.unwrap_err())


def test_deploy_files_batch_failure_falls_back_to_individual(
    config_multi_dataset: PipelineConfig, secrets: DummySecrets
):
    uploader = _make_uploader(config_multi_dataset, secrets)

    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(
            uploader,
            "_upload_files_batch",
            return_value=Failure(ProviderError(message="batch failed", code="BATCH_FAILED")),
        ),
        patch.object(uploader, "_upload_files_individual", return_value=Ok(None)) as mock_individual,
        patch.object(uploader._code_syncer, "sync", return_value=Ok(None)),
    ):
        result = uploader.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_ok()
    mock_individual.assert_called_once()


def test_deploy_files_skips_unused_datasets(secrets: DummySecrets):
    """Invariant: deploy_files uploads only datasets referenced by training.strategies."""
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
            strategies=[{"strategy_type": "sft", "dataset": "default"}],
        ),
        datasets={
            "default": DatasetConfig(
                source_type="local",
                source_local={"local_paths": {"train": dataset_a, "eval": None}},
            ),
            # Not referenced -> must not be uploaded
            "unused": DatasetConfig(
                source_type="local",
                source_local={"local_paths": {"train": dataset_b, "eval": None}},
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
    )

    uploader = _make_uploader(cfg, secrets)
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(uploader, "_upload_files_batch", return_value=Ok(None)) as mock_batch,
        patch.object(uploader._code_syncer, "sync", return_value=Ok(None)),
    ):
        result = uploader.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_ok()
    files_to_upload: list[tuple[str, str]] = mock_batch.call_args[0][1]
    assert (str(Path(dataset_a).resolve()), "data/sft/test_chat.jsonl") in files_to_upload
    assert not any(remote == "data/cot/test_instruction.jsonl" for _local, remote in files_to_upload)


def test_deploy_files_individual_fallback_failure_is_returned(
    config_multi_dataset: PipelineConfig, secrets: DummySecrets
):
    uploader = _make_uploader(config_multi_dataset, secrets)
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(
            uploader,
            "_upload_files_batch",
            return_value=Failure(ProviderError(message="batch failed", code="BATCH_FAILED")),
        ),
        patch.object(
            uploader,
            "_upload_files_individual",
            return_value=Failure(ProviderError(message="individual failed", code="INDIV_FAILED")),
        ),
        patch.object(uploader._code_syncer, "sync", return_value=Ok(None)),
    ):
        result = uploader.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_err()
    assert "individual failed" in str(result.unwrap_err())


def test_deploy_files_sync_source_code_failure_is_returned(
    config_multi_dataset: PipelineConfig, secrets: DummySecrets
):
    uploader = _make_uploader(config_multi_dataset, secrets)
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(uploader, "_upload_files_batch", return_value=Ok(None)),
        patch.object(
            uploader._code_syncer,
            "sync",
            return_value=Failure(ProviderError(message="sync failed", code="SYNC_FAILED")),
        ),
    ):
        result = uploader.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_err()
    assert "sync failed" in str(result.unwrap_err())


def test_deploy_files_huggingface_only_uploads_config_and_syncs_code(secrets: DummySecrets):
    cfg = PipelineConfig(
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
    )

    uploader = _make_uploader(cfg, secrets)
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with (
        patch.object(uploader, "_upload_files_batch", return_value=Ok(None)) as mock_batch,
        patch.object(uploader._code_syncer, "sync", return_value=Ok(None)) as mock_sync,
    ):
        result = uploader.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_ok()
    # No dataset files; only the config
    files_to_upload: list[tuple[str, str]] = mock_batch.call_args[0][1]
    assert files_to_upload == [(CONFIG_FIXTURE, "config/pipeline_config.yaml")]
    mock_sync.assert_called_once()


def test_deploy_files_os_error_in_batch_upload_returns_provider_error(
    config_multi_dataset: PipelineConfig, secrets: DummySecrets
):
    uploader = _make_uploader(config_multi_dataset, secrets)
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")

    with patch.object(uploader, "_upload_files_batch", side_effect=OSError("disk full")):
        result = uploader.deploy_files(ssh_client, {"config_path": CONFIG_FIXTURE})

    assert result.is_err()
    assert "Failed to upload files" in str(result.unwrap_err())


# ============================================================================
# _upload_files_batch
# ============================================================================


def test_upload_files_batch_no_existing_files_returns_err(uploader: FileUploader):
    ssh_client = MagicMock()
    result = uploader._upload_files_batch(
        ssh_client,
        files_to_upload=[
            ("definitely_missing_1.txt", "a.txt"),
            ("definitely_missing_2.txt", "b.txt"),
        ],
    )

    assert result.is_err()
    assert "No files to upload" in str(result.unwrap_err())


def test_upload_files_batch_rejects_absolute_remote_name(uploader: FileUploader):
    """remote_name must be relative; absolute would bypass tmpdir staging."""

    class SSHClientStub:
        _is_alias_mode = True
        key_path = ""
        port = 22
        ssh_target = "pc"

        def exec_command(self, command: str, background: bool = False, timeout: int = 30, silent: bool = False):
            return True, "EXISTS", ""

    ssh_client = SSHClientStub()

    result = uploader._upload_files_batch(
        ssh_client,
        files_to_upload=[("requirements.txt", "/abs/requirements.txt")],
    )
    assert result.is_err()
    assert "must be relative" in str(result.unwrap_err())


def test_upload_files_batch_ownership_warning_is_non_critical(uploader: FileUploader):
    class SSHClientStub:
        _is_alias_mode = True
        key_path = ""
        port = 22
        ssh_target = "pc"

        def exec_command(self, command: str, background: bool = False, timeout: int = 30, silent: bool = False):
            return True, "EXISTS", ""

    ssh_client = SSHClientStub()

    mocked_completed = MagicMock()
    mocked_completed.returncode = 1
    mocked_completed.stdout = ""
    mocked_completed.stderr = "tar: Cannot change ownership to uid 1000, gid 1000: Operation not permitted"

    with patch("src.pipeline.stages.managers.deployment.file_uploader.subprocess.run", return_value=mocked_completed):
        result = uploader._upload_files_batch(
            ssh_client,
            files_to_upload=[("requirements.txt", "requirements.txt")],
        )

    assert result.is_ok()


def test_upload_files_batch_hard_failure_returns_err(uploader: FileUploader):
    class SSHClientStub:
        _is_alias_mode = True
        key_path = ""
        port = 22
        ssh_target = "pc"

        def exec_command(self, command: str, background: bool = False, timeout: int = 30, silent: bool = False):
            return True, "EXISTS", ""

    ssh_client = SSHClientStub()

    mocked_completed = MagicMock()
    mocked_completed.returncode = 1
    mocked_completed.stdout = ""
    mocked_completed.stderr = "tar: real disk failure"

    with patch("src.pipeline.stages.managers.deployment.file_uploader.subprocess.run", return_value=mocked_completed):
        result = uploader._upload_files_batch(
            ssh_client,
            files_to_upload=[("requirements.txt", "requirements.txt")],
        )

    assert result.is_err()
    assert "Batch upload failed" in str(result.unwrap_err())


def test_upload_files_batch_verification_missing_is_non_fatal(uploader: FileUploader):
    """Verify command returns OK status but file shows MISS::; should not error."""

    class SSHClientStub:
        _is_alias_mode = True
        key_path = ""
        port = 22
        ssh_target = "pc"

        def exec_command(self, command: str, background: bool = False, timeout: int = 30, silent: bool = False):
            # Return MISS::requirements.txt for the verify command
            return True, "MISS::requirements.txt", ""

    ssh_client = SSHClientStub()

    mocked_completed = MagicMock()
    mocked_completed.returncode = 0
    mocked_completed.stdout = ""
    mocked_completed.stderr = ""

    with patch("src.pipeline.stages.managers.deployment.file_uploader.subprocess.run", return_value=mocked_completed):
        result = uploader._upload_files_batch(
            ssh_client,
            files_to_upload=[("requirements.txt", "requirements.txt")],
        )

    assert result.is_ok()


# ============================================================================
# _upload_files_individual
# ============================================================================


def test_upload_files_individual_success_uploads_dataset_and_aux_files(uploader: FileUploader):
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")
    ssh_client.upload_file.return_value = (True, "")

    result = uploader._upload_files_individual(
        ssh_client,
        dataset_files=[(DATASET_CHAT_FIXTURE, "data/sft/test_chat.jsonl")],
        config_path=CONFIG_FIXTURE,
    )

    assert result.is_ok()
    # dataset upload + config upload (only existing local files)
    assert ssh_client.upload_file.call_count >= 1


def test_upload_files_individual_failure_and_warning_paths(uploader: FileUploader):
    """First upload_file fails -> error; subsequent warnings non-fatal."""
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "", "")
    ssh_client.upload_file.return_value = (False, "boom")

    result = uploader._upload_files_individual(
        ssh_client,
        dataset_files=[(DATASET_CHAT_FIXTURE, "data/sft/test_chat.jsonl")],
        config_path=CONFIG_FIXTURE,
    )

    assert result.is_err()
    assert "Failed to upload dataset" in str(result.unwrap_err())


# ============================================================================
# misc
# ============================================================================


def test_set_workspace_propagates(uploader: FileUploader):
    uploader.set_workspace("/tmp/run_42")
    assert uploader.workspace == "/tmp/run_42"


def test_get_training_path_basename_with_strategy(uploader: FileUploader):
    assert uploader._get_training_path("data/datasets/train.jsonl", "sft") == "data/sft/train.jsonl"
    assert uploader._get_training_path("/abs/path/my.jsonl", "dpo") == "data/dpo/my.jsonl"

"""Unit tests for src.pipeline.stages.managers.deployment.code_syncer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from ryotenkai_engines.vllm.config import VLLMEngineConfig

from ryotenkai_control.pipeline.stages.managers.deployment.code_syncer import CodeSyncer
from ryotenkai_shared.config import (
    DatasetConfig,
    DatasetLocalPaths,
    DatasetSourceLocal,
    GlobalHyperparametersConfig,
    InferenceConfig,
    ModelConfig,
    PipelineConfig,
    QLoRAConfig,
    TrainingOnlyConfig,
)
from ryotenkai_shared.errors import SSHTransferFailedError

pytestmark = pytest.mark.unit


DATASET_CHAT_FIXTURE = "src/tests/fixtures/datasets/test_chat.jsonl"

SINGLE_NODE_PROVIDER_CFG: dict = {
    "connect": {"ssh": {"alias": "pc"}},
    "training": {"workspace_path": "/tmp/workspace"},
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
            adapter=QLoRAConfig(
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
                source=DatasetSourceLocal(local_paths=DatasetLocalPaths(train=DATASET_CHAT_FIXTURE, eval=None)),
            )
        },
        inference=InferenceConfig(
            enabled=False,
            provider="single_node",
            engine=VLLMEngineConfig(),
        ),
    )


@pytest.fixture
def syncer(base_config: PipelineConfig, secrets: DummySecrets) -> CodeSyncer:
    s = CodeSyncer(config=base_config, secrets=secrets)
    s.set_workspace("/workspace")
    return s


def test_sync_tar_fallback_failure_is_returned(syncer: CodeSyncer):
    """Phase A2 Batch 9 (raise-based): tar-fallback failure raises
    :class:`SSHTransferFailedError` rather than returning ``Err``."""
    ssh_client = MagicMock()
    ssh_client._is_alias_mode = True
    ssh_client.ssh_target = "pc"
    ssh_client.key_path = ""
    ssh_client.port = 22
    ssh_client.exec_command.return_value = (True, "OK", "")

    failing = SimpleNamespace(returncode=1, stdout="", stderr="rsync failed")

    def _raise_tar_failure(*args, **kwargs):
        raise SSHTransferFailedError(detail="tar failed", context={"reason": "TAR_FAILED"})

    with (
        patch("ryotenkai_control.pipeline.stages.managers.deployment.code_syncer.subprocess.run", return_value=failing),
        patch.object(syncer, "_sync_module_tar", side_effect=_raise_tar_failure),
        pytest.raises(SSHTransferFailedError) as exc_info,
    ):
        syncer.sync(ssh_client)

    assert "tar failed" in str(exc_info.value)


def test_set_workspace_propagates():
    syncer = CodeSyncer(config=MagicMock(), secrets=MagicMock())
    assert syncer.workspace == "/workspace"
    syncer.set_workspace("/tmp/run_42")
    assert syncer.workspace == "/tmp/run_42"


# ---------------------------------------------------------------------------
# Thin-image migration: src/runner is now an rsync target
# ---------------------------------------------------------------------------


def test_excludes_cover_dev_and_test_artefacts() -> None:
    """The "ship everything" policy is only safe if the exclude list
    keeps real noise off the pod: tests, byte-caches, and markdown.
    Drift-guard for the four critical patterns.
    """
    expected = {"__pycache__", "*.pyc", "tests", "*.md"}
    assert expected.issubset(set(CodeSyncer.EXCLUDE_PATTERNS))


# ---------------------------------------------------------------------------
# Phase A2 Batch 9 — raise-based contract coverage
# ---------------------------------------------------------------------------


def test_sync_success_returns_none(syncer: CodeSyncer):
    """Positive: a green rsync run returns ``None`` (no Result wrapper)."""
    ssh_client = MagicMock()
    ssh_client._is_alias_mode = True
    ssh_client.ssh_target = "pc"
    ssh_client.key_path = ""
    ssh_client.port = 22
    ssh_client.exec_command.return_value = (True, "OK", "")

    completed = SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch(
        "ryotenkai_control.pipeline.stages.managers.deployment.code_syncer.subprocess.run",
        return_value=completed,
    ):
        result = syncer.sync(ssh_client)

    assert result is None


def test_sync_tar_fallback_path_raises_ssh_transfer_failed_with_context(syncer: CodeSyncer):
    """Boundary: the raised exception carries the canonical
    ``reason``/``local``/``dest`` context for diagnostic surfacing."""
    ssh_client = MagicMock()
    ssh_client._is_alias_mode = True
    ssh_client.ssh_target = "pc"
    ssh_client.key_path = ""
    ssh_client.port = 22
    # First call: mkdir -p; subsequent: verify dirs missing → raise.
    ssh_client.exec_command.return_value = (False, "", "")

    failing = SimpleNamespace(returncode=1, stdout="", stderr="rsync failed")

    with (
        patch(
            "ryotenkai_control.pipeline.stages.managers.deployment.code_syncer.subprocess.run",
            return_value=failing,
        ),
        pytest.raises(SSHTransferFailedError) as exc_info,
    ):
        syncer.sync(ssh_client)

    assert "Failed to sync" in str(exc_info.value)
    assert exc_info.value.context.get("tar_returncode") == 1
    assert "remote_dir" in exc_info.value.context

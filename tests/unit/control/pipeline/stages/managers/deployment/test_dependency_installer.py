"""Unit tests for src.pipeline.stages.managers.deployment.dependency_installer."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from ryotenkai_engines.vllm.config import VLLMEngineConfig

from ryotenkai_control.pipeline.stages.managers.deployment.dependency_installer import DependencyInstaller
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
from ryotenkai_shared.errors import ProviderUnavailableError, SSHExecFailedError

pytestmark = pytest.mark.unit

SINGLE_NODE_PROVIDER_CFG: dict[str, Any] = {
    "connect": {"ssh": {"alias": "pc"}},
    "training": {"workspace_path": "/tmp/workspace"},
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
def installer(base_config: PipelineConfig, secrets: DummySecrets) -> DependencyInstaller:
    return DependencyInstaller(config=base_config, secrets=secrets)


def test_install_single_node_uses_runtime_image_verify(installer: DependencyInstaller):
    """single_node: install() must verify runtime image via docker on host."""
    ssh_client = SimpleNamespace()

    with patch.object(installer, "_verify_single_node_docker_runtime", return_value=None) as mock_verify:
        result = installer.install(ssh_client)

    # Phase A2 Batch 9 (raise-based): success returns None.
    assert result is None
    mock_verify.assert_called_once()


def test_install_cloud_verify_ok_skips_install(base_config: PipelineConfig, secrets: DummySecrets):
    """runpod/cloud: if deps already present in image, we only verify."""
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    installer = DependencyInstaller(config=cfg, secrets=secrets)
    ssh_client = SimpleNamespace()

    with patch.object(DependencyInstaller, "verify_prebuilt_dependencies", return_value=None) as mock_verify:
        result = installer.install(ssh_client)

    assert result is None
    mock_verify.assert_called_once()


def test_install_cloud_verify_fail_raises_ssh_exec_failed(
    base_config: PipelineConfig, secrets: DummySecrets,
):
    """runpod/cloud: if deps missing in the image, we FAIL (no fallback install).

    Phase A2 Batch 9 (raise-based): a failed runtime contract check
    in the cloud path is re-wrapped as :class:`SSHExecFailedError`
    with ``reason="RUNTIME_DEPS_MISSING"`` so observability stays
    distinct from a transport-level SSH exec failure.
    """
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    installer = DependencyInstaller(config=cfg, secrets=secrets)
    ssh_client = SimpleNamespace()

    def _raise_underlying(*args, **kwargs):
        raise SSHExecFailedError(
            detail="missing packages",
            context={"reason": "RUNTIME_CONTRACT_CHECK_FAILED", "output": "ImportError"},
        )

    with (
        patch.object(
            DependencyInstaller,
            "verify_prebuilt_dependencies",
            side_effect=_raise_underlying,
        ),
        pytest.raises(SSHExecFailedError) as exc_info,
    ):
        installer.install(ssh_client)

    assert exc_info.value.context.get("reason") == "RUNTIME_DEPS_MISSING"
    assert exc_info.value.context.get("underlying_reason") == "RUNTIME_CONTRACT_CHECK_FAILED"
    # ``cause`` chain preserves the original exception.
    assert isinstance(exc_info.value.__cause__, SSHExecFailedError)


def test_verify_prebuilt_dependencies_success():
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "OK\nversion=1.0", "")

    # Phase A2 Batch 9: returns None on success.
    assert DependencyInstaller.verify_prebuilt_dependencies(ssh_client) is None


def test_verify_prebuilt_dependencies_failure_raises_ssh_exec_failed():
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (False, "", "ImportError")

    with pytest.raises(SSHExecFailedError) as exc_info:
        DependencyInstaller.verify_prebuilt_dependencies(ssh_client)
    assert "Runtime contract check failed" in str(exc_info.value)
    assert exc_info.value.context.get("reason") == "RUNTIME_CONTRACT_CHECK_FAILED"


@pytest.mark.xfail(
    strict=True,
    reason="xfail-debt:dependency-installer-attr-drift — Pre-existing failure pre-packagization: dependency_installer module attribute access drifted (legacy patched attribute removed).",
)
def test_verify_single_node_docker_runtime_no_image_returns_config_error(installer: DependencyInstaller):
    """_verify_single_node_docker_runtime returns ConfigError when docker_image is absent."""
    ssh_client = SimpleNamespace()

    with patch(
        "ryotenkai_control.pipeline.stages.managers.deployment.dependency_installer.get_single_node_training_cfg",
        return_value={"workspace_path": "/tmp/w"},
    ):
        result = installer._verify_single_node_docker_runtime(ssh_client)

    assert result.is_err()
    err = result.unwrap_err()
    assert "docker_image is required" in str(err)


def test_verify_single_node_docker_runtime_pull_failure_propagates(installer: DependencyInstaller):
    """Phase A2 Batch 9: pull failure propagates the underlying
    :class:`ProviderUnavailableError` directly (no Result wrapping)."""
    ssh_client = SimpleNamespace()
    pull_err = ProviderUnavailableError(
        detail="pull failed",
        context={"reason": "DOCKER_PULL_FAILED"},
    )

    with (
        patch.object(installer, "_ensure_docker_image_present", side_effect=pull_err),
        pytest.raises(ProviderUnavailableError) as exc_info,
    ):
        installer._verify_single_node_docker_runtime(ssh_client)

    assert "pull failed" in str(exc_info.value)
    assert exc_info.value.context.get("reason") == "DOCKER_PULL_FAILED"


def test_verify_single_node_docker_runtime_check_failed_no_ok_in_stdout(installer: DependencyInstaller):
    """Phase A2 Batch 9 (raise-based): contract check failure raises
    :class:`SSHExecFailedError`."""
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "no marker", "")

    with (
        patch.object(installer, "_ensure_docker_image_present", return_value=None),
        pytest.raises(SSHExecFailedError) as exc_info,
    ):
        installer._verify_single_node_docker_runtime(ssh_client)

    assert "missing required packages" in str(exc_info.value)
    assert exc_info.value.context.get("reason") == "DOCKER_RUNTIME_CHECK_FAILED"


def test_verify_single_node_docker_runtime_check_failed_exec_returns_false(installer: DependencyInstaller):
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (False, "", "")

    with (
        patch.object(installer, "_ensure_docker_image_present", return_value=None),
        pytest.raises(SSHExecFailedError),
    ):
        installer._verify_single_node_docker_runtime(ssh_client)


def test_set_workspace_propagates(installer: DependencyInstaller):
    installer.set_workspace("/tmp/run_42")
    assert installer.workspace == "/tmp/run_42"


# ---------------------------------------------------------------------------
# Phase A2 Batch 9 — additional 7-class coverage for raise contract
# ---------------------------------------------------------------------------


def test_verify_single_node_docker_runtime_success_returns_none(installer: DependencyInstaller):
    """Positive: when pull + contract check both succeed, returns ``None``."""
    ssh_client = MagicMock()
    ssh_client.exec_command.return_value = (True, "OK\nversion=1.0", "")

    with patch.object(installer, "_ensure_docker_image_present", return_value=None):
        assert installer._verify_single_node_docker_runtime(ssh_client) is None


def test_install_cloud_verify_chains_cause_for_diagnosis(
    base_config: PipelineConfig, secrets: DummySecrets,
):
    """Invariant: the re-raised :class:`SSHExecFailedError` chains the
    original via ``__cause__`` so a traceback walker can find both."""
    cfg = base_config.model_copy(deep=True)
    cfg.training.provider = "runpod"
    cfg.providers = {"runpod": RUNPOD_PROVIDER_CFG}

    installer = DependencyInstaller(config=cfg, secrets=secrets)
    ssh_client = SimpleNamespace()

    underlying = SSHExecFailedError(
        detail="missing flash-attn",
        context={"reason": "RUNTIME_CONTRACT_CHECK_FAILED", "output": "no flash-attn module"},
    )

    with (
        patch.object(
            DependencyInstaller,
            "verify_prebuilt_dependencies",
            side_effect=underlying,
        ),
        pytest.raises(SSHExecFailedError) as exc_info,
    ):
        installer.install(ssh_client)

    assert exc_info.value.__cause__ is underlying
    assert exc_info.value.context.get("output") == "no flash-attn module"


def test_install_single_node_propagates_provider_unavailable(installer: DependencyInstaller):
    """Dependency error: a docker-pull ``ProviderUnavailableError``
    propagates verbatim through ``install()`` (no re-wrapping)."""
    ssh_client = SimpleNamespace()
    pull_err = ProviderUnavailableError(
        detail="docker daemon down",
        context={"reason": "DOCKER_PULL_FAILED"},
    )

    with (
        patch.object(installer, "_ensure_docker_image_present", side_effect=pull_err),
        pytest.raises(ProviderUnavailableError) as exc_info,
    ):
        installer.install(ssh_client)

    assert exc_info.value is pull_err

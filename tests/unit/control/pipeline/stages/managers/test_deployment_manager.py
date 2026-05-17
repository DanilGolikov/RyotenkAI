"""Facade smoke tests for TrainingDeploymentManager.

After Wave 3 decomposition the deployment concern is split into four
components living under ``deployment/``. This file keeps only smoke
tests for the facade itself: instantiation wiring, workspace
propagation, and that the public API delegates to the right
components.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from ryotenkai_engines.vllm.config import VLLMEngineConfig

from ryotenkai_control.pipeline.stages.managers.deployment.code_syncer import CodeSyncer
from ryotenkai_control.pipeline.stages.managers.deployment.dependency_installer import DependencyInstaller
from ryotenkai_control.pipeline.stages.managers.deployment.file_uploader import FileUploader
from ryotenkai_control.pipeline.stages.managers.deployment.training_launcher import TrainingLauncher
from ryotenkai_control.pipeline.stages.managers.deployment_manager import TrainingDeploymentManager
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
from ryotenkai_shared.errors import PipelineStageFailedError, SSHTransferFailedError

pytestmark = pytest.mark.unit


DATASET_CHAT_FIXTURE = "src/tests/fixtures/datasets/test_chat.jsonl"

SINGLE_NODE_PROVIDER_CFG: dict[str, Any] = {
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
def manager(base_config: PipelineConfig, secrets: DummySecrets) -> TrainingDeploymentManager:
    return TrainingDeploymentManager(config=base_config, secrets=secrets)


def test_manager_constructs_all_four_components(manager: TrainingDeploymentManager):
    """Facade owns one of each component, all wired through the same config."""
    assert isinstance(manager._code_syncer, CodeSyncer)
    assert isinstance(manager._file_uploader, FileUploader)
    assert isinstance(manager._deps_installer, DependencyInstaller)
    assert isinstance(manager._launcher, TrainingLauncher)
    # Phase 3 PR-3.3 (transport-unification-v2): the cross-component
    # FileUploader↔CodeSyncer DI is gone — file upload is HTTP and
    # has no SSH-side companion. The launcher now receives the
    # FileUploader and the deps_installer.
    assert manager._launcher._deps_installer is manager._deps_installer
    assert manager._launcher._file_uploader is manager._file_uploader


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


def test_deploy_code_delegates_to_code_syncer(manager: TrainingDeploymentManager):
    """Phase 3 PR-3.3: the legacy ``deploy_files`` (chain of file
    upload + code sync) is replaced with ``deploy_code`` — pure
    rsync, pre-launch. File upload is HTTP, called from
    ``start_training`` after /healthz.

    Phase A2 Batch 9 (raise-based): success returns ``None``.
    """
    ssh = SimpleNamespace()
    with patch.object(manager._code_syncer, "sync", return_value=None) as mock:
        result = manager.deploy_code(ssh)
    assert result is None
    mock.assert_called_once_with(ssh)


def test_install_dependencies_delegates_to_deps_installer(manager: TrainingDeploymentManager):
    ssh = SimpleNamespace()
    with patch.object(manager._deps_installer, "install", return_value=None) as mock:
        result = manager.install_dependencies(ssh)
    assert result is None
    mock.assert_called_once_with(ssh)


def test_start_training_delegates_to_launcher(manager: TrainingDeploymentManager):
    ssh = SimpleNamespace()
    ctx: dict[str, Any] = {}
    with patch.object(manager._launcher, "start_training", return_value={"mode": "docker"}) as mock:
        result = manager.start_training(ssh, ctx)
    assert result == {"mode": "docker"}
    mock.assert_called_once_with(ssh, ctx, None)


# ---------------------------------------------------------------------------
# Phase A2 Batch 9 — typed-exception propagation through the facade
# ---------------------------------------------------------------------------


def test_deploy_code_propagates_ssh_transfer_failed(manager: TrainingDeploymentManager):
    """Negative: typed exceptions from CodeSyncer propagate verbatim
    through the facade (no Result wrapping, no re-classification)."""
    ssh = SimpleNamespace()
    typed = SSHTransferFailedError(detail="rsync failed", context={"reason": "TAR_FAILED"})
    with (
        patch.object(manager._code_syncer, "sync", side_effect=typed),
        pytest.raises(SSHTransferFailedError) as exc_info,
    ):
        manager.deploy_code(ssh)
    assert exc_info.value is typed


def test_deploy_code_wraps_unexpected_exception(manager: TrainingDeploymentManager):
    """Boundary: a non-typed exception escapes are re-wrapped as
    :class:`PipelineStageFailedError` so the upstream stage-execution
    loop always sees a typed error (no untyped escape)."""
    ssh = SimpleNamespace()
    with (
        patch.object(manager._code_syncer, "sync", side_effect=RuntimeError("oops")),
        pytest.raises(PipelineStageFailedError) as exc_info,
    ):
        manager.deploy_code(ssh)
    assert exc_info.value.context.get("reason") == "CODE_SYNC_FAILED"
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_install_dependencies_wraps_unexpected_exception(manager: TrainingDeploymentManager):
    ssh = SimpleNamespace()
    with (
        patch.object(manager._deps_installer, "install", side_effect=RuntimeError("oops")),
        pytest.raises(PipelineStageFailedError) as exc_info,
    ):
        manager.install_dependencies(ssh)
    assert exc_info.value.context.get("reason") == "DEPS_INSTALL_FAILED"


def test_start_training_wraps_unexpected_exception(manager: TrainingDeploymentManager):
    ssh = SimpleNamespace()
    with (
        patch.object(manager._launcher, "start_training", side_effect=RuntimeError("oops")),
        pytest.raises(PipelineStageFailedError) as exc_info,
    ):
        manager.start_training(ssh, {})
    assert exc_info.value.context.get("reason") == "TRAINING_START_FAILED"


# ---------------------------------------------------------------------------
# Phase 5 — typed event emission for the deployment-manager surface
# ---------------------------------------------------------------------------


class TestPhase5EventEmission:
    """Pin the Phase 5 contract: a successful ``deploy_code`` emits
    ``ryotenkai.control.gpu.code_synced``; failures do not.
    """

    def test_deploy_code_success_emits_code_synced(
        self, manager: TrainingDeploymentManager,
    ) -> None:
        from tests._fakes.event_emitter import FakeEventEmitter

        emitter = FakeEventEmitter()
        manager.set_emitter(emitter)
        manager.set_run_id("run-42")
        ssh = SimpleNamespace()
        with patch.object(manager._code_syncer, "sync", return_value=None):
            manager.deploy_code(ssh)
        kinds = [ev.kind for ev in emitter.emitted]
        assert "ryotenkai.control.gpu.code_synced" in kinds
        ev = next(
            e for e in emitter.emitted
            if e.kind == "ryotenkai.control.gpu.code_synced"
        )
        assert ev.run_id == "run-42"
        # Severity is the typed default — pin it as an invariant.
        assert ev.severity == "info"

    def test_deploy_code_failure_emits_no_envelope(
        self, manager: TrainingDeploymentManager,
    ) -> None:
        """Negative: rsync failure → no code_synced envelope; the
        error propagates as the typed exception."""
        from tests._fakes.event_emitter import FakeEventEmitter

        emitter = FakeEventEmitter()
        manager.set_emitter(emitter)
        manager.set_run_id("run-42")
        ssh = SimpleNamespace()
        typed_err = SSHTransferFailedError(
            detail="rsync failed", context={"reason": "TAR_FAILED"},
        )
        with (
            patch.object(manager._code_syncer, "sync", side_effect=typed_err),
            pytest.raises(SSHTransferFailedError),
        ):
            manager.deploy_code(ssh)
        assert not any(
            ev.kind == "ryotenkai.control.gpu.code_synced"
            for ev in emitter.emitted
        )

    def test_no_emit_when_emitter_absent(
        self, manager: TrainingDeploymentManager,
    ) -> None:
        """Legacy path: no emitter wired → no exception, no envelope.
        Mirrors the contract documented on the constructor.
        """
        ssh = SimpleNamespace()
        with patch.object(manager._code_syncer, "sync", return_value=None):
            manager.deploy_code(ssh)
        # No emitter attached → nothing to assert about, just ensure
        # the method completed cleanly. Re-call set_emitter with None
        # to pin the no-emit contract explicitly.
        manager.set_emitter(None)  # type: ignore[arg-type]
        with patch.object(manager._code_syncer, "sync", return_value=None):
            manager.deploy_code(ssh)

    def test_set_run_id_rejects_empty_value(
        self, manager: TrainingDeploymentManager,
    ) -> None:
        """Pin invariant: empty / non-string values silently retain
        the previous cached run_id. The caller can't accidentally
        downgrade a real run_id to ``""``.
        """
        manager.set_run_id("run-42")
        manager.set_run_id("")  # ignored
        manager.set_run_id(None)  # type: ignore[arg-type]
        assert manager._cached_run_id == "run-42"

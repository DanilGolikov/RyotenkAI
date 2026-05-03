"""
Tests for the GPUDeployer stage.

Covers:
- Deployer initialization
- Successful deploy workflow
- Error handling at each step
- Callbacks integration
- Cleanup and disconnect

Approach: integration-style tests with mocked provider and deployment manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.state import RunContext
from src.pipeline.stages.gpu_deployer import GPUDeployer, GPUDeployerEventCallbacks
from src.utils.result import Failure, ProviderError, Success

# =========================================================================
# HELPER CLASSES
# =========================================================================


@dataclass
class MockSSHInfo:
    """Mock SSH connection info."""

    host: str = "test.host.com"
    port: int = 22
    user: str = "test_user"
    key_path: str = "/path/to/key"
    workspace_path: str = "/workspace"
    resource_id: str = "test_resource_123"
    is_alias_mode: bool = False


# =========================================================================
# FIXTURES
# =========================================================================


@pytest.fixture
def mock_config_with_gpu():
    """Config with GPU provider settings."""
    config = MagicMock()

    # Provider config
    config.get_active_provider_name.return_value = "runpod"

    mock_provider_config = MagicMock()
    mock_provider_config.gpu_type = "NVIDIA RTX 4090"
    mock_provider_config.container_disk_gb = 50
    config.get_provider_config.return_value = mock_provider_config

    return config


@pytest.fixture
def mock_secrets():
    """Mock secrets with API keys."""
    secrets = MagicMock()
    secrets.runpod_api_key = "test_runpod_key"
    secrets.hf_token = "test_hf_token"
    return secrets


@pytest.fixture
def mock_callbacks():
    """Mock callbacks for tests."""
    return GPUDeployerEventCallbacks(
        on_provider_created=MagicMock(),
        on_connected=MagicMock(),
        on_files_uploaded=MagicMock(),
        on_deps_installed=MagicMock(),
        on_training_started=MagicMock(),
        on_error=MagicMock(),
        on_cleanup=MagicMock(),
    )


@pytest.fixture
def mock_provider():
    """Mock GPU provider."""
    provider = MagicMock()
    provider.provider_type = "cloud"
    provider.get_base_workspace.return_value = "/workspace"

    # Default: successful connect
    ssh_info = MockSSHInfo()
    provider.connect.return_value = Success(ssh_info)
    provider.disconnect.return_value = None

    return provider


@pytest.fixture
def mock_deployment_manager():
    """Mock deployment manager."""
    manager = MagicMock()

    # Default: successful operations
    manager.deploy_files.return_value = Success(None)
    manager.install_dependencies.return_value = Success(None)
    manager.start_training.return_value = Success({"mode": "venv"})

    return manager


@pytest.fixture
def run_ctx() -> RunContext:
    """Deterministic run context for stage tests."""
    return RunContext(
        name="run_20260120_123456_abc12",
        created_at_utc=datetime(2026, 1, 20, 12, 34, 56, tzinfo=timezone.utc),
    )


# =========================================================================
# INITIALIZATION TESTS
# =========================================================================


@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_deployer_initialization(mock_tdm_class, mock_config_with_gpu, mock_secrets):
    """GPUDeployer initialization."""
    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets)

    assert deployer.config == mock_config_with_gpu
    assert deployer.secrets == mock_secrets
    assert deployer.stage_name == "GPU Deployer"
    assert deployer._provider_name == "runpod"
    assert deployer._provider_config is not None
    assert deployer._provider is None  # Not created until execute

    # Deployment manager constructed
    mock_tdm_class.assert_called_once_with(config=mock_config_with_gpu, secrets=mock_secrets)


@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_deployer_with_callbacks(mock_tdm_class, mock_config_with_gpu, mock_secrets, mock_callbacks):
    """Initialization with custom callbacks."""
    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)

    assert deployer._callbacks is mock_callbacks


# =========================================================================
# SUCCESSFUL WORKFLOW TESTS
# =========================================================================


@patch("src.pipeline.stages.gpu_deployer.SSHClient")
@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_execute_successful_deployment(
    mock_tdm_class,
    mock_factory,
    mock_ssh_class,
    mock_config_with_gpu,
    mock_secrets,
    mock_provider,
    mock_deployment_manager,
    run_ctx,
):
    """Successful full deployment."""
    # Setup mocks
    mock_factory.create.return_value = Success(mock_provider)
    mock_tdm_class.return_value = mock_deployment_manager

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets)
    context = {"run": run_ctx}

    result = deployer.execute(context)

    # Assert success
    assert result.is_ok()

    # All expected calls happened
    mock_factory.create.assert_called_once()
    mock_provider.connect.assert_called_once_with(run=run_ctx)
    mock_deployment_manager.set_workspace.assert_called_once()
    mock_deployment_manager.deploy_files.assert_called_once()
    mock_deployment_manager.install_dependencies.assert_called_once()
    mock_deployment_manager.start_training.assert_called_once()


@patch("src.pipeline.stages.gpu_deployer.SSHClient")
@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_execute_returns_correct_context(
    mock_tdm_class,
    mock_factory,
    mock_ssh_class,
    mock_config_with_gpu,
    mock_secrets,
    mock_provider,
    mock_deployment_manager,
    run_ctx,
):
    """execute() returns the expected context shape."""
    mock_factory.create.return_value = Success(mock_provider)
    mock_tdm_class.return_value = mock_deployment_manager

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets)
    context = {"run": run_ctx}

    result = deployer.execute(context)

    assert result.is_ok()
    result_data = result.unwrap()

    # Key fields present
    gpu_deployer_data = result_data.get("GPU Deployer", {})
    assert "provider_name" in gpu_deployer_data
    assert "provider_type" in gpu_deployer_data
    assert "resource_id" in gpu_deployer_data
    assert "ssh_host" in gpu_deployer_data
    assert "ssh_port" in gpu_deployer_data
    assert "workspace_path" in gpu_deployer_data
    assert "upload_duration_seconds" in gpu_deployer_data
    assert "deps_duration_seconds" in gpu_deployer_data

    # Values
    assert gpu_deployer_data["provider_name"] == "runpod"
    assert gpu_deployer_data["ssh_host"] == "test.host.com"
    assert gpu_deployer_data["ssh_port"] == 22


# =========================================================================
# ERROR HANDLING TESTS
# =========================================================================


@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_execute_provider_creation_failure(
    mock_tdm_class, mock_factory, mock_config_with_gpu, mock_secrets, mock_callbacks, run_ctx
):
    """Provider creation error."""
    mock_factory.create.return_value = Failure(ProviderError(message="Invalid provider config", code="PROVIDER_NOT_REGISTERED"))

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)
    context = {"run": run_ctx}

    result = deployer.execute(context)

    # Error asserted
    assert result.is_err()
    assert "Invalid provider config" in str(result.unwrap_err())

    # on_error callback
    mock_callbacks.on_error.assert_called_once()
    call_args = mock_callbacks.on_error.call_args[0]
    assert call_args[0] == "provider_create"


@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_execute_connection_failure(
    mock_tdm_class, mock_factory, mock_config_with_gpu, mock_secrets, mock_provider, mock_callbacks, run_ctx
):
    """Connection error."""
    # Mock connect returns Failure
    mock_provider.connect.return_value = Failure(ProviderError(message="Connection timeout", code="SSH_CONNECTION_FAILED"))
    mock_factory.create.return_value = Success(mock_provider)

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)
    context = {"run": run_ctx}

    result = deployer.execute(context)

    assert result.is_err()
    assert "Connection failed" in str(result.unwrap_err())

    # Callback asserted
    mock_callbacks.on_error.assert_called_once()
    call_args = mock_callbacks.on_error.call_args[0]
    assert call_args[0] == "connect"
    assert "Connection timeout" in call_args[1]


@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_execute_ssh_info_none(
    mock_tdm_class, mock_factory, mock_config_with_gpu, mock_secrets, mock_provider, mock_callbacks, run_ctx
):
    """connect returns None SSH info."""
    # Mock connect returns Success(None)
    mock_provider.connect.return_value = Success(None)
    mock_factory.create.return_value = Success(mock_provider)

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)
    context = {"run": run_ctx}

    result = deployer.execute(context)

    assert result.is_err()
    assert "SSH info is None" in str(result.unwrap_err())

    # Callback asserted
    mock_callbacks.on_error.assert_called_once()


@patch("src.pipeline.stages.gpu_deployer.SSHClient")
@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_execute_upload_failure(
    mock_tdm_class,
    mock_factory,
    mock_ssh_class,
    mock_config_with_gpu,
    mock_secrets,
    mock_provider,
    mock_deployment_manager,
    mock_callbacks,
    run_ctx,
):
    """File upload error."""
    mock_factory.create.return_value = Success(mock_provider)

    # Mock upload failure
    mock_deployment_manager.deploy_files.return_value = Failure("Upload failed: disk full")
    mock_tdm_class.return_value = mock_deployment_manager

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)
    context = {"run": run_ctx}

    result = deployer.execute(context)

    assert result.is_err()

    # disconnect was called
    mock_provider.disconnect.assert_called_once()

    # Callback asserted
    mock_callbacks.on_error.assert_called()


@patch("src.pipeline.stages.gpu_deployer.SSHClient")
@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_execute_deps_failure(
    mock_tdm_class,
    mock_factory,
    mock_ssh_class,
    mock_config_with_gpu,
    mock_secrets,
    mock_provider,
    mock_deployment_manager,
    mock_callbacks,
    run_ctx,
):
    """Dependency install error."""
    mock_factory.create.return_value = Success(mock_provider)

    # Mock deps failure
    mock_deployment_manager.install_dependencies.return_value = Failure("pip install failed")
    mock_tdm_class.return_value = mock_deployment_manager

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)
    context = {"run": run_ctx}

    result = deployer.execute(context)

    assert result.is_err()
    mock_provider.disconnect.assert_called_once()
    mock_callbacks.on_error.assert_called()


@patch("src.pipeline.stages.gpu_deployer.SSHClient")
@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_execute_training_start_failure(
    mock_tdm_class,
    mock_factory,
    mock_ssh_class,
    mock_config_with_gpu,
    mock_secrets,
    mock_provider,
    mock_deployment_manager,
    mock_callbacks,
    run_ctx,
):
    """Training start error."""
    mock_factory.create.return_value = Success(mock_provider)

    # Mock training start failure
    mock_deployment_manager.start_training.return_value = Failure("Failed to start training script")
    mock_tdm_class.return_value = mock_deployment_manager

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)
    context = {"run": run_ctx}

    result = deployer.execute(context)

    assert result.is_err()
    assert "Failed to start training script" in str(result.unwrap_err())

    mock_provider.disconnect.assert_called_once()
    mock_callbacks.on_error.assert_called()


# =========================================================================
# CALLBACK TESTS
# =========================================================================


@patch("src.pipeline.stages.gpu_deployer.SSHClient")
@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_callbacks_on_provider_created(
    mock_tdm_class,
    mock_factory,
    mock_ssh_class,
    mock_config_with_gpu,
    mock_secrets,
    mock_provider,
    mock_deployment_manager,
    mock_callbacks,
    run_ctx,
):
    """on_provider_created callback."""
    mock_factory.create.return_value = Success(mock_provider)
    mock_tdm_class.return_value = mock_deployment_manager

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)

    result = deployer.execute({"run": run_ctx})

    assert result.is_ok()

    # Callback asserted
    mock_callbacks.on_provider_created.assert_called_once()
    call_args = mock_callbacks.on_provider_created.call_args[0]
    assert call_args[0] == "runpod"  # provider_name
    assert call_args[1] == "cloud"  # provider_type


@patch("src.pipeline.stages.gpu_deployer.SSHClient")
@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_callbacks_on_connected(
    mock_tdm_class,
    mock_factory,
    mock_ssh_class,
    mock_config_with_gpu,
    mock_secrets,
    mock_provider,
    mock_deployment_manager,
    mock_callbacks,
    run_ctx,
):
    """on_connected callback."""
    mock_factory.create.return_value = Success(mock_provider)
    mock_tdm_class.return_value = mock_deployment_manager

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)

    result = deployer.execute({"run": run_ctx})

    assert result.is_ok()

    # Callback asserted
    mock_callbacks.on_connected.assert_called_once()
    call_args = mock_callbacks.on_connected.call_args[0]
    assert call_args[0] == "runpod"
    assert call_args[1] == "test.host.com"  # host
    assert call_args[2] == 22  # port


@patch("src.pipeline.stages.gpu_deployer.SSHClient")
@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_callbacks_on_files_uploaded(
    mock_tdm_class,
    mock_factory,
    mock_ssh_class,
    mock_config_with_gpu,
    mock_secrets,
    mock_provider,
    mock_deployment_manager,
    mock_callbacks,
    run_ctx,
):
    """on_files_uploaded callback."""
    mock_factory.create.return_value = Success(mock_provider)
    mock_tdm_class.return_value = mock_deployment_manager

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)

    result = deployer.execute({"run": run_ctx})

    assert result.is_ok()

    # Callback asserted
    mock_callbacks.on_files_uploaded.assert_called_once()
    # Argument: duration_seconds (float)
    duration = mock_callbacks.on_files_uploaded.call_args[0][0]
    assert isinstance(duration, float)
    assert duration >= 0


@patch("src.pipeline.stages.gpu_deployer.SSHClient")
@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_callbacks_on_training_started(
    mock_tdm_class,
    mock_factory,
    mock_ssh_class,
    mock_config_with_gpu,
    mock_secrets,
    mock_provider,
    mock_deployment_manager,
    mock_callbacks,
    run_ctx,
):
    """on_training_started callback."""
    mock_factory.create.return_value = Success(mock_provider)
    mock_tdm_class.return_value = mock_deployment_manager

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)

    result = deployer.execute({"run": run_ctx})

    assert result.is_ok()

    # Callback asserted
    mock_callbacks.on_training_started.assert_called_once()
    resource_id = mock_callbacks.on_training_started.call_args[0][0]
    assert resource_id == "test_resource_123"


# =========================================================================
# CLEANUP TESTS
# =========================================================================


@patch("src.pipeline.stages.gpu_deployer.SSHClient")
@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_cleanup_disconnects_provider(
    mock_tdm_class,
    mock_factory,
    mock_ssh_class,
    mock_config_with_gpu,
    mock_secrets,
    mock_provider,
    mock_deployment_manager,
    mock_callbacks,
    run_ctx,
):
    """cleanup() invokes disconnect."""
    mock_factory.create.return_value = Success(mock_provider)
    mock_tdm_class.return_value = mock_deployment_manager

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)

    # execute() to create provider
    result = deployer.execute({"run": run_ctx})
    assert result.is_ok()

    # Best-effort log download must be attempted for runpod before disconnect.
    deployer._download_remote_logs = MagicMock()  # type: ignore[method-assign]

    # Invoke cleanup
    deployer.cleanup()

    deployer._download_remote_logs.assert_called_once()  # type: ignore[attr-defined]

    # disconnect asserted
    assert mock_provider.disconnect.call_count >= 1

    # Callback asserted
    mock_callbacks.on_cleanup.assert_called_once_with("runpod")

    # Provider cleared
    assert deployer._provider is None


@patch("src.pipeline.stages.gpu_deployer.SSHClient")
@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_get_provider_returns_active_provider(
    mock_tdm_class,
    mock_factory,
    mock_ssh_class,
    mock_config_with_gpu,
    mock_secrets,
    mock_provider,
    mock_deployment_manager,
    run_ctx,
):
    """get_provider returns active provider."""
    mock_factory.create.return_value = Success(mock_provider)
    mock_tdm_class.return_value = mock_deployment_manager

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets)

    # Before execute: provider is None
    assert deployer.get_provider() is None

    # After execute: provider available
    result = deployer.execute({"run": run_ctx})
    assert result.is_ok()

    active_provider = deployer.get_provider()
    assert active_provider is mock_provider


# =========================================================================
# UNCOVERED BRANCHES
# =========================================================================


@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_execute_missing_run_context_returns_error(
    mock_tdm_class, mock_config_with_gpu, mock_secrets
):
    """Line 173: execute() without RunContext in context → ProviderError MISSING_RUN_CONTEXT."""
    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets)

    result = deployer.execute({"no_run": "here"})

    assert result.is_err()
    err = result.unwrap_err()
    assert err.code == "MISSING_RUN_CONTEXT"
    assert "RunContext" in err.message


@patch("src.pipeline.stages.gpu_deployer.SSHClient")
@patch("src.pipeline.stages.gpu_deployer.GPUProviderFactory")
@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_execute_image_sha_added_to_context(
    mock_tdm_class,
    mock_factory,
    mock_ssh_class,
    mock_config_with_gpu,
    mock_secrets,
    mock_provider,
    mock_deployment_manager,
    run_ctx,
):
    """Lines 292-294: start_training returns image_sha → docker_image_sha added to context."""
    image_sha = "sha256:abc1234567890deadbeef0000cafebabe0000feedface0000"
    mock_deployment_manager.start_training.return_value = Success(
        {"mode": "docker", "image_sha": image_sha}
    )
    mock_factory.create.return_value = Success(mock_provider)
    mock_tdm_class.return_value = mock_deployment_manager

    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets)
    context: dict = {"run": run_ctx}

    result = deployer.execute(context)

    assert result.is_ok()
    assert result.unwrap().get("docker_image_sha") == image_sha


@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_release_calls_download_logs_when_ssh_client_set(
    mock_tdm_class, mock_config_with_gpu, mock_secrets, mock_provider
):
    """Lines 346-349: release() with _ssh_client set → _download_remote_logs called."""
    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets)
    deployer._provider = mock_provider
    deployer._provider_name = "runpod"
    deployer._ssh_client = MagicMock()
    deployer._download_remote_logs = MagicMock()  # type: ignore[method-assign]

    deployer.release()

    deployer._download_remote_logs.assert_called_once_with("early_release")  # type: ignore[attr-defined]
    mock_provider.disconnect.assert_called_once()
    assert deployer._released is True


@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_release_swallows_download_logs_exception(
    mock_tdm_class, mock_config_with_gpu, mock_secrets, mock_provider
):
    """Lines 347-349: release() if _download_remote_logs raises → swallowed."""
    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets)
    deployer._provider = mock_provider
    deployer._provider_name = "runpod"
    deployer._ssh_client = MagicMock()
    deployer._download_remote_logs = MagicMock(side_effect=Exception("log download crashed"))  # type: ignore[method-assign]

    # Must not raise
    deployer.release()

    mock_provider.disconnect.assert_called_once()
    assert deployer._released is True


@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_handle_error_and_disconnect_calls_mark_error_on_provider(
    mock_tdm_class, mock_config_with_gpu, mock_secrets
):
    """Line 375: _handle_error_and_disconnect with mark_error on provider → mark_error() called."""
    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets)

    # Provider that implements _SupportsErrorMarking (has mark_error method)
    mock_provider = MagicMock()
    mock_provider.mark_error = MagicMock()
    deployer._provider = mock_provider
    deployer._ssh_client = None  # skip log download

    deployer._handle_error_and_disconnect("test error reason")

    mock_provider.mark_error.assert_called_once()
    mock_provider.disconnect.assert_called_once()


@patch("src.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_cleanup_swallows_download_logs_exception(
    mock_tdm_class, mock_config_with_gpu, mock_secrets, mock_provider
):
    """Lines 452-453: cleanup() if _download_remote_logs raises → does not propagate."""
    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets)
    deployer._provider = mock_provider
    deployer._provider_name = "runpod"
    deployer._ssh_client = MagicMock()
    deployer._download_remote_logs = MagicMock(side_effect=Exception("logs unavailable"))  # type: ignore[method-assign]

    # Must not raise
    deployer.cleanup()

    mock_provider.disconnect.assert_called_once()
    assert deployer._provider is None

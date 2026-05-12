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

from ryotenkai_shared.pipeline_context import RunContext
from ryotenkai_control.pipeline.stages.gpu_deployer import GPUDeployer, GPUDeployerEventCallbacks
from ryotenkai_shared.utils.result import Failure, ProviderError, Success

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
    manager.deploy_code.return_value = Success(None)
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


@patch("ryotenkai_control.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
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


@patch("ryotenkai_control.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
def test_deployer_with_callbacks(mock_tdm_class, mock_config_with_gpu, mock_secrets, mock_callbacks):
    """Initialization with custom callbacks."""
    deployer = GPUDeployer(config=mock_config_with_gpu, secrets=mock_secrets, callbacks=mock_callbacks)

    assert deployer._callbacks is mock_callbacks


# =========================================================================
# SUCCESSFUL WORKFLOW TESTS
# =========================================================================


# =========================================================================
# ERROR HANDLING TESTS
# =========================================================================


# =========================================================================
# CALLBACK TESTS
# =========================================================================


# =========================================================================
# CLEANUP TESTS
# =========================================================================


# =========================================================================
# UNCOVERED BRANCHES
# =========================================================================


@patch("ryotenkai_control.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
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


@patch("ryotenkai_control.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
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


@patch("ryotenkai_control.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
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


@patch("ryotenkai_control.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
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


@patch("ryotenkai_control.pipeline.stages.gpu_deployer.TrainingDeploymentManager")
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

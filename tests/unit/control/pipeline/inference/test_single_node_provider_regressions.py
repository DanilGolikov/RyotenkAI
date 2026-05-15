"""
Regression tests for SingleNodeInferenceProvider bugs.

This module tests specific bugs that were found and fixed during development:
1. Docker pull timeout (600s → 1200s)
2. Multi-line shell command with trailing backslash when trust_remote_code=False
3. Platform mismatch (ARM64 vs AMD64)
4. Docker image verification after pull (race)
5. Health check always returning Ok(bool) instead of proper error handling
6. Health check command NameError (double json.dumps quote escaping)
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from ryotenkai_providers.single_node.inference.provider import SingleNodeInferenceProvider
from ryotenkai_providers.single_node.training.config import (
    SingleNodeProviderConfig,
    SingleNodeConnectConfig,
    SingleNodeInferenceConfig,
    SingleNodeTrainingConfig,
)
from ryotenkai_shared.config import (
    InferenceSingleNodeServeConfig,
    PipelineConfig,
    Secrets,
    SSHConfig,
)
from ryotenkai_engines.vllm.config import VLLMEngineConfig


def _create_merge_mock_responses(
    *,
    container_id: str = "container123",
    merge_success: bool = True,
    exit_code: int = 0,
    num_polls: int = 1,
) -> list[tuple[bool, str, str]]:
    """
    Helper to create SSH mock responses for _run_merge_container with new polling logic.
    
    New flow:
    1. rm -rf output_path
    2. mkdir cache_dir
    3. docker run --detach (returns container ID)
    4-N. Polling loop:
       - docker ps -q (check if running)
       - docker logs (collect logs)
    N+1. docker ps -q (container stopped, returns empty)
    N+2. docker logs (final collection)
    N+3. docker inspect exit code
    N+4. docker rm -f cleanup
    N+5. verify config.json exists
    """
    responses = [
        (True, "", ""),  # rm -rf
        (True, "", ""),  # mkdir
        (True, container_id, ""),  # docker run --detach
    ]

    # Polling loop (container running)
    logs_content = "Merge in progress..."
    for i in range(num_polls):
        responses.append((True, container_id, ""))  # docker ps -q (running)
        if i == num_polls - 1 and merge_success:
            logs_content = "MERGE_SUCCESS\nCompleted"
        responses.append((True, logs_content, ""))  # docker logs

    # Container stopped
    responses.append((True, "", ""))  # docker ps -q (empty = stopped)
    responses.append((True, logs_content, ""))  # docker logs (final)
    responses.append((True, str(exit_code), ""))  # docker inspect exit code
    responses.append((True, "", ""))  # docker rm -f
    responses.append((True, "OK", ""))  # verify config.json

    return responses


@pytest.fixture
def mock_ssh_config():
    """Mock SSH configuration."""
    return SSHConfig(
        alias="test-node",
        host="192.168.1.100",
        port=22,
        user="testuser",
    )


@pytest.fixture
def mock_config(mock_ssh_config):
    """Mock provider configuration (NEW v3 structure)."""
    return SingleNodeProviderConfig(
        connect=SingleNodeConnectConfig(ssh=mock_ssh_config),
        training=SingleNodeTrainingConfig(
            workspace_path="/home/testuser/workspace",
        ),
        inference=SingleNodeInferenceConfig(
            serve=InferenceSingleNodeServeConfig(
                host="192.168.1.100",
                port=8000,
                workspace="/home/testuser/inference",
            )
        ),
    )


@pytest.fixture
def mock_engine_config():
    """Mock engine configuration."""
    return VLLMEngineConfig(
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )


@pytest.fixture
def mock_secrets():
    """Mock secrets."""
    return Secrets(hf_token="test-token-123")


@pytest.fixture(autouse=True)
def _stamp_provider_manifest_classvars():
    """Production: ProviderRegistry stamps ``_manifest_*`` ClassVars on the
    provider class before instantiation. Tests bypassing the registry need
    to do this themselves."""
    SingleNodeInferenceProvider._manifest_provider_id = "single_node"
    SingleNodeInferenceProvider._manifest_provider_name = "single_node"
    SingleNodeInferenceProvider._manifest_provider_type = "local"
    yield


@pytest.fixture
def provider(mock_config, mock_engine_config, mock_secrets):
    """Create provider instance via ProviderContext (post PR-1.5 API)."""
    from ryotenkai_providers.registry import ProviderContext

    config_dict = mock_config.model_dump(mode="python")

    mock_pipeline_config = Mock()
    mock_pipeline_config.training = Mock()
    mock_pipeline_config.training.provider = "single_node"
    mock_pipeline_config.get_provider_config = lambda *args, **kwargs: config_dict

    # Post-discriminated-union: cfg.inference.engine IS the typed engine config.
    mock_pipeline_config.inference = Mock()
    mock_pipeline_config.inference.engine = mock_engine_config
    mock_pipeline_config.inference.common = Mock()

    mock_pipeline_config.model = Mock()
    mock_pipeline_config.model.name = "test-model"
    mock_pipeline_config.model.trust_remote_code = False

    ctx = ProviderContext(
        provider_id="single_node",
        pipeline_config=mock_pipeline_config,
        provider_block=config_dict,
        secrets=mock_secrets,
    )
    provider_instance = SingleNodeInferenceProvider(ctx)
    provider_instance._run_id = "test-run-123"
    return provider_instance


class TestDockerPullTimeout:
    """
    Test: Docker pull timeout increased from 600s to 1200s.
    
    Bug: Images (~13GB total) took longer than 10 minutes to pull on slow networks.
    Fix: Increased timeout to 1200s (20 minutes).
    """

    def test_ensure_docker_image_uses_1200s_timeout_for_pull(self, provider):
        """Verify that docker pull uses 1200s timeout (not 600s)."""
        # NEW v3: SSH config is already set in provider from fixtures, just mock exec
        mock_ssh = MagicMock()
        provider._ssh_client = mock_ssh

        # Mock: image not found locally (needs pull)
        mock_ssh.exec_command.side_effect = [
            (False, "", ""),  # docker image inspect → not found
            (True, "Pull complete", ""),  # docker pull → success
            (True, "", ""),  # docker image inspect (verification) → found
        ]

        # Phase A2 Batch 12: _ensure_docker_image raises on failure, returns None on success.
        provider._ensure_docker_image(
            ssh=mock_ssh,
            image="test-image:v1.0",
        )

        # Check that docker pull was called with timeout=1200
        pull_call = mock_ssh.exec_command.call_args_list[1]
        assert pull_call[0][0] == "docker pull test-image:v1.0"
        assert pull_call[1]["timeout"] == 1200  # CRITICAL: must be 1200, not 600
        assert pull_call[1]["silent"] is False

    # NOTE: test_merge_container_uses_1200s_timeout DELETED in PR-16.
    # The legacy ``_run_merge_container`` is gone; the merge step is now an
    # engine-described :class:`PrepareStep` executed by
    # ``SingleNodeInferenceProvider._run_prepare_plan``. The 1200s pull
    # timeout regression is still pinned (``PULL_TIMEOUT = 1200`` in
    # provider.py). Per-step timeouts (``timeout_seconds`` field) are
    # covered in ``packages/providers/tests/unit/providers/single_node/test_run_prepare_plan.py``.


# ---------------------------------------------------------------------------
# PR-16 migration note: ``TestMergeCommandFormatting`` and
# ``TestMergePathMapping`` were deleted from this file. Their coverage moved:
#
#   * Shell-string formatting (single-line, no backslash continuation,
#     trust_remote_code flag conditional) → ``packages/engines/tests/unit/vllm/test_prepare_model.py``
#     (TestRegression + TestLogicSpecific) and
#     ``packages/providers/tests/unit/providers/inference/test_format_prepare_step.py``
#     (TestPositive + TestInvariants).
#   * Path mapping (host ↔ container under /workspace) →
#     ``packages/providers/tests/unit/providers/single_node/test_run_prepare_plan.py``
#     (TestPathMapping, parametrized over both directions).
# ---------------------------------------------------------------------------


class TestDockerImageVerification:
    """
    Test: Docker image verification after pull.
    
    Bug: After docker pull, the image was not immediately available in registry,
         causing "Unable to find image" error on docker run.
    Fix: Added 2-second delay and explicit docker image inspect verification.
    """

    def test_ensure_docker_image_verifies_after_pull(self, provider):
        """Verify that docker image inspect is called after docker pull."""
        mock_ssh = MagicMock()
        provider._ssh_client = mock_ssh

        # Mock: image not found → pull → verify
        mock_ssh.exec_command.side_effect = [
            (False, "", ""),  # docker image inspect → not found (initial check)
            (True, "Pull complete", ""),  # docker pull → success
            (True, "", ""),  # docker image inspect → found (verification)
        ]

        with patch("time.sleep") as mock_sleep:
            # Phase A2 Batch 12: _ensure_docker_image raises on failure, returns None on success.
            provider._ensure_docker_image(
                ssh=mock_ssh,
                image="test-image:v1.0",
            )

            # CRITICAL: sleep(2) must be called after pull
            mock_sleep.assert_called_once_with(2)

            # CRITICAL: docker image inspect must be called AFTER pull for verification
            assert mock_ssh.exec_command.call_count == 3
            verify_call = mock_ssh.exec_command.call_args_list[2]
            assert "docker image inspect test-image:v1.0" in verify_call[0][0]

    def test_ensure_docker_image_fails_if_verification_fails(self, provider):
        """Verify that pull fails if image is not available after verification."""
        mock_ssh = MagicMock()
        provider._ssh_client = mock_ssh

        # Mock: pull succeeds but verification fails (image still not found)
        mock_ssh.exec_command.side_effect = [
            (False, "", ""),  # docker image inspect → not found (initial)
            (True, "Pull complete", ""),  # docker pull → success
            # docker image inspect → still not found (verification retries; 5 attempts)
            (False, "", ""),
            (False, "", ""),
            (False, "", ""),
            (False, "", ""),
            (False, "", ""),
        ]

        from ryotenkai_shared.errors import InferenceUnavailableError

        with patch("time.sleep"):
            with pytest.raises(InferenceUnavailableError) as exc_info:
                provider._ensure_docker_image(
                    ssh=mock_ssh,
                    image="test-image:v1.0",
                )

            error_msg = str(exc_info.value.detail or exc_info.value)

            # CRITICAL: Error message should mention that image is not available after pull
            assert "not available in Docker registry" in error_msg
            assert "pulled but" in error_msg.lower()


class TestEnsureDockerImageIfNotPresent:
    """
    Fixed behavior (no user-configured pull policy):
    - if image exists locally -> skip pull
    - if missing -> pull + verify
    """

    def test_skips_pull_if_image_exists(self, provider):
        mock_ssh = MagicMock()
        provider._ssh_client = mock_ssh

        mock_ssh.exec_command.side_effect = [
            (True, "", ""),  # docker image inspect -> found
        ]

        # Phase A2 Batch 12: _ensure_docker_image raises on failure, returns None on success.
        provider._ensure_docker_image(ssh=mock_ssh, image="test-image:v1.0")
        assert mock_ssh.exec_command.call_count == 1


class TestHealthCheckBugFix:
    """
    Tests for bug fix: health_check always returned Ok(bool) even on errors.
    
    Bug: health_check returned Ok(ok and "OK" in stdout), which always returns Ok.
    This caused inference deployer to timeout even when vLLM was running,
    because failed health checks were not properly reported as errors.
    
    Fixed: Return Err() when SSH command fails, Ok(True) when "OK" in stdout,
    Ok(False) when command succeeds but service not ready.
    """

    def test_health_check_with_uninitialized_ssh_client(self, mock_config):
        """Test that health_check fails when SSH client is not initialized."""
        from ryotenkai_shared.errors import InferenceUnavailableError

        provider = self._create_provider(mock_config)
        # Don't call _connect_ssh(), so _ssh_client is None

        with pytest.raises(InferenceUnavailableError) as exc_info:
            provider.health_check()
        error_msg = str(exc_info.value.detail or exc_info.value)
        assert "SSH client not initialized" in error_msg

    def test_health_check_command_fails(self, mock_config):
        """Test that health_check raises when SSH command fails."""
        from ryotenkai_shared.errors import InferenceUnavailableError

        provider = self._create_provider(mock_config)

        # Mock SSH client directly
        mock_ssh = MagicMock()
        provider._ssh_client = mock_ssh

        # Mock: health check command fails (connection refused)
        mock_ssh.exec_command.return_value = (False, "", "Connection refused")

        with pytest.raises(InferenceUnavailableError) as exc_info:
            provider.health_check()
        error_msg = str(exc_info.value.detail or exc_info.value)
        assert "Health check command failed" in error_msg
        assert "Connection refused" in error_msg

    def test_health_check_success_with_ok_in_stdout(self, mock_config):
        """Test that health_check returns True when service is ready."""
        provider = self._create_provider(mock_config)

        # Mock SSH client directly
        mock_ssh = MagicMock()
        provider._ssh_client = mock_ssh

        # Mock: health check succeeds
        mock_ssh.exec_command.return_value = (True, "1", "")

        # Phase A2 Batch 12: health_check returns bool.
        assert provider.health_check() is True

    def test_health_check_command_succeeds_but_service_not_ready(self, mock_config):
        """Test that health_check returns False when command succeeds but no 'OK' in output."""
        provider = self._create_provider(mock_config)

        # Mock SSH client directly
        mock_ssh = MagicMock()
        provider._ssh_client = mock_ssh

        # Mock: command succeeds but service not ready (returns "0")
        mock_ssh.exec_command.return_value = (True, "0", "")

        # Phase A2 Batch 12: health_check returns bool.
        assert provider.health_check() is False

    def test_health_check_uses_config_host_port(self, mock_config):
        """Test that health_check uses host and port from provider config."""
        # Set explicit values for host and port (NEW v3: .inference.serve)
        mock_config.inference.serve.host = "192.168.1.100"
        mock_config.inference.serve.port = 8000

        provider = self._create_provider(mock_config)

        # Mock SSH client directly
        mock_ssh = MagicMock()
        provider._ssh_client = mock_ssh

        # Mock: health check succeeds
        mock_ssh.exec_command.return_value = (True, "OK", "")

        provider.health_check()

        # Verify command was called with correct host:port
        call_args = mock_ssh.exec_command.call_args
        command = call_args[0][0] if call_args else ""

        # Command should contain host:port from config
        assert "http://192.168.1.100:8000" in command

    def test_health_check_partial_ok_in_output(self, mock_config):
        """Test that health_check requires exact '1', not partial match."""
        provider = self._create_provider(mock_config)

        # Mock SSH client directly
        mock_ssh = MagicMock()
        provider._ssh_client = mock_ssh

        # Mock: output with other digits (not exact "1")
        mock_ssh.exec_command.return_value = (True, "Some debug info\n10\nMore info", "")

        # Phase A2 Batch 12: health_check returns False (no exact match).
        assert provider.health_check() is False

    @staticmethod
    def _create_provider(config):
        """Helper to create provider instance (NEW v3 structure)."""

        # Prepare config dict
        config_dict = config.model_dump(mode="python") if hasattr(config, 'model_dump') else config

        # Create minimal pipeline config (post-discriminated-union shape).
        full_config = Mock(spec=PipelineConfig)
        full_config.training = Mock()
        full_config.training.provider = "single_node"
        full_config.get_provider_config = lambda *args, **kwargs: config_dict

        full_config.inference = Mock()
        full_config.inference.engine = VLLMEngineConfig()
        full_config.inference.common = Mock()

        full_config.model = Mock()
        full_config.model.name = "test-model"
        full_config.model.trust_remote_code = False

        from ryotenkai_providers.registry import ProviderContext

        secrets = Secrets(hf_token="test-token")
        ctx = ProviderContext(
            provider_id="single_node",
            pipeline_config=full_config,
            provider_block=config_dict,
            secrets=secrets,
        )
        return SingleNodeInferenceProvider(ctx)

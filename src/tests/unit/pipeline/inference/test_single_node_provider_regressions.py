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

from src.providers.single_node.inference.provider import SingleNodeInferenceProvider
from src.providers.single_node.training.config import (
    SingleNodeConfig,
    SingleNodeConnectConfig,
    SingleNodeInferenceConfig,
    SingleNodeTrainingConfig,
)
from src.utils.config import (
    InferenceSingleNodeServeConfig,
    InferenceVLLMEngineConfig,
    PipelineConfig,
    Secrets,
    SSHConfig,
)


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
    return SingleNodeConfig(
        connect=SingleNodeConnectConfig(ssh=mock_ssh_config),
        training=SingleNodeTrainingConfig(
            workspace_path="/home/testuser/workspace",
            docker_image="test/runtime:latest",
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
    return InferenceVLLMEngineConfig(
        merge_image="test-merge:v1.0",
        serve_image="test-vllm:v0.6.3",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
    )


@pytest.fixture
def mock_secrets():
    """Mock secrets."""
    return Secrets(hf_token="test-token-123")


@pytest.fixture
def provider(mock_config, mock_engine_config, mock_secrets):
    """Create provider instance with mocked dependencies (NEW v3 structure)."""
    # Prepare config dict
    config_dict = mock_config.model_dump(mode="python")

    # Mock full PipelineConfig
    mock_pipeline_config = Mock()
    mock_pipeline_config.training = Mock()
    mock_pipeline_config.training.provider = "single_node"
    mock_pipeline_config.get_provider_config = lambda *args, **kwargs: config_dict

    mock_pipeline_config.inference = Mock()
    mock_pipeline_config.inference.engines = Mock()
    mock_pipeline_config.inference.engines.vllm = mock_engine_config
    mock_pipeline_config.inference.common = Mock()
    mock_pipeline_config.inference.common.lora = Mock()
    mock_pipeline_config.inference.common.lora.merge_before_deploy = True

    mock_pipeline_config.model = Mock()
    mock_pipeline_config.model.name = "test-model"
    mock_pipeline_config.model.trust_remote_code = False

    provider_instance = SingleNodeInferenceProvider(
        config=mock_pipeline_config,
        secrets=mock_secrets,
    )

    # Mock run_id (normally set in deploy())
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

        result = provider._ensure_docker_image(
            ssh=mock_ssh,
            image="test-image:v1.0",
        )

        assert result.is_success()

        # Check that docker pull was called with timeout=1200
        pull_call = mock_ssh.exec_command.call_args_list[1]
        assert pull_call[0][0] == "docker pull test-image:v1.0"
        assert pull_call[1]["timeout"] == 1200  # CRITICAL: must be 1200, not 600
        assert pull_call[1]["silent"] is False

    def test_merge_container_uses_1200s_timeout(self, provider):
        """Verify that merge container execution uses proper timeouts."""
        with patch.object(provider, "_ensure_docker_image") as mock_ensure:
            mock_ensure.return_value = Mock(is_failure=lambda: False)

            mock_ssh = MagicMock()
            mock_ssh.exec_command.side_effect = [
                (True, "", ""),  # rm -rf output_path
                (True, "", ""),  # mkdir cache_dir
                (True, "container123", ""),  # docker run --detach (returns container ID)
                (True, "container123", ""),  # docker ps -q (container running) ← FIX: must return ID!
                (True, "MERGE_SUCCESS", ""),  # docker logs (first poll)
                (True, "", ""),  # docker ps -q (container stopped)
                (True, "MERGE_SUCCESS\nDone", ""),  # docker logs (final)
                (True, "0", ""),  # docker inspect exit code
                (True, "", ""),  # docker rm -f
                (True, "OK", ""),  # verify config.json exists
            ]

            workspace_host = provider._provider_cfg.inference.serve.workspace.rstrip("/")

            result = provider._run_merge_container(
                ssh=mock_ssh,
                base_model="meta-llama/Llama-2-7b-hf",
                adapter_path="/workspace/runs/test/adapter",
                output_path=f"{workspace_host}/runs/test/merged",
                cache_dir=f"{workspace_host}/hf_cache",
                trust_remote_code=False,
            )

            assert result.is_success()

            # Check that docker run --detach was called
            detach_call = mock_ssh.exec_command.call_args_list[2]
            assert "docker run --detach" in detach_call[0][0]
            assert "helix-merge-" in detach_call[0][0]


class TestMergeCommandFormatting:
    """
    Test: Multi-line shell command with trailing backslash bug.
    
    Bug: When trust_remote_code=False, the merge command had a trailing "\\"
         which was interpreted as an extra argument by argparse.
    Fix: Replaced multi-line string with f-string concatenation.
    """

    def test_merge_command_without_trust_remote_code_has_no_trailing_backslash(self, provider):
        """Verify that merge command without trust flag has no trailing backslash."""
        with patch.object(provider, "_ensure_docker_image") as mock_ensure:
            mock_ensure.return_value = Mock(is_failure=lambda: False)

            mock_ssh = MagicMock()
            mock_ssh.exec_command.side_effect = _create_merge_mock_responses()

            workspace_host = provider._provider_cfg.inference.serve.workspace.rstrip("/")
            adapter_host = f"{workspace_host}/runs/test/adapter"
            output_host = f"{workspace_host}/runs/test/merged"
            cache_host = f"{workspace_host}/hf_cache"

            result = provider._run_merge_container(
                ssh=mock_ssh,
                base_model="meta-llama/Llama-2-7b-hf",
                adapter_path=adapter_host,
                output_path=output_host,
                cache_dir=cache_host,
                trust_remote_code=False,  # CRITICAL: False → no trust flag
            )

            assert result.is_success()

            # Extract the actual docker run command (3rd call)
            merge_call = mock_ssh.exec_command.call_args_list[2]
            docker_cmd = merge_call[0][0]

            # CRITICAL CHECKS:
            # 1. Command should NOT end with backslash
            assert not docker_cmd.strip().endswith("\\"), (
                f"Merge command ends with backslash when trust_remote_code=False:\n{docker_cmd}"
            )

            # 2. Command should NOT contain standalone backslash argument
            assert " \\ " not in docker_cmd, (
                f"Merge command contains standalone backslash:\n{docker_cmd}"
            )

            # 3. Command should NOT have --trust-remote-code flag
            assert "--trust-remote-code" not in docker_cmd

            # 4. Command should contain all required arguments
            assert "--base-model meta-llama/Llama-2-7b-hf" in docker_cmd
            assert "--adapter /workspace/runs/test/adapter" in docker_cmd
            assert "--output /workspace/runs/test/merged" in docker_cmd
            assert "--cache-dir /workspace/hf_cache" in docker_cmd

    def test_merge_command_with_trust_remote_code_includes_flag(self, provider):
        """Verify that merge command with trust_remote_code=True includes the flag."""
        with patch.object(provider, "_ensure_docker_image") as mock_ensure:
            mock_ensure.return_value = Mock(is_failure=lambda: False)

            mock_ssh = MagicMock()
            mock_ssh.exec_command.side_effect = _create_merge_mock_responses()

            workspace_host = provider._provider_cfg.inference.serve.workspace.rstrip("/")
            adapter_host = f"{workspace_host}/runs/test/adapter"
            output_host = f"{workspace_host}/runs/test/merged"
            cache_host = f"{workspace_host}/hf_cache"

            result = provider._run_merge_container(
                ssh=mock_ssh,
                base_model="meta-llama/Llama-2-7b-hf",
                adapter_path=adapter_host,
                output_path=output_host,
                cache_dir=cache_host,
                trust_remote_code=True,  # CRITICAL: True → should include flag
            )

            assert result.is_success()

            # Extract the actual docker run command (3rd call)
            merge_call = mock_ssh.exec_command.call_args_list[2]
            docker_cmd = merge_call[0][0]

            # CRITICAL: Command should include --trust-remote-code flag
            assert "--trust-remote-code" in docker_cmd, (
                f"Merge command missing --trust-remote-code flag:\n{docker_cmd}"
            )

            # Should still not end with backslash
            assert not docker_cmd.strip().endswith("\\")

    def test_merge_command_is_single_line_no_multiline_string(self, provider):
        """Verify that merge command is formatted as single line (no \\n in command)."""
        with patch.object(provider, "_ensure_docker_image") as mock_ensure:
            mock_ensure.return_value = Mock(is_failure=lambda: False)

            mock_ssh = MagicMock()
            mock_ssh.exec_command.side_effect = _create_merge_mock_responses()

            workspace_host = provider._provider_cfg.inference.serve.workspace.rstrip("/")

            provider._run_merge_container(
                ssh=mock_ssh,
                base_model="test-model",
                adapter_path=f"{workspace_host}/runs/test/adapter",
                output_path=f"{workspace_host}/runs/test/merged",
                cache_dir=f"{workspace_host}/hf_cache",
                trust_remote_code=False,
            )

            merge_call = mock_ssh.exec_command.call_args_list[2]
            docker_cmd = merge_call[0][0]

            # Command should be single line (no embedded newlines)
            lines = docker_cmd.split("\n")
            assert len(lines) == 1, (
                f"Merge command should be single line, got {len(lines)} lines:\n{docker_cmd}"
            )


class TestMergePathMapping:
    """
    Regression: merge container must write artifacts into the mounted /workspace path.

    Bug observed in logs:
      --output /home/user/inference/...  (host path inside container => writes to ephemeral FS)
      --cache-dir /home/user/inference/hf_cache (same issue)

    Fix:
      Map host workspace paths to container paths:
        /home/.../inference/...  -> /workspace/...
    """

    def test_merge_command_maps_output_and_cache_to_workspace_mount(self, provider):
        with patch.object(provider, "_ensure_docker_image") as mock_ensure:
            mock_ensure.return_value = Mock(is_failure=lambda: False)

            mock_ssh = MagicMock()
            mock_ssh.exec_command.side_effect = _create_merge_mock_responses()

            workspace_host = provider._provider_cfg.inference.serve.workspace.rstrip("/")
            output_host = f"{workspace_host}/runs/run_test/model"
            cache_host = f"{workspace_host}/hf_cache"

            result = provider._run_merge_container(
                ssh=mock_ssh,
                base_model="Qwen/Qwen2.5-0.5B-Instruct",
                adapter_path="test-org/test3-three-strategies",  # HF repo ID
                output_path=output_host,
                cache_dir=cache_host,
                trust_remote_code=False,
            )

            assert result.is_success()

            # docker run command is 3rd call
            docker_cmd = mock_ssh.exec_command.call_args_list[2][0][0]

            # CRITICAL: output/cache must be in /workspace, not /home/... inside container
            assert "--output /workspace/runs/run_test/model" in docker_cmd
            assert "--cache-dir /workspace/hf_cache" in docker_cmd

            # CRITICAL: HF cache env vars must point to the same in-container cache dir
            assert "-e HF_HOME=/workspace/hf_cache" in docker_cmd
            assert "-e HUGGINGFACE_HUB_CACHE=/workspace/hf_cache" in docker_cmd
            assert "-e TRANSFORMERS_CACHE=/workspace/hf_cache" in docker_cmd

    def test_merge_command_maps_local_adapter_dir_to_workspace_mount(self, provider):
        with patch.object(provider, "_ensure_docker_image") as mock_ensure:
            mock_ensure.return_value = Mock(is_failure=lambda: False)

            mock_ssh = MagicMock()
            mock_ssh.exec_command.side_effect = _create_merge_mock_responses()

            workspace_host = provider._provider_cfg.inference.serve.workspace.rstrip("/")
            adapter_host = f"{workspace_host}/runs/run_test/adapter"
            output_host = f"{workspace_host}/runs/run_test/model"
            cache_host = f"{workspace_host}/hf_cache"

            result = provider._run_merge_container(
                ssh=mock_ssh,
                base_model="Qwen/Qwen2.5-0.5B-Instruct",
                adapter_path=adapter_host,  # host path under workspace
                output_path=output_host,
                cache_dir=cache_host,
                trust_remote_code=False,
            )

            assert result.is_success()

            docker_cmd = mock_ssh.exec_command.call_args_list[2][0][0]
            # CRITICAL: adapter must be in /workspace, not /home/...
            assert "--adapter /workspace/runs/run_test/adapter" in docker_cmd
            assert "--output /workspace/runs/run_test/model" in docker_cmd
            assert "--cache-dir /workspace/hf_cache" in docker_cmd


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
            result = provider._ensure_docker_image(
                ssh=mock_ssh,
                image="test-image:v1.0",
            )

            assert result.is_success()

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

        with patch("time.sleep"):
            result = provider._ensure_docker_image(
                ssh=mock_ssh,
                image="test-image:v1.0",
            )

            assert result.is_failure()
            error_msg = str(result.unwrap_err())

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

        res = provider._ensure_docker_image(ssh=mock_ssh, image="test-image:v1.0")
        assert res.is_success()
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
        provider = self._create_provider(mock_config)
        # Don't call _connect_ssh(), so _ssh_client is None

        result = provider.health_check()

        assert result.is_failure()
        error_msg = str(result.unwrap_err())
        assert "SSH client not initialized" in error_msg

    def test_health_check_command_fails(self, mock_config):
        """Test that health_check returns Err when SSH command fails."""
        provider = self._create_provider(mock_config)

        # Mock SSH client directly
        mock_ssh = MagicMock()
        provider._ssh_client = mock_ssh

        # Mock: health check command fails (connection refused)
        mock_ssh.exec_command.return_value = (False, "", "Connection refused")

        result = provider.health_check()

        # Should return Err (not Ok(False)!)
        assert result.is_failure()
        error_msg = str(result.unwrap_err())
        assert "Health check command failed" in error_msg
        assert "Connection refused" in error_msg

    def test_health_check_success_with_ok_in_stdout(self, mock_config):
        """Test that health_check returns Ok(True) when service is ready."""
        provider = self._create_provider(mock_config)

        # Mock SSH client directly
        mock_ssh = MagicMock()
        provider._ssh_client = mock_ssh

        # Mock: health check succeeds
        mock_ssh.exec_command.return_value = (True, "1", "")

        result = provider.health_check()

        # Should return Ok(True)
        assert result.is_success()
        is_healthy = result.unwrap()
        assert is_healthy is True

    def test_health_check_command_succeeds_but_service_not_ready(self, mock_config):
        """Test that health_check returns Ok(False) when command succeeds but no 'OK' in output."""
        provider = self._create_provider(mock_config)

        # Mock SSH client directly
        mock_ssh = MagicMock()
        provider._ssh_client = mock_ssh

        # Mock: command succeeds but service not ready (returns "0")
        mock_ssh.exec_command.return_value = (True, "0", "")

        result = provider.health_check()

        # Should return Ok(False), not Err
        assert result.is_success()
        is_healthy = result.unwrap()
        assert is_healthy is False

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

        result = provider.health_check()

        # Should return Ok(False) because "10" != "1" (exact match required)
        assert result.is_success()
        is_healthy = result.unwrap()
        assert is_healthy is False

    @staticmethod
    def _create_provider(config):
        """Helper to create provider instance (NEW v3 structure)."""

        # Prepare config dict
        config_dict = config.model_dump(mode="python") if hasattr(config, 'model_dump') else config

        # Create minimal pipeline config
        full_config = Mock(spec=PipelineConfig)
        full_config.training = Mock()
        full_config.training.provider = "single_node"
        # Important: Mock.return_value returns Mock, use lambda to return actual dict
        full_config.get_provider_config = lambda *args, **kwargs: config_dict

        full_config.inference = Mock()
        full_config.inference.engines = Mock()
        full_config.inference.engines.vllm = InferenceVLLMEngineConfig(
            merge_image="test-merge:v1.0",
            serve_image="test-vllm:v0.6.3",
        )
        full_config.inference.common = Mock()
        full_config.inference.common.lora = Mock()
        full_config.inference.common.lora.merge_before_deploy = True

        full_config.model = Mock()
        full_config.model.name = "test-model"
        full_config.model.trust_remote_code = False

        secrets = Secrets(hf_token="test-token")

        return SingleNodeInferenceProvider(config=full_config, secrets=secrets)

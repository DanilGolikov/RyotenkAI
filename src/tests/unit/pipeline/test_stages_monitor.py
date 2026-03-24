"""
Unit tests for Training Monitor Stage.

Tests cover:
- Initialization (with/without callbacks)
- Execute method (success, errors, mock mode)
- Wait for training start (timeout, fast completion)
- Monitor training (completion, failure, timeout, race conditions)
- Helper methods (process checks, markers, resources)
- Callbacks integration
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.stages.training_monitor import (
    TrainingMonitor,
    TrainingMonitorEventCallbacks,
)
from src.utils.result import Ok

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_config():
    """Mock PipelineConfig with provider settings."""
    config = MagicMock()
    config.get_provider_config.return_value = {
        "training_start_timeout": 30,
        "mock_mode": False,
    }
    config.training.get_strategy_chain.return_value = []  # For mock mode
    return config


@pytest.fixture
def mock_callbacks():
    """Mock TrainingMonitorEventCallbacks."""
    return TrainingMonitorEventCallbacks(
        on_training_started=MagicMock(),
        on_training_completed=MagicMock(),
        on_training_failed=MagicMock(),
        on_process_died=MagicMock(),
        on_resource_check=MagicMock(),
    )


@pytest.fixture
def mock_context():
    """Mock context with GPU Deployer data."""
    return {
        "GPU Deployer": {
            "resource_id": "test-pod-123",
            "ssh_host": "192.168.1.100",
            "ssh_port": 22,
            "ssh_key_path": "/path/to/key",
            "ssh_user": "root",
            "is_alias_mode": False,
            "workspace_path": "/workspace",
            "provider_info": {},
        }
    }


@pytest.fixture
def mock_ssh_client():
    """Mock SSHClient."""
    client = MagicMock()
    client.exec_command.return_value = (True, "", "")
    return client


@pytest.fixture
def mock_log_manager():
    """Mock LogManager."""
    manager = MagicMock()
    manager.download = MagicMock()
    manager.download_on_error = MagicMock()
    manager.get_last_lines.return_value = ["log line 1", "log line 2"]
    manager.local_path.exists.return_value = True
    return manager


# =============================================================================
# BATCH 1: INITIALIZATION + EXECUTE (10 tests)
# =============================================================================


class TestInitialization:
    """Tests for TrainingMonitor initialization."""

    def test_init_with_callbacks(self, mock_config, mock_callbacks):
        """Test initialization with full config and callbacks."""
        monitor = TrainingMonitor(mock_config, callbacks=mock_callbacks)

        assert monitor.config == mock_config
        assert monitor.stage_name == "Training Monitor"
        assert monitor.log_download_interval == 30
        assert monitor._callbacks == mock_callbacks
        assert monitor._training_start_time == 0.0

    def test_init_without_callbacks(self, mock_config):
        """Test initialization without callbacks (should use defaults)."""
        monitor = TrainingMonitor(mock_config)

        assert monitor._callbacks is not None
        assert isinstance(monitor._callbacks, TrainingMonitorEventCallbacks)
        assert monitor._callbacks.on_training_started is None
        assert monitor._callbacks.on_training_completed is None
        assert monitor._callbacks.on_training_failed is None

    def test_init_provider_config_params(self):
        """Test that provider_config is retained but training params use hardcoded constants."""
        config = MagicMock()
        config.get_provider_config.return_value = {
            "training_start_timeout": 60,
        }

        monitor = TrainingMonitor(config)

        # training_start_timeout is now a hardcoded constant, not read from config
        assert isinstance(monitor.training_start_timeout, int)
        assert monitor.training_start_timeout > 0


class TestExecuteMethod:
    """Tests for TrainingMonitor.execute() method."""

    def test_execute_missing_ssh_info(self, mock_config):
        """Test execute with missing SSH connection info."""
        monitor = TrainingMonitor(mock_config)

        # Missing ssh_host
        context = {"GPU Deployer": {"ssh_port": 22}}
        result = monitor.execute(context)

        assert result.is_err()
        assert "Missing SSH/workspace connection info from GPU Deployer" in str(result.unwrap_err())

        # Missing ssh_port
        context = {"GPU Deployer": {"ssh_host": "192.168.1.100"}}
        result = monitor.execute(context)

        assert result.is_err()
        assert "Missing SSH/workspace connection info from GPU Deployer" in str(result.unwrap_err())

    def test_execute_mock_mode(self, mock_config, mock_context):
        """Test execute in mock mode."""
        # Activate mock mode via provider_info
        mock_context["GPU Deployer"]["provider_info"] = {"mock": True}

        monitor = TrainingMonitor(mock_config)
        result = monitor.execute(mock_context)

        assert result.is_ok()
        data = result.unwrap()
        assert "Training Monitor" in data
        assert data["Training Monitor"]["status"] == "completed"
        assert data["Training Monitor"]["training_info"]["mock"] is True

    def test_execute_mock_mode_from_config(self, mock_context):
        """Test that mock mode no longer works via config (only via provider_info)."""
        config = MagicMock()
        config.get_provider_config.return_value = {
            "mock_mode": True,  # This is now ignored — mock only via provider_info
        }
        config.training.get_strategy_chain.return_value = []

        # Remove ssh_host so the monitor fails early at SSH validation
        context_no_ssh = {"GPU Deployer": {"resource_id": "pod-123", "provider_info": {}}}
        monitor = TrainingMonitor(config)
        result = monitor.execute(context_no_ssh)

        # Without provider_info.mock=True, it falls through to SSH check which fails
        assert result.is_err()


    @patch("src.pipeline.stages.training_monitor.SSHClient")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    def test_execute_ssh_port_conversion(self, mock_log_manager_cls, mock_ssh_cls, mock_config, mock_callbacks):
        """Test SSH port conversion from string to int."""
        context = {
            "GPU Deployer": {
                "resource_id": "test-pod",
                "ssh_host": "192.168.1.100",
                "ssh_port": "2222",  # String instead of int
                "ssh_key_path": "/path/to/key",
                "ssh_user": "root",
                "is_alias_mode": False,
                "workspace_path": "/workspace",
                "provider_info": {},
            }
        }

        mock_ssh_instance = MagicMock()
        mock_ssh_cls.return_value = mock_ssh_instance

        # Mock wait_for_training_start to return False (to exit early)
        monitor = TrainingMonitor(mock_config, callbacks=mock_callbacks)
        with patch.object(monitor, "_wait_for_training_start", return_value=False):
            result = monitor.execute(context)

        # Check SSHClient was created with int port
        mock_ssh_cls.assert_called_once_with(
            host="192.168.1.100",
            port=2222,  # Should be converted to int
            username="root",
            key_path="/path/to/key",
        )

        assert result.is_err()  # Failed to start

    @patch("src.pipeline.stages.training_monitor.SSHClient")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    def test_execute_alias_mode(self, mock_log_manager_cls, mock_ssh_cls, mock_config, mock_context):
        """Test alias mode (username=None for SSH config)."""
        mock_context["GPU Deployer"]["is_alias_mode"] = True

        mock_ssh_instance = MagicMock()
        mock_ssh_cls.return_value = mock_ssh_instance

        monitor = TrainingMonitor(mock_config)
        with patch.object(monitor, "_wait_for_training_start", return_value=False):
            result = monitor.execute(mock_context)

        # Check SSHClient was created with username=None
        mock_ssh_cls.assert_called_once_with(
            host="192.168.1.100",
            port=22,
            username=None,  # Alias mode should use None
            key_path="/path/to/key",
        )

        assert result.is_err()

    @patch("src.pipeline.stages.training_monitor.SSHClient")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    def test_execute_training_start_timeout(self, mock_log_manager_cls, mock_ssh_cls, mock_config, mock_callbacks):
        """Test training start timeout."""
        mock_ssh_instance = MagicMock()
        mock_ssh_cls.return_value = mock_ssh_instance

        mock_log_manager_instance = MagicMock()
        mock_log_manager_cls.return_value = mock_log_manager_instance

        context = {
            "GPU Deployer": {
                "resource_id": "test-pod",
                "ssh_host": "192.168.1.100",
                "ssh_port": 22,
                "ssh_key_path": "/path/to/key",
                "ssh_user": "root",
                "is_alias_mode": False,
                "workspace_path": "/workspace",
                "provider_info": {},
            }
        }

        monitor = TrainingMonitor(mock_config, callbacks=mock_callbacks)

        # Mock _wait_for_training_start to return False (timeout)
        with patch.object(monitor, "_wait_for_training_start", return_value=False):
            result = monitor.execute(context)

        assert result.is_err()
        assert "Training process failed to start within 30s" in str(result.unwrap_err())

        # Check error handling
        mock_log_manager_instance.download_on_error.assert_called_once()
        assert "Training failed to start" in str(mock_log_manager_instance.download_on_error.call_args)

    @patch("src.pipeline.stages.training_monitor.SSHClient")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    @patch("time.sleep")
    def test_execute_success_flow(
        self,
        mock_sleep,
        mock_log_manager_cls,
        mock_ssh_cls,
        mock_config,
        mock_callbacks,
    ):
        """Test successful execute flow."""
        mock_ssh_instance = MagicMock()
        mock_ssh_cls.return_value = mock_ssh_instance

        mock_log_manager_instance = MagicMock()
        mock_log_manager_cls.return_value = mock_log_manager_instance

        context = {
            "GPU Deployer": {
                "resource_id": "test-pod",
                "ssh_host": "192.168.1.100",
                "ssh_port": 22,
                "ssh_key_path": "/path/to/key",
                "ssh_user": "root",
                "is_alias_mode": False,
                "workspace_path": "/workspace",
                "provider_info": {},
            }
        }

        monitor = TrainingMonitor(mock_config, callbacks=mock_callbacks)

        # Mock training start and immediate completion
        with patch.object(monitor, "_wait_for_training_start", return_value=True):
            with patch.object(monitor, "_monitor_training", return_value=Ok({"status": "completed"})):
                result = monitor.execute(context)

        assert result.is_ok()
        assert monitor._training_start_time > 0  # Training start time should be set

        # Check callback was fired
        mock_callbacks.on_training_started.assert_called_once()

    def test_execute_mock_mode_with_callbacks(self, mock_context, mock_callbacks):
        """Test mock mode with callbacks fires when triggered via provider_info."""
        config = MagicMock()
        config.get_provider_config.return_value = {}
        config.training.get_strategy_chain.return_value = []

        mock_context["GPU Deployer"]["provider_info"] = {"mock": True}
        monitor = TrainingMonitor(config, callbacks=mock_callbacks)
        result = monitor.execute(mock_context)

        assert result.is_ok()

        # In mock mode, callbacks are NOT fired (simplified logic)
        mock_callbacks.on_training_started.assert_not_called()
        mock_callbacks.on_training_completed.assert_not_called()


# =============================================================================
# BATCH 2: WAIT + MONITOR (12 tests)
# =============================================================================


class TestWaitForTrainingStart:
    """Tests for TrainingMonitor._wait_for_training_start()."""

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_training_start_success(self, mock_time, mock_sleep, mock_config, mock_ssh_client):
        """Test successful wait for training start."""
        # Simulate time progression - need enough calls for the while loop
        mock_time.side_effect = [0] + list(range(5, 100, 5))  # Start + many iterations

        # First attempt: no training.log yet
        # Second attempt: training.log exists + training process running (via _is_training_alive)
        mock_ssh_client.exec_command.side_effect = [
            (False, "", ""),  # TRAINING_COMPLETE check - not exists
            (False, "", ""),  # TRAINING_FAILED check - not exists
            (False, "", ""),  # training.log check - not exists yet
            (False, "", ""),  # TRAINING_COMPLETE check
            (False, "", ""),  # TRAINING_FAILED check
            (True, "Some log content", ""),  # training.log exists
            (False, "", ""),  # docker ps check (no container / docker unavailable)
            (True, "python train.py running", ""),  # Python process found
        ]

        monitor = TrainingMonitor(mock_config)
        monitor._workspace_path = "/workspace"

        result = monitor._wait_for_training_start(mock_ssh_client, timeout=120)

        assert result is True
        assert mock_sleep.call_count == 1  # One 5s delay before second attempt

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_training_start_docker_container_running(self, mock_time, mock_sleep, mock_config, mock_ssh_client):
        """Training start detected via Docker container (docker ps)."""
        mock_time.return_value = 0

        mock_ssh_client.exec_command.side_effect = [
            (False, "", ""),  # TRAINING_COMPLETE check - not exists
            (False, "", ""),  # TRAINING_FAILED check - not exists
            (True, "Some log content", ""),  # training.log exists
            (True, "RUNNING", ""),  # docker ps indicates helix_training container running
        ]

        monitor = TrainingMonitor(mock_config)
        monitor._workspace_path = "/workspace"

        result = monitor._wait_for_training_start(mock_ssh_client, timeout=120)

        assert result is True
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_training_start_timeout(self, mock_time, mock_sleep, mock_config, mock_ssh_client):
        """Test wait for training start timeout."""
        # Simulate time progression past timeout - need more values
        mock_time.side_effect = [0] + list(range(5, 200, 5))  # Exceeds 120s timeout

        # Never find training.log or markers
        mock_ssh_client.exec_command.return_value = (False, "", "")

        monitor = TrainingMonitor(mock_config)
        monitor._workspace_path = "/workspace"

        result = monitor._wait_for_training_start(mock_ssh_client, timeout=120)

        assert result is False
        assert mock_sleep.call_count >= 1  # Multiple polling attempts

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_training_start_fast_completion(self, mock_time, mock_sleep, mock_config, mock_ssh_client):
        """Test wait when TRAINING_COMPLETE already exists (fast training)."""
        mock_time.return_value = 0

        # TRAINING_COMPLETE marker already exists
        mock_ssh_client.exec_command.side_effect = [
            (True, "EXISTS", ""),  # TRAINING_COMPLETE check - found immediately!
        ]

        monitor = TrainingMonitor(mock_config)
        monitor._workspace_path = "/workspace"

        result = monitor._wait_for_training_start(mock_ssh_client, timeout=120)

        assert result is True
        mock_sleep.assert_not_called()  # Should return immediately

    @patch("time.sleep")
    @patch("time.time")
    def test_wait_for_training_start_already_failed(self, mock_time, mock_sleep, mock_config, mock_ssh_client):
        """Test wait when TRAINING_FAILED already exists."""
        mock_time.return_value = 0

        # TRAINING_FAILED marker already exists
        mock_ssh_client.exec_command.side_effect = [
            (False, "", ""),  # TRAINING_COMPLETE check - not found
            (True, "EXISTS", ""),  # TRAINING_FAILED check - found!
        ]

        monitor = TrainingMonitor(mock_config)
        monitor._workspace_path = "/workspace"

        result = monitor._wait_for_training_start(mock_ssh_client, timeout=120)

        assert result is True  # Return True to let _monitor_training handle error
        mock_sleep.assert_not_called()


class TestMonitorTraining:
    """Tests for TrainingMonitor._monitor_training()."""

    @patch("time.sleep")
    @patch("time.time")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    def test_monitor_training_completion_success(
        self,
        mock_log_manager_cls,
        mock_time,
        mock_sleep,
        mock_config,
        mock_ssh_client,
        mock_callbacks,
    ):
        """Test successful training completion."""
        # Time progression - use itertools.cycle for unlimited calls
        from itertools import count

        time_counter = count(start=0, step=1)
        mock_time.side_effect = lambda: next(time_counter)

        mock_log_manager_instance = MagicMock()
        mock_log_manager_cls.return_value = mock_log_manager_instance
        mock_log_manager_instance.get_last_lines.return_value = ["log1", "log2"]

        monitor = TrainingMonitor(mock_config, callbacks=mock_callbacks)
        monitor._log_manager = mock_log_manager_instance
        monitor._workspace_path = "/workspace"

        context = {}

        # Use counter to control when marker appears
        check_call_count = [0]  # Mutable to modify in closure

        def check_marker_side_effect(client, marker_name):
            check_call_count[0] += 1
            # After 2 calls (1 loop: COMPLETE + FAILED), return True for TRAINING_COMPLETE
            if check_call_count[0] >= 3 and marker_name == "TRAINING_COMPLETE":
                return True
            return False

        with patch.object(monitor, "_check_marker", side_effect=check_marker_side_effect):
            result = monitor._monitor_training(mock_ssh_client, context)

        assert result.is_ok()
        data = result.unwrap()
        assert "Training Monitor" in data
        assert data["Training Monitor"]["status"] == "completed"
        assert "training_duration_seconds" in data["Training Monitor"]

        # Check callbacks
        mock_callbacks.on_training_completed.assert_called_once()
        duration_arg = mock_callbacks.on_training_completed.call_args[0][0]
        assert duration_arg >= 0

        # Check log download
        mock_log_manager_instance.download.assert_called_once()

    @patch("time.sleep")
    @patch("time.time")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    def test_monitor_training_failed_marker(
        self,
        mock_log_manager_cls,
        mock_time,
        mock_sleep,
        mock_config,
        mock_ssh_client,
        mock_callbacks,
    ):
        """Test training failed with marker."""
        mock_time.side_effect = [0, 5, 10]

        mock_log_manager_instance = MagicMock()
        mock_log_manager_cls.return_value = mock_log_manager_instance
        mock_log_manager_instance.get_last_lines.return_value = []

        monitor = TrainingMonitor(mock_config, callbacks=mock_callbacks)
        monitor._log_manager = mock_log_manager_instance
        monitor._workspace_path = "/workspace"

        # Mock _check_marker and _read_marker_content
        with patch.object(monitor, "_check_marker") as mock_check:
            with patch.object(monitor, "_read_marker_content") as mock_read:
                mock_check.side_effect = [
                    False,  # TRAINING_COMPLETE - not found
                    True,  # TRAINING_FAILED - found!
                ]
                mock_read.return_value = "Out of memory error"

                result = monitor._monitor_training(mock_ssh_client, {})

        assert result.is_err()
        assert "Out of memory error" in str(result.unwrap_err())

        # Check callbacks
        mock_callbacks.on_training_failed.assert_called_once()
        error_msg, duration = mock_callbacks.on_training_failed.call_args[0]
        assert "Out of memory error" in error_msg
        assert duration >= 0

        # Check error handling
        mock_log_manager_instance.download_on_error.assert_called_once()

    @patch("time.sleep")
    @patch("time.time")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    def test_monitor_training_process_died_no_marker(
        self,
        mock_log_manager_cls,
        mock_time,
        mock_sleep,
        mock_config,
        mock_callbacks,
        mock_ssh_client,
    ):
        """Test process died without marker (after 5 retry attempts)."""
        # Time progression: initial check + 2s wait + 5 attempts with 3s each
        mock_time.side_effect = [0, 5, 7] + list(range(10, 26, 3))  # Multiple checks

        mock_log_manager_instance = MagicMock()
        mock_log_manager_cls.return_value = mock_log_manager_instance
        mock_log_manager_instance.get_last_lines.return_value = []

        monitor = TrainingMonitor(mock_config, callbacks=mock_callbacks)
        monitor._log_manager = mock_log_manager_instance
        monitor._workspace_path = "/workspace"

        # Mock methods
        with patch.object(monitor, "_is_training_alive", return_value=False):
            with patch.object(monitor, "_check_marker", return_value=False):
                result = monitor._monitor_training(mock_ssh_client, {})

        assert result.is_err()
        assert "Training process died without completion marker" in str(result.unwrap_err())

        # Check callback
        mock_callbacks.on_process_died.assert_called_once()
        duration = mock_callbacks.on_process_died.call_args[0][0]
        assert duration >= 0

        # Check 5 retry attempts (3s sleep each) + initial 2s wait
        assert mock_sleep.call_count >= 5

    @patch("time.sleep")
    @patch("time.time")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    def test_monitor_training_process_died_late_marker(
        self,
        mock_log_manager_cls,
        mock_time,
        mock_sleep,
        mock_config,
        mock_callbacks,
        mock_ssh_client,
    ):
        """Test process died but TRAINING_COMPLETE appears on 3rd retry (race condition)."""
        mock_time.side_effect = [0, 5, 7, 10, 13]

        mock_log_manager_instance = MagicMock()
        mock_log_manager_cls.return_value = mock_log_manager_instance
        mock_log_manager_instance.get_last_lines.return_value = []

        monitor = TrainingMonitor(mock_config, callbacks=mock_callbacks)
        monitor._log_manager = mock_log_manager_instance
        monitor._workspace_path = "/workspace"

        context = {}

        # Process is dead
        with patch.object(monitor, "_is_training_alive", return_value=False):
            # Markers: not found, not found, then TRAINING_COMPLETE on 3rd attempt
            with patch.object(monitor, "_check_marker") as mock_check:
                mock_check.side_effect = [
                    False,  # Initial TRAINING_COMPLETE check
                    False,  # Initial TRAINING_FAILED check
                    False,  # After 2s wait: TRAINING_COMPLETE
                    False,  # TRAINING_FAILED
                    True,  # Attempt 1: TRAINING_COMPLETE - FOUND!
                ]

                result = monitor._monitor_training(mock_ssh_client, context)

        assert result.is_ok()
        data = result.unwrap()
        assert data["Training Monitor"]["status"] == "completed"

        # Check callback (race condition resolved, training completed)
        mock_callbacks.on_training_completed.assert_called_once()
        mock_callbacks.on_process_died.assert_not_called()  # Should NOT fire

    @patch("time.sleep")
    @patch("time.time")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    def test_monitor_training_rate_limiting_status_logs(
        self,
        mock_log_manager_cls,
        mock_time,
        mock_sleep,
        mock_config,
        mock_ssh_client,
    ):
        """Test status logs rate limiting (every 15s)."""
        # Simulate time progression
        from itertools import count

        time_counter = count(start=0, step=1)
        mock_time.side_effect = lambda: next(time_counter)

        mock_log_manager_instance = MagicMock()
        mock_log_manager_cls.return_value = mock_log_manager_instance

        monitor = TrainingMonitor(mock_config)
        monitor._log_manager = mock_log_manager_instance
        monitor._workspace_path = "/workspace"

        check_call_count = [0]

        def check_marker_side_effect(client, marker_name):
            check_call_count[0] += 1
            # Complete after 30 loops (60 calls: 30 x (COMPLETE + FAILED))
            # This should give enough time for 4 status logs (0s, 15s, 30s, 45s)
            if check_call_count[0] >= 60 and marker_name == "TRAINING_COMPLETE":
                return True
            return False

        with patch.object(monitor, "_is_training_alive", return_value=True):
            with patch.object(monitor, "_check_marker", side_effect=check_marker_side_effect):
                with patch.object(monitor, "_get_resources") as mock_resources:
                    mock_resources.return_value = {
                        "gpu_util": 85.0,
                        "vram_used_gb": 12.0,
                        "vram_total_gb": 16.0,
                        "vram_pct": 75.0,
                        "gpu_temp": 75.0,
                        "ram_used_gb": 8.0,
                        "ram_total_gb": 32.0,
                    }

                    # Complete after some checks (need many False values)
                    # Each iteration: TRAINING_COMPLETE, TRAINING_FAILED
                    # Already handled by check_marker_side_effect above

                    with patch("src.pipeline.stages.training_monitor.logger") as mock_logger:
                        result = monitor._monitor_training(mock_ssh_client, {})

                    # Count INFO logs with "[MONITOR]" (status logs)
                    status_log_count = sum(1 for call in mock_logger.info.call_args_list if "[MONITOR]" in str(call))

                    # Should log at 0s, 15s, 30s, 45s = 4 times
                    # (Plus completion log, so >= 4)
                    assert status_log_count >= 4

        assert result.is_ok()

    @patch("time.sleep")
    @patch("time.time")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    def test_monitor_training_rate_limiting_downloads(
        self,
        mock_log_manager_cls,
        mock_time,
        mock_sleep,
        mock_config,
        mock_ssh_client,
    ):
        """Test log downloads rate limiting (configurable interval)."""
        # log_download_interval = 30 (hardcoded constant)
        from itertools import count

        time_counter = count(start=0, step=1)
        mock_time.side_effect = lambda: next(time_counter)

        mock_log_manager_instance = MagicMock()
        mock_log_manager_cls.return_value = mock_log_manager_instance

        monitor = TrainingMonitor(mock_config)
        monitor._log_manager = mock_log_manager_instance
        monitor._workspace_path = "/workspace"
        # log_download_interval = 30 is hardcoded constant

        check_call_count = [0]

        def check_marker_side_effect(client, marker_name):
            check_call_count[0] += 1
            # Complete after 20 loops (40 calls)
            if check_call_count[0] >= 40 and marker_name == "TRAINING_COMPLETE":
                return True
            return False

        with patch.object(monitor, "_is_training_alive", return_value=True):
            with patch.object(monitor, "_check_marker", side_effect=check_marker_side_effect):
                with patch.object(monitor, "_get_resources") as mock_resources:
                    mock_resources.return_value = {
                        "gpu_util": 0.0,
                        "vram_used_gb": 0.0,
                        "vram_total_gb": 0.0,
                        "vram_pct": 0.0,
                        "gpu_temp": 0.0,
                        "ram_used_gb": 0.0,
                        "ram_total_gb": 0.0,
                    }

                    # Complete after some checks
                    # Already handled by check_marker_side_effect above

                    result = monitor._monitor_training(mock_ssh_client, {})

        assert result.is_ok()

        # Check download was called periodically (at 30s, 60s = 2 times during monitoring)
        # Plus 1 final download on completion = 3 total
        assert mock_log_manager_instance.download.call_count >= 2

    @patch("time.sleep")
    @patch("time.time")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    def test_monitor_training_resource_check_callback(
        self,
        mock_log_manager_cls,
        mock_time,
        mock_sleep,
        mock_config,
        mock_ssh_client,
        mock_callbacks,
    ):
        """Test on_resource_check callback with correct metrics."""
        from itertools import count

        time_counter = count(start=0, step=1)
        mock_time.side_effect = lambda: next(time_counter)

        mock_log_manager_instance = MagicMock()
        mock_log_manager_cls.return_value = mock_log_manager_instance

        monitor = TrainingMonitor(mock_config, callbacks=mock_callbacks)
        monitor._log_manager = mock_log_manager_instance
        monitor._workspace_path = "/workspace"

        expected_resources = {
            "gpu_util": 90.0,
            "vram_used_gb": 14.5,
            "vram_total_gb": 16.0,
            "vram_pct": 90.6,
            "gpu_temp": 78.0,
            "ram_used_gb": 10.0,
            "ram_total_gb": 32.0,
        }

        check_call_count = [0]

        def check_marker_side_effect(client, marker_name):
            check_call_count[0] += 1
            # Complete after 10 loops (20 calls)
            if check_call_count[0] >= 20 and marker_name == "TRAINING_COMPLETE":
                return True
            return False

        with patch.object(monitor, "_is_training_alive", return_value=True):
            with patch.object(monitor, "_check_marker", side_effect=check_marker_side_effect):
                with patch.object(monitor, "_get_resources", return_value=expected_resources):
                    # Complete after some checks
                    # Already handled by check_marker_side_effect above

                    result = monitor._monitor_training(mock_ssh_client, {})

        assert result.is_ok()

        # Check callback was called with correct resources (at 15s mark)
        mock_callbacks.on_resource_check.assert_called()
        callback_calls = mock_callbacks.on_resource_check.call_args_list

        # At least one call should have the expected resources
        assert any(
            call[0][0] == expected_resources for call in callback_calls
        ), "on_resource_check should be called with correct resources dict"


# =============================================================================
# BATCH 3: HELPERS + INTEGRATION (10 tests)
# =============================================================================


class TestHelperMethods:
    """Tests for TrainingMonitor helper methods."""

    def test_is_training_alive_docker_found(self, mock_config):
        """Test _is_training_alive when Docker container is found."""
        mock_ssh_client = MagicMock()
        # Docker ps returns container name with RUNNING
        mock_ssh_client.exec_command.return_value = (True, "helix_training_pod123\nRUNNING", "")

        monitor = TrainingMonitor(mock_config)
        result = monitor._is_training_alive(mock_ssh_client)

        assert result is True

    def test_is_training_alive_python_fallback(self, mock_config):
        """Test _is_training_alive fallback to Python process check."""
        mock_ssh_client = MagicMock()
        # Docker check fails, but Python process found
        mock_ssh_client.exec_command.side_effect = [
            (False, "", ""),  # Docker check fails
            (True, "root 1234 python /workspace/train.py", ""),  # Python process found
        ]

        monitor = TrainingMonitor(mock_config)
        result = monitor._is_training_alive(mock_ssh_client)

        assert result is True

    def test_is_training_alive_dead(self, mock_config):
        """Test _is_training_alive when no process found."""
        mock_ssh_client = MagicMock()
        # Docker check fails, Python check returns empty
        mock_ssh_client.exec_command.side_effect = [
            (False, "", ""),  # Docker check fails
            (True, "", ""),  # Python process check returns empty
        ]

        monitor = TrainingMonitor(mock_config)
        result = monitor._is_training_alive(mock_ssh_client)

        assert result is False

    def test_check_marker_exists(self, mock_config):
        """Test _check_marker when marker file exists."""
        mock_ssh_client = MagicMock()
        mock_ssh_client.exec_command.return_value = (True, "EXISTS", "")

        monitor = TrainingMonitor(mock_config)
        monitor._workspace_path = "/workspace"
        result = monitor._check_marker(mock_ssh_client, "TRAINING_COMPLETE")

        assert result is True
        # Verify correct command
        assert "test -f /workspace/TRAINING_COMPLETE" in str(mock_ssh_client.exec_command.call_args)

    def test_check_marker_not_found(self, mock_config):
        """Test _check_marker when marker file doesn't exist."""
        mock_ssh_client = MagicMock()
        mock_ssh_client.exec_command.return_value = (False, "", "")

        monitor = TrainingMonitor(mock_config)
        monitor._workspace_path = "/workspace"
        result = monitor._check_marker(mock_ssh_client, "TRAINING_FAILED")

        assert result is False

    def test_read_marker_content_success(self, mock_config):
        """Test _read_marker_content successful read."""
        mock_ssh_client = MagicMock()
        mock_ssh_client.exec_command.return_value = (True, "Out of memory error\n", "")

        monitor = TrainingMonitor(mock_config)
        monitor._workspace_path = "/workspace"
        result = monitor._read_marker_content(mock_ssh_client, "TRAINING_FAILED")

        assert result == "Out of memory error"

    def test_read_marker_content_failure(self, mock_config):
        """Test _read_marker_content when read fails."""
        mock_ssh_client = MagicMock()
        mock_ssh_client.exec_command.return_value = (False, "", "File not found")

        monitor = TrainingMonitor(mock_config)
        monitor._workspace_path = "/workspace"
        result = monitor._read_marker_content(mock_ssh_client, "TRAINING_FAILED")

        assert result == "Unknown error"

    def test_get_resources_gpu_metrics_linux(self, mock_config):
        """Test _get_resources with GPU metrics parsing."""
        mock_ssh_client = MagicMock()
        # nvidia-smi output: gpu_util, vram_used_mb, vram_total_mb, temp
        mock_ssh_client.exec_command.side_effect = [
            (True, "85, 12500, 16000, 75", ""),  # GPU metrics
            (True, "", ""),  # cgroup probe (no data)
            (False, "", ""),  # RAM check (fail for this test)
            (False, "", ""),  # macOS fallback #1 (sysctl)
            (False, "", ""),  # macOS fallback #2 (vm_stat) - in case
        ]

        monitor = TrainingMonitor(mock_config)
        resources = monitor._get_resources(mock_ssh_client)

        assert resources["gpu_util"] == 85.0
        assert abs(resources["vram_used_gb"] - 12.2) < 0.1  # 12500 MB / 1024
        assert abs(resources["vram_total_gb"] - 15.6) < 0.1  # 16000 MB / 1024
        assert abs(resources["vram_pct"] - 78.125) < 0.1  # 12500/16000 * 100
        assert resources["gpu_temp"] == 75.0

    def test_get_resources_ram_linux(self, mock_config):
        """Test _get_resources with Linux RAM parsing (/proc/meminfo)."""
        mock_ssh_client = MagicMock()
        # Realistic /proc/meminfo output
        meminfo_output = """MemTotal:       33554432 kB
MemFree:        16777216 kB
MemAvailable:   20971520 kB
Buffers:         1048576 kB
Cached:          3145728 kB
"""
        mock_ssh_client.exec_command.side_effect = [
            (True, "50, 8000, 16000, 70", ""),  # GPU metrics
            (True, "", ""),  # cgroup probe (no data)
            (True, meminfo_output, ""),  # RAM via /proc/meminfo
        ]

        monitor = TrainingMonitor(mock_config)
        resources = monitor._get_resources(mock_ssh_client)

        # MemTotal = 33554432 kB = ~32 GB
        # MemAvailable = 20971520 kB = ~20 GB
        # Used = Total - Available = ~12 GB
        assert abs(resources["ram_total_gb"] - 32.0) < 1.0
        assert abs(resources["ram_used_gb"] - 12.0) < 1.0

    def test_get_resources_ram_cgroup_v1(self, mock_config):
        """Prefer cgroup v1 memory limit/usage over /proc/meminfo when available."""
        mock_ssh_client = MagicMock()

        limit_bytes = 30_999_998_464  # ~31GB decimal, ~28.9GiB
        usage_bytes = 2_147_483_648  # 2GiB
        cgroup_out = f"memory.limit_in_bytes={limit_bytes}\nmemory.usage_in_bytes={usage_bytes}\n"

        # Only 2 calls expected: nvidia-smi + cgroup probe. If /proc/meminfo is called, test should fail.
        mock_ssh_client.exec_command.side_effect = [
            (True, "50, 8000, 16000, 70", ""),  # GPU metrics
            (True, cgroup_out, ""),  # cgroup probe provides v1 metrics
        ]

        monitor = TrainingMonitor(mock_config)
        resources = monitor._get_resources(mock_ssh_client)

        assert abs(resources["ram_total_gb"] - (limit_bytes / (1024**3))) < 0.2
        assert abs(resources["ram_used_gb"] - (usage_bytes / (1024**3))) < 0.05

    def test_get_resources_ram_cgroup_v2(self, mock_config):
        """Prefer cgroup v2 memory.max/current over /proc/meminfo when available."""
        mock_ssh_client = MagicMock()

        max_bytes = 34_359_738_368  # 32GiB
        cur_bytes = 4_294_967_296  # 4GiB
        cgroup_out = f"memory.max={max_bytes}\nmemory.current={cur_bytes}\n"

        mock_ssh_client.exec_command.side_effect = [
            (True, "50, 8000, 16000, 70", ""),  # GPU metrics
            (True, cgroup_out, ""),  # cgroup probe provides v2 metrics
        ]

        monitor = TrainingMonitor(mock_config)
        resources = monitor._get_resources(mock_ssh_client)

        assert abs(resources["ram_total_gb"] - 32.0) < 0.1
        assert abs(resources["ram_used_gb"] - 4.0) < 0.1

    def test_get_resources_ram_cgroup_unlimited_falls_back_to_meminfo(self, mock_config):
        """If cgroup reports an effectively unlimited limit, fall back to /proc/meminfo."""
        mock_ssh_client = MagicMock()

        # Typical "unlimited" style value in cgroup v1 inside containers
        cgroup_out = "memory.limit_in_bytes=9223372036854771712\nmemory.usage_in_bytes=123\n"

        meminfo_output = """MemTotal:       33554432 kB
MemAvailable:   20971520 kB
"""
        mock_ssh_client.exec_command.side_effect = [
            (True, "50, 8000, 16000, 70", ""),  # GPU metrics
            (True, cgroup_out, ""),  # cgroup probe (ignored)
            (True, meminfo_output, ""),  # /proc/meminfo
        ]

        monitor = TrainingMonitor(mock_config)
        resources = monitor._get_resources(mock_ssh_client)
        assert abs(resources["ram_total_gb"] - 32.0) < 1.0
        assert abs(resources["ram_used_gb"] - 12.0) < 1.0

    def test_get_resources_ram_cgroup_partial_data_falls_back_to_meminfo(self, mock_config):
        """If cgroup output is incomplete, fall back to /proc/meminfo."""
        mock_ssh_client = MagicMock()
        cgroup_out = "memory.limit_in_bytes=30999998464\n"  # missing usage
        meminfo_output = """MemTotal:       33554432 kB
MemAvailable:   20971520 kB
"""
        mock_ssh_client.exec_command.side_effect = [
            (True, "50, 8000, 16000, 70", ""),  # GPU metrics
            (True, cgroup_out, ""),  # cgroup probe (incomplete)
            (True, meminfo_output, ""),  # /proc/meminfo
        ]
        monitor = TrainingMonitor(mock_config)
        resources = monitor._get_resources(mock_ssh_client)
        assert abs(resources["ram_total_gb"] - 32.0) < 1.0

    def test_display_last_log_lines(self, mock_config):
        """Test _display_last_log_lines output."""
        mock_log_manager = MagicMock()
        mock_log_manager.get_last_lines.return_value = [
            "Epoch 1/3 completed",
            "Loss: 2.5",
            "Epoch 2/3 completed",
        ]
        mock_log_manager.local_path.exists.return_value = True
        mock_log_manager.local_path = MagicMock()
        mock_log_manager.local_path.__str__.return_value = "/logs/training.log"

        monitor = TrainingMonitor(mock_config)
        monitor._log_manager = mock_log_manager

        # Just call the method and verify log_manager was used
        monitor._display_last_log_lines(n=30)

        # Verify log manager methods were called
        mock_log_manager.get_last_lines.assert_called_once_with(30)
        mock_log_manager.local_path.exists.assert_called_once()

    def test_display_last_log_lines_empty(self, mock_config):
        """Test _display_last_log_lines with no log content."""
        mock_log_manager = MagicMock()
        mock_log_manager.get_last_lines.return_value = []

        monitor = TrainingMonitor(mock_config)
        monitor._log_manager = mock_log_manager

        # Just call the method
        monitor._display_last_log_lines(n=30)

        # Verify early return (local_path.exists should not be called)
        mock_log_manager.get_last_lines.assert_called_once_with(30)
        mock_log_manager.local_path.exists.assert_not_called()

    def test_display_last_log_lines_no_manager(self, mock_config):
        """Test _display_last_log_lines when log_manager is None."""
        monitor = TrainingMonitor(mock_config)
        monitor._log_manager = None

        # Should return early without error
        monitor._display_last_log_lines(n=30)
        # No assertion needed - just verify no exception

    def test_get_resources_ram_parsing_exception(self, mock_config):
        """Test _get_resources graceful handling of RAM parsing exception."""
        mock_ssh_client = MagicMock()

        bad_meminfo = """MemTotal:
MemFree:        16777216 kB
"""

        mock_ssh_client.exec_command.side_effect = [
            (True, "80, 12000, 16000, 70", ""),  # GPU metrics
            (True, "", ""),  # cgroup probe (no data)
            (True, bad_meminfo, ""),  # Bad meminfo - will trigger exception
            (False, "", ""),  # macOS sysctl fails
        ]

        monitor = TrainingMonitor(mock_config)
        resources = monitor._get_resources(mock_ssh_client)

        # Should gracefully degrade to 0.0 for RAM
        assert resources["gpu_util"] == 80.0  # GPU still works
        assert resources["ram_total_gb"] == 0.0  # RAM parsing failed gracefully
        assert resources["ram_used_gb"] == 0.0


class TestParseCgroupRam:
    """Unit tests for TrainingMonitor._parse_cgroup_ram (static method)."""

    def test_cgroup_v2_valid(self):
        out = "memory.max=34359738368\nmemory.current=4294967296\n"
        result = TrainingMonitor._parse_cgroup_ram(out)
        assert result is not None
        total, used = result
        assert abs(total - 32.0) < 0.01
        assert abs(used - 4.0) < 0.01

    def test_cgroup_v1_valid(self):
        limit = 30_999_998_464
        usage = 2_147_483_648
        out = f"memory.limit_in_bytes={limit}\nmemory.usage_in_bytes={usage}\n"
        result = TrainingMonitor._parse_cgroup_ram(out)
        assert result is not None
        total, used = result
        assert abs(total - limit / (1024**3)) < 0.01
        assert abs(used - usage / (1024**3)) < 0.01

    def test_cgroup_v2_prefers_over_v1(self):
        """When both v1 and v2 keys are present, v2 wins."""
        v2_max = 34_359_738_368  # 32 GiB
        v2_cur = 1_073_741_824   # 1 GiB
        v1_lim = 16_106_127_360  # 15 GiB
        v1_use = 2_147_483_648   # 2 GiB
        out = (
            f"memory.max={v2_max}\n"
            f"memory.current={v2_cur}\n"
            f"memory.limit_in_bytes={v1_lim}\n"
            f"memory.usage_in_bytes={v1_use}\n"
        )
        result = TrainingMonitor._parse_cgroup_ram(out)
        assert result is not None
        total, _ = result
        assert abs(total - 32.0) < 0.01

    def test_unlimited_cgroup_v1_returns_none(self):
        out = "memory.limit_in_bytes=9223372036854771712\nmemory.usage_in_bytes=1000\n"
        assert TrainingMonitor._parse_cgroup_ram(out) is None

    def test_max_string_returns_none(self):
        out = "memory.max=max\nmemory.current=1073741824\n"
        assert TrainingMonitor._parse_cgroup_ram(out) is None

    def test_empty_output_returns_none(self):
        assert TrainingMonitor._parse_cgroup_ram("") is None
        assert TrainingMonitor._parse_cgroup_ram("   \n  ") is None

    def test_partial_v1_missing_usage_returns_none(self):
        out = "memory.limit_in_bytes=30999998464\n"
        assert TrainingMonitor._parse_cgroup_ram(out) is None

    def test_partial_v2_missing_current_returns_none(self):
        out = "memory.max=34359738368\n"
        assert TrainingMonitor._parse_cgroup_ram(out) is None

    def test_used_clamped_to_total(self):
        """usage > limit should not produce used > total."""
        total = 1_073_741_824   # 1 GiB
        usage = 2_147_483_648   # 2 GiB (exceeds limit)
        out = f"memory.max={total}\nmemory.current={usage}\n"
        result = TrainingMonitor._parse_cgroup_ram(out)
        assert result is not None
        total_gb, used_gb = result
        assert used_gb <= total_gb

    def test_noisy_output_with_extra_lines(self):
        """Extra non-kv lines should be silently ignored."""
        out = "some random line\nmemory.max=34359738368\nmemory.current=1073741824\nanother line\n"
        result = TrainingMonitor._parse_cgroup_ram(out)
        assert result is not None
        assert abs(result[0] - 32.0) < 0.01


class TestParseMemInfoRam:
    """Unit tests for TrainingMonitor._parse_meminfo_ram (static method)."""

    def test_valid_meminfo_uses_memavailable(self):
        out = "MemTotal:       33554432 kB\nMemFree:        8388608 kB\nMemAvailable:   20971520 kB\n"
        result = TrainingMonitor._parse_meminfo_ram(out)
        assert result is not None
        total, used = result
        assert abs(total - 32.0) < 0.5
        # Used = Total - Available = 33554432 - 20971520 = 12582912 kB ≈ 12 GiB
        assert abs(used - 12.0) < 0.5

    def test_valid_meminfo_falls_back_to_memfree(self):
        out = "MemTotal:       16777216 kB\nMemFree:        8388608 kB\n"
        result = TrainingMonitor._parse_meminfo_ram(out)
        assert result is not None
        total, used = result
        assert abs(total - 16.0) < 0.1
        assert abs(used - 8.0) < 0.1

    def test_missing_memtotal_returns_none(self):
        out = "MemFree:        8388608 kB\nMemAvailable:   12582912 kB\n"
        assert TrainingMonitor._parse_meminfo_ram(out) is None

    def test_empty_string_returns_none(self):
        assert TrainingMonitor._parse_meminfo_ram("") is None

    def test_malformed_memtotal_line_returns_none(self):
        """MemTotal: with no value should not crash and return None."""
        out = "MemTotal:\nMemFree: 8388608 kB\n"
        result = TrainingMonitor._parse_meminfo_ram(out)
        assert result is None

    def test_non_digit_value_skipped(self):
        out = "MemTotal:       bad_value kB\nMemFree:        8388608 kB\n"
        assert TrainingMonitor._parse_meminfo_ram(out) is None


    @patch("time.sleep")
    @patch("time.time")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    def test_monitor_training_process_died_failed_marker_in_retry(
        self,
        mock_log_manager_cls,
        mock_time,
        mock_sleep,
        mock_config,
        mock_callbacks,
        mock_ssh_client,
    ):
        """Test process died but TRAINING_FAILED appears during retry loop (lines 316-323)."""
        from itertools import count

        time_counter = count(start=0, step=1)
        mock_time.side_effect = lambda: next(time_counter)

        mock_log_manager_instance = MagicMock()
        mock_log_manager_cls.return_value = mock_log_manager_instance
        mock_log_manager_instance.get_last_lines.return_value = []

        monitor = TrainingMonitor(mock_config, callbacks=mock_callbacks)
        monitor._log_manager = mock_log_manager_instance
        monitor._workspace_path = "/workspace"

        check_call_count = [0]

        def check_marker_side_effect(client, marker_name):
            check_call_count[0] += 1
            # Process died scenario:
            # Initial checks (1-2): not found
            # After 2s wait checks (3-4): not found
            # Retry attempt 1 checks (5-6): TRAINING_FAILED found on check 6!
            if check_call_count[0] == 6 and marker_name == "TRAINING_FAILED":
                return True
            return False

        with patch.object(monitor, "_is_training_alive", return_value=False):
            with patch.object(monitor, "_check_marker", side_effect=check_marker_side_effect):
                with patch.object(monitor, "_read_marker_content", return_value="GPU out of memory"):
                    result = monitor._monitor_training(mock_ssh_client, {})

        assert result.is_err()
        assert "GPU out of memory" in str(result.unwrap_err())

        # Check callback
        mock_callbacks.on_training_failed.assert_called_once()
        error_msg, duration = mock_callbacks.on_training_failed.call_args[0]
        assert "GPU out of memory" in error_msg

        # Check log download
        mock_log_manager_instance.download_on_error.assert_called_once()

    @patch("time.sleep")
    @patch("time.time")
    @patch("src.pipeline.stages.training_monitor.LogManager")
    def test_monitor_training_process_died_failed_marker_immediate(
        self,
        mock_log_manager_cls,
        mock_time,
        mock_sleep,
        mock_config,
        mock_callbacks,
        mock_ssh_client,
    ):
        """Test process died with TRAINING_FAILED found immediately after 2s wait (lines 282-290)."""
        from itertools import count

        time_counter = count(start=0, step=1)
        mock_time.side_effect = lambda: next(time_counter)

        mock_log_manager_instance = MagicMock()
        mock_log_manager_cls.return_value = mock_log_manager_instance
        mock_log_manager_instance.get_last_lines.return_value = []

        monitor = TrainingMonitor(mock_config, callbacks=mock_callbacks)
        monitor._log_manager = mock_log_manager_instance
        monitor._workspace_path = "/workspace"

        check_call_count = [0]

        def check_marker_side_effect(client, marker_name):
            check_call_count[0] += 1
            # Process died scenario:
            # Initial checks (1-2): not found
            # After 2s wait: TRAINING_FAILED found immediately (check 4)
            if check_call_count[0] == 4 and marker_name == "TRAINING_FAILED":
                return True
            return False

        with patch.object(monitor, "_is_training_alive", return_value=False):
            with patch.object(monitor, "_check_marker", side_effect=check_marker_side_effect):
                with patch.object(monitor, "_read_marker_content", return_value="Disk full error"):
                    result = monitor._monitor_training(mock_ssh_client, {})

        assert result.is_err()
        assert "Disk full error" in str(result.unwrap_err())

        # Check callback
        mock_callbacks.on_training_failed.assert_called_once()
        error_msg, duration = mock_callbacks.on_training_failed.call_args[0]
        assert "Disk full error" in error_msg

        # Check log download (if condition on line 285-286)
        mock_log_manager_instance.download_on_error.assert_called_once()

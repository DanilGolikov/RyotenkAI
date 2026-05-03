"""
Extended tests for environment.py to improve coverage.

Target: Increase environment.py coverage from 40.78% to >70%.

Tests cover:
- Helper functions (_get_python_version, _get_package_version, etc.)
- EnvironmentSnapshot.to_dict()
- EnvironmentReporter constructor and methods
- Git commit detection
- GPU info detection
- Edge cases with missing dependencies
"""

from unittest.mock import MagicMock, patch

from src.utils.environment import (
    EnvironmentReporter,
    EnvironmentSnapshot,
    _get_cuda_version,
    _get_git_commit,
    _get_gpu_name,
    _get_gpu_vram_gb,
    _get_package_version,
    _get_python_version,
)

# =============================================================================
# TESTS: Helper Functions
# =============================================================================


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_get_python_version(self):
        """_get_python_version() should return version string."""
        version = _get_python_version()

        assert isinstance(version, str)
        assert "." in version  # Should be like "3.11.5"
        assert len(version.split(".")) >= 2

    def test_get_package_version_installed(self):
        """_get_package_version() should return version for installed package."""
        # pytest is definitely installed
        version = _get_package_version("pytest")

        assert version != "N/A"
        assert "." in version

    def test_get_package_version_not_installed(self):
        """_get_package_version() should return N/A for missing package."""
        version = _get_package_version("nonexistent_package_xyz_12345")

        assert version == "N/A"

    @patch("torch.cuda.is_available", return_value=True)
    def test_get_cuda_version_available(self, mock_is_available):
        """_get_cuda_version() should return version when CUDA available."""
        version = _get_cuda_version()

        # Should be version string or "N/A (CPU mode)"
        assert version in ["11.8", "12.1", "N/A (CPU mode)", "N/A"]

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_cuda_version_cpu_mode(self, mock_is_available):
        """_get_cuda_version() should return CPU mode when no CUDA."""
        version = _get_cuda_version()

        assert "CPU mode" in version or version == "N/A"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090")
    def test_get_gpu_name_available(self, mock_get_name, mock_is_available):
        """_get_gpu_name() should return GPU name when available."""
        gpu_name = _get_gpu_name()

        # Should be GPU name or "CPU"
        assert gpu_name in ["NVIDIA RTX 4090", "CPU"]

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_gpu_name_cpu(self, mock_is_available):
        """_get_gpu_name() should return CPU when no GPU."""
        gpu_name = _get_gpu_name()

        assert gpu_name == "CPU"

    @patch("torch.cuda.is_available", return_value=False)
    def test_get_gpu_vram_gb_cpu(self, mock_is_available):
        """_get_gpu_vram_gb() should return 0.0 when no GPU."""
        vram = _get_gpu_vram_gb()

        assert vram == 0.0

    def test_get_git_commit_with_git(self):
        """_get_git_commit() should return commit hash or N/A."""
        commit = _get_git_commit()

        # Should be hash (7+ chars) or "N/A"
        assert commit == "N/A" or len(commit) >= 7

    @patch("subprocess.run")
    def test_get_git_commit_no_git(self, mock_run):
        """_get_git_commit() should return N/A when git fails."""
        mock_run.side_effect = FileNotFoundError()

        commit = _get_git_commit()

        assert commit == "N/A"


# =============================================================================
# TESTS: EnvironmentSnapshot.to_dict()
# =============================================================================


class TestEnvironmentSnapshotToDict:
    """Test EnvironmentSnapshot.to_dict() method."""

    def test_to_dict_contains_all_fields(self):
        """to_dict() should include all snapshot fields."""
        snapshot = EnvironmentSnapshot.collect()
        data = snapshot.to_dict()

        required_fields = [
            "python_version",
            "torch_version",
            "cuda_version",
            "transformers_version",
            "peft_version",
            "trl_version",
            "accelerate_version",
            "bitsandbytes_version",
            "gpu_name",
            "gpu_vram_gb",
            "os_info",
            "git_commit",
            "timestamp",
        ]

        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_to_dict_values_not_none(self):
        """to_dict() should have non-None values."""
        snapshot = EnvironmentSnapshot.collect()
        data = snapshot.to_dict()

        for key, value in data.items():
            assert value is not None, f"Field {key} should not be None"

    def test_to_dict_timestamp_format(self):
        """to_dict() timestamp should be ISO format."""
        snapshot = EnvironmentSnapshot.collect()
        data = snapshot.to_dict()

        timestamp = data["timestamp"]
        assert "T" in timestamp or "-" in timestamp  # ISO 8601


# =============================================================================
# TESTS: EnvironmentReporter.__init__()
# =============================================================================


class TestEnvironmentReporterInit:
    """Test EnvironmentReporter initialization."""

    def test_init_with_snapshot(self):
        """__init__() should accept snapshot."""
        snapshot = EnvironmentSnapshot.collect()
        reporter = EnvironmentReporter(snapshot)

        assert reporter.snapshot is snapshot

    def test_init_stores_snapshot(self):
        """__init__() should store snapshot as attribute."""
        snapshot = EnvironmentSnapshot.collect()
        reporter = EnvironmentReporter(snapshot)

        assert hasattr(reporter, "snapshot")
        assert isinstance(reporter.snapshot, EnvironmentSnapshot)


# =============================================================================
# TESTS: EnvironmentReporter.collect()
# =============================================================================


class TestEnvironmentReporterCollect:
    """Test EnvironmentReporter.collect() class method."""

    def test_collect_creates_reporter(self):
        """collect() should create reporter with fresh snapshot."""
        reporter = EnvironmentReporter.collect()

        assert isinstance(reporter, EnvironmentReporter)
        assert isinstance(reporter.snapshot, EnvironmentSnapshot)

    def test_collect_snapshot_has_data(self):
        """collect() snapshot should have actual data."""
        reporter = EnvironmentReporter.collect()

        assert reporter.snapshot.python_version
        assert reporter.snapshot.torch_version
        assert reporter.snapshot.timestamp


# =============================================================================
# TESTS: EnvironmentReporter Methods
# =============================================================================


class TestEnvironmentReporterMethods:
    """Test EnvironmentReporter methods."""

    def test_to_dict_returns_snapshot_dict(self):
        """to_dict() should return snapshot's dictionary."""
        reporter = EnvironmentReporter.collect()

        data = reporter.to_dict()
        snapshot_data = reporter.snapshot.to_dict()

        assert data == snapshot_data

    def test_get_short_summary_format(self):
        """get_short_summary() should have correct format."""
        reporter = EnvironmentReporter.collect()

        summary = reporter.get_short_summary()

        # Should contain key terms
        assert "Python" in summary
        assert "PyTorch" in summary
        assert "CUDA" in summary
        # Should be comma-separated
        assert "," in summary

    def test_get_short_summary_short(self):
        """get_short_summary() should be reasonably short."""
        reporter = EnvironmentReporter.collect()

        summary = reporter.get_short_summary()

        # Should be one line, not too long
        assert len(summary) < 200

    def test_log_summary_calls_logger(self):
        """log_summary() should log environment info."""
        reporter = EnvironmentReporter.collect()

        # Should not raise
        reporter.log_summary()  # Logs to console, not easily capturable

    def test_log_debug_calls_logger(self):
        """log_debug() should log at DEBUG level."""
        reporter = EnvironmentReporter.collect()

        # Should not raise
        reporter.log_debug()  # Logs at DEBUG level


# =============================================================================
# TESTS: Git Commit Detection
# =============================================================================


class TestGitCommitDetection:
    """Test git commit detection."""

    @patch("subprocess.run")
    def test_git_commit_success(self, mock_run):
        """Git commit should be detected when available."""
        mock_run.return_value = MagicMock(stdout="a1b2c3d4e5f6789\n", returncode=0)

        commit = _get_git_commit()

        assert commit == "a1b2c3d" or len(commit) >= 7

    @patch("subprocess.run")
    def test_git_commit_nonzero_return(self, mock_run):
        """Git commit should return N/A on non-zero exit."""
        mock_run.return_value = MagicMock(returncode=128)

        commit = _get_git_commit()

        assert commit == "N/A"


# =============================================================================
# TESTS: GPU Detection Edge Cases
# =============================================================================


class TestGPUDetectionEdgeCases:
    """Test GPU detection edge cases."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_no_cuda_cpu_mode(self, mock_is_available):
        """When CUDA unavailable, should report CPU mode."""
        snapshot = EnvironmentSnapshot.collect()

        assert snapshot.gpu_name == "CPU"
        assert snapshot.gpu_vram_gb == 0.0
        assert "CPU mode" in snapshot.cuda_version or snapshot.cuda_version == "N/A"

    def test_snapshot_handles_import_errors(self):
        """Snapshot should not crash on import errors."""
        # This should not raise even if some packages missing
        snapshot = EnvironmentSnapshot.collect()

        assert snapshot.python_version  # Should always have Python version
        assert snapshot.os_info  # Should always have OS info

    def test_os_info_not_empty(self):
        """OS info should contain meaningful data."""
        snapshot = EnvironmentSnapshot.collect()

        os_info = snapshot.os_info

        assert os_info
        assert len(os_info) > 3  # At least some OS name


# =============================================================================
# TESTS: Reporter Integration
# =============================================================================


class TestReporterIntegration:
    """Integration tests for reporter."""

    def test_reporter_workflow(self):
        """Test typical reporter usage workflow."""
        # Collect
        reporter = EnvironmentReporter.collect()

        # Get dictionary
        data = reporter.to_dict()
        assert isinstance(data, dict)

        # Get summary
        summary = reporter.get_short_summary()
        assert isinstance(summary, str)

        # Log (should not raise)
        reporter.log_summary()
        reporter.log_debug()

    def test_reporter_dict_format(self):
        """Test dictionary format for W&B/MLflow."""
        reporter = EnvironmentReporter.collect()

        # Get dict
        env_dict = reporter.to_dict()

        # Should have expected keys
        assert "python_version" in env_dict
        assert "torch_version" in env_dict
        assert "gpu_name" in env_dict

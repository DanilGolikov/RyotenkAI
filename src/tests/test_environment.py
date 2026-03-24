"""
Tests for EnvironmentReporter.

Verifies environment collection and reporting functionality
for training reproducibility.
"""

from __future__ import annotations

import pytest

from src.utils.environment import EnvironmentReporter, EnvironmentSnapshot

# =============================================================================
# TESTS: EnvironmentSnapshot
# =============================================================================


class TestEnvironmentSnapshot:
    """Tests for EnvironmentSnapshot dataclass."""

    def test_collect_returns_snapshot(self):
        """collect() should return EnvironmentSnapshot."""
        snapshot = EnvironmentSnapshot.collect()

        assert isinstance(snapshot, EnvironmentSnapshot)

    def test_snapshot_has_python_version(self):
        """Snapshot should contain Python version."""
        snapshot = EnvironmentSnapshot.collect()

        assert snapshot.python_version
        assert "." in snapshot.python_version  # e.g., "3.11.5"

    def test_snapshot_has_torch_version(self):
        """Snapshot should contain PyTorch version."""
        snapshot = EnvironmentSnapshot.collect()

        # Should be version or "N/A"
        assert snapshot.torch_version

    def test_snapshot_has_transformers_version(self):
        """Snapshot should contain Transformers version."""
        snapshot = EnvironmentSnapshot.collect()

        assert snapshot.transformers_version

    def test_snapshot_has_timestamp(self):
        """Snapshot should have ISO timestamp."""
        snapshot = EnvironmentSnapshot.collect()

        assert snapshot.timestamp
        # Should be ISO format
        assert "T" in snapshot.timestamp or "-" in snapshot.timestamp

    def test_snapshot_to_dict(self):
        """to_dict() should return all fields."""
        snapshot = EnvironmentSnapshot.collect()
        data = snapshot.to_dict()

        assert isinstance(data, dict)
        assert "python_version" in data
        assert "torch_version" in data
        assert "transformers_version" in data
        assert "timestamp" in data
        assert "gpu_name" in data

    def test_snapshot_is_frozen(self):
        """Snapshot should be immutable (frozen dataclass)."""
        snapshot = EnvironmentSnapshot.collect()

        with pytest.raises(AttributeError):
            snapshot.python_version = "changed"  # type: ignore


# =============================================================================
# TESTS: EnvironmentReporter
# =============================================================================


class TestEnvironmentReporter:
    """Tests for EnvironmentReporter class."""

    def test_collect_returns_reporter(self):
        """collect() should return EnvironmentReporter."""
        reporter = EnvironmentReporter.collect()

        assert isinstance(reporter, EnvironmentReporter)

    def test_reporter_has_snapshot(self):
        """Reporter should contain snapshot."""
        reporter = EnvironmentReporter.collect()

        assert hasattr(reporter, "snapshot")
        assert isinstance(reporter.snapshot, EnvironmentSnapshot)

    def test_to_dict_returns_dict(self):
        """to_dict() should return dictionary."""
        reporter = EnvironmentReporter.collect()
        data = reporter.to_dict()

        assert isinstance(data, dict)
        assert "python_version" in data

    def test_get_short_summary(self):
        """get_short_summary() should return one-line summary."""
        reporter = EnvironmentReporter.collect()
        summary = reporter.get_short_summary()

        assert isinstance(summary, str)
        assert "Python" in summary
        assert "PyTorch" in summary

    def test_log_summary_does_not_raise(self, capsys):
        """log_summary() should not raise exception."""
        reporter = EnvironmentReporter.collect()

        # Should not raise
        reporter.log_summary()

    def test_log_debug_does_not_raise(self, capsys):
        """log_debug() should not raise exception."""
        reporter = EnvironmentReporter.collect()

        # Should not raise
        reporter.log_debug()


# =============================================================================
# TESTS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_git_commit_handles_missing_git(self):
        """Should handle missing git gracefully."""
        snapshot = EnvironmentSnapshot.collect()

        # Should be hash or "N/A", not raise
        assert snapshot.git_commit is not None

    def test_gpu_info_handles_no_gpu(self):
        """Should handle no GPU gracefully."""
        snapshot = EnvironmentSnapshot.collect()

        # Should be GPU name or "CPU", not raise
        assert snapshot.gpu_name is not None
        # VRAM should be >= 0
        assert snapshot.gpu_vram_gb >= 0

    def test_os_info_not_empty(self):
        """OS info should not be empty."""
        snapshot = EnvironmentSnapshot.collect()

        assert snapshot.os_info
        assert len(snapshot.os_info) > 0


# =============================================================================
# TESTS: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for environment reporting."""

    def test_reporter_snapshot_consistency(self):
        """Reporter should use same snapshot throughout."""
        reporter = EnvironmentReporter.collect()

        dict_data = reporter.to_dict()
        assert "python_version" in dict_data
        assert dict_data["python_version"] == reporter.snapshot.python_version

    def test_multiple_collections_independent(self):
        """Multiple collect() calls should be independent."""
        reporter1 = EnvironmentReporter.collect()
        reporter2 = EnvironmentReporter.collect()

        # Should have same Python version (same environment)
        assert reporter1.snapshot.python_version == reporter2.snapshot.python_version

        # But different snapshot objects (independent)
        assert reporter1.snapshot is not reporter2.snapshot

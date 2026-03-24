"""
Marker File Notifier.

Creates marker files to signal training completion status.
Used for RunPod integration where the monitoring script polls for these files.

Marker files:
- TRAINING_COMPLETE: Created on successful completion
- TRAINING_FAILED: Created on failure

Example:
    >>> notifier = MarkerFileNotifier(base_path="/workspace")
    >>> notifier.notify_complete({"output_path": "/workspace/model"})
    # Creates /workspace/TRAINING_COMPLETE with JSON content
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.training.constants import TRUNCATE_ERROR_DEBUG
from src.utils.logger import logger


class MarkerFileNotifier:
    """
    Creates marker files for training completion notification.

    Used in RunPod deployments where the orchestrator monitors
    for marker files to detect training completion.

    Attributes:
        base_path: Directory where marker files are created (default: /workspace)

    Example:
        >>> notifier = MarkerFileNotifier(base_path="/workspace")
        >>> notifier.notify_complete({"output_path": "/workspace/output"})
        # Creates /workspace/TRAINING_COMPLETE

        >>> notifier.notify_failed("OOM Error", {"phase": 2})
        # Creates /workspace/TRAINING_FAILED
    """

    COMPLETE_MARKER = "TRAINING_COMPLETE"
    FAILED_MARKER = "TRAINING_FAILED"

    def __init__(self, base_path: str = "/workspace"):
        """
        Initialize MarkerFileNotifier.

        Args:
            base_path: Directory for marker files (default: /workspace for RunPod)
        """
        self.base_path = Path(base_path)
        logger.debug(f"[NOTIFIER:INIT] MarkerFileNotifier with base_path={base_path}")

    def notify_complete(self, data: dict[str, Any]) -> None:
        """
        Create success marker file.

        Args:
            data: Metadata to include in marker file
                  (output_path, model_name, strategies, etc.)
        """
        marker_path = self.base_path / self.COMPLETE_MARKER
        content = {
            "status": "complete",
            "timestamp": time.time(),
            **data,
        }

        try:
            # Ensure base directory exists
            self.base_path.mkdir(parents=True, exist_ok=True)

            # Write marker file
            marker_path.write_text(json.dumps(content, indent=2))
            logger.info(f"✅ Created success marker: {marker_path}")
            logger.debug(f"[NOTIFIER:COMPLETE] path={marker_path}, data={list(data.keys())}")

        except Exception as e:
            logger.warning(f"Failed to create success marker: {e}")

    def notify_failed(self, error: str, data: dict[str, Any]) -> None:
        """
        Create failure marker file.

        Args:
            error: Error message
            data: Additional metadata (phase, model, etc.)
        """
        marker_path = self.base_path / self.FAILED_MARKER
        content = {
            "status": "failed",
            "timestamp": time.time(),
            "error": error,
            "error_type": data.get("error_type", "Unknown"),
            **data,
        }

        try:
            # Ensure base directory exists
            self.base_path.mkdir(parents=True, exist_ok=True)

            # Write marker file
            marker_path.write_text(json.dumps(content, indent=2))
            logger.info(f"❌ Created failure marker: {marker_path}")
            logger.debug(f"[NOTIFIER:FAILED] path={marker_path}, error={error[:TRUNCATE_ERROR_DEBUG]}...")

        except Exception as e:
            logger.warning(f"Failed to create failure marker: {e}")

    def cleanup(self) -> None:
        """Remove any existing marker files."""
        for marker in [self.COMPLETE_MARKER, self.FAILED_MARKER]:
            marker_path = self.base_path / marker
            if marker_path.exists():
                marker_path.unlink()
                logger.debug(f"[NOTIFIER:CLEANUP] Removed {marker_path}")

    def get_status(self) -> str | None:
        """
        Check current marker status.

        Returns:
            "complete", "failed", or None if no marker
        """
        if (self.base_path / self.COMPLETE_MARKER).exists():
            return "complete"
        if (self.base_path / self.FAILED_MARKER).exists():
            return "failed"
        return None


__all__ = ["MarkerFileNotifier"]

"""
Log Notifier.

Simple notifier that just logs completion status.
Useful for local development and testing where marker files aren't needed.

Example:
    >>> notifier = LogNotifier()
    >>> notifier.notify_complete({"output_path": "/tmp/model"})
    # Just logs: "Training completed successfully: /tmp/model"
"""

from __future__ import annotations

from typing import Any

from src.training.constants import DEFAULT_UNKNOWN
from src.utils.logger import logger


class LogNotifier:
    """
    Notifier that logs completion status to stdout.

    No file creation - just structured logging.
    Useful for local development and testing.

    Example:
        >>> notifier = LogNotifier()
        >>> notifier.notify_complete({"output_path": "/tmp/model"})
        INFO: Training completed successfully: /tmp/model
    """

    @staticmethod
    def notify_complete(data: dict[str, Any]) -> None:
        """
        Log successful completion.

        Args:
            data: Metadata to log
        """
        output_path = data.get("output_path", DEFAULT_UNKNOWN)
        model_name = data.get("model_name", DEFAULT_UNKNOWN)
        strategies = data.get("strategies", [])
        total_phases = data.get("total_phases", 0)

        logger.info("🎉 Training completed successfully!")
        logger.info(f"   📦 Output: {output_path}")
        logger.info(f"   🤖 Model: {model_name}")
        if strategies:
            logger.info(f"   🔧 Strategies: {' → '.join(strategies)}")
        if total_phases:
            logger.info(f"   📊 Phases: {total_phases}")

    @staticmethod
    def notify_failed(error: str, data: dict[str, Any]) -> None:
        """
        Log failure.

        Args:
            error: Error message
            data: Additional metadata
        """
        error_type = data.get("error_type", "Unknown")
        model_name = data.get("model_name", DEFAULT_UNKNOWN)
        phase = data.get("phase")

        logger.error("❌ Training failed!")
        logger.error(f"   Error: {error}")
        logger.error(f"   Type: {error_type}")
        if model_name != DEFAULT_UNKNOWN:
            logger.error(f"   Model: {model_name}")
        if phase is not None:
            logger.error(f"   Failed at phase: {phase}")


__all__ = ["LogNotifier"]
